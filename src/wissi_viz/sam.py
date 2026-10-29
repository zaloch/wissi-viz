"""Click / box segmentation for annotation (Segment Anything 2.1, Apache-2.0) with three interchangeable backends.

    svc = SamService()                       # backend='auto': torch → onnx → classic
    polys, info = svc.segment(source, rect=(x0, y0, x1, y1), points=[(x, y)], labels=[1])

The viewer sends the pane's visible rectangle (level-0 px) and the prompts; the region is read at the pyramid level
where it is ≤ ``size`` px (1024, SAM's input), encoded once (cached per slide + region, so further clicks in the same
view only run the small mask decoder), and the mask comes back as polygons in level-0 px.

Backends (all permissive):

* ``torch``   — Hugging Face ``transformers`` Sam2Model + ``facebook/sam2.1-hiera-*`` weights (Apache-2.0).
               ``pip install "wissi-viz[sam]"``; CUDA / Apple MPS / CPU. Best quality, ~10–50 ms per click on an RTX.
* ``onnx``    — the same SAM 2.1 weights exported to ONNX, run by ONNX Runtime (MIT, a core dependency): CUDA,
               DirectML (any Windows GPU) or CPU. **SAM 2.1 tiny ships inside wissi-viz** (int8, 33 MB,
               ``models/sam2.1-tiny-int8``; ``scripts/bundle_sam.py``), so it works offline with the plain install;
               ``WISSI_SAM_MODEL=small`` (or ``tiny-fp32``) downloads a larger float32 export once from this
               project's GitHub release. For a GPU, replace onnxruntime with onnxruntime-gpu / onnxruntime-directml.
* ``classic`` — no model: a colour flood fill from the click / Otsu inside the box. Always available, so the tool
               works without extra installs; masks are rougher.

Environment: ``WISSI_SAM_BACKEND`` (auto | torch | onnx | classic), ``WISSI_SAM_MODEL`` (tiny | small | base-plus |
large; default: the bundled tiny for onnx, small for torch), ``WISSI_SAM_DEVICE`` (auto | cuda | mps | cpu), ``WISSI_SAM_ONNX_DIR`` (pre-downloaded files).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import threading
import time
import urllib.request
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.filters import gaussian
from skimage.measure import find_contours
from skimage.segmentation import flood

from . import certs
from ._compat import otsu, user_dir
from .weights import ort_session

MODELS = {
    'tiny': 'facebook/sam2.1-hiera-tiny',
    'small': 'facebook/sam2.1-hiera-small',
    'base-plus': 'facebook/sam2.1-hiera-base-plus',
    'large': 'facebook/sam2.1-hiera-large',
}
ONNX_RELEASE = 'https://github.com/zaloch/wissi-viz/releases/download/sam2.1-onnx-v1'
BUNDLED = Path(__file__).resolve().parent / 'models' / 'sam2.1-tiny-int8'  # shipped with the package
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------------------------------------ backends
class _Classic:
    """Model-free fallback. Point: flood fill in Lab colour space from the click (tolerance from the local spread);
    box: Otsu on darkness inside the box, the component under the box centre (or the largest)."""

    name = 'classic'
    device = 'cpu'

    def embed(self, rgb: np.ndarray):
        lab = rgb2lab(gaussian(rgb, 1.0, channel_axis=-1, preserve_range=True).astype(np.uint8))
        return {'rgb': rgb, 'lab': lab}

    def predict(self, emb, points, labels, box):
        lab, rgb = emb['lab'], emb['rgb']
        h, w = lab.shape[:2]
        mask = np.zeros((h, w), bool)
        pos = [p for p, lb in zip(points, labels) if lb == 1]
        neg = [p for p, lb in zip(points, labels) if lb == 0]
        if box is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 > 2 and y1 - y0 > 2:
                dark = 255 - rgb[y0:y1, x0:x1].astype(np.float64).mean(-1)
                t = otsu(dark)
                fg = dark > t if t is not None else np.ones_like(dark, bool)
                lab_, n = ndi.label(ndi.binary_opening(fg, iterations=1))
                if n:
                    cy, cx = (y1 - y0) // 2, (x1 - x0) // 2
                    k = lab_[cy, cx] or (np.argmax(np.bincount(lab_.ravel())[1:]) + 1)
                    mask[y0:y1, x0:x1] = ndi.binary_fill_holes(lab_ == k)
        for x, y in pos:
            xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            r = 4
            patch = lab[max(0, yi - r) : yi + r + 1, max(0, xi - r) : xi + r + 1].reshape(-1, 3)
            tol = float(np.clip(2.5 * patch.std(0).mean() + 6, 8, 30))
            dist = np.sqrt(((lab - lab[yi, xi]) ** 2).sum(-1))
            m = flood(dist, (yi, xi), tolerance=tol)
            m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=1))
            lab_, n = ndi.label(m)
            if n:  # only the piece under the click (or the nearest piece)
                k = lab_[yi, xi]
                if not k:
                    _, (iy, ix) = ndi.distance_transform_edt(lab_ == 0, return_indices=True)
                    k = lab_[iy[yi, xi], ix[yi, xi]]
                mask |= lab_ == k
        for x, y in neg:
            xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            lab_, _ = ndi.label(mask)
            if lab_[yi, xi]:
                mask[lab_ == lab_[yi, xi]] = False
        return mask, 1.0


class _Torch:
    """transformers Sam2Model. The image is encoded once per region; prompts reuse the embeddings."""

    name = 'torch'

    def __init__(self, model: str, device: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        self.torch = torch
        mid = MODELS.get(model, model)
        self.processor = Sam2Processor.from_pretrained(mid)
        self.model = Sam2Model.from_pretrained(mid).to(device).eval()

    def embed(self, rgb: np.ndarray):
        # the first prompt runs the image encoder (forward with pixel_values) and keeps its image_embeddings;
        # later prompts on the same view pass those embeddings and only run the mask decoder
        inputs = self.processor(images=rgb, return_tensors='pt')
        return {'pixel_values': inputs['pixel_values'].to(self.device), 'sizes': inputs['original_sizes'], 'emb': None}

    def predict(self, emb, points, labels, box):
        torch = self.torch
        kw: dict[str, Any] = {'original_sizes': emb['sizes'], 'return_tensors': 'pt'}
        if points:
            kw['input_points'] = [[[[float(x), float(y)] for x, y in points]]]
            kw['input_labels'] = [[[int(v) for v in labels]]]
        if box is not None:
            kw['input_boxes'] = [[[float(v) for v in box]]]
        inputs = self.processor(**kw).to(self.device)
        inputs.pop('original_sizes', None)
        inputs.pop('reshaped_input_sizes', None)
        multi = box is None and len(points) == 1
        with torch.inference_mode():
            if emb['emb'] is None:
                out = self.model(pixel_values=emb['pixel_values'], **inputs, multimask_output=multi)
                emb['emb'] = out.image_embeddings
            else:
                out = self.model(**inputs, image_embeddings=emb['emb'], multimask_output=multi)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), emb['sizes'])[0]  # (objects, masks, H, W)
        scores = out.iou_scores.detach().float().cpu().numpy().reshape(-1)
        k = int(np.argmax(scores)) if multi else 0
        return np.asarray(masks[0, k], bool), float(scores[k])


class _Onnx:
    """SAM 2.1 exported to ONNX (encoder + prompt decoder), run by ONNX Runtime."""

    name = 'onnx'

    def __init__(self, model: str, device: str) -> None:
        d = _onnx_files(model)
        self.files = d
        self.enc = ort_session(d / 'encoder.onnx', device)  # CUDA / DirectML when there, else CPU (not CoreML)
        self.dec = ort_session(d / 'decoder.onnx', device)
        self.device = self.enc.get_providers()[0].replace('ExecutionProvider', '').lower()

    def embed(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        x = np.asarray(Image.fromarray(rgb).resize((1024, 1024), Image.BILINEAR), np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        e, h0, h1 = self.enc.run(None, {'image': x})
        return {'e': e, 'h0': h0, 'h1': h1, 'hw': (h, w)}

    def predict(self, emb, points, labels, box):
        h, w = emb['hw']
        pts, lab = [], []
        if box is not None:
            pts += [[box[0], box[1]], [box[2], box[3]]]
            lab += [2, 3]
        pts += [list(p) for p in points]
        lab += [int(v) for v in labels]
        c = np.asarray(pts, np.float32).reshape(1, -1, 2) * np.array([1024 / w, 1024 / h], np.float32)
        masks, iou = self.dec.run(
            None,
            {
                'image_embed': emb['e'],
                'high_res_feats_0': emb['h0'],
                'high_res_feats_1': emb['h1'],
                'point_coords': c,
                'point_labels': np.asarray(lab, np.float32).reshape(1, -1),
            },
        )
        k = int(np.argmax(iou[0]))
        low = Image.fromarray(masks[0, k].astype(np.float32))
        up = np.asarray(low.resize((w, h), Image.BILINEAR))
        return up > 0, float(iou[0, k])


def _onnx_files(model: str | None) -> Path:
    """Directory with encoder.onnx / decoder.onnx for ``model``: WISSI_SAM_ONNX_DIR; the bundled int8 tiny (no model
    asked for, or ``tiny``); else the user cache (downloaded once from this project's GitHub release, checked against
    its published SHA-256 sums)."""
    env = os.environ.get('WISSI_SAM_ONNX_DIR')
    if env:
        return Path(env)
    if model in (None, '', 'tiny', 'tiny-int8', 'bundled') and (BUNDLED / 'encoder.onnx').exists():
        return BUNDLED
    model = 'tiny' if model == 'tiny-fp32' else model
    d = user_dir('cache') / 'sam' / f'sam2.1-{model}-onnx'
    if (d / 'encoder.onnx').exists() and (d / 'decoder.onnx').exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    sums = urllib.request.urlopen(f'{ONNX_RELEASE}/SHA256SUMS', timeout=60).read().decode()
    want = {ln.split()[1]: ln.split()[0] for ln in sums.splitlines() if ln.strip()}
    for part in ('encoder', 'decoder'):
        name = f'sam2.1-{model}-{part}.onnx'
        if name not in want:
            raise FileNotFoundError(f'no ONNX export of sam2.1-{model} in {ONNX_RELEASE}')
        tmp = d / f'{part}.onnx.part'
        hsh = hashlib.sha256()
        with urllib.request.urlopen(f'{ONNX_RELEASE}/{name}', timeout=600) as r, open(tmp, 'wb') as f:
            while chunk := r.read(1 << 20):
                hsh.update(chunk)
                f.write(chunk)
        if hsh.hexdigest() != want[name]:
            tmp.unlink(missing_ok=True)
            raise ValueError(f'{name}: checksum mismatch')
        os.replace(tmp, d / f'{part}.onnx')
    return d


# ------------------------------------------------------------------------------------------------ service
class SamService:
    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        device: str | None = None,
        size: int = 1024,
        cache: int = 32,
    ) -> None:
        self.want = (backend or os.environ.get('WISSI_SAM_BACKEND') or 'auto').lower()
        self.model_name = (model or os.environ.get('WISSI_SAM_MODEL') or '').lower() or None  # None: the default
        self.device_pref = (device or os.environ.get('WISSI_SAM_DEVICE') or 'auto').lower()
        self.size = size
        self._impl = None
        self._error: str | None = None
        self._lock = threading.Lock()
        self._cache: OrderedDict = OrderedDict()  # (slide key, level, x, y, w, h) → embedding
        self._cache_max = cache  # one per pane of a 32-pane grid

    # -- backend selection (lazy: loading a model takes seconds)
    def _load(self):
        if self._impl is not None:
            return self._impl
        order = {
            'auto': ['torch', 'onnx', 'classic'],
            'torch': ['torch', 'classic'],
            'onnx': ['onnx', 'classic'],
            'classic': ['classic'],
        }.get(self.want, ['classic'])
        errs = []
        for b in order:
            try:
                if b == 'torch':
                    self._impl = _Torch(self.model_name or 'small', self.device_pref)
                elif b == 'onnx':
                    self._impl = _Onnx(self.model_name, self.device_pref)
                else:
                    self._impl = _Classic()
                break
            except Exception as e:  # noqa: BLE001  (missing package, no weights, no network → next backend)
                errs.append(f'{b}: {type(e).__name__}: {e}'[:300] + certs.hint(e))
        self._error = '; '.join(errs) or None
        return self._impl

    def status(self, load: bool = False) -> dict:
        impl = self._load() if load else self._impl
        return {
            'backend': impl.name if impl else None,
            'requested': self.want,
            'model': _model_label(impl, self.model_name),
            'device': getattr(impl, 'device', None),
            'loaded': impl is not None,
            'notes': self._error,
            'available': _available(),
        }

    # -- segmentation
    def segment(
        self,
        source,
        rect: Sequence[float],
        points: Sequence[Sequence[float]] = (),
        labels: Sequence[int] = (),
        box: Sequence[float] | None = None,
        key: str = '',
        min_area_px: float = 12.0,
    ) -> tuple[list, dict]:
        """Prompts in level-0 px. Returns ([polygon (n, 2) level-0 px, …], info)."""
        t0 = time.perf_counter()
        with self._lock:
            impl = self._load()
            lv, x0, y0, w, h, ds = self._region(source, rect)
            ck = (key or id(source), lv, x0, y0, w, h)
            emb = self._cache.get(ck)
            cached = emb is not None
            if emb is None:
                rgb = _rgb(source, lv, x0, y0, w, h)
                f = max(rgb.shape[:2]) / self.size
                if f > 1.05:  # level step coarser than needed: shrink to the model size
                    rgb = np.asarray(
                        Image.fromarray(rgb).resize((max(1, round(w / f)), max(1, round(h / f))), Image.LANCZOS)
                    )
                emb = (impl.embed(rgb), w / rgb.shape[1])
                self._cache[ck] = emb
                while len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(ck)
            t1 = time.perf_counter()
            emb, shrink = emb
            px = ds * shrink  # level-0 px per image px
            loc = lambda x, y: ((x - x0 * ds) / px, (y - y0 * ds) / px)
            pts = [loc(*p) for p in points]
            bx = None if box is None else (*loc(box[0], box[1]), *loc(box[2], box[3]))
            mask, score = impl.predict(emb, pts, list(labels), bx)
            t2 = time.perf_counter()
        polys = _polygons(mask, min_area_px)
        out = [np.stack([p[:, 0] * px + x0 * ds, p[:, 1] * px + y0 * ds], 1) for p in polys]
        ds = px
        return out, {
            'backend': impl.name,
            'device': getattr(impl, 'device', None),
            'score': score,
            'level': lv,
            'downsample': ds,
            'encode_ms': None if cached else round((t1 - t0) * 1000, 1),
            'decode_ms': round((t2 - t1) * 1000, 1),
            'cached': cached,
        }

    def _region(self, source, rect):
        """The pyramid level where the rect is ≤ size px on its long side, and the rect in that level's px."""
        x0, y0, x1, y1 = [float(v) for v in rect]
        info = source.info
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(info.width), x1), min(float(info.height), y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError('empty region')
        want = max(x1 - x0, y1 - y0) / self.size
        lv = 0
        for i, L in enumerate(info.levels):
            if L.downsample <= want * 1.0001:
                lv = i
        L = info.levels[lv]
        ds = L.downsample
        lx0, ly0 = int(math.floor(x0 / ds)), int(math.floor(y0 / ds))
        w = int(min(L.width, math.ceil(x1 / ds)) - lx0)
        h = int(min(L.height, math.ceil(y1 / ds)) - ly0)
        return lv, lx0, ly0, max(1, w), max(1, h), ds


def _rgb(source, lv: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    info = source.info
    if info.mode == 'rgb':
        return np.ascontiguousarray(np.asarray(source.read_region(lv, x, y, w, h))[..., :3])
    # fluorescence: up to three channels as R, G, B (each stretched to its 1–99.5 percentile)
    chans = []
    for c in range(min(3, len(info.channels))):
        a = np.asarray(source.read_region(lv, x, y, w, h, c), np.float32)
        lo, hi = np.percentile(a, [1, 99.5])
        chans.append(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1))
    while len(chans) < 3:
        chans.append(chans[0] if len(chans) == 1 else np.zeros_like(chans[0]))
    return (np.stack(chans, -1) * 255).astype(np.uint8)


def _polygons(mask: np.ndarray, min_area_px: float) -> list[np.ndarray]:
    """Mask → outer contours of its components (holes filled), simplified to ~0.75 px."""
    lab, n = ndi.label(mask)
    out = []
    for k in range(1, n + 1):
        comp = ndi.binary_fill_holes(lab == k)
        if comp.sum() < min_area_px:
            continue
        cs = find_contours(np.pad(comp, 1).astype(np.float32), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)[:, ::-1] - 1  # (row, col) → (x, y)
        try:
            import shapely

            g = shapely.Polygon(c).simplify(0.75, preserve_topology=True)
            if g.geom_type == 'Polygon' and not g.is_empty:
                c = np.asarray(g.exterior.coords)[:-1]
        except Exception:  # noqa: BLE001  (simplification is cosmetic: keep the raw contour)
            pass
        if len(c) >= 3:
            out.append(c)
    out.sort(key=lambda p: -len(p))
    return out


def _model_label(impl, model: str | None) -> str | None:
    if impl is None or impl.name == 'classic':
        return model
    if impl.name == 'onnx' and Path(getattr(impl, 'files', '')) == BUNDLED:
        return 'sam2.1-tiny (int8, bundled)'
    return f'sam2.1-{model or ("small" if impl.name == "torch" else "tiny")}'


def _available() -> list[str]:
    av = ['classic']
    if importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'):
        av.insert(0, 'torch')
    if importlib.util.find_spec('onnxruntime'):
        av.insert(len(av) - 1, 'onnx')
    return av


_SERVICE: SamService | None = None


def service() -> SamService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = SamService()
    return _SERVICE
