"""SAM 3 (Meta): name a concept ("nucleus", "blood vessel") and get every instance in a picture, outlined. Opt-in.

SAM 3 is not under a permissive licence: Meta releases it under the **SAM License** (free to use, change and share,
with conditions: a copy of the licence with anything you share, acknowledgement in publications, no military,
weapons, nuclear or espionage use, trade-control compliance). So wissi-viz never ships it: the app shows the licence
(``models/SAM-License.txt``) and downloads the model only after you accept it (:func:`accept`, stored in the user
config folder). Without accepting, "find by text" uses Grounding DINO + SAM 2.1 (Apache-2.0).

The model comes straight from the public ONNX export on Hugging Face (``nrl-ai/anylearning-sam3-onnx``,
``sam3_vit_h.zip``, ~3.3 GB, made from Meta's checkpoint with samexporter, MIT), checked against its SHA-256: wissi-viz
does not host or redistribute it. With the ``onnx`` package installed (``[sam3]`` extra, in ``[all]``), a faster int8
copy is made once on this computer (matrix multiplications quantized; the same instances and masks as float32 on the
check pictures) and the float32 files are removed. ``WISSI_SAM3_DIR`` points at an export of your own instead (the
three ``sam3_*.onnx`` files from ``samexporter.export_sam3``, e.g. made by ``scripts/export_sam3_onnx.py``).

Interface (samexporter's): image encoder ``image`` uint8 (3, 1008, 1008) RGB → vision_pos_enc_0‥2, backbone_fpn_0‥2;
language encoder ``tokens`` int64 (1, 32) CLIP BPE → text_attention_mask, text_memory, text_embeds; decoder (those +
original_height/width + box prompts) → pred_boxes (cx, cy, w, h; 0–1), pred_logits, pred_masks, presence_logit_dec.
"""

from __future__ import annotations

import datetime as _dt
import gc
import gzip
import html
import json
import os
import re
import shutil
import threading
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import weights
from ._compat import user_dir

SIZE = 1008
CONTEXT = 32
LICENSE_FILE = Path(__file__).resolve().parent / 'models' / 'SAM-License.txt'
LICENSE_NAME = 'SAM License'
LICENSE_URL = 'https://github.com/facebookresearch/sam3/blob/main/LICENSE'
#: the public float32 export (made from facebook/sam3 with samexporter); a changed file is refused
HF_REPO, HF_FILE = 'nrl-ai/anylearning-sam3-onnx', 'sam3_vit_h.zip'
HF_URL = f'https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}'
HF_SHA256 = '1617436f9db7eb9dd69029c2c465d54b316c36c21143f0fde0d1be513f15b7d5'
VOCAB = Path(__file__).resolve().parent / 'models' / 'bpe_simple_vocab_16e6.txt.gz'  # OpenAI CLIP (MIT)
PARTS = {'image_encoder': 'image-encoder', 'language_encoder': 'language-encoder', 'decoder': 'decoder'}
GRAPHS = {  # part → file names it may have: this release's, the export script's links, samexporter's own output
    'image_encoder': ('sam3-image-encoder.onnx', 'image_encoder.onnx', 'sam3_image_encoder.onnx'),
    'language_encoder': ('sam3-language-encoder.onnx', 'language_encoder.onnx', 'sam3_language_encoder.onnx'),
    'decoder': ('sam3-decoder.onnx', 'decoder.onnx', 'sam3_decoder.onnx'),
}


# ================================================================================================ licence
def _consent_file() -> Path:
    return user_dir('config') / 'licenses.json'


def accepted() -> bool:
    """Whether the SAM License was accepted on this computer (or ``WISSI_SAM3_ACCEPT_LICENSE=1``)."""
    if os.environ.get('WISSI_SAM3_ACCEPT_LICENSE', '').strip().lower() in ('1', 'true', 'yes'):
        return True
    try:
        return bool(json.loads(_consent_file().read_text(encoding='utf-8')).get('sam3', {}).get('accepted'))
    except (OSError, ValueError):
        return False


def accept(yes: bool = True) -> dict:
    """Record (or withdraw) acceptance of the SAM License for SAM 3."""
    p = _consent_file()
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        d = {}
    if yes:
        d['sam3'] = {'accepted': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
                     'license': LICENSE_NAME, 'updated': 'November 19, 2025'}  # fmt: skip
    else:
        d.pop('sam3', None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding='utf-8')
    return status()


def license_text() -> str:
    return LICENSE_FILE.read_text(encoding='utf-8')


# ================================================================================================ CLIP tokenizer
@lru_cache
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ClipBPE:
    """CLIP's byte-level BPE tokenizer (what SAM 3's text encoder reads), from ``bpe_simple_vocab_16e6.txt.gz``."""

    # CLIP splits letters runs, single digits and other non-space runs; [^\W\d_] is "a letter" for the re module
    PAT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+", re.I)

    def __init__(self, merges: Path | list[str]) -> None:
        if isinstance(merges, list):
            lines = merges
        else:
            with gzip.open(merges, 'rt', encoding='utf-8') as f:
                lines = f.read().splitlines()[1 : 49152 - 256 - 2 + 1]
        pairs = [tuple(m.split()) for m in lines if len(m.split()) == 2]
        b2u = _bytes_to_unicode()
        vocab = list(b2u.values())
        vocab += [v + '</w>' for v in vocab]
        vocab += [''.join(p) for p in pairs]
        vocab += ['<|startoftext|>', '<|endoftext|>']
        self.enc = {v: i for i, v in enumerate(vocab)}
        self.ranks = {p: i for i, p in enumerate(pairs)}
        self.b2u = b2u
        self.sot, self.eot = self.enc['<|startoftext|>'], self.enc['<|endoftext|>']
        self._cache: dict[str, str] = {}

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        s = ' '.join(word)
        self._cache[token] = s
        return s

    def encode(self, text: str) -> list[int]:
        text = html.unescape(html.unescape(text))
        text = re.sub(r'\s+', ' ', text).strip().lower()
        ids = []
        for tok in self.PAT.findall(text):
            t = ''.join(self.b2u[b] for b in tok.encode('utf-8'))
            ids += [self.enc[p] for p in self._bpe(t).split(' ')]
        return ids

    def tokens(self, text: str, context: int = CONTEXT) -> np.ndarray:
        ids = [self.sot, *self.encode(text)][: context - 1] + [self.eot]
        out = np.zeros((1, context), np.int64)
        out[0, : len(ids)] = ids
        return out


# ================================================================================================ outputs
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def postprocess(raw: dict, size: tuple[int, int], min_score: float, max_n: int = 300):
    """Decoder heads → (boxes xyxy px, scores, masks (n, h, w) bool) above ``min_score``, best first. The score is
    the query's probability times the presence probability (as Meta's ``Sam3Processor``); masks are resized only for
    the queries kept. Older exports (the decoder graph holds ``Sam3Processor``'s post-processing, outputs
    ``boxes, scores, masks``, already filtered at 0.5) are read as they are."""
    if 'pred_logits' not in raw and 'scores' in raw:
        return _processed(raw, size, min_score, max_n)
    logits = np.asarray(raw['pred_logits'], np.float32).reshape(-1)
    presence = float(_sigmoid(np.asarray(raw['presence_logit_dec'], np.float32).reshape(-1)[0]))
    scores = _sigmoid(logits) * presence
    keep = np.nonzero(scores > min_score)[0]
    keep = keep[np.argsort(-scores[keep], kind='stable')][:max_n]
    h, w = size
    b = np.asarray(raw['pred_boxes'], np.float32).reshape(-1, 4)[keep]
    boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                      (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)  # fmt: skip
    pm = np.asarray(raw['pred_masks'], np.float32)
    pm = pm.reshape(-1, *pm.shape[-2:])[keep]
    masks = np.zeros((len(keep), h, w), bool)
    for i, m in enumerate(pm):
        masks[i] = np.asarray(Image.fromarray(m).resize((w, h), Image.BILINEAR)) > 0  # sigmoid > 0.5
    return boxes.astype(np.float32), scores[keep].astype(np.float32), masks


def _processed(raw: dict, size: tuple[int, int], min_score: float, max_n: int):
    """Outputs of a decoder with the processor inside: boxes xyxy px, scores, masks (n, 1, h, w) at the input size."""
    h, w = size
    scores = np.asarray(raw['scores'], np.float32).reshape(-1)
    boxes = np.asarray(raw['boxes'], np.float32).reshape(-1, 4)
    pm = np.asarray(raw['masks'])
    pm = pm.reshape(-1, *pm.shape[-2:]) if pm.size else np.zeros((0, h, w), bool)
    keep = np.nonzero(scores > min_score)[0]
    keep = keep[np.argsort(-scores[keep], kind='stable')][:max_n]
    masks = np.zeros((len(keep), h, w), bool)
    for i, k in enumerate(keep):
        m = pm[k] if pm.dtype == np.bool_ else pm[k] > 0.5
        if m.shape != (h, w):
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127
        masks[i] = m
    return boxes[keep], scores[keep], masks


# ================================================================================================ model
class Sam3:
    name = 'sam3'
    label = 'SAM 3'
    tile = SIZE
    masks = True  # outlines come with the detections (no SAM 2.1 pass)

    def __init__(self, folder: Path, device: str | None = None) -> None:
        self.folder = Path(folder)

        def path(part):
            for n in GRAPHS[part]:
                if (self.folder / n).exists():
                    return self.folder / n
            raise FileNotFoundError(f'no SAM 3 {part.replace("_", " ")} in {self.folder} ({" or ".join(GRAPHS[part])})')

        self.enc = weights.ort_session(path('image_encoder'), device)
        self.lang = weights.ort_session(path('language_encoder'), device)
        self.dec = weights.ort_session(path('decoder'), device)
        self.dec_in = {i.name: i for i in self.dec.get_inputs()}
        self.dec_out = [o.name for o in self.dec.get_outputs()]
        bc = self.dec_in.get('box_coords')
        cap = bc.shape[0] if bc is not None else None
        self.capacity = cap if isinstance(cap, int) and cap > 0 else 1
        self.enc_in = self.enc.get_inputs()[0]
        vocab = self.folder / 'bpe_simple_vocab_16e6.txt.gz'
        self.tok = ClipBPE(vocab if vocab.exists() else VOCAB)
        self.device = self.enc.get_providers()[0].replace('ExecutionProvider', '').lower()
        self._text: dict[str, list[np.ndarray]] = {}
        self._lock = threading.Lock()

    def text(self, phrase: str) -> list[np.ndarray]:
        if phrase not in self._text:
            self._text[phrase] = self.lang.run(None, {'tokens': self.tok.tokens(phrase)})
        return self._text[phrase]

    def image(self, rgb: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., :3])).resize((SIZE, SIZE), Image.BILINEAR))
        x = x.transpose(2, 0, 1)
        if self.enc_in.type == 'tensor(float)':
            x = ((x / 255.0 - 0.5) / 0.5).astype(np.float32)
        else:
            x = x.astype(np.uint8)
        return self.enc.run(None, {self.enc_in.name: np.ascontiguousarray(x)})

    def decode(self, img: list[np.ndarray], txt: list[np.ndarray], size: tuple[int, int]) -> dict:
        h, w = size
        n = self.capacity
        feed = {
            'original_height': np.array(h, np.int64),
            'original_width': np.array(w, np.int64),
            **{f'vision_pos_enc_{i}': img[i] for i in range(3)},
            **{f'backbone_fpn_{i}': img[3 + i] for i in range(3)},
            'language_mask': txt[0],
            'language_features': txt[1],
            'language_embeds': txt[2],
            'box_coords': np.zeros((n, 1, 4), np.float32),
            'box_labels': np.ones((n, 1), np.int64),
            'box_masks': np.ones((1, n), np.bool_),  # every geometry slot is padding: text only
        }
        out = self.dec.run(None, {k: v for k, v in feed.items() if k in self.dec_in})
        return dict(zip(self.dec_out, out))

    def detect(self, rgb: np.ndarray, names: list[str], min_score: float = 0.15):
        """Boxes (xyxy px), scores, phrase indices and masks of every instance of each of ``names`` in ``rgb``."""
        h, w = rgb.shape[:2]
        B, S, L, M = [], [], [], []
        with self._lock:
            img = self.image(rgb)
            for k, n in enumerate(names):
                b, s, m = postprocess(self.decode(img, self.text(n), (h, w)), (h, w), min_score)
                B.append(b)
                S.append(s)
                L.append(np.full(len(s), k, np.int64))
                M.append(m)
        if not B:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32),
                np.zeros(0, np.int64),
                np.zeros((0, h, w), bool),
            )
        return np.concatenate(B), np.concatenate(S), np.concatenate(L), np.concatenate(M)


def folder() -> Path | None:
    env = os.environ.get('WISSI_SAM3_DIR')
    if env:
        return Path(env)
    d = weights.cache_dir('sam3')
    return d if (d / '.complete').exists() else None


def _graph_bytes(graph: Path) -> int:
    """A graph's size with the external-data files it refers to."""
    import onnx
    from onnx.external_data_helper import uses_external_data

    m = onnx.load(str(graph), load_external_data=False)
    locs = {
        e.value for t in m.graph.initializer if uses_external_data(t) for e in t.external_data if e.key == 'location'
    }
    return graph.stat().st_size + sum((graph.parent / loc).stat().st_size for loc in locs)


def quantize(src: Path, out: Path, parts=tuple(PARTS), progress=None) -> dict:
    """Write ``out/sam3-<part>.onnx`` (+ ``.data`` when large) from the float32 ``src/sam3_<part>.onnx``: the matrix
    multiplications of ``parts`` as int8 (ONNX Runtime dynamic quantization), the others as they are. Needs the
    ``onnx`` package. Returns {part: precision}."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.mkdir(parents=True, exist_ok=True)
    prec = {}
    for k, (local, name) in enumerate(PARTS.items()):
        if progress:
            progress(k / len(PARTS), f'making a faster int8 copy of SAM 3 (once): {local.replace("_", " ")}')
        s, dst = src / f'sam3_{local}.onnx', out / f'sam3-{name}.onnx'
        for p in out.glob(f'sam3-{name}.onnx*'):
            p.unlink()
        big = _graph_bytes(s) > 1.5 * 2**30
        if local in parts:
            quantize_dynamic(str(s), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'],
                             use_external_data_format=big)  # fmt: skip
        m = onnx.load(str(dst if local in parts else s))
        # the IR version of the source: a newer onnx package may write one the installed ONNX Runtime cannot read
        m.ir_version = min(m.ir_version, onnx.load(str(s), load_external_data=False).ir_version)
        for p in out.glob(f'sam3-{name}.onnx?*'):  # the quantizer's own external-data names
            p.unlink()
        if big:  # the weights next to the graph under a stable name
            onnx.save_model(m, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                            location=f'sam3-{name}.onnx.data', size_threshold=1024)  # fmt: skip
        else:
            onnx.save_model(m, str(dst))
        prec[local] = 'int8' if local in parts else 'float32'
        del m
        gc.collect()
    return prec


def download(progress=None) -> Path:
    """Fetch the public export from Hugging Face (resumable, SHA-256 checked), unpack it into the cache and, if the
    ``onnx`` package is installed, replace it with an int8 copy. Returns the model folder."""
    d = weights.cache_dir('sam3')
    d.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda *a: None)
    z = weights.download(HF_URL, d / HF_FILE, HF_SHA256, lambda f, m: say(0.9 * f, m), 'SAM 3 from Hugging Face')
    say(0.9, 'unpacking SAM 3…')
    fp32 = d / 'float32'
    fp32.mkdir(exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name  # flat: the graphs name their external-data files without folders
            if info.is_dir() or not base or base.startswith('.'):
                continue
            with zf.open(info) as r, open(fp32 / base, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 24)
    missing = [p for p in PARTS if not (fp32 / f'sam3_{p}.onnx').exists()]
    if missing:
        raise FileNotFoundError(f'{HF_FILE} from {HF_REPO} has no sam3_{"/".join(missing)}.onnx: it changed')
    z.unlink()
    note = ''
    try:
        quantize(fp32, d, progress=lambda f, m: say(0.92 + 0.07 * f, m))
        shutil.rmtree(fp32)
        note = 'int8 (quantized on this computer)'
    except ImportError:  # no onnx package: use the float32 graphs (slower)
        for p in fp32.iterdir():
            os.replace(p, d / p.name)
        fp32.rmdir()
        note = 'float32 (install the onnx package for a faster int8 copy)'
    shutil.copy(LICENSE_FILE, d / 'LICENSE-SAM.txt')
    (d / 'NOTICE.txt').write_text(
        'SAM 3 by Meta Platforms, Inc., under the SAM License (LICENSE-SAM.txt), which applies to these files.\n'
        f'Downloaded from https://huggingface.co/{HF_REPO} ({HF_FILE}, sha256 {HF_SHA256}): an ONNX export of the\n'
        f'official checkpoint made with samexporter (MIT). Here: {note}.\n',
        encoding='utf-8',
    )
    (d / '.complete').write_text(note, encoding='utf-8')
    return d


_DET: Sam3 | None = None
_LOCK = threading.Lock()


def detector(progress=None, device: str | None = None) -> Sam3:
    """The loaded model; downloads it on first use. Raises ``PermissionError`` until the SAM License is accepted."""
    global _DET
    if not accepted():
        raise PermissionError(
            'SAM 3 is under the SAM License (Meta): accept it first (Annotate → Find all → By text → SAM 3), or use '
            'Grounding DINO + SAM 2.1'
        )
    with _LOCK:
        if _DET is None:
            d = folder() or download(progress)
            _DET = Sam3(d, device or os.environ.get('WISSI_SAM3_DEVICE'))
        return _DET


def status() -> dict:
    here = folder() is not None
    return {
        'model': 'sam3',
        'license': LICENSE_NAME,
        'license_url': LICENSE_URL,
        'accepted': accepted(),
        'downloaded': here,
        'source': f'https://huggingface.co/{HF_REPO}',
        'precision': (
            (d / '.complete').read_text(encoding='utf-8') if (d := folder()) and (d / '.complete').exists() else None
        ),  # fmt: skip
        'loaded': _DET is not None,
        'size_mb': 3300,
    }
