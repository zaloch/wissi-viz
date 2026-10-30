"""Find all against reference outlines, to look at and to measure.

    wissi-viz validate run [--cases nuinsseg,camelyon,xenium] [--yes]   # computes (long without a GPU)
    wissi-viz validate open [--yes]    # the latest published run (or yours) in the app: reference + every method

Three cases, each a slide with the reference and one layer per method:

* **NuInsSeg** (H&E nuclei outlined by hand, 31 human and mouse organs, CC BY 4.0): a mosaic slide of patches, one or
  two per organ. Methods: Grounding DINO + SAM 2.1, SAM 3, Grounding DINO → SAM 3, segment everything, the cell
  models. Score: nuclei matched one-to-one at IoU ≥ 0.5 (precision, recall, F1).
* **Rat DRG** (neurofilament immunofluorescence, neurons outlined by three experts and merged into a consensus, CC BY
  4.0): a mosaic of six fields. Methods: Grounding DINO → SAM 3, SAM 3, Grounding DINO + SAM 2.1, segment everything,
  sized for neurons (15–60 µm), so that satellite glia and Schwann cells are not counted. Score: neurons matched at
  IoU ≥ 0.5.
* **CAMELYON16 test_065** (lymph-node metastases outlined by pathologists, CC0): the region around the lesions.
  Methods: areas (a pixel classifier trained on two small boxes of each kind, as a user would brush them), SAM 3 and
  Grounding DINO → SAM 3 by name. Score: Dice of the tumour area.
* **Xenium breast** (10x nuclei from the DAPI stain, CC BY 4.0: an instrument's segmentation, not hand-drawn): a 1 mm
  field. Methods: the cell models. Score: nuclei matched at IoU ≥ 0.5.

Results land in ``<pack>/validation``: GeoParquet layers, the mosaic slide, ``validation.json`` / ``validation.md``,
side-by-side pictures and a session for the app. The ``validate`` workflow publishes them as the GitHub release
``validation``; ``open`` downloads that when nothing was computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import shapely
import tifffile
from PIL import Image, ImageDraw

from . import annio, certs, evaluate, findall, pack
from .annotations import AnnotationLayer
from .desktop import main as desktop_main
from .session import Session
from .sources import open_slide

RELEASE = 'https://github.com/zaloch/wissi-viz/releases/download/validation'
NUINSSEG_RECORD = 'https://zenodo.org/api/records/10518968'  # NuInsSeg (Mahbod et al., Sci Data 2024), CC BY 4.0
NUINSSEG_MPP = 0.25  # µm per pixel (40× objective); approximate: used for object sizes only
CAMELYON = 'camelyon16-test-065'
XENIUM = 'xenium-breast'
CASES = ('nuinsseg', 'drg', 'camelyon', 'xenium')
DRG_RECORD = 'https://zenodo.org/api/records/6546069'  # rat DRG, Schulte, Troidl et al. 2023, CC BY 4.0
DRG_NEURON_UM = 30.0  # typical rat DRG neuron diameter: sets the pixel size (the images carry none)
COLORS = {'Reference': '#3ddc84', 'Expert': '#3ddc84'}
METHOD_COLORS = ['#ff5ca8', '#5ac8fa', '#ffb454', '#c38cff', '#ffd166', '#7ee2a8', '#ff7b7b']


_PACK: str | None = None  # the sample pack's folder (--dir), for the slides the cases use


def vdir(directory: str | None = None) -> Path:
    global _PACK
    _PACK = directory
    d = pack.pack_dir(directory) / 'validation'
    (d / 'layers').mkdir(parents=True, exist_ok=True)
    return d


# ================================================================================================ data
def _get(url: str, dest: Path, log) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log(f'downloading {url}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'wissi-viz'}), timeout=120) as r:
        with open(tmp, 'wb') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    return dest


def nuinsseg_zip(d: Path, log) -> Path:
    """NuInsSeg's archive from Zenodo, after checking the record's licence (CC BY 4.0)."""
    meta = json.loads(urllib.request.urlopen(NUINSSEG_RECORD, timeout=60).read().decode())
    lic = str((meta.get('metadata') or {}).get('license', {}).get('id', '')).lower()
    if not lic.startswith('cc-by') or '-nc' in lic or '-nd' in lic:
        raise SystemExit(f'NuInsSeg licence is {lic!r} on Zenodo now, not CC BY: not used')
    files = sorted(meta.get('files') or [], key=lambda f: -int(f.get('size', 0)))
    z = next((f for f in files if str(f.get('key', '')).lower().endswith('.zip')), None)
    if z is None:
        raise SystemExit(f'no zip in the NuInsSeg record: {[f.get("key") for f in files]}')
    url = (z.get('links') or {}).get('self') or f'https://zenodo.org/records/10518968/files/{z["key"]}?download=1'
    return _get(url, d / 'downloads' / z['key'], log)


def _nuinsseg_pairs(zf: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    """(organ, image member, label-mask member): the H&E patches and their labelled nuclei masks."""
    names = [n for n in zf.namelist() if not n.endswith('/')]
    imgs = [n for n in names if re.search(r'tissue[ _-]?images?/', n, re.I) and n.lower().endswith(('.png', '.tif'))]
    masks = {}
    for n in names:
        if re.search(r'label[ _-]?masks?', n, re.I) and not re.search(r'modify|ambig|vague', n, re.I):
            masks[(Path(n).parts[0].lower() if len(Path(n).parts) > 2 else '', Path(n).stem.lower())] = n
    out = []
    for n in sorted(imgs):
        parts = Path(n).parts
        organ = parts[-3] if len(parts) >= 3 else 'tissue'
        key = (parts[0].lower() if len(parts) > 2 else '', Path(n).stem.lower())
        m = masks.get(key) or next((v for (o, s), v in masks.items() if s == key[1]), None)
        if m:
            out.append((organ, n, m))
    if not out:
        raise SystemExit(f'NuInsSeg: no image / label mask pairs found (first entries: {names[:12]})')
    return out


def _label_polygons(lab: np.ndarray, x0: float, y0: float) -> list[np.ndarray]:
    from skimage.measure import find_contours, regionprops

    out = []
    for rp in regionprops(lab):
        r0, c0, _, _ = rp.bbox
        cs = find_contours(np.pad(rp.image, 1).astype(np.float32), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        if len(c) < 4:
            continue
        out.append(np.stack([c[:, 1] - 1 + c0 + x0, c[:, 0] - 1 + r0 + y0], 1))
    return out


def nuinsseg_mosaic(d: Path, per_organ: int = 1, limit: int = 16, log=print) -> tuple[Path, Path, Path]:
    """A mosaic slide of NuInsSeg patches (``per_organ`` from each organ, at most ``limit``), the hand-drawn nuclei on
    it and the patches (organ names) as layers."""
    out = d / 'nuinsseg-mosaic.ome.tif'
    ref = d / 'layers' / 'nuinsseg-mosaic.reference.parquet'
    pat = d / 'layers' / 'nuinsseg-mosaic.patches.parquet'
    if out.exists() and ref.exists() and pat.exists():
        return out, ref, pat
    zf = zipfile.ZipFile(nuinsseg_zip(d, log))
    pairs = _nuinsseg_pairs(zf)
    chosen, seen = [], {}
    for organ, im, m in pairs:
        if seen.get(organ, 0) < per_organ:
            chosen.append((organ, im, m))
            seen[organ] = seen.get(organ, 0) + 1
    chosen = chosen[:limit]
    tiles = []
    for organ, im, m in chosen:
        rgb = np.asarray(Image.open(io.BytesIO(zf.read(im))).convert('RGB'))
        lab = np.asarray(Image.open(io.BytesIO(zf.read(m))))
        if lab.ndim == 3:
            lab = lab[..., 0]
        if lab.max() <= 1:  # a binary mask: label its connected parts
            from scipy import ndimage as ndi

            lab = ndi.label(lab > 0)[0]
        tiles.append((organ, rgb, lab.astype(np.int32)))
    P = max(max(t[1].shape[:2]) for t in tiles)
    gap = 48
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    H, W = rows * (P + gap) + gap, cols * (P + gap) + gap
    canvas = np.full((H, W, 3), 246, np.uint8)
    nuclei, boxes, organs = [], [], []
    for k, (organ, rgb, lab) in enumerate(tiles):
        y0, x0 = gap + (k // cols) * (P + gap), gap + (k % cols) * (P + gap)
        h, w = rgb.shape[:2]
        canvas[y0 : y0 + h, x0 : x0 + w] = rgb
        nuclei += _label_polygons(lab, x0, y0)
        boxes.append(np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float))
        organs.append(organ)
    _write_rgb(out, canvas, NUINSSEG_MPP)
    L = AnnotationLayer(name='Reference · NuInsSeg (hand-drawn)')
    for p in nuclei:
        L.add_polygon(p, cls='Nucleus (expert)', color=COLORS['Reference'])
    annio.export(L, ref, slide=out.name)
    Pl = AnnotationLayer(name='NuInsSeg patches')
    for b, organ in zip(boxes, organs):
        Pl.add_polygon(b, cls=organ, props={'organ': organ})
    Pl.fill_opacity = 0.0
    annio.export(Pl, pat, slide=out.name)
    (d / 'NUINSSEG-ATTRIBUTION.md').write_text(
        'nuinsseg-mosaic.ome.tif and nuinsseg-mosaic.reference.parquet are made from NuInsSeg (A. Mahbod et al., '
        '"NuInsSeg: A fully annotated dataset for nuclei instance segmentation in H&E-stained histological images", '
        'Scientific Data 11, 295 (2024), https://doi.org/10.5281/zenodo.10518968), licensed CC BY 4.0. Patches were '
        'arranged into a mosaic and the label masks converted to outlines.\n',
        encoding='utf-8',
    )
    log(f'NuInsSeg: {len(tiles)} patches from {len(set(organs))} organs, {len(nuclei):,} hand-drawn nuclei')
    return out, ref, pat


def _write_rgb(path: Path, rgb: np.ndarray, mpp: float) -> None:
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tw:
        opts = {'tile': (256, 256), 'compression': 'zlib', 'photometric': 'rgb'}
        nlev = max(1, int(np.ceil(np.log2(max(rgb.shape[:2]) / 512))) + 1)
        tw.write(rgb, subifds=nlev - 1, metadata={'axes': 'YXS', 'PhysicalSizeX': mpp, 'PhysicalSizeY': mpp}, **opts)
        lv = rgb
        for _ in range(nlev - 1):
            lv = lv[::2, ::2]
            tw.write(np.ascontiguousarray(lv), subfiletype=1, **opts)


def _zenodo_file(record: str, key: str, d: Path, log) -> Path:
    """One file of a Zenodo record, after checking the record's licence (CC BY)."""
    meta = json.loads(urllib.request.urlopen(record, timeout=60).read().decode())
    lic = str((meta.get('metadata') or {}).get('license', {}).get('id', '')).lower()
    if not lic.startswith('cc-by') or '-nc' in lic or '-nd' in lic:
        raise SystemExit(f'{record}: licence is {lic!r}, not CC BY: not used')
    f = next((f for f in meta.get('files') or [] if f.get('key') == key), None)
    if f is None:
        raise SystemExit(f'{key} is not in {record}')
    return _get(f['links']['self'], d / 'downloads' / key, log)


def _unrar(archive: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for tool in ('7z', '7zz', 'bsdtar', 'unrar'):
        exe = shutil.which(tool)
        if not exe:
            continue
        cmd = {'bsdtar': [exe, '-xf', str(archive), '-C', str(out)],
               'unrar': [exe, 'x', '-o+', str(archive), str(out) + os.sep]}.get(tool,
              [exe, 'x', '-y', f'-o{out}', str(archive)])  # fmt: skip
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            return
    raise SystemExit('the DRG annotations are a RAR archive: install 7-Zip (7z) or bsdtar to unpack them')


def _instances(mask: np.ndarray) -> np.ndarray:
    """A mask → instance labels: labelled already, else its connected parts (touching cells split by distance)."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    m = mask[..., 0] if mask.ndim == 3 else mask
    vals = np.unique(m)
    if len(vals) > 3:  # one value per cell
        return m.astype(np.int32)
    fg = m > 0
    dist = ndi.distance_transform_edt(fg)
    lab = ndi.label(fg)[0]
    peaks = peak_local_max(dist, min_distance=12, threshold_abs=4, labels=lab, exclude_border=False)
    markers = np.zeros(m.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=fg).astype(np.int32)


def drg_mosaic(d: Path, n: int = 6, log=print) -> tuple[Path, Path, float]:
    """A mosaic of ``n`` rat DRG neurofilament images (one fluorescence channel) and the experts' consensus neuron
    outlines on it. Returns (slide, reference, µm per pixel: estimated from the neurons' size, see DRG_NEURON_UM)."""
    out = d / 'drg-mosaic.ome.tif'
    ref = d / 'layers' / 'drg-mosaic.reference.parquet'
    meta = d / 'drg-mosaic.json'
    if out.exists() and ref.exists() and meta.exists():
        return out, ref, json.loads(meta.read_text())['mpp']
    root = d / 'drg'
    if not (root / 'Annotations_ratDRG' / 'NF').exists():
        _unrar(_zenodo_file(DRG_RECORD, 'Annotations_ratDRG.rar', d, log), root)
    nf = root / 'Annotations_ratDRG' / 'NF'
    imgs = sorted((nf / 'images').glob('*.tif'))[:n]
    tiles = []
    for p in imgs:
        a = tifffile.imread(p)
        a = a[0] if a.ndim == 3 else a
        mp = nf / 'masks' / 'GT_Estimation' / f'{p.stem}.png'
        lab = _instances(np.asarray(Image.open(mp)))
        tiles.append((p.stem, a.astype(np.uint16), lab))
    if not tiles:
        raise SystemExit(f'no NF images in {nf}')
    areas = np.concatenate([np.bincount(t[2].ravel())[1:] for t in tiles])
    areas = areas[areas > 50]
    px = float(np.median(2 * np.sqrt(areas / math.pi))) if len(areas) else 60.0
    mpp = round(DRG_NEURON_UM / px, 3)
    H = max(t[1].shape[0] for t in tiles)
    W = max(t[1].shape[1] for t in tiles)
    gap = 64
    cols = 3 if len(tiles) > 4 else 2
    rows = math.ceil(len(tiles) / cols)
    canvas = np.zeros((rows * (H + gap) + gap, cols * (W + gap) + gap), np.uint16)
    neurons, boxes = [], []
    for k, (stem, a, lab) in enumerate(tiles):
        y0, x0 = gap + (k // cols) * (H + gap), gap + (k % cols) * (W + gap)
        canvas[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = a
        neurons += _label_polygons(lab, x0, y0)
        boxes.append((stem, x0, y0, a.shape[1], a.shape[0]))
    with tifffile.TiffWriter(out, bigtiff=True, ome=True) as tw:
        opts = {'tile': (512, 512), 'compression': 'zlib', 'photometric': 'minisblack'}
        data = canvas[None]
        nlev = max(1, int(np.ceil(np.log2(max(canvas.shape) / 1024))) + 1)
        tw.write(data, subifds=nlev - 1, metadata={'axes': 'CYX', 'Channel': {'Name': ['Neurofilament']},
                                                     'PhysicalSizeX': mpp, 'PhysicalSizeY': mpp}, **opts)  # fmt: skip
        lv = data
        for _ in range(nlev - 1):
            lv = lv[:, ::2, ::2]
            tw.write(np.ascontiguousarray(lv), subfiletype=1, **opts)
    L = AnnotationLayer(name='Reference · rat DRG neurons (3 experts, consensus)')
    for q in neurons:
        L.add_polygon(q, cls='Neuron (expert)', color=COLORS['Reference'])
    annio.export(L, ref, slide=out.name)
    meta.write_text(json.dumps({'mpp': mpp, 'images': [b[0] for b in boxes], 'neuron_px': px}), encoding='utf-8')
    (d / 'DRG-ATTRIBUTION.md').write_text(
        'drg-mosaic.ome.tif and drg-mosaic.reference.parquet are made from "Data from: Unbiased analysis of cellular '
        'plasticity in the dorsal root ganglion after peripheral nerve injury" (Zenodo, '
        'https://doi.org/10.5281/zenodo.6546069), licensed CC BY 4.0: neurofilament images of rat DRG and the '
        "consensus (GT_Estimation) of three experts' neuron outlines. Images were arranged into a mosaic and the "
        f'masks converted to outlines. The pixel size ({mpp} µm) is estimated from the neurons (median diameter taken '
        f'as {DRG_NEURON_UM:g} µm); the files carry none.\n',
        encoding='utf-8',
    )
    log(f'DRG: {len(tiles)} neurofilament images, {len(neurons):,} expert neurons, ~{mpp} µm/px (estimated)')
    return out, ref, mpp


DRG_METHODS = {
    'dino-sam3': ('Grounding DINO → SAM 3', {'mode': 'text', 'model': 'dino-sam3', 'sam3_name': 'neuron',
                                             'prompt': 'large round neuron cell bodies'}),
    'sam3': ('SAM 3', {'mode': 'text', 'model': 'sam3', 'prompt': 'neuron'}),
    'grounding': ('Grounding DINO + SAM 2.1', {'mode': 'text', 'model': 'grounding', 'prompt': 'neuron'}),
    'segment': ('Segment everything', {'mode': 'segment'}),
}  # fmt: skip


def run_drg(session, d: Path, methods: list[str], log) -> dict:
    slide, ref, mpp = drg_mosaic(d, log=log)
    sid = session.open([slide, ref], trusted=True)['opened'][0]
    reg = session.registry
    R = annio.read_geoparquet(ref)
    out = {'reference': len(R), 'mpp_estimated': mpp, 'methods': {}}
    for k, key in enumerate(m for m in methods if m in DRG_METHODS):
        label, body = DRG_METHODS[key]

        def go(body=body, label=label, k=k, key=key):
            b = {'scope': {'kind': 'slide'}, 'strictness': 0.4, 'object_um': 15, 'object_um_max': 60, **body}
            r = findall.run(reg, sid, b, None)
            p = _export(reg, sid, r['layer'], d / 'layers' / f'drg-mosaic.{key}.parquet', f'{label} · neurons',
                        METHOD_COLORS[k % len(METHOD_COLORS)])  # fmt: skip
            return evaluate.match(annio.read_geoparquet(p), R)

        _try(out['methods'], key, go, log)
    return out


def xenium_nuclei(d: Path, log) -> Path:
    """10x's nucleus outlines (from DAPI) of the Xenium breast bundle as a layer on the pack's slide."""
    out = d / 'layers' / f'{XENIUM}.nuclei-10x.parquet'
    if out.exists():
        return out
    x = next(i for i in pack.manifest()['items'] if i['id'] == XENIUM)
    src = pack.pack_dir(_PACK) / 'downloads' / x['file']
    with zipfile.ZipFile(src) as z:
        member = next(n for n in z.namelist() if n.endswith('nucleus_boundaries.parquet'))
        tmp = d / 'downloads' / 'xenium-nucleus_boundaries.parquet'
        tmp.write_bytes(z.read(member))
    slide = pack._slide(x, pack.pack_dir(_PACK))
    with tifffile.TiffFile(slide) as tf:
        m = re.search(r'PhysicalSizeX="([0-9.]+)"', tf.ome_metadata or '')
    mpp = float(m.group(1)) if m else 0.2125
    pack._xenium_cells(tmp, out, mpp, slide.name)
    F = annio.read_geoparquet(out)
    F.class_names = ['Nucleus (10x)']
    F.name = 'Reference · 10x nuclei (DAPI)'
    annio.write_geoparquet(F, out, slide=slide.name)
    log(f'Xenium: {len(F):,} 10x nuclei')
    return out


# ================================================================================================ running
def _export(
    reg, sid: str, layer: str, path: Path, name: str, color: str, keep_classes: list[str] | None = None
) -> Path:
    L = reg.layer(sid, layer)
    F = annio.from_layer(L)
    if keep_classes:
        F = annio.subset(F, np.nonzero(np.isin(F.class_of(), keep_classes))[0])
    F.class_names = [name.split(' · ')[0]] if not keep_classes else list(F.class_names)
    if not keep_classes:
        F.cls = np.zeros(len(F), np.int32)
    F.class_colors = dict.fromkeys(F.class_names, color)
    F.name = name
    annio.write_geoparquet(F, path, slide=None)
    return path


def _try(results: dict, key: str, fn, log) -> None:
    t0 = time.perf_counter()
    try:
        results[key] = fn() | {'seconds': round(time.perf_counter() - t0, 1)}
        log(f'  {key}: {results[key]}')
    except Exception as e:  # noqa: BLE001  (a method that cannot run here: reported, the others go on)
        results[key] = {'error': f'{e.__class__.__name__}: {e}'[:300], 'seconds': round(time.perf_counter() - t0, 1)}
        log(f'  {key}: failed: {results[key]["error"]}')


NUCLEUS_METHODS = {
    'grounding': ('Grounding DINO + SAM 2.1', {'mode': 'text', 'model': 'grounding', 'prompt': 'nucleus',
                                               'expand': True}),
    'sam3': ('SAM 3', {'mode': 'text', 'model': 'sam3', 'prompt': 'nucleus'}),
    'dino-sam3': ('Grounding DINO → SAM 3', {'mode': 'text', 'model': 'dino-sam3', 'sam3_name': 'nucleus',
                                             'prompt': 'cell nuclei stained purple with hematoxylin'}),
    'segment': ('Segment everything', {'mode': 'segment'}),
    'classic': ('Classic threshold', {'mode': 'cells', 'model': 'classic'}),
    'cellpose': ('Cellpose-SAM', {'mode': 'cells', 'model': 'cellpose'}),
}  # fmt: skip


def run_nuinsseg(session, d: Path, methods: list[str], log) -> dict:
    slide, ref, pat = nuinsseg_mosaic(d, log=log)
    sid = session.open([slide, ref, pat], trusted=True)['opened'][0]
    reg = session.registry
    R = annio.read_geoparquet(ref)
    out = {'reference': len(R), 'methods': {}}
    for k, key in enumerate(m for m in methods if m in NUCLEUS_METHODS):
        label, body = NUCLEUS_METHODS[key]

        def go(body=body, label=label, k=k, key=key):
            b = {'scope': {'kind': 'slide'}, 'strictness': 0.5, 'object_um': 4, 'object_um_max': 14, **body}
            if key == 'segment':
                b['object_um_max'] = None  # segment everything: the clicks' size (a range would also filter)
                b['object_um'] = 6
            r = findall.run(reg, sid, b, None)
            p = _export(reg, sid, r['layer'], d / 'layers' / f'nuinsseg-mosaic.{key}.parquet', f'{label} · nuclei',
                        METHOD_COLORS[k % len(METHOD_COLORS)])  # fmt: skip
            return evaluate.match(annio.read_geoparquet(p), R)

        _try(out['methods'], key, go, log)
    return out


def _camelyon_region(R: annio.Features, W: int, H: int, mpp: float):
    g = shapely.union_all(evaluate.polygons(R, ['Tumor']))
    if g.is_empty:
        raise SystemExit('CAMELYON16: no Tumor outlines')
    x0, y0, x1, y1 = g.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(max(x1 - x0, y1 - y0) * 1.3, 8000 / mpp)  # at most 8 mm
    side = max(side, 1500 / mpp)
    return shapely.box(max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2)), g


def run_camelyon(session, d: Path, methods: list[str], log) -> dict:
    x = next(i for i in pack.manifest()['items'] if i['id'] == CAMELYON)
    pd = pack.pack_dir(_PACK)
    slide, expert = pack._slide(x, pd), pd / 'layers' / f'{CAMELYON}.expert.parquet'
    sid = session.open([slide, expert], trusted=True)['opened'][0]
    reg = session.registry
    src = findall._source(reg, sid)
    mpp = findall._mpp(src)
    R = annio.read_geoparquet(expert)
    region, tumour = _camelyon_region(R, src.info.width, src.info.height, mpp)
    tumour = tumour.intersection(region)
    rx0, ry0, rx1, ry1 = region.bounds
    scope = {'kind': 'view', 'rect': [rx0, ry0, rx1, ry1]}
    out = {'region_mm': [round((rx1 - rx0) * mpp / 1000, 2), round((ry1 - ry0) * mpp / 1000, 2)],
           'tumour_mm2': round(tumour.area * mpp * mpp / 1e6, 3), 'methods': {}}  # fmt: skip
    # what a user would brush: two small boxes inside the largest lesion, two in the tissue around it
    big = max(getattr(tumour, 'geoms', [tumour]), key=lambda p: p.area)
    s = max(60 / mpp, min(250 / mpp, math.sqrt(big.area) / 6))
    inner = [shapely.box(px - s / 2, py - s / 2, px + s / 2, py + s / 2) for px, py in _inside_points(big, 2, s)]
    tissue = shapely.make_valid(region.difference(tumour.buffer(200 / mpp)))
    tis, _ = findall.scope_geometry(reg, sid, {'kind': 'slide'}, tissue=True)
    if tis is not None:
        tissue = tissue.intersection(tis)
    outer = [shapely.box(px - s / 2, py - s / 2, px + s / 2, py + s / 2) for px, py in _inside_points(tissue, 2, s)]
    T = AnnotationLayer(name='Training boxes (areas)')
    for b in inner:
        T.add_polygon(np.asarray(b.exterior.coords), cls='Tumour', color='#ff5ca8')
    for b in outer:
        T.add_polygon(np.asarray(b.exterior.coords), cls='Other', color='#5ac8fa')
    reg.add_annotations(sid, T)
    annio.export(T, d / 'layers' / f'{CAMELYON}.training.parquet', slide=slide.name)
    jobs = {
        'areas': ('Areas (4 brushed boxes)', lambda: findall.find_areas(reg, sid, training=T.name, scope=scope,
                                                                          detail='coarse'), ['Tumour']),
        'sam3': ('SAM 3', lambda: findall.find_by_text(reg, sid, prompt='tumor', model='sam3', scope=scope,
                                                      object_um=150, object_um_max=4000, strictness=0.4), None),
        'dino-sam3': ('Grounding DINO → SAM 3', lambda: findall.find_by_text(
            reg, sid, prompt='metastatic tumor region in a lymph node', model='dino-sam3', sam3_name='tumor',
            scope=scope, object_um=150, object_um_max=4000, strictness=0.4), None),
    }  # fmt: skip
    for k, (key, (label, fn, classes)) in enumerate((k, v) for k, v in jobs.items() if k in methods):

        def go(fn=fn, label=label, classes=classes, k=k, key=key):
            r = fn()
            p = _export(reg, sid, r['layer'], d / 'layers' / f'{CAMELYON}.{key}.parquet', f'{label} · tumour',
                        METHOD_COLORS[k % len(METHOD_COLORS)], classes)  # fmt: skip
            got = evaluate.polygons(annio.read_geoparquet(p))
            return {
                'dice': evaluate.dice(got, tumour, region),
                'found_mm2': round(shapely.union_all(got).intersection(region).area * mpp * mpp / 1e6, 3)
                if len(got)
                else 0.0,
            }

        _try(out['methods'], key, go, log)
    return out


def _inside_points(g, n: int, margin: float) -> list[tuple[float, float]]:
    """``n`` points well inside ``g`` (deterministic): the inner poles of the largest parts, then a grid."""
    inner = g.buffer(-margin)
    if inner.is_empty:
        inner = g
    parts = sorted(getattr(inner, 'geoms', [inner]), key=lambda p: -p.area)
    pts = []
    for p in parts:
        c = p.representative_point()
        pts.append((c.x, c.y))
        if len(pts) == n:
            return pts
    x0, y0, x1, y1 = inner.bounds
    for fy in np.linspace(0.2, 0.8, 4):
        for fx in np.linspace(0.2, 0.8, 4):
            q = shapely.Point(x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))
            if inner.contains(q) and all(math.dist((q.x, q.y), p) > 3 * margin for p in pts):
                pts.append((q.x, q.y))
                if len(pts) == n:
                    return pts
    return pts


def run_xenium(session, d: Path, methods: list[str], log) -> dict:
    x = next(i for i in pack.manifest()['items'] if i['id'] == XENIUM)
    pd = pack.pack_dir(_PACK)
    ref = xenium_nuclei(d, log)
    slide = pack._slide(x, pd)
    sid = session.open([slide, ref], trusted=True)['opened'][0]
    reg = session.registry
    src = findall._source(reg, sid)
    mpp = findall._mpp(src)
    tis, _ = findall.scope_geometry(reg, sid, {'kind': 'slide'}, tissue=True)
    R = annio.read_geoparquet(ref)
    c = R.centroid()
    cx, cy = np.median(c[:, 0]), np.median(c[:, 1])  # a 1 mm field where the nuclei are
    half = 500 / mpp
    region = shapely.box(cx - half, cy - half, cx + half, cy + half)
    if tis is not None and not region.intersects(tis):
        region = tis.envelope
    out = {'reference': int(shapely.contains_xy(region, c[:, 0], c[:, 1]).sum()), 'methods': {}}
    scope = {'kind': 'view', 'rect': list(region.bounds)}
    for k, key in enumerate(m for m in methods if m in ('classic', 'cellpose')):
        label = NUCLEUS_METHODS[key][0]

        def go(key=key, label=label, k=k):
            r = findall.find_cells(reg, sid, model=key, scope=scope)
            p = _export(reg, sid, r['layer'], d / 'layers' / f'{XENIUM}.{key}.parquet', f'{label} · nuclei',
                        METHOD_COLORS[k % len(METHOD_COLORS)])  # fmt: skip
            return evaluate.match(annio.read_geoparquet(p), R, region=region)

        _try(out['methods'], key, go, log)
    out['region'] = [round(v) for v in region.bounds]
    return out


def run(cases=CASES, methods=None, directory: str | None = None, yes: bool = False, log=print) -> Path:
    """Compute every case (see the module docstring); returns validation.json."""
    d = vdir(directory)
    methods = methods or list(NUCLEUS_METHODS) + ['areas']
    ids = [i for c, i in (('camelyon', CAMELYON), ('xenium', XENIUM)) if c in cases]
    need = [i for i in pack.manifest()['items'] if i['id'] in ids]
    if need:
        pd = pack.pack_dir(directory)
        missing = sum(x['size'] for x in need if not pack._complete(pd / 'downloads' / x['file'], x))
        if missing and not yes:
            raise SystemExit(f'needs {pack._human(missing)} of the public sample pack: run again with --yes')
        pack.download(need, pd, log=log)
        pack.prepare(need, pd, log=log)
    report: dict = {'when': time.strftime('%Y-%m-%d %H:%M'), 'cases': {}}
    s = Session(cache_dir=d / 'cache', remember=False)
    try:
        for case, fn in (('nuinsseg', run_nuinsseg), ('drg', run_drg), ('camelyon', run_camelyon),
                         ('xenium', run_xenium)):  # fmt: skip
            if case not in cases:
                continue
            log(f'{case}:')
            t0 = time.perf_counter()
            try:
                report['cases'][case] = fn(s, d, methods, log)
            except (SystemExit, Exception) as e:  # noqa: BLE001  (one case failing: reported, the others run)
                report['cases'][case] = {'error': f'{e.__class__.__name__}: {e}'[:400]}
                log(f'{case}: failed: {report["cases"][case]["error"]}')
            report['cases'][case]['seconds'] = round(time.perf_counter() - t0, 1)
    finally:
        s.shutdown()
    p = d / 'validation.json'
    p.write_text(json.dumps(report, indent=1), encoding='utf-8')
    (d / 'validation.md').write_text(markdown(report), encoding='utf-8')
    pictures(d, log)
    session_file(d)
    return p


# ================================================================================================ report, pictures
def markdown(rep: dict) -> str:
    lines = [f'# Find all against reference outlines ({rep.get("when", "")})', '']
    names = {k: v[0] for k, v in (NUCLEUS_METHODS | DRG_METHODS).items()} | {'areas': 'Areas (4 brushed boxes)'}
    for case, title, cols in (
        ('nuinsseg', 'NuInsSeg: H&E nuclei, hand-drawn (F1 at IoU ≥ 0.5)', ('f1', 'precision', 'recall', 'mean_iou',
                                                                            'found', 'reference')),
        ('drg', 'Rat DRG: neurons vs 3 experts\' consensus (F1 at IoU ≥ 0.5)', ('f1', 'precision', 'recall', 'mean_iou',
                                                                             'found', 'reference')),
        ('camelyon', 'CAMELYON16 test_065: metastasis area (Dice)', ('dice', 'found_mm2')),
        ('xenium', 'Xenium breast: nuclei vs 10x (F1 at IoU ≥ 0.5)', ('f1', 'precision', 'recall', 'mean_iou', 'found',
                                                                     'reference')),
    ):  # fmt: skip
        c = rep['cases'].get(case)
        if not c:
            continue
        lines += [f'## {title}', '']
        if c.get('error'):
            lines += [f'Not run: {c["error"]}', '']
            continue
        lines += ['| method | ' + ' | '.join(cols) + ' | seconds |', '|---' * (len(cols) + 2) + '|']
        for k, m in c.get('methods', {}).items():
            if m.get('error'):
                lines.append(f'| {names.get(k, k)} | ' + ' | '.join('–' for _ in cols) + f' | failed: {m["error"]} |')
            else:
                lines.append(f'| {names.get(k, k)} | ' + ' | '.join(str(m.get(x, '')) for x in cols)
                             + f' | {m.get("seconds", "")} |')  # fmt: skip
        lines.append('')
    return '\n'.join(lines)


def pictures(d: Path, log=print) -> None:
    """For each case and method: a crop with the reference (green) and the method (magenta) outlines."""
    jobs = [
        ('nuinsseg-mosaic', d / 'nuinsseg-mosaic.ome.tif', d / 'layers' / 'nuinsseg-mosaic.reference.parquet'),
        ('drg-mosaic', d / 'drg-mosaic.ome.tif', d / 'layers' / 'drg-mosaic.reference.parquet'),
    ]
    pd = pack.pack_dir(_PACK)
    for xid, ref in ((CAMELYON, pd / 'layers' / f'{CAMELYON}.expert.parquet'),
                     (XENIUM, d / 'layers' / f'{XENIUM}.nuclei-10x.parquet')):  # fmt: skip
        x = next(i for i in pack.manifest()['items'] if i['id'] == xid)
        jobs.append((xid, pack._slide(x, pd), ref))
    for stem, slide, ref in jobs:
        if not slide.exists() or not ref.exists():
            continue
        found = sorted((d / 'layers').glob(f'{stem}.*.parquet'))
        found = [p for p in found if not p.name.endswith(('.reference.parquet', '.patches.parquet', '.training.parquet',
                                                          '.nuclei-10x.parquet'))]  # fmt: skip
        if not found:
            continue
        src = open_slide(slide)
        R = annio.read_geoparquet(ref)
        c = R.centroid()
        if not len(c):
            continue
        cx, cy = float(np.median(c[:, 0])), float(np.median(c[:, 1]))
        side = {'nuinsseg-mosaic': 1150, 'drg-mosaic': 2000}.get(stem, 700)  # a few patches; a field of nuclei
        if stem == CAMELYON:  # the lesions, whole
            bb = R.bbox()
            ex = [bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()]
            cx, cy = (ex[0] + ex[2]) / 2, (ex[1] + ex[3]) / 2
            side = int(max(ex[2] - ex[0], ex[3] - ex[1]) * 1.3)
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        lv = 0
        for i, L in enumerate(src.info.levels):
            if side / L.downsample >= 900:
                lv = i
        ds = src.info.levels[lv].downsample
        w = int(min(side, src.info.width - x0) / ds)
        h = int(min(side, src.info.height - y0) / ds)
        base = np.asarray(src.read_region(lv, int(x0 / ds), int(y0 / ds), w, h))
        if base.ndim == 2 or base.shape[-1] not in (3, 4):  # fluorescence: the first channel, as grey
            base = np.asarray(src.read_region(lv, int(x0 / ds), int(y0 / ds), w, h, 0), np.float32)
            base = np.clip(255 * base / max(1.0, np.percentile(base, 99.5)), 0, 255).astype(np.uint8)
            base = np.stack([base] * 3, -1)
        base = base[..., :3]
        for p in found:
            im = Image.fromarray(base.copy())
            dr = ImageDraw.Draw(im)
            for F, col in ((R, (61, 220, 132)), (annio.read_geoparquet(p), (255, 92, 168))):
                for i in range(len(F)):
                    ring = (np.asarray(F.rings(i)[0], float) - [x0, y0]) / ds
                    if len(ring) < 3 or ring[:, 0].max() < 0 or ring[:, 1].max() < 0 or ring[:, 0].min() > w \
                            or ring[:, 1].min() > h:  # fmt: skip
                        continue
                    dr.line([tuple(q) for q in ring] + [tuple(ring[0])], fill=col, width=2)
            im.save(d / f'{p.stem}.jpg', quality=88)
        log(f'pictures: {stem}')


def session_file(d: Path) -> Path:
    """validation-session.json: the three slides with the reference and every method as layers (for the app)."""
    pd = pack.pack_dir(_PACK)
    entries = []
    mosaic = d / 'nuinsseg-mosaic.ome.tif'
    if mosaic.exists():
        entries.append({'id': 'nuinsseg-mosaic', 'title': 'NuInsSeg mosaic (H&E nuclei, hand-drawn)',
                        'slide': str(mosaic), 'layers': [str(p) for p in sorted((d / 'layers').glob('nuinsseg-mosaic.*.parquet'))],
                        'licence': 'CC-BY-4.0'})  # fmt: skip
    drg = d / 'drg-mosaic.ome.tif'
    if drg.exists():
        entries.append({'id': 'drg-mosaic', 'title': 'Rat DRG neurofilament (neurons, 3 experts)', 'slide': str(drg),
                        'layers': [str(p) for p in sorted((d / 'layers').glob('drg-mosaic.*.parquet'))],
                        'licence': 'CC-BY-4.0'})  # fmt: skip
    for xid, extra in ((CAMELYON, [pd / 'layers' / f'{CAMELYON}.expert.parquet']), (XENIUM, [])):
        x = next(i for i in pack.manifest()['items'] if i['id'] == xid)
        slide = pack._slide(x, pd)
        if not slide.exists():
            continue
        layers = [p for p in extra if p.exists()] + sorted((d / 'layers').glob(f'{xid}.*.parquet'))
        entries.append({'id': xid, 'title': x['title'], 'slide': str(slide), 'layers': [str(p) for p in layers],
                        'licence': x['licence']})  # fmt: skip
    doc = {'name': 'wissi-viz validation', 'slides': entries, 'show': [e['id'] for e in entries],
           'projects': [{'name': 'Validation', 'slides': [e['id'] for e in entries]}]}  # fmt: skip
    p = d / 'validation-session.json'
    p.write_text(json.dumps(doc, indent=1), encoding='utf-8')
    return p


# ================================================================================================ the published run
def fetch_published(d: Path, log=print) -> None:
    """The latest run of the validate workflow: layers, the NuInsSeg mosaic, the report and pictures."""
    sums = urllib.request.urlopen(f'{RELEASE}/SHA256SUMS', timeout=60).read().decode()
    for line in sums.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        want, name = parts
        dest = (d / 'layers' / name) if name.endswith('.parquet') else d / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == want:
            continue
        _get(f'{RELEASE}/{name}', dest, log)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != want:
            dest.unlink()
            raise SystemExit(f'{name}: checksum mismatch (try again)')


# ------------------------------------------------------------------------------------------------ measured evidence
#: which analysis tool each validated method belongs to (the tools an agent or a person chooses between)
TOOL_OF_METHOD = {'grounding': 'find_by_text', 'sam3': 'find_by_text', 'dino-sam3': 'find_by_text',
                  'cellpose': 'find_cells', 'classic': 'find_cells', 'segment': 'segment_everything',
                  'areas': 'find_areas'}  # fmt: skip
#: what each case tests, in a few words
CASE_TEXT = {'nuinsseg': 'H&E nuclei (brightfield, dense, ~8 µm)',
             'drg': 'neuron cell bodies (fluorescence, one channel, 20-60 µm)',
             'camelyon': 'tumour areas in an H&E lymph node',
             'xenium': 'nuclei (fluorescence, DAPI)'}  # fmt: skip
METHOD_NAMES = {'grounding': 'Grounding DINO + SAM 2.1', 'sam3': 'SAM 3', 'dino-sam3': 'Grounding DINO → SAM 3',
                'cellpose': 'Cellpose-SAM', 'classic': 'classic threshold', 'segment': 'segment everything',
                'areas': 'areas from brushed examples'}  # fmt: skip


def evidence(results: dict) -> dict[str, str]:
    """What the validation measured, one sentence per tool: for each case, the tool's best method and its score
    (F1 at IoU ≥ 0.5 for objects, Dice for areas), or that it found nothing. ``results``: validation.json."""
    per: dict[str, list[str]] = {}
    for case, r in (results.get('cases') or {}).items():
        best: dict[str, tuple[float, str]] = {}
        for m, v in (r.get('methods') or {}).items():
            tool = TOOL_OF_METHOD.get(m)
            if tool is None:
                continue
            if 'error' in v:
                score, txt = 0.0, f'{METHOD_NAMES.get(m, m)} found nothing'
            elif 'dice' in v:
                score, txt = float(v['dice']), f'Dice {v["dice"]:.2f} ({METHOD_NAMES.get(m, m)})'
            else:
                slow = f', ~{v["seconds"] / 60:.0f} min on a CPU' if v.get('seconds', 0) > 900 else ''
                score = float(v.get('f1', 0))
                txt = (f'F1 {v["f1"]:.2f}, recall {v.get("recall", 0):.2f} ({METHOD_NAMES.get(m, m)}{slow})'
                       if v.get('found') else f'{METHOD_NAMES.get(m, m)} found nothing')  # fmt: skip
            if tool not in best or score > best[tool][0]:
                best[tool] = (score, txt)
        for tool, (_, txt) in best.items():
            per.setdefault(tool, []).append(f'{CASE_TEXT.get(case, case)}: {txt}')
    return {t: 'Measured: ' + '; '.join(v) + '.' for t, v in per.items()}


def published_results(timeout: float = 60) -> dict | None:
    """validation.json of the latest validate run (None when it cannot be fetched)."""
    try:
        return json.loads(urllib.request.urlopen(f'{RELEASE}/validation.json', timeout=timeout).read().decode())
    except Exception:  # noqa: BLE001  (offline, or not published yet)
        return None


def publish_list(d: Path) -> list[Path]:
    """What the workflow publishes (and SHA256SUMS for them)."""
    files = sorted((d / 'layers').glob('*.parquet')) + [p for p in (d / 'nuinsseg-mosaic.ome.tif',
                   d / 'NUINSSEG-ATTRIBUTION.md', d / 'drg-mosaic.ome.tif', d / 'drg-mosaic.json',
                   d / 'DRG-ATTRIBUTION.md', d / 'validation.json', d / 'validation.md') if p.exists()]  # fmt: skip
    files += sorted(d.glob('*.jpg'))
    (d / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
    return [*files, d / 'SHA256SUMS']


def open_app(directory: str | None = None, yes: bool = False, browser: bool = False, log=print) -> None:
    d = vdir(directory)
    if not (d / 'validation.json').exists():
        log('fetching the latest published validation run…')
        fetch_published(d, log)
    need = [i for i in pack.manifest()['items'] if i['id'] in (CAMELYON, XENIUM)]
    pd = pack.pack_dir(directory)
    missing = sum(x['size'] for x in need if not pack._complete(pd / 'downloads' / x['file'], x))
    if missing and not yes:
        if not os.isatty(0):
            raise SystemExit(f'the CAMELYON16 and Xenium slides need {pack._human(missing)}: run again with --yes')
        if input(f'Download {pack._human(missing)} of public slides? [y/N] ').strip().lower() not in ('y', 'yes'):
            raise SystemExit('cancelled')
    pack.download(need, pd, log=log)
    pack.prepare(need, pd, log=log)
    if (d / 'validation.md').exists():
        log((d / 'validation.md').read_text(encoding='utf-8'))
    desktop_main(['--session', str(session_file(d))] + (['--browser'] if browser else []))


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog='wissi-viz validate', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument('step', choices=['run', 'open', 'publish-list'])
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--methods', default=None, help='comma-separated (default: all that can run here)')
    ap.add_argument('--dir', default=None)
    ap.add_argument('--yes', action='store_true')
    ap.add_argument('--browser', action='store_true')
    certs.add_arg(ap)
    a = ap.parse_args(certs.take_flag(argv))
    if a.step == 'run':
        p = run(a.cases.split(','), a.methods.split(',') if a.methods else None, a.dir, a.yes)
        print((p.parent / 'validation.md').read_text(encoding='utf-8'))
    elif a.step == 'publish-list':
        print('\n'.join(str(p) for p in publish_list(vdir(a.dir))))
    else:
        open_app(a.dir, a.yes, a.browser)
