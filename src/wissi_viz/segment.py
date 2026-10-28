"""Whole-slide cell / nucleus segmentation into a GeoParquet layer, tile by tile, with per-cell measurements.

    wissi-viz segment slide.svs cells.parquet                      # best available: cellpose → classic
    wissi-viz segment slide.svs cells.parquet --model cellpose     # Cellpose-SAM (BSD-3), GPU if present
    wissi-viz segment slide.svs cells.parquet --model classpose    # Classpose cell types on H&E (see licence note)
    wissi-viz segment slide.svs cells.parquet --model classic      # no model: stain / DAPI threshold (CPU)

Every backend writes the same table: an outline polygon per cell (level-0 px), a class, and measurements — area
(µm²), perimeter, circularity, eccentricity, and the mean of every stain (haematoxylin, eosin, DAB optical density)
or fluorescence channel inside the cell. Tiles overlap; a cell belongs to the tile that holds its centre, so cells
are neither cut nor counted twice.

Classes: Classpose assigns cell types. The others classify by measurement, which is honest about what a stain alone
can tell: H&E nuclei by size (small / medium / large), fluorescence cells by their brightest positive marker
(e.g. ``CD3+``; a per-channel Otsu threshold), else ``negative``.

Licences: Cellpose-SAM code and weights BSD-3 (``pip install cellpose``). Classpose weights are labelled MIT but are
trained on datasets with non-commercial terms (CoNIC/Lizard, MoNuSAC: CC BY-NC-SA); it runs in its own environment
(``uvx``) and is opt-in — check with your legal team before commercial use of its output. wissi-viz never uses
GrandQC (non-commercial).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import _slug, _stain_od

STAINS = ('hematoxylin', 'eosin', 'dab')


def available() -> list[str]:
    import importlib.util
    out = []
    if importlib.util.find_spec('cellpose'):
        out.append('cellpose')
    if shutil.which('uvx') or shutil.which('uv'):
        out.append('classpose')
    out.append('classic')
    return out


# ------------------------------------------------------------------------------------------------ tiling
def _tissue_tiles(src, tile_l0: int, overlap_l0: int) -> list[tuple[int, int, int, int]]:
    """Level-0 tiles (x, y, w, h) that contain tissue (brightfield: not white; fluorescence: above background)."""
    info = src.info
    lv = max((i for i, L in enumerate(info.levels) if L.width <= 4096), default=len(info.levels) - 1)
    L = info.levels[lv]
    if info.mode == 'rgb':
        rgb = np.asarray(src.read_region(lv, 0, 0, L.width, L.height))[..., :3].astype(np.float32) / 255
        tissue = ((rgb.max(-1) - rgb.min(-1)) > 0.06) & (rgb.mean(-1) < 0.9)
    else:
        a = np.asarray(src.read_region(lv, 0, 0, L.width, L.height, 0), np.float32)
        from skimage.filters import threshold_otsu
        tissue = a > (threshold_otsu(a) * 0.5 if a.max() > a.min() else 0)
    ds = L.downsample
    out = []
    for y in range(0, info.height, tile_l0):
        for x in range(0, info.width, tile_l0):
            t = tissue[int(y / ds):int((y + tile_l0) / ds) + 1, int(x / ds):int((x + tile_l0) / ds) + 1]
            if t.size and t.mean() > 0.02:
                x0, y0 = max(0, x - overlap_l0), max(0, y - overlap_l0)
                out.append((x0, y0, min(info.width, x + tile_l0 + overlap_l0) - x0, min(info.height, y + tile_l0 + overlap_l0) - y0))
    return out


def _read(src, x, y, w, h, scale: float) -> np.ndarray:
    """Level-0 rectangle resampled by ``scale`` (< 1 shrinks): RGB uint8 or (h, w, C) float32."""
    from PIL import Image
    if src.info.mode == 'rgb':
        a = np.asarray(src.read_region(0, x, y, w, h))[..., :3]
        if abs(scale - 1) > 1e-3:
            a = np.asarray(Image.fromarray(a).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR))
        return a
    chans = []
    for c in range(len(src.info.channels)):
        a = np.asarray(src.read_region(0, x, y, w, h, c), np.float32)
        if abs(scale - 1) > 1e-3:
            a = np.asarray(Image.fromarray(a).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR))
        chans.append(a)
    return np.stack(chans, -1)


# ------------------------------------------------------------------------------------------------ backends
def _classic_labels(img: np.ndarray, rgb: bool, thr: float | None) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from skimage.segmentation import watershed
    sig = gaussian(_stain_od(img)['hematoxylin'] if rgb else img[..., 0], 1.0)
    m = ndi.binary_opening(sig > thr, iterations=1)
    m = ndi.binary_fill_holes(m)
    dist = ndi.distance_transform_edt(m)
    peaks = peak_local_max(dist, min_distance=4, labels=ndi.label(m)[0], exclude_border=False)
    markers = np.zeros(m.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=m).astype(np.int32)


class _Cellpose:
    def __init__(self, gpu: bool | None = None) -> None:
        from cellpose import models
        import torch
        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        self.model = models.CellposeModel(gpu=use_gpu)
        self.device = 'cuda' if use_gpu else 'cpu'

    def labels(self, imgs: list[np.ndarray], rgb: bool) -> list[np.ndarray]:
        x = []
        for im in imgs:
            if rgb:
                x.append(im)
            else:                               # nuclear channel first, then the brightest other marker
                a = im[..., :min(3, im.shape[-1])]
                x.append(a)
        masks, *_ = self.model.eval(x, batch_size=16, flow_threshold=0.4, cellprob_threshold=0.0, normalize=True)
        return [np.asarray(m, np.int32) for m in masks]


# ------------------------------------------------------------------------------------------------ per-tile objects
def _objects(lab: np.ndarray, img: np.ndarray, rgb: bool, core: tuple, x0: float, y0: float, inv: float,
             names: list[str], min_px: int) -> dict:
    """Outlines + measurements of the labels whose centre lies in the tile core (tile px → level-0 via inv)."""
    from skimage.measure import find_contours, regionprops
    cx0, cy0, cx1, cy1 = core
    chans = [_stain_od(img)[s] for s in STAINS] if rgb else [img[..., c] for c in range(img.shape[-1])]
    n = int(lab.max())
    if n == 0:
        return {}
    flat = lab.ravel()
    cnt = np.bincount(flat, minlength=n + 1)
    means = [np.bincount(flat, ch.ravel().astype(np.float64), minlength=n + 1) / np.maximum(cnt, 1) for ch in chans]
    rings, rows = [], []
    for rp in regionprops(lab):
        if rp.area < min_px:
            continue
        cy, cx = rp.centroid
        if not (cx0 <= cx < cx1 and cy0 <= cy < cy1):
            continue
        r0, c0, r1, c1 = rp.bbox
        sub = np.pad(rp.image, 1)
        cs = find_contours(sub.astype(np.float32), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        ring = np.stack([(c[:, 1] - 1 + c0) * inv + x0, (c[:, 0] - 1 + r0) * inv + y0], 1)
        if len(ring) > 24:                      # keep outlines light: ≤ ~24 vertices per cell
            ring = ring[np.linspace(0, len(ring) - 1, 24).astype(int)]
        rings.append(ring)
        rows.append((rp.label, rp.area, rp.perimeter, rp.eccentricity))
    if not rows:
        return {}
    lab_ids = np.array([r[0] for r in rows])
    out = {'rings': rings, 'area_px': np.array([r[1] for r in rows], float) * inv * inv,
           'perimeter_px': np.array([r[2] for r in rows], float) * inv,
           'eccentricity': np.array([r[3] for r in rows], float)}
    for nm, m in zip(names, means):
        out[f'{nm}_mean'] = m[lab_ids]
    return out


def _tile_job(args):
    path, tile, scale, overlap_l0, thr, names, min_px = args
    from .sources import open_slide
    src = open_slide(path)
    x, y, w, h = tile
    img = _read(src, x, y, w, h, scale)
    rgb = src.info.mode == 'rgb'
    lab = _classic_labels(img, rgb, thr)
    return _finish_tile(lab, img, rgb, tile, scale, overlap_l0, src.info, names, min_px)


def _finish_tile(lab, img, rgb, tile, scale, overlap_l0, info, names, min_px):
    x, y, w, h = tile
    inv = 1 / scale
    m = overlap_l0 / 2 * scale                  # core = tile minus half the overlap (centre ownership)
    core = (0 if x == 0 else m, 0 if y == 0 else m,
            lab.shape[1] - (0 if x + w >= info.width else m), lab.shape[0] - (0 if y + h >= info.height else m))
    return _objects(lab, img, rgb, core, x, y, inv, names, min_px)


# ------------------------------------------------------------------------------------------------ public API
def segment_slide(slide: str | Path, out: str | Path, model: str = 'auto', target_mpp: float = 0.5, tile: int = 1024,
                  overlap: int = 64, workers: int | None = None, log=print) -> Path:
    """Segment every cell of a slide into a GeoParquet layer (see module docstring). Returns the output path."""
    from . import annio
    from .sources import open_slide
    src = open_slide(slide)
    info = src.info
    mpp = (info.mpp or (None,))[0] or target_mpp
    model = model if model != 'auto' else available()[0] if available()[0] != 'classpose' else 'classic'
    if model == 'classpose':
        return _classpose(slide, out, log)
    scale = min(1.0, mpp / target_mpp)          # resample so a pixel is ~target_mpp µm
    tile_l0, ov_l0 = int(tile / scale), int(overlap / scale)
    tiles = _tissue_tiles(src, tile_l0, ov_l0)
    rgb = info.mode == 'rgb'
    names = list(STAINS) if rgb else [_slug(c.name or f'ch{i}') for i, c in enumerate(info.channels)]
    min_px = max(6, int(8 * (0.5 / target_mpp) ** 2))
    log(f'{Path(slide).name}: {len(tiles)} tissue tiles, model {model}, {mpp * (1 / scale):.2f} µm/px')
    t0 = time.time()
    parts = []
    if model == 'classic':
        from skimage.filters import threshold_otsu
        rng = np.random.default_rng(0)
        samp = [tiles[i] for i in rng.choice(len(tiles), min(10, len(tiles)), replace=False)] if tiles else []
        vals = []
        for t in samp:
            im = _read(src, *t, scale)
            v = (_stain_od(im)['hematoxylin'] if rgb else im[..., 0]).ravel()[::5]
            vals.append(v)
        thr = float(threshold_otsu(np.concatenate(vals))) if vals else 0.0
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(str(slide), t, scale, ov_l0, thr, names, min_px) for t in tiles]
        with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
            for k, r in enumerate(ex.map(_tile_job, jobs, chunksize=2)):
                if r:
                    parts.append(r)
                if k % 50 == 0:
                    log(f'  {k}/{len(tiles)} tiles, {time.time() - t0:.0f} s')
    elif model == 'cellpose':
        cp = _Cellpose()
        log(f'  Cellpose-SAM on {cp.device}')
        batch = 8
        for b0 in range(0, len(tiles), batch):
            ts = tiles[b0:b0 + batch]
            imgs = [_read(src, *t, scale) for t in ts]
            for t, im, lab in zip(ts, imgs, cp.labels(imgs, rgb)):
                r = _finish_tile(lab, im, rgb, t, scale, ov_l0, info, names, min_px)
                if r:
                    parts.append(r)
            log(f'  {min(b0 + batch, len(tiles))}/{len(tiles)} tiles, {time.time() - t0:.0f} s')
    else:
        raise ValueError(f'unknown model {model!r} (cellpose | classpose | classic)')
    F = _features(parts, rgb, names, mpp)
    F.name = 'cells'
    p = annio.write_geoparquet(F, out, slide=Path(slide).name)
    log(f'{len(F):,} cells → {p} ({time.time() - t0:.0f} s)')
    return p


def _features(parts: list[dict], rgb: bool, names: list[str], mpp: float):
    from . import annio
    rings = [r for p in parts for r in p['rings']]
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != 'rings'} if parts else {}
    n = len(rings)
    ring_off = np.zeros(n + 1, np.int64)
    if n:
        np.cumsum([len(r) for r in rings], out=ring_off[1:])
    coords = np.concatenate(rings) if n else np.zeros((0, 2))
    props = {}
    if n:
        props['area_um2'] = cols['area_px'] * mpp * mpp
        props['perimeter_um'] = cols['perimeter_px'] * mpp
        props['circularity'] = np.clip(4 * np.pi * cols['area_px'] / np.maximum(cols['perimeter_px'], 1e-9) ** 2, 0, 1)
        props['eccentricity'] = cols['eccentricity']
        for nm in names:
            props[f'{nm}_mean'] = cols[f'{nm}_mean']
    labels = classify(props, rgb, names)
    class_names = sorted(set(labels.tolist()))
    idx = {c: i for i, c in enumerate(class_names)}
    F = annio.Features(np.full(n, annio.GEO_POLYGON, np.uint8), coords, ring_off, np.arange(n + 1, dtype=np.int64),
                       np.array([idx[c] for c in labels], np.int32), class_names, props={k: v.astype(np.float32) for k, v in props.items()})
    F.class_colors = {c: col for c, col in zip(class_names, _palette(len(class_names)))}
    F.mpp = mpp
    F.style = {'line_width': 1.2, 'fill_opacity': 0.25}
    return F


def classify(props: dict, rgb: bool, names: list[str]) -> np.ndarray:
    """Measurement-based classes: H&E nuclei by size; fluorescence cells by their brightest positive marker."""
    n = len(props.get('area_um2', []))
    if n == 0:
        return np.zeros(0, object)
    if rgb:
        a = props['area_um2']
        return np.where(a < 25, 'Small nucleus', np.where(a < 50, 'Medium nucleus', 'Large nucleus')).astype(object)
    from skimage.filters import threshold_otsu
    markers = [nm for nm in names[1:]] or names
    z = []
    for nm in markers:
        v = props[f'{nm}_mean'].astype(np.float64)
        t = threshold_otsu(v) if v.max() > v.min() else np.inf
        z.append((v - t) / (v.std() + 1e-9))
    z = np.stack(z, 1)
    best = z.argmax(1)
    out = np.array([f'{markers[b]}+' for b in best], dtype=object)
    out[z.max(1) <= 0] = 'negative'
    return out


def _palette(n: int) -> list[str]:
    from .annotations import _auto_color
    return [_auto_color(i) for i in range(n)]


def _classpose(slide, out, log) -> Path:
    """Run Classpose in its own environment (uvx) and convert its cell contours (GeoJSON) to GeoParquet."""
    import tempfile

    from . import annio
    log('Classpose: weights are labelled MIT but were trained on datasets with non-commercial terms '
        '(e.g. CoNIC/Lizard, MoNuSAC: CC BY-NC-SA). Check before commercial use. GrandQC is not used.')
    uvx = shutil.which('uvx') or shutil.which('uv')
    if not uvx:
        raise RuntimeError('Classpose runs through uv: install uv first (https://docs.astral.sh/uv/)')
    cmd = ([uvx] if uvx.endswith('uvx') or uvx.endswith('uvx.exe') else [uvx, 'tool', 'run']) + [
        '--from', 'git+https://github.com/sohmandal/classpose', 'classpose-predict-wsi', '--model_config', 'conic',
        '--slide_path', str(slide), '--output_folder', '', '--device', 'cuda:0' if _cuda() else 'cpu']
    with tempfile.TemporaryDirectory() as d:
        cmd[cmd.index('--output_folder') + 1] = d
        log('  ' + ' '.join(cmd))
        subprocess.run(cmd, check=True)
        gj = next(Path(d).rglob('*cell_contours.geojson'))
        layer = annio.load(gj, name='cells')
        return annio.export(layer, out, slide=Path(slide).name)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return shutil.which('nvidia-smi') is not None


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog='wissi-viz segment')
    ap.add_argument('slide')
    ap.add_argument('out', help='output .parquet (GeoParquet)')
    ap.add_argument('--model', default='auto', choices=['auto', 'cellpose', 'classpose', 'classic'])
    ap.add_argument('--mpp', type=float, default=0.5, help='resolution to segment at (µm/px)')
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args(argv)
    segment_slide(a.slide, a.out, a.model, a.mpp, workers=a.workers)


if __name__ == '__main__':
    main(sys.argv[1:])
