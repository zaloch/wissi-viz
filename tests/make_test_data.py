"""Generate synthetic test slides: an Aperio-style SVS (brightfield, H&E-like) and a
3-channel 16-bit fluorescence CZI, plus a GeoJSON file with ~cell detections.

    python tests/make_test_data.py OUTDIR [--size 16000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

CELL = 28  # nucleus grid cell size (px)


def _noise(h: int, w: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    small = rng.random((h // scale + 3, w // scale + 3)).astype(np.float32)
    small = ndimage.gaussian_filter(small, 1.2)
    big = ndimage.zoom(small, scale, order=1)[:h, :w]
    big -= big.min()
    return big / max(big.max(), 1e-6)


def nuclei_field(h: int, w: int, seed: int = 1):
    """Jittered-grid nuclei: returns per-cell centers/radii and a tissue mask fn."""
    rng = np.random.default_rng(seed)
    gh, gw = h // CELL + 1, w // CELL + 1
    tissue = _noise(gh, gw, 16, rng)
    present = (tissue > 0.42) & (rng.random((gh, gw)) < 0.75)
    cy = (np.arange(gh)[:, None] * CELL + CELL / 2 + rng.uniform(-7, 7, (gh, gw))).astype(np.float32)
    cx = (np.arange(gw)[None, :] * CELL + CELL / 2 + rng.uniform(-7, 7, (gh, gw))).astype(np.float32)
    r = rng.uniform(5, 10, (gh, gw)).astype(np.float32)
    cls = (tissue * 3 + rng.random((gh, gw)) * 0.8).astype(np.int32) % 3
    return dict(present=present, cx=cx, cy=cy, r=r, tissue=tissue, cls=cls)


def render_band(f, y0: int, y1: int, w: int):
    """Return (nucleus_intensity, tissue_intensity) float32 for rows [y0, y1)."""
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    gy, gx = (ys // CELL).astype(int), (xs // CELL).astype(int)
    best = np.full((y1 - y0, w), 1e9, np.float32)
    gh, gw = f['present'].shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            iy, ix = np.clip(gy + dy, 0, gh - 1), np.clip(gx + dx, 0, gw - 1)
            p = f['present'][iy, ix]
            d = np.hypot(xs - f['cx'][iy, ix], ys - f['cy'][iy, ix]) / f['r'][iy, ix]
            best = np.minimum(best, np.where(p, d, 1e9))
    nuc = np.clip(1.3 - best, 0, 1)
    tis = f['tissue'][np.clip(gy, 0, gh - 1), np.clip(gx, 0, gw - 1)]
    tis = np.clip((tis - 0.35) * 6, 0, 1)
    return nuc, tis.astype(np.float32)


def make_svs(path: Path, w: int, h: int, f):
    rng = np.random.default_rng(3)
    img = np.empty((h, w, 3), np.uint8)
    for y0 in range(0, h, 1024):
        y1 = min(h, y0 + 1024)
        nuc, tis = render_band(f, y0, y1, w)
        grain = rng.normal(0, 0.04, nuc.shape).astype(np.float32)
        eos = np.array([236, 150, 190], np.float32)  # pink
        hem = np.array([70, 50, 140], np.float32)  # purple
        white = np.array([245, 245, 245], np.float32)
        base = white * (1 - tis[..., None]) + eos * tis[..., None]
        px = base * (1 - nuc[..., None] * tis[..., None]) + hem * (nuc * tis)[..., None]
        px *= (1 + grain[..., None])
        img[y0:y1] = np.clip(px, 0, 255).astype(np.uint8)

    desc = (f'Aperio Image Library v12.0.15 \r\n{w}x{h} [0,0 {w}x{h}] (256x256) JPEG/RGB Q=85'
            f'|AppMag = 20|MPP = 0.5000|Filename = synthetic')
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        opts = dict(tile=(256, 256), compression='jpeg', compressionargs={'level': 85},
                    photometric='rgb', subsampling=(2, 2))
        tw.write(img, description=desc, metadata=None, **opts)
        thumb = img[::32, ::32]
        tw.write(np.ascontiguousarray(thumb), description=f'Aperio Image Library v12.0.15 \r\n{w}x{h} -> {thumb.shape[1]}x{thumb.shape[0]}',
                 photometric='rgb', compression='jpeg', metadata=None)
        lvl = img
        for _ in range(2):
            lvl = lvl[::4, ::4]
            tw.write(np.ascontiguousarray(lvl), description=f'Aperio Image Library v12.0.15 \r\n{w}x{h} -> {lvl.shape[1]}x{lvl.shape[0]}', metadata=None, **opts)
    return img


def make_czi(path: Path, w: int, h: int, f):
    from pylibCZIrw import czi as pyczi
    rng = np.random.default_rng(5)
    tile = 2048
    if path.exists():
        path.unlink()
    with pyczi.create_czi(str(path), exist_ok=True) as czi:
        for y0 in range(0, h, tile):
            y1 = min(h, y0 + tile)
            nuc, tis = render_band(f, y0, y1, w)
            for x0 in range(0, w, tile):
                x1 = min(w, x0 + tile)
                n, t = nuc[:, x0:x1], tis[:, x0:x1]
                dapi = 400 + n * t * 20000 + rng.normal(0, 150, n.shape)
                ring = np.clip(ndimage.uniform_filter(n, 9) - n, 0, 1) * 4
                fitc = 300 + t * 2500 + ring * t * 9000 + rng.normal(0, 150, n.shape)
                gx = (np.arange(x0, x1)[None, :] / w)
                cy5 = 200 + t * gx * 6000 * (n > 0.2) + rng.normal(0, 100, n.shape)
                for c, ch in enumerate((dapi, fitc, cy5)):
                    data = np.clip(ch, 0, 65535).astype(np.uint16)[..., None]
                    czi.write(data=data, plane={'C': c, 'Z': 0, 'T': 0}, location=(x0, y0),
                              compression_options='zstd1:ExplicitLevel=1')
        czi.write_metadata('synthetic', channel_names={0: 'DAPI', 1: 'AF488', 2: 'AF647'},
                           scale_x=0.325e-6, scale_y=0.325e-6)


def cells_geojson(f, path: Path, w: int, h: int, limit: int | None = None):
    p = f['present']
    cx, cy, r, cls = f['cx'][p], f['cy'][p], f['r'][p], f['cls'][p]
    keep = (cx < w) & (cy < h)
    cx, cy, r, cls = cx[keep], cy[keep], r[keep], cls[keep]
    if limit:
        cx, cy, r, cls = cx[:limit], cy[:limit], r[:limit], cls[:limit]
    names = ['Tumor', 'Stroma', 'Immune']
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    feats = []
    for i in range(cx.size):
        ring = np.stack([cx[i] + r[i] * np.cos(ang), cy[i] + r[i] * 0.85 * np.sin(ang)], 1).round(1)
        ring = np.vstack([ring, ring[:1]]).tolist()
        feats.append({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [ring]},
                      'properties': {'classification': {'name': names[cls[i]]}, 'area': float(np.pi * r[i] ** 2 * .85)}})
    # A few large ROIs
    feats.append({'type': 'Feature', 'properties': {'classification': {'name': 'ROI'}, 'name': 'Tumor region'},
                  'geometry': {'type': 'Polygon', 'coordinates': [[[w * .1, h * .1], [w * .45, h * .12], [w * .5, h * .5], [w * .2, h * .45], [w * .1, h * .1]]]}})
    feats.append({'type': 'Feature', 'properties': {'classification': {'name': 'ROI'}, 'name': 'Margin'},
                  'geometry': {'type': 'LineString', 'coordinates': [[w * .55, h * .2], [w * .7, h * .35], [w * .8, h * .3], [w * .9, h * .6]]}})
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')
    return len(feats)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--size', type=int, default=16000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H = a.size, int(a.size * 0.75)
    f = nuclei_field(H, W)
    make_svs(out / 'synthetic_he.svs', W, H, f)
    print('svs ok')
    n = cells_geojson(f, out / 'synthetic_he.geojson', W, H)
    print('geojson features', n)
    cw, chh = W // 2, H // 2
    fc = nuclei_field(chh, cw, seed=7)
    make_czi(out / 'synthetic_if.czi', cw, chh, fc)
    print('czi ok')
