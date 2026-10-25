"""Build full-size benchmark slides from a real H&E texture (mirror-tiled, so the codec sees real
tissue entropy; ~35 % background like a typical biopsy section).

    python benchmarks/make_bench_slides.py SOURCE_H&E.svs OUTDIR [--scale 1.0]

Writes:
  bench_he_64k.svs         64000×48000 RGB, JPEG Q80, 256² tiles, Aperio pyramid 1/4/16/64
  bench_if_40k.ome.tif     40000×30000 ×3 ch uint16, zlib, 512² tiles, SubIFD pyramid ÷2 …
  bench_if_24k.czi         24000×18000 ×3 ch uint16, zstd, 2048² mosaic (10 % overlap) + pyramid subblocks ÷2 …
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))


def load_texture(path: str) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from wissi_viz.sources import open_slide
    s = open_slide(path)
    lv = s.info.levels[0]
    return s.read_region(0, 0, 0, lv.width, lv.height)


class Mirror:
    """Infinite mirror-tiled view of a texture + a smooth tissue mask; sample(x0, y0, w, h, step)."""

    def __init__(self, tex: np.ndarray, W: int, H: int, seed: int = 0):
        self.tex = tex
        self.th, self.tw = tex.shape[:2]
        rng = np.random.default_rng(seed)
        g = rng.random((H // 4096 + 3, W // 4096 + 3)).astype(np.float32)
        from scipy import ndimage
        self.mask_lo = ndimage.gaussian_filter(g, 0.8)
        self.mask_scale = 4096
        self.thr = np.quantile(self.mask_lo, 0.35)

    def _mirror(self, v, n):
        v = np.mod(v, 2 * n)
        return np.where(v < n, v, 2 * n - 1 - v)

    def sample(self, x0, y0, w, h, step=1):
        ys = self._mirror(np.arange(y0, y0 + h * step, step)[:h], self.th)
        xs = self._mirror(np.arange(x0, x0 + w * step, step)[:w], self.tw)
        rgb = self.tex[ys[:, None], xs[None, :]]
        my = np.clip((np.arange(y0, y0 + h * step, step)[:h] / self.mask_scale), 0, self.mask_lo.shape[0] - 1.001)
        mx = np.clip((np.arange(x0, x0 + w * step, step)[:w] / self.mask_scale), 0, self.mask_lo.shape[1] - 1.001)
        from scipy import ndimage
        m = ndimage.map_coordinates(self.mask_lo, np.meshgrid(my, mx, indexing='ij'), order=1)
        t = np.clip((m - self.thr) * 12 + 0.5, 0, 1)[..., None]
        return (rgb * t + 245 * (1 - t)).astype(np.uint8)


_MINV = None


def he_to_if_channel(rgb: np.ndarray, c: int) -> np.ndarray:
    """One pseudo-fluorescence channel (cheaper than he_to_if when channels are written plane by plane)."""
    global _MINV
    if _MINV is None:
        M = np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]])
        _MINV = np.linalg.inv(M / np.linalg.norm(M, axis=1, keepdims=True)).astype(np.float32)
    od = -np.log((rgb.astype(np.float32) + 1) / 256)
    k = 1 if c == 1 else 0
    conc = np.clip(od @ _MINV[:, k], 0, None)
    if c == 0:
        v = conc * 9000 + 120
    elif c == 1:
        v = conc * 5000 + 200
    else:
        v = np.clip(conc - 0.9, 0, None) * 20000 + 80
    return v.clip(0, 65535).astype(np.uint16)


def he_to_if(rgb: np.ndarray) -> np.ndarray:
    """Pseudo-fluorescence from H&E: hematoxylin → DAPI, eosin → AF488, a blob marker → AF647."""
    od = -np.log((rgb.astype(np.float32) + 1) / 256)
    M = np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    c = od.reshape(-1, 3) @ np.linalg.inv(M)
    c = np.clip(c, 0, None).reshape(rgb.shape)
    dapi = c[..., 0] * 9000 + 120
    fitc = c[..., 1] * 5000 + 200
    cy5 = np.clip(c[..., 0] - 0.9, 0, None) * 20000 + 80
    return np.stack([dapi, fitc, cy5]).clip(0, 65535).astype(np.uint16)


def make_svs(out: Path, src: Mirror, W: int, H: int, mpp: float = 0.25):
    import tifffile
    T = 256

    def tiles(step, w, h):
        for y in range(0, h, T):
            for x in range(0, w, T):
                yield src.sample(x * step, y * step, T, T, step)
    desc = f'Aperio Image Library v12.0.16\r\n{W}x{H} [0,0 {W}x{H}] ({T}x{T}) JPEG/RGB Q=80|AppMag = 40|MPP = {mpp}'
    opts = dict(tile=(T, T), compression='jpeg', compressionargs={'level': 80}, photometric='rgb', metadata=None)
    with tifffile.TiffWriter(out, bigtiff=True) as tw:
        tw.write(tiles(1, W, H), shape=(H, W, 3), dtype=np.uint8, description=desc, **opts)
        th = src.sample(0, 0, W // 64, H // 64, 64)
        tw.write(th, description=f'Aperio Image Library v12.0.16\r\n{W}x{H} -> {th.shape[1]}x{th.shape[0]}',
                 photometric='rgb', compression='jpeg', metadata=None)
        for f in (4, 16, 64):
            w, h = W // f, H // f
            tw.write(tiles(f, w, h), shape=(h, w, 3), dtype=np.uint8,
                     description=f'Aperio Image Library v12.0.16\r\n{W}x{H} -> {w}x{h}', **opts)


def make_ome(out: Path, src: Mirror, W: int, H: int, mpp: float = 0.325):
    import tifffile
    T = 512
    nlev = 1
    while max(W, H) >> nlev > 1024:
        nlev += 1

    def tiles(step, w, h):
        for c in range(3):
            for y in range(0, h, T):
                for x in range(0, w, T):
                    yield he_to_if_channel(src.sample(x * step, y * step, T, T, step), c)
    meta = {'axes': 'CYX', 'Channel': {'Name': ['DAPI', 'AF488', 'AF647'], 'Color': [-16776961, 16711935, -65281]},
            'PhysicalSizeX': mpp, 'PhysicalSizeY': mpp}
    opts = dict(tile=(T, T), compression='zlib', compressionargs={'level': 1}, photometric='minisblack')
    with tifffile.TiffWriter(out, bigtiff=True, ome=True) as tw:
        tw.write(tiles(1, W, H), shape=(3, H, W), dtype=np.uint16, subifds=nlev - 1, metadata=meta, **opts)
        for k in range(1, nlev):
            f = 2 ** k
            tw.write(tiles(f, W // f, H // f), shape=(3, H // f, W // f), dtype=np.uint16, subfiletype=1, **opts)


def make_czi(out: Path, src: Mirror, W: int, H: int, mpp: float = 0.325):
    from czi_writer import write_czi_stream
    T, ov = 2048, 205
    nlev = 1
    while max(W, H) >> nlev > 1024:
        nlev += 1

    def blocks():
        m = 0
        for s in [2 ** k for k in range(nlev)]:
            step = T - ov if s == 1 else T * s
            for y in range(0, H, step):
                for x in range(0, W, step):
                    w, h = min(T * s, W - x), min(T * s, H - y)
                    w -= w % s
                    h -= h % s
                    if w <= 0 or h <= 0:
                        continue
                    chans = he_to_if(src.sample(x, y, w // s, h // s, s))
                    for c in range(3):
                        yield dict(c=c, x=x, y=y, w=w, h=h, data=chans[c], m=m, pyramid=s > 1)
                    m += 1
    write_czi_stream(out, blocks(), ['DAPI', 'AF488', 'AF647'], mpp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('out')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--only', default='svs,ome,czi')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tex = load_texture(a.source)
    s = a.scale
    jobs = {'svs': ('bench_he_64k.svs', make_svs, int(64000 * s), int(48000 * s)),
            'ome': ('bench_if_40k.ome.tif', make_ome, int(40000 * s), int(30000 * s)),
            'czi': ('bench_if_24k.czi', make_czi, int(24000 * s), int(18000 * s))}
    for k in a.only.split(','):
        name, fn, W, H = jobs[k]
        t = time.time()
        fn(out / name, Mirror(tex, W, H, seed=len(name)), W, H)
        print(f'{name}: {W}x{H}  {(out / name).stat().st_size / 2**20:.0f} MB  {time.time() - t:.0f} s', flush=True)
