"""Profile the object-overlay pipeline for a 1M-cell slide: server side (parse → layer → compile → encode, memory)
and what one browser must download to show it at fit and at 10×.

    python benchmarks/overlay_profile.py /var/tmp/objbench/cells.parquet
"""

from __future__ import annotations

import gzip
import json
import resource
import sys
import time

import numpy as np


def rss_mb():
    with open('/proc/self/status', encoding='utf-8') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def pyramid_views(c, dpr=1.0):
    """Bytes (raw, gzip) a browser fetches for one view, replicating the viewer's level choice."""
    import math

    pm = c.pyr_meta
    res = {}
    W = max(float(c.bb[:, 2].max()), 1.0)
    H = max(float(c.bb[:, 3].max()), 1.0)
    for label, (pw, ph, scale) in {
        'fit_1600x1000': (1600, 1000, None),
        'fit_400x250 (1 of 16)': (400, 250, None),
        '10x_1600x1000': (1600, 1000, 1.0),
        '10x_400x250': (400, 250, 1.0),
        '2.5x_1600x1000': (1600, 1000, 0.25),
    }.items():
        sc = scale or min(pw / W, ph / H) * 0.96
        sd = sc * dpr
        k_need = math.ceil(math.log2(4 / sd))
        lvl = 0 if k_need <= pm['k0'] else min(pm['levels'] - 1, k_need - pm['k0'])
        k = pm['k0'] + lvl
        span = 2**k * pm['tile_bins']
        cx, cy = (W / 2, H / 2) if scale is None else (float(np.median(c.cx)), float(np.median(c.cy)))
        hw, hh = pw / 2 / sc, ph / 2 / sc
        tiles = [
            (tx, ty)
            for tx in range(max(0, int((cx - hw) // span)), int((cx + hw) // span) + 1)
            for ty in range(max(0, int((cy - hh) // span)), int((cy + hh) // span) + 1)
            if (tx, ty) in c._pyr[lvl]['index']
        ]
        raw = [c.pyramid_tile(lvl, tx, ty) for tx, ty in tiles]
        res[label] = {
            'level': lvl,
            'tiles': len(tiles),
            'points': sum(c._pyr[lvl]['index'][t][1] for t in tiles),
            'KB': round(sum(map(len, raw)) / 1024, 1),
            'gzip_KB': round(sum(len(gzip.compress(b, 3)) for b in raw) / 1024, 1),
        }
    return res


def main(path, shape='auto'):
    from wissi_viz import read_objects

    out = {}
    r0 = rss_mb()
    t = time.perf_counter()
    objs = read_objects(path, mpp=0.25, units='px')
    out['read_s'] = round(time.perf_counter() - t, 2)
    out['rss_after_read_MB'] = round(rss_mb() - r0)
    t = time.perf_counter()
    layer = objs.to_layer(shape=shape)
    out['to_layer_s'] = round(time.perf_counter() - t, 2)
    out['rss_after_layer_MB'] = round(rss_mb() - r0)
    t = time.perf_counter()
    c = layer.compiled()
    out['compile_s'] = round(time.perf_counter() - t, 2)
    out['rss_after_compile_MB'] = round(rss_mb() - r0)
    t = time.perf_counter()
    ov = layer.overview_bytes()
    out['overview_encode_s'] = round(time.perf_counter() - t, 2)
    out['overview_MB'] = round(len(ov) / 2**20, 2)
    t = time.perf_counter()
    gz = gzip.compress(ov, 3)
    out['overview_gzip_MB'] = round(len(gz) / 2**20, 2)
    out['overview_gzip_s'] = round(time.perf_counter() - t, 2)
    out['features'] = int(c.n)
    out['buckets'] = len(c.buckets)
    # detail needed for one 1600×1000 view at 10× (scale = 0.25*4 = 1 → 1600×1000 level-0 px)
    rng = np.random.default_rng(0)
    sizes, times, gzs, nb = [], [], [], []
    bb = np.asarray([b[:4] for b in c.buckets])
    for _ in range(10):
        i = rng.integers(0, c.n)
        x, y = float(c.cx[i]), float(c.cy[i])
        v = (x - 800, y - 500, x + 800, y + 500)
        hit = np.nonzero((bb[:, 0] <= v[2]) & (bb[:, 2] >= v[0]) & (bb[:, 1] <= v[3]) & (bb[:, 3] >= v[1]))[0]
        tb = time.perf_counter()
        data = [c.bucket(int(k)) for k in hit]
        times.append(time.perf_counter() - tb)
        sizes.append(sum(len(d) for d in data))
        gzs.append(sum(len(gzip.compress(d, 3)) for d in data))
        nb.append(len(hit))
        c.release()
    out['view10x_buckets'] = float(np.mean(nb))
    out['view10x_detail_MB'] = round(float(np.mean(sizes)) / 2**20, 2)
    out['view10x_detail_gzip_MB'] = round(float(np.mean(gzs)) / 2**20, 2)
    out['view10x_encode_s'] = round(float(np.mean(times)), 2)
    out['bytes_per_feature_overview'] = round(len(ov) / c.n, 1)
    if c.pyr_meta:
        out['pyramid'] = {k: v for k, v in c.pyr_meta.items() if k != 'slots'}
        out['pyramid_view'] = pyramid_views(c)
    out['rss_total_MB'] = round(rss_mb() - r0)
    print(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[2:3] or []))
