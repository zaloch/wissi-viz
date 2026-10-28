"""Where does a PathCore-backed wissi-viz server's memory go? Open many different slides one after another (as users
reviewing cases do), each with a cell table from objects(), and record process RSS and every cache after each slide.

    python benchmarks/memory_harness.py --slides 25 --cells 300000 [--disk /tmp/cache] [--revisit 10] --out mem.json
    python benchmarks/memory_harness.py --registry-kw '{"annotation_memory_mb": 400}'     # budgets to try

Results: benchmarks/results-memory.md. The slide behind every connector image is generated once if --svs is missing.
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'scripts'))
import wissi_loadtest as LT  # noqa: E402
from mock_remote import MockPathCore  # noqa: E402

from wissi_viz import SlideRegistry, TileService  # noqa: E402
from wissi_viz.connectors import AnnotationPayload  # noqa: E402
from wissi_viz.server import create_app  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--slides', type=int, default=25)
ap.add_argument('--cells', type=int, default=300_000)
ap.add_argument('--meas', type=int, default=20)
ap.add_argument('--svs', default=str(Path.home() / '.cache/wissi-probe/probe_1_24000x18000.svs'))
ap.add_argument('--registry-kw', default='{}', help='JSON kwargs for SlideRegistry (e.g. new memory options)')
ap.add_argument('--tile-mb', type=int, default=1024)
ap.add_argument('--disk', default=None, help='TileService disk_dir (a production cache_dir)')
ap.add_argument('--revisit', type=int, default=0, help='then reopen this many earlier slides')
ap.add_argument('--out', default=None)
a = ap.parse_args()


def rss_mb():
    from wissi_viz.memory import rss_bytes  # Linux, macOS and Windows

    return rss_bytes() / 2**20


def deep_nbytes(obj, seen=None, depth=0):
    """Bytes held in numpy arrays / bytes reachable from obj (each buffer counted once)."""
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 6:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        base = obj
        while isinstance(base.base, np.ndarray):
            base = base.base
        if id(base) in seen and base is not obj:
            return 0
        seen.add(id(base))
        return base.nbytes if not isinstance(base.base, (bytes, memoryview)) or True else 0
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, dict):
        return sum(deep_nbytes(v, seen, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        if len(obj) > 5000:  # e.g. per-feature property lists: estimate from a sample
            sample = list(obj)[:200]
            return int(sum(deep_nbytes(v, seen, depth + 1) for v in sample) * len(obj) / max(1, len(sample)))
        return sum(deep_nbytes(v, seen, depth + 1) for v in obj)
    d = getattr(obj, '__dict__', None)
    if d is not None and type(obj).__module__.startswith('wissi_viz'):
        return deep_nbytes(d, seen, depth + 1)
    return 0


class PathCoreWithCells(MockPathCore):
    """PathCore-style async connector whose objects() returns a cell table (positions, class, measurements)."""

    async def objects(self, slide_id):
        rng = np.random.default_rng(abs(hash(slide_id)) % 2**32)
        n, W, H = a.cells, self.b.info.width, self.b.info.height
        cx, cy, r = rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(4, 10, n)
        cols = {
            'XMin': cx - r,
            'XMax': cx + r,
            'YMin': cy - r,
            'YMax': cy + r,
            'Classification': rng.choice(['Tumor', 'Stroma', 'Immune', 'Necrosis', 'Other'], n),
        }
        for k in range(a.meas):
            cols[f'Marker {k} Intensity'] = rng.random(n) * 255
        return [AnnotationPayload('Cells', cols, format='table')]


if not Path(a.svs).exists():
    import importlib.util

    spec = importlib.util.spec_from_file_location('wissi_probe', ROOT / 'scripts' / 'wissi_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    Path(a.svs).parent.mkdir(parents=True, exist_ok=True)
    print(f'generating {a.svs} …', flush=True)
    probe.make_svs(Path(a.svs), 24000, 18000, seed=1)
conn = PathCoreWithCells(a.svs)
reg = SlideRegistry(TileService(memory_bytes=a.tile_mb * 2**20, disk_dir=a.disk), **json.loads(a.registry_kw))
ids = [conn.register(reg, f'S-{k:03d}') for k in range(a.slides)]
from starlette.testclient import TestClient  # noqa: E402

cli = TestClient(create_app(reg))
B = '/wissi-viz'
rng = random.Random(0)
rows = []
r0 = rss_mb()
print(f'start RSS {r0:.0f} MB', flush=True)


def visit(k, sid):
    t0 = time.perf_counter()
    info = cli.get(f'{B}/slides/{sid}/info.json').json()
    metas = cli.get(f'{B}/slides/{sid}/annotations').json()  # waits for annotations + objects
    # image tiles: overview + a zoomed region (as one user session does)
    lv = info['levels']
    for L in (len(lv) - 1, len(lv) // 2, 0):
        ts = info['tile_size']
        nx, ny = -(-lv[L]['width'] // ts), -(-lv[L]['height'] // ts)
        for _ in range(12):
            cli.get(f'{B}/slides/{sid}/tiles/{L}/{rng.randrange(nx)}/{rng.randrange(ny)}?v={info["version"]}')
    # overlays: index, pyramid tiles at two levels around a few places, outline buckets
    for m in metas:
        q = f'layer={quote(m["name"])}&v={m["version"]}'
        buf = cli.get(f'{B}/slides/{sid}/annotations/overview?{q}').content
        idx = LT.parse_index(buf, m)
        if idx['pyr']:
            for lvl in (0, 1, 2):
                if lvl < len(idx['tiles']):
                    tiles = sorted(idx['tiles'][lvl])
                    for tx, ty in rng.sample(tiles, min(40, len(tiles))):
                        cli.get(f'{B}/slides/{sid}/annotations/ptile/{lvl}/{tx}/{ty}?{q}')
        for i in rng.sample(range(len(idx['buckets'])), min(30, len(idx['buckets']))):
            cli.get(f'{B}/slides/{sid}/annotations/bucket/{i}?{q}')
    gc.collect()
    m = reg.memory_stats()
    row = {
        'slides_opened': k + 1,
        'rss_mb': round(rss_mb()),
        'tile_cache_mb': m['tile_cache_mb'],
        'decoded_mb': m['decoded_mb'],
        'rss_anon_mb': m.get('rss_anon_mb'),
        'rss_file_mb': m.get('rss_file_mb'),
        'overlay_mb': m.get('overlay_mb'),
        'auto_layers': m['auto_layers'],
        'auto_layers_ram_mb': m['auto_layers_ram_mb'],
        'mapped_mb': m.get('auto_layers_mapped_mb'),
        'evictions': m['layer_evictions'],
        'disk_hits': m['layer_disk_hits'],
        'calls': conn.stats().get('calls'),
        'sec': round(time.perf_counter() - t0, 1),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)


for k, sid in enumerate(ids):
    visit(k, sid)
for j in range(a.revisit):
    visit(len(ids) + j, ids[j % len(ids)])
if a.out:
    Path(a.out).write_text(json.dumps(rows, indent=1), encoding='utf-8')
