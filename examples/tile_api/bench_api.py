"""Benchmark 1–16 panes of API-backed slides: cold (every tile comes from the vendor API, with
latency/429s/503s) vs warm (same views again → wissi-viz cache, no upstream traffic).

    python examples/tile_api/mock_tile_api.py SLIDES... --port 9000      # the vendor, separate process
    python examples/tile_api/bench_api.py --api http://127.0.0.1:9000 --panes 1,4,9,16 --out api_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connector import DemoApiConnection  # noqa: E402

from wissi_viz import SlideRegistry, TileService  # noqa: E402
from wissi_viz.bench import browser_bench, format_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--api', default='http://127.0.0.1:9000')
ap.add_argument('--token', default='secret')
ap.add_argument('--panes', default='1,4,9,16')
ap.add_argument('--seconds', type=float, default=6)
ap.add_argument('--magnification', type=float, default=10)
ap.add_argument('--viewport', default='1600x1000')
ap.add_argument('--headed', action='store_true')
ap.add_argument('--http2', action='store_true', help='serve wissi-viz over HTTP/2 (TLS), as behind a production proxy')
ap.add_argument('--out', default=None)
a = ap.parse_args()
vp = tuple(int(v) for v in a.viewport.split('x'))
panes = [int(x) for x in a.panes.split(',')]

conn = DemoApiConnection(a.api, a.token)
reg = SlideRegistry(TileService(memory_bytes=3072 * 2**20, workers=16), warm=True)
ids = [conn.register(reg, i['id'], name=i['name']) for i in conn.list_images()]
api0 = httpx.get(a.api + '/api/_stats').json()
results = {}
for phase in ('cold', 'warm'):
    print(f'--- {phase}: ' + ('tiles come from the vendor API' if phase == 'cold' else 'same views again (wissi-viz cache)'))
    before = httpx.get(a.api + '/api/_stats').json()
    r = browser_bench(registry=reg, slide_ids=ids, panes=panes, pan_seconds=a.seconds, magnification=a.magnification,
                      viewport=vp, headed=a.headed, http2=a.http2)
    after = httpx.get(a.api + '/api/_stats').json()
    r['upstream'] = {k: after[k] - before[k] for k in ('requests', 'tiles', 'r429', 'r503', 'r404', 'bytes')}
    r['upstream']['MB'] = round(r['upstream'].pop('bytes') / 2**20, 1)
    results[phase] = r
    print(format_report(None, r))
    print('upstream during this phase:', r['upstream'])
results['connector'] = conn.stats()
print('\nconnector totals:', results['connector'])
if a.out:
    Path(a.out).write_text(json.dumps(results, indent=2, default=str))
