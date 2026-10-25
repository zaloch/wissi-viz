"""NiceGUI app showing 1–16 images from a tile API (the mock one, or yours via your connector).

    python examples/tile_api/mock_tile_api.py SLIDES... --port 9000 --token secret     # terminal 1 (the "vendor")
    python examples/tile_api/app.py --api http://127.0.0.1:9000 --token secret          # terminal 2 (your app)

The browser never talks to the vendor API: tiles go API → wissi-viz (cache) → browser, so the
vendor token stays on the server and repeat views cost no upstream requests.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from nicegui import run, ui

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connector import DemoApiConnection  # noqa: E402

from wissi_viz.nicegui import WissiViz, configure  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--api', default=os.environ.get('TILE_API', 'http://127.0.0.1:9000'))
ap.add_argument('--token', default=os.environ.get('TILE_API_TOKEN', 'secret'))
ap.add_argument('--port', type=int, default=8100)
ap.add_argument('--cache-dir', default=None, help='persistent tile cache (a Docker volume in production)')
args = ap.parse_args()

registry = configure(cache_dir=args.cache_dir, memory_mb=2048)
conn = DemoApiConnection(args.api, args.token)
IMAGES = conn.list_images()
IDS = {i['id']: conn.register(registry, i['id'], name=i['name']) for i in IMAGES}  # lazy: nothing fetched yet
LABELS = {i['id']: f"{i['id']} · {i['name']}" for i in IMAGES}


@ui.page('/')
def index():
    with ui.row().classes('w-full items-center gap-3'):
        n = ui.select([1, 2, 3, 4, 6, 8, 9, 12, 16], value=4, label='panes').classes('w-24')
        pick = ui.select(LABELS, value=list(LABELS)[:4], multiple=True, label='images').classes('w-[34rem]').props('use-chips')
        sync = ui.select(['none', 'all', 'pan', 'zoom'], value='none', label='sync').classes('w-24')
        mag = ui.select([None, 1.25, 2.5, 5, 10, 20, 40], value=None, label='go to (×)').classes('w-28')
    grid = WissiViz([IDS[i] for i in pick.value], layout='auto').classes('w-full').style('height: calc(100vh - 150px)')
    stats = ui.label().classes('text-xs font-mono opacity-80')

    def apply():
        chosen = list(pick.value)
        k = int(n.value)
        if len(chosen) < k:  # fill the grid with the next images
            chosen += [i for i in LABELS if i not in chosen][:k - len(chosen)]
        grid.set_slides([IDS[i] for i in chosen[:k]])
    n.on_value_change(lambda _: apply())
    pick.on_value_change(lambda _: apply())
    sync.on_value_change(lambda e: grid.set_sync(e.value))

    async def goto(e):
        if e.value:
            views = await grid.get_views()
            for v in views:
                if v:
                    grid.go_to(v['pane'], v['x'], v['y'], magnification=e.value)
    mag.on_value_change(goto)

    async def refresh():
        api = await run.io_bound(lambda: httpx.get(args.api + '/api/_stats', timeout=2).json())
        c, t = conn.stats(), registry.tiles.stats()
        stats.text = (f"upstream: {int(c.get('calls', 0))} calls ({c.get('mean_call_ms', 0)} ms avg), "
                      f"{int(c.get('rate_limited', 0))} rate-limited, {int(c.get('retries', 0))} retried, "
                      f"{int(c.get('not_found', 0))} empty · API saw {api['requests']} requests "
                      f"({api['r429']}×429, {api['r503']}×503) · wissi-viz cache: {t['memory']['hits']} hits / "
                      f"{t['memory']['misses']} misses, {t['memory']['bytes'] / 2**20:.0f} MB")
    ui.timer(2.0, refresh)


ui.run(title='wissi-viz × tile API', port=args.port, reload=False, show=False)
