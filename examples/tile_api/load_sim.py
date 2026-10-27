"""Simulate 10–100 users on a wissi-viz deployment whose slides come from a vendor tile API (or local files).

The pieces run as three processes, as they would in production:

    load generator (scripts/wissi_loadtest.py) ──HTTP──▶ wissi-viz server ──DemoApiConnection──▶ mock vendor API
         N virtual users, each ≤ R requests                caches, retries, 502 on upstream failure    latency, 429, 503, 404

    python examples/tile_api/load_sim.py --backend api   --users 10,25,50,75,100 --requests 200 --out load_api.json
    python examples/tile_api/load_sim.py --backend api   --vendor-errors 0.05 --vendor-limit 8 ...   # a struggling vendor
    python examples/tile_api/load_sim.py --backend files --files a.svs b.czi ...                      # no vendor, local files

For a real deployment, skip this file and run scripts/wissi_loadtest.py against your server.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 90) -> None:
    import httpx
    t = time.time()
    while time.time() - t < timeout:
        try:
            if httpx.get(url, timeout=2).status_code < 500:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    raise RuntimeError(f'{url} did not come up')


def serve(args) -> None:
    """(child process) the wissi-viz server: registry + caches, slides from the vendor API or local files."""
    import uvicorn
    from fastapi.responses import JSONResponse

    from wissi_viz import SlideRegistry, TileService
    from wissi_viz.server import create_app
    sys.path.insert(0, str(HERE))
    reg = SlideRegistry(TileService(memory_bytes=args.cache_mb * 2**20, workers=args.workers), warm=False)
    conn = None
    if args.backend == 'synthetic':   # overlay-focused: 16 blank 100k × 80k slides (image tiles cost ~nothing)
        import numpy as np

        from wissi_viz import Level, SlideInfo
        from wissi_viz.sources.callback import CallbackSource
        W, H = 100_000, 80_000
        info = SlideInfo(width=W, height=H, levels=[Level(-(-W // 2**k), -(-H // 2**k), float(2**k)) for k in range(10)],
                         tile_size=512, mpp=(0.25, 0.25))
        blank = np.full((512, 512, 3), 235, np.uint8)
        for i in range(16):
            reg.add(CallbackSource(info, read_tile=lambda L, x, y, c: blank), slide_id=f'syn{i}')
    elif args.backend == 'api':
        from connector import DemoApiConnection
        conn = DemoApiConnection(args.vendor_url, args.token)
        for im in conn.list_images():
            conn.register(reg, im['id'], name=im['name'])
    else:
        for i, f in enumerate(args.files):
            reg.add(f, slide_id=f's{i}')
    if args.overlay_table:   # the same object table on every slide, compiled once (memory-mapped, shared)
        from wissi_viz import cached_object_layer
        layer = cached_object_layer(args.overlay_table, args.layer_cache, mpp=0.25, read_kw={'units': 'px'}, name='cells')
        for sid in reg.ids():
            reg.add_annotations(sid, layer)
    app = create_app(reg)

    @app.get('/_sim/stats')
    def sim_stats():
        return JSONResponse({'tiles': reg.tiles.stats(), 'connector': conn.stats() if conn else {}})
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning', timeout_keep_alive=30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--backend', choices=['api', 'files', 'synthetic'], default='api')
    ap.add_argument('--overlay-table', default=None, help='object table (parquet/csv/…) shown on every slide; users fetch overlays')
    ap.add_argument('--layer-cache', default=os.path.join(tempfile.gettempdir(), 'wissi-layer-cache'), help='compiled overlay cache directory')
    ap.add_argument('--overlay-mode', choices=['server', 'client'], default='server', help='how simulated viewers get overlays')
    ap.add_argument('--files', nargs='*', default=[], help='slide files (the vendor\'s images, or served directly)')
    ap.add_argument('--users', default='10,25,50,75,100')
    ap.add_argument('--requests', type=int, default=200)
    ap.add_argument('--panes', type=int, default=4)
    ap.add_argument('--conn', type=int, default=6)
    ap.add_argument('--think', default='0.3,1.5')
    ap.add_argument('--vendor-latency', default='40,120')
    ap.add_argument('--vendor-limit', type=int, default=12)
    ap.add_argument('--vendor-errors', type=float, default=0.01)
    ap.add_argument('--cache-mb', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--out', default=None)
    # internal
    ap.add_argument('--_serve', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--port', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--vendor-url', default='', help=argparse.SUPPRESS)
    ap.add_argument('--token', default='secret', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._serve:
        return serve(args)
    if not args.files and args.backend != 'synthetic':
        sys.exit('pass --files (slide files)')
    procs = []
    try:
        vendor_url = ''
        if args.backend == 'api':
            vp = free_port()
            vendor_url = f'http://127.0.0.1:{vp}'
            procs.append(subprocess.Popen([sys.executable, str(HERE / 'mock_tile_api.py'), *args.files, '--port', str(vp),
                                           '--token', args.token, '--latency', args.vendor_latency,
                                           '--limit', str(args.vendor_limit), '--errors', str(args.vendor_errors)],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
            wait_http(f'{vendor_url}/api/_stats')
        sp = free_port()
        procs.append(subprocess.Popen([sys.executable, __file__, '--_serve', '--backend', args.backend, '--port', str(sp),
                                       '--vendor-url', vendor_url, '--token', args.token, '--cache-mb', str(args.cache_mb),
                                       '--workers', str(args.workers), '--layer-cache', args.layer_cache,
                                       *(['--overlay-table', args.overlay_table] if args.overlay_table else []),
                                       '--files', *args.files],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        base = f'http://127.0.0.1:{sp}/wissi-viz'
        wait_http(f'{base}/slides')
        import wissi_loadtest
        argv = [base, '--users', args.users, '--requests', str(args.requests), '--panes', str(args.panes),
                '--conn', str(args.conn), '--think', args.think, '--procs', str(args.procs),
                '--stats-url', f'http://127.0.0.1:{sp}/_sim/stats'] + (['--overlays', '--overlay-mode', args.overlay_mode] if args.overlay_table else [])
        if vendor_url:
            argv += ['--stats-url', f'{vendor_url}/api/_stats']
        if args.out:
            argv += ['--out', args.out]
        stages, _ = wissi_loadtest.main(argv)
        # upstream summary per stage
        print('\nupstream per stage (vendor calls, retries, rate-limited, 503s answered by the vendor):', file=sys.stderr)
        for s in stages:
            sv = s.get('server') or {}
            c = sv.get(f'http://127.0.0.1:{sp}/_sim/stats', {}).get('connector', {})
            v = sv.get(f'{vendor_url}/api/_stats', {}) if vendor_url else {}
            print(f"  {s['users']:>4} users: connector {json.dumps(c)}  vendor {json.dumps(v)}", file=sys.stderr)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except Exception:  # noqa: BLE001
                p.kill()


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
