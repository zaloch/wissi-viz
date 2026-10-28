"""Object overlays at scale, legacy vs new: one browser, 1 and 16 panes, a 1M-cell layer on every slide.

Server side: build time and memory. Browser side:
- **ready**: time until every pane shows its overlay at fit;
- **transfer**: annotation bytes actually sent (gzip, counted by server middleware);
- GPU memory held for overlays;
- frame time while panning at fit;
- zoom steps (2.5×, 10×, 40×): time and bytes per step.

    python benchmarks/overlay_bench.py /var/tmp/objbench/cells.parquet --mode new|legacy --out o.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table')
    ap.add_argument('--mode', choices=['new', 'legacy'], default='new')
    ap.add_argument('--panes', default='1,16')
    ap.add_argument(
        '--overlays', choices=['server', 'client'], default='server', help='new mode: where the pyramid is built'
    )
    ap.add_argument('--viewport', default='1600x1000')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.mode == 'legacy':
        os.environ['WISSI_LEGACY_OVERLAYS'] = '1'
    import numpy as np
    import uvicorn
    from playwright.sync_api import sync_playwright

    from wissi_viz import Level, SlideInfo, SlideRegistry, read_objects
    from wissi_viz.server import create_app
    from wissi_viz.sources.callback import CallbackSource

    def rss():
        return (
            int([x for x in open('/proc/self/status', encoding='utf-8') if x.startswith('VmRSS')][0].split()[1]) / 1024
        )

    W, H = 100_000, 80_000
    info = SlideInfo(
        width=W,
        height=H,
        levels=[Level(-(-W // 2**k), -(-H // 2**k), float(2**k)) for k in range(10)],
        tile_size=512,
        mpp=(0.25, 0.25),
    )
    blank = np.full((512, 512, 3), 235, np.uint8)
    reg = SlideRegistry(warm=False)
    panes = [int(x) for x in a.panes.split(',')]
    ids = [
        reg.add(CallbackSource(info, read_tile=lambda L, x, y, c: blank), slide_id=f's{i}') for i in range(max(panes))
    ]
    res = {'mode': a.mode if a.mode == 'legacy' else f'new/{a.overlays}'}
    r0 = rss()
    t = time.perf_counter()
    objs = read_objects(a.table, mpp=0.25, units='px')
    res['server_read_s'] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    layer = objs.to_layer(name='cells')
    res['server_layer_s'] = round(time.perf_counter() - t, 2)
    del objs
    t = time.perf_counter()
    layer.compiled()
    res['server_compile_s'] = round(time.perf_counter() - t, 2)
    import gc

    gc.collect()
    res['server_rss_MB'] = round(rss() - r0)
    for sid in ids:
        reg.add_annotations(sid, layer)

    app = create_app(reg)
    counted = {'ann': 0, 'ann_req': 0}

    class Count:  # ASGI middleware: bytes actually sent for /annotations/ responses
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http' or '/annotations' not in scope['path']:
                return await self.app(scope, receive, send)
            counted['ann_req'] += 1

            async def s2(msg):
                if msg['type'] == 'http.response.body':
                    counted['ann'] += len(msg.get('body', b''))
                await send(msg)

            return await self.app(scope, receive, s2)

    with socket.socket() as s_:
        s_.bind(('127.0.0.1', 0))
        port = s_.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(Count(app), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    vw, vh = (int(v) for v in a.viewport.split('x'))
    idle = """() => { const g = grid; if (g.stats().inflight || g.stats().queuedUploads || g._pendingWork) return false;
        return g.panes.every(p => { const L = p.slide && p.slide.layers.get('cells'); return L && L.state === 'ready' && (!L.client || L.client === 'ready'); }); }"""
    res['runs'] = []
    with sync_playwright() as pw:
        for n in panes:
            b = pw.chromium.launch(
                args=['--ignore-gpu-blocklist', '--enable-unsafe-swiftshader', '--enable-precise-memory-info']
            )
            pg = b.new_page(viewport={'width': vw, 'height': vh})
            pg.goto(f'http://127.0.0.1:{port}/demo')
            pg.wait_for_function('() => window.grid')
            run = {'panes': n}

            def until_idle(timeout=120):
                t0 = time.perf_counter()
                pg.wait_for_function(idle, timeout=timeout * 1000, polling=50)
                # idle must hold for 300 ms (the scheduler may still be about to request)
                time.sleep(0.3)
                pg.wait_for_function(idle, timeout=timeout * 1000, polling=50)
                return time.perf_counter() - t0

            c0 = dict(counted)
            t0 = time.perf_counter()
            pg.evaluate(
                '([ids, om]) => { grid.setAttribute("controls", "false"); grid.setLod({overlays: om}); grid.setSync("all"); grid.setSlides(ids); }',
                [ids[:n], a.overlays],
            )
            until_idle()
            run['ready_ms'] = round(1000 * (time.perf_counter() - t0))
            run['fit_transfer_MB'] = round((counted['ann'] - c0['ann']) / 2**20, 2)
            run['fit_requests'] = counted['ann_req'] - c0['ann_req']
            run['gpu_overlay_MB'] = pg.evaluate('() => grid.stats().annotationMB')
            run['js_heap_MB'] = pg.evaluate(
                '() => performance.memory ? Math.round(performance.memory.usedJSHeapSize / 2 ** 20) : null'
            )
            run['client_build_ms'] = pg.evaluate(
                "() => { const b = grid.panes.map(p => p.slide.layers.get('cells').buildMs).filter(Boolean); return b.length ? Math.round(b.reduce((x, y) => x + y) / b.length) : null; }"
            )
            click = """async ([x, y]) => { const p = grid.panes[0], r = p.rect, el = grid.$canvas.getBoundingClientRect();
                const sx = r.x + r.w / 2 + (x - p.view.cx) * p.view.scale, sy = r.y + r.h / 2 + (y - p.view.cy) * p.view.scale;
                const t0 = performance.now();
                const got = new Promise((res) => { grid.addEventListener('objectclick', (e) => res(e.detail), { once: true }); setTimeout(() => res(null), 10000); });
                grid._click(p, { clientX: el.left + sx, clientY: el.top + sy, button: 0 });
                const d = await got; return d ? Math.round(performance.now() - t0) : null; }"""
            hit = pg.evaluate(
                "async () => (await fetch(grid.panes[0].slide.url + '/annotations/hit?layer=cells&x=52000&y=41000&tol=2000')).json()"
            )
            target = [hit['cx'], hit['cy']] if hit.get('index') is not None else [52000, 41000]
            run['click_fit_ms'] = pg.evaluate(click, target)
            # pan at fit: frame cost of drawing every pane's overlay
            fr = pg.evaluate("""async () => { const g = grid, out = [], js0 = (g.jsFrameMs = []);
                let last = performance.now(); const t0 = last;
                while (performance.now() - t0 < 3000) {
                  await new Promise((r) => requestAnimationFrame(r));
                  for (const p of g.panes) if (p.view) p.view.cx += 30 / p.view.scale * 0.016;
                  g.requestRender();
                  const t = performance.now(); out.push(t - last); last = t; }
                out.sort((a, b) => a - b); const js = g.jsFrameMs.slice().sort((a, b) => a - b);
                return { p50: out[out.length >> 1], p95: out[Math.floor(out.length * 0.95)], js50: js[js.length >> 1], js95: js[Math.floor(js.length * 0.95)] }; }""")
            run['pan_frame_ms'] = {k: round(v, 1) for k, v in fr.items()}
            run['zoom'] = []
            for mag in (2.5, 10, 40):
                c0 = dict(counted)
                t0 = time.perf_counter()
                pg.evaluate(f'() => grid.goTo("all", 52000, 41000, {{magnification: {mag}, animate: false}})')
                until_idle()
                run['zoom'].append(
                    {
                        'mag': mag,
                        'ms': round(1000 * (time.perf_counter() - t0)),
                        'transfer_MB': round((counted['ann'] - c0['ann']) / 2**20, 3),
                        'requests': counted['ann_req'] - c0['ann_req'],
                    }
                )
            pg.evaluate(f'() => grid.goTo("all", {target[0]}, {target[1]}, {{magnification: 40, animate: false}})')
            until_idle()
            run['click_40x_ms'] = pg.evaluate(click, target)
            run['gpu_overlay_MB_end'] = pg.evaluate('() => grid.stats().annotationMB')
            run['js_heap_MB_end'] = pg.evaluate(
                '() => performance.memory ? Math.round(performance.memory.usedJSHeapSize / 2 ** 20) : null'
            )
            res['runs'].append(run)
            print(json.dumps(run), flush=True)
            b.close()
    srv.should_exit = True
    print(json.dumps({k: v for k, v in res.items() if k != 'runs'}))
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.exit(main())
