"""Where does the viewer's main thread stall? Real input, Chromium traces, CPU profiles.

Drives the viewer with real mouse input (drag-pan, wheel zoom, flings, hover) over real slides (SVS / OME-TIFF / CZI)
with a 1M-cell object overlay, in 1 and 16 panes, and records:

- **frames**: an independent requestAnimationFrame clock (p50 / p95 / p99 / max interval, % of frames > 33 ms)
- **input → frame**: delay from each pointer / wheel event to the next frame (main-thread queueing)
- **long tasks** (> 50 ms) from PerformanceObserver
- **main-thread breakdown** from a Chromium trace: busy %, JS, style/layout, GC (minor/major, longest pause)
- **JS hotspots** from a CPU profile (self time by function)

    python benchmarks/smoothness.py --slides a.svs,b.ome.tif,c.czi --overlay cells.parquet --out smooth.json
    (slides: benchmarks/make_bench_slides.py; overlay: any object table; --compositor gpu forces GPU-path compositing)
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path

S = 'bench/slides/'  # defaults used for results-smoothness.md
SLIDES = [S + 'bench_he_64k.svs', S + 'bench_if_40k.ome.tif', S + 'bench_if_24k.czi']
OVERLAY = '/var/tmp/objbench/cells_64k.parquet'

INSTRUMENT = r"""
(() => {
  const R = window.__perf = { rec: false, frames: [], lat: [], pending: [], long: [] };
  let last = 0;
  const tick = (t) => {
    if (R.rec) {
      if (last) R.frames.push(t - last);
      const now = performance.now();
      for (const ts of R.pending) R.lat.push(now - ts);
      R.pending.length = 0;
    }
    last = t;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  for (const ev of ['pointermove', 'wheel', 'pointerdown', 'pointerup'])
    addEventListener(ev, (e) => { if (R.rec) R.pending.push(e.timeStamp); }, { capture: true, passive: true });
  try {
    new PerformanceObserver((l) => { if (R.rec) for (const e of l.getEntries()) R.long.push([e.startTime, e.duration]); })
      .observe({ type: 'longtask', buffered: false });
  } catch (e) { /* no long-task API */ }
})();
"""


def pct(v, q):
    import numpy as np

    return float(np.percentile(v, q)) if len(v) else None


def trace_breakdown(raw: bytes, t0_us: float | None = None) -> dict:
    """Main-thread time by event type from a Chromium trace (devtools.timeline)."""
    ev = json.loads(raw)
    ev = ev['traceEvents'] if isinstance(ev, dict) else ev
    main = None
    for e in ev:
        if e.get('name') == 'thread_name' and e.get('args', {}).get('name') == 'CrRendererMain':
            main = (e['pid'], e['tid'])
            break
    tot = defaultdict(float)
    cnt = defaultdict(int)
    gc_max = 0.0
    tasks = []
    ts_min, ts_max = None, None
    for e in ev:
        if main is None or (e.get('pid'), e.get('tid')) != main or e.get('ph') != 'X':
            continue
        d = e.get('dur', 0) / 1000
        n = e['name']
        ts_min = e['ts'] if ts_min is None else min(ts_min, e['ts'])
        ts_max = e['ts'] + e.get('dur', 0) if ts_max is None else max(ts_max, e['ts'] + e.get('dur', 0))
        if n in ('RunTask', 'ThreadControllerImpl::RunTask'):
            tasks.append(d)
        tot[n] += d
        cnt[n] += 1
        if 'GC' in n:
            gc_max = max(gc_max, d)
    wall = ((ts_max or 0) - (ts_min or 0)) / 1000 or 1
    keys = {
        'js': ['FunctionCall', 'EvaluateScript', 'v8.callFunction'],
        'raf': ['FireAnimationFrame'],
        'style': ['UpdateLayoutTree', 'RecalculateStyles'],
        'layout': ['Layout'],
        'paint': ['Paint', 'PrePaint', 'Layerize'],
        'gc': ['MinorGC', 'MajorGC', 'V8.GC_SCAVENGER', 'V8.GC_MARK_COMPACTOR', 'BlinkGC.AtomicPhase'],
        'events': ['EventDispatch'],
        'timers': ['TimerFire'],
        'gpu_upload': ['WebGLRenderingContext::texImage2D'],
    }
    out = {
        'wall_ms': round(wall),
        'busy_pct': round(100 * sum(tasks) / wall, 1),
        'tasks_over_16ms': sum(1 for t in tasks if t > 16.7),
        'tasks_over_50ms': sum(1 for t in tasks if t > 50),
        'longest_task_ms': round(max(tasks), 1) if tasks else 0,
        'gc_longest_ms': round(gc_max, 1),
    }
    for k, names in keys.items():
        out[k + '_ms'] = round(sum(tot[n] for n in names), 1)
    out['top_events'] = sorted(
        ((n, round(v, 1), cnt[n]) for n, v in tot.items() if n not in ('RunTask', 'ThreadControllerImpl::RunTask')),
        key=lambda x: -x[1],
    )[:12]
    return out


def profile_hotspots(prof: dict, top=15) -> list:
    nodes = {n['id']: n for n in prof['nodes']}
    self_t = defaultdict(float)
    samples, deltas = prof.get('samples', []), prof.get('timeDeltas', [])
    for sid, dt in zip(samples, deltas):
        n = nodes[sid]
        cf = n['callFrame']
        url = cf.get('url', '').split('/')[-1]
        name = cf.get('functionName') or '(anonymous)'
        key = f'{name} {url}:{cf.get("lineNumber", -1) + 1}' if url else name
        self_t[key] += dt / 1000
    tot = sum(v for k, v in self_t.items() if k not in ('(idle)', '(program)')) or 1
    return [
        (k, round(v, 1), round(100 * v / tot, 1))
        for k, v in sorted(self_t.items(), key=lambda kv: -kv[1])
        if k not in ('(idle)',)
    ][:top]


def scenario(pg, browser, cdp, name, action, seconds):
    pg.evaluate(
        '() => { const R = window.__perf; R.frames = []; R.lat = []; R.pending = []; R.long = []; R.rec = true; }'
    )
    browser.start_tracing(
        page=pg, categories=['devtools.timeline', 'v8', 'disabled-by-default-devtools.timeline', 'blink.user_timing']
    )
    cdp.send('Profiler.enable')
    cdp.send('Profiler.setSamplingInterval', {'interval': 250})
    cdp.send('Profiler.start')
    js0 = pg.evaluate(
        '() => { grid.jsFrameMs = []; grid.fx.lowres = 0; grid.fx.frames = 0; return grid.counters.requests; }'
    )
    t0 = time.perf_counter()
    action(seconds)
    dt = time.perf_counter() - t0
    prof = cdp.send('Profiler.stop')['profile']
    raw = browser.stop_tracing()
    r = pg.evaluate(
        '() => { const R = window.__perf; R.rec = false; return { frames: R.frames, lat: R.lat, long: R.long, js: grid.jsFrameMs.slice() }; }'
    )
    req = pg.evaluate('() => grid.counters.requests') - js0
    lowres = pg.evaluate('() => [grid.fx.lowres || 0, grid.fx.frames]')
    f, lat = r['frames'], r['lat']
    out = {
        'scenario': name,
        'seconds': round(dt, 1),
        'frames': len(f),
        'fps': round(len(f) / dt, 1),
        'frame_ms': {
            'p50': round(pct(f, 50), 1),
            'p95': round(pct(f, 95), 1),
            'p99': round(pct(f, 99), 1),
            'max': round(max(f), 1),
        }
        if f
        else None,
        'frames_over_33ms_pct': round(100 * sum(1 for x in f if x > 33.4) / max(1, len(f)), 1),
        'input_to_frame_ms': {'p50': round(pct(lat, 50), 1), 'p95': round(pct(lat, 95), 1), 'max': round(max(lat), 1)}
        if lat
        else None,
        'input_events': len(lat),
        'long_tasks': len(r['long']),
        'long_task_ms_total': round(sum(d for _, d in r['long'])),
        'long_task_max_ms': round(max([d for _, d in r['long']] or [0])),
        'viewer_js_ms': {
            'p50': round(pct(r['js'], 50), 2),
            'p95': round(pct(r['js'], 95), 2),
            'max': round(max(r['js']), 1),
        }
        if r['js']
        else {'p50': 0, 'p95': 0, 'max': 0},
        'requests': req,
        'viewer_frames': lowres[1],
        'lowres_frames': lowres[0],
        'trace': trace_breakdown(raw),
        'hotspots': profile_hotspots(prof),
    }
    return out


def main():
    from wissi_viz._compat import utf8_stdio

    utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default='')
    ap.add_argument('--viewport', default='960x600')
    ap.add_argument('--seconds', type=float, default=4)
    ap.add_argument('--panes', default='1,16')
    ap.add_argument('--slides', default=','.join(SLIDES), help='comma-separated slide paths, cycled over the panes')
    ap.add_argument('--overlay', default=OVERLAY, help="object table shown on every 3rd pane ('' for none)")
    ap.add_argument('--layer-cache', default='/var/tmp/layercache')
    ap.add_argument(
        '--compositor',
        choices=('default', 'gpu'),
        default='default',
        help='gpu: force GPU-path compositing (no per-frame canvas readback when there is no real GPU)',
    )
    a = ap.parse_args()
    import uvicorn
    from playwright.sync_api import sync_playwright

    from wissi_viz import SlideRegistry, TileService, cached_object_layer
    from wissi_viz.server import create_app

    reg = SlideRegistry(TileService(memory_bytes=2048 * 2**20), warm=True)
    slides = a.slides.split(',')
    layer = (
        cached_object_layer(a.overlay, a.layer_cache, mpp=0.25, read_kw={'units': 'px'}, name='cells')
        if a.overlay
        else None
    )
    ids = []
    for i in range(16):
        sid = reg.add(slides[i % len(slides)], slide_id=f'p{i}')
        if layer is not None and i % 3 == 0:
            reg.add_annotations(sid, layer)
        ids.append(sid)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    vw, vh = (int(v) for v in a.viewport.split('x'))
    results = {'label': a.label, 'viewport': a.viewport, 'compositor': a.compositor, 'runs': []}
    with sync_playwright() as pw:
        flags = ['--ignore-gpu-blocklist', '--enable-unsafe-swiftshader']
        if a.compositor == 'gpu':
            flags += ['--enable-gpu-compositing', '--enable-gpu-rasterization', '--use-angle=swiftshader']
        browser = pw.chromium.launch(args=flags)
        for n in [int(x) for x in a.panes.split(',')]:
            ctx = browser.new_context(viewport={'width': vw, 'height': vh})
            pg = ctx.new_page()
            pg.add_init_script(INSTRUMENT)
            pg.goto(f'http://127.0.0.1:{port}/demo')
            pg.wait_for_function('() => window.grid')
            pg.evaluate(
                '(ids) => { grid.setAttribute("controls", "false"); grid.setSync("all"); grid.setSlides(ids); }',
                ids[:n],
            )
            pg.wait_for_function('() => grid.panes.every(p => p.view) && grid.stats().inflight === 0', timeout=120000)
            # start at ~5x on tissue with overlays visible, everything loaded
            pg.evaluate('() => grid.goTo("all", 32000, 24000, {magnification: 5, animate: false})')
            pg.wait_for_function(
                '() => grid.stats().inflight === 0 && grid.stats().queuedUploads === 0', timeout=120000
            )
            time.sleep(1)
            cdp = ctx.new_cdp_session(pg)
            box = pg.evaluate(
                '() => { const r = grid.getBoundingClientRect(); const p = grid.panes[0].rect; return [r.left + p.x + p.w / 2, r.top + p.y + p.h / 2]; }'
            )
            cx, cy = box

            def pan(sec):  # continuous drag, ~120 events/s, a slow circle
                import math

                pg.mouse.move(cx, cy)
                pg.mouse.down()
                t0 = time.perf_counter()
                k = 0
                while time.perf_counter() - t0 < sec:
                    k += 1
                    nx, ny = cx + 120 * math.cos(k / 40), cy + 80 * math.sin(k / 40)
                    pg.mouse.move(nx, ny)
                    time.sleep(0.008)
                pg.mouse.up()

            def wheel(sec):  # bursts of wheel ticks in and out
                pg.mouse.move(cx, cy)
                t0 = time.perf_counter()
                k = 0
                while time.perf_counter() - t0 < sec:
                    k += 1
                    pg.mouse.wheel(0, -60 if (k // 25) % 2 == 0 else 60)
                    time.sleep(0.016)

            def fling(sec):  # quick flicks, let them coast
                t0 = time.perf_counter()
                k = 0
                while time.perf_counter() - t0 < sec:
                    k += 1
                    sgn = 1 if k % 2 else -1
                    pg.mouse.move(cx, cy)
                    pg.mouse.down()
                    for j in range(8):
                        pg.mouse.move(cx + sgn * 25 * j, cy + 10 * j)
                        time.sleep(0.01)
                    pg.mouse.up()
                    time.sleep(0.35)

            def hover(sec):  # pointer moving over the panes, no buttons
                import math

                t0 = time.perf_counter()
                k = 0
                while time.perf_counter() - t0 < sec:
                    k += 1
                    pg.mouse.move(vw / 2 + (vw / 2 - 20) * math.cos(k / 30), vh / 2 + (vh / 2 - 20) * math.sin(k / 23))
                    time.sleep(0.008)

            for name, act in (('drag-pan', pan), ('wheel-zoom', wheel), ('fling', fling), ('hover', hover)):
                r = scenario(pg, browser, cdp, name, act, a.seconds)
                r['panes'] = n
                results['runs'].append(r)
                t = r['trace']
                print(
                    f'{n:>2}p {name:<10} fps {r["fps"]:>5} frame p95 {r["frame_ms"]["p95"] if r["frame_ms"] else None:>6} max {r["frame_ms"]["max"] if r["frame_ms"] else None:>6} | '
                    f'input→frame p95 {r["input_to_frame_ms"]["p95"] if r["input_to_frame_ms"] else None} | long {r["long_tasks"]} (max {r["long_task_max_ms"]}) | '
                    f'busy {t["busy_pct"]}% js {t["js_ms"]} style {t["style_ms"]} layout {t["layout_ms"]} gc {t["gc_ms"]} (max {t["gc_longest_ms"]}) | '
                    f'viewer js p95 {r["viewer_js_ms"]["p95"]} | lowres {r["lowres_frames"]}/{r["viewer_frames"]}',
                    flush=True,
                )
                time.sleep(0.5)
            ctx.close()
        browser.close()
    srv.should_exit = True
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
