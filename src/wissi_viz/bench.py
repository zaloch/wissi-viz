"""Benchmarks: slide opening, tile serving, and in-browser panning with 1–16 panes.

    wissi-viz bench SLIDE [SLIDE ...] [--panes 1,4,9,16] [--server-only] [--headed] [--out report.json]
    wissi-viz fetch-samples DIR [--set small|standard|full] [--max-mb 2000]

Server stage (Python, no browser)
  open        time to open the file and build the pyramid (first open in this process)
  cold tile   produce one 512² tile that is in no cache: read + decode + re-encode, per level (p50/p95)
  throughput  tiles/s with N concurrent requests (what panning several panes generates)
  warm tile   the same tiles again, served from the memory cache

Browser stage (Chromium via Playwright, the real <wissi-viz> element, real HTTP)
  open        setSlides → every pane fully sharp at the fit view
  zoom        jump to 20× → every pane fully sharp
  pan         continuous scripted pan (all panes, speed in screen px/s): frame times from
              requestAnimationFrame (p50/p95/p99, % frames > 33 ms), how much of the view is
              sharp / blank while moving, then time to fully sharp after stopping
  revisit     the same pan again → browser/server caches warm

The WebGL renderer string is reported: "SwiftShader" means *software* rendering (no GPU,
e.g. a headless server/container); a vendor string (NVIDIA, AMD, Apple, Intel, ANGLE (…)) means
the GPU is used. Run with --headed on a desktop to measure your real GPU.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _pct(v: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(v, float), q)) if len(v) else float('nan')


# ============================================================================= server stage
def server_bench(paths: Sequence[str], n_tiles: int = 48, concurrency: int = 8, seed: int = 0) -> list[dict]:
    from .registry import SlideRegistry
    from .tiles import TileService
    rng = random.Random(seed)
    out = []
    for path in paths:
        tiles = TileService(memory_bytes=512 * 2**20, workers=max(concurrency, 4))
        reg = SlideRegistry(tiles, warm=False)
        t0 = time.perf_counter()
        sid = reg.add(path)
        e, pyr = reg.get(sid)
        open_ms = (time.perf_counter() - t0) * 1000
        info = pyr.info
        chans = [None] if info.mode == 'rgb' else list(range(min(3, len(info.channels))))
        res: dict[str, Any] = {'slide': Path(path).name, 'reader': type(e.source).__name__,
                               'size': f'{info.width}x{info.height}', 'mode': info.mode, 'levels': len(info.levels),
                               'file_MB': round(os.path.getsize(path) / 2**20, 1) if os.path.isfile(path) else None,
                               'open_ms': round(open_ms, 1)}
        nl = len(info.levels)
        picks = sorted({0, nl // 2, nl - 1})
        per_level = []
        for L in picks:
            lv = info.levels[L]
            nx, ny = math.ceil(lv.width / info.tile_size), math.ceil(lv.height / info.tile_size)
            pool = [(tx, ty, c) for tx in range(nx) for ty in range(ny) for c in chans]
            rng.shuffle(pool)
            coords = pool[: 2 * n_tiles]
            cold_set, tput_set = coords[: max(4, len(coords) // 3)], coords[max(4, len(coords) // 3):]
            lat, sizes = [], []
            for tx, ty, c in cold_set:
                t = time.perf_counter()
                data, _ = tiles.get(sid, pyr, e.source.version, L, tx, ty, c).result()
                lat.append((time.perf_counter() - t) * 1000)
                sizes.append(len(data))
            tput = None
            if len(tput_set) >= 8:  # only distinct, never-requested tiles count
                t = time.perf_counter()
                futs = [tiles.get(sid, pyr, e.source.version, L, tx, ty, c) for tx, ty, c in tput_set]
                for f in futs:
                    f.result()
                tput = round(len(tput_set) / (time.perf_counter() - t), 1)
            warm = []
            for tx, ty, c in coords[:16]:
                t1 = time.perf_counter()
                tiles.get(sid, pyr, e.source.version, L, tx, ty, c).result()
                warm.append((time.perf_counter() - t1) * 1000)
            per_level.append({'level': L, 'downsample': round(lv.downsample, 2), 'cold_p50_ms': round(_pct(lat, 50), 2),
                              'cold_p95_ms': round(_pct(lat, 95), 2), f'throughput_x{concurrency}_tiles_s': tput,
                              'n_tiles': len(cold_set) + len(tput_set), 'warm_p50_ms': round(_pct(warm, 50), 3),
                              'tile_KB': round(statistics.mean(sizes) / 1024, 1) if sizes else None})
        res['levels_bench'] = per_level
        out.append(res)
        reg.remove(sid)
    return out


# ============================================================================= browser stage
_PAN_JS = """
async ([seconds, speed, sampleEvery]) => {
  const g = document.querySelector('wissi-viz');
  const frames = [], sharp = [], blank = [];
  const c0 = {...g.counters};
  g.jsFrameMs = [];
  let last = null, t0 = null, k = 0;
  await new Promise((resolve) => {
    const step = (ts) => {
      if (t0 === null) { t0 = ts; last = ts; }
      const dt = ts - last; last = ts;
      if (dt > 0) frames.push(dt);
      const t = (ts - t0) / 1000;
      // Lissajous path in screen space, the same for every pane
      const vx = speed * Math.cos(t * 0.9), vy = speed * 0.6 * Math.sin(t * 1.3);
      for (const p of g.panes) if (p.view) { p.view.cx += vx * dt / 1000 / p.view.scale; p.view.cy += vy * dt / 1000 / p.view.scale; p.target = null; }
      g.requestRender();
      if (k++ % sampleEvery === 0) { const c = g.coverage(); sharp.push(c.sharp); blank.push(c.blank); }
      if (t < seconds) requestAnimationFrame(step); else resolve();
    };
    requestAnimationFrame(step);
  });
  const tStop = performance.now();
  // settle: time until fully sharp again
  let settle = null;
  while (performance.now() - tStop < 30000) {
    await new Promise((r) => requestAnimationFrame(r));
    if (g.coverage().sharp >= 0.999) { settle = performance.now() - tStop; break; }
  }
  const c1 = g.counters;
  return { frames, sharp, blank, settle_ms: settle, js: g.jsFrameMs.slice(),
           requests: c1.requests - c0.requests, completed: c1.completed - c0.completed,
           cancelled: c1.cancelled - c0.cancelled, MB: (c1.bytes - c0.bytes) / 2 ** 20 };
}
"""

_WAIT_SHARP_JS = """
async (timeout) => {
  const g = document.querySelector('wissi-viz');
  const t0 = performance.now();
  while (performance.now() - t0 < timeout) {
    await new Promise((r) => requestAnimationFrame(r));
    const c = g.coverage();
    if (c.panes.length && c.panes.every(Boolean) && c.sharp >= 0.999) return performance.now() - t0;
  }
  const c = g.coverage();  // timed out: report what is missing
  window.__wissiTimeout = { coverage: c.panes.filter((p) => p && !p.offSlide && p.sharp < 0.999),
    stats: g.stats(), failed: [...g.failed.keys()].slice(0, 10), inflight: [...g.inflight.keys()].slice(0, 10) };
  console.warn('wissi-bench timeout', JSON.stringify(window.__wissiTimeout));
  return null;
}
"""


def _summ_pan(r: dict) -> dict:
    f = r['frames'][2:] or r['frames']
    return {'fps_median': round(1000 / _pct(f, 50), 1) if f else None, 'frame_p50_ms': round(_pct(f, 50), 1),
            'frame_p95_ms': round(_pct(f, 95), 1), 'frame_p99_ms': round(_pct(f, 99), 1),
            'long_frames_pct': round(100 * float(np.mean(np.asarray(f) > 33.4)), 1) if f else None,
            'sharp_while_moving_pct': round(100 * float(np.mean(r['sharp'])), 1) if r['sharp'] else None,
            'blank_while_moving_pct': round(100 * float(np.mean(r['blank'])), 2) if r['blank'] else None,
            'settle_ms': None if r['settle_ms'] is None else round(r['settle_ms']),
            'js_frame_p50_ms': round(_pct(r.get('js') or [], 50), 2), 'js_frame_p95_ms': round(_pct(r.get('js') or [], 95), 2),
            'tiles_requested': r['requests'], 'tiles_cancelled': r['cancelled'], 'MB': round(r['MB'], 1)}


def browser_bench(paths: Sequence[str] | None = None, panes: Sequence[int] = (1, 4, 9, 16), headed: bool = False,
                  pan_seconds: float = 8, pan_speed: float = 900, magnification: float = 20, port: int = 0,
                  browser_args: Sequence[str] | None = None, viewport=(1600, 1000), cache_dir: str | None = None,
                  registry=None, slide_ids: Sequence[str] | None = None, http2: bool = False, log=print) -> dict:
    """``paths``: slide files; or pass an existing ``registry`` + ``slide_ids`` (e.g. API-backed slides)."""
    import socket

    import uvicorn
    from playwright.sync_api import sync_playwright

    from .registry import SlideRegistry
    from .server import create_app
    from .tiles import TileService

    if registry is not None:
        reg = registry
        ids = [slide_ids[i % len(slide_ids)] for i in range(max(panes))]
    else:
        reg = SlideRegistry(TileService(memory_bytes=2048 * 2**20, disk_dir=cache_dir), warm=True)
        ids = []
        for i in range(max(panes)):  # distinct ids → each pane is a separate slide (no shared caches)
            p = paths[i % len(paths)]
            ids.append(reg.add(p, slide_id=f'bench{i}', name=f'{Path(p).name} #{i}'))
    if not port:
        with socket.socket() as s_:
            s_.bind(('127.0.0.1', 0))
            port = s_.getsockname()[1]
    if http2:  # browsers only speak HTTP/2 over TLS → hypercorn with a throwaway self-signed certificate
        th, stop = _serve_h2(create_app(reg), port)
        scheme = 'https'
    else:
        server = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        while not server.started:
            time.sleep(0.05)
        stop = lambda: setattr(server, 'should_exit', True)  # noqa: E731
        scheme = 'http'
    args = list(browser_args or ['--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--enable-unsafe-swiftshader'])
    if http2:
        args.append('--ignore-certificate-errors')
    report: dict[str, Any] = {'runs': []}
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(headless=not headed, args=args)
            pg = b.new_page(viewport={'width': viewport[0], 'height': viewport[1]}, ignore_https_errors=True)
            pg.goto(f'{scheme}://127.0.0.1:{port}/')
            pg.wait_for_function('() => window.grid && customElements.get("wissi-viz")', timeout=30000)
            report['webgl_renderer'] = pg.evaluate("""() => { const gl = document.createElement('canvas').getContext('webgl2');
                const e = gl && gl.getExtension('WEBGL_debug_renderer_info');
                return gl ? (e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) : 'no WebGL2'; }""")
            report['software_rendering'] = 'swiftshader' in report['webgl_renderer'].lower() or 'llvmpipe' in report['webgl_renderer'].lower()
            report['hardware_concurrency'] = pg.evaluate('navigator.hardwareConcurrency')
            report['http_protocol'] = pg.evaluate(
                "() => (performance.getEntriesByType('resource').find(e => e.name.includes('/static/')) || {}).nextHopProtocol || ''")
            log(f"WebGL renderer: {report['webgl_renderer']}{'  (software — no GPU)' if report['software_rendering'] else ''}")
            for n in panes:
                pg.evaluate("""(ids) => { grid.setAttribute('controls','false'); grid.setSync('none');
                    grid.setLayout('auto'); grid.setSlides(ids); }""", ids[:n])
                open_ms = pg.evaluate(_WAIT_SHARP_JS, 60000)
                if open_ms is None:
                    log('  open timed out: ' + json.dumps(pg.evaluate('() => window.__wissiTimeout')))
                info = pg.evaluate('() => grid.panes[0].slide.info && [grid.panes[0].slide.info.width, grid.panes[0].slide.info.height]')
                cx, cy = (info[0] / 2, info[1] / 2) if info else (0, 0)
                pg.evaluate(f'() => grid.goTo("all", {cx}, {cy}, {{magnification: {magnification}, animate: false}})')
                zoom_ms = pg.evaluate(_WAIT_SHARP_JS, 60000)
                if zoom_ms is None:
                    log('  zoom timed out: ' + json.dumps(pg.evaluate('() => window.__wissiTimeout')))
                pan = pg.evaluate(_PAN_JS, [pan_seconds, pan_speed, 2])
                pg.evaluate(f'() => grid.goTo("all", {cx}, {cy}, {{magnification: {magnification}, animate: false}})')
                pg.evaluate(_WAIT_SHARP_JS, 60000)
                again = pg.evaluate(_PAN_JS, [pan_seconds, pan_speed, 2])
                run = {'panes': n, 'open_ms': None if open_ms is None else round(open_ms),
                       'zoom_to_sharp_ms': None if zoom_ms is None else round(zoom_ms),
                       'pan': _summ_pan(pan), 'revisit': _summ_pan(again),
                       'stats': pg.evaluate('() => grid.stats()')}
                report['runs'].append(run)
                log(f"{n:>2} panes  open {run['open_ms']} ms  zoom {run['zoom_to_sharp_ms']} ms  pan: "
                    f"{run['pan']['fps_median']} fps (p95 {run['pan']['frame_p95_ms']} ms), sharp {run['pan']['sharp_while_moving_pct']}%, "
                    f"blank {run['pan']['blank_while_moving_pct']}%, settle {run['pan']['settle_ms']} ms | revisit "
                    f"{run['revisit']['fps_median']} fps, sharp {run['revisit']['sharp_while_moving_pct']}%")
            b.close()
    finally:
        stop()
        th.join(timeout=5)
    report['server_tile_cache'] = reg.tiles.stats()
    report['settings'] = {'pan_seconds': pan_seconds, 'pan_speed_px_s': pan_speed, 'magnification': magnification,
                          'viewport': viewport, 'headed': headed}
    return report


# ============================================================================= zoom / level-of-detail stage
_ZOOM_JS = """
async ([rate, toScales, maxSeconds, sampleEvery]) => {
  // continuous zoom about each pane's centre at `rate`× per second (the way a wheel/trackpad zoom drives the
  // viewer: _zoomAt with animation), until every pane's target reaches toScales[i]; then wait for full res
  const g = document.querySelector('wissi-viz');
  const S = { frames: [], sharp: [], detail: [], blank: [] };
  g.jsFrameMs = [];
  let last = null, t0 = null, k = 0;
  await new Promise((resolve) => {
    const step = (ts) => {
      if (t0 === null) { t0 = ts; last = ts; }
      const dt = ts - last; last = ts;
      if (dt > 0) S.frames.push(dt);
      let busy = false;
      g.panes.forEach((p, i) => {
        if (!p.view || toScales[i] == null) return;
        const cur = (p.target || p.view).scale, goal = toScales[i];
        const lr = Math.log(goal / cur);
        if (Math.abs(lr) > 1e-3) {
          const stepL = Math.sign(lr) * Math.min(Math.abs(lr), Math.log(rate) * dt / 1000);
          g._zoomAt(p, p.rect.w / 2, p.rect.h / 2, Math.exp(stepL), true);
          busy = true;
        } else if (p.target) busy = true;
      });
      if (k++ % sampleEvery === 0) { const c = g.coverage(); S.sharp.push(c.sharp); S.detail.push(c.detail); S.blank.push(c.blank); }
      if (busy && (ts - t0) / 1000 < maxSeconds) requestAnimationFrame(step); else resolve();
    };
    requestAnimationFrame(step);
  });
  S.motion_ms = performance.now() - (t0 || performance.now());
  S.js = g.jsFrameMs.slice();
  const tStop = performance.now();
  S.settle_ms = null;
  while (performance.now() - tStop < 30000) {
    await new Promise((r) => requestAnimationFrame(r));
    if (g.coverage().sharp >= 0.999) { S.settle_ms = performance.now() - tStop; break; }
  }
  return S;
}
"""

_JUMP_JS = """
async ([targets, timeout, sampleEvery]) => {
  // animated goTo (double-click / 'centre on object' style) to each [fx, fy, magnification]; time to full res
  const g = document.querySelector('wissi-viz');
  const out = [];
  for (const [fx, fy, mag] of targets) {
    for (const p of g.panes) if (p.slide && p.slide.info) g.goTo(p.index, fx * p.slide.info.width, fy * p.slide.info.height, { magnification: mag });
    const t0 = performance.now();
    const S = { detail: [], blank: [] };
    let k = 0, ms = null;
    while (performance.now() - t0 < timeout) {
      await new Promise((r) => requestAnimationFrame(r));
      const c = g.coverage();
      if (k++ % sampleEvery === 0) { S.detail.push(c.detail); S.blank.push(c.blank); }
      if (!g.panes.some((p) => p.target) && c.sharp >= 0.999) { ms = performance.now() - t0; break; }
    }
    out.push({ mag, ms, detail: S.detail, blank: S.blank });
  }
  return out;
}
"""

_SNAP_JS = """() => { const g = document.querySelector('wissi-viz');
  let undrawn = 0; for (const t of g.tex.values()) if (!t.drawn) undrawn++;
  return { ...g.counters, ...g.fx, undrawn, textures: g.tex.size, textureMB: g.texBytes / 2 ** 20 }; }"""


def _delta(a: dict, b: dict) -> dict:
    d = {k: b[k] - a[k] for k in ('requests', 'completed', 'cancelled', 'aborted', 'failed', 'uploaded', 'wasted',
                                  'frames', 'draws', 'fill', 'area')}
    return {'requests': d['requests'], 'cancelled': d['cancelled'] + d['aborted'], 'MB': round((b['bytes'] - a['bytes']) / 2**20, 2),
            'uploaded': d['uploaded'], 'evicted_undrawn': d['wasted'],
            'overdraw': round(d['fill'] / d['area'], 3) if d['area'] else None,
            'draws_per_frame': round(d['draws'] / d['frames'], 1) if d['frames'] else None}


def _summ_zoom(r: dict) -> dict:
    f = r['frames'][2:] or r['frames']
    js = r.get('js') or []
    return {'motion_ms': round(r['motion_ms']), 'fps_median': round(1000 / _pct(f, 50), 1) if f else None,
            'frame_p95_ms': round(_pct(f, 95), 1) if f else None,
            'js_frame_p50_ms': round(_pct(js, 50), 2) if js else None, 'js_frame_p95_ms': round(_pct(js, 95), 2) if js else None,
            'detail_pct': round(100 * float(np.mean(r['detail'])), 1) if r['detail'] else None,
            'full_res_pct': round(100 * float(np.mean(r['sharp'])), 1) if r['sharp'] else None,
            'blank_pct': round(100 * float(np.mean(r['blank'])), 2) if r['blank'] else None,
            'settle_ms': None if r['settle_ms'] is None else round(r['settle_ms'])}


def zoom_bench(configs: Sequence[tuple[str, Sequence[str], int]], modes: Sequence[str] = ('legacy', 'default'),
               rate: float = 4.0, magnification: float = 40, viewport=(1600, 1000), repeats: int = 1,
               headed: bool = False, browser_args: Sequence[str] | None = None, http2: bool = False,
               save_to: str | None = None, log=print) -> dict:
    """Zooming through the pyramid: continuous zoom fit→``magnification``→fit at ``rate``×/s, then jump zooms.

    ``configs``: (label, slide paths, panes). ``modes``: viewer LOD policies (``'legacy'``, ``'default'`` or a JSON
    dict for ``setLod``). Every (mode, config, repeat) gets a fresh server registry and a fresh browser, so tile
    caches start cold on both sides and the comparison is fair."""
    import socket

    import uvicorn
    from playwright.sync_api import sync_playwright

    from .registry import SlideRegistry
    from .server import create_app
    from .tiles import TileService

    args = list(browser_args or ['--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--enable-unsafe-swiftshader'])
    if http2:
        args.append('--ignore-certificate-errors')
    jumps = [[0.5, 0.5, magnification], [0.35, 0.6, magnification / 4], [0.62, 0.42, magnification / 2],
             [0.5, 0.5, max(1.25, magnification / 16)], [0.45, 0.55, magnification]]
    report: dict[str, Any] = {'runs': [], 'settings': {'rate_per_s': rate, 'magnification': magnification,
                                                       'viewport': viewport, 'jumps': jumps, 'http2': http2}}
    with sync_playwright() as pw:
        for rep in range(repeats):
            for cfg in configs:
                label, paths, n = cfg[:3]
                reg_kw = cfg[3] if len(cfg) > 3 else {}
                for mode in modes:
                    reg = SlideRegistry(TileService(memory_bytes=2048 * 2**20), warm=True, **reg_kw)
                    ids = [reg.add(paths[i % len(paths)], slide_id=f'z{i}') for i in range(n)]
                    with socket.socket() as s_:
                        s_.bind(('127.0.0.1', 0))
                        port = s_.getsockname()[1]
                    if http2:
                        th, stop = _serve_h2(create_app(reg), port)
                        scheme = 'https'
                    else:
                        server = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
                        th = threading.Thread(target=server.run, daemon=True)
                        th.start()
                        while not server.started:
                            time.sleep(0.05)
                        stop = lambda server=server: setattr(server, 'should_exit', True)  # noqa: E731
                        scheme = 'http'
                    b = pw.chromium.launch(headless=not headed, args=args)
                    try:
                        pg = b.new_page(viewport={'width': viewport[0], 'height': viewport[1]}, ignore_https_errors=True)
                        pg.goto(f'{scheme}://127.0.0.1:{port}/')
                        pg.wait_for_function('() => window.grid && customElements.get("wissi-viz")', timeout=30000)
                        lod = mode if mode in ('legacy', 'default') else json.loads(mode)
                        pg.evaluate("""([ids, lod]) => { grid.setAttribute('controls','false'); grid.setSync('none');
                            grid.setLod(lod); grid.setLayout('auto'); grid.setSlides(ids); }""", [ids, lod])
                        open_ms = pg.evaluate(_WAIT_SHARP_JS, 60000)
                        fit = pg.evaluate('() => grid.panes.map((p) => p.view && p.view.scale)')
                        mpp = pg.evaluate('() => grid.panes.map((p) => p.slide.info.mpp ? p.slide.info.mpp[0] : 1)')
                        deep = [m * magnification / 10 for m in mpp]
                        s0 = pg.evaluate(_SNAP_JS)
                        zin = pg.evaluate(_ZOOM_JS, [rate, deep, 30, 2])
                        s1 = pg.evaluate(_SNAP_JS)
                        zout = pg.evaluate(_ZOOM_JS, [rate, fit, 30, 2])
                        s2 = pg.evaluate(_SNAP_JS)
                        jump = pg.evaluate(_JUMP_JS, [jumps, 30000, 2])
                        s3 = pg.evaluate(_SNAP_JS)
                    finally:
                        b.close()
                        stop()
                        th.join(timeout=5)
                        reg.close()   # release caches + workers before the next (cold) run
                    run = {'config': label, 'panes': n, 'mode': mode, 'repeat': rep,
                           'open_ms': None if open_ms is None else round(open_ms),
                           'zoom_in': {**_summ_zoom(zin), **_delta(s0, s1)},
                           'zoom_out': {**_summ_zoom(zout), **_delta(s1, s2)},
                           'jumps': {'ms': [None if j['ms'] is None else round(j['ms']) for j in jump],
                                     'detail_pct': round(100 * float(np.mean([x for j in jump for x in j['detail']] or [0])), 1),
                                     'blank_pct': round(100 * float(np.mean([x for j in jump for x in j['blank']] or [0])), 2),
                                     **_delta(s2, s3)},
                           'end': {'textures': s3['textures'], 'textureMB': round(s3['textureMB'], 1),
                                   'undrawn_textures': s3['undrawn'], 'requests': s3['requests'],
                                   'MB': round(s3['bytes'] / 2**20, 1)}}
                    report['runs'].append(run)
                    if save_to:  # keep partial results of long sweeps
                        Path(save_to).write_text(json.dumps(report, indent=1))
                    zi, zo, jp = run['zoom_in'], run['zoom_out'], run['jumps']
                    log(f"{label:<14} {n:>2}p {str(mode)[:8]:<8} in: detail {zi['detail_pct']}% blank {zi['blank_pct']}% "
                        f"settle {zi['settle_ms']} ms {zi['MB']} MB req {zi['requests']} overdraw {zi['overdraw']} js {zi['js_frame_p50_ms']} | "
                        f"out: detail {zo['detail_pct']}% blank {zo['blank_pct']}% {zo['MB']} MB req {zo['requests']} | "
                        f"jumps {jp['ms']} {jp['MB']} MB | total {run['end']['MB']} MB")
    return report


def format_zoom_report(rep: dict) -> str:
    rows = ['| config | panes | LOD | phase | detail % | full-res % | blank % | settle ms | MB | requests | cancelled | '
            'uploaded | overdraw | draws/frame | JS ms p50 (p95) | fps |', '|' + '---|' * 16]
    for r in rep['runs']:
        for ph in ('zoom_in', 'zoom_out'):
            z = r[ph]
            rows.append(f"| {r['config']} | {r['panes']} | {r['mode']} | {ph.replace('_', ' ')} | {z['detail_pct']} | {z['full_res_pct']} | "
                        f"{z['blank_pct']} | {z['settle_ms']} | {z['MB']} | {z['requests']} | {z['cancelled']} | {z['uploaded']} | "
                        f"{z['overdraw']} | {z['draws_per_frame']} | {z['js_frame_p50_ms']} ({z['js_frame_p95_ms']}) | {z['fps_median']} |")
        j = r['jumps']
        rows.append(f"| {r['config']} | {r['panes']} | {r['mode']} | jumps ({', '.join(str(x) for x in j['ms'])} ms) | {j['detail_pct']} | | "
                    f"{j['blank_pct']} | {sum(x or 0 for x in j['ms'])} | {j['MB']} | {j['requests']} | {j['cancelled']} | {j['uploaded']} | "
                    f"{j['overdraw']} | {j['draws_per_frame']} | | |")
    return '\n'.join(rows)


def _serve_h2(app, port: int):
    """Serve ``app`` over HTTPS + HTTP/2 (hypercorn, MIT) with a self-signed certificate (cryptography)."""
    import asyncio
    import datetime
    import tempfile

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, '127.0.0.1')])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=2)).sign(key, hashes.SHA256()))
    d = tempfile.mkdtemp(prefix='wissi-h2-')
    Path(d, 'cert.pem').write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    Path(d, 'key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                     serialization.NoEncryption()))
    cfg = Config()
    cfg.bind = [f'127.0.0.1:{port}']
    cfg.certfile, cfg.keyfile = str(Path(d, 'cert.pem')), str(Path(d, 'key.pem'))
    cfg.alpn_protocols = ['h2', 'http/1.1']
    cfg.loglevel = 'WARNING'
    loop = asyncio.new_event_loop()
    done = asyncio.Event()
    started = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_until_complete(serve(app, cfg, shutdown_trigger=done.wait))
    th = threading.Thread(target=run, daemon=True)
    th.start()
    started.wait(10)
    time.sleep(0.5)
    return th, lambda: loop.call_soon_threadsafe(done.set)


# ============================================================================= public samples
SAMPLE_SETS = {
    # (url, approx MB) — OpenSlide test data (various licences, see their README), CAMELYON16 (CC0)
    'small': [
        ('https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs', 2),
        ('https://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff', 200),
    ],
    'standard': [
        ('https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs', 170),
        ('https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-2.svs', 390),
        ('https://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff', 200),
        ('https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/CMU-1.ndpi', 190),
        ('dir:https://openslide.cs.cmu.edu/download/openslide-testdata/Leica/', 0),
        ('dir:https://openslide.cs.cmu.edu/download/openslide-testdata/Zeiss/', 0),
    ],
    'full': [
        ('dir:https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/', 0),
        ('dir:https://openslide.cs.cmu.edu/download/openslide-testdata/Leica/', 0),
        ('dir:https://openslide.cs.cmu.edu/download/openslide-testdata/Zeiss/', 0),
        ('dir:https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/', 0),
        ('dir:https://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/', 0),
        ('https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/images/normal_001.tif', 900),
    ],
}
SLIDE_SUFFIXES = ('.svs', '.tif', '.tiff', '.ndpi', '.scn', '.czi', '.ome.tif', '.ome.tiff', '.vsi', '.zip')


def fetch_samples(dest: str | Path, which: str = 'standard', max_mb: float = 2000, extra_urls: Sequence[str] = (),
                  log=print) -> list[Path]:
    """Download public whole-slide test images (skips files already present / larger than ``max_mb``)."""
    import re
    import urllib.request
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    for u, _ in SAMPLE_SETS[which]:
        if u.startswith('dir:'):
            base = u[4:]
            try:
                html = urllib.request.urlopen(base, timeout=60).read().decode('utf-8', 'replace')
            except Exception as e:  # noqa: BLE001
                log(f'  cannot list {base}: {e}')
                continue
            for href in sorted(set(re.findall(r'href="([^"?/][^"]*)"', html))):
                if href.lower().endswith(SLIDE_SUFFIXES):
                    urls.append(base + href)
        else:
            urls.append(u)
    urls += list(extra_urls)
    got = []
    for u in urls:
        name = u.rsplit('/', 1)[-1]
        sub = dest / u.split('/')[-2]
        sub.mkdir(exist_ok=True)
        target = sub / name
        if target.exists():
            got.append(target)
            continue
        try:
            req = urllib.request.Request(u, method='HEAD')
            size = int(urllib.request.urlopen(req, timeout=60).headers.get('Content-Length', 0)) / 2**20
        except Exception as e:  # noqa: BLE001
            log(f'  skip {u}: {e}')
            continue
        if size > max_mb:
            log(f'  skip {name}: {size:.0f} MB > --max-mb {max_mb}')
            continue
        log(f'  downloading {name} ({size:.0f} MB) …')
        tmp = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(u, timeout=120) as r, open(tmp, 'wb') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(target)
        got.append(target)
    return got


def format_report(server: list[dict] | None, browser: dict | None) -> str:
    lines = []
    if server:
        lines.append('## Server: open + tiles\n')
        lines.append('| slide | reader | size | MB | open ms | level (÷) | cold p50/p95 ms | tiles/s (×8) | warm p50 ms | tile KB |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for r in server:
            for i, L in enumerate(r['levels_bench']):
                tp = next((v for k, v in L.items() if k.startswith('throughput')), None) or '–'
                head = [r['slide'], r['reader'], r['size'], str(r['file_MB']), str(r['open_ms'])] if i == 0 else [''] * 5
                lines.append('| ' + ' | '.join(head + [f"{L['level']} (÷{L['downsample']:g})",
                                                       f"{L['cold_p50_ms']} / {L['cold_p95_ms']}", str(tp),
                                                       str(L['warm_p50_ms']), str(L['tile_KB'])]) + ' |')
    if browser:
        vw, vh = browser['settings']['viewport']
        proto = browser.get('http_protocol') or 'http/1.1'
        mag = browser['settings'].get('magnification', 20)
        lines.append(f"\n## Browser ({vw}×{vh}, {proto}): {browser.get('webgl_renderer')}"
                     f"{' — SOFTWARE rendering (no GPU)' if browser.get('software_rendering') else ''}\n")
        lines.append(f'| panes | open→sharp ms | {mag:g}×→sharp ms | pan fps (p95 ms) | viewer JS ms/frame p50 (p95) | long frames % | sharp while panning % | blank % | settle ms | revisit fps / sharp % | MB |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in browser['runs']:
            p, v = r['pan'], r['revisit']
            lines.append(f"| {r['panes']} | {r['open_ms']} | {r['zoom_to_sharp_ms']} | {p['fps_median']} ({p['frame_p95_ms']}) | "
                         f"{p['js_frame_p50_ms']} ({p['js_frame_p95_ms']}) | "
                         f"{p['long_frames_pct']} | {p['sharp_while_moving_pct']} | {p['blank_while_moving_pct']} | {p['settle_ms']} | "
                         f"{v['fps_median']} / {v['sharp_while_moving_pct']} | {p['MB']} |")
    return '\n'.join(lines)


def main(args) -> None:
    server = None if args.browser_only else server_bench(args.slides, concurrency=args.concurrency)
    if server:
        for r in server:
            print(json.dumps(r))
    browser = None
    if not args.server_only:
        browser = browser_bench(args.slides, [int(x) for x in args.panes.split(',')], headed=args.headed,
                                pan_seconds=args.seconds, pan_speed=args.speed, magnification=args.magnification,
                                browser_args=args.browser_args.split() if args.browser_args else None,
                                viewport=tuple(int(v) for v in args.viewport.lower().split('x')), http2=args.http2)
    md = format_report(server, browser)
    print('\n' + md)
    if args.out:
        Path(args.out).write_text(json.dumps({'server': server, 'browser': browser}, indent=2, default=str))
        Path(args.out).with_suffix('.md').write_text(md)
