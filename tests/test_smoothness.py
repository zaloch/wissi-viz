"""Main-loop smoothness regressions (real headless Chromium): adaptive render resolution only while moving and
never at rest, one batched ``views`` event instead of a per-frame event storm, no 2-D overlay work when idle,
and no per-frame tile-key strings. Skipped when Playwright's Chromium isn't available."""
from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip('playwright')
pytest.importorskip('uvicorn')


@pytest.fixture(scope='module')
def page():
    import uvicorn
    from playwright.sync_api import sync_playwright

    from wissi_viz import Level, SlideInfo, SlideRegistry
    from wissi_viz.server import create_app
    from wissi_viz.sources.callback import CallbackSource

    def tile(level, tx, ty, c):
        i = np.arange(256, dtype=np.uint16)
        t = np.empty((256, 256, 3), np.uint8)
        t[..., 0] = ((i[None, :] + tx * 256) % 251).astype(np.uint8)
        t[..., 1] = ((i[:, None] + ty * 256) % 241).astype(np.uint8)
        t[..., 2] = (level * 60) % 256
        return t
    W, H = 8192, 6144
    info = SlideInfo(width=W, height=H, levels=[Level(W >> k, H >> k, float(2 ** k)) for k in range(6)], tile_size=256, mpp=(0.5, 0.5))
    reg = SlideRegistry(warm=False)
    reg.add(CallbackSource(info, read_tile=tile), slide_id='s')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
        except Exception as e:  # noqa: BLE001
            srv.should_exit = True
            pytest.skip(f'no Chromium: {e}')
        pg = b.new_page(viewport={'width': 640, 'height': 400})
        pg.goto(f'http://127.0.0.1:{port}/')
        pg.wait_for_function('() => window.grid')
        pg.evaluate('() => { grid.setAttribute("controls", "false"); grid.setSync("all"); grid.setSlides(["s", "s", "s", "s"]); }')
        pg.wait_for_function('() => grid.panes.every(p => p.view)', timeout=30000)
        yield pg
        b.close()
    srv.should_exit = True


def _settle(pg):
    pg.evaluate(f'() => grid.goTo("all", {4096}, {3072}, {{zoom: 0.5, animate: false}})')
    pg.wait_for_function('() => grid.stats().inflight === 0 && grid.stats().queuedUploads === 0 && grid.coverage().blank === 0',
                         timeout=30000)
    time.sleep(0.3)


SNAP = """() => { const g = grid; g._draw(); const gl = g.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight,
    px = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px); return Array.from(px); }"""


def test_adaptive_resolution_full_res_at_rest_low_res_only_while_moving(page):
    pg = page
    _settle(pg)
    assert pg.evaluate('() => grid.lod.adaptive') is True
    full = np.array(pg.evaluate(SNAP), np.int16)
    # at rest a "slow" frame time must never lower the resolution
    pg.evaluate('() => { grid._frameMs = 80; grid._rs = 1; }')
    assert pg.evaluate('() => grid._renderScale()') == 1
    rest = np.array(pg.evaluate(SNAP), np.int16)
    assert np.array_equal(full, rest)
    # moving + slow frames → renders below full resolution, stretched to the whole canvas (no black/garbage)
    n0 = pg.evaluate('() => grid.stats().lowresFrames')
    low = pg.evaluate("""() => { const t = performance.now(); for (const p of grid.panes) p.movedAt = t;
        grid._frameMs = 80; grid._rs = 1; for (let i = 0; i < 6; i++) grid._renderScale(); return grid._rs; }""")
    assert low in (0.5, 0.75)
    moving = np.array(pg.evaluate("""() => { const t = performance.now(); for (const p of grid.panes) p.movedAt = t;
        grid._frameMs = 80; const g = grid; g._draw(); const gl = g.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight,
        px = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px); return Array.from(px); }"""), np.int16)
    assert pg.evaluate('() => grid.stats().lowresFrames') > n0
    # the backing store itself shrinks (the CSS size doesn't): less to render *and* less for the compositor
    css = pg.evaluate('() => [grid.$canvas.clientWidth, grid.$canvas.clientHeight, grid.$canvas.width, grid.$canvas.height, grid._cw, grid._ch]')
    assert css[2] < css[4] and css[3] < css[5] and (css[0], css[1]) == (css[4], css[5])   # dpr 1
    assert moving.shape[0] < full.shape[0]
    # same picture, softer: compare against the full frame downsampled to the low-res size
    fw, fh = css[4], css[5]
    mw, mh = css[2], css[3]
    f3 = full.reshape(fh, fw, 4)[(np.arange(mh) * fh // mh)][:, (np.arange(mw) * fw // mw)]
    assert np.abs(moving.reshape(mh, mw, 4) - f3).mean() < 40      # not blank, not shifted by a pane
    # once motion stops the next frame is full resolution again and pixel-identical to before
    pg.evaluate('() => { for (const p of grid.panes) { p.movedAt = 0; p.vel = null; p.target = null; } }')
    back = np.array(pg.evaluate(SNAP), np.int16)
    assert np.array_equal(full, back)
    assert pg.evaluate('() => [grid.$canvas.width, grid.$canvas.height]') == [css[4], css[5]]
    # legacy LOD keeps the old behaviour
    pg.evaluate('() => grid.setLod("legacy")')
    assert pg.evaluate('() => { const t = performance.now(); for (const p of grid.panes) p.movedAt = t; grid._frameMs = 80; return grid._renderScale(); }') == 1
    pg.evaluate('() => { for (const p of grid.panes) p.movedAt = 0; grid.setLod("default"); }')


def test_views_event_is_batched_and_throttled_with_trailing_snapshot(page):
    pg = page
    _settle(pg)
    pg.evaluate("""() => { window.__ev = { views: [], viewchange: 0 };
        window.__evt = [];
        grid.addEventListener('views', (e) => { __ev.views.push(e.detail.views); __evt.push(performance.now()); });
        grid.addEventListener('viewchange', () => __ev.viewchange++); }""")
    box = pg.evaluate('() => { const r = grid.getBoundingClientRect(), p = grid.panes[0].rect; return [r.left + p.x + p.w / 2, r.top + p.y + p.h / 2]; }')
    pg.mouse.move(*box)
    pg.mouse.down()
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < 1.0:
        k += 1
        pg.mouse.move(box[0] + (k % 60) * 2, box[1] + (k % 40))
        time.sleep(0.008)
    pg.mouse.up()
    # wait for any fling (inertia) to finish — slow CI machines glide for longer — then for the trailing snapshot
    pg.wait_for_function('() => grid.panes.every(p => !p.vel && !p.target)', timeout=10_000)
    time.sleep(0.5)
    ev = pg.evaluate('() => __ev')
    final = pg.evaluate('() => grid.panes.map(p => [p.view.cx, p.view.cy])')
    times = pg.evaluate('() => __evt')
    gaps = np.diff(times)
    assert len(ev['views']) >= 2 and gaps.min() >= 100, gaps      # throttled to ≤ ~8/s, not one per frame
    assert all(len(v) == 4 for v in ev['views'])                   # every pane in each snapshot
    last = ev['views'][-1]
    for v, (cx, cy) in zip(last, final):                           # the trailing snapshot is the resting view
        assert abs(v['x'] - cx) < 1 and abs(v['y'] - cy) < 1
    assert ev['viewchange'] <= 4 * len(ev['views'])


def test_idle_overlay_canvas_is_not_touched_every_frame(page):
    pg = page
    _settle(pg)
    n = pg.evaluate("""() => { let n = 0; const c = grid.ctx2d, orig = c.clearRect.bind(c);
        c.clearRect = (...a) => { n++; return orig(...a); };
        for (let i = 0; i < 30; i++) grid._draw();
        c.clearRect = orig; return n; }""")
    assert n == 0


def test_tile_keys_are_cached_and_distinct(page):
    r = page.evaluate("""() => { const s = grid.panes[0].slide, k = grid._key.bind(grid);
        const a = k(s, 2, 3, 4, null), b = k(s, 2, 3, 4, null);
        const keys = new Set([a, k(s, 2, 3, 4, 0), k(s, 2, 3, 4, 14), k(s, 2, 3, 4, 15), k(s, 2, 3, 4, 40), k(s, 2, 4, 3, null), k(s, 3, 3, 4, null)]);
        return { same: a === b, distinct: keys.size, a }; }""")
    assert r['same'] and r['distinct'] == 7 and r['a'].endswith('|2|3|4|rgb')


def test_merge_views_reports_only_changed_panes():
    pytest.importorskip('nicegui')
    from wissi_viz.nicegui import merge_views
    views: dict = {}
    snap = {'views': [{'pane': 0, 'x': 1, 'y': 2, 'zoom': 0.5}, {'pane': 1, 'x': 3, 'y': 4, 'zoom': 0.5}, None]}
    assert [v['pane'] for v in merge_views(views, snap)] == [0, 1]
    snap['views'][1] = {'pane': 1, 'x': 9, 'y': 4, 'zoom': 0.5}
    assert [v['pane'] for v in merge_views(views, snap)] == [1]
    assert views[1]['x'] == 9 and merge_views(views, None) == []
