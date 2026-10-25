"""Real browser (headless Chromium, software WebGL) against the real server: loading, panning and
request hygiene. Skipped when Playwright's Chromium isn't available."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

pytest.importorskip('playwright')
pytest.importorskip('uvicorn')


@pytest.fixture(scope='module')
def slide(tmp_path_factory):
    d = tmp_path_factory.mktemp('br')
    rng = np.random.default_rng(0)
    img = (rng.random((6000, 8000, 3)) * 255).astype(np.uint8)
    desc = 'Aperio Image Library v12 \r\n8000x6000 [0,0 8000x6000] (256x256) JPEG/RGB Q=80|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 'b.svs', bigtiff=True) as tw:
        o = dict(tile=(256, 256), compression='jpeg', photometric='rgb', metadata=None)
        tw.write(img, description=desc, **o)
        tw.write(img[::16, ::16].copy(), description='Aperio thumb', photometric='rgb', metadata=None)
        tw.write(np.ascontiguousarray(img[::4, ::4]), description='Aperio Image Library level', **o)
    return d / 'b.svs'


def _run(slide, **kw):
    from wissi_viz.bench import browser_bench
    try:
        return browser_bench([str(slide)], log=lambda *a: None, viewport=(800, 500), pan_seconds=1.5, **kw)
    except Exception as e:  # noqa: BLE001
        if 'Executable doesn' in str(e) or 'BrowserType.launch' in str(e):
            pytest.skip(f'no Chromium: {e}')
        raise


@pytest.mark.parametrize('http2', [False, True])
def test_no_duplicate_requests_and_never_blank(slide, http2):
    if http2:
        pytest.importorskip('hypercorn')
    r = _run(slide, panes=[1, 4], http2=http2)
    assert r['http_protocol'] == ('h2' if http2 else 'http/1.1')
    for run in r['runs']:
        st = run['stats']
        # every completed request became a texture (no re-requests of tiles waiting for upload)
        assert st['requests'] - st['cancelled'] - st['aborted'] - st['failed'] <= st['textures'] + 2, st
        assert run['open_ms'] is not None and run['zoom_to_sharp_ms'] is not None
        assert run['pan']['blank_while_moving_pct'] < 5 and run['revisit']['sharp_while_moving_pct'] > 90


def test_cross_origin_embed(slide, tmp_path):
    """Any web app (not NiceGUI): a page on origin A embeds <wissi-viz> served by a tile server on origin B."""
    import http.server
    import socket
    import threading
    import time

    import uvicorn

    from wissi_viz import SlideRegistry
    from wissi_viz.server import create_app

    def free_port():
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]
    reg = SlideRegistry(warm=False)
    reg.add(str(slide), slide_id='b')
    pb, pa = free_port(), free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(reg, cors='*'), host='127.0.0.1', port=pb, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    tiles = f'http://127.0.0.1:{pb}/wissi-viz'
    (tmp_path / 'index.html').write_text(f'''<!doctype html><html><body style="margin:0">
<h3>Some other app</h3>
<script type="module" src="{tiles}/static/index.js"></script>
<wissi-viz id="v" api="{tiles}" slides="b,b" layout="1x2" style="display:block;width:900px;height:450px"></wissi-viz>
</body></html>''')
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp_path), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', pa), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:  # noqa: BLE001
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page()
            errors = []
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.goto(f'http://127.0.0.1:{pa}/index.html')
            pg.wait_for_function('() => { const v = document.getElementById("v"); return v && v.coverage && '
                                 'v.coverage().panes.length === 2 && v.coverage().sharp >= 0.999 }', timeout=60000)
            pg.evaluate('() => document.getElementById("v").goTo("all", 4000, 3000, {magnification: 10, animate: false})')
            pg.wait_for_function('() => document.getElementById("v").coverage().sharp >= 0.999', timeout=60000)
            assert not [e for e in errors if 'CORS' in e or 'Failed' in e], errors
            b.close()
    finally:
        httpd.shutdown()
        srv.should_exit = True


def test_failed_tile_is_retried_until_sharp(tmp_path):
    """A tile whose first request fails (HTTP 500/502) must be retried after back-off even when the viewer is idle."""
    import socket
    import threading
    import time

    import uvicorn

    from wissi_viz import Level, SlideInfo, SlideRegistry
    from wissi_viz.server import create_app
    from wissi_viz.sources.callback import CallbackSource

    calls = {'n': 0}

    def tile(level, tx, ty, c):
        calls['n'] += 1
        if (level, tx, ty) == (0, 1, 1) and calls.get('failed') is None:   # a tile the fit view needs fails once
            calls['failed'] = True
            raise ConnectionError('transient')
        return np.full((256, 256, 3), 40 + 50 * level, np.uint8)
    info = SlideInfo(width=1000, height=700, levels=[Level(1000, 700, 1.0), Level(250, 175, 4.0)], tile_size=256, mpp=(0.5, 0.5))
    reg = SlideRegistry(warm=False)
    reg.add(CallbackSource(info, read_tile=tile), slide_id='flaky')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:  # noqa: BLE001
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 600, 'height': 400})
            pg.goto(f'http://127.0.0.1:{port}/')
            pg.wait_for_function('() => window.grid')
            pg.evaluate('() => grid.setSlides(["flaky"])')
            pg.wait_for_function('() => grid.panes[0].view && grid.coverage().panes[0] && grid.coverage().panes[0].level === 0',
                                 timeout=20000)
            pg.wait_for_function('() => grid.coverage().sharp >= 0.999', timeout=20000)
            assert pg.evaluate('() => grid.counters.failed') >= 1 and calls.get('failed')
            b.close()
    finally:
        srv.should_exit = True
