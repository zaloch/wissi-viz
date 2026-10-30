"""Desktop app (wissi_viz.desktop): session, local server, page, native-window wiring, shortcuts and icons.
The native window itself needs a GUI; its wiring is checked against a stand-in pywebview module."""

from __future__ import annotations

import json
import os
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import tifffile

from wissi_viz import desktop as D
from wissi_viz import session as S

pytest.importorskip('fastapi')


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    """Keep caches, config and shortcuts inside tmp_path on every OS."""

    def user_dir(kind, app='wissi-viz'):
        d = tmp_path / 'user' / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(D, 'user_dir', user_dir)
    monkeypatch.setattr(S, 'user_dir', user_dir)
    return tmp_path


def _svs(path: Path, w=1200, h=900) -> Path:
    rng = np.random.default_rng(0)
    img = (rng.random((h, w, 3)) * 200 + 30).astype(np.uint8)
    desc = f'Aperio Image Library v12 \r\n{w}x{h} [0,0 {w}x{h}] (256x256) JPEG/RGB Q=95|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    return path


@pytest.fixture()
def files(dirs):
    d = dirs / 'data with spaces'  # paths with spaces work everywhere
    d.mkdir()
    a, b = _svs(d / 'a.svs'), _svs(d / 'b.svs', 800, 600)
    rng = np.random.default_rng(1)
    n = 500
    (d / 'cells.csv').write_text(
        'x,y,class,area\n'
        + '\n'.join(
            f'{x:.1f},{y:.1f},{c},{s:.1f}'
            for x, y, c, s in zip(
                rng.uniform(0, 1200, n),
                rng.uniform(0, 900, n),
                rng.choice(['Tumor', 'Immune'], n),
                rng.uniform(20, 80, n),
            )
        ),
        encoding='utf-8',
    )
    (d / 'rois.geojson').write_text(
        json.dumps(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'properties': {'classification': {'name': 'Tumor'}},
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [[[100, 100], [400, 100], [400, 300], [100, 100]]],
                        },
                    }
                ],
            }
        ),
        encoding='utf-8',
    )
    (d / 'broken.svs').write_bytes(b'not a slide')
    return {'dir': d, 'a': a, 'b': b, 'csv': d / 'cells.csv', 'geojson': d / 'rois.geojson', 'bad': d / 'broken.svs'}


def test_session_opens_slides_layers_and_reports_errors(files):
    s = D.Session()
    r = s.open([files['a'], files['csv'], files['geojson'], files['bad'], files['dir'] / 'missing.svs'])
    assert len(r['opened']) == 1 and [x['name'] for x in r['slides']] == ['a.svs']
    sid = r['opened'][0]
    assert {L['layer'] for L in r['layers']} == {'cells', 'rois'} and all(L['slide'] == sid for L in r['layers'])
    assert len(r['errors']) == 2 and any('broken.svs' in e for e in r['errors'])
    assert set(s.registry.layers(sid)) >= {'cells', 'rois'}
    assert len(s.registry.layer(sid, 'cells')) == 500
    assert s.registry.ids() == [sid]  # the broken file left nothing behind
    # cell tables are compiled once into the cache
    assert any((s.cache_dir / 'objects').iterdir())
    # a folder opens every slide in it; layers go to the slide named, else the first one opened
    r2 = s.open([files['dir']])
    assert {x['name'] for x in r2['slides']} >= {'a.svs', 'b.svs'}
    other = next(x['id'] for x in r2['slides'] if x['name'] == 'b.svs')
    r3 = s.open([files['geojson']], slide=other)
    assert r3['layers'][0]['slide'] == other
    # recent files survive a restart
    assert str(files['a']) in D.Session().recent()
    assert s.close(other)['slides'] and other not in s.registry
    s.shutdown()


def test_layer_before_slide_is_an_error(files):
    r = D.Session().open([files['csv']])
    assert not r['opened'] and 'open a slide first' in r['errors'][0]


def test_http_endpoints_need_token_and_loopback_host(files):
    from starlette.testclient import TestClient

    s = D.Session()
    c = TestClient(s.app())
    page = c.get('/app').text
    assert s.token in page and '/wissi-viz/static/index.js' in page and '__API__' not in page
    assert c.get('/app/icon.svg').headers['content-type'].startswith('image/svg')
    assert c.get('/app/state').status_code == 403
    h = {'X-Wissi-Token': s.token}
    assert c.post('/app/open', json={'paths': [str(files['a'])]}, headers={'X-Wissi-Token': 'nope'}).status_code == 403
    r = c.post('/app/open', json={'paths': [str(files['a']), str(files['csv'])]}, headers=h).json()
    assert len(r['opened']) == 1 and r['layers'][0]['layer'] == 'cells'
    st = c.get('/app/state', headers=h).json()
    assert st['slides'][0]['name'] == 'a.svs' and st['version']
    sid = r['opened'][0]
    assert c.get(f'/wissi-viz/slides/{sid}/info.json').json()['width'] == 1200
    assert c.get(f'/wissi-viz/slides/{sid}/tiles/0/0/0').status_code == 200
    # DNS-rebinding guard: other Host names are refused
    assert c.get('/app', headers={'host': 'evil.example'}).status_code == 400
    assert c.post('/app/close', json={'slide': sid}, headers=h).json()['slides'] == []


def test_local_server_starts_and_stops(files):
    s = D.Session()
    s.open([files['a']])
    srv = D._Server(s.app())
    url = srv.start()
    assert url.startswith('http://127.0.0.1:')
    with urllib.request.urlopen(url + '/app', timeout=10) as r:
        assert b'wissi-viz' in r.read()
    srv.stop()
    assert not srv.thread.is_alive()
    s.shutdown()


def test_bridge_uses_native_dialogs(files):
    class Win:
        def __init__(self):
            self.calls = []

        def create_file_dialog(self, kind, allow_multiple=False, file_types=()):
            self.calls.append((kind, allow_multiple, file_types))
            if kind == 'folder':
                return [str(files['dir'])]
            return [str(files['a'])] if any('Whole-slide' in t for t in file_types) else [str(files['csv'])]

    fake = types.SimpleNamespace(FileDialog=types.SimpleNamespace(OPEN='open', FOLDER='folder'))
    sys.modules['webview'], saved = fake, sys.modules.get('webview')
    try:
        b = D.Bridge(D.Session())
        b._window = Win()
        r = b.pick_slides()
        assert r['opened'] and not r['errors']
        kinds = b._window.calls[0][2]
        assert '*.svs' in kinds[0] and '*.czi' in kinds[0] and '*.csv' in kinds[1]
        r = b.pick_layer(r['opened'][0])
        assert r['layers'][0]['layer'] == 'cells'
        r = b.pick_folder()
        assert len(r['slides']) >= 2
        assert b.state()['slides']
    finally:
        if saved is None:
            sys.modules.pop('webview', None)
        else:
            sys.modules['webview'] = saved


def test_native_window_wiring(files, monkeypatch):
    """run_native: window on /app?native=1 with the bridge, File menu, icon, persistent storage, drop + icon hooks."""
    seen = {}

    class Ev(list):
        def __iadd__(self, fn):
            self.append(fn)
            return self

    class Window:
        def __init__(self, **kw):
            seen['window'] = kw
            self.events = types.SimpleNamespace(shown=Ev(), loaded=Ev())

    def create_window(title, url, **kw):
        seen['title'], seen['url'] = title, url
        return Window(**kw)

    def start(**kw):
        seen['start'] = kw

    class MenuAction:
        def __init__(self, title, fn):
            self.title, self.fn = title, fn

    menu_mod = types.SimpleNamespace(
        Menu=lambda title, items: (title, items), MenuAction=MenuAction, MenuSeparator=lambda: None
    )
    fake = types.SimpleNamespace(create_window=create_window, start=start, menu=menu_mod)
    monkeypatch.setitem(sys.modules, 'webview', fake)
    monkeypatch.setitem(sys.modules, 'webview.menu', menu_mod)
    s = D.Session()
    D.run_native('http://127.0.0.1:9', s)
    assert seen['title'] == 'wissi-viz' and seen['url'] == 'http://127.0.0.1:9/app?native=1'
    assert isinstance(seen['window']['js_api'], D.Bridge)
    st = seen['start']
    want = 'icon.ico' if sys.platform == 'win32' else 'icon.png'  # WinForms reads .ico files only
    assert Path(st['icon']).name == want and Path(st['icon']).exists()
    assert st['private_mode'] is False and st['storage_path'].endswith('webview')
    ((title, items),) = st['menu']
    assert title == 'File' and [i.title for i in items if i] == [
        'Open slides…',
        'Open folder…',
        'Add cells / annotations…',
        'Install app shortcut',
    ]


def test_icons_are_valid():
    from PIL import Image

    png = Image.open(D.icon_path('png'))
    assert png.size == (512, 512) and png.mode == 'RGBA'
    ico = Image.open(D.icon_path('ico'))
    assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= set(ico.info['sizes'])
    icns = Image.open(D.icon_path('icns'))
    assert icns.size[0] >= 512
    svg = D.icon_path('svg').read_text(encoding='utf-8')
    assert svg.startswith('<svg') and 'viewBox="0 0 1024 1024"' in svg


def test_linux_shortcut(dirs, monkeypatch):
    monkeypatch.setenv('XDG_DATA_HOME', str(dirs / 'share'))
    monkeypatch.setattr(D, '_launch_command', lambda: ['/opt/my apps/bin/wissi-viz-app'])
    monkeypatch.setattr(D.sys, 'platform', 'linux')
    (p,) = D.install_shortcut()
    text = p.read_text(encoding='utf-8')
    assert p == dirs / 'share/applications/wissi-viz.desktop'
    assert 'Exec="/opt/my apps/bin/wissi-viz-app" %F' in text and 'Icon=wissi-viz' in text
    assert (dirs / 'share/icons/hicolor/512x512/apps/wissi-viz.png').exists()
    assert D.uninstall_shortcut() == [p] and not p.exists()


def test_macos_app_bundle(dirs, monkeypatch):
    import plistlib

    app = dirs / 'Applications' / 'wissi-viz.app'
    monkeypatch.setattr(D, 'shortcut_paths', lambda: [app])
    (p,) = D._install_macos(['/Users/me/.local/bin/wissi-viz-app'])
    info = plistlib.loads((p / 'Contents/Info.plist').read_bytes())
    assert info['CFBundleIconFile'] == 'wissi-viz' and info['CFBundleExecutable'] == 'wissi-viz'
    assert (p / 'Contents/Resources/wissi-viz.icns').stat().st_size > 10_000
    exe = (p / 'Contents/MacOS/wissi-viz').read_text(encoding='utf-8')
    assert exe.startswith('#!/bin/sh') and 'exec \'/Users/me/.local/bin/wissi-viz-app\' "$@"' in exe


def test_windows_shortcut_command(dirs, monkeypatch):
    """The PowerShell call that writes the .lnk files (run for real on Windows CI)."""
    runs = []

    def run(cmd, env=None, **kw):
        runs.append((cmd, env))
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(D.subprocess, 'run', run)
    monkeypatch.setattr(
        D, 'shortcut_paths', lambda: [dirs / 'Start' / 'wissi-viz.lnk', dirs / 'Desk' / 'wissi-viz.lnk']
    )
    out = D._install_windows([r'C:\Users\me\venv\Scripts\wissi-viz-app.exe'])
    assert len(out) == len(runs) == 2
    cmd, env = runs[0]
    assert cmd[0] == 'powershell' and 'WScript.Shell' in cmd[-1]
    assert env['WV_TARGET'].endswith('wissi-viz-app.exe') and env['WV_ICON'].endswith('.ico')


def test_cli_dispatches_desktop(monkeypatch):
    from wissi_viz import cli

    got = {}
    monkeypatch.setattr(D, 'main', lambda argv: got.setdefault('argv', argv))
    cli.main(['desktop', '--browser', 'x.svs'])
    assert got['argv'] == ['--browser', 'x.svs']


def test_page_in_browser_mode(files, tmp_path):
    """The app page with no native bridge: empty state, paste-a-path, slide list, viewer draws."""
    pytest.importorskip('playwright')
    from playwright.sync_api import sync_playwright

    s = D.Session(cache_dir=tmp_path / 'cache', remember=False)  # not this computer's remembered slides
    srv = D._Server(s.app())
    url = srv.start()
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1200, 'height': 800})
            slow = float(os.environ.get('WISSI_SWEEP_CPU_SLOWDOWN', '0') or 0)
            if slow > 1:  # a slow machine
                pg.context.new_cdp_session(pg).send('Emulation.setCPUThrottlingRate', {'rate': slow})
            errors = []
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.goto(url + '/app')
            pg.wait_for_selector('#empty .card h1')
            assert pg.is_visible('#empty')
            pg.click('#empty button[data-act=open]')  # no native dialogs here → the path box
            pg.wait_for_selector('#pathbox textarea', state='visible')
            # the button's handler is bound when the app finishes starting, which may also reset the box: on slow
            # machines a click (or text typed) before that is lost
            pg.wait_for_function('() => !!document.getElementById("b-paths").onclick', timeout=60000)
            pg.fill('#paths', f'"{files["a"]}"\n{files["b"]}\n{files["csv"]}')
            pg.wait_for_function('() => document.getElementById("paths").value.includes("csv")')
            pg.click('#b-paths')
            try:
                pg.wait_for_function('() => document.querySelectorAll("#slides li").length === 2', timeout=90000)
            except Exception as e:  # say what the page shows
                state = pg.evaluate('() => ({slides: document.querySelectorAll("#slides li").length, '
                                    'log: document.getElementById("log").textContent, '
                                    'paths: document.getElementById("paths").value, '
                                    'toasts: [...document.querySelectorAll(".toast")].map((t) => t.textContent)})')  # fmt: skip
                raise AssertionError(
                    f'slides did not open: {state}; page errors {errors}; server has '
                    f'{s.registry.layers and list(s.registry._entries)}'
                ) from e
            assert pg.is_hidden('#empty')
            assert pg.eval_on_selector('#n', 'e => e.value') == '2'
            pg.wait_for_function('() => document.getElementById("grid").panes.every(p => p.slide && p.slide.info)')
            pg.select_option('#n', '1')
            pg.wait_for_function('() => document.getElementById("grid").panes.length === 1')
            assert not errors, errors
            b.close()
    finally:
        srv.stop()
        s.shutdown()


def test_serve_page_lists_slides_and_opens_32_panes(files):
    """`wissi-viz serve` serves the same app UI: slide list from the registry, no file opening, up to 32 panes."""
    pytest.importorskip('playwright')
    from playwright.sync_api import sync_playwright

    from wissi_viz import SlideRegistry, TileService
    from wissi_viz.server import create_app

    reg = SlideRegistry(TileService())
    for p in (files['a'], files['b']):
        reg.add(p)
    srv = D._Server(create_app(reg))
    url = srv.start()
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1400, 'height': 900})
            errors = []
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.goto(url + '/')
            pg.wait_for_function('() => document.querySelectorAll("#slides li").length === 2')
            assert pg.is_hidden('#b-open') and pg.is_hidden('#b-layer')
            assert list(pg.eval_on_selector_all('#n option', 'os => os.map(o => +o.value)'))[-1] == 32
            pg.select_option('#n', '32')
            pg.wait_for_function('() => document.getElementById("grid").panes.length === 32')
            pg.wait_for_function('() => document.getElementById("grid").panes.every(p => p.slide && p.slide.info)')
            rects = pg.evaluate('() => document.getElementById("grid").panes.map(p => [p.rect.w, p.rect.h])')
            assert all(w > 50 and h > 50 for w, h in rects)  # a real grid, every pane visible
            # clicking a slide puts it in the active pane
            pg.click('#slides li:nth-child(2)')
            pg.wait_for_function('() => document.querySelector("#slides li:nth-child(2) .pane").textContent.length > 0')
            assert pg.request.get(url + '/demo').ok
            assert not errors, errors
            b.close()
    finally:
        srv.stop()
        reg.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows taskbar properties')
def test_windows_taskbar_name_and_icon_properties():
    """The window's AppUserModel properties (name, icon, relaunch command, id) are accepted by the shell."""
    import ctypes

    ctypes.windll.ole32.CoInitialize(None)
    user32 = ctypes.windll.user32
    user32.CreateWindowExW.restype = ctypes.c_void_p
    user32.CreateWindowExW.argtypes = [ctypes.c_ulong, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]  # fmt: skip
    hwnd = user32.CreateWindowExW(0, 'STATIC', 'wissi-viz test', 0, 0, 0, 10, 10, None, None, None, None)
    assert hwnd
    try:
        props = {D._AUMID['relaunch_command']: 'wissi-viz-app', D._AUMID['relaunch_icon']: f'{D.icon_path("ico")},0',
                 D._AUMID['relaunch_name']: D.APP_NAME, D._AUMID['id']: D.BUNDLE_ID}  # fmt: skip
        assert D._set_window_props(int(hwnd), props)
    finally:
        user32.DestroyWindow(ctypes.c_void_p(hwnd))


def test_browser_fallback_says_why(monkeypatch):
    """Without the app window, the browser page is told why (it shows it), so the fallback is never silent."""
    opened = []
    monkeypatch.setattr(D, '_chromium', lambda: None)
    monkeypatch.setattr(D.webbrowser, 'open', opened.append)
    srv = types.SimpleNamespace(thread=types.SimpleNamespace(is_alive=lambda: False))
    D.run_browser('http://127.0.0.1:9', srv, 'pywebview is not installed here')
    assert opened == ['http://127.0.0.1:9/app?fallback=pywebview%20is%20not%20installed%20here']
    D.run_browser('http://127.0.0.1:9', srv)
    assert opened[-1] == 'http://127.0.0.1:9/app'
