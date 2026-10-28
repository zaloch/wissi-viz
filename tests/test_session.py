"""Opening things from the app page: folder limits and the in-page file picker on a server, connecting to tile APIs
(command line and the Connect dialog), the server key, and CORS that never exposes the app page."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from wissi_viz import session as S
from wissi_viz.session import Session, create_session_app

pytest.importorskip('fastapi')
ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG = ROOT / 'scripts' / 'probe_configs' / 'mock_api.json'


def _svs(path: Path, w=1200, h=900) -> Path:
    img = (np.random.default_rng(0).random((h, w, 3)) * 200 + 30).astype(np.uint8)
    desc = f'Aperio Image Library v12 \r\n{w}x{h} [0,0 {w}x{h}] (256x256) JPEG/RGB Q=95|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    return path


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        S,
        'user_dir',
        lambda kind, app='wissi-viz': (
            (tmp_path / 'user' / kind).mkdir(parents=True, exist_ok=True) or tmp_path / 'user' / kind
        ),
    )
    allowed = tmp_path / 'slides'
    (allowed / 'sub').mkdir(parents=True)
    _svs(allowed / 'a.svs')
    _svs(allowed / 'sub' / 'b.svs', 800, 600)
    (allowed / 'cells.csv').write_text('x,y,class\n10,10,A\n20,20,B\n', encoding='utf-8')
    (allowed / 'notes.docx').write_text('not a slide', encoding='utf-8')
    (allowed / '.hidden.svs').write_text('', encoding='utf-8')
    outside = tmp_path / 'private'
    outside.mkdir()
    _svs(outside / 'secret.svs')
    return allowed, outside


@pytest.fixture(scope='module')
def mock_api(tmp_path_factory):
    """The simulated vendor tile API (examples/tile_api) over real HTTP."""
    pytest.importorskip('httpx')
    import uvicorn

    sys.path.insert(0, str(ROOT / 'examples' / 'tile_api'))
    from mock_tile_api import build_catalog, create_api

    d = tmp_path_factory.mktemp('api')
    cat = build_catalog([str(_svs(d / 'he.svs', 2600, 1800))], n=3)
    app = create_api(cat, 'tok', latency=(0.0, 0.002), limit=50, error_rate=0.0, seed=1)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    yield f'http://127.0.0.1:{port}'
    server.should_exit = True


def _inline_spec(url: str) -> dict:
    """The mock API's description with the URL and token written out (no ${ENV}), as a user would paste it."""
    spec = json.loads(MOCK_CONFIG.read_text(encoding='utf-8'))
    spec.pop('_comment', None)
    spec['base_url'] = url
    spec['headers'] = {'Authorization': 'Bearer tok'}
    return spec


# ------------------------------------------------------------------------------------------- folder limits
def test_server_opens_only_inside_allowed_folders(tree):
    allowed, outside = tree
    s = Session(roots=[allowed], connect=False, remember=False)
    r = s.open([allowed / 'a.svs', outside / 'secret.svs', allowed / '..' / 'private' / 'secret.svs'])
    assert len(r['opened']) == 1 and len(r['errors']) == 2
    assert all('outside the folders' in e for e in r['errors'])
    assert s.open([outside / 'secret.svs'], trusted=True)['opened']  # command-line paths are trusted
    nothing = Session(roots=[], remember=False)
    assert not nothing.can_open and 'outside' in nothing.open([allowed / 'a.svs'])['errors'][0]


def test_browse_lists_what_opens_and_stays_inside(tree):
    allowed, outside = tree
    s = Session(roots=[allowed], remember=False)
    top = s.browse()
    assert top['dir'] == str(allowed.resolve()) and top['parent'] is None
    assert [(e['name'], e['kind']) for e in top['entries']] == [
        ('sub', 'folder'),
        ('a.svs', 'slide'),
        ('cells.csv', 'table'),
    ]
    sub = s.browse(str(allowed / 'sub'))
    assert sub['parent'] == str(allowed.resolve()) and [e['name'] for e in sub['entries']] == ['b.svs']
    for bad in (str(outside), str(allowed / '..')):
        with pytest.raises(PermissionError):
            s.browse(bad)
    two = Session(roots=[allowed, outside], remember=False)
    assert {e['path'] for e in two.browse()['entries']} == {str(allowed.resolve()), str(outside.resolve())}
    assert two.browse(str(allowed))['parent'] == ''  # back to the list of allowed folders
    with pytest.raises(PermissionError):
        Session(roots=[], remember=False).browse()


@pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need extra rights on Windows')
def test_symlink_out_of_an_allowed_folder_is_refused(tree):
    allowed, outside = tree
    os.symlink(outside / 'secret.svs', allowed / 'link.svs')
    s = Session(roots=[allowed], remember=False)
    assert 'link.svs' not in [e['name'] for e in s.browse()['entries']]
    assert s.open([allowed / 'link.svs'])['errors']


# ------------------------------------------------------------------------------------------- tile APIs
def test_connect_from_the_command_line_lists_and_registers_lazily(mock_api, monkeypatch, tmp_path):
    monkeypatch.setenv('MOCK_API_URL', mock_api)
    monkeypatch.setenv('MOCK_API_TOKEN', 'tok')
    s = Session(tmp_path / 'c', remember=False, roots=[], connect=False)
    r = s.connect(str(MOCK_CONFIG), trusted=True)  # trusted: ${ENV} expanded, even though visitors may not connect
    assert r['connected'] == 3 and r['service'] == 'custom' and len(r['opened']) == 1
    paths = [x['path'] for x in s.slides()]
    assert paths == ['custom:img-01', 'custom:img-02', 'custom:img-03']
    info = s.registry.info(r['opened'][0])
    assert info['width'] == 2600 and info['height'] == 1800
    one = s.connect(str(MOCK_CONFIG), images='img-02', trusted=True)
    assert one['connected'] == 1


def test_connect_failures_are_clear_and_leave_nothing_behind(mock_api, tmp_path):
    s = Session(tmp_path / 'c', remember=False)
    bad = _inline_spec(mock_api)
    bad['headers'] = {'Authorization': 'Bearer wrong'}
    with pytest.raises(Exception, match='401|Unauthorized'):
        s.connect(bad, images='img-01')
    assert s.slides() == [] and s.registry.ids() == []
    with pytest.raises(ValueError, match='no images'):
        s.connect({**_inline_spec(mock_api), 'list': None})


def test_what_a_server_visitor_may_not_do(mock_api, tree):
    allowed, outside = tree
    s = Session(roots=[allowed], connect=True, remember=False)
    leak = _inline_spec(mock_api)
    leak['headers'] = {'X-Steal': '${HOME}'}
    with pytest.raises(PermissionError, match='environment'):
        s.connect(json.dumps(leak))
    with pytest.raises(PermissionError, match='command line'):
        s.connect('os:getcwd')
    (outside / 'svc.json').write_text(json.dumps(_inline_spec(mock_api)), encoding='utf-8')
    with pytest.raises(PermissionError, match='outside'):
        s.connect(str(outside / 'svc.json'))
    with pytest.raises(PermissionError):
        Session(roots=[allowed], connect=False, remember=False).connect(_inline_spec(mock_api))
    assert s.connect(json.dumps(_inline_spec(mock_api)))['connected'] == 3  # plain JSON is fine


def test_http_connector_images_from_a_fixed_list():
    pytest.importorskip('httpx')
    from wissi_viz.connectors import HttpTileConnection, image_list

    c = HttpTileConnection({'base_url': 'http://x', 'tile': {'path': '/t'}, 'images': ['a', {'id': 'b', 'name': 'B'}]})
    assert c.images() == [('a', 'a'), ('b', 'B')]
    assert image_list(c, 'x, y\nz') == [('x', 'x'), ('y', 'y'), ('z', 'z')]


# ------------------------------------------------------------------------------------------- HTTP endpoints
def test_server_endpoints_key_token_and_cors(tree, mock_api):
    from starlette.testclient import TestClient

    allowed, outside = tree
    s = Session(roots=[allowed], connect=True, remember=False)
    c = TestClient(create_session_app(s, mode='server', cors='*', key='k3y'))
    page = c.get('/').text
    assert '"open": true' in page and '"key": true' in page and s.token in page
    tok = {'x-wissi-token': s.token}
    assert c.get('/app/state').status_code == 403
    assert c.get('/app/state', headers=tok).json()['slides'] == []
    assert c.post('/app/open', json={'paths': [str(allowed / 'a.svs')]}, headers=tok).status_code == 401
    both = {**tok, 'x-wissi-key': 'k3y'}
    r = c.post('/app/open', json={'paths': [str(allowed / 'a.svs'), str(outside / 'secret.svs')]}, headers=both).json()
    assert len(r['opened']) == 1 and 'outside' in r['errors'][0]
    assert (
        c.get('/app/browse', params={'dir': str(allowed / 'sub')}, headers=both).json()['entries'][0]['name'] == 'b.svs'
    )
    assert c.get('/app/browse', params={'dir': str(outside)}, headers=both).status_code == 403
    r = c.post(
        '/app/connect', json={'spec': json.dumps(_inline_spec(mock_api)), 'images': 'img-01,img-03'}, headers=both
    )
    assert r.status_code == 200 and r.json()['connected'] == 2
    assert c.post('/app/close', json={'slide': r.json()['opened'][0]}, headers=tok).status_code == 403  # shared list
    # CORS: other sites may use the tile API, never read the page (it holds the token)
    origin = {'origin': 'https://elsewhere.example'}
    assert c.get('/wissi-viz/slides', headers=origin).headers.get('access-control-allow-origin') == '*'
    assert 'access-control-allow-origin' not in c.get('/', headers=origin).headers


def test_server_without_permissions_offers_nothing(tree):
    from starlette.testclient import TestClient

    allowed, _ = tree
    s = Session(roots=[], connect=False, remember=False)
    s.open([allowed / 'a.svs'], trusted=True)
    c = TestClient(create_session_app(s, mode='server'))
    page = c.get('/').text
    assert '"open": false' in page and '"connect": false' in page
    tok = {'x-wissi-token': s.token}
    assert len(c.get('/app/state', headers=tok).json()['slides']) == 1
    assert c.post('/app/open', json={'paths': [str(allowed / 'a.svs')]}, headers=tok).status_code == 403
    assert c.post('/app/connect', json={'spec': '{}'}, headers=tok).status_code == 403


# ------------------------------------------------------------------------------------------- command line
def test_cli_serve_wires_the_options(tree, mock_api, monkeypatch, capsys):
    pytest.importorskip('uvicorn')
    import uvicorn

    from wissi_viz import cli

    allowed, _ = tree
    got = {}
    monkeypatch.setattr(uvicorn, 'run', lambda app, **kw: got.update(app=app, **kw))
    monkeypatch.setenv('MOCK_API_URL', mock_api)
    monkeypatch.setenv('MOCK_API_TOKEN', 'tok')
    cli.main(
        [
            'serve',
            str(allowed / 'a.svs'),
            '--allow-open',
            str(allowed),
            '--connect',
            str(MOCK_CONFIG),
            '--images',
            'img-01',
            '--key',
            'k',
            '--port',
            '8123',
        ]
    )
    from starlette.testclient import TestClient

    c = TestClient(got['app'])
    page = c.get('/').text
    assert '"open": true' in page and '"connect": false' in page and got['port'] == 8123
    token = page.split("token: '")[1].split("'")[0]
    names = [x['path'] for x in c.get('/app/state', headers={'x-wissi-token': token}).json()['slides']]
    assert any(n.endswith('a.svs') for n in names) and 'custom:img-01' in names
    assert 'custom: 1 image' in capsys.readouterr().err


def test_cli_help_lists_every_command(capsys):
    from wissi_viz import cli

    with pytest.raises(SystemExit):
        cli.main(['--help'])
    out = capsys.readouterr().out
    for cmd in ('--demo', '--benchmark', 'desktop', 'pack', 'segment', 'serve', 'convert', 'info'):
        assert cmd in out


# ------------------------------------------------------------------------------------------- the page, in a browser
def test_page_picker_and_connect_dialog(tree, mock_api):
    pytest.importorskip('playwright')
    import uvicorn
    from playwright.sync_api import sync_playwright

    allowed, _ = tree
    s = Session(roots=[allowed], connect=True, remember=False)
    with socket.socket() as sk:
        sk.bind(('127.0.0.1', 0))
        port = sk.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(
            create_session_app(s, mode='server', key='k3y'), host='127.0.0.1', port=port, log_level='warning'
        )
    )
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.02)
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1280, 'height': 800})
            pg.on('dialog', lambda d: d.accept('k3y'))  # the key prompt
            pg.goto(f'http://127.0.0.1:{port}/')
            pg.wait_for_selector('#b-open:not([hidden])')
            assert pg.is_visible('#b-connect') and not pg.is_visible('#slides li button')
            pg.click('#b-open')
            pg.wait_for_selector('#browser a:has-text("a.svs")')
            pg.click('#browser a:has-text("sub")')
            pg.wait_for_selector('#browser a:has-text("b.svs")')
            pg.dblclick('#browser a:has-text("b.svs")')  # click selects, double-click opens
            pg.wait_for_function('() => document.querySelectorAll("#slides li").length === 1')
            pg.wait_for_function('() => document.getElementById("grid").panes.length === 1')
            pg.click('#b-connect')
            pg.fill('#c-spec', '{"base_url": "nope"')
            pg.click('#c-go')
            pg.wait_for_function('() => document.getElementById("c-msg").className === "err"')
            pg.fill('#c-spec', json.dumps(_inline_spec(mock_api)))
            pg.fill('#c-images', 'img-02, img-03')
            pg.click('#c-go')
            pg.wait_for_function('() => document.querySelectorAll("#slides li").length === 3')
            assert not pg.is_visible('#connect')
            assert 'custom: 2 images' in pg.text_content('#log')
            pg.wait_for_function(
                '() => { const g = document.getElementById("grid"); return g.panes[0].slide && g.panes[0].slide.info; }',
                timeout=30000,
            )
            b.close()
    finally:
        srv.should_exit = True


def test_app_mounted_under_a_path(tree):
    """The app inside another ASGI app (NiceGUI's mount_app, a reverse-proxy subpath): every URL in the page
    carries the mount path, and CORS still covers the API only."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    allowed, _ = tree
    s = Session(roots=[], connect=False, remember=False)
    sid = s.open([allowed / 'a.svs'], trusted=True)['opened'][0]
    parent = FastAPI()
    parent.mount('/viewer', create_session_app(s, mode='server', cors='*'))
    c = TestClient(parent)
    page = c.get('/viewer/').text
    assert '/viewer/wissi-viz/static/index.js' in page and '/viewer/wissi-viz/static/app.js' in page
    assert "base: '/viewer'" in page and 'src="/viewer/app/icon.svg"' in page
    token = page.split("token: '")[1].split("'")[0]
    assert c.get('/viewer/app/state', headers={'x-wissi-token': token}).json()['slides'][0]['id'] == sid
    assert c.get('/viewer/app/icon.svg').status_code == 200
    origin = {'origin': 'https://elsewhere.example'}
    assert c.get('/viewer/wissi-viz/slides', headers=origin).headers.get('access-control-allow-origin') == '*'
    assert 'access-control-allow-origin' not in c.get('/viewer/', headers=origin).headers
