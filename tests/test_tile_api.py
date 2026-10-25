"""End-to-end against a simulated vendor tile API over real HTTP: auth, DeepZoom/padded/16-bit tiles,
429 + Retry-After, random 503s, 404 background, annotations/objects in µm, and caching."""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip('httpx')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples' / 'tile_api'))

from wissi_viz import SlideRegistry, TileService, open_slide  # noqa: E402
from wissi_viz.sources.callback import _decode  # noqa: E402


@pytest.fixture(scope='module')
def api(tmp_path_factory):
    import uvicorn
    from mock_tile_api import build_catalog, create_api
    d = tmp_path_factory.mktemp('api')
    rng = np.random.default_rng(0)
    from scipy import ndimage
    noise = ndimage.gaussian_filter(rng.random((1800, 2600, 3)).astype(np.float32), (1.5, 1.5, 0))
    rgb = ((noise - noise.min()) / np.ptp(noise) * 200 + 30).astype(np.uint8)
    rgb[:, 2000:] = 250  # background strip → 404 tiles in the DeepZoom style
    desc = 'Aperio Image Library v12 \r\n2600x1800 [0,0 2600x1800] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 'he.svs', bigtiff=True) as tw:
        tw.write(rgb, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    fl = (rng.random((3, 1500, 2100)) * 50000).astype(np.uint16)
    with tifffile.TiffWriter(d / 'fl.ome.tif', ome=True) as tw:
        o = dict(tile=(256, 256), compression='zlib', photometric='minisblack')
        tw.write(fl, subifds=2, metadata={'axes': 'CYX', 'PhysicalSizeX': 0.5, 'PhysicalSizeY': 0.5}, **o)
        tw.write(fl[:, ::2, ::2].copy(), subfiletype=1, **o)
        tw.write(fl[:, ::4, ::4].copy(), subfiletype=1, **o)
    roi = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'name': 'T'},
           'geometry': {'type': 'Polygon', 'coordinates': [[[100, 100], [600, 100], [600, 400], [100, 100]]]}}]}  # µm
    cells = {'Class': ['A', 'B', 'A'], 'Centroid X µm': [10.0, 20.0, 30.0], 'Centroid Y µm': [5.0, 5.0, 5.0]}
    extras = {str(d / 'he.svs'): {'annotations': roi, 'objects': cells}}
    cat = build_catalog([str(d / 'he.svs'), str(d / 'fl.ome.tif')], n=4, annotations=extras)
    app = create_api(cat, 'tok', latency=(0.0, 0.004), limit=3, error_rate=0.05, seed=1)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.02)
    yield f'http://127.0.0.1:{port}', d, rgb, fl
    server.should_exit = True


def _conn(url, token='tok'):
    from connector import DemoApiConnection
    return DemoApiConnection(url, token)


def test_auth_and_catalog(api):
    import httpx
    url, *_ = api
    assert httpx.get(url + '/api/images').status_code == 401
    imgs = _conn(url).list_images()
    assert [i['style'] for i in imgs] == ['dzi', 'channels', 'tiles', 'render']


def test_pixels_match_source_and_cache(api):
    import httpx
    url, d, rgb, fl = api
    conn = _conn(url)
    reg = SlideRegistry(TileService(memory_bytes=256 * 2**20, workers=8), warm=False)
    ids = {i['style']: conn.register(reg, i['id']) for i in conn.list_images()}

    # 16-bit channels (lossless PNG, padded edge tiles, coarsest-first ids): exact at every level
    e, pyr = reg.get(ids['channels'])
    assert pyr.info.mode == 'channels' and pyr.info.dtype == 'uint16' and pyr.info.mpp == (0.5, 0.5)
    for lv, f in ((0, 1), (1, 2), (2, 4)):
        L = pyr.info.levels[lv]
        a = e.source.read_region(lv, 0, 0, L.width, L.height, 1)
        assert np.array_equal(a, fl[1, ::f, ::f][:L.height, :L.width]), lv

    # DeepZoom (254 + 1 px overlap, JPEG): every tile fetched through the full pipeline lines up
    e2, pyr2 = reg.get(ids['dzi'])
    assert pyr2.info.overlap == 1 and pyr2.info.tile_size == 254
    src = e2.source
    t = src.read_tile(0, 2, 3).astype(int)
    h, w = t.shape[:2]
    errs = {(dx, dy): np.abs(t[2:-2, 2:-2] - rgb[762 + dy + 2:762 + dy + h - 2, 508 + dx + 2:508 + dx + w - 2]).mean()
            for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)}
    assert min(errs, key=errs.get) == (0, 0)          # geometry exact: best alignment at zero offset
    assert errs[(0, 0)] < 8 and errs[(1, 1)] > 1.5 * errs[(0, 0)]  # remaining error is JPEG noise
    data, mt = src.fetch_tile(0, 2, 3)
    assert mt == 'image/jpeg' and _decode(data).shape[:2] == (256, 256)  # passed through with overlap
    bg = src.read_tile(0, 9, 1)  # background strip → API 404 → drawn as background
    assert (bg == 255).all()

    # many concurrent tiles: some requests hit the 429 limit / random 503s and are retried
    futs = [reg.tiles.get(sid, reg.get(sid)[1], reg.get(sid)[0].source.version, 0, tx, ty, c)
            for sid, c in ((ids['dzi'], None), (ids['tiles'], None), (ids['channels'], 0), (ids['render'], None))
            for tx in range(3) for ty in range(3)]
    for f in futs:
        f.result(timeout=60)
    st = conn.stats()
    api_stats = httpx.get(url + '/api/_stats').json()
    assert api_stats['r429'] > 0 and api_stats['r503'] > 0 and api_stats['r404'] > 0
    assert st['rate_limited'] > 0 and st['retries'] > 0 and st['errors'] == 0

    # the same tiles again: served from wissi-viz's cache, zero upstream requests
    before = httpx.get(url + '/api/_stats').json()['requests']
    for sid, c in ((ids['dzi'], None), (ids['tiles'], None), (ids['channels'], 0)):
        e3, p3 = reg.get(sid)
        for tx in range(3):
            for ty in range(3):
                reg.tiles.get(sid, p3, e3.source.version, 0, tx, ty, c).result()
    assert httpx.get(url + '/api/_stats').json()['requests'] == before


def test_annotations_and_objects_in_microns(api):
    url, *_ = api
    conn = _conn(url)
    reg = SlideRegistry(warm=False)
    sid = conn.register(reg, 'img-01')
    layers = reg.layers(sid)
    assert set(layers) == {'ROIs (API)', 'Cells (API)'}
    bb = layers['ROIs (API)'].compiled().bbox
    assert bb == pytest.approx([200, 200, 1200, 800])        # µm → px with mpp 0.5
    cells = layers['Cells (API)']
    assert len(cells) == 3 and not cells.visible and cells.compiled().bbox[0] == pytest.approx(20)


def test_bad_token_fails_cleanly(api):
    from wissi_viz.connectors import RemoteError
    url, *_ = api
    conn = _conn(url, token='wrong')
    reg = SlideRegistry(warm=False)
    sid = conn.register(reg, 'img-01')
    with pytest.raises(Exception) as ei:
        reg.get(sid)
    assert '401' in str(ei.value) or isinstance(ei.value, RemoteError)
    _ = open_slide
