"""Remote (PathCore / HALOLink-style) slides: pyramid mapping, overlap/padding, async, retries,
concurrency limits, annotations. Uses mock connections backed by a synthetic SVS."""
from __future__ import annotations

import concurrent.futures as cf
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).parent))

from mock_remote import MockHaloLink, MockPathCore, _Backing  # noqa: E402

from wissi_viz import SlideRegistry, TileService  # noqa: E402
from wissi_viz.connectors import RemoteError, RemoteImageInfo, RemotePyramid, RemoteSlideSource, TileNotFound  # noqa: E402
from wissi_viz.sources.callback import _decode  # noqa: E402


@pytest.fixture(scope='module')
def svs(tmp_path_factory):
    d = tmp_path_factory.mktemp('remote')
    rng = np.random.default_rng(3)
    H, W = 2400, 3200
    from scipy import ndimage
    noise = ndimage.gaussian_filter(rng.random((H, W, 3)).astype(np.float32), (1.5, 1.5, 0))
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    img = np.ascontiguousarray((noise * 220 + 20).astype(np.uint8))  # fine texture: 1-px misalignment is obvious
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=95|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 'r.svs', bigtiff=True) as tw:
        o = dict(tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
        tw.write(img, description=desc, **o)
        tw.write(img[::16, ::16].copy(), description='Aperio thumb', photometric='rgb', metadata=None)
        tw.write(np.ascontiguousarray(img[::4, ::4]), description='Aperio Image Library level', **o)
    return d / 'r.svs'


def test_deepzoom_layout():
    p = RemotePyramid.deepzoom(16000, 12000, 254, 1)
    assert p.level_ids[0] == 14 and p.levels[0] == (16000, 12000)
    assert p.levels[1] == (8000, 6000) and p.level_ids[1] == 13
    w, h = p.levels[-1]
    assert w <= 254 and h <= 254 and p.levels[-2][0] > 254
    q = RemotePyramid.from_dzi('<?xml version="1.0"?><Image TileSize="510" Overlap="2" Format="jpg" '
                               'xmlns="http://schemas.microsoft.com/deepzoom/2008"><Size Width="5000" Height="3000"/></Image>')
    assert (q.tile_size, q.overlap, q.levels[0]) == (510, 2, (5000, 3000))
    e = RemotePyramid.explicit(1000, 800, 256, [(250, 200), (1000, 800), (500, 400)], level_ids=['c', 'f', 'm'])
    assert e.levels == [(1000, 800), (500, 400), (250, 200)] and e.level_ids == ['f', 'm', 'c']


def _mad(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def test_halolink_overlap_crop_is_pixel_exact(svs):
    halo = MockHaloLink(svs, fmt='PNG')  # lossless → geometry must match exactly
    src = halo.source('img-1')
    b = _Backing(svs)
    assert src.info.overlap == 1 and src.info.tile_size == 254
    for level in range(len(src.info.levels)):
        ds = src.info.levels[level].downsample
        lv = src.info.levels[level]
        nx, ny = -(-lv.width // 254), -(-lv.height // 254)
        for tx, ty in {(0, 0), (min(1, nx - 1), min(1, ny - 1)), (nx - 1, ny - 1), (min(2, nx - 1), 0)}:
            got = src.read_tile(level, tx, ty)
            h, w = got.shape[:2]
            assert (w, h) == (min(254, lv.width - tx * 254), min(254, lv.height - ty * 254))
            assert np.array_equal(got, b.region(ds, tx * 254, ty * 254, w, h)), (level, tx, ty)
    # pass-through: interior tiles keep their overlap (256 = 254 + 2); the GPU crops them
    data, mt = MockHaloLink(svs).source('img-1').fetch_tile(0, 1, 1)
    assert mt == 'image/jpeg' and _decode(data).shape[:2] == (256, 256)


def test_pathcore_async_padded_coarsest_first(svs):
    pc = MockPathCore(svs)
    src = pc.source('S-1')
    b = _Backing(svs)
    assert src.pyramid.level_ids == ['L0', 'L1']  # re-ordered finest-first, ids preserved
    lv0 = src.info.levels[0]
    last = (lv0.width - 1) // 256
    t = src.read_tile(0, last, 0)
    assert t.shape == (256, lv0.width - last * 256, 3)    # padding removed
    assert _mad(t, b.region(1, last * 256, 0, t.shape[1], 256)) < 1e-6  # PNG → exact
    data, mt = src.fetch_tile(0, last, 0)
    assert mt == 'image/png' and _decode(data).shape[:2] == (256, 256)  # padded bytes passed through


def test_retries_not_found_and_errors(svs):
    halo = MockHaloLink(svs, fail_rate=0.3)
    halo.retries = 8
    src = halo.source('img')
    for tx in range(4):
        src.read_tile(1, tx, 0)
    assert src.stats['retries'] > 0 and src.stats['errors'] == 0

    def missing(level, c, r, ch):
        raise TileNotFound
    info = RemoteImageInfo(width=600, height=400, tile_size=256, mpp=0.5, background='#f0f0f0')
    s2 = RemoteSlideSource(info.build_pyramid(), missing, background='#f0f0f0')
    assert (s2.read_tile(0, 1, 1) == 240).all()

    def broken(level, c, r, ch):
        raise ConnectionError('down')
    s3 = RemoteSlideSource(info.build_pyramid(), broken, retries=1)
    with pytest.raises(RemoteError):
        s3.read_tile(0, 0, 0)


def test_concurrency_limit_and_coalescing(svs):
    halo = MockHaloLink(svs)
    halo.max_concurrency = 3
    halo.__init__(svs)  # re-create the semaphore with the new limit
    reg = SlideRegistry(TileService(memory_bytes=64 * 2**20, workers=16), warm=False)
    sid = halo.register(reg, 'img-7')
    e, pyr = reg.get(sid)
    futs = [reg.tiles.get(sid, pyr, e.source.version, 0, tx, ty, None) for tx in range(6) for ty in range(4)]
    futs += [reg.tiles.get(sid, pyr, e.source.version, 0, 0, 0, None) for _ in range(10)]  # duplicates
    cf.wait(futs)
    assert all(f.exception() is None for f in futs)
    assert halo.max_active <= 3
    assert halo.calls == 24  # duplicates coalesced


def test_lazy_registration(svs):
    calls = []

    class Counting(MockHaloLink):
        def image_info(self, image_id):
            calls.append(image_id)
            return super().image_info(image_id)
    c = Counting(svs)
    reg = SlideRegistry(warm=False)
    sid = c.register(reg, 'abc/def 1')
    assert sid == 'halo-abc_def_1' and calls == []
    reg.info(sid)
    assert calls == ['abc/def 1']


def test_annotations_from_api(svs):
    geo = svs.parent / 'cells.geojson'
    import json
    feats = []
    for i in range(500):
        x, y = 50 + (i % 25) * 100, 50 + (i // 25) * 100
        ring = [[x, y], [x + 20, y], [x + 20, y + 15], [x, y + 15], [x, y]]
        feats.append({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [ring]},
                      'properties': {'classification': {'name': ['Tumor', 'Immune'][i % 2]}, 'area': 300.0}})
    geo.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}))
    reg = SlideRegistry(warm=False)
    halo, pc = MockHaloLink(svs, geojson_path=geo), MockPathCore(svs)
    h_id, p_id = halo.register(reg, 'h'), pc.register(reg, 'p')
    hl = reg.layers(h_id)
    assert set(hl) == {'HALO annotations', 'HALO objects'}
    objs = hl['HALO objects']
    assert len(objs) == 500 and set(objs.classes) == {'Tumor', 'Immune'}
    assert objs.properties(3)['Area'] == 300.0
    roi = hl['HALO annotations']
    assert roi.line_width == 3 and roi.classes['Tumor ROI'] == '#00ff00'
    pl = reg.layers(p_id)['PathCore annotations']
    bb = pl.compiled().bbox  # µm → px through the slide's mpp
    W, H = 3200, 2400
    assert bb[0] == pytest.approx(W * .55, abs=1) and bb[3] == pytest.approx(H * .6, abs=1)


def test_http_info_and_upstream_errors(svs):
    from fastapi.testclient import TestClient

    from wissi_viz import create_app
    reg = SlideRegistry(warm=False)
    sid = MockHaloLink(svs).register(reg, 'x')

    class Down(MockHaloLink):
        retries = 0

        def tile(self, *a, **k):
            raise ConnectionError('upstream down')
    bad = Down(svs).register(reg, 'down')
    c = TestClient(create_app(reg))
    info = c.get(f'/wissi-viz/slides/{sid}/info.json').json()
    assert info['overlap'] == 1 and info['levels'][0]['remote_level'] == '12'
    assert all(not lv['virtual'] for lv in info['levels'])
    r = c.get(f'/wissi-viz/slides/{sid}/tiles/0/1/1?v={info["version"]}')
    assert r.status_code == 200 and r.headers['content-type'] == 'image/jpeg'
    assert c.get(f'/wissi-viz/slides/{bad}/tiles/0/0/0').status_code == 502


def test_async_client_on_bound_loop(svs):
    """An async client bound to the app's loop (e.g. NiceGUI's) keeps working from worker threads."""
    import asyncio
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    seen = []

    class Bound(MockPathCore):
        async def tile(self, *a, **k):
            seen.append(asyncio.get_running_loop())
            return await super().tile(*a, **k)
    pc = Bound(svs)
    pc.loop = loop
    src = pc.source('s')
    src.read_tile(0, 0, 0)
    assert seen and all(lp is loop for lp in seen)
    loop.call_soon_threadsafe(loop.stop)
