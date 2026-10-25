"""Run: pytest -q tests  (creates small synthetic slides in a temp dir)."""
from __future__ import annotations

import gzip
import json
import struct
import zlib

import numpy as np
import pytest
import tifffile

from wissi_viz import AnnotationLayer, CallbackSource, Level, SlideInfo, SlideRegistry, TileService, open_slide
from wissi_viz.tiles import PyramidSlide


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp('slides')
    rng = np.random.default_rng(0)
    img = (rng.random((3000, 4000, 3)) * 255).astype(np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 4000, dtype=np.uint8)[None, :]
    desc = 'Aperio Image Library v12 \r\n4000x3000 [0,0 4000x3000] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 'a.svs', bigtiff=True) as tw:
        o = dict(tile=(256, 256), compression='jpeg', photometric='rgb', metadata=None)
        tw.write(img, description=desc, **o)
        tw.write(img[::16, ::16].copy(), description='Aperio thumb', photometric='rgb', metadata=None)
        tw.write(img[::4, ::4].copy(), description='Aperio Image Library level', **o)
    # single-level plain tiled TIFF (no pyramid at all) → virtual levels
    tifffile.imwrite(d / 'flat.tif', img[:, :, 1].copy(), tile=(256, 256))
    gj = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[10, 10], [100, 10], [100, 80], [10, 80], [10, 10]]]},
         'properties': {'classification': {'name': 'Tumor', 'color': [255, 0, 0]}, 'score': 0.9}},
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [500, 600]}, 'properties': {'class': 'Lymph'}},
        {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': [[0, 0], [300, 300], [600, 0]]}, 'properties': {}},
    ]}
    (d / 'a.geojson').write_text(json.dumps(gj))
    return d


def test_svs_reader(data):
    s = open_slide(data / 'a.svs')
    assert s.info.mode == 'rgb' and s.info.mpp == (0.5, 0.5)
    assert [lv.downsample for lv in s.info.levels] == [1.0, 4.0]
    t = s.read_tile(0, 7, 5)
    assert t.shape == (3000 - 5 * 512, 4000 - 7 * 512, 3)  # edge tile clipped


def test_virtual_levels_single_level_tiff(data):
    s = open_slide(data / 'flat.tif')
    assert len(s.info.levels) == 1 and s.info.mode == 'channels'
    p = PyramidSlide(s)
    assert max(p.info.levels[-1].width, p.info.levels[-1].height) <= 512
    assert all(b.downsample / a.downsample == 2 for a, b in zip(p.info.levels, p.info.levels[1:]))
    top = p.fetch_tile(len(p.info.levels) - 1, 0, 0, 0)
    assert top.shape == (p.info.levels[-1].height, p.info.levels[-1].width)
    full = tifffile.imread(data / 'flat.tif').astype(np.float64)
    f = 2 ** (len(p.info.levels) - 1)
    ref = full[:f * (3000 // f), :f * (4000 // f)].reshape(3000 // f, f, 4000 // f, f).mean((1, 3))
    assert np.abs(top[:ref.shape[0], :ref.shape[1]] - ref).mean() < 1.5


def test_annotations_bucketing_and_encoding():
    rng = np.random.default_rng(1)
    xy = rng.random((50000, 2)).astype(np.float32) * 20000
    t = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    polys = [np.stack([x + 6 * np.cos(t), y + 6 * np.sin(t)], 1) for x, y in xy[:20000]]
    L = AnnotationLayer.from_polygons(polys, classes=['a', 'b'] * 10000)
    L.add_rect(100, 100, 15000, 15000, cls='roi')
    c = L.compiled()
    assert sum(b[5] for b in c.buckets) == len(L) == 20001
    assert sorted(c.order.tolist()) == list(range(len(L)))
    roi_bucket = [b for b in c.buckets if b[7] > 10000]
    assert len(roi_bucket) == 1 and roi_bucket[0][5] == 1  # huge ROI isolated in its own size tier
    ov = L.overview_bytes()
    assert ov[:4] == b'WSA1'
    jl = struct.unpack('<I', ov[4:8])[0]
    meta = json.loads(ov[8:8 + jl])
    assert meta['count'] == 20001 and meta['buckets'] == len(c.buckets)
    det = L.bucket_bytes(0)
    jl = struct.unpack('<I', det[4:8])[0]
    m = json.loads(det[8:8 + jl])
    nfeat = int(c.buckets[0][5])
    assert m['segments'] == nfeat * 9  # closed 9-gons
    assert m['tri_verts'] == nfeat * 7 * 3


def test_geojson_and_sidecar(data):
    L = AnnotationLayer.from_geojson(data / 'a.geojson')
    assert len(L) == 3
    assert L.classes['Tumor'] == '#ff0000'
    assert L.properties(0)['score'] == 0.9
    back = L.to_geojson()
    assert back['features'][1]['geometry']['type'] == 'Point'


def test_aperio_and_halo_xml(tmp_path):
    ap = tmp_path / 's.xml'
    ap.write_text('<Annotations><Annotation Id="1" Name="Tumor" LineColor="255"><Regions>'
                  '<Region Id="1" Type="0"><Vertices><Vertex X="1" Y="1"/><Vertex X="50" Y="1"/><Vertex X="50" Y="40"/></Vertices></Region>'
                  '<Region Id="2" Type="2"><Vertices><Vertex X="100" Y="100"/><Vertex X="140" Y="120"/></Vertices></Region>'
                  '</Regions></Annotation></Annotations>')
    L = AnnotationLayer.load(ap)
    assert len(L) == 2 and L.classes['Tumor'] == '#ff0000'
    hx = tmp_path / 's.annotations'
    hx.write_text('<Annotations><Annotation Name="Layer 1" LineColor="65280"><Regions>'
                  '<Region Type="Rectangle" NegativeROA="0"><Vertices><V X="0" Y="0"/><V X="10" Y="10"/></Vertices></Region>'
                  '<Region Type="Polygon" NegativeROA="1"><Vertices><V X="0" Y="0"/><V X="5" Y="0"/><V X="5" Y="5"/></Vertices></Region>'
                  '</Regions></Annotation></Annotations>')
    H = AnnotationLayer.load(hx)
    assert len(H) == 2 and 'Layer 1 (negative)' in H.classes


def test_callback_source_passthrough():
    calls = []

    def tile(level, x, y, c):
        calls.append((level, x, y))
        return b'\xff\xd8\xff\xe0fakejpeg'

    info = SlideInfo(width=1000, height=800, levels=[Level(1000, 800, 1.0), Level(250, 200, 4.0)], tile_size=256)
    src = CallbackSource(info, read_tile=tile)
    reg = SlideRegistry(TileService(memory_bytes=10 * 2**20), warm=False)
    sid = reg.add(src, slide_id='remote-1')
    _, pyr = reg.get(sid)
    f1 = reg.tiles.get(sid, pyr, src.version, 0, 1, 1, None)
    f2 = reg.tiles.get(sid, pyr, src.version, 0, 1, 1, None)
    assert f1.result() == (b'\xff\xd8\xff\xe0fakejpeg', 'image/jpeg')
    assert f2.result()[0] == f1.result()[0]
    assert calls.count((0, 1, 1)) == 1  # coalesced + cached → one remote call


def test_http_api(data, tmp_path):
    from fastapi.testclient import TestClient

    from wissi_viz import create_app
    reg = SlideRegistry(TileService(disk_dir=tmp_path / 'cache'), warm=False)
    sid = reg.add(data / 'a.svs', slide_id='a')
    c = TestClient(create_app(reg))
    info = c.get('/wissi-viz/slides/a/info.json').json()
    assert info['width'] == 4000 and info['version']
    r = c.get(f'/wissi-viz/slides/a/tiles/0/0/0?v={info["version"]}')
    assert r.status_code == 200 and r.headers['content-type'] == 'image/jpeg'
    assert 'immutable' in r.headers['cache-control']
    assert c.get('/wissi-viz/slides/a/tiles/0/99/0').status_code == 404
    assert c.get('/wissi-viz/slides/nope/info.json').status_code == 404
    layers = c.get('/wissi-viz/slides/a/annotations').json()
    assert layers and layers[0]['count'] == 3
    ov = c.get('/wissi-viz/slides/a/annotations/overview', params={'layer': layers[0]['name']})
    assert ov.content[:4] == b'WSA1'  # TestClient transparently gunzips
    feat = c.get('/wissi-viz/slides/a/annotations/feature/0', params={'layer': layers[0]['name']}).json()
    assert feat['class'] == 'Tumor'
    r = c.post('/wissi-viz/slides/a/annotations/add', params={'layer': 'Drawings'},
               json={'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [1, 2]}, 'properties': {}})
    assert r.json()['ok'] and 'Drawings' in reg.layers('a')
    assert c.get('/wissi-viz/static/index.js').status_code == 200
    assert c.get('/wissi-viz/static/../server.py').status_code == 404
    assert list((tmp_path / 'cache').rglob('*.jpg'))  # persisted to the disk cache


def test_channel_tile_encoding(data):
    reg = SlideRegistry(warm=False)
    sid = reg.add(data / 'flat.tif', slide_id='flat')
    e, pyr = reg.get(sid)
    b, mt = reg.tiles.get(sid, pyr, e.source.version, 0, 0, 0, 0).result()
    raw = np.frombuffer(zlib.decompress(b), np.uint8).reshape(512, 512)
    assert mt == 'application/x-wissi-raw'
    assert np.array_equal(raw, tifffile.imread(data / 'flat.tif')[:512, :512])
    assert reg.info(sid)['channels'][0]['window'][1] > 0
    _ = gzip


def test_czi_reader(tmp_path):
    pyczi = pytest.importorskip("pylibCZIrw.czi")  # LGPL: only used to *write* a test file
    p = tmp_path / 'x.czi'
    a = (np.arange(1500 * 2000, dtype=np.uint32).reshape(1500, 2000) % 4096).astype(np.uint16)
    with pyczi.create_czi(str(p), exist_ok=True) as w:
        for c in range(2):
            w.write(data=(a + c)[..., None], plane={'C': c, 'Z': 0, 'T': 0}, location=(0, 0))
        w.write_metadata('t', channel_names={0: 'DAPI', 1: 'CD8'}, scale_x=0.5e-6, scale_y=0.5e-6)
    s = open_slide(p)
    assert s.info.mode == 'channels' and [c.name for c in s.info.channels] == ['DAPI', 'CD8']
    assert s.info.mpp == pytest.approx((0.5, 0.5))
    t = s.read_tile(0, 1, 1, 1)
    assert np.array_equal(t, (a + 1)[512:1024, 512:1024])
    top = PyramidSlide(s).fetch_tile(len(s.info.levels) - 1, 0, 0, 0)
    assert top.dtype == np.uint16 and top.shape[1] <= 512


def test_czifile_reader_mosaic_and_pyramid(tmp_path):
    """Permissive CZI reader: overlapping mosaic + pyramid subblocks, exact at every level."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from czi_writer import mosaic_with_pyramid

    from wissi_viz.sources.czi import CziSource
    rng = np.random.default_rng(0)
    H, W = 1100, 1700
    img = (rng.random((2, H, W)) * 4000).astype(np.uint16)
    p = tmp_path / 'm.czi'
    mosaic_with_pyramid(p, img, tile=300, overlap=20, levels=(1, 2, 4), channel_names=['DAPI', 'CD3'], mpp=0.5)
    s = CziSource(p)
    assert [lv.downsample for lv in s.info.levels] == [1.0, 2.0, 4.0]
    assert [c.name for c in s.info.channels] == ['DAPI', 'CD3'] and s.info.mpp == (0.5, 0.5)
    assert np.array_equal(s.read_region(0, 0, 0, W, H, 1), img[1])
    for lv, ds in ((1, 2), (2, 4)):
        ref = img[0][:H // ds * ds, :W // ds * ds].reshape(H // ds, ds, W // ds, ds).mean((1, 3)).astype(np.uint16)
        assert np.array_equal(s.read_region(lv, 0, 0, W // ds, H // ds, 0), ref)
    rgb = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    mosaic_with_pyramid(tmp_path / 'rgb.czi', rgb, levels=(1,))
    r = CziSource(tmp_path / 'rgb.czi')
    assert r.info.mode == 'rgb' and np.array_equal(r.read_region(0, 0, 0, W, H), rgb)


def test_slideio_backend_matches(tmp_path):
    pytest.importorskip('slideio')
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from czi_writer import mosaic_with_pyramid
    rng = np.random.default_rng(1)
    img = (rng.random((1, 700, 900)) * 4000).astype(np.uint16)
    p = tmp_path / 'x.czi'
    mosaic_with_pyramid(p, img, levels=(1, 2), channel_names=['DAPI'])
    a, b = open_slide(p), open_slide(p, backend='slideio')
    assert type(b).__name__ == 'SlideioSource'
    for lv in range(2):
        assert np.array_equal(a.read_tile(lv, 0, 0, 0), b.read_tile(lv, 0, 0, 0))


def test_tiff_fast_path_matches_zarr(tmp_path, data):
    """Direct tile decode (bypassing zarr) must be pixel-identical, incl. OME SubIFD pyramids and edges."""
    rng = np.random.default_rng(5)
    img = (rng.random((3, 1100, 1500)) * 60000).astype(np.uint16)
    p = tmp_path / 'p.ome.tif'
    with tifffile.TiffWriter(p, ome=True) as tw:
        o = dict(tile=(256, 256), compression='zlib', photometric='minisblack')
        tw.write(img, subifds=2, metadata={'axes': 'CYX'}, **o)
        tw.write(img[:, ::2, ::2].copy(), subfiletype=1, **o)
        tw.write(img[:, ::4, ::4].copy(), subfiletype=1, **o)
    for path in (p, data / 'a.svs'):
        s = open_slide(path)
        chans = range(len(s.info.channels)) if s.info.mode == 'channels' else [None]
        for lv, L in enumerate(s.info.levels):
            for c in chans:
                for x, y, w, h in ((0, 0, L.width, L.height), (37, 11, 300, 260), (max(0, L.width - 200), max(0, L.height - 150), 200, 150)):
                    w, h = min(w, L.width - x), min(h, L.height - y)
                    assert np.array_equal(s.read_region(lv, x, y, w, h, c), s._read_zarr(lv, x, y, w, h, c)), (path, lv, c)
        assert s._fast, f'fast path disabled for {path}'
    s = open_slide(p)
    assert len(s.info.levels) == 3 and np.array_equal(s.read_region(1, 0, 0, 750, 550, 2), img[2, ::2, ::2])
