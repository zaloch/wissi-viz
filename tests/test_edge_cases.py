"""Awkward files through every stage: slides without resolution, odd sizes and types, blank or broken files, and
annotation / cell files with invalid shapes. Each either works or fails with a message that says what is wrong —
never a crash deep inside, a blank pane, or silently wrong pixels."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

pytest.importorskip('scipy')
pytest.importorskip('shapely')
pytest.importorskip('skimage')

P = '/wissi-viz'


def _tissue(h, w, rng):
    img = np.full((h, w, 3), 240, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy - h / 2) / (h * 0.35)) ** 2 + ((xx - w / 2) / (w * 0.35)) ** 2 < 1
    img[m] = (210, 140, 190)
    for _ in range(max(1, h * w // 900)):
        y, x = rng.integers(0, h), rng.integers(0, w)
        if m[y, x]:
            img[max(0, y - 4) : y + 4, max(0, x - 4) : x + 4] = (80, 45, 135)
    return img


def _svs(w, h):
    return f'Aperio Image Library v12 \r\n{w}x{h} [0,0 {w}x{h}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'


#: name → (how to write it, None if it opens, else a phrase its error must contain)
SLIDES: dict = {}


def slide(name, error=None):
    def deco(fn):
        SLIDES[name] = (fn, error)
        return fn

    return deco


@slide('no_resolution.tif')
def _(p, rng):
    tifffile.imwrite(p, _tissue(700, 900, rng), tile=(256, 256), photometric='rgb', metadata=None)


@slide('resolution_in_cm.tif')
def _(p, rng):
    tifffile.imwrite(p, _tissue(600, 600, rng), tile=(256, 256), photometric='rgb', resolution=(20000, 20000),
                     resolutionunit='CENTIMETER', metadata=None)  # fmt: skip


@slide('absurd_resolution.tif')
def _(p, rng):
    tifffile.imwrite(p, _tissue(300, 300, rng), photometric='rgb', resolution=(1, 1), resolutionunit='CENTIMETER',
                     metadata=None)  # fmt: skip


@slide('one_pixel.tif')
def _(p, rng):
    tifffile.imwrite(p, np.full((1, 1, 3), 128, np.uint8), photometric='rgb', metadata=None)


@slide('strip_7x300.tif')
def _(p, rng):
    tifffile.imwrite(p, _tissue(7, 300, rng), photometric='rgb', metadata=None)


@slide('odd size pyramid ü.svs')
def _(p, rng):
    big = _tissue(1003, 1501, rng)
    with tifffile.TiffWriter(p, bigtiff=True) as tw:
        tw.write(
            big, description=_svs(1501, 1003), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None
        )
        tw.write(big[::4, ::4].copy(), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None,
                 subfiletype=1)  # fmt: skip


@slide('grey.tif')
def _(p, rng):
    tifffile.imwrite(p, _tissue(500, 500, rng).mean(-1).astype(np.uint8), tile=(256, 256), metadata=None)


@slide('rgba.tif')
def _(p, rng):
    t = _tissue(400, 400, rng)
    tifffile.imwrite(p, np.concatenate([t, np.full((400, 400, 1), 255, np.uint8)], -1), tile=(256, 256),
                     photometric='rgb', extrasamples=['unassalpha'], metadata=None)  # fmt: skip


@slide('rgb16.tif')
def _(p, rng):
    tifffile.imwrite(p, _tissue(400, 400, rng).astype(np.uint16) * 257, tile=(256, 256), photometric='rgb',
                     metadata=None)  # fmt: skip


@slide('float_channels.ome.tif')
def _(p, rng):
    tifffile.imwrite(p, rng.random((2, 300, 400)).astype(np.float32), tile=(256, 256), metadata={'axes': 'CYX'})


@slide('seven_channels.ome.tif')
def _(p, rng):
    tifffile.imwrite(p, (rng.random((7, 300, 300)) * 60000).astype(np.uint16), tile=(256, 256),
                     metadata={'axes': 'CYX'})  # fmt: skip


@slide('time_z_channels.ome.tif')
def _(p, rng):
    tifffile.imwrite(p, (rng.random((2, 2, 2, 200, 300)) * 1000).astype(np.uint16),
                     metadata={'axes': 'TZCYX', 'PhysicalSizeX': 0.65, 'PhysicalSizeY': 0.65})  # fmt: skip


@slide('blank_glass.svs')
def _(p, rng):
    tifffile.imwrite(p, np.full((600, 800, 3), 245, np.uint8), description=_svs(800, 600), tile=(256, 256),
                     photometric='rgb', metadata=None)  # fmt: skip


@slide('all_black.svs')
def _(p, rng):
    tifffile.imwrite(p, np.zeros((600, 800, 3), np.uint8), description=_svs(800, 600), tile=(256, 256),
                     photometric='rgb', metadata=None)  # fmt: skip


@slide('empty.svs', error='empty (0 bytes)')
def _(p, rng):
    p.write_bytes(b'')


@slide('picture.tif', error='PNG picture')
def _(p, rng):
    p.write_bytes(b'\x89PNG\r\n\x1a\n' + b'0' * 64)


@slide('text.svs', error='no slide reader recognises it')
def _(p, rng):
    p.write_text('hello')


@slide('truncated.svs', error='truncated')
def _(p, rng):
    q = p.with_name('whole.svs')
    big = _tissue(1003, 1501, rng)
    with tifffile.TiffWriter(q, bigtiff=True) as tw:
        tw.write(
            big, description=_svs(1501, 1003), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None
        )
        tw.write(big[::4, ::4].copy(), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None,
                 subfiletype=1)  # fmt: skip
    raw = q.read_bytes()
    q.unlink()
    p.write_bytes(raw[: len(raw) // 3])


@pytest.fixture(scope='module')
def slides(tmp_path_factory):
    d = tmp_path_factory.mktemp('awkward')
    rng = np.random.default_rng(5)
    for name, (make, _) in SLIDES.items():
        make(d / name, rng)
    return d


@pytest.mark.parametrize('name', list(SLIDES))
def test_awkward_slide(slides, tmp_path, name):
    from starlette.testclient import TestClient

    from wissi_viz import findall, spatial
    from wissi_viz.metrics import region_metrics
    from wissi_viz.sam import SamService
    from wissi_viz.session import Session

    error = SLIDES[name][1]
    s = Session(cache_dir=tmp_path / 'cache', remember=False)
    try:
        r = s.open([slides / name], trusted=True)
        if error:
            assert not r['opened'] and len(r['errors']) == 1 and error in r['errors'][0], r['errors']
            assert '.cpp:' not in r['errors'][0]  # no reader build paths in messages
            return
        assert r['opened'] and not r['errors'], r['errors']
        sid = r['opened'][0]
        c = TestClient(s.app())
        info = c.get(f'{P}/slides/{sid}/info.json').json()
        assert c.get(f'{P}/slides/{sid}/thumbnail?size=128').status_code == 200
        h = c.get(f'{P}/slides/{sid}/histogram')
        assert h.status_code == 200 and h.json()['channels']
        ts = info['tile_size']
        chans = [None] if info['mode'] == 'rgb' else range(len(info['channels']))
        for li, lv in enumerate(info['levels']):  # the first and last tile of every level, every channel
            for tx, ty in ((0, 0), ((lv['width'] - 1) // ts, (lv['height'] - 1) // ts)):
                for ch in chans:
                    q = '' if ch is None else f'?c={ch}'
                    assert c.get(f'{P}/slides/{sid}/tiles/{li}/{tx}/{ty}{q}').status_code == 200, (li, tx, ty, ch)
        src = findall._source(s.registry, sid)
        geom, tissue = spatial.tissue_mask(src)
        if len(tissue):
            region_metrics(tissue, source=src, mpp=(src.info.mpp or (0.5,))[0])
        W, H = src.info.width, src.info.height
        rect = [0, 0, min(W, 400), min(H, 400)]
        try:
            SamService(backend='classic').segment(src, rect, [[rect[2] / 2, rect[3] / 2]], [1])
        except ValueError as e:  # a clear refusal is fine (a 1-px slide)
            assert 'nothing to segment' in str(e)
        findall.sam_service = lambda: SamService(backend='classic')
        try:
            findall.segment_everything(s.registry, sid, scope={'kind': 'view', 'rect': rect}, object_um=8)
        except ValueError as e:
            assert any(w in str(e) for w in ('no objects', 'nothing to segment', 'outside the slide')), e
    finally:
        s.shutdown()


def test_pixels_are_scaled_not_wrapped(slides):
    """16-bit RGB is scaled to 8 bit (it wrapped around); float channels are mapped from their range (they were
    cut to 0); extra dimensions read their first plane."""
    from wissi_viz.sources import open_slide

    a = np.asarray(open_slide(slides / 'rgb16.tif').read_region(0, 0, 0, 400, 400))
    want = np.asarray(_tissue(400, 400, np.random.default_rng(0)))  # same palette: 240 glass, 210/140/190 tissue
    assert a.dtype == np.uint8 and a.max() == 240 and set(np.unique(a)) <= set(np.unique(want)) | {80, 45, 135}
    f = open_slide(slides / 'float_channels.ome.tif')
    b = np.asarray(f.read_region(0, 0, 0, 400, 300, 1))
    assert f.info.dtype == 'uint16' and b.dtype == np.uint16 and b.max() > 60000 and np.percentile(b, 50) > 20000
    t = open_slide(slides / 'time_z_channels.ome.tif')
    assert len(t.info.channels) == 2 and t.info.mpp == (0.65, 0.65)


def test_segment_and_measure_without_resolution(slides, tmp_path):
    """No resolution in the file: segmentation assumes 0.5 µm/px and stores it; measuring the cells later uses it."""
    from wissi_viz import annio, spatial
    from wissi_viz.metrics import objects_from_layer, region_metrics
    from wissi_viz.segment import segment_slide
    from wissi_viz.sources import open_slide

    out = tmp_path / 'cells.parquet'
    segment_slide(slides / 'no_resolution.tif', out, model='classic', log=lambda *a: None)
    cells = annio.load(out)
    assert cells.mpp == 0.5 and len(cells) > 10
    src = open_slide(slides / 'no_resolution.tif')
    assert src.info.mpp is None
    objs = objects_from_layer(cells)
    assert objs.mpp == 0.5
    _, tissue = spatial.tissue_mask(src)
    df = region_metrics(tissue, source=src, objects=objs, mpp=objs.mpp)
    assert len(df) and df['obj_count'].iloc[0] > 0


# ------------------------------------------------------------------------------------------------ annotations
SQ = [[100, 100], [300, 100], [300, 300], [100, 300], [100, 100]]


def _fc(*feats):
    return json.dumps({'type': 'FeatureCollection', 'features': list(feats)})


def _f(geom, props=None):
    return {
        'type': 'Feature',
        'geometry': geom,
        'properties': {'classification': {'name': 'T'}} if props is None else props,
    }


ANNOTATIONS = {  # name → (content, expected features, phrase in the open message or None)
    'empty.geojson': (_fc(), 0, None),
    'null_geometry.geojson': (_fc(_f(None), _f({'type': 'Polygon', 'coordinates': [SQ]})), 1, None),
    'two_point_polygon.geojson': (
        _fc(
            _f({'type': 'Polygon', 'coordinates': [[[0, 0], [9, 9], [0, 0]]]}),
            _f({'type': 'Polygon', 'coordinates': [SQ]}),
        ),
        1,
        '1 invalid shape left out',
    ),  # fmt: skip
    'nan.geojson': (
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":'
        '{"type":"Polygon","coordinates":[[[1,1],[NaN,2],[3,3],[1,1]]]}}]}',
        0,
        '1 invalid shape left out',
    ),  # fmt: skip
    'bowtie.geojson': (
        _fc(_f({'type': 'Polygon', 'coordinates': [[[100, 100], [300, 300], [300, 100], [100, 300], [100, 100]]]})),
        1,
        None,
    ),  # fmt: skip
    'holes.geojson': (
        _fc(_f({'type': 'MultiPolygon', 'coordinates': [[SQ, [[150, 150], [200, 150], [200, 200], [150, 150]]]]})),
        1,
        None,
    ),  # fmt: skip
    'outside.geojson': (
        _fc(_f({'type': 'Polygon', 'coordinates': [[[1e7, 1e7], [1e7 + 50, 1e7], [1e7, 1e7 + 50], [1e7, 1e7]]]})),
        1,
        None,
    ),  # fmt: skip
    'mixed.geojson': (
        _fc(
            _f({'type': 'Point', 'coordinates': [50, 50]}),
            _f({'type': 'LineString', 'coordinates': [[0, 0], [100, 100]]}),
            _f({'type': 'Polygon', 'coordinates': [SQ]}),
        ),
        3,
        None,
    ),  # fmt: skip
    'numeric_class.geojson': (
        _fc(
            _f({'type': 'Polygon', 'coordinates': [SQ]}),
            _f({'type': 'Point', 'coordinates': [5, 5]}, {'classification': {'name': 7}}),
        ),
        2,
        None,
    ),  # fmt: skip
    'broken.geojson': ('{"type": "FeatureCollection", "features": [', None, 'not valid JSON'),
    'header_only.csv': ('x,y,class\n', 0, None),
    'bad_rows.csv': ('x,y,class\n10,20,T\n,30,T\nabc,40,S\n50,NaN,S\n60,70,\n', 2, None),
    'no_position.csv': ('a,b,class\n1,2,T\n', None, 'no centroid'),
    'semicolons.csv': ('x;y;class\n10;20;T\n30;40;S\n', 2, None),
    'bom.csv': ('﻿X,Y,Classification\n10,20,Tümor\n30,40,Stroma 2\n', 2, None),
}


@pytest.fixture(scope='module')
def one_slide(slides):
    return slides / 'odd size pyramid ü.svs'


@pytest.mark.parametrize('name', list(ANNOTATIONS))
def test_awkward_annotations(one_slide, tmp_path, name):
    from starlette.testclient import TestClient

    from wissi_viz.session import Session

    content, n, phrase = ANNOTATIONS[name]
    f = tmp_path / name
    f.write_text(content, encoding='utf-8')
    s = Session(cache_dir=tmp_path / 'cache', remember=False)
    try:
        r = s.open([one_slide, f], trusted=True)
        sid = r['opened'][0]
        if phrase:
            assert any(phrase in e for e in r['errors']), r['errors']
        else:
            assert not r['errors'], r['errors']
        if n is None:
            assert not r['layers']
            return
        lname = r['layers'][0]['layer']
        assert len(s.registry.layer(sid, lname)) == n
        c = TestClient(s.app())
        q = f'layer={lname}'
        for path in ('annotations', 'annotations/overview', 'annotations/summary', 'annotations/columns',
                     'annotations/metrics', 'annotations/object_metrics'):  # fmt: skip
            code = c.get(f'{P}/slides/{sid}/{path}?{q}').status_code
            assert code == 200 or (code == 409 and path.endswith('columns')), (path, code)  # columns: tables only
        assert c.get(f'{P}/slides/{sid}/annotations/hit?{q}&x=200&y=200&r=50').status_code == 200
        for fmt in ('geoparquet', 'geojson', 'qupath', 'halo', 'aperio', 'csv', 'coco'):
            assert c.get(f'{P}/slides/{sid}/annotations/export?{q}&fmt={fmt}').status_code == 200, fmt
    finally:
        s.shutdown()
