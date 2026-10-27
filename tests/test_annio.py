"""Annotation file formats: GeoParquet (lossless, fast, spatially filterable) and conversions."""
from __future__ import annotations

import json

import numpy as np
import pytest

from wissi_viz import annio
from wissi_viz.annotations import AnnotationLayer


def _mixed() -> AnnotationLayer:
    L = AnnotationLayer(name='review', classes={'Tumor': '#ff0000', 'Stroma': '#00aa00'})
    L.add_polygon([[0, 0], [100, 0], [100, 80], [0, 80]], holes=[[[20, 20], [40, 20], [40, 40]]], cls='Tumor',
                  props={'score': 0.5, 'note': 'margin'})
    L.add_polygon([[200, 200], [260, 210], [230, 280]], cls='Stroma', props={'score': 1.5})
    L.add_point(5, 7, cls='Pin')
    L.add_polyline([[1, 1], [50, 60], [70, 10]], cls='Ruler')
    return L


def _cells(n=2000, seed=0) -> AnnotationLayer:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 4000, (n, 2))
    r = rng.uniform(4, 9, n)
    L = AnnotationLayer.from_boxes(xy[:, 0] - r, xy[:, 1] - r, xy[:, 0] + r, xy[:, 1] + r,
                                   classes=rng.choice(['Tumor', 'Immune'], n), name='cells')
    L._columns = {'area': (np.pi * r * r).astype(np.float32), 'marker': rng.random(n)}
    L._col_offset = 0
    return L


def test_wkb_is_valid_and_round_trips():
    F = annio.from_layer(_mixed())
    offs, data = annio.encode_wkb(F)
    G = annio.decode_wkb((offs, data))
    assert G.kind.tolist() == F.kind.tolist() and np.allclose(G.coords, F.coords)
    assert G.feat_ring_off.tolist() == F.feat_ring_off.tolist()
    shapely = pytest.importorskip('shapely')
    geoms = shapely.from_wkb([bytes(data[offs[i]:offs[i + 1]]) for i in range(len(F))])
    assert [g.geom_type for g in geoms] == ['Polygon', 'Polygon', 'Point', 'LineString']
    assert all(g.is_valid for g in geoms)
    assert geoms[0].area == pytest.approx(100 * 80 - 200)       # the hole is subtracted
    assert F.shell_area()[0] == pytest.approx(100 * 80 - 200)


def test_geoparquet_round_trip_is_lossless(tmp_path):
    pytest.importorskip('pyarrow')
    L = _mixed()
    p = L.export(tmp_path / 'a.parquet', mpp=0.25)
    M = AnnotationLayer.load(p)
    F, G = annio.from_layer(L), annio.from_layer(M)
    assert len(G) == 4 and sorted(G.class_names) == sorted(F.class_names)
    by = lambda X: {c: X.coords[X.ring_off[X.feat_ring_off[i]]].tolist() for i, c in enumerate(X.class_of())}  # noqa: E731
    assert by(F) == by(G)
    assert M.classes['Tumor'] == '#ff0000'
    props = [M.properties(i) for i in range(len(M))]
    assert {'score': 0.5, 'note': 'margin'}.items() <= {k: v for p in props for k, v in p.items() if p.get('note')}.items()
    meta = json.loads(__import__('pyarrow.parquet', fromlist=['x']).read_schema(p).metadata[b'geo'])
    assert meta['version'] == '1.1.0' and meta['columns']['geometry']['encoding'] == 'WKB'
    assert set(meta['columns']['geometry']['geometry_types']) == {'Point', 'LineString', 'Polygon'}


def test_geoparquet_big_layer_and_bbox_read(tmp_path):
    pytest.importorskip('pyarrow')
    L = _cells(50_000)
    p = L.export(tmp_path / 'cells.parquet')
    F = annio.read_geoparquet(p)
    assert len(F) == 50_000 and set(F.class_names) == {'Tumor', 'Immune'} and set(F.props) == {'area', 'marker'}
    assert np.allclose(F.props['area'], L._columns['area'])
    sub = annio.read_geoparquet(p, bbox=(0, 0, 1000, 1000))
    c = sub.centroid()
    assert 0 < len(sub) < 50_000 and (c.max(0) < 1000 + 10).all()
    gj = L.export(tmp_path / 'cells.geojson')
    assert p.stat().st_size < 0.5 * gj.stat().st_size                # far smaller than GeoJSON


def test_geopandas_reads_it(tmp_path):
    gpd = pytest.importorskip('geopandas')
    pytest.importorskip('pyarrow')
    p = _mixed().export(tmp_path / 'a.parquet')
    df = gpd.read_parquet(p)
    assert len(df) == 4 and set(df['class']) >= {'Tumor', 'Stroma'}


@pytest.mark.parametrize('fmt,ext', [('geojson', '.geojson'), ('qupath', '.geojson'), ('halo', '.annotations'),
                                     ('aperio', '.xml'), ('coco', '.coco.json')])
def test_round_trip_through_other_formats(tmp_path, fmt, ext):
    L = _mixed()
    p = L.export(tmp_path / f'a{ext}', fmt=fmt)
    M = AnnotationLayer.load(p)
    F, G = annio.from_layer(L), annio.from_layer(M)
    polys = lambda X: sorted(X.coords[X.ring_off[X.feat_ring_off[i]]:X.ring_off[X.feat_ring_off[i] + 1]].round(1).tolist()  # noqa: E731
                             for i in range(len(X)) if X.kind[i] == annio.GEO_POLYGON)
    assert polys(G) == polys(F) if fmt != 'halo' else len(polys(G)) >= 2
    names = set(G.class_names)
    assert {'Tumor', 'Stroma'} <= names or fmt == 'halo' and {'Tumor', 'Stroma'} <= {n.replace(' (negative)', '') for n in names}
    if fmt == 'qupath':
        doc = json.loads(p.read_text())
        f0 = doc['features'][0]['properties']
        assert f0['objectType'] == 'annotation' and f0['classification']['color'] == [255, 0, 0]


def test_csv_and_mask(tmp_path):
    L = _mixed()
    p = L.export(tmp_path / 'a.csv', wkt=True)
    rows = p.read_text().splitlines()
    assert rows[0].startswith('id,class,type,centroid_x') and rows[0].endswith('wkt') and len(rows) == 5
    m = L.export(tmp_path / 'm.tif', width=300, height=300, downsample=2)
    import tifffile
    a = tifffile.imread(m)
    assert a.shape == (150, 150)
    assert a[20, 40] == 1 and a[15, 15] == 0 and a[110, 115] == 2      # feature 1, its hole, feature 2


def test_cli_convert(tmp_path, capsys):
    pytest.importorskip('pyarrow')
    from wissi_viz.cli import main
    src = _mixed().export(tmp_path / 'a.geojson')
    main(['convert', str(src), str(tmp_path / 'a.parquet')])
    main(['convert', str(tmp_path / 'a.parquet'), str(tmp_path / 'a.xml'), '--to', 'aperio'])
    assert len(AnnotationLayer.load(tmp_path / 'a.xml')) == 4
    assert '→' in capsys.readouterr().out


def test_export_endpoint(tmp_path):
    pytest.importorskip('pyarrow')
    import tifffile
    from starlette.testclient import TestClient

    from wissi_viz import SlideRegistry, TileService
    from wissi_viz.server import create_app
    img = np.full((512, 512, 3), 200, np.uint8)
    tifffile.imwrite(tmp_path / 's.tif', img, tile=(256, 256), photometric='rgb')
    reg = SlideRegistry(TileService())
    sid = reg.add(tmp_path / 's.tif')
    reg.add_annotations(sid, _mixed())
    c = TestClient(create_app(reg))
    r = c.get(f'/wissi-viz/slides/{sid}/annotations/export', params={'layer': 'review'})
    assert r.status_code == 200 and r.content[:4] == b'PAR1'
    assert 'review.parquet' in r.headers['content-disposition']
    r = c.get(f'/wissi-viz/slides/{sid}/annotations/export', params={'layer': 'review', 'fmt': 'qupath'})
    assert r.json()['type'] == 'FeatureCollection'
    assert c.get(f'/wissi-viz/slides/{sid}/annotations/export', params={'layer': 'review', 'fmt': 'nope'}).status_code == 400
