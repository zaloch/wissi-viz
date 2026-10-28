"""Region and object measurements, checked against geometry and images with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')
pytest.importorskip('skimage')
pytest.importorskip('pandas')

from wissi_viz import AnnotationLayer, open_slide
from wissi_viz.metrics import object_metrics, region_metrics
from wissi_viz.objects import ObjectTable


@pytest.fixture(scope='module')
def slide(tmp_path_factory):
    import tifffile

    d = tmp_path_factory.mktemp('m')
    img = np.full((2000, 2000, 3), 240, np.uint8)  # white glass
    img[200:1200, 200:1200] = (180, 90, 160)  # pink-purple tissue square
    yy, xx = np.mgrid[0:2000, 0:2000]
    nuclei = ((xx - 600) % 40 - 20) ** 2 + ((yy - 600) % 40 - 20) ** 2 < 36
    img[nuclei & (xx >= 200) & (xx < 1200) & (yy >= 200) & (yy < 1200)] = (70, 50, 140)  # dark purple dots
    desc = 'Aperio Image Library v12 \r\n2000x2000 [0,0 2000x2000] (256x256) JPEG/RGB Q=95|AppMag = 20|MPP = 0.5'
    tifffile.imwrite(d / 's.svs', img, tile=(256, 256), photometric='rgb', description=desc, metadata=None)
    return open_slide(d / 's.svs')


def _grid_objects(spacing=20, x0=200, x1=1200):
    g = np.arange(x0 + spacing / 2, x1, spacing)
    X, Y = np.meshgrid(g, g)
    n = X.size
    cls = np.where((np.arange(n) % 2) == 0, 'Tumor', 'Immune')
    return ObjectTable(X.ravel(), Y.ravel(), cls, {'marker': np.linspace(0, 1, n)}, mpp=0.5)


def test_shape_metrics_of_a_square_and_a_circle(slide):
    L = AnnotationLayer(name='rois')
    L.add_rect(200, 200, 1000, 1000, cls='Square')
    L.add_ellipse(1500, 1500, 200, 200, n=256, cls='Circle')
    t = region_metrics(L, mpp=0.5, intensity=False)
    sq, ci = t.iloc[0], t.iloc[1]
    assert sq['shape_area_um2'] == pytest.approx(500 * 500)
    assert sq['shape_perimeter_um'] == pytest.approx(2000)
    assert sq['shape_circularity'] == pytest.approx(math.pi / 4, rel=1e-6)
    assert sq['shape_solidity'] == pytest.approx(1) and sq['shape_extent'] == pytest.approx(1)
    assert sq['shape_aspect_ratio'] == pytest.approx(1) and sq['shape_eccentricity'] == pytest.approx(0, abs=1e-6)
    assert sq['shape_max_feret_um'] == pytest.approx(500 * math.sqrt(2))
    assert ci['shape_circularity'] == pytest.approx(1, abs=1e-3)
    assert ci['shape_area_um2'] == pytest.approx(math.pi * 100**2, rel=1e-3)
    assert ci['shape_major_axis_um'] == pytest.approx(200, rel=0.01)


def test_intensity_and_texture(slide):
    L = AnnotationLayer(name='rois')
    L.add_rect(200, 200, 1000, 1000, cls='Tissue')
    L.add_rect(1300, 1300, 600, 600, cls='Glass')
    t = region_metrics(L, source=slide)
    tis, gl = t.iloc[0], t.iloc[1]
    assert tis['analysis_mpp'] == pytest.approx(0.5)  # small enough to measure at full resolution
    assert tis['int_tissue_frac'] > 0.99 and gl['int_tissue_frac'] < 0.01
    assert tis['int_hematoxylin_od_mean'] > gl['int_hematoxylin_od_mean'] + 0.1
    assert 0.05 < tis['int_hematoxylin_od_pos_frac'] < 0.2  # the dark nuclei: ~7 % of the area
    assert tis['tex_contrast'] > gl.get('tex_contrast', 0) and 0 <= tis['tex_homogeneity'] <= 1
    assert tis['int_R_mean'] < 190 and gl['int_R_mean'] == pytest.approx(240)


def test_object_metrics_inside_regions(slide):
    objs = _grid_objects()  # regular grid: 10 µm apart, alternating classes
    L = AnnotationLayer(name='rois')
    L.add_rect(200, 200, 1000, 1000, cls='All')
    L.add_rect(200, 200, 500, 500, cls='Quarter')
    t = region_metrics(L, objects=objs, intensity=False)
    a, q = t.iloc[0], t.iloc[1]
    assert a['obj_count'] == 2500 and q['obj_count'] == 625
    assert a['obj_density_mm2'] == pytest.approx(2500 / 0.25)
    assert a['obj_pct_Tumor'] == pytest.approx(50) and a['obj_simpson'] == pytest.approx(0.5)
    assert a['obj_shannon'] == pytest.approx(math.log(2)) and a['obj_evenness'] == pytest.approx(1)
    assert a['obj_nn_mean_um'] == pytest.approx(10)
    assert a['obj_clark_evans'] == pytest.approx(2.0, rel=0.01)  # regular pattern: R = 2
    assert a['obj_morisita'] < 1  # dispersed
    assert a['obj_ripley_L_minus_r_25um'] < 5  # not clustered
    assert a['obj_dist_mean_um_Tumor_to_Immune'] == pytest.approx(10)
    assert a['obj_contact_frac_Tumor_with_Immune'] == pytest.approx(1)
    assert a['obj_mean_marker'] == pytest.approx(0.5, abs=0.01)
    assert 0 < a['obj_depth_mean_um'] < 250


def test_clustered_points_score_as_clustered():
    rng = np.random.default_rng(0)
    centres = rng.uniform(300, 1700, (20, 2))
    pts = np.concatenate([c + rng.normal(0, 15, (100, 2)) for c in centres])
    objs = ObjectTable(pts[:, 0], pts[:, 1], np.full(len(pts), 'A'), mpp=0.5)
    L = AnnotationLayer(name='r')
    L.add_rect(0, 0, 2000, 2000)
    t = region_metrics(L, objects=objs, intensity=False)
    assert t.iloc[0]['obj_clark_evans'] < 0.5 and t.iloc[0]['obj_morisita'] > 3
    assert t.iloc[0]['obj_ripley_L_minus_r_25um'] > 20


def test_object_metrics_shape_and_intensity(slide):
    L = AnnotationLayer(name='nuclei')
    for cx, cy in ((620, 620), (660, 620), (1500, 1500)):  # two on dark nuclei, one on glass
        L.add_ellipse(cx, cy, 6, 6, n=64, cls='Nucleus')
    L.add_ellipse(900, 900, 20, 10, n=64, cls='Elongated')
    t = object_metrics(L, source=slide)
    assert t['area_um2'].iloc[0] == pytest.approx(math.pi * 9, rel=0.02)
    assert t['circularity'].iloc[0] == pytest.approx(1, abs=0.01)
    assert t['eccentricity'].iloc[3] == pytest.approx(math.sqrt(1 - 0.25), abs=0.02)
    assert t['aspect_ratio'].iloc[3] == pytest.approx(2, rel=0.02)
    assert t['hematoxylin_mean'].iloc[0] > t['hematoxylin_mean'].iloc[2] + 0.2
    assert t['pixels'].iloc[0] == pytest.approx(math.pi * 36, rel=0.2)


def test_regions_from_a_cell_layer_and_geoparquet(tmp_path):
    pytest.importorskip('pyarrow')
    from wissi_viz import annio

    objs = _grid_objects()
    cells = objs.to_layer(name='cells')
    L = AnnotationLayer(name='rois')
    L.add_rect(200, 200, 1000, 1000, cls='All')
    p = L.export(tmp_path / 'rois.parquet')
    t = region_metrics(annio.read_geoparquet(p), objects=cells, mpp=0.5, intensity=False)
    assert t.iloc[0]['obj_count'] == 2500 and t.iloc[0]['region'] == 'All'


def test_metrics_endpoints(slide, tmp_path):
    from starlette.testclient import TestClient

    from wissi_viz import SlideRegistry, TileService
    from wissi_viz.server import create_app

    reg = SlideRegistry(TileService())
    sid = reg.add(slide.path if hasattr(slide, 'path') else slide.info.name)
    rois = AnnotationLayer(name='rois')
    rois.add_rect(200, 200, 1000, 1000, cls='Tissue')
    reg.add_annotations(sid, rois)
    o = _grid_objects()
    o.bbox = np.stack([o.x - 6, o.y - 6, o.x + 6, o.y + 6], 1)  # nuclei as 12-px ellipses
    reg.add_annotations(sid, o.to_layer(name='cells'))
    c = TestClient(create_app(reg))
    j = c.get(f'/wissi-viz/slides/{sid}/annotations/metrics', params={'layer': 'rois', 'objects': 'cells'}).json()
    row = dict(zip(j['columns'], j['rows'][0]))
    assert row['region'] == 'Tissue' and row['obj_count'] == 2500 and row['int_tissue_frac'] > 0.99
    csv = c.get(f'/wissi-viz/slides/{sid}/annotations/metrics', params={'layer': 'rois', 'fmt': 'csv'})
    assert csv.text.startswith('region,') and 'attachment' in csv.headers['content-disposition']
    o = c.get(f'/wissi-viz/slides/{sid}/annotations/object_metrics', params={'layer': 'cells', 'fmt': 'json'}).json()
    assert len(o['rows']) == 2500 and 'area_um2' in o['columns'] and 'hematoxylin_mean' in o['columns']
