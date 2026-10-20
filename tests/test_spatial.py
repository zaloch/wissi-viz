"""Spatial analyses against brute force, closed-form geometry and planted patterns."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')

from wissi_viz import spatial as sp  # noqa: E402
from wissi_viz.objects import ObjectTable  # noqa: E402

MPP = 0.5


def table(xy, classes=None, cols=None):
    xy = np.asarray(xy, float)
    return ObjectTable(xy[:, 0], xy[:, 1], None if classes is None else np.asarray(classes, object), cols or {}, mpp=MPP)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_nearest_neighbors_brute_force(rng):
    xy = rng.random((600, 2)) * 4000
    cl = rng.choice(['A', 'B', 'C'], 600)
    t = table(xy, cl)
    for s, g in (('A', 'B'), ('A', 'A'), ('B', ['A', 'C'])):
        r = sp.nearest_neighbors(t, s, g)
        S, T = np.nonzero(t.mask(s))[0], np.nonzero(t.mask(g))[0]
        D = np.hypot(*(xy[S][:, None] - xy[T][None]).transpose(2, 0, 1))
        if s == g:
            D[S[:, None] == T[None]] = np.inf
        assert np.allclose(r.dist_um, D.min(1) * MPP)
        assert np.array_equal(r.tgt_idx, T[D.argmin(1)])
    L = sp.nearest_neighbors(t, 'A', 'B').layer()
    assert len(L) == (cl == 'A').sum() and L.colormap is not None


def test_proximity_brute_force(rng):
    xy = rng.random((500, 2)) * 3000
    cl = rng.choice(['T', 'I'], 500)
    t = table(xy, cl)
    r = sp.proximity(t, 'T', 'I', radius_um=100)
    S, T = np.nonzero(cl == 'T')[0], np.nonzero(cl == 'I')[0]
    D = np.hypot(*(xy[S][:, None] - xy[T][None]).transpose(2, 0, 1)) * MPP
    assert np.array_equal(r.counts, (D <= 100).sum(1))
    same = sp.proximity(t, 'T', 'T', 100)
    D2 = np.hypot(*(xy[S][:, None] - xy[S][None]).transpose(2, 0, 1)) * MPP
    assert np.array_equal(same.counts, (D2 <= 100).sum(1) - 1)


def test_distance_to_roi_and_bands():
    from shapely.geometry import box
    roi = box(1000, 1000, 3000, 3000)  # 1000 µm square at mpp 0.5
    pts = [[2000, 2000], [1000, 2000], [900, 2000], [3200, 3000], [2900, 1500]]
    t = table(pts, ['A'] * 5)
    d = sp.distance_to_roi(t, roi)
    assert np.allclose(d, [-500, 0, 50, 100, -50])
    br = sp.margin_bands(roi, (-100, 0, 100), objs=t)
    # outward band 0..+100 µm: 1000² ... area = s·4·d + π d² (rounded corners); inward: s² − (s−2d)²
    s, dd = 1.0, 0.1  # mm
    rows = br.table.to_dict('records')
    assert rows[0]['area_mm2'] == pytest.approx(s * s - (s - 2 * dd) ** 2, rel=1e-3)
    assert rows[1]['area_mm2'] == pytest.approx(4 * s * dd + np.pi * dd * dd, rel=2e-3)
    assert rows[0]['n_total'] == 1 and rows[1]['n_total'] == 2  # −50 µm | 0 and +50 µm
    assert len(br.layer()) == 2


def test_infiltration_profile_uniform_density(rng):
    from shapely.geometry import box
    n = 40000
    xy = rng.random((n, 2)) * 8000  # 4 mm × 4 mm at 0.5 µm/px → 2500 cells/mm²
    t = table(xy, ['A'] * n)
    roi = box(2000, 2000, 6000, 6000)
    pr = sp.infiltration_profile(t, roi, ['A'], range_um=(-400, 400), bin_um=100, within=box(0, 0, 8000, 8000))
    assert np.allclose(pr.density['A'], n / 16, rtol=0.12)
    assert pr.plot()['series']


def test_roi_composition_part_of_whole(rng):
    from shapely.geometry import box
    xy = rng.random((4000, 2)) * 4000
    cl = np.where(xy[:, 0] < 2000, 'Tumor', 'Stroma')
    cl[rng.random(4000) < 0.2] = 'CD8'
    t = table(xy, cl)
    rois = {'left': box(0, 0, 2000, 4000), 'right': box(2000, 0, 4000, 4000)}
    res = sp.roi_composition(t, rois, whole=box(0, 0, 4000, 4000))
    rows = {r['roi']: r for r in res.table.to_dict('records')}
    assert rows['Whole']['n_objects'] == 4000
    assert rows['left']['n_objects'] + rows['right']['n_objects'] == 4000
    for c in ('Tumor', 'Stroma', 'CD8'):
        assert rows['left'][f'% of whole {c}'] + rows['right'][f'% of whole {c}'] == pytest.approx(100)
        assert rows['left'][f'n {c}'] == ((cl == c) & (xy[:, 0] < 2000)).sum()
    assert rows['left']['area % of whole'] == pytest.approx(50)
    assert rows['left']['% Stroma'] == 0 and rows['Whole']['area_mm2'] == pytest.approx(4.0)
    assert res.plot('sunburst')['series'][0]['type'] == 'sunburst'
    assert len(res.layer(value='density CD8 /mm2')) == 2


def test_density_grid_and_hotspot(rng):
    n = 20000
    xy = rng.random((n, 2)) * 6000
    hot = rng.normal([1500, 1500], 120, (1500, 2))
    t = table(np.vstack([xy, hot]), ['bg'] * n + ['hot'] * 1500)
    g = sp.density_grid(t, None, cell_um=100, hotspots=True)
    assert g.counts.sum() == n + 1500                                   # every object in exactly one cell
    bg = sp.density_grid(table(xy, ['bg'] * n), None, cell_um=100)
    assert np.median(bg.values) == pytest.approx(n / 9, rel=0.15)     # 3 mm × 3 mm
    centre = np.argmin(np.hypot(g.centers[:, 0] - 1500, g.centers[:, 1] - 1500))
    assert g.gi_z[centre] > 5 and np.median(g.gi_z) < 1
    assert len(g.hotspot_layer()) > 0 and g.layer().colormap is not None
    ratio = sp.density_grid(t, 'hot', denominator=None, cell_um=100)
    assert ratio.values.max() > 50 * np.median(ratio.values[ratio.values > 0]) or np.median(ratio.values) == 0


def test_ripley_csr_cluster_regular(rng):
    from shapely.geometry import box
    area = box(0, 0, 4000, 4000)
    csr = table(rng.random((1500, 2)) * 4000, ['A'] * 1500)
    r = sp.ripley(csr, 'A', area=area, radii_um=[25, 50, 100, 150], n_sim=19)
    assert np.all(np.abs(r.H) < 12)                                     # L(r) ≈ r under CSR
    parents = rng.random((30, 2)) * 4000
    cl = (parents[rng.integers(0, 30, 1500)] + rng.normal(0, 60, (1500, 2))).clip(0, 3999)
    rc = sp.ripley(table(cl, ['A'] * 1500), 'A', area=area, radii_um=[25, 50, 100], n_sim=19)
    assert np.all(rc.L > rc.L_hi)                                       # clustered: above envelope
    g = np.stack(np.meshgrid(np.arange(50, 4000, 100), np.arange(50, 4000, 100)), -1).reshape(-1, 2)
    rr = sp.ripley(table(g, ['A'] * len(g)), 'A', area=area, radii_um=[20, 40], n_sim=19)
    assert np.all(rr.L < rr.L_lo)                                       # lattice: below envelope (inhibition)
    assert r.plot()['series'] and rc.plot('g')['series']


def test_neighborhood_enrichment_planted(rng):
    a = rng.random((800, 2)) * 6000
    b = a[:400] + rng.normal(0, 10, (400, 2))   # B right next to A
    c = rng.random((800, 2)) * 6000
    t = table(np.vstack([a, b, c]), ['A'] * 800 + ['B'] * 400 + ['C'] * 800)
    e = sp.neighborhood_enrichment(t, radius_um=15, n_perm=99)
    i = {k: n for n, k in enumerate(e.classes)}
    # planted pairs dominate; under label permutation the other pairs look relatively depleted
    assert e.z[i['A'], i['B']] > 20 and e.z[i['A'], i['B']] > e.z[i['A'], i['C']] + 20
    assert np.allclose(e.z, e.z.T)
    rnd = table(rng.random((2000, 2)) * 6000, rng.choice(['A', 'B', 'C'], 2000))
    assert np.abs(sp.neighborhood_enrichment(rnd, radius_um=40, n_perm=99).z).max() < 4  # null: nothing enriched


def test_cellular_neighborhoods_recover_regions(rng):
    left = rng.random((3000, 2)) * [3000, 6000]
    right = rng.random((3000, 2)) * [3000, 6000] + [3000, 0]
    cl = np.concatenate([rng.choice(['T', 'S'], 3000, p=[.9, .1]), rng.choice(['I', 'S'], 3000, p=[.8, .2])])
    t = table(np.vstack([left, right]), cl)
    cn = sp.cellular_neighborhoods(t, k=15, n_neighborhoods=2)
    side = np.r_[np.zeros(3000), np.ones(3000)].astype(int)
    agree = max((cn.labels == side).mean(), (cn.labels != side).mean())
    assert agree > 0.95
    assert cn.composition.shape == (2, 3) and np.allclose(cn.composition.sum(1), 1, atol=1e-6)


def test_delaunay_graph_lattice():
    g = np.stack(np.meshgrid(np.arange(10) * 40.0, np.arange(10) * 40.0), -1).reshape(-1, 2)  # 20 µm spacing
    t = table(g, (['A', 'B'] * 50))
    gr = sp.delaunay_graph(t, max_edge_um=21)
    assert len(gr.edges) == 2 * 10 * 9                                  # only axis-aligned neighbours
    m, cls = gr.interactions()
    assert m.sum() - np.trace(m) // 1 >= 0 and gr.degree().max() == 4


def test_morans_i(rng):
    from scipy import sparse
    xy = rng.random((1500, 2)) * 6000
    smooth = np.sin(xy[:, 0] / 800) + np.cos(xy[:, 1] / 900)
    t = table(xy, None, {'smooth': smooth, 'noise': rng.normal(size=1500)})
    s, n = sp.morans_i(t, 'smooth', k=8, permutations=99), sp.morans_i(t, 'noise', k=8)
    assert s['I'] > 0.8 and s['z'] > 20 and s['p_perm'] <= 0.01
    assert abs(n['z']) < 3.5
    # brute-force I with dense weights
    _, nn = sp._tree(xy).query(xy, 9)
    W = np.zeros((1500, 1500))
    W[np.repeat(np.arange(1500), 8), nn[:, 1:].ravel()] = 1 / 8
    z = smooth - smooth.mean()
    assert s['I'] == pytest.approx(len(z) / W.sum() * z @ W @ z / (z @ z))
    _ = sparse


def test_h_score_and_positivity():
    od = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.05])
    t = table(np.arange(12).reshape(6, 2) * 10.0, ['Tumor'] * 6, {'DAB': od, 'Ki67': [0, 500, 0, 500, 500, 0]})
    h = sp.h_score(t, 'DAB', (0.2, 0.4, 0.6)).to_dict('records')[0]
    assert h['% 0'] == pytest.approx(200 / 6) and h['% 3+'] == pytest.approx(200 / 6)
    assert h['H-score'] == pytest.approx((1 * 1 + 2 * 1 + 3 * 2) / 6 * 100)
    p = sp.positivity(t, {'Ki67': ('Ki67', 250)}).to_dict('records')[0]
    assert p['% Ki67+'] == pytest.approx(50)


def test_phenotyping_rules():
    t = table(np.zeros((4, 2)), None, {'CD3': np.array([900, 900, 10, 10.]), 'CD8': np.array([900, 5, 900, 5.]),
                                       'CK': np.array([0, 0, 0, 800.])})
    lab = t.phenotype({'CD3': ('CD3', 100), 'CD8': ('CD8', 100), 'CK': ('CK', 100)},
                      rules={'CD8 T': {'CD3': True, 'CD8': True}, 'CD4 T': {'CD3': True, 'CD8': False}, 'Tumor': {'CK': True}})
    assert list(lab) == ['CD8 T', 'CD4 T', 'Other', 'Tumor']
    auto = t.phenotype({'CD3': ('CD3', 100), 'CD8': ('CD8', 100)}, inplace=False)
    assert list(auto) == ['CD3+CD8+', 'CD3+', 'CD8+', 'Negative']


def test_tissue_mask_and_stain_vectors(rng):
    pytest.importorskip('skimage')
    from wissi_viz.sources import Level, SlideInfo, SlideSource

    H = W = 800
    yy, xx = np.mgrid[:H, :W]
    tissue = ((xx - 300) ** 2 + (yy - 400) ** 2 < 220 ** 2) | ((xx - 620) ** 2 / 120 ** 2 + (yy - 300) ** 2 / 200 ** 2 < 1)
    h_vec, e_vec = np.array(sp.RUIFROK['hematoxylin']), np.array(sp.RUIFROK['eosin'])
    ch = rng.random((H, W)) * 0.8 * tissue
    ce = rng.random((H, W)) * 0.6 * tissue
    od = ch[..., None] * h_vec + ce[..., None] * e_vec
    img = (255 * 10 ** (-od)).clip(0, 255).astype(np.uint8)

    class Mem(SlideSource):
        def __init__(self):
            super().__init__()
            self.info = SlideInfo(W, H, [Level(W, H, 1.0)], mpp=(2.0, 2.0))

        def read_region(self, level, x, y, w, h, channel=None):
            return img[y:y + h, x:x + w]
    geom, layer = sp.tissue_mask(Mem(), min_area_um2=2000, smooth_um=6)
    import shapely
    truth = tissue.sum()
    inter = shapely.contains_xy(geom, xx.ravel() + .5, yy.ravel() + .5).reshape(H, W)
    iou = (inter & tissue).sum() / (inter | tissue).sum()
    assert iou > 0.97 and len(layer) == 1 and truth > 0  # the two blobs overlap → one region
    v = sp.estimate_stain_vectors(Mem())
    ang = lambda a, b: np.degrees(np.arccos(abs(np.dot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b)))  # noqa: E731
    assert ang(v['hematoxylin'], h_vec) < 6 and ang(v['second'], e_vec) < 6
