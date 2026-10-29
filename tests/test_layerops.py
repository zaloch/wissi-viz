"""Chained analyses: objects → regions → only the cells inside → any analysis; and one analysis on many images."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')
pytest.importorskip('skimage')
pytest.importorskip('pandas')

sys.path.insert(0, str(Path(__file__).parent))
from test_findall import data  # noqa: F401  (the synthetic slide + cells fixture)

P = '/wissi-viz'


@pytest.fixture()
def session(data, tmp_path):  # noqa: F811
    from wissi_viz.session import Session

    s = Session(cache_dir=tmp_path / 'cache', remember=False)
    r = s.open([data / 'slide.svs', data / 'cells.csv', data / 'nuclei.geojson'], trusted=True)
    assert not r['errors'], r['errors']
    yield s, r['opened'][0]
    s.shutdown()


def _classes(reg, sid, layer):
    from wissi_viz import annio

    return annio.from_layer(reg.layer(sid, layer)).class_of()


def test_objects_to_regions_and_cells_inside(session):
    import shapely

    from wissi_viz import annio, layerops

    s, sid = session
    reg = s.registry
    # the big nuclei, merged when closer than 150 µm: a few large regions
    name = layerops.to_regions(reg, sid, 'nuclei', classes=['Big'], join_um=150, min_area_um2=5000)
    assert name == 'Regions · nuclei' and reg.layer(sid, name).origin == 'found'
    R = annio.from_layer(reg.layer(sid, name))
    assert 1 <= len(R) < 50 and (R.kind == annio.GEO_POLYGON).all()
    # without joining, each nucleus is its own region (no merging)
    single = layerops.to_regions(reg, sid, 'nuclei', classes=['Big'], join_um=0)
    assert len(reg.layer(sid, single)) > 3 * len(R)
    # too strict a minimum: a clear error
    with pytest.raises(ValueError, match='no region'):
        layerops.to_regions(reg, sid, 'nuclei', classes=['Big'], join_um=10, min_area_um2=1e9)
    # the cells inside them, as a new layer that keeps classes and measurements
    inner = layerops.inside(reg, sid, 'cells', name)
    assert inner == 'cells in Regions · nuclei'
    g = layerops.geometry(reg, sid, name)
    c = annio.from_layer(reg.layer(sid, 'cells')).centroid()
    want = int(shapely.contains_xy(g, c[:, 0], c[:, 1]).sum())
    I = annio.from_layer(reg.layer(sid, inner))  # noqa: E741
    assert len(I) == want and 0 < want < len(c) and 'area_um2' in I.props
    assert set(I.class_of()) <= {'Big', 'Small'}


def test_analysis_inside_a_region(session):
    from wissi_viz import analyses, layerops

    s, sid = session
    reg = s.registry
    regions = layerops.to_regions(reg, sid, 'nuclei', classes=['Big'], join_um=150, min_area_um2=5000)
    whole = analyses.run(reg, sid, 'nn', 'cells', {'source': 'Big', 'target': 'Small', 'max_um': 200})
    part = analyses.run(reg, sid, 'nn', 'cells', {'source': 'Big', 'target': 'Small', 'max_um': 200},
                        within={'layer': regions})  # fmt: skip
    kp = {k['label']: k['value'] for k in part['kpis']}
    assert part['title'].endswith(f'· in {regions}') and 'cells in ROI' in kp and kp['ROI area'].endswith('mm²')
    n_in = int(kp['cells in ROI'].replace(',', ''))
    assert 0 < n_in < len(reg.layer(sid, 'cells'))
    assert all(n.endswith(f'· in {regions}') for n in part['layers']) and set(part['layers']).isdisjoint(
        whole['layers']
    )
    # Ripley's L uses the region as the study area
    rip = analyses.run(reg, sid, 'ripley', 'cells', {'a': 'Big', 'b': 'all'}, within={'layer': regions})
    assert rip['chart'] and 'in Regions' in rip['title']
    # a region class the layer does not have, or no cells inside: a clear error
    with pytest.raises(ValueError, match='no regions'):
        analyses.run(reg, sid, 'density', 'cells', {}, within={'layer': regions, 'classes': ['Nope']})


def test_one_analysis_on_many_images(data, tmp_path):  # noqa: F811
    from wissi_viz import analyses
    from wissi_viz.session import Session

    shutil.copy(data / 'slide.svs', tmp_path / 'second.svs')
    shutil.copy(data / 'slide.svs', tmp_path / 'bare.svs')
    s = Session(cache_dir=tmp_path / 'cache', remember=False)
    try:
        a = s.open([data / 'slide.svs', data / 'cells.csv'], trusted=True)['opened'][0]
        b = s.open([tmp_path / 'second.svs', data / 'cells.csv'], trusted=True)['opened'][0]
        c = s.open([tmp_path / 'bare.svs'], trusted=True)['opened'][0]
        out = analyses.run_batch(s.registry, [a, b, c], 'density', 'cells', {'cell_um': 100})
        t = out['table']
        assert out['batch'] and t['n'] == 2 and t['columns'][:2] == ['image', 'cell layer']
        assert [r[0] for r in t['rows']] == ['slide.svs', 'second.svs']
        assert 'bare.svs (no cell layer)' in out['note'] and out['kpis'][1]['value'] == '1'
        assert out['chart'] and len(out['chart']['series'][0]['values']) == 2
        # over HTTP, as a background job
        from starlette.testclient import TestClient

        cl = TestClient(s.app())
        j = cl.post(f'{P}/analyses/batch', json={'analysis': 'density', 'slides': [a, b], 'objects': 'cells',
                                                 'params': {'cell_um': 200}}).json()  # fmt: skip
        t0 = time.time()
        while j['state'] in ('queued', 'running') and time.time() - t0 < 60:
            time.sleep(0.2)
            j = cl.get(f'{P}/jobs/{j["id"]}').json()
        assert j['state'] == 'done' and j['result']['table']['n'] == 2, j
    finally:
        s.shutdown()


def test_chain_over_http(session):
    from starlette.testclient import TestClient

    s, sid = session
    c = TestClient(s.app())
    r = c.post(f'{P}/slides/{sid}/layers/regions', json={'layer': 'nuclei', 'classes': ['Big'], 'join_um': 150,
                                                          'min_area_um2': 5000})  # fmt: skip
    assert r.status_code == 200, r.text
    regions = r.json()['layer']
    r = c.post(f'{P}/slides/{sid}/layers/inside', json={'objects': 'cells', 'regions': regions})
    assert r.status_code == 200 and r.json()['layer'] == f'cells in {regions}'
    r = c.post(f'{P}/slides/{sid}/analyses', json={'analysis': 'density', 'objects': 'cells', 'params': {},
                                                    'within': {'layer': regions}})  # fmt: skip
    assert r.status_code == 200 and 'in Regions' in r.json()['title']
    bad = c.post(f'{P}/slides/{sid}/layers/inside', json={'objects': 'cells', 'regions': 'cells'})
    assert bad.status_code == 400 and 'no regions' in bad.json()['detail']
    assert np.isfinite(len(s.registry.layers(sid)))
