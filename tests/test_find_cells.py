"""Find all → By text → a cell model: every nucleus of a view, a region or the slide as a detected cell layer, on any
slide source (files and tile APIs alike); SAM 3 reports whether its download is published."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')
pytest.importorskip('skimage')

sys.path.insert(0, str(Path(__file__).parent))
from test_findall import data, session  # noqa: F401  (the synthetic slide + nuclei fixture)

P = '/wissi-viz'


def test_classic_cells_in_a_view(session):  # noqa: F811
    from wissi_viz import annio, findall

    s, sid = session
    reg = s.registry
    rect = [300, 300, 1300, 1300]
    out = findall.run(reg, sid, {'mode': 'text', 'model': 'classic', 'scope': {'kind': 'view', 'rect': rect}})
    L = reg.layer(sid, out['layer'])
    assert out['mode'] == 'cells' and out['result'] is None and L.origin == 'detected'
    assert out['layer'] == 'Cells · classic stain threshold' and out['where'] == 'this view'
    F = annio.from_layer(L)
    c = F.centroid()
    assert len(F) == out['count'] and (c >= 300).all() and (c <= 1300).all()
    # the nuclei of the fixture that lie in the view, found once each
    N = annio.from_layer(reg.layer(sid, 'nuclei')).centroid()
    truth = N[(N >= 300).all(1) & (N <= 1300).all(1)]
    d = np.sqrt(((truth[:, None] - c[None]) ** 2).sum(-1)).min(1)
    assert (d < 4).mean() > 0.85, (len(truth), len(F))
    assert len(F) < 1.3 * len(truth)
    assert {'area_um2', 'circularity', 'hematoxylin_mean'} <= set(F.props)
    assert set(F.class_of()) <= {'Small nucleus', 'Medium nucleus', 'Large nucleus'}
    kp = dict(out['kpis'])
    assert kp['cells'] == f'{len(F):,}' and kp['median area'].endswith('µm²')
    # runs never overwrite each other
    again = findall.find_cells(reg, sid, model='classic', scope={'kind': 'view', 'rect': rect})
    assert again['layer'] == 'Cells · classic stain threshold 2'


def test_cells_from_a_tile_api_source(data):  # noqa: F811
    """A source without a file (as from a tile API): the tile-reading models work, Classpose says why it cannot."""
    from wissi_viz import findall
    from wissi_viz.session import Session
    from wissi_viz.sources import open_slide

    s = Session(remember=False)
    try:
        sid = s.registry.add(open_slide(data / 'slide.svs'), slide_id='remote')
        out = findall.find_cells(s.registry, sid, model='classic', scope={'kind': 'slide'})
        assert out['count'] > 200 and out['where'] == 'the whole slide'
        if findall.cell_models()['classpose']['available']:
            with pytest.raises(ValueError, match='tile API'):
                findall.find_cells(s.registry, sid, model='classpose')
    finally:
        s.shutdown()


def test_cell_model_errors_and_capabilities(session, monkeypatch):  # noqa: F811
    from wissi_viz import findall, segment

    s, sid = session
    reg = s.registry
    with pytest.raises(ValueError, match='unknown cell model'):
        findall.find_cells(reg, sid, model='nope')
    monkeypatch.setattr(segment, 'available', lambda: ['classic'])
    caps = findall.cell_models()
    assert (
        caps['classic']['available']
        and not caps['cellpose']['available']
        and 'wissi-viz[cells]' in caps['cellpose']['why']
    )
    with pytest.raises(ValueError, match='not installed'):
        findall.run(reg, sid, {'mode': 'text', 'model': 'cellpose', 'scope': {'kind': 'slide'}})
    with pytest.raises(ValueError, match='no cells'):  # glass only
        findall.find_cells(reg, sid, model='classic', scope={'kind': 'view', 'rect': [0, 0, 30, 30]})


def test_cells_over_http_and_sam3_published(session, monkeypatch):  # noqa: F811
    from starlette.testclient import TestClient

    from wissi_viz import sam3, weights

    s, sid = session
    c = TestClient(s.app())
    monkeypatch.setattr(weights, 'published', lambda release, timeout=4.0: False)
    caps = c.get(f'{P}/find/capabilities').json()
    assert caps['cells']['classic']['available'] and set(caps['cells']) == {'cellpose', 'classic', 'classpose'}
    if sam3.folder() is None:
        assert caps['text']['sam3']['published'] is False
    j = c.post(f'{P}/slides/{sid}/find', json={'mode': 'text', 'model': 'classic', 'prompt': '',
                                               'scope': {'kind': 'view', 'rect': [300, 300, 900, 900]}}).json()  # fmt: skip
    assert j['title'] == 'Find cells · classic stain threshold'
    t0 = time.time()
    while j['state'] in ('queued', 'running') and time.time() - t0 < 60:
        time.sleep(0.2)
        j = c.get(f'{P}/jobs/{j["id"]}').json()
    assert j['state'] == 'done' and j['result']['count'] >= 5, j


def test_published_check(monkeypatch):
    import urllib.error

    from wissi_viz import weights

    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        if 'missing' in url:
            raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)
        if 'offline' in url:
            raise urllib.error.URLError('no route')

        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=-1):
                return b''

        return R()

    monkeypatch.setattr(weights.urllib.request, 'urlopen', fake)
    monkeypatch.setattr(weights, '_PUBLISHED', {})
    assert weights.published('there-v1') is True and weights.published('there-v1') is True
    assert weights.published('missing-v1') is False and weights.published('offline-v1') is None
    assert len(calls) == 3  # a yes is remembered; a no for ten minutes
    weights.published('missing-v1')
    assert len(calls) == 3
