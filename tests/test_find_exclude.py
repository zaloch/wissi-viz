"""Find all: a size range, phrases to exclude, and regions to leave out."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')
pytest.importorskip('skimage')

sys.path.insert(0, str(Path(__file__).parent))
from test_findall import data, session  # noqa: F401  (the synthetic slide: nuclei of ~4.5 and ~10 µm, 0.5 µm/px)

RECT = [300, 300, 1300, 1300]


class _Phrases:
    """Stands in for SAM 3: 'nucleus' matches every dark blob (0.6), 'big' only the big ones (0.7)."""

    name, label, tile, masks = 'fake', 'fake', 800, True

    def detect(self, rgb, names, min_score=0.1):
        from scipy import ndimage as ndi

        lab, _ = ndi.label(rgb.mean(-1) < 120)
        B, S, L, M = [], [], [], []
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            size = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            for j, n in enumerate(names):
                if n == 'big' and size <= 28:  # model px: the slide is read 2× enlarged here
                    continue
                B.append([sl[1].start - 1, sl[0].start - 1, sl[1].stop + 1, sl[0].stop + 1])
                S.append(0.7 if n == 'big' else 0.6)
                L.append(j)
                M.append(lab == k)
        return (np.asarray(B, np.float32).reshape(-1, 4), np.asarray(S, np.float32), np.asarray(L, np.int64),
                np.asarray(M, bool).reshape(-1, *rgb.shape[:2]))  # fmt: skip


def _diameters_um(reg, sid, layer):
    from wissi_viz import annio

    F = annio.from_layer(reg.layer(sid, layer))
    d = []
    for i in range(len(F)):
        r = np.asarray(F.rings(i)[0], float)
        a = 0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1)))
        d.append(2 * np.sqrt(a / np.pi) * 0.5)
    return np.asarray(d), F


def test_text_size_range_and_excluded_phrases(session):  # noqa: F811
    from wissi_viz import findall

    s, sid = session
    reg = s.registry
    scope = {'kind': 'view', 'rect': RECT}
    every = findall.find_by_text(reg, sid, prompt='nucleus', scope=scope, object_um=4, object_um_max=12,
                                 strictness=0.0, detector=_Phrases())  # fmt: skip
    d, _ = _diameters_um(reg, sid, every['layer'])
    assert (d < 7).any() and (d > 7).any()
    small = findall.find_by_text(reg, sid, prompt='nucleus', scope=scope, object_um=3, object_um_max=6,
                                 strictness=0.0, detector=_Phrases())  # fmt: skip
    d, _ = _diameters_um(reg, sid, small['layer'])
    assert len(d) and (d < 7.5).all() and 'outside 3–6 µm left out' in small['note']
    # the same by excluding the big ones by name
    nobig = findall.find_by_text(reg, sid, prompt='nucleus', exclude='big', scope=scope, object_um=4, object_um_max=12,
                                 strictness=0.0, detector=_Phrases())  # fmt: skip
    d, _ = _diameters_um(reg, sid, nobig['layer'])
    assert len(d) and (d < 7.5).all() and 'like "big" left out' in nobig['note']
    assert abs(nobig['count'] - small['count']) <= 2
    with pytest.raises(ValueError, match='left out'):
        findall.find_by_text(reg, sid, prompt='big', exclude='', scope=scope, object_um=1, object_um_max=2,
                             detector=_Phrases())  # fmt: skip


def test_regions_left_out(session, monkeypatch):  # noqa: F811
    import shapely

    from wissi_viz import annio, findall
    from wissi_viz.annotations import AnnotationLayer
    from wissi_viz.sam import SamService

    monkeypatch.setattr(findall, 'sam_service', lambda: SamService(backend='classic'))

    s, sid = session
    reg = s.registry
    L = AnnotationLayer(name='Ignore')
    L.add_polygon(np.array([[300, 300], [800, 300], [800, 1300], [300, 1300]], float), cls='Area 1')
    reg.add_annotations(sid, L)
    scope = {'kind': 'view', 'rect': RECT, 'exclude': {'layer': 'Ignore'}}
    g, where = findall.scope_geometry(reg, sid, scope)
    assert where == 'this view, outside Ignore' and abs(g.area - 500 * 1000) < 1
    out = findall.find_cells(reg, sid, model='classic', scope=scope)
    c = annio.from_layer(reg.layer(sid, out['layer'])).centroid()
    assert len(c) and (c[:, 0] >= 800).all() and out['where'] == where
    seg = findall.segment_everything(reg, sid, scope=scope, object_um=8, object_um_max=12)
    d, F = _diameters_um(reg, sid, seg['layer'])
    assert len(d) and (F.centroid()[:, 0] >= 790).all() and (d > 6).all() and '8–12 µm' in seg['note']
    assert shapely.box(*RECT).difference(g).area > 0
    with pytest.raises(ValueError, match='no layer'):
        findall.scope_geometry(reg, sid, {'kind': 'slide', 'exclude': {'layer': 'Nope'}})
    with pytest.raises(ValueError, match='outside Ignore'):
        findall.scope_geometry(reg, sid, {'kind': 'view', 'rect': [400, 400, 700, 700], 'exclude': {'layer': 'Ignore'}})


def test_matches_and_tiles_beside_the_result(session):  # noqa: F811
    """Every match before filtering and the tiles searched come as hidden layers; 0 found says why."""
    from wissi_viz import annio, findall

    s, sid = session
    reg = s.registry
    out = findall.find_by_text(reg, sid, prompt='nucleus', scope={'kind': 'view', 'rect': RECT}, object_um=4,
                               object_um_max=6, strictness=0.0, detector=_Phrases())  # fmt: skip
    comp = out['companions']
    assert set(comp) == {'matches', 'tiles'} and comp['matches'] == f'{out["layer"]} · matches'
    M, T = reg.layer(sid, comp['matches']), reg.layer(sid, comp['tiles'])
    assert not M.visible and not T.visible and M.origin == 'found'
    assert len(M) > out['count'] and len(T) == int(dict(out['kpis'])['tiles']) and out['why'] == ''
    assert 'score' in annio.from_layer(M).props
    # examples nothing is like: 0 found, and the reason
    L = annio.from_layer(reg.layer(sid, 'cells'))
    big = [{'layer': 'cells', 'index': int(i)} for i in range(len(L))][:2]
    zero = findall.find_by_text(reg, sid, prompt='nucleus', scope={'kind': 'view', 'rect': RECT}, strictness=1.0,
                                examples={'picks': big}, detector=_Phrases())  # fmt: skip
    assert zero['count'] == 0 and 'like your examples' in zero['why']
