"""Find all like the examples (similar objects, segment everything, areas), background jobs, histograms, CZI scenes,
the slide list's projects and where layers come from."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip('scipy')
pytest.importorskip('shapely')
pytest.importorskip('skimage')

sys.path.insert(0, str(Path(__file__).parent))
from czi_writer import write_czi

MPP = 0.5


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    """A pink tissue with dark round nuclei of two sizes, the nuclei as a cell table (with outlines in GeoJSON)."""
    d = tmp_path_factory.mktemp('find')
    rng = np.random.default_rng(11)
    W = H = 1600
    img = np.full((H, W, 3), 242, np.uint8)  # glass around the tissue
    img[250:-250, 250:-250] = (215, 150, 195)
    img[250:-250, 250 : W // 2] = (195, 115, 180)  # left half: a second kind of tissue for areas
    yy, xx = np.mgrid[0:H, 0:W]
    feats, rows = [], ['x,y,class,area_um2']
    for _ in range(500):
        x, y = rng.uniform(40, W - 40, 2)
        big = rng.random() < 0.5
        r = rng.uniform(9, 11) if big else rng.uniform(4, 5)
        sl = (slice(int(y - r - 1), int(y + r + 2)), slice(int(x - r - 1), int(x + r + 2)))
        m = (xx[sl] - x) ** 2 + (yy[sl] - y) ** 2 < r * r
        if (img[sl][m] < 120).any():  # overlaps a nucleus already drawn
            continue
        img[sl][m] = (80, 45, 135)
        cls = 'Big' if big else 'Small'
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([x + r * np.cos(t), y + r * np.sin(t)], 1).tolist()
        feats.append({'type': 'Feature', 'properties': {'classification': {'name': cls}},
                      'geometry': {'type': 'Polygon', 'coordinates': [ring + [ring[0]]]}})  # fmt: skip
        rows.append(f'{x:.1f},{y:.1f},{cls},{np.pi * (r * MPP) ** 2:.2f}')
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = {MPP}'
    with tifffile.TiffWriter(d / 'slide.svs', bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
        tw.write(
            img[::4, ::4].copy(), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None, subfiletype=1
        )
    (d / 'cells.csv').write_text('\n'.join(rows), encoding='utf-8')
    (d / 'nuclei.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')
    return d


@pytest.fixture()
def session(data, tmp_path):
    from wissi_viz.session import Session

    s = Session(cache_dir=tmp_path / 'cache', remember=False)
    r = s.open([data / 'slide.svs', data / 'cells.csv', data / 'nuclei.geojson'], trusted=True)
    assert not r['errors'], r['errors']
    yield s, r['opened'][0]
    s.shutdown()


def _classes(reg, sid, layer):
    from wissi_viz import annio

    return annio.from_layer(reg.layer(sid, layer)).class_of()


# ------------------------------------------------------------------------------------------------ similar objects
@pytest.mark.parametrize('layer', ['cells', 'nuclei'])  # points with an area column, and outlines
def test_find_similar_objects_from_picks(session, layer):
    from wissi_viz import annio, findall

    s, sid = session
    reg = s.registry
    cls = _classes(reg, sid, layer)
    picks = [{'layer': layer, 'index': int(i)} for i in np.nonzero(cls == 'Big')[0][:3]]
    out = findall.find_objects(reg, sid, examples={'picks': picks}, candidates=layer, scope={'kind': 'slide'})
    G = annio.from_layer(reg.layer(sid, out['layer']))
    got = cls[np.asarray(G.props['source_index'], int)]
    precision, recall = (got == 'Big').mean(), (got == 'Big').sum() / (cls == 'Big').sum()
    assert precision > 0.9 and recall > 0.8, (precision, recall, out)
    assert reg.layer(sid, out['layer']).origin == 'found'
    # stricter: fewer, still the right ones; looser: more
    strict = findall.rethreshold(reg, out['result'], 0.95)
    loose = findall.rethreshold(reg, out['result'], 0.0)
    assert strict['count'] <= out['count'] <= loose['count']


def test_examples_drawn_and_negatives(session):
    """Outlines drawn on the slide as examples (no picks), and "not like this" picks."""
    from wissi_viz import annio, findall
    from wissi_viz.annotations import AnnotationLayer

    s, sid = session
    reg = s.registry
    F = annio.from_layer(reg.layer(sid, 'nuclei'))
    cls = F.class_of()
    A = AnnotationLayer(name='Annotations')
    for i in np.nonzero(cls == 'Small')[0][:3]:
        A.add_polygon(F.rings(int(i))[0], cls='Tiny')
    reg.add_annotations(sid, A)
    out = findall.find_objects(reg, sid, examples={'layer': 'Annotations', 'classes': ['Tiny']}, candidates='nuclei')
    G = annio.from_layer(reg.layer(sid, out['layer']))
    got = cls[np.asarray(G.props['source_index'], int)]
    assert (got == 'Small').mean() > 0.9 and out['layer'] == 'Found · Tiny'
    # a negative example of the same kind as the positives removes them
    neg = [{'layer': 'nuclei', 'index': int(i), 'neg': True} for i in np.nonzero(cls == 'Small')[0][3:6]]
    out2 = findall.find_objects(reg, sid, examples={'layer': 'Annotations', 'picks': neg}, candidates='nuclei')
    assert out2['count'] < out['count'] and out2['layer'] == 'Found · Tiny 2'  # never overwrites a result


def test_find_objects_needs_examples_and_candidates_in_scope(session):
    from wissi_viz import findall

    s, sid = session
    with pytest.raises(ValueError, match='no examples'):
        findall.find_objects(s.registry, sid, examples={}, candidates='cells')
    with pytest.raises(ValueError, match='no objects'):
        findall.find_objects(s.registry, sid, examples={'picks': [{'layer': 'cells', 'index': 0}]}, candidates='cells',
                             scope={'kind': 'view', 'rect': [0, 0, 3, 3]})  # fmt: skip


# ------------------------------------------------------------------------------------------------ segment everything
@pytest.mark.parametrize('backend, recall', [('classic', 0.9), ('onnx', 0.8)])  # onnx: the bundled SAM 2.1
def test_segment_everything(session, monkeypatch, backend, recall):
    from wissi_viz import annio, findall
    from wissi_viz.sam import SamService

    svc = SamService(backend=backend)
    monkeypatch.setattr(findall, 'sam_service', lambda: svc)
    s, sid = session
    reg = s.registry
    rect = [200, 200, 900, 900]
    out = findall.segment_everything(reg, sid, scope={'kind': 'view', 'rect': rect}, object_um=5)  # the smaller nuclei
    G = annio.from_layer(reg.layer(sid, out['layer']))
    c = annio.from_layer(reg.layer(sid, 'cells')).centroid()
    inside = c[
        (c[:, 0] > rect[0] + 15) & (c[:, 0] < rect[2] - 15) & (c[:, 1] > rect[1] + 15) & (c[:, 1] < rect[3] - 15)
    ]
    import shapely

    polys = shapely.polygons([shapely.linearrings(G.rings(i)[0]) for i in range(len(G))])
    hit = shapely.union_all(polys)
    found = shapely.contains_xy(hit, inside[:, 0], inside[:, 1]).mean()
    assert found > recall, (found, out)  # (almost) every nucleus outlined
    assert svc.status()['backend'] == backend and out['note'].startswith(backend)
    assert reg.layer(sid, out['layer']).origin == 'found' and out['layer'] == 'Segmented objects'
    # with examples: only the big ones
    cls = _classes(reg, sid, 'cells')
    picks = [{'layer': 'cells', 'index': int(i)} for i in np.nonzero(cls == 'Big')[0][:3]]
    out2 = findall.segment_everything(reg, sid, examples={'picks': picks}, scope={'kind': 'view', 'rect': rect})
    assert 0 < out2['count'] < out['count'] and out2['result']


# ------------------------------------------------------------------------------------------------ by text
class _FakeText:
    """Stands in for Grounding DINO / SAM 3: every dark blob is a match, big ones score 0.6, small ones 0.2."""

    name, label, tile = 'fake', 'fake', 800

    def __init__(self, masks=False):
        self.masks = masks
        self.calls = 0

    def detect(self, rgb, names, min_score=0.1):
        from scipy import ndimage as ndi

        self.calls += 1
        lab, n = ndi.label(rgb.mean(-1) < 120)
        boxes, scores, labels, masks = [], [], [], []
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            boxes.append([sl[1].start - 1, sl[0].start - 1, sl[1].stop + 1, sl[0].stop + 1])
            scores.append(0.6 if max(h, w) > 16 else 0.2)
            labels.append(len(labels) % len(names))
            masks.append(lab == k)
        out = [np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
               np.asarray(labels, np.int64)]  # fmt: skip
        if self.masks:
            out.append(np.asarray(masks, bool).reshape(-1, *rgb.shape[:2]))
        return tuple(out)


@pytest.mark.parametrize('masks', [False, True])  # boxes outlined by SAM 2.1 (Grounding DINO) / masks (SAM 3)
def test_find_by_text(session, monkeypatch, masks):
    from wissi_viz import annio, findall
    from wissi_viz.sam import SamService

    monkeypatch.setattr(findall, 'sam_service', lambda: SamService(backend='classic'))
    s, sid = session
    reg = s.registry
    det = _FakeText(masks)
    rect = [200, 200, 1400, 1400]
    out = findall.find_by_text(reg, sid, prompt='Nucleus, gland', scope={'kind': 'view', 'rect': rect},
                               strictness=0.0, object_um=20, detector=det)  # fmt: skip
    assert out['mode'] == 'text' and out['layer'] == 'Text · nucleus, gland' and det.calls > 1
    L = reg.layer(sid, out['layer'])
    assert L.origin == 'found' and set(L.classes) == {'nucleus', 'gland'}
    G = annio.from_layer(L)
    c = annio.from_layer(reg.layer(sid, 'cells')).centroid()

    def within(m):
        return (
            (c[:, 0] > rect[0] + m) & (c[:, 0] < rect[2] - m) & (c[:, 1] > rect[1] + m) & (c[:, 1] < rect[3] - m)
        ).sum()

    assert 0.9 * within(15) <= len(G) <= within(-2), (len(G), within(15), within(-2))
    g = G.centroid()
    d = np.sqrt(((g[:, None] - g[None]) ** 2).sum(-1)) + np.eye(len(g)) * 1e9
    assert d.min() > 3  # each once, although the tiles overlap
    assert set(G.class_of()) == {'nucleus', 'gland'} and 'score' in G.props
    # stricter: only the high scores (the big nuclei)
    r = findall.rethreshold(reg, out['result'], 0.8)
    assert 0 < r['count'] < out['count']
    assert set(annio.from_layer(reg.layer(sid, out['layer'])).class_of()) <= {'nucleus', 'gland'}
    # with examples: also like them
    cls = _classes(reg, sid, 'cells')
    picks = [{'layer': 'cells', 'index': int(i)} for i in np.nonzero(cls == 'Small')[0][:3]]
    out2 = findall.find_by_text(reg, sid, prompt='nucleus', examples={'picks': picks},
                                scope={'kind': 'view', 'rect': rect}, strictness=0.0, object_um=20, detector=det)  # fmt: skip
    assert out2['layer'] == 'Text · nucleus' and 0 < out2['count'] < out['count'] and 'examples' in out2['note']
    with pytest.raises(ValueError, match='type what to find'):
        findall.find_by_text(reg, sid, prompt=' ,. ', detector=det)


def test_find_by_text_over_http(session, monkeypatch, tmp_path):
    from wissi_viz import findall, sam3

    monkeypatch.setattr(findall, 'text_detector', lambda model='grounding', progress=None: _FakeText(True))
    monkeypatch.setattr(sam3, 'user_dir', lambda kind: tmp_path / kind)
    monkeypatch.delenv('WISSI_SAM3_ACCEPT_LICENSE', raising=False)
    s, sid = session
    c = _client(s)
    caps = c.get('/wissi-viz/find/capabilities').json()
    assert caps['text']['grounding']['license'] == 'Apache-2.0' and caps['text']['sam3']['accepted'] is False
    j = _wait(c, c.post(f'/wissi-viz/slides/{sid}/find', json={
        'mode': 'text', 'prompt': 'nucleus', 'object_um': 5, 'scope': {'kind': 'view', 'rect': [900, 900, 1300, 1300]},
    }).json()['id'])  # fmt: skip
    assert j['state'] == 'done' and j['result']['count'] > 0, j['error']
    # SAM 3: the licence first
    lic = c.get('/wissi-viz/models/sam3/license').json()
    assert lic['name'] == 'SAM License' and 'Grant of Rights' in lic['text'] and not lic['accepted']
    assert c.post('/wissi-viz/models/sam3/license', json={'accept': True}).json()['accepted'] is True
    assert sam3.accepted() and c.get('/wissi-viz/find/capabilities').json()['text']['sam3']['accepted']
    assert c.post('/wissi-viz/models/sam3/license', json={'accept': False}).json()['accepted'] is False


# ------------------------------------------------------------------------------------------------ areas
def test_areas_from_two_classes(session):
    pytest.importorskip('sklearn')
    from wissi_viz import findall
    from wissi_viz.annotations import AnnotationLayer

    s, sid = session
    reg = s.registry
    T = AnnotationLayer(name='Training')
    T.add_rect(320, 400, 380, 400, cls='Left')
    T.add_rect(900, 400, 380, 400, cls='Right')
    reg.add_annotations(sid, T)
    out = findall.find_areas(reg, sid, training='Training', detail='coarse')
    L = reg.layer(sid, out['layer'])
    import shapely

    from wissi_viz import annio

    F = annio.from_layer(L)
    area = {}
    for i, c in enumerate(F.class_of()):
        rs = F.rings(i)
        area[c] = area.get(c, 0) + shapely.Polygon(rs[0], rs[1:]).area
    total = sum(area.values())
    assert set(area) == {'Left', 'Right'} and 0.35 < area['Left'] / total < 0.65, area  # the two halves
    with pytest.raises(ValueError, match='two'):
        T1 = AnnotationLayer(name='One')
        T1.add_rect(320, 400, 380, 400, cls='Left')
        reg.add_annotations(sid, T1)
        findall.find_areas(reg, sid, training='One')


# ------------------------------------------------------------------------------------------------ HTTP: jobs, histogram, origins, projects
def _client(s):
    from starlette.testclient import TestClient

    return TestClient(s.app())


def _wait(c, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f'/wissi-viz/jobs/{jid}').json()
        if j['state'] not in ('queued', 'running'):
            return j
        time.sleep(0.1)
    raise AssertionError('job did not finish')


def test_find_job_threshold_and_cancel_over_http(session):
    s, sid = session
    c = _client(s)
    caps = c.get('/wissi-viz/find/capabilities').json()
    assert caps['sam'] and 'areas' in caps
    cls = _classes(s.registry, sid, 'cells')
    body = {'mode': 'objects', 'candidates': 'cells', 'scope': {'kind': 'slide'},
            'examples': {'picks': [{'layer': 'cells', 'index': int(i)} for i in np.nonzero(cls == 'Big')[0][:2]]}}  # fmt: skip
    j = c.post(f'/wissi-viz/slides/{sid}/find', json=body).json()
    j = _wait(c, j['id'])
    assert j['state'] == 'done' and j['result']['count'] > 0, j['error']
    r = c.post(
        f'/wissi-viz/slides/{sid}/find/threshold', json={'result': j['result']['result'], 'strictness': 1.0}
    ).json()
    assert r['count'] <= j['result']['count']
    assert c.post(f'/wissi-viz/slides/{sid}/find/threshold', json={'result': 'nope'}).status_code == 400
    # a user error comes back as the job's error, without a traceback
    bad = _wait(
        c, c.post(f'/wissi-viz/slides/{sid}/find', json={'mode': 'objects', 'candidates': 'cells'}).json()['id']
    )
    assert bad['state'] == 'error' and 'no examples' in bad['error']
    # cancel a long run
    long = c.post(
        f'/wissi-viz/slides/{sid}/find', json={'mode': 'segment', 'object_um': 3, 'scope': {'kind': 'slide'}}
    ).json()
    c.post(f'/wissi-viz/jobs/{long["id"]}/cancel')
    assert _wait(c, long['id'])['state'] in ('cancelled', 'done')
    assert c.get('/wissi-viz/jobs/nope').status_code == 404
    assert any(x['id'] == long['id'] for x in c.get(f'/wissi-viz/slides/{sid}/jobs').json())


def test_histogram(session):
    s, sid = session
    c = _client(s)
    h = c.get(f'/wissi-viz/slides/{sid}/histogram').json()
    assert [x['name'] for x in h['channels']] == ['red', 'green', 'blue'] and h['domain'] == 255
    assert sum(h['channels'][0]['counts']) + h['channels'][0]['zeros'] == h['pixels']
    v = c.get(f'/wissi-viz/slides/{sid}/histogram?rect=0,0,400,400').json()
    assert v['pixels'] <= 400 * 400 and v['level'] == 0
    assert c.get(f'/wissi-viz/slides/{sid}/histogram?rect=5,5,5,5').status_code == 400


def test_czi_scenes_and_fluorescence_histogram(tmp_path):
    from starlette.testclient import TestClient

    from wissi_viz import SlideRegistry
    from wissi_viz.server import create_app
    from wissi_viz.sources import open_slide

    rng = np.random.default_rng(0)
    blocks = []
    for ch in range(2):
        plane = np.clip(rng.normal(300, 40, (200, 500)), 0, 4095).astype(np.uint16)
        plane[:, 200:300] = 0
        for sc, (x0, x1) in enumerate(((0, 200), (300, 500))):
            blocks.append(
                {'c': ch, 'x': x0, 'y': 0, 'w': x1 - x0, 'h': 200, 'data': plane[:, x0:x1], 'm': len(blocks), 's': sc}
            )
    write_czi(tmp_path / 'two.czi', blocks, ['DAPI', 'CD3'], mpp=0.5)
    src = open_slide(tmp_path / 'two.czi')
    assert src.info.extra['scenes'] == [
        {'index': 0, 'name': 'Scene 1', 'rect': [0, 0, 200, 200]},
        {'index': 1, 'name': 'Scene 2', 'rect': [300, 0, 500, 200]},
    ]
    assert open_slide(tmp_path / 'two.czi', scene=1).info.width == 200
    reg = SlideRegistry(warm=False)
    sid = reg.add(tmp_path / 'two.czi')
    c = TestClient(create_app(reg))
    assert c.get(f'/wissi-viz/slides/{sid}/info.json').json()['extra']['scenes'][1]['rect'] == [300, 0, 500, 200]
    h = c.get(f'/wissi-viz/slides/{sid}/histogram').json()
    assert [x['name'] for x in h['channels']] == ['DAPI', 'CD3'] and h['domain'] == 65535
    d = h['channels'][0]
    assert d['zeros'] >= 100 * 200 and 250 < d['p']['50'] < 350  # the unscanned gap does not skew the percentiles


def test_layer_origins(session, data, tmp_path):
    from wissi_viz import annio
    from wissi_viz.session import _origin_from_name

    s, sid = session
    c = _client(s)
    assert s.registry.layer(sid, 'cells').origin == 'imported'
    f = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [9, 0], [9, 9], [0, 0]]]}}
    c.post(f'/wissi-viz/slides/{sid}/annotations/add?layer=Mine', json={**f, 'properties': {'source': 'sam'}})
    assert s.registry.layer(sid, 'Mine').origin == 'sam'
    c.post(f'/wissi-viz/slides/{sid}/annotations/add?layer=Mine', json=f)  # drawn
    assert s.registry.layer(sid, 'Mine').origin == 'drawn+sam'
    metas = {L['name']: L['style']['origin'] for L in c.get(f'/wissi-viz/slides/{sid}/annotations').json()}
    assert metas['Mine'] == 'drawn+sam'
    # saved with the layer (GeoParquet) and read back
    pytest.importorskip('pyarrow')
    p = annio.export(s.registry.layer(sid, 'Mine'), tmp_path / 'mine.parquet')
    assert annio.load(p).origin == 'drawn+sam'
    # sample-pack files written before origins existed
    assert _origin_from_name('/x/cmu-1.cells.parquet') == 'detected'
    assert _origin_from_name('/x/cmu-1.hotspots.parquet') == 'analysis'
    assert _origin_from_name('/x/cmu-1.expert.parquet') == 'imported'


def test_projects(tmp_path, monkeypatch, data):
    from wissi_viz import session as S

    monkeypatch.setattr(S, 'user_dir', lambda kind: tmp_path / kind)
    (tmp_path / 'config').mkdir()
    s = S.Session(cache_dir=tmp_path / 'cache')
    c = TestClient_(s)
    tok = {'X-Wissi-Token': s.token}
    r = c.post('/app/projects', json={'projects': [{'name': 'A', 'slides': ['/a.svs', '/a.svs']}, {'name': 'A'}, {'name': ''}, 3]},
               headers=tok).json()  # fmt: skip
    assert r['projects'] == [{'name': 'A', 'slides': ['/a.svs']}]
    assert c.post('/app/projects', json={'projects': []}).status_code == 403  # token
    assert S.Session(cache_dir=tmp_path / 'cache').projects == [{'name': 'A', 'slides': ['/a.svs']}]  # remembered
    assert c.get('/app/state', headers=tok).json()['projects'] == r['projects']
    # a session file's projects (by session id) become paths
    doc = {
        'slides': [{'id': 'one', 'slide': str(data / 'slide.svs'), 'layers': []}],
        'projects': [{'name': 'Pack', 'slides': ['one', 'missing']}],
    }
    (tmp_path / 's.json').write_text(json.dumps(doc), encoding='utf-8')
    s.load_session(tmp_path / 's.json', log=lambda *a: None)
    assert {'name': 'Pack', 'slides': [str(data / 'slide.svs')]} in s.projects
    s.shutdown()


def TestClient_(s):
    from starlette.testclient import TestClient

    return TestClient(s.app())


def test_browse_places_and_sizes(session, data):
    s, _ = session
    r = s.browse(str(data))
    assert any(e['name'] == 'slide.svs' and e['size'] > 0 for e in r['entries'])
    assert r['places'][0]['name'] == 'Home'
