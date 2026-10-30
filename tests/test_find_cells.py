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
    assert caps['text']['sam3']['source'] == f'https://huggingface.co/{sam3.HF_REPO}'
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


def test_sam3_reads_exports_with_the_processor_inside():
    """Older SAM 3 exports return boxes, scores and masks already post-processed: read them as they are."""
    from wissi_viz import sam3

    m = np.zeros((3, 1, 40, 50), bool)
    m[0, 0, 5:10, 5:10] = m[2, 0, 20:30, 20:40] = True
    raw = {'boxes': np.array([[5, 5, 10, 10], [0, 0, 1, 1], [20, 20, 40, 30]], np.float32),
           'scores': np.array([0.7, 0.3, 0.9], np.float32), 'masks': m}  # fmt: skip
    b, s, k = sam3.postprocess(raw, (40, 50), 0.5)
    assert s.tolist() == pytest.approx([0.9, 0.7]) and b[0].tolist() == [20, 20, 40, 30] and k.shape == (2, 40, 50)
    assert k[0, 25, 30] and not k[0, 7, 7] and k[1, 7, 7]
    small = {**raw, 'masks': m[:, :, ::2, ::2].astype(np.float32)}  # another size, as probabilities
    assert sam3.postprocess(small, (40, 50), 0.5)[2].shape == (2, 40, 50)


def test_sam3_download_from_the_public_copy(tmp_path, monkeypatch):
    """SAM 3 comes from the public export (here a stand-in zip): checked, unpacked flat, licence and notice beside it;
    a file with another fingerprint is refused."""
    import hashlib
    import zipfile

    from wissi_viz import sam3, weights

    z = tmp_path / 'src' / 'sam3_vit_h.zip'
    z.parent.mkdir()
    with zipfile.ZipFile(z, 'w') as zf:
        for n in ('sam3_image_encoder.onnx', 'sam3_image_encoder.onnx.data', 'sam3_language_encoder.onnx',
                  'sam3_decoder.onnx', 'config.yaml'):  # fmt: skip
            zf.writestr(f'sam3_vit_h/{n}', b'x' * 10)
    monkeypatch.setattr(weights, 'cache_dir', lambda name: tmp_path / 'cache' / name)
    monkeypatch.setattr(sam3, 'HF_URL', z.as_uri())
    monkeypatch.delenv('WISSI_SAM3_DIR', raising=False)

    def no_onnx(*a, **k):
        raise ImportError('onnx')

    monkeypatch.setattr(sam3, 'quantize', no_onnx)
    monkeypatch.setattr(sam3, 'HF_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='checksum'):
        sam3.download()
    assert sam3.folder() is None
    monkeypatch.setattr(sam3, 'HF_SHA256', hashlib.sha256(z.read_bytes()).hexdigest())
    msgs = []
    d = sam3.download(lambda f, m: msgs.append(m))
    assert sam3.folder() == d and (d / 'sam3_decoder.onnx').exists() and (d / 'sam3_image_encoder.onnx.data').exists()
    assert not (d / 'sam3_vit_h.zip').exists() and 'Grant of Rights' in (d / 'LICENSE-SAM.txt').read_text('utf-8')
    assert 'huggingface.co/nrl-ai' in (d / 'NOTICE.txt').read_text('utf-8') and 'float32' in sam3.status()['precision']
    assert any('Hugging Face' in m for m in msgs)


def test_sam3_local_int8_copy(tmp_path):
    """The int8 copy made on this computer: each part quantized (or kept) and still runs, close to float32."""
    onnx = pytest.importorskip('onnx')
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from wissi_viz import sam3

    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 64)).astype(np.float32)
    src = tmp_path / 'fp32'
    src.mkdir()
    for local in sam3.PARTS:
        g = helper.make_graph(
            [helper.make_node('MatMul', ['x', 'w'], ['y'])], local,
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 64])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 64])], [numpy_helper.from_array(w, 'w')],
        )  # fmt: skip
        m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)  # what ONNX Runtime reads
        onnx.save(m, str(src / f'sam3_{local}.onnx'))
    out = tmp_path / 'int8'
    prec = sam3.quantize(src, out, parts=('image_encoder', 'language_encoder'))
    assert prec == {'image_encoder': 'int8', 'language_encoder': 'int8', 'decoder': 'float32'}
    x = rng.normal(size=(1, 64)).astype(np.float32)
    for name in sam3.PARTS.values():
        y = ort.InferenceSession(str(out / f'sam3-{name}.onnx')).run(None, {'x': x})[0]
        assert np.abs(y - x @ w).max() < 0.05 * np.abs(x @ w).max()
    assert (out / 'sam3-image-encoder.onnx').stat().st_size < (src / 'sam3_image_encoder.onnx').stat().st_size


def test_sam3_fixed_score_cut_is_lowered(tmp_path):
    """A decoder with Sam3Processor's fixed 0.5 cut (scores > 0.5 → NonZero → the outputs) returns lower scores once
    the cut is lowered; the masks' own 0.5 (no NonZero after it) is left alone."""
    onnx = pytest.importorskip('onnx')
    import onnxruntime as ort
    from onnx import TensorProto, helper

    from wissi_viz import sam3

    c = lambda name, v: helper.make_node(
        'Constant', [], [name], value=helper.make_tensor(name, TensorProto.FLOAT, [], [v])
    )
    nodes = [
        c('half', 0.5), c('mhalf', 0.5),
        helper.make_node('Greater', ['s', 'half'], ['keep']),
        helper.make_node('NonZero', ['keep'], ['nz']),
        helper.make_node('Transpose', ['nz'], ['idx'], perm=[1, 0]),
        helper.make_node('GatherND', ['s', 'idx'], ['scores']),
        helper.make_node('Greater', ['m', 'mhalf'], ['masks']),
    ]  # fmt: skip
    g = helper.make_graph(nodes, 'dec', [helper.make_tensor_value_info('s', TensorProto.FLOAT, [4]),
                                         helper.make_tensor_value_info('m', TensorProto.FLOAT, [4])],
                          [helper.make_tensor_value_info('scores', TensorProto.FLOAT, None),
                           helper.make_tensor_value_info('masks', TensorProto.BOOL, None)])  # fmt: skip
    p = tmp_path / 'sam3_decoder.onnx'
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)], ir_version=9), str(p))
    feed = {'s': np.array([0.04, 0.2, 0.45, 0.8], np.float32), 'm': np.array([0.2, 0.6, 0.4, 0.9], np.float32)}
    before = ort.InferenceSession(str(p)).run(None, feed)
    assert before[0].tolist() == pytest.approx([0.8])
    lowered = sam3.lower_score_cut(p, 0.1)
    after = ort.InferenceSession(lowered).run(None, feed)
    assert after[0].tolist() == pytest.approx([0.2, 0.45, 0.8]) and after[1].tolist() == before[1].tolist()
    raw = tmp_path / 'raw.onnx'  # a decoder with raw outputs: nothing to lower
    g2 = helper.make_graph([helper.make_node('Identity', ['s'], ['pred_logits'])], 'raw',
                           [helper.make_tensor_value_info('s', TensorProto.FLOAT, [4])],
                           [helper.make_tensor_value_info('pred_logits', TensorProto.FLOAT, [4])])  # fmt: skip
    onnx.save(helper.make_model(g2, opset_imports=[helper.make_opsetid('', 17)], ir_version=9), str(raw))
    assert sam3.lower_score_cut(raw) is None


def test_dino_to_sam3_passes_the_best_boxes_as_examples():
    from wissi_viz import findall

    assert findall.concept('nucleus') == 'nucleus' and findall.concept('blood vessel') == 'blood vessel'
    assert findall.concept('DAB stained and Unstained Neurons') == 'Neurons'
    assert findall.concept('cells with visible and not visible nucleus') == 'cells'
    assert findall.concept('a furry animal lying on a couch') == 'animal'
    assert findall.concept('large round nuclei in the tumour region') == 'nuclei'

    class Dino:
        def detect(self, rgb, names, min_score=0.1):
            b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [80, 80, 90, 90], [5, 5, 9, 9]], float)
            return b, np.array([0.9, 0.8, 0.7, 0.6, 0.5]), np.array([0, 0, 0, 0, 1])

    class Sam3:
        tile = 1008

        def detect(self, rgb, names, min_score=0.1, exemplars=None):
            self.seen = (names, exemplars)
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros((0, 4, 4), bool)

    s3 = Sam3()
    d = findall.DinoSam3(Dino(), s3, examples=3)
    d.detect(np.zeros((4, 4, 3), np.uint8), ['DAB stained and unstained neurons', 'glia'])
    names, ex = s3.seen
    assert names == ['neurons', 'glia'] and d.tile == 1008 and d.masks
    assert [list(x) for x in ex[0]] == [[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90]]  # overlapping 2nd skipped
    assert [list(x) for x in ex[1]] == [[5, 5, 9, 9]]
    d.concepts = {'DAB stained and unstained neurons': 'neuron'}
    d.detect(np.zeros((4, 4, 3), np.uint8), ['DAB stained and unstained neurons'])
    assert s3.seen[0] == ['neuron']


def test_sam3_old_int8_copies_are_remade_on_x86(tmp_path, monkeypatch):
    from wissi_viz import sam3, weights

    monkeypatch.setattr(weights, 'cache_dir', lambda name: tmp_path / name)
    monkeypatch.delenv('WISSI_SAM3_DIR', raising=False)
    d = tmp_path / 'sam3'
    d.mkdir()
    (d / '.complete').write_text('int8 (quantized on this computer)', encoding='utf-8')
    for machine, kept in (('x86_64', False), ('AMD64', False), ('arm64', True), ('aarch64', True)):
        monkeypatch.setattr(sam3.platform, 'machine', lambda m=machine: m)
        assert (sam3.folder() == d) is kept, machine
    monkeypatch.setattr(sam3.platform, 'machine', lambda: 'x86_64')
    for note in (f'{sam3.QUANT_TAG} (quantized on this computer)', 'float32 (install the onnx package…)'):
        (d / '.complete').write_text(note, encoding='utf-8')
        assert sam3.folder() == d
