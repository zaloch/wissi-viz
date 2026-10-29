"""Find by text: Grounding DINO's tokenizer, resizing and post-processing; SAM 3's tokenizer, post-processing and
licence gate; model downloads. The real models run in .github/workflows/grounding-onnx.yml / sam3-onnx.yml
(``WISSI_TEST_GROUNDING=1`` / ``WISSI_TEST_SAM3=1``)."""

from __future__ import annotations

import gzip
import hashlib
import os

import numpy as np
import pytest

from wissi_viz import grounding as G
from wissi_viz import sam3 as S3
from wissi_viz import weights


def test_phrases():
    assert G.phrases('Nucleus, gland.  fat   cell\nnucleus') == ['nucleus', 'gland', 'fat cell']
    assert G.phrases(' . , ') == []
    assert G.phrases(['A', 'b ']) == ['a', 'b']


def test_wordpiece_and_caption():
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '.', 'nuc', '##le', '##us', 'fat', 'cell', 'cafe', '-']
    tok = G.WordPiece(vocab)
    assert tok.basic('Café NUCLEUS, fat-cell') == ['cafe', 'nucleus', ',', 'fat', '-', 'cell']
    assert tok.encode('Nucleus') == [5, 6, 7]
    assert tok.encode('xyz') == [1]  # unknown word
    ids, spans = G.encode_prompt(tok, ['nucleus', 'fat cell'])
    assert ids.tolist() == [[2, 5, 6, 7, 4, 8, 9, 4, 3]]  # [CLS] nucleus . fat cell . [SEP]
    assert spans == [(1, 4), (5, 7)]


@pytest.mark.parametrize(
    'hw, out',
    [
        ((512, 512), (800, 800)),
        ((480, 640), (800, 1066)),
        ((640, 480), (1066, 800)),
        ((300, 2000), (200, 1333)),
        ((800, 1000), (800, 1000)),
    ],  # fmt: skip
)
def test_resize_like_the_processor(hw, out):
    assert G.resize_to(*hw) == out


def test_pixels_normalised_and_padded():
    x, m = G.pixels(np.full((400, 600, 3), 255, np.uint8))
    assert x.shape == (1, 3, 800, 1200) and x.dtype == np.float32 and m.all()
    assert np.allclose(x[0, :, 0, 0], (1 - G.MEAN) / G.STD, atol=1e-5)
    x, m = G.pixels(np.full((400, 600, 3), 255, np.uint8), fixed=(800, 800))  # a graph that takes 800 × 800
    assert x.shape == (1, 3, 800, 800) and m[0, :533].all() and not m[0, 534:].any() and not x[0, :, 600:].any()


def test_grounding_postprocess():
    logits = np.full((1, 4, 256), -20.0, np.float32)
    logits[0, 0, 1:4] = [2.0, 1.0, 0.0]  # query 0: phrase 0 (tokens 1–3)
    logits[0, 1, 5] = 0.5  # query 1: phrase 1 (tokens 5–6)
    logits[0, 2, 1] = -3.0  # query 2: too weak
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.5, 0.5, 1, 1], [0, 0, 0, 0]]], np.float32)
    b, s, lab = G.postprocess(logits, boxes, [(1, 4), (5, 7)], (100, 200), 0.3)
    assert lab.tolist() == [0, 1] and np.allclose(s, 1 / (1 + np.exp([-2.0, -0.5])))
    assert np.allclose(b[0], [80, 30, 120, 70]) and np.allclose(b[1], [40, 20, 60, 30])


class _FakeSession:
    def __init__(self, inputs, outputs, fn):
        self._in, self._out, self.fn = inputs, outputs, fn

    def get_inputs(self):
        shape = lambda n: [1, 3, 800, 800] if n == 'pixel_values' else [8, 1, 4]
        return [type('I', (), {'name': n, 'shape': shape(n), 'type': 'tensor(uint8)'})() for n in self._in]

    def get_outputs(self):
        return [type('O', (), {'name': n})() for n in self._out]

    def get_providers(self):
        return ['CPUExecutionProvider']

    def run(self, _, feed):
        return self.fn(feed)


def test_grounding_model_feeds_by_name(tmp_path, monkeypatch):
    (tmp_path / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '.', 'cell']), encoding='utf-8')
    seen = {}

    def run(feed):
        seen.update(feed)
        lg = np.full((1, 3, 256), -20.0, np.float32)
        lg[0, 0, 1] = 3.0
        return [lg, np.array([[[0.5, 0.5, 0.5, 0.5]] * 3], np.float32)]

    names = ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']
    monkeypatch.setattr(weights, 'ort_session', lambda p, d=None: _FakeSession(names, ['logits', 'pred_boxes'], run))
    det = G.GroundingDino(tmp_path)
    b, s, lab = det.detect(np.zeros((100, 200, 3), np.uint8), ['cell'])
    assert seen['input_ids'].tolist() == [[2, 5, 4, 3]] and seen['pixel_values'].shape == (1, 3, 800, 800)
    assert seen['pixel_mask'][0, :400].all() and not seen['pixel_mask'][0, 400:].any()
    assert len(b) == 1 and np.allclose(b[0], [50, 25, 150, 75])  # relative to the picture, not the padding


# ------------------------------------------------------------------------------------------------ SAM 3
def _merges(tmp_path):
    p = tmp_path / 'bpe.txt.gz'
    with gzip.open(p, 'wt', encoding='utf-8') as f:
        f.write('#version: 0.2\n' + '\n'.join(['c e', 'l l</w>', 'ce ll</w>', 'n u']) + '\n')
    return p


def test_clip_bpe(tmp_path):
    tok = S3.ClipBPE(_merges(tmp_path))
    ids = tok.encode('Cell')
    assert [k for k, v in tok.enc.items() if v in ids] == ['cell</w>']
    t = tok.tokens('cell cell', context=6)
    assert t.shape == (1, 6) and t[0, 0] == tok.sot and t[0, 3] == tok.eot and t[0, 4] == 0
    long = tok.tokens('cell ' * 50, context=8)
    assert long[0, -1] == tok.eot  # cut to the context, ending with end-of-text


def test_sam3_postprocess():
    raw = {
        'pred_logits': np.array([[[3.0], [-1.0], [5.0]]], np.float32),
        'presence_logit_dec': np.array([[4.0]], np.float32),
        'pred_boxes': np.array([[[0.5, 0.5, 0.5, 0.5], [0, 0, 0.1, 0.1], [0.25, 0.25, 0.1, 0.2]]], np.float32),
        'pred_masks': np.stack([np.full((1, 8, 8), v, np.float32) for v in (5.0, -5.0, -5.0)], 1),
    }
    raw['pred_masks'][0, 2, :4, :4] = 5.0
    b, s, m = S3.postprocess(raw, (16, 32), 0.5)
    assert len(s) == 2 and s[0] > s[1]  # best first: query 2, then 0
    assert np.allclose(b[0], [6.4, 2.4, 9.6, 5.6]) and np.allclose(b[1], [8, 4, 24, 12])
    assert m.shape == (2, 16, 32) and m[1].all() and m[0][:6, :14].all() and not m[0][10:, 18:].any()


def test_sam3_needs_the_licence(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, 'user_dir', lambda kind: tmp_path / kind)
    monkeypatch.delenv('WISSI_SAM3_ACCEPT_LICENSE', raising=False)
    assert not S3.accepted()
    with pytest.raises(PermissionError, match='SAM License'):
        S3.detector()
    assert 'Grant of Rights' in S3.license_text() and 'military' in S3.license_text()
    st = S3.accept()
    assert st['accepted'] and S3.accepted() and st['license'] == 'SAM License'
    monkeypatch.setenv('WISSI_SAM3_DIR', str(tmp_path / 'none'))
    with pytest.raises(FileNotFoundError, match='image encoder'):
        S3.detector()
    assert not S3.accept(False)['accepted']


def test_sam3_decoder_feed(tmp_path, monkeypatch):
    """Text only: every geometry slot is padding; the image goes in as uint8 at 1008²."""
    shutil_copy = _merges(tmp_path).read_bytes()
    (tmp_path / 'bpe_simple_vocab_16e6.txt.gz').write_bytes(shutil_copy)
    for n in ('sam3_image_encoder.onnx', 'sam3_language_encoder.onnx', 'sam3_decoder.onnx'):
        (tmp_path / n).write_bytes(b'')
    feeds = {}

    def enc(feed):
        feeds['image'] = feed['image']
        return [np.zeros(1)] * 6

    def lang(feed):
        feeds['tokens'] = feed['tokens']
        return [np.zeros((1, 32), bool), np.zeros((32, 1, 256)), np.zeros((32, 1, 1024))]

    def dec(feed):
        feeds['dec'] = feed
        return [np.array([[[0.5, 0.5, 0.2, 0.2]]], np.float32), np.array([[[4.0]]], np.float32),
                np.full((1, 1, 4, 4), 3.0, np.float32), np.array([[3.0]], np.float32)]  # fmt: skip

    dec_in = ['original_height', 'original_width', 'vision_pos_enc_2', 'backbone_fpn_0', 'backbone_fpn_1',
              'backbone_fpn_2', 'language_mask', 'language_features', 'language_embeds', 'box_coords', 'box_labels',
              'box_masks']  # fmt: skip
    sessions = {
        'sam3_image_encoder.onnx': _FakeSession(['image'], ['a'] * 6, enc),
        'sam3_language_encoder.onnx': _FakeSession(['tokens'], ['m', 'f', 'e'], lang),
        'sam3_decoder.onnx': _FakeSession(
            dec_in, ['pred_boxes', 'pred_logits', 'pred_masks', 'presence_logit_dec'], dec
        ),
    }
    monkeypatch.setattr(weights, 'ort_session', lambda p, d=None: sessions[p.name])
    m = S3.Sam3(tmp_path)
    b, s, lab, masks = m.detect(np.zeros((50, 70, 3), np.uint8), ['cell', 'nu'])
    assert feeds['image'].shape == (3, 1008, 1008) and feeds['image'].dtype == np.uint8
    d = feeds['dec']
    assert d['box_masks'].all() and d['box_coords'].shape == (8, 1, 4) and int(d['original_width']) == 70
    assert 'vision_pos_enc_0' not in d  # only the inputs the graph has
    assert lab.tolist() == [0, 1] and masks.shape == (2, 50, 70) and masks.all() and np.allclose(b[0], [28, 20, 42, 30])


# ------------------------------------------------------------------------------------------------ downloads
def test_fetch_checks_sums(tmp_path, monkeypatch):
    files = {'m.onnx': b'model', 'm.onnx.data': b'weights', 'vocab.txt': b'a\nb\n'}
    sums = ''.join(f'{hashlib.sha256(v).hexdigest()}  {k}\n' for k, v in files.items())
    served = {'SHA256SUMS': sums.encode(), **files}

    class R:
        def __init__(self, b):
            self.b, self.headers = b, {'Content-Length': str(len(b))}

        def read(self, n=-1):
            out, self.b = (self.b, b'') if n < 0 else (self.b[:n], self.b[n:])
            return out

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(weights, 'user_dir', lambda kind: tmp_path / kind)
    monkeypatch.setattr(weights.urllib.request, 'urlopen', lambda url, timeout=0: R(served[url.rsplit('/', 1)[1]]))
    got = []
    d = weights.fetch('rel', {'m.onnx': 'model.onnx', 'vocab.txt': 'vocab.txt'}, 'x', lambda f, m: got.append(f),
                      also=lambda n: n.endswith('.data'))  # fmt: skip
    assert (d / 'model.onnx').read_bytes() == b'model' and (d / 'm.onnx.data').read_bytes() == b'weights'
    assert got and weights.have('x', {'m.onnx': 'model.onnx'}) == d
    served['vocab.txt'] = b'tampered'
    with pytest.raises(ValueError, match='checksum'):
        weights.fetch('rel', {'vocab.txt': 'vocab.txt'}, 'y')
    assert not (tmp_path / 'cache' / 'models' / 'y' / 'vocab.txt').exists()
    assert weights.have('y', {'vocab.txt': 'vocab.txt'}) is None


# ------------------------------------------------------------------------------------------------ real models
real_grounding = pytest.mark.skipif(not os.environ.get('WISSI_TEST_GROUNDING'), reason='WISSI_TEST_GROUNDING=1')


@real_grounding
def test_real_grounding_finds_the_cat():
    from skimage import data

    det = G.detector()
    b, s, lab = det.detect(data.chelsea(), ['cat', 'dog'], min_score=0.3)
    assert len(b) and lab[0] == 0 and s[0] > 0.4
    x0, y0, x1, y1 = b[0]
    assert (x1 - x0) > 150 and (y1 - y0) > 150  # the cat fills most of the picture


@real_grounding
def test_real_grounding_status():
    st = G.status()
    assert st['downloaded'] and st['license'] == 'Apache-2.0'


@pytest.mark.skipif(not os.environ.get('WISSI_TEST_SAM3'), reason='WISSI_TEST_SAM3=1 (needs the SAM 3 release)')
def test_real_sam3_finds_the_cat(monkeypatch):
    from skimage import data

    monkeypatch.setenv('WISSI_SAM3_ACCEPT_LICENSE', '1')
    det = S3.detector()
    b, s, lab, m = det.detect(data.chelsea(), ['cat'], min_score=0.5)
    assert len(s) >= 1 and m[0].mean() > 0.2  # the cat fills much of the picture
