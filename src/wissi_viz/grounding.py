"""Find objects by name: Grounding DINO (IDEA Research, Apache-2.0) turns a text prompt ("nucleus. mitotic figure.")
into boxes; :mod:`wissi_viz.findall` then outlines each box with SAM 2.1.

    det = detector()                                   # downloads the model once (~230 MB, int8)
    boxes, scores, labels = det.detect(rgb, ['nucleus', 'gland'])

Runs on ONNX Runtime (a core dependency): no PyTorch. The model is Grounding DINO tiny (Swin-T + BERT,
``IDEA-Research/grounding-dino-tiny``) exported to ONNX from ``transformers`` by this project's ``grounding-onnx``
workflow, which also quantizes the weights to int8 and checks the results against the PyTorch model (release
``grounding-dino-tiny-onnx-v1``, ``check.json``). The graph takes 800 × 800 pixels and the caption's text masks.
``WISSI_GROUNDING_MODEL=fp32`` uses the float32 file; ``WISSI_GROUNDING_DIR`` points at files you already have
(``model.onnx``, ``vocab.txt``).

The model was trained on natural images: on slides it knows coarse things (cells, nuclei, glands, vessels, fat,
tissue folds) better than pathology terms. Pick a few examples as well to keep only those like them.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from . import weights

RELEASE = 'grounding-dino-tiny-onnx-v1'
VARIANTS = {'int8': 'grounding-dino-tiny-int8.onnx', 'fp32': 'grounding-dino-tiny-fp32.onnx'}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_TEXT = 256  # the model's text length


# ================================================================================================ prompt → tokens
def phrases(prompt: str | list[str]) -> list[str]:
    """``"Nucleus, gland. fat cell"`` → ``['nucleus', 'gland', 'fat cell']`` (one class per phrase)."""
    parts = prompt if isinstance(prompt, list) else re.split(r'[.,;\n]+', prompt or '')
    out: list[str] = []
    for p in parts:
        p = ' '.join(str(p).lower().split())
        if p and p not in out:
            out.append(p)
    return out


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith('P')


class WordPiece:
    """BERT's uncased WordPiece tokenizer (what Grounding DINO's text encoder reads), from its ``vocab.txt``."""

    def __init__(self, vocab: Path | list[str]) -> None:
        words = vocab if isinstance(vocab, list) else Path(vocab).read_text(encoding='utf-8').splitlines()
        self.ids = {w: i for i, w in enumerate(words)}
        self.cls, self.sep, self.unk = self.ids['[CLS]'], self.ids['[SEP]'], self.ids['[UNK]']

    @staticmethod
    def basic(text: str) -> list[str]:
        """Lower-case, strip accents, split on white space and punctuation (BERT's basic tokenizer)."""
        text = ''.join(
            ' ' if c.isspace() else c for c in text if not unicodedata.category(c).startswith('C') or c in '\t\n\r'
        )
        text = unicodedata.normalize('NFD', text.lower())
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        out: list[str] = []
        for word in text.split():
            cur = ''
            for c in word:
                if _is_punct(c):
                    if cur:
                        out.append(cur)
                    out.append(c)
                    cur = ''
                else:
                    cur += c
            if cur:
                out.append(cur)
        return out

    def word(self, w: str) -> list[int]:
        if len(w) > 100:
            return [self.unk]
        out, start = [], 0
        while start < len(w):
            end, hit = len(w), None
            while start < end:
                piece = w[start:end] if start == 0 else '##' + w[start:end]
                if piece in self.ids:
                    hit = self.ids[piece]
                    break
                end -= 1
            if hit is None:
                return [self.unk]
            out.append(hit)
            start = end
        return out

    def encode(self, text: str) -> list[int]:
        return [i for w in self.basic(text) for i in self.word(w)]


def text_masks(tok: WordPiece, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grounding DINO's per-phrase text attention and position ids (``[CLS] a b . c . [SEP]``: each phrase attends
    only to itself (with its full stop), word positions restart at each phrase), as transformers'
    ``generate_masks_with_special_tokens_and_transfer_map``. Computed here, not in the graph: traced into ONNX it would be frozen to the export's own caption."""
    special = {tok.cls, tok.sep, tok.ids.get('.', -1), tok.ids.get('?', -1)}
    row = ids[0]
    n = len(row)
    att = np.eye(n, dtype=bool)
    pos = np.zeros(n, np.int64)
    prev = 0
    for col in [i for i, t in enumerate(row) if int(t) in special]:
        if col == 0 or col == n - 1:
            att[col, col] = True
            pos[col] = 0
        else:
            att[prev + 1 : col + 1, prev + 1 : col + 1] = True
            pos[prev + 1 : col] = np.arange(0, col - prev - 1)  # the phrase's words; its '.' stays at 0
        prev = col
    return att[None], pos[None]


def encode_prompt(tok: WordPiece, names: list[str]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """``[CLS] name₁ . name₂ . [SEP]`` (Grounding DINO's caption) and each name's token span."""
    ids, spans = [tok.cls], []
    dot = tok.encode('.')
    for n in names:
        t = tok.encode(n)
        if len(ids) + len(t) + len(dot) + 1 > MAX_TEXT:
            break
        spans.append((len(ids), len(ids) + len(t)))
        ids += t + dot
    ids.append(tok.sep)
    return np.asarray(ids, np.int64)[None], spans


# ================================================================================================ image
def resize_to(h: int, w: int, short: int = 800, long: int = 1333) -> tuple[int, int]:
    """The processor's output size: shortest side ``short`` unless the longest would pass ``long`` (as DETR)."""
    size, raw = short, None
    lo, hi = float(min(h, w)), float(max(h, w))
    if hi / lo * size > long:
        raw = long * lo / hi
        size = int(round(raw))
    if (h <= w and h == size) or (w <= h and w == size):
        return h, w
    if w < h:
        return (int((raw if raw is not None else size) * h / w), size)
    return (size, int((raw if raw is not None else size) * w / h))


def pixels(
    rgb: np.ndarray, short: int = 800, long: int = 1333, fixed: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(pixel_values, pixel_mask). The processor's resizing; or, for a graph with a fixed input size (the published
    export takes 800 × 800), the picture scaled to fit it and padded with zeros at the bottom / right, the mask marking
    the picture (the model's boxes are relative to the picture, not the padding)."""
    h, w = rgb.shape[:2]
    if fixed:
        H, W = fixed
        k = min(H / h, W / w)
        oh, ow = max(1, min(H, int(round(h * k)))), max(1, min(W, int(round(w * k))))
    else:
        oh, ow = resize_to(h, w, short, long)
        H, W = oh, ow
    im = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    if (oh, ow) != (h, w):
        im = im.resize((ow, oh), Image.BILINEAR)
    x = np.zeros((1, 3, H, W), np.float32)
    x[0, :, :oh, :ow] = ((np.asarray(im, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    mask = np.zeros((1, H, W), np.int64)
    mask[0, :oh, :ow] = 1
    return x, mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def postprocess(
    logits: np.ndarray,
    boxes: np.ndarray,
    spans: list[tuple[int, int]],
    size: tuple[int, int],
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Queries → (boxes xyxy in px of an image of ``size`` (h, w), scores, phrase index), best first. A query's
    score is its highest token probability (as ``post_process_grounded_object_detection``); its phrase is the one
    whose tokens score highest."""
    p = _sigmoid(np.asarray(logits, np.float32)[0])  # (queries, 256)
    s = p.max(-1)
    keep = np.nonzero(s > min_score)[0]
    keep = keep[np.argsort(-s[keep], kind='stable')]
    h, w = size
    b = np.asarray(boxes, np.float32)[0][keep]
    xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)  # fmt: skip
    if spans:
        per = np.stack([p[keep, a:b_].max(-1) if b_ > a else np.zeros(len(keep)) for a, b_ in spans], 1)
        lab = per.argmax(1)
    else:
        lab = np.zeros(len(keep), np.int64)
    return xyxy.astype(np.float32), s[keep].astype(np.float32), lab.astype(np.int64)


# ================================================================================================ model
class GroundingDino:
    name = 'grounding-dino'
    label = 'Grounding DINO tiny + SAM 2.1'

    def __init__(self, folder: Path, device: str | None = None) -> None:
        self.folder = Path(folder)
        self.sess = weights.ort_session(self.folder / 'model.onnx', device)
        self.inputs = {i.name for i in self.sess.get_inputs()}
        pv = next((i for i in self.sess.get_inputs() if i.name == 'pixel_values'), None)
        hw = list(pv.shape[2:]) if pv is not None else []
        self.fixed = tuple(hw) if len(hw) == 2 and all(isinstance(v, int) and v > 0 for v in hw) else None
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.tok = WordPiece(self.folder / 'vocab.txt')
        self.device = self.sess.get_providers()[0].replace('ExecutionProvider', '').lower()
        self._lock = threading.Lock()

    def detect(
        self, rgb: np.ndarray, names: list[str], min_score: float = 0.15, short: int = 800
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (xyxy, px of ``rgb``), scores and phrase indices of everything in ``rgb`` matching ``names``."""
        ids, spans = encode_prompt(self.tok, names)
        x, mask = pixels(rgb, short, fixed=self.fixed)
        feed = {
            'input_ids': ids,
            'attention_mask': np.ones_like(ids),
            'token_type_ids': np.zeros_like(ids),
            'pixel_values': x,
            'pixel_mask': mask,
        }
        if 'text_self_attention_masks' in self.inputs or 'position_ids' in self.inputs:
            feed['text_self_attention_masks'], feed['position_ids'] = text_masks(self.tok, ids)
        with self._lock:
            out = dict(zip(self.outputs, self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})))
        logits = out.get('logits', next(iter(out.values())))
        boxes = out.get('pred_boxes')
        if boxes is None:
            boxes = next(v for v in out.values() if v.ndim == 3 and v.shape[-1] == 4)
        return postprocess(logits, boxes, spans, rgb.shape[:2], min_score)


def files(variant: str | None = None) -> dict[str, str]:
    v = variant or os.environ.get('WISSI_GROUNDING_MODEL') or 'int8'
    if v not in VARIANTS:
        raise ValueError(f'WISSI_GROUNDING_MODEL: {v!r} (int8 | fp32)')
    return {VARIANTS[v]: 'model.onnx', 'vocab.txt': 'vocab.txt', 'LICENSE-Apache-2.0.txt': 'LICENSE-Apache-2.0.txt',
            'NOTICE.txt': 'NOTICE.txt'}  # fmt: skip


def folder(variant: str | None = None) -> Path | None:
    """The model's folder if it is here (``WISSI_GROUNDING_DIR`` or downloaded before), else None."""
    env = os.environ.get('WISSI_GROUNDING_DIR')
    if env:
        return Path(env)
    v = variant or os.environ.get('WISSI_GROUNDING_MODEL') or 'int8'
    return weights.have(f'grounding-dino-tiny-{v}', files(v))


_DET: dict[str, GroundingDino] = {}
_LOCK = threading.Lock()


def detector(progress=None, variant: str | None = None, device: str | None = None) -> GroundingDino:
    """The loaded model (downloaded on first use; ``progress(fraction, message)`` while downloading)."""
    v = variant or os.environ.get('WISSI_GROUNDING_MODEL') or 'int8'
    with _LOCK:
        if v not in _DET:
            d = folder(v) or weights.fetch(RELEASE, files(v), f'grounding-dino-tiny-{v}', progress)
            _DET[v] = GroundingDino(d, device or os.environ.get('WISSI_GROUNDING_DEVICE'))
        return _DET[v]


def status() -> dict:
    v = os.environ.get('WISSI_GROUNDING_MODEL') or 'int8'
    return {
        'model': f'grounding-dino-tiny ({v})',
        'downloaded': folder(v) is not None,
        'loaded': v in _DET,
        'size_mb': 230 if v == 'int8' else 720,
        'license': 'Apache-2.0',
    }
