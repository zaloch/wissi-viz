"""SAM 3 (Meta): name a concept ("nucleus", "blood vessel") and get every instance in a picture, outlined. Opt-in.

SAM 3 is not under a permissive licence: Meta releases it under the **SAM License** (free to use, change and share,
with conditions: a copy of the licence with anything you share, acknowledgement in publications, no military,
weapons, nuclear or espionage use, trade-control compliance). So wissi-viz never ships it: the app shows the licence
(``models/SAM-License.txt``) and downloads the model only after you accept it (:func:`accept`, stored in the user
config folder). Without accepting, "find by text" uses Grounding DINO + SAM 2.1 (Apache-2.0).

The model files are ONNX exports of Meta's official checkpoint (``facebook/sam3``), made by this project's
``sam3-onnx`` workflow with samexporter (MIT) and quantized to int8 (release ``sam3-onnx-v1``, about 1.3 GB, with the
licence and a check against the PyTorch model). ``WISSI_SAM3_DIR`` points at your own export (the three
``sam3_*.onnx`` files from ``samexporter.export_sam3`` plus ``bpe_simple_vocab_16e6.txt.gz``).

Interface (samexporter's): image encoder ``image`` uint8 (3, 1008, 1008) RGB → vision_pos_enc_0‥2, backbone_fpn_0‥2;
language encoder ``tokens`` int64 (1, 32) CLIP BPE → text_attention_mask, text_memory, text_embeds; decoder (those +
original_height/width + box prompts) → pred_boxes (cx, cy, w, h; 0–1), pred_logits, pred_masks, presence_logit_dec.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import html
import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import weights
from ._compat import user_dir

RELEASE = 'sam3-onnx-v1'
SIZE = 1008
CONTEXT = 32
LICENSE_FILE = Path(__file__).resolve().parent / 'models' / 'SAM-License.txt'
LICENSE_NAME = 'SAM License'
LICENSE_URL = 'https://github.com/facebookresearch/sam3/blob/main/LICENSE'
FILES = {  # release assets, kept under their names (the graphs refer to their external-data files by name)
    'sam3-image-encoder-int8.onnx': 'sam3-image-encoder-int8.onnx',
    'sam3-language-encoder-int8.onnx': 'sam3-language-encoder-int8.onnx',
    'sam3-decoder-int8.onnx': 'sam3-decoder-int8.onnx',
    'bpe_simple_vocab_16e6.txt.gz': 'bpe_simple_vocab_16e6.txt.gz',
    'LICENSE-SAM.txt': 'LICENSE-SAM.txt',
    'NOTICE.txt': 'NOTICE.txt',
}
GRAPHS = {  # part → file names it may have: this release's, the export script's links, samexporter's own output
    'image_encoder': ('sam3-image-encoder-int8.onnx', 'image_encoder.onnx', 'sam3_image_encoder.onnx'),
    'language_encoder': ('sam3-language-encoder-int8.onnx', 'language_encoder.onnx', 'sam3_language_encoder.onnx'),
    'decoder': ('sam3-decoder-int8.onnx', 'decoder.onnx', 'sam3_decoder.onnx'),
}


# ================================================================================================ licence
def _consent_file() -> Path:
    return user_dir('config') / 'licenses.json'


def accepted() -> bool:
    """Whether the SAM License was accepted on this computer (or ``WISSI_SAM3_ACCEPT_LICENSE=1``)."""
    if os.environ.get('WISSI_SAM3_ACCEPT_LICENSE', '').strip().lower() in ('1', 'true', 'yes'):
        return True
    try:
        return bool(json.loads(_consent_file().read_text(encoding='utf-8')).get('sam3', {}).get('accepted'))
    except (OSError, ValueError):
        return False


def accept(yes: bool = True) -> dict:
    """Record (or withdraw) acceptance of the SAM License for SAM 3."""
    p = _consent_file()
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        d = {}
    if yes:
        d['sam3'] = {'accepted': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
                     'license': LICENSE_NAME, 'updated': 'November 19, 2025'}  # fmt: skip
    else:
        d.pop('sam3', None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding='utf-8')
    return status()


def license_text() -> str:
    return LICENSE_FILE.read_text(encoding='utf-8')


# ================================================================================================ CLIP tokenizer
@lru_cache
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ClipBPE:
    """CLIP's byte-level BPE tokenizer (what SAM 3's text encoder reads), from ``bpe_simple_vocab_16e6.txt.gz``."""

    # CLIP splits letters runs, single digits and other non-space runs; [^\W\d_] is "a letter" for the re module
    PAT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+", re.I)

    def __init__(self, merges: Path | list[str]) -> None:
        if isinstance(merges, list):
            lines = merges
        else:
            with gzip.open(merges, 'rt', encoding='utf-8') as f:
                lines = f.read().splitlines()[1 : 49152 - 256 - 2 + 1]
        pairs = [tuple(m.split()) for m in lines if len(m.split()) == 2]
        b2u = _bytes_to_unicode()
        vocab = list(b2u.values())
        vocab += [v + '</w>' for v in vocab]
        vocab += [''.join(p) for p in pairs]
        vocab += ['<|startoftext|>', '<|endoftext|>']
        self.enc = {v: i for i, v in enumerate(vocab)}
        self.ranks = {p: i for i, p in enumerate(pairs)}
        self.b2u = b2u
        self.sot, self.eot = self.enc['<|startoftext|>'], self.enc['<|endoftext|>']
        self._cache: dict[str, str] = {}

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        s = ' '.join(word)
        self._cache[token] = s
        return s

    def encode(self, text: str) -> list[int]:
        text = html.unescape(html.unescape(text))
        text = re.sub(r'\s+', ' ', text).strip().lower()
        ids = []
        for tok in self.PAT.findall(text):
            t = ''.join(self.b2u[b] for b in tok.encode('utf-8'))
            ids += [self.enc[p] for p in self._bpe(t).split(' ')]
        return ids

    def tokens(self, text: str, context: int = CONTEXT) -> np.ndarray:
        ids = [self.sot, *self.encode(text)][: context - 1] + [self.eot]
        out = np.zeros((1, context), np.int64)
        out[0, : len(ids)] = ids
        return out


# ================================================================================================ outputs
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def postprocess(raw: dict, size: tuple[int, int], min_score: float, max_n: int = 300):
    """Decoder heads → (boxes xyxy px, scores, masks (n, h, w) bool) above ``min_score``, best first. The score is
    the query's probability times the presence probability (as Meta's ``Sam3Processor``); masks are resized only for
    the queries kept."""
    logits = np.asarray(raw['pred_logits'], np.float32).reshape(-1)
    presence = float(_sigmoid(np.asarray(raw['presence_logit_dec'], np.float32).reshape(-1)[0]))
    scores = _sigmoid(logits) * presence
    keep = np.nonzero(scores > min_score)[0]
    keep = keep[np.argsort(-scores[keep], kind='stable')][:max_n]
    h, w = size
    b = np.asarray(raw['pred_boxes'], np.float32).reshape(-1, 4)[keep]
    boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                      (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)  # fmt: skip
    pm = np.asarray(raw['pred_masks'], np.float32)
    pm = pm.reshape(-1, *pm.shape[-2:])[keep]
    masks = np.zeros((len(keep), h, w), bool)
    for i, m in enumerate(pm):
        masks[i] = np.asarray(Image.fromarray(m).resize((w, h), Image.BILINEAR)) > 0  # sigmoid > 0.5
    return boxes.astype(np.float32), scores[keep].astype(np.float32), masks


# ================================================================================================ model
class Sam3:
    name = 'sam3'
    label = 'SAM 3'
    tile = SIZE
    masks = True  # outlines come with the detections (no SAM 2.1 pass)

    def __init__(self, folder: Path, device: str | None = None) -> None:
        self.folder = Path(folder)

        def path(part):
            for n in GRAPHS[part]:
                if (self.folder / n).exists():
                    return self.folder / n
            raise FileNotFoundError(f'no SAM 3 {part.replace("_", " ")} in {self.folder} ({" or ".join(GRAPHS[part])})')

        self.enc = weights.ort_session(path('image_encoder'), device)
        self.lang = weights.ort_session(path('language_encoder'), device)
        self.dec = weights.ort_session(path('decoder'), device)
        self.dec_in = {i.name: i for i in self.dec.get_inputs()}
        self.dec_out = [o.name for o in self.dec.get_outputs()]
        bc = self.dec_in.get('box_coords')
        cap = bc.shape[0] if bc is not None else None
        self.capacity = cap if isinstance(cap, int) and cap > 0 else 1
        self.enc_in = self.enc.get_inputs()[0]
        self.tok = ClipBPE(self.folder / 'bpe_simple_vocab_16e6.txt.gz')
        self.device = self.enc.get_providers()[0].replace('ExecutionProvider', '').lower()
        self._text: dict[str, list[np.ndarray]] = {}
        self._lock = threading.Lock()

    def text(self, phrase: str) -> list[np.ndarray]:
        if phrase not in self._text:
            self._text[phrase] = self.lang.run(None, {'tokens': self.tok.tokens(phrase)})
        return self._text[phrase]

    def image(self, rgb: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., :3])).resize((SIZE, SIZE), Image.BILINEAR))
        x = x.transpose(2, 0, 1)
        if self.enc_in.type == 'tensor(float)':
            x = ((x / 255.0 - 0.5) / 0.5).astype(np.float32)
        else:
            x = x.astype(np.uint8)
        return self.enc.run(None, {self.enc_in.name: np.ascontiguousarray(x)})

    def decode(self, img: list[np.ndarray], txt: list[np.ndarray], size: tuple[int, int]) -> dict:
        h, w = size
        n = self.capacity
        feed = {
            'original_height': np.array(h, np.int64),
            'original_width': np.array(w, np.int64),
            **{f'vision_pos_enc_{i}': img[i] for i in range(3)},
            **{f'backbone_fpn_{i}': img[3 + i] for i in range(3)},
            'language_mask': txt[0],
            'language_features': txt[1],
            'language_embeds': txt[2],
            'box_coords': np.zeros((n, 1, 4), np.float32),
            'box_labels': np.ones((n, 1), np.int64),
            'box_masks': np.ones((1, n), np.bool_),  # every geometry slot is padding: text only
        }
        out = self.dec.run(None, {k: v for k, v in feed.items() if k in self.dec_in})
        return dict(zip(self.dec_out, out))

    def detect(self, rgb: np.ndarray, names: list[str], min_score: float = 0.15):
        """Boxes (xyxy px), scores, phrase indices and masks of every instance of each of ``names`` in ``rgb``."""
        h, w = rgb.shape[:2]
        B, S, L, M = [], [], [], []
        with self._lock:
            img = self.image(rgb)
            for k, n in enumerate(names):
                b, s, m = postprocess(self.decode(img, self.text(n), (h, w)), (h, w), min_score)
                B.append(b)
                S.append(s)
                L.append(np.full(len(s), k, np.int64))
                M.append(m)
        if not B:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32),
                np.zeros(0, np.int64),
                np.zeros((0, h, w), bool),
            )
        return np.concatenate(B), np.concatenate(S), np.concatenate(L), np.concatenate(M)


def folder() -> Path | None:
    env = os.environ.get('WISSI_SAM3_DIR')
    if env:
        return Path(env)
    return weights.have('sam3-int8', FILES)


_DET: Sam3 | None = None
_LOCK = threading.Lock()


def detector(progress=None, device: str | None = None) -> Sam3:
    """The loaded model; downloads it on first use. Raises ``PermissionError`` until the SAM License is accepted."""
    global _DET
    if not accepted():
        raise PermissionError(
            'SAM 3 is under the SAM License (Meta): accept it first (Annotate → Find all → By text → SAM 3), or use '
            'Grounding DINO + SAM 2.1'
        )
    with _LOCK:
        if _DET is None:
            d = folder()
            if d is None:
                try:
                    d = weights.fetch(RELEASE, FILES, 'sam3-int8', progress, also=lambda n: n.endswith('.data'))
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f'{e}. SAM 3 has not been exported for wissi-viz yet; set WISSI_SAM3_DIR to an export of your '
                        'own (scripts/export_sam3_onnx.py), or use Grounding DINO + SAM 2.1'
                    ) from e
            _DET = Sam3(d, device or os.environ.get('WISSI_SAM3_DEVICE'))
        return _DET


def status() -> dict:
    return {
        'model': 'sam3 (int8)',
        'license': LICENSE_NAME,
        'license_url': LICENSE_URL,
        'accepted': accepted(),
        'downloaded': folder() is not None,
        'loaded': _DET is not None,
        'size_mb': 1300,
    }
