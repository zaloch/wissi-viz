"""Mirror Grounding DINO tiny (IDEA Research, Apache-2.0) as ONNX for wissi-viz's "find by text", and check it.

    pip install torch transformers huggingface_hub onnx onnxruntime scikit-image
    python scripts/build_grounding_onnx.py --out dist/grounding

1. Downloads the ONNX export of ``IDEA-Research/grounding-dino-tiny`` published by onnx-community on Hugging Face
   (``onnx/model.onnx``, float32), at a recorded revision, and its tokenizer vocabulary.
2. Quantizes the weights of the matrix multiplications to int8 (ONNX Runtime dynamic quantization; convolutions stay
   float32, since int8 convolutions need ONNX Runtime ≥ 1.24 on CPU).
3. Checks wissi-viz's own pipeline (:mod:`wissi_viz.grounding`: tokenizer, resizing, post-processing) with both
   files against the original PyTorch model and processor in ``transformers`` on a few pictures and prompts: the
   same tokens, pixels and boxes on square crops (the export takes 800 × 800, like wissi-viz's tiles), and close
   boxes on whole pictures (scaled into 800 × 800 and padded, against transformers' own resizing).

Writes ``grounding-dino-tiny-{fp32,int8}.onnx``, ``vocab.txt``, ``NOTICE.txt``, ``check.json`` and ``SHA256SUMS``;
.github/workflows/grounding-onnx.yml publishes them as the GitHub release ``grounding-dino-tiny-onnx-v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

ONNX_REPO = 'onnx-community/grounding-dino-tiny-ONNX'
TORCH_REPO = 'IDEA-Research/grounding-dino-tiny'
CASES = [  # (scikit-image picture, prompt)
    ('chelsea', 'cat'),
    ('coffee', 'cup. spoon. saucer'),
    ('astronaut', 'person. rocket. flag'),
    ('rocket', 'rocket. tower. tree'),
    ('immunohistochemistry', 'cell nucleus'),
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of each box in ``a`` with each in ``b`` (xyxy)."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def fetch(out: Path, revision: str | None) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(ONNX_REPO, revision=revision)
    rev = info.sha
    onnx_path = hf_hub_download(ONNX_REPO, 'onnx/model.onnx', revision=rev)
    shutil.copy(onnx_path, out / 'grounding-dino-tiny-fp32.onnx')
    tj = json.loads(Path(hf_hub_download(ONNX_REPO, 'tokenizer.json', revision=rev)).read_text(encoding='utf-8'))
    vocab = tj['model']['vocab']
    words = [''] * len(vocab)
    for w, i in vocab.items():
        words[i] = w
    (out / 'vocab.txt').write_text('\n'.join(words) + '\n', encoding='utf-8')
    for n in ('config.json', 'preprocessor_config.json'):
        try:
            shutil.copy(hf_hub_download(ONNX_REPO, n, revision=rev), out / n)
        except Exception:  # noqa: BLE001
            pass
    return {'onnx_repo': ONNX_REPO, 'onnx_revision': rev}


def describe(path: Path) -> None:
    """Inputs, outputs and operator counts of a graph (in the log, for the record)."""
    import collections

    import onnx

    m = onnx.load(str(path), load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print(
        'inputs', [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    )
    print('outputs', [o.name for o in m.graph.output])
    print('opset', [(o.domain, o.version) for o in m.opset_import], 'ops', dict(ops.most_common(40)), flush=True)


def quantize(out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(out / 'grounding-dino-tiny-fp32.onnx'),
        str(out / 'grounding-dino-tiny-int8.onnx'),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Gemm'],
        extra_options={'MatMulConstBOnly': True},  # weights only: activations × activations stay float
    )
    print('int8:', f'{(out / "grounding-dino-tiny-int8.onnx").stat().st_size / 2**20:.0f} MB', flush=True)


def check(out: Path) -> dict:
    import torch
    from skimage import data
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from wissi_viz import grounding as G

    proc = AutoProcessor.from_pretrained(TORCH_REPO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(TORCH_REPO).eval()
    dets = {}
    for v in ('fp32', 'int8'):
        d = out / f'_{v}'
        d.mkdir(exist_ok=True)
        (d / 'model.onnx').unlink(missing_ok=True)
        (d / 'model.onnx').symlink_to((out / f'grounding-dino-tiny-{v}.onnx').resolve())
        shutil.copy(out / 'vocab.txt', d / 'vocab.txt')
        dets[v] = G.GroundingDino(d, 'cpu')
    s = dets['fp32'].sess
    report: dict = {
        'inputs': {i.name: [str(i.type), list(i.shape)] for i in s.get_inputs()},
        'outputs': {o.name: [str(o.type), list(o.shape)] for o in s.get_outputs()},
        'cases': [],
    }
    print(json.dumps(report, indent=1))

    def ref_boxes(ref_in, ref_out, size):
        kw = {'input_ids': ref_in['input_ids'], 'text_threshold': 0.25, 'target_sizes': [size]}
        try:
            r = proc.post_process_grounded_object_detection(ref_out, threshold=0.3, **kw)[0]
        except TypeError:
            r = proc.post_process_grounded_object_detection(ref_out, box_threshold=0.3, **kw)[0]
        return r['boxes'].numpy(), r['scores'].numpy()

    def compare(rb, rs, det, rgb, names, iou):
        t0 = time.perf_counter()
        b, sc, lab = det.detect(rgb, names, min_score=0.3)
        dt = time.perf_counter() - t0
        if len(rb) and len(b):
            M = _iou(rb, b)
            recall = float((M.max(1) >= iou).mean())
            dscore = float(np.median(np.abs(rs - sc[M.argmax(1)])))
        else:
            recall, dscore = (1.0 if len(rb) == len(b) else 0.0), 0.0
        return {'boxes': int(len(b)), 'recall_of_reference': round(recall, 3), 'median_score_diff': round(dscore, 4),
                'seconds': round(dt, 2), 'labels': [names[i] for i in lab[:8]]}  # fmt: skip

    for pic, prompt in CASES:
        full = getattr(data, pic)()
        if full.ndim == 2:
            full = np.stack([full] * 3, -1)
        full = np.ascontiguousarray(full[..., :3]).astype(np.uint8)
        h, w = full.shape[:2]
        k = min(h, w)
        square = np.ascontiguousarray(full[(h - k) // 2 : (h - k) // 2 + k, (w - k) // 2 : (w - k) // 2 + k])
        names = G.phrases(prompt)
        caption = ' '.join(f'{n}.' for n in names)
        ids, spans = G.encode_prompt(dets['fp32'].tok, names)
        for variant, rgb in (('square', square), ('picture', full)):
            ref_in = proc(images=rgb, text=caption, return_tensors='pt')
            px, mask = G.pixels(rgb, fixed=dets['fp32'].fixed)
            case = {'picture': pic, 'variant': variant, 'size': list(rgb.shape[:2]), 'prompt': caption,
                    'same_tokens': ref_in['input_ids'][0].tolist() == ids[0].tolist()}  # fmt: skip
            if list(ref_in['pixel_values'].shape) == list(px.shape):
                case['pixel_max_diff'] = float(np.abs(ref_in['pixel_values'].numpy() - px).max())
            with torch.no_grad():
                ref_out = model(**ref_in)  # transformers end to end, its own resizing
                ours_in = {**ref_in, 'pixel_values': torch.from_numpy(px), 'pixel_mask': torch.from_numpy(mask)}
                torch_ours = model(**ours_in)  # PyTorch on exactly what the ONNX graph gets
            raw = dict(zip(dets['fp32'].outputs, dets['fp32'].sess.run(
                None, {k_: v.numpy() for k_, v in ours_in.items() if k_ in dets['fp32'].inputs})))  # fmt: skip
            case['onnx_vs_torch_logit_max_diff'] = float(np.abs(raw['logits'] - torch_ours.logits.numpy()).max())
            case['onnx_vs_torch_box_max_diff'] = float(np.abs(raw['pred_boxes'] - torch_ours.pred_boxes.numpy()).max())
            rb, rs = ref_boxes(ref_in, ref_out, rgb.shape[:2])
            case['reference'] = [[round(float(x), 1) for x in b] + [round(float(sc), 3)] for b, sc in zip(rb, rs)]
            for v, det in dets.items():
                case[v] = compare(rb, rs, det, rgb, names, 0.9 if (v == 'fp32' and variant == 'square') else 0.75)
            print(json.dumps(case), flush=True)
            report['cases'].append(case)
    C = [c for c in report['cases'] if c['variant'] == 'square']
    P = [c for c in report['cases'] if c['variant'] == 'picture']
    mean = lambda cs, v: round(float(np.mean([c[v]['recall_of_reference'] for c in cs])), 3)
    report['summary'] = {
        'tokens_match': all(c['same_tokens'] for c in report['cases']),
        'pixels_max_diff': max(c.get('pixel_max_diff', 0) for c in C),
        'onnx_vs_torch_logit_max_diff': max(c['onnx_vs_torch_logit_max_diff'] for c in report['cases']),
        'fp32_recall': mean(C, 'fp32'),
        'int8_recall': mean(C, 'int8'),
        'fp32_recall_padded': mean(P, 'fp32'),
        'int8_recall_padded': mean(P, 'int8'),
        'int8_median_score_diff': round(float(np.median([c['int8']['median_score_diff'] for c in C])), 4),
        'fp32_seconds': round(float(np.mean([c['fp32']['seconds'] for c in C])), 2),
        'int8_seconds': round(float(np.mean([c['int8']['seconds'] for c in C])), 2),
    }
    for v in ('fp32', 'int8'):
        shutil.rmtree(out / f'_{v}')
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='dist/grounding')
    ap.add_argument('--revision', default=None, help='onnx-community revision (default: the latest)')
    ap.add_argument('--step', choices=('fetch', 'quantize', 'check', 'all'), default='all')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.step in ('fetch', 'all'):
        src = fetch(out, a.revision)
        (out / 'source.json').write_text(json.dumps(src), encoding='utf-8')
        describe(out / 'grounding-dino-tiny-fp32.onnx')
    if a.step in ('quantize', 'all'):
        quantize(out)
    if a.step not in ('check', 'all'):
        return
    src = json.loads((out / 'source.json').read_text(encoding='utf-8'))
    rep = check(out)
    rep['source'] = src
    rep['sizes_mb'] = {p.name: round(p.stat().st_size / 2**20, 1) for p in sorted(out.glob('*.onnx'))}
    (out / 'check.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    (out / 'NOTICE.txt').write_text(
        'Grounding DINO tiny\n'
        'Copyright (c) 2023 IDEA Research. Licensed under the Apache License, Version 2.0 (LICENSE-Apache-2.0.txt).\n'
        f'Weights: https://huggingface.co/{TORCH_REPO}\n'
        f'ONNX export: https://huggingface.co/{ONNX_REPO} (revision {src["onnx_revision"]}).\n'
        'Modified by wissi-viz: grounding-dino-tiny-int8.onnx has the weights of its matrix multiplications quantized '
        'to 8-bit integers (scripts/build_grounding_onnx.py).\n',
        encoding='utf-8',
    )
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(f'{_sha(out / n)}  {n}\n' for n in names), encoding='utf-8')
    print(json.dumps(rep['summary'], indent=1), json.dumps(rep['sizes_mb']))
    s = rep['summary']
    if not s['tokens_match'] or s['fp32_recall'] < 0.9:
        raise SystemExit('wissi-viz pipeline does not match transformers: see check.json')


if __name__ == '__main__':
    main()
