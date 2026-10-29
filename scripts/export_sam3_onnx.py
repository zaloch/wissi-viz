"""Export SAM 3 (Meta, SAM License) to ONNX for wissi-viz's opt-in "find by text" model, quantize it, and check it.

SAM 3's weights are gated on Hugging Face: request access at https://huggingface.co/facebook/sam3 and log in
(``HF_TOKEN``). Then, with samexporter (MIT, https://github.com/vietanhdev/samexporter) checked out with its pinned
``sam3`` submodule and installed:

    python scripts/export_sam3_onnx.py --samexporter path/to/samexporter --out dist/sam3

1. ``samexporter.export_sam3`` writes the image encoder, language encoder and decoder graphs (float32).
2. The weights of their matrix multiplications are quantized to int8 (ONNX Runtime dynamic quantization).
3. :mod:`wissi_viz.sam3` (float32 and int8) is checked against Meta's own ``Sam3Processor`` in PyTorch on a few
   pictures and prompts: the same instances, the same masks.

Without access to the gated checkpoint, ``--mirror`` starts from the public float32 ONNX export of SAM 3 made with
samexporter (https://huggingface.co/nrl-ai/anylearning-sam3-onnx, ``sam3_vit_h.zip``,
checked against its SHA-256), quantizes it the same way and checks the int8 graphs against the float32 ones through
:mod:`wissi_viz.sam3` (no PyTorch needed):

    python scripts/export_sam3_onnx.py --samexporter path/to/samexporter --out dist/sam3 --mirror

Writes ``sam3-{image-encoder,language-encoder,decoder}.onnx`` (int8 where that keeps the results; ``.data`` files),
``bpe_simple_vocab_16e6.txt.gz``, ``LICENSE-SAM.txt``, ``NOTICE.txt``, ``check.json`` and ``SHA256SUMS``. Run by
.github/workflows/sam3-onnx.yml, which publishes them as the GitHub release ``sam3-onnx-v1``. You may run it yourself
and point ``WISSI_SAM3_DIR`` at the output instead.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

#: the public float32 export, made with an earlier samexporter (its decoder graph holds Sam3Processor's post-processing)
MIRROR = ('nrl-ai/anylearning-sam3-onnx', 'sam3_vit_h.zip',
          '1617436f9db7eb9dd69029c2c465d54b316c36c21143f0fde0d1be513f15b7d5')  # fmt: skip
PARTS = {'image_encoder': 'image-encoder', 'language_encoder': 'language-encoder', 'decoder': 'decoder'}
CASES = [
    ('chelsea', ['cat', 'eye']),
    ('coffee', ['cup', 'spoon']),
    ('astronaut', ['person', 'flag']),
    ('immunohistochemistry', ['cell nucleus', 'brown stain']),
    ('human_mitosis', ['cell nucleus']),
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _pictures():
    from skimage import data

    for pic, prompts in CASES:
        im = getattr(data, pic)()
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        if im.dtype != np.uint8:
            im = (255 * (im - im.min()) / max(1e-9, float(im.max() - im.min()))).astype(np.uint8)
        yield pic, np.ascontiguousarray(im[..., :3]), prompts


def reference(ckpt: str | None) -> dict:
    """Meta's Sam3Processor (PyTorch, CPU) on the test pictures: masks, boxes, scores per prompt."""
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device='cpu', checkpoint_path=ckpt, load_from_HF=ckpt is None)
    proc = Sam3Processor(model, device='cpu', confidence_threshold=0.4)
    out = {}
    for pic, rgb, prompts in _pictures():
        st = proc.set_image(Image.fromarray(rgb))
        for p in prompts:
            t0 = time.perf_counter()
            r = proc.set_text_prompt(prompt=p, state=st)
            out[(pic, p)] = {
                'masks': r['masks'].cpu().numpy().reshape(-1, *rgb.shape[:2]).astype(bool),
                'boxes': r['boxes'].cpu().numpy(),
                'scores': r['scores'].cpu().numpy().reshape(-1),
                'seconds': time.perf_counter() - t0,
            }
            print(pic, p, len(out[(pic, p)]['scores']), 'reference instances', flush=True)
    del model, proc
    gc.collect()
    torch.set_num_threads(os.cpu_count() or 4)
    return out


def export(samexporter: Path, work: Path, ckpt: str | None) -> None:
    cmd = [sys.executable, '-m', 'samexporter.export_sam3', '--output_dir', str(work), '--opset', '18',
           '--device', 'cpu', '--max-geometric-prompts', '8']  # fmt: skip
    if ckpt:
        cmd += ['--checkpoint', ckpt]
    subprocess.run(cmd, cwd=samexporter, check=True)


def from_mirror(work: Path) -> str:
    """The public float32 export, checked and unpacked into ``work`` (``sam3_<part>.onnx`` + external data)."""
    from huggingface_hub import hf_hub_download

    repo, name, sha = MIRROR
    if all((work / f'sam3_{p}.onnx').exists() for p in PARTS):
        _describe(work)
        return f'https://huggingface.co/{repo} ({name}, sha256 {sha})'
    z = Path(hf_hub_download(repo, name, cache_dir=str(work / 'hf')))
    got = _sha(z)
    if got != sha:
        raise SystemExit(
            f'{repo}/{name}: sha256 {got}, expected {sha} (the mirror changed: check it, then update MIRROR)'
        )
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name  # flat: the graphs name their external data files without folders
            if info.is_dir() or not base or base.startswith('.'):
                continue
            with zf.open(info) as r, open(work / base, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 24)
            print('  ', base, f'{(work / base).stat().st_size / 2**20:.0f} MB', flush=True)
    shutil.rmtree(work / 'hf', ignore_errors=True)
    missing = [p for p in PARTS if not (work / f'sam3_{p}.onnx').exists()]
    _describe(work)
    if missing:
        have = sorted(p.name for p in work.iterdir() if p.is_file())
        raise SystemExit(f'{name} has no sam3_{"/".join(missing)}.onnx (it has {have})')
    return f'https://huggingface.co/{repo} ({name}, sha256 {sha})'


def _describe(work: Path) -> None:
    """Print each graph's inputs and outputs (``IO`` lines: the workflow shows them)."""
    import onnx

    for g in sorted(work.glob('*.onnx')):
        m = onnx.load(str(g), load_external_data=False)

        def dims(v):
            return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]

        ins = [f'{v.name}{dims(v)}' for v in m.graph.input if v.name not in {i.name for i in m.graph.initializer}]
        outs = [f'{v.name}{dims(v)}' for v in m.graph.output]
        print(f'IO {g.name}: in {", ".join(ins)} | out {", ".join(outs)}', flush=True)


def _graph_bytes(graph: Path) -> int:
    """A graph's size with the external-data files it refers to."""
    import onnx
    from onnx.external_data_helper import uses_external_data

    m = onnx.load(str(graph), load_external_data=False)
    locs = {
        e.value for t in m.graph.initializer if uses_external_data(t) for e in t.external_data if e.key == 'location'
    }
    return graph.stat().st_size + sum((graph.parent / loc).stat().st_size for loc in locs)


def self_reference(fp32: Path) -> dict:
    """The float32 graphs through wissi-viz on the test pictures (the reference when Meta's PyTorch model is not
    available): the int8 graphs must find the same instances."""
    from wissi_viz import sam3

    m = sam3.Sam3(fp32, 'cpu')
    out = {}
    for pic, rgb, prompts in _pictures():
        for p in prompts:
            t0 = time.perf_counter()
            b, s, _, masks = m.detect(rgb, [p], min_score=0.4)
            out[(pic, p)] = {'masks': masks, 'boxes': b, 'scores': s, 'seconds': time.perf_counter() - t0}
            print(pic, p, int((s > 0.5).sum()), 'float32 instances', flush=True)
    if not (out[('chelsea', 'cat')]['scores'] > 0.5).any():
        raise SystemExit('the float32 graphs do not find the cat in skimage.data.chelsea(): not a working SAM 3')
    del m
    gc.collect()
    return out


#: tried in order; the first whose results match the reference is published (quantizing a graph that holds
#: Sam3Processor's fixed 0.5 cut can push borderline instances under it)
CONFIGS = [
    ('image_encoder', 'language_encoder', 'decoder'),
    ('image_encoder', 'language_encoder'),
    ('image_encoder',),
    (),
]


def quantize(work: Path, out: Path, quant=tuple(PARTS)) -> dict:
    """Write ``sam3-<part>.onnx`` (+ ``.data`` when large): the parts in ``quant`` with int8 matrix multiplications,
    the others as they are (float32). Returns {part: precision}."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    prec = {}
    for local, name in PARTS.items():
        src = work / f'sam3_{local}.onnx'
        dst = out / f'sam3-{name}.onnx'
        for p in out.glob(f'sam3-{name}.onnx*'):
            p.unlink()
        big = _graph_bytes(src) > 1.5 * 2**30
        if local in quant:
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'],
                             use_external_data_format=big)  # fmt: skip
            prec[local] = 'int8'
        else:
            prec[local] = 'float32'
        m = onnx.load(str(dst if local in quant else src))
        for p in out.glob(f'sam3-{name}.onnx?*'):  # the quantizer's own external-data names
            p.unlink()
        if big:  # the weights next to the graph under a stable name
            onnx.save_model(m, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                            location=f'sam3-{name}.onnx.data', size_threshold=1024)  # fmt: skip
        else:
            onnx.save_model(m, str(dst))
        del m
        gc.collect()
        size = sum(p.stat().st_size for p in out.glob(f'sam3-{name}.onnx*')) / 2**20
        print(local, '→', dst.name, prec[local], f'{size:.0f} MB', flush=True)
    return prec


def _wissi_dir(out: Path) -> Path:
    """The release files under the names :mod:`wissi_viz.sam3` looks for (links), for checking."""
    d = out / '_wissi'
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    for local, name in PARTS.items():
        for f in out.glob(f'sam3-{name}.onnx.data'):
            (d / f.name).symlink_to(f.resolve())
        (d / f'{local}.onnx').symlink_to((out / f'sam3-{name}.onnx').resolve())
    shutil.copy(out / 'bpe_simple_vocab_16e6.txt.gz', d / 'bpe_simple_vocab_16e6.txt.gz')
    return d


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = a.reshape(len(a), -1).astype(np.float32)
    B = b.reshape(len(b), -1).astype(np.float32)
    inter = A @ B.T
    return inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1)


def check(ref: dict, dirs: dict[str, Path]) -> dict:
    from wissi_viz import sam3

    rep: dict = {'cases': []}
    models = {v: sam3.Sam3(d, 'cpu') for v, d in dirs.items()}
    for pic, rgb, prompts in _pictures():
        for p in prompts:
            R = ref[(pic, p)]
            keep = R['scores'] > 0.5
            rm = R['masks'][keep]
            case = {
                'picture': pic,
                'prompt': p,
                'reference': int(keep.sum()),
                'reference_seconds': round(R['seconds'], 1),
            }
            for v, m in models.items():
                t0 = time.perf_counter()
                b, s, lab, masks = m.detect(rgb, [p], min_score=0.4)
                dt = time.perf_counter() - t0
                if len(rm) and len(masks):
                    M = _iou(rm, masks)
                    best = M.max(1)
                    case[v] = {
                        'instances': int((s > 0.5).sum()),
                        'recall_iou90': round(float((best >= 0.9).mean()), 3),
                        'recall_iou75': round(float((best >= 0.75).mean()), 3),
                        'median_iou': round(float(np.median(best)), 3),
                        'median_score_diff': round(float(np.median(np.abs(R['scores'][keep] - s[M.argmax(1)]))), 4),
                        'seconds': round(dt, 1),
                    }
                else:
                    ok = len(rm) == int((s > 0.5).sum())
                    case[v] = {'instances': int((s > 0.5).sum()), 'recall_iou90': 1.0 if ok else 0.0,
                               'recall_iou75': 1.0 if ok else 0.0, 'median_iou': 1.0 if ok else 0.0, 'seconds': round(dt, 1)}  # fmt: skip
            print(json.dumps(case), flush=True)
            rep['cases'].append(case)
    C = rep['cases']
    rep['summary'] = {
        v: {
            'recall_iou90': round(float(np.mean([c[v]['recall_iou90'] for c in C])), 3),
            'recall_iou75': round(float(np.mean([c[v]['recall_iou75'] for c in C])), 3),
            'median_iou': round(float(np.median([c[v]['median_iou'] for c in C])), 3),
            'seconds_per_prompt': round(float(np.mean([c[v]['seconds'] for c in C])), 1),
        }
        for v in models
    }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samexporter', required=True, help='samexporter checkout (with the sam3 submodule)')
    ap.add_argument('--out', default='dist/sam3')
    ap.add_argument('--work', default='work/sam3', help='float32 export (large; not published)')
    ap.add_argument('--checkpoint', default=None, help='a local sam3.pt (default: download from Hugging Face)')
    ap.add_argument('--skip-check', action='store_true')
    ap.add_argument('--mirror', action='store_true', help='start from the public float32 export (no gated access)')
    a = ap.parse_args()
    sx, out, work = Path(a.samexporter).resolve(), Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    source = None
    if a.mirror:
        source = from_mirror(work)
        ref = None
    else:
        ref = None if a.skip_check else reference(a.checkpoint)
        if not (work / 'sam3_decoder.onnx').exists():
            export(sx, work, a.checkpoint)
    shutil.copy(sx / 'samexporter' / 'assets' / 'bpe_simple_vocab_16e6.txt.gz', out / 'bpe_simple_vocab_16e6.txt.gz')
    shutil.copy(sx / 'sam3' / 'LICENSE', out / 'LICENSE-SAM.txt')
    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=sx, capture_output=True, text=True).stdout.strip()
    rev3 = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=sx / 'sam3', capture_output=True, text=True).stdout.strip()
    notice = (
        'SAM 3 by Meta Platforms, Inc., licensed under the SAM License (LICENSE-SAM.txt), which applies to these files.\n'
        'Weights: https://huggingface.co/facebook/sam3 (official checkpoint). Model code: '
        f'https://github.com/facebookresearch/sam3 ({rev3}).\n'
        + (
            f'Exported to ONNX with samexporter (MIT, https://github.com/vietanhdev/samexporter, {rev}).\n'
            if source is None
            else 'Exported to ONNX with samexporter (MIT, https://github.com/vietanhdev/samexporter): the public '
            f'float32 export at {source}, made with an earlier samexporter revision.\n'
        )
        + 'Modified by wissi-viz: the weights of the matrix multiplications of {quantized} were quantized to 8-bit '
        'integers (scripts/export_sam3_onnx.py; check.json lists each part).\n'
        'If you publish results obtained with SAM 3, the SAM License asks you to acknowledge its use.\n'
        'The CLIP tokenizer vocabulary (bpe_simple_vocab_16e6.txt.gz) is from OpenAI CLIP (MIT).\n'
    )
    rep: dict = {'samexporter': rev, 'sam3': rev3, 'float32_from': source or 'facebook/sam3 via samexporter'}
    if a.skip_check:
        rep['precision'] = quantize(work, out)
    else:
        fp32 = work / '_wissi'
        fp32.mkdir(exist_ok=True)
        for f in work.iterdir():  # the graphs and their external data
            if f.is_file() and not (fp32 / f.name).exists():
                (fp32 / f.name).symlink_to(f.resolve())
        shutil.copy(out / 'bpe_simple_vocab_16e6.txt.gz', fp32 / 'bpe_simple_vocab_16e6.txt.gz')
        if ref is None:  # mirror: the float32 graphs through wissi-viz are the reference for the release
            ref = self_reference(fp32)
            rep['reference'] = 'the float32 graphs through wissi_viz.sam3 (no PyTorch reference without HF access)'
            need = 0.85
        else:
            base = check(ref, {'fp32': fp32})
            rep['fp32'] = base
            if base['summary']['fp32']['recall_iou75'] < 0.8:
                raise SystemExit("wissi-viz SAM 3 pipeline does not match Meta's Sam3Processor: see the log")
            need = base['summary']['fp32']['recall_iou75'] - 0.1
        rep['tried'] = []
        for quant in CONFIGS:
            prec = quantize(work, out, quant)
            r = check(ref, {'release': _wissi_dir(out)})
            ok = r['summary']['release']['recall_iou75'] >= need
            rep['tried'].append({'precision': prec, 'summary': r['summary']['release'], 'kept': ok})
            print('tried', json.dumps(rep['tried'][-1]), flush=True)
            if ok:
                rep.update(r)
                rep['precision'] = prec
                break
        shutil.rmtree(out / '_wissi', ignore_errors=True)
        if 'precision' not in rep:
            raise SystemExit('no configuration finds what the reference finds: see the log')
    q = [k.replace('_', ' ') for k, v in rep['precision'].items() if v == 'int8']
    (out / 'NOTICE.txt').write_text(
        notice.replace('{quantized}', ', '.join(q) if q else 'no part (all float32)'), encoding='utf-8'
    )
    rep['sizes_mb'] = {p.name: round(p.stat().st_size / 2**20, 1) for p in sorted(out.iterdir()) if p.is_file()}
    (out / 'check.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(f'{_sha(out / n)}  {n}\n' for n in names), encoding='utf-8')
    print(json.dumps(rep.get('summary'), indent=1), json.dumps(rep['precision']), json.dumps(rep['sizes_mb']))


if __name__ == '__main__':
    main()
