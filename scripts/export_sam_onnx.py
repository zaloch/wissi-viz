"""Export SAM 2.1 (Apache-2.0) to ONNX for wissi-viz's ``onnx`` SAM backend, and check it against PyTorch.

    pip install torch "SAM-2 @ git+https://github.com/facebookresearch/sam2.git" huggingface_hub onnx onnxruntime
    python scripts/export_sam_onnx.py --model small --out dist/sam

Writes ``sam2.1-<model>-encoder.onnx`` (image → embeddings) and ``sam2.1-<model>-decoder.onnx`` (embeddings + point /
box prompts → 3 masks + IoU scores), a ``SHA256SUMS`` file and the licence. Run by .github/workflows/sam-onnx.yml,
which publishes them as a GitHub release; ``wissi_viz.sam`` downloads them on first use and verifies the checksums.

Interface (what wissi_viz/sam.py feeds it):
  encoder: image float32 (1, 3, 1024, 1024), RGB resized to 1024², /255, ImageNet mean/std
           → image_embed (1, 256, 64, 64), high_res_feats_0 (1, 32, 256, 256), high_res_feats_1 (1, 64, 128, 128)
  decoder: image_embed, high_res_feats_0, high_res_feats_1, point_coords float32 (1, N, 2) in 1024-px space,
           point_labels float32 (1, N): 1 positive, 0 negative, 2 / 3 box corners
           → masks (1, 3, 256, 256) logits, iou_predictions (1, 3)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = {'tiny': 'facebook/sam2.1-hiera-tiny', 'small': 'facebook/sam2.1-hiera-small',
          'base-plus': 'facebook/sam2.1-hiera-base-plus', 'large': 'facebook/sam2.1-hiera-large'}


def build(model: str):
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor.from_pretrained(MODELS[model], device='cpu')
    m = pred.model.eval()

    class Encoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):
            out = self.m.forward_image(image)
            _, feats, _, _ = self.m._prepare_backbone_features(out)
            if self.m.directly_add_no_mem_embed:
                feats[-1] = feats[-1] + self.m.no_mem_embed
            sizes = [(256, 256), (128, 128), (64, 64)]
            maps = [f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(feats[::-1], sizes[::-1])][::-1]
            return maps[2], maps[0], maps[1]

    class Decoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels):
            sparse, dense = self.m.sam_prompt_encoder(points=(point_coords, point_labels.int()), boxes=None, masks=None)
            masks, iou, _, _ = self.m.sam_mask_decoder(
                image_embeddings=image_embed, image_pe=self.m.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True,
                repeat_image=False, high_res_features=[high_res_feats_0, high_res_feats_1])
            return masks, iou

    return pred, Encoder(m).eval(), Decoder(m).eval()


def export(model: str, out: Path) -> dict:
    import torch
    pred, enc, dec = build(model)
    out.mkdir(parents=True, exist_ok=True)
    fe, fd = out / f'sam2.1-{model}-encoder.onnx', out / f'sam2.1-{model}-decoder.onnx'
    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        e, h0, h1 = enc(x)
        torch.onnx.export(enc, (x,), fe, input_names=['image'],
                          output_names=['image_embed', 'high_res_feats_0', 'high_res_feats_1'], opset_version=17,
                          dynamo=False)
        pc = torch.tensor([[[500.0, 400.0], [600.0, 420.0]]])
        pl = torch.tensor([[1.0, 0.0]])
        torch.onnx.export(dec, (e, h0, h1, pc, pl), fd,
                          input_names=['image_embed', 'high_res_feats_0', 'high_res_feats_1', 'point_coords', 'point_labels'],
                          output_names=['masks', 'iou_predictions'], opset_version=17,
                          dynamic_axes={'point_coords': {1: 'n'}, 'point_labels': {1: 'n'}}, dynamo=False)
    return {'encoder': fe, 'decoder': fd, 'pred': pred}


def check(files: dict, model: str) -> dict:
    """Same image and prompts through SAM2ImagePredictor (PyTorch) and through wissi_viz.sam's ONNX backend: the
    masks must agree (IoU ≥ 0.95)."""
    import os
    import torch
    rng = np.random.default_rng(0)
    img = np.full((768, 1000, 3), (235, 170, 205), np.uint8)
    yy, xx = np.mgrid[0:768, 0:1000]
    for cx, cy, r in ((300, 300, 60), (650, 420, 90), (820, 200, 40)):
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = (80, 50, 140)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    pred = files['pred']
    with torch.inference_mode():
        pred.set_image(img)
    os.environ['WISSI_SAM_ONNX_DIR'] = str(files['encoder'].parent / '_check')
    d = Path(os.environ['WISSI_SAM_ONNX_DIR'])
    d.mkdir(exist_ok=True)
    for k in ('encoder', 'decoder'):
        (d / f'{k}.onnx').write_bytes(files[k].read_bytes())
    from wissi_viz.sam import _Onnx
    ox = _Onnx(model, 'cpu')
    emb = ox.embed(img)
    res = {}
    for name, pts, labels, box in (('point', [(650, 420)], [1], None), ('box', [], [], (250, 230, 370, 370)),
                                   ('point+neg', [(300, 300), (650, 420)], [1, 0], None)):
        with torch.inference_mode():
            m_t, s_t, _ = pred.predict(point_coords=np.asarray(pts, np.float32) if pts else None,
                                       point_labels=np.asarray(labels) if pts else None,
                                       box=np.asarray(box, np.float32) if box else None, multimask_output=True)
        mt = m_t[int(np.argmax(s_t))] > 0
        mo, _ = ox.predict(emb, pts, labels, box)
        iou = float((mt & mo).sum() / max((mt | mo).sum(), 1))
        res[name] = round(iou, 4)
        assert iou >= 0.95, f'{name}: ONNX vs PyTorch IoU {iou:.3f}'
    import shutil
    shutil.rmtree(d)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', nargs='+', default=['tiny', 'small'], choices=list(MODELS))
    ap.add_argument('--out', default='dist/sam')
    a = ap.parse_args()
    out = Path(a.out)
    report = {}
    for model in a.model:
        files = export(model, out)
        report[model] = check(files, model)
        print(model, report[model], flush=True)
    sums = []
    for f in sorted(out.glob('*.onnx')):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        sums.append(f'{h}  {f.name}')
    (out / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    (out / 'check.json').write_text(json.dumps(report, indent=1))
    (out / 'NOTICE.txt').write_text(
        'SAM 2.1 model weights and code by Meta Platforms, Inc., licensed under the Apache License 2.0 '
        '(https://github.com/facebookresearch/sam2). These ONNX files are a format conversion of those weights, '
        'distributed under the same licence.\n')


if __name__ == '__main__':
    main()
