"""Why SAM 3 finds nothing on slides: its raw scores for short names on real microscopy (NuInsSeg H&E nuclei, rat DRG
neurofilament neurons) at several scales, against the reference outlines, with the cat photo as a control; then
Find by text on the NuInsSeg mosaic with every tile's raw and kept detections logged.

    WISSI_SAM3_ACCEPT_LICENSE=1 uv run python scripts/sam3_slide_probe.py --out sam3-probe

Writes probe.json, pictures of the best detections and prints one summary per case.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wissi_viz import sam3, validation, weights

NUCLEUS_NAMES = ['nucleus', 'nuclei', 'cell nucleus', 'cell', 'purple nucleus', 'round purple spot']
NEURON_NAMES = ['neuron', 'cell', 'cell body', 'round cell', 'bright circle']


def notice(title: str, text: str) -> None:
    print(f'{title}: {text}', flush=True)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::notice title={title}::{" ".join(text.split())[:4000]}', flush=True)


def lowered_detector():
    """SAM 3 with its decoder's score cut at 0.001 instead of 0.05: every query's score can be read."""
    det = sam3.detector()
    dpath = next(det.folder / n for n in sam3.GRAPHS['decoder'] if (det.folder / n).exists())
    low = sam3.lower_score_cut(dpath, 0.001)
    if low is not None:
        det.dec = weights.ort_session(low, None)
        det.score_cut = 0.001
    return det


def match(masks: np.ndarray, labels: np.ndarray, iou: float = 0.5) -> tuple[int, float]:
    """Detections whose mask overlaps one labelled object at IoU ≥ ``iou`` (one-to-one, greedy): (matches, mean IoU)."""
    used, ious = set(), []
    for m in masks:
        ids, n = np.unique(labels[m], return_counts=True)
        best, bi = 0.0, 0
        for i, c in zip(ids, n):
            if i == 0 or i in used:
                continue
            v = c / ((labels == i).sum() + m.sum() - c)
            if v > best:
                best, bi = v, i
        if best >= iou:
            used.add(bi)
            ious.append(best)
    return len(ious), float(np.mean(ious)) if ious else 0.0


def probe(det, rgb: np.ndarray, labels: np.ndarray | None, names: list[str]) -> list[dict]:
    """Every name on ``rgb``: how many detections at each score, the top scores, and matches with ``labels``."""
    out = []
    for n in names:
        t0 = time.time()
        b, s, _, m = det.detect(rgb, [n], min_score=0.001)
        r = {'name': n, 'seconds': round(time.time() - t0, 1), 'top': [round(float(v), 3) for v in s[:5]],
             **{f'n>={c}': int((s >= c).sum()) for c in (0.01, 0.05, 0.1, 0.3, 0.5)}}  # fmt: skip
        if labels is not None:
            for c in (0.05, 0.1, 0.3):
                k = s >= c
                r[f'tp@{c}'], r[f'iou@{c}'] = match(m[k], labels)
            r['median_px'] = float(np.median(np.sqrt(m[s >= 0.05].sum((1, 2))))) if (s >= 0.05).any() else 0
        out.append(r)
    return out


def picture(rgb: np.ndarray, det, name: str, path: Path, cut: float = 0.1) -> None:
    b, s, _, m = det.detect(rgb, [name], min_score=cut)
    im = Image.fromarray(rgb).convert('RGB')
    d = ImageDraw.Draw(im)
    for bx, sc in zip(b, s):
        d.rectangle([float(v) for v in bx], outline=(0, 255, 120), width=2)
        d.text((float(bx[0]) + 2, float(bx[1]) + 1), f'{sc:.2f}', fill=(255, 255, 0))
    im.save(path)


def short(rows: list[dict], ref: int | None) -> str:
    return ' | '.join(
        f'{r["name"]}: n≥.05={r["n>=0.05"]} n≥.1={r["n>=0.1"]} n≥.3={r["n>=0.3"]} top={r["top"][:3]}'
        + (
            f' tp@.1={r.get("tp@0.1")}/{ref} iou={r.get("iou@0.1", 0):.2f} ~{r.get("median_px", 0):.0f}px'
            if ref
            else ''
        )
        for r in rows
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default='sam3-probe')
    p.add_argument('--work', default='sam3-probe-work')
    a = p.parse_args(argv)
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    (work / 'layers').mkdir(parents=True, exist_ok=True)
    det = lowered_detector()
    rec: dict = {'device': det.device, 'score_cut': det.score_cut}
    from skimage import data

    rec['cat'] = probe(det, data.chelsea(), None, ['cat', 'animal'])
    notice('sam3 control: cat photo', short(rec['cat'], None))

    # ---- NuInsSeg: four H&E patches of different organs (512 px at ~0.25 µm/px), labelled nuclei
    zf = zipfile.ZipFile(validation.nuinsseg_zip(work, print))
    pairs, seen = [], set()
    for organ, im, mk in validation._nuinsseg_pairs(zf):
        if organ not in seen:
            seen.add(organ)
            pairs.append((organ, im, mk))
        if len(pairs) == 4:
            break
    rec['nuinsseg'] = []
    for organ, im, mk in pairs:
        rgb = np.asarray(Image.open(io.BytesIO(zf.read(im))).convert('RGB'))
        lab = np.asarray(Image.open(io.BytesIO(zf.read(mk))))
        lab = lab[..., 0] if lab.ndim == 3 else lab
        ref = len(np.unique(lab)) - 1
        h, w = rgb.shape[:2]
        c = min(h, w) // 2
        variants = {
            'as given (SAM 3 enlarges to 1008)': (rgb, lab),
            'quarter, 4x enlarged': (rgb[:c, :c], lab[:c, :c]),
            'halved (nuclei ~half size)': (np.asarray(Image.fromarray(rgb).resize((w // 2, h // 2))),
                                           np.asarray(Image.fromarray(lab).resize((w // 2, h // 2), Image.NEAREST))),
        }  # fmt: skip
        lines = []  # one notice per organ (a step keeps only 10)
        for vname, (x, lb) in variants.items():
            rows = probe(det, x, lb, NUCLEUS_NAMES)
            nref = len(np.unique(lb)) - 1
            rec['nuinsseg'].append({'organ': organ, 'variant': vname, 'reference': nref, 'rows': rows})
            lines.append(f'[{vname}] {short(rows, nref)}')
        notice(f'sam3 H&E {organ}', ' || '.join(lines))
        picture(rgb, det, 'nucleus', out / f'nuinsseg-{organ.replace(" ", "_")}.nucleus.png')
        Image.fromarray(rgb).save(out / f'nuinsseg-{organ.replace(" ", "_")}.png')
        print(f'{organ}: {ref} nuclei', flush=True)

    # ---- rat DRG: neurofilament, one channel shown as grey, two full-resolution crops
    slide, refp, mpp = validation.drg_mosaic(work, n=2, log=print)
    import tifffile

    from wissi_viz import annio

    img = tifffile.imread(slide)
    img = img[0] if img.ndim == 3 else img
    lo, hi = np.percentile(img[img > 0], (0.5, 99.8))
    g = (np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
    R = annio.read_geoparquet(refp)
    from skimage.draw import polygon as fill

    lab = np.zeros(g.shape, np.int32)
    for i in range(len(R)):
        ring = R.rings(i)[0]
        rr, cc = fill(ring[:, 1], ring[:, 0], g.shape)
        lab[rr, cc] = i + 1
    ys, xs = np.nonzero(lab)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    rec['drg'] = []
    lines = []
    for side in (512, 1024):
        y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
        x = np.stack([g[y0 : y0 + side, x0 : x0 + side]] * 3, -1)
        lb = lab[y0 : y0 + side, x0 : x0 + side]
        rows = probe(det, x, lb, NEURON_NAMES)
        nref = len(np.unique(lb)) - 1
        rec['drg'].append({'crop': side, 'mpp': mpp, 'reference': nref, 'rows': rows})
        lines.append(f'[{side}px crop, ~{mpp:.2f} µm/px] {short(rows, nref)}')
        picture(x, det, 'neuron', out / f'drg-{side}.neuron.png')
        Image.fromarray(x).save(out / f'drg-{side}.png')
    notice('sam3 rat DRG', ' || '.join(lines))

    (out / 'probe.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
