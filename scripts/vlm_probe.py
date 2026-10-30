"""A small test of a vision-language model through Ollama: shown an image and a goal, does it describe the image
correctly and plan an analysis with wissi-viz's tools? Nothing is run or measured; this only reads the model's answers.

    ollama pull qwen3-vl:4b
    uv run --with tifffile --with scipy python scripts/vlm_probe.py      # rat DRG images, downloaded once (~Zenodo)
    uv run --with tifffile --with scipy python scripts/vlm_probe.py --model qwen3-vl:8b

The images: rat dorsal root ganglion, neurofilament immunofluorescence, neurons outlined by three experts (Schulte,
Troidl et al. 2023, https://doi.org/10.5281/zenodo.6546069, CC BY 4.0). Unpacking needs 7-Zip (7z) or bsdtar.

Writes one JSON file per image (the answers, timings) to --out and prints a short summary.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS = {
    'find_by_text': 'outline every object matching a short name (e.g. "nucleus", "gland"), within a size range in µm',
    'find_cells': 'outline every nucleus or cell with a cell model, measuring area, shape and stain or channel intensity',
    'segment_everything': 'outline every object of a size, then keep those similar to examples',
    'find_areas': 'map tissue areas (e.g. tumour, stroma) from a few drawn examples of each',
    'split_by_marker': 'split found objects into positive and negative by a stain or channel (e.g. DAB, a fluorescent marker)',
    'count': 'count objects of a layer, overall or per class or per area',
    'measure': 'measure objects or areas: size, shape, intensity, density',
}

LOOK = {
    'type': 'object',
    'properties': {
        'is_microscopy': {'type': 'boolean'},
        'modality': {'type': 'string', 'enum': ['brightfield', 'fluorescence', 'other']},
        'stain': {'type': 'string'},
        'tissue': {'type': 'string'},
        'structures': {'type': 'array', 'items': {'type': 'string'}},
        'approx_object_count': {'type': 'integer'},
        'notes': {'type': 'string'},
    },
    'required': ['is_microscopy', 'modality', 'stain', 'tissue', 'structures', 'approx_object_count', 'notes'],
}

PLAN = {
    'type': 'object',
    'properties': {
        'feasible': {'type': 'string', 'enum': ['yes', 'partly', 'no']},
        'not_possible': {'type': 'array', 'items': {'type': 'string'}},
        'steps': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'tool': {'type': 'string', 'enum': list(TOOLS)},
                    'what': {'type': 'string'},
                    'size_um_min': {'type': 'number'},
                    'size_um_max': {'type': 'number'},
                    'marker': {'type': 'string'},
                    'why': {'type': 'string'},
                },
                'required': ['tool', 'what', 'why'],
            },
        },
        'result': {'type': 'string'},
        'risks': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['feasible', 'steps', 'not_possible', 'result', 'risks'],
}

SYSTEM = (
    'You assist with microscopy image analysis. You look at images and plan analyses, but you never measure or count '
    'precisely yourself: deterministic tools do that. Answer only with the JSON requested. Be factual; if the image '
    'does not fit the goal, say so.'
)


DRG_RECORD = 'https://zenodo.org/api/records/6546069'  # rat DRG, Schulte, Troidl et al. 2023, CC BY 4.0
DRG_FILE = 'Annotations_ratDRG.rar'
DRG_NEURON_UM = 30.0  # typical rat DRG neuron diameter: the images carry no pixel size, so it is estimated from this


def drg_folder(work: Path) -> Path:
    """The rat DRG neurofilament images and the experts' consensus masks (downloaded once, licence checked)."""
    nf = work / 'drg' / 'Annotations_ratDRG' / 'NF'
    if nf.exists():
        return nf
    meta = json.loads(urllib.request.urlopen(DRG_RECORD, timeout=60).read().decode())
    lic = str((meta.get('metadata') or {}).get('license', {}).get('id', '')).lower()
    if not lic.startswith('cc-by') or '-nc' in lic or '-nd' in lic:
        raise SystemExit(f'{DRG_RECORD}: licence {lic!r} is not CC BY')
    f = next(f for f in meta['files'] if f['key'] == DRG_FILE)
    rar = work / DRG_FILE
    if not rar.exists():
        print(f'downloading {DRG_FILE} ({f["size"] / 1e6:.0f} MB)', flush=True)
        urllib.request.urlretrieve(f['links']['self'], rar)
    for tool, cmd in (('7z', ['7z', 'x', '-y', f'-o{work / "drg"}', str(rar)]),
                      ('bsdtar', ['bsdtar', '-xf', str(rar), '-C', str(work / 'drg')])):  # fmt: skip
        if shutil.which(tool):
            (work / 'drg').mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True, capture_output=True)
            break
    if not nf.exists():
        raise SystemExit('could not unpack the DRG archive: install 7-Zip (7z) or bsdtar')
    return nf


def drg_image(nf: Path, k: int) -> tuple[str, np.ndarray, np.ndarray, int, float]:
    """Image ``k``: (name, 8-bit RGB as the model sees it, experts' mask, their neuron count, µm per pixel)."""
    import tifffile
    from scipy import ndimage as ndi

    imgs = sorted((nf / 'images').glob('*.tif'))
    p = imgs[k % len(imgs)]
    a = tifffile.imread(p).astype(np.float32)
    a = a[0] if a.ndim == 3 else a
    lo, hi = np.percentile(a, (0.5, 99.8))
    g = (np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
    m = np.asarray(Image.open(nf / 'masks' / 'GT_Estimation' / f'{p.stem}.png'))
    m = (m[..., 0] if m.ndim == 3 else m) > 0
    lab, n = ndi.label(m)
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes > 50]
    px = float(np.median(2 * np.sqrt(sizes / np.pi))) if len(sizes) else 60.0
    return f'drg-{p.stem}', np.stack([g] * 3, -1), m, int(len(sizes)), DRG_NEURON_UM / px


def outlined(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The image with the experts' neuron outlines drawn on in yellow."""
    from scipy import ndimage as ndi

    edge = mask & ~ndi.binary_erosion(mask, iterations=3)
    out = rgb.copy()
    out[edge] = (255, 210, 0)
    return out


def views(rgb: np.ndarray, mask: np.ndarray, mpp: float, side: int = 1280, crop: int = 512, zoom: int = 2) -> list:
    """What the model is shown: the whole section (longest side ``side`` px) and two details at full resolution, each
    ``crop`` px enlarged ``zoom`` times so that neurons are ~60 px across, chosen where the tissue is densest. Returns
    [(label, image, the same with the experts' outlines, µm per shown pixel)]."""
    from scipy import ndimage as ndi

    H, W = rgb.shape[:2]
    k = min(1.0, side / max(H, W))
    size = (max(1, round(W * k)), max(1, round(H * k)))
    out = [('the whole section', np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS)),
            np.asarray(Image.fromarray(outlined(rgb, mask)).resize(size, Image.LANCZOS)), mpp / k)]  # fmt: skip
    g = ndi.uniform_filter(rgb[..., 0].astype(np.float32), 31)
    tissue = g > g.mean() + 0.25 * g.std()
    frac = ndi.uniform_filter(tissue.astype(np.float32), crop)
    picked = []
    for y, x in zip(*np.unravel_index(np.argsort(frac, axis=None)[::-1], frac.shape)):
        if len(picked) == 2:
            break
        if all(max(abs(y - py), abs(x - px)) >= crop for py, px in picked):
            picked.append((int(y), int(x)))
    for j, (y, x) in enumerate(picked):
        y0 = min(max(0, y - crop // 2), max(0, H - crop))
        x0 = min(max(0, x - crop // 2), max(0, W - crop))
        box = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        big = (min(crop, W - x0) * zoom, min(crop, H - y0) * zoom)
        out.append((f'detail {j + 1}', np.asarray(Image.fromarray(rgb[box]).resize(big, Image.BICUBIC)),
                    np.asarray(Image.fromarray(outlined(rgb[box], mask[box])).resize(big, Image.BICUBIC)),
                    mpp / zoom))  # fmt: skip
    return out


def samples(work: Path) -> list[dict]:
    """Rat DRG neurofilament immunofluorescence (CC BY 4.0) with three goals: one plain, one with sizes, and one
    that can only partly be done (the images have one marker, so CGRP-positive neurons cannot be told apart)."""
    nf = drg_folder(work)
    n_imgs = len(list((nf / 'images').glob('*.tif')))
    out = []
    goals = [
        ('Outline every neuron cell body and count them, leaving out the satellite glia and Schwann cells around them.',
         {'tools': ['find_by_text|find_cells|segment_everything'], 'feasible': ['yes', 'partly']}),
        ('Measure the size of each neuron and report how many are large (more than 40 µm across).',
         {'tools': ['find_by_text|find_cells|segment_everything', 'measure|count'], 'feasible': ['yes', 'partly']}),
        ('Count the neurons and report what fraction of them are CGRP-positive nociceptors.',
         {'tools': ['find_by_text|find_cells|segment_everything'], 'feasible': ['partly', 'no'],
          'not_possible': True}),
    ]  # fmt: skip
    for j, (goal, exp) in enumerate(goals):
        name, rgb, mask, n, mpp = drg_image(nf, j * max(n_imgs // 3, 1))
        out.append({
            'name': f'{name}-goal{j + 1}',
            'views': views(rgb, mask, mpp),
            'goal': goal,
            'context': ('Sample: a section of rat dorsal root ganglion (DRG). One fluorescence channel: Neurofilament '
                        f'(an antibody marking neurons). Pixel size about {mpp:.2f} µm.'),
            'reference_count': n,
            'expect': {'is_microscopy': True, 'modality': 'fluorescence',
                       'stain_has': ['neurofil', 'fluoresc', 'immuno', 'antibod', 'marker', 'neuron'], **exp},
        })  # fmt: skip
    return out


def png_b64(rgb: np.ndarray) -> str:
    im = Image.fromarray(rgb).convert('RGB')
    b = io.BytesIO()
    im.save(b, format='PNG')
    return base64.b64encode(b.getvalue()).decode()


def notice(title: str, text: str) -> None:
    print(f'{title}: {text}', flush=True)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::notice title={title}::{" ".join(text.split())[:3500]}', flush=True)


def chat(host: str, model: str, messages: list[dict], schema: dict | None, limit: int, budget: float = 600,
         think: bool | None = None) -> tuple:  # fmt: skip
    """One streamed chat: (the answer, parsed if a schema was given; stats). Stops at ``limit`` tokens (thinking
    included) or after ``budget`` seconds; an unparsable answer comes back as {'_raw': its end}."""
    body = {
        'model': model,
        'messages': messages,
        'stream': True,
        'options': {'temperature': 0, 'num_ctx': 8192, 'num_predict': limit},
    }
    if think is not None:
        body['think'] = think
    if schema:
        body['format'] = schema
    req = urllib.request.Request(
        f'{host}/api/chat', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    t0, first, content, thinking, n, last, stats = time.time(), None, [], [], 0, {}, {}
    with urllib.request.urlopen(req, timeout=budget + 60) as r:
        for line in r:
            if not line.strip():
                continue
            last = json.loads(line)
            if 'error' in last:
                stats['error'] = last['error']
                break
            m = last.get('message', {})
            if first is None and (m.get('content') or m.get('thinking')):
                first = time.time() - t0
            content.append(m.get('content') or '')
            thinking.append(m.get('thinking') or '')
            n += 1
            if time.time() - t0 > budget:
                stats['stopped'] = f'over {budget:.0f} s'
                break
    rate = last['eval_count'] / (last['eval_duration'] / 1e9) if last.get('eval_duration') else None
    stats |= {
        'seconds': round(time.time() - t0, 1),
        'first_token_s': round(first, 1) if first is not None else None,
        'prompt_tokens': last.get('prompt_eval_count'),
        'answer_tokens': last.get('eval_count', n),
        'tokens_per_s': round(rate, 1) if rate else None,
        'done_reason': last.get('done_reason'),
        'thinking_chars': len(''.join(thinking)),
    }
    text = ''.join(content)
    if not schema:
        return text, stats
    try:
        return json.loads(text), stats
    except ValueError:
        return {'_raw': text[-1500:]}, stats


def check(look: dict, plan: dict, expect: dict) -> list[str]:
    """Plain checks of the answers against what the image is; returns the problems found."""
    bad = []
    if 'is_microscopy' in expect and look.get('is_microscopy') != expect['is_microscopy']:
        bad.append(f'is_microscopy={look.get("is_microscopy")}')
    if expect.get('is_microscopy') is False and plan.get('feasible') != 'no':
        bad.append('planned an analysis of a non-microscopy image')
    if 'modality' in expect and look.get('modality') != expect['modality']:
        bad.append(f'modality={look.get("modality")}')
    stain = (look.get('stain') or '').lower()
    if 'stain_has' in expect and not any(w in stain for w in expect['stain_has']):
        bad.append(f'stain="{look.get("stain")}"')
    if expect.get('is_microscopy', True) and not plan.get('steps'):
        bad.append('no steps planned')
    used = {st.get('tool') for st in plan.get('steps', [])}
    for want in expect.get('tools', []):
        if not used & set(want.split('|')):
            bad.append(f'no {want} step')
    if 'feasible' in expect and plan.get('feasible') not in expect['feasible']:
        bad.append(f'feasible={plan.get("feasible")}')
    if expect.get('not_possible') and not plan.get('not_possible'):
        bad.append('nothing flagged as not possible')
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', default=os.environ.get('WISSI_VLM_MODEL', 'qwen3-vl:4b'))
    p.add_argument('--host', default=os.environ.get('OLLAMA_HOST_URL', 'http://localhost:11434'))
    p.add_argument('--out', default='vlm-probe')
    p.add_argument('--work', default='vlm-probe-work', help='where the DRG images are downloaded and unpacked')
    p.add_argument('--budget', type=float, default=float(os.environ.get('WISSI_VLM_BUDGET', 900)),
                   help='most seconds per answer')  # fmt: skip
    p.add_argument('--limit', type=int, default=0, help='most tokens per answer (default 1024; 4096 if thinking)')
    p.add_argument('--think', choices=['on', 'off', 'default'], default=os.environ.get('WISSI_VLM_THINK', 'off'),
                   help="the model's reasoning before answering (default off)")  # fmt: skip
    a = p.parse_args(argv)
    think = {'on': True, 'off': False, 'default': None}[a.think]
    limit = a.limit or (4096 if think or (think is None and 'thinking' in a.model) else 1024)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tools = '\n'.join(f'- {k}: {v}' for k, v in TOOLS.items())
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sample = samples(work)

    # speed first (plain text, then one image without the JSON format), so a slow or stuck answer can be told apart
    _, t = chat(a.host, a.model, [{'role': 'user', 'content': 'Reply with one word: ready.'}], None, 200, think=think)
    notice('vlm speed', f'{a.model} think={a.think} text: {t}')
    img = png_b64(sample[0]['views'][0][1])
    q = [{'role': 'user', 'content': 'In two sentences, what is this image?', 'images': [img]}]
    text, t = chat(a.host, a.model, q, None, min(limit, 1024), a.budget, think=think)
    notice('vlm image', f'{a.model} {t} · {text[-1500:]}')

    failed = 0
    for s in sample:
        imgs = []
        for j, (_, v, ref, _) in enumerate(s['views']):
            imgs.append(png_b64(v))
            Image.fromarray(v).save(out / f'{s["name"]}.{j}.png')  # exactly what the model is shown
            Image.fromarray(ref).save(out / f'{s["name"]}.{j}.experts.png')
        shown = '; '.join(f'image {j + 1}: {lab}, {um:.2f} µm per shown pixel' for j, (lab, _, _, um) in
                          enumerate(s['views']))  # fmt: skip
        look_q = (
            f'{s.get("context", "")}\nYou are shown {len(imgs)} images of it ({shown}).\nDescribe it. Is it '
            'microscopy? Brightfield or fluorescence? Which stain or markers? Which tissue or sample? Which structures '
            'are visible? Roughly how many cells (here: neurons) are in the whole section?'
        )
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': look_q, 'images': imgs}]
        plan, t2 = {}, {}
        try:
            look, t1 = chat(a.host, a.model, msgs, LOOK, limit, a.budget, think=think)
            if '_raw' not in look:
                plan_q = (
                    f'Goal: {s["goal"]}\nTools available:\n{tools}\nPlan the analysis as steps using only these '
                    'tools. For each step give the tool, what it applies to (a short name), sizes in µm where relevant, '
                    'the marker for split_by_marker, and why. Say what the result will be, and the risks (what could '
                    'go wrong). Plan every part of the goal this image allows, even if other parts are not possible: list those '
                    'parts under not_possible with the reason, and set feasible to yes, partly or no. Only if nothing '
                    'of the goal fits this image, set feasible to no and give no steps.'
                )
                msgs += [{'role': 'assistant', 'content': json.dumps(look)}, {'role': 'user', 'content': plan_q}]
                plan, t2 = chat(a.host, a.model, msgs, PLAN, limit, a.budget, think=think)
        except Exception as e:  # noqa: BLE001 - report it and go on to the next image
            failed += 1
            notice(f'vlm {s["name"]}', f'{a.model} ERROR {type(e).__name__}: {e}')
            continue
        raw = look.get('_raw') or plan.get('_raw')
        problems = ['answer unparsable'] if raw else check(look, plan, s['expect'])
        failed += bool(problems)
        rec = {'model': a.model, 'name': s['name'], 'goal': s['goal'], 'look': look, 'plan': plan,
               'reference_count': s.get('reference_count'), 'timing': {'look': t1, 'plan': t2},
               'problems': problems}  # fmt: skip
        (out / f'{s["name"]}.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
        steps = ' → '.join(f'{st.get("tool")}({st.get("what")})' for st in plan.get('steps', []))
        notice(
            f'vlm {s["name"]}',
            f'{a.model} {"OK" if not problems else "PROBLEMS: " + "; ".join(problems)} · '
            f'look {t1.get("seconds")} s ({t1.get("answer_tokens")} tok, {t1.get("done_reason")}), '
            f'plan {t2.get("seconds")} s ({t2.get("answer_tokens")} tok, {t2.get("done_reason")}) · '
            f'{look.get("modality")}, stain "{look.get("stain")}", tissue "{look.get("tissue")}", '
            f'~{look.get("approx_object_count")} objects (experts: {s.get("reference_count")}) · feasible={plan.get("feasible")} · {steps or "no steps"} · '
            f'not possible: {"; ".join(plan.get("not_possible", [])) or "-"} · '
            f'result: {plan.get("result")} · risks: {"; ".join(plan.get("risks", []))}'
            + (f' · raw: {raw}' if raw else ''),
        )
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
