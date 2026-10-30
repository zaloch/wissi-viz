"""A small test of a vision-language model through Ollama: shown an image and a goal, does it describe the image
correctly and plan an analysis with wissi-viz's tools? Nothing is run or measured; this only reads the model's answers.

    ollama pull qwen3-vl:4b
    uv run python scripts/vlm_probe.py                 # the built-in sample images
    uv run python scripts/vlm_probe.py --model qwen3-vl:8b --out probe-out

Writes one JSON file per image (the answers, timings) to --out and prints a short summary.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
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
        'feasible': {'type': 'boolean'},
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
    'required': ['feasible', 'steps', 'result', 'risks'],
}

SYSTEM = (
    'You assist with microscopy image analysis. You look at images and plan analyses, but you never measure or count '
    'precisely yourself: deterministic tools do that. Answer only with the JSON requested. Be factual; if the image '
    'does not fit the goal, say so.'
)


def samples() -> list[dict]:
    """Built-in test images (scikit-image's sample data) with a goal each; the last is not microscopy at all."""
    from skimage import data

    mitosis = data.human_mitosis()  # fluorescent nuclei, fetched once by scikit-image
    mitosis = (np.clip(mitosis / np.percentile(mitosis, 99.5), 0, 1) * 255).astype(np.uint8)
    return [
        {
            'name': 'ihc-colon',
            'image': data.immunohistochemistry(),
            'goal': 'Count the DAB-positive (brown) cells and report the fraction of all cells that are positive.',
            'expect': {'modality': 'brightfield', 'stain_has': ['dab', 'hematoxylin', 'ihc', 'immuno']},
        },
        {
            'name': 'mitosis-nuclei',
            'image': np.stack([mitosis] * 3, -1),
            'goal': 'Count the nuclei and find the ones that are dividing (mitotic).',
            'expect': {'modality': 'fluorescence', 'stain_has': ['dapi', 'hoechst', 'nuclear', 'fluoresc']},
        },
        {
            'name': 'not-microscopy',
            'image': data.coffee(),
            'goal': 'Count the nuclei.',
            'expect': {'is_microscopy': False},
        },
    ]


def png_b64(rgb: np.ndarray, side: int = 512) -> str:
    im = Image.fromarray(rgb).convert('RGB')
    im.thumbnail((side, side))
    b = io.BytesIO()
    im.save(b, format='PNG')
    return base64.b64encode(b.getvalue()).decode()


def notice(title: str, text: str) -> None:
    print(f'{title}: {text}', flush=True)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::notice title={title}::{" ".join(text.split())[:3500]}', flush=True)


def chat(host: str, model: str, messages: list[dict], schema: dict | None, limit: int, budget: float = 600) -> tuple:
    """One streamed chat: (the answer, parsed if a schema was given; stats). Stops at ``limit`` tokens (thinking
    included) or after ``budget`` seconds; an unparsable answer comes back as {'_raw': its end}."""
    body = {
        'model': model,
        'messages': messages,
        'stream': True,
        'options': {'temperature': 0, 'num_ctx': 8192, 'num_predict': limit},
    }
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
    if expect.get('is_microscopy') is False and plan.get('feasible'):
        bad.append('planned an analysis of a non-microscopy image')
    if 'modality' in expect and look.get('modality') != expect['modality']:
        bad.append(f'modality={look.get("modality")}')
    stain = (look.get('stain') or '').lower()
    if 'stain_has' in expect and not any(w in stain for w in expect['stain_has']):
        bad.append(f'stain="{look.get("stain")}"')
    if expect.get('is_microscopy', True) and not plan.get('steps'):
        bad.append('no steps planned')
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', default=os.environ.get('WISSI_VLM_MODEL', 'qwen3-vl:4b'))
    p.add_argument('--host', default=os.environ.get('OLLAMA_HOST_URL', 'http://localhost:11434'))
    p.add_argument('--out', default='vlm-probe')
    p.add_argument('--limit', type=int, default=0, help='most tokens per answer (default 1024; 4096 if thinking)')
    a = p.parse_args(argv)
    limit = a.limit or (4096 if 'thinking' in a.model else 1024)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tools = '\n'.join(f'- {k}: {v}' for k, v in TOOLS.items())
    sample = samples()

    # speed first (plain text, then one image without the JSON format), so a slow or stuck answer can be told apart
    _, t = chat(a.host, a.model, [{'role': 'user', 'content': 'Reply with one word: ready.'}], None, 200)
    notice('vlm speed', f'{a.model} text: {t}')
    img = png_b64(sample[0]['image'])
    q = [{'role': 'user', 'content': 'In two sentences, what is this image?', 'images': [img]}]
    text, t = chat(a.host, a.model, q, None, min(limit, 1024))
    notice('vlm image', f'{a.model} {t} · {text[-1500:]}')

    failed = 0
    for s in sample:
        img = png_b64(s['image'])
        Image.open(io.BytesIO(base64.b64decode(img))).save(out / f'{s["name"]}.png')
        look_q = (
            'Describe this image. Is it microscopy? Brightfield or fluorescence? Which stain or markers? Which tissue '
            'or sample? Which structures are visible? Roughly how many cells or nuclei are visible?'
        )
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': look_q, 'images': [img]}]
        plan, t2 = {}, {}
        try:
            look, t1 = chat(a.host, a.model, msgs, LOOK, limit)
            if '_raw' not in look:
                plan_q = (
                    f'Goal: {s["goal"]}\nTools available:\n{tools}\nPlan the analysis as steps using only these '
                    'tools. For each step give the tool, what it applies to (a short name), sizes in µm where relevant, '
                    'the marker for split_by_marker, and why. Say what the result will be, and the risks (what could '
                    'go wrong). If the goal does not fit this image, set feasible to false and give no steps.'
                )
                msgs += [{'role': 'assistant', 'content': json.dumps(look)}, {'role': 'user', 'content': plan_q}]
                plan, t2 = chat(a.host, a.model, msgs, PLAN, limit)
        except Exception as e:  # noqa: BLE001 - report it and go on to the next image
            failed += 1
            notice(f'vlm {s["name"]}', f'{a.model} ERROR {type(e).__name__}: {e}')
            continue
        raw = look.get('_raw') or plan.get('_raw')
        problems = ['answer unparsable'] if raw else check(look, plan, s['expect'])
        failed += bool(problems)
        rec = {'model': a.model, 'name': s['name'], 'goal': s['goal'], 'look': look, 'plan': plan,
               'timing': {'look': t1, 'plan': t2}, 'problems': problems}  # fmt: skip
        (out / f'{s["name"]}.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
        steps = ' → '.join(f'{st.get("tool")}({st.get("what")})' for st in plan.get('steps', []))
        notice(
            f'vlm {s["name"]}',
            f'{a.model} {"OK" if not problems else "PROBLEMS: " + "; ".join(problems)} · '
            f'look {t1.get("seconds")} s ({t1.get("answer_tokens")} tok, {t1.get("done_reason")}), '
            f'plan {t2.get("seconds")} s ({t2.get("answer_tokens")} tok, {t2.get("done_reason")}) · '
            f'{look.get("modality")}, stain "{look.get("stain")}", tissue "{look.get("tissue")}", '
            f'~{look.get("approx_object_count")} objects · feasible={plan.get("feasible")} · {steps or "no steps"} · '
            f'result: {plan.get("result")} · risks: {"; ".join(plan.get("risks", []))}'
            + (f' · raw: {raw}' if raw else ''),
        )
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
