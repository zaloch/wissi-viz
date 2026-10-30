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


def chat(host: str, model: str, messages: list[dict], schema: dict, timeout: float = 900) -> tuple[dict, dict]:
    body = {
        'model': model,
        'messages': messages,
        'format': schema,
        'stream': False,
        'options': {'temperature': 0, 'num_ctx': 8192},
    }
    req = urllib.request.Request(
        f'{host}/api/chat', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    msg = out.get('message', {})
    stats = {
        'seconds': round(out.get('total_duration', 0) / 1e9, 1),
        'prompt_tokens': out.get('prompt_eval_count'),
        'answer_tokens': out.get('eval_count'),
    }
    if msg.get('thinking'):
        stats['thinking_chars'] = len(msg['thinking'])
    return json.loads(msg.get('content') or '{}'), stats


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
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tools = '\n'.join(f'- {k}: {v}' for k, v in TOOLS.items())
    failed = 0
    for s in samples():
        img = png_b64(s['image'])
        Image.open(io.BytesIO(base64.b64decode(img))).save(out / f'{s["name"]}.png')
        look_q = (
            'Describe this image. Is it microscopy? Brightfield or fluorescence? Which stain or markers? Which tissue '
            'or sample? Which structures are visible? Roughly how many cells or nuclei are visible?'
        )
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': look_q, 'images': [img]}]
        look, t1 = chat(a.host, a.model, msgs, LOOK)
        plan_q = (
            f'Goal: {s["goal"]}\nTools available:\n{tools}\nPlan the analysis as steps using only these tools. For '
            'each step give the tool, what it applies to (a short name), sizes in µm where relevant, the marker for '
            'split_by_marker, and why. Say what the result will be, and the risks (what could go wrong). If the goal '
            'does not fit this image, set feasible to false and give no steps.'
        )
        msgs += [
            {'role': 'assistant', 'content': json.dumps(look)},
            {'role': 'user', 'content': plan_q},
        ]
        plan, t2 = chat(a.host, a.model, msgs, PLAN)
        problems = check(look, plan, s['expect'])
        failed += bool(problems)
        rec = {'model': a.model, 'name': s['name'], 'goal': s['goal'], 'look': look, 'plan': plan,
               'timing': {'look': t1, 'plan': t2}, 'problems': problems}  # fmt: skip
        (out / f'{s["name"]}.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
        steps = ' → '.join(f'{st.get("tool")}({st.get("what")})' for st in plan.get('steps', []))
        line = (
            f'{s["name"]} [{a.model}] {"OK" if not problems else "PROBLEMS: " + "; ".join(problems)} · '
            f'look {t1["seconds"]} s, plan {t2["seconds"]} s · {look.get("modality")}, stain "{look.get("stain")}", '
            f'tissue "{look.get("tissue")}", ~{look.get("approx_object_count")} objects · '
            f'feasible={plan.get("feasible")} · {steps or "no steps"} · result: {plan.get("result")} · '
            f'risks: {"; ".join(plan.get("risks", []))}'
        )
        print(line, flush=True)
        if os.environ.get('GITHUB_ACTIONS'):
            print(f'::notice title=vlm {s["name"]}::{line[:3500]}', flush=True)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
