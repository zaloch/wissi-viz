"""A small test of Nimble (Bespoke Labs, Apache-2.0, a decision model fine-tuned from Qwen3.5-9B) through Ollama: given
what a vision model said about an image and the goal, does it make the planning decisions an analyst would, and,
given a result that is off, the right correction? Each answer comes with the probability of every option.

    ollama pull nimble
    uv run python scripts/nimble_probe.py --looks vlm-probe-out/vlm-probe-2      # Qwen3-VL's descriptions (JSON)

Nothing is run or measured; this only reads Nimble's decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ---- the planning decisions: each maps to a wissi-viz parameter
PLAN = {
    'tool': {
        'type': 'choice',
        'instructions': 'Which wissi-viz tool should outline the objects the goal is about?',
        'criteria': {
            'find_by_text': 'a text-prompted detector outlines every object matching a short name (e.g. "neuron"), '
            'within a size range; good for large distinctive objects such as neuron cell bodies or glands',
            'find_cells': 'a nucleus / cell model outlines every nucleus in the image; good for counting nuclei',
            'segment_everything': 'outline every object of a given size, then keep those similar to a few examples',
            'find_areas': 'map tissue regions (e.g. tumour vs stroma) from brushed examples; not single objects',
        },
    },
    'size': {
        'type': 'choice',
        'instructions': 'What is the diameter range of the objects the goal is about?',
        'criteria': {
            'under_10um': 'smaller than 10 µm (e.g. small nuclei, glial nuclei)',
            '10_20um': '10 to 20 µm (e.g. most cell nuclei, small cells)',
            '20_60um': '20 to 60 µm (e.g. sensory neuron cell bodies)',
            '60_200um': '60 to 200 µm (e.g. glands, vessels, nerve fascicles)',
            'over_200um': 'larger than 200 µm (tissue regions)',
        },
    },
    'exclude_neighbours': {
        'type': 'noul',
        'instructions': 'Must similar-looking cells next to the target objects be explicitly left out (for example '
        'satellite glia or Schwann cells around neurons)?',
        'criteria': {'false': 'no need to exclude anything', 'true': 'yes, exclude the neighbouring cells'},
    },
    'needs_missing_marker': {
        'type': 'noul',
        'instructions': 'Does the goal need a stain or marker that the image does not have?',
        'criteria': {
            'false': 'everything the goal needs is in the image',
            'true': 'part of the goal needs a marker or stain that is not in the image',
        },  # fmt: skip
    },
    'feasible': {
        'type': 'choice',
        'instructions': 'How much of the goal can be done with this image?',
        'criteria': {'all': 'all of it', 'part': 'part of it; the rest needs other data', 'none': 'none of it'},
    },
}

# ---- after a run: what to change (the adjust-and-re-run loop)
CHECK = {
    'next': {
        'type': 'choice',
        'instructions': 'Given the goal and the result of the last run, what should happen next?',
        'criteria': {
            'accept': 'the result fits the goal and the expected objects; keep it',
            'raise_min_size': 'many objects found are smaller than the targets; raise the smallest size',
            'lower_max_size': 'objects found are merged or larger than the targets; lower the largest size',
            'lower_strictness': 'many targets are missed; keep fainter or less certain matches',
            'raise_strictness': 'many false matches; keep only the most certain ones',
            'other_tool': 'the tool finds the wrong kind of object; use another tool',
        },
    }
}

GOALS = {
    1: ('Outline every neuron cell body and count them, leaving out the satellite glia and Schwann cells around them.',
        {'tool': {'find_by_text', 'segment_everything'}, 'size': {'20_60um'}, 'exclude_neighbours': True,
         'needs_missing_marker': False, 'feasible': {'all'}}),
    2: ('Measure the size of each neuron and report how many are large (more than 40 µm across).',
        {'tool': {'find_by_text', 'segment_everything'}, 'size': {'20_60um'}, 'needs_missing_marker': False,
         'feasible': {'all'}}),
    3: ('Count the neurons and report what fraction of them are CGRP-positive nociceptors.',
        {'tool': {'find_by_text', 'segment_everything'}, 'size': {'20_60um'}, 'needs_missing_marker': True,
         'feasible': {'part'}}),
}  # fmt: skip
CONTEXT = ('Sample: a section of rat dorsal root ganglion. One fluorescence channel: Neurofilament (an antibody '
           'marking neurons). Pixel size about 1 µm.')  # fmt: skip
FALLBACK_LOOK = ('Fluorescence microscopy of a tissue section: a large oval ganglion packed with round, bright cell '
                 'bodies 20-50 µm across, with smaller dots between them and a few nerve bundles at the edge.')  # fmt: skip

RESULTS = [  # (what the last run gave, the right next step)
    ('Goal: count neuron cell bodies (20-60 µm). Found 1,240 objects, median diameter 9 µm; the overlay shows small '
     'dots between the large round cells outlined, and most large round cells are not outlined.', 'raise_min_size'),
    ('Goal: count neuron cell bodies (20-60 µm). Found 41 objects, median diameter 34 µm; the overlay shows only the '
     'brightest large round cells outlined and many similar but fainter cells left out.', 'lower_strictness'),
    ('Goal: count neuron cell bodies (20-60 µm). Found 176 objects, median diameter 31 µm; the overlay shows nearly '
     'every large round cell outlined once, nothing between them.', 'accept'),
    ('Goal: count neuron cell bodies (20-60 µm). Found 22 objects, median diameter 140 µm; each outline covers a '
     'cluster of several round cells.', 'lower_max_size'),
]  # fmt: skip


def notice(title: str, text: str) -> None:
    print(f'{title}: {text}', flush=True)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::notice title={title}::{" ".join(text.split())[:3500]}', flush=True)


def decide(host: str, model: str, state: str, questions: dict, timeout: float = 1800) -> tuple[dict, float]:
    body = {'model': model, 'state': state, 'questions': questions}
    req = urllib.request.Request(
        f'{host}/v1/systemone', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out.get('answers', out), round(time.time() - t0, 1)


def value(a) -> object:
    """An answer's value: the chosen option, or for yes / no questions True / False (probability ≥ 0.5)."""
    if isinstance(a, dict):
        if 'choice' in a:
            return a['choice']
        if 'noul' in a:
            v = a['noul']
            return v >= 0.5 if isinstance(v, int | float) else bool(v)
    return a


def shown(a) -> str:
    if isinstance(a, dict) and 'probabilities' in a:
        ps = sorted(a['probabilities'].items(), key=lambda kv: -kv[1])[:3]
        return f'{a.get("choice")} (' + ', '.join(f'{k} {v:.2f}' for k, v in ps) + ')'
    if isinstance(a, dict) and 'noul' in a:
        return f'{"yes" if value(a) else "no"} ({a["noul"]:.2f})'
    return json.dumps(a)[:200]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', default=os.environ.get('WISSI_DECIDER', 'nimble'))
    p.add_argument('--host', default=os.environ.get('OLLAMA_HOST_URL', 'http://localhost:11434'))
    p.add_argument('--looks', default='', help="a folder of the vision probe's JSON answers (its descriptions)")
    p.add_argument('--out', default='nimble-probe')
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    looks = {}
    for f in sorted(Path(a.looks).glob('*.json')) if a.looks else []:
        r = json.loads(f.read_text(encoding='utf-8'))
        k = int(r['name'].rsplit('goal', 1)[-1])
        lk = r.get('look') or {}
        if '_raw' not in lk and lk:
            looks[k] = (f'{r["model"]} described it: {lk.get("modality")}, stain {lk.get("stain")}, tissue '
                        f'{lk.get("tissue")}; visible: {", ".join(lk.get("structures") or [])}; about '
                        f'{lk.get("approx_object_count")} cells. {lk.get("notes") or ""}')  # fmt: skip
    bad = 0
    rec = {'model': a.model, 'plans': {}, 'checks': []}
    for k, (goal, expect) in GOALS.items():
        look = looks.get(k) or f'(no vision model description; a fixed one) {FALLBACK_LOOK}'
        state = f'{CONTEXT}\nWhat the image shows: {look}\nGoal: {goal}'
        ans, sec = decide(a.host, a.model, state, PLAN)
        wrong = []
        for q, want in expect.items():
            v = value(ans.get(q))
            if (isinstance(want, set) and v not in want) or (isinstance(want, bool) and v is not want):
                wrong.append(q)
        bad += bool(wrong)
        rec['plans'][k] = {'state': state, 'answers': ans, 'seconds': sec, 'wrong': wrong}
        notice(f'nimble plan {k}', f'{"OK" if not wrong else "WRONG: " + ", ".join(wrong)} · {sec} s · '
               + ' · '.join(f'{q}: {shown(ans.get(q))}' for q in PLAN) + f' · from: {look[:300]}')  # fmt: skip
    for j, (result, right) in enumerate(RESULTS, 1):
        ans, sec = decide(a.host, a.model, result, CHECK)
        got = value(ans.get('next'))
        bad += got != right
        rec['checks'].append({'result': result, 'answers': ans, 'right': right, 'seconds': sec})
        notice(f'nimble check {j}', f'{"OK" if got == right else "WRONG"} (right: {right}) · {sec} s · '
               f'{shown(ans.get("next"))} · {result[:160]}')  # fmt: skip
    (out / 'nimble.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
