"""Trial the bundled agent: small open models on Ollama drive wissi-viz's tools through a set of tasks.

    python scripts/agent_trial.py --model qwen3:4b [--base-url http://127.0.0.1:11434/v1] [--out trial.json]

A test session (two brightfield slides with a cell layer, a fluorescence slide, saved views) and ten tasks a person
might ask, each checked by what the model did (which tools, with which arguments) and what it answered. Writes a
JSON report and a Markdown table; prints a one-line summary (and a GitHub annotation in CI)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from czi_writer import write_czi  # noqa: E402

from wissi_viz.agent import llm, tools  # noqa: E402
from wissi_viz.agent.ui import UiBus  # noqa: E402
from wissi_viz.annotations import AnnotationLayer  # noqa: E402
from wissi_viz.session import Session  # noqa: E402


class Spy(UiBus):
    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, cmd):
        self.sent.append(cmd)
        return True


def build(d: Path) -> tuple[Session, dict]:
    rng = np.random.default_rng(0)
    for name in ('Liver_A.svs', 'Liver_B.svs'):
        W, H = 2000, 1500
        img = np.full((H, W, 3), 235, np.uint8)
        desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
        with tifffile.TiffWriter(d / name, bigtiff=True) as tw:
            tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    planes = [np.clip(rng.normal(300, 40, (800, 800)), 0, 4095).astype(np.uint16) for _ in range(3)]
    write_czi(d / 'Nerve_IF.czi', [{'c': c, 'x': 0, 'y': 0, 'w': 800, 'h': 800, 'data': p, 'm': c} for c, p in enumerate(planes)],
              ['DAPI', 'NeuN', 'GFAP'], mpp=0.325)  # fmt: skip
    s = Session(roots=[d], remember=False, cache_dir=d / 'cache')
    ids = s.open([d / 'Liver_A.svs', d / 'Liver_B.svs', d / 'Nerve_IF.czi'])['opened']
    counts = {}
    for sid, n_t in zip(ids[:2], (47, 21)):
        L = AnnotationLayer(name='cells', classes={'Tumour': '#ff0000', 'Immune': '#00ff00'})
        for i in range(n_t + 30):
            L.add_point(*rng.uniform(100, 1400, 2), cls='Tumour' if i < n_t else 'Immune')
        s.registry.add_annotations(sid, L, name='cells')
        counts[sid] = n_t
    s.add_view(ids[0], [200, 200, 800, 650], 'A core', {'Region': 'Core', 'Grade': 2})
    s.add_view(ids[1], [300, 300, 900, 750], 'B core', {'Region': 'Core', 'Grade': 3})
    s.add_view(ids[0], [1200, 900, 1800, 1350], 'A edge', {'Region': 'Edge', 'Grade': 2})
    return s, {'ids': ids, 'tumour': counts}


def called(steps, name, **want):
    for st in steps:
        if st['tool'] != name or (isinstance(st['result'], dict) and 'error' in st['result']):
            continue
        a = st['args']
        if all(str(v).lower() in json.dumps(a.get(k), default=str).lower() for k, v in want.items()):
            return True
    return False


def tasks(ctx):
    a, b, fl = ctx['ids']
    return [
        ('How many slides are open?', lambda st, r, ui, s: called(st, 'list_slides') and '3' in r),
        ('Which layers does Liver_A have?', lambda st, r, ui, s: called(st, 'list_layers') and 'cells' in r.lower()),
        ('How many Tumour cells are in the cells layer of Liver_A?',
         lambda st, r, ui, s: (called(st, 'layer_summary') or called(st, 'list_layers')) and str(ctx['tumour'][a]) in r),
        ('Show Liver_B in the window.', lambda st, r, ui, s: any(c['type'] == 'show' and b in c['slides'] for c in ui.sent)),
        ('Go to x 600, y 450 on Liver_A, 200 µm wide.',
         lambda st, r, ui, s: any(c['type'] == 'goto' and c['slide'] == a and abs((c['bounds'][2] - c['bounds'][0]) - 400) < 2
                                  and abs((c['bounds'][0] + c['bounds'][2]) / 2 - 600) < 2 for c in ui.sent)),
        ('Hide the cells layer on Liver_B.',
         lambda st, r, ui, s: any(c['type'] == 'layer' and c['slide'] == b and c['visible'] is False for c in ui.sent)),
        ('Which saved views are in the Core region? Give their names.',
         lambda st, r, ui, s: called(st, 'list_views') and 'a core' in r.lower() and 'b core' in r.lower() and 'a edge' not in r.lower()),
        ('Make a figure called Cores of the saved views whose Region is Core.',
         lambda st, r, ui, s: any(f['name'] == 'Cores' and len(s.figure_layout(f['recipe'], f['name'], open_slides=False)['members']) == 2
                                  for f in s.store.figures())),
        ('Split the channels of Nerve_IF into separate panes.',
         lambda st, r, ui, s: any(c['type'] == 'channels' and c['slide'] == fl and c.get('split') is True for c in ui.sent)),
        ('Save the region 0, 0, 500, 400 of Liver_B as a view named Test.',
         lambda st, r, ui, s: any(v['name'] == 'Test' and v['bounds'] == [0, 0, 500, 400] for v in s.store.views())),
    ]  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:11434/v1')
    ap.add_argument('--max-steps', type=int, default=6)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    cfg = {**llm.DEFAULT, 'base_url': a.base_url, 'model': a.model, 'max_steps': a.max_steps, 'temperature': 0.0}
    rows = []
    with tempfile.TemporaryDirectory() as td:
        s, ctx = build(Path(td))
        try:
            for i, (prompt, ok) in enumerate(tasks(ctx), 1):
                ui = Spy()
                h = tools.Host(s, ui, exports=Path(td) / 'out')
                t0 = time.perf_counter()
                try:
                    out = llm.chat(h, [{'role': 'user', 'content': prompt}], cfg)
                    err = None
                except Exception as e:  # noqa: BLE001  (a model server error: the task fails, the trial goes on)
                    out, err = {'reply': '', 'steps': []}, f'{type(e).__name__}: {e}'
                sec = time.perf_counter() - t0
                try:
                    good = bool(ok(out['steps'], out['reply'] or '', ui, s)) and err is None
                except Exception as e:  # noqa: BLE001
                    good, err = False, f'check: {e}'
                rows.append({'task': i, 'prompt': prompt, 'ok': good, 'seconds': round(sec, 1),
                             'tools': [f"{st['tool']}({json.dumps(st['args'])[:80]})" for st in out['steps']],
                             'errors': [st['result']['error'] for st in out['steps'] if isinstance(st['result'], dict) and 'error' in st['result']],
                             'reply': (out['reply'] or '')[:300], 'error': err})  # fmt: skip
                print(f"{'✓' if good else '✗'} {i:2d} {sec:6.1f}s {prompt} → {[t.split('(')[0] for t in rows[-1]['tools']]}", flush=True)
        finally:
            s.shutdown()
    n = sum(r['ok'] for r in rows)
    summary = {'model': a.model, 'passed': n, 'tasks': len(rows), 'seconds': round(sum(r['seconds'] for r in rows), 1), 'rows': rows}
    line = f'{a.model}: {n}/{len(rows)} tasks, {summary["seconds"]:.0f} s'
    print(line)
    print(f'::notice::{line} · ' + ' '.join(f"{r['task']}{'✓' if r['ok'] else '✗'}" for r in rows))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
        md = [f'### {a.model}: {n}/{len(rows)}', '', '| # | task | ok | s | tools | reply |', '|---|---|---|---|---|---|']
        md += [f"| {r['task']} | {r['prompt']} | {'✓' if r['ok'] else '✗'} | {r['seconds']} | {'; '.join(r['tools'])[:160]} | "
               f"{(r['error'] or r['reply']).replace('|', '/').replace(chr(10), ' ')[:160]} |" for r in rows]  # fmt: skip
        Path(a.out).with_suffix('.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
