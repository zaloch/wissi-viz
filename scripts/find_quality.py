"""How well find all works on a real H&E slide: outlines drawn on the slide, as pictures, and counts.

    python scripts/find_quality.py --out dist/quality

Downloads CMU-1-Small-Region.svs (OpenSlide test data, H&E, 0.5 µm/px), then in its central 1024 × 1024 px runs
segment everything (no examples), find by text ("nucleus") and find by text with expand (the best matches as
examples for segment everything). Writes ``<run>.jpg`` (the region with the outlines) and ``quality.json`` (counts,
seconds). .github/workflows/find-quality.yml publishes them as the GitHub release ``find-quality``, to look at.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

URL = 'https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs'


def main() -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from wissi_viz import annio, findall
    from wissi_viz.sam import _rgb
    from wissi_viz.session import Session

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='dist/quality')
    ap.add_argument('--slide', default=None, help='a local slide instead of CMU-1-Small-Region')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slide = Path(a.slide) if a.slide else out / 'CMU-1-Small-Region.svs'
    if not slide.exists():
        urllib.request.urlretrieve(URL, slide)
    s = Session(cache_dir=out / 'cache', remember=False)
    sid = s.open([slide], trusted=True)['opened'][0]
    reg = s.registry
    src = findall._source(reg, sid)
    W, H = src.info.width, src.info.height
    x0, y0 = max(0, W // 2 - 512), max(0, H // 2 - 512)
    rect = [x0, y0, min(W, x0 + 1024), min(H, y0 + 1024)]
    scope = {'kind': 'view', 'rect': rect}
    base = _rgb(src, 0, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
    Image.fromarray(base).save(out / 'region.jpg', quality=90)
    runs = {
        'segment-everything': lambda: findall.segment_everything(reg, sid, scope=scope, object_um=8),
        'text-nucleus': lambda: findall.find_by_text(reg, sid, prompt='nucleus', scope=scope, object_um=8),
        'text-nucleus-expand': lambda: findall.find_by_text(
            reg, sid, prompt='nucleus', scope=scope, object_um=8, expand=True
        ),  # fmt: skip
    }
    report = {'slide': slide.name, 'mpp': list(src.info.mpp or ()), 'region': rect, 'runs': {}}
    for name, run in runs.items():
        t0 = time.perf_counter()
        try:
            r = run()
        except ValueError as e:
            report['runs'][name] = {'error': str(e)}
            print(name, 'error', e, flush=True)
            continue
        G = annio.from_layer(reg.layer(sid, r['layer']))
        im = Image.fromarray(base.copy())
        d = ImageDraw.Draw(im)
        dia = []
        for i in range(len(G)):
            ring = np.asarray(G.rings(i)[0]) - [rect[0], rect[1]]
            d.line([tuple(p) for p in ring] + [tuple(ring[0])], fill=(0, 255, 255), width=2)
            dia.append(float(np.sqrt(np.ptp(ring[:, 0]) * np.ptp(ring[:, 1]))))
        im.save(out / f'{name}.jpg', quality=88)
        report['runs'][name] = {
            'count': r['count'],
            'of': r['of'],
            'seconds': round(time.perf_counter() - t0, 1),
            'median_size_um': round(float(np.median(dia)) * float((src.info.mpp or (1,))[0]), 1) if dia else None,
            'note': r.get('note'),
            'kpis': r.get('kpis'),
        }
        print(name, json.dumps(report['runs'][name]), flush=True)
    s.shutdown()
    (out / 'quality.json').write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
