"""NiceGUI integration, end to end: examples/nicegui_inspector.py in a real NiceGUI server and browser.
The embedded app (WissiApp) loads its slide and layers; the custom panel's layer switch, summary, analysis
chart, hover and right-click callbacks all reach Python exactly once, without errors."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip('nicegui')
pytest.importorskip('playwright')
pytest.importorskip('scipy')
ROOT = Path(__file__).resolve().parents[1]


def _data(d: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(5)
    W = H = 2400
    img = np.full((H, W, 3), 245, np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    img[(xx - 1200) ** 2 + (yy - 1200) ** 2 < 1000**2] = (214, 150, 190)
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 'slide.svs', bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    xy = np.vstack([rng.normal(1100, 150, (500, 2)), rng.uniform(400, 2000, (700, 2))])
    cls = ['Tumor'] * 500 + list(rng.choice(['Immune', 'Stroma'], 700))
    rows = ['x,y,class,area_um2'] + [
        f'{x:.1f},{y:.1f},{c},{a:.1f}' for (x, y), c, a in zip(xy, cls, rng.normal(50, 9, 1200))
    ]
    (d / 'cells.csv').write_text('\n'.join(rows), encoding='utf-8')
    roi = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'properties': {'classification': {'name': 'Tumor'}},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[900, 900], [1300, 900], [1300, 1300], [900, 1300], [900, 900]]],
                },
            }
        ],
    }
    (d / 'regions.geojson').write_text(json.dumps(roi), encoding='utf-8')
    return d / 'slide.svs', d / 'cells.csv', d / 'regions.geojson'


def test_nicegui_inspector_example(tmp_path):
    from playwright.sync_api import sync_playwright

    slide, cells, regions = _data(tmp_path)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    log = tmp_path / 'server.log'
    with open(log, 'w') as fh:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / 'examples' / 'nicegui_inspector.py'),
                str(slide),
                str(cells),
                str(regions),
                '--port',
                str(port),
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=tmp_path,
            # NiceGUI switches to its own test mode when it sees pytest's variables
            env={k: v for k, v in os.environ.items() if not k.startswith('PYTEST')},
        )
    url = f'http://127.0.0.1:{port}'
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(url + '/', timeout=1)
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail('the example exited:\n' + log.read_text()[-3000:])
                time.sleep(0.25)
        errors = []
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1440, 'height': 900})
            pg.on('pageerror', lambda e: errors.append(str(e)))
            # the whole app in an iframe, sharing the registry
            pg.goto(url + '/')
            app = pg.frame_locator('iframe')
            app.locator('#l-list .lcard[data-layer="cells"]').wait_for(timeout=60_000)
            assert app.locator('#l-list .lcard[data-layer="regions"]').count() == 1
            # the custom panel
            pg.goto(url + '/custom')
            pg.wait_for_function(
                '() => { const g = document.querySelector("wissi-viz"); return g && g.panes && g.panes.every(p => p.slide && p.slide.info); }',
                timeout=60_000,
            )
            pg.click('text=Summarise')
            pg.wait_for_selector('td:has-text("Tumor")', timeout=60_000)
            pg.click('#run-density')
            pg.wait_for_selector('text=hotspots', timeout=120_000)
            assert pg.locator('.nicegui-echart').count() == 1
            pg.locator('.q-checkbox:has-text("regions")').click()
            pg.wait_for_function(
                '() => { const p = document.querySelector("wissi-viz").panes[0]; return !p.layerVisible(p.slide.layers.get("regions")); }'
            )
            box = pg.evaluate("""() => { const g = document.querySelector("wissi-viz"), p = g.panes[0], r = g.getBoundingClientRect();
                return [r.left + p.rect.x + p.rect.w / 2, r.top + p.rect.y + p.rect.h / 2]; }""")
            pg.mouse.click(box[0], box[1], button='right')
            pg.wait_for_selector('text=right-click at', timeout=10_000)
            b.close()
        assert not errors, errors
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    text = log.read_text()
    # every Python callback got the event's data; Windows' asyncio may log a closed pipe at shutdown (not ours)
    ours = [i for i in range(len(text)) if text.startswith('Traceback', i)
            and '_ProactorBasePipeTransport._call_connection_lost' not in text[max(0, i - 300) : i]]  # fmt: skip
    assert not ours, text[-3000:]
