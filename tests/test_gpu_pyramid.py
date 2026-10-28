"""The experimental WebGPU pyramid build must reproduce the worker's client-mode pyramid: every level, tile and point,
representatives, counts, classes, histograms, sizes and outline shapes exactly; positions within 1/65535 of a tile.
Skipped when Chromium or a WebGPU adapter (SwiftShader counts) isn't available."""

from __future__ import annotations

import functools
import http.server
import importlib.util
import re
import shutil
import socket
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip('playwright')
ROOT = Path(__file__).resolve().parents[1]


def _ab():
    spec = importlib.util.spec_from_file_location('gpu_pyramid_ab', ROOT / 'benchmarks' / 'gpu_pyramid_ab.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gpu_pyramid_matches_worker(tmp_path):
    m = _ab()
    shutil.copy(ROOT / 'benchmarks' / 'gpu_pyramid' / 'gpu_pyramid.js', tmp_path)
    shutil.copy(ROOT / 'benchmarks' / 'gpu_pyramid' / 'ab.html', tmp_path)
    src = (ROOT / 'src' / 'wissi_viz' / 'static' / 'index.js').read_text(encoding='utf-8')
    (tmp_path / 'worker.js').write_text(
        re.search(r'const PYR_WORKER_SRC = `(.*?)`;', src, re.S).group(1), encoding='utf-8'
    )
    (tmp_path / 'c.bin').write_bytes(m.synthetic_payload(60_000))
    with socket.socket() as so:
        so.bind(('127.0.0.1', 0))
        port = so.getsockname()[1]

    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Q, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            try:
                args = ['--enable-unsafe-swiftshader', '--enable-unsafe-webgpu']
                if sys.platform.startswith('linux'):
                    args.append('--enable-features=Vulkan')  # headless Linux: SwiftShader WebGPU via Vulkan
                b = pw.chromium.launch(args=args)
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page()
            pg.goto(f'http://127.0.0.1:{port}/ab.html')
            pg.wait_for_function('() => window.ready')
            if not pg.evaluate('async () => !!(navigator.gpu && await navigator.gpu.requestAdapter())'):
                pytest.skip('no WebGPU adapter')
            r = pg.evaluate('() => ab("c.bin", 1)')
            b.close()
    finally:
        httpd.shutdown()
    assert r['pyrG'] is not None
    assert r['pyrG']['points'] == r['pyrW']['points'] and r['pyrG']['tiles'] == r['pyrW']['tiles']
    assert r['pyrG']['slots'] == r['pyrW']['slots'] and r['pyrG']['other_class'] == r['pyrW']['other_class']
    pts = 0
    for L in r['parity']:
        assert not L.get('levelMismatch') and L['tileSetDiff'] == 0 and L['tileNDiff'] == 0, L
        assert L['missing'] == 0, L  # same representative object for every dot
        assert L['field'] == {}, L  # counts, classes, histograms, sizes, shapes identical
        assert L['posMax'] <= 1, L  # f32 vs f64 rounding: ≤ 1/65535 of the tile
        pts += L['points'][0]
    assert sum(L['posDiff'] for L in r['parity']) < 0.01 * pts
