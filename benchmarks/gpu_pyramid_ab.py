"""Client-mode object pyramid: the current Web Worker build vs an experimental WebGPU compute build.

Both run in the same browser on the same /annotations/columns payload (exactly what the server sends in client mode).
Reported per size: median build time of each, the GPU build's phases, and a full parity check — every level, tile
and point: representatives, counts, classes, histograms, sizes, outline shapes, positions (in 1/65535 of a tile).

    python benchmarks/gpu_pyramid_ab.py                          # headless (no GPU here → SwiftShader, CPU-emulated)
    python benchmarks/gpu_pyramid_ab.py --headed                 # a real browser window on your desktop GPU
    python benchmarks/gpu_pyramid_ab.py --headed --channel chrome --cells 100k,1M
    python benchmarks/gpu_pyramid_ab.py --table cells.parquet    # your own object table (any read_objects format)

The adapter line tells you what ran the GPU build: architecture 'swiftshader' means CPU emulation, not your GPU.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import shutil
import socket
import statistics
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX_JS = HERE.parent / 'src' / 'wissi_viz' / 'static' / 'index.js'


def _n(s: str) -> int:
    s = s.strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith('k') else int(float(s[:-1]) * 1_000_000) if s.endswith('m') else int(s)


def synthetic_payload(n: int, W: int = 92_000, H: int = 75_000) -> bytes:
    """Tissue-like clustered cells, 12 classes (skewed, so the 'other' slot is used)."""
    import numpy as np

    from wissi_viz import AnnotationLayer
    rng = np.random.default_rng(n)
    k = 60
    cxs, cys = rng.uniform(0, W, k), rng.uniform(0, H, k)
    which = rng.integers(0, k, n)
    cx = np.clip(cxs[which] + rng.normal(0, 2500, n), 0, W - 1)
    cy = np.clip(cys[which] + rng.normal(0, 2500, n), 0, H - 1)
    r = rng.uniform(4, 10, n)
    cls = [f'c{int(c)}' for c in np.minimum(rng.geometric(0.25, n) - 1, 11)]
    L = AnnotationLayer.from_boxes(cx - r, cy - r, cx + r, cy + r, classes=cls, name='cells')
    return L.compiled().columns_payload(L)


def table_payload(path: str, mpp: float | None) -> bytes:
    from wissi_viz import read_objects
    L = read_objects(path, mpp=mpp, units='px' if mpp is None else 'auto').to_layer()
    return L.compiled().columns_payload(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells', default='100k,300k,1M', help='synthetic sizes (ignored with --table)')
    ap.add_argument('--table', help='use this object table instead of synthetic cells')
    ap.add_argument('--mpp', type=float, default=None, help='µm per px of the table coordinates (default: they are px)')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--headed', action='store_true', help='open a real browser window (uses your GPU)')
    ap.add_argument('--channel', default=None, help="e.g. 'chrome' to use your installed Chrome instead of Playwright's Chromium")
    ap.add_argument('--out', default=None, help='write the raw results as JSON')
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright
    work = Path(tempfile.mkdtemp(prefix='gpupyr-'))
    try:
        shutil.copy(HERE / 'gpu_pyramid' / 'gpu_pyramid.js', work)
        shutil.copy(HERE / 'gpu_pyramid' / 'ab.html', work)
        src = INDEX_JS.read_text()
        (work / 'worker.js').write_text(re.search(r'const PYR_WORKER_SRC = `(.*?)`;', src, re.S).group(1))
        files = []
        if a.table:
            (work / 'table.bin').write_bytes(table_payload(a.table, a.mpp))
            files.append(('table.bin', Path(a.table).name))
        else:
            for s in a.cells.split(','):
                n = _n(s)
                print(f'building {n:,} synthetic cells …', flush=True)
                (work / f'c{n}.bin').write_bytes(synthetic_payload(n))
                files.append((f'c{n}.bin', f'{n:,} cells'))

        with socket.socket() as so:
            so.bind(('127.0.0.1', 0))
            port = so.getsockname()[1]

        class Q(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def end_headers(self):
                self.send_header('Cache-Control', 'no-store')
                super().end_headers()
        httpd = http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Q, directory=str(work)))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        flags = ['--enable-unsafe-webgpu']
        if sys.platform.startswith('linux'):
            flags += ['--enable-features=Vulkan']
        if not a.headed:
            flags += ['--enable-unsafe-swiftshader']
        results = []
        with sync_playwright() as pw:
            b = pw.chromium.launch(headless=not a.headed, channel=a.channel, args=flags)
            pg = b.new_page()
            pg.on('console', lambda m: print('  console:', m.text[:400]) if m.type == 'error' else None)
            pg.goto(f'http://127.0.0.1:{port}/ab.html')
            pg.wait_for_function('() => window.ready')
            for fname, label in files:
                r = pg.evaluate('([f, n]) => ab(f, n)', [fname, a.repeats])
                r['label'] = label
                results.append(r)
                if len(results) == 1:
                    print('adapter:', r.get('adapter'))
                if not r['gpu'] or r.get('pyrG') is None:
                    print(f'{label}: GPU build unavailable (no WebGPU adapter, or an unsupported layer)')
                    continue
                med = lambda v: statistics.median(v)
                ph = {k: med([t[k] for t in r['gpuT']]) for k in r['gpuT'][0]}
                par = r.get('parity') or []
                exact = all(not L.get('levelMismatch') and not L['tileSetDiff'] and not L['tileNDiff'] and not L['missing']
                            and not L['field'] for L in par)
                pos_max = max((L.get('posMax', 0) for L in par), default=0)
                pos_n = sum(L.get('posDiff', 0) for L in par)
                pts = sum(L['points'][0] for L in par if 'points' in L)
                print(f"{label:>14}: worker {med(r['worker']):7.0f} ms | GPU {med(r['gpu']):7.0f} ms "
                      f"(compute {ph['gpu_compute']:.0f}, upload {ph['setup_upload_encode']:.0f}, readback {ph['readback']:.0f}, "
                      f"tiles {ph['tile_views']:.0f}; {ph['gpuMB']:.0f} MB on GPU) | "
                      f"parity: {'identical' if exact else 'MISMATCH'} structure/classes/counts/histograms, "
                      f"positions ≤ {pos_max} LSB ({pos_n / max(pts, 1):.2%} of points differ)", flush=True)
                if not exact:
                    for L in par:
                        print('   ', json.dumps(L))
            b.close()
        httpd.shutdown()
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
