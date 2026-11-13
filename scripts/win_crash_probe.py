"""Windows access-violation probe: a cell table read with Polars while the slide's tiles are being produced.

Each variant runs in its own process (a crash ends only that process), several times; the exit codes are counted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

LOOPS = 40


def make(d: Path) -> None:
    import tifffile

    rng = np.random.default_rng(11)
    W = H = 1600
    img = np.full((H, W, 3), 242, np.uint8)
    img[250:-250, 250:-250] = (215, 150, 195)
    rows = ['x,y,class,area_um2']
    for _ in range(500):
        x, y = rng.uniform(40, W - 40, 2)
        rows.append(f'{x:.1f},{y:.1f},{"Big" if rng.random() < 0.5 else "Small"},{rng.uniform(10, 90):.2f}')
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 'slide.svs', bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
        tw.write(img[::4, ::4].copy(), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None,
                 subfiletype=1)  # fmt: skip
    (d / 'cells.csv').write_text('\n'.join(rows), encoding='utf-8')


def child(variant: str, d: Path) -> None:
    if variant == 'polars-alone':  # Polars with busy numpy threads, no session
        import polars as pl

        stop = threading.Event()

        def busy():
            a = np.ones((1024, 1024, 3), np.uint8)
            while not stop.is_set():
                a.reshape(256, 4, 256, 4, 3).mean(axis=(1, 3))

        ts = [threading.Thread(target=busy, daemon=True) for _ in range(8)]
        for t in ts:
            t.start()
        for _ in range(LOOPS * 5):
            pl.read_csv(d / 'cells.csv', separator=',', infer_schema_length=10000)
        stop.set()
        return
    from wissi_viz import objects
    from wissi_viz.session import Session

    if variant == 'session-pyarrow':
        objects._engine = lambda engine: 'pyarrow'
    for i in range(LOOPS):
        s = Session(cache_dir=d / f'cache{i}', remember=False)
        files = [d / 'slide.svs'] if variant == 'session-no-csv' else [d / 'slide.svs', d / 'cells.csv']
        r = s.open(files, trusted=True)
        assert not r['errors'], r['errors']
        s.shutdown()


VARIANTS = {
    'session': {},
    'session-polars-1-thread': {'POLARS_MAX_THREADS': '1'},
    'session-pyarrow': {},
    'session-no-csv': {},
    'polars-alone': {},
}

if __name__ == '__main__':
    if len(sys.argv) > 2:
        child(sys.argv[1], Path(sys.argv[2]))
        sys.exit(0)
    runs = int(os.environ.get('PROBE_RUNS', '6'))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make(d)
        for name, env in VARIANTS.items():
            codes = []
            for _ in range(runs):
                p = subprocess.run([sys.executable, '-X', 'faulthandler', __file__, name, str(d)],
                                   env={**os.environ, **env}, capture_output=True, text=True)  # fmt: skip
                codes.append(p.returncode)
                if p.returncode:
                    tail = ' / '.join(ln.strip() for ln in p.stderr.splitlines() if 'File' in ln or 'fatal' in ln)
                    print(f'{name} exit {p.returncode}: {tail[:900]}', flush=True)
            bad = sum(c != 0 for c in codes)
            print(f'::notice title={name}::{bad} of {runs} processes crashed ({runs * LOOPS} opens) codes={codes}')
