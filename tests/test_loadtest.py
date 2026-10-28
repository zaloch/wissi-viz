"""scripts/wissi_loadtest.py: the scalability model recovers known parameters, and a tiny live run against a
real wissi-viz server counts statuses, respects the per-user request cap and reports view latency."""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import wissi_loadtest as lt


def test_usl_fit_and_little_extrapolation_recover_a_known_system():
    """Synthetic closed-loop users (think T, then a burst of b tiles taking view time V; X from a known USL):
    the fit must recover σ, κ, and the extrapolated view time at 150 users must match the closed-form answer."""
    lam, sig, kap, b, T = 20.0, 0.03, 0.0004, 20.0, 0.9
    N = np.array([10, 25, 50, 75, 100])
    X = lam * N / (1 + sig * (N - 1) + kap * N * (N - 1))
    m = lt.usl_fit(N, X)
    assert abs(m['sigma'] - sig) < 0.003 and abs(m['kappa'] - kap) < 4e-5 and m['r2'] > 0.999
    V = N * b / X - T
    st = [
        {
            'users': int(n),
            'requests': 1000,
            'throughput_rps': float(x),
            'latency_mean_ms': 100.0 * v,
            'latency_ms': {'p95': 200.0 * v},
            'view_ms': {'p50': 1000 * v, 'p95': 2000 * v},
            'view_mean_ms': 1000 * v,
            'error_rate_pct': 0.0,
        }
        for n, x, v in zip(N, X, V)
    ]
    mdl = lt.model(st, [60, 150])
    assert abs(mdl['tiles_per_move'] - b) < 0.5 and abs(mdl['think_s'] - T) < 0.05
    p = {q['users']: q for q in mdl['predictions']}
    assert p[60]['mode'] == 'interpolated' and p[150]['mode'].startswith('EXTRAPOLATED')
    x150 = lam * 150 / (1 + sig * 149 + kap * 150 * 149)
    v150 = 150 * b / x150 - T
    assert abs(p[150]['view_p95_ms'] / 2000 - v150) / v150 < 0.05, (p[150], v150)


def test_pyramid_and_visible_tiles():
    s = lt._template_slide('http://x/{image}/{level}/{col}/{row}', 'a', '10000x8000:256:2')
    assert s.levels[0][:2] == (10000, 8000) and max(s.levels[-1][:2]) <= 256
    assert lt.level_for(s, 1.0) == 0 and lt.level_for(s, 0.25) == 2
    t = lt.visible_tiles(s, 0, 5000, 4000, 1.0, 512, 512)
    assert len(t) == 9 and all(L == 0 for L, _, _ in t)  # 512 px window, 256 tiles, unaligned → 3×3


def test_live_run_counts_statuses_and_caps_requests(tmp_path):
    pytest.importorskip('httpx')
    import tifffile
    import uvicorn

    from wissi_viz import SlideRegistry
    from wissi_viz.server import create_app

    img = (np.random.default_rng(0).random((3000, 4000, 3)) * 255).astype(np.uint8)
    with tifffile.TiffWriter(tmp_path / 's.svs', bigtiff=True) as tw:
        tw.write(
            img,
            description='Aperio Image Library v12 \r\n4000x3000 (256x256) JPEG/RGB Q=80|MPP = 0.5',
            tile=(256, 256),
            compression='jpeg',
            photometric='rgb',
            metadata=None,
        )
    reg = SlideRegistry(warm=False)
    reg.add(str(tmp_path / 's.svs'), slide_id='s')
    with socket.socket() as so:
        so.bind(('127.0.0.1', 0))
        port = so.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    try:
        out = tmp_path / 'load.json'
        stages, mdl = lt.main(
            [
                f'http://127.0.0.1:{port}/wissi-viz',
                '--users',
                '1,2,3',
                '--requests',
                '30',
                '--panes',
                '2',
                '--think',
                '0,0.05',
                '--ramp',
                '0.2',
                '--stats-url',
                f'http://127.0.0.1:{port}/wissi-viz/stats',
                '--out',
                str(out),
            ]
        )
        for s, n in zip(stages, (1, 2, 3)):
            assert s['users'] == n and 0 < s['requests'] <= 30 * n
            assert s['status'].get('200', 0) == s['requests'] and s['error_rate_pct'] == 0
            assert s['view_ms']['p50'] is not None and 'server' in s
        assert out.exists() and out.with_suffix('.md').exists() and mdl['usl'] is not None
        assert lt.MAX_REQUESTS_PER_USER == 999
    finally:
        srv.should_exit = True
