"""scripts/wissi_probe.py: smoke + direct sweep + viewer sessions stay within the API-call budget (checked against the
vendor's own counter), produce a report and settings, and send nothing to a non-local service without --yes.
Also the configurable HTTP connector."""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip('uvicorn')
httpx = pytest.importorskip('httpx')
ROOT = Path(__file__).resolve().parents[1]


def _probe():
    spec = importlib.util.spec_from_file_location('wissi_probe', ROOT / 'scripts' / 'wissi_probe.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def slides(tmp_path_factory):
    m = _probe()
    d = tmp_path_factory.mktemp('probe')
    return [m.make_svs(d / f'probe_{k}_3072x2304.svs', 3072, 2304, seed=k) for k in (1, 2)], d


def test_mock_api_stays_within_budget_and_reports(slides, tmp_path):
    m = _probe()
    files, d = slides
    r = m.main(
        [
            'mock',
            '--workdir',
            str(d),
            '--local-size',
            '3072x2304',
            '--panes',
            '1,4',
            '--conn',
            '5',
            '--max-calls',
            '160',
            '--seconds',
            '10',
            '--mock-errors',
            '0',
            '--out',
            str(tmp_path / 'out'),
        ]
    )
    assert r['vendor']['requests'] <= 160  # the vendor's own count, not ours
    assert r['budget'].used == r['vendor']['requests']  # every call went through the guard
    assert all(x['status'] in ('PASS', 'INFO') for x in r['smoke']), r['smoke']
    assert r['direct']['c5']['status'].get('ok')
    done = [x for x in r['runs'] if 'cold' in x]
    assert done and all(x['cold']['error_pct'] == 0 for x in done)
    assert all(x['warm']['upstream']['calls'] == 0 for x in done)  # warm reruns: cache only
    assert all(x['cold']['upstream']['calls'] <= x['cold']['tile_requests'] for x in done)  # ≤ 1 call per tile
    for f in ('report.md', 'results.json', 'recommended_settings.py'):
        assert (tmp_path / 'out' / f).exists()
    assert 'max_concurrency' in r['rec']['settings']
    compile((tmp_path / 'out' / 'recommended_settings.py').read_text(encoding='utf-8'), 'settings', 'exec')


def test_local_slides(slides, tmp_path):
    m = _probe()
    files, _ = slides
    r = m.main(
        [
            f'local:{files[0]},{files[1]}',
            '--panes',
            '1,4',
            '--conn',
            '5',
            '--requests',
            '60',
            '--seconds',
            '8',
            '--out',
            str(tmp_path / 'out'),
        ]
    )
    assert all(x['status'] == 'PASS' for x in r['smoke'])
    assert r['direct']['c5']['tiles_per_s'] > 0
    brief = [(x.get('panes'), x.get('conn'), x['cold']['error_pct'], x['warm']['view_p95_ms']) for x in r['runs']]
    assert all(x['cold']['error_pct'] < 5 for x in r['runs']), brief  # a loaded CI runner may time out a few
    # on a slow CI machine a run can end before any view settles; most must settle
    assert sum(x['warm']['view_p95_ms'] is not None for x in r['runs']) >= max(1, len(r['runs']) - 1), brief


def test_no_yes_sends_nothing(tmp_path, monkeypatch):
    m = _probe()
    cfg = tmp_path / 'api.json'
    cfg.write_text(
        json.dumps(
            {
                'base_url': 'https://tiles.example.invalid',
                'info': {'path': '/i/{image}'},
                'tile': {'path': '/t/{image}/{level}/{col}/{row}'},
            }
        ),
        encoding='utf-8',
    )
    sent = []
    monkeypatch.setattr(
        httpx.Client, 'send', lambda *a, **k: sent.append(a) or (_ for _ in ()).throw(AssertionError('sent'))
    )
    assert m.main([f'http:{cfg}', '--images', 'a,b', '--out', str(tmp_path / 'o')]) is None
    assert not sent and not (tmp_path / 'o').exists()


def test_http_connector_config(slides, monkeypatch):
    from wissi_viz.connectors import HttpTileConnection

    with pytest.raises(ValueError, match='TODO'):
        HttpTileConnection(ROOT / 'scripts' / 'probe_configs' / 'halolink.example.json')
    files, _ = slides
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    p = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / 'examples' / 'tile_api' / 'mock_tile_api.py'),
            *map(str, files),
            '--port',
            str(port),
            '--n',
            '4',
            '--latency',
            '1,3',
            '--errors',
            '0',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                httpx.get(f'http://127.0.0.1:{port}/api/_stats', timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        monkeypatch.setenv('MOCK_API_URL', f'http://127.0.0.1:{port}')
        monkeypatch.setenv('MOCK_API_TOKEN', 'secret')
        c = HttpTileConnection(ROOT / 'scripts' / 'probe_configs' / 'mock_api.json')
        for img, overlap in (('img-01', 1), ('img-04', 0)):  # DeepZoom, and opaque coarsest-first level ids
            src = c.source(img)
            assert src.info.width == 3072 and src.pyramid.overlap == overlap
            a = src.read_tile(0, 1, 1)
            assert a.shape[:2] == (src.tile_rect(0, 1, 1)[3], src.tile_rect(0, 1, 1)[2])
    finally:
        p.kill()


def test_halolink_and_pathcore_modalities(slides, tmp_path):
    """The two connector modalities (sync DeepZoom HALO-style, async coarsest-first padded-PNG PathCore-style) pass the
    smoke test; a shallow API pyramid is flagged and runs that cannot fit the budget are skipped, never overrun."""
    m = _probe()
    files, d = slides
    common = [
        '--workdir',
        str(d),
        '--local-size',
        '3072x2304',
        '--mock-latency',
        '2,5',
        '--mock-errors',
        '0',
        '--seconds',
        '8',
    ]
    h = m.main(
        ['mock:halolink', '--panes', '1,4', '--conn', '5', '--max-calls', '200', '--out', str(tmp_path / 'h'), *common]
    )
    assert all(x['status'] in ('PASS', 'INFO') for x in h['smoke']), h['smoke']
    assert h['budget'].used <= 200 and all('cold' in r for r in h['runs'])
    p = m.main(
        ['mock:pathcore', '--panes', '1,16', '--conn', '5', '--max-calls', '220', '--out', str(tmp_path / 'p'), *common]
    )
    depth = [x for x in p['smoke'] if 'pyramid depth' in x['check']]
    assert depth and all(x['status'] == 'WARN' for x in depth)  # 768×576 coarsest level = 9 tiles
    assert p['budget'].used <= 220 and p['budget'].refused == 0  # planned, not cut off
    r16 = [r for r in p['runs'] if r['panes'] == 16]
    assert r16 and 'to open 16 slide' in r16[0].get('skipped', '')
    assert any('cold' in r for r in p['runs'] if r['panes'] == 1)
