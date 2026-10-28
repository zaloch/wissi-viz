"""Server memory stays bounded however many slides are opened: auto-loaded annotation/object layers are memory-mapped
from the disk cache (no API calls to reopen), evicted least-recently-used beyond annotation_memory_mb, and overlay
tiles share one byte budget. Also: save()/open() with text columns (used to fail to reopen)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wissi_viz import AnnotationLayer, SlideRegistry, TileService
from wissi_viz.connectors import AnnotationPayload
from wissi_viz.connectors.base import to_layer
from wissi_viz.memory import anon_nbytes, overlay_cache, set_overlay_cache_mb


def _table(n, seed=0, W=8000, H=6000):
    rng = np.random.default_rng(seed)
    cx, cy, r = rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(4, 10, n)
    cols = {
        'XMin': cx - r,
        'XMax': cx + r,
        'YMin': cy - r,
        'YMax': cy + r,
        'Classification': rng.choice(['Tumor', 'Stroma', 'Immune'], n),
        'Note': rng.choice(['a', 'b c'], n),
    }
    for k in range(6):
        cols[f'M{k}'] = rng.random(n)
    return cols


@pytest.fixture(scope='module')
def slide(tmp_path_factory):
    d = tmp_path_factory.mktemp('mem')
    rng = np.random.default_rng(0)
    img = (rng.random((3072, 4096, 3)) * 255).astype(np.uint8)
    desc = 'Aperio Image Library v12 \r\n4096x3072 [0,0 4096x3072] (256x256) JPEG/RGB Q=80|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 'm.svs', bigtiff=True) as tw:
        o = {'tile': (256, 256), 'compression': 'jpeg', 'photometric': 'rgb', 'metadata': None}
        tw.write(img, description=desc, **o)
        tw.write(img[::16, ::16].copy(), description='Aperio thumb', photometric='rgb', metadata=None)
        tw.write(np.ascontiguousarray(img[::4, ::4]), description='Aperio Image Library level', **o)
    return d / 'm.svs'


def _connector(slide, n):
    from mock_remote import MockPathCore

    class PC(MockPathCore):
        calls = 0

        async def objects(self, slide_id):
            PC.calls += 1
            return [AnnotationPayload('Cells', _table(n, seed=abs(hash(slide_id)) % 1000), format='table')]

    return PC(str(slide)), PC


def test_save_open_with_text_columns_roundtrips(tmp_path):
    L = to_layer(AnnotationPayload('Cells', _table(20_000), format='table'))
    L.save(tmp_path / 'L')
    M = AnnotationLayer.open(tmp_path / 'L', version=L.version)
    for i in (0, 4321, 19_999):
        assert L.properties(i) == M.properties(i)
    assert L.compiled().bucket(0) == M.compiled().bucket(0)
    assert anon_nbytes(M) < anon_nbytes(L) / 3  # memory-mapped: mostly not anonymous RAM


def test_auto_layers_are_memory_mapped_and_reopen_without_api_calls(slide, tmp_path):
    conn, PC = _connector(slide, 30_000)
    reg = SlideRegistry(
        TileService(memory_bytes=64 * 2**20, disk_dir=tmp_path / 'cache'), warm=False, annotation_memory_mb=0.01
    )
    ids = [conn.register(reg, f'S{k}') for k in range(4)]
    in_ram = AnnotationLayer  # noqa: F841
    for sid in ids:
        names = list(reg.layers(sid))
        assert 'Cells' in names
        L = reg.layers(sid)['Cells']
        assert anon_nbytes(L) < 8 * 2**20  # 30k cells × 12 columns would be ~10 MB in RAM
    m = reg.memory_stats()
    assert m['auto_layers_ram_mb'] < 30 and m['layer_cache_dir']
    assert (tmp_path / 'cache' / '_layers').exists()
    # the tiny budget (10 kB) evicted earlier slides; reopening them hits the disk copy — no new objects() calls
    assert m['layer_evictions'] > 0
    calls = PC.calls
    for sid in ids:
        assert 'Cells' in reg.layers(sid)
    assert PC.calls == calls and reg.memory_stats()['layer_disk_hits'] > 0


def test_budget_evicts_without_disk_cache_and_reloads_on_demand(slide):
    conn, PC = _connector(slide, 20_000)
    reg = SlideRegistry(TileService(memory_bytes=64 * 2**20), warm=False, annotation_memory_mb=3)
    ids = [conn.register(reg, f'T{k}') for k in range(5)]
    for sid in ids:
        reg.layers(sid)
    m = reg.memory_stats()
    assert m['layer_evictions'] >= 3 and m['auto_layers_ram_mb'] <= 3 + 6  # budget + the slide just opened
    calls = PC.calls
    assert reg.layer(ids[0], 'Cells').compiled().n == 20_000  # evicted → fetched again transparently
    assert PC.calls == calls + 1


def test_overlay_tiles_share_one_byte_budget(slide):
    set_overlay_cache_mb(1)
    try:
        layers = [to_layer(AnnotationPayload('Cells', _table(60_000, seed=s), format='table')) for s in range(3)]
        for L in layers:
            c = L.compiled()
            for tx, ty in list(c._pyr[0]['index'])[:40]:
                c.pyramid_tile(0, tx, ty)
            for i in range(min(40, len(c.buckets))):
                c.bucket(i)
        st = overlay_cache().stats()
        assert st['bytes'] <= 2**20 and st['evictions'] > 0
        for L in layers:
            L.compiled().release()
    finally:
        set_overlay_cache_mb(128)


def test_tile_disk_trim_leaves_layers_alone(tmp_path):
    from wissi_viz.tiles import LAYER_DIR, DiskCache

    dc = DiskCache(tmp_path, max_bytes=1)
    keep = tmp_path / LAYER_DIR / 's' / 'v' / 'x.npy'
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b'0' * 1000)
    (tmp_path / 'ab').mkdir()
    (tmp_path / 'ab' / 'tile').write_bytes(b'0' * 1000)
    dc.trim()
    assert keep.exists() and not (tmp_path / 'ab' / 'tile').exists()


def test_stats_endpoint_reports_memory(slide):
    pytest.importorskip('httpx')
    from starlette.testclient import TestClient

    from wissi_viz.server import create_app

    reg = SlideRegistry(warm=False)
    reg.add(str(slide), slide_id='m')
    m = TestClient(create_app(reg)).get('/wissi-viz/stats').json()['memory']
    for k in (
        'rss_mb',
        'tile_cache_mb',
        'decoded_mb',
        'overlay_mb',
        'auto_layers_ram_mb',
        'gzip_cache_mb',
        'annotation_budget_mb',
    ):
        assert k in m
