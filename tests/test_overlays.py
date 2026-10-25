"""Object overlays at scale: the object pyramid, GPU-drawn bbox shapes, the persistent compiled cache, readers."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from wissi_viz import AnnotationLayer
from wissi_viz.annotations import ELLIPSE, PYR_SLOTS


def _unpack(buf):
    jl = struct.unpack_from('<I', buf, 4)[0]
    import json
    meta = json.loads(buf[8:8 + jl])
    off = [8 + jl]

    def take(dtype, n):
        a = np.frombuffer(buf, dtype, n, off[0])
        off[0] += -(-(n * np.dtype(dtype).itemsize) // 4) * 4
        return a
    return meta, take


def _cells(n=120_000, seed=0, W=40_000, H=30_000, ncls=10):
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(0, W, n), rng.uniform(0, H, n)
    r = rng.uniform(4, 10, n)
    cls = [f'c{int(k)}' for k in rng.integers(0, ncls, n)]
    return AnnotationLayer.from_boxes(cx - r, cy - r, cx + r, cy + r, classes=cls, name='cells'), cx, cy, cls


def _tiles(c, level):
    out = []
    for (tx, ty) in c._pyr[level]['index']:
        out.append(_unpack(c.pyramid_tile(level, tx, ty)))
    return out


def test_pyramid_levels_account_for_every_feature():
    L, cx, cy, cls = _cells()
    c = L.compiled()
    pm = c.pyr_meta
    assert pm and pm['levels'] >= 4 and pm['mode'] == 'class' and len(pm['slots']) == PYR_SLOTS
    # level 0: every feature exactly once, at its centre (u16 quantised within its tile)
    seen = []
    for meta, take in _tiles(c, 0):
        n = meta['n']
        pos = take('<u2', 2 * n).reshape(-1, 2)
        take('<u2', n), take('<u2', n), take('<u2', n)
        orig = take('<u4', n)
        x = meta['x0'] + pos[:, 0] / 65535 * meta['span']
        assert np.abs(x - cx[orig]).max() <= meta['span'] / 65535 + 1e-3
        seen.append(orig)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(len(cx)))
    # aggregated levels: counts add up to n; the majority colour is the class with the largest count
    for lv in range(1, pm['levels']):
        total = 0
        for meta, take in _tiles(c, lv):
            n = meta['n']
            take('<u2', 2 * n)
            cnt = take('<u2', n)
            col = take('<u2', n)
            take('<u4', n)
            hist = take('u1', n * 8).reshape(-1, 8)
            total += int(cnt.sum())
            slots = np.asarray([s if s >= 0 else pm['other_class'] for s in pm['slots']])
            assert (hist.max(1) == 255).all()
            # the majority colour belongs to a slot holding the (scaled) maximum count
            slot_of_col = {int(c): j for j, c in enumerate(slots)}
            assert all(hist[r, slot_of_col[int(col[r])]] == 255 for r in range(0, n, 7))
        assert total == len(cx), (lv, total)
    # payload a browser needs is bounded by screen area, not by n: the coarsest level is one small tile
    assert pm['points'][-1] < 4000 and pm['tiles'][-1] == 1


def test_bbox_shapes_are_sent_as_parameters_not_vertices():
    L, cx, cy, _ = _cells(20_000)
    c = L.compiled()
    assert L._block is not None and not L._rings          # columnar: no per-cell vertex arrays
    meta, take = _unpack(c.bucket(0))
    assert meta['segments'] == 0 and meta['tri_verts'] == 0 and meta['shapes'] > 0
    take('<f4', 0), take('<f4', 0), take('<u2', 0), take('<u2', 0)
    shp = take('<f4', meta['shapes'] * 4).reshape(-1, 4)
    take('<u2', meta['shapes'])
    kinds = take('u1', meta['shapes'])
    assert (kinds == ELLIPSE).all()
    feats = c.order[int(c.buckets[0][4]):int(c.buckets[0][4]) + int(c.buckets[0][5])]
    assert np.allclose(shp[:, 0], cx[feats], atol=1e-2) and np.allclose(shp[:, 1], cy[feats], atol=1e-2)
    assert len(c.bucket(0)) < 25 * meta['shapes'] + 200    # ~19 bytes per cell


def test_hit_test_skips_hidden_classes_and_mixed_edits_still_work():
    L, cx, cy, cls = _cells(5000)
    c = L.compiled()
    i = int(np.argmin((cx - 20000) ** 2 + (cy - 15000) ** 2))
    assert c.hit_test(cx[i], cy[i], 1.0) is not None
    hid = L._class_index[cls[i]]
    j = c.hit_test(cx[i], cy[i], 1.0, hidden=[hid])
    assert j is None or cls[j] != cls[i]
    # an incremental edit converts the columnar block to classic storage, keeping every feature
    L.add_polygon([[0, 0], [10, 0], [10, 10]], cls='roi')
    assert len(L) == 5001 and L.compiled().n == 5001


def test_save_open_roundtrip_is_byte_identical(tmp_path):
    L, *_ = _cells(80_000, ncls=12)               # > 8 classes → an 'other' slot
    L._columns = {'area': np.arange(80_000, dtype=np.float32)}
    a = L.compiled()
    L.save(tmp_path / 'lay')
    M = AnnotationLayer.open(tmp_path / 'lay', version=4242)
    b = M.compiled()
    assert M.version == 4242 and a.pyr_meta == b.pyr_meta and -1 in a.pyr_meta['slots']
    for lv in range(a.pyr_meta['levels']):
        k = next(iter(a._pyr[lv]['index']))
        assert a.pyramid_tile(lv, *k) == b.pyramid_tile(lv, *k)
    assert a.bucket(3) == b.bucket(3) and M.properties(777)['area'] == 777.0
    assert M.meta()['count'] == L.meta()['count'] and M.meta()['pyramid'] == L.meta()['pyramid']


def test_cached_object_layer_rebuilds_when_the_file_changes(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq

    from wissi_viz import cached_object_layer
    rng = np.random.default_rng(1)
    n = 60_000
    x = rng.uniform(0, 20000, n)
    t = pa.table({'XMin': x - 5, 'XMax': x + 5, 'YMin': x * 0.5 - 5, 'YMax': x * 0.5 + 5,
                  'Classification': np.asarray(['a', 'b'])[rng.integers(0, 2, n)], 'Area': rng.random(n)})
    f = tmp_path / 'cells.parquet'
    pq.write_table(t, f)
    L1 = cached_object_layer(f, tmp_path / 'cache', mpp=0.5, read_kw={'units': 'px'})
    L2 = cached_object_layer(f, tmp_path / 'cache', mpp=0.5, read_kw={'units': 'px'})
    assert L1.version == L2.version and len(L2) == n and L2.compiled().pyr_meta
    pq.write_table(t.slice(0, n // 2), f)
    L3 = cached_object_layer(f, tmp_path / 'cache', mpp=0.5, read_kw={'units': 'px'})
    assert len(L3) == n // 2 and L3.version != L1.version


def test_delimited_reader_detects_separator_and_compacts_text(tmp_path):
    from wissi_viz import read_objects
    rows = ['Centroid X px\tCentroid Y px\tClass\tScore'] + [f'{i}\t{2 * i}\t{"T" if i % 2 else "S"}\t{i / 10}' for i in range(500)]
    (tmp_path / 'o.txt').write_text('\n'.join(rows))
    o = read_objects(tmp_path / 'o.txt', units='px')
    assert len(o) == 500 and set(o.class_names()) == {'T', 'S'}
    L = o.to_layer()
    assert L.properties(3)['Score'] == pytest.approx(0.3, abs=1e-6)
    (tmp_path / 'o.csv').write_text('\n'.join(r.replace('\t', ';') for r in rows))
    assert len(read_objects(tmp_path / 'o.csv', units='px')) == 500


def test_sixteen_panes_with_pyramid_layers_load_draw_and_click(tmp_path):
    """Regression: overview requests were cancelled while 'loading' and could leave a layer stuck forever."""
    pytest.importorskip('playwright')
    import socket
    import threading
    import time

    import uvicorn

    from wissi_viz import Level, SlideInfo, SlideRegistry
    from wissi_viz.server import create_app
    from wissi_viz.sources.callback import CallbackSource
    W, H = 40_000, 30_000
    info = SlideInfo(width=W, height=H, levels=[Level(-(-W // 2**k), -(-H // 2**k), float(2**k)) for k in range(8)],
                     tile_size=512, mpp=(0.5, 0.5))
    blank = np.full((512, 512, 3), 255, np.uint8)
    reg = SlideRegistry(warm=False)
    L, cx, cy, cls = _cells(60_000, W=W, H=H, ncls=3)
    for i in range(16):
        reg.add(CallbackSource(info, read_tile=lambda *a: blank), slide_id=f's{i}')
        reg.add_annotations(f's{i}', L)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:  # noqa: BLE001
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 800, 'height': 600})
            errs = []
            pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' or 'GL_INVALID' in m.text else None)
            pg.goto(f'http://127.0.0.1:{port}/')
            pg.wait_for_function('() => window.grid')
            pg.evaluate('() => { grid.setAttribute("controls", "false"); grid.setSlides([...Array(16).keys()].map(i => "s" + i)); }')
            pg.wait_for_function('''() => grid.panes.length === 16 && grid.panes.every(p => { const L = p.slide && p.slide.layers.get('cells');
                return L && L.state === 'ready' && L.ptiles.size > 0; })''', timeout=60000)
            snap = """() => { grid._draw(); const gl = grid.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
                const px = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
                let n = 0; for (let i = 0; i < px.length; i += 4) if (px[i] < 250 || px[i + 1] < 250 || px[i + 2] < 250) n++; return n / (w * h); }"""
            pg.wait_for_timeout(500)
            shown = pg.evaluate(snap)
            assert shown > 0.05, shown                       # aggregated dots drawn in every pane
            pg.evaluate('() => grid.panes.forEach(p => { const L = p.slide.layers.get("cells"); L.classes.forEach(c => L.setClassVisible(c.name, false)); })')
            assert pg.evaluate(snap) < shown * 0.2           # hiding every class hides the aggregated dots
            pg.evaluate('() => grid.panes.forEach(p => { const L = p.slide.layers.get("cells"); L.classes.forEach(c => L.setClassVisible(c.name, true)); })')
            # click on a cell at an aggregated zoom → server hit test → a real object with properties
            i = int(np.argmin((cx - W / 2) ** 2 + (cy - H / 2) ** 2))
            got = pg.evaluate(f'''() => new Promise((res) => {{
                grid.addEventListener('objectclick', (e) => res(e.detail), {{ once: true }});
                const p = grid.panes[0], r = p.rect, el = grid.$canvas.getBoundingClientRect();
                const sx = r.x + r.w / 2 + ({cx[i]} - p.view.cx) * p.view.scale, sy = r.y + r.h / 2 + ({cy[i]} - p.view.cy) * p.view.scale;
                grid._click(p, {{ clientX: el.left + sx, clientY: el.top + sy, button: 0 }});
                setTimeout(() => res(null), 8000); }})''')
            assert got and got['layer'] == 'cells' and got['index'] is not None and 'properties' in got, got
            assert not [e for e in errs if 'GL_INVALID' in e], errs
            b.close()
    finally:
        srv.should_exit = True


def test_client_mode_builds_the_same_pyramid_in_the_browser(tmp_path):
    """overlays: 'client' — the layer's columns are downloaded once and the pyramid is built in a worker: every
    aggregated level accounts for every object, clicks resolve locally, zooming makes no overlay requests."""
    pytest.importorskip('playwright')
    import socket
    import threading
    import time

    import uvicorn

    from wissi_viz import Level, SlideInfo, SlideRegistry
    from wissi_viz.server import create_app
    from wissi_viz.sources.callback import CallbackSource
    W, H = 40_000, 30_000
    info = SlideInfo(width=W, height=H, levels=[Level(-(-W // 2**k), -(-H // 2**k), float(2**k)) for k in range(8)],
                     tile_size=512, mpp=(0.5, 0.5))
    reg = SlideRegistry(warm=False)
    L, cx, cy, cls = _cells(90_000, W=W, H=H, ncls=11)
    reg.add(CallbackSource(info, read_tile=lambda *a: np.full((512, 512, 3), 255, np.uint8)), slide_id='s0')
    reg.add_annotations('s0', L)
    server_points = L.compiled().pyr_meta['points']
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(reg), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:  # noqa: BLE001
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 800, 'height': 600})
            seen = []
            pg.on('request', lambda r: seen.append(r.url) if '/annotations' in r.url else None)
            pg.goto(f'http://127.0.0.1:{port}/')
            pg.wait_for_function('() => window.grid')
            pg.evaluate('() => { grid.setAttribute("controls", "false"); grid.setLod({overlays: "client"}); grid.setSlides(["s0"]); }')
            pg.wait_for_function('() => { const L = grid.panes[0].slide && grid.panes[0].slide.layers.get("cells"); return L && L.client === "ready"; }',
                                 timeout=60000)
            st = pg.evaluate('''() => { const L = grid.panes[0].slide.layers.get('cells');
                return { pyr: L.pyr, sums: L.local.map((m, i) => i === 0 ? null : [...m.values()].reduce((a, t) => a + t.count.reduce((x, y) => x + y, 0), 0)) }; }''')
            assert all(v == len(cx) for v in st['sums'][1:]), st['sums']
            assert st['pyr']['points'][0] == server_points[0] == len(cx)
            assert -1 in st['pyr']['slots']                       # 11 classes → an 'other' slot, as on the server
            i = int(np.argmin((cx - W / 2) ** 2 + (cy - H / 2) ** 2))
            n0 = len(seen)
            for mag in (2.5, 10, 40):
                pg.evaluate(f'() => grid.goTo(0, {cx[i]}, {cy[i]}, {{magnification: {mag}, animate: false}})')
                pg.wait_for_timeout(400)
            got = pg.evaluate('''() => new Promise((res) => {
                grid.addEventListener('objectclick', (e) => res(e.detail), { once: true });
                const p = grid.panes[0], r = p.rect, el = grid.$canvas.getBoundingClientRect();
                grid._click(p, { clientX: el.left + r.x + r.w / 2, clientY: el.top + r.y + r.h / 2, button: 0 });
                setTimeout(() => res(null), 8000); })''')
            assert got and got['index'] == i, (got, i)
            later = [u for u in seen[n0:] if '/annotations/' in u and '/feature/' not in u]
            assert not later, later                               # no tiles, buckets or hit tests after the download
            b.close()
    finally:
        srv.should_exit = True
