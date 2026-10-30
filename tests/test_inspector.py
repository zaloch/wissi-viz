"""The app's inspector: layer summaries and object percentiles, colour-by and removal, the spatial-analysis
catalogue and runner, and the Layers / Analyze / Measure / Draw page with its overlays, in a real browser."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip('fastapi')

MPP = 0.5


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp('insp')
    rng = np.random.default_rng(3)
    W = H = 3000
    img = np.full((H, W, 3), 245, np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    tissue = (xx - 1500) ** 2 / 1200**2 + (yy - 1500) ** 2 / 1100**2 < 1
    img[tissue] = (214, 150, 190)
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = {MPP}'
    with tifffile.TiffWriter(d / 'slide.svs', bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
        tw.write(
            img[::4, ::4].copy(), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None, subfiletype=1
        )
    n = 1500
    tumor = rng.normal(1300, 180, (700, 2))
    rest = rng.uniform(500, 2500, (n - 700, 2))
    xy = np.vstack([tumor, rest])
    cls = ['Tumor'] * 700 + list(rng.choice(['Immune', 'Stroma'], n - 700))
    area = np.r_[rng.normal(80, 10, 700), rng.normal(40, 8, n - 700)]
    rows = ['x,y,class,area_um2,marker'] + [
        f'{x:.1f},{y:.1f},{c},{a:.2f},{m:.4f}' for (x, y), c, a, m in zip(xy, cls, area, rng.random(n))
    ]
    (d / 'cells.csv').write_text('\n'.join(rows), encoding='utf-8')
    roi = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'properties': {'name': 'Tumor nest', 'classification': {'name': 'Tumor'}},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[1000, 1000], [1600, 1000], [1600, 1600], [1000, 1600], [1000, 1000]]],
                },
            }
        ],
    }
    (d / 'regions.geojson').write_text(json.dumps(roi), encoding='utf-8')
    return d


@pytest.fixture()
def session(data, tmp_path):
    from wissi_viz import desktop as D

    s = D.Session(cache_dir=tmp_path / 'cache', remember=False)
    r = s.open([data / 'slide.svs', data / 'cells.csv', data / 'regions.geojson'], trusted=True)
    assert not r['errors'], r['errors']
    sid = r['opened'][0]
    yield s, sid
    s.shutdown()


def _client(s):
    from starlette.testclient import TestClient

    return TestClient(s.app())


# ------------------------------------------------------------------------------------------------ summaries
def test_summary_counts_means_and_density(session):
    s, sid = session
    c = _client(s)
    r = c.get(f'/wissi-viz/slides/{sid}/annotations/summary', params={'layer': 'cells'})
    assert r.status_code == 200, r.text
    sm = r.json()
    assert sm['n'] == 1500 and {k['name'] for k in sm['classes']} == {'Tumor', 'Immune', 'Stroma'}
    tumor = next(k for k in sm['classes'] if k['name'] == 'Tumor')
    assert tumor['n'] == 700 and tumor['pct'] == pytest.approx(46.7, abs=0.1)
    assert sm['columns'][0] == 'area_um2' and set(sm['numeric']) >= {'area_um2', 'marker'}
    assert tumor['means']['area_um2'] == pytest.approx(80, abs=2)
    if pytest.importorskip('skimage', reason='tissue area needs scikit-image'):
        true_mm2 = np.pi * 1200 * 1100 * MPP * MPP / 1e6
        assert sm['tissue_mm2'] == pytest.approx(true_mm2, rel=0.08)
        assert sm['per_mm2'] == pytest.approx(1500 / sm['tissue_mm2'], rel=1e-3)
    regions = c.get(
        f'/wissi-viz/slides/{sid}/annotations/summary', params={'layer': 'regions', 'tissue': 'false'}
    ).json()
    assert regions['area_mm2'] == pytest.approx(600 * 600 * MPP * MPP / 1e6, rel=1e-3)


def test_object_percentiles(session):
    s, sid = session
    c = _client(s)
    L = s.registry.layer(sid, 'cells')
    areas = L.column_values('area_um2')
    i = int(np.argmax(areas))
    f = c.get(f'/wissi-viz/slides/{sid}/annotations/feature/{i}', params={'layer': 'cells', 'pct': 1}).json()
    assert f['percentiles']['area_um2'] == 100.0 and f['order'][0] == 'area_um2'
    j = int(np.argmin(areas))
    g = c.get(f'/wissi-viz/slides/{sid}/annotations/feature/{j}', params={'layer': 'cells', 'pct': 1}).json()
    assert g['percentiles']['area_um2'] == pytest.approx(100 / 1500, abs=0.1)
    assert (
        'percentiles' not in c.get(f'/wissi-viz/slides/{sid}/annotations/feature/{j}', params={'layer': 'cells'}).json()
    )


def test_colour_by_and_remove(session):
    s, sid = session
    c = _client(s)
    url = f'/wissi-viz/slides/{sid}/annotations'
    r = c.post(f'{url}/style', json={'layer': 'cells', 'color_by': 'marker', 'cmap': 'magma'})
    assert r.status_code == 200 and r.json()['colormap']['name'] == 'magma'
    assert s.registry.layer(sid, 'cells').colormap is not None
    assert c.post(f'{url}/style', json={'layer': 'cells', 'color_by': 'nope'}).status_code == 400
    assert c.post(f'{url}/style', json={'layer': 'cells', 'color_by': None}).json()['colormap'] is None
    assert c.delete(url, params={'layer': 'regions'}).status_code == 200
    assert 'regions' not in s.registry.layers(sid)
    assert c.delete(url, params={'layer': 'regions'}).status_code == 404


# ------------------------------------------------------------------------------------------------ analyses
def test_catalogue_and_runs(session):
    pytest.importorskip('scipy')
    pytest.importorskip('shapely')
    s, sid = session
    c = _client(s)
    cat = c.get('/wissi-viz/analyses').json()
    ids = {a['id'] for a in cat}
    assert {
        'density',
        'nn',
        'prox',
        'bands',
        'infiltration',
        'composition',
        'ripley',
        'enrichment',
        'hoods',
        'contacts',
        'moran',
    } <= ids
    assert next(a for a in cat if a['id'] == 'bands')['needs_regions'] is True
    url = f'/wissi-viz/slides/{sid}/analyses'
    r = c.post(url, json={'analysis': 'density', 'objects': 'cells', 'params': {'cls': 'Tumor', 'cell_um': 100}})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res['layers'] == ['Density · Tumor', 'Hotspots · Tumor'] and res['chart']['kind'] == 'bars'
    assert s.registry.layer(sid, 'Density · Tumor').meta()['style']['origin'] == 'analysis'
    nn = c.post(
        url,
        json={'analysis': 'nn', 'objects': 'cells', 'params': {'source': 'Immune', 'target': 'Tumor', 'max_um': 200}},
    ).json()
    assert nn['kpis'][0]['label'] == 'median' and nn['kpis'][0]['value'].endswith('µm') and nn['table']['n'] > 0
    bands = c.post(
        url, json={'analysis': 'bands', 'objects': 'cells', 'params': {'regions': 'regions', 'width_um': 50}}
    ).json()
    assert bands['layers'] and bands['chart']['series']
    comp = c.post(url, json={'analysis': 'composition', 'objects': 'cells', 'params': {'regions': 'regions'}})
    assert comp.status_code == 200, comp.text  # its layer used to fail on an unknown style option
    # clear errors for wrong input
    assert (
        c.post(
            url, json={'analysis': 'nn', 'objects': 'cells', 'params': {'source': 'Nope', 'target': 'Tumor'}}
        ).status_code
        == 400
    )
    assert 'region' in c.post(url, json={'analysis': 'bands', 'objects': 'cells', 'params': {}}).json()['detail']
    assert c.post(url, json={'analysis': 'nope', 'objects': 'cells'}).status_code == 400


def test_every_analysis_runs(session):
    pytest.importorskip('scipy')
    pytest.importorskip('shapely')
    from wissi_viz import analyses

    s, sid = session
    params = {
        'nn': {'source': 'Immune', 'target': 'Tumor'},
        'prox': {'source': 'Immune', 'target': 'Tumor'},
        'bands': {'regions': 'regions'},
        'infiltration': {'regions': 'regions'},
        'composition': {'regions': 'regions'},
        'ripley': {'a': 'Tumor'},
        'moran': {'column': 'marker'},
    }
    for key in analyses.CATALOG:
        out = analyses.run(s.registry, sid, key, 'cells', params.get(key, {}))
        json.dumps(out)  # what the page receives
        assert out['title'] and (out['chart'] or out['kpis']), key


# ------------------------------------------------------------------------------------------------ the page
def test_inspector_in_the_browser(session):
    pytest.importorskip('playwright')
    pytest.importorskip('scipy')
    from playwright.sync_api import sync_playwright

    from wissi_viz import desktop as D

    s, sid = session
    srv = D._Server(s.app())
    url = srv.start()
    errors = []
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1440, 'height': 900})
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.goto(url + '/app')
            g = '() => document.getElementById("grid")'
            pg.wait_for_function(f'() => ({g})().panes.every(p => p.slide && p.slide.info)')
            pg.wait_for_selector('#l-list .lcard[data-layer="cells"]')
            # tools: number keys light the slot
            pg.keyboard.press('9')
            assert pg.evaluate(f'() => ({g})().tool') == 'sam' and pg.is_visible('#tools .slot.on[data-tool=sam]')
            pg.keyboard.press('1')
            # layer eye, solo, class toggle, colour by
            vis = f'(n) => {{ const p = ({g})().panes[0]; return p.layerVisible(p.slide.layers.get(n)); }}'
            pg.click('.lcard[data-layer="regions"] .eye')
            assert pg.evaluate(vis, 'regions') is False
            pg.click('.lcard[data-layer="regions"] .eye')
            pg.click('.lcard[data-layer="cells"] .solo-b')
            assert pg.evaluate(vis, 'cells') and not pg.evaluate(vis, 'regions')
            pg.click('.lcard[data-layer="cells"] .solo-b')
            assert pg.evaluate(vis, 'regions')
            pg.click('.lcard[data-layer="cells"] .nm')
            pg.wait_for_selector('.lcard[data-layer="cells"] .cls .chip')  # counts from the summary
            pg.click('.lcard[data-layer="cells"] .cls:has-text("Stroma") .box')
            assert pg.evaluate(f'() => ({g})().panes[0].slide.layers.get("cells").hidden.has("Stroma")')
            pg.select_option('.lcard[data-layer="cells"] .lbody select', 'area_um2')
            pg.wait_for_function(f'() => ({g})().panes[0].slide.layers.get("cells").meta.colormap')
            # object card with percentiles, then the radial menu
            L = s.registry.layer(sid, 'cells')
            c = L.compiled()
            i = int(np.argmax(L.column_values('area_um2')))
            pg.evaluate(
                f'([x, y]) => ({g})().goTo(0, x, y, {{zoom: 2, animate: false}})', [float(c.cx[i]), float(c.cy[i])]
            )
            pg.wait_for_timeout(600)
            box = pg.evaluate(f"""() => {{ const e = ({g})(), p = e.panes[0], r = e.getBoundingClientRect();
                return [r.left + p.rect.x + p.rect.w / 2, r.top + p.rect.y + p.rect.h / 2]; }}""")
            pg.mouse.click(*box)
            pg.wait_for_selector('#card .bars')
            assert 'area_um2' in pg.inner_text('#card')
            pg.keyboard.press('Escape')
            assert not pg.is_visible('#card')
            pg.mouse.click(box[0] - 150, box[1] + 100, button='right')
            pg.wait_for_selector('#radial[style*="block"]')
            pg.click('#radial [data-r=measure]')
            assert pg.get_attribute('#t-measure', 'aria-selected') == 'true'
            pg.wait_for_selector('#ms-body .tiles')
            assert '1,500' in pg.inner_text('#ms-body')
            # Analyze: run density & hotspots → result card, layers and a toast
            pg.keyboard.press('a')
            pg.click('.an-card:has-text("Density")')
            pg.click('#an-run')
            pg.wait_for_selector('.result .chart svg', timeout=60_000)
            pg.wait_for_selector('.toast.analysis')
            pg.wait_for_function(f'() => ({g})().panes[0].slide.layers.has("Hotspots · all cells")')
            pg.keyboard.press('l')
            pg.wait_for_selector('.lcard.analysis')
            # help sheet
            pg.keyboard.press('?')
            assert pg.is_visible('#help')
            b.close()
    finally:
        srv.stop()
    assert not errors, errors


# ------------------------------------------------------------------------------------------------ drawing and measuring
def test_drawing_measuring_undo_delete_in_the_browser(session):
    """Shapes keep their size (µm), Shift makes squares, the measure tools show a length / area that Enter keeps,
    Backspace removes a point, ⌘Z / Ctrl+Z undoes, Delete removes the selected drawing and undo puts it back; the
    tools have tooltips and an on-slide hint."""
    pytest.importorskip('playwright')
    from playwright.sync_api import sync_playwright

    from wissi_viz import desktop as D

    s, sid = session
    srv = D._Server(s.app())
    url = srv.start()
    errors = []
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1440, 'height': 900})
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.on('response', lambda r: errors.append(f'{r.status} {r.url}') if r.status >= 400 else None)
            pg.goto(url + '/app')
            g = '() => document.getElementById("grid")'
            pg.wait_for_function(f'() => ({g})().panes.every(p => p.slide && p.slide.info)')
            pg.evaluate(f'() => ({g})().fit(0, false)')
            pg.wait_for_timeout(300)

            def at(x, y):
                return pg.evaluate(
                    f"""([x, y]) => {{ const e = ({g})(), p = e.panes[0], r = e.getBoundingClientRect(), v = p.view;
                    return [r.left + p.rect.x + p.rect.w / 2 + (x - v.cx) * v.scale,
                            r.top + p.rect.y + p.rect.h / 2 + (y - v.cy) * v.scale]; }}""",
                    [x, y],
                )

            def drag(a, z, shift=False):
                pg.mouse.move(*at(*a))
                if shift:
                    pg.keyboard.down('Shift')
                pg.mouse.down()
                pg.mouse.move(*at(*z), steps=6)
                pg.mouse.up()
                if shift:
                    pg.keyboard.up('Shift')

            def layer(name):
                return s.registry.layers(sid).get(name)

            shots = os.environ.get('WISSI_SWEEP_SHOTS')  # a folder: save what the steps look like

            def shot(name):
                if shots:
                    pg.screenshot(path=str(Path(shots) / f'draw-{name}.png'))

            def wait_n(name, n):
                pg.wait_for_function(
                    f'([n, k]) => {{ const L = ({g})().panes[0].slide.layers.get(n); return k ? L && L.meta.count === k && L.state === "ready" : !L; }}',
                    arg=[name, n],
                )

            # tooltip and hint
            pg.hover('#tools .slot[data-tool=measure]')
            pg.wait_for_selector('#ttip', state='visible')
            assert 'Measure length' in pg.inner_text('#ttip') and 'Enter keeps it' in pg.inner_text('#ttip')
            shot('tooltip')
            # a rectangle keeps its size in µm; Shift makes a square
            pg.keyboard.press('3')
            assert pg.is_visible('#toolhint') and 'Rectangle' in pg.inner_text('#toolhint')
            pg.mouse.move(*at(1000, 1000))
            pg.mouse.down()
            pg.mouse.move(*at(1400, 1200), steps=6)
            shot('rect-live')
            pg.mouse.up()
            wait_n('Annotations', 1)
            props = layer('Annotations').properties(0)
            assert props['area_um2'] == pytest.approx(400 * 200 * MPP**2, rel=0.03), props
            assert props['width_um'] == pytest.approx(400 * MPP, rel=0.03)
            drag((1800, 1000), (2100, 1150), shift=True)
            wait_n('Annotations', 2)
            sq = layer('Annotations').properties(1)
            assert sq['width_um'] == pytest.approx(sq['height_um'], rel=0.01), sq
            # undo: the square, then the rectangle (the empty layer goes)
            pg.keyboard.press('Control+z')
            wait_n('Annotations', 1)
            pg.keyboard.press('Control+z')
            wait_n('Annotations', 0)
            # measure a length by dragging; Enter keeps it with its length
            pg.keyboard.press('7')
            drag((500, 500), (500, 1500))
            m = pg.evaluate(f'() => ({g})()._measured')
            assert m['kind'] == 'measure' and len(m['pts']) == 2
            shot('length')
            pg.keyboard.press('Enter')
            wait_n('Measurements', 1)
            assert layer('Measurements').properties(0)['length_um'] == pytest.approx(1000 * MPP, rel=0.03)
            # measure an area by clicking points; Backspace removes the last one; Esc clears
            pg.keyboard.press('8')
            for q in ((2000, 2000), (2400, 2000), (2400, 2300), (2600, 2600)):
                pg.mouse.click(*at(*q))
            pg.keyboard.press('Backspace')
            assert len(pg.evaluate(f'() => ({g})()._drawing.pts')) == 3
            pg.keyboard.press('Enter')
            m = pg.evaluate(f'() => ({g})()._measured')
            assert m['kind'] == 'area' and len(m['pts']) == 3
            shot('area')
            pg.keyboard.press('Escape')
            assert pg.evaluate(f'() => ({g})()._measured') is None
            # a polygon, selected with a click and deleted; undo puts it back
            pg.keyboard.press('5')
            for q in ((1000, 2000), (1500, 2000), (1500, 2400), (1000, 2400)):
                pg.mouse.click(*at(*q))
            pg.keyboard.press('Enter')
            wait_n('Annotations', 1)
            pg.keyboard.press('1')
            assert not pg.is_visible('#toolhint')
            pg.mouse.click(*at(1250, 2200))
            pg.wait_for_function(f'() => ({g})().panes[0].selection?.layer === "Annotations"')
            pg.wait_for_selector('#card', state='visible')  # its sizes: the object card
            assert 'area_um2' in pg.inner_text('#card')
            pg.keyboard.press('Escape')
            pg.keyboard.press('Delete')
            wait_n('Annotations', 0)
            pg.keyboard.press('Control+z')
            wait_n('Annotations', 1)
            assert layer('Annotations').properties(0)['area_um2'] == pytest.approx(500 * 400 * MPP**2, rel=0.03)
            b.close()
    finally:
        srv.stop()
    # undo can empty (and so remove) a layer while the inspector still asks for its summary: a 404 nobody sees
    gone = [e for e in errors if '/annotations/summary?layer=Annotations' in e]
    errors = [
        e for e in errors if e not in gone and not (gone and e.startswith('Failed to load resource') and '404' in e)
    ]
    assert not errors, errors


def test_remove_and_geometry_endpoints(session):
    s, sid = session
    c = _client(s)
    base = f'/wissi-viz/slides/{sid}/annotations'
    feats = [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [k, 0], [k, k], [0, 0]]]},
              'properties': {'classification': {'name': f'c{k}'}}} for k in (10, 20, 30)]  # fmt: skip
    r = c.post(base + '/add', params={'layer': 'D'}, json={'type': 'FeatureCollection', 'features': feats}).json()
    assert r['indices'] == [0, 1, 2]
    f = c.get(base + '/feature/1', params={'layer': 'D', 'geometry': 1}).json()
    assert f['geometry']['coordinates'][0][1] == [20, 0] and f['properties']['classification']['name'] == 'c20'
    assert c.post(base + '/remove', params={'layer': 'D', 'index': 1}).json()['left'] == 2
    assert s.registry.layer(sid, 'D').class_of(1) == 'c30'
    assert c.post(base + '/remove', params={'layer': 'D', 'index': 9}).status_code == 400
    # an object table is not edited one object at a time
    assert c.post(base + '/remove', params={'layer': 'cells', 'index': 0}).status_code == 400
    c.post(base + '/remove', params={'layer': 'D', 'index': 0})
    c.post(base + '/remove', params={'layer': 'D', 'index': 0})
    assert 'D' not in s.registry.layers(sid)
