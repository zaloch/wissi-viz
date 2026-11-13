"""The slide list arranged by metadata (group and sort chips, a metadata table), checked slides, and pages of panes:
only the shown page's slides load, each slide reopens where it was left, a page per group keeps its layout."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip('playwright')
pytest.importorskip('uvicorn')

GROUPS = ['Untreated', '1 MPK Free siRNA', '3 MPK LNP']
G = '() => document.getElementById("grid")'


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp('slidelist')
    rows = ['Slide,Treatment group,Animal,Dose (mg/kg)']
    for i in range(10):
        W, H = 800, 600
        img = np.full((H, W, 3), 240, np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        img[(xx - 400) ** 2 / 250**2 + (yy - 300) ** 2 / (120 + 10 * i) ** 2 < 1] = (215, 140, 200)
        desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
        with tifffile.TiffWriter(d / f'S{i:02d}.svs', bigtiff=True) as tw:
            tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
        rows.append(f'S{i:02d},{GROUPS[i % 3]},{1040 + i:06d},{[0, 1, 3][i % 3]}')  # the stem names the slide
    (d / 'meta.csv').write_text('\n'.join(rows), encoding='utf-8')
    return d


def _names(pg) -> list:
    """The slide shown in each pane (None: an empty pane), once every pane with a slide has read it."""
    pg.wait_for_function(f'() => ({G})().panes.every((p) => !p.slide || p.slide.info)')
    return pg.evaluate(f'() => ({G})().panes.map((p) => p.slide?.info?.name ?? null)')


def test_list_by_metadata_and_pages_of_panes(data, tmp_path):
    from playwright.sync_api import sync_playwright

    from wissi_viz import desktop as D

    s = D.Session(cache_dir=tmp_path / 'cache', remember=False)
    s.open(sorted(data.glob('*.svs')))
    assert s.add_metadata([data / 'meta.csv'])['meta_tables'][0]['matched'] == 10
    srv = D._Server(s.app())
    url = srv.start()
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1400, 'height': 900})
            pg.set_default_timeout(60000)
            pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.goto(url + '/app')
            pg.wait_for_selector('#slides li')
            pg.evaluate('() => localStorage.clear()')

            # group by treatment, sort by dose descending then animal: nested sections in that order
            pg.click('#s-add-group')
            pg.click('.menu button:has-text("Treatment group")')
            pg.wait_for_function(
                '() => document.querySelectorAll("#slides section[data-kind=\'Treatment group\']").length === 3'
            )
            heads = pg.eval_on_selector_all('#slides section .ghead .gn', 'els => els.map((e) => e.textContent)')
            assert heads == sorted(GROUPS, key=str.lower)  # '1 MPK…' < '3 MPK…' < 'Untreated'
            pg.click('#s-add-sort')
            pg.click('.menu button:has-text("Animal")')
            pg.click('#s-chips .lchip.group .dir')  # groups descending
            pg.wait_for_function('() => document.querySelector("#slides section .gn").textContent === "Untreated"')
            first = pg.eval_on_selector_all(
                '#slides section >> nth=0 >> li .name', 'els => els.map((e) => e.textContent)'
            )
            assert first == ['S00.svs', 'S03.svs', 'S06.svs', 'S09.svs']
            assert pg.inner_text('#slides li >> nth=0 >> .sub') == '001040'  # the sort field tells slides apart
            # switch Animal to a group and back; drag it before Treatment group; remove it
            pg.click('#s-chips .lchip.sort .role')
            pg.wait_for_function('() => document.querySelectorAll("#slides .kids section").length === 10')
            pg.click('#s-chips .lchip >> nth=1 >> .role')
            pg.drag_and_drop('#s-chips .lchip >> nth=1', '#s-chips .lchip >> nth=0')
            assert pg.inner_text('#s-chips .lchip >> nth=0 >> .f') == 'Animal'
            pg.click('#s-chips .lchip >> nth=0 >> .x')
            pg.wait_for_function('() => document.querySelectorAll("#s-chips .lchip").length === 1')
            pg.click('#s-add-sort')
            pg.click('.menu button:has-text("Animal")')

            # the filter matches metadata values
            pg.fill('#s-filter', 'lnp')
            pg.wait_for_function('() => document.querySelectorAll("#slides li[draggable]").length === 3')
            pg.fill('#s-filter', '')

            # 4 panes, all 10 slides: 3 pages; only the shown page's slides are in panes
            pg.select_option('#n', '4')
            pg.wait_for_function(f'() => ({G})().panes.length === 4')
            assert pg.inner_text('#s-pages') == '3 pages × 4'
            pg.click('#s-openall')
            pg.wait_for_function('() => document.getElementById("deck-pos").textContent === "1 / 3"')
            pg.wait_for_function(f'() => ({G})().panes.every((p) => p.slide?.info)')
            assert _names(pg) == ['S00.svs', 'S03.svs', 'S06.svs', 'S09.svs']
            assert pg.inner_text('#deck-title') == 'Untreated'
            loaded = pg.evaluate(f'() => [...({G})().slides.keys()].map((u) => u.split("/").pop())')
            assert len(loaded) == 4, loaded  # the other pages are not loaded
            label = pg.evaluate(f'() => ({G})().panes[0].$label.textContent')
            assert label == 'Untreated · S00.svs'
            # zoom pane 1, go to page 2 and back: it reopens where it was left
            pg.evaluate(f'() => ({G})().goTo(0, 200, 150, {{ zoom: 4, animate: false }})')
            pg.keyboard.press('PageDown')
            pg.wait_for_function('() => document.getElementById("deck-pos").textContent === "2 / 3"')
            pg.wait_for_function(f'() => ({G})().panes.every((p) => p.slide?.info)')
            assert _names(pg) == ['S02.svs', 'S05.svs', 'S08.svs', 'S01.svs']  # pages run on across groups
            pg.click('#deck-prev')
            pg.wait_for_function(f'() => ({G})().panes[0].view && Math.abs(({G})().panes[0].view.scale - 4) < 0.05')
            # 2 panes: re-divided around the first slide shown (5 pages)
            pg.select_option('#n', '2')
            pg.wait_for_function('() => document.getElementById("deck-pos").textContent === "1 / 5"')

            # checked slides: one, then a Shift-range; Open selected → pages of the checked slides only
            pg.select_option('#n', '4')
            pg.hover('#slides li[draggable] >> nth=1')
            pg.click('#slides li[draggable] >> nth=1 >> .pick')
            pg.click('#slides li[draggable] >> nth=5 >> .pick', modifiers=['Shift'])
            pg.wait_for_function('() => document.querySelectorAll("#slides li.checked").length === 5')
            assert pg.inner_text('#s-openall') == 'Open 5 selected'
            pg.click('#slides li[draggable] >> nth=7', modifiers=['Control'])  # Ctrl-click checks too
            pg.click('#s-openall')
            pg.wait_for_function('() => document.getElementById("deck-pos").textContent === "1 / 2"')
            assert len(_names(pg)) == 4
            # a page per group: the short last page keeps 4 panes, the rest empty
            pg.click('#s-clear')
            pg.check('#s-breaks')
            pg.click('#s-openall')
            pg.wait_for_function('() => document.getElementById("deck-pos").textContent === "1 / 3"')
            pg.click('#deck-next')
            pg.wait_for_function('() => document.getElementById("deck-title").textContent === "3 MPK LNP"')
            names = _names(pg)
            assert len(names) == 4 and names[3] is None and names[:3] == ['S02.svs', 'S05.svs', 'S08.svs']
            # a group's ▸ opens that group alone
            pg.hover('#slides section >> nth=0 >> .ghead')
            pg.click('#slides section >> nth=0 >> .ghead .gopen')
            pg.wait_for_function('() => document.getElementById("deck-pos").textContent === "1 / 1"')
            assert _names(pg) == ['S00.svs', 'S03.svs', 'S06.svs', 'S09.svs']
            pg.click('#deck-close')
            pg.wait_for_function('() => document.getElementById("deck").hidden')
            assert pg.evaluate(f'() => ({G})().panes[0].$label.textContent') == 'S00.svs'
            time.sleep(0.3)
            b.close()
    finally:
        srv.stop()
        s.shutdown()
    assert not errors, errors


def test_metadata_table_by_http(data, tmp_path):
    from starlette.testclient import TestClient

    from wissi_viz.session import Session, create_session_app

    s = Session(roots=[data], remember=False)
    c = TestClient(create_session_app(s, mode='server', key='k'))
    h = {'x-wissi-token': s.token, 'x-wissi-key': 'k'}
    c.post('/app/open', json={'paths': [str(data / 'S00.svs'), str(data / 'S01.svs')]}, headers=h)
    r = c.post('/app/metadata', json={'paths': [str(data / 'meta.csv')]}, headers=h).json()
    assert r['meta_tables'][0]['matched'] == 2 and r['slides'][1]['meta']['Treatment group'] == '1 MPK Free siRNA'
    assert c.post('/app/metadata', json={'paths': [str(Path(data).parent / 'x.csv')]}, headers=h).json()['errors']
    assert c.post('/app/metadata', json={'remove': str(data / 'meta.csv')}, headers=h).json()['meta_tables'] == []
    assert c.post('/app/metadata', json={'paths': []}).status_code == 403  # the page's token is needed
