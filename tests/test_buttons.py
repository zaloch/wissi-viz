"""Every button of the app page, clicked in a real browser, does what its label or tooltip says.

The sweep walks the app through its states (no slides, the file picker, a brightfield slide with cells and regions,
a two-scene fluorescence CZI, menus, dialogs, the object card, the radial menu, running jobs) and, in each, checks
every visible button it knows. Any visible button the sweep does not check fails the test: a new button needs a
check here."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip('playwright')
pytest.importorskip('uvicorn')
pytest.importorskip('scipy')
pytest.importorskip('sklearn')

sys.path.insert(0, str(Path(__file__).parent))
from czi_writer import write_czi

MPP = 0.5


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp('buttons')
    rng = np.random.default_rng(7)
    W = H = 2400
    img = np.full((H, W, 3), 240, np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    tissue = (xx - 1200) ** 2 / 1000**2 + (yy - 1200) ** 2 / 900**2 < 1
    img[tissue] = (220, 160, 200)
    img[tissue & (xx < 1200)] = (205, 120, 185)  # two textures for the area classifier
    cells = []
    for _ in range(900):  # dark nuclei
        x, y = rng.uniform(300, 2100, 2)
        if not tissue[int(y), int(x)]:
            continue
        r = rng.uniform(5, 11)
        sl = (slice(int(y - r), int(y + r) + 1), slice(int(x - r), int(x + r) + 1))
        m = (xx[sl] - x) ** 2 + (yy[sl] - y) ** 2 < r * r
        img[sl][m] = (90, 50, 140)
        cells.append((x, y, 'Large' if r > 8 else 'Small', np.pi * (r * MPP) ** 2))
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = {MPP}'
    with tifffile.TiffWriter(d / 'he.svs', bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
        tw.write(
            img[::4, ::4].copy(), tile=(256, 256), compression='zlib', photometric='rgb', metadata=None, subfiletype=1
        )
    rows = ['x,y,class,area_um2'] + [f'{x:.1f},{y:.1f},{c},{a:.2f}' for x, y, c, a in cells]
    (d / 'cells.csv').write_text('\n'.join(rows), encoding='utf-8')
    roi = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'properties': {'classification': {'name': 'ROI'}},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[800, 800], [1600, 800], [1600, 1600], [800, 1600], [800, 800]]],
                },
            }
        ],
    }
    (d / 'regions.geojson').write_text(json.dumps(roi), encoding='utf-8')
    # a fluorescence CZI: two scenes, three 12-bit channels
    blocks, m = [], 0
    for c, peak in enumerate((3000, 1500, 800)):
        plane = np.clip(rng.normal(200, 30, (600, 1300)) + peak * (rng.random((600, 1300)) > 0.995), 0, 4095).astype(
            np.uint16
        )
        plane[:, 600:700] = 0
        for s, (x0, x1) in enumerate(((0, 600), (700, 1300))):
            blocks.append({'c': c, 'x': x0, 'y': 0, 'w': x1 - x0, 'h': 600, 'data': plane[:, x0:x1], 'm': m, 's': s})
            m += 1
    write_czi(d / 'fl.czi', blocks, ['DAPI', 'CD8', 'PanCK'], mpp=0.325)
    return d


def _js(name):
    return f'() => document.getElementById("{name}")'


G = '() => document.getElementById("grid")'


class Sweep:
    """Clicks and checks; remembers which selectors were checked, and finds visible buttons that were not."""

    def __init__(self, pg, url):
        self.pg, self.url = pg, url
        self.covered: set[str] = set()  # exercised so far
        self.missed: dict[str, str] = {}
        # every selector this file checks (plain strings; the loops over tools / tabs / magnifications use f-strings
        # and are declared with cover() before they run): a button is "known" if any of them matches it
        src = Path(__file__).read_text(encoding='utf-8')
        calls = re.findall(r"(?:w\.(?:check|cover)\(|_open_picker\(w, )((?:\s*'[^']*',?)+)", src)
        self.known = {q for c in calls for q in re.findall(r"'([^']*)'", c) if q.startswith(('#', '.'))}

    def check(self, selector: str, cond, *, nth: int | None = None, timeout: int = 15000, button: str = 'left'):
        """Click ``selector`` (the first visible one, or the nth) and wait until ``cond`` (JS) is true."""
        self.covered.add(selector)
        loc = self.pg.locator(selector)
        loc = loc.nth(nth) if nth is not None else loc.first
        loc.click(button=button)
        if cond:
            try:
                self.pg.wait_for_function(cond, timeout=timeout)
            except Exception as e:
                state = self.pg.evaluate(
                    """() => ({ log: document.getElementById("log").textContent,
                        picker: document.getElementById("picker").className, toasts: [...document.querySelectorAll(".toast")].map((t) => t.textContent),
                        panes: (document.getElementById("grid").panes || []).map((p) => p.slide?.info?.id || null) })"""
                )
                raise AssertionError(f'{selector}: {cond} never became true; page: {state}') from e

    def cover(self, *selectors: str) -> None:
        self.covered.update(selectors)

    def audit(self, where: str) -> None:
        """Visible, enabled buttons on the page that no check covered so far."""
        found = self.pg.evaluate(
            r"""(covered) => {
                const vis = (e) => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
                    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && !e.closest('[hidden]'); };
                const all = [...document.querySelectorAll('button, [role=button], a[href="#"]')].filter((e) => vis(e) && !e.disabled);
                const match = (e, sel) => {   // CSS plus Playwright's :has-text("…"); ">> …" parts are dropped
                    sel = sel.split('>>')[0].trim();
                    const t = /:has-text\("([^"]*)"\)/.exec(sel);
                    const base = sel.replace(/:has-text\("[^"]*"\)/g, '').trim() || '*';
                    try { return e.matches(base) && (!t || e.textContent.includes(t[1])); } catch (_) { return false; }
                };
                return all.filter((e) => !covered.some((s) => match(e, s)))
                    .map((e) => (e.id ? '#' + e.id : e.tagName.toLowerCase() + (e.className ? '.' + String(e.className).split(' ').join('.') : ''))
                        + ' «' + (e.getAttribute('aria-label') || e.textContent || '').trim().slice(0, 40) + '» in '
                        + (e.closest('[id]')?.id || '?'));
            }""",
            sorted(self.known | self.covered),
        )
        for f in found:
            self.missed.setdefault(f, where)


class _FakeText:
    """Stands in for the text-prompted models (no download in the sweep): three objects in every tile."""

    name, label, tile, masks = 'fake', 'fake model', 800, True

    def detect(self, rgb, names, min_score=0.1):
        h, w = rgb.shape[:2]
        boxes = np.array([[w * f - 10, h * f - 10, w * f + 10, h * f + 10] for f in (0.3, 0.5, 0.7)], np.float32)
        masks = np.zeros((3, h, w), bool)
        for k, (x0, y0, x1, y1) in enumerate(boxes.astype(int)):
            masks[k, max(0, y0) : y1, max(0, x0) : x1] = True
        return boxes, np.array([0.6, 0.5, 0.4], np.float32), np.zeros(3, np.int64), masks


def test_every_button_does_what_it_says(data, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright

    from wissi_viz import desktop as D
    from wissi_viz import findall, sam3

    monkeypatch.setattr(findall, 'text_detector', lambda model='grounding', progress=None: _FakeText())
    monkeypatch.setattr(sam3, 'user_dir', lambda kind: tmp_path / kind)  # the licence choice stays in the test
    monkeypatch.delenv('WISSI_SAM3_ACCEPT_LICENSE', raising=False)

    s = D.Session(cache_dir=tmp_path / 'cache', remember=False)
    srv = D._Server(s.app())
    url = srv.start()
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:
                pytest.skip(f'no Chromium: {e}')
            ctx = b.new_context(viewport={'width': 1500, 'height': 920}, accept_downloads=True,
                                permissions=['clipboard-read', 'clipboard-write'])  # fmt: skip
            pg = ctx.new_page()
            slow = float(os.environ.get('WISSI_SWEEP_CPU_SLOWDOWN', '0') or 0)
            if slow > 1:  # find races: a slow machine (e.g. WISSI_SWEEP_CPU_SLOWDOWN=4)
                ctx.new_cdp_session(pg).send('Emulation.setCPUThrottlingRate', {'rate': slow})
            pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            w = Sweep(pg, url)
            _empty_and_picker(w, data, s)
            _brightfield(w, data, s)
            _fluorescence(w, data, s)
            _slide_list(w, data, s)
            b.close()
    finally:
        srv.stop()
        s.shutdown()
    assert not errors, errors
    assert not w.missed, 'buttons without a check in tests/test_buttons.py:\n' + '\n'.join(
        f'{k}  ({v})' for k, v in w.missed.items()
    )
    assert not w.known - w.covered, f'checks that never ran: {sorted(w.known - w.covered)}'


# ------------------------------------------------------------------------------------------------ states
def _open_picker(w, act_selector):
    w.check(act_selector, '() => document.getElementById("picker").classList.contains("on")')


def _empty_and_picker(w, data, s):
    pg = w.pg
    pg.goto(w.url + '/app')
    pg.wait_for_selector('#empty')
    w.pg.wait_for_function('() => document.getElementById("log").textContent.length > 0')
    # header / empty-screen actions that need no slide
    w.check('#b-help', '() => document.getElementById("help").open')
    w.check('#help form button', '() => !document.getElementById("help").open')
    w.check('#b-connect', '() => document.getElementById("connect").open')
    w.check('#c-go', '() => /service description/.test(document.getElementById("c-msg").textContent)')  # validation
    w.check('#c-form button[value=cancel]', '() => !document.getElementById("connect").open')
    w.check('#empty button[data-act=connect]', '() => document.getElementById("connect").open')
    pg.keyboard.press('Escape')
    w.check('#b-side', '() => document.getElementById("side").classList.contains("hidden")')
    w.check('#b-side', '() => !document.getElementById("side").classList.contains("hidden")')
    w.check('#b-insp', '() => document.getElementById("insp").classList.contains("hidden")')
    w.check('#b-insp', '() => !document.getElementById("insp").classList.contains("hidden")')
    # the file picker: open slides / open folder / places / up / filter-path / cancel
    _open_picker(w, '#empty button[data-act=open]')
    w.check('#pk-places button', '() => document.querySelector("#pk-places button.on")')
    # a folder asked for while another is still listing wins (the slow Home listing must not overwrite it)
    pg.click('#pk-places button')
    pg.fill('#pk-filter', str(data))
    pg.keyboard.press('Enter')
    pg.wait_for_selector('#browser a:has-text("he.svs")')
    w.check('#pk-up', f'() => document.getElementById("browser").dataset.dir === {json.dumps(str(data.parent))}')
    pg.fill('#pk-filter', str(data))
    pg.keyboard.press('Enter')
    pg.wait_for_selector('#browser a:has-text("he.svs")')
    w.audit('file picker')
    w.check('#pk-close', '() => !document.getElementById("picker").classList.contains("on")')
    _open_picker(w, '#b-open')
    assert pg.is_hidden('#pk-folder') is False and 'Open' in pg.inner_text('#pk-open')
    pg.click('#pk-close')
    _open_picker(w, '#b-folder')
    assert 'Open this folder' in pg.inner_text('#pk-folder')
    pg.click('#pk-close')
    _open_picker(w, '#empty button[data-act=folder]')
    assert pg.is_visible('#pk-folder')
    pg.fill('#pk-filter', str(data))
    pg.keyboard.press('Enter')
    pg.wait_for_selector('#browser a:has-text("fl.czi")')
    w.check('#pk-crumbs button', None)  # crumbs: a parent folder
    pg.wait_for_function(f'() => document.getElementById("browser").dataset.dir !== {json.dumps(str(data))}')
    pg.fill('#pk-filter', str(data))
    pg.keyboard.press('Enter')
    pg.wait_for_selector('#browser a:has-text("fl.czi")')
    w.check('#browser a[data-kind=folder]', None) if pg.locator('#browser a[data-kind=folder]').count() else w.cover(
        '#browser a[data-kind=folder]'
    )
    pg.fill('#pk-filter', str(data))
    pg.keyboard.press('Enter')
    pg.wait_for_selector('#browser a:has-text("fl.czi")')
    w.check('#pk-folder', '() => document.querySelectorAll("#slides li").length === 2')  # every slide in the folder
    assert not pg.is_visible('#picker')
    w.cover('#browser a')


def _brightfield(w, data, s):
    pg = w.pg
    sid = next(x['id'] for x in s.slides() if x['name'] == 'he.svs')
    _open_picker(w, '#b-open')  # the paste box: a path, then Open
    pg.fill('#paths', str(data / 'he.svs'))
    w.check(
        '#b-paths',
        f'() => ({G})().panes.some((p) => p.slide?.info?.id === {json.dumps(sid)}) && !document.getElementById("picker").classList.contains("on")',
    )
    pg.select_option('#n', '1')
    pg.click('#slides li[title$="he.svs"]')
    pg.wait_for_function(f'() => ({G})().panes.length === 1 && ({G})().panes[0].slide?.info?.id === {json.dumps(sid)}')
    # add cells + regions through the picker in layer mode (the "add layer" buttons)
    for btn in ('#b-layer', '#l-add'):
        _open_picker(w, btn)
        assert 'Add' in pg.inner_text('#pk-open')
        w.check('#pk-close', '() => !document.getElementById("picker").classList.contains("on")')
    _open_picker(w, '#b-layer')
    pg.fill('#pk-filter', str(data))
    pg.keyboard.press('Enter')
    pg.click('#browser a:has-text("cells.csv")')
    pg.click('#browser a:has-text("regions.geojson")', modifiers=['Shift'])
    w.check(
        '#pk-open', f'() => ({G})().panes[0].slide.layers.has("cells") && ({G})().panes[0].slide.layers.has("regions")'
    )
    pg.wait_for_selector('#l-list .lcard[data-layer="cells"]')
    vis = f'(n) => {{ const p = ({G})().panes[0]; const L = p.slide.layers.get(n); return !!L && p.layerVisible(L); }}'
    view = f'() => ({G})().panes[0].view'
    # ---- toolbar tools 1–8
    w.cover(
        '#tools .slot',
        '#vtools [data-v=mag]',
        '#tabs button',
        '.lcard .eye',
        '.lcard .solo-b',
        '.lcard .nm',
        '.ghead .tw',
    )
    for t in ('point', 'rect', 'ellipse', 'polygon', 'freehand', 'line', 'sam', 'pan'):
        w.check(f'#tools .slot[data-tool={t}]', f'() => ({G})().tool === "{t}"')
    # ---- view tools on the slide
    sc0 = pg.evaluate(f'() => ({view})().scale')
    w.check('#vtools [data-v=in]', f'() => ({G})().panes[0].view.scale > {sc0 * 1.9}')
    w.check('#vtools [data-v=out]', f'() => Math.abs(({G})().panes[0].view.scale / {sc0} - 1) < 0.05')
    w.check('#vtools [data-v=one]', f'() => Math.abs(({G})().panes[0].view.scale * devicePixelRatio - 1) < 0.02')
    for m in ('2.5', '5', '10', '20', '40'):
        w.check(
            f'#vtools [data-v=mag][data-m="{m}"]',
            f'() => document.getElementById("v-mag").textContent.startsWith("{m.split(".")[0]}")',
        )
    w.check('#vtools [data-v=fit]', f'() => ({G})().panes[0].view.scale < 0.6')
    w.check('#fit', f'() => ({G})().panes[0].view.scale < 0.6')
    w.check('#vtools [data-v=map]', '() => !document.getElementById("minimap").hidden')
    w.check('#vtools [data-v=map]', '() => document.getElementById("minimap").hidden')
    w.check('#vtools [data-v=scale]', f'() => ({G})().getAttribute("scalebar") === "false"')
    w.check(
        '#vtools [data-v=scale]',
        f'() => ({G})().getAttribute("scalebar") === "true" && ({G})().panes[0].$scale.style.display !== "none"',
    )
    with pg.expect_download() as dl:
        w.check('#vtools [data-v=snap]', None)
    assert dl.value.suggested_filename.endswith('.png')
    w.check('#vtools [data-v=full]', '() => !!document.fullscreenElement')
    w.check('#vtools [data-v=full]', '() => !document.fullscreenElement')
    w.check('#fps', '() => document.getElementById("perf").style.display === "block"')
    w.check('#fps', '() => document.getElementById("perf").style.display === "none"')
    # ---- inspector tabs
    for t in ('image', 'draw', 'analyze', 'measure', 'layers'):
        w.check(f'#tabs [data-tab={t}]', f'() => document.getElementById("p-{t}").classList.contains("on")')
    # ---- Layers tab
    w.check(
        '#l-scope [data-scope=all]',
        '() => document.querySelector("#l-scope [data-scope=all]").getAttribute("aria-pressed") === "true"',
    )
    w.check(
        '#l-scope [data-scope=pane]',
        '() => document.querySelector("#l-scope [data-scope=pane]").getAttribute("aria-pressed") === "true"',
    )
    w.check('#l-mode [data-mode=flat]', '() => document.querySelector("#l-list .lflat")')
    w.check('.lflat .eye', f'() => !({vis})("{pg.locator(".lflat").first.get_attribute("data-layer")}")')
    w.check('.lflat .eye', None)
    w.check('#l-mode [data-mode=source]', '() => document.querySelector("#l-list .lgroup-h")')
    w.check('.lgroup-h button', f'() => !({vis})("cells") && !({vis})("regions")')
    w.check('.lgroup-h button', f'() => ({vis})("cells") && ({vis})("regions")')
    w.check('#l-mode [data-mode=nested]', '() => !document.querySelector("#l-list .lgroup-h, #l-list .lflat")')
    w.check(
        '#l-chips .fchip[data-src=imported]',
        '() => document.querySelector("#l-chips [data-src=imported]").getAttribute("aria-pressed") === "true"',
    )
    w.check(
        '#l-chips .fchip[data-src=all]',
        '() => document.querySelector("#l-chips [data-src=all]").getAttribute("aria-pressed") === "true"',
    )
    w.check('.lcard[data-layer="regions"] .eye', f'() => !({vis})("regions")')
    w.check('.lcard[data-layer="regions"] .eye', f'() => ({vis})("regions")')
    w.check('.lcard[data-layer="cells"] .solo-b', f'() => ({vis})("cells") && !({vis})("regions")')
    w.check('.lcard[data-layer="cells"] .solo-b', f'() => ({vis})("regions")')
    w.check('.lcard[data-layer="cells"] .nm', '() => document.querySelector(\'.lcard[data-layer="cells"] .lbody\')')
    pg.wait_for_selector('.lcard[data-layer="cells"] .cls .chip')
    w.check(
        '.lcard[data-layer="cells"] .cls .box', f'() => ({G})().panes[0].slide.layers.get("cells").hidden?.size === 1'
    )
    w.check('.lcard[data-layer="cells"] .cls .box', f'() => !({G})().panes[0].slide.layers.get("cells").hidden?.size')
    # layer ⋯ menu: every item
    menu = lambda: pg.click('.lcard[data-layer="regions"] .lhead button[aria-label^="More"]')
    w.cover('.lhead button[aria-label^="More"]', '.lflat button[aria-label^="More"]')
    menu()
    w.audit('layer menu')
    sc = pg.evaluate(f'() => ({view})().scale')
    w.check('.menu button:has-text("Zoom to layer")', f'() => Math.abs(({G})().panes[0].view.scale - {sc}) > 0.01')
    menu()
    with pg.expect_download() as dl:
        w.check('.menu button:has-text("Export as")', None)
    assert dl.value.suggested_filename
    menu()
    w.check('.menu button:has-text("Summary")', '() => document.getElementById("p-measure").classList.contains("on")')
    pg.click('#t-layers')
    menu()
    w.check(
        '.menu button:has-text("Measure its regions")', '() => document.getElementById("m-regions").value === "regions"'
    )
    pg.click('#t-layers')
    menu()
    w.check(
        '.menu button:has-text("Analyse these cells")',
        '() => document.getElementById("p-analyze").classList.contains("on")',
    )
    w.cover('.menu button:has-text("Remove from slide")')  # checked on a found layer below
    pg.click('#t-layers')
    # chain steps: cells → regions (merged), cells → only those inside a region layer
    cells_menu = lambda: pg.click('.lcard[data-layer="cells"] .lhead button[aria-label^="More"]')
    cells_menu()
    w.check('.menu button:has-text("Make regions")', '() => document.getElementById("lo").open')
    w.check('#lo-cancel', '() => !document.getElementById("lo").open')
    cells_menu()
    pg.click('.menu button:has-text("Make regions")')
    w.check('#lo-run', f'() => !document.getElementById("lo").open && ({G})().panes[0].slide.layers.has("Regions · cells")',
            timeout=30000)  # fmt: skip
    pg.click('#t-layers')
    cells_menu()
    w.check('.menu button:has-text("Keep only inside")', '() => document.getElementById("lo").open')
    pg.select_option('#lo-body select >> nth=0', 'regions')
    pg.click('#lo-run')
    pg.wait_for_function(f'() => ({G})().panes[0].slide.layers.has("cells in regions")', timeout=30000)
    pg.click('#t-layers')
    w.check(  # the made regions are listed with found layers
        '#l-chips .fchip[data-src=found]',
        '() => document.querySelector("#l-chips [data-src=found]").getAttribute("aria-pressed") === "true" '
        '&& document.querySelector(\'.lcard[data-layer="Regions · cells"]\')',
    )
    pg.click('#l-chips .fchip[data-src=all]')
    w.audit('layers tab')
    # ---- object card and radial menu
    L = s.registry.layer(next(x['id'] for x in s.slides() if x['name'] == 'he.svs'), 'cells')
    c = L.compiled()
    pg.evaluate(f'([x, y]) => ({G})().goTo(0, x, y, {{zoom: 2, animate: false}})', [float(c.cx[5]), float(c.cy[5])])
    pg.wait_for_timeout(500)
    box = pg.evaluate(f"""() => {{ const e = ({G})(), p = e.panes[0], r = e.getBoundingClientRect();
        return [r.left + p.rect.x + p.rect.w / 2, r.top + p.rect.y + p.rect.h / 2]; }}""")
    pg.mouse.click(*box)
    pg.wait_for_selector('#card .bars, #card b')
    w.audit('object card')
    w.check(
        '#card button:has-text("Copy")',
        '() => [...document.querySelectorAll(".toast b")].some((b) => b.textContent === "Copied")',
    )
    w.check(
        '.toast button[aria-label=Dismiss]',
        '() => ![...document.querySelectorAll(".toast b")].some((b) => b.textContent === "Copied")',
    )
    pg.evaluate(f'([x, y]) => ({G})().goTo(0, x, y, {{zoom: 0.5, animate: false}})', [float(c.cx[5]), float(c.cy[5])])
    w.check(
        '#card button:has-text("Zoom in")', f'() => ({G})().panes[0].view.scale > 1.5'
    )  # 40×: 0.25 µm per screen px
    pg.evaluate(f'([x, y]) => ({G})().goTo(0, x, y, {{zoom: 2, animate: false}})', [float(c.cx[5]), float(c.cy[5])])
    pg.wait_for_timeout(400)
    pg.mouse.click(*box)
    pg.wait_for_selector('#card .btn')
    w.check('#card button[aria-label=Close]', '() => document.getElementById("card").style.display === "none"')
    radial = lambda: (pg.mouse.click(*box, button='right'), pg.wait_for_selector('#radial[style*="block"]'))
    radial()
    w.audit('radial menu')
    w.check(
        '#radial [data-r=copy]',
        '() => [...document.querySelectorAll(".toast b")].some((b) => b.textContent === "Copied")',
    )
    radial()
    w.check('#radial [data-r=hide]', f'() => !({vis})("cells")')
    pg.click('.lcard[data-layer="cells"] .eye')
    radial()
    w.check('#radial [data-r=solo]', f'() => ({vis})("cells") && !({vis})("regions")')
    radial()
    w.check('#radial [data-r=solo]', f'() => ({vis})("regions")')
    radial()
    w.check('#radial [data-r=sam]', f'() => ({G})().tool === "sam"')
    pg.keyboard.press('1')
    radial()
    w.check('#radial [data-r=measure]', '() => document.getElementById("p-measure").classList.contains("on")')
    radial()
    w.check('#radial [data-r=fit]', f'() => ({G})().panes[0].view.scale < 0.6')
    # ---- Measure
    w.check('#m-run', '() => document.querySelector("#mtable table")', timeout=60000)
    w.cover('#m-csv')
    # ---- Analyze
    pg.click('#t-analyze')
    pg.wait_for_selector('.an-card')
    w.audit('analyze tab')
    w.cover('.an-card')
    w.check('.an-card:has-text("Density")', '() => document.getElementById("an-run")')
    w.check('#an-run', '() => document.querySelector("#an-results .result .chart svg")', timeout=90000)
    w.audit('analysis result')
    # the same analysis on the cells inside a region only
    pg.select_option('#an-within', 'regions')
    pg.wait_for_selector('.an-card:has-text("Density")')
    pg.click('.an-card:has-text("Density")') if not pg.is_visible('#an-run') else None
    pg.click('#an-run')
    pg.wait_for_function(
        '() => /in regions/.test(document.querySelector("#an-results .result b").textContent)', timeout=90000
    )
    pg.select_option('#an-within', '')
    w.check(
        '#an-results .result button:has-text("Table")',
        '() => document.querySelector("#an-results .tablebox:not([hidden]) table")',
    )
    with pg.expect_download() as dl:
        w.check('#an-results .result button:has-text("CSV")', None)
    assert dl.value.suggested_filename.endswith('.csv')
    w.check(
        '#an-results .result button:has-text("Show layer")',
        '() => document.getElementById("p-layers").classList.contains("on")',
    )
    w.cover('.toast button.btn:not([aria-label])')  # a toast's own action ("Show"): the same as the result's
    # ---- Annotate: SAM, accept / discard, drawing, picking, find all (objects, areas, cancel), result cards
    pg.click('#t-draw')
    w.check('#a-sam', f'() => ({G})().tool === "sam"')
    pg.evaluate(f'([x, y]) => ({G})().goTo(0, x, y, {{zoom: 1, animate: false}})', [float(c.cx[5]), float(c.cy[5])])
    pg.wait_for_timeout(500)
    pg.mouse.click(*box)
    pg.wait_for_function(f'() => ({G})().panes[0].sam?.polys?.length')
    w.check('#a-discard', f'() => !({G})().panes[0].sam')
    pg.mouse.click(*box)
    pg.wait_for_function(f'() => ({G})().panes[0].sam?.polys?.length')
    w.check('#a-accept', f'() => ({G})().panes[0].slide.layers.get("Annotations")?.meta.count >= 1')
    w.check('#a-sam', f'() => ({G})().tool === "pan"')
    # two classes of rectangles for the area classifier
    pg.evaluate(f'() => ({G})().fit(0, false)')
    pg.wait_for_timeout(300)
    ex = pg.evaluate(f"""() => {{ const e = ({G})(), p = e.panes[0], r = e.getBoundingClientRect(), v = p.view;
        const sx = (x) => r.left + p.rect.x + p.rect.w / 2 + (x - v.cx) * v.scale, sy = (y) => r.top + p.rect.y + p.rect.h / 2 + (y - v.cy) * v.scale;
        return [sx(500), sy(1000), sx(800), sy(1300), sx(1500), sy(1000), sx(1800), sy(1300)]; }}""")
    for k, cls in enumerate(('Left', 'Right')):
        pg.fill('#a-class', cls)
        pg.dispatch_event('#a-class', 'change')
        pg.click('#tools .slot[data-tool=rect]')
        pg.mouse.move(ex[4 * k], ex[4 * k + 1])
        pg.mouse.down()
        pg.mouse.move(ex[4 * k + 2], ex[4 * k + 3], steps=4)
        pg.mouse.up()
        pg.wait_for_timeout(500)
    pg.click('#tools .slot[data-tool=pan]')
    w.check('#f-modes [data-mode=areas]', '() => !document.getElementById("f-detail-row").hidden')
    pg.select_option('#f-detail', 'coarse')
    pg.select_option('#f-where', 'slide')
    pg.wait_for_function('() => !document.getElementById("f-run").disabled')
    w.check('#f-run', '() => document.querySelector("#f-results .result")', timeout=120000)
    pg.wait_for_function(f'() => [...({G})().panes[0].slide.layers.keys()].some((n) => n.startsWith("Areas"))')
    w.check('#f-modes [data-mode=segment]', '() => !document.getElementById("f-size-row").hidden')
    pg.select_option('#f-where', 'slide')
    pg.fill('#f-size', '4')
    pg.dispatch_event('#f-size', 'input')
    pg.wait_for_function('() => !document.getElementById("f-run").disabled')
    pg.click('#f-run')
    pg.wait_for_selector('#f-cancel', state='visible')
    w.check('#f-cancel', '() => document.getElementById("f-job").hidden', timeout=60000)
    # by text: a prompt, the model choice, SAM 3's licence (shown, declined, accepted, withdrawn), a run
    w.check('#f-modes [data-mode=text]', '() => !document.getElementById("f-text-row").hidden')
    pg.wait_for_function('() => document.getElementById("f-run").disabled')  # nothing typed yet
    pg.select_option('#f-model', 'sam3')
    pg.wait_for_selector('#f-lic', state='visible')
    w.check(
        '#f-lic',
        '() => document.getElementById("sam3-lic").open && /Grant of Rights/.test(document.getElementById("lic-text").textContent)',
    )
    w.check('#lic-cancel', '() => !document.getElementById("sam3-lic").open')
    pg.click('#f-lic')
    pg.wait_for_function('() => /Grant of Rights/.test(document.getElementById("lic-text").textContent)')
    w.check(
        '#lic-accept',
        '() => !document.getElementById("sam3-lic").open && /accepted/.test(document.getElementById("f-lic-msg").textContent)',
    )
    pg.click('#f-lic')
    pg.wait_for_selector('#lic-withdraw', state='visible')
    w.check(
        '#lic-withdraw',
        '() => !document.getElementById("sam3-lic").open && document.getElementById("f-model").value === "grounding"',
    )
    pg.fill('#f-prompt', 'nucleus. gland')
    pg.dispatch_event('#f-prompt', 'input')
    pg.select_option('#f-where', 'view')
    pg.wait_for_function('() => !document.getElementById("f-run").disabled')
    n_before = pg.locator('#f-results .result').count()
    w.check('#f-run', f'() => document.querySelectorAll("#f-results .result").length > {n_before}', timeout=60000)
    assert pg.locator('#f-results .result >> nth=0').get_attribute('data-layer').startswith('Text · nucleus, gland')
    w.check('#f-modes [data-mode=objects]', '() => !document.getElementById("f-cand-row").hidden')
    w.check('#a-pick', '() => document.body.classList.contains("picking")')
    for k in (5, 6, 7):
        pg.evaluate(f'([x, y]) => ({G})().goTo(0, x, y, {{zoom: 2, animate: false}})', [float(c.cx[k]), float(c.cy[k])])
        pg.wait_for_timeout(400)
        pg.mouse.click(*box)
    pg.wait_for_function('() => document.querySelectorAll("#ex-picks .exc").length === 3')
    w.check('#ex-picks .exc button', '() => document.querySelectorAll("#ex-picks .exc").length === 2')
    w.check('#a-pick', '() => !document.body.classList.contains("picking")')
    pg.select_option('#f-cand', 'cells')
    pg.select_option('#f-where', 'slide')
    pg.wait_for_function('() => !document.getElementById("f-run").disabled')
    n_before = pg.locator('#f-results .result').count()
    w.check('#f-run', f'() => document.querySelectorAll("#f-results .result").length > {n_before}', timeout=90000)
    first = '#f-results .result >> nth=0'
    found = pg.locator(first).get_attribute('data-layer')
    pg.click(f'.lcard[data-layer="{found}"] .eye') if pg.is_visible(f'.lcard[data-layer="{found}"] .eye') else None
    w.check('#f-results .result button:has-text("Show")', f'() => ({vis})({json.dumps(found)})')
    w.audit('annotate tab')
    w.check('#ex-picks button:has-text("clear")', '() => !document.querySelector("#ex-picks .exc")')
    w.check(
        '#f-results .result button:has-text("Remove")', f'() => !({G})().panes[0].slide.layers.has({json.dumps(found)})'
    )
    # the layer menu's Remove (two clicks)
    pg.click('#t-layers')
    area = pg.evaluate(f'() => [...({G})().panes[0].slide.layers.keys()].find((n) => n.startsWith("Areas"))')
    pg.click(f'.lcard[data-layer="{area}"] .lhead button[aria-label^="More"]')
    pg.click('.menu button:has-text("Remove from slide")')
    w.check('.menu button:has-text("Click again")', f'() => !({G})().panes[0].slide.layers.has({json.dumps(area)})')
    # Image tab, brightfield
    pg.click('#t-image')
    pg.wait_for_selector('#disp-body .hist canvas')
    w.check('#i-auto', f'() => Math.abs(({G})().panes[0].adjust.contrast - 1) > 0.01')
    w.check('#i-reset', f'() => ({G})().panes[0].adjust.contrast === 1')
    w.audit('image tab (brightfield)')


def _fluorescence(w, data, s):
    pg = w.pg
    pg.click('#slides li[title$="fl.czi"]')
    pg.wait_for_function(f'() => ({G})().panes[0].slide?.info?.mode === "channels"')
    pg.click('#t-image')
    pg.wait_for_selector('#disp-body .hist[data-ch="2"] canvas')
    assert pg.locator('#i-scope option').count() == 4  # whole image, view, two scenes
    ch = f'(i) => ({G})().panes[0].channels[i]'
    w.check(
        '#disp-body .hist[data-ch="1"] .solo-b:has-text("S")',
        f'() => !({ch})(0).visible && ({ch})(1).visible && !({ch})(2).visible',
    )
    w.check('#disp-body .hist[data-ch="1"] .solo-b:has-text("S")', f'() => ({ch})(0).visible && ({ch})(2).visible')
    pg.evaluate(f'() => ({G})().setChannel(0, 0, {{min: 0, max: 60000}})')
    w.check('#disp-body .hist[data-ch="0"] .solo-b:has-text("A")', f'() => ({ch})(0).hi < 5000')
    pg.evaluate(f'() => ({G})().setChannel(0, 1, {{min: 0, max: 60000}})')
    w.check('#i-auto', f'() => ({ch})(1).hi < 5000')
    w.check('#i-link', '() => document.getElementById("i-link").getAttribute("aria-pressed") === "true"')
    pg.fill('#disp-body .hist[data-ch="0"] input[aria-label$="maximum"]', '1234')
    pg.dispatch_event('#disp-body .hist[data-ch="0"] input[aria-label$="maximum"]', 'change')
    pg.wait_for_function(f'() => [0, 1, 2].every((i) => Math.round(({ch})(i).hi) === 1234)')  # linked
    w.check('#i-link', '() => document.getElementById("i-link").getAttribute("aria-pressed") === "false"')
    w.check('#i-reset', f'() => Math.round(({ch})(0).hi) !== 1234')
    w.check('#i-scenes a:has-text("Scene 2")', '() => document.getElementById("i-scope").value === "scene:1"')
    w.cover('#i-scenes a', '#disp-body .solo-b')  # every scene link / channel's A and S work like the ones checked
    w.audit('image tab (fluorescence)')


def _slide_list(w, data, s):
    pg = w.pg
    # projects: new (named inline), drag a slide in, group menus, slide menus, collapse
    w.check('#s-newproj', '() => document.querySelector("#slides section[data-kind=project] input")')
    pg.keyboard.type('Cohort A')
    pg.keyboard.press('Enter')
    pg.wait_for_selector('#slides section[data-kind=project] .gn:has-text("Cohort A")')
    pg.locator('#slides section[data-kind=folder] li').first.drag_to(pg.locator('#slides section[data-kind=project]'))
    pg.wait_for_function(
        '() => document.querySelectorAll("#slides section[data-kind=project] li[draggable]").length === 1'
    )
    assert s.projects[0]['name'] == 'Cohort A' and len(s.projects[0]['slides']) == 1
    w.audit('slide list')
    w.check(
        '#slides section[data-kind=project] .ghead .tw',
        '() => document.querySelector("#slides section[data-kind=project]").classList.contains("closed")',
    )
    w.check(
        '#slides section[data-kind=project] .ghead .tw',
        '() => !document.querySelector("#slides section[data-kind=project]").classList.contains("closed")',
    )
    w.cover('.ghead .gn')
    pg.hover('#slides section[data-kind=folder] .ghead')
    pg.click('#slides section[data-kind=folder] .ghead .more')
    w.check(
        '.menu button:has-text("Make a project of these")',
        '() => document.querySelectorAll("#slides section[data-kind=project]").length === 2',
    )
    pg.keyboard.press('Enter')
    pg.hover('#slides section[data-kind=project] >> nth=0 >> .ghead')
    pg.click('#slides section[data-kind=project] >> nth=0 >> .ghead .more')
    w.audit('project menu')
    w.check('.menu button:has-text("Show in panes")', f'() => ({G})().panes.length >= 1')
    pg.hover('#slides section[data-kind=project] >> nth=0 >> .ghead')
    pg.click('#slides section[data-kind=project] >> nth=0 >> .ghead .more')
    w.check(
        '.menu button:has-text("Rename")',
        '() => document.activeElement?.matches("#slides section[data-kind=project] input")',
    )
    pg.keyboard.press('Escape')
    pg.wait_for_function('() => !document.querySelector("#slides section[data-kind=project] input")')
    pg.hover('#slides li[draggable] >> nth=0')
    pg.click('#slides li[draggable] .more >> nth=0')
    w.audit('slide menu')
    n = pg.evaluate(f'() => ({G})().panes.length')
    w.check(
        '.menu button:has-text("Add as a new pane")',
        f'() => ({G})().panes.length === {n + 1} || document.getElementById("n").value === "{n}"',
    )
    pg.hover('#slides li[draggable] >> nth=0')
    pg.click('#slides li[draggable] .more >> nth=0')
    w.check('.menu button:has-text("Show in the active pane")', None)
    pg.hover('#slides li[draggable] >> nth=0')
    pg.click('#slides li[draggable] .more >> nth=0')
    w.cover('.menu button:has-text("Add to")', '.menu button:has-text("New project with this slide")')
    if pg.locator('.menu button:has-text("Remove from this project")').count():
        w.check('.menu button:has-text("Remove from this project")', None)
    else:
        pg.keyboard.press('Escape')
    pg.hover('#slides section[data-kind=project] >> nth=0 >> .ghead')
    pg.click('#slides section[data-kind=project] >> nth=0 >> .ghead .more')
    k = len(s.projects)
    w.check(
        '.menu button:has-text("Delete project")',
        f'() => document.querySelectorAll("#slides section[data-kind=project]").length === {k - 1}',
    )
    w.check('#slides li[draggable]', None)  # click: show in the active pane
    n_open = len(s.slides())
    pg.hover('#slides li[draggable] >> nth=0')
    w.check('#slides li[draggable] button[aria-label^="Close"]', None)
    pg.wait_for_function(f'() => document.querySelectorAll("#slides li[draggable]").length === {n_open - 1}')
    assert len(s.slides()) == n_open - 1
    w.cover('.more', '#slides li button')
    w.audit('end')
