"""Click / box segmentation: the model-free backend always works; SAM 2.1 (torch / onnx) when installed."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('skimage')

from wissi_viz import open_slide  # noqa: E402
from wissi_viz.sam import SamService  # noqa: E402


@pytest.fixture(scope='module')
def slide(tmp_path_factory):
    import tifffile
    d = tmp_path_factory.mktemp('sam')
    img = np.full((3000, 3000, 3), (235, 170, 205), np.uint8)       # eosin-pink background
    yy, xx = np.mgrid[0:3000, 0:3000]
    for cx, cy, r in ((1000, 1000, 40), (1200, 1050, 25), (2000, 2200, 60)):
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = (80, 50, 140)  # haematoxylin-dark nuclei
    desc = 'Aperio Image Library v12 \r\n3000x3000 [0,0 3000x3000] (256x256) JPEG/RGB Q=95|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(d / 's.svs', bigtiff=True) as tw:
        tw.write(img, tile=(256, 256), photometric='rgb', description=desc, metadata=None)
        tw.write(img[::4, ::4].copy(), tile=(256, 256), photometric='rgb', description='Aperio level', metadata=None)
    return open_slide(d / 's.svs')


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2


def test_classic_click_and_box(slide):
    svc = SamService(backend='classic')
    polys, info = svc.segment(slide, rect=(700, 700, 1500, 1500), points=[(1000, 1000)], labels=[1], key='s')
    assert info['backend'] == 'classic' and info['level'] == 0 and len(polys) == 1
    assert _area(polys[0]) == pytest.approx(math.pi * 40 ** 2, rel=0.1)
    c = polys[0].mean(0)
    assert abs(c[0] - 1000) < 3 and abs(c[1] - 1000) < 3
    polys2, info2 = svc.segment(slide, rect=(700, 700, 1500, 1500), box=(1160, 1010, 1240, 1090), key='s')
    assert info2['cached'] and _area(polys2[0]) == pytest.approx(math.pi * 25 ** 2, rel=0.15)
    # a whole-slide view is read from the 4× level and still lands in level-0 px
    polys3, info3 = svc.segment(slide, rect=(0, 0, 3000, 3000), points=[(2000, 2200)], labels=[1], key='s')
    assert 2.5 < info3['downsample'] < 3.2 and _area(polys3[0]) == pytest.approx(math.pi * 60 ** 2, rel=0.2)


def test_negative_click_removes_a_part(slide):
    svc = SamService(backend='classic')
    polys, _ = svc.segment(slide, rect=(700, 700, 1500, 1500), points=[(1000, 1000), (1200, 1050)], labels=[1, 1], key='n')
    assert len(polys) == 2
    polys, _ = svc.segment(slide, rect=(700, 700, 1500, 1500), points=[(1000, 1000), (1200, 1050), (1200, 1050)],
                           labels=[1, 1, 0], key='n')
    assert len(polys) == 1


def test_sam_endpoint(slide):
    from starlette.testclient import TestClient

    from wissi_viz import SlideRegistry, TileService, sam
    from wissi_viz.server import create_app
    sam._SERVICE = SamService(backend='classic')
    reg = SlideRegistry(TileService())
    sid = reg.add(slide.path)
    c = TestClient(create_app(reg))
    st = c.get('/wissi-viz/sam/status', params={'load': True}).json()
    assert st['backend'] == 'classic' and 'classic' in st['available']
    r = c.post(f'/wissi-viz/slides/{sid}/sam', json={'rect': [700, 700, 1500, 1500], 'points': [[1000, 1000]], 'labels': [1]})
    j = r.json()
    assert r.status_code == 200 and len(j['polygons']) == 1 and j['backend'] == 'classic'
    assert c.post(f'/wissi-viz/slides/{sid}/sam', json={'rect': [5000, 5000, 6000, 6000], 'points': []}).status_code == 400
    sam._SERVICE = None


@pytest.mark.skipif(os.environ.get('WISSI_TEST_SAM') not in ('torch', 'onnx'), reason='set WISSI_TEST_SAM=torch|onnx')
def test_real_sam(slide):
    svc = SamService(backend=os.environ['WISSI_TEST_SAM'], model=os.environ.get('WISSI_SAM_MODEL', 'tiny'))
    polys, info = svc.segment(slide, rect=(700, 700, 1500, 1500), points=[(1000, 1000)], labels=[1], key='r')
    assert info['backend'] == os.environ['WISSI_TEST_SAM'], svc.status()['notes']
    assert _area(polys[0]) == pytest.approx(math.pi * 40 ** 2, rel=0.25)
    polys, info = svc.segment(slide, rect=(700, 700, 1500, 1500), box=(1160, 1010, 1240, 1090), key='r')
    assert info['cached'] and _area(polys[0]) == pytest.approx(math.pi * 25 ** 2, rel=0.3)


def test_app_sam_accept_export_and_measure(slide, tmp_path):
    """In the app page: SAM click → preview → Accept saves it to the layer with the class; export and measure it."""
    pytest.importorskip('playwright')
    pytest.importorskip('pyarrow')
    from playwright.sync_api import sync_playwright

    from wissi_viz import desktop as D, sam
    sam._SERVICE = SamService(backend='classic')
    s = D.Session(cache_dir=tmp_path / 'c')
    r = s.open([slide.path])
    sid = r['opened'][0]
    srv = D._Server(s.app())
    url = srv.start()
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            except Exception as e:  # noqa: BLE001
                pytest.skip(f'no Chromium: {e}')
            pg = b.new_page(viewport={'width': 1400, 'height': 900})
            errors = []
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.goto(url + '/app')
            pg.wait_for_function('() => document.getElementById("grid").panes.every(p => p.slide && p.slide.info)')
            pg.click('#b-annot')
            pg.fill('#a-class', 'Nucleus')
            pg.dispatch_event('#a-class', 'change')
            pg.evaluate('() => { const g = document.getElementById("grid"); g.goTo(0, 1000, 1000, {zoom: 1, animate: false}); }')
            pg.wait_for_timeout(300)
            pg.click('#a-sam')
            box = pg.evaluate('''() => { const g = document.getElementById("grid"), p = g.panes[0], r = g.getBoundingClientRect();
                return [r.left + p.rect.x + p.rect.w / 2, r.top + p.rect.y + p.rect.h / 2]; }''')
            pg.mouse.click(*box)                                    # the 40-px nucleus is at the view centre
            pg.wait_for_function('() => { const p = document.getElementById("grid").panes[0]; return p.sam && !p.sam.busy && p.sam.polys.length === 1; }')
            pg.click('#a-accept')
            pg.wait_for_function(f'''async () => {{ const r = await fetch("/wissi-viz/slides/{sid}/annotations?wait=false");
                const m = await r.json(); return m.some(x => x.name === "Annotations" && x.count === 1); }}''')
            L = s.registry.layer(sid, 'Annotations')
            assert L.class_of(0) == 'Nucleus' and L.properties(0).get('source') == 'sam'
            # export link and measurements
            exp = pg.request.get(f'{url}/wissi-viz/slides/{sid}/annotations/export?layer=Annotations&fmt=geoparquet')
            assert exp.ok and exp.body()[:4] == b'PAR1'
            pg.wait_for_function('() => [...document.querySelectorAll("#m-regions option")].some(o => o.value === "Annotations")')
            pg.select_option('#m-regions', 'Annotations')
            pg.click('#m-run')
            pg.wait_for_selector('#mtable table')
            txt = pg.inner_text('#mtable')
            assert 'shape_area_um2' in txt and 'int_hematoxylin_od_mean' in txt
            assert not errors, errors
            b.close()
    finally:
        srv.stop()
        s.shutdown()
        sam._SERVICE = None
