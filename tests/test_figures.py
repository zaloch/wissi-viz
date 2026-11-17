"""Figures: recipes laid out into pages of windows (wissi_viz.figures), PowerPoint templates (pptxio), exports
(figexport), channel pictures (regions) and the HTTP API."""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from wissi_viz import figures as F
from wissi_viz.regions import region_rgb


def _views(fl=False, n_per=3):
    out = []
    i = 0
    for t in ('Untreated', 'siRNA'):
        for _ in range(n_per):
            i += 1
            v = {'id': i, 'name': f'V{i}', 'slide': f'/d/S{i}.svs', 'sid': f's{i}', 'bounds': [0, 0, 1200, 900],
                 'meta': {'Treatment': t, 'Animal': f'00{i}', 'Slide': f'S{i}.svs', 'n': 10 - i}, 'mpp': 0.5,
                 'fluorescence': fl}  # fmt: skip
            if fl:
                v['channels'] = [{'name': 'DAPI', 'color': '#0000ff'}, {'name': 'NeuN', 'color': '#00ff00'},
                                 {'name': 'GFAP', 'color': '#ff0000'}]  # fmt: skip
            out.append(v)
    return out


def test_query_keeps_its_members_until_update():
    vs = _views()
    rec = {'source': {'kind': 'query', 'rules': [{'field': 'Treatment', 'op': 'in', 'values': ['siRNA']}]}}
    sel = F.select(rec, vs)
    assert sel['ids'] == [4, 5, 6] and sel['pending'] == {'add': [], 'remove': []}
    rec['members'] = sel['ids']
    vs.append({**vs[-1], 'id': 7, 'name': 'V7'})  # a new view that matches: waits for Update
    vs[3]['meta'] = {**vs[3]['meta'], 'Treatment': 'Untreated'}  # one that no longer matches: stays until Update
    sel = F.select(rec, vs)
    assert sel['ids'] == [4, 5, 6] and sel['pending'] == {'add': [7], 'remove': [4]}
    rec['members'] = None  # Update
    assert F.select(rec, vs)['ids'] == [5, 6, 7]
    assert F.select({'source': {'kind': 'views', 'ids': [3, 99, 1]}}, vs)['missing'] == [99]
    # rules: ranges, set / empty
    q = lambda rules: F.select({'source': {'kind': 'query', 'rules': rules}}, vs)['ids']
    assert q([{'field': 'n', 'op': '>=', 'values': ['7']}]) == [1, 2, 3]
    assert q([{'field': 'Missing', 'op': 'missing'}]) == [1, 2, 3, 4, 5, 6, 7]
    assert q([{'field': 'Treatment', 'op': 'not in', 'values': ['siRNA']}]) == [1, 2, 3, 4]


def test_order_labels_and_a_journal_page_by_rows():
    vs = _views()
    rec = {'source': {'kind': 'views', 'ids': [6, 5, 4, 3, 2, 1]}, 'layout': {'mode': 'field', 'rows': 'Treatment',
           'order': [{'field': 'Animal', 'desc': True}]}, 'labels': '{Animal} · {Nothing}', 'chart': {'kind': 'box'}}  # fmt: skip
    sel = F.select(rec, vs)
    assert [v['id'] for v in sel['members']] == [6, 5, 4, 3, 2, 1]  # siRNA < Untreated; animals descending
    L = F.layout(rec, sel['members'])
    (p,) = L['pages']
    assert p['w'] == 180 and 100 < p['h'] < 180 and p['bg'] == 'white'
    assert [h['text'] for h in p['heads']] == ['siRNA', 'Untreated']
    ws = [w for w in p['windows'] if w['kind'] == 'view']
    assert [w['letter'] for w in p['windows']] == list('ABCDEFG') and p['windows'][-1]['kind'] == 'chart'
    assert ws[0]['label'] == '006' and ws[0]['rect'][1] == ws[2]['rect'][1] < ws[3]['rect'][1]  # 3 per row
    x, y, w, h = ws[0]['rect']
    assert abs(w / h - 4 / 3) < 1e-6 and ws[0]['bounds'] == [0, 0, 1200, 900]
    sb = ws[0]['scalebar']
    assert sb['um'] == 100 and sb['text'] == '100 µm' and abs(sb['mm'] - w * 100 / 600) < 1e-6  # 600 µm wide
    assert F.label('{Animal} – {Treatment}', vs[0]) == '001 – Untreated'


def test_slides_overflow_and_the_background_rule():
    vs = _views()
    rec = {'template': 'ppt', 'source': {'kind': 'views', 'ids': [1, 2, 3, 4, 5, 6]}, 'layout': {'per_page': 4},
           'chart': {'kind': 'bar'}}  # fmt: skip
    L = F.layout(rec, F.select(rec, vs)['members'], name='Fig 3')
    assert [len(p['windows']) for p in L['pages']] == [4, 3] and L['pages'][0]['title']['text'] == 'Fig 3'
    assert L['pages'][1]['windows'][-1]['kind'] == 'chart' and L['pages'][0]['w'] == pytest.approx(338.667)
    for p in L['pages']:
        for w in p['windows']:
            x, y, ww, hh = w['rect']
            assert 0 <= x and x + ww <= p['w'] + 1e-6 and 0 <= y and y + hh <= p['h'] + 1e-6
    fl = _views(fl=True)
    assert F.layout({**rec, 'chart': None}, fl)['pages'][0]['bg'] == 'black'  # every view fluorescence
    assert F.layout({**rec, 'chart': None}, fl[:2] + vs[2:])['pages'][0]['bg'] == 'white'  # mixed
    assert F.layout({**rec, 'background': 'black'}, vs)['pages'][0]['ink'] == '#ffffff'
    t = {'size': [254.0, 190.5], 'layouts': [{'name': 'Grid', 'slots': [[10, 40, 100, 70], [120, 40, 100, 70]],
         'title': [10, 5, 230, 20]}]}  # fmt: skip
    L = F.layout({'template': 'pptx', 'pptx': {'id': 'x', 'layout': 'Grid'}}, fl, template=t)
    assert L['pages'][0]['bg'] is None and len(L['pages']) == 3 and L['pages'][0]['windows'][1]['rect'][0] == 120


def test_fluorescence_split_into_channels_plus_merge():
    fl = _views(fl=True, n_per=1)
    rec = {'source': {'kind': 'views', 'ids': [1, 2]}, 'channels': {'mode': 'split', 'gray': True}}
    L = F.layout(rec, fl)
    p = L['pages'][0]
    assert p['bg'] == 'black' and [c['text'] for c in p['cols']] == ['DAPI', 'NeuN', 'GFAP', 'Merge']
    assert [h['text'] for h in p['heads']] == ['V1', 'V2']
    ws = p['windows']
    assert [w.get('channels') for w in ws[:4]] == [[0], [1], [2], [0, 1, 2]] and ws[0]['gray'] and not ws[3].get('gray')
    assert [w.get('letter') for w in ws] == ['A', None, None, None, 'B', None, None, None]
    assert [bool(w.get('scalebar')) for w in ws] == [False, False, False, True] * 2
    assert ws[4]['rect'][1] > ws[0]['rect'][1] and ws[3]['scalebar']['color'] == '#ffffff'
    # only some channels, no merge window; merge mode with a subset
    L = F.layout({**rec, 'channels': {'mode': 'split', 'merge': False, 'show': ['NeuN', 'DAPI']}}, fl)
    assert [w['channels'] for w in L['pages'][0]['windows'][:2]] == [[0], [1]] and len(L['pages'][0]['windows']) == 4
    L = F.layout({**rec, 'channels': {'mode': 'merge', 'show': ['GFAP']}}, fl)
    assert [w['channels'] for w in L['pages'][0]['windows']] == [[2], [2]]


def test_windows_moved_changed_and_added_by_hand():
    vs = _views()
    rec = {'source': {'kind': 'views', 'ids': [1, 2, 3]},
           'windows': {'0:0': {'rect': [10, 10, 50, 40]}, '0:1': {'shows': {'kind': 'text', 'text': 'Note'}},
                       '0:2': {'shows': {'kind': 'view', 'view': 5}}},
           'extra': [{'page': 0, 'rect': [5, 60, 30, 20], 'shows': {'kind': 'empty'}}, {'page': 9}]}  # fmt: skip
    p = F.layout(rec, F.select(rec, vs)['members'] + [vs[4]])['pages'][0]
    ws = {w['key']: w for w in p['windows']}
    assert (
        ws['0:0']['rect'] == [10, 10, 50, 40]
        and ws['0:0']['edited']
        and ws['0:0']['bounds'][2] - ws['0:0']['bounds'][0] == 1200
    )
    assert ws['0:1']['kind'] == 'text' and ws['0:1']['text'] == 'Note' and 'letter' not in ws['0:1']
    assert ws['0:2']['view'] == 5 and ws['0:2']['label'] == 'V5' and ws['0:2']['letter'] == 'B'
    assert ws['x0']['kind'] == 'empty' and 'x1' not in ws
    fl = _views(fl=True, n_per=1)
    rec = {'source': {'kind': 'views', 'ids': [1]}, 'windows': {'0:0': {'shows': {'kind': 'view', 'channels': [1]}}}}
    w = F.layout(rec, fl)['pages'][0]['windows'][0]
    assert w['view'] == 1 and w['channels'] == [1]  # the same view, one channel


def test_chart_values_and_csv():
    vs = _views()

    class Objs:
        x = np.array([10.0, 20, 2000, 30])
        y = np.array([10.0, 20, 20, 30])
        classes = np.array(['A', 'B', 'A', 'A'], dtype=object)

    vals = F.measure_values({'measure': {'kind': 'count', 'layer': 'cells', 'class': 'A'}}, vs[:2],
                            lambda sid, layer: Objs() if layer == 'cells' else None)  # fmt: skip
    assert vals == {1: 2.0, 2: 2.0}
    d = F.measure_values({'measure': {'kind': 'density', 'layer': 'cells'}}, vs[:1], lambda s, layer: Objs())
    assert d[1] == pytest.approx(3 / (1200 * 900 * 0.25 / 1e6))
    rec = {'chart': {'kind': 'dots', 'measure': {'kind': 'field', 'field': 'n'}, 'group': 'Treatment'}}
    s = F.chart_series(rec, vs, F.measure_values(F.normalize(rec)['chart'], vs, lambda *a: None))
    assert [g['name'] for g in s['groups']] == ['siRNA', 'Untreated'] and s['ylabel'] == 'n'
    assert F.chart_csv(s).splitlines()[:2] == ['Treatment,view,label,n', 'siRNA,4,V4,6.0']
    assert F.box_stats([1, 2, 3, 4, 100])['hi'] == 4 and F.box_stats([]) is None


class _ChSource:
    """Three channels, 400 × 300, a 2 × downsampled level."""

    class info:
        width, height, mode = 400, 300, 'channels'

        class L:
            def __init__(self, ds):
                self.downsample = ds

        levels = [L(1.0), L(2.0)]

        class C:
            def __init__(self, n, c):
                self.name, self.color = n, c

        channels = [C('DAPI', '#0000ff'), C('NeuN', '#00ff00'), C('GFAP', '#ff0000')]

    def read_region(self, level, x, y, w, h, c):
        return np.full((h, w), 1000 * (c + 1), np.uint16)

    def channel_stats(self):
        return [(0, 3000)] * 3


def test_channel_pictures_and_outside_the_slide():
    src = _ChSource()
    a = region_rgb(src, (0, 0, 400, 300), (40, 30))
    assert a.shape == (30, 40, 3) and tuple(a[5, 5]) == (255, 170, 85)  # GFAP red, NeuN green, DAPI blue
    a = region_rgb(src, (0, 0, 400, 300), (40, 30), channels=[1])
    assert tuple(a[5, 5]) == (0, 170, 0)
    a = region_rgb(src, (0, 0, 400, 300), (40, 30), channels=[1], gray=True)
    assert tuple(a[5, 5]) == (170, 170, 170)
    a = region_rgb(src, (-200, 0, 200, 300), (40, 30), channels=[0])  # half outside: black there
    assert tuple(a[5, 2]) == (0, 0, 0) and tuple(a[5, 35]) == (0, 0, 85)


# ------------------------------------------------------------------------------------------- PowerPoint, export
pptx = pytest.importorskip('pptx')
pytest.importorskip('matplotlib')


def _template(path, pictures=True):
    from PIL import Image
    from pptx.util import Mm

    prs = pptx.Presentation()
    prs.slide_width, prs.slide_height = Mm(338.667), Mm(190.5)
    buf = io.BytesIO()
    Image.new('RGB', (40, 30), (200, 50, 50)).save(buf, 'PNG')
    s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only, pictures placed by hand: 2 × 2
    s.shapes.title.text = 'Old talk'
    if pictures:
        for r in range(2):
            for c in range(2):
                buf.seek(0)
                s.shapes.add_picture(buf, Mm(20 + c * 110), Mm(50 + r * 65), Mm(100), Mm(60))
    prs.save(path)
    return path


def test_template_layouts_learned_and_slides_written(tmp_path):
    from wissi_viz import pptxio

    info = pptxio.read_template(_template(tmp_path / 'talk.pptx'))
    assert info['size'] == [pytest.approx(338.67, abs=0.01), pytest.approx(190.5)]
    assert info['colors']['accent1'].startswith('#') and info['fonts']['heading']
    names = [x['name'] for x in info['layouts']]
    assert 'Title Only' in names and 'Slide 1 · 2 × 2 pictures' in names
    learned = info['layouts'][-1]
    assert learned['source'] == 'slides' and learned['base'] == 'Title Only' and learned['slots'][1][0] > 120
    two = next(x for x in info['layouts'] if x['name'] == 'Two Content')
    assert len(two['slots']) == 2 and two['title']
    vs = _views(fl=True)
    rec = {'template': 'pptx', 'pptx': {'id': 'x', 'layout': learned['name']}, 'source': {'kind': 'views', 'ids': [1, 2, 3, 4, 5]},
           'chart': {'kind': 'box', 'measure': {'kind': 'field', 'field': 'n'}}, 'channels': {'mode': 'split'}}  # fmt: skip
    L = F.layout(rec, vs[:5], name='Results', template=info)
    assert len(L['pages']) == 6 and all(len(p['windows']) == 4 for p in L['pages'])  # 5 × 4 + chart in slots of 4
    from wissi_viz import figexport

    pic = lambda w, pw, ph: _png(pw, ph)
    ser = F.chart_series(rec, vs[:5], {v['id']: float(v['meta']['n']) for v in vs[:5]})
    data = pptxio.write_pptx(L, pic, lambda w, pw, ph: figexport.chart_png(ser, pw, ph), template=tmp_path / 'talk.pptx',
                             info=info, layout_name=learned['name'], dpi=60)  # fmt: skip
    out = pptx.Presentation(io.BytesIO(data))
    assert len(out.slides) == 6  # the template's own slide is not kept
    s0 = out.slides[0]
    assert s0.slide_layout.name == 'Title Only' and s0.shapes.title.text == 'Results'
    assert sum(sh.shape_type == 13 for sh in s0.shapes) == 4  # pictures
    texts = [sh.text_frame.text for sh in s0.shapes if sh.has_text_frame]
    assert {'V1 · DAPI', 'V1 · NeuN', 'V1 · GFAP', 'V1 · Merge', 'A', '100 µm'} <= set(
        texts
    )  # in the template's places
    # a plain deck (no template)
    L = F.layout({'template': 'ppt', 'source': {'kind': 'views', 'ids': [1, 2]}}, vs[:2], name='Plain')
    out = pptx.Presentation(io.BytesIO(pptxio.write_pptx(L, pic, None, dpi=60)))
    assert len(out.slides) == 1 and round(out.slide_width / 36000) == 339


def _png(w, h):
    from PIL import Image

    b = io.BytesIO()
    Image.new('RGB', (w, h), (30, 30, 30)).save(b, 'PNG')
    return b.getvalue()


def test_exports_in_every_format():
    from wissi_viz import figexport

    vs = _views()
    rec = {'source': {'kind': 'views', 'ids': [1, 2, 3, 4]}, 'chart': {'kind': 'bar', 'measure': {'kind': 'field', 'field': 'n'},
           'group': 'Treatment'}, 'layout': {'mode': 'field', 'rows': 'Treatment'}}  # fmt: skip
    L = F.layout(rec, vs[:4])
    ser = F.chart_series(rec, vs[:4], {v['id']: float(v['meta']['n']) for v in vs[:4]})
    pic = lambda w, pw, ph: np.full((ph, pw, 3), 128, np.uint8)
    png = figexport.render(L, pic, 'png', 100, ser)[0]
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    assert abs(im.size[0] - 180 / 25.4 * 100) <= 1 and abs(im.size[1] - L['pages'][0]['h'] / 25.4 * 100) <= 1  # 180 mm
    pdf = figexport.render(L, pic, 'pdf', 100, ser)[0]
    assert pdf.startswith(b'%PDF') and b'Untreated' in pdf or b'FontFile' in pdf
    svg = figexport.render(L, pic, 'svg', 100, ser)[0].decode()
    assert '<svg' in svg and 'Untreated' in svg  # text stays text
    tif = figexport.render(L, pic, 'tiff', 150, ser)[0]
    th, tw = tifffile.TiffFile(io.BytesIO(tif)).pages[0].shape[:2]
    assert abs(tw - 180 / 25.4 * 150) <= 1 and abs(th - L['pages'][0]['h'] / 25.4 * 150) <= 1
    data, name, mt = figexport.bundle('Fig 1', 'png', [png], snap=figexport.snapshot('Fig 1', rec, vs[:4]), series=ser)
    z = zipfile.ZipFile(io.BytesIO(data))
    assert name == 'Fig 1.zip' and sorted(z.namelist()) == ['Fig 1.csv', 'Fig 1.png', 'Fig 1.recipe.json']
    snap = json.loads(z.read('Fig 1.recipe.json'))
    assert snap['recipe'] == rec and snap['views'][0]['bounds'] == [0, 0, 1200, 900]
    assert figexport.bundle('Fig 1', 'tiff', [tif])[1:] == ('Fig 1.tif', 'image/tiff')


# ------------------------------------------------------------------------------------------------------- the API
@pytest.fixture()
def slides(tmp_path):
    for i, name in enumerate(('A1.svs', 'B2.svs')):
        W, H = 1200, 900
        img = np.full((H, W, 3), 230, np.uint8)
        img[200:400, 300:600] = (120, 40, 120 + 60 * i)
        desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
        with tifffile.TiffWriter(tmp_path / name, bigtiff=True) as tw:
            tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    return tmp_path


def test_figures_api(slides, tmp_path):
    from starlette.testclient import TestClient

    from wissi_viz.session import Session, create_session_app

    s = Session(roots=[slides], remember=False, cache_dir=tmp_path / 'cache')
    c = TestClient(create_session_app(s, mode='server', key='k'))
    h = {'x-wissi-token': s.token, 'x-wissi-key': 'k'}
    sids = c.post('/app/open', json={'paths': [str(slides / 'A1.svs'), str(slides / 'B2.svs')]}, headers=h).json()[
        'opened'
    ]
    for i, (sid, g) in enumerate(((sids[0], 'ctrl'), (sids[0], 'ctrl'), (sids[1], 'kd'))):
        c.post('/app/views', json={'slide': sid, 'bounds': [100 * i, 100, 400 + 100 * i, 400], 'name': f'F{i}',
                                   'meta': {'G': g, 'Score': i + 1}}, headers=h)  # fmt: skip
    rec = {'source': {'kind': 'query', 'rules': [{'field': 'G', 'values': ['ctrl', 'kd']}]}, 'layout': {'mode': 'field', 'rows': 'G'},
           'chart': {'kind': 'bar', 'measure': {'kind': 'field', 'field': 'Score'}, 'group': 'G'}}  # fmt: skip
    j = c.post('/app/figures/layout', json={'recipe': rec, 'name': 'Fig 1'}, headers=h).json()
    assert j['members'] == [1, 2, 3] and j['values']['G'] == ['ctrl', 'kd'] and 'Score' in j['fields']
    assert [w['kind'] for w in j['pages'][0]['windows']] == ['view', 'view', 'view', 'chart']
    assert j['series']['groups'][1]['values'][0]['value'] == 3.0 and j['background'] == 'white'
    w0 = j['pages'][0]['windows'][0]
    th = c.get(
        f'/wissi-viz/slides/{w0["sid"]}/thumbnail', params={'size': 64, 'bounds': ','.join(map(str, w0['bounds']))}
    )
    assert th.status_code == 200 and th.headers['content-type'] == 'image/jpeg'
    r = c.post('/app/figures/export', json={'recipe': rec, 'name': 'Fig 1', 'fmt': 'pdf', 'dpi': 100}, headers=h)
    assert r.status_code == 200 and r.content.startswith(b'%PDF') and 'Fig 1.pdf' in r.headers['content-disposition']
    r = c.post('/app/figures/export', json={'recipe': rec, 'name': 'Fig 1', 'fmt': 'png', 'dpi': 100, 'snapshot': True,
                                            'data': True}, headers=h)  # fmt: skip
    assert sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == ['Fig 1.csv', 'Fig 1.png', 'Fig 1.recipe.json']
    assert c.post('/app/figures/export', json={'recipe': rec, 'fmt': 'gif'}, headers=h).status_code == 400
    # a template: uploaded, listed, used; a .potx is kept as a .pptx
    tpl = _template(tmp_path / 'lab.pptx').read_bytes()
    r = c.post('/app/figures/templates?name=lab.potx', content=_as_potx(tpl), headers=h)
    assert r.status_code == 200, r.text
    t = r.json()['template']
    assert (
        t['name'] == 'lab.potx' and c.get('/app/figures/templates', headers=h).json()['templates'][0]['id'] == t['id']
    )
    rec2 = {**rec, 'template': 'pptx', 'pptx': {'id': t['id'], 'layout': 'Slide 1 · 2 × 2 pictures'}}
    j = c.post('/app/figures/layout', json={'recipe': rec2, 'name': 'Fig 1'}, headers=h).json()
    assert j['template']['id'] == t['id'] and len(j['pages'][0]['windows']) == 4
    r = c.post('/app/figures/export', json={'recipe': rec2, 'name': 'Fig 1', 'fmt': 'pptx', 'dpi': 72}, headers=h)
    assert r.status_code == 200 and len(pptx.Presentation(io.BytesIO(r.content)).slides) == 1
    assert c.post('/app/figures/templates?name=x.pdf', content=b'%PDF', headers=h).status_code == 400
    assert c.post('/app/figures/layout', json={'recipe': rec}).status_code == 403  # the token
    # a snapshot opens as a figure on the views as they were: later changes to the views do not move it
    r = c.post('/app/figures/export', json={'recipe': rec, 'name': 'Fig 1', 'fmt': 'svg', 'snapshot': True}, headers=h)
    snap = zipfile.ZipFile(io.BytesIO(r.content)).read('Fig 1.recipe.json')
    c.post('/app/views/1', json={'bounds': [0, 0, 50, 50], 'meta': {'G': 'moved'}}, headers=h)
    f = c.post('/app/figures/snapshot', content=snap, headers=h).json()['figure']
    assert f['name'].startswith('Fig 1 (as of ') and f['recipe']['source']['kind'] == 'frozen'
    j = c.post('/app/figures/layout', json={'recipe': f['recipe'], 'name': f['name']}, headers=h).json()
    assert (
        j['members'] == [1, 2, 3]
        and j['pages'][0]['windows'][0]['bounds'][2] - j['pages'][0]['windows'][0]['bounds'][0] > 100
    )
    assert c.post('/app/figures/snapshot', content=b'{}', headers=h).status_code == 400
    # a layer drawn in the windows, with its key; on the thumbnail too
    from wissi_viz.annotations import AnnotationLayer

    L = AnnotationLayer(name='cells', classes={'Tumour': '#ff0000'})
    L.add_rect(320, 220, 40, 40, cls='Tumour')
    s.registry.add_annotations(sids[0], L, name='cells')
    j = c.post('/app/figures/layout', json={'recipe': {**rec, 'layers': ['cells']}}, headers=h).json()
    assert j['pages'][0]['windows'][0]['legend'] == [{'name': 'Tumour', 'color': '#ff0000'}]
    w0 = j['pages'][0]['windows'][0]
    th = c.get(f'/wissi-viz/slides/{w0["sid"]}/thumbnail', params={'size': 64, 'bounds': ','.join(map(str, w0['bounds'])),
                                                                  'layers': 'cells|nope'})  # fmt: skip
    assert th.status_code == 200
    r = c.post('/app/figures/export', json={'recipe': {**rec, 'layers': ['cells'], 'windows': {'0:0': {'inset': {}}}},
                                            'fmt': 'pptx', 'dpi': 72}, headers=h)  # fmt: skip
    assert r.status_code == 200, r.text
    # relink: the views of a slide that moved follow it
    (slides / 'moved').mkdir()
    shutil.copy(slides / 'B2.svs', slides / 'moved' / 'B2.svs')  # a copy: Windows does not move a file that is open
    r = c.post(
        '/app/views-relink', json={'old': str(slides / 'B2.svs'), 'new': str(slides / 'moved' / 'B2.svs')}, headers=h
    )
    assert r.json()['relinked'] == 1 and Path(s.store.view(3)['slide']) == slides / 'moved' / 'B2.svs'
    assert c.post('/app/views-relink', json={'old': 'x', 'new': str(slides / 'nope.svs')}, headers=h).status_code == 400
    s.shutdown()


def _as_potx(data: bytes) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for it in src.infolist():
            b = src.read(it.filename)
            if it.filename == '[Content_Types].xml':
                b = b.replace(b'presentationml.presentation.main+xml', b'presentationml.template.main+xml')
            z.writestr(it, b)
    return out.getvalue()


def test_insets_layers_and_frozen_views():
    vs = _views()
    rec = {'source': {'kind': 'views', 'ids': [1, 2]}, 'layers': ['cells'],
           'windows': {'0:0': {'inset': {'src': [0.5, 0.5, 0.1, 0.1], 'corner': 'bl', 'size': 0.4}, 'layers': []}}}  # fmt: skip
    ws = F.layout(rec, F.select(rec, vs)['members'])['pages'][0]['windows']
    w = ws[0]
    x, y, ww, wh = w['rect']
    ins = w['inset']
    assert ins['frame'] == pytest.approx([x + 0.5 * ww, y + 0.5 * wh, 0.1 * ww, 0.1 * wh])
    assert ins['rect'][0] < x + ww / 2 and ins['rect'][1] > y + wh / 2  # bottom left
    assert ins['rect'][2] == pytest.approx(0.4 * ww) and ins['bounds'] == pytest.approx([600, 450, 720, 540])
    assert w['layers'] == [] and ws[1]['layers'] == ['cells'] and 'inset' not in ws[1]
    # a snapshot's views: as given, in the recipe's order
    frozen = {'source': {'kind': 'frozen', 'views': [{'id': 9, 'bounds': [0, 0, 10, 10]}]}}
    assert F.normalize(frozen)['source']['kind'] == 'frozen'
    assert F.select(frozen, [vs[1], vs[0]])['ids'] == [2, 1]


def test_layer_outlines_and_key():
    from wissi_viz import annio
    from wissi_viz.annotations import AnnotationLayer
    from wissi_viz.regions import draw_layers, layer_key

    L = AnnotationLayer(name='cells', classes={'Tumour': '#ff0000', 'Immune': '#00ff00'})
    L.add_rect(10, 10, 20, 20, cls='Tumour')
    L.add_point(60, 60, cls='Immune')
    L.add_rect(500, 500, 10, 10, cls='Tumour')  # outside
    Fs = annio.from_layer(L)
    img = np.full((100, 100, 3), 255, np.uint8)
    out = draw_layers(img, (0, 0, 100, 100), [Fs])
    assert tuple(out[10, 20]) == (255, 0, 0) and (out[55:66, 55:66] == (0, 255, 0)).all(-1).any()
    assert layer_key([Fs], (0, 0, 100, 100)) == [
        {'name': 'Tumour', 'color': '#ff0000'},
        {'name': 'Immune', 'color': '#00ff00'},
    ]
    assert layer_key([Fs], (0, 0, 40, 40)) == [{'name': 'Tumour', 'color': '#ff0000'}]
