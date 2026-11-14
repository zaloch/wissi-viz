"""Saved views (points of interest) and figure recipes: the app's SQLite store, tables in and out, the HTTP API."""

from __future__ import annotations

import io

import numpy as np
import pytest
import tifffile

from wissi_viz.store import Store


@pytest.fixture()
def slides(tmp_path):
    for name in ('A1.svs', 'B2.svs'):
        W, H = 1200, 900
        img = np.full((H, W, 3), 230, np.uint8)
        desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
        with tifffile.TiffWriter(tmp_path / name, bigtiff=True) as tw:
            tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    return tmp_path


def test_views_and_figures_in_a_file(tmp_path):
    st = Store(tmp_path / 'db' / 'wissi.db')
    v = st.add_view('/data/A1.svs', [10, 20, 110, 70], 'Islet', meta={'Treatment': 'siRNA', 'Score': 3})
    assert v['bounds'] == [10, 20, 110, 70] and v['meta'] == {'Treatment': 'siRNA', 'Score': 3}
    v = st.update_view(v['id'], name='Islet 1', meta={'Score': None, 'Reader': 'GZ'}, bounds=[0, 0, 50, 50])
    assert v['name'] == 'Islet 1' and v['meta'] == {'Treatment': 'siRNA', 'Reader': 'GZ'} and v['bounds'][2] == 50
    with pytest.raises(ValueError, match='width and height'):
        st.add_view('/data/A1.svs', [10, 10, 10, 30])
    f = st.save_figure('Figure 2', {'views': [v['id']], 'layout': {'cols': 2}})
    st.save_figure('Figure 2b', {'views': []}, f['id'])
    st.close()
    again = Store(tmp_path / 'db' / 'wissi.db')  # kept
    assert again.views()[0]['name'] == 'Islet 1' and again.figures()[0]['name'] == 'Figure 2b'
    with pytest.raises(KeyError):
        again.save_figure('x', {}, 999)
    assert again.delete_views([v['id']]) == 1 and again.views() == []


def test_views_from_and_to_tables(tmp_path):
    """Rectangles, or centres with a size in pixels or µm; other columns become metadata; bad rows are reported."""
    import pyarrow.parquet as pq

    (tmp_path / 'poi.csv').write_text(
        'slide,name,x0,y0,x1,y1,x,y,width,width_um,Treatment,Animal\n'
        'A1.svs,box,0,0,100,50,,,,,Untreated,001042\n'
        'A1,centre px,,,,,500,400,200,,siRNA,7\n'
        'B2,centre um,,,,,300,300,,50,siRNA,8\n'
        ',no slide,0,0,1,1,,,,,,\n'
        'A1,no size,,,,,10,10,,,,\n',
        encoding='utf-8',
    )
    st = Store()
    r = st.import_views(tmp_path / 'poi.csv', mpp_of=lambda s: 0.5 if s == 'B2' else None)
    assert r['added'] == 3 and len(r['errors']) == 2 and r['errors'][0].startswith('row 5')
    vs = st.views()
    assert vs[1]['bounds'] == [400, 300, 600, 500] and vs[2]['bounds'] == [250, 250, 350, 350]  # 50 µm at 0.5 µm/px
    assert vs[0]['meta'] == {'Treatment': 'Untreated', 'Animal': '001042'}
    t = pq.read_table(io.BytesIO(st.export_views('parquet')))
    assert t.column_names[:7] == ['id', 'slide', 'name', 'x0', 'y0', 'x1', 'y1'] and t.num_rows == 3
    (tmp_path / 'back.parquet').write_bytes(st.export_views('parquet'))
    st2 = Store()
    assert st2.import_views(tmp_path / 'back.parquet')['added'] == 3 and st2.views()[2]['meta']['Treatment'] == 'siRNA'
    assert st.export_views('csv').decode().splitlines()[0].startswith('id,slide,name,x0,y0,x1,y1')


def test_views_api(slides):
    from starlette.testclient import TestClient

    from wissi_viz.session import Session, create_session_app

    s = Session(roots=[slides], remember=False)
    c = TestClient(create_session_app(s, mode='server', key='k'))
    h = {'x-wissi-token': s.token, 'x-wissi-key': 'k'}
    sid = c.post('/app/open', json={'paths': [str(slides / 'A1.svs')]}, headers=h).json()['opened'][0]
    r = c.post('/app/views', json={'slide': sid, 'bounds': [0, 0, 300, 200], 'name': 'Field 1', 'meta': {'G': 'a'}},
               headers=h).json()  # fmt: skip
    vid = r['view']['id']
    assert r['views'][0]['sid'] == sid and r['views'][0]['slide'].endswith('A1.svs')
    (slides / 'more.csv').write_text('slide,x,y,width_um,name\nA1,600,450,100,Centre\nB2,10,10,20,Elsewhere\n')
    r = c.post('/app/views-import', json={'paths': [str(slides / 'more.csv')]}, headers=h).json()
    assert r['added'] == 1 and 'pixel size' in r['errors'][0]  # B2 is not open: its µm size cannot be converted
    assert r['views'][1]['bounds'] == [500, 350, 700, 550]
    assert c.post(f'/app/views/{vid}', json={'name': 'Renamed'}, headers=h).json()['views'][0]['name'] == 'Renamed'
    assert c.post('/app/views/999', json={'name': 'x'}, headers=h).status_code == 404
    csv = c.get('/app/views-export', headers=h)
    assert csv.headers['content-type'].startswith('text/csv') and 'Renamed' in csv.text
    assert len(c.post('/app/views-delete', json={'ids': [vid]}, headers=h).json()['views']) == 1
    f = c.post('/app/figures', json={'name': 'Fig 1', 'recipe': {'views': [vid]}}, headers=h).json()
    assert f['figure']['recipe'] == {'views': [vid]} and len(c.get('/app/figures', headers=h).json()['figures']) == 1
    assert c.post('/app/figures', json={'delete': f['figure']['id']}, headers=h).json()['figures'] == []
    assert c.post('/app/views', json={'slide': sid, 'bounds': [0, 0, 1, 1]}).status_code == 403  # the token
    s.shutdown()
