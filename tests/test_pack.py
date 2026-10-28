"""The sample pack pipeline end to end on local stand-ins (file:// URLs): download with verification, Xenium
bundle merge with 10x cells, CAMELYON-style ASAP annotations, classic segmentation, analyses, session, desktop."""
from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pytest

pytest.importorskip('pyarrow')
pytest.importorskip('scipy')
pytest.importorskip('skimage')
pytest.importorskip('pandas')

from wissi_viz import pack  # noqa: E402


@pytest.fixture()
def fake_pack(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tifffile
    src = tmp_path / 'src'
    src.mkdir()
    rng = np.random.default_rng(0)
    # an H&E-like slide with nuclei
    H, W = 1500, 2000
    img = np.full((H, W, 3), (232, 170, 200), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    for cx, cy in rng.uniform(100, 1400, (300, 2)):
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < 49] = (80, 50, 140)
    img[:, 1600:] = 245                                              # glass
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=95|AppMag = 20|MPP = 0.5'
    tifffile.imwrite(src / 'he.svs', img, tile=(256, 256), photometric='rgb', description=desc, metadata=None)
    (src / 'he.xml').write_text('''<?xml version="1.0"?><ASAP_Annotations><Annotations>
<Annotation Name="A" Type="Polygon" PartOfGroup="_0"><Coordinates><Coordinate Order="0" X="200" Y="200"/>
<Coordinate Order="1" X="900" Y="200"/><Coordinate Order="2" X="900" Y="900"/><Coordinate Order="3" X="200" Y="900"/>
</Coordinates></Annotation></Annotations></ASAP_Annotations>''')
    # a Xenium-like bundle: two morphology channels + cell boundaries in µm
    mpp = 0.2125
    for k, val in ((0, 1000), (1, 400)):
        a = (rng.random((800, 1000)) * 50).astype(np.uint16)
        for cx, cy in ((300, 300), (600, 500)):
            a[(np.mgrid[0:800, 0:1000][1] - cx) ** 2 + (np.mgrid[0:800, 0:1000][0] - cy) ** 2 < 400] = val
        tifffile.imwrite(src / f'morphology_focus_000{k}.ome.tif', a, photometric='minisblack',
                         metadata={'axes': 'YX', 'Channel': {'Name': ['DAPI' if k == 0 else 'CD45']}, 'PhysicalSizeX': mpp})
    def ring(cx, cy):
        t = np.linspace(0, 2 * np.pi, 13)
        return list((cx + 20 * np.cos(t)) * mpp), list((cy + 20 * np.sin(t)) * mpp)
    xs, ys, ids = [], [], []
    for i, (cx, cy) in enumerate(((300, 300), (600, 500))):
        x, y = ring(cx, cy)
        xs += x
        ys += y
        ids += [f'c{i}'] * len(x)
    pq.write_table(pa.table({'cell_id': ids, 'vertex_x': xs, 'vertex_y': ys}), src / 'cell_boundaries.parquet')
    with zipfile.ZipFile(src / 'x_outs.zip', 'w') as z:
        for k in (0, 1):
            z.write(src / f'morphology_focus_000{k}.ome.tif', f'outs/morphology_focus/morphology_focus_000{k}.ome.tif')
        z.write(src / 'cell_boundaries.parquet', 'outs/cell_boundaries.parquet')
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()  # noqa: E731
    items = [
        {'id': 'he', 'title': 'H&E', 'kind': 'he', 'url': (src / 'he.svs').as_uri(), 'file': 'he.svs',
         'size': (src / 'he.svs').stat().st_size, 'md5': md5(src / 'he.svs'), 'annotations_url': (src / 'he.xml').as_uri(),
         'source': 'test', 'licence': 'CC0-1.0'},
        {'id': 'xen', 'title': 'Xenium', 'kind': 'if', 'url': (src / 'x_outs.zip').as_uri(), 'file': 'x_outs.zip',
         'size': (src / 'x_outs.zip').stat().st_size, 'unpack': 'xenium', 'source': 'test', 'licence': 'CC-BY-4.0',
         'attribution': 'test data'},
    ]
    m = tmp_path / 'pack32.json'
    m.write_text(json.dumps({'items': items, 'quick': ['he', 'xen'], 'total_bytes': 1}))
    monkeypatch.setattr(pack, 'MANIFEST', m)
    monkeypatch.setattr(pack, 'RESULTS_RELEASE', (tmp_path / 'no-release').as_uri())
    return tmp_path / 'pack'


def test_demo_pipeline(fake_pack):
    s = pack.run_demo('quick', str(fake_pack), model='classic', yes=True, open_app=False, log=lambda *a: None)
    doc = json.loads(s.read_text())
    assert [e['id'] for e in doc['slides']] == ['he', 'xen'] and doc['show'] == ['he', 'xen']
    he, xen = doc['slides']
    layers = {p.rsplit('/', 1)[-1].split('.', 1)[1] for p in he['layers']}
    assert {'cells.parquet', 'tissue.parquet', 'expert.parquet', 'density.parquet', 'hotspots.parquet'} <= layers
    from wissi_viz import annio, open_slide
    cells = annio.read_geoparquet(fake_pack / 'layers' / 'he.cells.parquet')
    assert 250 < len(cells) < 340 and 'hematoxylin_mean' in cells.props          # ~300 nuclei painted
    exp = annio.read_geoparquet(fake_pack / 'layers' / 'he.expert.parquet')
    assert exp.class_names == ['Tumor']
    x = open_slide(fake_pack / 'slides' / 'xen.ome.tif')
    assert x.info.mode == 'channels' and [c.name for c in x.info.channels] == ['DAPI', 'CD45']
    xc = annio.read_geoparquet(fake_pack / 'layers' / 'xen.cells-10x.parquet')
    assert len(xc) == 2 and abs(xc.centroid()[0, 0] - 300) < 2                     # µm → px
    import pandas as pd
    allr = pd.read_csv(fake_pack / 'metrics' / 'all_regions.csv')
    tumour = allr[(allr.image == 'he') & allr.region.str.startswith('Tumor')].iloc[0]
    assert tumour['obj_count'] > 50 and tumour['shape_area_um2'] == pytest.approx(350 * 350)
    assert 'CC0-1.0' in (fake_pack / 'ATTRIBUTION.md').read_text()
    st = pack.status(str(fake_pack))
    assert st['quick']['downloaded'] == 2 and st['quick']['analyzed'] == 2
    # the desktop app loads the session: both slides with their layers, shown in panes
    from wissi_viz import desktop as D
    sess = D.Session(cache_dir=fake_pack / 'appcache')
    r = sess.load_session(s, log=lambda *a: None)
    assert not r['errors'] and len(sess.show) == 2
    assert {'cells', 'Expert annotations', 'Tissue'} <= set(sess.registry.layers(sess.show[0]))


def test_bad_download_is_rejected(fake_pack, monkeypatch):
    m = json.loads(pack.MANIFEST.read_text())
    m['items'][0]['md5'] = '0' * 32
    pack.MANIFEST.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='MD5'):
        pack.download(pack.items('quick'), pack.pack_dir(fake_pack), log=lambda *a: None)


def test_cli_entry_points_parse(monkeypatch):
    from wissi_viz import cli
    got = {}
    monkeypatch.setattr(pack, 'run_demo', lambda *a, **k: got.setdefault('demo', a))
    monkeypatch.setattr(pack, 'run_benchmark', lambda *a, **k: got.setdefault('bench', a))
    cli.main(['--demo', 'full', '--yes', '--no-open'])
    cli.main(['--benchmark', '--panes', '1,32', '--yes'])
    assert got['demo'][0] == 'full' and got['bench'][0] == 'quick' and got['bench'][3] == '1,32'
