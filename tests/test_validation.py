"""wissi-viz validate: matching outlines to a reference (F1, Dice), the NuInsSeg mosaic and a run of it."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')
pytest.importorskip('skimage')


def test_match_and_dice():
    import shapely

    from wissi_viz import evaluate

    ref = np.array([shapely.box(x, 0, x + 10, 10) for x in range(0, 100, 20)], dtype=object)  # 5 squares
    found = np.array([shapely.box(1, 0, 11, 10), shapely.box(20, 0, 30, 10), shapely.box(45, 0, 48, 3),
                      shapely.box(200, 0, 210, 10)], dtype=object)  # fmt: skip
    m = evaluate.match(found, ref)
    assert (m['tp'], m['fp'], m['fn']) == (2, 2, 3) and m['precision'] == 0.5 and m['recall'] == 0.4
    assert m['f1'] == pytest.approx(2 * 2 / 9, abs=1e-3) and 0.8 < m['mean_iou'] <= 1
    # one-to-one: a duplicate of a matched square does not count twice
    assert evaluate.match(np.array([ref[0], ref[0]], dtype=object), ref[:1])['tp'] == 1
    # only inside a region
    assert evaluate.match(found, ref, region=shapely.box(0, 0, 35, 10))['reference'] == 2
    assert evaluate.dice(shapely.box(0, 0, 10, 10), shapely.box(5, 0, 15, 10)) == 0.5
    assert evaluate.dice([], []) == 1.0


def _fake_nuinsseg(path):
    """Two organs, two patches each: dark round nuclei on pink, with labelled masks (as NuInsSeg lays them out)."""
    from PIL import Image

    rng = np.random.default_rng(3)
    with zipfile.ZipFile(path, 'w') as zf:
        for organ in ('human liver', 'mouse heart'):
            for k in (1, 2):
                img = np.full((256, 256, 3), (225, 160, 200), np.uint8)
                lab = np.zeros((256, 256), np.uint16)
                yy, xx = np.mgrid[0:256, 0:256]
                n = 0
                for _ in range(25):
                    x, y = rng.uniform(20, 236, 2)
                    m = (xx - x) ** 2 + (yy - y) ** 2 < 8**2
                    if lab[m].any():
                        continue
                    n += 1
                    lab[m] = n
                    img[m] = (90, 50, 140)
                stem = f'{organ.replace(" ", "_")}_{k:02d}'
                b = io.BytesIO()
                Image.fromarray(img).save(b, format='PNG')
                zf.writestr(f'{organ}/tissue images/{stem}.png', b.getvalue())
                b = io.BytesIO()
                Image.fromarray(lab).save(b, format='TIFF')
                zf.writestr(f'{organ}/label masks modify/{stem}.tif', b.getvalue())  # the "modify" set: skipped
                zf.writestr(f'{organ}/label masks/{stem}.tif', b.getvalue())


def test_nuinsseg_mosaic_and_a_run(tmp_path, monkeypatch):
    from wissi_viz import annio, validation
    from wissi_viz.session import Session
    from wissi_viz.sources import open_slide

    z = tmp_path / 'nuinsseg.zip'
    _fake_nuinsseg(z)
    monkeypatch.setattr(validation, 'nuinsseg_zip', lambda d, log: z)
    d = tmp_path / 'validation'
    (d / 'layers').mkdir(parents=True)
    slide, ref, pat = validation.nuinsseg_mosaic(d, per_organ=2, log=lambda m: None)
    src = open_slide(slide)
    assert src.info.mode == 'rgb' and abs(src.info.mpp[0] - validation.NUINSSEG_MPP) < 1e-6
    R = annio.read_geoparquet(ref)
    assert 60 < len(R) <= 100 and set(annio.read_geoparquet(pat).class_of()) == {'human liver', 'mouse heart'}
    s = Session(cache_dir=tmp_path / 'cache', remember=False)
    try:
        out = validation.run_nuinsseg(s, d, ['classic'], log=lambda m: None)
    finally:
        s.shutdown()
    m = out['methods']['classic']
    assert out['reference'] == len(R) and m['f1'] > 0.8 and m['mean_iou'] > 0.6, m
    assert (d / 'layers' / 'nuinsseg-mosaic.classic.parquet').exists()
    md = validation.markdown({'when': 'now', 'cases': {'nuinsseg': out}})
    assert '| Classic threshold |' in md and 'F1 at IoU' in md
    validation.pictures(d, log=lambda m: None)
    assert (d / 'nuinsseg-mosaic.classic.jpg').exists()
    doc = validation.session_file(d)
    assert 'nuinsseg-mosaic' in doc.read_text(encoding='utf-8')
    names = [p.name for p in validation.publish_list(d)]
    assert 'nuinsseg-mosaic.ome.tif' in names and 'SHA256SUMS' in names and 'nuinsseg-mosaic.classic.jpg' in names


def test_cli_lists_validate(capsys):
    from wissi_viz import cli

    with pytest.raises(SystemExit):
        cli.main(['validate', '--help'])
    assert 'NuInsSeg' in capsys.readouterr().out


def test_drg_mosaic_and_a_run(tmp_path, monkeypatch):
    """The rat DRG case from its archive's layout (NF/images, NF/masks/GT_Estimation): the mosaic, the experts'
    neurons, the pixel size estimated from them, and a run."""
    import tifffile
    from PIL import Image

    from wissi_viz import annio, validation
    from wissi_viz.session import Session

    d = tmp_path / 'validation'
    (d / 'layers').mkdir(parents=True)
    nf = d / 'drg' / 'Annotations_ratDRG' / 'NF'
    (nf / 'images').mkdir(parents=True)
    (nf / 'masks' / 'GT_Estimation').mkdir(parents=True)
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:400, 0:500]
    for k in (1, 2):
        img = rng.normal(300, 30, (400, 500)).clip(0)
        mask = np.zeros((400, 500), np.uint8)
        for cx, cy in ((100, 100), (300, 120), (200, 280), (400, 300)):
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < 40**2
            img[m] += 3000
            mask[m] = 255
        tifffile.imwrite(nf / 'images' / f'000{k}.tif', img.astype(np.uint16))
        Image.fromarray(mask).save(nf / 'masks' / 'GT_Estimation' / f'000{k}.png')
    monkeypatch.setattr(validation, '_zenodo_file', lambda *a, **k: pytest.fail('no download expected'))
    slide, ref, mpp = validation.drg_mosaic(d, log=lambda m: None)
    R = annio.read_geoparquet(ref)
    assert len(R) == 8 and mpp == pytest.approx(validation.DRG_NEURON_UM / 80, rel=0.08)
    s = Session(cache_dir=tmp_path / 'cache', remember=False)
    try:
        out = validation.run_drg(s, d, ['nothing-here'], log=lambda m: None)
        assert out['reference'] == 8 and out['methods'] == {}
        sid = s.open([slide], trusted=True)['opened'][0]
        assert s.registry.info(sid)['channels'][0]['name'] == 'Neurofilament'
    finally:
        s.shutdown()
    assert 'drg-mosaic' in validation.session_file(d).read_text(encoding='utf-8')
