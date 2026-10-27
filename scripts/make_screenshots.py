"""README / website screenshots from real whole-slide images.

    python scripts/make_screenshots.py SLIDES_DIR OUT_DIR [--quick]

SLIDES_DIR holds CMU-1.svs, CMU-2.svs and CMU-3.svs from the OpenSlide test data (Aperio brightfield, CC0 1.0):
https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/ . Any .svs files work; the first one gets the cells.

1. Nuclei are detected on the first slide at full resolution: haematoxylin optical density (colour deconvolution),
   Otsu threshold, watershed-free connected components, size filter. Each nucleus gets its area, haematoxylin OD and
   eccentricity; the class is its size bin (small / medium / large), so every overlay shows real measurements.
2. The desktop app page, the stain-deconvolution view and the analysis workbench are screenshotted in headless
   Chromium (WebGL2 via SwiftShader).

Run by .github/workflows/screenshots.yml, which downloads the slides.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TILE = 2048


# ------------------------------------------------------------------------------------------------ nuclei
def _thumb(src):
    """The coarsest level at most ~4k wide, its downsample, and a tissue mask from saturation + darkness."""
    lv = max((i for i, L in enumerate(src.info.levels) if L.width <= 4096), default=len(src.info.levels) - 1)
    L = src.info.levels[lv]
    rgb = src.read_region(lv, 0, 0, L.width, L.height).astype(np.float32) / 255
    sat = rgb.max(-1) - rgb.min(-1)
    tissue = (sat > 0.07) & (rgb.mean(-1) < 0.92)
    return rgb, L.downsample, tissue


def _detect_tile(args):
    path, x, y, w, h, thr = args
    from skimage.color import rgb2hed
    from skimage.filters import gaussian
    from skimage.measure import label, regionprops_table
    from skimage.morphology import binary_opening, disk

    from wissi_viz.sources import open_slide
    src = open_slide(path)
    rgb = src.read_region(0, x, y, w, h)
    hem = rgb2hed(rgb)[..., 0]
    hs = gaussian(hem, 1.0)
    m = binary_opening(hs > thr, disk(1))
    lab = label(m)
    if lab.max() == 0:
        return None
    t = regionprops_table(lab, intensity_image=hem, properties=('centroid', 'area', 'bbox', 'mean_intensity', 'eccentricity'))
    return {k: np.asarray(v) for k, v in t.items()}, x, y


def detect_nuclei(path: Path, out: Path, workers: int | None = None, quick: bool = False) -> Path:
    from skimage.color import rgb2hed
    from skimage.filters import gaussian, threshold_otsu

    from wissi_viz.sources import open_slide
    src = open_slide(path)
    W, H = src.info.width, src.info.height
    mpp = (src.info.mpp or (0.5,))[0]
    _, ds, tissue = _thumb(src)
    tiles = []
    for y in range(0, H, TILE):
        for x in range(0, W, TILE):
            ty0, tx0 = int(y / ds), int(x / ds)
            ty1, tx1 = int(min(H, y + TILE) / ds) + 1, int(min(W, x + TILE) / ds) + 1
            if tissue[ty0:ty1, tx0:tx1].mean() > 0.15:
                tiles.append((x, y, min(TILE, W - x), min(TILE, H - y)))
    if quick:
        tiles = tiles[:: max(1, len(tiles) // 12)]
    # one global threshold on haematoxylin OD from a sample of tissue tiles
    rng = np.random.default_rng(0)
    samp = [tiles[i] for i in rng.choice(len(tiles), min(12, len(tiles)), replace=False)]
    vals = np.concatenate([gaussian(rgb2hed(src.read_region(0, x, y, w, h))[..., 0], 1.0).ravel()[::7] for x, y, w, h in samp])
    thr = float(threshold_otsu(vals))
    print(f'{path.name}: {len(tiles)} tissue tiles, haematoxylin threshold {thr:.4f}', flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for k, r in enumerate(ex.map(_detect_tile, [(str(path), *t, thr) for t in tiles], chunksize=2)):
            if r is None:
                continue
            t, x, y = r
            # keep nuclei whose centre is inside the tile (no double counting at edges) and of nucleus size
            area_um2 = t['area'] * mpp * mpp
            keep = (area_um2 >= 12) & (area_um2 <= 150)
            rows.append({'x': t['centroid-1'][keep] + x, 'y': t['centroid-0'][keep] + y,
                         'XMin': t['bbox-1'][keep] + x, 'YMin': t['bbox-0'][keep] + y,
                         'XMax': t['bbox-3'][keep] + x, 'YMax': t['bbox-2'][keep] + y,
                         'Area µm²': area_um2[keep], 'Haematoxylin OD': t['mean_intensity'][keep],
                         'Eccentricity': t['eccentricity'][keep]})
            if k % 50 == 0:
                print(f'  {k}/{len(tiles)} tiles, {time.time() - t0:.0f} s', flush=True)
    cols = {c: np.concatenate([r[c] for r in rows]) for c in rows[0]}
    a = cols['Area µm²']
    cls = np.where(a < 25, 'Small nucleus', np.where(a < 50, 'Medium nucleus', 'Large nucleus'))
    import csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Classification', *cols])
        for i in range(len(a)):
            w.writerow([cls[i], *(f'{cols[c][i]:.2f}' for c in cols)])
    print(f'{len(a):,} nuclei → {out}', flush=True)
    return out


def hotspot(cells_csv: Path, bin_px: float = 1500) -> tuple[float, float]:
    """Centre of the densest bin of nuclei: where the zoomed-in views go."""
    xy = np.loadtxt(cells_csv, delimiter=',', skiprows=1, usecols=(1, 2))
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[np.arange(0, xy[:, 0].max() + bin_px, bin_px),
                                                          np.arange(0, xy[:, 1].max() + bin_px, bin_px)])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    return float((xe[i] + xe[i + 1]) / 2), float((ye[j] + ye[j + 1]) / 2)


# ------------------------------------------------------------------------------------------------ screenshots
def _settle(pg, ms=2500):
    """Wait until every pane has its tiles (no requests in flight), then a little longer."""
    pg.wait_for_function('() => { const g = document.getElementById("grid"); return g.panes.every(p => p.slide && p.slide.info) }',
                         timeout=120_000)
    t0 = time.time()
    while time.time() - t0 < 60:
        busy = pg.evaluate('() => { const g = document.getElementById("grid"); return g.inflight.size + g.queued.size + g.uploads.length }')
        if not busy:
            break
        pg.wait_for_timeout(300)
    pg.wait_for_timeout(ms)


def app_shots(slides: list[Path], cells: Path, out: Path, spot: tuple[float, float]) -> None:
    from playwright.sync_api import sync_playwright

    from wissi_viz import desktop as D
    s = D.Session(cache_dir=out.parent / '.shot-cache')
    r = s.open([slides[0], cells])
    assert not r['errors'], r['errors']
    first = r['opened'][0]
    layer = r['layers'][0]['layer']
    ids = [first] + [s.open([p])['opened'][0] for p in slides[1:]]
    srv = D._Server(s.app())
    url = srv.start()
    sx, sy = spot
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])

        def page(w, h):
            pg = b.new_page(viewport={'width': w, 'height': h}, device_scale_factor=1)
            pg.goto(url + '/app')
            pg.wait_for_function(f'() => window.wissiApp && document.querySelectorAll("#slides li").length === {len(set(ids))}')
            pg.evaluate('() => { document.getElementById("log").textContent = "" }')
            return pg

        def panes(pg, ids_):
            pg.evaluate('(ids) => window.wissiApp.showSlides(ids)', ids_)

        # hero: three slides + a 20× close-up with nuclei outlines
        pg = page(1600, 1000)
        panes(pg, [ids[0], ids[1] if len(ids) > 1 else ids[0], ids[2] if len(ids) > 2 else ids[0], ids[0]])
        pg.evaluate("""([L, x, y]) => { const g = document.getElementById('grid');
            for (const i of [0, 1, 2]) g.setLayerVisible(i, L, false);
            g.goTo(3, x, y, {magnification: 20, animate: false}); }""", [layer, sx, sy])
        _settle(pg, 4000)
        pg.screenshot(path=str(out / 'hero.png'))

        # desktop app: overview with nuclei coloured by size class, and 40× outlines
        pg = page(1440, 900)
        panes(pg, [ids[0], ids[0]])
        pg.evaluate("""([x, y]) => { const g = document.getElementById('grid');
            g.goTo(0, x, y, {magnification: 2.5, animate: false});
            g.goTo(1, x, y, {magnification: 40, animate: false}); }""", [sx, sy])
        _settle(pg, 4000)
        pg.screenshot(path=str(out / 'desktop-app.png'))

        # stain deconvolution: the same field, RGB vs haematoxylin only
        pg = page(1600, 900)
        panes(pg, [ids[0], ids[0]])
        pg.evaluate("""([L, x, y]) => { const g = document.getElementById('grid');
            g.setLayerVisible('all', L, false);
            g.goTo('all', x, y, {magnification: 10, animate: false});
            g.setStain(1, 'hematoxylin'); }""", [layer, sx, sy])
        _settle(pg, 3000)
        pg.screenshot(path=str(out / 'stain-deconvolution.png'))

        b.close()
    srv.stop()
    s.shutdown()


def analysis_shot(slide: Path, cells: Path, out: Path, port: int = 8093) -> None:
    from playwright.sync_api import sync_playwright
    proc = subprocess.Popen([sys.executable, str(ROOT / 'examples' / 'analysis_app.py'), str(slide), str(cells),
                             '--port', str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        import urllib.request
        for _ in range(240):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        with sync_playwright() as pw:
            b = pw.chromium.launch(args=['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
            pg = b.new_page(viewport={'width': 1700, 'height': 950})
            pg.goto(f'http://127.0.0.1:{port}/')
            pg.wait_for_selector('.q-tab', timeout=120_000)
            pg.get_by_role('tab', name='Density').click()
            panel = pg.locator('.q-tab-panel', has=pg.get_by_label('cell µm'))
            panel.get_by_role('button', name='Run').click()
            panel.locator('canvas').first.wait_for(timeout=300_000)          # the chart
            pg.wait_for_timeout(8000)
            pg.screenshot(path=str(out / 'analysis-density.png'))
            b.close()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('slides_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--quick', action='store_true', help='detect on a sample of tiles (for trying the script)')
    a = ap.parse_args()
    slides = sorted(Path(a.slides_dir).glob('*.svs'))
    if not slides:
        sys.exit('no .svs files')
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = out.parent / '.shot-data' / f'{slides[0].stem} nuclei.csv'
    if not cells.exists():
        detect_nuclei(slides[0], cells, quick=a.quick)
    spot = hotspot(cells)
    print('close-up at', spot, flush=True)
    app_shots(slides, cells, out, spot)
    analysis_shot(slides[0], cells, out)
    meta = {'slides': [p.name for p in slides], 'nuclei': sum(1 for _ in open(cells, encoding='utf-8')) - 1,
            'source': 'OpenSlide test data (Aperio), CC0 1.0: https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/'}
    (out / 'SOURCES.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    for p in sorted(out.glob('*.png')):
        print(p, p.stat().st_size)


if __name__ == '__main__':
    main()
