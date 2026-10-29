"""The public sample pack: 32 whole-slide images (H&E and multiplex immunofluorescence), each segmented into cells
and run through wissi-viz's analyses, opened as one 32-pane session.

    uv run wissi-viz --demo            # quick: 5 images (~1 GB), opens in the desktop app
    uv run wissi-viz --demo full       # all 32 (~20 GB)
    uv run wissi-viz --benchmark       # benchmarks on the same images, report written to ./wissi-viz-benchmark.md
    wissi-viz pack status | download | prepare | segment | analyze | open    (step by step, --set quick|full)

Every image allows reuse including commercial use (CC0, TCGA open access, CC BY 4.0 with attribution); the pack
writes ATTRIBUTION.md next to the images. Segmentation uses the best installed backend (Cellpose-SAM on a GPU,
else the CPU classic detector); ``--model classpose`` is opt-in (see wissi_viz.segment for its licence note).
Precomputed results from a GPU run are used when a ``pack32-results`` release exists, so CPU-only machines get the
same cells.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from . import annio, certs
from ._compat import user_dir
from ._version import __version__
from .annotations import AnnotationLayer
from .bench import browser_bench, server_bench
from .desktop import main as desktop_main
from .sources import open_slide

MANIFEST = Path(__file__).resolve().parent / 'data' / 'pack32.json'
RESULTS_RELEASE = 'https://github.com/zaloch/wissi-viz/releases/download/pack32-results-v1'
UA = {'User-Agent': 'wissi-viz sample pack'}
CAMELYON_GROUPS = {'_0': 'Tumor', '_1': 'Tumor', '_2': 'Exclusion', 'Tumor': 'Tumor', 'Exclusion': 'Exclusion'}


def manifest() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError('the sample-pack manifest is missing from this install (wissi_viz/data/pack32.json)')
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def items(which: str = 'quick') -> list[dict]:
    m = manifest()
    if which == 'full':
        return m['items']
    ids = set(m.get('quick') or [])
    return [x for x in m['items'] if x['id'] in ids]


def pack_dir(path: str | os.PathLike | None = None) -> Path:
    if path:
        d = Path(path)
    elif os.environ.get('WISSI_PACK_DIR'):
        d = Path(os.environ['WISSI_PACK_DIR'])
    else:
        d = user_dir('cache') / 'pack32'
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------ download
def _human(n: float) -> str:
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB':
            return f'{n:.1f} {u}' if u != 'B' else f'{int(n)} B'
        n /= 1024
    return str(n)


def download(sel: list[dict], d: Path, workers: int = 4, log=print) -> None:
    """Resumable parallel download with size (and MD5, when the source publishes one) verification."""
    raw = d / 'downloads'
    raw.mkdir(exist_ok=True)
    todo = [x for x in sel if not _complete(raw / x['file'], x)]
    total = sum(x['size'] for x in todo)
    if not todo:
        log('all images already downloaded')
        return
    log(f'downloading {len(todo)} file(s), {_human(total)} → {raw}')
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def one(x, attempts: int = 6):
        """One file; a dropped connection or a short read resumes where it stopped (up to ``attempts`` times)."""
        for k in range(attempts):
            try:
                return fetch(x)
            except OSError as e:  # network errors (URLError, resets, timeouts) and short reads
                if 'MD5' in str(e) or k == attempts - 1:
                    raise
                log(f'  {x["file"]}: {e} — resuming in {2**k} s')
                time.sleep(2**k)

    def fetch(x):
        dst = raw / x['file']
        part = dst.with_name(dst.name + '.part')
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(x['url'], headers={**UA, **({'Range': f'bytes={have}-'} if have else {})})
        with urllib.request.urlopen(req, timeout=600) as r:
            if have and r.status != 206:  # server ignored the range: start over
                have = 0
            with open(part, 'ab' if have else 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    with lock:
                        done[0] += len(chunk)
        if x.get('size') and part.stat().st_size != x['size']:
            raise OSError(f'{x["file"]}: got {part.stat().st_size} bytes, expected {x["size"]}')
        if x.get('md5') and _md5(part) != x['md5']:
            part.unlink()
            raise OSError(f'{x["file"]}: MD5 mismatch (deleted; run again)')
        os.replace(part, dst)
        return x['id']

    stop = threading.Event()

    def progress():
        while not stop.wait(5):
            el = time.time() - t0
            rate = done[0] / max(el, 1e-6)
            log(
                f'  {_human(done[0])} / {_human(total)}  ({_human(rate)}/s, ~{(total - done[0]) / max(rate, 1):.0f} s left)'
            )

    th = threading.Thread(target=progress, daemon=True)
    th.start()
    errors = []
    try:
        with ThreadPoolExecutor(workers) as ex:
            for f in [ex.submit(one, x) for x in todo]:
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
    finally:
        stop.set()
    if errors:
        raise RuntimeError('download failed for: ' + '; '.join(errors) + certs.hint(errors))
    log(f'downloaded in {time.time() - t0:.0f} s')


def _complete(p: Path, x: dict) -> bool:
    return p.exists() and (not x.get('size') or p.stat().st_size == x['size'])


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ prepare
def prepare(sel: list[dict], d: Path, log=print) -> None:
    """Slides into ``slides/`` (Xenium bundles → one multi-channel OME-TIFF + 10x cell outlines), expert
    annotations into GeoParquet, ATTRIBUTION.md."""
    sd, ld = d / 'slides', d / 'layers'
    sd.mkdir(exist_ok=True)
    ld.mkdir(exist_ok=True)
    for x in sel:
        src = d / 'downloads' / x['file']
        if x.get('unpack') == 'xenium':
            out = sd / f'{x["id"]}.ome.tif'
            if not out.exists():
                log(f'{x["id"]}: merging Xenium morphology channels')
                _xenium(src, out, ld / f'{x["id"]}.cells-10x.parquet')
        else:
            out = sd / x['file']
            if not out.exists():
                try:
                    os.link(src, out)  # no copy: hard link where the file system allows it
                except OSError:
                    shutil.copyfile(src, out)
        if x.get('annotations_url'):
            gp = ld / f'{x["id"]}.expert.parquet'
            if not gp.exists():
                xml = d / 'downloads' / Path(x['annotations_url']).name
                if not xml.exists():
                    urllib.request.urlretrieve(x['annotations_url'], xml)
                L = AnnotationLayer.from_asap_xml(xml, name='Expert annotations', groups=CAMELYON_GROUPS)
                annio.export(L, gp, slide=out.name)
        x['_slide'] = str(out)
    _attribution(sel, d)


def _xenium(zpath: Path, out: Path, cells_out: Path) -> None:
    """Xenium output bundle → pyramidal multi-channel OME-TIFF (DAPI + boundary / interior / protein stains) and the
    10x cell boundaries as a GeoParquet layer (µm → px)."""
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        focus = sorted(n for n in names if 'morphology_focus' in n and n.endswith('.ome.tif'))
        if not focus:
            focus = [n for n in names if n.endswith('morphology_focus.ome.tif') or n.endswith('morphology.ome.tif')][:1]
        tmp = out.with_suffix('.tmpdir')
        tmp.mkdir(exist_ok=True)
        for n in focus:
            z.extract(n, tmp)
        boundaries = next((n for n in names if n.endswith('cell_boundaries.parquet')), None)
        if boundaries:
            z.extract(boundaries, tmp)
    chans, chnames, mpp = [], [], 0.2125
    for n in focus:
        with tifffile.TiffFile(tmp / n) as tf:
            a = tf.series[0].levels[0].asarray()
            meta = tf.ome_metadata or ''
        a = a.reshape(-1, *a.shape[-2:])
        chans.extend(list(a))
        nm = re.findall(r'<Channel[^>]*Name="([^"]+)"', meta)
        chnames.extend(nm[: len(a)] if len(nm) >= len(a) else [f'ch{len(chnames) + i}' for i in range(len(a))])
        m = re.search(r'PhysicalSizeX="([0-9.]+)"', meta)
        if m:
            mpp = float(m.group(1))
    data = np.stack(chans)
    with tifffile.TiffWriter(out, bigtiff=True, ome=True) as tw:
        opts = {'tile': (512, 512), 'compression': 'zlib', 'photometric': 'minisblack'}
        nlev = max(1, int(np.ceil(np.log2(max(data.shape[1:]) / 1024))) + 1)
        tw.write(
            data,
            subifds=nlev - 1,
            metadata={'axes': 'CYX', 'Channel': {'Name': chnames}, 'PhysicalSizeX': mpp, 'PhysicalSizeY': mpp},
            **opts,
        )
        lv = data
        for _ in range(nlev - 1):
            lv = lv[:, ::2, ::2]
            tw.write(lv, subfiletype=1, **opts)
    if boundaries:
        _xenium_cells(tmp / boundaries, cells_out, mpp, out.name)
    shutil.rmtree(tmp, ignore_errors=True)


def _xenium_cells(parquet: Path, out: Path, mpp: float, slide: str) -> None:
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=['cell_id', 'vertex_x', 'vertex_y'])
    cid = t.column('cell_id').to_numpy(zero_copy_only=False)
    x = t.column('vertex_x').to_numpy() / mpp
    y = t.column('vertex_y').to_numpy() / mpp
    change = np.nonzero(cid[1:] != cid[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(cid)]])
    keep = []
    for s, e in zip(starts, ends):  # drop the repeated closing vertex of each ring
        if e - s >= 4 and x[s] == x[e - 1] and y[s] == y[e - 1]:
            e -= 1
        keep.append((s, e))
    idx = np.concatenate([np.arange(s, e) for s, e in keep])
    lens = np.array([e - s for s, e in keep])
    ring_off = np.zeros(len(keep) + 1, np.int64)
    np.cumsum(lens, out=ring_off[1:])
    n = len(keep)
    F = annio.Features(
        np.full(n, annio.GEO_POLYGON, np.uint8),
        np.stack([x[idx], y[idx]], 1),
        ring_off,
        np.arange(n + 1, dtype=np.int64),
        np.zeros(n, np.int32),
        ['Cell (10x)'],
        props={'cell_id': np.asarray([str(cid[s]) for s, _ in keep], dtype=object)},
    )
    F.name, F.mpp = '10x cell segmentation', mpp
    annio.write_geoparquet(F, out, slide=slide)


def _attribution(sel: list[dict], d: Path) -> None:
    lines = [
        '# Sample pack: sources and licences',
        '',
        'Every image below allows reuse, including commercial use. Keep this file with the images.',
        '',
        '| Image | Source | Licence | Attribution |',
        '|---|---|---|---|',
    ]
    for x in sel:
        lines.append(
            f'| {x["title"]} | {x["source"]} ([link]({x["url"]})) | {x["licence"]} | {x.get("attribution", "")} |'
        )
    (d / 'ATTRIBUTION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ------------------------------------------------------------------------------------------------ segment + analyze
def segment(sel: list[dict], d: Path, model: str = 'auto', log=print) -> None:
    from .segment import segment_slide

    ld = d / 'layers'
    _fetch_results(sel, d, log)
    for x in sel:
        out = ld / f'{x["id"]}.cells.parquet'
        if out.exists():
            continue
        if (ld / f'{x["id"]}.cells-10x.parquet').exists() and model == 'auto':
            continue  # Xenium bundles bring their own cell segmentation
        segment_slide(_slide(x, d), out, model=model, log=log)


def _fetch_results(sel: list[dict], d: Path, log) -> None:
    """Precomputed cells (from a GPU run of this same pack) if this project published them."""
    ld = d / 'layers'
    try:
        sums = (
            urllib.request.urlopen(urllib.request.Request(f'{RESULTS_RELEASE}/SHA256SUMS', headers=UA), timeout=20)
            .read()
            .decode()
        )
    except Exception:  # noqa: BLE001  (no release yet / offline: segment locally)
        return
    want = {ln.split()[1]: ln.split()[0] for ln in sums.splitlines() if ln.strip()}
    for x in sel:
        name = f'{x["id"]}.cells.parquet'
        if name in want and not (ld / name).exists():
            log(f'{x["id"]}: precomputed cells')
            tmp = ld / (name + '.part')
            urllib.request.urlretrieve(f'{RESULTS_RELEASE}/{name}', tmp)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() == want[name]:
                os.replace(tmp, ld / name)
            else:
                tmp.unlink()


def _slide(x: dict, d: Path) -> Path:
    if x.get('unpack') == 'xenium':
        return d / 'slides' / f'{x["id"]}.ome.tif'
    return d / 'slides' / x['file']


def _cells(x: dict, d: Path) -> Path | None:
    ld = d / 'layers'
    for n in (f'{x["id"]}.cells.parquet', f'{x["id"]}.cells-10x.parquet'):
        if (ld / n).exists():
            return ld / n
    return None


def _mpp_of(src, cells, name: str, log) -> float:
    """µm/px for measuring a slide: its own; else what its cells were segmented at; else 0.5 µm/px, the
    segmenter's assumption for slides without resolution metadata (said in the log: the µm are then approximate)."""
    mpp = (src.info.mpp or (None,))[0] or getattr(cells, 'mpp', None)
    if not mpp:
        log(f'{name}: no resolution in the file; measuring at an assumed 0.5 µm/px (sizes and densities approximate)')
    return float(mpp or 0.5)


def analyze(sel: list[dict], d: Path, log=print) -> None:
    """Per image: tissue outline, cell density + Getis-Ord hotspots, and region metrics (tissue and expert ROIs, with
    cells) → layers/*.parquet and metrics/<id>.csv."""
    from . import annio, spatial
    from .metrics import objects_from_layer, region_metrics

    ld, md = d / 'layers', d / 'metrics'
    md.mkdir(exist_ok=True)
    for x in sel:
        out = md / f'{x["id"]}.csv'
        if out.exists():
            continue
        t0 = time.time()
        src = open_slide(_slide(x, d))
        cp = _cells(x, d)
        cells = annio.load(cp) if cp else None
        mpp = _mpp_of(src, cells, x['id'], log)
        tis_p = ld / f'{x["id"]}.tissue.parquet'
        if not tis_p.exists():
            _, tis_layer = spatial.tissue_mask(src)
            tis_layer.origin = 'detected'
            annio.export(tis_layer, tis_p, slide=Path(_slide(x, d)).name)
        tissue = annio.load(tis_p)
        objs = objects_from_layer(cells, mpp) if cells is not None else None
        if objs is not None and len(objs) and not (ld / f'{x["id"]}.density.parquet').exists():
            cls = None
            g = spatial.density_grid(objs, cls, cell_um=100, hotspots=True)
            for gl, suffix in (
                (g.layer(name='Cell density'), 'density'),
                (g.hotspot_layer(name='Hotspots'), 'hotspots'),
            ):
                gl.origin = 'analysis'
                annio.export(gl, ld / f'{x["id"]}.{suffix}.parquet')
        tables = [region_metrics(tissue, source=src, objects=objs, mpp=mpp)]
        exp = ld / f'{x["id"]}.expert.parquet'
        if exp.exists():
            tables.append(region_metrics(annio.load(exp), source=src, objects=objs, mpp=mpp))
        import pandas as pd

        df = pd.concat(tables, ignore_index=True)
        df.insert(0, 'image', x['id'])
        df.to_csv(out, index=False)
        log(f'{x["id"]}: {len(df)} regions × {len(df.columns)} measurements ({time.time() - t0:.0f} s)')
    _summary(sel, d)


def _summary(sel, d: Path) -> None:
    import pandas as pd

    fs = [d / 'metrics' / f'{x["id"]}.csv' for x in sel if (d / 'metrics' / f'{x["id"]}.csv').exists()]
    if fs:
        pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(
            d / 'metrics' / 'all_regions.csv', index=False
        )


def session(sel: list[dict], d: Path) -> Path:
    """session.json: the slides (in pack order) and their layers, for ``wissi-viz desktop --session``."""
    ld = d / 'layers'
    entries = []
    for x in sel:
        layers = [str(p) for p in sorted(ld.glob(f'{x["id"]}.*.parquet'))]
        entries.append(
            {'id': x['id'], 'title': x['title'], 'slide': str(_slide(x, d)), 'layers': layers, 'licence': x['licence']}
        )
    groups: dict[str, list[str]] = {}
    for x in sel:  # the slide list's projects: one per collection
        groups.setdefault(_collection(x), []).append(x['id'])
    p = d / f'session-{len(sel)}.json'
    doc = {
        'name': 'wissi-viz sample pack',
        'slides': entries,
        'show': [e['id'] for e in entries][:32],
        'projects': [{'name': f'Sample pack · {k}', 'slides': v} for k, v in groups.items()],
    }
    p.write_text(json.dumps(doc, indent=1), encoding='utf-8')
    return p


def _collection(x: dict) -> str:
    src = (x.get('source') or '').lower()
    if 'xenium' in src or x.get('kind') == 'if':
        return 'Multiplex IF (Xenium)'
    if 'camelyon' in src or 'camelyon' in x.get('id', ''):
        return 'Lymph nodes (CAMELYON16)'
    if 'tcga' in src or 'gdc' in src:
        return 'Tumour types (TCGA)'
    return 'H&E (OpenSlide)'


# ------------------------------------------------------------------------------------------------ entry points
def _need_extras() -> None:
    missing = [m for m in ('pyarrow', 'scipy', 'skimage', 'pandas', 'shapely') if not importlib.util.find_spec(m)]
    if missing:
        raise SystemExit(
            f'the demo needs {", ".join(missing)}: install "wissi-viz[demo]" '
            '(uv tool install "wissi-viz[demo]", or uv sync --extra demo)'
        )


def run_demo(
    which: str = 'quick',
    directory: str | None = None,
    model: str = 'auto',
    yes: bool = False,
    open_app: bool = True,
    browser: bool = False,
    log=print,
) -> Path:
    _need_extras()
    sel = items(which)
    d = pack_dir(directory)
    need = sum(x['size'] for x in sel if not _complete(d / 'downloads' / x['file'], x))
    log(f'wissi-viz demo ({which}): {len(sel)} images in {d}')
    if need and not yes:
        if not sys.stdin or not sys.stdin.isatty():
            raise SystemExit(f'the demo needs to download {_human(need)}; run again with --yes')
        ans = input(f'Download {_human(need)} of public slides (licences in ATTRIBUTION.md)? [y/N] ').strip().lower()
        if ans not in ('y', 'yes'):
            raise SystemExit('cancelled')
    download(sel, d, log=log)
    prepare(sel, d, log=log)
    segment(sel, d, model=model, log=log)
    analyze(sel, d, log=log)
    s = session(sel, d)
    log(f'ready: {s}  (metrics: {d / "metrics" / "all_regions.csv"})')
    if open_app:
        desktop_main(['--session', str(s)] + (['--browser'] if browser else []))
    return s


def publish(d: Path) -> str:
    """Checksums for the segmented cells, and the command that shares them (a GitHub release this package reads):
    after one GPU run, every other machine downloads these cells instead of segmenting."""
    ld = d / 'layers'
    files = sorted(ld.glob('*.cells.parquet'))
    if not files:
        raise SystemExit('no segmented cells yet: run `wissi-viz pack segment` first')
    lines = [f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}' for f in files]
    (ld / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    tag = RESULTS_RELEASE.rsplit('/', 1)[-1]
    return (
        f'{len(files)} cell layers, {_human(sum(f.stat().st_size for f in files))}. To share them:\n'
        f'  cd "{ld}"\n  gh release create {tag} --repo zaloch/wissi-viz --title "Sample pack cells" '
        f'--notes "Cells of the public sample pack" *.cells.parquet SHA256SUMS'
    )


def status(directory: str | None = None) -> dict:
    d = pack_dir(directory)
    out = {}
    for which in ('quick', 'full'):
        sel = items(which)
        out[which] = {
            'images': len(sel),
            'bytes': sum(x['size'] for x in sel),
            'downloaded': sum(_complete(d / 'downloads' / x['file'], x) for x in sel),
            'segmented': sum(_cells(x, d) is not None for x in sel),
            'analyzed': sum((d / 'metrics' / f'{x["id"]}.csv').exists() for x in sel),
        }
    out['dir'] = str(d)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog='wissi-viz pack')
    ap.add_argument('step', choices=['status', 'download', 'prepare', 'segment', 'analyze', 'open', 'publish', 'all'])
    ap.add_argument('--set', default='quick', choices=['quick', 'full'])
    ap.add_argument('--dir', default=None, help='where the pack lives (default: the user cache, or $WISSI_PACK_DIR)')
    ap.add_argument('--model', default='auto', choices=['auto', 'cellpose', 'classpose', 'classic'])
    ap.add_argument('--yes', action='store_true')
    ap.add_argument('--browser', action='store_true')
    a = ap.parse_args(argv)
    d = pack_dir(a.dir)
    sel = items(a.set)
    if a.step == 'status':
        print(json.dumps(status(a.dir), indent=1))
    elif a.step == 'download':
        download(sel, d)
    elif a.step == 'prepare':
        prepare(sel, d)
    elif a.step == 'segment':
        prepare(sel, d)
        segment(sel, d, a.model)
    elif a.step == 'analyze':
        analyze(sel, d)
    elif a.step == 'publish':
        print(publish(d))
    elif a.step == 'open':
        desktop_main(['--session', str(session(sel, d))] + (['--browser'] if a.browser else []))
    else:
        run_demo(a.set, a.dir, a.model, a.yes, browser=a.browser)


# ------------------------------------------------------------------------------------------------ benchmark
def run_benchmark(
    which: str = 'quick',
    directory: str | None = None,
    out: str = 'wissi-viz-benchmark',
    panes: str = '1,4,16,32',
    yes: bool = False,
    headed: bool = False,
    seconds: float = 6,
    log=print,
) -> Path:
    """Benchmarks on the sample pack: slide opening and tile serving, panning 1–32 panes in a browser, SAM click
    latency, region metrics, GeoParquet vs GeoJSON. Writes <out>.md and <out>.json."""

    _need_extras()
    sel = items(which)
    d = pack_dir(directory)
    need = sum(x['size'] for x in sel if not _complete(d / 'downloads' / x['file'], x))
    if need and not yes:
        if not sys.stdin or not sys.stdin.isatty():
            raise SystemExit(f'the benchmark needs to download {_human(need)}; run again with --yes')
        if input(f'Download {_human(need)} of public slides? [y/N] ').strip().lower() not in ('y', 'yes'):
            raise SystemExit('cancelled')
    download(sel, d, log=log)
    prepare(sel, d, log=log)
    slides = [str(_slide(x, d)) for x in sel]
    rep: dict[str, Any] = {
        'version': __version__,
        'set': which,
        'images': len(sel),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'cpu': platform.processor() or platform.machine(),
        'cpus': os.cpu_count(),
    }
    log('1/5 slide opening and tile serving')
    rep['server'] = server_bench(slides)
    log('2/5 panning in the browser (' + panes + ' panes)')
    try:
        rep['browser'] = browser_bench(slides, [int(p) for p in panes.split(',')], headed=headed, pan_seconds=seconds)
    except Exception as e:  # noqa: BLE001  (no Chromium: install with `playwright install chromium`)
        rep['browser'] = {'error': f'{type(e).__name__}: {e}'}
    log('3/5 SAM click latency')
    rep['sam'] = _bench_sam(slides[:3])
    log('4/5 region metrics')
    rep['metrics'] = _bench_metrics(sel[:2], d)
    log('5/5 GeoParquet vs GeoJSON')
    rep['formats'] = _bench_formats(sel, d)
    md = _report_md(rep)
    Path(out).with_suffix('.json').write_text(json.dumps(rep, indent=1, default=str), encoding='utf-8')
    p = Path(out).with_suffix('.md')
    p.write_text(md, encoding='utf-8')
    print('\n' + md)
    log(f'report: {p.resolve()}')
    return p


def _bench_sam(slides: list[str]) -> dict:
    from .sam import SamService

    svc = SamService()
    rows = []
    for path in slides:
        src = open_slide(path)
        W, H = src.info.width, src.info.height
        cx, cy = W / 2, H / 2
        for zoom, span in (('overview', max(W, H)), ('20x field', 2000)):
            rect = (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)
            try:
                _, a = svc.segment(src, rect, points=[(cx, cy)], labels=[1], key=path)
                _, b = svc.segment(src, rect, points=[(cx, cy), (cx + span / 10, cy)], labels=[1, 1], key=path)
                rows.append(
                    {
                        'slide': Path(path).name,
                        'view': zoom,
                        'backend': a['backend'],
                        'device': a['device'],
                        'encode_ms': a['encode_ms'],
                        'first_click_ms': round((a['encode_ms'] or 0) + a['decode_ms'], 1),
                        'next_click_ms': b['decode_ms'],
                    }
                )
            except Exception as e:  # noqa: BLE001
                rows.append({'slide': Path(path).name, 'view': zoom, 'error': str(e)})
    return {'status': svc.status(), 'rows': rows}


def _bench_metrics(sel, d: Path) -> list[dict]:
    from . import annio, spatial
    from .metrics import objects_from_layer, region_metrics

    rows = []
    for x in sel:
        src = open_slide(_slide(x, d))
        t0 = time.perf_counter()
        _, tissue = spatial.tissue_mask(src)
        t1 = time.perf_counter()
        cp = _cells(x, d)
        cells = annio.load(cp) if cp else None
        mpp = _mpp_of(src, cells, x['id'], lambda *_: None)
        objs = objects_from_layer(cells, mpp) if cells is not None else None
        t2 = time.perf_counter()
        df = region_metrics(tissue, source=src, objects=objs, mpp=mpp)
        t3 = time.perf_counter()
        rows.append(
            {
                'image': x['id'],
                'regions': len(df),
                'measurements': len(df.columns),
                'cells': len(objs) if objs is not None else 0,
                'tissue_s': round(t1 - t0, 2),
                'load_cells_s': round(t2 - t1, 2),
                'metrics_s': round(t3 - t2, 2),
            }
        )
    return rows


def _bench_formats(sel, d: Path) -> dict:

    cps = [c for c in (_cells(x, d) for x in sel) if c]
    if not cps:
        return {'note': 'no cell layers yet (run wissi-viz --demo first)'}
    big = max(cps, key=lambda p: p.stat().st_size)
    t0 = time.perf_counter()
    F = annio.read_geoparquet(big)
    t1 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        gj = Path(tmp) / 'cells.geojson'
        annio.write_geojson(F, gj)
        t2 = time.perf_counter()
        AnnotationLayer.from_geojson(gj)
        t3 = time.perf_counter()
        pq_ = annio.write_geoparquet(F, Path(tmp) / 'cells.parquet')
        t4 = time.perf_counter()
        return {
            'layer': big.name,
            'features': len(F),
            'geoparquet_mb': round(pq_.stat().st_size / 2**20, 1),
            'geojson_mb': round(gj.stat().st_size / 2**20, 1),
            'geoparquet_read_s': round(t1 - t0, 2),
            'geoparquet_write_s': round(t4 - t3, 2),
            'geojson_write_s': round(t2 - t1, 2),
            'geojson_read_s': round(t3 - t2, 2),
        }


def _report_md(r: dict) -> str:
    L = [
        f'# wissi-viz {r["version"]} benchmark',
        '',
        f'{r["images"]} public slides ({r["set"]} set) · {r["platform"]} · Python {r["python"]} · {r["cpus"]} CPUs',
        '',
    ]
    if r.get('server'):
        L += [
            '## Slide opening and tile serving',
            '',
            '| slide | open (ms) | cold tile p50 (ms) | warm tile p50 (ms) | tiles/s |',
            '|---|---|---|---|---|',
        ]
        for s in r['server']:
            lv = (s.get('levels_bench') or [{}])[0]  # full resolution
            tput = next((v for k, v in lv.items() if k.startswith('throughput')), '')
            L.append(
                f'| {s.get("slide", "")} ({s.get("size", "")}) | {s.get("open_ms", "")} | {lv.get("cold_p50_ms", "")} | '
                f'{lv.get("warm_p50_ms", "")} | {tput} |'
            )
        L.append('')
    b = r.get('browser') or {}
    if b.get('error'):
        L += ['## Panning', '', f'Not run: {b["error"]}', '']
    elif b:
        L += [
            '## Panning in the browser',
            '',
            f'{b.get("webgl_renderer", "")} ({"software" if b.get("software_rendering") else "GPU"}), {b.get("http_protocol", "")}',
            '',
            '| panes | open (ms) | zoom to sharp (ms) | fps (median) | frame p95 (ms) | sharp while moving | settle (ms) | GPU textures (MB) |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for run in b.get('runs', []):
            pn = run.get('pan', {})
            L.append(
                f'| {run.get("panes")} | {run.get("open_ms")} | {run.get("zoom_to_sharp_ms")} | {pn.get("fps_median")} | '
                f'{pn.get("frame_p95_ms")} | {pn.get("sharp_while_moving_pct")} % | {pn.get("settle_ms")} | '
                f'{(run.get("stats") or {}).get("textureMB")} |'
            )
        L.append('')
    s = r.get('sam') or {}
    if s.get('rows'):
        st = s['status']
        L += [
            '## SAM click segmentation',
            '',
            f'backend **{st.get("backend")}** on {st.get("device")} (model {st.get("model")})',
            '',
            '| slide | view | first click (ms) | next click (ms) |',
            '|---|---|---|---|',
        ]
        for row in s['rows']:
            L.append(
                f'| {row["slide"]} | {row["view"]} | {row.get("first_click_ms", row.get("error", ""))} | {row.get("next_click_ms", "")} |'
            )
        L.append('')
    if r.get('metrics'):
        L += [
            '## Region metrics',
            '',
            '| image | cells | measurements | tissue (s) | metrics (s) |',
            '|---|---|---|---|---|',
        ]
        for m in r['metrics']:
            L.append(f'| {m["image"]} | {m["cells"]:,} | {m["measurements"]} | {m["tissue_s"]} | {m["metrics_s"]} |')
        L.append('')
    f = r.get('formats') or {}
    if f.get('features'):
        L += [
            '## Annotation formats',
            '',
            f'{f["features"]:,} cells ({f["layer"]})',
            '',
            '| | size (MB) | write (s) | read (s) |',
            '|---|---|---|---|',
            f'| GeoParquet | {f["geoparquet_mb"]} | {f["geoparquet_write_s"]} | {f["geoparquet_read_s"]} |',
            f'| GeoJSON | {f["geojson_mb"]} | {f["geojson_write_s"]} | {f["geojson_read_s"]} |',
            '',
        ]
    return '\n'.join(L)
