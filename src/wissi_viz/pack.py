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

import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

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
        from ._compat import user_dir
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

    def one(x):
        dst = raw / x['file']
        part = dst.with_name(dst.name + '.part')
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(x['url'], headers={**UA, **({'Range': f'bytes={have}-'} if have else {})})
        with urllib.request.urlopen(req, timeout=600) as r:
            if have and r.status != 206:            # server ignored the range: start over
                have = 0
            with open(part, 'ab' if have else 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    with lock:
                        done[0] += len(chunk)
        if x.get('size') and part.stat().st_size != x['size']:
            raise IOError(f'{x["file"]}: got {part.stat().st_size} bytes, expected {x["size"]}')
        if x.get('md5') and _md5(part) != x['md5']:
            part.unlink()
            raise IOError(f'{x["file"]}: MD5 mismatch (deleted; run again)')
        os.replace(part, dst)
        return x['id']

    stop = threading.Event()

    def progress():
        while not stop.wait(5):
            el = time.time() - t0
            rate = done[0] / max(el, 1e-6)
            log(f'  {_human(done[0])} / {_human(total)}  ({_human(rate)}/s, ~{(total - done[0]) / max(rate, 1):.0f} s left)')
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
        raise RuntimeError('download failed for: ' + '; '.join(errors))
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
    from . import annio
    from .annotations import AnnotationLayer
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
                    os.link(src, out)                  # no copy: hard link where the file system allows it
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
    import tifffile
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
        import re
        nm = re.findall(r'<Channel[^>]*Name="([^"]+)"', meta)
        chnames.extend(nm[:len(a)] if len(nm) >= len(a) else [f'ch{len(chnames) + i}' for i in range(len(a))])
        m = re.search(r'PhysicalSizeX="([0-9.]+)"', meta)
        if m:
            mpp = float(m.group(1))
    data = np.stack(chans)
    with tifffile.TiffWriter(out, bigtiff=True, ome=True) as tw:
        opts = dict(tile=(512, 512), compression='zlib', photometric='minisblack')
        nlev = max(1, int(np.ceil(np.log2(max(data.shape[1:]) / 1024))) + 1)
        tw.write(data, subifds=nlev - 1, metadata={'axes': 'CYX', 'Channel': {'Name': chnames},
                                                    'PhysicalSizeX': mpp, 'PhysicalSizeY': mpp}, **opts)
        lv = data
        for _ in range(nlev - 1):
            lv = lv[:, ::2, ::2]
            tw.write(lv, subfiletype=1, **opts)
    if boundaries:
        _xenium_cells(tmp / boundaries, cells_out, mpp, out.name)
    shutil.rmtree(tmp, ignore_errors=True)


def _xenium_cells(parquet: Path, out: Path, mpp: float, slide: str) -> None:
    import pyarrow.parquet as pq

    from . import annio
    t = pq.read_table(parquet, columns=['cell_id', 'vertex_x', 'vertex_y'])
    cid = t.column('cell_id').to_numpy(zero_copy_only=False)
    x = t.column('vertex_x').to_numpy() / mpp
    y = t.column('vertex_y').to_numpy() / mpp
    change = np.nonzero(cid[1:] != cid[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(cid)]])
    keep = []
    for s, e in zip(starts, ends):                  # drop the repeated closing vertex of each ring
        if e - s >= 4 and x[s] == x[e - 1] and y[s] == y[e - 1]:
            e -= 1
        keep.append((s, e))
    idx = np.concatenate([np.arange(s, e) for s, e in keep])
    lens = np.array([e - s for s, e in keep])
    ring_off = np.zeros(len(keep) + 1, np.int64)
    np.cumsum(lens, out=ring_off[1:])
    n = len(keep)
    F = annio.Features(np.full(n, annio.GEO_POLYGON, np.uint8), np.stack([x[idx], y[idx]], 1), ring_off,
                       np.arange(n + 1, dtype=np.int64), np.zeros(n, np.int32), ['Cell (10x)'],
                       props={'cell_id': np.asarray([str(cid[s]) for s, _ in keep], dtype=object)})
    F.name, F.mpp = '10x cell segmentation', mpp
    annio.write_geoparquet(F, out, slide=slide)


def _attribution(sel: list[dict], d: Path) -> None:
    lines = ['# Sample pack: sources and licences', '',
             'Every image below allows reuse, including commercial use. Keep this file with the images.', '',
             '| Image | Source | Licence | Attribution |', '|---|---|---|---|']
    for x in sel:
        lines.append(f'| {x["title"]} | {x["source"]} ([link]({x["url"]})) | {x["licence"]} | {x.get("attribution", "")} |')
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
            continue                                   # Xenium bundles bring their own cell segmentation
        segment_slide(_slide(x, d), out, model=model, log=log)


def _fetch_results(sel: list[dict], d: Path, log) -> None:
    """Precomputed cells (from a GPU run of this same pack) if this project published them."""
    ld = d / 'layers'
    try:
        sums = urllib.request.urlopen(urllib.request.Request(f'{RESULTS_RELEASE}/SHA256SUMS', headers=UA), timeout=20).read().decode()
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


def analyze(sel: list[dict], d: Path, log=print) -> None:
    """Per image: tissue outline, cell density + Getis-Ord hotspots, and region metrics (tissue and expert ROIs, with
    cells) → layers/*.parquet and metrics/<id>.csv."""
    from . import annio, spatial
    from .metrics import objects_from_layer, region_metrics
    from .sources import open_slide
    ld, md = d / 'layers', d / 'metrics'
    md.mkdir(exist_ok=True)
    for x in sel:
        out = md / f'{x["id"]}.csv'
        if out.exists():
            continue
        t0 = time.time()
        src = open_slide(_slide(x, d))
        mpp = (src.info.mpp or (None,))[0]
        tis_p = ld / f'{x["id"]}.tissue.parquet'
        if not tis_p.exists():
            _, tis_layer = spatial.tissue_mask(src)
            annio.export(tis_layer, tis_p, slide=Path(_slide(x, d)).name)
        tissue = annio.load(tis_p)
        cp = _cells(x, d)
        cells = annio.load(cp) if cp else None
        objs = objects_from_layer(cells, mpp) if cells is not None else None
        if objs is not None and len(objs) and not (ld / f'{x["id"]}.density.parquet').exists():
            cls = None
            g = spatial.density_grid(objs, cls, cell_um=100, hotspots=True)
            annio.export(g.layer(name='Cell density'), ld / f'{x["id"]}.density.parquet')
            annio.export(g.hotspot_layer(name='Hotspots'), ld / f'{x["id"]}.hotspots.parquet')
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
        pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(d / 'metrics' / 'all_regions.csv', index=False)


def session(sel: list[dict], d: Path) -> Path:
    """session.json: the slides (in pack order) and their layers, for ``wissi-viz desktop --session``."""
    ld = d / 'layers'
    entries = []
    for x in sel:
        layers = [str(p) for p in sorted(ld.glob(f'{x["id"]}.*.parquet'))]
        entries.append({'id': x['id'], 'title': x['title'], 'slide': str(_slide(x, d)), 'layers': layers,
                        'licence': x['licence']})
    p = d / f'session-{len(sel)}.json'
    p.write_text(json.dumps({'name': 'wissi-viz sample pack', 'slides': entries, 'show': [e['id'] for e in entries][:32]},
                            indent=1), encoding='utf-8')
    return p


# ------------------------------------------------------------------------------------------------ entry points
def run_demo(which: str = 'quick', directory: str | None = None, model: str = 'auto', yes: bool = False,
             open_app: bool = True, browser: bool = False, log=print) -> Path:
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
        from .desktop import main as desktop_main
        desktop_main(['--session', str(s)] + (['--browser'] if browser else []))
    return s


def status(directory: str | None = None) -> dict:
    d = pack_dir(directory)
    out = {}
    for which in ('quick', 'full'):
        sel = items(which)
        out[which] = {'images': len(sel), 'bytes': sum(x['size'] for x in sel),
                      'downloaded': sum(_complete(d / 'downloads' / x['file'], x) for x in sel),
                      'segmented': sum(_cells(x, d) is not None for x in sel),
                      'analyzed': sum((d / 'metrics' / f'{x["id"]}.csv').exists() for x in sel)}
    out['dir'] = str(d)
    return out


def main(argv: list[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog='wissi-viz pack')
    ap.add_argument('step', choices=['status', 'download', 'prepare', 'segment', 'analyze', 'open', 'all'])
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
    elif a.step == 'open':
        from .desktop import main as desktop_main
        desktop_main(['--session', str(session(sel, d))] + (['--browser'] if a.browser else []))
    else:
        run_demo(a.set, a.dir, a.model, a.yes, browser=a.browser)
