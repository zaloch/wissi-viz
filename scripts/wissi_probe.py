#!/usr/bin/env python3
"""wissi_probe — check an integration and pick its settings, in one command, with a hard cap on API calls.

    python scripts/wissi_probe.py TARGET [--images a,b] [--panes 1,4,9,16] [--conn 5,8,10] [--max-calls 1000] [--yes]

TARGET is where the slides come from:

    local                     2 synthetic SVS slides (generated once, cached) served by an in-process wissi-viz
    local:a.svs,b.svs         your own files
    mock                      the bundled mock vendor API (HTTP; latency, 429s, 503s) through a connector
    mock:halolink             HALO-Link-style connector modality over the 2 SVS: sync client, DeepZoom + overlap, HALO XML
    mock:pathcore             PathCore-style connector modality: async client, coarsest-first level ids, padded PNG, µm GeoJSON
    http:my_api.json          any tile API that has plain tile URLs, described in JSON (wissi_viz.connectors.http)
    mypkg.halo:make_conn      YOUR connector: a RemoteConnection subclass or a factory returning one — e.g. the
                              HALO Link / PathCore connection wissi already uses (--connector-arg key=value …)
    https://host/wissi-viz    an already deployed wissi-viz server (--slides id1,id2)

Phases, each with its own share of the call budget:

1. **smoke**: per image ≤ 7 calls. Metadata and pyramid, coarsest / middle / full-resolution tiles (format, size,
   padding), how "no tile" is signalled, annotations and objects.
2. **api**: direct calls at each concurrency in --conn (distinct full-resolution tiles, no retries): latency,
   tiles/s, MB/s, 429 / 5xx / timeouts → the connector's best ``max_concurrency``.
3. **e2e**: viewer sessions through an in-process wissi-viz server for every (panes × conn): open 1–16 different
   images, then pan / zoom / jump. Each cold run uses a fresh cache; a warm rerun of the same session shows
   cache-served speed and makes no API calls. Reports view latency (time until a move is sharp), tile latency,
   errors, and upstream calls vs viewer requests.

Output (--out DIR, default ./wissi-probe-<target>): report.md, results.json, recommended_settings.py.

Without --yes the script prints the plan (including the exact API-call budget) and stops; local and mock targets
run without it. The call budget is enforced by a guard around the connector, not just by planning: once it is
used up, no further call reaches the API.

Needs: wissi-viz (this repo), httpx, numpy, uvicorn; tifffile + imagecodecs for generated slides.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import wissi_loadtest as LT  # noqa: E402  (same folder)

SMOKE_CALLS_PER_IMAGE = 7


# ============================================================================================ budget guard
class BudgetExhausted(Exception):
    pass


class Budget:
    """Every call to the API goes through take(); past the limit the call is refused before it is made."""

    def __init__(self, limit: int):
        self.limit, self.used, self.refused = int(limit), 0, 0
        self.by_kind: dict = {}
        self._lock = threading.Lock()

    def take(self, kind: str) -> None:
        with self._lock:
            if self.used >= self.limit:
                self.refused += 1
                raise BudgetExhausted(f'API call budget of {self.limit} used up')
            self.used += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1

    @property
    def left(self) -> int:
        return max(0, self.limit - self.used)


def guard(conn, budget: Budget):
    """Wrap the connector's API methods (sync or async) so each call is counted and capped."""
    from wissi_viz.connectors import RemoteConnection
    for name, kind in (('image_info', 'info'), ('tile', 'tile'), ('region', 'tile'), ('annotations', 'annotations'),
                       ('objects', 'objects')):
        if getattr(type(conn), name) is getattr(RemoteConnection, name):
            continue                                             # not implemented by this connector
        fn = getattr(conn, name)
        if inspect.iscoroutinefunction(fn):
            async def w(*a, _fn=fn, _k=kind, **kw):
                budget.take(_k)
                return await _fn(*a, **kw)
        else:
            def w(*a, _fn=fn, _k=kind, **kw):
                budget.take(_k)
                return _fn(*a, **kw)
        setattr(conn, name, w)
    return conn


def msg(e: BaseException) -> str:
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:200]


def classify(e: BaseException) -> str:
    from wissi_viz.connectors import RetryAfter, TileNotFound
    if isinstance(e, BudgetExhausted):
        return 'budget'
    if isinstance(e, TileNotFound):
        return 'empty'
    if isinstance(e, RetryAfter):
        return 'rate_limited'
    n = type(e).__name__.lower()
    if 'timeout' in n:
        return 'timeout'
    sc = getattr(getattr(e, 'response', None), 'status_code', None)
    if sc:
        return f'http_{sc}'
    if 'connect' in n:
        return 'connection'
    return 'error'


# ============================================================================================ slides
def make_svs(path: Path, W: int, H: int, seed: int) -> Path:
    """A synthetic Aperio-style SVS (H&E-like texture, 256² JPEG tiles, levels 1/4/16 + thumbnail), written
    tile by tile so memory stays small."""
    import numpy as np
    import tifffile
    rng = np.random.default_rng(seed)
    G = 512
    gh, gw = H // G + 3, W // G + 3
    c = rng.random((gh, gw)).astype(np.float32)
    for _ in range(3):
        c = (c + np.roll(c, 1, 0) + np.roll(c, -1, 0) + np.roll(c, 1, 1) + np.roll(c, -1, 1)) / 5
    c = (c - c.min()) / (np.ptp(c) + 1e-6)

    def render(ds: float, x0: int, y0: int, w: int, h: int):
        xs = (x0 + np.arange(w, dtype=np.float32) + 0.5) * ds
        ys = (y0 + np.arange(h, dtype=np.float32) + 0.5) * ds
        fx, fy = np.clip(xs / G, 0, gw - 2), np.clip(ys / G, 0, gh - 2)
        ix, iy = fx.astype(int), fy.astype(int)
        ax, ay = (fx - ix)[None, :], (fy - iy)[:, None]
        t = (c[iy][:, ix] * (1 - ax) * (1 - ay) + c[iy][:, ix + 1] * ax * (1 - ay)
             + c[iy + 1][:, ix] * (1 - ax) * ay + c[iy + 1][:, ix + 1] * ax * ay)
        tis = np.clip((t - 0.42) * 5, 0, 1)
        C = 24.0
        gx, gy = np.floor(xs / C)[None, :], np.floor(ys / C)[:, None]
        hsh = np.modf(np.abs(np.sin(gx * 12.9898 + gy * 78.233 + seed) * 43758.5453))[0]
        dx = xs[None, :] / C - gx - 0.5 - (hsh - 0.5) * 0.4
        dy = ys[:, None] / C - gy - 0.5 - (np.modf(hsh * 7.31)[0] - 0.5) * 0.4
        nuc = np.clip(1.4 - np.sqrt(dx * dx + dy * dy) / (0.18 + 0.1 * hsh), 0, 1) * (hsh > 0.25)
        nuc *= min(1.0, 2.0 / ds)
        white, eos, hem = (np.array(v, np.float32) for v in ((245, 245, 245), (236, 150, 190), (70, 50, 140)))
        base = white * (1 - tis[..., None]) + eos * tis[..., None]
        px = base * (1 - (nuc * tis)[..., None]) + hem * (nuc * tis)[..., None]
        px *= 1 + rng.normal(0, 0.03, px.shape[:2]).astype(np.float32)[..., None]
        return np.clip(px, 0, 255).astype(np.uint8)

    def tiles(ds: int, lw: int, lh: int):
        for ty in range(math.ceil(lh / 256)):
            for tx in range(math.ceil(lw / 256)):
                yield render(ds, tx * 256, ty * 256, 256, 256)
    head = f'Aperio Image Library v12.0.15 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=85'
    o = dict(tile=(256, 256), compression='jpeg', compressionargs={'level': 85}, photometric='rgb', metadata=None, dtype='uint8')
    tmp = path.with_suffix('.tmp')
    with tifffile.TiffWriter(tmp, bigtiff=True) as tw:
        tw.write(tiles(1, W, H), shape=(H, W, 3), description=f'{head}|AppMag = 40|MPP = 0.2500|Filename = {path.stem}', **o)
        tds = max(W, H) / 1024
        tw.write(render(tds, 0, 0, int(W / tds), int(H / tds)), description=f'{head} -> thumbnail', photometric='rgb',
                 compression='jpeg', metadata=None)
        for ds in (4, 16):
            lw, lh = math.ceil(W / ds), math.ceil(H / ds)
            if max(lw, lh) < 512:
                break
            tw.write(tiles(ds, lw, lh), shape=(lh, lw, 3), description=f'{head} -> {lw}x{lh}', **o)
    tmp.replace(path)
    return path


def local_slides(spec: str, workdir: Path, size: str) -> list[Path]:
    if ':' in spec:
        files = [Path(p).expanduser() for p in spec.split(':', 1)[1].split(',') if p]
        for f in files:
            if not f.exists():
                sys.exit(f'{f}: not found')
        return files
    W, H = (int(v) for v in size.lower().split('x'))
    workdir.mkdir(parents=True, exist_ok=True)
    out = []
    for k in (1, 2):
        p = workdir / f'probe_{k}_{W}x{H}.svs'
        if not p.exists():
            print(f'  generating {p.name} ({W}×{H}) …', file=sys.stderr, flush=True)
            make_svs(p, W, H, seed=k)
        out.append(p)
    return out


# ============================================================================================ targets
def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Target:
    kind = ''            # local | connector | wissi
    label = ''
    conn = None          # RemoteConnection (connector targets)
    images: list = []
    files: list = []
    base = ''            # deployed wissi-viz
    vendor_stats = None  # callable → dict (mock: the vendor's own request counter)
    cleanup: list = []


def start_mock(files: list[Path], args) -> tuple[str, subprocess.Popen]:
    import httpx
    import tempfile
    port = free_port()
    log = tempfile.NamedTemporaryFile('w+', suffix='.log', prefix='wissi-mock-', delete=False, encoding='utf-8')
    p = subprocess.Popen([sys.executable, str(ROOT / 'examples' / 'tile_api' / 'mock_tile_api.py'), *map(str, files),
                          '--port', str(port), '--token', 'probe', '--n', '4', '--latency', args.mock_latency,
                          '--limit', str(args.mock_limit), '--errors', str(args.mock_errors)],
                         stdout=subprocess.DEVNULL, stderr=log)   # a file, not a pipe: never blocks the child
    url = f'http://127.0.0.1:{port}'
    for _ in range(300):
        try:
            if httpx.get(url + '/api/_stats', timeout=1).status_code == 200:
                return url, p
        except Exception:  # noqa: BLE001
            pass
        if p.poll() is not None:
            log.flush()
            sys.exit('mock API failed to start:\n' + Path(log.name).read_text(encoding='utf-8', errors='replace')[-2000:])
        time.sleep(0.2)
    p.kill()
    sys.exit('mock API did not start')


def mock_modality(kind: str, files: list[Path], latency: str, fail_rate: float):
    """In-process HALO-Link-style (sync client, DeepZoom 254+1 overlap, HALO XML + cell table) or PathCore-style
    (async client, coarsest-first string level ids, padded PNG tiles, µm GeoJSON) connector over the local slides —
    the same mocks tests/test_remote.py verifies pixel for pixel — with simulated network latency."""
    import asyncio
    sys.path.insert(0, str(ROOT / 'tests'))
    from mock_remote import MockHaloLink, MockPathCore

    from wissi_viz.connectors import HaloLinkConnection, PathCoreConnection
    lo, hi = (float(v) / 1000 for v in latency.split(','))
    rng = random.Random(0)
    if kind == 'halolink':
        inner = {f'halo-{k + 1}': MockHaloLink(str(f), fail_rate=fail_rate) for k, f in enumerate(files)}

        class ProbeHaloLink(HaloLinkConnection):            # a sync client, as HALO integrations usually are
            def image_info(self, image_id):
                time.sleep(rng.uniform(lo, hi))
                return inner[image_id].image_info(image_id)

            def tile(self, image_id, level, col, row, channel=None):
                time.sleep(rng.uniform(lo, hi))
                return inner[image_id].tile(image_id, level, col, row, channel)

            def annotations(self, image_id):
                return inner[image_id].annotations(image_id)

            def objects(self, image_id):
                return inner[image_id].objects(image_id)
        return ProbeHaloLink(), list(inner)
    inner = {f'pathcore-{k + 1}': MockPathCore(str(f)) for k, f in enumerate(files)}

    class ProbePathCore(PathCoreConnection):                # an async client
        async def image_info(self, slide_id):
            await asyncio.sleep(rng.uniform(lo, hi))
            return await inner[slide_id].image_info(slide_id)

        async def tile(self, slide_id, level, col, row, channel=None):
            await asyncio.sleep(rng.uniform(lo, hi))
            return await inner[slide_id].tile(slide_id, level, col, row, channel)

        async def annotations(self, slide_id):
            return await inner[slide_id].annotations(slide_id)
    return ProbePathCore(), list(inner)


def load_factory(spec: str, kwargs: dict):
    mod, _, attr = spec.partition(':')
    sys.path.insert(0, os.getcwd())
    obj = getattr(importlib.import_module(mod), attr)
    from wissi_viz.connectors import RemoteConnection
    conn = obj(**kwargs) if callable(obj) else obj
    if not isinstance(conn, RemoteConnection):
        sys.exit(f'{spec} returned {type(conn).__name__}, not a wissi_viz.connectors.RemoteConnection')
    return conn


def connector_kind(conn) -> str:
    from wissi_viz.connectors import HaloLinkConnection, PathCoreConnection
    if isinstance(conn, HaloLinkConnection):
        return 'halolink'
    if isinstance(conn, PathCoreConnection):
        return 'pathcore'
    return getattr(conn, 'kind', None) or 'custom'


def make_target(args) -> Target:
    t = Target()
    t.cleanup = []
    spec = args.target
    kw = dict(x.split('=', 1) for x in args.connector_arg)
    if spec == 'local' or spec.startswith('local:'):
        t.kind, t.label = 'local', 'local files'
        t.files = local_slides(spec, Path(args.workdir), args.local_size)
        t.images = [str(f) for f in t.files]
    elif spec in ('mock:halolink', 'mock:pathcore'):
        kind = spec.split(':')[1]
        files = local_slides('local', Path(args.workdir), args.local_size)
        t.kind = 'connector'
        t.conn, t.images = mock_modality(kind, files, args.mock_latency, args.mock_errors)
        t.label = f'{kind} connector, mock modality (in-process, {args.mock_latency} ms latency)'
    elif spec == 'mock':
        sys.path.insert(0, str(ROOT / 'examples' / 'tile_api'))
        from connector import DemoApiConnection
        files = local_slides('local', Path(args.workdir), args.local_size)
        url, proc = start_mock(files, args)
        t.cleanup.append(proc.kill)
        t.kind, t.label = 'connector', f'mock vendor API ({url})'
        t.conn = DemoApiConnection(url, 'probe')
        t.images = (args.images or 'img-01,img-04').split(',')   # a DeepZoom image and a 'tiles'-style one
        import httpx
        t.vendor_stats = lambda: httpx.get(url + '/api/_stats', timeout=5).json()
    elif spec.startswith('http:') and not spec.startswith('http://'):
        from wissi_viz.connectors import HttpTileConnection
        t.kind, t.conn = 'connector', HttpTileConnection(spec[5:])
        t.label = f"{t.conn.kind} via HTTP config {spec[5:]} ({t.conn.config['base_url']})"
    elif spec.startswith(('http://', 'https://')):
        t.kind, t.label, t.base = 'wissi', f'deployed wissi-viz {spec}', spec.rstrip('/')
    elif ':' in spec:
        t.kind, t.conn = 'connector', load_factory(spec, kw)
        t.label = f'{connector_kind(t.conn)} connector {spec}'
    else:
        sys.exit(f'unknown target {spec!r} (local, local:a.svs,b.svs, mock, http:config.json, module:factory, https://…/wissi-viz)')
    if t.kind == 'connector' and not t.images:
        if not args.images:
            lister = getattr(t.conn, 'list_images', None)
            sys.exit('pass --images id1,id2' + (' (or let the connector list them: it has list_images())' if lister else ''))
        t.images = args.images.split(',')
    if t.kind == 'wissi' and args.slides:
        t.images = args.slides.split(',')
    return t


def args_target_is_mock(t: Target) -> bool:
    return 'mock modality' in t.label


def is_local_host(t: Target) -> bool:
    if t.kind == 'local' or t.label.startswith('mock') or args_target_is_mock(t):
        return True
    url = t.base or (getattr(t.conn, 'config', {}) or {}).get('base_url', '')
    return (urlparse(url).hostname or '') in ('localhost', '127.0.0.1', '::1')


# ============================================================================================ phase 1: smoke
def check(rows, name, status, detail='', ms=None):
    rows.append({'check': name, 'status': status, 'detail': detail, 'ms': None if ms is None else round(ms, 1)})


def tile_check(rows, label, fetch, expect):
    """fetch() → raw tile (bytes / array / None). expect = (w, h) of the tile once overlap/padding are removed."""
    import numpy as np
    from wissi_viz.sources.callback import _decode, sniff_media_type
    t0 = time.perf_counter()
    try:
        out = fetch()
    except Exception as e:  # noqa: BLE001
        check(rows, label, 'FAIL', f'{classify(e)}: {msg(e)}', 1000 * (time.perf_counter() - t0))
        return None
    ms = 1000 * (time.perf_counter() - t0)
    if out is None:
        check(rows, label, 'WARN', 'no data (None) for a tile inside the image', ms)
        return None
    if isinstance(out, (bytes, bytearray, memoryview)):
        b = bytes(out)
        mt = sniff_media_type(b)
        try:
            a = np.asarray(_decode(b))
        except Exception as e:  # noqa: BLE001
            check(rows, label, 'FAIL', f'{len(b)} bytes, not decodable ({e})', ms)
            return None
        kind = f'{mt or "encoded"} {len(b) / 1024:.0f} KB'
    else:
        a, kind = np.asarray(out), f'array {np.asarray(out).dtype}'
    h, w = a.shape[:2]
    ew, eh = expect
    note = 'exact size' if (w, h) == (ew, eh) else ('padded/overlap (cropped by wissi-viz)' if w >= ew and h >= eh else 'SMALLER than expected')
    check(rows, label, 'PASS' if w >= ew and h >= eh else 'FAIL', f'{kind}, {w}×{h} for a {ew}×{eh} area: {note}', ms)
    return a


def smoke_connector(t: Target, budget: Budget, args) -> tuple[list, dict]:
    from wissi_viz.sources.remote import resolve
    rows, sources = [], {}
    conn = t.conn
    for img in t.images:
        pre = f'[{img}] '
        t0 = time.perf_counter()
        try:
            src = conn.source(img)                     # image_info, with the connector's retry policy
        except Exception as e:  # noqa: BLE001
            check(rows, pre + 'metadata', 'FAIL', f'{classify(e)}: {msg(e)}', 1000 * (time.perf_counter() - t0))
            continue
        sources[img] = src
        i, P = src.info, src.pyramid
        lv = ', '.join(f'{L.width}×{L.height}' for L in i.levels[:6]) + (' …' if len(i.levels) > 6 else '')
        check(rows, pre + 'metadata', 'PASS', f'{i.width}×{i.height}, tile {i.tile_size}{f" overlap {P.overlap}" if P.overlap else ""}, '
              f'{len(i.levels)} levels ({lv}), mode {i.mode}', 1000 * (time.perf_counter() - t0))
        check(rows, pre + 'mpp', 'PASS' if i.mpp else 'WARN', f'{i.mpp}' if i.mpp else 'missing: µm units, scale bar and magnification unavailable')
        top_w, top_h = i.levels[-1].width, i.levels[-1].height
        n_top = math.ceil(top_w / i.tile_size) * math.ceil(top_h / i.tile_size)
        check(rows, pre + 'pyramid depth', 'PASS' if n_top <= 4 else 'WARN',
              f'coarsest level {top_w}×{top_h} = {n_top} tile(s)' + ('' if n_top <= 4 else
              f': the first (zoomed-out) view of each slide needs all {n_top} — wissi-viz builds the smaller overview levels '
              'from them, once per slide (then cached). If the API has a thumbnail / overview endpoint, expose it as an extra '
              'coarse level in image_info(); a persistent cache_dir makes this a one-time cost per slide'))
        check(rows, pre + 'version', 'PASS' if src.version not in ('', '0', 'None') else 'WARN',
              src.version if src.version not in ('', '0', 'None') else 'missing: cached tiles never invalidate when the image changes')
        ch = 0 if i.mode == 'channels' else None

        def fetch(L, c, r, _img=img, _src=src):
            return lambda: resolve(conn.tile(_img, _src.pyramid.level_ids[L], c, r, ch), conn.loop, conn.timeout)
        top = len(i.levels) - 1
        mid = top // 2
        for name, L in (('coarsest tile', top), ('middle-level tile', mid), ('full-resolution tile', 0)):
            lw, lh = i.levels[L].width, i.levels[L].height
            c, r = (lw // 2) // i.tile_size, (lh // 2) // i.tile_size
            _, _, ew, eh = src.tile_rect(L, c, r)
            tile_check(rows, pre + f'{name} (level {L}, {c},{r})', fetch(L, c, r), (ew, eh))
        nc = math.ceil(i.levels[0].width / i.tile_size)
        t0 = time.perf_counter()
        try:
            out = fetch(0, nc + 3, 0)()
            how = 'returns None (treated as background)' if out is None else f'returns data ({type(out).__name__}) — fine if intentional'
            check(rows, pre + 'tile outside the image', 'INFO', how, 1000 * (time.perf_counter() - t0))
        except Exception as e:  # noqa: BLE001
            k = classify(e)
            check(rows, pre + 'tile outside the image', 'PASS' if k == 'empty' else 'INFO',
                  'TileNotFound → drawn as background, cached' if k == 'empty' else f'{k}: {msg(e)} (wissi-viz never asks for these)',
                  1000 * (time.perf_counter() - t0))
        if args.no_annotations:
            continue
        t0 = time.perf_counter()
        try:
            layers = conn.load_annotations(img, i.mpp)
            src._ann = (lambda _l=layers: _l)          # the viewer runs reuse these (no per-pane re-fetch)
            desc = ', '.join(f'{L.name} ({L.compiled().n} features)' for L in layers) or 'none'
            check(rows, pre + 'annotations + objects', 'PASS', desc + ' — wissi-viz fetches these once per slide when a viewer '
                  'first opens it (2 calls, in the background)', 1000 * (time.perf_counter() - t0))
        except NotImplementedError:
            check(rows, pre + 'annotations + objects', 'INFO', 'not implemented by this connector')
        except Exception as e:  # noqa: BLE001
            check(rows, pre + 'annotations + objects', 'FAIL', f'{classify(e)}: {msg(e)}', 1000 * (time.perf_counter() - t0))
    return rows, sources


def smoke_local(t: Target) -> tuple[list, dict]:
    from wissi_viz.sources import open_slide
    rows, sources = [], {}
    for f in t.files:
        t0 = time.perf_counter()
        try:
            src = open_slide(str(f))
        except Exception as e:  # noqa: BLE001
            check(rows, f'[{f.name}] open', 'FAIL', str(e)[:300])
            continue
        i = src.info
        check(rows, f'[{f.name}] open', 'PASS', f'{i.width}×{i.height}, {len(i.levels)} native levels, tile {i.tile_size}, '
              f'mpp {i.mpp}, {f.stat().st_size / 2**20:.0f} MB', 1000 * (time.perf_counter() - t0))
        sources[str(f)] = src
    return rows, sources


def smoke_wissi(t: Target, args, headers) -> tuple[list, list]:
    import httpx
    rows = []
    with httpx.Client(timeout=args.timeout, headers=headers, verify=not args.insecure) as cli:
        t0 = time.perf_counter()
        try:
            ids = t.images or [d['id'] for d in cli.get(t.base + '/slides').raise_for_status().json()][:16]
            check(rows, 'slide list', 'PASS', f'{len(ids)} slide(s)', 1000 * (time.perf_counter() - t0))
        except Exception as e:  # noqa: BLE001
            check(rows, 'slide list', 'FAIL', str(e)[:300])
            return rows, []
        ok = []
        for sid in ids[:args.max_images]:
            t0 = time.perf_counter()
            r = cli.get(f'{t.base}/slides/{sid}/info.json')
            if r.status_code != 200:
                check(rows, f'[{sid}] info.json', 'FAIL', f'HTTP {r.status_code}', 1000 * (time.perf_counter() - t0))
                continue
            d = r.json()
            check(rows, f'[{sid}] info.json', 'PASS', f"{d['width']}×{d['height']}, {len(d['levels'])} levels", 1000 * (time.perf_counter() - t0))
            top = len(d['levels']) - 1
            for L in (top, 0):
                lw, lh = d['levels'][L]['width'], d['levels'][L]['height']
                c, rr = (lw // 2) // d['tile_size'], (lh // 2) // d['tile_size']
                t0 = time.perf_counter()
                q = cli.get(f"{t.base}/slides/{sid}/tiles/{L}/{c}/{rr}?v={d.get('version', '')}")
                check(rows, f'[{sid}] tile level {L}', 'PASS' if q.status_code == 200 else 'FAIL',
                      f"HTTP {q.status_code}, {q.headers.get('content-type')}, {len(q.content) / 1024:.0f} KB, cache-control: {q.headers.get('cache-control')}",
                      1000 * (time.perf_counter() - t0))
            ok.append(sid)
        r = cli.get(t.base + '/stats')
        check(rows, 'server stats', 'PASS' if r.status_code == 200 else 'INFO', f'HTTP {r.status_code}')
    return rows, ok


# ============================================================================================ phase 2: direct API / server
def probe_tiles(src, n: int, rng: random.Random) -> list:
    """n distinct full-resolution tiles from the central 60 % of the image (tissue, not background)."""
    i = src.info
    ts = i.tile_size
    nx, ny = math.ceil(i.levels[0].width / ts), math.ceil(i.levels[0].height / ts)
    cand = [(c, r) for r in range(int(ny * 0.2), max(int(ny * 0.8), int(ny * 0.2) + 1))
            for c in range(int(nx * 0.2), max(int(nx * 0.8), int(nx * 0.2) + 1))]
    rng.shuffle(cand)
    return cand[:n]


def direct_phase(t: Target, sources: dict, budget: Budget, args) -> dict:
    """Sequential latency, then a concurrency sweep — each call a distinct, never-fetched full-resolution tile."""
    from wissi_viz.sources.remote import resolve
    rng = random.Random(args.seed)
    imgs = [k for k in t.images if k in sources]
    if not imgs:
        return {}
    plan = [('sequential', 1, args.seq_calls)] + [(f'c{c}', c, max(20, 3 * c)) for c in args.conn]
    need = sum(n for _, _, n in plan)
    pool = {img: probe_tiles(sources[img], need, rng) for img in imgs}
    out = {}
    if t.kind == 'local':
        from wissi_viz import TileService
        from wissi_viz.tiles import PyramidSlide
        svc = TileService(memory_bytes=512 * 2**20, workers=max(args.conn) + 2)
        pyrs = {img: PyramidSlide(sources[img]) for img in imgs}

        def call(img, c, r):
            return svc.get(f'probe-{img}', pyrs[img], '0', 0, c, r, None).result()[0]
    else:
        conn = t.conn

        def call(img, c, r):
            src = sources[img]
            return resolve(conn.tile(img, src.pyramid.level_ids[0], c, r, None), conn.loop, conn.timeout)
    k = 0
    for name, conc, n in plan:
        if t.kind != 'local' and budget.left < n:
            out[name] = {'skipped': f'budget: {budget.left} calls left, needs {n}'}
            continue
        jobs = []
        for _ in range(n):
            img = imgs[k % len(imgs)]
            c, r = pool[img][(k // len(imgs)) % len(pool[img])]
            jobs.append((img, c, r))
            k += 1
        res = []

        def one(j):
            t0 = time.perf_counter()
            try:
                b = call(*j)
                size = len(b) if isinstance(b, (bytes, bytearray)) else (getattr(b, 'nbytes', 0) if b is not None else 0)
                return ('ok' if b is not None else 'empty', time.perf_counter() - t0, size)
            except Exception as e:  # noqa: BLE001
                return (classify(e), time.perf_counter() - t0, 0)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(one, jobs))
        wall = time.perf_counter() - t0
        lat = sorted(x[1] for x in res if x[0] in ('ok', 'empty'))
        st: dict = {}
        for x in res:
            st[x[0]] = st.get(x[0], 0) + 1
        bad = sum(v for kk, v in st.items() if kk not in ('ok', 'empty'))
        out[name] = {'concurrency': conc, 'calls': n, 'wall_s': round(wall, 2), 'tiles_per_s': round(n / wall, 1),
                     'MB_per_s': round(sum(x[2] for x in res) / 2**20 / wall, 2),
                     'kB_per_tile': round(sum(x[2] for x in res) / 1024 / max(1, st.get('ok', 0)), 1),
                     'p50_ms': round(1000 * LT._pct(lat, 50), 1) if lat else None,
                     'p95_ms': round(1000 * LT._pct(lat, 95), 1) if lat else None,
                     'p99_ms': round(1000 * LT._pct(lat, 99), 1) if lat else None,
                     'status': st, 'error_pct': round(100 * bad / n, 1)}
        print(f"    {name:>10}: {out[name]['tiles_per_s']:>6} tiles/s  p50 {out[name]['p50_ms']} ms  p95 {out[name]['p95_ms']} ms  "
              f"{out[name]['MB_per_s']} MB/s  {st}", file=sys.stderr, flush=True)
    if t.kind == 'local':
        svc.close()
    return out


def direct_phase_wissi(t: Target, ids: list, args, headers, budget: Budget) -> dict:
    """Deployed server: distinct full-resolution tiles (cold unless someone viewed them) at each concurrency."""
    import httpx
    rng = random.Random(args.seed)
    with httpx.Client(timeout=args.timeout, headers=headers, verify=not args.insecure) as cli:
        infos = {}
        for sid in ids:
            budget.take('info')
            infos[sid] = cli.get(f'{t.base}/slides/{sid}/info.json').json()
    urls = []
    for sid, d in infos.items():
        ts = d['tile_size']
        nx, ny = math.ceil(d['levels'][0]['width'] / ts), math.ceil(d['levels'][0]['height'] / ts)
        cand = [(c, r) for r in range(int(ny * 0.2), int(ny * 0.8) + 1) for c in range(int(nx * 0.2), int(nx * 0.8) + 1)]
        rng.shuffle(cand)
        urls += [f"{t.base}/slides/{sid}/tiles/0/{c}/{r}?v={d.get('version', '')}" for c, r in cand[:400]]
    rng.shuffle(urls)
    out, k = {}, 0
    for name, conc, n in [('sequential', 1, args.seq_calls)] + [(f'c{c}', c, max(20, 3 * c)) for c in args.conn]:
        if budget.left < n:
            out[name] = {'skipped': f'budget: {budget.left} requests left'}
            continue
        jobs = urls[k:k + n]
        k += n
        for _ in jobs:
            budget.take('tile')
        lim = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
        with httpx.Client(timeout=args.timeout, headers=headers, verify=not args.insecure, limits=lim) as cli:
            def one(u):
                t0 = time.perf_counter()
                try:
                    r = cli.get(u)
                    return ('ok' if r.status_code == 200 else f'http_{r.status_code}', time.perf_counter() - t0, len(r.content))
                except Exception as e:  # noqa: BLE001
                    return (classify(e), time.perf_counter() - t0, 0)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                res = list(ex.map(one, jobs))
            wall = time.perf_counter() - t0
        lat = sorted(x[1] for x in res if x[0] == 'ok')
        st: dict = {}
        for x in res:
            st[x[0]] = st.get(x[0], 0) + 1
        out[name] = {'concurrency': conc, 'calls': n, 'tiles_per_s': round(n / wall, 1),
                     'MB_per_s': round(sum(x[2] for x in res) / 2**20 / wall, 2),
                     'p50_ms': round(1000 * LT._pct(lat, 50), 1) if lat else None,
                     'p95_ms': round(1000 * LT._pct(lat, 95), 1) if lat else None,
                     'p99_ms': round(1000 * LT._pct(lat, 99), 1) if lat else None,
                     'status': st, 'error_pct': round(100 * (n - st.get('ok', 0)) / n, 1)}
        print(f"    {name:>10}: {out[name]['tiles_per_s']:>6} tiles/s  p95 {out[name]['p95_ms']} ms  {st}", file=sys.stderr, flush=True)
    return out


# ============================================================================================ phase 3: end to end
def serve(registry) -> tuple[str, object]:
    import uvicorn

    from wissi_viz.server import create_app
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(registry), host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.02)
    return f'http://127.0.0.1:{port}/wissi-viz', srv


def lt_args(base: str, panes: int, conn: int, cap: int, args, slides_ids: list, stats: bool) -> SimpleNamespace:
    return SimpleNamespace(requests=cap, panes=panes, conn=conn, think=(0.15, 0.5), ramp=0.0, max_seconds=args.seconds,
                           timeout=args.timeout, http2=False, insecure=args.insecure, mode='wissi', base=base,
                           overlays=False, overlay_mode='server', ok_404=False, stats_url=[base + '/stats'] if stats else [],
                           procs=1, distinct=True, slides=','.join(slides_ids), max_slides=64, header=[])


def e2e_caps(panes: list, conns: list, users: int, available: int) -> dict:
    """Split the budget over the runs, larger pane counts get more (opening 16 images costs more requests)."""
    w = {p: 4 + p for p in panes}
    unit = available / max(1, sum(w.values()) * len(conns) * users)
    return {p: int(unit * w[p]) for p in panes}


def e2e_phase(t: Target, sources: dict, budget: Budget, args, headers) -> list:
    from wissi_viz import SlideRegistry, TileService
    runs = []
    P = max(args.panes)
    for src in sources.values():                       # one annotation fetch per image for the whole probe
        if t.kind == 'connector' and not getattr(src, '_probe_memo', False):
            a = src._ann
            if args.no_annotations:
                src._ann = list
            elif callable(a):
                box = {}
                src._ann = (lambda _a=a, _b=box: _b.setdefault('v', _a()))
            src._probe_memo = True
    stop = False
    w = {p: 4 + p for p in args.panes}
    # what opening one slide costs the API: its coarsest level's tiles (wissi-viz builds the overview from them)
    keys = list(sources)
    open_cost = {}
    for img, src in sources.items():
        i = src.info
        open_cost[img] = max(1, math.ceil(i.levels[-1].width / i.tile_size) * math.ceil(i.levels[-1].height / i.tile_size))

    def open_calls(n):
        return sum(open_cost[keys[k % len(keys)]] for k in range(n)) if keys else n

    def need(p):                                       # open p slides + a few moves (≈ 2 tiles per pane per move × 3)
        return open_calls(min(P, p + args.users - 1)) + 6 * p * args.users + 10
    selected, skipped = [], {}
    per_req = 1.0                                      # API calls per viewer request, measured as runs complete
    if t.kind == 'connector':
        # priority: every pane count once (first --conn value), then the other connection counts; keep what fits
        prio = [(p, args.conn[0]) for p in args.panes] + [(p, c) for c in args.conn[1:] for p in args.panes]
        left = budget.left
        for run in prio:
            if need(run[0]) <= left:
                selected.append(run)
                left -= need(run[0])
            else:
                skipped[run] = (f'needs ≈ {need(run[0])} API calls (≈ {open_calls(run[0])} to open {run[0]} slide(s): '
                                f'{", ".join(sorted({str(v) for v in open_cost.values()}))} per slide) and only {left} remain — raise --max-calls')
        print(f'   plan: {len(selected)} of {len(prio)} runs fit the remaining {budget.left} API calls'
              + (f'; opening one slide costs ≈ {max(open_cost.values())} calls' if open_cost else ''), file=sys.stderr, flush=True)
    caps = (e2e_caps(args.panes, args.conn, args.users, budget.left) if t.kind == 'wissi'
            else {p: args.requests for p in args.panes})
    for p in args.panes:
        for c in args.conn:
            if t.kind == 'connector':
                if (p, c) in skipped:
                    runs.append({'panes': p, 'conn': c, 'skipped': skipped[(p, c)]})
                    print(f"    {p:>2} panes × {c:>2} conn: skipped ({skipped[(p, c)]})", file=sys.stderr, flush=True)
                    continue
                rest = [r for r in selected if (r[0], args.conn.index(r[1])) >= (p, args.conn.index(c))]
                spare = max(0, budget.left - sum(need(q) for q, _ in rest))
                allow = need(p) + spare * w[p] / sum(w[q] for q, _ in rest)
                # requests: what the allowance buys at the measured calls per request, keeping a 15 % margin
                caps[p] = max(3 * p, int(0.85 * (allow - open_calls(p)) / per_req / args.users) + p)
            cap = min(LT.MAX_REQUESTS_PER_USER, caps[p] if t.kind != 'local' else args.requests)
            if stop or (t.kind != 'local' and (cap < 12 or budget.left < 12)):
                runs.append({'panes': p, 'conn': c, 'skipped': f'budget ({budget.left} calls left)'})
                continue
            row = {'panes': p, 'conn': c, 'users': args.users, 'cap_per_user': cap}
            if t.kind == 'wissi':
                ids = t.images or []
                base, srv, reg = t.base, None, None
                for _ in range(args.users * cap):
                    try:
                        budget.take('viewer')
                    except BudgetExhausted:
                        break
            else:
                reg = SlideRegistry(TileService(memory_bytes=args.memory_mb * 2**20, workers=args.workers), warm=False)
                ids = []
                keys = list(sources)
                for k in range(P):
                    sid = f'p{k:02d}'
                    reg.add(sources[keys[k % len(keys)]] if t.kind == 'connector' else str(keys[k % len(keys)]), slide_id=sid)
                    ids.append(sid)
                base, srv = serve(reg)
            a = lt_args(base, p, c, cap, args, ids, stats=t.kind != 'wissi')
            slides = LT.discover(a, headers)
            for phase in ('cold', 'warm') if t.kind != 'wissi' else ('cold',):
                b0, r0, c0 = budget.used, budget.refused, (t.conn.stats() if t.conn else {})
                v0 = t.vendor_stats() if t.vendor_stats else None
                s = LT.run_stage(args.users, slides, a, args.seed + p * 100 + c, headers)
                up = {'calls': budget.used - b0}
                if t.conn:
                    c1 = t.conn.stats()
                    up.update({k: round(c1.get(k, 0) - c0.get(k, 0), 1) for k in ('retries', 'rate_limited', 'not_found', 'errors')})
                if v0 is not None:
                    v1 = t.vendor_stats()
                    up['vendor_requests'] = v1['requests'] - v0['requests']
                    up['vendor_429'] = v1['r429'] - v0['r429']
                    up['vendor_503'] = v1['r503'] - v0['r503']
                tiles = (s.get('by_kind', {}).get('tile') or {}).get('requests', 0)
                srv_stats = next(iter((s.get('server') or {}).values()), {})
                row[phase] = {
                    'requests': s.get('requests'), 'tile_requests': tiles, 'MB': s.get('MB'), 'throughput_rps': s.get('throughput_rps'),
                    'tile_p50_ms': s.get('latency_ms', {}).get('p50'), 'tile_p95_ms': s.get('latency_ms', {}).get('p95'),
                    'view_p50_ms': s.get('view_ms', {}).get('p50'), 'view_p95_ms': s.get('view_ms', {}).get('p95'),
                    'error_pct': s.get('error_rate_pct'), 'errors': s.get('errors'), 'upstream': up,
                    'upstream_per_tile': round(up['calls'] / tiles, 3) if tiles and t.kind == 'connector' else None,
                    'server_produce_ms': srv_stats.get('avg_produce_ms'),
                    'budget_exhausted': budget.refused > r0,
                }
                if phase == 'cold' and t.kind == 'connector' and s.get('requests'):
                    per_req = max(per_req, (up['calls'] - open_calls(p)) / max(1, s['requests'] - p))
                if budget.refused > r0:                # numbers of this run are not valid; stop the matrix
                    stop = True
                    print('    ! API call budget used up during this run — its numbers are not valid; remaining runs skipped',
                          file=sys.stderr, flush=True)
                    break
                print(f"    {p:>2} panes × {c:>2} conn {phase:>4}: view p50/p95 {row[phase]['view_p50_ms']}/{row[phase]['view_p95_ms']} ms  "
                      f"tile p95 {row[phase]['tile_p95_ms']} ms  {s.get('requests')} req  errors {s.get('error_rate_pct')}%  "
                      f"upstream {up}", file=sys.stderr, flush=True)
            if srv is not None:
                srv.should_exit = True
                reg.tiles.close()
            runs.append(row)
    return runs


# ============================================================================================ recommendations
def recommend(t: Target, direct: dict, runs: list, args) -> dict:
    rec, why = {}, {}
    sweep = [v for k, v in direct.items() if k.startswith('c') and 'tiles_per_s' in v]
    if sweep and t.kind != 'connector':
        best = max(sweep, key=lambda v: v['tiles_per_s'])
        why['tile_server'] = (f"cold tiles: {best['tiles_per_s']} tiles/s at {best['concurrency']} concurrent requests "
                              f"(p95 {best['p95_ms']} ms) — the server's own capacity for never-seen tiles")
    if sweep and t.kind == 'connector':
        clean = [v for v in sweep if v['error_pct'] <= 1.0 and not v['status'].get('rate_limited')]
        limited = [v['concurrency'] for v in sweep if v['status'].get('rate_limited')]
        if clean:
            best = max(v['tiles_per_s'] for v in clean)
            pick = min((v for v in clean if v['tiles_per_s'] >= 0.9 * best), key=lambda v: v['concurrency'])
            rec['max_concurrency'] = pick['concurrency']
            why['max_concurrency'] = (f"{pick['tiles_per_s']} tiles/s at {pick['concurrency']} concurrent calls, within 10 % of the best "
                                      f"clean rate ({best}/s){'; 429s at ' + ', '.join(map(str, limited)) if limited else ''}")
        else:
            lo = min(v['concurrency'] for v in sweep)
            rec['max_concurrency'] = max(2, lo - 2)
            why['max_concurrency'] = f'errors or 429s at every tested concurrency (lowest {lo}); test lower values (--conn 2,3,4)'
        p99 = max((v['p99_ms'] or 0) for v in sweep)
        rec['timeout_s'] = int(min(60, max(10, math.ceil(5 * p99 / 1000))))
        why['timeout_s'] = f'5 × the worst p99 call latency ({p99:.0f} ms), between 10 and 60 s'
        allv = [v for v in direct.values() if 'status' in v]
        transient = sum(n for v in allv for k, n in v['status'].items() if k not in ('ok', 'empty', 'budget'))
        transient += sum(int((r.get(ph) or {}).get('upstream', {}).get('retries', 0) or 0) + int((r.get(ph) or {}).get('upstream', {}).get('rate_limited', 0) or 0)
                         for r in runs for ph in ('cold', 'warm'))
        rec['retries'] = 3 if transient else 2
        why['retries'] = (f'{transient} transient failures (429 / 5xx / timeouts / retried calls) across all phases'
                          if transient else 'no transient failures seen')
        if clean and rec['max_concurrency'] == max(v['concurrency'] for v in sweep) and not limited:
            why['max_concurrency'] += ('; throughput was still rising at the highest tested value — if the vendor allows it, '
                                       'test higher (e.g. --conn 10,12,16)')
        seq = direct.get('sequential', {})
        if seq.get('p50_ms'):
            why['network'] = f"one tile ≈ {seq['p50_ms']} ms (p50, sequential) · {seq.get('kB_per_tile', '?')} kB · {seq.get('MB_per_s')} MB/s sequential"
    done = [r for r in runs if 'cold' in r and not r['cold'].get('budget_exhausted')]
    if done:
        by = {}
        for r in done:
            by.setdefault(r['panes'], {})[r['conn']] = r
        gains = []
        for p, d in by.items():
            if p >= 4 and len(d) > 1:
                lo, hi = min(d), max(d)
                a, b = d[lo]['cold']['view_p95_ms'], d[hi]['cold']['view_p95_ms']
                if a and b:
                    gains.append((p, lo, hi, a, b))
        if gains:
            g = [(1 - b / a) for (_, _, _, a, b) in gains]
            avg = sum(g) / len(g)
            rec['http2'] = 'recommended' if avg >= 0.15 else 'optional'
            why['http2'] = ('; '.join(f'{p} panes: view p95 {a} → {b} ms with {lo} → {hi} connections' for p, lo, hi, a, b in gains) +
                            (' → more parallel requests help: serve wissi-viz over HTTP/2 (browsers allow only 6 HTTP/1.1 connections per host)'
                             if rec['http2'] == 'recommended' else ' → little gain from more connections; HTTP/1.1 is adequate (HTTP/2 still recommended behind a proxy)'))
        ok_panes = [p for p, d in by.items() if all((r['cold']['view_p95_ms'] or 1e9) <= args.view_target_ms for r in d.values())]
        rec['comfortable_panes'] = max(ok_panes) if ok_panes else min(by)
        why['comfortable_panes'] = (f'largest pane count whose first views are sharp within {args.view_target_ms} ms (p95) at every tested connection count'
                                    if ok_panes else f'no pane count met {args.view_target_ms} ms p95 on first view')
        warm = [r['warm']['view_p95_ms'] for r in done if r.get('warm') and r['warm']['view_p95_ms']]
        cold = [r['cold']['view_p95_ms'] for r in done if r['cold']['view_p95_ms']]
        if warm and cold:
            why['cache'] = f'cold views p95 {statistics.median(cold):.0f} ms vs cached {statistics.median(warm):.0f} ms (medians over runs)'
        mb16 = max((r['cold']['MB'] or 0) for r in done if r['panes'] == max(by))
        rec['memory_mb'] = int(min(16384, max(1024, 2 ** math.ceil(math.log2(max(1.0, mb16 * args.expected_users * 2))))))
        why['memory_mb'] = (f'a {max(by)}-pane session fetched {mb16} MB; × {args.expected_users} expected concurrent users × 2 for history, '
                            'rounded to a power of two (≥ 1 GB)')
        if t.kind == 'connector':
            rec['disk_cache'] = True
            why['disk_cache'] = 'every cache miss is an API round trip; a persistent cache_dir keeps tiles across restarts'
            ups = [r['cold']['upstream_per_tile'] for r in done if r['cold'].get('upstream_per_tile') is not None]
            oc = max((math.ceil(src.info.levels[-1].width / src.info.tile_size) * math.ceil(src.info.levels[-1].height / src.info.tile_size))
                     for src in (getattr(t, '_sources', None) or {}).values()) if getattr(t, '_sources', None) else None
            if oc and oc > 4:
                why['open_cost'] = (f'opening a slide costs ≈ {oc} API calls (its coarsest API level is {oc} tiles; wissi-viz builds the '
                                    'overview levels from them), once per slide while it stays in the cache')
            if ups and max(ups) > 1.2:
                why['upstream_per_tile'] = (f'up to {max(ups):.1f} API calls per viewer tile: wissi-viz assembles some tiles from several API '
                                            'tiles — typically overview levels below a shallow API pyramid (see "pyramid depth" in the smoke '
                                            'test), or API tiles smaller than wissi-viz tiles. Warm reruns made 0 calls.')
            elif ups:
                why['upstream_per_tile'] = (f'{max(ups):.2f} API calls per viewer tile request at most (≤ 1: each tile fetched once, '
                                            'concurrent requests coalesced); warm reruns made 0 calls' if all(
                                                (r.get('warm') or {}).get('upstream', {}).get('calls', 0) == 0 for r in done) else '')
        rec['workers'] = max(16, 2 * rec.get('max_concurrency', 8)) if t.kind == 'connector' else None
        if rec['workers']:
            why['workers'] = 'server threads mostly wait on the API: at least 2 × max_concurrency'
    return {'settings': rec, 'why': why}


def settings_py(t: Target, rec: dict) -> str:
    s = rec['settings']
    cls = {'halolink': 'HaloLinkConnection', 'pathcore': 'PathCoreConnection'}.get(connector_kind(t.conn) if t.conn else '', 'RemoteConnection')
    lines = ['# Generated by scripts/wissi_probe.py — measured settings for this integration. Review before use.',
             '# Every value is explained in report.md.', '']
    if t.kind == 'connector':
        lines += [f'from wissi_viz.connectors import {cls}', '',
                  f'class MyConnection({cls}):          # your existing connector class',
                  f"    max_concurrency = {s.get('max_concurrency', 8)}",
                  f"    retries = {s.get('retries', 2)}",
                  f"    timeout = {float(s.get('timeout_s', 30))}", '']
    lines += ['from wissi_viz.nicegui import configure      # NiceGUI; or TileService(...) / SlideRegistry(...) in your own app', '',
              'registry = configure(',
              "    cache_dir='/cache/wissi-viz',                # persistent tile cache (a Docker volume)" if s.get('disk_cache', True) else '    cache_dir=None,',
              f"    memory_mb={s.get('memory_mb', 1024)},",
              *([f"    workers={s['workers']},"] if s.get('workers') else []),
              ')', '']
    if 'http2' in s:
        lines += [f"# HTTP/2 in front of wissi-viz: {'recommended — measurable gain with many panes' if s['http2'] == 'recommended' else 'optional at this latency'}"]
    if 'comfortable_panes' in s:
        lines += [f"# panes per user that open within the view-latency target: up to {s['comfortable_panes']}"]
    return '\n'.join(lines) + '\n'


# ============================================================================================ report
def report_md(t: Target, args, budget: Budget, smoke: list, direct: dict, runs: list, rec: dict, vendor: dict | None) -> str:
    L = [f'# wissi-viz integration probe: {t.label}', '',
         f"_{time.strftime('%Y-%m-%d %H:%M')} · images: {', '.join(map(str, t.images))[:300]} · panes {args.panes} · "
         f'connections {args.conn} · users {args.users}_', '']
    if t.kind != 'local':
        L += [f'**API calls: {budget.used} of {budget.limit} allowed** ({", ".join(f"{k} {v}" for k, v in sorted(budget.by_kind.items()))})'
              + (f'; {budget.refused} refused by the budget guard' if budget.refused else '')
              + (f"; the vendor's own counter: {vendor['requests']} requests" if vendor else ''), '']
    L += ['## Recommended settings', '', '| setting | value | why |', '|---|---|---|']
    for k, v in rec['settings'].items():
        if v is not None:
            L.append(f"| `{k}` | **{v}** | {rec['why'].get(k, '')} |")
    for k in ('network', 'tile_server', 'cache', 'open_cost', 'upstream_per_tile'):
        if rec['why'].get(k):
            L.append(f"| {k.replace('_', ' ')} | | {rec['why'][k]} |")
    L += ['', 'Ready-to-paste: `recommended_settings.py`.', '', '## 1. Smoke test', '', '| | check | detail | ms |', '|---|---|---|---|']
    for r in smoke:
        icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'INFO': 'ℹ️'}[r['status']]
        L.append(f"| {icon} | {r['check']} | {r['detail']} | {'' if r['ms'] is None else r['ms']} |")
    if direct:
        L += ['', '## 2. ' + ('API, called directly' if t.kind == 'connector' else 'Tile server, cold tiles' if t.kind == 'local' else 'Server, distinct tiles'),
              '', 'Distinct full-resolution tiles, no retries, so the service\'s own behaviour shows.', '',
              '| run | concurrency | calls | tiles/s | MB/s | p50 ms | p95 ms | p99 ms | statuses |', '|---|---|---|---|---|---|---|---|---|']
        for k, v in direct.items():
            if 'skipped' in v:
                L.append(f'| {k} | | | | | | | | skipped: {v["skipped"]} |')
                continue
            L.append(f"| {k} | {v['concurrency']} | {v['calls']} | {v['tiles_per_s']} | {v['MB_per_s']} | {v['p50_ms']} | {v['p95_ms']} | "
                     f"{v['p99_ms']} | {', '.join(f'{a} {b}' for a, b in v['status'].items())} |")
    if runs:
        L += ['', '## 3. Viewer sessions end to end', '',
              'Each pane shows a different image and moves on its own (open, then pan / zoom / jump). *Cold*: fresh cache, '
              'everything from the source. *Warm*: the same session again, served from the wissi-viz cache. '
              '*View*: time until every tile of a move has arrived.', '',
              '| panes | conn | cold view p50 / p95 ms | cold tile p95 ms | warm view p95 ms | requests | MB | errors % | API calls | API calls per tile |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for r in runs:
            if 'skipped' in r:
                L.append(f"| {r['panes']} | {r['conn']} | skipped: {r['skipped']} | | | | | | | |")
                continue
            c, w = r['cold'], r.get('warm') or {}
            if c.get('budget_exhausted'):
                L.append(f"| {r['panes']} | {r['conn']} | invalid: the API call budget ran out during this run | | | | | | {c['upstream']['calls']} | |")
                continue
            L.append(f"| {r['panes']} | {r['conn']} | {c['view_p50_ms']} / **{c['view_p95_ms']}** | {c['tile_p95_ms']} | {w.get('view_p95_ms', '—')} | "
                     f"{c['requests']} | {c['MB']} | {c['error_pct']} | {c['upstream']['calls'] if t.kind != 'local' else '—'}"
                     f"{' (+' + str(w['upstream']['calls']) + ' warm)' if w and t.kind == 'connector' else ''} | {c['upstream_per_tile'] if c['upstream_per_tile'] is not None else '—'} |")
    L += ['', '## Notes', '',
          '- The load generator opens up to --conn connections per user over HTTP/1.1, like a browser does behind an HTTP/2 proxy '
          '(a browser on plain HTTP/1.1 is limited to 6 per host).',
          '- Request caps per run are small by design (the API-call budget); view latencies from few moves are indicative, not precise.',
          '- Registry warm-up (background pre-fetch of coarse levels) was off, so every API call is user-driven and counted.']
    if t.kind == 'local':
        L.append('- Local target: no API involved; this measures wissi-viz itself (file decoding, encoding, caching).')
    return '\n'.join(L) + '\n'


# ============================================================================================ main
def main(argv=None):
    from wissi_viz._compat import utf8_stdio
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target')
    ap.add_argument('--images', default=None, help='connector targets: image ids (comma-separated; 2 is plenty)')
    ap.add_argument('--slides', default=None, help='deployed wissi-viz: slide ids')
    ap.add_argument('--connector-arg', action='append', default=[], help='key=value passed to your connector factory (repeatable)')
    ap.add_argument('--panes', default='1,4,9,16', help='panes (= different images) per viewer, per run')
    ap.add_argument('--conn', default='5,8,10', help='concurrent connections / calls to test')
    ap.add_argument('--users', type=int, default=1, help='simultaneous viewers per run (server load)')
    ap.add_argument('--max-calls', type=int, default=1000, help='hard cap on calls to the API (connector / deployed targets)')
    ap.add_argument('--requests', type=int, default=300, help='local target: max requests per viewer per run')
    ap.add_argument('--seconds', type=float, default=40, help='max seconds per viewer run')
    ap.add_argument('--seq-calls', type=int, default=8, help='sequential calls for the latency baseline')
    ap.add_argument('--skip', default='', help='phases to skip: smoke,api,e2e')
    ap.add_argument('--no-annotations', action='store_true', help='smoke: do not fetch annotations / objects')
    ap.add_argument('--max-images', type=int, default=4, help='deployed target: slides to smoke-test')
    ap.add_argument('--view-target-ms', type=int, default=1500, help='"comfortable" = first views sharp within this (p95)')
    ap.add_argument('--expected-users', type=int, default=10, help='for the memory-cache recommendation')
    ap.add_argument('--memory-mb', type=int, default=1024, help='wissi-viz tile memory cache during the runs')
    ap.add_argument('--workers', type=int, default=None, help='wissi-viz tile threads during the runs (default: its own)')
    ap.add_argument('--local-size', default='24000x18000', help='generated slides: WIDTHxHEIGHT')
    ap.add_argument('--workdir', default=str(Path.home() / '.cache' / 'wissi-probe'), help='where generated slides are kept')
    ap.add_argument('--mock-latency', default='40,120', help='mock target: min,max ms per request')
    ap.add_argument('--mock-limit', type=int, default=12, help='mock target: concurrent requests before 429')
    ap.add_argument('--mock-errors', type=float, default=0.01, help='mock target: fraction of 503s')
    ap.add_argument('--header', action='append', default=[], help="deployed target: 'Name: value' (repeatable)")
    ap.add_argument('--timeout', type=float, default=30)
    ap.add_argument('--insecure', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--yes', action='store_true', help='run against a non-local service (otherwise: print the plan only)')
    args = ap.parse_args(argv)
    args.panes = sorted({max(1, min(16, int(x))) for x in args.panes.split(',')})
    args.conn = sorted({int(x) for x in args.conn.split(',')})
    skip = set(filter(None, args.skip.split(',')))
    headers = {k.strip(): v.strip() for k, _, v in (h.partition(':') for h in args.header)}

    t = make_target(args)
    budget = Budget(args.max_calls if t.kind != 'local' else 10**12)
    out_dir = Path(args.out or f"wissi-probe-{t.kind}")
    n_img = len(t.images) if t.images else args.max_images
    plan_smoke = 0 if 'smoke' in skip else (SMOKE_CALLS_PER_IMAGE - (2 if args.no_annotations else 0)) * n_img
    plan_api = 0 if 'api' in skip else args.seq_calls + sum(max(20, 3 * c) for c in args.conn)
    plan_e2e = 0 if 'e2e' in skip else max(0, args.max_calls - plan_smoke - plan_api)
    print(f'target: {t.label}', file=sys.stderr)
    if t.kind != 'local':
        caps = e2e_caps(args.panes, args.conn, args.users, plan_e2e)
        print(f'plan (API calls, hard cap {args.max_calls}): smoke ≤ {plan_smoke} · direct sweep ≤ {plan_api} '
              f'({args.seq_calls} sequential + {", ".join(f"{max(20, 3 * c)} at {c}" for c in args.conn)}) · '
              f'viewer runs ≤ {plan_e2e} ({len(args.panes)} pane counts × {len(args.conn)} connection counts; '
              f'requests per viewer: {", ".join(f"{p} panes → {caps[p]}" for p in args.panes)}); warm reruns: 0', file=sys.stderr)
        if min(caps.values()) < 12 and 'e2e' not in skip:
            print('  ! too little budget per viewer run: raise --max-calls or test fewer --panes / --conn', file=sys.stderr)
    if not args.yes and not is_local_host(t):
        print('\nnothing sent yet. Re-run with --yes to run this plan (get permission to load-test the service first).', file=sys.stderr)
        for f in t.cleanup:
            f()
        return None
    try:
        if t.conn is not None:
            guard(t.conn, budget)
        t0 = time.perf_counter()
        print('\n1. smoke test', file=sys.stderr, flush=True)
        smoke, sources, ids = [], {}, []
        if t.kind == 'connector':
            smoke, sources = smoke_connector(t, budget, args) if 'smoke' not in skip else ([], {img: t.conn.source(img) for img in t.images})
        elif t.kind == 'local':
            smoke, sources = smoke_local(t)
        else:
            smoke, ids = smoke_wissi(t, args, headers)
            t.images = ids
        for r in smoke:
            print(f"   {r['status']:<4} {r['check']}: {r['detail']}" + (f" ({r['ms']} ms)" if r['ms'] is not None else ''), file=sys.stderr)
        direct = {}
        if 'api' not in skip:
            print('\n2. ' + ('direct API calls' if t.kind == 'connector' else 'cold tiles'), file=sys.stderr, flush=True)
            direct = direct_phase_wissi(t, ids, args, headers, budget) if t.kind == 'wissi' else direct_phase(t, sources, budget, args)
        runs = []
        t._sources = sources
        if 'e2e' not in skip:
            print('\n3. viewer sessions', file=sys.stderr, flush=True)
            runs = e2e_phase(t, sources, budget, args, headers)
        rec = recommend(t, direct, runs, args)
        vendor = t.vendor_stats() if t.vendor_stats else None
        out_dir.mkdir(parents=True, exist_ok=True)
        md = report_md(t, args, budget, smoke, direct, runs, rec, vendor)
        (out_dir / 'report.md').write_text(md, encoding='utf-8')
        (out_dir / 'recommended_settings.py').write_text(settings_py(t, rec), encoding='utf-8')
        (out_dir / 'results.json').write_text(json.dumps({
            'target': t.label, 'kind': t.kind, 'images': list(map(str, t.images)), 'settings': {k: v for k, v in vars(args).items() if k != 'header'},
            'budget': {'limit': budget.limit if t.kind != 'local' else None, 'used': budget.used, 'by_kind': budget.by_kind, 'refused': budget.refused},
            'vendor_stats': vendor, 'smoke': smoke, 'direct': direct, 'runs': runs, 'recommendation': rec,
            'seconds': round(time.perf_counter() - t0, 1)}, indent=1, default=str), encoding='utf-8')
        print('\n' + md, flush=True)
        print(f'wrote {out_dir}/report.md, results.json, recommended_settings.py ({time.perf_counter() - t0:.0f} s)', file=sys.stderr)
        return {'budget': budget, 'smoke': smoke, 'direct': direct, 'runs': runs, 'rec': rec, 'vendor': vendor, 'out': out_dir}
    finally:
        for f in t.cleanup:
            f()


if __name__ == '__main__':
    main()
