#!/usr/bin/env python3
"""wissi_loadtest — multi-user load test for a wissi-viz tile server (or any XYZ tile API).

Each virtual user behaves like a person at a viewer:
- opens 1–16 slides (``--panes``);
- loads each slide's metadata and the coarse overview tiles;
- then pans, zooms and jumps with linked panes.

After every move the user fetches the tiles that became visible: at most ``--conn`` at once, like a
browser, and never re-fetching a tile it already has, like a browser cache. It then pauses for a think time.

Users are added in stages (``--users 10,25,50,100``). Each user stops after ``--requests`` requests;
this is hard-capped at 999 so a test stays small. For each stage the report gives:
- throughput;
- per-request latency (p50/p95/p99);
- **view latency**: how long a user waits for a move to become sharp, which is what people feel;
- every HTTP status (404, 429, 500, 502, 503, 504, …), timeouts and connection errors.

It then fits the Universal Scalability Law (Gunther) to throughput against users, and Little's law to
latency. From these it estimates capacity, the user count where adding users stops helping, and the
expected latency at user counts you did not test. Every extrapolated number is flagged.

Requirements: Python ≥ 3.9, ``httpx`` (BSD-3), ``numpy`` (BSD-3); ``h2`` (MIT) only for ``--http2``.

Examples::

    # a wissi-viz deployment (slides discovered from /wissi-viz/slides; or pass --slides id1,id2)
    python wissi_loadtest.py https://wissi.example.org/wissi-viz --users 10,25,50,100 --requests 300 \\
        --header "Cookie: session=…" --out load.json

    # a raw vendor tile API (no wissi-viz in between)
    python wissi_loadtest.py "https://api.example.org/images/{image}/tiles/{level}/{col}/{row}" \\
        --images img1,img2 --pyramid 100000x80000:256:2 --header "Authorization: Bearer …"

Load testing a server you do not operate can look like an attack. Get permission first. For any host
other than localhost the script asks for ``--yes``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import quote, urlparse

MAX_REQUESTS_PER_USER = 999
SCREEN = (1600, 1000)
GRIDS = {
    1: (1, 1),
    2: (2, 1),
    3: (3, 1),
    4: (2, 2),
    5: (3, 2),
    6: (3, 2),
    7: (4, 2),
    8: (4, 2),
    9: (3, 3),
    10: (4, 3),
    11: (4, 3),
    12: (4, 3),
    13: (4, 4),
    14: (4, 4),
    15: (4, 4),
    16: (4, 4),
}


# ----------------------------------------------------------------------------------------------- slides
@dataclass
class Slide:
    id: str
    width: int
    height: int
    tile: int
    levels: list  # [(width, height, downsample)]
    channels: int = 0  # 0 = RGB tiles, n = one request per channel
    version: str = ''
    url: str = ''  # template with {level} {col} {row} (+ {c})
    mpp: float = 0.25  # µm per level-0 px (sets the 40× zoom limit)


def _wissi_slide(base: str, sid: str, d: dict) -> Slide:
    lv = [(x['width'], x['height'], x['downsample']) for x in d['levels']]
    ch = len(d.get('channels') or []) if d.get('mode') == 'channels' else 0
    url = f'{base}/slides/{sid}/tiles/{{level}}/{{col}}/{{row}}?v={d.get("version", "")}' + ('&c={c}' if ch else '')
    mpp = (d.get('mpp') or [0.25])[0] or 0.25
    return Slide(sid, d['width'], d['height'], d['tile_size'], lv, min(ch, 3), d.get('version', ''), url, float(mpp))


def _template_slide(template: str, image: str, pyramid: str, mpp: float = 0.25) -> Slide:
    size, tile, step = pyramid.split(':')
    w, h = (int(v) for v in size.lower().split('x'))
    tile, step = int(tile), float(step)
    lv, ds = [], 1.0
    while True:
        lw, lh = max(1, math.ceil(w / ds)), max(1, math.ceil(h / ds))
        lv.append((lw, lh, ds))
        if lw <= tile and lh <= tile:
            break
        ds *= step
    url = template.replace('{image}', image).replace('{x}', '{col}').replace('{y}', '{row}')
    return Slide(image, w, h, tile, lv, 0, '', url, mpp)


# ----------------------------------------------------------------------------------------------- user model
def level_for(s: Slide, scale: float) -> int:
    """Coarsest level whose texels are ≤ 1 screen px (what the viewer requests at rest)."""
    want, L = 1 / scale, 0
    for i, (_, _, ds) in enumerate(s.levels):
        if ds <= want * 1.0001:
            L = i
    return L


def visible_tiles(s: Slide, L: int, cx: float, cy: float, scale: float, pw: int, ph: int):
    lw, lh, ds = s.levels[L]
    hw, hh = pw / 2 / scale, ph / 2 / scale
    x0, x1 = max(0.0, cx - hw), min(s.width, cx + hw)
    y0, y1 = max(0.0, cy - hh), min(s.height, cy + hh)
    if x1 <= x0 or y1 <= y0:
        return []
    t = s.tile * ds
    nx, ny = math.ceil(lw / s.tile), math.ceil(lh / s.tile)
    return [
        (L, c, r)
        for r in range(int(y0 // t), min(ny - 1, int((y1 - 1e-9) // t)) + 1)
        for c in range(int(x0 // t), min(nx - 1, int((x1 - 1e-9) // t)) + 1)
    ]


def parse_index(buf: bytes, meta: dict) -> dict:
    """A layer's overview payload: bucket table (+ which pyramid tiles exist), as the viewer reads it."""
    import struct

    import numpy as np

    jl = struct.unpack_from('<I', buf, 4)[0]
    m = json.loads(buf[8 : 8 + jl])
    off = 8 + jl

    def take(dtype, n):
        nonlocal off
        a = np.frombuffer(buf, dtype, n, off)
        off += -(-(n * np.dtype(dtype).itemsize) // 4) * 4
        return a

    n, B = m['count'], m['buckets']
    pyr = m.get('pyramid') if m.get('index_only') else None
    if not pyr:  # full overview (small layers / older servers): skip the per-feature arrays
        for dt, k in (('<f4', 2 * n), ('<f4', n), ('<u4', n), ('<u2', n), ('<u2', n), ('u1', n)):
            take(dt, k)
    buckets = take('<f4', B * 8).reshape(-1, 8)
    tiles = []
    if pyr:
        for lv in range(pyr['levels']):
            t = take('<i4', pyr['tiles'][lv] * 2).reshape(-1, 2)
            tiles.append({(int(a), int(b)) for a, b in t})
    return {
        'name': meta['name'],
        'version': meta['version'],
        'pyr': pyr,
        'tiles': tiles,
        'buckets': buckets.tolist(),
        'min_px': (meta.get('style') or {}).get('detail_min_px') or 6,
    }


@dataclass
class UserResult:
    req: list = field(
        default_factory=list
    )  # (t_start, latency_s, status, bytes, kind)  status: int | 'timeout' | 'conn'
    views: list = field(default_factory=list)  # (t_start, seconds until every tile of the move arrived)
    stopped: str = ''


async def run_user(uid: int, slides: list, args, t_end: float, seed: int, headers: dict) -> UserResult:
    import httpx

    rnd = random.Random(seed * 100003 + uid)
    res = UserResult()
    budget = min(args.requests, MAX_REQUESTS_PER_USER)
    n = max(1, min(16, args.panes))
    cols, rows = GRIDS[n]
    pw, ph = SCREEN[0] // cols, SCREEN[1] // rows
    # distinct: pane k shows slides[k] and each pane moves on its own (n different images, as in a review of n
    # cases); default: random slides with linked panes (the same relative move in every pane)
    distinct = getattr(args, 'distinct', False)
    mine = (
        [slides[(uid + k) % len(slides)] for k in range(n)]
        if distinct
        else [slides[rnd.randrange(len(slides))] for _ in range(n)]
    )
    seen: set = set()
    sem = asyncio.Semaphore(args.conn)
    limits = httpx.Limits(max_connections=args.conn, max_keepalive_connections=args.conn)
    async with httpx.AsyncClient(
        http2=args.http2,
        limits=limits,
        timeout=args.timeout,
        headers=headers,
        verify=not args.insecure,
        follow_redirects=True,
    ) as cli:

        async def get(url: str, kind: str, body: bool = False):
            if len(res.req) >= budget:
                return None
            async with sem:
                if len(res.req) >= budget:
                    return None
                t0 = time.perf_counter()
                res.req.append(None)  # reserve the slot (keeps the budget exact under concurrency)
                i = len(res.req) - 1
                try:
                    r = await cli.get(url)
                    res.req[i] = (t0, time.perf_counter() - t0, r.status_code, len(r.content), kind)
                    return r.content if body and r.status_code == 200 else None
                except httpx.TimeoutException:
                    res.req[i] = (t0, time.perf_counter() - t0, 'timeout', 0, kind)
                except httpx.HTTPError:
                    res.req[i] = (t0, time.perf_counter() - t0, 'conn', 0, kind)
                return None

        overlays: dict = {}  # slide id → [layer state] (wissi mode, --overlays)

        async def open_overlays(s: Slide):
            raw = await get(f'{args.base}/slides/{s.id}/annotations', 'ann_meta', body=True)
            if not raw:
                return
            layers = []
            for m in json.loads(raw):
                if not m.get('style', {}).get('visible', True):
                    continue
                if args.overlay_mode == 'client' and m.get('client_ok'):
                    # client mode: the whole layer once (the browser builds its own pyramid), then nothing per move
                    await get(
                        f'{args.base}/slides/{s.id}/annotations/columns?layer={quote(m["name"])}&v={m["version"]}',
                        'ann_columns',
                    )
                    continue
                b = await get(
                    f'{args.base}/slides/{s.id}/annotations/overview?layer={quote(m["name"])}&v={m["version"]}',
                    'ann_index',
                    body=True,
                )
                if b:
                    layers.append(parse_index(b, m))
            overlays[s.id] = layers

        def overlay_urls(s: Slide, v):
            out = []
            for L in overlays.get(s.id, ()):
                hw, hh = pw / 2 / v[2], ph / 2 / v[2]
                r = (v[0] - hw, v[1] - hh, v[0] + hw, v[1] + hh)
                q = f'layer={quote(L["name"])}&v={L["version"]}'
                P = L['pyr']
                if P:  # object pyramid: the viewer's level choice
                    k_need = math.ceil(math.log2(4 / v[2]))
                    lvl = 0 if k_need <= P['k0'] else min(P['levels'] - 1, k_need - P['k0'])
                    for lv in {P['levels'] - 1, lvl}:
                        span = 2 ** (P['k0'] + lv) * P['tile_bins']
                        for tx in range(max(0, int(r[0] // span)), int(r[2] // span) + 1):
                            for ty in range(max(0, int(r[1] // span)), int(r[3] // span) + 1):
                                key = (s.id, L['name'], 'pt', lv, tx, ty)
                                if (tx, ty) in L['tiles'][lv] and key not in seen:
                                    seen.add(key)
                                    out.append(
                                        (f'{args.base}/slides/{s.id}/annotations/ptile/{lv}/{tx}/{ty}?{q}', 'ann_ptile')
                                    )
                B = L['buckets']
                for i in range(len(B)):
                    x0, y0, x1, y1, _, _, med, _ = B[i]
                    if x0 > r[2] or x1 < r[0] or y0 > r[3] or y1 < r[1] or med * v[2] < L['min_px']:
                        continue
                    key = (s.id, L['name'], 'b', i)
                    if key not in seen:
                        seen.add(key)
                        out.append((f'{args.base}/slides/{s.id}/annotations/bucket/{i}?{q}', 'ann_bucket'))
            return out

        def urls(s: Slide, tiles):
            out = []
            for L, c, r in tiles:
                key = (s.id, L, c, r)
                if key in seen:
                    continue
                seen.add(key)
                base = s.url.replace('{level}', str(L)).replace('{col}', str(c)).replace('{row}', str(r))
                out += [base.replace('{c}', str(k)) for k in range(s.channels)] if s.channels else [base]
            return out

        async def view(moves, views=None):
            t0 = time.perf_counter()
            todo = [(u, 'tile') for (s, tiles) in moves for u in urls(s, tiles)]
            if views is not None and overlays:
                todo += [x for s, v in views for x in overlay_urls(s, v)]
            if todo:
                await asyncio.gather(*(get(u, k) for u, k in todo))
                res.views.append((t0, time.perf_counter() - t0))

        # open: metadata (wissi mode) + whole coarsest level, then the fit view
        if args.mode == 'wissi':
            await asyncio.gather(*(get(f'{args.base}/slides/{s.id}/info.json', 'info') for s in mine))
            if args.overlays:
                await asyncio.gather(*(open_overlays(s) for s in {s.id: s for s in mine}.values()))
        fit = [min(pw / s.width, ph / s.height) * 0.96 for s in mine]
        st = [[s.width / 2, s.height / 2, f] for s, f in zip(mine, fit)]
        await view(
            [(s, visible_tiles(s, len(s.levels) - 1, s.width / 2, s.height / 2, 1e-9, 10**9, 10**9)) for s in mine]
        )
        await view(
            [(s, visible_tiles(s, level_for(s, v[2]), *v, pw, ph)) for s, v in zip(mine, st)], list(zip(mine, st))
        )
        # navigate (linked panes: the same relative move in every pane)
        while len(res.req) < budget and time.perf_counter() < t_end:
            await asyncio.sleep(rnd.uniform(*args.think))
            a = rnd.random()
            for s, v, f in zip(mine, st, fit):
                if distinct:
                    a = rnd.random()
                mx = s.mpp * 40 / 10  # 40× as screen px per level-0 px
                if a < 0.55:  # pan 0.3–1 screen
                    ang, d = rnd.uniform(0, 2 * math.pi), rnd.uniform(0.3, 1.0)
                    v[0] += math.cos(ang) * d * pw / v[2]
                    v[1] += math.sin(ang) * d * ph / v[2]
                elif a < 0.75:
                    v[2] = min(mx, v[2] * 2)  # zoom in ×2
                elif a < 0.9:
                    v[2] = max(f, v[2] / 2)  # zoom out ×2
                else:  # jump: new place, 2.5–40×
                    v[0], v[1] = rnd.uniform(0.1, 0.9) * s.width, rnd.uniform(0.1, 0.9) * s.height
                    v[2] = max(f, min(mx, f * 2 ** rnd.uniform(1, math.log2(max(2.0, mx / f)))))
                v[0], v[1] = min(max(v[0], 0), s.width), min(max(v[1], 0), s.height)
            await view(
                [(s, visible_tiles(s, level_for(s, v[2]), *v, pw, ph)) for s, v in zip(mine, st)], list(zip(mine, st))
            )
        res.stopped = 'budget' if len(res.req) >= budget else 'time'
    res.req = [x for x in res.req if x is not None]
    return res


# ----------------------------------------------------------------------------------------------- stages
async def _lag_probe(stop: asyncio.Event, out: list):
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.05)
        out.append(time.perf_counter() - t - 0.05)


async def _stage_async(users: list, slides, args, seed, headers):
    t_end = time.perf_counter() + args.max_seconds
    stop, lag = asyncio.Event(), []
    probe = asyncio.create_task(_lag_probe(stop, lag))

    async def start(uid, delay):
        await asyncio.sleep(delay)
        return await run_user(uid, slides, args, t_end, seed, headers)

    out = await asyncio.gather(*(start(u, args.ramp * k / max(1, len(users))) for k, u in enumerate(users)))
    stop.set()
    await probe
    return out, lag


def _stage_proc(payload):
    users, slides, args, seed, headers = payload
    out, lag = asyncio.run(_stage_async(users, slides, args, seed, headers))
    return [(r.req, r.views, r.stopped) for r in out], lag


def _snap(args, headers) -> dict:
    import httpx

    out = {}
    for u in args.stats_url:
        try:
            out[u] = httpx.get(u, headers=headers, timeout=10, verify=not args.insecure).json()
        except Exception as e:  # noqa: BLE001
            out[u] = {'error': str(e)}
    return out


_GAUGE = ('mean', 'avg', 'inflight', 'max_bytes', 'bytes', 'entries', 'size', 'slides_opened')


def _diff(a, b, key=''):
    """Counters are differenced; gauges (averages, sizes, in-flight) are reported as the value after the stage."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {
            k: _diff(a.get(k), v, k)
            for k, v in b.items()
            if isinstance(v, (int, float, dict)) and not isinstance(v, bool)
        }
    if any(g in key for g in _GAUGE) and not key.startswith('r'):
        return b
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return round(b - a, 3)
    return b


def run_stage(n_users: int, slides, args, seed: int, headers: dict) -> dict:
    ids = list(range(n_users))
    before = _snap(args, headers) if args.stats_url else None
    t0 = time.perf_counter()
    if args.procs > 1:
        chunks = [ids[i :: args.procs] for i in range(args.procs)]
        with mp.get_context('spawn').Pool(args.procs) as pool:
            parts = pool.map(_stage_proc, [(c, slides, args, seed, headers) for c in chunks if c])
    else:
        parts = [_stage_proc((ids, slides, args, seed, headers))]
    wall = time.perf_counter() - t0
    req = [x for p in parts for u in p[0] for x in u[0]]
    views = [x for p in parts for u in p[0] for x in u[1]]
    lag = [x for p in parts for x in p[1]]
    out = summarise(n_users, req, views, lag, wall, args)
    if before is not None:  # server-side counters during this stage (e.g. upstream calls, 429/503, cache hits)
        after = _snap(args, headers)
        out['server'] = {u: _diff(before[u], after[u]) for u in after}
    return out


def _pct(v, q):
    import numpy as np

    return float(np.percentile(v, q)) if len(v) else None


def summarise(n_users, req, views, lag, wall, args) -> dict:
    import numpy as np

    if not req:
        return {'users': n_users, 'requests': 0}
    t = np.array([r[0] for r in req])
    lat = np.array([r[1] for r in req])
    status: dict = {}
    for r in req:
        status[str(r[2])] = status.get(str(r[2]), 0) + 1
    good = lambda c: isinstance(c, int) and (200 <= c < 400 or (args.ok_404 and c == 404))
    ok = np.array([good(r[2]) for r in req])
    # steady window: drop the ramp-up and the tail where users finish → throughput at full concurrency
    lo, hi = t.min() + args.ramp, np.percentile(t, 90)
    win = (t >= lo) & (t <= hi)
    span = max(1e-3, hi - lo)
    thr = float(win.sum() / span) if win.sum() > 50 else float(len(req) / max(1e-3, t.max() - t.min()))
    v = np.array([x[1] for x in views]) if views else np.array([])
    errs = {k: c for k, c in status.items() if not (k.isdigit() and good(int(k)))}
    return {
        'users': n_users,
        'requests': len(req),
        'wall_s': round(wall, 1),
        'throughput_rps': round(thr, 1),
        'MB': round(sum(r[3] for r in req) / 2**20, 1),
        'latency_ms': {
            q: round(1000 * _pct(lat[ok], p), 1) if ok.any() else None
            for q, p in (('p50', 50), ('p90', 90), ('p95', 95), ('p99', 99), ('max', 100))
        },
        'latency_mean_ms': round(1000 * float(lat[ok].mean()), 1) if ok.any() else None,
        'view_mean_ms': round(1000 * float(v.mean())) if len(v) else None,
        'view_ms': {
            q: round(1000 * _pct(v, p)) if len(v) else None for q, p in (('p50', 50), ('p95', 95), ('p99', 99))
        },
        'status': dict(sorted(status.items())),
        'errors': errs,
        'error_rate_pct': round(100 * (1 - ok.mean()), 3),
        'rate_pct': {k: round(100 * c / len(req), 3) for k, c in errs.items()},
        'client_loop_lag_ms_p95': round(1000 * _pct(lag, 95), 1) if lag else None,
        'by_kind': {
            k: {
                'requests': int(sum(1 for r in req if r[4] == k)),
                'MB': round(sum(r[3] for r in req if r[4] == k) / 2**20, 2),
                'p50_ms': round(1000 * _pct([r[1] for r in req if r[4] == k and good(r[2])], 50), 1)
                if any(r[4] == k and good(r[2]) for r in req)
                else None,
                'p95_ms': round(1000 * _pct([r[1] for r in req if r[4] == k and good(r[2])], 95), 1)
                if any(r[4] == k and good(r[2]) for r in req)
                else None,
                'errors': int(sum(1 for r in req if r[4] == k and not good(r[2]))),
            }
            for k in sorted({r[4] for r in req})
        },
    }


# ----------------------------------------------------------------------------------------------- modelling
def usl_fit(N, X):
    """Universal Scalability Law X(N) = λN / (1 + σ(N−1) + κN(N−1)), σ, κ ≥ 0: bounded least squares
    (grid over σ, κ; λ in closed form), so a fit never needs impossible negative coefficients."""
    import numpy as np

    N, X = np.asarray(N, float), np.asarray(X, float)
    if len(N) < 3:
        return None
    best = None
    sig_grid = np.concatenate([[0.0], np.geomspace(1e-5, 1.0, 120)])
    kap_grid = np.concatenate([[0.0], np.geomspace(1e-8, 1e-1, 120)])
    for sg in sig_grid:
        for kp in kap_grid:
            f = N / (1 + sg * (N - 1) + kp * N * (N - 1))
            lam_ = float((X * f).sum() / (f * f).sum())
            sse = float(((X - lam_ * f) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, lam_, sg, kp)
    _, lam, sigma, kappa = best
    pred = lam * N / (1 + sigma * (N - 1) + kappa * N * (N - 1))
    r2 = 1 - float(((X - pred) ** 2).sum() / max(1e-12, ((X - X.mean()) ** 2).sum()))
    n_star = math.sqrt((1 - sigma) / kappa) if kappa > 1e-12 else None
    x_max = (
        (lam * n_star / (1 + sigma * (n_star - 1) + kappa * n_star * (n_star - 1)))
        if n_star
        else (lam / sigma if sigma > 0 else None)
    )
    return {'lambda': lam, 'sigma': sigma, 'kappa': kappa, 'r2': r2, 'peak_users': n_star, 'max_throughput_rps': x_max}


def usl_x(m, n):
    return m['lambda'] * n / (1 + m['sigma'] * (n - 1) + m['kappa'] * n * (n - 1))


def model(stages: list, predict: list) -> dict:
    """Interpolate (inside the tested range) or extrapolate (outside, flagged) throughput, latency, errors."""
    import numpy as np

    st = [s for s in stages if s.get('requests')]
    if len(st) < 2:
        return {}
    N = np.array([s['users'] for s in st], float)
    X = np.array([s['throughput_rps'] for s in st], float)
    R = np.array([s['latency_mean_ms'] or 0 for s in st], float) / 1000
    m = usl_fit(N, X) if len(st) >= 3 else None
    good = bool(m and m['r2'] >= 0.8)
    # Closed-loop users: think T, then a burst of b tile requests that takes the view time Vm. Little's law per
    # user: X = N·b / (T + Vm)  →  N/X = T/b + Vm/b, a straight line in Vm whose slope and intercept give b and T.
    Vm = np.array([(s.get('view_mean_ms') or s['view_ms']['p50'] or 0) for s in st], float) / 1000
    b = T = None
    if len(st) >= 2 and np.ptp(Vm) > 1e-6:
        slope, icpt = np.polyfit(Vm, N / X, 1)
        if slope > 0:
            b, T = 1 / slope, max(0.0, icpt / slope)
    out = {
        'usl': m,
        'tiles_per_move': None if b is None else round(b, 1),
        'think_s': None if T is None else round(T, 3),
        'tested_users': [int(n) for n in N],
        'predictions': [],
    }
    E = np.array([s['error_rate_pct'] for s in st], float)
    V = np.array([s['view_ms']['p95'] or 0 for s in st], float)
    P95 = np.array([s['latency_ms']['p95'] or 0 for s in st], float)
    for n in predict:
        inside = N.min() <= n <= N.max()
        if inside:
            x = float(np.interp(n, N, X))
            r = float(np.interp(n, N, R))
            p95, view, err = (float(np.interp(n, N, a)) for a in (P95, V, E))
        else:
            # a poor fit (e.g. throughput still rising because shared caches warm up) is not trusted: assume no
            # further throughput gain beyond the last stage, so latency grows by Little's law (pessimistic)
            x = usl_x(m, n) if good else float(X[-1])
            if b is not None:  # view time from Little's law, never better than the last measured stage
                v = max(float(Vm[-1]), n * b / x - T) if n > N[-1] else max(0.0, n * b / x - T)
                k = v / Vm[-1] if Vm[-1] > 0 else 1.0
            else:  # no usable fit: scale with users per unit throughput
                k = max(1.0, (n / x) / (N[-1] / X[-1]))
            r = float(R[-1]) * k
            p95, view = max(0.0, P95[-1] * k), max(0.0, V[-1] * k)
            slope = np.polyfit(N, E, 1)[0] if len(N) >= 2 else 0
            err = max(0.0, float(E[-1] + slope * (n - N[-1])))
        how = (
            'interpolated'
            if inside
            else ('EXTRAPOLATED (USL)' if good else 'EXTRAPOLATED (pessimistic: no more throughput)')
        )
        out['predictions'].append(
            {
                'users': n,
                'mode': how,
                'throughput_rps': round(x, 1),
                'latency_mean_ms': round(1000 * r, 1),
                'latency_p95_ms': round(p95, 1),
                'view_p95_ms': round(view),
                'error_rate_pct': round(err, 3),
            }
        )
    return out


# ----------------------------------------------------------------------------------------------- report
def report(stages: list, mdl: dict, args) -> str:
    L = [
        f'# Load test: {args.target}',
        '',
        f'- {args.panes} pane(s) per user; ≤ {min(args.requests, MAX_REQUESTS_PER_USER)} requests per user; '
        f'{args.conn} connections per user ({"HTTP/2" if args.http2 else "HTTP/1.1"}); think {args.think[0]}–{args.think[1]} s; '
        f'{"overlays on; " if getattr(args, "overlays", False) else ""}'
        f'ramp {args.ramp} s',
        '',
        '| users | requests | req/s | MB | latency p50 / p95 / p99 ms | view p50 / p95 ms | error % | statuses ≠ 2xx | client lag p95 ms |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for s in stages:
        if not s.get('requests'):
            L.append(f'| {s["users"]} | 0 | | | | | | | |')
            continue
        lt, vw = s['latency_ms'], s['view_ms']
        errs = ', '.join(f'{k}×{v}' for k, v in s['errors'].items()) or '—'
        L.append(
            f'| {s["users"]} | {s["requests"]} | {s["throughput_rps"]} | {s["MB"]} | {lt["p50"]} / {lt["p95"]} / {lt["p99"]} | '
            f'{vw["p50"]} / {vw["p95"]} | {s["error_rate_pct"]} | {errs} | {s["client_loop_lag_ms_p95"]} |'
        )
    if any((s.get('client_loop_lag_ms_p95') or 0) > 50 for s in stages):
        L += [
            '',
            '> **Client saturated:** event-loop lag > 50 ms, so the load generator itself was a bottleneck in some '
            'stages. Use `--procs N` or run from a larger machine.',
        ]
    big = max((st for st in stages if st.get('by_kind')), key=lambda st: st['users'], default=None)
    if big and len(big['by_kind']) > 1:
        L += [
            '',
            f'Per request type at {big["users"]} users:',
            '',
            '| type | requests | MB | p50 ms | p95 ms | errors |',
            '|---|---|---|---|---|---|',
        ]
        for k, v in big['by_kind'].items():
            L.append(f'| {k} | {v["requests"]} | {v["MB"]} | {v["p50_ms"]} | {v["p95_ms"]} | {v["errors"]} |')
    m = mdl.get('usl') if mdl else None
    if m:
        L += [
            '',
            '## Scalability model (Universal Scalability Law)',
            '',
            f'X(N) = {m["lambda"]:.2f}·N / (1 + {m["sigma"]:.4f}·(N−1) + {m["kappa"]:.6f}·N(N−1)),  fit R² = {m["r2"]:.3f}',
            '',
            f'- **σ (contention)** = {m["sigma"]:.4f}: the part of the work that serialises (locks, a single worker, '
            f'an upstream rate limit).',
            f'- **κ (coherency)** = {m["kappa"]:.6f}: crosstalk that makes throughput fall when more users are added '
            f'(cache thrash, retry storms).',
            '- **Peak**: '
            + (
                f'≈ {m["peak_users"]:.0f} users at ≈ {m["max_throughput_rps"]:.0f} req/s; beyond that '
                f'throughput *falls*.'
                if m['peak_users']
                else (
                    f'no retrograde region found; throughput levels off near {m["max_throughput_rps"]:.0f} req/s.'
                    if m['max_throughput_rps']
                    else 'scaling linearly in the tested range.'
                )
            ),
        ]
        top = max(st['users'] for st in stages if st.get('requests'))
        if m['max_throughput_rps'] and (m['peak_users'] or m['max_throughput_rps'] / max(1e-9, m['lambda'])) > 3 * top:
            L.append(
                f'- No limit is visible within the tested {top} users. The fitted ceiling lies far beyond that range, '
                f'so treat it as unknown and test higher (or with more distinct slides per user) to find it.'
            )
        if m['r2'] < 0.8:
            L.append(
                f'- The fit is poor (R² {m["r2"]:.2f}): add stages or longer runs before trusting the predictions.'
            )
    if mdl and mdl.get('predictions'):
        L += [
            '',
            '## Predictions',
            '',
            '| users | how | req/s | latency mean / p95 ms | view p95 ms | error % |',
            '|---|---|---|---|---|---|',
        ]
        for p in mdl['predictions']:
            L.append(
                f'| {p["users"]} | {p["mode"]} | {p["throughput_rps"]} | {p["latency_mean_ms"]} / {p["latency_p95_ms"]} | '
                f'{p["view_p95_ms"]} | {p["error_rate_pct"]} |'
            )
        L += [
            '',
            "Interpolated rows sit between measured stages. EXTRAPOLATED rows take throughput from the USL fit and view time from Little's law for closed-loop users (each move: think, then a burst of tiles). "
            'Treat them as an estimate, roughly reliable to about 2× the largest tested load. Error-rate extrapolation '
            'is a linear trend: real systems usually fail abruptly at saturation.',
        ]
    return '\n'.join(L)


# ----------------------------------------------------------------------------------------------- main
def discover(args, headers) -> list:
    import httpx

    with httpx.Client(timeout=args.timeout, headers=headers, verify=not args.insecure, http2=args.http2) as cli:
        if args.mode == 'template':
            return [_template_slide(args.target, im, args.pyramid, args.mpp) for im in args.images.split(',')]
        ids = (
            args.slides.split(',')
            if args.slides
            else [d['id'] for d in cli.get(f'{args.base}/slides').raise_for_status().json()]
        )
        if not ids:
            sys.exit('no slides found; pass --slides')
        out = []
        for sid in ids[: args.max_slides]:
            r = cli.get(f'{args.base}/slides/{sid}/info.json')
            if r.status_code != 200:
                print(f'  skip {sid}: info.json → HTTP {r.status_code}', file=sys.stderr)
                continue
            out.append(_wissi_slide(args.base, sid, r.json()))
        return out


def main(argv=None):
    for s in (sys.stdout, sys.stderr):  # UTF-8 output on any OS / pipe (reports use →, ≤, µ)
        try:
            if (getattr(s, 'encoding', '') or '').lower().replace('-', '') != 'utf8':
                s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        'target', help='wissi-viz base URL (…/wissi-viz) or a tile URL template with {image} {level} {col} {row}'
    )
    ap.add_argument('--users', default='10,25,50,100', help='comma-separated user counts, one stage each')
    ap.add_argument(
        '--requests', type=int, default=300, help=f'max requests per user (hard cap {MAX_REQUESTS_PER_USER})'
    )
    ap.add_argument('--panes', type=int, default=4, help='slides open per user (1–16)')
    ap.add_argument(
        '--conn',
        type=int,
        default=6,
        help='concurrent requests per user (6 ≈ browser HTTP/1.1, 24 ≈ wissi-viz over HTTP/2)',
    )
    ap.add_argument('--think', default='0.3,1.5', help='pause between moves, seconds (min,max)')
    ap.add_argument('--ramp', type=float, default=5, help='seconds over which each stage starts its users')
    ap.add_argument('--max-seconds', type=float, default=180, help='per-stage time cap')
    ap.add_argument('--timeout', type=float, default=30)
    ap.add_argument('--http2', action='store_true', help='use HTTP/2 (needs the h2 package)')
    ap.add_argument(
        '--header', action='append', default=[], help="extra header, e.g. 'Authorization: Bearer …' (repeatable)"
    )
    ap.add_argument('--slides', default=None, help='wissi mode: slide ids (default: all from /slides)')
    ap.add_argument('--max-slides', type=int, default=64)
    ap.add_argument('--images', default=None, help='template mode: image ids')
    ap.add_argument('--pyramid', default='100000x80000:256:2', help='template mode: WIDTHxHEIGHT:TILE:STEP')
    ap.add_argument('--mpp', type=float, default=0.25, help='template mode: µm per px (sets the 40× zoom limit)')
    ap.add_argument(
        '--overlays',
        action='store_true',
        help='wissi mode: also fetch annotation/object overlays like the viewer (layer index, pyramid tiles, outline buckets)',
    )
    ap.add_argument(
        '--overlay-mode',
        choices=['server', 'client'],
        default='server',
        help='--overlays: server = pyramid tiles per move (default); client = download each layer once',
    )
    ap.add_argument(
        '--ok-404', action='store_true', help='count HTTP 404 as success (APIs that answer 404 for empty tiles)'
    )
    ap.add_argument(
        '--stats-url',
        action='append',
        default=[],
        help='JSON endpoint to snapshot before/after each stage, e.g. …/wissi-viz/stats (repeatable)',
    )
    ap.add_argument('--procs', type=int, default=1, help='load-generator processes (use >1 when client lag is high)')
    ap.add_argument(
        '--predict', default='', help='extra user counts to estimate, e.g. 150,200 (default: 1.5× and 2× the largest)'
    )
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--insecure', action='store_true', help='skip TLS verification (self-signed test servers)')
    ap.add_argument('--yes', action='store_true', help='confirm load-testing a non-local host')
    ap.add_argument('--out', default=None, help='write JSON (+ .md report)')
    ap.add_argument(
        '--from-json', default=None, help='re-render the report/model from a saved --out JSON (no requests)'
    )
    args = ap.parse_args(argv)
    args.think = tuple(float(v) for v in args.think.split(','))
    if args.from_json:
        d = json.load(open(args.from_json, encoding='utf-8-sig'))
        for k, v in d['settings'].items():
            if k not in ('from_json', 'out'):
                setattr(args, k, tuple(v) if k == 'think' else v)
        stages = d['stages']
        users = [s['users'] for s in stages]
        pred = [int(x) for x in args.predict.split(',') if x] or sorted({int(users[-1] * 1.5), users[-1] * 2})
        mid = [(a + b) // 2 for a, b in zip(users, users[1:]) if b - a > 1][:2]
        mdl = model(stages, sorted(set(mid + pred)))
        md = report(stages, mdl, args)
        print(md)
        if args.out:
            json.dump({**d, 'model': mdl}, open(args.out, 'w', encoding='utf-8'), indent=1)
            open(os.path.splitext(args.out)[0] + '.md', 'w', encoding='utf-8').write(md + '\n')
        return stages, mdl
    args.requests = min(args.requests, MAX_REQUESTS_PER_USER)
    args.mode = 'template' if '{level}' in args.target else 'wissi'
    args.base = args.target.rstrip('/')
    if args.mode == 'template' and not args.images:
        sys.exit('template mode needs --images')
    users = [int(u) for u in args.users.split(',')]
    host = urlparse(args.target).hostname or ''
    if host not in ('localhost', '127.0.0.1', '::1') and not args.yes:
        sys.exit(
            f'{host}: this sends up to {sum(users) * args.requests} requests. Re-run with --yes if you have permission.'
        )
    headers = {}
    for h in args.header:
        k, _, v = h.partition(':')
        headers[k.strip()] = v.strip()
    slides = discover(args, headers)
    print(
        f'{len(slides)} slide(s); stages {users} users × ≤{args.requests} requests; {args.panes} pane(s)/user',
        file=sys.stderr,
    )
    stages = []
    for k, n in enumerate(users):
        s = run_stage(n, slides, args, args.seed + k, headers)
        stages.append(s)
        if s.get('requests'):
            print(
                f'{n:>4} users  {s["requests"]:>6} req  {s["throughput_rps"]:>7} req/s  p95 {s["latency_ms"]["p95"]} ms  '
                f'view p95 {s["view_ms"]["p95"]} ms  errors {s["error_rate_pct"]}% {s["errors"] or ""}  lag {s["client_loop_lag_ms_p95"]} ms',
                file=sys.stderr,
            )
    pred = [int(x) for x in args.predict.split(',') if x] or sorted({int(users[-1] * 1.5), users[-1] * 2})
    mid = [(a + b) // 2 for a, b in zip(users, users[1:]) if b - a > 1][:2]
    mdl = model(stages, sorted(set(mid + pred)))
    md = report(stages, mdl, args)
    print('\n' + md)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(
                {'settings': {k: v for k, v in vars(args).items() if k != 'header'}, 'stages': stages, 'model': mdl},
                f,
                indent=1,
            )
        with open(os.path.splitext(args.out)[0] + '.md', 'w', encoding='utf-8') as f:
            f.write(md + '\n')
    return stages, mdl


if __name__ == '__main__':
    main()
