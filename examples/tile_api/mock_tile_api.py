"""A simulated vendor tile API (think PathCore / HALO Link / OMERO), as a separate service.

It knows nothing about wissi-viz. It behaves like a real image server, awkward parts included:

* Bearer-token auth (401 without it).
* Several pyramid styles, all common in practice:
    dzi       DeepZoom: 254 px tiles with 1 px overlap, level ids = DZI level numbers, JPEG,
              404 for empty background tiles
    tiles     512 px tiles, levels listed coarsest-first with opaque string ids, JPEG, edge tiles padded to 512
    channels  fluorescence as raw 16-bit PNG tiles, one request per channel
    render    fluorescence rendered server-side to an RGB JPEG (OMERO-style "render_image_region")
* Network latency (uniform random, default 40–120 ms per request).
* A concurrency limit → HTTP 429 + Retry-After when exceeded.
* Random transient 503s (default 1 %).
* Annotations (GeoJSON in µm) and cell tables (columnar JSON, µm) for some images.

    python examples/tile_api/mock_tile_api.py --port 9000 --token secret [--latency 40,120] [--limit 12]

Endpoints
    GET /api/images                                   list
    GET /api/images/{id}                              metadata (pyramid, mpp, version)
    GET /api/images/{id}/tiles/{level}/{col}/{row}    tile (?channel=k for 'channels')
    GET /api/images/{id}/annotations                  GeoJSON, µm
    GET /api/images/{id}/objects                      {"columns": {...}} cell table, µm
    GET /api/_stats                                   request counters (no auth; for the demo)
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import random
import sys
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from wissi_viz.sources import open_slide  # noqa: E402  (the *vendor* reads its own files however it likes)

STYLES = ('dzi', 'tiles', 'channels', 'render')


class Backing:
    """Reads any (downsample, region) of a slide file, box-filtering from the best native level."""

    def __init__(self, path: str):
        self.path = path
        self.src = open_slide(path)
        self.info = self.src.info
        self.lock = threading.Lock()

    def region(self, ds: float, x: int, y: int, w: int, h: int, channel=None) -> np.ndarray:
        lv_i = max((i for i, lv in enumerate(self.info.levels) if lv.downsample <= ds + 1e-6),
                   key=lambda i: self.info.levels[i].downsample)
        nl = self.info.levels[lv_i]
        f = max(1, int(round(ds / nl.downsample)))
        X, Y, W, H = x * f, y * f, w * f, h * f
        rgb = self.info.mode == 'rgb'
        out = np.full((H, W, 3), 255, np.uint8) if rgb else np.zeros((H, W), np.uint16)
        x1, y1 = min(X + W, nl.width), min(Y + H, nl.height)
        if x1 > X and y1 > Y:
            a = self.src.read_region(lv_i, X, Y, x1 - X, y1 - Y, channel)
            out[:y1 - Y, :x1 - X] = a
        if f > 1:
            sh = (h, f, w, f) + out.shape[2:]
            out = out[:h * f, :w * f].reshape(sh).mean((1, 3)).astype(out.dtype)
        return out


class Image:
    def __init__(self, image_id: str, backing: Backing, style: str, name: str, extras: dict):
        self.id, self.b, self.style, self.name, self.extras = image_id, backing, style, name, extras
        i = backing.info
        self.W, self.H = i.width, i.height
        self.mpp = (i.mpp or (0.5, 0.5))[0]
        self.version = f'{int(os.path.getmtime(backing.path))}-{style}'
        if style == 'dzi':
            self.ts, self.ov = 254, 1
            self.max_level = math.ceil(math.log2(max(self.W, self.H)))
        else:
            self.ts, self.ov = 512, 0
            self.levels = []  # finest-first internally: (id, w, h, ds)
            ds = 1
            k = 0
            while True:
                w, h = math.ceil(self.W / ds), math.ceil(self.H / ds)
                self.levels.append((f'r{k}-{random.Random(image_id).randint(100, 999)}', w, h, ds))
                if max(w, h) <= self.ts:
                    break
                ds *= 4 if style == 'tiles' else 2
                k += 1

    def meta(self) -> dict:
        d = {'id': self.id, 'name': self.name, 'width': self.W, 'height': self.H, 'mpp': self.mpp,
             'modified': self.version, 'has_annotations': 'annotations' in self.extras,
             'has_objects': 'objects' in self.extras}
        if self.style == 'dzi':
            d.update(format='dzi', tile_size=self.ts, overlap=self.ov, tile_format='jpeg')
        else:
            d.update(format='tiles', tile_size=self.ts, tile_format='png16' if self.style == 'channels' else 'jpeg',
                     levels=[{'id': lid, 'width': w, 'height': h} for lid, w, h, _ in reversed(self.levels)])
        if self.style == 'channels':
            d['channels'] = [{'name': c.name, 'color': c.color} for c in self.b.info.channels]
            d['dtype'] = 'uint16'
        return d

    def tile(self, level: str, col: int, row: int, channel: int | None) -> tuple[bytes, str] | None:
        from PIL import Image as PImage
        if self.style == 'dzi':
            L = int(level)
            if not 0 <= L <= self.max_level:
                raise KeyError(level)
            ds = 2 ** (self.max_level - L)
            lw, lh = math.ceil(self.W / ds), math.ceil(self.H / ds)
            x0, y0 = col * self.ts - (self.ov if col else 0), row * self.ts - (self.ov if row else 0)
            x1, y1 = min(lw, (col + 1) * self.ts + self.ov), min(lh, (row + 1) * self.ts + self.ov)
            if x0 >= lw or y0 >= lh or col < 0 or row < 0:
                raise KeyError('tile out of range')
            a = self.b.region(ds, x0, y0, x1 - x0, y1 - y0)
            if a.min() > 238 and a.std() < 2.5:
                return None  # empty background → 404, like many tile servers
            buf = io.BytesIO()
            PImage.fromarray(a).save(buf, 'JPEG', quality=85)
            return buf.getvalue(), 'image/jpeg'
        match = [lv for lv in self.levels if lv[0] == level]
        if not match:
            raise KeyError(level)
        _, lw, lh, ds = match[0]
        if col * self.ts >= lw or row * self.ts >= lh or col < 0 or row < 0:
            raise KeyError('tile out of range')
        if self.style == 'channels':
            a = self.b.region(ds, col * self.ts, row * self.ts, self.ts, self.ts, channel or 0)  # padded to 512
            buf = io.BytesIO()
            PImage.fromarray(a.astype(np.uint16)).save(buf, 'PNG')
            return buf.getvalue(), 'image/png'
        if self.style == 'render':  # fluorescence composite → RGB, server-side
            info = self.b.info
            comp = np.zeros((self.ts, self.ts, 3), np.float32)
            for c, ch in enumerate(info.channels[:3]):
                v = self.b.region(ds, col * self.ts, row * self.ts, self.ts, self.ts, c).astype(np.float32)
                lo, hi = self.extras['windows'][c]
                t = np.clip((v - lo) / max(hi - lo, 1), 0, 1)
                col_rgb = np.array([int(ch.color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
                comp += t[..., None] * col_rgb
            a = np.clip(comp, 0, 255).astype(np.uint8)
        else:
            a = self.b.region(ds, col * self.ts, row * self.ts, self.ts, self.ts)
        buf = io.BytesIO()
        PImage.fromarray(a).save(buf, 'JPEG', quality=85)
        return buf.getvalue(), 'image/jpeg'


def build_catalog(files: list[str], n: int = 16, annotations: dict | None = None) -> dict[str, Image]:
    backings = {f: Backing(f) for f in files}
    cat = {}
    seen: dict[str, int] = {}
    for k in range(n):
        f = files[k % len(files)]
        b = backings[f]
        j = seen[f] = seen.get(f, -1) + 1  # alternate styles per source file
        if b.info.mode == 'rgb':
            style = 'dzi' if j % 2 == 0 else 'tiles'
        else:
            style = 'channels' if j % 2 == 0 else 'render'
        extras = dict((annotations or {}).get(f, {}))
        if style == 'render':
            extras['windows'] = b.src.channel_stats()
        iid = f'img-{k + 1:02d}'
        cat[iid] = Image(iid, b, style, f'{Path(f).stem} ({style})', extras)
    return cat


def create_api(catalog: dict[str, Image], token: str, latency=(0.04, 0.12), limit: int = 12,
               error_rate: float = 0.01, seed: int = 0) -> FastAPI:
    api = FastAPI(title='Mock tile API')
    rng = random.Random(seed)
    state = {'inflight': 0, 'requests': 0, 'tiles': 0, 'r429': 0, 'r503': 0, 'r404': 0, 'bytes': 0, 'meta': 0}

    def auth(authorization: str | None):
        if authorization != f'Bearer {token}':
            raise HTTPException(401, 'invalid token')

    def image(iid: str) -> Image:
        if iid not in catalog:
            raise HTTPException(404, 'no such image')
        return catalog[iid]

    async def network():
        lo, hi = latency
        if hi > 0:
            await asyncio.sleep(rng.uniform(lo, hi))

    @api.middleware('http')
    async def limiter(request: Request, call_next):
        if not request.url.path.startswith('/api/images'):
            return await call_next(request)
        state['requests'] += 1
        if state['inflight'] >= limit:
            state['r429'] += 1
            return JSONResponse({'detail': 'too many requests'}, status_code=429, headers={'Retry-After': '0.2'})
        if error_rate and rng.random() < error_rate:
            state['r503'] += 1
            return JSONResponse({'detail': 'upstream hiccup'}, status_code=503)
        state['inflight'] += 1
        try:
            await network()
            return await call_next(request)
        finally:
            state['inflight'] -= 1

    @api.get('/api/images')
    def list_images(authorization: str | None = Header(None)):
        auth(authorization)
        return [{'id': i.id, 'name': i.name, 'style': i.style} for i in catalog.values()]

    @api.get('/api/images/{iid}')
    def meta(iid: str, authorization: str | None = Header(None)):
        auth(authorization)
        state['meta'] += 1
        return image(iid).meta()

    @api.get('/api/images/{iid}/tiles/{level}/{col}/{row}')
    async def tile(iid: str, level: str, col: int, row: int, channel: int | None = None,
                   authorization: str | None = Header(None)):
        auth(authorization)
        img = image(iid)
        try:
            out = await run_in_threadpool(img.tile, level, col, row, channel)
        except KeyError as e:
            raise HTTPException(400, f'bad tile request: {e}') from None
        if out is None:
            state['r404'] += 1
            raise HTTPException(404, 'no tile here')
        state['tiles'] += 1
        state['bytes'] += len(out[0])
        return Response(out[0], media_type=out[1])

    @api.get('/api/images/{iid}/annotations')
    def annotations(iid: str, authorization: str | None = Header(None)):
        auth(authorization)
        img = image(iid)
        return img.extras.get('annotations', {'type': 'FeatureCollection', 'features': []})

    @api.get('/api/images/{iid}/objects')
    def objects(iid: str, authorization: str | None = Header(None)):
        auth(authorization)
        img = image(iid)
        if 'objects' not in img.extras:
            raise HTTPException(404, 'no objects')
        return {'units': 'um', 'columns': img.extras['objects']}

    @api.get('/api/_stats')
    def stats():
        return dict(state)

    return api


def demo_annotations(slide: str, cells_tsv: str | None, rois_geojson: str | None, mpp: float) -> dict:
    """Package ROIs (px → µm) and a cell table (µm) the way an analysis platform would expose them."""
    out = {}
    if rois_geojson and Path(rois_geojson).exists():
        gj = json.loads(Path(rois_geojson).read_text(encoding='utf-8'))
        for f in gj['features']:
            g = f['geometry']
            g['coordinates'] = [[[x * mpp, y * mpp] for x, y in ring] for ring in g['coordinates']]
        out['annotations'] = gj
    if cells_tsv and Path(cells_tsv).exists():
        import pandas as pd
        df = pd.read_csv(cells_tsv, sep='\t')
        keep = ['Class', 'Centroid X µm', 'Centroid Y µm', 'Cell: CD3 mean', 'Cell: CD8 mean', 'Nucleus: Ki67 mean']
        out['objects'] = {k: df[k].tolist() for k in keep if k in df}
    return {slide: out} if out else {}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='slide files the vendor "stores"')
    ap.add_argument('--port', type=int, default=9000)
    ap.add_argument('--token', default='secret')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--latency', default='40,120', help='min,max ms per request')
    ap.add_argument('--limit', type=int, default=12, help='max concurrent requests before 429')
    ap.add_argument('--errors', type=float, default=0.01, help='fraction of requests answered 503')
    ap.add_argument('--cells', default=None, help='QuPath TSV for the first slide (exposed as /objects)')
    ap.add_argument('--rois', default=None, help='ROI GeoJSON (px) for the first slide (exposed as /annotations)')
    a = ap.parse_args()
    import uvicorn
    first = a.files[0]
    mpp = (open_slide(first).info.mpp or (0.5,))[0]
    cat = build_catalog(a.files, a.n, demo_annotations(first, a.cells, a.rois, mpp))
    lo, hi = (float(v) / 1000 for v in a.latency.split(','))
    print(f'mock tile API: {len(cat)} images on http://127.0.0.1:{a.port}/api/images (token {a.token!r})')
    uvicorn.run(create_api(cat, a.token, (lo, hi), a.limit, a.errors), host='127.0.0.1', port=a.port, log_level='warning')
