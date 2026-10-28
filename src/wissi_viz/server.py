"""FastAPI router serving tiles, annotations and the viewer's JavaScript.

Mount it on any FastAPI app (NiceGUI's ``app`` is a FastAPI app)::

    from wissi_viz import SlideRegistry, create_router
    registry = SlideRegistry()
    app.include_router(create_router(registry), prefix='/wissi-viz')
"""
from __future__ import annotations

import asyncio
import gzip
import json
import mimetypes
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .annotations import AnnotationLayer
from .registry import SlideRegistry
from .sources.remote import RemoteError
from .tiles import encode_rgb

STATIC_DIR = Path(__file__).parent / 'static'
ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
IMMUTABLE = 'public, max-age=31536000, immutable'


class _GzCache:
    def __init__(self, max_bytes=256 * 2**20):
        self.d: OrderedDict[tuple, bytes] = OrderedDict()
        self.size, self.max = 0, max_bytes
        self.lock = threading.Lock()

    def get(self, key, make):
        with self.lock:
            v = self.d.get(key)
            if v is not None:
                self.d.move_to_end(key)
                return v
        v = gzip.compress(make(), compresslevel=3)
        with self.lock:
            self.d[key] = v
            self.size += len(v)
            while self.size > self.max and self.d:
                _, old = self.d.popitem(last=False)
                self.size -= len(old)
        return v

    def stats(self) -> dict:
        with self.lock:
            return {'entries': len(self.d), 'bytes': self.size, 'max_bytes': self.max}


def create_router(registry: SlideRegistry, cors: str | None = '*', allow_edit: bool = True) -> APIRouter:
    r = APIRouter()
    gz = _GzCache(int(getattr(registry, 'gzip_cache_mb', 256) * 2**20))
    tiles = registry.tiles

    def hdrs(cache: str | None = None, extra: dict | None = None) -> dict:
        h = {}
        if cors:
            h['Access-Control-Allow-Origin'] = cors
        if cache:
            h['Cache-Control'] = cache
        if extra:
            h.update(extra)
        return h

    def entry(sid: str):
        try:
            return registry.get(sid)
        except KeyError:
            raise HTTPException(404, f'unknown slide {sid}') from None
        except Exception as e:
            raise HTTPException(500, f'cannot open slide {sid}: {e}') from e

    @r.get('/static/{path:path}')
    def static(path: str):
        p = (STATIC_DIR / path).resolve()
        if not p.is_relative_to(STATIC_DIR.resolve()) or not p.is_file():
            raise HTTPException(404)
        mt = 'text/javascript' if p.suffix in ('.js', '.mjs') else (mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
        return FileResponse(p, media_type=mt, headers=hdrs('no-cache'))

    @r.get('/slides')
    def slides():
        return JSONResponse([{'id': i, 'name': registry._entries[i].name} for i in registry.ids()], headers=hdrs('no-cache'))

    @r.get('/stats')
    def stats():
        mem = registry.memory_stats()
        mem['gzip_cache_mb'] = round(gz.stats()['bytes'] / 2**20, 1)
        return JSONResponse({**tiles.stats(), 'memory': mem}, headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/info.json')
    async def info(sid: str):
        entry(sid)
        try:
            d = await asyncio.to_thread(registry.info, sid)
        except Exception as e:
            raise HTTPException(500, str(e)) from e
        return JSONResponse(d, headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/tiles/{level}/{tx}/{ty}')
    async def tile(sid: str, level: int, tx: int, ty: int, c: int | None = None, v: str | None = None):
        e, pyr = entry(sid)
        info = pyr.info
        if not (0 <= level < len(info.levels)):
            raise HTTPException(404)
        lv, ts = info.levels[level], info.tile_size
        if tx < 0 or ty < 0 or tx * ts >= lv.width or ty * ts >= lv.height:
            raise HTTPException(404)
        if info.mode == 'rgb':
            c = None
        elif c is None or not (0 <= c < len(info.channels)):
            raise HTTPException(400, 'channel required')
        version = e.source.version
        try:
            data, mt = await asyncio.wrap_future(tiles.get(sid, pyr, version, level, tx, ty, c))
        except RemoteError as ex:  # upstream API failed after retries; not cached, browser retries later
            raise HTTPException(502, f'remote tile error: {ex}') from ex
        except Exception as ex:
            raise HTTPException(500, f'tile error: {ex}') from ex
        cache = IMMUTABLE if v == version else 'public, max-age=60'
        return Response(data, media_type=mt, headers=hdrs(cache))

    @r.get('/slides/{sid}/thumbnail')
    async def thumbnail(sid: str, size: int = 512):
        e, pyr = entry(sid)
        size = max(16, min(size, 2048))

        def make():
            src = e.source
            if src.info.mode == 'rgb':
                a = src.thumbnail(size)
            else:
                import numpy as np
                comp = None
                wins = e.windows or src.channel_stats()
                for ci, ch in enumerate(src.info.channels):
                    t = src.thumbnail(size, ci).astype(np.float32)
                    lo, hi = wins[ci]
                    t = np.clip((t - lo) / max(hi - lo, 1), 0, 1)
                    col = np.array([int(ch.color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
                    comp = t[..., None] * col if comp is None else comp + t[..., None] * col
                a = np.clip(comp, 0, 255).astype(np.uint8)
            from PIL import Image
            im = Image.fromarray(a)
            im.thumbnail((size, size))
            return encode_rgb(__import__('numpy').asarray(im), 'jpeg', 85)[0]

        data = await asyncio.to_thread(make)
        return Response(data, media_type='image/jpeg', headers=hdrs('public, max-age=3600'))

    # ---------------------------------------------------------------- annotations
    def layer(sid: str, name: str) -> AnnotationLayer:
        entry(sid)
        try:
            return registry.layer(sid, name)
        except KeyError:
            raise HTTPException(404, f'unknown layer {name}') from None

    def gz_response(data: bytes, v: str | None, version: int):
        cache = IMMUTABLE if v == str(version) else 'no-cache'
        return Response(data, media_type='application/octet-stream',
                        headers=hdrs(cache, {'Content-Encoding': 'gzip'}))

    @r.get('/slides/{sid}/annotations')
    async def ann_list(sid: str, wait: bool = True):
        """Layer metadata. With ``wait=false`` never blocks on sidecar parsing; the
        ``X-Pending`` header tells the client to ask again."""
        e, _ = entry(sid)
        loaded = registry._load_file_layers(e, block=False)
        if wait and not loaded:
            await asyncio.wrap_future(e.file_layers_future)
            loaded = True
        metas = await asyncio.to_thread(lambda: [L.meta() for L in list(e.layers.values())])
        return JSONResponse(metas, headers=hdrs('no-cache', {'X-Pending': '0' if loaded else '1',
                                                             'Access-Control-Expose-Headers': 'X-Pending'}))

    @r.get('/slides/{sid}/annotations/overview')
    async def ann_overview(sid: str, layer_name: str = Query(alias='layer'), v: str | None = None):
        L = layer(sid, layer_name)
        c = await asyncio.to_thread(L.compiled)
        ver = L.version
        data = await asyncio.to_thread(gz.get, (sid, layer_name, ver, 'ov'), L.overview_bytes)
        return gz_response(data, v, ver)

    @r.get('/slides/{sid}/annotations/bucket/{i}')
    async def ann_bucket(sid: str, i: int, layer_name: str = Query(alias='layer'), v: str | None = None):
        L = layer(sid, layer_name)
        c = await asyncio.to_thread(L.compiled)
        if not (0 <= i < len(c.buckets)):
            raise HTTPException(404)
        ver = L.version
        data = await asyncio.to_thread(gz.get, (sid, layer_name, ver, i), lambda: c.bucket(i))
        return gz_response(data, v, ver)

    @r.get('/slides/{sid}/annotations/ptile/{level}/{tx}/{ty}')
    async def ann_ptile(sid: str, level: int, tx: int, ty: int, layer_name: str = Query(alias='layer'), v: str | None = None):
        """Object-pyramid tile: aggregated dots (level ≥ 1) or the raw features (level 0) of one tile."""
        L = layer(sid, layer_name)
        c = await asyncio.to_thread(L.compiled)
        if c.pyr_meta is None or not (0 <= level < c.pyr_meta['levels']):
            raise HTTPException(404)
        ver = L.version
        data = await asyncio.to_thread(gz.get, (sid, layer_name, ver, 'pt', level, tx, ty), lambda: c.pyramid_tile(level, tx, ty))
        return gz_response(data, v, ver)

    @r.get('/slides/{sid}/annotations/columns')
    async def ann_columns(sid: str, layer_name: str = Query(alias='layer'), v: str | None = None):
        """Client-side overlay mode: the whole layer as compact columns (bulk object layers only)."""
        L = layer(sid, layer_name)
        c = await asyncio.to_thread(L.compiled)
        if L._block is None or L._kinds:
            raise HTTPException(409, 'client mode needs a bulk object layer')
        ver = L.version
        data = await asyncio.to_thread(gz.get, (sid, layer_name, ver, 'cols'), lambda: c.columns_payload(L))
        return gz_response(data, v, ver)

    @r.get('/slides/{sid}/annotations/hit')
    async def ann_hit(sid: str, x: float, y: float, tol: float = 0.0, hidden: str = '',
                      layer_name: str = Query(alias='layer')):
        """Smallest feature at (x, y) ± tol (level-0 px), skipping hidden class ids — clicks on aggregated dots."""
        L = layer(sid, layer_name)
        c = await asyncio.to_thread(L.compiled)
        hid = [int(h) for h in hidden.split(',') if h.strip().lstrip('-').isdigit()]
        i = await asyncio.to_thread(c.hit_test, x, y, tol, hid)
        if i is None:
            return JSONResponse({'index': None}, headers=hdrs('no-cache'))
        return JSONResponse({'index': int(i), 'cx': float(c.cx[i]), 'cy': float(c.cy[i]), 'size': float(c.size[i]),
                             'cls': int(c.cls[i])}, headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/annotations/feature/{idx}')
    def ann_feature(sid: str, idx: int, layer_name: str = Query(alias='layer')):
        L = layer(sid, layer_name)
        if not (0 <= idx < len(L)):
            raise HTTPException(404)
        c = L.compiled()
        return JSONResponse({'index': idx, 'class': L.class_of(idx), 'properties': _jsonable(L.properties(idx)),
                             'centroid': [float(c.cx[idx]), float(c.cy[idx])], 'size': float(c.size[idx])},
                            headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/annotations/geojson')
    def ann_geojson(sid: str, layer_name: str = Query(alias='layer')):
        return JSONResponse(_jsonable(layer(sid, layer_name).to_geojson()), headers=hdrs('no-cache'))

    EXPORT_EXT = {'geoparquet': '.parquet', 'geojson': '.geojson', 'qupath': '.geojson', 'halo': '.annotations',
                  'aperio': '.xml', 'csv': '.csv', 'coco': '.coco.json'}

    @r.get('/slides/{sid}/annotations/export')
    def ann_export(sid: str, layer_name: str = Query(alias='layer'), fmt: str = 'geoparquet'):
        """Download a layer as GeoParquet (default), GeoJSON, QuPath GeoJSON, HALO XML, Aperio XML, CSV or COCO."""
        import tempfile

        from fastapi.responses import FileResponse
        from starlette.background import BackgroundTask
        if fmt not in EXPORT_EXT:
            raise HTTPException(400, f'fmt must be one of {", ".join(EXPORT_EXT)}')
        L = layer(sid, layer_name)
        e, pyr = registry.get(sid)
        from . import annio
        tmpdir = Path(tempfile.mkdtemp(prefix='wissi-export-'))
        fname = f'{e.name.rsplit(".", 1)[0]} - {layer_name}{EXPORT_EXT[fmt]}'.replace('/', '_').replace('\\', '_')
        out = annio.export(L, tmpdir / 'out', fmt, mpp=(pyr.info.mpp or (None,))[0], slide=e.name,
                           width=pyr.info.width, height=pyr.info.height)
        import shutil
        return FileResponse(out, filename=fname, headers=hdrs('no-cache'),
                            background=BackgroundTask(shutil.rmtree, tmpdir, True))

    def _table_response(df, fmt: str, filename: str):
        """A pandas table as JSON ({columns, rows}), CSV or Parquet."""
        from fastapi.responses import Response
        if fmt == 'csv':
            return Response(df.to_csv(index=False), media_type='text/csv',
                            headers=hdrs('no-cache', {'Content-Disposition': f'attachment; filename="{filename}.csv"'}))
        if fmt == 'parquet':
            import io
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            return Response(buf.getvalue(), media_type='application/vnd.apache.parquet',
                            headers=hdrs('no-cache', {'Content-Disposition': f'attachment; filename="{filename}.parquet"'}))
        import math
        rows = [[None if isinstance(v, float) and not math.isfinite(v) else _jsonable(v) for v in r]
                for r in df.itertuples(index=False, name=None)]
        return JSONResponse({'columns': list(df.columns), 'rows': rows}, headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/annotations/metrics')
    def ann_metrics(sid: str, layer_name: str = Query(alias='layer'), objects: str | None = None, fmt: str = 'json',
                    intensity: bool = True, texture: bool = True):
        """Measure every region of a layer: shape, intensity / stain, texture and — with ``objects`` (a layer of
        cells) — counts, densities, composition, diversity, distances and clustering (wissi_viz.metrics)."""
        from .metrics import region_metrics
        e, pyr = registry.get(sid)
        regions = layer(sid, layer_name)
        objs = layer(sid, objects) if objects else None
        try:
            df = region_metrics(regions, source=pyr.source, objects=objs, intensity=intensity, texture=texture)
        except Exception as ex:  # noqa: BLE001  (say what failed instead of a bare 500)
            import logging
            logging.getLogger('wissi_viz').exception('region metrics failed')
            raise HTTPException(500, f'{type(ex).__name__}: {ex}')
        return _table_response(df, fmt, f'{e.name.rsplit(".", 1)[0]} - {layer_name} metrics')

    @r.get('/slides/{sid}/annotations/object_metrics')
    def ann_object_metrics(sid: str, layer_name: str = Query(alias='layer'), fmt: str = 'csv', intensity: bool = True,
                           limit: int = 5000):
        """Per-object shape and intensity of a segmented layer (JSON: the first ``limit`` rows; CSV / Parquet: all)."""
        from .metrics import object_metrics
        e, pyr = registry.get(sid)
        try:
            df = object_metrics(layer(sid, layer_name), source=pyr.source, intensity=intensity)
        except Exception as ex:  # noqa: BLE001
            raise HTTPException(500, f'{type(ex).__name__}: {ex}')
        if fmt == 'json':
            df = df.head(limit)
        return _table_response(df, fmt, f'{e.name.rsplit(".", 1)[0]} - {layer_name} objects')

    @r.get('/sam/status')
    def sam_status(load: bool = False):
        """Which SAM backend serves click / box segmentation (torch, onnx or the model-free classic fallback)."""
        from .sam import service
        return JSONResponse(service().status(load=load), headers=hdrs('no-cache'))

    @r.post('/slides/{sid}/sam')
    async def sam_segment(sid: str, request: Request):
        """Segment from prompts: {rect: [x0, y0, x1, y1] (the pane's view, level-0 px), points: [[x, y], …],
        labels: [1 | 0, …], box: [x0, y0, x1, y1] | null} → {polygons: [[[x, y], …], …], score, backend, …}."""
        import anyio

        from .sam import service
        body = await request.json()
        e, pyr = registry.get(sid)

        def run():
            return service().segment(pyr.source, body['rect'], body.get('points') or [], body.get('labels') or [],
                                     body.get('box'), key=f'{sid}@{e.source.version}')
        try:
            polys, info = await anyio.to_thread.run_sync(run)
        except ValueError as ex:
            raise HTTPException(400, str(ex))
        return JSONResponse({'polygons': [p.round(2).tolist() for p in polys], **info}, headers=hdrs('no-cache'))

    if allow_edit:
        @r.post('/slides/{sid}/annotations/add')
        async def ann_add(sid: str, request: Request, layer_name: str = Query(alias='layer')):
            """Append GeoJSON features (e.g. drawings made in the viewer) to a layer, creating it if needed."""
            entry(sid)
            body = await request.json()
            layers = registry.layers(sid)
            L = layers.get(layer_name)
            new = AnnotationLayer.from_geojson(body, name=layer_name)
            if L is None:
                registry.add_annotations(sid, new)
            else:
                from .annotations import _add_geometry, _class_from_props
                for f in (body.get('features') or [body]):
                    props = f.get('properties') or {}
                    cn, col = _class_from_props(props)
                    _add_geometry(L, f.get('geometry') or f, cn, col, props or None)
                registry.changed(sid, layer_name)
            return JSONResponse({'ok': True}, headers=hdrs('no-cache'))

    return r


def _jsonable(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))


def create_app(registry: SlideRegistry, prefix: str = '/wissi-viz', cors: str | None = '*'):
    """Standalone FastAPI app (used by ``wissi-viz serve``)."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse

    app = FastAPI(title='wissi-viz')
    if cors:
        app.add_middleware(CORSMiddleware, allow_origins=[cors] if cors != '*' else ['*'],
                           allow_methods=['*'], allow_headers=['*'])
    app.include_router(create_router(registry, cors=cors), prefix=prefix)

    @app.get('/', response_class=HTMLResponse)
    def index():   # the app UI: slide list, 1–32 panes, tools
        return app_page(prefix, mode='server')

    @app.get('/demo', response_class=HTMLResponse)
    def demo():    # the minimal element demo (go-to controls, no slide list)
        return (STATIC_DIR / 'demo.html').read_text(encoding='utf-8').replace('__API__', prefix)

    @app.get('/app/icon.{kind}')
    def app_icon(kind: str):
        from fastapi import HTTPException
        from fastapi.responses import FileResponse
        if kind not in ('svg', 'png'):
            raise HTTPException(404)
        return FileResponse(ASSETS_DIR / f'icon.{kind}', headers={'Cache-Control': 'max-age=86400'})

    return app


def app_page(prefix: str = '/wissi-viz', mode: str = 'server', token: str = '') -> str:
    """The app page (assets/desktop.html). mode='server': slides from the registry, no file opening;
    mode='desktop': local files through the desktop window's bridge or token-protected endpoints."""
    from . import __version__
    page = (ASSETS_DIR / 'desktop.html').read_text(encoding='utf-8')
    return (page.replace('__API__', prefix).replace('__TOKEN__', token).replace('__MODE__', mode)
            .replace('__VERSION__', __version__))
