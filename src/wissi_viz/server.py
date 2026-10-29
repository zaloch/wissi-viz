"""FastAPI router serving tiles, annotations and the viewer's JavaScript.

Mount it on any FastAPI app (NiceGUI's ``app`` is a FastAPI app)::

    from wissi_viz import SlideRegistry, create_router
    registry = SlideRegistry()
    app.include_router(create_router(registry), prefix='/wissi-viz')
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import math
import mimetypes
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

import anyio
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image
from starlette.background import BackgroundTask

from . import annio, sam3, summary
from ._version import __version__
from .annotations import AnnotationLayer, _add_geometry, _class_from_props
from .jobs import JobRunner
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
    jobs = getattr(registry, 'jobs', None)
    if jobs is None:  # one runner per registry: every router on it sees the same jobs
        jobs = registry.jobs = JobRunner()

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
        mt = (
            'text/javascript'
            if p.suffix in ('.js', '.mjs')
            else (mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
        )
        return FileResponse(p, media_type=mt, headers=hdrs('no-cache'))

    @r.get('/slides')
    def slides():
        return JSONResponse(
            [{'id': i, 'name': registry._entries[i].name} for i in registry.ids()], headers=hdrs('no-cache')
        )

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

    @r.get('/slides/{sid}/histogram')
    async def histogram(sid: str, rect: str | None = None, bins: int = 256, max_pixels: int = 4_000_000):
        """Per-channel (fluorescence) or R, G, B (brightfield) histograms of the whole slide or ``rect``
        (x0,y0,x1,y1 in level-0 px), read at the level where that is ≤ ``max_pixels``: {channels: [{name, counts,
        lo, hi, zeros, p: {0.1, 1, 50, 99, 99.9}}], level, domain}. Zero pixels (unscanned mosaic area) are counted
        apart, so they don't skew the percentiles."""
        e, pyr = entry(sid)
        src = e.source
        info = src.info
        bins = max(16, min(int(bins), 1024))
        if rect:
            x0, y0, x1, y1 = (float(v) for v in rect.split(','))
        else:
            x0, y0, x1, y1 = 0.0, 0.0, float(info.width), float(info.height)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(info.width), x1), min(float(info.height), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise HTTPException(400, 'empty rectangle')

        def make():
            lv = len(info.levels) - 1  # the finest level where the rectangle is ≤ max_pixels
            for i, L in enumerate(info.levels):
                if (x1 - x0) * (y1 - y0) / L.downsample**2 <= max_pixels:
                    lv = i
                    break
            L = info.levels[lv]
            ds = L.downsample
            lx, ly = int(x0 / ds), int(y0 / ds)
            w = max(1, min(L.width - lx, int(math.ceil((x1 - x0) / ds))))
            h = max(1, min(L.height - ly, int(math.ceil((y1 - y0) / ds))))
            if info.mode == 'rgb':
                a = np.asarray(src.read_region(lv, lx, ly, w, h))[..., :3]
                planes = [('red', a[..., 0]), ('green', a[..., 1]), ('blue', a[..., 2])]
                top = 255
            else:
                planes = [
                    (c.name or f'channel {i + 1}', np.asarray(src.read_region(lv, lx, ly, w, h, i)))
                    for i, c in enumerate(info.channels)
                ]
                top = 255 if info.dtype == 'uint8' else 65535
            out = []
            for name, v in planes:
                v = v.ravel()
                nz = v[v > 0]
                zeros = int(v.size - nz.size)
                vals = nz if nz.size > 100 else v
                hi = int(vals.max()) if vals.size else top
                hi = max(hi, 1)
                counts, _ = np.histogram(vals, bins=bins, range=(0, hi + 1))
                ps = np.percentile(vals, [0.1, 1, 50, 99, 99.9]) if vals.size else [0] * 5
                out.append({
                    'name': name, 'counts': counts.tolist(), 'lo': 0, 'hi': hi + 1, 'zeros': zeros,
                    'min': int(vals.min()) if vals.size else 0, 'max': hi,
                    'p': dict(zip(('0.1', '1', '50', '99', '99.9'), (float(x) for x in ps))),
                })  # fmt: skip
            return {'channels': out, 'level': lv, 'domain': top, 'rect': [x0, y0, x1, y1], 'pixels': int(w * h)}

        try:
            d = await asyncio.to_thread(make)
        except Exception as ex:
            raise HTTPException(500, f'histogram: {ex}') from ex
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
                comp = None
                wins = e.windows or src.channel_stats()
                for ci, ch in enumerate(src.info.channels):
                    t = src.thumbnail(size, ci).astype(np.float32)
                    lo, hi = wins[ci]
                    t = np.clip((t - lo) / max(hi - lo, 1), 0, 1)
                    col = np.array([int(ch.color[i : i + 2], 16) for i in (1, 3, 5)], np.float32)
                    comp = t[..., None] * col if comp is None else comp + t[..., None] * col
                a = np.clip(comp, 0, 255).astype(np.uint8)
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
        return Response(data, media_type='application/octet-stream', headers=hdrs(cache, {'Content-Encoding': 'gzip'}))

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
        return JSONResponse(
            metas,
            headers=hdrs(
                'no-cache', {'X-Pending': '0' if loaded else '1', 'Access-Control-Expose-Headers': 'X-Pending'}
            ),
        )

    @r.get('/slides/{sid}/annotations/overview')
    async def ann_overview(sid: str, layer_name: str = Query(alias='layer'), v: str | None = None):
        L = layer(sid, layer_name)
        await asyncio.to_thread(L.compiled)  # compile off the event loop
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
    async def ann_ptile(
        sid: str, level: int, tx: int, ty: int, layer_name: str = Query(alias='layer'), v: str | None = None
    ):
        """Object-pyramid tile: aggregated dots (level ≥ 1) or the raw features (level 0) of one tile."""
        L = layer(sid, layer_name)
        c = await asyncio.to_thread(L.compiled)
        if c.pyr_meta is None or not (0 <= level < c.pyr_meta['levels']):
            raise HTTPException(404)
        ver = L.version
        data = await asyncio.to_thread(
            gz.get, (sid, layer_name, ver, 'pt', level, tx, ty), lambda: c.pyramid_tile(level, tx, ty)
        )
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
    async def ann_hit(
        sid: str, x: float, y: float, tol: float = 0.0, hidden: str = '', layer_name: str = Query(alias='layer')
    ):
        """Smallest feature at (x, y) ± tol (level-0 px), skipping hidden class ids — clicks on aggregated dots."""
        L = layer(sid, layer_name)
        c = await asyncio.to_thread(L.compiled)
        hid = [int(h) for h in hidden.split(',') if h.strip().lstrip('-').isdigit()]
        i = await asyncio.to_thread(c.hit_test, x, y, tol, hid)
        if i is None:
            return JSONResponse({'index': None}, headers=hdrs('no-cache'))
        return JSONResponse(
            {
                'index': int(i),
                'cx': float(c.cx[i]),
                'cy': float(c.cy[i]),
                'size': float(c.size[i]),
                'cls': int(c.cls[i]),
            },
            headers=hdrs('no-cache'),
        )

    @r.get('/slides/{sid}/annotations/feature/{idx}')
    def ann_feature(sid: str, idx: int, layer_name: str = Query(alias='layer'), pct: bool = False):
        """One object: class, properties, centroid, size; ``pct=1`` adds each measurement's percentile within
        its layer (the app's object card)."""
        L = layer(sid, layer_name)
        if not (0 <= idx < len(L)):
            raise HTTPException(404)
        c = L.compiled()
        props = L.properties(idx)
        out = {
            'index': idx,
            'class': L.class_of(idx),
            'properties': _jsonable(props),
            'centroid': [float(c.cx[idx]), float(c.cy[idx])],
            'size': float(c.size[idx]),
        }
        if pct:
            out['percentiles'] = summary.percentiles(L, props)
            out['order'] = summary.preferred_columns(L, 8)
        return JSONResponse(out, headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/annotations/summary')
    async def ann_summary(sid: str, layer_name: str = Query(alias='layer'), tissue: bool = True):
        """Counts and shares per class, per-class means of the main measurements, region areas and objects per
        mm² of tissue (the app's Measure tab)."""
        e, pyr = registry.get(sid)
        L = layer(sid, layer_name)
        mpp = (registry.info(sid).get('mpp') or [None])[0]

        def work():
            area = summary.tissue_area_mm2(f'{sid}@{e.source.version}', pyr.source) if tissue else None
            return summary.summary(L, mpp, area)

        return JSONResponse(_jsonable(await anyio.to_thread.run_sync(work)), headers=hdrs('no-cache'))

    @r.get('/analyses')
    def analyses_catalog():
        """The spatial analyses the app can run (``wissi_viz.analyses``; needs the [analysis] extra)."""
        try:
            from . import analyses
        except ImportError as ex:
            raise HTTPException(501, f'spatial analyses need: pip install "wissi-viz[analysis]" ({ex.name})') from ex
        return JSONResponse(analyses.catalog(), headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/annotations/geojson')
    def ann_geojson(sid: str, layer_name: str = Query(alias='layer')):
        return JSONResponse(_jsonable(layer(sid, layer_name).to_geojson()), headers=hdrs('no-cache'))

    EXPORT_EXT = {
        'geoparquet': '.parquet',
        'geojson': '.geojson',
        'qupath': '.geojson',
        'halo': '.annotations',
        'aperio': '.xml',
        'csv': '.csv',
        'coco': '.coco.json',
    }

    @r.get('/slides/{sid}/annotations/export')
    def ann_export(sid: str, layer_name: str = Query(alias='layer'), fmt: str = 'geoparquet'):
        """Download a layer as GeoParquet (default), GeoJSON, QuPath GeoJSON, HALO XML, Aperio XML, CSV or COCO."""

        if fmt not in EXPORT_EXT:
            raise HTTPException(400, f'fmt must be one of {", ".join(EXPORT_EXT)}')
        L = layer(sid, layer_name)
        e, pyr = registry.get(sid)
        tmpdir = Path(tempfile.mkdtemp(prefix='wissi-export-'))
        fname = f'{e.name.rsplit(".", 1)[0]} - {layer_name}{EXPORT_EXT[fmt]}'.replace('/', '_').replace('\\', '_')
        out = annio.export(
            L,
            tmpdir / 'out',
            fmt,
            mpp=(pyr.info.mpp or (None,))[0],
            slide=e.name,
            width=pyr.info.width,
            height=pyr.info.height,
        )
        return FileResponse(
            out, filename=fname, headers=hdrs('no-cache'), background=BackgroundTask(shutil.rmtree, tmpdir, True)
        )

    def _table_response(df, fmt: str, filename: str):
        """A pandas table as JSON ({columns, rows}), CSV or Parquet."""
        if fmt == 'csv':
            return Response(
                df.to_csv(index=False),
                media_type='text/csv',
                headers=hdrs('no-cache', {'Content-Disposition': _attachment(f'{filename}.csv')}),
            )
        if fmt == 'parquet':
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            return Response(
                buf.getvalue(),
                media_type='application/vnd.apache.parquet',
                headers=hdrs('no-cache', {'Content-Disposition': _attachment(f'{filename}.parquet')}),
            )
        rows = [
            [None if isinstance(v, float) and not math.isfinite(v) else _jsonable(v) for v in r]
            for r in df.itertuples(index=False, name=None)
        ]
        return JSONResponse({'columns': list(df.columns), 'rows': rows}, headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/annotations/metrics')
    def ann_metrics(
        sid: str,
        layer_name: str = Query(alias='layer'),
        objects: str | None = None,
        fmt: str = 'json',
        intensity: bool = True,
        texture: bool = True,
    ):
        """Measure every region of a layer: shape, intensity / stain, texture and — with ``objects`` (a layer of
        cells) — counts, densities, composition, diversity, distances and clustering (wissi_viz.metrics)."""
        from .metrics import region_metrics

        e, pyr = registry.get(sid)
        regions = layer(sid, layer_name)
        objs = layer(sid, objects) if objects else None
        try:
            df = region_metrics(regions, source=pyr.source, objects=objs, intensity=intensity, texture=texture)
        except Exception as ex:  # say what failed instead of a bare 500
            logging.getLogger('wissi_viz').exception('region metrics failed')
            raise HTTPException(500, f'{type(ex).__name__}: {ex}') from ex
        return _table_response(df, fmt, f'{e.name.rsplit(".", 1)[0]} - {layer_name} metrics')

    @r.get('/slides/{sid}/annotations/object_metrics')
    def ann_object_metrics(
        sid: str, layer_name: str = Query(alias='layer'), fmt: str = 'csv', intensity: bool = True, limit: int = 5000
    ):
        """Per-object shape and intensity of a segmented layer (JSON: the first ``limit`` rows; CSV / Parquet: all)."""
        from .metrics import object_metrics

        e, pyr = registry.get(sid)
        try:
            df = object_metrics(layer(sid, layer_name), source=pyr.source, intensity=intensity)
        except Exception as ex:
            raise HTTPException(500, f'{type(ex).__name__}: {ex}') from ex
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

        from .sam import service

        body = await request.json()
        e, pyr = registry.get(sid)

        def run():
            return service().segment(
                pyr.source,
                body['rect'],
                body.get('points') or [],
                body.get('labels') or [],
                body.get('box'),
                key=f'{sid}@{e.source.version}',
            )

        try:
            polys, info = await anyio.to_thread.run_sync(run)
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex
        return JSONResponse({'polygons': [p.round(2).tolist() for p in polys], **info}, headers=hdrs('no-cache'))

    if allow_edit:

        @r.post('/slides/{sid}/annotations/add')
        async def ann_add(sid: str, request: Request, layer_name: str = Query(alias='layer')):
            """Append GeoJSON features (e.g. drawings made in the viewer) to a layer, creating it if needed."""
            entry(sid)
            body = await request.json()
            layers = registry.layers(sid)
            L = layers.get(layer_name)
            feats = body.get('features') or [body]
            # where the shapes came from: SAM masks carry source=sam, the drawing tools source=drawn
            srcs = {str((f.get('properties') or {}).get('source') or 'drawn') for f in feats}
            origin = 'sam' if srcs == {'sam'} else 'drawn' if srcs <= {'drawn'} else 'drawn+sam'
            if L is None:
                new = AnnotationLayer.from_geojson(body, name=layer_name)
                new.origin = origin
                registry.add_annotations(sid, new)
            else:
                for f in feats:
                    props = f.get('properties') or {}
                    cn, col = _class_from_props(props)
                    _add_geometry(L, f.get('geometry') or f, cn, col, props or None)
                if L.origin in ('drawn', 'sam', 'drawn+sam'):
                    L.origin = L.origin if L.origin == origin else 'drawn+sam'
                elif not L.origin:
                    L.origin = origin
                registry.changed(sid, layer_name)
            return JSONResponse({'ok': True}, headers=hdrs('no-cache'))

        @r.post('/slides/{sid}/annotations/style')
        async def ann_style(sid: str, request: Request):
            """Colour a layer by a measurement ({layer, color_by: column | null, cmap}); null: back to classes."""
            body = await request.json()
            L = layer(sid, body.get('layer') or '')
            col = body.get('color_by') or None
            if col is not None and col not in L.numeric_columns():
                raise HTTPException(400, f'no numeric column {col!r} in {L.name}')
            await anyio.to_thread.run_sync(lambda: L.color_by(col, cmap=body.get('cmap') or 'viridis'))
            registry.changed(sid, L.name)
            return JSONResponse({'ok': True, 'colormap': _jsonable(L.colormap)}, headers=hdrs('no-cache'))

        @r.delete('/slides/{sid}/annotations')
        def ann_remove(sid: str, layer_name: str = Query(alias='layer')):
            layer(sid, layer_name)
            registry.remove_annotations(sid, layer_name)
            return JSONResponse({'ok': True}, headers=hdrs('no-cache'))

        @r.post('/slides/{sid}/analyses')
        async def run_analysis(sid: str, request: Request):
            """Run a spatial analysis on a cell layer: {analysis, objects, params} → headline numbers, chart, table;
            result layers are added to the slide."""
            try:
                from . import analyses
            except ImportError as ex:
                raise HTTPException(
                    501, f'spatial analyses need: pip install "wissi-viz[analysis]" ({ex.name})'
                ) from ex
            body = await request.json()
            entry(sid)
            try:
                out = await anyio.to_thread.run_sync(
                    lambda: analyses.run(
                        registry,
                        sid,
                        body.get('analysis'),
                        body.get('objects'),
                        body.get('params'),
                        within=body.get('within'),
                    )  # fmt: skip
                )
            except ValueError as ex:
                raise HTTPException(400, str(ex)) from ex
            except Exception as ex:
                logging.getLogger('wissi_viz').exception('analysis failed')
                raise HTTPException(500, f'{type(ex).__name__}: {ex}') from ex
            return JSONResponse(_jsonable(out), headers=hdrs('no-cache'))

        @r.post('/analyses/batch')
        async def analyses_batch(request: Request):
            """The same analysis on several images, in the background: {analysis, slides: [id…], objects, params,
            within} → the job; its result: one row of headline numbers per image (``analyses.run_batch``)."""
            try:
                from . import analyses
            except ImportError as ex:
                raise HTTPException(
                    501, f'spatial analyses need: pip install "wissi-viz[analysis]" ({ex.name})'
                ) from ex
            b = await request.json()
            slides = [s_ for s_ in (b.get('slides') or []) if s_ in registry.ids()]
            job = jobs.submit(
                'analysis', lambda job: _jsonable(analyses.run_batch(registry, slides, b.get('analysis'),
                                                                     b.get('objects'), b.get('params'),
                                                                     b.get('within'), job)),
                title=f'{b.get("analysis")} on {len(slides)} images',
            )  # fmt: skip
            return JSONResponse(job.to_dict(), headers=hdrs('no-cache'))

        @r.post('/slides/{sid}/layers/regions')
        async def layer_regions(sid: str, request: Request):
            """Merge a layer's objects into regions: {layer, classes, join_um, min_area_um2, name} → {layer}."""
            lo = _layerops()
            b = await request.json()
            entry(sid)
            try:
                name = await anyio.to_thread.run_sync(lambda: lo.to_regions(
                    registry, sid, b.get('layer'), classes=b.get('classes') or None,
                    join_um=float(b.get('join_um', 20)), min_area_um2=float(b.get('min_area_um2', 0)),
                    name=b.get('name')))  # fmt: skip
            except ValueError as ex:
                raise HTTPException(400, str(ex)) from ex
            return JSONResponse({'layer': name}, headers=hdrs('no-cache'))

        @r.post('/slides/{sid}/layers/inside')
        async def layer_inside(sid: str, request: Request):
            """The objects of a layer inside a region layer, as a new layer: {objects, regions, classes, name}."""
            lo = _layerops()
            b = await request.json()
            entry(sid)
            try:
                name = await anyio.to_thread.run_sync(lambda: lo.inside(
                    registry, sid, b.get('objects'), b.get('regions'), classes=b.get('classes') or None,
                    name=b.get('name')))  # fmt: skip
            except ValueError as ex:
                raise HTTPException(400, str(ex)) from ex
            return JSONResponse({'layer': name}, headers=hdrs('no-cache'))

        @r.post('/slides/{sid}/find')
        async def find_start(sid: str, request: Request):
            """Find all like the examples, or by name, in the background: {mode: objects | segment | areas | text,
            examples: {layer, classes, picks}, candidates, training, classes, scope: {kind: view | region | slide,
            rect, layer}, strictness, object_um, detail, prompt, model: grounding | sam3, name} → the job (poll
            ``GET /jobs/{id}``)."""
            findall = _findall()
            body = await request.json()
            entry(sid)
            titles = {'objects': 'Find similar objects', 'segment': 'Segment everything', 'areas': 'Find areas',
                      'text': 'Find by text'}  # fmt: skip
            job = jobs.submit(
                'find', lambda job: _jsonable(findall.run(registry, sid, body, job)), slide=sid,
                title=titles.get(body.get('mode'), 'Find all'),
            )  # fmt: skip
            return JSONResponse(job.to_dict(), headers=hdrs('no-cache'))

        @r.post('/slides/{sid}/find/threshold')
        async def find_threshold(sid: str, request: Request):
            """Apply another strictness to a finished result: {result, strictness} → {layer, count, of}."""
            findall = _findall()
            body = await request.json()
            try:
                out = await anyio.to_thread.run_sync(
                    lambda: findall.rethreshold(registry, body.get('result') or '', float(body.get('strictness', 0.5)))
                )
            except ValueError as ex:
                raise HTTPException(400, str(ex)) from ex
            return JSONResponse(out, headers=hdrs('no-cache'))

        @r.post('/models/sam3/license')
        async def sam3_license_accept(request: Request):
            """Accept ({accept: true}) or withdraw ({accept: false}) the SAM License for SAM 3 on this computer."""
            body = await request.json()
            return JSONResponse(sam3.accept(bool(body.get('accept'))), headers=hdrs('no-cache'))

        @r.post('/jobs/{jid}/cancel')
        def job_cancel(jid: str):
            job = jobs.get(jid)
            if job is None:
                raise HTTPException(404, 'no such job')
            job.cancel()
            return JSONResponse(job.to_dict(), headers=hdrs('no-cache'))

    @r.get('/find/capabilities')
    def find_caps():
        """What finding can use here: the SAM backend, whether area classification (scikit-learn) is installed."""
        return JSONResponse({**_findall().capabilities(), 'edit': allow_edit}, headers=hdrs('no-cache'))

    @r.get('/models/sam3/license')
    def sam3_license():
        """SAM 3's licence (shown before it is downloaded) and whether it was accepted here."""
        return JSONResponse({'name': sam3.LICENSE_NAME, 'url': sam3.LICENSE_URL, 'text': sam3.license_text(),
                             **sam3.status()}, headers=hdrs('no-cache'))  # fmt: skip

    @r.get('/jobs/{jid}')
    def job_status(jid: str):
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, 'no such job (the server restarted?)')
        return JSONResponse(job.to_dict(), headers=hdrs('no-cache'))

    @r.get('/slides/{sid}/jobs')
    def slide_jobs(sid: str):
        return JSONResponse([j.to_dict() for j in jobs.list(sid)], headers=hdrs('no-cache'))

    return r


def _attachment(name: str) -> str:
    """Content-Disposition for a download named ``name``: an ASCII fallback plus the UTF-8 name (RFC 6266), so
    slides called 'Präparat 1' download as such (a raw non-ASCII header breaks the response)."""
    ascii_name = name.encode('ascii', 'replace').decode().replace('?', '_').replace('"', '')
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(name)}'


def _layerops():
    try:
        from . import layerops
    except ImportError as ex:
        raise HTTPException(501, f'layer operations need: pip install "wissi-viz[analysis]" ({ex.name})') from ex
    return layerops


def _findall():
    try:
        from . import findall
    except ImportError as ex:
        raise HTTPException(501, f'finding needs: pip install "wissi-viz[analysis]" ({ex.name})') from ex
    return findall


def _jsonable(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))


class _ApiCORS(CORSMiddleware):
    """CORS for the tile / annotation API only: other sites may embed the viewer, but never read the app page
    (it carries the token for the app's own endpoints)."""

    def __init__(self, app, path_prefix: str, **kw) -> None:
        super().__init__(app, **kw)
        self.path_prefix = path_prefix

    async def __call__(self, scope, receive, send) -> None:
        path, root = scope.get('path', ''), scope.get('root_path', '')
        if root and path.startswith(root):
            path = path[len(root) :]
        if scope['type'] == 'http' and not path.startswith(self.path_prefix):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


def create_app(
    registry: SlideRegistry,
    prefix: str = '/wissi-viz',
    cors: str | None = '*',
    index: bool = True,
    dependencies: list | None = None,
):
    """Standalone FastAPI app: the API under ``prefix``, the app page at ``/`` (``index=False``: none; see
    :func:`wissi_viz.session.create_session_app` for the page with file opening and connections). It also works
    mounted under a path (``parent.mount('/viewer', app)``). ``dependencies``: e.g. your login check, on every
    route."""

    app = FastAPI(title='wissi-viz', dependencies=list(dependencies or []))
    if cors:
        app.add_middleware(
            _ApiCORS,
            path_prefix=prefix,
            allow_origins=[cors] if cors != '*' else ['*'],
            allow_methods=['*'],
            allow_headers=['*'],
        )
    app.include_router(create_router(registry, cors=cors), prefix=prefix)

    if index:

        @app.get('/', response_class=HTMLResponse)
        def index_page(request: Request):  # the app UI: slide list, 1–32 panes, tools
            root = request.scope.get('root_path', '')
            return app_page(root + prefix, mode='server', base=root)

    @app.get('/demo', response_class=HTMLResponse)
    def demo(request: Request):  # the minimal element demo (go-to controls, no slide list)
        root = request.scope.get('root_path', '')
        return (STATIC_DIR / 'demo.html').read_text(encoding='utf-8').replace('__API__', root + prefix)

    @app.get('/app/icon.{kind}')
    def app_icon(kind: str):
        if kind not in ('svg', 'png'):
            raise HTTPException(404)
        return FileResponse(ASSETS_DIR / f'icon.{kind}', headers={'Cache-Control': 'max-age=86400'})

    return app


def app_page(
    prefix: str = '/wissi-viz', mode: str = 'server', token: str = '', caps: dict | None = None, base: str = ''
) -> str:
    """The app page (assets/desktop.html). mode='server': a server's slides; mode='desktop': local files through
    the desktop window's bridge. ``caps``: what the page may offer (open / browse / connect / close / key); none
    without a session (:func:`wissi_viz.session.create_session_app`). ``base``: the path the app is mounted at."""
    page = (ASSETS_DIR / 'desktop.html').read_text(encoding='utf-8')
    return (
        page.replace('__API__', prefix)
        .replace('__TOKEN__', token)
        .replace('__MODE__', mode)
        .replace('__CAPS__', json.dumps(caps or {}))
        .replace('__BASE__', base)
        .replace('__VERSION__', __version__)
    )
