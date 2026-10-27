"""Mock PathCore / HALOLink connections backed by a local slide — for tests and demos.

They deliberately reproduce the awkward parts of real tile APIs:

* MockHaloLink: DeepZoom pyramid (tile 254 + 1 px overlap, level ids = DZI numbers),
  JPEG bytes, sync client, HALO XML annotations + a cell table (XMin/XMax/YMin/YMax),
  occasional transient failures, "no tile" for empty background.
* MockPathCore: *async* client, levels listed coarsest-first with string ids, fixed
  256×256 PNG tiles padded past the image edge, GeoJSON annotations in microns.
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import random
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from wissi_viz.connectors import AnnotationPayload, HaloLinkConnection, PathCoreConnection, RemoteImageInfo
from wissi_viz.sources import open_slide


class _Backing:
    """Reads any (downsample, region) from a local slide, box-filtering from the nearest native level."""

    def __init__(self, path):
        self.src = open_slide(path)
        self.info = self.src.info
        self.lock = threading.Lock()

    def region(self, ds: float, x: int, y: int, w: int, h: int) -> np.ndarray:
        """(x, y, w, h) in the coordinates of a level with downsample ``ds``; zeros past the image."""
        nat = max((i for i, lv in enumerate(self.info.levels) if lv.downsample <= ds + 1e-6), key=lambda i: self.info.levels[i].downsample)
        nl = self.info.levels[nat]
        f = int(round(ds / nl.downsample))
        X, Y, W, H = x * f, y * f, w * f, h * f
        out = np.zeros((H, W, 3), np.uint8)
        x1, y1 = min(X + W, nl.width), min(Y + H, nl.height)
        if x1 > X and y1 > Y:
            with self.lock:
                a = self.src.read_region(nat, X, Y, x1 - X, y1 - Y)
            out[:y1 - Y, :x1 - X] = a
        if f > 1:
            out = out.reshape(h, f, w, f, 3).mean((1, 3)).astype(np.uint8)
        return out


def _enc(a: np.ndarray, fmt: str) -> bytes:
    b = io.BytesIO()
    Image.fromarray(a).save(b, fmt, quality=92) if fmt == 'JPEG' else Image.fromarray(a).save(b, fmt)
    return b.getvalue()


class MockHaloLink(HaloLinkConnection):
    def __init__(self, slide_path, geojson_path=None, fail_rate=0.0, fmt='JPEG'):
        super().__init__()
        self.fmt = fmt
        self.b = _Backing(slide_path)
        self.geojson_path = geojson_path
        self.fail_rate = fail_rate
        self._rng = random.Random(0)  # deterministic failures
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._l = threading.Lock()
        i = self.b.info
        self.W, self.H = i.width, i.height
        self.max_level = math.ceil(math.log2(max(self.W, self.H)))

    def image_info(self, image_id):
        return RemoteImageInfo(width=self.W, height=self.H, tile_size=254, overlap=1, deepzoom=True,
                               mpp=self.b.info.mpp, name=f'HALO {image_id}', version='halo-v1')

    def tile(self, image_id, level, col, row, channel=None):
        with self._l:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.fail_rate and self._rng.random() < self.fail_rate:
                raise ConnectionError('transient upstream error')
            ds = 2 ** (self.max_level - int(level))
            lw, lh = math.ceil(self.W / ds), math.ceil(self.H / ds)
            ts, ov = 254, 1
            x0 = col * ts - (ov if col > 0 else 0)
            y0 = row * ts - (ov if row > 0 else 0)
            x1 = min(lw, (col + 1) * ts + ov)
            y1 = min(lh, (row + 1) * ts + ov)
            a = self.b.region(ds, x0, y0, x1 - x0, y1 - y0)
            return _enc(a, self.fmt)
        finally:
            with self._l:
                self.active -= 1

    def annotations(self, image_id):
        # ROI as HALO XML (level-0 px)
        w, h = self.W, self.H
        xml = (f'<Annotations><Annotation Name="Tumor ROI" LineColor="65280"><Regions><Region Type="Polygon">'
               f'<Vertices><V X="{w*.1}" Y="{h*.1}"/><V X="{w*.45}" Y="{h*.12}"/><V X="{w*.5}" Y="{h*.5}"/>'
               f'<V X="{w*.2}" Y="{h*.45}"/></Vertices></Region></Regions></Annotation></Annotations>')
        return [AnnotationPayload('HALO annotations', xml, format='halo_xml', style={'line_width': 3})]

    def objects(self, image_id):
        if not self.geojson_path:
            return []
        feats = json.loads(Path(self.geojson_path).read_text(encoding='utf-8'))['features']
        rows = {'XMin': [], 'XMax': [], 'YMin': [], 'YMax': [], 'Classification': [], 'Area': []}
        for f in feats:
            if f['geometry']['type'] != 'Polygon' or f['properties']['classification']['name'] == 'ROI':
                continue
            c = np.asarray(f['geometry']['coordinates'][0])
            rows['XMin'].append(c[:, 0].min()); rows['XMax'].append(c[:, 0].max())
            rows['YMin'].append(c[:, 1].min()); rows['YMax'].append(c[:, 1].max())
            rows['Classification'].append(f['properties']['classification']['name'])
            rows['Area'].append(f['properties'].get('area', 0))
        return [AnnotationPayload('HALO objects', rows, format='table', style={'fill_opacity': 0.0})]


class MockPathCore(PathCoreConnection):
    def __init__(self, slide_path):
        super().__init__()
        self.b = _Backing(slide_path)
        i = self.b.info
        self.levels = [(lv.width, lv.height, lv.downsample) for lv in i.levels]
        self.ids = [f'L{k}' for k in range(len(self.levels))]

    async def image_info(self, slide_id):
        await asyncio.sleep(0.01)
        i = self.b.info
        # coarsest-first ordering with opaque string ids, as some APIs return them
        return RemoteImageInfo(width=i.width, height=i.height, tile_size=256,
                               levels=[lv[:2] for lv in reversed(self.levels)], level_ids=list(reversed(self.ids)),
                               mpp=i.mpp[0], name=f'PathCore {slide_id}', version='pc-etag-1')

    async def tile(self, slide_id, level, col, row, channel=None):
        await asyncio.sleep(0.005)  # network latency
        k = self.ids.index(level)
        w, h, ds = self.levels[k]
        a = self.b.region(ds, col * 256, row * 256, 256, 256)  # always 256×256: padded past the edge
        return _enc(a, 'PNG')

    async def annotations(self, slide_id):
        i = self.b.info
        mx = i.mpp[0]
        w, h = i.width * mx, i.height * mx  # µm
        gj = {'type': 'FeatureCollection', 'features': [{
            'type': 'Feature', 'properties': {'classification': {'name': 'Margin', 'color': [255, 0, 255]}},
            'geometry': {'type': 'LineString', 'coordinates': [[w * .55, h * .2], [w * .7, h * .35], [w * .8, h * .3], [w * .9, h * .6]]}}]}
        return [AnnotationPayload('PathCore annotations', gj, units='um', style={'line_width': 3})]
