"""Annotation layers: ROIs, drawings and (up to millions of) detected objects.

Storage is columnar (numpy) so layers with 10^6 cells stay cheap. For the browser,
each layer is split into a quadtree of *buckets* (≈ vector tiles):

* ``overview.bin`` — one point per feature (centroid, size, class, original index)
  plus the bucket table. Small, loaded once, drawn as GPU points when zoomed out.
* ``bucket/{i}.bin`` — exact outlines (as line segments) and triangulated fills for
  the features of one bucket. Fetched only for buckets inside the view *and* zoomed
  in far enough for outlines to be visible; cached on the server and in the browser.

Coordinates are always level-0 pixels of the slide.
"""
from __future__ import annotations

import colorsys
import json
import os
import struct
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .colormaps import CATEGORICAL as _CAT, DIVERGING as _DIVERGING, lut as _lut

POINT, POLYLINE, POLYGON = 0, 1, 2
#: parametric kinds (bulk objects): defined by their bounding box alone, drawn analytically on the GPU
ELLIPSE, RECT = 3, 4
#: benchmarks only: WISSI_LEGACY_OVERLAYS=1 restores the pre-0.5 path (vertex rings per object, one overview)
LEGACY_OVERLAYS = os.environ.get('WISSI_LEGACY_OVERLAYS') == '1'
MAGIC = b'WSA1'



def _color_hex(c: Any) -> str | None:
    if c is None:
        return None
    if isinstance(c, str):
        c = c.strip()
        if c.startswith('#') and len(c) in (7, 9):
            return c[:7].lower()
        return None
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        r, g, b = (int(v * 255) if isinstance(v, float) and max(c[:3]) <= 1 else int(v) for v in c[:3])
        return '#%02x%02x%02x' % (r & 255, g & 255, b & 255)
    if isinstance(c, (int, np.integer)):  # packed int, assumed 0xRRGGBB (QuPath uses signed ARGB)
        v = int(c) & 0xFFFFFF
        return '#%06x' % v
    return None


def _auto_color(i: int) -> str:
    if i < len(_CAT):
        return _CAT[i]
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class AnnotationLayer:
    """A named set of features with per-class styling.

    Build incrementally (``add_polygon`` / ``add_point`` / …) or in bulk
    (``from_points`` / ``from_polygons`` / ``from_boxes`` / ``from_geojson`` /
    ``from_aperio_xml`` / ``from_halo_xml``).
    """
    name: str = 'annotations'
    #: class name → '#rrggbb'
    classes: dict[str, str] = field(default_factory=dict)
    line_width: float = 1.5
    fill_opacity: float = 0.15
    point_size: float = 4.0
    visible: bool = True
    #: Below this on-screen size (px) a feature is drawn as a dot instead of its outline.
    detail_min_px: float = 6.0
    #: draw order across layers (higher on top); default: layers of smaller objects first
    z: float | None = None
    #: zoomed-out stand-in for features below ``detail_min_px``: 'points' (a dot each) or 'none' (hidden —
    #: right for lines/edges, which mean nothing as dots)
    overview: str = 'points'
    #: object pyramid ('auto' = for layers with ≥ PYRAMID_MIN features): zoomed out, the viewer gets aggregated
    #: dots (one per ~4 screen px, with per-class counts) and, zoomed in, only the visible cells, tile by tile —
    #: so what a browser downloads scales with screen area, not with the number of objects
    pyramid: Any = 'auto'

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._kinds: list[int] = []
        self._cls: list[int] = []
        self._radius: list[float] = []
        self._rings: list[np.ndarray] = []  # each (n, 2) float32, without closing vertex
        self._feat_nrings: list[int] = []
        self._props: list[dict | None] = []
        #: columnar per-feature properties (bulk imports: object tables with 10^6 rows)
        self._columns: dict[str, np.ndarray] = {}
        self._col_offset = 0  # feature index where the columnar block starts
        #: continuous colouring (``color_by``): per-feature values + colormap; None → colour by class
        self._values: np.ndarray | None = None
        self.colormap: dict | None = None
        #: bulk parametric features (object tables: points / bbox ellipses / rects) kept columnar:
        #: {'kind': u8[n], 'bb': f32[n, 4], 'cls': u16[n]} — no per-feature arrays for 10^6 cells
        self._block: dict | None = None
        self._dirty = True
        self._compiled: _Compiled | None = None
        # versions key immutable browser caches: start at a random offset so a restarted server never reuses one
        self.version = int.from_bytes(os.urandom(3), 'little') * 1024
        self._class_index: dict[str, int] = {}
        for k in list(self.classes):
            self._class_id(k)

    # ------------------------------------------------------------- building
    def _class_id(self, cls: str | None, color: str | None = None) -> int:
        cls = cls or (color or 'default')
        if cls not in self._class_index:
            self._class_index[cls] = len(self._class_index)
            if cls not in self.classes:
                self.classes[cls] = color or _auto_color(len(self._class_index) - 1)
        elif color and cls == color:
            pass
        return self._class_index[cls]

    def _add(self, kind: int, rings: Sequence[np.ndarray], cls: str | None, color: str | None,
             props: dict | None, radius: float = 0.0) -> int:
        with self._lock:
            self._materialize()
            if color and not cls:
                cls = color
            cid = self._class_id(cls, _color_hex(color))
            self._kinds.append(kind)
            self._cls.append(cid)
            self._radius.append(float(radius))
            self._rings.extend(rings)
            self._feat_nrings.append(len(rings))
            self._props.append(props)
            self._dirty = True
            return len(self._kinds) - 1

    def add_polygon(self, coords: Any, holes: Iterable[Any] = (), cls: str | None = None,
                    color: str | None = None, props: dict | None = None) -> int:
        rings = [_ring(coords, closed=True)] + [_ring(h, closed=True) for h in holes]
        return self._add(POLYGON, rings, cls, color, props)

    def add_polyline(self, coords: Any, cls: str | None = None, color: str | None = None, props: dict | None = None) -> int:
        return self._add(POLYLINE, [_ring(coords, closed=False)], cls, color, props)

    def add_point(self, x: float, y: float, cls: str | None = None, color: str | None = None,
                  props: dict | None = None, radius: float = 0.0) -> int:
        return self._add(POINT, [np.array([[x, y]], np.float32)], cls, color, props, radius)

    def add_rect(self, x: float, y: float, w: float, h: float, **kw) -> int:
        return self.add_polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], **kw)

    def add_ellipse(self, cx: float, cy: float, rx: float, ry: float, n: int = 48, **kw) -> int:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self.add_polygon(np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1), **kw)

    def clear(self) -> None:
        with self._lock:
            v = self.version
            self.__post_init__()
            self.version = v + 1  # versions are cache keys: never reuse one

    def __len__(self) -> int:
        return len(self._kinds) + (len(self._block['kind']) if self._block is not None else 0)

    # ------------------------------------------------------------- columnar bulk shapes
    def _bulk_shapes(self, kind: Any, bb: Any, classes: Sequence[str | None] | None) -> None:
        """Bulk points / ellipses / rects given by bounding boxes (n, 4) = x0, y0, x1, y1, stored columnar."""
        bb = np.ascontiguousarray(np.asarray(bb, np.float32).reshape(-1, 4))
        n = len(bb)
        with self._lock:
            if len(self._kinds) or self._block is not None or LEGACY_OVERLAYS:   # mixing with other features: classic storage
                kinds = np.broadcast_to(np.asarray(kind, np.uint8), (n,))
                for k in np.unique(kinds):
                    sel = np.nonzero(kinds == k)[0]
                    cl = [None] * len(sel) if classes is None else [classes[i] for i in sel]
                    kind_, rings, rad = self._rings_for(int(k), bb[sel])
                    self._bulk(kind_, rings, cl, rad, None)
                return
            if classes is None:
                ids = np.full(n, self._class_id(None), np.uint16)
            else:
                arr = np.asarray([c if c is not None else 'default' for c in classes], dtype=object)
                uniq, inv = np.unique(arr.astype(str), return_inverse=True)
                lut = np.asarray([self._class_id(u, _color_hex(u)) for u in uniq], np.uint16)
                ids = lut[inv]
            self._block = {'kind': np.ascontiguousarray(np.broadcast_to(np.asarray(kind, np.uint8), (n,))), 'bb': bb, 'cls': ids}
            self._dirty = True

    @staticmethod
    def _rings_for(kind: int, bb: np.ndarray):
        """(kind, rings, radius) in classic storage for parametric features."""
        if kind == POINT:
            c = np.stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2], 1).astype(np.float32)
            return POINT, [c[i:i + 1] for i in range(len(c))], ((bb[:, 2] - bb[:, 0]) / 2).astype(np.float32)
        return POLYGON, _box_rings(bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3], 'ellipse' if kind == ELLIPSE else 'rect'), None

    def _materialize(self) -> None:
        """Convert the columnar block to classic per-feature storage (before incremental edits)."""
        b = self._block
        if b is None:
            return
        self._block = None
        inv = {v: k for k, v in self._class_index.items()}
        for k in np.unique(b['kind']):
            sel = np.nonzero(b['kind'] == k)[0]
            kind, rings, rad = self._rings_for(int(k), b['bb'][sel])
            self._kinds.extend([kind] * len(sel))
            self._cls.extend(b['cls'][sel].tolist())
            self._radius.extend(rad.tolist() if rad is not None else [0.0] * len(sel))
            self._rings.extend(rings)
            self._feat_nrings.extend([1] * len(sel))
            self._props.extend([None] * len(sel))
        _ = inv
        self._dirty = True

    def properties(self, index: int) -> dict:
        p = self._props[index] if 0 <= index < len(self._props) else None
        out = dict(p or {})
        j = index - self._col_offset
        if self._columns and 0 <= j < len(next(iter(self._columns.values()))):
            for k, col in self._columns.items():
                v = col[j]
                out[k] = v.item() if hasattr(v, 'item') else v
        return out

    def transform(self, scale: float | tuple[float, float] = 1.0, offset: tuple[float, float] = (0.0, 0.0)) -> 'AnnotationLayer':
        """In place: ``p' = p * scale + offset`` for every vertex (e.g. µm → px: ``scale=1/mpp``;
        coordinates given at a pyramid level: ``scale=downsample``; ROI-relative: ``offset``)."""
        sx, sy = (scale, scale) if isinstance(scale, (int, float)) else scale
        ox, oy = offset
        with self._lock:
            m = np.array([sx, sy], np.float32)
            o = np.array([ox, oy], np.float32)
            self._rings = [r * m + o for r in self._rings]
            self._radius = [r * (abs(sx) + abs(sy)) / 2 for r in self._radius]
            if self._block is not None:
                bb = self._block['bb'] * np.array([sx, sy, sx, sy], np.float32) + np.array([ox, oy, ox, oy], np.float32)
                self._block['bb'] = np.stack([np.minimum(bb[:, 0], bb[:, 2]), np.minimum(bb[:, 1], bb[:, 3]),
                                              np.maximum(bb[:, 0], bb[:, 2]), np.maximum(bb[:, 1], bb[:, 3])], 1)
            self._dirty = True
        return self

    def class_of(self, index: int) -> str:
        inv = {v: k for k, v in self._class_index.items()}
        m = len(self._cls)
        c = self._cls[index] if index < m else int(self._block['cls'][index - m])
        return inv.get(c, '')

    def _all_cls(self) -> np.ndarray:
        a = np.asarray(self._cls, np.uint16)
        return np.concatenate([a, self._block['cls']]) if self._block is not None else a

    # ------------------------------------------------------------- bulk constructors
    @classmethod
    def from_points(cls, xy: Any, classes: Sequence[str] | str | None = None, radius: Any = 0.0,
                    name: str = 'points', props: Sequence[dict] | None = None, **kw) -> 'AnnotationLayer':
        layer = cls(name=name, **kw)
        xy = np.asarray(xy, np.float32).reshape(-1, 2)
        n = len(xy)
        cl = _broadcast(classes, n)
        rad = np.broadcast_to(np.asarray(radius, np.float32), (n,))
        if props is None:   # columnar: no per-point arrays
            layer._bulk_shapes(POINT, np.stack([xy[:, 0] - rad, xy[:, 1] - rad, xy[:, 0] + rad, xy[:, 1] + rad], 1), cl)
        else:
            layer._bulk(POINT, [xy[i:i + 1] for i in range(n)], cl, rad, props)
        return layer

    @classmethod
    def from_polygons(cls, polygons: Sequence[Any], classes: Sequence[str] | str | None = None,
                      name: str = 'objects', props: Sequence[dict] | None = None, **kw) -> 'AnnotationLayer':
        layer = cls(name=name, **kw)
        rings = [_ring(p, closed=True) for p in polygons]
        layer._bulk(POLYGON, rings, _broadcast(classes, len(rings)), None, props)
        return layer

    @classmethod
    def from_boxes(cls, xmin: Any, ymin: Any, xmax: Any, ymax: Any, classes: Sequence[str] | str | None = None,
                   shape: str = 'ellipse', name: str = 'objects', props: Sequence[dict] | None = None,
                   **kw) -> 'AnnotationLayer':
        """Cell/object bounding boxes, e.g. HALO object-data exports (XMin, XMax, YMin, YMax)."""
        layer = cls(name=name, **kw)
        n = len(np.atleast_1d(xmin))
        if props is None:   # columnar, drawn as exact ellipses/rects on the GPU
            bb = np.stack([np.asarray(a, np.float32).reshape(-1) for a in (xmin, ymin, xmax, ymax)], 1)
            layer._bulk_shapes(ELLIPSE if shape == 'ellipse' else RECT, bb, _broadcast(classes, n))
        else:
            layer._bulk(POLYGON, _box_rings(xmin, ymin, xmax, ymax, shape), _broadcast(classes, n), None, props)
        return layer

    @classmethod
    def from_table(cls, table: Any, *, name: str = 'objects', shape: str = 'auto', radius: float = 0.0,
                   x: str | None = None, y: str | None = None, xmin: str | None = None, ymin: str | None = None,
                   xmax: str | None = None, ymax: str | None = None, class_column: str | None = None,
                   columns: str | Sequence[str] | None = 'auto', max_columns: int = 24, **kw) -> 'AnnotationLayer':
        """Object/cell tables (HALO object data, PathCore/QuPath measurements, model outputs).

        ``table``: pandas DataFrame, dict of columns, or list of row dicts. Geometry columns are
        found case-insensitively (``XMin/XMax/YMin/YMax``, ``x/y``, ``Centroid X µm`` …) unless named.
        Remaining columns (up to ``max_columns``) are kept *columnar* as per-object properties, so
        a click on any of 10^6 objects can show its measurements without 10^6 Python dicts.
        """
        cols = _columns_of(table)
        n = len(next(iter(cols.values()))) if cols else 0
        find = lambda given, *cands: given if given is not None else _find_col(cols, cands)  # noqa: E731
        bx0, by0 = find(xmin, 'xmin', 'x_min', 'minx', 'bbox_x0', 'left'), find(ymin, 'ymin', 'y_min', 'miny', 'bbox_y0', 'top')
        bx1, by1 = find(xmax, 'xmax', 'x_max', 'maxx', 'bbox_x1', 'right'), find(ymax, 'ymax', 'y_max', 'maxy', 'bbox_y1', 'bottom')
        cx = find(x, 'x', 'cx', 'centroid_x', 'centroid x', 'x_centroid', 'xcenter', 'center_x', 'centroid x px', 'centroid x µm')
        cy = find(y, 'y', 'cy', 'centroid_y', 'centroid y', 'y_centroid', 'ycenter', 'center_y', 'centroid y px', 'centroid y µm')
        ccol = find(class_column, 'classification', 'class', 'label', 'phenotype', 'cell_type', 'celltype',
                    'cell type', 'type', 'name')
        classes = None if ccol is None else [None if v is None else str(v) for v in np.asarray(cols[ccol]).tolist()]
        have_box = None not in (bx0, by0, bx1, by1)
        if shape == 'auto':
            shape = 'ellipse' if have_box else 'point'
        if have_box and shape in ('ellipse', 'rect'):
            layer = cls.from_boxes(cols[bx0], cols[by0], cols[bx1], cols[by1], classes=classes,
                                   shape=shape, name=name, **kw)
            geo = {bx0, by0, bx1, by1}
        elif cx is not None and cy is not None:
            xy = np.stack([np.asarray(cols[cx], np.float32), np.asarray(cols[cy], np.float32)], 1)
            layer = cls.from_points(xy, classes=classes, radius=radius, name=name, **kw)
            geo = {cx, cy}
        elif have_box:
            x0, y0, x1, y1 = (np.asarray(cols[k], np.float32) for k in (bx0, by0, bx1, by1))
            layer = cls.from_points(np.stack([(x0 + x1) / 2, (y0 + y1) / 2], 1), classes=classes,
                                    radius=np.maximum(x1 - x0, y1 - y0) / 2, name=name, **kw)
            geo = {bx0, by0, bx1, by1}
        else:
            raise ValueError(f'no geometry columns found in {list(cols)[:20]}')
        if columns:
            keep = [c for c in cols if c not in geo] if columns == 'auto' else list(columns)
            layer._columns = {c: np.asarray(cols[c]) for c in keep[:max_columns] if len(cols[c]) == n}
            layer._col_offset = 0
        return layer

    @classmethod
    def parse(cls, data: Any, fmt: str = 'auto', name: str | None = None, **kw) -> 'AnnotationLayer':
        """Annotations as returned by an API: GeoJSON (dict/list/text), HALO XML text, Aperio XML text,
        or an object table. ``fmt``: 'auto' | 'geojson' | 'halo_xml' | 'aperio_xml' | 'table'."""
        if isinstance(data, AnnotationLayer):
            return data
        if fmt == 'table' or (fmt == 'auto' and _looks_tabular(data)):
            return cls.from_table(data, name=name or 'objects', **kw)
        if isinstance(data, (bytes, bytearray)):
            data = bytes(data).decode('utf-8', 'replace')
        if isinstance(data, str):
            t = data.lstrip()
            if fmt == 'auto':
                fmt = 'geojson' if t[:1] in '{[' else ('halo_xml' if ('<V ' in t[:20000] or '<V>' in t[:20000])
                                                       else 'aperio_xml')
        if fmt == 'halo_xml':
            return cls.from_halo_xml(data, name=name, **kw)
        if fmt == 'aperio_xml':
            return cls.from_aperio_xml(data, name=name, **kw)
        return cls.from_geojson(data, name=name, **kw)

    def _bulk(self, kind, rings, classes, radius, props) -> None:
        with self._lock:
            n = len(rings)
            uniq: dict[str, int] = {}
            ids = np.empty(n, np.int64)
            for i, c in enumerate(classes):
                k = c if c is not None else 'default'
                if k not in uniq:
                    uniq[k] = self._class_id(k, _color_hex(k))
                ids[i] = uniq[k]
            self._kinds.extend([kind] * n)
            self._cls.extend(ids.tolist())
            self._radius.extend((radius if radius is not None else np.zeros(n)).tolist())
            self._rings.extend(rings)
            self._feat_nrings.extend([1] * n)
            self._props.extend(props if props is not None else [None] * n)
            self._dirty = True

    @classmethod
    def from_geojson(cls, data: Any, name: str | None = None, **kw) -> 'AnnotationLayer':
        """GeoJSON Feature/FeatureCollection/geometry (QuPath, HALO GeoJSON export, …), or a path."""
        if isinstance(data, (str, Path)) and Path(str(data)).exists():
            p = Path(data)
            name = name or p.stem
            data = _load_json(p)
        elif isinstance(data, (str, bytes)):
            data = json.loads(data)
        layer = cls(name=name or 'geojson', **kw)
        feats = data.get('features') if isinstance(data, dict) and 'features' in data else (
            data if isinstance(data, list) else [data])
        for f in feats:
            geom = f.get('geometry') if f.get('type') == 'Feature' else f
            if not geom:
                continue
            props = f.get('properties') or {}
            cname, color = _class_from_props(props)
            _add_geometry(layer, geom, cname, color, props or None)
        return layer

    @classmethod
    def from_aperio_xml(cls, path: str | Path, name: str | None = None, **kw) -> 'AnnotationLayer':
        """Aperio ImageScope / eSlide Manager XML (path, or XML text/bytes from an API)."""
        root, stem = _xml_root(path)
        layer = cls(name=name or stem, **kw)
        for ann in root.iter('Annotation'):
            aname = ann.get('Name') or ann.get('Id') or 'annotation'
            color = _colorref(ann.get('LineColor'))
            for reg in ann.iter('Region'):
                v = np.array([[float(e.get('X')), float(e.get('Y'))] for e in reg.iter('Vertex')], np.float32)
                if len(v) == 0:
                    continue
                t = reg.get('Type', '0')
                props = {k: reg.get(k) for k in ('Id', 'Text', 'Area', 'Length') if reg.get(k)}
                cname = aname + (' (negative)' if reg.get('NegativeROA') == '1' else '')
                if t == '2' and len(v) >= 2:  # ellipse given by its bounding box corners
                    (x0, y0), (x1, y1) = v.min(0), v.max(0)
                    layer.add_ellipse((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, cls=cname,
                                      color=color, props=props)
                elif t in ('3', '4') or len(v) == 2:
                    layer.add_polyline(v, cls=cname, color=color, props=props)
                elif len(v) == 1:
                    layer.add_point(*v[0], cls=cname, color=color, props=props)
                else:
                    layer.add_polygon(v, cls=cname, color=color, props=props)
        return layer

    @classmethod
    def from_halo_xml(cls, path: str | Path, name: str | None = None, **kw) -> 'AnnotationLayer':
        """HALO ``.annotations`` XML (path, or XML text/bytes as returned by an API)."""
        root, stem = _xml_root(path)
        layer = cls(name=name or stem, **kw)
        for ann in root.iter('Annotation'):
            aname = ann.get('Name') or 'annotation'
            color = _colorref(ann.get('LineColor'))
            for reg in ann.iter('Region'):
                v = np.array([[float(e.get('X')), float(e.get('Y'))] for e in reg.iter('V')], np.float32)
                if len(v) == 0:
                    continue
                t = (reg.get('Type') or 'Polygon').lower()
                cname = aname + (' (negative)' if reg.get('NegativeROA') == '1' else '')
                if t == 'rectangle' and len(v) >= 2:
                    (x0, y0), (x1, y1) = v.min(0), v.max(0)
                    layer.add_rect(x0, y0, x1 - x0, y1 - y0, cls=cname, color=color)
                elif t == 'ellipse' and len(v) >= 2:
                    (x0, y0), (x1, y1) = v.min(0), v.max(0)
                    layer.add_ellipse((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, cls=cname, color=color)
                elif t in ('ruler', 'line', 'polyline') or len(v) == 2:
                    layer.add_polyline(v, cls=cname, color=color)
                elif t in ('pin', 'point') or len(v) == 1:
                    layer.add_point(*v[0], cls=cname, color=color)
                else:
                    layer.add_polygon(v, cls=cname, color=color)
        return layer

    @classmethod
    def load(cls, path: str | Path, **kw) -> 'AnnotationLayer':
        """Load by extension: .geojson/.json, .annotations (HALO), .xml (Aperio or HALO)."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext in ('.geojson', '.json'):
            return cls.from_geojson(p, **kw)
        if ext == '.annotations':
            return cls.from_halo_xml(p, **kw)
        if ext == '.xml':
            head = p.read_text(errors='ignore')[:4000]
            return cls.from_halo_xml(p, **kw) if '<V ' in head or '<V\n' in head else cls.from_aperio_xml(p, **kw)
        raise ValueError(f'unsupported annotation file: {p.name}')

    def to_geojson(self) -> dict:
        c = self.compiled()
        inv = {v: k for k, v in self._class_index.items()}
        feats = []
        for f in range(len(self._kinds)):
            r0, r1 = c.feat_ring_off[f], c.feat_ring_off[f + 1]
            rings = [c.coords[c.ring_off[r]:c.ring_off[r + 1]].tolist() for r in range(r0, r1)]
            k = self._kinds[f]
            if k == POINT:
                g = {'type': 'Point', 'coordinates': rings[0][0]}
            elif k == POLYLINE:
                g = {'type': 'LineString', 'coordinates': rings[0]}
            else:
                g = {'type': 'Polygon', 'coordinates': [r + r[:1] for r in rings]}
            cname = inv.get(self._cls[f], '')
            props = dict(self._props[f] or {})
            props.setdefault('classification', {'name': cname, 'color': self.classes.get(cname)})
            feats.append({'type': 'Feature', 'geometry': g, 'properties': props})
        if self._block is not None:
            b = self._block
            t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            for k, bb, ci in zip(b['kind'].tolist(), b['bb'].tolist(), b['cls'].tolist()):
                x0, y0, x1, y1 = bb
                if k == POINT:
                    g = {'type': 'Point', 'coordinates': [(x0 + x1) / 2, (y0 + y1) / 2]}
                elif k == ELLIPSE:
                    ring = np.stack([(x0 + x1) / 2 + (x1 - x0) / 2 * np.cos(t), (y0 + y1) / 2 + (y1 - y0) / 2 * np.sin(t)], 1).tolist()
                    g = {'type': 'Polygon', 'coordinates': [ring + ring[:1]]}
                else:
                    g = {'type': 'Polygon', 'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
                cname = inv.get(ci, '')
                feats.append({'type': 'Feature', 'geometry': g,
                              'properties': {'classification': {'name': cname, 'color': self.classes.get(cname)}}})
        return {'type': 'FeatureCollection', 'features': feats}

    # ------------------------------------------------------------- continuous colouring
    def color_by(self, values: Any, cmap: str = 'viridis', vmin: float | None = None, vmax: float | None = None,
                 label: str | None = None, log: bool = False, nan_color: str = '#7f7f7f',
                 percentiles: tuple[float, float] = (1.0, 99.0)) -> 'AnnotationLayer':
        """Colour every feature by a number (distance, density, marker intensity, score …).

        ``values``: one number per feature, or the name of a property column. Limits default to the
        1st–99th percentile (diverging maps: symmetric around 0). ``values=None`` → back to class colours.
        """
        with self._lock:
            if values is None:
                self._values, self.colormap = None, None
                self._dirty = True
                return self
            if isinstance(values, str):
                label = label or values
                col = values
                if col in self._columns:
                    v = np.full(len(self), np.nan, np.float64)
                    c = np.asarray(self._columns[col], np.float64)
                    v[self._col_offset:self._col_offset + len(c)] = c
                    values = v
                else:
                    values = np.array([float(q) if isinstance(q := (p or {}).get(col), (int, float)) else np.nan
                                       for p in self._props], np.float64)
            v = np.asarray(values, np.float64).ravel()
            if len(v) != len(self):
                raise ValueError(f'color_by: {len(v)} values for {len(self)} features')
            fin = v[np.isfinite(v)]
            if log:
                fin = np.log10(fin[fin > 0])
            if vmin is None or vmax is None:
                lo, hi = np.percentile(fin, percentiles) if fin.size else (0.0, 1.0)
                if cmap.lower().removesuffix('_r') in _DIVERGING:
                    m = max(abs(lo), abs(hi))
                    lo, hi = -m, m
                vmin = lo if vmin is None else (np.log10(vmin) if log else vmin)
                vmax = hi if vmax is None else (np.log10(vmax) if log else vmax)
            elif log:
                vmin, vmax = np.log10(vmin), np.log10(vmax)
            self._values = v
            self.colormap = {'name': cmap, 'vmin': float(vmin), 'vmax': float(vmax if vmax > vmin else vmin + 1e-9),
                             'label': label or '', 'log': bool(log), 'lut': _lut(cmap, 256), 'nan': nan_color}
            self._dirty = True
        return self

    def _color_index(self) -> np.ndarray:
        """Per-feature index into the colour texture: class id, or (#classes + LUT bin) when colour-by-value."""
        cls = self._all_cls()
        if self._values is None or self.colormap is None:
            return cls
        cm = self.colormap
        v = self._values.copy()
        if cm['log']:
            with np.errstate(divide='ignore', invalid='ignore'):
                v = np.where(v > 0, np.log10(v), np.nan)
        t = (v - cm['vmin']) / (cm['vmax'] - cm['vmin'])
        b = np.where(np.isfinite(t), np.clip(np.floor(np.nan_to_num(t) * 256), 0, 255), 256)  # 256 = NaN colour
        return (len(self._class_index) + b).astype(np.uint16)

    def values(self) -> np.ndarray | None:
        return self._values

    # ------------------------------------------------------------- compile / encode
    def touch(self) -> None:
        """Call after mutating ``classes``/style so viewers refetch."""
        with self._lock:
            self._dirty = True

    def compiled(self) -> '_Compiled':
        with self._lock:
            if self._dirty or self._compiled is None:
                self._compiled = _Compiled(self)
                self._dirty = False
                self.version += 1
            return self._compiled

    def meta(self) -> dict:
        c = self.compiled()
        inv = sorted(self._class_index.items(), key=lambda kv: kv[1])
        return {'name': self.name, 'version': self.version, 'count': int(c.n),
                'pyramid': c.pyr_meta, 'median_size': c.median_size,
                'client_ok': bool(self._block is not None and not self._kinds),
                'buckets': int(len(c.buckets)), 'bbox': c.bbox,
                'classes': [{'name': k, 'color': self.classes.get(k, '#ffffff')} for k, _ in inv],
                'colormap': self.colormap,
                'style': {'line_width': self.line_width, 'fill_opacity': self.fill_opacity,
                          'point_size': self.point_size, 'visible': self.visible,
                          'detail_min_px': self.detail_min_px, 'z': self.z, 'overview': self.overview}}

    def overview_bytes(self) -> bytes:
        return self.compiled().overview(self.meta())

    def bucket_bytes(self, i: int) -> bytes:
        return self.compiled().bucket(i)

    # ------------------------------------------------------------- persistent compiled form
    _SAVE_FORMAT = 2   # bump whenever the compiled/pyramid layout changes (old caches are rebuilt)

    def save(self, path: str | Path) -> Path:
        """Write the layer *and its compiled form* (buckets, object pyramid) as a folder of .npy files. ``open()``
        memory-maps it back in milliseconds — no parsing, no recompiling — and several server processes share one
        copy through the OS page cache. Columnar (bulk object) layers only."""
        if len(self._kinds) or self._block is None:
            raise ValueError('save() supports bulk object layers (from_table / from_boxes / from_points / ObjectTable.to_layer)')
        c = self.compiled()
        d = Path(path)
        tmp = d.with_name(d.name + '.tmp')
        tmp.mkdir(parents=True, exist_ok=True)
        arrs = {'block_kind': self._block['kind'], 'block_bb': self._block['bb'], 'block_cls': self._block['cls'],
                'order': c.order.astype(np.int64), 'buckets': np.asarray(c.buckets, np.float64).reshape(-1, 8)}
        if self._values is not None:
            arrs['values'] = np.asarray(self._values)
        cols = []
        for i, (k, v) in enumerate(self._columns.items()):
            if hasattr(v, 'codes'):
                arrs[f'colcodes_{i}'], arrs[f'colcats_{i}'] = v.codes, np.asarray(v.categories, dtype=str)
                cols.append([k, 'cat'])
            elif np.asarray(v).dtype.kind in 'biuf':
                arrs[f'col_{i}'] = np.asarray(v)
                cols.append([k, 'num'])
        if c.pyr_meta is not None:
            for li, L in enumerate(c._pyr):
                for name in ('x', 'y', 'feat', 'count', 'col', 'orig', 'hist'):
                    if name in L:
                        arrs[f'pyr{li}_{name}'] = L[name]
                ks = sorted(L['index'])
                arrs[f'pyr{li}_tiles'] = np.asarray([[tx, ty, *L['index'][(tx, ty)]] for tx, ty in ks], np.int64).reshape(-1, 4)
            arrs['slot_of'] = c._slot_of
        for k, v in arrs.items():
            np.save(tmp / f'{k}.npy', np.ascontiguousarray(v), allow_pickle=False)
        meta = {'format': self._SAVE_FORMAT, 'name': self.name, 'classes': self.classes,
                'class_index': self._class_index, 'colormap': self.colormap, 'columns': cols,
                'style': {k: getattr(self, k) for k in ('line_width', 'fill_opacity', 'point_size', 'visible',
                                                        'detail_min_px', 'z', 'overview', 'pyramid')},
                'median_size': c.median_size, 'bbox': c.bbox, 'pyr_meta': c.pyr_meta,
                'pyr_levels': [{'k': L['k'], 'span': L['span'], 'raw': L['raw']} for L in c._pyr] if c.pyr_meta else []}
        (tmp / 'meta.json').write_text(json.dumps(meta))
        if d.exists():
            import shutil
            shutil.rmtree(d)
        tmp.rename(d)
        return d

    @classmethod
    def open(cls, path: str | Path, version: int | None = None) -> 'AnnotationLayer':
        """Memory-map a layer written by ``save()``."""
        d = Path(path)
        meta = json.loads((d / 'meta.json').read_text())
        if meta.get('format') != cls._SAVE_FORMAT:
            raise ValueError(f'{d}: unsupported layer format {meta.get("format")}')
        ld = lambda k: np.load(d / f'{k}.npy', mmap_mode='r', allow_pickle=False)  # noqa: E731
        st = meta['style']
        L = cls(name=meta['name'], classes=dict(meta['classes']), **st)
        L._class_index = {k: int(v) for k, v in meta['class_index'].items()}
        L._block = {'kind': ld('block_kind'), 'bb': ld('block_bb'), 'cls': ld('block_cls')}
        L.colormap = meta['colormap']
        if (d / 'values.npy').exists():
            L._values = np.asarray(ld('values'))
        from .objects import _Categorical
        cols = {}
        for i, (k, kind) in enumerate(meta['columns']):
            if kind == 'cat':
                cat = _Categorical.__new__(_Categorical)
                cat.codes, cat.categories, cat.dtype = ld(f'colcodes_{i}'), np.load(d / f'colcats_{i}.npy'), np.dtype(object)
                cols[k] = cat
            else:
                cols[k] = ld(f'col_{i}')
        L._columns = cols
        L._compiled = _Compiled._from_saved(L, meta, ld, d)
        L._dirty = False
        if version is not None:
            L.version = version
        return L

    def pyramid_tile_bytes(self, level: int, tx: int, ty: int) -> bytes:
        return self.compiled().pyramid_tile(level, tx, ty)

    def use_pyramid(self, n: int) -> bool:
        return not LEGACY_OVERLAYS and bool(self.pyramid) and (self.pyramid != 'auto' or n >= PYRAMID_MIN)


PYRAMID_MIN = 50_000   # layers at least this large get an object pyramid (smaller ones ship one overview)
PYR_TILE_BINS = 128    # pyramid tiles are 128 × 128 bins (a 400×250 pane of a 16-grid needs ~100×62)
PYR_SLOTS = 8          # aggregated dots carry counts for the 8 most common classes (+ 'other' folded into slot 7)


class _Compiled:
    """Immutable, bucketed, GPU-ready form of a layer."""

    MAX_FEATS = 6000
    MAX_VERTS = 60000
    pyr_meta: dict | None = None
    median_size: float = 0.0

    def __init__(self, layer: AnnotationLayer) -> None:
        m = len(layer._kinds)                      # classic features (with vertex rings) come first …
        blk = layer._block                         # … then the columnar parametric block (no rings)
        nb = 0 if blk is None else len(blk['kind'])
        n = m + nb
        self.n = n
        self.kinds = np.asarray(layer._kinds, np.uint8)
        self.cls = np.asarray(layer._cls, np.uint16)
        if nb:
            self.kinds = np.concatenate([self.kinds, blk['kind']])
            self.cls = np.concatenate([self.cls, blk['cls']])
        self.col = layer._color_index()  # colour-texture index per feature
        self.radius = np.concatenate([np.asarray(layer._radius, np.float32), np.zeros(nb, np.float32)])
        lens = np.fromiter((len(r) for r in layer._rings), np.int64, len(layer._rings))
        self.ring_off = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=self.ring_off[1:])
        self.coords = (np.concatenate(layer._rings).astype(np.float32) if layer._rings
                       else np.zeros((0, 2), np.float32))
        nr = np.concatenate([np.asarray(layer._feat_nrings, np.int64), np.zeros(nb, np.int64)])
        self.feat_ring_off = np.zeros(n + 1, np.int64)
        np.cumsum(nr, out=self.feat_ring_off[1:])
        self._bucket_cache: OrderedDict[int, bytes] = OrderedDict()
        self._lock = threading.Lock()
        if n == 0:
            self.bb = np.zeros((0, 4), np.float32)
            self.order = np.zeros(0, np.int64)
            self.buckets = []
            self.bbox = [0, 0, 0, 0]
            return
        # per-feature bbox: reduceat over [first vertex of feature i, first vertex of feature i+1)
        if m:
            vs = self.ring_off[self.feat_ring_off[:m]]
            x, y = self.coords[:, 0], self.coords[:, 1]
            bb = np.stack([np.minimum.reduceat(x, vs), np.minimum.reduceat(y, vs),
                           np.maximum.reduceat(x, vs), np.maximum.reduceat(y, vs)], 1).astype(np.float32)
            r = self.radius[:m]
            bb[:, 0] -= r
            bb[:, 1] -= r
            bb[:, 2] += r
            bb[:, 3] += r
        else:
            bb = np.zeros((0, 4), np.float32)
        if nb:
            bb = np.concatenate([bb, blk['bb']])
        self.bb = bb
        self.size = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
        self.cx, self.cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        self.nverts = (self.ring_off[self.feat_ring_off[1:]] - self.ring_off[self.feat_ring_off[:-1]])
        self.bbox = [float(bb[:, 0].min()), float(bb[:, 1].min()), float(bb[:, 2].max()), float(bb[:, 3].max())]
        # Separate size tiers first (cells vs. regions vs. huge ROIs) so each bucket's median
        # size is representative and the level-of-detail switch is per tier.
        groups: list[np.ndarray] = []
        edges = [0, 48, 512, 4096, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = np.nonzero((self.size >= lo) & (self.size < hi))[0]
            if len(idx):
                b = bb[idx]
                self._split(idx, float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()),
                            float(b[:, 3].max()), depth=0, out=groups)
        self.order = np.concatenate(groups)
        self.buckets = []
        start = 0
        for g in groups:
            b = bb[g]
            self.buckets.append((float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max()),
                                 start, len(g), float(np.median(self.size[g])), float(self.size[g].max())))
            start += len(g)
        self.median_size = float(np.median(self.size))
        self._pyr_cache: OrderedDict = OrderedDict()
        if layer.use_pyramid(n):
            self._build_pyramid(layer)

    # ------------------------------------------------------------- object pyramid
    def _build_pyramid(self, layer: AnnotationLayer) -> None:
        """Level 0 = every feature (tiled); level i ≥ 1 = features binned at 2^(k0+i-1) level-0 px, one dot per
        occupied bin with its count, a per-class histogram (class colouring) or mean colour bin (colour-by-value)."""
        n = self.n
        cx, cy = self.cx.astype(np.float64), self.cy.astype(np.float64)
        k0 = int(np.clip(np.ceil(np.log2(max(self.median_size, 1.0))), 0, 24))
        extent = max(float(self.bb[:, 2].max()), float(self.bb[:, 3].max()), 1.0)
        # coarsest: one tile of 2048-px-ish bins (small panes of a 16-grid need very coarse dots)
        kmax = max(k0 + 1, int(np.ceil(np.log2(extent / 64))))
        value_mode = layer.colormap is not None and layer._values is not None
        ncls = len(layer._class_index)
        counts = np.bincount(self.cls.astype(np.int64), minlength=max(ncls, 1))
        top = [int(c) for c in np.argsort(-counts)[:PYR_SLOTS] if counts[c] > 0]
        slot_of = np.full(max(ncls, 1), min(len(top), PYR_SLOTS) - 1 if len(top) >= PYR_SLOTS else 0, np.uint8)
        other = ncls > PYR_SLOTS
        for j, c in enumerate(top):
            slot_of[c] = j
        other_class = None
        if other:  # rarer classes share the last slot ('other'), drawn in the colour of the most common of them
            mask = np.ones(ncls, bool)
            mask[top[:PYR_SLOTS - 1]] = False
            slot_of[mask] = PYR_SLOTS - 1
            top = top[:PYR_SLOTS - 1]
            other_class = int(np.argmax(np.where(mask, counts, -1)))
        self._slot_of = slot_of
        levels = []
        # level 0: raw features, tiled at 2^k0-px bins
        levels.append(self._pyr_tiles(cx, cy, k0, np.arange(n), raw=True))
        # levels 1…: bins of 2^(k0+i) px, each built from the level below (2×2 parents merge), so only the
        # first aggregation touches all n features
        x, y, w = cx, cy, np.ones(n)
        rep = np.arange(n)
        if value_mode:
            nanc = len(layer._class_index) + 256
            col0 = self.col.astype(np.int64)
            fin = col0 != nanc
            vs, vc = np.where(fin, col0, 0).astype(np.float64), fin.astype(np.float64)
        else:
            H = None
            slot = slot_of[self.cls.astype(np.int64)].astype(np.int64)
        tc = np.asarray(top + [other_class if other_class is not None else (top[-1] if top else 0)] * (PYR_SLOTS - len(top)), np.int64)
        for k in range(k0 + 1, kmax + 1):
            s_ = float(2 ** k)
            bx, by = np.floor(x / s_).astype(np.int64), np.floor(y / s_).astype(np.int64)
            key = by * (int(bx.max()) + 2) + bx
            o = np.argsort(key, kind='stable')
            ks = key[o]
            starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
            W = np.add.reduceat(w[o], starts)
            x, y = np.add.reduceat((x * w)[o], starts) / W, np.add.reduceat((y * w)[o], starts) / W
            rep = rep[o][starts]
            agg = {'count': np.minimum(W, 65535).astype(np.uint16), 'orig': rep.astype(np.uint32)}
            if value_mode:
                vs, vc = np.add.reduceat(vs[o], starts), np.add.reduceat(vc[o], starts)
                agg['col'] = np.where(vc > 0, np.round(vs / np.maximum(vc, 1)), nanc).astype(np.uint16)
            else:
                if H is None:
                    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(o)]))
                    H = np.bincount(gid * PYR_SLOTS + slot[o], minlength=len(starts) * PYR_SLOTS).reshape(-1, PYR_SLOTS)
                else:
                    H = np.add.reduceat(H[o], starts, axis=0)
                hmax = H.max(1, keepdims=True)
                agg['hist'] = np.where(H > 0, np.maximum(1, np.round(255 * H / np.maximum(hmax, 1))), 0).astype(np.uint8)
                agg['col'] = tc[H.argmax(1)].astype(np.uint16)
            w = W
            levels.append(self._pyr_tiles(x, y, k, agg, raw=False))
        self._pyr = levels
        self.pyr_meta = {'k0': k0, 'kmax': kmax, 'levels': len(levels), 'tile_bins': PYR_TILE_BINS,
                         'mode': 'value' if value_mode else 'class',
                         'slots': [int(c) for c in top] + ([-1] if other else []), 'other_class': other_class,
                         'tiles': [len(L['index']) for L in levels], 'points': [int(len(L['x'])) for L in levels]}

    @classmethod
    def _from_saved(cls, layer: AnnotationLayer, meta: dict, ld, d: Path) -> '_Compiled':
        """Rebuild the compiled form of a saved bulk layer without recomputing buckets or the pyramid."""
        c = cls.__new__(cls)
        b = layer._block
        n = len(b['kind'])
        c.n = n
        c.kinds, c.cls = b['kind'], b['cls']
        c.col = layer._color_index()
        c.radius = np.zeros(n, np.float32)
        c.ring_off = np.zeros(1, np.int64)
        c.coords = np.zeros((0, 2), np.float32)
        c.feat_ring_off = np.zeros(n + 1, np.int64)
        c.bb = b['bb']
        c.size = np.maximum(c.bb[:, 2] - c.bb[:, 0], c.bb[:, 3] - c.bb[:, 1])
        c.cx, c.cy = (c.bb[:, 0] + c.bb[:, 2]) / 2, (c.bb[:, 1] + c.bb[:, 3]) / 2
        c.nverts = np.zeros(n, np.int64)
        c.bbox = meta['bbox']
        c.order = ld('order')
        c.buckets = [tuple(float(v) for v in row) for row in ld('buckets')]
        c.median_size = meta['median_size']
        c._bucket_cache = OrderedDict()
        c._pyr_cache = OrderedDict()
        c._lock = threading.Lock()
        c.pyr_meta = meta['pyr_meta']
        if c.pyr_meta:
            c._slot_of = np.asarray(ld('slot_of'))
            c._pyr = []
            for li, lm in enumerate(meta['pyr_levels']):
                L = dict(lm)
                for name in ('x', 'y', 'feat', 'count', 'col', 'orig', 'hist'):
                    if (d / f'pyr{li}_{name}.npy').exists():
                        L[name] = ld(f'pyr{li}_{name}')
                t = np.load(d / f'pyr{li}_tiles.npy')
                L['index'] = {(int(a), int(b_)): (int(s_), int(m)) for a, b_, s_, m in t}
                c._pyr.append(L)
        return c

    def _pyr_tiles(self, x, y, k, data, raw: bool) -> dict:
        """Sort points of one level by tile; index (tx, ty) → (start, count)."""
        span = float(2 ** k) * PYR_TILE_BINS
        tx, ty = np.floor(x / span).astype(np.int64), np.floor(y / span).astype(np.int64)
        key = ty * (int(tx.max()) + 2) + tx
        o = np.argsort(key, kind='stable')
        ks = key[o]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if len(ks) else np.zeros(0, np.int64)
        ends = np.r_[starts[1:], len(ks)]
        index = {(int(tx[o[a]]), int(ty[o[a]])): (int(a), int(b - a)) for a, b in zip(starts, ends)}
        L = {'k': k, 'span': span, 'raw': raw, 'x': x[o].astype(np.float32), 'y': y[o].astype(np.float32), 'index': index}
        if raw:
            L['feat'] = data[o]
        else:
            for name, arr in data.items():
                L[name] = arr[o]
        return L

    def columns_payload(self, layer: AnnotationLayer) -> bytes:
        """Client-side overlay mode: every object as centre (f32) + radii (u16, ¼ px) + colour/class/kind, ~17 bytes
        each — the browser builds its own pyramid in a worker and never asks again for this layer."""
        nc = len(layer._class_index)
        rx = np.clip(np.round((self.bb[:, 2] - self.bb[:, 0]) * 2), 0, 65535).astype('<u2')   # ¼ px units of the radius
        ry = np.clip(np.round((self.bb[:, 3] - self.bb[:, 1]) * 2), 0, 65535).astype('<u2')
        meta = {'n': int(self.n), 'nclasses': nc, 'mode': 'value' if (layer.colormap is not None and layer._values is not None) else 'class',
                'nan': nc + 256, 'median_size': self.median_size, 'bbox': self.bbox, 'tile_bins': PYR_TILE_BINS, 'slots': PYR_SLOTS}
        return _pack(meta, [self.cx.astype('<f4'), self.cy.astype('<f4'), rx, ry, self.col.astype('<u2'),
                            self.cls.astype('<u2'), self.kinds.astype('u1')])

    def pyramid_tile(self, level: int, tx: int, ty: int) -> bytes:
        key = (level, tx, ty)
        with self._lock:
            if key in self._pyr_cache:
                self._pyr_cache.move_to_end(key)
                return self._pyr_cache[key]
        L = self._pyr[level]
        a, m = L['index'].get((tx, ty), (0, 0))
        sl = slice(a, a + m)
        x0, y0, span = tx * L['span'], ty * L['span'], L['span']
        pos = np.stack([np.clip(np.round((L['x'][sl] - x0) / span * 65535), 0, 65535),
                        np.clip(np.round((L['y'][sl] - y0) / span * 65535), 0, 65535)], 1).astype('<u2')
        meta = {'n': int(m), 'level': level, 'x0': x0, 'y0': y0, 'span': span, 'raw': L['raw'], 'k': L['k']}
        if L['raw']:
            f = L['feat'][sl]
            arrays = [pos, np.minimum(self.size[f], 65535).astype('<u2'), self.col[f].astype('<u2'), self.cls[f].astype('<u2'),
                      f.astype('<u4'), self.kinds[f].astype('u1')]
        else:
            arrays = [pos, L['count'][sl].astype('<u2'), L['col'][sl].astype('<u2'), L['orig'][sl].astype('<u4')]
            if 'hist' in L:
                arrays.append(np.ascontiguousarray(L['hist'][sl]).astype('u1'))
                meta['hist'] = True
        data = _pack(meta, arrays)
        with self._lock:
            self._pyr_cache[key] = data
            while len(self._pyr_cache) > 4096:
                self._pyr_cache.popitem(last=False)
        return data

    def _split(self, idx, x0, y0, x1, y1, depth, out) -> None:
        if (len(idx) <= self.MAX_FEATS and self.nverts[idx].sum() <= self.MAX_VERTS) or depth >= 14 or len(idx) <= 1:
            if len(idx):
                out.append(idx)
            return
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        cx, cy = self.cx[idx], self.cy[idx]
        right, bottom = cx >= mx, cy >= my
        for sel, rect in (((~right) & (~bottom), (x0, y0, mx, my)), (right & (~bottom), (mx, y0, x1, my)),
                          ((~right) & bottom, (x0, my, mx, y1)), (right & bottom, (mx, my, x1, y1))):
            self._split(idx[sel], *rect, depth=depth + 1, out=out)

    def overview(self, meta: dict) -> bytes:
        if self.pyr_meta is not None:  # pyramid layers: bucket table + which pyramid tiles exist (points come as tiles)
            bt = np.asarray(self.buckets, '<f4').reshape(-1, 8)
            tiles = [np.asarray(sorted(L['index']), '<i4').reshape(-1, 2) for L in self._pyr]
            return _pack(dict(meta, buckets=len(self.buckets), index_only=True), [bt] + tiles)
        n, o = self.n, self.order
        cen = np.stack([self.cx[o], self.cy[o]], 1).astype('<f4') if n else np.zeros((0, 2), '<f4')
        size = (self.size[o] if n else np.zeros(0)).astype('<f4')
        cls = self.cls[o].astype('<u2')
        col = self.col[o].astype('<u2')
        kinds = self.kinds[o].astype('u1')
        orig = o.astype('<u4')
        bt = np.asarray(self.buckets, '<f4').reshape(-1, 8)
        meta = dict(meta, buckets=len(self.buckets))
        return _pack(meta, [cen, size, orig, cls, col, kinds, bt])

    def bucket(self, i: int) -> bytes:
        with self._lock:
            if i in self._bucket_cache:
                self._bucket_cache.move_to_end(i)
                return self._bucket_cache[i]
        b = self.buckets[i]
        feats = self.order[int(b[4]):int(b[4]) + int(b[5])]
        data = self._encode_detail(feats)
        with self._lock:
            self._bucket_cache[i] = data
            while len(self._bucket_cache) > 2048:
                self._bucket_cache.popitem(last=False)
        return data

    def _encode_detail(self, feats: np.ndarray) -> bytes:
        k = self.kinds[feats]
        lines = feats[(k == POLYLINE) | (k == POLYGON)]
        # rings of those features
        rs, re_ = self.feat_ring_off[lines], self.feat_ring_off[lines + 1]
        nr = re_ - rs
        ring_ids = np.repeat(rs, nr) + (np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr))
        ring_feat = np.repeat(lines, nr)
        ring_closed = self.kinds[ring_feat] == POLYGON
        a_, b_ = self.ring_off[ring_ids], self.ring_off[ring_ids + 1]
        cnt = b_ - a_
        tot = int(cnt.sum())
        base = np.repeat(a_, cnt)
        local = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ncnt = np.repeat(cnt, cnt)
        nxt = base + (local + 1) % np.maximum(ncnt, 1)
        keep = np.repeat(ring_closed, cnt) | (local < ncnt - 1)
        keep &= ncnt > 1
        sa, sb = (base + local)[keep], nxt[keep]
        seg = np.concatenate([self.coords[sa], self.coords[sb]], 1).astype('<f4')
        seg_cls = self.col[np.repeat(ring_feat, cnt)[keep]].astype('<u2')
        # fills
        tris, tri_cls = [], []
        polys = feats[k == POLYGON]
        if len(polys):
            try:
                import mapbox_earcut as earcut
            except ImportError:  # fills are optional
                earcut = None
            if earcut is not None:
                for f in polys:
                    r0, r1 = self.feat_ring_off[f], self.feat_ring_off[f + 1]
                    v0, v1 = self.ring_off[r0], self.ring_off[r1]
                    verts = self.coords[v0:v1]
                    if len(verts) < 3:
                        continue
                    ends = (self.ring_off[r0 + 1:r1 + 1] - v0).astype(np.uint32)
                    idx = earcut.triangulate_float32(verts, ends)
                    if len(idx):
                        tris.append(verts[idx])
                        tri_cls.append(np.full(len(idx), self.col[f], '<u2'))
        tv = np.concatenate(tris).astype('<f4') if tris else np.zeros((0, 2), '<f4')
        tc = np.concatenate(tri_cls) if tri_cls else np.zeros(0, '<u2')
        # parametric shapes: centre + radii, drawn analytically (outline + fill) by the GPU — 19 bytes per cell
        sf = feats[(k == ELLIPSE) | (k == RECT)]
        b = self.bb[sf]
        shp = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, (b[:, 2] - b[:, 0]) / 2, (b[:, 3] - b[:, 1]) / 2], 1).astype('<f4')
        return _pack({'segments': int(len(seg)), 'tri_verts': int(len(tv)), 'shapes': int(len(sf))},
                     [seg, tv, seg_cls, tc, shp, self.col[sf].astype('<u2'), self.kinds[sf].astype('u1')])

    def hit_test(self, x: float, y: float, tol: float, hidden: Iterable[int] = ()) -> int | None:
        """Index of the smallest feature whose bbox contains (x, y) (± tol), or None. Only buckets around the
        point are scanned; features of ``hidden`` class ids are skipped."""
        if not self.n:
            return None
        parts = [self.order[int(b[4]):int(b[4]) + int(b[5])] for b in self.buckets
                 if b[0] - tol <= x <= b[2] + tol and b[1] - tol <= y <= b[3] + tol]
        if not parts:
            return None
        idx = np.concatenate(parts)
        bb = self.bb[idx]
        m = (bb[:, 0] - tol <= x) & (x <= bb[:, 2] + tol) & (bb[:, 1] - tol <= y) & (y <= bb[:, 3] + tol)
        hidden = list(hidden)
        if hidden:
            m &= ~np.isin(self.cls[idx], np.asarray(hidden, np.uint16))
        cand = idx[np.nonzero(m)[0]]
        if not len(cand):
            return None
        return int(cand[np.argmin(self.size[cand])])


def _columns_of(table: Any) -> dict[str, np.ndarray]:
    if hasattr(table, 'columns') and hasattr(table, 'to_numpy'):  # pandas / polars-like
        return {str(c): np.asarray(table[c]) for c in table.columns}
    if isinstance(table, dict):
        return {str(k): np.asarray(v) for k, v in table.items()}
    rows = list(table)
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {str(k): np.asarray([r.get(k) for r in rows]) for k in keys}


def _norm_col(c: str) -> str:
    return ' '.join(str(c).lower().replace('_', ' ').replace('(', ' ').replace(')', ' ').split())


def _find_col(cols: dict, cands: Sequence[str]) -> str | None:
    norm = {_norm_col(c): c for c in cols}
    for cand in cands:
        k = _norm_col(cand)
        if k in norm:
            return norm[k]
    return None


def _looks_tabular(data: Any) -> bool:
    if hasattr(data, 'columns') and hasattr(data, 'to_numpy'):
        return True
    if isinstance(data, dict) and 'type' not in data and data and all(
            isinstance(v, (list, tuple, np.ndarray)) for v in data.values()):
        return True
    return False


def _box_rings(xmin, ymin, xmax, ymax, shape: str = 'ellipse') -> list[np.ndarray]:
    x0, y0, x1, y1 = (np.asarray(a, np.float32) for a in (xmin, ymin, xmax, ymax))
    if shape == 'ellipse':
        t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        xs = cx[:, None] + rx[:, None] * np.cos(t)[None]
        ys = cy[:, None] + ry[:, None] * np.sin(t)[None]
    else:
        xs = np.stack([x0, x1, x1, x0], 1)
        ys = np.stack([y0, y0, y1, y1], 1)
    return list(np.stack([xs, ys], 2).astype(np.float32))


def _xml_root(src: Any) -> tuple[ET.Element, str]:
    if isinstance(src, bytes) or (isinstance(src, str) and src.lstrip().startswith('<')):
        return ET.fromstring(src), 'annotations'
    return ET.parse(src).getroot(), Path(src).stem


def _pack(meta: dict, arrays: list[np.ndarray]) -> bytes:
    """MAGIC | u32 json_len | json (padded to 4) | arrays, each padded to 4 bytes."""
    j = json.dumps(meta, separators=(',', ':')).encode()
    j += b' ' * (-len(j) % 4)
    parts = [MAGIC, struct.pack('<I', len(j)), j]
    for a in arrays:
        b = np.ascontiguousarray(a).tobytes()
        parts.append(b + b'\0' * (-len(b) % 4))
    return b''.join(parts)


# ---------------------------------------------------------------- helpers
def _ring(coords: Any, closed: bool) -> np.ndarray:
    a = np.asarray(coords, np.float32).reshape(-1, 2)
    if len(a) == 0:
        raise ValueError('empty geometry')
    if closed and len(a) > 2 and np.array_equal(a[0], a[-1]):
        a = a[:-1]
    return a


def _broadcast(v, n):
    if v is None or isinstance(v, str):
        return [v] * n
    v = list(v)
    if len(v) != n:
        raise ValueError('classes length mismatch')
    return [None if c is None else str(c) for c in v]


def _colorref(v: str | None) -> str | None:
    """Windows COLORREF (0x00BBGGRR as decimal), used by Aperio and HALO XML."""
    if v is None or v == '':
        return None
    try:
        n = int(v) & 0xFFFFFF
    except ValueError:
        return _color_hex(v)
    r, g, b = n & 255, (n >> 8) & 255, (n >> 16) & 255
    return '#%02x%02x%02x' % (r, g, b)


def _class_from_props(props: dict) -> tuple[str | None, str | None]:
    c = props.get('classification')
    if isinstance(c, dict):
        return c.get('name'), _color_hex(c.get('color') or c.get('colorRGB'))
    for key in ('class', 'classification', 'label', 'Class', 'Classification', 'name', 'Name', 'type'):
        if isinstance(props.get(key), str):
            return props[key], _color_hex(props.get('color'))
    return None, _color_hex(props.get('color'))


def _add_geometry(layer: AnnotationLayer, g: dict, cname, color, props) -> None:
    t, c = g.get('type'), g.get('coordinates')
    if t == 'Polygon' and c:
        layer.add_polygon(c[0], holes=c[1:], cls=cname, color=color, props=props)
    elif t == 'MultiPolygon':
        for poly in c:
            if poly:
                layer.add_polygon(poly[0], holes=poly[1:], cls=cname, color=color, props=props)
    elif t == 'LineString':
        layer.add_polyline(c, cls=cname, color=color, props=props)
    elif t == 'MultiLineString':
        for ln in c:
            layer.add_polyline(ln, cls=cname, color=color, props=props)
    elif t == 'Point':
        layer.add_point(c[0], c[1], cls=cname, color=color, props=props)
    elif t == 'MultiPoint':
        for p in c:
            layer.add_point(p[0], p[1], cls=cname, color=color, props=props)
    elif t == 'GeometryCollection':
        for sub in g.get('geometries', []):
            _add_geometry(layer, sub, cname, color, props)


def _load_json(p: Path):
    return json.loads(p.read_bytes())


def discover_sidecars(slide_path: Path) -> list[AnnotationLayer]:
    """Annotation files next to a slide with the same stem: .geojson/.json, .annotations, .xml."""
    out = []
    base = slide_path.with_suffix('')
    if base.suffix.lower() == '.ome':
        base = base.with_suffix('')
    for ext in ('.geojson', '.json', '.annotations', '.xml'):
        p = base.with_name(base.name + ext)
        if p.exists():
            try:
                out.append(AnnotationLayer.load(p, name=p.name))
            except Exception:
                pass
    return out


def from_czi_metadata(meta: dict, offset=(0, 0)) -> AnnotationLayer | None:
    """Best-effort import of graphics layers embedded in CZI metadata (rectangles, circles,
    ellipses, polygons, polylines, lines, points)."""
    md = (meta or {}).get('ImageDocument', {}).get('Metadata', {})
    layers = md.get('Layers', {}) or {}
    lay = layers.get('Layer') if isinstance(layers, dict) else None
    if not lay:
        return None
    lay = lay if isinstance(lay, list) else [lay]
    ox, oy = offset
    out = AnnotationLayer(name='CZI graphics')

    def f(v, k, d=0.0):
        try:
            return float(v.get(k, d))
        except (TypeError, ValueError, AttributeError):
            return d

    def pts(s):
        a = np.array([[float(t) for t in p.split(',')] for p in str(s).split() if ',' in p], np.float32)
        return a + [ox, oy] if len(a) else a

    for L in lay:
        lname = (L.get('@Name') or L.get('Name') or 'layer') if isinstance(L, dict) else 'layer'
        el = (L or {}).get('Elements') or {}
        for kind, items in el.items():
            for it in (items if isinstance(items, list) else [items]):
                if not isinstance(it, dict):
                    continue
                g = it.get('Geometry') or {}
                color = _color_hex(('#' + str((it.get('Attributes') or {}).get('Stroke', ''))[-6:])
                                   if (it.get('Attributes') or {}).get('Stroke') else None)
                k = kind.lower()
                if k == 'rectangle':
                    out.add_rect(f(g, 'Left') + ox, f(g, 'Top') + oy, f(g, 'Width'), f(g, 'Height'), cls=lname, color=color)
                elif k == 'circle':
                    r = f(g, 'Radius')
                    out.add_ellipse(f(g, 'CenterX') + ox, f(g, 'CenterY') + oy, r, r, cls=lname, color=color)
                elif k == 'ellipse':
                    out.add_ellipse(f(g, 'CenterX') + ox, f(g, 'CenterY') + oy, f(g, 'RadiusX'), f(g, 'RadiusY'),
                                    cls=lname, color=color)
                elif k in ('polygon', 'closedpolyline', 'closedbezier'):
                    p = pts(g.get('Points', ''))
                    if len(p) > 2:
                        out.add_polygon(p, cls=lname, color=color)
                elif k in ('polyline', 'openpolyline', 'bezier', 'openbezier'):
                    p = pts(g.get('Points', ''))
                    if len(p) > 1:
                        out.add_polyline(p, cls=lname, color=color)
                elif k in ('line', 'arrow', 'scalebar'):
                    out.add_polyline([[f(g, 'X1') + ox, f(g, 'Y1') + oy], [f(g, 'X2') + ox, f(g, 'Y2') + oy]],
                                     cls=lname, color=color)
                elif k in ('marker', 'point', 'events'):
                    out.add_point(f(g, 'X', f(g, 'CenterX')) + ox, f(g, 'Y', f(g, 'CenterY')) + oy, cls=lname, color=color)
    return out if len(out) else None
