"""Annotation files: GeoParquet (the efficient default) and conversions to and from other tools' formats.

    from wissi_viz.annio import export, load
    export(layer, 'cells.parquet')                  # GeoParquet 1.1 (WKB + bbox covering, zstd)
    export(layer, 'cells.geojson', fmt='qupath')    # QuPath-style GeoJSON
    layer = load('cells.parquet', bbox=(x0, y0, x1, y1))   # only the features in a region

    wissi-viz convert in.geojson out.parquet        # any → any, by extension (or --to)

Formats (read / write):

| fmt | extensions | read | write | notes |
|---|---|---|---|---|
| ``geoparquet`` | .parquet .geoparquet | ✔ | ✔ | needs pyarrow (``[io]``); opens in GeoPandas, QGIS (GDAL ≥ 3.8), DuckDB |
| ``geojson`` / ``qupath`` | .geojson .json | ✔ | ✔ | ``qupath`` adds ``objectType`` and RGB class colours |
| ``halo`` | .annotations .xml | ✔ | ✔ | HALO annotation XML |
| ``aperio`` | .xml | ✔ | ✔ | Aperio ImageScope XML |
| ``asap`` | .xml | ✔ | | ASAP XML (CAMELYON and other challenge datasets) |
| ``csv`` | .csv .tsv | ✔ (object tables) | ✔ | one row per feature: class, centroid, bbox, area, properties (+ WKT) |
| ``coco`` | .coco.json | ✔ | ✔ | COCO instance segmentation (polygons) |
| ``mask`` | .tif .tiff | ✔ (label masks) | ✔ | label image (feature ids or class ids) at a chosen downsample |

Coordinates are level-0 pixels everywhere (y down). GeoParquet stores them in a "crs": null (engineering) space and
records the slide's µm/px, the layer's classes and colours and its style in file metadata, so a round trip is
lossless for geometry, classes and properties.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from .annotations import ELLIPSE, POINT, POLYGON, POLYLINE, RECT, AnnotationLayer, _auto_color
from .objects import from_label_mask, read_objects
from .sources import open_slide

GEO_POINT, GEO_LINE, GEO_POLYGON = 1, 2, 3  # WKB / OGC geometry type codes
_ELLIPSE_VERTS = 32
FORMATS = ('geoparquet', 'geojson', 'qupath', 'halo', 'aperio', 'csv', 'coco', 'mask')


# ------------------------------------------------------------------------------------------------ canonical form
@dataclass
class Features:
    """Features as flat arrays: ``kind`` (GEO_POINT / GEO_LINE / GEO_POLYGON) per feature; rings of feature *i* are
    ``feat_ring_off[i]:feat_ring_off[i+1]`` (polygons: shell first, then holes; rings are *not* closed); vertices of
    ring *r* are ``coords[ring_off[r]:ring_off[r+1]]``."""

    kind: np.ndarray
    coords: np.ndarray
    ring_off: np.ndarray
    feat_ring_off: np.ndarray
    cls: np.ndarray  # int32 index into class_names (-1: none)
    class_names: list
    class_colors: dict = field(default_factory=dict)
    props: dict = field(default_factory=dict)  # name → array (len n); object arrays may hold None
    name: str = 'annotations'
    style: dict = field(default_factory=dict)
    mpp: float | None = None

    def __len__(self) -> int:
        return len(self.kind)

    # per-feature geometry summaries (vectorised)
    def bbox(self) -> np.ndarray:
        n = len(self)
        if n == 0:
            return np.zeros((0, 4))
        starts = self.ring_off[self.feat_ring_off[:-1]]
        x, y = self.coords[:, 0], self.coords[:, 1]
        return np.stack(
            [
                np.minimum.reduceat(x, starts),
                np.minimum.reduceat(y, starts),
                np.maximum.reduceat(x, starts),
                np.maximum.reduceat(y, starts),
            ],
            1,
        )

    def shell_area(self) -> np.ndarray:
        """Signed-area-free polygon area (shell minus holes) in px²; 0 for points and lines."""
        out = np.zeros(len(self))
        if not len(self.coords):
            return out
        x, y = self.coords[:, 0], self.coords[:, 1]
        # shoelace per ring: sum x_i*y_{i+1} - x_{i+1}*y_i with wrap-around inside each ring
        nxt = np.arange(len(x)) + 1
        ends = self.ring_off[1:] - 1
        nxt[ends] = self.ring_off[:-1]
        cross = x * y[nxt] - x[nxt] * y
        ring_area = np.abs(np.add.reduceat(cross, self.ring_off[:-1])) / 2 if len(self.ring_off) > 1 else np.zeros(0)
        ring_area[np.diff(self.ring_off) < 3] = 0
        poly = np.nonzero(self.kind == GEO_POLYGON)[0]
        for f in poly:  # shell − holes (holes are rare; loop only over polygons)
            r0, r1 = self.feat_ring_off[f], self.feat_ring_off[f + 1]
            out[f] = ring_area[r0] - ring_area[r0 + 1 : r1].sum()
        return out

    def centroid(self) -> np.ndarray:
        b = self.bbox()
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1) if len(b) else np.zeros((0, 2))

    def class_of(self) -> np.ndarray:
        names = np.asarray(list(self.class_names) + [''], dtype=object)
        return names[np.where(self.cls < 0, len(self.class_names), self.cls)]

    def rings(self, f: int) -> list:
        return [
            self.coords[self.ring_off[r] : self.ring_off[r + 1]]
            for r in range(self.feat_ring_off[f], self.feat_ring_off[f + 1])
        ]


def from_layer(layer: AnnotationLayer, ellipse_vertices: int = _ELLIPSE_VERTS) -> Features:
    """Flatten a layer (classic per-feature storage, the columnar bulk block, or a memory-mapped saved layer)."""
    inv = {v: k for k, v in layer._class_index.items()}
    class_names = [inv.get(i, f'class {i}') for i in range(len(inv))]
    m = len(layer._kinds)
    parts = []  # (kind, coords, ring_off, feat_ring_off, cls)
    if m:
        c = layer.compiled()
        lk = np.asarray(layer._kinds, np.uint8)
        kind = np.where(lk == POINT, GEO_POINT, np.where(lk == POLYLINE, GEO_LINE, GEO_POLYGON)).astype(np.uint8)
        nrings = int(c.feat_ring_off[m])
        parts.append(
            (
                kind,
                np.asarray(c.coords[: c.ring_off[nrings]], np.float64),
                np.asarray(c.ring_off[: nrings + 1], np.int64),
                np.asarray(c.feat_ring_off[: m + 1], np.int64),
                np.asarray(layer._cls, np.int32),
            )
        )
    b = layer._block
    if b is not None and len(b['kind']):
        bk = np.asarray(b['kind'])
        bb = np.asarray(b['bb'], np.float64)
        cls = np.asarray(b['cls'], np.int32)
        nb = len(bk)
        nv = np.where(bk == POINT, 1, np.where(bk == RECT, 4, ellipse_vertices))
        ring_off = np.zeros(nb + 1, np.int64)
        np.cumsum(nv, out=ring_off[1:])
        coords = np.empty((ring_off[-1], 2))
        cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        rx, ry = (bb[:, 2] - bb[:, 0]) / 2, (bb[:, 3] - bb[:, 1]) / 2
        for k in (POINT, RECT, ELLIPSE):
            sel = np.nonzero(bk == k)[0]
            if not len(sel):
                continue
            start = ring_off[sel]
            if k == POINT:
                coords[start] = np.stack([cx[sel], cy[sel]], 1)
            elif k == RECT:
                q = np.stack(
                    [
                        np.stack([bb[sel, 0], bb[sel, 1]], 1),
                        np.stack([bb[sel, 2], bb[sel, 1]], 1),
                        np.stack([bb[sel, 2], bb[sel, 3]], 1),
                        np.stack([bb[sel, 0], bb[sel, 3]], 1),
                    ],
                    1,
                )
                coords[(start[:, None] + np.arange(4)).ravel()] = q.reshape(-1, 2)
            else:
                t = np.linspace(0, 2 * np.pi, ellipse_vertices, endpoint=False)
                e = np.stack([cx[sel, None] + rx[sel, None] * np.cos(t), cy[sel, None] + ry[sel, None] * np.sin(t)], 2)
                coords[(start[:, None] + np.arange(ellipse_vertices)).ravel()] = e.reshape(-1, 2)
        kind = np.where(bk == POINT, GEO_POINT, GEO_POLYGON).astype(np.uint8)
        parts.append((kind, coords, ring_off, np.arange(nb + 1, dtype=np.int64), cls))
    if not parts:
        F = Features(
            np.zeros(0, np.uint8),
            np.zeros((0, 2)),
            np.zeros(1, np.int64),
            np.zeros(1, np.int64),
            np.zeros(0, np.int32),
            class_names,
        )
    elif len(parts) == 1:
        F = Features(*parts[0], class_names)
    else:  # classic features first, then the block (the layer's own order)
        (k1, c1, r1, f1, s1), (k2, c2, r2, f2, s2) = parts
        F = Features(
            np.concatenate([k1, k2]),
            np.concatenate([c1, c2]),
            np.concatenate([r1, r2[1:] + r1[-1]]),
            np.concatenate([f1, f2[1:] + f1[-1]]),
            np.concatenate([s1, s2]),
            class_names,
        )
    F.class_colors = dict(layer.classes)
    F.name = layer.name
    F.style = {
        k: getattr(layer, k)
        for k in ('line_width', 'fill_opacity', 'point_size', 'visible', 'detail_min_px', 'z', 'origin')
    }
    F.props = _layer_props(layer, len(F))
    F.mpp = F.mpp or layer.mpp
    return F


def _layer_props(layer: AnnotationLayer, n: int) -> dict:
    """Per-feature properties as columns: columnar block columns as they are, sparse dict properties gathered."""
    cols: dict[str, Any] = {}
    off = layer._col_offset
    for k, v in (layer._columns or {}).items():
        if hasattr(v, 'codes'):  # categorical: decode once
            v = np.asarray(v.categories, dtype=object)[np.asarray(v.codes)]
        v = np.asarray(v)
        if off == 0 and len(v) == n:
            cols[k] = v
        else:
            full = np.full(n, None, dtype=object) if v.dtype.kind in 'OUS' else np.full(n, np.nan)
            full[off : off + len(v)] = v
            cols[k] = full
    dict_props = [p for p in layer._props if p]
    if dict_props:
        keys: dict[str, None] = {}
        for p in dict_props:
            for k in p:
                if k not in ('classification', 'objectType', 'isLocked') and k not in cols:
                    keys[k] = None
        for k in keys:
            arr = np.full(n, None, dtype=object)
            for i, p in enumerate(layer._props):
                if p and k in p:
                    v = p[k]
                    arr[i] = json.dumps(v) if isinstance(v, (dict, list)) else v
            cols[k] = _tighten(arr)
    return cols


def _tighten(arr: np.ndarray) -> np.ndarray:
    """object column → float64 if every value is numeric (None → NaN), else keep."""
    vals = [v for v in arr if v is not None]
    if vals and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in vals):
        return np.array([np.nan if v is None else float(v) for v in arr])
    return arr


def to_layer(F: Features, name: str | None = None) -> AnnotationLayer:
    """Features → AnnotationLayer. Point-only sets become a columnar block (fast for millions of cells)."""
    layer = AnnotationLayer(
        name=name or F.name, classes={c: F.class_colors.get(c) or _hexcol(i) for i, c in enumerate(F.class_names)}
    )
    for k, v in (F.style or {}).items():
        if hasattr(layer, k) and v is not None:
            setattr(layer, k, v)
    layer.mpp = float(F.mpp) if F.mpp else None
    names = F.class_of()
    cls = [None if c == '' else c for c in names.tolist()]
    n = len(F)
    if n == 0:
        return layer
    if (F.kind == GEO_POINT).all():
        xy = F.coords[F.ring_off[F.feat_ring_off[:-1]]]
        bb = np.concatenate([xy, xy], 1)
        layer._bulk_shapes(POINT, bb, cls)
    else:
        for kind_geo, kind_layer in ((GEO_POINT, POINT), (GEO_LINE, POLYLINE), (GEO_POLYGON, POLYGON)):
            sel = np.nonzero(F.kind == kind_geo)[0]
            if not len(sel):
                continue
            single = (np.diff(F.feat_ring_off)[sel] == 1).all()
            if single:
                rings = [
                    F.coords[F.ring_off[F.feat_ring_off[f]] : F.ring_off[F.feat_ring_off[f] + 1]].astype(np.float32)
                    for f in sel
                ]
                layer._bulk(kind_layer, rings, [cls[f] for f in sel], None, None)
            else:
                for f in sel:
                    rs = F.rings(int(f))
                    if kind_layer == POLYGON:
                        layer.add_polygon(rs[0], holes=rs[1:], cls=cls[f])
                    elif kind_layer == POLYLINE:
                        layer.add_polyline(rs[0], cls=cls[f])
                    else:
                        layer.add_point(*rs[0][0], cls=cls[f])
        # features were added grouped by kind: reorder the property columns the same way
        order = np.concatenate([np.nonzero(F.kind == k)[0] for k in (GEO_POINT, GEO_LINE, GEO_POLYGON)])
        if not np.array_equal(order, np.arange(n)):
            F = _take_props(F, order)
    if F.props:
        layer._columns = {k: np.asarray(v) for k, v in F.props.items()}
        layer._col_offset = 0
    return layer


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    lens = np.asarray(lens, np.int64)
    if not len(lens) or lens.sum() == 0:
        return np.zeros(0, np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return np.repeat(np.asarray(starts, np.int64) - offs, lens) + np.arange(int(lens.sum()))


def subset(F: Features, idx: np.ndarray) -> Features:
    """The features ``idx`` of F (geometry, classes, properties), vectorised."""
    idx = np.asarray(idx, np.int64)
    fr0, nr = F.feat_ring_off[idx], F.feat_ring_off[idx + 1] - F.feat_ring_off[idx]
    rings = _ranges(fr0, nr)
    v0 = F.ring_off[rings]
    nv = F.ring_off[rings + 1] - v0
    G = Features(
        F.kind[idx],
        F.coords[_ranges(v0, nv)],
        np.concatenate([[0], np.cumsum(nv)]).astype(np.int64),
        np.concatenate([[0], np.cumsum(nr)]).astype(np.int64),
        F.cls[idx],
        list(F.class_names),
    )
    G.class_colors = dict(F.class_colors)
    G.props = {k: np.asarray(v)[idx] for k, v in F.props.items()}
    return G


def _take_props(F: Features, idx: np.ndarray) -> Features:
    G = copy.copy(F)
    G.props = {k: np.asarray(v)[idx] for k, v in F.props.items()}
    return G


def _hexcol(i: int) -> str:
    return _auto_color(i)


# ------------------------------------------------------------------------------------------------ WKB (vectorised)
def _slice(F: Features, f0: int, f1: int) -> Features:
    r0, r1 = F.feat_ring_off[f0], F.feat_ring_off[f1]
    v0, v1 = F.ring_off[r0], F.ring_off[r1]
    return Features(
        F.kind[f0:f1],
        F.coords[v0:v1],
        F.ring_off[r0 : r1 + 1] - v0,
        F.feat_ring_off[f0 : f1 + 1] - r0,
        F.cls[f0:f1],
        F.class_names,
    )


def _chunks(F: Features, max_vertices: int = 1 << 21):
    """Feature ranges holding at most ~max_vertices vertices each (bounded memory for 10⁷-vertex layers)."""
    n = len(F)
    if n == 0:
        return
    vend = F.ring_off[F.feat_ring_off[1:]]
    f0 = 0
    while f0 < n:
        f1 = int(np.searchsorted(vend, F.ring_off[F.feat_ring_off[f0]] + max_vertices, 'right'))
        f1 = max(f1, f0 + 1)
        yield f0, min(f1, n)
        f0 = min(f1, n)


def encode_wkb(F: Features) -> tuple[np.ndarray, np.ndarray]:
    """All features → little-endian WKB. Returns (offsets int64 [n+1], bytes uint8). Polygon rings are closed.
    Vectorised per chunk of ~2M vertices: coordinates in one gather, headers inserted with one ``np.insert``."""
    if len(F) == 0:
        return np.zeros(1, np.int64), np.zeros(0, np.uint8)
    offs_parts, data_parts, base = [np.zeros(1, np.int64)], [], 0
    for f0, f1 in _chunks(F):
        o, d = _encode_wkb(_slice(F, f0, f1))
        offs_parts.append(o[1:] + base)
        data_parts.append(d)
        base += len(d)
    return np.concatenate(offs_parts), (np.concatenate(data_parts) if len(data_parts) > 1 else data_parts[0])


def _encode_wkb(F: Features) -> tuple[np.ndarray, np.ndarray]:
    n = len(F)
    if n == 0:
        return np.zeros(1, np.int64), np.zeros(0, np.uint8)
    nrings_f = np.diff(F.feat_ring_off)
    ring_feat = np.repeat(np.arange(n), nrings_f)  # feature of each ring
    ring_kind = F.kind[ring_feat]
    nv = np.diff(F.ring_off)
    close = (ring_kind == GEO_POLYGON).astype(np.int64)
    nout = np.where(ring_kind == GEO_POINT, 1, nv + close)  # vertices written per ring
    # coordinate stream in output order (closing vertex = first vertex)
    T = int(nout.sum())
    out_start = np.zeros(len(nout) + 1, np.int64)
    np.cumsum(nout, out=out_start[1:])
    within = np.arange(T) - np.repeat(out_start[:-1], nout)
    src = np.repeat(F.ring_off[:-1], nout) + np.where(within >= np.repeat(nv, nout), 0, within)
    cs = np.ascontiguousarray(F.coords[src].astype('<f8')).view(np.uint8).ravel()
    # headers before each ring: first ring of a feature carries the feature header
    first = np.zeros(len(nv), bool)
    first[F.feat_ring_off[:-1][nrings_f > 0]] = True
    H = np.zeros((len(nv), 13), np.uint8)
    hlen = np.zeros(len(nv), np.int64)

    def put32(rows, col, vals):
        v = np.asarray(vals, np.uint32)
        for b in range(4):
            H[rows, col + b] = (v >> (8 * b)) & 0xFF

    pt = first & (ring_kind == GEO_POINT)
    ln = first & (ring_kind == GEO_LINE)
    pf = first & (ring_kind == GEO_POLYGON)
    ph = ~first & (ring_kind == GEO_POLYGON)
    H[pt | ln | pf, 0] = 1
    put32(pt, 1, GEO_POINT)
    hlen[pt] = 5
    put32(ln, 1, GEO_LINE)
    put32(ln, 5, nout[ln])
    hlen[ln] = 9
    put32(pf, 1, GEO_POLYGON)
    put32(pf, 5, nrings_f[ring_feat[pf]])
    put32(pf, 9, nout[pf])
    hlen[pf] = 13
    put32(ph, 0, nout[ph])
    hlen[ph] = 4
    mask = np.arange(13)[None, :] < hlen[:, None]
    hdr = H[mask]
    pos = np.repeat(out_start[:-1] * 16, hlen)
    data = np.insert(cs, pos, hdr)
    # feature byte offsets: start of first ring + all header bytes before it
    hcum = np.zeros(len(nv) + 1, np.int64)
    np.cumsum(hlen, out=hcum[1:])
    ring_final = out_start[:-1] * 16 + hcum[:-1]
    offs = np.empty(n + 1, np.int64)
    offs[:n] = ring_final[F.feat_ring_off[:-1]] if len(ring_final) else 0
    offs[n] = len(data)
    return offs, data


def decode_wkb(blobs) -> Features:
    """WKB (list of bytes, or (offsets, data)) → Features. Point / LineString / Polygon in either byte order;
    Multi* geometries are split into one feature each (their index is in ``props['part_of']``)."""
    if isinstance(blobs, tuple):
        offs, data = blobs
        blobs = [bytes(data[offs[i] : offs[i + 1]]) for i in range(len(offs) - 1)]
    kinds, coords, ring_off, feat_ring_off, part_of = [], [], [0], [0], []

    def geom(mv, pos, owner):
        bo = '<' if mv[pos] == 1 else '>'
        (t,) = struct.unpack_from(bo + 'I', mv, pos + 1)
        t &= 0xFFFF
        t = t % 1000  # ISO Z/M codes: 1001, 2001, …  (only 2-D kept)
        pos += 5
        if t == GEO_POINT:
            x, y = struct.unpack_from(bo + 'dd', mv, pos)
            coords.append(np.array([[x, y]]))
            ring_off.append(ring_off[-1] + 1)
            kinds.append(GEO_POINT)
            feat_ring_off.append(feat_ring_off[-1] + 1)
            part_of.append(owner)
            return pos + 16
        if t == GEO_LINE:
            (m,) = struct.unpack_from(bo + 'I', mv, pos)
            a = np.frombuffer(mv, bo + 'f8', 2 * m, pos + 4).reshape(m, 2)
            coords.append(a)
            ring_off.append(ring_off[-1] + m)
            kinds.append(GEO_LINE)
            feat_ring_off.append(feat_ring_off[-1] + 1)
            part_of.append(owner)
            return pos + 4 + 16 * m
        if t == GEO_POLYGON:
            (nr,) = struct.unpack_from(bo + 'I', mv, pos)
            pos += 4
            for _ in range(nr):
                (m,) = struct.unpack_from(bo + 'I', mv, pos)
                a = np.frombuffer(mv, bo + 'f8', 2 * m, pos + 4).reshape(m, 2)
                if m > 1 and (a[0] == a[-1]).all():
                    a = a[:-1]
                coords.append(a)
                ring_off.append(ring_off[-1] + len(a))
                pos += 4 + 16 * m
            kinds.append(GEO_POLYGON)
            feat_ring_off.append(feat_ring_off[-1] + nr)
            part_of.append(owner)
            return pos
        if t in (4, 5, 6, 7):  # Multi* / GeometryCollection: one feature per part
            (np_,) = struct.unpack_from(bo + 'I', mv, pos)
            pos += 4
            for _ in range(np_):
                pos = geom(mv, pos, owner)
            return pos
        raise ValueError(f'unsupported WKB geometry type {t}')

    for i, b in enumerate(blobs):
        if b is not None:
            geom(memoryview(b), 0, i)
    F = Features(
        np.asarray(kinds, np.uint8),
        np.concatenate(coords).astype(np.float64) if coords else np.zeros((0, 2)),
        np.asarray(ring_off, np.int64),
        np.asarray(feat_ring_off, np.int64),
        np.zeros(len(kinds), np.int32),
        [],
    )
    F.props['__row__'] = np.asarray(part_of, np.int64)
    return F


# ------------------------------------------------------------------------------------------------ GeoParquet
def write_geoparquet(
    F: Features, path: str | Path, *, compression: str = 'zstd', row_group_size: int = 65536, slide: str | None = None
) -> Path:
    """GeoParquet 1.1: WKB ``geometry``, a ``bbox`` covering struct (fast spatial filtering), ``class`` as a
    dictionary column, every property as its own column; wissi-viz metadata (classes, colours, style, mpp)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    offs, data = encode_wkb(F)
    large = len(data) >= 2**31 - 1
    geom = (pa.LargeBinaryArray if large else pa.BinaryArray).from_buffers(
        pa.large_binary() if large else pa.binary(),
        len(F),
        [None, pa.py_buffer(offs.astype(np.int64 if large else np.int32)), pa.py_buffer(data)],
    )
    bb = F.bbox().astype(np.float32)
    bbox = pa.StructArray.from_arrays([pa.array(bb[:, i]) for i in range(4)], names=['xmin', 'ymin', 'xmax', 'ymax'])
    names = pa.array(list(F.class_names) or [''], pa.string())
    cls = pa.DictionaryArray.from_arrays(
        pa.array(np.where(F.cls < 0, 0, F.cls).astype(np.int32)),
        names,
        mask=pa.array(F.cls < 0) if (F.cls < 0).any() else None,
    )
    cols = {'class': cls}
    for k, v in F.props.items():
        if k.startswith('__'):
            continue
        cols[k] = _arrow_col(v)
    cols['bbox'] = bbox
    cols['geometry'] = geom
    table = pa.table(cols)
    types = sorted({['Point', 'LineString', 'Polygon'][k - 1] for k in np.unique(F.kind).tolist()})
    geo = {
        'version': '1.1.0',
        'primary_column': 'geometry',
        'columns': {
            'geometry': {
                'encoding': 'WKB',
                'geometry_types': types,
                'crs': None,
                'edges': 'planar',
                'bbox': [float(bb[:, 0].min()), float(bb[:, 1].min()), float(bb[:, 2].max()), float(bb[:, 3].max())]
                if len(bb)
                else [0, 0, 0, 0],
                'covering': {
                    'bbox': {
                        'xmin': ['bbox', 'xmin'],
                        'ymin': ['bbox', 'ymin'],
                        'xmax': ['bbox', 'xmax'],
                        'ymax': ['bbox', 'ymax'],
                    }
                },
            }
        },
    }
    wissi = {
        'format': 'wissi-viz annotations',
        'version': 1,
        'units': 'pixel (level 0, y down)',
        'mpp': F.mpp,
        'name': F.name,
        'classes': {c: F.class_colors.get(c) for c in F.class_names},
        'style': F.style,
        'slide': slide,
    }
    table = table.replace_schema_metadata(
        {b'geo': json.dumps(geo).encode(), b'wissi': json.dumps(wissi, default=str).encode()}
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, p, compression=compression, row_group_size=row_group_size, write_statistics=True)
    return p


def _arrow_col(v):
    import pyarrow as pa

    v = np.asarray(v)
    if v.dtype.kind in 'biuf':
        return pa.array(v)
    vals = [None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x) for x in v.tolist()]
    arr = pa.array(vals, pa.string())
    uniq = len(set(vals))
    return arr.dictionary_encode() if uniq <= max(1, len(vals) // 4) else arr


def read_geoparquet(path: str | Path, bbox: tuple | None = None, columns: list | None = None) -> Features:
    """GeoParquet (ours or any tool's) → Features. ``bbox`` = (x0, y0, x1, y1): read only intersecting features,
    using the bbox covering column and row-group statistics when present."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    md = pf.schema_arrow.metadata or {}
    geo = json.loads(md.get(b'geo', b'{}') or b'{}')
    wissi = json.loads(md.get(b'wissi', b'{}') or b'{}')
    gcol = geo.get('primary_column', 'geometry')
    cov = (geo.get('columns', {}).get(gcol, {}).get('covering') or {}).get('bbox')
    filt = None
    if bbox is not None and cov:
        import pyarrow.dataset as ds

        x0, y0, x1, y1 = bbox
        f = lambda k: ds.field(*cov[k])
        filt = (f('xmax') >= x0) & (f('xmin') <= x1) & (f('ymax') >= y0) & (f('ymin') <= y1)
        table = ds.dataset(path, format='parquet').to_table(filter=filt, columns=columns)
    else:
        table = pq.read_table(path, columns=columns)
    if gcol not in table.column_names:
        raise ValueError(f'{Path(path).name}: no geometry column {gcol!r}')
    enc = geo.get('columns', {}).get(gcol, {}).get('encoding', 'WKB').upper()
    if enc != 'WKB':
        raise ValueError(f'{Path(path).name}: geometry encoding {enc} not supported (write it as WKB)')
    garr = table.column(gcol).combine_chunks()
    F = _decode_arrow_wkb(garr)
    rows = F.props.pop('__row__')
    if bbox is not None and not cov:  # no covering column: filter after decoding
        b = F.bbox()
        x0, y0, x1, y1 = bbox
        keep_rows = np.unique(rows[(b[:, 2] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 1] <= y1)])
        sub = table.take(keep_rows)
        return read_from_table_subset(sub, gcol, wissi)
    # classes and properties (repeated for parts of multi-geometries)
    ccol = next((c for c in ('class', 'classification', 'Class', 'name') if c in table.column_names), None)
    if ccol is not None:
        col = table.column(ccol).combine_chunks()
        if hasattr(col, 'dictionary'):
            names = [str(x) for x in col.dictionary.to_pylist()]
            idx = col.indices.to_numpy(zero_copy_only=False)
            valid = np.asarray(col.is_valid())
            codes = np.where(valid, np.nan_to_num(idx, nan=0), -1).astype(np.int32)
        else:
            vals = np.asarray(col.to_pylist(), dtype=object)
            vals = np.array(['' if v is None else str(v) for v in vals], dtype=object)
            names, codes = np.unique(vals, return_inverse=True)
            names = [str(x) for x in names]
            codes = codes.astype(np.int32)
        F.class_names = names
        F.cls = codes[rows]
    else:
        F.class_names, F.cls = [], np.full(len(F), -1, np.int32)
    for c in table.column_names:
        if c in (gcol, 'bbox', ccol):
            continue
        col = table.column(c).combine_chunks()
        if hasattr(col, 'dictionary'):
            col = pc.cast(col, col.type.value_type) if hasattr(col.type, 'value_type') else col
        try:
            v = col.to_numpy(zero_copy_only=False)
        except Exception:  # noqa: BLE001  (nested types: keep as JSON text)
            v = np.asarray([json.dumps(x) if x is not None else None for x in col.to_pylist()], dtype=object)
        F.props[c] = np.asarray(v)[rows]
    F.class_colors = {k: v for k, v in (wissi.get('classes') or {}).items() if v}
    F.name = wissi.get('name') or Path(path).stem
    F.style = wissi.get('style') or {}
    F.mpp = wissi.get('mpp')
    return F


def read_from_table_subset(sub, gcol, wissi) -> Features:  # helper for files without a covering column

    import pyarrow.parquet as pq

    buf = io.BytesIO()
    meta = {
        b'geo': json.dumps(
            {'version': '1.1.0', 'primary_column': gcol, 'columns': {gcol: {'encoding': 'WKB', 'geometry_types': []}}}
        ).encode(),
        b'wissi': json.dumps(wissi).encode(),
    }
    pq.write_table(sub.replace_schema_metadata(meta), buf)
    buf.seek(0)
    return read_geoparquet(buf)


def _decode_arrow_wkb(arr) -> Features:
    """Binary arrow column → Features; vectorised fast path for single-ring polygons / points / lines in
    little-endian (what wissi-viz and most tools write), the generic parser otherwise."""
    import pyarrow as pa

    n = len(arr)
    if n == 0:
        F = decode_wkb([])
        return F
    if arr.null_count:
        arr = arr.fill_null(b'')
    if pa.types.is_large_binary(arr.type):
        offs = np.frombuffer(arr.buffers()[1], np.int64, n + 1, arr.offset * 8)
    else:
        offs = np.frombuffer(arr.buffers()[1], np.int32, n + 1, arr.offset * 4).astype(np.int64)
    data = np.frombuffer(arr.buffers()[2], np.uint8)
    starts = offs[:-1]
    if (offs[1:] - starts < 5).any() or (data[starts] != 1).any():
        return decode_wkb((offs, data))
    t = data[starts + 1].astype(np.uint32) | (data[starts + 2].astype(np.uint32) << 8)
    if not np.isin(t, (GEO_POINT, GEO_LINE, GEO_POLYGON)).all():
        return decode_wkb((offs, data))

    def u32(at):
        return (
            data[at].astype(np.int64)
            | (data[at + 1].astype(np.int64) << 8)
            | (data[at + 2].astype(np.int64) << 16)
            | (data[at + 3].astype(np.int64) << 24)
        )

    poly = t == GEO_POLYGON
    if poly.any() and (u32(starts[poly] + 5) != 1).any():
        return decode_wkb((offs, data))  # holes present: generic path
    nv = np.where(t == GEO_POINT, 1, 0).astype(np.int64)
    cstart = starts + 5
    line = t == GEO_LINE
    nv[line] = u32(starts[line] + 5)
    cstart[line] = starts[line] + 9
    nv[poly] = u32(starts[poly] + 9)
    cstart[poly] = starts[poly] + 13
    keep = nv - poly.astype(np.int64)  # drop polygon closing vertex
    total = int(keep.sum())
    out_off = np.zeros(n + 1, np.int64)
    np.cumsum(keep, out=out_off[1:])
    coords = np.empty((total, 2), np.float64)
    lane = np.arange(16)
    a = 0
    while a < n:  # ≤ ~1M vertices per gather: bounded index memory
        b = max(a + 1, int(np.searchsorted(out_off, out_off[a] + (1 << 20), 'right')) - 1)
        b = min(b, n)
        k = keep[a:b]
        tot = int(k.sum())
        if not tot:
            a = b
            continue
        within = np.arange(tot) - np.repeat(out_off[a:b] - out_off[a], k)
        byte_at = np.repeat(cstart[a:b], k) + within * 16
        coords[out_off[a] : out_off[b]] = data[byte_at[:, None] + lane].view('<f8').reshape(tot, 2)
        a = b
    F = Features(t.astype(np.uint8), coords, out_off, np.arange(n + 1, dtype=np.int64), np.zeros(n, np.int32), [])
    F.props['__row__'] = np.arange(n)
    return F


# ------------------------------------------------------------------------------------------------ other formats
def _rgb(hexcol: str | None) -> list[int]:
    h = (hexcol or '#ffcc00').lstrip('#')
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def _colorref(hexcol: str | None) -> str:
    r, g, b = _rgb(hexcol)
    return str(r | (g << 8) | (b << 16))


def _scalar(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


def write_geojson(F: Features, path, qupath: bool = False) -> Path:
    """GeoJSON FeatureCollection. ``qupath``: QuPath's flavour (objectType, classification colour as RGB)."""
    p = Path(path)
    names = F.class_of()
    obj_type = 'detection' if len(F) > 5000 else 'annotation'
    keys = [k for k in F.props if not k.startswith('__')]
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('{"type":"FeatureCollection","features":[')
        for i in range(len(F)):
            rs = F.rings(i)
            k = F.kind[i]
            if k == GEO_POINT:
                g = {'type': 'Point', 'coordinates': [float(rs[0][0, 0]), float(rs[0][0, 1])]}
            elif k == GEO_LINE:
                g = {'type': 'LineString', 'coordinates': rs[0].round(2).tolist()}
            else:
                g = {'type': 'Polygon', 'coordinates': [np.vstack([r, r[:1]]).round(2).tolist() for r in rs]}
            props = {kk: _scalar(F.props[kk][i]) for kk in keys}
            cname = names[i]
            if qupath:
                props = {
                    'objectType': obj_type,
                    **({'classification': {'name': cname, 'color': _rgb(F.class_colors.get(cname))}} if cname else {}),
                    'measurements': {kk: v for kk, v in props.items() if isinstance(v, (int, float))},
                }
            elif cname:
                props['classification'] = {'name': cname, 'color': F.class_colors.get(cname)}
            fh.write(
                (',' if i else '')
                + json.dumps({'type': 'Feature', 'geometry': g, 'properties': props}, separators=(',', ':'))
            )
        fh.write(']}')
    return p


def write_halo_xml(F: Features, path) -> Path:
    root = ET.Element('Annotations')
    names = F.class_of()
    by_class: dict[str, list] = {}
    for i in range(len(F)):
        by_class.setdefault(names[i] or 'Annotation', []).append(i)
    for cname, idx in by_class.items():
        a = ET.SubElement(
            root, 'Annotation', Name=cname, Visible='True', LineColor=_colorref(F.class_colors.get(cname))
        )
        regs = ET.SubElement(a, 'Regions')
        for i in idx:
            rs = F.rings(i)
            k = F.kind[i]
            for j, r in enumerate(rs):
                typ = 'Pin' if k == GEO_POINT else ('Ruler' if k == GEO_LINE else 'Polygon')
                reg = ET.SubElement(regs, 'Region', Type=typ, HasEndcaps='0', NegativeROA='1' if j else '0')
                vs = ET.SubElement(reg, 'Vertices')
                for x, y in r:
                    ET.SubElement(vs, 'V', X=f'{x:.2f}', Y=f'{y:.2f}')
    p = Path(path)
    ET.ElementTree(root).write(p, encoding='utf-8', xml_declaration=True)
    return p


def write_aperio_xml(F: Features, path) -> Path:
    root = ET.Element('Annotations', MicronsPerPixel=str(F.mpp or 0))
    names = F.class_of()
    by_class: dict[str, list] = {}
    for i in range(len(F)):
        by_class.setdefault(names[i] or 'Annotation', []).append(i)
    rid = 0
    for aid, (cname, idx) in enumerate(by_class.items(), 1):
        a = ET.SubElement(
            root,
            'Annotation',
            Id=str(aid),
            Name=cname,
            ReadOnly='0',
            LineColor=_colorref(F.class_colors.get(cname)),
            Visible='1',
            Selected='0',
            Type='4',
        )
        ET.SubElement(a, 'Attributes')
        regs = ET.SubElement(a, 'Regions')
        for i in idx:
            for j, r in enumerate(F.rings(i)):
                rid += 1
                k = F.kind[i]
                reg = ET.SubElement(
                    regs,
                    'Region',
                    Id=str(rid),
                    Type='0' if k != GEO_LINE else '4',
                    NegativeROA='1' if j else '0',
                    Selected='0',
                    Analyze='1',
                )
                vs = ET.SubElement(reg, 'Vertices')
                for x, y in r:
                    ET.SubElement(vs, 'Vertex', X=f'{x:.2f}', Y=f'{y:.2f}', Z='0')
    p = Path(path)
    ET.ElementTree(root).write(p, encoding='utf-8', xml_declaration=True)
    return p


def write_csv(F: Features, path, wkt: bool = False) -> Path:
    p = Path(path)
    sep = '\t' if p.suffix.lower() in ('.tsv', '.txt') else ','
    b = F.bbox()
    area = F.shell_area()
    names = F.class_of()
    keys = [k for k in F.props if not k.startswith('__')]
    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter=sep)
        w.writerow(
            ['id', 'class', 'type', 'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax', 'area_px']
            + keys
            + (['wkt'] if wkt else [])
        )
        tname = {GEO_POINT: 'point', GEO_LINE: 'line', GEO_POLYGON: 'polygon'}
        for i in range(len(F)):
            row = [
                i,
                names[i],
                tname[int(F.kind[i])],
                f'{(b[i, 0] + b[i, 2]) / 2:.2f}',
                f'{(b[i, 1] + b[i, 3]) / 2:.2f}',
                *(f'{v:.2f}' for v in b[i]),
                f'{area[i]:.1f}',
            ] + [_scalar(F.props[k][i]) for k in keys]
            if wkt:
                row.append(_wkt(F, i))
            w.writerow(row)
    return p


def _wkt(F: Features, i: int) -> str:
    rs = F.rings(i)
    fmt = lambda r: ', '.join(f'{x:.2f} {y:.2f}' for x, y in r)
    k = F.kind[i]
    if k == GEO_POINT:
        return f'POINT ({rs[0][0, 0]:.2f} {rs[0][0, 1]:.2f})'
    if k == GEO_LINE:
        return f'LINESTRING ({fmt(rs[0])})'
    return 'POLYGON (' + ', '.join(f'({fmt(np.vstack([r, r[:1]]))})' for r in rs) + ')'


def write_coco(
    F: Features, path, width: int | None = None, height: int | None = None, file_name: str | None = None
) -> Path:
    b = F.bbox()
    area = F.shell_area()
    names = list(F.class_names) or ['object']
    cats = [{'id': i + 1, 'name': c, 'supercategory': 'object'} for i, c in enumerate(names)]
    W = int(width or (np.ceil(b[:, 2].max()) if len(b) else 0))
    H = int(height or (np.ceil(b[:, 3].max()) if len(b) else 0))
    anns = []
    for i in range(len(F)):
        if F.kind[i] != GEO_POLYGON:
            continue
        rs = F.rings(i)
        anns.append(
            {
                'id': len(anns) + 1,
                'image_id': 1,
                'category_id': int(max(F.cls[i], 0)) + 1,
                'iscrowd': 0,
                'segmentation': [rs[0].round(2).ravel().tolist()],
                'area': float(area[i]),
                'bbox': [float(b[i, 0]), float(b[i, 1]), float(b[i, 2] - b[i, 0]), float(b[i, 3] - b[i, 1])],
            }
        )
    doc = {
        'info': {'description': f'{F.name} (wissi-viz)', 'version': '1.0'},
        'licenses': [],
        'images': [{'id': 1, 'file_name': file_name or F.name, 'width': W, 'height': H}],
        'annotations': anns,
        'categories': cats,
    }
    p = Path(path)
    p.write_text(json.dumps(doc), encoding='utf-8')
    return p


def read_coco(path) -> Features:
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    cats = {c['id']: c['name'] for c in doc.get('categories', [])}
    names = list(dict.fromkeys(cats.values()))
    kinds, coords, ring_off, cls = [], [], [0], []
    for a in doc.get('annotations', []):
        seg = a.get('segmentation')
        if not isinstance(seg, list) or not seg:
            continue  # RLE masks: not supported
        r = np.asarray(seg[0], np.float64).reshape(-1, 2)
        if len(r) >= 2 and (r[0] == r[-1]).all():
            r = r[:-1]
        coords.append(r)
        ring_off.append(ring_off[-1] + len(r))
        kinds.append(GEO_POLYGON)
        nm = cats.get(a.get('category_id'))
        cls.append(names.index(nm) if nm in names else -1)
    n = len(kinds)
    return Features(
        np.asarray(kinds, np.uint8),
        np.concatenate(coords) if coords else np.zeros((0, 2)),
        np.asarray(ring_off, np.int64),
        np.arange(n + 1, dtype=np.int64),
        np.asarray(cls, np.int32),
        names,
        name=Path(path).stem.replace('.coco', ''),
    )


def write_mask(F: Features, path, width: int, height: int, downsample: float = 1.0, by: str = 'feature') -> Path:
    """Label image: ``by='feature'`` → each polygon gets its index + 1 (uint32); ``by='class'`` → class index + 1
    (uint16). Rasterised at ``downsample`` (level-0 px per mask px) into a tiled TIFF."""
    w, h = int(np.ceil(width / downsample)), int(np.ceil(height / downsample))
    img = Image.new('I', (w, h), 0)
    d = ImageDraw.Draw(img)
    for i in range(len(F)):
        if F.kind[i] != GEO_POLYGON:
            continue
        rs = F.rings(i)
        val = i + 1 if by == 'feature' else int(F.cls[i]) + 1
        d.polygon([tuple(v) for v in (rs[0] / downsample).tolist()], fill=val)
        for hole in rs[1:]:
            d.polygon([tuple(v) for v in (hole / downsample).tolist()], fill=0)
    a = np.asarray(img, np.int32)
    a = a.astype(np.uint32 if by == 'feature' else np.uint16)
    p = Path(path)
    tifffile.imwrite(
        p,
        a,
        tile=(512, 512) if min(w, h) >= 512 else None,
        compression='zlib',
        metadata={'downsample': downsample, 'classes': list(F.class_names), 'by': by},
    )
    return p


def read_mask(path, downsample: float = 1.0) -> Features:
    return from_layer(from_label_mask(tifffile.imread(path), downsample=downsample).to_layer())


# ------------------------------------------------------------------------------------------------ dispatch
def detect_format(path: str | Path) -> str:
    n = Path(path).name.lower()
    if n.endswith(('.parquet', '.geoparquet')):
        return 'geoparquet'
    if n.endswith('.coco.json'):
        return 'coco'
    if n.endswith(('.geojson', '.json')):
        if n.endswith('.json'):
            try:
                with open(path, 'rb') as fh:
                    head = fh.read(4096)
                if b'"annotations"' in head and b'"images"' in head:
                    return 'coco'
            except OSError:
                pass
        return 'geojson'
    if n.endswith('.annotations'):
        return 'halo'
    if n.endswith('.xml'):
        try:
            head = Path(path).read_text(encoding='utf-8-sig', errors='ignore')[:4000]
        except OSError:
            head = ''
        if '<ASAP_Annotations' in head:
            return 'asap'
        return 'halo' if '<V ' in head or '<V\n' in head or 'HALO' in head else 'aperio'
    if n.endswith(('.csv', '.tsv', '.txt')):
        return 'csv'
    if n.endswith(('.tif', '.tiff')):
        return 'mask'
    raise ValueError(f'unknown annotation format: {Path(path).name}')


def load(
    path: str | Path,
    fmt: str = 'auto',
    *,
    bbox: tuple | None = None,
    name: str | None = None,
    mpp: float | None = None,
    downsample: float = 1.0,
) -> AnnotationLayer:
    """Any supported file → AnnotationLayer."""
    fmt = detect_format(path) if fmt == 'auto' else fmt
    if fmt == 'geoparquet':
        return to_layer(read_geoparquet(path, bbox=bbox), name=name)
    if fmt == 'coco':
        return to_layer(read_coco(path), name=name)
    if fmt == 'mask':
        return to_layer(read_mask(path, downsample), name=name or Path(path).stem)
    if fmt == 'csv':
        return read_objects(path, mpp=mpp).to_layer(name=name or Path(path).stem)
    layer = AnnotationLayer.load(path)
    if name:
        layer.name = name
    return layer


def export(
    layer: AnnotationLayer | Features,
    path: str | Path,
    fmt: str = 'auto',
    *,
    mpp: float | None = None,
    slide: str | None = None,
    width: int | None = None,
    height: int | None = None,
    downsample: float = 1.0,
    by: str = 'feature',
    wkt: bool = False,
) -> Path:
    """Write a layer in any supported format (by extension, or ``fmt``)."""
    F = layer if isinstance(layer, Features) else from_layer(layer)
    if mpp:
        F.mpp = mpp
    fmt = detect_format(path) if fmt == 'auto' else fmt
    if fmt == 'geoparquet':
        return write_geoparquet(F, path, slide=slide)
    if fmt in ('geojson', 'qupath'):
        return write_geojson(F, path, qupath=fmt == 'qupath')
    if fmt == 'halo':
        return write_halo_xml(F, path)
    if fmt == 'aperio':
        return write_aperio_xml(F, path)
    if fmt == 'csv':
        return write_csv(F, path, wkt=wkt)
    if fmt == 'coco':
        return write_coco(F, path, width, height, file_name=slide)
    if fmt == 'mask':
        if not (width and height):
            b = F.bbox()
            width, height = int(np.ceil(b[:, 2].max())), int(np.ceil(b[:, 3].max()))
        return write_mask(F, path, width, height, downsample, by)
    raise ValueError(f'unknown format {fmt!r} (one of {", ".join(FORMATS)})')


def convert(
    src: str | Path,
    dst: str | Path,
    *,
    src_fmt: str = 'auto',
    dst_fmt: str = 'auto',
    slide: str | None = None,
    downsample: float = 1.0,
    by: str = 'feature',
    wkt: bool = False,
    mpp: float | None = None,
) -> Path:
    """File → file. ``slide``: a slide path, for the mask size, µm/px and COCO image name."""
    width = height = None
    if slide:
        s = open_slide(slide)
        width, height = s.info.width, s.info.height
        mpp = mpp or (s.info.mpp or (None,))[0]
    layer = load(src, src_fmt, mpp=mpp)
    return export(
        layer,
        dst,
        dst_fmt,
        mpp=mpp,
        slide=Path(slide).name if slide else None,
        width=width,
        height=height,
        downsample=downsample,
        by=by,
        wkt=wkt,
    )
