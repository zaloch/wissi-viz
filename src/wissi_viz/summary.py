"""Quick numbers about one layer for the app's inspector: count and share per class, per-class means of the
main measurements, area of drawn regions, objects per mm² of tissue, and where one object ranks within its
layer. Needs numpy only; the tissue area uses scikit-image when it is installed."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from . import annio
from .annotations import POLYGON, AnnotationLayer

#: measurement names shown first in summaries and object cards (matched as substrings, in this order)
PREFERRED = ('area', 'circular', 'eccentric', 'solidity', 'perimeter', 'intensity', 'mean', 'od', 'score')
MAX_REGION_FEATURES = 50_000  # above this a layer is objects, not regions: no per-class area

_lock = threading.Lock()
_sorted: OrderedDict = OrderedDict()  # (layer id, version, column) → sorted finite values (for percentiles)
_tissue: dict[str, float | None] = {}  # slide key → tissue area in mm²


def preferred_columns(layer: AnnotationLayer, n: int = 4) -> list[str]:
    cols = layer.numeric_columns()
    rank = lambda c: next((i for i, k in enumerate(PREFERRED) if k in c.lower()), len(PREFERRED))
    return sorted(cols, key=rank)[:n]


def summary(layer: AnnotationLayer, mpp: float | None = None, tissue_mm2: float | None = None) -> dict:
    """Counts and shares per class, per-class means of up to four measurements, region areas (drawn or
    polygon layers) and, given the tissue area, objects per mm²."""
    c = layer.compiled()
    n = int(c.n)
    ids = np.asarray(c.cls[:n], np.int64)
    names = {v: k for k, v in layer._class_index.items()}
    counts = np.bincount(ids, minlength=max(names, default=-1) + 1) if n else np.zeros(0, int)
    cols = preferred_columns(layer)
    means: dict[str, np.ndarray] = {}
    for col in cols:
        v = layer.column_values(col)
        ok = np.isfinite(v)
        tot = np.bincount(ids[ok], weights=v[ok], minlength=len(counts))
        cnt = np.bincount(ids[ok], minlength=len(counts))
        with np.errstate(invalid='ignore', divide='ignore'):
            means[col] = tot / cnt
    areas = None
    polygons = n and n <= MAX_REGION_FEATURES and bool((np.asarray(c.kinds[:n]) == POLYGON).any())
    if polygons and mpp:
        a = annio.from_layer(layer).shell_area() * mpp * mpp / 1e6  # mm²
        areas = np.bincount(ids, weights=a, minlength=len(counts))
    classes = []
    for k in np.argsort(-counts, kind='stable') if n else []:
        if counts[k] == 0:
            continue
        name = names.get(int(k), '')
        row = {
            'name': name,
            'color': layer.classes.get(name, '#9aa3ad'),
            'n': int(counts[k]),
            'pct': round(100.0 * counts[k] / n, 1),
            'means': {col: _num(means[col][k]) for col in cols},
        }
        if areas is not None:
            row['area_mm2'] = _num(areas[k])
        classes.append(row)
    out: dict[str, Any] = {
        'layer': layer.name,
        'n': n,
        'classes': classes,
        'columns': cols,
        'numeric': layer.numeric_columns(),
    }
    if areas is not None:
        out['area_mm2'] = _num(areas.sum())
    if tissue_mm2:
        out['tissue_mm2'] = round(tissue_mm2, 4)
        out['per_mm2'] = round(n / tissue_mm2, 1)
    return out


def percentiles(layer: AnnotationLayer, props: dict) -> dict[str, float]:
    """Rank (0–100) of each numeric property of one object among all objects of its layer."""
    out = {}
    cols = set(layer.numeric_columns())
    for k, v in props.items():
        if k not in cols or not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
            continue
        s = _sorted_column(layer, k)
        if s.size:
            out[k] = round(100.0 * float(np.searchsorted(s, v, side='right')) / s.size, 1)
    return out


def tissue_area_mm2(key: str, source: Any) -> float | None:
    """Tissue area of a slide (computed once per slide on a low-resolution level); None without
    scikit-image / shapely or without µm per pixel."""
    with _lock:
        if key in _tissue:
            return _tissue[key]
    area = None
    mpp = (getattr(source.info, 'mpp', None) or (None,))[0]
    if mpp:
        try:
            from .spatial import tissue_mask
        except ImportError:  # the [analysis] extra is not installed
            tissue_mask = None
        if tissue_mask is not None:
            geom, _ = tissue_mask(source)
            area = float(geom.area) * mpp * mpp / 1e6 or None
    with _lock:
        _tissue[key] = area
    return area


def _sorted_column(layer: AnnotationLayer, name: str) -> np.ndarray:
    key = (id(layer), layer.version, name)
    with _lock:
        hit = _sorted.get(key)
        if hit is not None:
            _sorted.move_to_end(key)
            return hit
    v = layer.column_values(name)
    s = np.sort(v[np.isfinite(v)])
    with _lock:
        _sorted[key] = s
        while len(_sorted) > 32:
            _sorted.popitem(last=False)
    return s


def _num(x: float) -> float | None:
    x = float(x)
    return round(x, 4) if np.isfinite(x) else None
