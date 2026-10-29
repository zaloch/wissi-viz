"""Chain steps between layers: objects → regions, and cells → only those inside regions.

    regions = to_regions(registry, sid, 'Found · gland', join_um=20, min_area_um2=500)   # a region layer
    inside(registry, sid, 'cells', regions)                                               # 'cells in Regions · …'
    analyses.run(registry, sid, 'nn', 'cells', {...}, within={'layer': regions})          # any analysis, ROI only

``to_regions`` merges objects (outlines, or points / boxes) into regions: objects closer than ``join_um`` become one
region (dilate, union, erode back: a morphological closing), holes smaller than ``min_area_um2`` are filled and
regions smaller than it dropped. A layer of areas (find areas, drawn ROIs) is already regions: pick its classes.
"""

from __future__ import annotations

import math

import numpy as np
import shapely

from . import annio
from .annotations import AnnotationLayer

ORIGIN = 'found'  # made from found / detected objects: listed with them, usable as a region everywhere


def _mpp(registry, sid: str) -> float:
    return float((registry.info(sid).get('mpp') or [None])[0] or 0.5)


def _layer(registry, sid: str, layer):
    if isinstance(layer, AnnotationLayer):
        return layer
    L = registry.layer(sid, layer)
    if L is None:
        raise ValueError(f'no layer {layer!r} on this slide')
    return L


def geometry(registry, sid: str, layer, classes: list[str] | None = None):
    """The union of a layer's polygons (only ``classes``, if given) in level-0 px; None if it has no polygons."""
    F = annio.from_layer(_layer(registry, sid, layer))
    names = F.class_of()
    want = set(classes or [])
    polys = []
    for i in range(len(F)):
        if F.kind[i] != annio.GEO_POLYGON or (want and names[i] not in want):
            continue
        rs = F.rings(i)
        if len(rs[0]) < 3:
            continue
        p = shapely.make_valid(shapely.Polygon(rs[0], [r for r in rs[1:] if len(r) >= 3]))
        if not p.is_empty:
            polys.append(p)
    if not polys:
        return None
    g = shapely.union_all(polys)
    g = shapely.make_valid(g)
    return g if not g.is_empty else None


def to_regions(
    registry,
    sid: str,
    layer,
    *,
    classes: list[str] | None = None,
    join_um: float = 20.0,
    min_area_um2: float = 0.0,
    name: str | None = None,
    cls: str = 'Region',
    color: str = '#ffd166',
) -> str:
    """Merge the objects of ``layer`` into regions (see module docstring); adds the region layer, returns its name."""
    L = _layer(registry, sid, layer)
    F = annio.from_layer(L)
    names = F.class_of()
    want = set(classes or [])
    mpp = _mpp(registry, sid)
    r = max(0.0, join_um / mpp / 2)  # half the gap: two objects join when their dilations touch
    shapes = []
    for i in range(len(F)):
        if want and names[i] not in want:
            continue
        rs = F.rings(i)
        if F.kind[i] == annio.GEO_POLYGON and len(rs[0]) >= 3:
            shapes.append(shapely.make_valid(shapely.Polygon(rs[0], [q for q in rs[1:] if len(q) >= 3])))
        else:  # points / lines: a small disc around them
            shapes.append(shapely.buffer(shapely.multipoints(rs[0]), max(r, 2.0)))
    shapes = [s for s in shapes if not s.is_empty]
    if not shapes:
        raise ValueError(f'{L.name} has no objects{" of " + ", ".join(sorted(want)) if want else ""} to merge')
    g = (
        shapely.union_all(shapely.buffer(np.asarray(shapes, dtype=object), r, quad_segs=4))
        if r
        else shapely.union_all(shapes)
    )
    if r:
        g = g.buffer(-r, quad_segs=4)
    g = shapely.make_valid(g)
    min_px = min_area_um2 / (mpp * mpp)
    parts = [p for p in getattr(g, 'geoms', [g]) if p.geom_type == 'Polygon' and not p.is_empty]
    out = AnnotationLayer(name='regions')
    n = 0
    for p in parts:
        if p.area < max(min_px, 1.0):
            continue
        holes = [h for h in p.interiors if shapely.Polygon(h).area >= max(min_px, 1.0)]
        p = p.simplify(0.5, preserve_topology=True)
        try:
            out.add_polygon(np.asarray(p.exterior.coords), holes=[np.asarray(h.coords) for h in holes], cls=cls,
                            color=color, props={'area_um2': round(p.area * mpp * mpp, 1)})  # fmt: skip
            n += 1
        except ValueError:
            continue
    if not n:
        raise ValueError(f'no region of at least {min_area_um2:g} µm² after merging {L.name}')
    out.name = _unique(registry, sid, name or f'Regions · {L.name}')
    out.origin = ORIGIN
    out.mpp = L.mpp
    out.fill_opacity = 0.12
    out.line_width = 2.0
    registry.add_annotations(sid, out)
    return out.name


def inside(
    registry,
    sid: str,
    objects,
    regions,
    *,
    classes: list[str] | None = None,
    name: str | None = None,
) -> str:
    """A new layer with the objects of ``objects`` whose centre lies inside ``regions`` (its ``classes`` only, if
    given); measurements and classes are kept. Returns the new layer's name."""
    L = _layer(registry, sid, objects)
    R = _layer(registry, sid, regions)
    g = geometry(registry, sid, R, classes)
    if g is None:
        raise ValueError(f'{R.name} has no regions (polygons) to select with')
    F = annio.from_layer(L)
    c = F.centroid()
    shapely.prepare(g)
    idx = np.nonzero(shapely.contains_xy(g, c[:, 0], c[:, 1]))[0]
    if not len(idx):
        raise ValueError(f'no objects of {L.name} inside {R.name}')
    G = annio.subset(F, idx)
    G.mpp = F.mpp
    G.style = dict(F.style or {})
    out = annio.to_layer(G, name=_unique(registry, sid, name or f'{L.name} in {R.name}'))
    out.origin = L.origin or 'imported'
    out.mpp = L.mpp
    registry.add_annotations(sid, out)
    return out.name


def select_inside(objs, g):
    """An ObjectTable restricted to the objects whose centre lies inside geometry ``g`` (level-0 px)."""
    shapely.prepare(g)
    keep = np.nonzero(shapely.contains_xy(g, objs.x, objs.y))[0]
    return objs.subset(keep)


def area_mm2(g, mpp: float) -> float:
    return float(g.area) * mpp * mpp / 1e6 if g is not None else math.nan


def _unique(registry, sid: str, base: str) -> str:
    have = set(registry.layers(sid))
    name, k = base, 1
    while name in have:
        k += 1
        name = f'{base} {k}'
    return name
