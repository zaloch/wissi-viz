"""How well a layer of outlines matches a reference layer: objects matched one-to-one (F1 at an IoU) and areas
overlapping (Dice). Used by ``wissi-viz validate``; useful for any two layers.

    m = match(found, reference, region=geom)      # {'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_iou'}
    d = dice(found_geometry, reference_geometry)  # 2 |A ∩ B| / (|A| + |B|)
"""

from __future__ import annotations

import numpy as np
import shapely

from . import annio


def polygons(F: annio.Features, classes: list[str] | None = None) -> np.ndarray:
    """The polygons of a Features (only ``classes``, if given) as shapely geometries (valid, non-empty)."""
    names = F.class_of()
    want = set(classes or [])
    out = []
    for i in range(len(F)):
        if F.kind[i] != annio.GEO_POLYGON or (want and names[i] not in want):
            continue
        rs = F.rings(i)
        if len(rs[0]) < 3:
            continue
        p = shapely.make_valid(shapely.Polygon(rs[0], [r for r in rs[1:] if len(r) >= 3]))
        if not p.is_empty and p.area > 0:
            out.append(p)
    return np.asarray(out, dtype=object)


def match(found, reference, region=None, iou: float = 0.5) -> dict:
    """Match outlines one-to-one (greedy, best IoU first, at least ``iou``): true / false positives, misses, precision,
    recall, F1 and the mean IoU of the matches. ``found`` / ``reference``: Features or arrays of shapely polygons;
    ``region``: only objects whose centre lies inside it count."""
    A = polygons(found) if isinstance(found, annio.Features) else np.asarray(found, dtype=object)
    B = polygons(reference) if isinstance(reference, annio.Features) else np.asarray(reference, dtype=object)
    if region is not None:
        shapely.prepare(region)
        A = A[shapely.contains(region, shapely.centroid(A))] if len(A) else A
        B = B[shapely.contains(region, shapely.centroid(B))] if len(B) else B
    tp, ious = 0, []
    if len(A) and len(B):
        i, j = shapely.STRtree(B).query(A, predicate='intersects')
        inter = shapely.area(shapely.intersection(A[i], B[j]))
        union = shapely.area(A[i]) + shapely.area(B[j]) - inter
        v = inter / np.maximum(union, 1e-9)
        order = np.argsort(-v, kind='stable')
        used_a, used_b = set(), set()
        for k in order:
            if v[k] < iou:
                break
            if i[k] in used_a or j[k] in used_b:
                continue
            used_a.add(int(i[k]))
            used_b.add(int(j[k]))
            ious.append(float(v[k]))
        tp = len(ious)
    fp, fn = len(A) - tp, len(B) - tp
    precision = tp / len(A) if len(A) else 0.0
    recall = tp / len(B) if len(B) else 0.0
    f1 = 2 * tp / (len(A) + len(B)) if len(A) + len(B) else 1.0
    return {
        'found': int(len(A)),
        'reference': int(len(B)),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3),
        'mean_iou': round(float(np.mean(ious)), 3) if ious else 0.0,
    }


def dice(a, b, region=None) -> float:
    """Dice overlap of two areas (shapely geometries or arrays of them), within ``region`` if given."""

    def one(g):
        if g is None:
            return shapely.Polygon()
        if isinstance(g, np.ndarray | list | tuple):
            g = shapely.union_all(np.asarray(g, dtype=object)) if len(g) else shapely.Polygon()
        g = shapely.make_valid(g)
        return g.intersection(region) if region is not None else g

    A, B = one(a), one(b)
    s = A.area + B.area
    return round(2 * A.intersection(B).area / s, 3) if s else 1.0
