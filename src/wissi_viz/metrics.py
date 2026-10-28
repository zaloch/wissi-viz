"""Measurements for annotated or segmented regions: one row per region, every meaningful variable of spatial
image analysis, grouped by prefix so a table stays readable.

    from wissi_viz.metrics import region_metrics, object_metrics
    table = region_metrics(rois, source=slide, objects=cells)   # ROIs: AnnotationLayer / Features / geometries
    cells_table = object_metrics(cells_layer, source=slide)     # per segmented object: shape + intensity

Region columns (µm units when the slide is calibrated):

* ``shape_*``     area, perimeter, circularity, solidity, convexity, extent, eccentricity, major / minor axis,
                  max / min Feret diameter, aspect ratio, holes and hole area, centroid.
* ``int_*``       per channel (fluorescence) or per stain (brightfield: haematoxylin, eosin, DAB optical density,
                  plus R/G/B): mean, sd, median, p10, p90, integrated, positive fraction (Otsu within the region);
                  ``int_tissue_frac`` (brightfield: non-background fraction).
* ``tex_*``       Haralick texture (GLCM over 4 directions at 1, 2 and 4 px of the analysis level): contrast,
                  dissimilarity, homogeneity, energy, correlation, ASM; histogram entropy. ``analysis_mpp`` says
                  at which resolution intensity and texture were measured.
* ``obj_*``       objects inside the region: count, density / mm², per class count, % and density, Shannon and
                  Simpson diversity, richness, evenness; nearest-neighbour distance (all, per class, and class →
                  class); contact fraction (A cells with a B cell within ``contact_um``); Clark–Evans aggregation
                  index; Morisita index; Ripley's L(r) − r at ``radii_um``; mean depth from the region boundary per
                  class; mean and median of every numeric object measurement (markers, areas, …).

Needs the ``[analysis]`` extra (scipy, shapely, scikit-image, pandas).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .annotations import AnnotationLayer

_STAINS = ('hematoxylin', 'eosin', 'dab')


# ------------------------------------------------------------------------------------------------ inputs
def _regions(regions: Any) -> list[tuple[str, Any]]:
    """[(name, shapely geometry)] from a layer, Features, geometry, dict or list."""
    import shapely
    from shapely.geometry import LineString, Point, Polygon
    from . import annio
    if isinstance(regions, AnnotationLayer):
        regions = annio.from_layer(regions)
    if isinstance(regions, annio.Features):
        F = regions
        names = F.class_of()
        name_prop = F.props.get('name')
        out = []
        for i in range(len(F)):
            rs = F.rings(i)
            k = F.kind[i]
            if k == annio.GEO_POLYGON and len(rs[0]) >= 3:
                g = shapely.make_valid(Polygon(rs[0], [h for h in rs[1:] if len(h) >= 3]))
            elif k == annio.GEO_LINE:
                g = LineString(rs[0])
            else:
                g = Point(rs[0][0])
            nm = name_prop[i] if name_prop is not None and name_prop[i] not in (None, '') else (names[i] or f'region {i + 1}')
            out.append((str(nm), g))
        return out
    from .spatial import roi_geometries
    return roi_geometries(regions)


def objects_from_layer(layer: AnnotationLayer, mpp: float | None = None):
    """An AnnotationLayer of segmented objects (cells, nuclei, …) → ObjectTable (centroids, classes, bbox,
    measurement columns, outlines for polygon layers)."""
    from . import annio
    from .objects import ObjectTable
    F = annio.from_layer(layer)
    c = F.centroid()
    outlines = None
    if len(F) and (F.kind == annio.GEO_POLYGON).all() and len(layer._kinds):   # real outlines (not bbox ellipses)
        outlines = [F.coords[F.ring_off[F.feat_ring_off[i]]:F.ring_off[F.feat_ring_off[i] + 1]] for i in range(len(F))]
    cols = {k: np.asarray(v) for k, v in F.props.items() if np.asarray(v).dtype.kind in 'biuf'}
    return ObjectTable(c[:, 0], c[:, 1], F.class_of(), cols, F.bbox(), outlines, mpp or F.mpp, layer.name)


# ------------------------------------------------------------------------------------------------ shape
def _shape(g, mpp: float) -> dict:
    import shapely
    u, u2 = mpp, mpp * mpp
    d: dict[str, float] = {}
    if g.is_empty:
        return d
    if g.geom_type in ('Point', 'MultiPoint'):
        c = g.centroid
        return {'shape_centroid_x_px': c.x, 'shape_centroid_y_px': c.y}
    if g.geom_type in ('LineString', 'MultiLineString'):
        c = g.centroid
        return {'shape_length_um': g.length * u, 'shape_centroid_x_px': c.x, 'shape_centroid_y_px': c.y}
    A, P = g.area, g.length
    hull = g.convex_hull
    minx, miny, maxx, maxy = g.bounds
    d['shape_area_um2'] = A * u2
    d['shape_area_mm2'] = A * u2 / 1e6
    d['shape_perimeter_um'] = P * u
    d['shape_circularity'] = 4 * math.pi * A / P ** 2 if P else np.nan
    d['shape_solidity'] = A / hull.area if hull.area else np.nan
    d['shape_convexity'] = hull.length / P if P else np.nan
    d['shape_extent'] = A / ((maxx - minx) * (maxy - miny)) if maxx > minx and maxy > miny else np.nan
    # exact second moments of area (Green's theorem over shells and holes) → equivalent ellipse, as in scikit-image
    polys = list(getattr(g, 'geoms', [g]))
    m = _area_moments([shapely.geometry.polygon.orient(p) for p in polys if p.geom_type == 'Polygon'])
    if m is not None:
        mu20, mu02, mu11 = m
        tr, det = mu20 + mu02, mu20 * mu02 - mu11 ** 2
        disc = math.sqrt(max(tr * tr / 4 - det, 0))
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 0)
        d['shape_major_axis_um'] = 4 * math.sqrt(l1) * u
        d['shape_minor_axis_um'] = 4 * math.sqrt(l2) * u
        d['shape_eccentricity'] = math.sqrt(max(0.0, 1 - l2 / l1)) if l1 > 0 else 0.0
        d['shape_orientation_deg'] = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))
    mrr = shapely.minimum_rotated_rectangle(g)
    if mrr.geom_type == 'Polygon':
        e = np.asarray(mrr.exterior.coords)
        s = np.hypot(*np.diff(e[:3], axis=0).T)
        d['shape_min_feret_um'] = float(s.min()) * u
        d['shape_aspect_ratio'] = float(s.max() / s.min()) if s.min() > 0 else np.nan
    hv = np.asarray(hull.exterior.coords) if hull.geom_type == 'Polygon' else np.asarray(g.envelope.exterior.coords)
    if len(hv) > 1:
        dd = hv[:, None, :] - hv[None, :, :]
        d['shape_max_feret_um'] = float(np.sqrt((dd ** 2).sum(-1)).max()) * u
    d['shape_holes'] = sum(len(p.interiors) for p in polys if p.geom_type == 'Polygon')
    d['shape_hole_area_um2'] = sum(abs(shapely.area(shapely.Polygon(h))) for p in polys if p.geom_type == 'Polygon'
                                   for h in p.interiors) * u2
    c = g.centroid
    d['shape_centroid_x_px'], d['shape_centroid_y_px'] = c.x, c.y
    return d


def _area_moments(polys) -> tuple | None:
    """Central second moments (mu20, mu02, mu11) of polygons with holes; shells counter-clockwise."""
    A = Sx = Sy = Sxx = Syy = Sxy = 0.0
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            v = np.asarray(ring.coords, np.float64)
            x, y = v[:-1, 0], v[:-1, 1]
            x1, y1 = v[1:, 0], v[1:, 1]
            c = x * y1 - x1 * y
            A += c.sum() / 2
            Sx += ((x + x1) * c).sum() / 6
            Sy += ((y + y1) * c).sum() / 6
            Sxx += ((x * x + x * x1 + x1 * x1) * c).sum() / 12
            Syy += ((y * y + y * y1 + y1 * y1) * c).sum() / 12
            Sxy += ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * c).sum() / 24
    if abs(A) < 1e-12:
        return None
    cx, cy = Sx / A, Sy / A
    return Sxx / A - cx * cx, Syy / A - cy * cy, Sxy / A - cx * cy


# ------------------------------------------------------------------------------------------------ pixels
def _pick_level(source, bounds, max_pixels: float) -> int:
    x0, y0, x1, y1 = bounds
    for i, L in enumerate(source.info.levels):
        if (x1 - x0) / L.downsample * (y1 - y0) / L.downsample <= max_pixels:
            return i
    return len(source.info.levels) - 1


def _read(source, g, max_pixels: float):
    """Pixels of the region's bounding box at a level fitting ``max_pixels``, its mask, and that level's
    downsample. Brightfield → (h, w, 3) uint8; fluorescence → (h, w, C)."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = g.bounds
    lv = _pick_level(source, g.bounds, max_pixels)
    L = source.info.levels[lv]
    ds = L.downsample
    lx0, ly0 = int(max(0, math.floor(x0 / ds))), int(max(0, math.floor(y0 / ds)))
    lx1, ly1 = int(min(L.width, math.ceil(x1 / ds))), int(min(L.height, math.ceil(y1 / ds)))
    w, h = max(1, lx1 - lx0), max(1, ly1 - ly0)
    if source.info.mode == 'rgb':
        img = np.asarray(source.read_region(lv, lx0, ly0, w, h))[..., :3]
    else:
        img = np.stack([np.asarray(source.read_region(lv, lx0, ly0, w, h, c)) for c in range(len(source.info.channels))], -1)
    ox, oy = lx0 * ds, ly0 * ds                      # bbox origin in level-0 px
    scale = w * h / max_pixels
    if scale > 1:                                    # coarsest level still too big: stride
        st = int(math.ceil(math.sqrt(scale)))
        img = img[::st, ::st]
        ds *= st
        h, w = img.shape[:2]
    m = Image.new('L', (w, h), 0)
    dr = ImageDraw.Draw(m)
    for p in getattr(g, 'geoms', [g]):
        if p.geom_type != 'Polygon':
            continue
        dr.polygon([((x - ox) / ds, (y - oy) / ds) for x, y in p.exterior.coords], fill=1)
        for hole in p.interiors:
            dr.polygon([((x - ox) / ds, (y - oy) / ds) for x, y in hole.coords], fill=0)
    return img, np.asarray(m, bool), ds


def _stats(prefix: str, v: np.ndarray, px_area_um2: float, positive: bool = True) -> dict:
    if not len(v):
        return {}
    v = v.astype(np.float64)
    q10, q50, q90 = np.percentile(v, [10, 50, 90])
    d = {f'{prefix}_mean': v.mean(), f'{prefix}_sd': v.std(), f'{prefix}_median': q50, f'{prefix}_p10': q10,
         f'{prefix}_p90': q90, f'{prefix}_integrated': v.sum() * px_area_um2}
    if positive and v.max() > v.min():
        from skimage.filters import threshold_otsu
        d[f'{prefix}_pos_frac'] = float((v > threshold_otsu(v)).mean())
    return d


def _stain_od(rgb: np.ndarray, vectors: dict | None = None) -> dict[str, np.ndarray]:
    """Per-pixel optical density of haematoxylin, eosin and DAB (colour deconvolution: Ruifrok & Johnston vectors,
    or ``vectors`` = {'hematoxylin': rgb, 'eosin': rgb[, 'dab': rgb]} e.g. from spatial.estimate_stain_vectors)."""
    od = -np.log10(np.clip(rgb.astype(np.float64), 1, 255) / 255.0)
    if vectors:
        M = np.array([vectors['hematoxylin'], vectors.get('eosin') or vectors.get('second'),
                      vectors.get('dab') or np.cross(vectors['hematoxylin'], vectors.get('eosin') or vectors.get('second'))], float)
    else:
        M = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    conc = od.reshape(-1, 3) @ np.linalg.pinv(M)
    return {s: np.clip(conc[:, i], 0, None).reshape(rgb.shape[:2]) for i, s in enumerate(_STAINS)}


def _texture(gray: np.ndarray, mask: np.ndarray) -> dict:
    from skimage.feature import graycomatrix, graycoprops
    if mask.sum() < 64:
        return {}
    g = gray.astype(np.float64)
    lo, hi = np.percentile(g[mask], [1, 99])
    q = np.clip((g - lo) / max(hi - lo, 1e-9) * 31, 0, 31).astype(np.uint8) + 1    # 32 levels, 0 = outside
    q[~mask] = 0
    P = graycomatrix(q, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33,
                     symmetric=True, normed=False)
    P = P[1:, 1:].astype(np.float64)                 # drop pairs touching the outside
    P /= max(P.sum(axis=(0, 1), keepdims=True).min(), 1)
    P = P / np.maximum(P.sum(axis=(0, 1), keepdims=True), 1e-12)
    d = {f'tex_{p}': float(graycoprops(P, p).mean()) for p in ('contrast', 'dissimilarity', 'homogeneity', 'energy',
                                                               'correlation', 'ASM')}
    h = np.bincount(q[mask], minlength=33)[1:].astype(np.float64)
    h /= h.sum()
    d['tex_entropy'] = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    return d


def _intensity(source, g, mpp: float, max_pixels: float, texture: bool, stain_vectors) -> dict:
    img, mask, ds = _read(source, g, max_pixels)
    if not mask.any():
        return {}
    a_um2 = (ds * mpp) ** 2
    d: dict[str, float] = {'analysis_mpp': ds * mpp}
    if source.info.mode == 'rgb':
        px = img[mask]
        bg = (px.min(1) > 215) & ((px.max(1) - px.min(1)) < 18)          # white glass
        d['int_tissue_frac'] = float(1 - bg.mean())
        for i, c in enumerate('RGB'):
            d.update(_stats(f'int_{c}', px[:, i], a_um2, positive=False))
        od = _stain_od(img, stain_vectors)
        for s in _STAINS:
            d.update(_stats(f'int_{s}_od', od[s][mask], a_um2))
        gray = od['hematoxylin']
    else:
        names = [c.name or f'ch{i}' for i, c in enumerate(source.info.channels)]
        for i, nm in enumerate(names):
            d.update(_stats(f'int_{_slug(nm)}', img[..., i][mask], a_um2))
        gray = img.astype(np.float64).max(-1)
    if texture:
        d.update(_texture(gray, mask))
    return d


def _slug(s: str) -> str:
    import re
    return re.sub(r'[^A-Za-z0-9]+', '_', str(s)).strip('_') or 'x'


# ------------------------------------------------------------------------------------------------ objects
def _object_metrics_in(g, objs, sel: np.ndarray, mpp: float, area_um2: float, classes: Sequence[str],
                       radii_um: Sequence[float], contact_um: float, quadrat_um: float, max_pairs_classes: int) -> dict:
    from scipy.spatial import cKDTree
    d: dict[str, float] = {}
    n = int(sel.sum())
    d['obj_count'] = n
    a_mm2 = area_um2 / 1e6 if area_um2 else np.nan
    d['obj_density_mm2'] = n / a_mm2 if a_mm2 else np.nan
    if n == 0:
        return d
    xy = objs.xy[sel] * mpp                           # µm
    cl = objs.classes[sel] if objs.classes is not None else np.full(n, '', object)
    counts = {c: int((cl == c).sum()) for c in classes}
    p = np.array([v / n for v in counts.values() if v], float)
    for c in classes:
        k = _slug(c)
        d[f'obj_n_{k}'] = counts[c]
        d[f'obj_pct_{k}'] = 100 * counts[c] / n
        d[f'obj_density_mm2_{k}'] = counts[c] / a_mm2 if a_mm2 else np.nan
    if len(p):
        H = float(-(p * np.log(p)).sum())
        d['obj_shannon'] = H
        d['obj_simpson'] = float(1 - (p ** 2).sum())
        d['obj_richness'] = int(len(p))
        d['obj_evenness'] = H / math.log(len(p)) if len(p) > 1 else np.nan
    tree = cKDTree(xy)
    if n >= 2:
        dist, _ = tree.query(xy, k=2)
        nn = dist[:, 1]
        d['obj_nn_mean_um'] = float(nn.mean())
        d['obj_nn_median_um'] = float(np.median(nn))
        if area_um2:
            d['obj_clark_evans'] = float(nn.mean() / (0.5 * math.sqrt(area_um2 / n)))
            for r in radii_um:                              # Ripley L(r) − r (no edge correction; > 0 = clustered)
                pairs = tree.count_neighbors(tree, r) - n
                K = area_um2 * pairs / (n * (n - 1))
                d[f'obj_ripley_L_minus_r_{int(r)}um'] = math.sqrt(max(K, 0) / math.pi) - r
        # Morisita index over square quadrats inside the region's bbox
        q = quadrat_um
        ix = np.floor((xy[:, 0] - xy[:, 0].min()) / q).astype(np.int64)
        iy = np.floor((xy[:, 1] - xy[:, 1].min()) / q).astype(np.int64)
        ni = np.bincount(ix * (iy.max() + 1) + iy)
        Q = max(1, int(round(area_um2 / (q * q)))) if area_um2 else len(ni)
        d['obj_morisita'] = float(Q * (ni * (ni - 1)).sum() / (n * (n - 1)))
    # per class: nearest-neighbour distance within the class, class → class distances and contacts
    present = [c for c in classes if counts[c]]
    trees = {c: cKDTree(xy[cl == c]) for c in present}
    for c in present:
        pts = xy[cl == c]
        if len(pts) >= 2:
            dd, _ = trees[c].query(pts, k=2)
            d[f'obj_nn_mean_um_{_slug(c)}'] = float(dd[:, 1].mean())
    top = sorted(present, key=lambda c: -counts[c])[:max_pairs_classes]
    for a in top:
        pa = xy[cl == a]
        for b in top:
            if a == b:
                continue
            dd, _ = trees[b].query(pa, k=1)
            d[f'obj_dist_mean_um_{_slug(a)}_to_{_slug(b)}'] = float(dd.mean())
            d[f'obj_contact_frac_{_slug(a)}_with_{_slug(b)}'] = float((dd <= contact_um).mean())
    # depth from the region boundary (infiltration): distance of each object to the outline
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        import shapely
        bnd = g.boundary
        depth = shapely.distance(shapely.points(objs.x[sel], objs.y[sel]), bnd) * mpp
        d['obj_depth_mean_um'] = float(depth.mean())
        for c in present:
            d[f'obj_depth_mean_um_{_slug(c)}'] = float(depth[cl == c].mean())
    # every numeric object measurement: mean and median
    for k, v in list(objs.columns.items())[:60]:
        v = np.asarray(v)
        if v.dtype.kind not in 'biuf':
            continue
        vv = v[sel].astype(np.float64)
        vv = vv[np.isfinite(vv)]
        if len(vv):
            d[f'obj_mean_{_slug(k)}'] = float(vv.mean())
            d[f'obj_median_{_slug(k)}'] = float(np.median(vv))
    return d


# ------------------------------------------------------------------------------------------------ public API
def region_metrics(regions: Any, *, source: Any = None, objects: Any = None, mpp: float | None = None,
                   classes: Sequence[str] | None = None, intensity: bool = True, texture: bool = True,
                   radii_um: Sequence[float] = (25, 50, 100), contact_um: float = 15.0, quadrat_um: float = 50.0,
                   max_pixels: float = 4e6, stain_vectors: dict | None = None, max_pairs_classes: int = 6):
    """One row per region (see module docstring). ``source``: the slide (SlideSource / PyramidSlide) for intensity
    and texture; ``objects``: an ObjectTable or a layer of segmented objects for the ``obj_*`` metrics."""
    from .spatial import _df
    if source is not None and hasattr(source, 'source') and not hasattr(source, 'read_region'):
        source = source.source
    if mpp is None and source is not None:
        mpp = (source.info.mpp or (None,))[0]
    if objects is not None and isinstance(objects, AnnotationLayer):
        objects = objects_from_layer(objects, mpp)
    if mpp is None and objects is not None:
        mpp = objects.mpp
    units = mpp or 1.0
    regs = _regions(regions)
    if objects is not None and classes is None:
        classes = objects.class_names()
    classes = list(classes or [])
    rows = []
    if objects is not None:
        import shapely
        pts = shapely.points(objects.x, objects.y)
    for i, (name, g) in enumerate(regs):
        row: dict[str, Any] = {'region': name, 'index': i, 'type': g.geom_type}
        row.update(_shape(g, units))
        area_um2 = g.area * units * units if g.geom_type in ('Polygon', 'MultiPolygon') else 0.0
        if source is not None and intensity and area_um2:
            row.update(_intensity(source, g, units, max_pixels, texture, stain_vectors))
        if objects is not None and area_um2:
            import shapely
            shapely.prepare(g)
            x0, y0, x1, y1 = g.bounds
            cand = (objects.x >= x0) & (objects.x <= x1) & (objects.y >= y0) & (objects.y <= y1)
            sel = np.zeros(len(objects), bool)
            idx = np.nonzero(cand)[0]
            if len(idx):
                sel[idx] = shapely.contains(g, pts[idx])
            row.update(_object_metrics_in(g, objects, sel, units, area_um2, classes, radii_um, contact_um,
                                          quadrat_um, max_pairs_classes))
        rows.append(row)
    keys: dict[str, None] = {}
    for r in rows:
        keys.update(dict.fromkeys(r))
    return _df({k: [r.get(k, np.nan) for r in rows] for k in keys})


def object_metrics(objects: Any, *, source: Any = None, mpp: float | None = None, intensity: bool = True,
                   level: int = 0, tile: int = 2048):
    """One row per segmented object: shape (area, perimeter, circularity, solidity, eccentricity, axes, Feret)
    computed vectorised, and — with ``source`` — mean / sd intensity per channel or stain inside each object,
    measured tile by tile at ``level`` from a rasterised label image (exact pixel masks, bounded memory)."""
    import shapely
    from . import annio
    from .spatial import _df
    if source is not None and hasattr(source, 'source') and not hasattr(source, 'read_region'):
        source = source.source
    F = annio.from_layer(objects) if isinstance(objects, AnnotationLayer) else objects
    if mpp is None:
        mpp = (source.info.mpp or (None,))[0] if source is not None else F.mpp
    u = mpp or 1.0
    n = len(F)
    polys = np.nonzero(F.kind == annio.GEO_POLYGON)[0]
    geoms = np.empty(n, object)
    if len(polys):
        geoms[polys] = [shapely.Polygon(F.rings(int(i))[0]) for i in polys]
    rest = np.setdiff1d(np.arange(n), polys)
    if len(rest):
        c = F.centroid()
        geoms[rest] = shapely.points(c[rest, 0], c[rest, 1])
    geoms = np.asarray(geoms)
    A, P = shapely.area(geoms), shapely.length(geoms)
    hull = shapely.convex_hull(geoms)
    c = F.centroid()
    out: dict[str, Any] = {'index': np.arange(n), 'class': F.class_of(), 'x_px': c[:, 0], 'y_px': c[:, 1],
                           'area_um2': A * u * u, 'perimeter_um': P * u}
    with np.errstate(divide='ignore', invalid='ignore'):
        out['circularity'] = np.where(P > 0, 4 * np.pi * A / P ** 2, np.nan)
        out['solidity'] = np.where(shapely.area(hull) > 0, A / shapely.area(hull), np.nan)
    mrr = shapely.minimum_rotated_rectangle(geoms)
    ext = shapely.get_coordinates(mrr, return_index=True)
    w = np.full(n, np.nan)
    hgt = np.full(n, np.nan)
    pts, idx = ext
    if len(idx):
        # first three corners of each rectangle → its two side lengths
        first = np.searchsorted(idx, np.arange(n))
        ok = (np.bincount(idx, minlength=n) >= 4)
        f = first[ok]
        w[ok] = np.hypot(*(pts[f + 1] - pts[f]).T)
        hgt[ok] = np.hypot(*(pts[f + 2] - pts[f + 1]).T)
    major, minor = np.fmax(w, hgt), np.fmin(w, hgt)
    out['major_axis_um'] = major * u
    out['minor_axis_um'] = minor * u
    with np.errstate(divide='ignore', invalid='ignore'):
        out['aspect_ratio'] = major / minor
        out['eccentricity'] = np.sqrt(np.clip(1 - (minor / major) ** 2, 0, 1))
    out['max_feret_um'] = np.hypot(major, minor) * u
    for k, v in F.props.items():
        if not k.startswith('__'):
            out[k] = v
    if source is not None and intensity and len(polys):
        out.update(_object_intensity(source, F, polys, level, tile))
    return _df(out)


def _object_intensity(source, F, polys: np.ndarray, level: int, tile: int) -> dict:
    """Per-object mean / sd of each channel (or H / E / DAB OD and grey for brightfield), tile by tile."""
    from PIL import Image, ImageDraw
    L = source.info.levels[level]
    ds = L.downsample
    n = len(F)
    b = F.bbox()
    rgb = source.info.mode == 'rgb'
    names = list(_STAINS) if rgb else [_slug(c.name or f'ch{i}') for i, c in enumerate(source.info.channels)]
    s1 = {k: np.zeros(n) for k in names}
    s2 = {k: np.zeros(n) for k in names}
    cnt = np.zeros(n)
    tx = np.floor((b[polys, 0] + b[polys, 2]) / 2 / ds / tile).astype(int)
    ty = np.floor((b[polys, 1] + b[polys, 3]) / 2 / ds / tile).astype(int)
    for key in np.unique(tx * 100000 + ty):
        mem = polys[(tx * 100000 + ty) == key]
        bx0, by0 = b[mem, 0].min() / ds, b[mem, 1].min() / ds
        bx1, by1 = b[mem, 2].max() / ds, b[mem, 3].max() / ds
        x0, y0 = int(max(0, math.floor(bx0))), int(max(0, math.floor(by0)))
        x1, y1 = int(min(L.width, math.ceil(bx1) + 1)), int(min(L.height, math.ceil(by1) + 1))
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        lab = Image.new('I', (w, h), 0)
        dr = ImageDraw.Draw(lab)
        for j, f in enumerate(mem, 1):
            dr.polygon([((x / ds) - x0, (y / ds) - y0) for x, y in F.rings(int(f))[0]], fill=j)
        lab = np.asarray(lab, np.int64)
        m = lab > 0
        if not m.any():
            continue
        li = lab[m]
        cnt[mem] += np.bincount(li, minlength=len(mem) + 1)[1:]
        if rgb:
            img = np.asarray(source.read_region(level, x0, y0, w, h))[..., :3]
            od = _stain_od(img)
            chans = [od[s] for s in _STAINS]
        else:
            chans = [np.asarray(source.read_region(level, x0, y0, w, h, ci), np.float64) for ci in range(len(names))]
        for k, a in zip(names, chans):
            v = a[m].astype(np.float64)
            s1[k][mem] += np.bincount(li, v, minlength=len(mem) + 1)[1:]
            s2[k][mem] += np.bincount(li, v * v, minlength=len(mem) + 1)[1:]
    out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in names:
            mean = np.where(cnt > 0, s1[k] / cnt, np.nan)
            out[f'{k}_mean'] = mean
            out[f'{k}_sd'] = np.sqrt(np.clip(np.where(cnt > 0, s2[k] / cnt, np.nan) - mean ** 2, 0, None))
    out['pixels'] = cnt
    return out
