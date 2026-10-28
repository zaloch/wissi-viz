"""Spatial analysis for brightfield and multiplex-fluorescence histology.

Every function takes an :class:`~wissi_viz.objects.ObjectTable` (and ROIs where relevant),
returns a result object with

* ``.table``  — a pandas DataFrame (or dict) of numbers,
* ``.layer()`` — an :class:`~wissi_viz.annotations.AnnotationLayer` to overlay on the slide,
* ``.plot()``  — an ECharts option dict (``ui.echart(result.plot())`` in NiceGUI),

and works in microns (the table's ``mpp``). Dependencies: scipy, shapely, scikit-image,
pandas — all BSD (``pip install wissi-viz[analysis]``).

Contents
  nearest_neighbors      distance from each A to the nearest B (+ lines on the slide)
  proximity              B within r of each A (infiltration / engagement)
  distance_to_roi        signed distance to an ROI boundary (inside < 0)
  margin_bands           invasive-margin bands + per-band densities
  infiltration_profile   density vs signed distance to a boundary
  roi_composition        part-of-whole: counts, %, density per ROI and of the whole
  density_grid           hex/square density or ratio maps + Getis-Ord Gi* hotspots
  ripley                 K / L / H / g(r) (uni- or bivariate) with CSR envelopes
  neighborhood_enrichment permutation z-scores of class co-location
  cellular_neighborhoods  kNN composition clustering (CN1…CNk)
  delaunay_graph         cell–cell contact graph / interaction counts
  morans_i               global spatial autocorrelation of a measurement
  h_score / positivity   IHC scoring and marker positivity per ROI
  tissue_mask            tissue detection from a slide (brightfield or fluorescence)
  estimate_stain_vectors Macenko stain vectors for GPU colour deconvolution
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Polygon
from skimage import filters, measure, morphology

from . import plots
from .annotations import POLYGON, AnnotationLayer
from .objects import ObjectTable


def _df(d):
    try:
        import pandas as pd

        return pd.DataFrame(d)
    except ImportError:
        return d


def _tree(xy):
    return cKDTree(xy)


# ============================================================================= ROIs
def roi_geometries(rois: Any, group: bool = False) -> list[tuple[str, Any]]:
    """Normalise ROIs to ``[(name, shapely geometry)]``.

    Accepts an :class:`AnnotationLayer` (polygons; name = ``properties['name']`` or class), a shapely
    geometry, ``{name: coords|geometry}``, or ``[(name, coords|geometry)]``. Coordinates: level-0 px.
    ``group=True`` unions features with the same name/class.
    """
    out: list[tuple[str, Any]] = []
    if isinstance(rois, AnnotationLayer):
        c = rois.compiled()
        inv = {v: k for k, v in rois._class_index.items()}
        for f in range(c.n):
            if c.kinds[f] != POLYGON:
                continue
            r0, r1 = c.feat_ring_off[f], c.feat_ring_off[f + 1]
            rings = [c.coords[c.ring_off[r] : c.ring_off[r + 1]].astype(np.float64) for r in range(r0, r1)]
            if len(rings[0]) < 3:
                continue
            g = shapely.make_valid(Polygon(rings[0], [h for h in rings[1:] if len(h) >= 3]))
            p = rois.properties(f)
            out.append((str(p.get('name') or inv.get(int(c.cls[f]), f'ROI {f + 1}')), g))
    elif hasattr(rois, 'geom_type'):
        out.append(('ROI', rois))
    else:
        items = rois.items() if isinstance(rois, dict) else rois
        for k, v in items:
            out.append((str(k), v if hasattr(v, 'geom_type') else shapely.make_valid(Polygon(np.asarray(v, float)))))
    if group:
        merged: dict[str, Any] = {}
        for k, g in out:
            merged[k] = g if k not in merged else merged[k].union(g)
        out = list(merged.items())
    return out


def _union(rois: Any):
    gs = [g for _, g in roi_geometries(rois)]
    return shapely.union_all(gs) if gs else None


def _inside(geom, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    shapely.prepare(geom)
    return shapely.contains_xy(geom, x, y)


def _geom_layer(
    items: Sequence[tuple[str, Any]],
    name: str,
    values: Sequence[float] | None = None,
    cmap: str = 'viridis',
    label: str | None = None,
    **style,
) -> AnnotationLayer:
    L = AnnotationLayer(name=name, **style)
    for k, g in items:
        for poly in getattr(g, 'geoms', [g]):
            if poly.geom_type != 'Polygon' or poly.is_empty:
                continue
            L.add_polygon(
                np.asarray(poly.exterior.coords)[:-1],
                holes=[np.asarray(h.coords)[:-1] for h in poly.interiors],
                cls=k,
                props={'name': k},
            )
    if values is not None:
        vals = []
        for (_, g), v in zip(items, values):
            vals += [v] * sum(1 for p in getattr(g, 'geoms', [g]) if p.geom_type == 'Polygon' and not p.is_empty)
        L.color_by(np.asarray(vals, float), cmap=cmap, label=label)
    return L


# ============================================================================= nearest neighbour
@dataclass
class NNResult:
    source: str
    target: str
    src_idx: np.ndarray
    tgt_idx: np.ndarray
    dist_um: np.ndarray
    objs: ObjectTable = field(repr=False)

    @property
    def table(self):
        return _df({'source_index': self.src_idx, 'target_index': self.tgt_idx, 'distance_um': self.dist_um})

    def summary(self, within_um: Sequence[float] = (10, 20, 50, 100)) -> dict:
        d = self.dist_um[np.isfinite(self.dist_um)]
        out = {
            'source': self.source,
            'target': self.target,
            'n_source': int(len(self.dist_um)),
            'median_um': float(np.median(d)) if d.size else np.nan,
            'mean_um': float(d.mean()) if d.size else np.nan,
            'q25_um': float(np.percentile(d, 25)) if d.size else np.nan,
            'q75_um': float(np.percentile(d, 75)) if d.size else np.nan,
        }
        for w in within_um:
            out[f'frac_within_{w}um'] = float((d <= w).mean()) if d.size else np.nan
        return out

    def layer(
        self,
        name: str | None = None,
        cmap: str = 'viridis_r',
        max_lines: int = 300_000,
        max_um: float | None = None,
        line_width: float = 1.5,
    ) -> AnnotationLayer:
        """A line from every source object to its nearest target, coloured by distance."""
        o = self.objs
        sel = np.isfinite(self.dist_um)
        if max_um is not None:
            sel &= self.dist_um <= max_um
        idx = np.nonzero(sel)[0]
        if len(idx) > max_lines:
            idx = np.random.default_rng(0).choice(idx, max_lines, replace=False)
        L = AnnotationLayer(
            name=name or f'NN {self.source} → {self.target}',
            line_width=line_width,
            fill_opacity=0,
            detail_min_px=4,
            overview='none',
        )
        s, t = self.src_idx[idx], self.tgt_idx[idx]
        for a, b in zip(s, t):
            L.add_polyline([[o.x[a], o.y[a]], [o.x[b], o.y[b]]])
        L.color_by(self.dist_um[idx], cmap=cmap, label=f'NN distance {self.source}→{self.target} (µm)')
        return L

    def plot(self, bins: int = 40, max_um: float | None = None):
        d = self.dist_um[np.isfinite(self.dist_um)]
        return plots.histogram(
            d if max_um is None else d[d <= max_um],
            bins=bins,
            title=f'Nearest {self.target} to each {self.source}',
            xlabel='distance (µm)',
        )


def nearest_neighbors(objs: ObjectTable, source: Any, target: Any, max_um: float | None = None) -> NNResult:
    """Distance (µm) from each ``source`` object to the nearest ``target`` object (self excluded)."""
    ms, mt = objs.mask(source), objs.mask(target)
    si, ti = np.nonzero(ms)[0], np.nonzero(mt)[0]
    mpp = objs._mpp()
    if not len(ti) or not len(si):
        return NNResult(str(source), str(target), si, np.full(len(si), -1), np.full(len(si), np.nan), objs)
    tree = _tree(objs.xy[ti])
    same = np.array_equal(ms, mt)
    k = 2 if same else 1
    ub = (max_um / mpp) if max_um else np.inf
    d, j = tree.query(objs.xy[si], k=k, distance_upper_bound=ub)
    if same:
        d, j = d[:, 1], j[:, 1]
    found = np.isfinite(d)
    tgt = np.full(len(si), -1)
    tgt[found] = ti[j[found]]
    return NNResult(str(source), str(target), si, tgt, np.where(found, d * mpp, np.nan), objs)


# ============================================================================= proximity
@dataclass
class ProximityResult:
    source: str
    target: str
    radius_um: float
    src_idx: np.ndarray
    counts: np.ndarray
    objs: ObjectTable = field(repr=False)

    @property
    def table(self):
        return _df({'source_index': self.src_idx, f'n_{self.target}_within_{self.radius_um:g}um': self.counts})

    def summary(self) -> dict:
        c = self.counts
        area_mm2 = np.pi * (self.radius_um / 1000) ** 2
        return {
            'source': self.source,
            'target': self.target,
            'radius_um': self.radius_um,
            'n_source': int(len(c)),
            'frac_with_any': float((c > 0).mean()) if len(c) else np.nan,
            'mean_count': float(c.mean()) if len(c) else np.nan,
            'mean_density_per_mm2': float(c.mean() / area_mm2) if len(c) else np.nan,
        }

    def layer(self, name: str | None = None, cmap: str = 'magma', point_size: float = 6) -> AnnotationLayer:
        o = self.objs
        L = AnnotationLayer.from_points(
            o.xy[self.src_idx],
            name=name or f'{self.target} within {self.radius_um:g} µm of {self.source}',
            point_size=point_size,
        )
        L.color_by(self.counts.astype(float), cmap=cmap, label=f'# {self.target} within {self.radius_um:g} µm')
        return L

    def plot(self):
        return plots.bar_counts(
            self.counts,
            title=f'{self.target} within {self.radius_um:g} µm of each {self.source}',
            xlabel=f'# {self.target}',
        )


def proximity(objs: ObjectTable, source: Any, target: Any, radius_um: float) -> ProximityResult:
    """For each ``source`` object, the number of ``target`` objects within ``radius_um`` (self excluded)."""
    ms, mt = objs.mask(source), objs.mask(target)
    si, ti = np.nonzero(ms)[0], np.nonzero(mt)[0]
    if not len(si) or not len(ti):
        return ProximityResult(str(source), str(target), radius_um, si, np.zeros(len(si), int), objs)
    r = radius_um / objs._mpp()
    counts = np.asarray(_tree(objs.xy[ti]).query_ball_point(objs.xy[si], r, return_length=True))
    if np.array_equal(ms, mt):
        counts = counts - 1
    return ProximityResult(str(source), str(target), radius_um, si, counts, objs)


# ============================================================================= ROI distance / margins
def distance_to_roi(objs: ObjectTable, roi: Any) -> np.ndarray:
    """Signed distance (µm) from every object to the ROI boundary: negative inside, positive outside."""
    g = _union(roi) if not hasattr(roi, 'geom_type') else roi
    pts = shapely.points(objs.x, objs.y)
    d = shapely.distance(g.boundary, pts) * objs._mpp()
    return np.where(_inside(g, objs.x, objs.y), -d, d)


@dataclass
class BandsResult:
    edges_um: list[float]
    bands: list[tuple[str, Any]]
    table: Any
    plot_data: dict

    def layer(self, name: str = 'Margin bands', cmap: str = 'coolwarm', fill_opacity: float = 0.25) -> AnnotationLayer:
        centers = [(a + b) / 2 for a, b in zip(self.edges_um[:-1], self.edges_um[1:])]
        return _geom_layer(
            self.bands,
            name,
            values=centers,
            cmap=cmap,
            label='distance to boundary (µm)',
            fill_opacity=fill_opacity,
            line_width=1,
            detail_min_px=0,
        )

    def plot(self):
        return plots.grouped_bars(
            self.plot_data['bands'], self.plot_data['series'], title='Density per margin band', ylabel='cells / mm²'
        )


def margin_bands(
    roi: Any,
    edges_um: Sequence[float] = (-200, -100, 0, 100, 200),
    mpp: float | None = None,
    objs: ObjectTable | None = None,
    within: Any = None,
    classes: Sequence[str] | None = None,
) -> BandsResult:
    """Bands at signed distances from an ROI boundary (e.g. invasive margin −200…+200 µm).

    With ``objs``: per-band counts and densities (cells/mm²) per class. ``within`` clips the bands
    (e.g. to the tissue mask) so densities use real tissue area.
    """
    g = _union(roi) if not hasattr(roi, 'geom_type') else roi
    mpp = mpp or (objs.mpp if objs is not None else None)
    if not mpp:
        raise ValueError('margin_bands needs mpp')
    clip = None if within is None else (_union(within) if not hasattr(within, 'geom_type') else within)
    edges = sorted(edges_um)
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        outer = g.buffer(b / mpp) if b != 0 else g
        inner = g.buffer(a / mpp) if a != 0 else g
        band = outer.difference(inner)
        if clip is not None:
            band = band.intersection(clip)
        lab = f'{a:+g}…{b:+g} µm'
        bands.append((lab, band))
    rows = []
    series: dict[str, list] = {}
    if objs is not None:
        d = distance_to_roi(objs, g)
        inside_clip = _inside(clip, objs.x, objs.y) if clip is not None else np.ones(len(objs), bool)
        cls = classes or objs.class_names() or ['objects']
        for (lab, band), a, b in zip(bands, edges[:-1], edges[1:]):
            area = band.area * (mpp / 1000) ** 2
            m = (d >= a) & (d < b) & inside_clip
            row = {'band': lab, 'from_um': a, 'to_um': b, 'area_mm2': area, 'n_total': int(m.sum())}
            for c in cls:
                n = int((m & objs.mask(c)).sum()) if objs.classes is not None else int(m.sum())
                row[f'n {c}'] = n
                row[f'density {c} /mm2'] = n / area if area > 0 else np.nan
                series.setdefault(c, []).append(row[f'density {c} /mm2'])
            rows.append(row)
    return BandsResult(edges, bands, _df(rows), {'bands': [b[0] for b in bands], 'series': series})


@dataclass
class ProfileResult:
    centers_um: np.ndarray
    density: dict[str, np.ndarray]
    table: Any

    def plot(self):
        return plots.lines(
            self.centers_um,
            self.density,
            title='Density vs distance to boundary',
            xlabel='signed distance (µm, inside < 0)',
            ylabel='cells / mm²',
            vline=0,
        )


def infiltration_profile(
    objs: ObjectTable,
    roi: Any,
    classes: Sequence[str] | None = None,
    range_um: tuple[float, float] = (-500, 500),
    bin_um: float = 25,
    within: Any = None,
) -> ProfileResult:
    """Cell density (/mm²) in thin bands as a function of signed distance to the ROI boundary."""
    edges = np.arange(range_um[0], range_um[1] + bin_um, bin_um)
    br = margin_bands(roi, edges, objs=objs, within=within, classes=classes)
    t = br.table
    get = (
        (lambda c: np.asarray(t[f'density {c} /mm2']))
        if hasattr(t, 'columns')
        else (lambda c: np.asarray([r[f'density {c} /mm2'] for r in t]))
    )
    cls = classes or objs.class_names()
    dens = {c: get(c) for c in cls}
    centers = (edges[:-1] + edges[1:]) / 2
    return ProfileResult(centers, dens, _df({'center_um': centers, **{f'{c} /mm2': v for c, v in dens.items()}}))


# ============================================================================= composition (part of whole)
@dataclass
class CompositionResult:
    table: Any
    rois: list[tuple[str, Any]]
    classes: list[str]
    whole_name: str

    def layer(self, name: str = 'ROIs', value: str | None = None, cmap: str = 'viridis') -> AnnotationLayer:
        """ROI outlines; ``value`` (a column, e.g. 'density CD8 /mm2') colours them as a choropleth."""
        if value is None:
            return _geom_layer(self.rois, name, fill_opacity=0.08, line_width=2.5, detail_min_px=0)
        t = self.table
        vals = [float(t.loc[t['roi'] == k, value].iloc[0]) if hasattr(t, 'loc') else np.nan for k, _ in self.rois]
        return _geom_layer(
            self.rois, name, values=vals, cmap=cmap, label=value, fill_opacity=0.35, line_width=2, detail_min_px=0
        )

    def plot(self, kind: str = 'percent'):
        t = self.table
        rows = list(t.to_dict('records') if hasattr(t, 'to_dict') else t)
        if kind == 'sunburst':
            return plots.sunburst(rows, self.classes, self.whole_name)
        key = {'percent': '% {c}', 'density': 'density {c} /mm2', 'count': 'n {c}'}[kind]
        return plots.stacked_bars(
            [r['roi'] for r in rows],
            {c: [r.get(key.format(c=c), 0) for r in rows] for c in self.classes},
            title={
                'percent': 'Composition per ROI (%)',
                'density': 'Density per ROI (cells/mm²)',
                'count': 'Cells per ROI',
            }[kind],
            percent=(kind == 'percent'),
        )


def roi_composition(
    objs: ObjectTable,
    rois: Any,
    classes: Sequence[str] | None = None,
    whole: Any = None,
    whole_name: str = 'Whole',
    group: bool = False,
) -> CompositionResult:
    """Counts, % of ROI, density (/mm²) and **% of whole** per class for each ROI.

    ``whole``: the reference region (e.g. the tissue mask); default = all objects / union of ROIs.
    Rows: one per ROI plus the whole. ``% of whole {c}`` = share of all class-c objects in that ROI;
    ``area % of whole`` = ROI area share.
    """
    mpp = objs._mpp()
    items = roi_geometries(rois, group=group)
    cls = list(classes or objs.class_names() or [])
    wg = None
    if whole is not None:
        wg = _union(whole) if not hasattr(whole, 'geom_type') else whole
    in_whole = _inside(wg, objs.x, objs.y) if wg is not None else np.ones(len(objs), bool)
    whole_area = (wg.area if wg is not None else _union(rois).area) * (mpp / 1000) ** 2
    masks = {c: objs.mask(c) for c in cls}
    totals = {c: int((masks[c] & in_whole).sum()) for c in cls}
    rows = []
    for name, g in [*items, (whole_name, wg)]:
        inside = in_whole if g is None else (_inside(g, objs.x, objs.y) & in_whole)
        area = whole_area if g is None else g.area * (mpp / 1000) ** 2
        n = int(inside.sum())
        row = {
            'roi': name,
            'area_mm2': area,
            'area % of whole': 100 * area / whole_area if whole_area else np.nan,
            'n_objects': n,
            'density /mm2': n / area if area else np.nan,
        }
        for c in cls:
            k = int((masks[c] & inside).sum())
            row[f'n {c}'] = k
            row[f'% {c}'] = 100 * k / n if n else 0.0
            row[f'density {c} /mm2'] = k / area if area else np.nan
            row[f'% of whole {c}'] = 100 * k / totals[c] if totals[c] else 0.0
        rows.append(row)
    return CompositionResult(_df(rows), items, cls, whole_name)


# ============================================================================= density grids / hotspots
@dataclass
class GridResult:
    cells: list[np.ndarray]
    centers: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    label: str
    cell_um: float
    gi_z: np.ndarray | None = None

    @property
    def table(self):
        d = {'x_px': self.centers[:, 0], 'y_px': self.centers[:, 1], 'value': self.values, 'count': self.counts}
        if self.gi_z is not None:
            d['gi_z'] = self.gi_z
        return _df(d)

    def layer(
        self,
        name: str | None = None,
        cmap: str = 'inferno',
        fill_opacity: float = 0.55,
        log: bool = False,
        min_value: float | None = None,
    ) -> AnnotationLayer:
        keep = np.isfinite(self.values) & (self.values > (min_value if min_value is not None else -np.inf))
        L = AnnotationLayer.from_polygons(
            [c for c, k in zip(self.cells, keep) if k],
            name=name or self.label,
            fill_opacity=fill_opacity,
            line_width=0,
            detail_min_px=0,
        )
        L.color_by(self.values[keep], cmap=cmap, label=self.label, log=log)
        return L

    def hotspot_layer(self, name: str | None = None, z: float = 1.96, fill_opacity: float = 0.5) -> AnnotationLayer:
        """Getis-Ord Gi* hot (z > 1.96) / cold (z < −1.96) spots."""
        if self.gi_z is None:
            raise ValueError('run density_grid(..., hotspots=True)')
        hot, cold = self.gi_z > z, self.gi_z < -z
        L = AnnotationLayer(
            name=name or f'Hotspots ({self.label})',
            classes={'Hotspot': '#ff1744', 'Coldspot': '#2979ff'},
            fill_opacity=fill_opacity,
            line_width=0,
            detail_min_px=0,
        )
        for c, h, k in zip(self.cells, hot, cold):
            if h or k:
                L.add_polygon(c, cls='Hotspot' if h else 'Coldspot')
        return L

    def plot(self):
        v = self.values[np.isfinite(self.values)]
        return plots.histogram(v, bins=40, title=self.label, xlabel=self.label)


def _grid(x0, y0, x1, y1, s, shape):
    if shape == 'hex':
        w, h = np.sqrt(3) * s, 1.5 * s  # pointy-top hexagons, circumradius s
        rows = np.arange(y0, y1 + h, h)
        cs, polys = [], []
        ang = np.deg2rad(np.arange(6) * 60 + 30)
        unit = np.stack([np.cos(ang), np.sin(ang)], 1) * s
        for i, cy in enumerate(rows):
            xs = np.arange(x0 + (w / 2 if i % 2 else 0), x1 + w, w)
            for cx in xs:
                cs.append((cx, cy))
        cs = np.asarray(cs)
        polys = [unit + c for c in cs]
        return cs, polys
    xs = np.arange(x0 + s / 2, x1 + s, s)
    ys = np.arange(y0 + s / 2, y1 + s, s)
    cs = np.array([(x, y) for y in ys for x in xs])
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * s / 2
    return cs, [sq + c for c in cs]


def density_grid(
    objs: ObjectTable,
    cls: Any = None,
    cell_um: float = 50,
    shape: str = 'hex',
    denominator: Any = None,
    within: Any = None,
    min_objects: int = 1,
    hotspots: bool = False,
    hotspot_radius_cells: float = 1.5,
) -> GridResult:
    """Density (cells/mm²) of ``cls`` on a hex (or square) grid; with ``denominator`` the ratio
    ``#cls / #denominator`` (e.g. CD8 / all cells, Ki67+ / Tumor). Optional Getis-Ord Gi* hotspots."""
    mpp = objs._mpp()
    # hex: flat-to-flat width = cell_um → circumradius R = cell_um / √3 ; square: side = cell_um
    R = cell_um / mpp / np.sqrt(3) if shape == 'hex' else cell_um / mpp
    x0, y0, x1, y1 = objs.x.min(), objs.y.min(), objs.x.max(), objs.y.max()
    centers, polys = _grid(x0, y0, x1, y1, R, shape)
    _, nearest = _tree(centers).query(objs.xy)  # Voronoi cells of a hex/square lattice are the hexes/squares
    num_m = objs.mask(cls)
    all_counts = np.bincount(nearest, minlength=len(centers))
    num = np.bincount(nearest[num_m], minlength=len(centers)).astype(float)
    cell_px2 = 1.5 * np.sqrt(3) * R**2 if shape == 'hex' else R**2
    area_mm2 = cell_px2 * (mpp / 1000) ** 2
    s = R
    if denominator is not None:
        den = np.bincount(nearest[objs.mask(denominator)], minlength=len(centers)).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.where(den > 0, num / den, np.nan)
        label = f'{cls} / {denominator}'
    else:
        vals = num / area_mm2
        label = f'{cls or "cells"} / mm²'
    keep = all_counts >= min_objects
    if within is not None:
        g = _union(within) if not hasattr(within, 'geom_type') else within
        keep &= _inside(g, centers[:, 0], centers[:, 1])
    idx = np.nonzero(keep)[0]
    gi = None
    if hotspots:
        gi = _getis_ord(centers[idx], np.nan_to_num(vals[idx]), hotspot_radius_cells * 2 * s)
    return GridResult([polys[i] for i in idx], centers[idx], vals[idx], all_counts[idx], label, cell_um, gi)


def _getis_ord(xy: np.ndarray, v: np.ndarray, radius: float) -> np.ndarray:
    """Getis-Ord Gi* z-scores with binary distance weights (self included)."""
    n = len(v)
    if n < 3:
        return np.zeros(n)
    tree = _tree(xy)
    nb = tree.query_ball_point(xy, radius)
    xbar, s = v.mean(), v.std()
    if s == 0:
        return np.zeros(n)
    z = np.empty(n)
    for i, js in enumerate(nb):
        w = len(js)
        num = v[js].sum() - xbar * w
        den = s * np.sqrt((n * w - w * w) / (n - 1))
        z[i] = num / den if den > 0 else 0.0
    return z


# ============================================================================= Ripley / pair correlation
@dataclass
class RipleyResult:
    r_um: np.ndarray
    K: np.ndarray
    L: np.ndarray
    g: np.ndarray
    L_lo: np.ndarray | None
    L_hi: np.ndarray | None
    label: str

    @property
    def H(self):
        return self.L - self.r_um

    @property
    def table(self):
        d = {'r_um': self.r_um, 'K': self.K, 'L': self.L, 'H': self.H, 'g': self.g}
        if self.L_lo is not None:
            d.update({'L_csr_lo': self.L_lo, 'L_csr_hi': self.L_hi})
        return _df(d)

    def plot(self, what: str = 'H'):
        if what == 'g':
            return plots.lines(
                self.r_um,
                {'g(r)': self.g},
                title=f'Pair correlation {self.label}',
                xlabel='r (µm)',
                ylabel='g(r)',
                hline=1,
            )
        y = self.H if what == 'H' else self.L
        band = None
        if self.L_lo is not None:
            band = (self.L_lo - (self.r_um if what == 'H' else 0), self.L_hi - (self.r_um if what == 'H' else 0))
        return plots.lines(
            self.r_um,
            {f'{what}(r)': y},
            band=band,
            band_label='CSR 95% envelope',
            title=f"Ripley's {what} {self.label}",
            xlabel='r (µm)',
            ylabel='L(r) − r (µm)' if what == 'H' else 'L(r) (µm)',
            hline=0 if what == 'H' else None,
        )


def ripley(
    objs: ObjectTable,
    a: Any,
    b: Any = None,
    radii_um: Sequence[float] | None = None,
    area: Any = None,
    n_sim: int = 39,
    max_points: int = 20000,
    seed: int = 0,
) -> RipleyResult:
    """Ripley's K/L (univariate, or bivariate a→b) with border edge-correction and a CSR envelope.

    ``area``: study region (ROI / tissue); default = convex hull of the points. Envelope: ``n_sim``
    simulations of complete spatial randomness of the same intensity in ``area`` (39 → ~95% pointwise).
    """
    mpp = objs._mpp()
    rng = np.random.default_rng(seed)
    ma = objs.mask(a)
    mb = ma if b is None else objs.mask(b)
    A_xy, B_xy = objs.xy[ma], objs.xy[mb]
    if area is None:
        g = shapely.convex_hull(shapely.multipoints(np.vstack([A_xy, B_xy])))
    else:
        g = _union(area) if not hasattr(area, 'geom_type') else area
        ins = _inside(g, A_xy[:, 0], A_xy[:, 1])
        A_xy = A_xy[ins]
        B_xy = B_xy[_inside(g, B_xy[:, 0], B_xy[:, 1])]
    if radii_um is None:
        side = np.sqrt(g.area) * mpp
        radii_um = np.linspace(0, min(250.0, side / 4), 26)[1:]
    r_px = np.asarray(radii_um, float) / mpp
    univariate = b is None

    def k_func(P, Q):
        if len(P) > max_points:
            P = P[rng.choice(len(P), max_points, replace=False)]
        if not len(P) or not len(Q):
            return np.full(len(r_px), np.nan)
        dist_b = shapely.distance(g.boundary, shapely.points(P[:, 0], P[:, 1]))
        tq = _tree(Q)
        tp = _tree(P)
        sdm = tp.sparse_distance_matrix(tq, r_px.max(), output_type='coo_matrix')
        di, dd = sdm.row, sdm.data
        if univariate:
            nz = dd > 0
            di, dd = di[nz], dd[nz]
        lam = len(Q) / g.area
        K = np.empty(len(r_px))
        for k, r in enumerate(r_px):
            elig = dist_b >= r
            ne = elig.sum()
            K[k] = ((dd <= r) & elig[di]).sum() / (ne * lam) if ne else np.nan
        return K

    K = k_func(A_xy, B_xy)
    L = np.sqrt(np.maximum(K, 0) / np.pi)
    lo = hi = None
    if n_sim:
        minx, miny, maxx, maxy = g.bounds

        def csr(n):
            out = np.empty((0, 2))
            while len(out) < n:
                p = rng.uniform([minx, miny], [maxx, maxy], (int(n * 1.5) + 16, 2))
                out = np.vstack([out, p[_inside(g, p[:, 0], p[:, 1])]])
            return out[:n]

        sims = []
        n_a = min(len(A_xy), max_points // 2)
        for _ in range(n_sim):
            P = csr(n_a)
            Q = P if univariate else csr(min(len(B_xy), max_points))
            sims.append(np.sqrt(np.maximum(k_func(P, Q), 0) / np.pi))
        sims = np.asarray(sims)
        lo, hi = np.nanmin(sims, 0), np.nanmax(sims, 0)
    # pair correlation from dK/dr
    rr = np.concatenate([[0], r_px])
    KK = np.concatenate([[0], K])
    dK = np.diff(KK) / np.diff(rr)
    rm = (rr[1:] + rr[:-1]) / 2
    gpc = dK / (2 * np.pi * rm)
    label = f'{a}' if univariate else f'{a} → {b}'
    return RipleyResult(
        np.asarray(radii_um, float),
        K * mpp**2,
        L * mpp,
        gpc,
        None if lo is None else lo * mpp,
        None if hi is None else hi * mpp,
        label,
    )


# ============================================================================= co-location / neighbourhoods
@dataclass
class EnrichmentResult:
    classes: list[str]
    z: np.ndarray
    observed: np.ndarray
    expected: np.ndarray

    @property
    def table(self):
        return _df({'class': self.classes, **{c: self.z[:, j] for j, c in enumerate(self.classes)}})

    def plot(self):
        return plots.heatmap(
            self.z, self.classes, self.classes, title='Neighbourhood enrichment (z-score)', diverging=True
        )


def neighborhood_enrichment(
    objs: ObjectTable, radius_um: float = 20, classes: Sequence[str] | None = None, n_perm: int = 199, seed: int = 0
) -> EnrichmentResult:
    """Are classes A and B neighbours (within ``radius_um``) more/less often than by chance?
    z-scores against random relabelling (positive = co-located / attraction, negative = avoidance)."""
    cls = list(classes or objs.class_names())
    code = np.full(len(objs), -1)
    for i, c in enumerate(cls):
        code[objs.mask(c)] = i
    keep = code >= 0
    xy, code = objs.xy[keep], code[keep]
    pairs = _tree(xy).query_pairs(radius_um / objs._mpp(), output_type='ndarray')
    K = len(cls)

    def count(c):
        a, b = c[pairs[:, 0]], c[pairs[:, 1]]
        m = np.bincount(a * K + b, minlength=K * K).reshape(K, K)
        return m + m.T - np.diag(np.diag(m))

    obs = count(code)
    rng = np.random.default_rng(seed)
    perm = np.stack([count(rng.permutation(code)) for _ in range(n_perm)])
    mu, sd = perm.mean(0), perm.std(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(sd > 0, (obs - mu) / sd, 0.0)
    return EnrichmentResult(cls, z, obs, mu)


@dataclass
class NeighborhoodResult:
    labels: np.ndarray
    composition: np.ndarray
    classes: list[str]
    names: list[str]
    objs: ObjectTable = field(repr=False)

    @property
    def table(self):
        return _df(
            {
                'neighborhood': self.names,
                **{c: self.composition[:, j] for j, c in enumerate(self.classes)},
                'n_cells': np.bincount(self.labels, minlength=len(self.names)),
            }
        )

    def layer(self, name: str = 'Cellular neighborhoods', point_size: float = 5, **style) -> AnnotationLayer:
        return AnnotationLayer.from_points(
            self.objs.xy, classes=[self.names[i] for i in self.labels], name=name, point_size=point_size, **style
        )

    def plot(self):
        return plots.heatmap(self.composition, self.names, self.classes, title='Neighborhood composition (fraction)')


def cellular_neighborhoods(
    objs: ObjectTable, k: int = 10, n_neighborhoods: int = 6, classes: Sequence[str] | None = None, seed: int = 0
) -> NeighborhoodResult:
    """Cellular neighbourhoods (Schürch et al. 2020): class composition of each cell's k nearest
    neighbours (incl. itself), clustered with k-means into ``n_neighborhoods`` recurring niches."""
    cls = list(classes or objs.class_names())
    code = np.full(len(objs), len(cls))
    for i, c in enumerate(cls):
        code[objs.mask(c)] = i
    _, nn = _tree(objs.xy).query(objs.xy, k=min(k, len(objs)))
    comp = np.zeros((len(objs), len(cls) + 1))
    np.add.at(comp, (np.repeat(np.arange(len(objs)), nn.shape[1]), code[nn].ravel()), 1)
    comp = comp[:, : len(cls)] / nn.shape[1]
    _, lab = kmeans2(comp, n_neighborhoods, minit='++', seed=seed)
    cent = np.stack(
        [comp[lab == i].mean(0) if (lab == i).any() else np.zeros(len(cls)) for i in range(n_neighborhoods)]
    )
    order = np.lexsort((-cent.max(1), cent.argmax(1)))  # group by dominant class
    remap = np.empty(n_neighborhoods, int)
    remap[order] = np.arange(n_neighborhoods)
    lab = remap[lab]
    cent = cent[order]
    overall = comp.mean(0) + 1e-9
    names = []
    for i in range(n_neighborhoods):
        dom = int(np.argmax(cent[i]))
        enr = cent[i] / overall
        enr[dom] = 0
        top = int(np.argmax(enr))
        tag = f'{cls[dom]}' + (f' + {cls[top]}↑' if enr[top] > 1.25 and cent[i, top] > 0.05 else '')
        names.append(f'CN{i + 1}: {tag}')
    seen: dict[str, int] = {}
    for i, nm in enumerate(names):  # keep labels unique
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            names[i] = f'{nm} ({seen[nm]})'
    return NeighborhoodResult(lab, cent, cls, names, objs)


@dataclass
class GraphResult:
    edges: np.ndarray
    length_um: np.ndarray
    objs: ObjectTable = field(repr=False)
    classes: list[str] = field(default_factory=list)

    @property
    def table(self):
        return _df({'i': self.edges[:, 0], 'j': self.edges[:, 1], 'length_um': self.length_um})

    def interactions(self):
        """Counts of edges per class pair (contact matrix)."""
        cls = self.classes or self.objs.class_names()
        code = {c: i for i, c in enumerate(cls)}
        K = len(cls)
        a = np.array([code.get(c, -1) for c in self.objs.classes[self.edges[:, 0]]])
        b = np.array([code.get(c, -1) for c in self.objs.classes[self.edges[:, 1]]])
        ok = (a >= 0) & (b >= 0)
        m = np.bincount(a[ok] * K + b[ok], minlength=K * K).reshape(K, K)
        return m + m.T - np.diag(np.diag(m)), cls

    def degree(self) -> np.ndarray:
        return np.bincount(self.edges.ravel(), minlength=len(self.objs))

    def layer(
        self, name: str = 'Contact graph', by: str = 'pair', max_edges: int = 400_000, **style
    ) -> AnnotationLayer:
        """Edges as lines, coloured by class pair (``by='pair'``) or by length (``by='length'``)."""
        o = self.objs
        e = self.edges
        if len(e) > max_edges:
            e = e[np.random.default_rng(0).choice(len(e), max_edges, replace=False)]
        style.setdefault('line_width', 1)
        style.setdefault('detail_min_px', 3)
        style.setdefault('overview', 'none')
        L = AnnotationLayer(name=name, fill_opacity=0, **style)
        if by == 'pair' and o.classes is not None:
            for i, j in e:
                a, b = sorted((o.classes[i], o.classes[j]))
                L.add_polyline([[o.x[i], o.y[i]], [o.x[j], o.y[j]]], cls=f'{a}–{b}')
        else:
            for i, j in e:
                L.add_polyline([[o.x[i], o.y[i]], [o.x[j], o.y[j]]])
            ln = np.hypot(o.x[e[:, 0]] - o.x[e[:, 1]], o.y[e[:, 0]] - o.y[e[:, 1]]) * o._mpp()
            L.color_by(ln, cmap='viridis', label='edge length (µm)')
        return L

    def plot(self):
        m, cls = self.interactions()
        return plots.heatmap(m, cls, cls, title='Cell–cell contacts (Delaunay edges)')


def delaunay_graph(objs: ObjectTable, max_edge_um: float = 30, classes: Sequence[str] | None = None) -> GraphResult:
    """Delaunay triangulation of cell centroids, pruned to edges ≤ ``max_edge_um`` (≈ touching cells)."""
    tri = Delaunay(objs.xy)
    s = tri.simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e.sort(1)
    e = np.unique(e, axis=0)
    ln = np.hypot(*(objs.xy[e[:, 0]] - objs.xy[e[:, 1]]).T) * objs._mpp()
    keep = ln <= max_edge_um
    return GraphResult(e[keep], ln[keep], objs, list(classes or objs.class_names()))


# ============================================================================= autocorrelation
def morans_i(objs: ObjectTable, column: str | np.ndarray, k: int = 8, permutations: int = 0, seed: int = 0) -> dict:
    """Global Moran's I of a measurement with row-standardised k-nearest-neighbour weights.
    Returns I, E[I], z under the normality assumption, and optionally a permutation p-value."""
    v = np.asarray(objs.column(column) if isinstance(column, str) else column, np.float64)
    ok = np.isfinite(v)
    xy, v = objs.xy[ok], v[ok]
    n = len(v)
    _, nn = _tree(xy).query(xy, k=k + 1)
    nn = nn[:, 1:]
    W = sparse.csr_matrix((np.full(n * k, 1.0 / k), (np.repeat(np.arange(n), k), nn.ravel())), shape=(n, n))
    S0 = float(W.sum())
    Ws = W + W.T
    S1 = 0.5 * float(Ws.multiply(Ws).sum())
    S2 = float(((np.asarray(W.sum(1)).ravel() + np.asarray(W.sum(0)).ravel()) ** 2).sum())
    z = v - v.mean()
    denom = float((z**2).sum())

    def moran(zz):
        return (n / S0) * float(zz @ (W @ zz)) / denom

    i_obs = moran(z)
    e_i = -1.0 / (n - 1)
    var = (n * n * S1 - n * S2 + 3 * S0 * S0) / ((n * n - 1) * S0 * S0) - e_i**2
    out = {'I': i_obs, 'expected': e_i, 'z': (i_obs - e_i) / np.sqrt(max(var, 1e-15)), 'n': n, 'k': k}
    if permutations:
        rng = np.random.default_rng(seed)
        sims = np.array([moran(rng.permutation(z)) for _ in range(permutations)])
        out['p_perm'] = float((np.sum(np.abs(sims - e_i) >= abs(i_obs - e_i)) + 1) / (permutations + 1))
    return out


# ============================================================================= IHC / positivity
def h_score(
    objs: ObjectTable,
    column: str,
    thresholds: Sequence[float] = (0.2, 0.4, 0.6),
    rois: Any = None,
    within: Any = None,
    cls: Any = None,
) -> Any:
    """IHC H-score (0–300) per ROI from a DAB intensity column (e.g. 'DAB: Nucleus: Mean' OD).
    ``cls`` restricts to e.g. tumour cells. Returns a table with % 0/1+/2+/3+ and H-score."""
    bins = np.digitize(np.asarray(objs.column(column), np.float64), thresholds)
    base = objs.mask(cls)
    regions = roi_geometries(rois) if rois is not None else []
    if within is not None:
        w = _union(within) if not hasattr(within, 'geom_type') else within
        regions = [(n, g.intersection(w)) for n, g in regions] or [('Whole', w)]
    if not regions:
        regions = [('All', None)]
    rows = []
    for name, g in regions:
        m = base & (_inside(g, objs.x, objs.y) if g is not None else True)
        n = int(m.sum())
        pct = [100 * float(((bins == k) & m).sum()) / n if n else 0.0 for k in range(4)]
        rows.append(
            {
                'roi': name,
                'n': n,
                '% 0': pct[0],
                '% 1+': pct[1],
                '% 2+': pct[2],
                '% 3+': pct[3],
                '% positive': pct[1] + pct[2] + pct[3],
                'H-score': pct[1] + 2 * pct[2] + 3 * pct[3],
            }
        )
    return _df(rows)


def positivity(objs: ObjectTable, markers: dict[str, tuple[str, float]], rois: Any = None, cls: Any = None) -> Any:
    """% positive per marker (``{'Ki67': ('Nucleus: Ki67 mean', 300)}``) per ROI, optionally within a class."""
    base = objs.mask(cls)
    regions = roi_geometries(rois) if rois is not None else [('All', None)]
    rows = []
    for name, g in regions:
        m = base & (_inside(g, objs.x, objs.y) if g is not None else True)
        n = int(m.sum())
        row = {'roi': name, 'n': n}
        for mk, (col, thr) in markers.items():
            row[f'% {mk}+'] = 100 * float((objs.gate(col, thr) & m).sum()) / n if n else 0.0
        rows.append(row)
    return _df(rows)


# ============================================================================= slide-level image analysis
def tissue_mask(
    source: Any,
    max_size: int = 2048,
    min_area_um2: float = 20000,
    smooth_um: float = 20,
    mode: str = 'auto',
    threshold: float | None = None,
    name: str = 'Tissue',
):
    """Detect tissue on a low-resolution level. Brightfield: optical density (Otsu); fluorescence:
    log of the channel maximum (Otsu). Returns ``(shapely geometry in level-0 px, AnnotationLayer)``."""
    info = source.info
    lvl = len(info.levels) - 1
    for i, lv in enumerate(info.levels):
        if max(lv.width, lv.height) <= max_size:
            lvl = i
            break
    lv = info.levels[lvl]
    ds = lv.downsample
    mpp = (info.mpp or (1.0, 1.0))[0]
    if mode == 'auto':
        mode = 'brightfield' if info.mode == 'rgb' else 'fluorescence'
    if mode == 'brightfield':
        a = source.read_region(lvl, 0, 0, lv.width, lv.height).astype(np.float32)
        od = -np.log10(np.clip(a / 255.0, 1 / 255, 1)).sum(2)
        img = od
    else:
        chans = [
            source.read_region(lvl, 0, 0, lv.width, lv.height, c).astype(np.float32) for c in range(len(info.channels))
        ]
        img = np.log1p(np.max(np.stack(chans), 0))
    img = filters.gaussian(img, sigma=max(0.5, smooth_um / (mpp * ds) / 2))
    t = threshold if threshold is not None else filters.threshold_otsu(img)
    m = img > t
    px_area = (mpp * ds) ** 2
    n_px = max(1, int(min_area_um2 / px_area))
    try:  # scikit-image ≥ 0.26
        m = morphology.remove_small_objects(m, max_size=n_px)
        m = morphology.remove_small_holes(m, max_size=n_px)
    except TypeError:  # older scikit-image
        m = morphology.remove_small_objects(m, min_size=n_px)
        m = morphology.remove_small_holes(m, area_threshold=n_px)
    cs = measure.find_contours(np.pad(m, 1).astype(np.uint8), 0.5)
    geom = None
    for c in cs:
        if len(c) < 4:
            continue
        p = Polygon((c[:, ::-1] - 1) * ds)
        p = p.buffer(0)
        geom = p if geom is None else geom.symmetric_difference(p)  # even–odd → holes
    if geom is None:
        geom = shapely.Polygon()
    geom = geom.simplify(ds * 0.75)
    layer = _geom_layer([(name, geom)], name, fill_opacity=0.0, line_width=2, detail_min_px=0)
    layer.classes[name] = '#9e9e9e'
    return geom, layer


RUIFROK = {  # optical-density stain vectors (Ruifrok & Johnston 2001)
    'hematoxylin': [0.650, 0.704, 0.286],
    'eosin': [0.072, 0.990, 0.105],
    'dab': [0.268, 0.570, 0.776],
}


def estimate_stain_vectors(source: Any, max_size: int = 1024, beta: float = 0.15, alpha: float = 1.0) -> dict:
    """Macenko stain-vector estimation (H&E or H-DAB) on a low-resolution level.
    Returns ``{'hematoxylin': [r,g,b], 'second': [r,g,b]}`` (OD unit vectors) for ``setStain``."""
    info = source.info
    lvl = next((i for i, lv in enumerate(info.levels) if max(lv.width, lv.height) <= max_size), len(info.levels) - 1)
    lv = info.levels[lvl]
    a = source.read_region(lvl, 0, 0, lv.width, lv.height).reshape(-1, 3).astype(np.float64)
    od = -np.log((a + 1) / 256.0)
    od = od[(od > beta).all(1)]
    if len(od) < 100:
        return {'hematoxylin': RUIFROK['hematoxylin'], 'second': RUIFROK['eosin']}
    _, _, vt = np.linalg.svd(od - od.mean(0), full_matrices=False)
    v = vt[:2].T
    if v[:, 0].sum() < 0:
        v[:, 0] *= -1
    if v[:, 1].sum() < 0:
        v[:, 1] *= -1
    proj = od @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = v @ [np.cos(lo), np.sin(lo)]
    v2 = v @ [np.cos(hi), np.sin(hi)]
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    # hematoxylin absorbs red most strongly of the common stains (OD_R: H 0.65 vs eosin 0.07, DAB 0.27)
    h, s = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    return {'hematoxylin': h.tolist(), 'second': s.tolist()}


_ = Iterable
