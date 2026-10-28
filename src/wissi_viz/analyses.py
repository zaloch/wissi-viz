"""Spatial analyses the app runs on a slide's cell layer: one catalogue, one ``run()``.

Each analysis is a small adapter around :mod:`wissi_viz.spatial`. It turns the user's choices into a call,
adds the result as layers (``origin='analysis'``, so they toggle, export and measure like any other layer),
and returns what the inspector shows: a few headline numbers, a chart and a table.

Adding one: write a function ``(ctx, **params) -> Result`` and decorate it with :func:`analysis`.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import spatial
from .annotations import AnnotationLayer
from .metrics import objects_from_layer

MAX_ROWS = 200  # table rows sent to the page (the CSV has them all)


@dataclass
class Param:
    key: str
    label: str
    kind: str  # 'class' | 'class?' (or all) | 'number' | 'column' | 'regions'
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ''


@dataclass
class Spec:
    id: str
    name: str
    group: str  # 'Density' | 'Distances' | 'Regions' | 'Patterns'
    desc: str
    params: list[Param]
    speed: str  # rough cost shown on the card
    fn: Callable[..., Result] = field(repr=False)

    @property
    def needs_regions(self) -> bool:
        return any(p.kind == 'regions' for p in self.params)

    def describe(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'group': self.group,
            'desc': self.desc,
            'speed': self.speed,
            'needs_regions': self.needs_regions,
            'params': [p.__dict__ for p in self.params],
        }


@dataclass
class Result:
    title: str
    kpis: list[tuple[str, str]] = field(default_factory=list)
    note: str = ''
    layers: list[AnnotationLayer] = field(default_factory=list)
    chart: dict | None = None
    table: Any = None


@dataclass
class Context:
    registry: Any
    slide: str
    objects: Any  # spatial.ObjectTable
    layer: str
    mpp: float | None


CATALOG: OrderedDict[str, Spec] = OrderedDict()


def analysis(key: str, name: str, group: str, desc: str, params: list[Param], speed: str = '<1 s'):
    """Register ``fn(ctx, **params) -> Result`` in the catalogue under ``key``."""

    def deco(fn):
        CATALOG[key] = Spec(key, name, group, desc, params, speed, fn)
        return fn

    return deco


def catalog() -> list[dict]:
    return [s.describe() for s in CATALOG.values()]


def echarts_option(chart: dict | None, dark: bool = False) -> dict | None:
    """A result's ``chart`` as an Apache ECharts option, for ``ui.echart`` in NiceGUI, Dash, Panel, Streamlit
    or plain ECharts. Bars, lines (with the random-layout band as a shaded area) and heat maps."""
    if not chart:
        return None
    text = '#c9d1d9' if dark else '#333'
    if chart['kind'] == 'heat':
        vals = chart.get('values') or []
        data = [[j, i, v] for i, row in enumerate(vals) for j, v in enumerate(row) if v is not None]
        nums = [d[2] for d in data] or [0.0]
        lo, hi = min(nums), max(nums)
        diverging = lo < 0 < hi
        m = max(abs(lo), abs(hi)) or 1.0
        return {
            'tooltip': {'position': 'top'},
            'grid': {'left': 100, 'right': 16, 'top': 10, 'bottom': 70},
            'xAxis': {'type': 'category', 'data': chart.get('cols', []), 'axisLabel': {'rotate': 30, 'color': text}},
            'yAxis': {'type': 'category', 'data': chart.get('rows', []), 'axisLabel': {'color': text}},
            'visualMap': {
                'min': -m if diverging else min(0.0, lo),
                'max': m if diverging else hi,
                'calculable': True,
                'orient': 'horizontal',
                'left': 'center',
                'bottom': 0,
                'text': [chart.get('label', ''), ''],
                'inRange': {'color': ['#2b8cbe', '#f7f7f7', '#e66101'] if diverging else ['#f2f0f7', '#54278f']},
            },
            'series': [{'type': 'heatmap', 'data': data, 'label': {'show': len(chart.get('cols', [])) <= 8}}],
        }
    x = chart.get('x') or []
    numeric = chart['kind'] == 'lines' and all(isinstance(v, (int, float)) for v in x)
    series = []
    for sr in chart.get('series') or []:
        vals = sr.get('values') or []
        if chart['kind'] == 'bars':
            series.append({'name': sr['name'], 'type': 'bar', 'data': vals})
        else:
            data = [[xv, yv] for xv, yv in zip(x, vals)] if numeric else vals
            style = {'type': 'dashed', 'width': 1, 'color': '#9aa3ad'} if sr.get('band') else {'width': 2}
            series.append({'name': sr['name'], 'type': 'line', 'data': data, 'showSymbol': False, 'lineStyle': style})
    if chart.get('marker') is not None and series:
        mk = chart['marker']
        pos = (
            mk
            if numeric
            else min(range(len(x)), key=lambda i: abs(x[i] - mk))
            if x and isinstance(x[0], (int, float))
            else None
        )
        if pos is not None:
            series[0]['markLine'] = {
                'symbol': 'none',
                'label': {'formatter': chart.get('marker_label') or ''},
                'data': [{'xAxis': pos}],
            }
    xaxis = (
        {'type': 'value', 'name': chart.get('xlabel', ''), 'nameLocation': 'middle', 'nameGap': 26}
        if numeric
        else {
            'type': 'category',
            'data': [f'{v:g}' if isinstance(v, (int, float)) else str(v) for v in x],
            'name': chart.get('xlabel', ''),
            'nameLocation': 'middle',
            'nameGap': 26,
        }
    )
    return {
        'tooltip': {'trigger': 'axis'},
        'legend': {'show': len(series) > 1, 'textStyle': {'color': text}},
        'grid': {'left': 50, 'right': 16, 'top': 30 if len(series) > 1 else 12, 'bottom': 40},
        'xAxis': xaxis,
        'yAxis': {'type': 'value', 'name': chart.get('ylabel', '')},
        'series': series,
    }


_cache_lock = threading.Lock()
_objects: OrderedDict = OrderedDict()  # (slide, layer, version) → ObjectTable


def run(registry, slide: str, analysis_id: str, objects: str, params: dict | None = None) -> dict:
    """Run one analysis on the cell layer ``objects`` of ``slide``; result layers are added to the slide."""
    spec = CATALOG.get(analysis_id)
    if spec is None:
        raise ValueError(f'unknown analysis {analysis_id!r}')
    layer = registry.layer(slide, objects)
    if layer is None:
        raise ValueError(f'no layer {objects!r} on this slide')
    mpp = (registry.info(slide).get('mpp') or [None])[0]
    ctx = Context(registry, slide, _object_table(slide, layer, mpp), objects, mpp)
    kw = _coerce(spec, params or {}, ctx)
    t0 = time.perf_counter()
    res = spec.fn(ctx, **kw)
    names = []
    for L in res.layers:
        L.origin = 'analysis'
        registry.add_annotations(slide, L)
        names.append(L.name)
    return {
        'analysis': spec.id,
        'title': res.title,
        'kpis': [{'label': k, 'value': v} for k, v in res.kpis],
        'note': res.note,
        'layers': names,
        'chart': res.chart,
        'table': _table(res.table),
        'seconds': round(time.perf_counter() - t0, 2),
    }


# ----------------------------------------------------------------------------------------------- helpers
def _object_table(slide: str, layer: AnnotationLayer, mpp):
    key = (slide, layer.name, layer.version)
    with _cache_lock:
        if key in _objects:
            _objects.move_to_end(key)
            return _objects[key]
    objs = objects_from_layer(layer, mpp)
    with _cache_lock:
        _objects[key] = objs
        while len(_objects) > 4:
            _objects.popitem(last=False)
    return objs


def _coerce(spec: Spec, params: dict, ctx: Context) -> dict:
    classes = set(ctx.objects.class_names())
    out = {}
    for p in spec.params:
        v = params.get(p.key, p.default)
        if p.kind in ('class', 'class?'):
            if v in (None, '', 'all') and p.kind == 'class?':
                v = None
            elif v not in classes:
                raise ValueError(f'{p.label}: choose one of {", ".join(sorted(classes))}')
        elif p.kind == 'number':
            v = float(v if v not in (None, '') else p.default)
            if p.min is not None and v < p.min or p.max is not None and v > p.max:
                raise ValueError(f'{p.label}: between {p.min:g} and {p.max:g} {p.unit}'.strip())
        elif p.kind == 'column':
            if v not in ctx.objects.columns:
                raise ValueError(f'{p.label}: choose a measurement column')
        elif p.kind == 'regions':
            L = ctx.registry.layer(ctx.slide, v) if v else None
            if L is None:
                raise ValueError('pick a region layer (draw one, or add an annotation file)')
            v = L
        out[p.key] = v
    return out


def _hist(values, bins: int = 24, label: str = '', unit: str = '') -> dict | None:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    hi = float(np.percentile(v, 99.5)) if v.size > 20 else float(v.max())
    counts, edges = np.histogram(np.clip(v, v.min(), hi), bins=bins, range=(float(v.min()), hi or 1.0))
    return {
        'kind': 'bars',
        'x': [round(float(e), 3) for e in edges[:-1]],
        'x_end': round(float(edges[-1]), 3),
        'series': [{'name': label or 'count', 'values': counts.tolist()}],
        'xlabel': f'{label} ({unit})' if unit else label,
        'ylabel': 'objects',
        'marker': round(float(np.median(v)), 3),
        'marker_label': 'median',
    }


def _table(t: Any) -> dict | None:
    if t is None:
        return None
    if hasattr(t, 'to_dict') and hasattr(t, 'columns'):  # pandas
        cols = [str(c) for c in t.columns]
        rows = t.head(MAX_ROWS).to_numpy().tolist()
        n = len(t)
    elif isinstance(t, dict):
        cols = list(t)
        n = len(next(iter(t.values()), []))
        rows = [[t[c][i] for c in cols] for i in range(min(n, MAX_ROWS))]
    else:  # list of dicts
        t = list(t)
        cols = list(t[0]) if t else []
        rows = [[r.get(c) for c in cols] for r in t[:MAX_ROWS]]
        n = len(t)
    clean = lambda x: (
        (round(float(x), 4) if np.isfinite(x) else None)
        if isinstance(x, (float, np.floating))
        else (int(x) if isinstance(x, (np.integer,)) else x)
    )
    return {'columns': cols, 'rows': [[clean(x) for x in r] for r in rows], 'n': int(n)}


def _fmt(x: float, unit: str = '', digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return '–'
    s = f'{x:,.0f}' if abs(x) >= 100 else f'{x:.{digits}g}'
    return f'{s} {unit}'.strip()


def _cls_label(c) -> str:
    return c if c else 'all cells'


# ----------------------------------------------------------------------------------------------- the catalogue
CLASS_ALL = Param('cls', 'Cells', 'class?', 'all')


@analysis(
    'density',
    'Density & hotspots',
    'Density',
    'Cells per mm² on a hex grid, and Getis-Ord hotspots',
    [CLASS_ALL, Param('cell_um', 'Grid', 'number', 100, 20, 1000, 10, 'µm')],
    '~1–3 s',
)
def _density(ctx: Context, cls, cell_um):
    g = spatial.density_grid(ctx.objects, cls, cell_um=cell_um, hotspots=True)
    label = _cls_label(cls)
    hot = g.hotspot_layer(name=f'Hotspots · {label}', fill_opacity=0.25)
    v = np.asarray(g.values, np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    return Result(
        f'Density · {label}',
        [
            ('median', _fmt(np.median(v) if v.size else np.nan, '/mm²')),
            ('peak', _fmt(v.max() if v.size else np.nan, '/mm²')),
            ('hotspots', f'{len(hot):,}'),
        ],
        f'{cell_um:g} µm hexagons; hotspots: Getis-Ord Gi* z > 1.96.',
        [g.layer(name=f'Density · {label}', fill_opacity=0.4), hot],
        _hist(v, label='cells / mm² per hexagon'),
        g.table,
    )


@analysis(
    'nn',
    'Nearest neighbours',
    'Distances',
    'Distance from each A to the closest B',
    [
        Param('source', 'From', 'class'),
        Param('target', 'To', 'class'),
        Param('max_um', 'Up to', 'number', 100, 5, 1000, 5, 'µm'),
    ],
)
def _nn(ctx: Context, source, target, max_um):
    r = spatial.nearest_neighbors(ctx.objects, source, target, max_um=max_um)
    s = r.summary()
    return Result(
        f'{source} → {target}',
        [
            ('median', _fmt(s['median_um'], 'µm')),
            ('IQR', f'{_fmt(s["q25_um"])}–{_fmt(s["q75_um"], "µm")}'),
            ('within 20 µm', f'{100 * s["frac_within_20um"]:.0f}%' if np.isfinite(s['frac_within_20um']) else '–'),
        ],
        f'{s["n_source"]:,} {source} cells; distances beyond {max_um:g} µm are left out of the lines layer.',
        [r.layer(name=f'Nearest {target} · from {source}', max_um=max_um)],
        _hist(r.dist_um, label='distance', unit='µm'),
        r.table,
    )


@analysis(
    'prox',
    'Proximity',
    'Distances',
    'How many B within r of each A',
    [
        Param('source', 'Around', 'class'),
        Param('target', 'Count', 'class'),
        Param('radius_um', 'Radius', 'number', 30, 5, 500, 5, 'µm'),
    ],
)
def _prox(ctx: Context, source, target, radius_um):
    r = spatial.proximity(ctx.objects, source, target, radius_um)
    c = np.asarray(r.counts, np.float64)
    return Result(
        f'{target} within {radius_um:g} µm of {source}',
        [
            ('mean', _fmt(c.mean() if c.size else np.nan)),
            ('none nearby', f'{100 * (c == 0).mean():.0f}%' if c.size else '–'),
        ],
        f'{len(c):,} {source} cells, coloured by how many {target} cells are within {radius_um:g} µm.',
        [r.layer(name=f'{target} near {source} · {radius_um:g} µm')],
        _hist(c, bins=min(24, int(c.max()) + 1 if c.size else 1), label=f'{target} within r'),
        r.table,
    )


@analysis(
    'bands',
    'Margin bands',
    'Regions',
    'Cell density in bands inside and outside a region boundary',
    [Param('regions', 'Region', 'regions'), Param('width_um', 'Band width', 'number', 100, 10, 1000, 10, 'µm')],
    '~1 s',
)
def _bands(ctx: Context, regions, width_um):
    edges = [-2 * width_um, -width_um, 0, width_um, 2 * width_um]
    r = spatial.margin_bands(regions, edges, mpp=ctx.mpp, objs=ctx.objects)
    pd = r.plot_data or {}
    series = pd.get('series') or {}
    items = series.items() if isinstance(series, dict) else ((s.get('name'), s.get('values')) for s in series)
    return Result(
        f'Margin bands · {regions.name}',
        [('bands', str(len(edges) - 1)), ('width', _fmt(width_um, 'µm'))],
        'Negative distances are inside the region.',
        [r.layer(name=f'Margin bands · {regions.name}')],
        {
            'kind': 'bars',
            'x': list(pd.get('bands') or []),
            'series': [
                {'name': str(k), 'values': [None if v is None else float(v) for v in vals]} for k, vals in items
            ],
            'ylabel': 'cells / mm²',
        },
        r.table,
    )


@analysis(
    'infiltration',
    'Infiltration',
    'Regions',
    'Density of each class against distance to a region boundary',
    [Param('regions', 'Region', 'regions'), Param('range_um', 'Out to', 'number', 500, 50, 3000, 50, 'µm')],
    '~1–2 s',
)
def _infiltration(ctx: Context, regions, range_um):
    r = spatial.infiltration_profile(
        ctx.objects, regions, range_um=(-range_um, range_um), bin_um=max(5.0, range_um / 20)
    )
    return Result(
        f'Infiltration · {regions.name}',
        [('classes', str(len(r.density))), ('range', f'±{range_um:g} µm')],
        'Negative distances are inside the region.',
        [],
        {
            'kind': 'lines',
            'x': [round(float(x), 1) for x in r.centers_um],
            'series': [
                {'name': k, 'values': [None if not np.isfinite(v) else round(float(v), 2) for v in vals]}
                for k, vals in r.density.items()
            ],
            'xlabel': 'distance to boundary (µm)',
            'ylabel': 'cells / mm²',
            'marker': 0,
            'marker_label': 'boundary',
        },
        r.table,
    )


@analysis(
    'composition',
    'Composition',
    'Regions',
    'Counts, density and share of each class per region',
    [Param('regions', 'Regions', 'regions')],
)
def _composition(ctx: Context, regions):
    r = spatial.roi_composition(ctx.objects, regions)
    t = _table(r.table) or {'columns': [], 'rows': []}
    cols = t['columns']
    roi_i = cols.index('roi') if 'roi' in cols else 0
    pct_cols = [c for c in cols if c.endswith('%') or c.startswith('% ')][:8]
    series = [{'name': c, 'values': [row[cols.index(c)] for row in t['rows']]} for c in pct_cols]
    return Result(
        f'Composition · {regions.name}',
        [('regions', f'{len(r.rois):,}'), ('classes', str(len(r.classes)))],
        '',
        [r.layer(name=f'Composition · {regions.name}')],
        {'kind': 'bars', 'x': [str(row[roi_i]) for row in t['rows']], 'series': series, 'ylabel': '%'}
        if series
        else None,
        r.table,
    )


@analysis(
    'ripley',
    "Ripley's L",
    'Patterns',
    'Clustering against random, across distances',
    [Param('a', 'Cells', 'class'), Param('b', 'Against', 'class?', 'all')],
    '~2–5 s',
)
def _ripley(ctx: Context, a, b):
    r = spatial.ripley(ctx.objects, a, b)
    rr, L = np.asarray(r.r_um, float), np.asarray(r.L, float)
    above = rr[L > np.asarray(r.L_hi, float)]
    return Result(
        f"Ripley's L · {a}" + (f' → {b}' if b else ''),
        [('clustered at', f'{_fmt(above.min())}–{_fmt(above.max(), "µm")}' if above.size else 'no distance')],
        'L above the grey band (39 random layouts) means more clustered than chance at that distance.',
        [],
        {
            'kind': 'lines',
            'x': [round(float(x), 1) for x in rr],
            'series': [
                {'name': 'L(r) − r', 'values': [round(float(v), 3) for v in L - rr]},
                {'name': 'random low', 'values': [round(float(v), 3) for v in np.asarray(r.L_lo) - rr], 'band': True},
                {'name': 'random high', 'values': [round(float(v), 3) for v in np.asarray(r.L_hi) - rr], 'band': True},
            ],
            'xlabel': 'distance (µm)',
            'ylabel': 'L(r) − r',
        },
        r.table,
    )


@analysis(
    'enrichment',
    'Neighbour enrichment',
    'Patterns',
    'Which classes sit next to each other more than chance',
    [Param('radius_um', 'Within', 'number', 20, 5, 200, 5, 'µm')],
    '~2–6 s',
)
def _enrichment(ctx: Context, radius_um):
    r = spatial.neighborhood_enrichment(ctx.objects, radius_um=radius_um)
    z = np.asarray(r.z, float)
    k = np.unravel_index(np.nanargmax(z), z.shape) if np.isfinite(z).any() else None
    top = f'{r.classes[k[0]]} · {r.classes[k[1]]}' if k else '–'
    return Result(
        f'Neighbour enrichment · {radius_um:g} µm',
        [('most enriched', top), ('z', _fmt(float(np.nanmax(z)) if k else np.nan))],
        'z > 2: neighbours more often than by chance; z < −2: avoid each other (199 shuffles).',
        [],
        {
            'kind': 'heat',
            'rows': list(r.classes),
            'cols': list(r.classes),
            'values': [[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in z],
            'label': 'z-score',
        },
        r.table,
    )


@analysis(
    'hoods',
    'Cellular neighbourhoods',
    'Patterns',
    'Recurring local mixes of cell types (k nearest)',
    [Param('k', 'Neighbours', 'number', 10, 3, 50, 1), Param('n', 'Neighbourhoods', 'number', 6, 2, 12, 1)],
    '~2–8 s',
)
def _hoods(ctx: Context, k, n):
    r = spatial.cellular_neighborhoods(ctx.objects, k=int(k), n_neighborhoods=int(n))
    comp = np.asarray(r.composition, float)
    return Result(
        f'{int(n)} neighbourhoods · k = {int(k)}',
        [('cells', f'{len(r.labels):,}'), ('neighbourhoods', str(int(n)))],
        'Each cell is coloured by the neighbourhood its k nearest cells look like.',
        [r.layer(name=f'Neighbourhoods · k {int(k)}')],
        {
            'kind': 'heat',
            'rows': list(r.names),
            'cols': list(r.classes),
            'values': [[round(float(v), 3) for v in row] for row in comp],
            'label': 'share',
        },
        r.table,
    )


@analysis(
    'contacts',
    'Contact graph',
    'Patterns',
    'Touching neighbours (Delaunay, pruned by length)',
    [Param('max_edge_um', 'Longest edge', 'number', 30, 5, 200, 5, 'µm')],
    '~1–4 s',
)
def _contacts(ctx: Context, max_edge_um):
    r = spatial.delaunay_graph(ctx.objects, max_edge_um=max_edge_um)
    ln = np.asarray(r.length_um, float)
    return Result(
        f'Contact graph · ≤ {max_edge_um:g} µm',
        [('edges', f'{len(ln):,}'), ('median length', _fmt(np.median(ln) if ln.size else np.nan, 'µm'))],
        '',
        [r.layer(name=f'Contacts · {max_edge_um:g} µm')],
        _hist(ln, label='edge length', unit='µm'),
        r.table,
    )


@analysis(
    'moran',
    "Moran's I",
    'Patterns',
    'Is a measurement spatially autocorrelated?',
    [Param('column', 'Measurement', 'column')],
)
def _moran(ctx: Context, column):
    d = spatial.morans_i(ctx.objects, column, permutations=99)
    i = d.get('I', d.get('i', np.nan))
    p = d.get('p_perm', np.nan)
    return Result(
        f"Moran's I · {column}",
        [('I', _fmt(i)), ('p', _fmt(p))],
        'I near 0: no spatial pattern; towards 1: similar values cluster together (8 nearest neighbours).',
        [],
        None,
        [d],
    )
