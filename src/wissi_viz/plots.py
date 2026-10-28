"""ECharts option builders for analysis results (Apache ECharts, Apache-2.0 — bundled with NiceGUI).

Every function returns a plain dict: ``ui.echart(opts)`` in NiceGUI, or
``echarts.init(el).setOption(opts)`` in any page. Values are converted to plain Python.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .colormaps import categorical, lut

_BASE = {
    'animation': False,
    'textStyle': {'fontFamily': 'system-ui, sans-serif'},
    'grid': {'left': 60, 'right': 20, 'top': 72, 'bottom': 50, 'containLabel': True},
    'tooltip': {'trigger': 'axis'},
    'toolbox': {'feature': {'saveAsImage': {}, 'dataZoom': {}}},
}


def _py(v):
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _opts(title: str, **kw) -> dict:
    o = dict(_BASE, title={'text': title, 'left': 'center', 'top': 2, 'textStyle': {'fontSize': 14}}, **kw)
    if 'legend' in o:
        o['legend'] = dict({'top': 30}, **{k: v for k, v in o['legend'].items() if k != 'top'})
    y = o.get('yAxis')
    if isinstance(y, dict) and y.get('name') and y.get('type') != 'category':  # rotated, beside the axis
        o['yAxis'] = dict(y, nameLocation='middle', nameRotate=90, nameGap=48)
    return _py(o)


def histogram(
    values: Sequence[float], bins: int = 40, title: str = '', xlabel: str = '', density: bool = False
) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=bins, density=density)
    centers = (edges[:-1] + edges[1:]) / 2
    med = float(np.median(v)) if v.size else 0.0
    return _opts(
        title,
        xAxis={'type': 'value', 'name': xlabel, 'nameLocation': 'middle', 'nameGap': 28},
        yAxis={'type': 'value', 'name': 'density' if density else 'count'},
        series=[
            {
                'type': 'bar',
                'data': [[c, n] for c, n in zip(centers, counts)],
                'barWidth': '95%',
                'itemStyle': {'color': '#56b4e9'},
                'markLine': {
                    'symbol': 'none',
                    'data': [{'xAxis': med, 'name': 'median'}],
                    'label': {'formatter': f'median {med:.3g}'},
                },
            }
        ],
    )


def bar_counts(counts: Sequence[int], title: str = '', xlabel: str = '') -> dict:
    c = np.bincount(np.asarray(counts, int))
    return _opts(
        title,
        xAxis={'type': 'category', 'data': list(range(len(c))), 'name': xlabel},
        yAxis={'type': 'value', 'name': 'objects'},
        series=[{'type': 'bar', 'data': c, 'itemStyle': {'color': '#e69f00'}}],
    )


def lines(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    band: tuple[Sequence[float], Sequence[float]] | None = None,
    band_label: str = 'envelope',
    hline: float | None = None,
    vline: float | None = None,
) -> dict:
    x = np.asarray(x, float)
    out = []
    if band is not None:
        lo, hi = np.asarray(band[0], float), np.asarray(band[1], float)
        out += [
            {
                'type': 'line',
                'name': band_label,
                'data': [[a, b] for a, b in zip(x, lo)],
                'stack': 'env',
                'lineStyle': {'opacity': 0},
                'symbol': 'none',
                'tooltip': {'show': False},
            },
            {
                'type': 'line',
                'name': band_label,
                'data': [[a, b] for a, b in zip(x, hi - lo)],
                'stack': 'env',
                'lineStyle': {'opacity': 0},
                'symbol': 'none',
                'areaStyle': {'color': 'rgba(150,150,150,0.3)'},
                'tooltip': {'show': False},
            },
        ]
    for i, (name, y) in enumerate(series.items()):
        s = {
            'type': 'line',
            'name': name,
            'data': [[a, b] for a, b in zip(x, np.asarray(y, float))],
            'symbol': 'none',
            'lineStyle': {'width': 2},
            'itemStyle': {'color': categorical(i)},
        }
        marks = []
        if hline is not None and i == 0:
            marks.append({'yAxis': hline})
        if vline is not None and i == 0:
            marks.append({'xAxis': vline})
        if marks:
            s['markLine'] = {
                'symbol': 'none',
                'silent': True,
                'lineStyle': {'type': 'dashed', 'color': '#888'},
                'data': marks,
                'label': {'show': False},
            }
        out.append(s)
    return _opts(
        title,
        legend={'top': 26, 'data': list(series)},
        xAxis={'type': 'value', 'name': xlabel, 'nameLocation': 'middle', 'nameGap': 28},
        yAxis={'type': 'value', 'name': ylabel},
        series=out,
    )


def heatmap(
    m: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str = '', diverging: bool = False, fmt: str = '{c}'
) -> dict:
    m = np.asarray(m, float)
    data = [[j, i, m[i, j]] for i in range(m.shape[0]) for j in range(m.shape[1])]
    fin = m[np.isfinite(m)]
    if diverging:
        a = float(np.abs(fin).max()) if fin.size else 1.0
        lo, hi, colors = -a, a, lut('coolwarm', 11)  # blue = avoidance, red = enrichment
    else:
        lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        colors = lut('viridis', 9)
    return _opts(
        title,
        tooltip={'position': 'top'},
        grid={'left': 110, 'right': 30, 'top': 50, 'bottom': 90, 'containLabel': True},
        xAxis={'type': 'category', 'data': list(cols), 'axisLabel': {'rotate': 40}, 'splitArea': {'show': True}},
        yAxis={'type': 'category', 'data': list(rows), 'splitArea': {'show': True}, 'inverse': True},
        visualMap={
            'min': lo,
            'max': hi,
            'calculable': True,
            'orient': 'horizontal',
            'left': 'center',
            'bottom': 0,
            'inRange': {'color': colors},
        },
        series=[
            {
                'type': 'heatmap',
                'data': [[a, b, (round(c, 2) if np.isfinite(c) else None)] for a, b, c in data],
                'label': {'show': len(rows) * len(cols) <= 100},
            }
        ],
    )


def stacked_bars(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    title: str = '',
    percent: bool = False,
    horizontal: bool = True,
) -> dict:
    cat_axis = {'type': 'category', 'data': list(categories)}
    val_axis = {'type': 'value', 'max': 100 if percent else None, 'name': '%' if percent else ''}
    s = [
        {
            'type': 'bar',
            'name': k,
            'stack': 'total',
            'data': list(v),
            'itemStyle': {'color': categorical(i)},
            'emphasis': {'focus': 'series'},
        }
        for i, (k, v) in enumerate(series.items())
    ]
    return _opts(
        title,
        legend={'top': 26, 'type': 'scroll'},
        tooltip={'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
        grid={'left': 20, 'right': 30, 'top': 72, 'bottom': 30, 'containLabel': True},
        xAxis=val_axis if horizontal else cat_axis,
        yAxis=cat_axis if horizontal else val_axis,
        series=s,
    )


def grouped_bars(
    categories: Sequence[str], series: dict[str, Sequence[float]], title: str = '', ylabel: str = ''
) -> dict:
    s = [
        {'type': 'bar', 'name': k, 'data': list(v), 'itemStyle': {'color': categorical(i)}}
        for i, (k, v) in enumerate(series.items())
    ]
    return _opts(
        title,
        legend={'top': 26, 'type': 'scroll'},
        tooltip={'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
        xAxis={'type': 'category', 'data': list(categories), 'axisLabel': {'rotate': 30}},
        yAxis={'type': 'value', 'name': ylabel},
        series=s,
    )


def sunburst(rows: list[dict], classes: Sequence[str], whole_name: str = 'Whole') -> dict:
    """Part-of-whole: whole → ROIs → classes (counts)."""
    color = {c: categorical(i) for i, c in enumerate(classes)}
    children = []
    for r in rows:
        if r['roi'] == whole_name:
            continue
        children.append(
            {
                'name': r['roi'],
                'children': [
                    {'name': c, 'value': r.get(f'n {c}', 0), 'itemStyle': {'color': color[c]}}
                    for c in classes
                    if r.get(f'n {c}', 0)
                ],
            }
        )
    whole = next((r for r in rows if r['roi'] == whole_name), None)
    if whole is not None:
        rest = {
            c: whole.get(f'n {c}', 0) - sum(r.get(f'n {c}', 0) for r in rows if r['roi'] != whole_name) for c in classes
        }
        rest_children = [{'name': c, 'value': v, 'itemStyle': {'color': color[c]}} for c, v in rest.items() if v > 0]
        if rest_children:
            children.append({'name': 'outside ROIs', 'itemStyle': {'color': '#bbbbbb'}, 'children': rest_children})
    return _opts(
        'Part of whole',
        tooltip={'trigger': 'item'},
        series=[
            {
                'type': 'sunburst',
                'radius': ['12%', '92%'],
                'sort': None,
                'data': [{'name': whole_name, 'children': children}],
                'label': {'rotate': 'radial', 'minAngle': 8},
                'levels': [
                    {},
                    {'r0': '12%', 'r': '22%'},
                    {'r0': '22%', 'r': '60%'},
                    {'r0': '60%', 'r': '92%', 'label': {'align': 'right'}},
                ],
            }
        ],
    )


def scatter(
    x: Sequence[float],
    y: Sequence[float],
    classes: Sequence[str] | None = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    max_points: int = 20000,
    log: bool = False,
    xgate: float | None = None,
    ygate: float | None = None,
    seed: int = 0,
) -> dict:
    """Marker-vs-marker scatter (gating). Large tables are randomly subsampled; gates drawn as dashed lines.
    Brush selection is enabled (``ui.echart(..., on_point_click / 'brushselected')``)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.arange(len(x))
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_points, replace=False))
    groups: dict[str, list] = {}
    cl = np.asarray(classes, dtype=object) if classes is not None else None
    for i in idx:
        groups.setdefault(str(cl[i]) if cl is not None else 'objects', []).append([x[i], y[i], int(i)])
    series = [
        {
            'type': 'scatter',
            'name': k,
            'data': v,
            'symbolSize': 3,
            'large': True,
            'itemStyle': {'color': categorical(j)},
        }
        for j, (k, v) in enumerate(groups.items())
    ]
    marks = [
        m
        for m in ({'xAxis': xgate} if xgate is not None else None, {'yAxis': ygate} if ygate is not None else None)
        if m
    ]
    if marks:
        series[0]['markLine'] = {'symbol': 'none', 'lineStyle': {'type': 'dashed', 'color': '#d55e00'}, 'data': marks}
    ax = {'type': 'log' if log else 'value', 'scale': True}
    return _opts(
        title,
        legend={'top': 26, 'type': 'scroll'},
        tooltip={'trigger': 'item'},
        brush={'toolbox': ['rect', 'polygon', 'clear'], 'xAxisIndex': 0},
        xAxis=dict(ax, name=xlabel, nameLocation='middle', nameGap=28),
        yAxis=dict(ax, name=ylabel),
        series=series,
    )


def table_rows(table: Any) -> list[dict]:
    """DataFrame / dict → rows for ``ui.table`` (numbers rounded)."""
    rows = table.to_dict('records') if hasattr(table, 'to_dict') else list(table)
    return [
        {
            k: (round(float(v), 3) if isinstance(v, (float, np.floating)) and np.isfinite(v) else _py(v))
            for k, v in r.items()
        }
        for r in rows
    ]
