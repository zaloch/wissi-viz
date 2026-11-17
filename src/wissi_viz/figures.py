"""Figures: a recipe — which saved views, laid out how, labelled and charted how, on which template — turned into pages
of windows (rectangles in mm). The composer draws these pages; the exports (:mod:`wissi_viz.figexport`,
:mod:`wissi_viz.pptxio`) draw the same pages, so what is shown is what is written.

A recipe never holds pixels. Its views are references to saved views (:mod:`wissi_viz.store`): change a view's frame
or its metadata and the figure follows. Which views are in it is ``source`` — a fixed list (``views``) or rules over
the views' fields (``query``). For a query, ``members`` keeps the views as of the last Update, so that nothing joins or
leaves a figure without being asked; :func:`select` reports what an Update would add or remove.

Templates:

* ``pub`` — a journal page: 85 / 180 mm wide (``page``: ``single`` / ``double``), as tall as its content, or
  ``full`` (180 × 240 mm; more pages when it overflows).
* ``ppt`` — 16:9 or 4:3 slides with a title; windows fit the slide; more views than fit go on to the next slide.
* ``pptx`` — the slides of a PowerPoint template (:func:`wissi_viz.pptxio.read_template`): its size, and the windows
  of one of its layouts (picture placeholders, or picture positions learned from its slides).

Every window can be moved, resized or changed by hand (``windows``: by ``'<page>:<slot>'``), and windows can be added
(``extra``). The page is black when every view is fluorescence (``background``: ``auto``), or as set.
"""

from __future__ import annotations

import copy
import csv
import io
import math
import re
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

PT = 25.4 / 72  # mm per point
PAGES = {'single': (85.0, None), 'double': (180.0, None), 'full': (180.0, 240.0),
         '16:9': (338.667, 190.5), '4:3': (254.0, 190.5)}  # fmt: skip
TEMPLATES = ('pub', 'ppt', 'pptx')
SHOWS = ('view', 'chart', 'text', 'empty')
CHARTS = ('box', 'bar', 'dots')
#: group colours (blue, orange, teal, purple, red, olive): distinct in lightness too; lighter ones on a black page
COLORS = ('#2f63a8', '#d9822b', '#2a9d8f', '#7b4fa0', '#c0392b', '#7f8c2d')
COLORS_DARK = ('#7fb0ff', '#ffae5c', '#5fd3c4', '#c39bf0', '#ff7f6e', '#c4d36a')

DEFAULT: dict[str, Any] = {
    'template': 'pub',
    'page': 'double',  # pub: single | double | full · ppt: 16:9 | 4:3 · pptx: the template's size
    'title': '',  # slides: their title (default: the figure's name)
    'source': {'kind': 'views', 'ids': [], 'rules': []},
    'members': None,  # query: the views in the figure as of the last Update (None: every view that matches now)
    'layout': {
        'mode': 'grid',
        'rows': None,
        'order': [],
        'cols': 'auto',
        'per_page': 6,
        'aspect': 'auto',
        'gap': None,
    },  # fmt: skip
    'labels': '{Name}',
    'letters': True,
    'scalebar': 'auto',  # 'auto' | µm | None (none)
    'background': 'auto',  # auto (black when every view is fluorescence) | white | black
    'font': None,  # label size in pt (default: 8 on a page, 14 on a slide)
    'chart': None,  # {'kind': box | bar | dots, 'measure': {...}, 'group': field}
    'pptx': None,  # {'id': template id, 'layout': layout name}
    #: fluorescence: 'merge' (a window per view) or 'split' (a window per channel, plus the merge: a row per view);
    #: ``show``: channel names shown (None: all); ``gray``: single channels in grey
    'channels': {'mode': 'merge', 'merge': True, 'show': None, 'gray': False},
    'windows': {},  # '<page>:<slot>' → {'rect': [x, y, w, h] mm, 'shows': {'kind', 'view' | 'text'}}
    'extra': [],  # [{'page', 'rect', 'shows'}]: windows added by hand
}


# ------------------------------------------------------------------------------------------------------- recipes
def normalize(recipe: dict | None) -> dict:
    """A complete recipe: defaults filled in, unknown values replaced by defaults (never raises)."""
    r = copy.deepcopy(DEFAULT)
    src = recipe if isinstance(recipe, dict) else {}
    for k, v in src.items():
        if k in ('source', 'layout') and isinstance(v, dict):
            r[k].update({kk: vv for kk, vv in v.items() if kk in r[k]})
        elif k in r:
            r[k] = copy.deepcopy(v)
    if r['template'] not in TEMPLATES:
        r['template'] = 'pub'
    pages = ('single', 'double', 'full') if r['template'] == 'pub' else ('16:9', '4:3')
    if r['template'] != 'pptx' and r['page'] not in pages:
        r['page'] = pages[1] if r['template'] == 'pub' else pages[0]
    s = r['source']
    s['kind'] = 'query' if s.get('kind') == 'query' else 'views'
    s['ids'] = [int(i) for i in s.get('ids') or [] if _is_int(i)]
    s['rules'] = [q for q in (s.get('rules') or []) if isinstance(q, dict) and q.get('field')]
    if r['members'] is not None:
        r['members'] = [int(i) for i in r['members'] if _is_int(i)]
    L = r['layout']
    L['mode'] = 'field' if L.get('mode') == 'field' and L.get('rows') else 'grid'
    L['order'] = [o for o in (L.get('order') or []) if isinstance(o, dict) and o.get('field')]
    L['cols'] = int(L['cols']) if _is_int(L.get('cols')) and 1 <= int(L['cols']) <= 12 else 'auto'
    L['per_page'] = int(L['per_page']) if _is_int(L.get('per_page')) and 1 <= int(L['per_page']) <= 24 else 6
    a = L.get('aspect')
    L['aspect'] = float(a) if isinstance(a, (int, float)) and 0.2 <= a <= 5 else 'auto'
    g = L.get('gap')
    L['gap'] = float(g) if isinstance(g, (int, float)) and 0 <= g <= 20 else None
    if r['background'] not in ('auto', 'white', 'black'):
        r['background'] = 'auto'
    sb = r['scalebar']
    r['scalebar'] = sb if sb is None or sb == 'auto' or (isinstance(sb, (int, float)) and sb > 0) else 'auto'
    if not isinstance(r['labels'], str):
        r['labels'] = DEFAULT['labels']
    f = r['font']
    r['font'] = float(f) if isinstance(f, (int, float)) and 4 <= f <= 48 else None
    c = r['chart']
    if isinstance(c, dict):
        c = {'kind': c.get('kind') if c.get('kind') in CHARTS else 'box',
             'measure': c.get('measure') if isinstance(c.get('measure'), dict) else {'kind': 'count'},
             'group': c.get('group') or None}  # fmt: skip
    r['chart'] = c if isinstance(c, dict) else None
    if not isinstance(r['windows'], dict):
        r['windows'] = {}
    if not isinstance(r['extra'], list):
        r['extra'] = []
    if not isinstance(r['pptx'], dict):
        r['pptx'] = None
    ch = r['channels'] if isinstance(r['channels'], dict) else {}
    show = ch.get('show')
    r['channels'] = {'mode': 'split' if ch.get('mode') == 'split' else 'merge', 'merge': ch.get('merge') is not False,
                     'show': [str(x) for x in show] if isinstance(show, list) else None, 'gray': bool(ch.get('gray'))}  # fmt: skip
    return r


def _is_int(v: Any) -> bool:
    try:
        int(v)
    except (TypeError, ValueError):
        return False
    return not isinstance(v, bool)


# ---------------------------------------------------------------------------------------------- fields and order
def field_value(view: dict, field: str) -> Any:
    """A view's value for a field: ``Name`` (the view's name), ``Slide`` (its slide's file name), else its metadata
    (the slide's fields, overridden by the view's own)."""
    if field == 'Name':
        return view.get('name') or view.get('meta', {}).get('Slide') or f'View {view.get("id")}'
    return (view.get('meta') or {}).get(field)


def fields_of(views: Iterable[dict]) -> list[str]:
    """Every field the views have, in first-seen order, ``Name`` and ``Slide`` first."""
    out = {'Name': None, 'Slide': None}
    for v in views:
        out.update(dict.fromkeys(k for k in (v.get('meta') or {}) if k))
    return list(out)


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sort_key(v: Any) -> tuple:
    """Numbers before text, numbers by value, text case-insensitively, missing last."""
    if v is None or v == '':
        return (2, 0, '')
    n = _num(v)
    return (
        (0, n, '')
        if n is not None and not (isinstance(v, str) and v[:1] == '0' and len(v) > 1)
        else (1, 0, str(v).lower())
    )


def matches(view: dict, rules: list[dict]) -> bool:
    """Whether a view passes every rule: {'field', 'op': in | not in | >= | <= | has | missing, 'values'}."""
    for q in rules:
        v = field_value(view, q['field'])
        op = q.get('op') or 'in'
        vals = q.get('values') or []
        if op == 'has':
            ok = v is not None and v != ''
        elif op == 'missing':
            ok = v is None or v == ''
        elif op in ('>=', '<='):
            n, t = _num(v), _num(vals[0] if vals else None)
            ok = n is not None and t is not None and (n >= t if op == '>=' else n <= t)
        else:
            inside = str(v) in {str(x) for x in vals} if v is not None else ('' in vals or None in vals)
            ok = inside if op == 'in' else not inside
        if not ok:
            return False
    return True


def select(recipe: dict, views: list[dict]) -> dict:
    """The views in the figure, in order, and what an Update would change.

    Returns {'members': [view, …], 'ids': [id, …], 'pending': {'add': [id], 'remove': [id]}, 'missing': [id]}:
    ``missing`` are ids the figure keeps that no longer exist (deleted views)."""
    r = normalize(recipe)
    by = {int(v['id']): v for v in views}
    src = r['source']
    pending = {'add': [], 'remove': []}
    if src['kind'] == 'views':
        ids = src['ids']
    else:
        now = [int(v['id']) for v in views if matches(v, src['rules'])]
        if r['members'] is None:
            ids = now
        else:
            ids = r['members']
            have = set(ids)
            pending['add'] = [i for i in now if i not in have]
            nowset = set(now)
            pending['remove'] = [i for i in ids if i not in nowset and i in by]
    missing = [i for i in ids if i not in by]
    members = [by[i] for i in ids if i in by]
    members = order(members, r)
    return {'members': members, 'ids': [int(v['id']) for v in members], 'pending': pending, 'missing': missing}


def order(members: list[dict], recipe: dict) -> list[dict]:
    """Sorted by the layout's order fields (stable: else as listed); the rows field first when laid out by field."""
    L = recipe['layout']
    keys = list(L['order'])
    if L['mode'] == 'field' and not any(o['field'] == L['rows'] for o in keys):
        keys.insert(0, {'field': L['rows']})
    out = list(members)
    for o in reversed(keys):
        out.sort(key=lambda v, f=o['field']: _sort_key(field_value(v, f)), reverse=bool(o.get('desc')))
    return out


_FIELD = re.compile(r'\{([^{}]+)\}')


def _fmt(v: Any) -> str:
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:g}'
    return str(v)


def label(template: str, view: dict) -> str:
    """A label from fields: ``'{Animal} · {Region}'`` (a missing field leaves its separator out)."""
    text = _FIELD.sub(lambda m: _fmt(field_value(view, m.group(1).strip())), template or '')
    text = re.sub(r'(\s*[·,;|/-]\s*){2,}', ' · ', text)
    return text.strip().strip('·,;|/-').strip()


# ----------------------------------------------------------------------------------------------------- geometry
def _nice(x: float) -> float:
    """1, 2 or 5 × 10^k at or below x."""
    if x <= 0:
        return 0.0
    e = 10 ** math.floor(math.log10(x))
    return max(m for m in (1, 2, 5, 10) if m * e <= x * (1 + 1e-9)) * e


def expand(bounds, aspect: float) -> list[float]:
    """The view's rectangle widened (or heightened) about its centre to ``aspect`` (w / h): all of the view stays
    in sight, as in a pane of another shape."""
    x0, y0, x1, y1 = (float(b) for b in bounds)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _aspect(members: list[dict], L: dict) -> float:
    if L['aspect'] != 'auto':
        return float(L['aspect'])
    a = sorted((b[2] - b[0]) / (b[3] - b[1]) for b in (v['bounds'] for v in members) if b[3] > b[1] and b[2] > b[0])
    return min(2.5, max(0.5, a[len(a) // 2])) if a else 4 / 3


def _auto_cols(n: int) -> int:
    return {0: 1, 1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}.get(n, 4)


def _rows(items: list[dict], cols: int, rows_field: str | None) -> list[list[dict]]:
    """Items in rows of at most ``cols``; a new row at each new value of ``rows_field``, and for each view whose
    channels are split (its items share a ``row``)."""
    out: list[list[dict]] = []
    last, last_row = object(), object()
    for it in items:
        g = it.get('group', None) if rows_field else None
        rk = it.get('row')
        if not out or len(out[-1]) >= cols or (rows_field and g != last) or rk != last_row:
            out.append([])
        out[-1].append(it)
        last, last_row = g, rk
    return out


def _channel_items(v: dict, group: Any, ch: dict) -> list[dict]:
    """A view's items: one, or — split — one per shown channel plus the merge (fluorescence with 2+ channels)."""
    chans = v.get('channels') or []
    shown = [i for i, c in enumerate(chans) if ch['show'] is None or c['name'] in ch['show']] if chans else []
    base = {'kind': 'view', 'v': v, 'group': group}
    if not v.get('fluorescence') or len(chans) < 2:
        return [base]
    if ch['mode'] != 'split':
        return [{**base, 'ch': shown if ch['show'] is not None else None}]
    out = [{**base, 'ch': [i], 'gray': ch['gray'], 'colhead': (chans[i]['name'], None if ch['gray'] else chans[i]['color']),
            'row': int(v['id'])} for i in shown]  # fmt: skip
    if ch['merge'] or not out:
        out.append({**base, 'ch': shown, 'colhead': ('Merge', None), 'row': int(v['id'])})
    out[0]['first'] = True
    out[-1]['last'] = True
    return out


def layout(
    recipe: dict,
    members: list[dict],
    *,
    name: str = '',
    template: dict | None = None,
) -> dict:
    """The figure's pages: {'pages': [{'w', 'h', 'bg', 'ink', 'title', 'heads', 'windows'}], 'warnings', 'letters'}.

    ``members``: the views in order (:func:`select`), each with ``id``, ``bounds`` (level-0 px), ``meta``, and,
    when its slide is open, ``mpp`` (µm / px) and ``fluorescence``. ``template``: a PowerPoint template's description
    (:func:`wissi_viz.pptxio.read_template`) for ``pptx``. Windows: {'key', 'page', 'rect' [x, y, w, h] mm, 'kind',
    'view' (id), 'bounds' (what to read, widened to the window's shape), 'letter', 'label', 'label_rect',
    'scalebar': {'um', 'mm', 'color'}, 'text', 'edited'}."""
    r = normalize(recipe)
    L = r['layout']
    tpl = r['template']
    warnings: list[str] = []
    if tpl == 'pptx' and not template:
        warnings.append('no PowerPoint template: add one, or choose Presentation')
        tpl = 'ppt'
    slide = tpl != 'pub'
    fl = [v.get('fluorescence') for v in members]
    all_fl = bool(members) and all(f is True for f in fl)
    bg = r['background']
    if bg == 'auto':
        bg = None if tpl == 'pptx' else ('black' if all_fl else 'white')
    dark = bg == 'black'
    ink = '#ffffff' if dark else '#111111'
    font = r['font'] or (14.0 if slide else 8.0)
    gap = L['gap'] if L['gap'] is not None else (4.0 if slide else 1.5)
    lab_h = (font * PT * 1.25 + (1.2 if slide else 0.6)) if r['labels'].strip() else 0.0
    aspect = _aspect(members, L)
    rows_field = L['rows'] if L['mode'] == 'field' else None
    items = [it for v in members for it in _channel_items(v, field_value(v, rows_field) if rows_field else None,
                                                          r['channels'])]  # fmt: skip
    split = any('row' in it for it in items)
    head_w = (font * PT * 1.6 + 1.0) if rows_field or split else 0.0
    col_h = font * PT * 1.6 if split else 0.0
    per_view = max([sum(1 for it in items if it.get('row') == k) for k in {it.get('row') for it in items}] or [1])
    chart_on = r['chart'] is not None

    def head_text(row: list[dict]) -> str:
        if split and row and 'row' in row[0]:
            return label(r['labels'], row[0]['v']) or str(row[0]['v'].get('name') or '')
        return _fmt(row[0].get('group')) if row else ''

    pages: list[dict] = []
    if tpl == 'pptx':
        W, H = template['size']
        lay = _template_layout(template, (r['pptx'] or {}).get('layout'))
        if lay is None:
            warnings.append('the template has no layouts')
    elif tpl == 'ppt':
        W, H = PAGES[r['page']]
        lay = None
    else:
        W, H = PAGES[r['page']]
    slots_mode = tpl == 'pptx' and lay is not None and bool(lay['slots'])
    if split and not slots_mode:  # a row per view: its name heads the row, channel names head the columns
        lab_h = 0.0
    if slots_mode:  # windows in the template's places: each says what it shows
        col_h = 0.0

    def new_page(h=None) -> dict:
        p = {'w': W, 'h': h if h is not None else H, 'bg': bg, 'ink': ink, 'title': None, 'heads': [], 'windows': []}
        pages.append(p)
        return p

    def window(page: dict, rect, it: dict | None, kind: str | None = None) -> dict:
        w = {'key': f'{len(pages) - 1}:{len(page["windows"])}', 'page': len(pages) - 1, 'rect': [*map(float, rect)],
             'kind': kind or (it['kind'] if it else 'empty'), 'edited': False}  # fmt: skip
        if it and it['kind'] == 'view':
            w['view'] = int(it['v']['id'])
            if it.get('ch') is not None:
                w['channels'] = list(it['ch'])
            if it.get('gray'):
                w['gray'] = True
            for k in ('first', 'last'):
                if it.get(k):
                    w[k] = True
            if 'row' in it:
                w['split'] = True
            if it.get('colhead') and slots_mode:
                w['chlabel'] = it['colhead'][0]
            elif it.get('colhead') and not any(c['x'] == round(rect[0], 2) for c in page.setdefault('cols', [])):
                if len(page['windows']) == 0 or page.get('_colrow') in (None, it.get('row')):
                    page['_colrow'] = it.get('row')
                    t, col = it['colhead']
                    page['cols'].append({'text': t, 'color': col, 'x': round(rect[0], 2),
                                         'rect': [rect[0], rect[1] - col_h, rect[2], col_h]})  # fmt: skip
        page['windows'].append(w)
        return w

    if tpl == 'pub':
        m = 2.0
        cw = W - 2 * m
        n_cols = L['cols'] if L['cols'] != 'auto' else (
            min(8, per_view) if split else
            min(6, max([sum(1 for it in items if it['group'] == g) for g in {it['group'] for it in items}] or [1]))
            if rows_field else _auto_cols(len(items)))  # fmt: skip
        ww = (cw - head_w - gap * (n_cols - 1)) / n_cols
        wh = ww / aspect
        y = m + col_h
        page = new_page(None)
        for row in _rows(items, n_cols, rows_field):
            if H is not None and y + wh + lab_h > H - m and page['windows']:
                page['h'] = H
                page = new_page(None)
                y = m + col_h
            ht = head_text(row)
            if (rows_field or split) and (not page['heads'] or page['heads'][-1]['text'] != ht or split):
                page['heads'].append({'text': ht, 'rect': [m, y, head_w, wh]})
            elif rows_field:  # a group that wraps onto another row: the head spans its rows
                hd = page['heads'][-1]
                hd['rect'][3] = y + wh - hd['rect'][1]
            for j, it in enumerate(row):
                window(page, [m + head_w + j * (ww + gap), y, ww, wh], it)
            y += wh + lab_h + gap
        if chart_on:
            cwid = min(cw - head_w, max(60.0, (cw - head_w) * 0.5))
            ch = cwid / 1.6
            if H is not None and y + ch > H - m and page['windows']:
                page['h'] = H
                page = new_page(None)
                y = m
            window(page, [m + head_w, y, cwid, ch], None, 'chart')
            y += ch + gap
        for p in pages:
            if p['h'] is None:
                p['h'] = H if H is not None else round(y - gap + m, 2)
    else:
        title_text = r['title'] or name
        if tpl == 'pptx' and lay is not None and lay['slots']:
            slots = lay['slots']
            seq = items + ([{'kind': 'chart'}] if chart_on else [])
            for k in range(0, max(1, len(seq)), len(slots)):
                page = new_page()
                page['title'] = {'text': title_text, 'rect': lay.get('title')} if lay.get('title') else None
                for j, rect in enumerate(slots):
                    it = seq[k + j] if k + j < len(seq) else None
                    x, y0, w, h = rect
                    window(page, [x, y0, w, max(1.0, h - lab_h)], it, None if it else 'empty')
        else:
            m = 12.0
            title_h = 22.0
            box = [m, m + title_h, W - 2 * m, H - 2 * m - title_h - 6.0]
            if tpl == 'pptx' and lay is not None:
                t = lay.get('title')
                top = (t[1] + t[3] + 4.0) if t else m
                box = [m, top, W - 2 * m, H - top - 14.0]
            box = [box[0], box[1] + col_h, box[2], box[3] - col_h]
            seq = items + ([{'kind': 'chart', 'group': object()}] if chart_on else [])
            cap = L['per_page']
            bx, by, bw, bh = box
            if split:
                n_cols = L['cols'] if L['cols'] != 'auto' else min(8, per_view)
                n_rows = max(1, cap // n_cols) if cap >= n_cols else 1
            elif rows_field:
                n_cols = (
                    L['cols']
                    if L['cols'] != 'auto'
                    else min(
                        cap,
                        max([sum(1 for it in items if it['group'] == g) for g in {it['group'] for it in items}] or [1]),
                    )
                )
                n_rows = max(1, cap // n_cols)
            else:
                k = min(cap, max(1, len(seq)))
                best = None
                for c in range(1, k + 1) if L['cols'] == 'auto' else (L['cols'],):
                    rr = math.ceil(k / c)
                    w_ = min((bw - head_w - gap * (c - 1)) / c, ((bh - gap * (rr - 1)) / rr - lab_h) * aspect)
                    if best is None or w_ > best[0] + 1e-6:
                        best = (w_, c, rr)
                n_cols, n_rows = best[1], best[2]
            ww = min((bw - head_w - gap * (n_cols - 1)) / n_cols, ((bh - gap * (n_rows - 1)) / n_rows - lab_h) * aspect)
            ww = max(ww, 5.0)
            wh = ww / aspect
            x_off = bx + (bw - (head_w + n_cols * ww + (n_cols - 1) * gap)) / 2
            rows = _rows(seq, n_cols, rows_field) if seq else [[]]
            for k in range(0, len(rows), n_rows):
                page = new_page()
                page['title'] = {'text': title_text, 'rect': (lay.get('title') if tpl == 'pptx' and lay else None)
                                 or [m, m, W - 2 * m, title_h - 4.0]}  # fmt: skip
                for i, row in enumerate(rows[k : k + n_rows]):
                    y = by + i * (wh + lab_h + gap)
                    if (rows_field or split) and row and row[0]['kind'] == 'view':
                        g = head_text(row)
                        if page['heads'] and page['heads'][-1]['text'] == g and not split:
                            hd = page['heads'][-1]
                            hd['rect'][3] = y + wh - hd['rect'][1]
                        else:
                            page['heads'].append({'text': g, 'rect': [x_off, y, head_w, wh]})
                    for j, it in enumerate(row):
                        window(page, [x_off + head_w + j * (ww + gap), y, ww, wh], it)

    # by hand: moved / resized / changed windows, and added ones
    by_key = {w['key']: w for p in pages for w in p['windows']}
    for key, o in (r['windows'] or {}).items():
        w = by_key.get(key)
        if w is None or not isinstance(o, dict):
            continue
        _apply(w, o)
    for i, x in enumerate(r['extra']):
        if not isinstance(x, dict) or not _is_int(x.get('page')) or not (0 <= int(x['page']) < len(pages)):
            continue
        p = pages[int(x['page'])]
        w = {'key': f'x{i}', 'page': int(x['page']), 'rect': [10.0, 10.0, 40.0, 30.0], 'kind': 'empty',
             'edited': True, 'extra': True}  # fmt: skip
        _apply(w, x)
        p['windows'].append(w)

    # what each window shows: the view's picture (its frame widened to the window's shape), its label and scale bar
    by_id = {int(v['id']): v for v in members}
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sb_um = r['scalebar']
    view_ws = []
    for p in pages:
        n = 0
        for w in p['windows']:
            if w['kind'] == 'view' and w.get('view') not in by_id:
                w['kind'] = 'empty'
                w['missing'] = True
            if w['kind'] in ('view', 'chart') and r['letters'] and (not w.get('split') or w.get('first')):
                w['letter'] = letters[n % 26] * (1 + n // 26)
                n += 1
            x, y, ww_, wh_ = w['rect']
            if w['kind'] == 'view':
                v = by_id[w['view']]
                w['bounds'] = expand(v['bounds'], ww_ / max(wh_, 1e-6))
                w['sid'] = v.get('sid')
                if not w.get('split') or w.get('chlabel'):
                    lab = label(r['labels'], v)
                    w['label'] = f'{lab} · {w["chlabel"]}' if w.get('chlabel') and lab else w.get('chlabel') or lab
                    w['label_rect'] = [x, y + wh_ + 0.3, ww_, lab_h]
                w['fluorescence'] = bool(v.get('fluorescence'))
                view_ws.append((w, v))
        p.pop('_colrow', None)
        for c in p.get('cols', []):
            c.pop('x', None)
    if sb_um == 'auto':
        widths = [(w['bounds'][2] - w['bounds'][0]) * v['mpp'] for w, v in view_ws if v.get('mpp')]
        sb_um = _nice(min(widths) / 5) if widths else None
    no_mpp = 0
    for w, v in view_ws:
        if not sb_um or (w.get('split') and not w.get('last')):
            continue
        if not v.get('mpp'):
            no_mpp += 1
            continue
        um_w = (w['bounds'][2] - w['bounds'][0]) * v['mpp']
        w['scalebar'] = {'um': sb_um, 'mm': w['rect'][2] * sb_um / um_w,
                         'color': '#ffffff' if v.get('fluorescence') else '#111111',
                         'text': _um_text(sb_um)}  # fmt: skip
    if no_mpp:
        warnings.append(f'{no_mpp} view{"s" if no_mpp > 1 else ""} without a pixel size: no scale bar')
    return {'pages': pages, 'warnings': warnings, 'background': bg, 'dark': dark, 'font': font,
            'scalebar_um': sb_um, 'aspect': aspect}  # fmt: skip


def _um_text(um: float) -> str:
    if um >= 1000:
        return f'{um / 1000:g} mm'
    return f'{um:g} µm'


def _apply(w: dict, o: dict) -> None:
    rect = o.get('rect')
    if isinstance(rect, (list, tuple)) and len(rect) == 4 and all(isinstance(v, (int, float)) for v in rect):
        if rect[2] > 1 and rect[3] > 1:
            w['rect'] = [float(v) for v in rect]
            w['edited'] = True
    sh = o.get('shows')
    if isinstance(sh, dict) and sh.get('kind') in SHOWS:
        keep = w.get('view') if sh['kind'] == 'view' and 'view' not in sh else None  # channels of the same view
        w['kind'] = sh['kind']
        w['edited'] = True
        if keep is None:
            w.pop('view', None)
            w.pop('channels', None)
        if sh['kind'] == 'view' and _is_int(sh.get('view')):
            w['view'] = int(sh['view'])
        if sh['kind'] == 'view' and isinstance(sh.get('channels'), list):
            w['channels'] = [int(c) for c in sh['channels'] if _is_int(c)]
        if sh['kind'] == 'text':
            w['text'] = str(sh.get('text') or '')


def _template_layout(template: dict, name: str | None) -> dict | None:
    lays = template.get('layouts') or []
    if not lays:
        return None
    for lay in lays:
        if lay['name'] == name:
            return lay
    return next((x for x in lays if x['slots']), lays[0])


# -------------------------------------------------------------------------------------------------------- charts
def chart_series(recipe: dict, members: list[dict], values: dict[int, float | None], dark: bool = False) -> dict:
    """The chart window's data: one value per view, in groups (the ``group`` field; one group without it)."""
    r = normalize(recipe)
    c = r['chart'] or {'kind': 'box', 'measure': {'kind': 'count'}, 'group': None}
    gf = c.get('group')
    groups: dict[str, list[dict]] = {}
    for v in members:
        g = _fmt(field_value(v, gf)) if gf else 'All'
        groups.setdefault(g or '–', []).append({'view': int(v['id']), 'value': values.get(int(v['id'])),
                                                 'label': label(r['labels'], v)})  # fmt: skip
    names = sorted(groups, key=_sort_key) if gf else list(groups)
    pal = COLORS_DARK if dark else COLORS
    return {'kind': c['kind'], 'ylabel': measure_label(c['measure']), 'group': gf,
            'groups': [{'name': g, 'color': pal[i % len(pal)], 'values': groups[g]} for i, g in enumerate(names)]}  # fmt: skip


def measure_label(m: dict) -> str:
    k = m.get('kind')
    what = m.get('class') or m.get('layer') or 'objects'
    if k == 'density':
        return f'{what} per mm²'
    if k == 'field':
        return str(m.get('field') or 'value')
    return f'{what} per view'


def box_stats(vals: list[float]) -> dict | None:
    """Quartiles (linear), whiskers at the most extreme values within 1.5 IQR, and the mean."""
    v = sorted(x for x in vals if x is not None and math.isfinite(x))
    if not v:
        return None

    def q(p: float) -> float:
        i = (len(v) - 1) * p
        lo, hi = math.floor(i), math.ceil(i)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)

    q1, md, q3 = q(0.25), q(0.5), q(0.75)
    iqr = q3 - q1
    lo = min(x for x in v if x >= q1 - 1.5 * iqr)
    hi = max(x for x in v if x <= q3 + 1.5 * iqr)
    return {'q1': q1, 'median': md, 'q3': q3, 'lo': lo, 'hi': hi, 'mean': sum(v) / len(v), 'n': len(v)}


def chart_csv(series: dict) -> str:
    """The chart's data as CSV: group, view id, label, value."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([series.get('group') or 'group', 'view', 'label', series.get('ylabel') or 'value'])
    for g in series['groups']:
        for x in g['values']:
            w.writerow([g['name'], x['view'], x['label'], '' if x['value'] is None else x['value']])
    return out.getvalue()


def count_in(xs, ys, bounds) -> int:
    """How many centroids lie inside a rectangle (level-0 px)."""
    x0, y0, x1, y1 = bounds
    return int(((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).sum())


def measure_values(chart: dict | None, members: list[dict], objects_of: Callable[[str, str], Any]) -> dict:
    """{view id: value} for the chart's measure. ``objects_of(sid, layer)``: an object table (``x``, ``y``,
    ``classes``) of an open slide's layer, or None."""
    out: dict[int, float | None] = {}
    if not chart:
        return out
    m = chart.get('measure') or {}
    for v in members:
        vid = int(v['id'])
        if m.get('kind') == 'field':
            out[vid] = _num(field_value(v, m.get('field') or ''))
            continue
        sid = v.get('sid')
        objs = objects_of(sid, m.get('layer') or '') if sid and m.get('layer') else None
        if objs is None:
            out[vid] = None
            continue
        xs, ys = np.asarray(objs.x), np.asarray(objs.y)
        if m.get('class'):
            cls = getattr(objs, 'classes', None)
            keep = np.asarray(cls) == m['class'] if cls is not None else np.zeros(len(xs), bool)
            xs, ys = xs[keep], ys[keep]
        n = count_in(xs, ys, v['bounds'])
        if m.get('kind') == 'density':
            x0, y0, x1, y1 = v['bounds']
            mpp = v.get('mpp')
            out[vid] = n / ((x1 - x0) * (y1 - y0) * mpp * mpp / 1e6) if mpp else None
        else:
            out[vid] = float(n)
    return out
