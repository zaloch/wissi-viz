"""wissi-viz's tools for a language model, defined once: each has a name, a description written for the model, a JSON
schema for its arguments and a function of the :class:`Host` (the app's session and window). :mod:`.mcp_server`
serves them over MCP; :mod:`.llm` hands them to a local model as OpenAI-style functions.

Tools that change what the window shows send a command to the open app pages (:class:`.ui.UiBus`); the others read
or change the session (slides, layers, saved views, figures) and work without a window too."""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .. import figures, summary
from ..regions import draw_layers, region_rgb
from ..server import layer_features
from .ui import UiBus


@dataclass
class Picture:
    """A tool result that is a picture (PNG): an image for models that see, a sentence for those that do not."""

    png: bytes
    text: str


@dataclass
class Tool:
    name: str
    description: str
    params: dict = field(default_factory=dict)
    required: tuple = ()
    fn: Callable | None = None
    #: changes what the window shows (needs an open app page to be seen)
    ui: bool = False
    #: changes data (files, layers, saved views): a client may ask before running it
    writes: bool = False

    def schema(self) -> dict:
        return {'type': 'object', 'properties': self.params, 'required': list(self.required),
                'additionalProperties': False}  # fmt: skip


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, params: dict | None = None, required=(), ui: bool = False, writes: bool = False):
    def deco(fn):
        TOOLS[name] = Tool(name, description, params or {}, tuple(required), fn, ui, writes)
        return fn

    return deco


def S(t: str, desc: str, **kw) -> dict:
    return {'type': t, 'description': desc, **kw}


BOUNDS = S('array', 'Rectangle in level-0 pixels: [x0, y0, x1, y1]', items={'type': 'number'}, minItems=4, maxItems=4)


class Host:
    """What the tools act on: the app's :class:`~wissi_viz.session.Session` and its window(s)."""

    def __init__(self, session, ui: UiBus | None = None, exports: str | Path | None = None) -> None:
        self.s = session
        self.ui = ui or getattr(session, 'ui', None) or UiBus()
        self.exports = Path(exports) if exports else Path(session.cache_dir) / 'exports'

    def slide(self, ref: str) -> str:
        """An open slide's id from its id, name, file name, stem or path."""
        by = self.s._slide_of()
        sid = by.get(str(ref).strip().lower())
        if sid is None:
            names = ', '.join(x['name'] for x in self.s.slides()[:12])
            raise ValueError(f'no open slide {ref!r}; open: {names or "none"}')
        return sid

    def tell(self, cmd: dict) -> str:
        return '' if self.ui.send(cmd) else ' (no app window is open: open one to see it)'


def call(host: Host, name: str, args: dict | None) -> Any:
    """Run a tool; errors come back as {'error': message} (for the model to read and recover from)."""
    t = TOOLS.get(name)
    if t is None:
        return {'error': f'no tool {name!r}; tools: {", ".join(TOOLS)}'}
    try:
        return t.fn(host, **(args or {}))
    except TypeError as e:
        return {'error': f'{name}: {e}'}
    except (ValueError, KeyError, PermissionError, FileNotFoundError, ImportError) as e:
        return {'error': str(e).strip("'")}


def schemas() -> list[dict]:
    """The tools as OpenAI-style function definitions (for local models)."""
    return [{'type': 'function', 'function': {'name': t.name, 'description': t.description, 'parameters': t.schema()}}
            for t in TOOLS.values()]  # fmt: skip


def _num(v):
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    return _num(x)


# ------------------------------------------------------------------------------------------------------- slides
@tool('list_slides', 'The slides open in the app: id, name, path, size, pixel size (µm/px), mode (brightfield / '
      'fluorescence channels), their metadata fields, and which panes of the window show them.')  # fmt: skip
def list_slides(h: Host) -> dict:
    out = []
    panes = h.ui.state.get('panes') or []
    for x in h.s.slides():
        info = h.s.registry.info(x['id'])
        out.append({'id': x['id'], 'name': x['name'], 'path': x['path'], 'width': info.get('width'),
                    'height': info.get('height'), 'mpp': (info.get('mpp') or [None])[0], 'mode': info.get('mode'),
                    'channels': [c.get('name') for c in info.get('channels') or []] or None,
                    'meta': dict(list(x['meta'].items())[:16]),
                    'panes': [i for i, p in enumerate(panes) if p and p.get('slide') == x['id']]})  # fmt: skip
    return {'slides': out}


@tool('open_slides', 'Open slide files or folders (paths on this computer or server) in the app.',
      {'paths': S('array', 'File or folder paths', items={'type': 'string'})}, ['paths'], writes=True)  # fmt: skip
def open_slides(h: Host, paths: list[str]) -> dict:
    if not h.s.can_open:
        raise PermissionError('this server does not open files')
    r = h.s.open(paths)
    if r['opened']:
        h.tell({'type': 'slides', 'slides': h.s.slides()})
    return {'opened': r['opened'], 'errors': r['errors']}


@tool('show_slides', 'Show slides in the window, one per pane (up to 32), replacing what the panes show.',
      {'slides': S('array', 'Slide ids or names, one per pane', items={'type': 'string'})}, ['slides'], ui=True)  # fmt: skip
def show_slides(h: Host, slides: list[str]) -> dict:
    ids = [h.slide(x) for x in slides][:32]
    note = h.tell({'type': 'show', 'slides': ids})
    return {'shown': ids, 'note': note.strip() or None}


@tool('go_to', 'Move a pane to a region of its slide: a rectangle in level-0 pixels, or a centre (x, y) with a width '
      'in µm. Opens the slide in that pane if needed.',
      {'slide': S('string', 'Slide id or name'), 'bounds': BOUNDS, 'x': S('number', 'Centre x, level-0 px'),
       'y': S('number', 'Centre y, level-0 px'), 'width_um': S('number', 'Width of the region in µm'),
       'pane': S('integer', 'Pane index (0 = first); default: the active pane')}, ['slide'], ui=True)  # fmt: skip
def go_to(h: Host, slide: str, bounds=None, x=None, y=None, width_um=None, pane=None) -> dict:
    sid = h.slide(slide)
    info = h.s.registry.info(sid)
    if bounds is None:
        if x is None or y is None:
            bounds = [0, 0, info['width'], info['height']]
        else:
            mpp = (info.get('mpp') or [None])[0] or 0.5
            w = (width_um or 500) / mpp
            hgt = w * 0.75
            bounds = [x - w / 2, y - hgt / 2, x + w / 2, y + hgt / 2]
    b = [float(v) for v in bounds]
    note = h.tell({'type': 'goto', 'slide': sid, 'bounds': b, 'pane': pane})
    return {'slide': sid, 'bounds': [round(v, 1) for v in b], 'note': note.strip() or None}


@tool('current_view', 'What the window shows now: each pane\'s slide and visible rectangle (level-0 px), and which '
      'pane is active.')  # fmt: skip
def current_view(h: Host) -> dict:
    if not h.ui.state:
        return {'panes': [], 'note': 'no app window has reported its view'}
    return _clean(h.ui.state)


@tool('view_image', 'A picture (PNG) of a region of a slide, with chosen layers outlined. For models that see.',
      {'slide': S('string', 'Slide id or name'), 'bounds': BOUNDS,
       'size': S('integer', 'Longer side in pixels (64–1024, default 768)'),
       'layers': S('array', 'Layers to outline', items={'type': 'string'}),
       'channels': S('array', 'Fluorescence channels to show (names); default: all', items={'type': 'string'})},
      ['slide'])  # fmt: skip
def view_image(h: Host, slide: str, bounds=None, size: int = 768, layers=None, channels=None) -> Picture:
    sid = h.slide(slide)
    e, pyr = h.s.registry.get(sid)
    info = pyr.info
    b = bounds or [0, 0, info.width, info.height]
    chans = None
    if channels and info.mode != 'rgb':
        names = [c.name for c in info.channels]
        chans = [names.index(c) for c in channels if c in names]
    a = region_rgb(e.source, b, max(64, min(int(size or 768), 1024)), e.windows, chans)
    if layers:
        a = draw_layers(a, b, [layer_features(h.s.registry, sid, n) for n in layers])
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, 'PNG')
    mpp = (info.mpp or (None,))[0]
    w_um = f', {(b[2] - b[0]) * mpp:.0f} µm wide' if mpp else ''
    return Picture(
        buf.getvalue(), f'{info.name}: [{", ".join(f"{v:.0f}" for v in b)}]{w_um}, {a.shape[1]}×{a.shape[0]} px'
    )


# ------------------------------------------------------------------------------------------------------ layers
@tool('list_layers', 'The layers of a slide (cells, annotations, analysis results): name, number of objects, the '
      'number of objects of each class, where they came from.', {'slide': S('string', 'Slide id or name')}, ['slide'])  # fmt: skip
def list_layers(h: Host, slide: str) -> dict:
    sid = h.slide(slide)
    out = []
    for n, L in h.s.registry.layers(sid).items():
        out.append({'name': n, 'objects': len(L), 'classes': _class_counts(L), 'origin': L.origin or None})
    return {'slide': sid, 'layers': out}


def _class_counts(L) -> dict:
    """{class: number of objects}, the 24 largest classes."""
    try:
        ids = np.bincount(np.asarray(L._all_cls(), np.int64))
        inv = {i: k for k, i in L._class_index.items()}
    except (AttributeError, TypeError, ValueError):  # a layer kept another way: class names only
        return dict.fromkeys(list(L.classes)[:24])
    counts = {inv.get(i, f'class {i}'): int(c) for i, c in enumerate(ids) if c}
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:24])


@tool('layer_summary', 'Counts and shares per class, per-class means of the main measurements, region areas and '
      'objects per mm² of tissue for one layer.',
      {'slide': S('string', 'Slide id or name'), 'layer': S('string', 'Layer name')}, ['slide', 'layer'])  # fmt: skip
def layer_summary(h: Host, slide: str, layer: str) -> dict:
    sid = h.slide(slide)
    L = h.s.registry.layer(sid, layer)
    mpp = (h.s.registry.info(sid).get('mpp') or [None])[0]
    return _clean(summary.summary(L, mpp, None))


@tool('show_layer', 'Show or hide a layer (optionally one class of it) in every pane of a slide.',
      {'slide': S('string', 'Slide id or name'), 'layer': S('string', 'Layer name'),
       'visible': S('boolean', 'Show (true) or hide (false)'), 'class': S('string', 'Only this class of the layer')},
      ['slide', 'layer', 'visible'], ui=True)  # fmt: skip
def show_layer(h: Host, slide: str, layer: str, visible: bool, **kw) -> dict:
    sid = h.slide(slide)
    h.s.registry.layer(sid, layer)
    note = h.tell({'type': 'layer', 'slide': sid, 'layer': layer, 'visible': bool(visible), 'class': kw.get('class')})
    return {'ok': True, 'note': note.strip() or None}


@tool('set_channels', 'Fluorescence slides. Split into a pane per channel plus the merge: {slide, split: true}. Back '
      'to one pane: {slide, split: false}. Show only some channels: {slide, show: [channel names, e.g. DAPI]}.',
      {'slide': S('string', 'Slide id or name'), 'split': S('boolean', 'true: a pane per channel + merge; false: one pane'),
       'show': S('array', 'Channel names to show (not slide names)', items={'type': 'string'})}, ['slide'], ui=True)  # fmt: skip
def set_channels(h: Host, slide: str, show=None, split=None) -> dict:
    sid = h.slide(slide)
    info = h.s.registry.info(sid)
    if info.get('mode') != 'channels':
        raise ValueError(f'{info.get("name")} is brightfield: it has no channels')
    names = [c['name'] for c in info['channels']]
    bad = [c for c in show or [] if c not in names]
    if bad:
        raise ValueError(f'no channel {bad[0]!r}; channels: {", ".join(names)} (to split into panes: split: true)')
    note = h.tell({'type': 'channels', 'slide': sid, 'show': show, 'split': split})
    return {'channels': names, 'note': note.strip() or None}


# ---------------------------------------------------------------------------------------------------- analyses
@tool('list_analyses', 'The spatial analyses that can be run on a layer of cells, with their parameters.')
def list_analyses(h: Host) -> dict:
    from .. import analyses

    return {'analyses': _clean(analyses.catalog())}


@tool('run_analysis', 'Run a spatial analysis (see list_analyses) on a layer of cells of a slide. Adds result layers '
      'to the slide and returns headline numbers.',
      {'slide': S('string', 'Slide id or name'), 'analysis': S('string', 'Analysis id'),
       'objects': S('string', 'Layer of cells'), 'params': S('object', 'Parameters by name'),
       'within_layer': S('string', 'Only cells inside the regions of this layer')},
      ['slide', 'analysis', 'objects'], writes=True)  # fmt: skip
def run_analysis(h: Host, slide: str, analysis: str, objects: str, params=None, within_layer=None) -> dict:
    from .. import analyses

    sid = h.slide(slide)
    out = analyses.run(h.s.registry, sid, analysis, objects, params or {},
                       within={'layer': within_layer} if within_layer else None)  # fmt: skip
    h.tell({'type': 'layers', 'slide': sid})
    return _clean({k: out[k] for k in ('title', 'kpis', 'note', 'layers', 'seconds')})


# ------------------------------------------------------------------------------------------------- saved views
@tool('list_views', 'Saved views (regions of slides with metadata): id, name, slide, rectangle, fields. Optional '
      'rules filter them like a figure query.',
      {'rules': S('array', 'Rules: {field, op: in | not in | >= | <= | has | missing, values}', items={'type': 'object'})})  # fmt: skip
def list_views(h: Host, rules=None) -> dict:
    every = h.s.figure_views()
    vs = [v for v in every if figures.matches(v, rules)] if rules else every
    out = {'views': [{'id': v['id'], 'name': v['name'], 'slide': v['slide'], 'open': bool(v.get('sid')),
                      'bounds': v['bounds'], 'fields': v['meta']} for v in vs[:200]],
           'fields': figures.fields_of(vs)}  # fmt: skip
    if rules and not vs and every:
        vals: dict[str, set] = {}
        for v in every:
            for k, x in (v.get('meta') or {}).items():
                vals.setdefault(k, set()).add(str(x))
        out['note'] = f'no view matches; {len(every)} views; their fields and values: ' + '; '.join(
            f'{k}: {", ".join(sorted(x)[:8])}' for k, x in list(vals.items())[:12]
        )
    return out


@tool('save_view', 'Save a region of an open slide as a view (point of interest), with a name and metadata fields.',
      {'slide': S('string', 'Slide id or name'), 'bounds': BOUNDS, 'name': S('string', 'Name of the view'),
       'fields': S('object', 'Metadata: {field: value}')}, ['slide', 'bounds'], writes=True)  # fmt: skip
def save_view(h: Host, slide: str, bounds, name: str = '', fields=None) -> dict:
    r = h.s.add_view(h.slide(slide), bounds, name, fields or {})
    h.tell({'type': 'views'})
    return {'view': r['view']}


# ------------------------------------------------------------------------------------------------------ figures
@tool('make_figure', 'Make a figure of saved views (the Figures screen): a fixed list of view ids or rules over their '
      'fields, a template (pub: journal page, ppt: slides), rows by a field, labels from fields, an optional chart.',
      {'name': S('string', 'Figure name'), 'views': S('array', 'View ids', items={'type': 'integer'}),
       'rules': S('array', 'Or rules: {field, op, values}', items={'type': 'object'}),
       'template': S('string', 'pub | ppt', enum=['pub', 'ppt']), 'rows_by': S('string', 'A row per value of this field'),
       'labels': S('string', "Label template, e.g. '{Animal} · {Region}'"),
       'chart_field': S('string', 'Chart: one value per view from this numeric field'),
       'channels': S('string', 'Fluorescence: merge | split', enum=['merge', 'split'])}, ['name'], writes=True)  # fmt: skip
def make_figure(h: Host, name: str, views=None, rules=None, template='pub', rows_by=None, labels=None, chart_field=None,
                channels=None) -> dict:  # fmt: skip
    recipe: dict = {'template': template or 'pub',
                    'source': {'kind': 'query', 'rules': rules} if rules else {'kind': 'views', 'ids': list(views or [])}}  # fmt: skip
    if rows_by:
        recipe['layout'] = {'mode': 'field', 'rows': rows_by}
    if labels:
        recipe['labels'] = labels
    if chart_field:
        recipe['chart'] = {'kind': 'box', 'measure': {'kind': 'field', 'field': chart_field}, 'group': rows_by}
    if channels:
        recipe['channels'] = {'mode': channels}
    L = h.s.figure_layout(recipe, name, open_slides=False)
    if not L['members']:
        raise ValueError('no saved view matches: list_views shows the views and their fields')
    if recipe['source']['kind'] == 'query':
        recipe['members'] = L['members']
    f = h.s.store.save_figure(name, recipe)
    note = h.tell({'type': 'figure', 'id': f['id']})
    return {'figure': f['id'], 'views': L['members'], 'pages': len(L['pages']), 'warnings': L['warnings'],
            'note': note.strip() or None}  # fmt: skip


@tool('export_figure', 'Export a saved figure to a file: png, tiff, pdf, svg or pptx. Returns where it was written.',
      {'figure': S('string', 'Figure id or name'), 'format': S('string', 'png | tiff | pdf | svg | pptx',
       enum=['png', 'tiff', 'pdf', 'svg', 'pptx']), 'dpi': S('integer', '72–1200 (default 300)')},
      ['figure'], writes=True)  # fmt: skip
def export_figure(h: Host, figure: str, format: str = 'pdf', dpi: int = 300) -> dict:
    fs = h.s.store.figures()
    f = next((x for x in fs if str(x['id']) == str(figure) or x['name'] == figure), None)
    if f is None:
        raise ValueError(f'no figure {figure!r}; figures: {", ".join(x["name"] for x in fs[:12]) or "none"}')
    data, fname, _ = h.s.figure_export(f['recipe'], f['name'], format, dpi)
    h.exports.mkdir(parents=True, exist_ok=True)
    p = h.exports / fname
    p.write_bytes(data)
    return {'path': str(p), 'bytes': len(data)}


def tool_list() -> list[dict]:
    """The tools for people: name, what it does, whether it changes the window or data."""
    return [{'name': t.name, 'description': t.description, 'ui': t.ui, 'writes': t.writes} for t in TOOLS.values()]


def picture_b64(p: Picture) -> str:
    return base64.b64encode(p.png).decode()
