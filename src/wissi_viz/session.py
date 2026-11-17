"""What a viewer window has open, and how things get opened: slide files and folders, cell tables and
annotation files attached to a slide, and images from tile APIs. Shared by the desktop app and
``wissi-viz serve``; the HTTP endpoints the app page talks to are built by :func:`create_session_app`.

Who may open what:

* desktop (``roots=None``): anything the local user can read, plus any tile API;
* server: only files inside the folders given with ``--allow-open`` (none by default), and tile APIs only
  with ``--allow-connect``. Paths named on the command line are trusted.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import secrets
import threading
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import anyio
import numpy as np
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request  # module level: FastAPI resolves the string annotations here

from . import annio, figures, slidemeta
from ._compat import user_dir
from ._version import __version__
from .annotations import AnnotationLayer
from .connectors import RemoteConnection, image_entries, open_connection
from .objects import cached_object_layer
from .regions import region_rgb
from .registry import SLIDE_EXTS, SlideRegistry
from .server import app_page, create_app
from .store import Store
from .tiles import TileService

log = logging.getLogger(__name__)

PREFIX = '/wissi-viz'
ANNOTATION_EXTS = ('.geojson', '.json', '.xml', '.annotations', '.geoparquet', '.coco.json')
TABLE_EXTS = (
    '.csv',
    '.tsv',
    '.txt',
    '.parquet',
    '.pq',
    '.feather',
    '.arrow',
    '.ipc',
    '.h5ad',
)  # .parquet: GeoParquet if it has geometry
MAX_CONNECT_IMAGES = 1000  # per connection: the slide list stays usable, and nothing is fetched until shown


def _slide_exts() -> tuple[str, ...]:
    return SLIDE_EXTS


def known_kind(p: Path) -> str | None:
    """'folder' | 'slide' | 'annotations' | 'table', or None for files wissi-viz does not read."""
    n = p.name.lower()
    if p.is_dir():
        return 'folder'
    if n.endswith(_slide_exts()):
        return 'slide'
    if n.endswith(ANNOTATION_EXTS):
        return 'annotations'
    if n.endswith(TABLE_EXTS):
        return 'table'
    return None


def _kind(p: Path) -> str:
    return known_kind(p) or 'slide'  # let open_slide decide (it raises a clear error for unknown formats)


class Session:
    """The slides and layers open in one app window (or on one server).

    ``registry``: share an existing registry (``wissi-viz serve`` builds one from its cache options); else one is
    created. ``roots``: folders the user interface may open from (None = anywhere: the desktop app). ``connect``:
    whether the user interface may connect to tile APIs. ``remember``: keep a recent-files list.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike | None = None,
        memory_mb: int = 512,
        cache_gb: float = 10,
        decoded_cache_mb: int = 512,
        *,
        registry: SlideRegistry | None = None,
        roots: Sequence[str | os.PathLike] | None = None,
        connect: bool = True,
        remember: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else user_dir('cache')
        if registry is None:
            tiles = TileService(
                memory_bytes=memory_mb * 2**20, disk_dir=self.cache_dir / 'tiles', disk_bytes=int(cache_gb * 2**30)
            )
            registry = SlideRegistry(tiles, decoded_cache_mb=decoded_cache_mb)
        self.registry = registry
        self.tiles = registry.tiles
        self.roots = None if roots is None else [Path(r).expanduser().resolve() for r in roots]
        self.can_connect = connect
        self.remember = remember
        self.token = secrets.token_urlsafe(24)
        self._paths: dict[str, str] = {}  # slide id → file path, or 'service:image id' for API images
        self._connections: list[RemoteConnection] = []
        self._lock = threading.Lock()
        self._recent_file = user_dir('config') / 'recent.json'
        self._projects_file = user_dir('config') / 'projects.json'
        self.show: list[str] = []  # slide ids to show in panes on load (a session)
        self.views: dict[str, list[float]] = {}  # slide id → [x0, y0, x1, y1] to open on (a session's "view")
        #: the user's own groups of slides: [{'name', 'slides': [path, …]}] (paths, so they survive restarts)
        self.projects: list[dict] = self._load_projects()
        #: per-image metadata a tile API gave (slide id → {field: value}); metadata tables matched to slides by a key
        self.slide_meta: dict[str, dict] = {}
        self._meta_file = user_dir('config') / 'metadata.json'
        self.meta_tables: list[slidemeta.MetaTable] = []
        #: saved views (points of interest) and figure recipes: a file for a session that remembers, else in memory
        self.store = Store(user_dir('config') / 'wissi.db' if remember else ':memory:')
        self._auto_meta: dict[str, dict] = {}
        self._matched: tuple[tuple, list[dict]] | None = None
        self._fig_objects: dict[tuple, tuple] = {}
        self._template_info: dict[str, dict] = {}
        if remember:
            try:
                for t in json.loads(self._meta_file.read_text(encoding='utf-8')):
                    try:
                        self.meta_tables.append(slidemeta.MetaTable(t['path'], t.get('key')))
                    except Exception:  # noqa: BLE001  (moved or deleted since: dropped)
                        pass
            except (OSError, ValueError, TypeError):
                pass

    # -- permissions
    @property
    def can_open(self) -> bool:
        return self.roots is None or bool(self.roots)

    def allowed(self, p: str | os.PathLike) -> bool:
        """Inside one of the allowed folders (symlinks resolved), or anywhere when there are no limits."""
        if self.roots is None:
            return True
        rp = Path(p).expanduser().resolve()
        return any(rp == r or rp.is_relative_to(r) for r in self.roots)

    # -- opening files
    def open(
        self,
        paths: Iterable[str | os.PathLike] | None,
        slide: str | None = None,
        *,
        trusted: bool = False,
        eager: bool = True,
    ) -> dict:
        """Open slide files / folders; attach annotation files and cell tables to ``slide`` (else the first slide
        opened in this call, else the most recent one). ``trusted``: skip the folder limits (command-line paths).
        ``eager``: read each slide now, so unreadable files are reported here (a server with thousands of slides
        skips this and opens them when first shown). Returns {'opened', 'layers', 'errors', 'slides'}."""
        opened, layers, errors, extras = [], [], [], []
        for raw in paths or []:
            p = Path(str(raw)).expanduser()
            try:
                if not (trusted or self.allowed(p)):
                    raise PermissionError('outside the folders this server allows')
                if not p.exists():
                    raise FileNotFoundError(f'not found: {p}')
                k = _kind(p)
                if k in ('annotations', 'table'):
                    extras.append(p)
                    continue
                same = self._open_path(p) if k != 'folder' else None
                if same is not None:  # already open: show it (re-registering would close it under the viewer)
                    opened.append(same)
                    continue
                ids = self.registry.add_directory(p) if k == 'folder' else [self.registry.add(p)]
                if k == 'folder' and not ids:
                    raise ValueError(f'no slides in {p}')
                for sid in ids:
                    if eager:
                        self.registry.info(sid)  # open now: unreadable files fail here, not as a blank pane
                    self._paths[sid] = str(p if k != 'folder' else self.registry._entries[sid].path)
                    opened.append(sid)
            except Exception as e:  # noqa: BLE001  (reported to the user per file)
                errors.append(f'{p.name}: {e}')
                for sid in [s for s in self.registry.ids() if s not in self._paths]:
                    self.registry.remove(sid)
        target = (
            slide
            if slide in self._paths
            else (opened[0] if opened else (list(self._paths)[-1] if self._paths else None))
        )
        for p in extras:
            if target is None:
                errors.append(f'{p.name}: open a slide first · Then add cells / annotations')
                continue
            try:
                name = self.add_layer(target, p)
                layers.append({'slide': target, 'layer': name})
                n = getattr(self.registry.layer(target, name), 'skipped', 0)
                if n:
                    errors.append(f'{p.name}: {n} invalid shape{"s" if n > 1 else ""} left out (missing or NaN '
                                  'coordinates, or too few points); the rest is shown')  # fmt: skip
            except Exception as e:  # noqa: BLE001  (reported to the user per file)
                errors.append(f'{p.name}: {e}')
        self._remember([self._paths[s] for s in opened] + [str(p) for p in extras])
        return {'opened': opened, 'layers': layers, 'errors': errors, 'slides': self.slides()}

    def _open_path(self, p: Path) -> str | None:
        """The id of an open slide with this file, if any."""
        try:
            rp = p.resolve()
        except OSError:
            return None
        for sid, q in self._paths.items():
            if sid in self.registry and ':' not in q[2:] and Path(q).resolve() == rp:
                return sid
        return None

    def add_layer(self, slide: str, path: str | os.PathLike) -> str:
        """Attach an annotation file (GeoJSON / HALO XML / …) or a cell table (CSV / Parquet / AnnData) to a slide.
        Tables are compiled once into the cache and memory-mapped on later opens."""
        p = Path(path)
        try:
            fmt = annio.detect_format(p)
        except ValueError:
            fmt = None
        if fmt == 'coco' or (fmt == 'geoparquet' and _is_geoparquet(p)):
            layer = annio.load(p)
        elif _kind(p) == 'table' or (p.suffix.lower() == '.json' and _looks_like_table(p)):
            mpp = (self.registry.info(slide).get('mpp') or [None])[0]
            layer = cached_object_layer(p, self.cache_dir / 'objects', mpp=mpp, name=p.stem)
        else:
            layer = AnnotationLayer.load(p)
        if not layer.origin:
            layer.origin = 'imported'
        name = layer.name or p.stem
        if name in self.registry.layers(slide):
            name = f'{name} ({p.name})'
        self.registry.add_annotations(slide, layer, name=name)
        return name

    def browse(self, folder: str | None = None) -> dict:
        """One folder for the in-page file picker: sub-folders and the files wissi-viz reads, within the limits.
        No folder: the allowed folders (server) or the home folder (desktop)."""
        if not self.can_open:
            raise PermissionError('this server does not open files')
        if not folder:
            if self.roots is not None and len(self.roots) > 1:
                return {
                    'dir': '',
                    'parent': None,
                    'entries': [{'name': str(r), 'path': str(r), 'kind': 'folder'} for r in self.roots],
                }
            folder = str(self.roots[0] if self.roots else Path.home())
        base = Path(folder).expanduser().resolve()
        if not self.allowed(base):
            raise PermissionError('outside the folders this server allows')
        if not base.is_dir():
            raise NotADirectoryError(f'not a folder: {base}')
        entries = []
        for c in sorted(base.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if c.name.startswith('.') or not self.allowed(c):
                continue
            try:
                k = known_kind(c)
            except OSError:
                continue
            if k:
                e = {'name': c.name, 'path': str(c), 'kind': k}
                if k != 'folder':
                    try:
                        e['size'] = c.stat().st_size
                    except OSError:
                        pass
                entries.append(e)
            if len(entries) >= 2000:
                break
        if self.roots is None:
            parent = str(base.parent) if base.parent != base else None
        elif any(base == r for r in self.roots):
            parent = '' if len(self.roots) > 1 else None
        else:
            parent = str(base.parent)
        return {'dir': str(base), 'parent': parent, 'entries': entries, 'places': self.places()}

    def places(self) -> list[dict]:
        """Shortcuts for the file picker: home and its usual folders, mounted volumes and the folders of recent
        files (desktop); the allowed folders (server)."""
        if self.roots is not None:
            return [{'name': r.name or str(r), 'path': str(r), 'group': 'Folders'} for r in self.roots]
        home = Path.home()
        out = [{'name': 'Home', 'path': str(home), 'group': 'Places'}]
        for n in ('Desktop', 'Documents', 'Downloads', 'Pictures'):
            if (home / n).is_dir():
                out.append({'name': n, 'path': str(home / n), 'group': 'Places'})
        for vol in (Path('/Volumes'), Path('/media') / home.name, Path('/mnt')):
            try:
                for v in sorted(vol.iterdir()) if vol.is_dir() else []:
                    if v.is_dir() and not v.name.startswith('.'):
                        out.append({'name': v.name, 'path': str(v), 'group': 'Drives'})
            except OSError:
                pass
        if os.name == 'nt':
            for d in 'CDEFGH':
                if Path(f'{d}:/').exists():
                    out.append({'name': f'{d}:', 'path': f'{d}:\\', 'group': 'Drives'})
        seen = set()
        for f in self.recent():
            d = str(Path(f).parent) if not Path(f).is_dir() else f
            if d not in seen:
                seen.add(d)
                out.append({'name': Path(d).name or d, 'path': d, 'group': 'Recent'})
            if len(seen) >= 6:
                break
        return out

    # -- tile APIs
    def connect(self, spec: Any, images: Any = None, *, trusted: bool = False, **factory_kwargs: Any) -> dict:
        """Show the images of a tile API: ``spec`` is a service description (dict, JSON text, ``.json`` path) or,
        when ``trusted`` (the command line), ``module:factory``. ``images``: ids (list or comma-separated);
        default: every image the service lists. Images are registered lazily, so nothing is fetched until a pane
        shows one; the first is checked straight away so a wrong URL or token fails here."""
        if not (trusted or self.can_connect):
            raise PermissionError('this server does not connect to tile APIs')
        text = str(spec).strip() if isinstance(spec, (str, os.PathLike)) else ''
        is_file = (
            bool(text)
            and not text.startswith('{')
            and (text.lower().endswith('.json') or Path(text).expanduser().exists())
        )
        if is_file and not trusted and not self.allowed(text):
            raise PermissionError('the service file is outside the folders this server allows')
        # ${ENV} secrets: on the command line and in the desktop app (the user's own), not from a server's visitors
        conn = open_connection(spec, allow_code=trusted, expand_env=trusted or self.roots is None, **factory_kwargs)
        label = getattr(conn, 'kind', None) or conn.id_prefix
        pairs = image_entries(conn, images)
        if not pairs:
            raise ValueError('service lists no images · Give image ids')
        errors = []
        if len(pairs) > MAX_CONNECT_IMAGES:
            errors.append(f'{label}: showing the first {MAX_CONNECT_IMAGES} of {len(pairs)} images')
            pairs = pairs[:MAX_CONNECT_IMAGES]
        ids = []
        for iid, name, meta in pairs:
            sid = conn.register(self.registry, iid, name=name)
            self._paths[sid] = f'{label}:{iid}'
            if meta:
                self.slide_meta[sid] = meta
            ids.append(sid)
        try:
            self.registry.info(ids[0])
        except Exception as e:
            for sid in ids:
                self._paths.pop(sid, None)
                self.slide_meta.pop(sid, None)
                self.registry.remove(sid)
            raise ConnectionError(f'{label}: {e}') from e
        self._connections.append(conn)
        return {
            'opened': ids[:1],
            'connected': len(ids),
            'service': label,
            'layers': [],
            'errors': errors,
            'slides': self.slides(),
        }

    # -- the rest
    def close(self, slide: str) -> dict:
        self._paths.pop(slide, None)
        self.slide_meta.pop(slide, None)
        self._auto_meta.pop(slide, None)
        self.registry.remove(slide)
        return {'slides': self.slides()}

    def load_session(self, path: str | os.PathLike, log=print) -> dict:
        """A session file (``wissi-viz --demo`` writes one): slides with their layers, and which to show."""
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        errors, ids = [], {}
        for e in doc.get('slides', []):
            r = self.open([e['slide']], trusted=True)
            errors += r['errors']
            if not r['opened']:
                continue
            sid = r['opened'][0]
            ids[e.get('id', sid)] = sid
            if e.get('view') and len(e['view']) == 4:
                self.views[sid] = [float(v) for v in e['view']]
            for lp in e.get('layers', []):
                try:
                    layer = annio.load(lp)
                    layer.origin = layer.origin or _origin_from_name(lp)
                    self.registry.add_annotations(sid, layer, name=layer.name)
                except Exception as ex:  # noqa: BLE001  (reported; the other layers still load)
                    errors.append(f'{Path(lp).name}: {ex}')
        self.show = [ids[k] for k in doc.get('show', []) if k in ids][:32] or list(ids.values())[:32]
        if doc.get('projects'):  # session projects name slides by their session id: store their paths
            paths = {k: self._paths.get(v) for k, v in ids.items()}
            self.add_projects(
                [{'name': p.get('name'), 'slides': [paths[s] for s in p.get('slides', []) if paths.get(s)]}
                 for p in doc['projects']]
            )  # fmt: skip
        for m in errors:
            log(f'wissi-viz: {m}')
        return {'slides': self.slides(), 'errors': errors}

    def slides(self) -> list[dict]:
        """The open slides with their metadata: file fields, then a tile API's, then metadata tables' (later wins)."""
        ids = [s for s in self._paths if s in self.registry]
        names = {s: self.registry._entries[s].name for s in ids}
        tables = self._table_meta(tuple((s, names[s], self._paths[s]) for s in ids))
        out = []
        for s in ids:
            p = self._paths[s]
            if s not in self._auto_meta:
                self._auto_meta[s] = slidemeta.auto_meta(p, remote=self.registry._entries[s].path is None)
            meta = {**self._auto_meta[s], **self.slide_meta.get(s, {})}
            for t in tables:
                meta.update(t.get(s, {}))
            out.append({'id': s, 'name': names[s], 'path': p, 'meta': meta})
        return out

    def _table_meta(self, slides: tuple) -> list[dict]:
        key = (slides, tuple(id(t) for t in self.meta_tables))
        if self._matched is None or self._matched[0] != key:
            self._matched = (key, [t.match(list(slides)) for t in self.meta_tables])
        return self._matched[1]

    # -- saved views (points of interest) and figures
    def _slide_of(self) -> dict[str, str]:
        """{key: slide id} for every open slide: its id, name, file name, stem and path (lower case)."""
        out: dict[str, str] = {}
        for s in self._paths:
            if s in self.registry:
                for k in slidemeta.slide_keys(s, self.registry._entries[s].name, self._paths[s]):
                    out.setdefault(k, s)
        return out

    def saved_views(self) -> dict:
        """Every saved view, with ``sid``: the open slide it is of (None: that slide is not open)."""
        by = self._slide_of()
        return {'views': [{**v, 'sid': by.get(str(v['slide']).strip().lower())} for v in self.store.views()]}

    def add_view(self, sid: str, bounds, name: str = '', meta: dict | None = None, display: dict | None = None) -> dict:
        """Save a view of an open slide (its rectangle in level-0 pixels); the slide is kept by its path."""
        if sid not in self._paths:
            raise ValueError(f'no open slide {sid!r}')
        v = self.store.add_view(self._paths[sid] or sid, bounds, name, meta, display)
        return {**self.saved_views(), 'view': {**v, 'sid': sid}}

    def update_view(self, vid: int, fields: dict) -> dict:
        self.store.update_view(int(vid), **{k: fields[k] for k in ('name', 'bounds', 'meta', 'display') if k in fields})
        return self.saved_views()

    def delete_views(self, ids) -> dict:
        self.store.delete_views(ids or [])
        return self.saved_views()

    def import_views(self, paths, *, trusted: bool = False) -> dict:
        """Add views from tables (CSV / TSV / Parquet): see :mod:`wissi_viz.store` for the columns."""
        by = self._slide_of()

        def mpp_of(slide: str) -> float | None:
            sid = by.get(str(slide).strip().lower())
            try:
                return (self.registry.get(sid)[1].info.mpp or (None,))[0] if sid else None
            except Exception:  # noqa: BLE001  (no pixel size: the row says so)
                return None

        def slide_of(slide: str) -> str:  # an open slide named by its id, file name or stem: kept by its path
            sid = by.get(str(slide).strip().lower())
            return (self._paths.get(sid) or str(slide)) if sid else str(slide)

        added, errors = 0, []
        for raw in paths or []:
            p = Path(str(raw)).expanduser()
            try:
                if not (trusted or self.allowed(p)):
                    raise PermissionError('outside the folders this server allows')
                r = self.store.import_views(p, mpp_of, slide_of)
                added += r['added']
                errors += [f'{p.name}: {e}' for e in r['errors']]
            except Exception as e:  # noqa: BLE001  (reported per file)
                errors.append(f'{p.name}: {e}')
        return {**self.saved_views(), 'added': added, 'errors': errors[:50]}

    # -- figures: recipes laid out (wissi_viz.figures), exported (figexport, pptxio), PowerPoint templates
    def figure_views(self) -> list[dict]:
        """Every saved view for a figure: its fields (the slide's, then the view's own, plus ``Slide``), and, when
        its slide is open, ``sid``, ``mpp`` and ``fluorescence``."""
        slides = {s['id']: s for s in self.slides()}
        out = []
        for v in self.saved_views()['views']:
            sid = v.get('sid')
            name = Path(str(v['slide'])).name if v.get('slide') else ''
            meta = {**(slides[sid]['meta'] if sid in slides else {}), 'Slide': name, **(v.get('meta') or {})}
            item = {**v, 'meta': meta, 'mpp': None, 'fluorescence': None}
            if sid:
                try:
                    info = self.registry.get(sid)[1].info
                    item['mpp'] = (info.mpp or (None,))[0]
                    item['fluorescence'] = info.mode != 'rgb'
                    if item['fluorescence']:
                        item['channels'] = [{'name': c.name, 'color': c.color} for c in info.channels]
                except Exception:  # noqa: BLE001  (an unreadable slide: no size, no scale bar)
                    pass
            out.append(item)
        return out

    def _objects(self, sid: str, layer: str):
        """Centroids and classes of an open slide's layer (cached per layer version)."""
        try:
            L = self.registry.layer(sid, layer)
        except KeyError:
            return None
        key = (sid, layer)
        hit = self._fig_objects.get(key)
        if hit is not None and hit[0] == L.version:
            return hit[1]
        F = annio.from_layer(L)
        c = F.centroid()
        obj = type('Objects', (), {})()
        obj.x, obj.y, obj.classes = c[:, 0], c[:, 1], F.class_of()
        self._fig_objects[key] = (L.version, obj)
        return obj

    def figure_layout(self, recipe: dict | None, name: str = '', *, open_slides: bool = True) -> dict:
        """A figure's pages and what goes with them: the views it can use (``views``, ``fields``, ``values``), its
        members and what an Update would change, the chart's data and the template. Slides of its views that are
        not open are opened (when this session may open them)."""
        r = figures.normalize(recipe)
        views = self.figure_views()
        sel = figures.select(r, views)
        opened = False
        closed = sorted({str(v['slide']) for v in sel['members'] if not v.get('sid') and v.get('slide')})
        if closed and open_slides and self.can_open:
            paths = [p for p in closed if Path(p).exists() and self.allowed(p)]
            if paths:
                opened = bool(self.open(paths)['opened'])
                views = self.figure_views()
                sel = figures.select(r, views)
        tinfo = None
        if r['template'] == 'pptx' and (r['pptx'] or {}).get('id'):
            try:
                tinfo = self.template_info(r['pptx']['id'])
            except KeyError:
                tinfo = None
        L = figures.layout(r, sel['members'], name=name, template=tinfo)
        series = None
        if r['chart']:
            vals = figures.measure_values(r['chart'], sel['members'], self._objects)
            series = figures.chart_series(r, sel['members'], vals, dark=L['dark'])
        fields = figures.fields_of(views)
        values = {}
        for f in fields:
            seen: dict[str, None] = {}
            for v in views:
                x = figures.field_value(v, f)
                if x is not None and x != '':
                    seen[str(x)] = None
                if len(seen) >= 200:
                    break
            values[f] = sorted(seen, key=figures._sort_key)
        layers: dict[str, set] = {}
        for sid in {v['sid'] for v in sel['members'] if v.get('sid')}:
            for n, lay in self.registry.layers(sid).items():
                layers.setdefault(n, set()).update(lay.classes)
        return {**L, 'members': sel['ids'], 'pending': sel['pending'], 'missing': sel['missing'],
                'closed': [str(v['slide']) for v in sel['members'] if not v.get('sid')],
                'views': [{'id': v['id'], 'name': v['name'], 'sid': v.get('sid'), 'slide': v['slide'],
                           'bounds': v['bounds'], 'meta': v['meta'], 'fluorescence': v['fluorescence'],
                           'channels': v.get('channels')} for v in views],
                'fields': fields, 'values': values, 'series': series, 'template': tinfo,
                'layers': [{'name': n, 'classes': sorted(c)} for n, c in sorted(layers.items())],
                'opened': opened, 'slides': self.slides() if opened else None}  # fmt: skip

    def figure_export(
        self, recipe: dict | None, name: str = 'Figure', fmt: str = 'png', dpi: int = 300, *,
        snapshot: bool = False, data: bool = False,
    ) -> tuple[bytes, str, str]:  # fmt: skip
        """The figure as a file: (bytes, file name, media type). PNG / TIFF / SVG pages, one PDF, or one .pptx; a
        zip when there are several pages, or with the recipe snapshot / the chart's data beside it."""
        try:
            from . import figexport
        except ImportError as e:
            raise ImportError('figure export needs the figures extra: pip install "wissi-viz[figures]"') from e
        if fmt not in figexport.FORMATS:
            raise ValueError(f'format: {", ".join(figexport.FORMATS)}')
        dpi = max(72, min(int(dpi or 300), 1200))
        r = figures.normalize(recipe)
        views = self.figure_views()
        sel = figures.select(r, views)
        tinfo = None
        if r['template'] == 'pptx' and (r['pptx'] or {}).get('id'):
            tinfo = self.template_info(r['pptx']['id'])
        L = figures.layout(r, sel['members'], name=name, template=tinfo)
        series = None
        if r['chart']:
            vals = figures.measure_values(r['chart'], sel['members'], self._objects)
            series = figures.chart_series(r, sel['members'], vals, dark=L['dark'])

        def picture(w: dict, pw: int, ph: int):
            sid = w.get('sid')
            if not sid:
                return np.full((ph, pw, 3), 0 if w.get('fluorescence') else 255, np.uint8)
            e, _ = self.registry.get(sid)
            return region_rgb(e.source, w['bounds'], (pw, ph), e.windows, w.get('channels'), bool(w.get('gray')))

        stem = re.sub(r'[\\/:*?"<>|]+', '_', name or 'Figure').strip() or 'Figure'
        if fmt == 'pptx':
            from . import pptxio

            def jpeg(w: dict, pw: int, ph: int) -> bytes:
                buf = io.BytesIO()
                Image.fromarray(picture(w, pw, ph)).save(buf, 'JPEG', quality=90)
                return buf.getvalue()

            ink = '#ffffff' if L['dark'] else '#111111'
            chart = (lambda w, pw, ph: figexport.chart_png(series, pw, ph, ink=ink, font=L['font'],
                                                           fonts=(tinfo or {}).get('fonts'))) if series else None  # fmt: skip
            tpath = self._template_path(r['pptx']['id']) if tinfo else None
            files = [pptxio.write_pptx(L, jpeg, chart, template=tpath, info=tinfo,
                                       layout_name=(r['pptx'] or {}).get('layout'), dpi=min(dpi, 300))]  # fmt: skip
        else:
            files = figexport.render(L, picture, fmt, dpi, series, (tinfo or {}).get('fonts'))
        snap = figexport.snapshot(name, r, sel['members']) if snapshot else None
        return figexport.bundle(stem, fmt, files, snap=snap, series=series if data else None)

    def _templates_dir(self) -> Path:
        d = (user_dir('config') if self.remember else self.cache_dir / 'session-templates') / 'templates'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def templates(self) -> list[dict]:
        """The PowerPoint templates added: [{'id', 'name'}], newest first."""
        try:
            items = json.loads((self._templates_dir() / 'templates.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return []
        return [t for t in items if isinstance(t, dict) and (self._templates_dir() / f'{t.get("id")}.pptx').exists()]

    def _template_path(self, tid: str) -> Path:
        if not re.fullmatch(r'[0-9a-f]{8,40}', str(tid)):
            raise KeyError(f'no template {tid}')
        p = self._templates_dir() / f'{tid}.pptx'
        if not p.exists():
            raise KeyError(f'no template {tid}')
        return p

    def template_info(self, tid: str) -> dict:
        """A template's description (:func:`wissi_viz.pptxio.read_template`), with its id."""
        if tid in self._template_info:
            return self._template_info[tid]
        from . import pptxio

        info = pptxio.read_template(self._template_path(tid))
        name = next((t['name'] for t in self.templates() if t['id'] == tid), info['name'])
        info = {**info, 'id': tid, 'name': name}
        self._template_info[tid] = info
        return info

    def add_template_bytes(self, name: str, data: bytes) -> dict:
        """Add a PowerPoint template (.pptx, or .potx: kept as a .pptx) from its bytes."""
        try:
            from . import pptxio  # noqa: F401  (python-pptx present)
        except ImportError as e:
            raise ImportError('PowerPoint templates need the figures extra: pip install "wissi-viz[figures]"') from e
        name = Path(str(name or 'template.pptx')).name
        if not name.lower().endswith(('.pptx', '.potx')):
            raise ValueError('a PowerPoint template: .pptx or .potx')
        if len(data) > 200 * 2**20:
            raise ValueError('template larger than 200 MB')
        data = _potx_as_pptx(data) if name.lower().endswith('.potx') else data
        tid = hashlib.sha1(data).hexdigest()[:16]
        d = self._templates_dir()
        (d / f'{tid}.pptx').write_bytes(data)
        try:
            info = self.template_info(tid)
        except Exception as e:
            (d / f'{tid}.pptx').unlink(missing_ok=True)
            self._template_info.pop(tid, None)
            raise ValueError(f'{name}: not a PowerPoint file this can read ({e})') from e
        items = [t for t in self.templates() if t['id'] != tid]
        items.insert(0, {'id': tid, 'name': name})
        (d / 'templates.json').write_text(json.dumps(items[:50], ensure_ascii=False), encoding='utf-8')
        self._template_info.pop(tid, None)
        info = self.template_info(tid)
        return {'template': info, 'templates': self.templates()}

    def add_templates(self, paths, *, trusted: bool = False) -> dict:
        """Add PowerPoint templates by path (desktop file dialog, or a server's allowed folders)."""
        out: dict = {'templates': self.templates(), 'errors': []}
        for raw in paths or []:
            p = Path(str(raw)).expanduser()
            try:
                if not (trusted or self.allowed(p)):
                    raise PermissionError('outside the folders this server allows')
                out = {**self.add_template_bytes(p.name, p.read_bytes()), 'errors': out['errors']}
            except Exception as e:  # noqa: BLE001  (reported per file)
                out['errors'].append(f'{p.name}: {e}')
        return out

    # -- metadata tables
    def add_metadata(
        self, paths: Iterable[str | os.PathLike], key: str | None = None, *, trusted: bool = False
    ) -> dict:
        """Add metadata tables (CSV / TSV / Parquet): the column naming most open slides is the key (by id, name,
        file name or path), the other columns become fields of the slides it names."""
        errors = []
        for raw in paths or []:
            p = Path(str(raw)).expanduser()
            try:
                if not (trusted or self.allowed(p)):
                    raise PermissionError('outside the folders this server allows')
                if p.suffix.lower() not in slidemeta.TABLE_EXTS:
                    raise ValueError('metadata table: .csv, .tsv or .parquet')
                t = slidemeta.MetaTable(p, key)
                self.meta_tables = [x for x in self.meta_tables if Path(x.path) != p] + [t]
            except Exception as e:  # noqa: BLE001  (reported per file)
                errors.append(f'{p.name}: {e}')
        self._save_meta()
        return {**self.metadata_state(), 'errors': errors}

    def remove_metadata(self, path: str) -> dict:
        self.meta_tables = [t for t in self.meta_tables if t.path != path]
        self._save_meta()
        return self.metadata_state()

    def metadata_state(self) -> dict:
        slides = self.slides()
        tables = self._matched[1] if self._matched else []
        return {'slides': slides, 'meta_tables': [t.describe(len(m)) for t, m in zip(self.meta_tables, tables)]}

    def _save_meta(self) -> None:
        self._matched = None
        if self.remember:
            with self._lock:
                try:
                    data = [{'path': t.path, 'key': t.key} for t in self.meta_tables]
                    self._meta_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
                except OSError:
                    pass

    def state(self) -> dict:
        return {
            **self.metadata_state(),
            'recent': self.recent(),
            'version': __version__,
            'show': self.show,
            'views': self.views,
            'projects': self.projects,
        }

    # -- projects (the slide list's own groups)
    def _load_projects(self) -> list[dict]:
        if not self.remember:
            return []
        try:
            return _clean_projects(json.loads(self._projects_file.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            return []

    def set_projects(self, projects: Any) -> dict:
        """Replace the projects: [{'name': str, 'slides': [path, …]}] (names unique, order kept)."""
        self.projects = _clean_projects(projects)
        if self.remember:
            with self._lock:
                try:
                    self._projects_file.write_text(json.dumps(self.projects, ensure_ascii=False), encoding='utf-8')
                except OSError:
                    pass
        return {'projects': self.projects}

    def add_projects(self, projects: Any) -> None:
        """Merge projects (a session file's): new names are appended, existing ones gain the slides."""
        cur = {p['name']: p for p in self.projects}
        for p in _clean_projects(projects):
            if p['name'] in cur:
                cur[p['name']]['slides'] += [s for s in p['slides'] if s not in cur[p['name']]['slides']]
            else:
                self.projects.append(p)
                cur[p['name']] = p
        self.set_projects(self.projects)

    def recent(self) -> list[str]:
        if not self.remember:
            return []
        try:
            items = json.loads(self._recent_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return []
        return [p for p in items if isinstance(p, str) and Path(p).exists()][:12]

    def _remember(self, paths: list[str]) -> None:
        if not paths or not self.remember:
            return
        with self._lock:
            items = list(paths) + [p for p in self.recent() if p not in set(paths)]
            try:
                self._recent_file.write_text(json.dumps(items[:12], ensure_ascii=False), encoding='utf-8')
            except OSError:
                pass

    def shutdown(self) -> None:
        try:
            self.store.close()
        except Exception:  # noqa: BLE001  (best-effort cleanup at exit)
            pass
        try:
            self.registry.close()
        except Exception:  # noqa: BLE001  (best-effort cleanup at exit)
            pass

    def app(self, **kw):
        """The desktop app's server: see :func:`create_session_app`."""
        return create_session_app(self, **kw)


def add_connect_args(ap) -> None:
    """``--connect`` / ``--images`` / ``--connector-arg``, shared by ``serve`` and ``desktop``."""
    ap.add_argument(
        '--connect',
        action='append',
        default=[],
        metavar='SERVICE',
        help='show the images of a tile API: a service .json (see docs/INTEGRATION.md) or module:factory (repeatable)',
    )
    ap.add_argument('--images', default=None, help='image ids for --connect, comma-separated (default: all it lists)')
    ap.add_argument(
        '--connector-arg', action='append', default=[], metavar='KEY=VALUE', help='passed to a module:factory'
    )


def apply_connects(session: Session, args, log=print) -> list[str]:
    """Run the ``--connect`` options; returns the errors (also logged)."""
    kw = dict(kv.split('=', 1) for kv in args.connector_arg if '=' in kv)
    errors = []
    for spec in args.connect:
        try:
            r = session.connect(spec, args.images, trusted=True, **kw)
            log(f'wissi-viz: {r["service"]}: {r["connected"]} image(s)')
            errors += r['errors']
        except Exception as e:  # noqa: BLE001  (one service failing must not stop the others)
            errors.append(f'{spec}: {e}')
            log(f'wissi-viz: {spec}: {e}')
    return errors


def create_session_app(
    session: Session,
    *,
    mode: str = 'desktop',
    prefix: str = PREFIX,
    cors: str | None = None,
    key: str | None = None,
    local_only: bool | None = None,
    dependencies: list | None = None,
):
    """The viewer server plus the app page (``/app``, and ``/`` in server mode) and the endpoints it uses.

    Every ``/app/*`` call carries the page's token (so other web pages cannot drive it). ``key``: a password the
    user types once before opening files or connecting (``wissi-viz serve --key``). ``local_only``: accept only
    127.0.0.1 / localhost as the Host (default: in desktop mode). ``dependencies``: e.g. your login check, on every
    route. Works mounted under a path (``parent.mount('/viewer', app)``; see :func:`wissi_viz.nicegui.mount_app`)."""
    app = create_app(session.registry, prefix=prefix, cors=cors, index=False, dependencies=dependencies)
    if local_only if local_only is not None else mode == 'desktop':
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
    caps = {
        'open': session.can_open,
        'browse': session.can_open,
        'connect': session.can_connect,
        'close': mode == 'desktop',
        'key': bool(key),
    }

    def check(request: Request, needs_key: bool = False) -> None:
        if not secrets.compare_digest(request.headers.get('x-wissi-token', ''), session.token):
            raise HTTPException(403, 'bad token')
        if needs_key and key and not secrets.compare_digest(request.headers.get('x-wissi-key', ''), key):
            raise HTTPException(401, 'key required')

    def page(request: Request) -> str:
        root = request.scope.get('root_path', '')
        return app_page(root + prefix, mode=mode, token=session.token, caps=caps, base=root)

    app.add_api_route('/app', page, response_class=HTMLResponse, methods=['GET'])
    if mode == 'server':
        app.add_api_route('/', page, response_class=HTMLResponse, methods=['GET'])

    @app.get('/app/state')
    def app_state(request: Request):
        check(request)
        return session.state()

    async def run(fn):
        try:
            return JSONResponse(await anyio.to_thread.run_sync(fn))
        except PermissionError as e:
            raise HTTPException(403, str(e)) from e
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        except KeyError as e:  # no such view / figure
            raise HTTPException(404, str(e).strip("'")) from e
        except (ConnectionError, ImportError) as e:
            raise HTTPException(502, str(e)) from e

    @app.post('/app/open')
    async def app_open(request: Request):
        check(request, needs_key=True)
        if not session.can_open:
            raise HTTPException(403, 'this server does not open files')
        body = await request.json()
        return await run(lambda: session.open(body.get('paths') or [], body.get('slide')))

    @app.get('/app/browse')
    async def app_browse(request: Request, folder: str | None = Query(None, alias='dir')):
        check(request, needs_key=True)
        return await run(lambda: session.browse(folder))

    @app.post('/app/connect')
    async def app_connect(request: Request):
        check(request, needs_key=True)
        body = await request.json()
        return await run(lambda: session.connect(body.get('spec'), body.get('images')))

    @app.post('/app/projects')
    async def app_projects(request: Request):
        """Replace the slide list's projects: {projects: [{name, slides: [path, …]}]}."""
        check(request, needs_key=mode == 'server')
        body = await request.json()
        return await run(lambda: session.set_projects(body.get('projects')))

    @app.post('/app/metadata')
    async def app_metadata(request: Request):
        """Add metadata tables {paths: [...], key?} or remove one {remove: path}."""
        check(request, needs_key=True)
        body = await request.json()
        if body.get('remove'):
            return await run(lambda: session.remove_metadata(body['remove']))
        if not session.can_open:
            raise HTTPException(403, 'this server does not open files')
        return await run(lambda: session.add_metadata(body.get('paths') or [], body.get('key')))

    # saved views (points of interest) and figure recipes: the app's own database (wissi_viz.store)
    @app.get('/app/views')
    async def app_views(request: Request):
        check(request)
        return await run(session.saved_views)

    @app.post('/app/views')
    async def app_views_add(request: Request):
        """Save a view {slide (open slide id), bounds [x0, y0, x1, y1], name?, meta?, display?}."""
        check(request, needs_key=mode == 'server')
        b = await request.json()
        return await run(lambda: session.add_view(b.get('slide'), b.get('bounds') or [], b.get('name', ''),
                                                  b.get('meta'), b.get('display')))  # fmt: skip

    @app.post('/app/views/{vid}')
    async def app_views_update(vid: int, request: Request):
        """Change a view: {name?, bounds?, meta? (merged; null removes a field), display?}."""
        check(request, needs_key=mode == 'server')
        b = await request.json()
        return await run(lambda: session.update_view(vid, b))

    @app.post('/app/views-delete')
    async def app_views_delete(request: Request):
        """Delete views {ids: [...]}."""
        check(request, needs_key=mode == 'server')
        ids = (await request.json()).get('ids') or []
        return await run(lambda: session.delete_views(ids))

    @app.post('/app/views-import')
    async def app_views_import(request: Request):
        """Add views from tables {paths: [...]}: slide, x0, y0, x1, y1 (or x, y, width / width_um), name, metadata."""
        check(request, needs_key=True)
        if not session.can_open:
            raise HTTPException(403, 'this server does not open files')
        b = await request.json()
        return await run(lambda: session.import_views(b.get('paths') or []))

    @app.get('/app/views-export')
    async def app_views_export(request: Request, fmt: str = 'csv'):
        check(request)
        pq = fmt == 'parquet'
        data = await anyio.to_thread.run_sync(lambda: session.store.export_views('parquet' if pq else 'csv'))
        name = f'wissi-viz views.{"parquet" if pq else "csv"}'
        return Response(data, media_type='application/vnd.apache.parquet' if pq else 'text/csv',
                        headers={'Content-Disposition': f'attachment; filename="{name}"'})  # fmt: skip

    @app.get('/app/figures')
    async def app_figures(request: Request):
        check(request)
        return await run(lambda: {'figures': session.store.figures()})

    @app.post('/app/figures')
    async def app_figures_save(request: Request):
        """Save a figure recipe {name, recipe, id? (replace)} or delete one {delete: id}."""
        check(request, needs_key=mode == 'server')
        b = await request.json()

        def go():
            if b.get('delete') is not None:
                session.store.delete_figure(b['delete'])
                return {'figures': session.store.figures()}
            f = session.store.save_figure(b.get('name') or 'Figure', b.get('recipe') or {}, b.get('id'))
            return {'figure': f, 'figures': session.store.figures()}

        return await run(go)

    @app.post('/app/figures/layout')
    async def app_figures_layout(request: Request):
        """A figure's pages {recipe, name}: windows in mm, the views it can use, what an Update would change."""
        check(request)
        b = await request.json()
        return await run(lambda: session.figure_layout(b.get('recipe'), str(b.get('name') or '')))

    @app.post('/app/figures/export')
    async def app_figures_export(request: Request):
        """The figure as a file {recipe, name, fmt: png | tiff | pdf | svg | pptx, dpi, snapshot, data}."""
        check(request)
        b = await request.json()

        def go():
            return session.figure_export(b.get('recipe'), str(b.get('name') or 'Figure'), str(b.get('fmt') or 'png'),
                                         int(b.get('dpi') or 300), snapshot=bool(b.get('snapshot')),
                                         data=bool(b.get('data')))  # fmt: skip

        try:
            data, fname, mt = await anyio.to_thread.run_sync(go)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e).strip("'")) from e
        except ImportError as e:
            raise HTTPException(501, str(e)) from e
        return Response(data, media_type=mt, headers={'Content-Disposition': f'attachment; filename="{fname}"',
                                                      'X-Filename': quote(fname)})  # fmt: skip

    @app.get('/app/figures/templates')
    async def app_figures_templates(request: Request):
        check(request)
        return await run(lambda: {'templates': session.templates()})

    @app.post('/app/figures/templates')
    async def app_figures_template_add(request: Request, name: str = 'template.pptx'):
        """Add a PowerPoint template: the file's bytes as the body, ``?name=`` its file name."""
        check(request, needs_key=mode == 'server')
        data = await request.body()
        try:
            return await run(lambda: session.add_template_bytes(name, data))
        except HTTPException as e:
            if e.status_code == 502:  # python-pptx missing
                raise HTTPException(501, e.detail) from e
            raise

    @app.post('/app/close')
    async def app_close(request: Request):
        check(request)
        if not caps['close']:
            raise HTTPException(403, 'slides on a server stay open for everyone')
        return session.close((await request.json()).get('slide'))

    return app


#: sample-pack layer files written before layers recorded their origin: <id>.<what>.parquet
_PACK_ORIGINS = {
    'cells': 'detected',
    'cells-10x': 'detected',
    'tissue': 'detected',
    'density': 'analysis',
    'hotspots': 'analysis',
}


def _clean_projects(projects: Any) -> list[dict]:
    out, seen = [], set()
    for p in projects if isinstance(projects, list) else []:
        if not isinstance(p, dict):
            continue
        name = str(p.get('name') or '').strip()[:120]
        if not name or name in seen:
            continue
        seen.add(name)
        slides = [str(s) for s in p.get('slides') or [] if isinstance(s, str)]
        out.append({'name': name, 'slides': list(dict.fromkeys(slides))[:5000]})
    return out[:200]


def _origin_from_name(path: str | os.PathLike) -> str:
    parts = Path(path).name.split('.')
    return _PACK_ORIGINS.get(parts[-2] if len(parts) >= 3 else '', 'imported')


def _is_geoparquet(p: Path) -> bool:
    """A .parquet file with GeoParquet metadata (annotations), as opposed to a plain cell table."""
    try:
        import pyarrow.parquet as pq

        return b'geo' in (pq.read_schema(p).metadata or {})
    except Exception:  # noqa: BLE001  (no pyarrow, or not parquet: treat as a table)
        return False


def _looks_like_table(p: Path) -> bool:
    """A .json export that is a column table (not GeoJSON)."""
    try:
        with open(p, 'rb') as f:
            head = f.read(4096).lstrip()
    except OSError:
        return False
    return head.startswith(b'[') or (b'"type"' not in head and b'"features"' not in head)


def _potx_as_pptx(data: bytes) -> bytes:
    """A PowerPoint template (.potx) as a presentation (.pptx): the same parts, the main part's type changed."""
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for item in src.infolist():
            b = src.read(item.filename)
            if item.filename == '[Content_Types].xml':
                b = b.replace(b'presentationml.template.main+xml', b'presentationml.presentation.main+xml')
            z.writestr(item, b)
    return out.getvalue()
