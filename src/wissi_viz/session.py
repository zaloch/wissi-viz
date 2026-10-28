"""What a viewer window has open, and how things get opened: slide files and folders, cell tables and
annotation files attached to a slide, and images from tile APIs. Shared by the desktop app and
``wissi-viz serve``; the HTTP endpoints the app page talks to are built by :func:`create_session_app`.

Who may open what:

* desktop (``roots=None``): anything the local user can read, plus any tile API;
* server: only files inside the folders given with ``--allow-open`` (none by default), and tile APIs only
  with ``--allow-connect``. Paths named on the command line are trusted.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import anyio
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request  # module level: FastAPI resolves the string annotations here

from . import annio
from ._compat import user_dir
from ._version import __version__
from .annotations import AnnotationLayer
from .connectors import RemoteConnection, image_list, open_connection
from .objects import cached_object_layer
from .registry import SLIDE_EXTS, SlideRegistry
from .server import app_page, create_app
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
        self.show: list[str] = []  # slide ids to show in panes on load (a session)

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
                errors.append(f'{p.name}: open a slide first, then add cells / annotations to it')
                continue
            try:
                layers.append({'slide': target, 'layer': self.add_layer(target, p)})
            except Exception as e:  # noqa: BLE001  (reported to the user per file)
                errors.append(f'{p.name}: {e}')
        self._remember([self._paths[s] for s in opened] + [str(p) for p in extras])
        return {'opened': opened, 'layers': layers, 'errors': errors, 'slides': self.slides()}

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
                entries.append({'name': c.name, 'path': str(c), 'kind': k})
            if len(entries) >= 2000:
                break
        if self.roots is None:
            parent = str(base.parent) if base.parent != base else None
        elif any(base == r for r in self.roots):
            parent = '' if len(self.roots) > 1 else None
        else:
            parent = str(base.parent)
        return {'dir': str(base), 'parent': parent, 'entries': entries}

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
        pairs = image_list(conn, images)
        if not pairs:
            raise ValueError('the service lists no images: give image ids')
        errors = []
        if len(pairs) > MAX_CONNECT_IMAGES:
            errors.append(f'{label}: showing the first {MAX_CONNECT_IMAGES} of {len(pairs)} images')
            pairs = pairs[:MAX_CONNECT_IMAGES]
        ids = []
        for iid, name in pairs:
            sid = conn.register(self.registry, iid, name=name)
            self._paths[sid] = f'{label}:{iid}'
            ids.append(sid)
        try:
            self.registry.info(ids[0])
        except Exception as e:
            for sid in ids:
                self._paths.pop(sid, None)
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
            for lp in e.get('layers', []):
                try:
                    layer = annio.load(lp)
                    self.registry.add_annotations(sid, layer, name=layer.name)
                except Exception as ex:  # noqa: BLE001  (reported; the other layers still load)
                    errors.append(f'{Path(lp).name}: {ex}')
        self.show = [ids[k] for k in doc.get('show', []) if k in ids][:32] or list(ids.values())[:32]
        for m in errors:
            log(f'wissi-viz: {m}')
        return {'slides': self.slides(), 'errors': errors}

    def slides(self) -> list[dict]:
        return [
            {'id': s, 'name': self.registry._entries[s].name, 'path': p}
            for s, p in self._paths.items()
            if s in self.registry
        ]

    def state(self) -> dict:
        return {'slides': self.slides(), 'recent': self.recent(), 'version': __version__, 'show': self.show}

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
):
    """The viewer server plus the app page (``/app``, and ``/`` in server mode) and the endpoints it uses.

    Every ``/app/*`` call carries the page's token (so other web pages cannot drive it). ``key``: a password the
    user types once before opening files or connecting (``wissi-viz serve --key``). ``local_only``: accept only
    127.0.0.1 / localhost as the Host (default: in desktop mode)."""
    app = create_app(session.registry, prefix=prefix, cors=cors, index=False)
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

    def page() -> str:
        return app_page(prefix, mode=mode, token=session.token, caps=caps)

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

    @app.post('/app/close')
    async def app_close(request: Request):
        check(request)
        if not caps['close']:
            raise HTTPException(403, 'slides on a server stay open for everyone')
        return session.close((await request.json()).get('slide'))

    return app


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
