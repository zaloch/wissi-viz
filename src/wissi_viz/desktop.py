"""wissi-viz as a desktop app: the same viewer in its own native window, with its own icon.

    uv tool install "wissi-viz[desktop]"      # or: pip install "wissi-viz[desktop]"
    wissi-viz-app                             # empty window → File ▸ Open slides…  (Ctrl/⌘+O)
    wissi-viz desktop a.svs b.czi cells.csv   # open slides (tables / annotation files attach to the first one)
    wissi-viz desktop --install-shortcut      # Start menu + desktop (Windows), ~/Applications (macOS), app menu (Linux)

How it works: the tile server runs in this process on 127.0.0.1 (random port, Host-checked, a per-launch token on the
app's own endpoints), and a pywebview window shows ``/app``: WebView2 on Windows, WKWebView on macOS, WebKitGTK or
Qt WebEngine on Linux. Native file dialogs, a File menu and drag-and-drop of files from the OS come through the
pywebview bridge. Without a GUI backend (e.g. Linux with no GTK/Qt bindings) the page opens as a Chromium/Edge
``--app`` window, or in the default browser, with a path box instead of dialogs.

The embedded element (``<wissi-viz>`` / NiceGUI) is unchanged; this is just another host for it.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from starlette.requests import Request  # module level: FastAPI resolves the string annotations here

from . import __version__
from ._compat import user_dir, utf8_stdio

ASSETS = Path(__file__).resolve().parent / 'assets'
APP_NAME = 'wissi-viz'
BUNDLE_ID = 'org.wissiviz.desktop'
ANNOTATION_EXTS = ('.geojson', '.json', '.xml', '.annotations', '.geoparquet', '.coco.json')
TABLE_EXTS = ('.csv', '.tsv', '.txt', '.parquet', '.pq', '.feather', '.arrow', '.ipc', '.h5ad')   # .parquet: GeoParquet if it has geometry
PREFIX = '/wissi-viz'


def icon_path(kind: str = 'png') -> Path:
    """The app icon: 'png' (512 px), 'ico' (Windows, 16–256 px), 'icns' (macOS), 'svg'."""
    return ASSETS / f'icon.{kind}'


def _slide_exts() -> tuple[str, ...]:
    from .registry import SLIDE_EXTS
    return SLIDE_EXTS


def _kind(p: Path) -> str:
    n = p.name.lower()
    if p.is_dir():
        return 'folder'
    if n.endswith(_slide_exts()):
        return 'slide'
    if n.endswith(ANNOTATION_EXTS):
        return 'annotations'
    if n.endswith(TABLE_EXTS):
        return 'table'
    return 'slide'          # let open_slide decide (it raises a clear error for unknown formats)


# ------------------------------------------------------------------------------------------------ session
class Session:
    """The slides and layers open in one app window. Shared by the native bridge and the HTTP fallback."""

    def __init__(self, cache_dir: str | os.PathLike | None = None, memory_mb: int = 512, cache_gb: float = 10,
                 decoded_cache_mb: int = 512) -> None:
        from .registry import SlideRegistry
        from .tiles import TileService
        self.cache_dir = Path(cache_dir) if cache_dir else user_dir('cache')
        self.tiles = TileService(memory_bytes=memory_mb * 2**20, disk_dir=self.cache_dir / 'tiles',
                                 disk_bytes=int(cache_gb * 2**30))
        self.registry = SlideRegistry(self.tiles, decoded_cache_mb=decoded_cache_mb)
        self.token = secrets.token_urlsafe(24)
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._recent_file = user_dir('config') / 'recent.json'
        self.show: list[str] = []          # slide ids to show in panes on load (a session)

    # -- opening
    def open(self, paths, slide: str | None = None) -> dict:
        """Open slide files / folders; attach annotation files and cell tables to ``slide`` (else the first slide
        opened in this call, else the most recent one). Returns {'opened', 'layers', 'errors', 'slides'}."""
        opened, layers, errors, extras = [], [], [], []
        for raw in paths or []:
            p = Path(str(raw)).expanduser()
            try:
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
                    self.registry.info(sid)          # open now: unreadable files fail here, not as a blank pane
                    self._paths[sid] = p if k != 'folder' else self.registry._entries[sid].path
                    opened.append(sid)
            except Exception as e:  # noqa: BLE001
                errors.append(f'{p.name}: {e}')
                for sid in [s for s in self.registry.ids() if s not in self._paths]:
                    self.registry.remove(sid)
        target = slide if slide in self._paths else (opened[0] if opened else (list(self._paths)[-1] if self._paths else None))
        for p in extras:
            if target is None:
                errors.append(f'{p.name}: open a slide first, then add cells / annotations to it')
                continue
            try:
                layers.append({'slide': target, 'layer': self.add_layer(target, p)})
            except Exception as e:  # noqa: BLE001
                errors.append(f'{p.name}: {e}')
        self._remember([self._paths[s] for s in opened] + extras)
        return {'opened': opened, 'layers': layers, 'errors': errors, 'slides': self.slides()}

    def add_layer(self, slide: str, path: str | os.PathLike) -> str:
        """Attach an annotation file (GeoJSON / HALO XML / …) or a cell table (CSV / Parquet / AnnData) to a slide.
        Tables are compiled once into the cache and memory-mapped on later opens."""
        p = Path(path)
        from . import annio
        try:
            fmt = annio.detect_format(p)
        except ValueError:
            fmt = None
        if fmt == 'coco' or (fmt == 'geoparquet' and _is_geoparquet(p)):
            layer = annio.load(p)
        elif _kind(p) == 'table' or (p.suffix.lower() == '.json' and _looks_like_table(p)):
            from .objects import cached_object_layer
            mpp = (self.registry.info(slide).get('mpp') or [None])[0]
            layer = cached_object_layer(p, self.cache_dir / 'objects', mpp=mpp, name=p.stem)
        else:
            from .annotations import AnnotationLayer
            layer = AnnotationLayer.load(p)
        name = layer.name or p.stem
        if name in self.registry.layers(slide):
            name = f'{name} ({p.name})'
        self.registry.add_annotations(slide, layer, name=name)
        return name

    def close(self, slide: str) -> dict:
        self._paths.pop(slide, None)
        self.registry.remove(slide)
        return {'slides': self.slides()}

    def load_session(self, path: str | os.PathLike, log=print) -> dict:
        """A session file (``wissi-viz --demo`` writes one): slides with their layers, and which to show."""
        from . import annio
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        errors, ids = [], {}
        for e in doc.get('slides', []):
            r = self.open([e['slide']])
            errors += r['errors']
            if not r['opened']:
                continue
            sid = r['opened'][0]
            ids[e.get('id', sid)] = sid
            for lp in e.get('layers', []):
                try:
                    layer = annio.load(lp)
                    self.registry.add_annotations(sid, layer, name=layer.name)
                except Exception as ex:  # noqa: BLE001
                    errors.append(f'{Path(lp).name}: {ex}')
        self.show = [ids[k] for k in doc.get('show', []) if k in ids][:32] or list(ids.values())[:32]
        for m in errors:
            log(f'wissi-viz: {m}')
        return {'slides': self.slides(), 'errors': errors}

    def slides(self) -> list[dict]:
        return [{'id': s, 'name': self.registry._entries[s].name, 'path': str(p)}
                for s, p in self._paths.items() if s in self.registry]

    # -- recent files
    def recent(self) -> list[str]:
        try:
            items = json.loads(self._recent_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return []
        return [p for p in items if isinstance(p, str) and Path(p).exists()][:12]

    def _remember(self, paths) -> None:
        if not paths:
            return
        with self._lock:
            items = [str(p) for p in paths] + [p for p in self.recent() if p not in {str(q) for q in paths}]
            try:
                self._recent_file.write_text(json.dumps(items[:12], ensure_ascii=False), encoding='utf-8')
            except OSError:
                pass

    def shutdown(self) -> None:
        try:
            self.registry.close()
        except Exception:  # noqa: BLE001
            pass

    # -- HTTP
    def app(self):
        """The viewer server plus the app page (``/app``) and its token-protected endpoints."""
        from fastapi import HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
        from starlette.middleware.trustedhost import TrustedHostMiddleware

        from .server import app_page, create_app
        app = create_app(self.registry, prefix=PREFIX, cors=None)
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

        def check(request: Request):
            if not secrets.compare_digest(request.headers.get('x-wissi-token', ''), self.token):
                raise HTTPException(403, 'bad token')

        @app.get('/app', response_class=HTMLResponse)
        def desktop_page():
            return app_page(PREFIX, mode='desktop', token=self.token)

        @app.get('/app/state')
        def app_state(request: Request):
            check(request)
            return {'slides': self.slides(), 'recent': self.recent(), 'version': __version__, 'show': self.show}

        @app.post('/app/open')
        async def app_open(request: Request):
            check(request)
            body = await request.json()
            import anyio
            res = await anyio.to_thread.run_sync(lambda: self.open(body.get('paths') or [], body.get('slide')))
            return JSONResponse(res)

        @app.post('/app/close')
        async def app_close(request: Request):
            check(request)
            return self.close((await request.json()).get('slide'))

        return app


def _is_geoparquet(p: Path) -> bool:
    """A .parquet file with GeoParquet metadata (annotations), as opposed to a plain cell table."""
    try:
        import pyarrow.parquet as pq
        return b'geo' in (pq.read_schema(p).metadata or {})
    except Exception:  # noqa: BLE001
        return False


def _looks_like_table(p: Path) -> bool:
    """A .json export that is a column table (not GeoJSON)."""
    try:
        with open(p, 'rb') as f:
            head = f.read(4096).lstrip()
    except OSError:
        return False
    return head.startswith(b'[') or (b'"type"' not in head and b'"features"' not in head)


# ------------------------------------------------------------------------------------------------ server
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class _Server:
    """uvicorn in a daemon thread on 127.0.0.1."""

    def __init__(self, app, port: int | None = None) -> None:
        import uvicorn
        self.port = port or _free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=self.port, log_level='warning',
                                                    lifespan='off'))
        self.server.install_signal_handlers = lambda: None   # not the main thread
        self.thread = threading.Thread(target=self.server.run, name='wissi-viz-server', daemon=True)

    def start(self, timeout: float = 15) -> str:
        self.thread.start()
        t0 = time.monotonic()
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() - t0 > timeout:
                raise RuntimeError('wissi-viz: the local server did not start')
            time.sleep(0.02)
        return f'http://127.0.0.1:{self.port}'

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


# ------------------------------------------------------------------------------------------------ native window
SLIDE_FILTER = 'Whole-slide images ({})'
LAYER_FILTER = 'Cells and annotations ({})'


def _patterns(exts) -> str:
    return ';'.join(f'*{e}' for e in exts)


class Bridge:
    """Methods the page calls as ``window.pywebview.api.<name>()`` (each runs on a pywebview worker thread)."""

    def __init__(self, session: Session) -> None:
        self._s = session
        self._window = None          # set by run_native; underscore → not exposed to JavaScript

    def _dialog(self, kind: str, multiple: bool = True, types=()):
        import webview
        fd = getattr(webview, 'FileDialog', None)
        t = {'open': fd.OPEN if fd else webview.OPEN_DIALOG, 'folder': fd.FOLDER if fd else webview.FOLDER_DIALOG}[kind]
        try:
            return list(self._window.create_file_dialog(t, allow_multiple=multiple, file_types=types) or [])
        except Exception:  # noqa: BLE001  (a backend that rejects the filter syntax: no filter)
            return list(self._window.create_file_dialog(t, allow_multiple=multiple) or [])

    def pick_slides(self):
        types = (SLIDE_FILTER.format(_patterns(_slide_exts())),
                 LAYER_FILTER.format(_patterns(ANNOTATION_EXTS + TABLE_EXTS)), 'All files (*.*)')
        return self._s.open(self._dialog('open', True, types))

    def pick_folder(self):
        return self._s.open(self._dialog('folder', False))

    def pick_layer(self, slide: str | None = None):
        types = (LAYER_FILTER.format(_patterns(ANNOTATION_EXTS + TABLE_EXTS)), 'All files (*.*)')
        return self._s.open(self._dialog('open', True, types), slide=slide)

    def open_paths(self, paths, slide: str | None = None):
        return self._s.open(paths, slide=slide)

    def close_slide(self, slide: str):
        return self._s.close(slide)

    def state(self):
        return {'slides': self._s.slides(), 'recent': self._s.recent(), 'version': __version__, 'show': self._s.show}

    def install_shortcut(self):
        try:
            return {'installed': [str(p) for p in install_shortcut()]}
        except Exception as e:  # noqa: BLE001
            return {'errors': [str(e)]}


def _push(window, result: dict) -> None:
    """Hand a result produced on the Python side (menu, drop) to the page."""
    try:
        window.evaluate_js(f'window.wissiApp && window.wissiApp.onResult({json.dumps(result)})')
    except Exception:  # noqa: BLE001
        pass


def _active_slide(window) -> str | None:
    try:
        return window.evaluate_js('window.wissiApp ? window.wissiApp.activeSlide() : null')
    except Exception:  # noqa: BLE001
        return None


def _menu(bridge: Bridge, window):
    from webview.menu import Menu, MenuAction, MenuSeparator

    def act(fn):
        return lambda: _push(window, fn())
    return [Menu('File', [
        MenuAction('Open slides…', act(bridge.pick_slides)),
        MenuAction('Open folder…', act(bridge.pick_folder)),
        MenuAction('Add cells / annotations…', act(lambda: bridge.pick_layer(_active_slide(window)))),
        MenuSeparator(),
        MenuAction('Install app shortcut', act(bridge.install_shortcut)),
    ])]


def _enable_drop(window, session: Session) -> None:
    """Files dropped from Explorer / Finder / a file manager → full paths (pywebview ≥ 5 DOM events)."""
    try:
        from webview.dom import DOMEventHandler
    except ImportError:
        return

    def on_drop(e):
        files = (e.get('dataTransfer') or {}).get('files') or []
        paths = [f.get('pywebviewFullPath') for f in files if f.get('pywebviewFullPath')]
        if paths:
            _push(window, session.open(paths, slide=_active_slide(window)))
    try:
        window.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True)
        window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)
    except Exception:  # noqa: BLE001
        pass


def _native_icon(window) -> None:
    """Window / Dock icon where pywebview's ``start(icon=)`` doesn't reach (it covers GTK and Qt)."""
    try:
        if sys.platform == 'win32':
            from System import Action  # pythonnet, installed with pywebview on Windows
            from System.Drawing import Icon
            form = getattr(window, 'native', None)
            if form is not None:
                ico = Icon(str(icon_path('ico')))
                form.Invoke(Action(lambda: setattr(form, 'Icon', ico)))
        elif sys.platform == 'darwin':
            from AppKit import NSApplication, NSImage
            from PyObjCTools import AppHelper
            img = NSImage.alloc().initWithContentsOfFile_(str(icon_path('icns')))
            AppHelper.callAfter(lambda: NSApplication.sharedApplication().setApplicationIconImage_(img))
    except Exception:  # noqa: BLE001  (cosmetic)
        pass


def _windows_app_id() -> None:
    """Group the window under its own taskbar button and icon instead of python.exe's."""
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(BUNDLE_ID)
        except Exception:  # noqa: BLE001
            pass


def run_native(url: str, session: Session, debug: bool = False, width: int = 1440, height: int = 920) -> None:
    import webview
    bridge = Bridge(session)
    _windows_app_id()
    window = webview.create_window(APP_NAME, f'{url}/app?native=1', js_api=bridge, width=width, height=height,
                                   min_size=(720, 480), background_color='#0b0d10', text_select=True)
    bridge._window = window
    window.events.shown += lambda: _native_icon(window)
    window.events.loaded += lambda: _enable_drop(window, session)
    kw = {'debug': debug, 'icon': str(icon_path('png')), 'private_mode': False,
          'storage_path': str(session.cache_dir / 'webview')}
    try:
        webview.start(menu=_menu(bridge, window), **kw)
    except TypeError:          # pywebview < 5: no app menu / icon arguments
        webview.start(debug=debug)


# ------------------------------------------------------------------------------------------------ browser fallback
def _chromium() -> str | None:
    """A Chromium-family browser for a chromeless ``--app`` window."""
    names = ['msedge', 'chrome', 'chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable', 'brave']
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    cands = []
    if sys.platform == 'win32':
        for base in filter(None, (os.environ.get('PROGRAMFILES(X86)'), os.environ.get('PROGRAMFILES'),
                                  os.environ.get('LOCALAPPDATA'))):
            cands += [Path(base) / 'Microsoft/Edge/Application/msedge.exe', Path(base) / 'Google/Chrome/Application/chrome.exe']
    elif sys.platform == 'darwin':
        cands += [Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
                  Path('/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'),
                  Path('/Applications/Chromium.app/Contents/MacOS/Chromium')]
    return next((str(c) for c in cands if c.exists()), None)


def run_browser(url: str, server: _Server) -> None:
    page = f'{url}/app'
    exe = _chromium()
    if exe:
        prof = user_dir('cache') / 'app-window'
        proc = subprocess.Popen([exe, f'--app={page}', f'--user-data-dir={prof}', '--no-first-run',
                                 '--no-default-browser-check', '--window-size=1440,920'])
        print(f'wissi-viz: {page}  (close the window to quit)')
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return
    import webbrowser
    webbrowser.open(page)
    print(f'wissi-viz: {page}  (Ctrl+C to quit)')
    try:
        while server.thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def _has_native() -> tuple[bool, str]:
    try:
        import webview  # noqa: F401
    except ImportError:
        return False, 'pywebview is not installed (install "wissi-viz[desktop]")'
    if sys.platform.startswith('linux'):
        import importlib.util
        if not (importlib.util.find_spec('gi') or importlib.util.find_spec('qtpy')):
            return False, ('no GUI toolkit for pywebview: install GTK bindings (e.g. apt install python3-gi '
                           'gir1.2-webkit2-4.1, and create the venv with --system-site-packages) or Qt '
                           '(uv pip install "pywebview[qt]")')
    return True, ''


# ------------------------------------------------------------------------------------------------ shortcuts
def _launch_command() -> list[str]:
    """How a shortcut starts the app: the ``wissi-viz-app`` GUI launcher if installed, else ``python -m``."""
    import sysconfig
    exe = 'wissi-viz-app.exe' if sys.platform == 'win32' else 'wissi-viz-app'
    for d in (Path(sys.executable).parent, Path(sysconfig.get_path('scripts'))):
        if (d / exe).exists():
            return [str(d / exe)]
    found = shutil.which('wissi-viz-app')
    if found:
        return [found]
    py = Path(sys.executable)
    if sys.platform == 'win32' and (py.parent / 'pythonw.exe').exists():
        py = py.parent / 'pythonw.exe'                    # no console window
    return [str(py), '-m', 'wissi_viz.desktop']


def _stable_icon(kind: str) -> Path:
    """A copy of the icon outside site-packages, so shortcuts keep it across upgrades."""
    d = user_dir('config')
    dst = d / f'{APP_NAME}.{kind}'
    try:
        shutil.copyfile(icon_path(kind), dst)
    except OSError:
        return icon_path(kind)
    return dst


def shortcut_paths() -> list[Path]:
    """Where install_shortcut writes, per OS."""
    home = Path.home()
    if sys.platform == 'win32':
        start = Path(os.environ.get('APPDATA') or home / 'AppData/Roaming') / 'Microsoft/Windows/Start Menu/Programs'
        return [start / f'{APP_NAME}.lnk', _windows_desktop() / f'{APP_NAME}.lnk']
    if sys.platform == 'darwin':
        return [home / 'Applications' / f'{APP_NAME}.app']
    data = Path(os.environ.get('XDG_DATA_HOME') or home / '.local/share')
    return [data / 'applications' / f'{APP_NAME}.desktop']


def _windows_desktop() -> Path:
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', "[Environment]::GetFolderPath('Desktop')"],
                             capture_output=True, text=True, timeout=20, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())     # follows OneDrive folder redirection
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.home() / 'Desktop'


def install_shortcut() -> list[Path]:
    """Add a launcher with the app icon: Start menu + desktop .lnk (Windows), ~/Applications/wissi-viz.app (macOS),
    a .desktop entry in the app menu (Linux). Returns the paths written."""
    cmd = _launch_command()
    if sys.platform == 'win32':
        return _install_windows(cmd)
    if sys.platform == 'darwin':
        return _install_macos(cmd)
    return _install_linux(cmd)


def uninstall_shortcut() -> list[Path]:
    removed = []
    for p in shortcut_paths():
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed.append(p)
    return removed


def _install_windows(cmd: list[str]) -> list[Path]:
    ico = _stable_icon('ico')
    out = []
    script = ('$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:WV_LNK); $s.TargetPath = $env:WV_TARGET; '
              '$s.Arguments = $env:WV_ARGS; $s.IconLocation = $env:WV_ICON + ",0"; $s.WorkingDirectory = $env:USERPROFILE; '
              '$s.Description = "wissi-viz whole-slide image viewer"; $s.Save()')
    args = subprocess.list2cmdline(cmd[1:])
    for lnk in shortcut_paths():
        lnk.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'WV_LNK': str(lnk), 'WV_TARGET': cmd[0], 'WV_ARGS': args, 'WV_ICON': str(ico)}
        r = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script], env=env,
                           capture_output=True, text=True, timeout=60, check=False)
        if r.returncode != 0:
            raise RuntimeError(f'could not create {lnk}: {r.stderr.strip()}')
        out.append(lnk)
    return out


def _install_macos(cmd: list[str]) -> list[Path]:
    import plistlib
    app = shortcut_paths()[0]
    if app.exists():
        shutil.rmtree(app)
    (app / 'Contents/MacOS').mkdir(parents=True)
    (app / 'Contents/Resources').mkdir(parents=True)
    shutil.copyfile(icon_path('icns'), app / 'Contents/Resources' / f'{APP_NAME}.icns')
    exe = app / 'Contents/MacOS' / APP_NAME
    quoted = ' '.join("'" + c.replace("'", "'\\''") + "'" for c in cmd)
    exe.write_text(f'#!/bin/sh\nexec {quoted} "$@"\n', encoding='utf-8')
    exe.chmod(0o755)
    info = {'CFBundleName': APP_NAME, 'CFBundleDisplayName': APP_NAME, 'CFBundleIdentifier': BUNDLE_ID,
            'CFBundleExecutable': APP_NAME, 'CFBundleIconFile': APP_NAME, 'CFBundlePackageType': 'APPL',
            'CFBundleShortVersionString': __version__, 'CFBundleVersion': __version__,
            'NSHighResolutionCapable': True, 'LSMinimumSystemVersion': '11.0'}
    with open(app / 'Contents/Info.plist', 'wb') as f:
        plistlib.dump(info, f)
    os.utime(app)                       # nudge Finder / LaunchServices to pick up the icon
    return [app]


def _install_linux(cmd: list[str]) -> list[Path]:
    data = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share')
    icon_dir = data / 'icons/hicolor/512x512/apps'
    icon_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(icon_path('png'), icon_dir / f'{APP_NAME}.png')
    scal = data / 'icons/hicolor/scalable/apps'
    scal.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(icon_path('svg'), scal / f'{APP_NAME}.svg')

    def q(c):
        return '"' + c.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$') + '"' if any(
            ch in c for ch in ' "\'\\`$') else c
    entry = '\n'.join([
        '[Desktop Entry]', 'Type=Application', f'Name={APP_NAME}', 'GenericName=Whole-slide image viewer',
        'Comment=GPU viewer for SVS / CZI / OME-TIFF histology slides', f'Exec={" ".join(q(c) for c in cmd)} %F',
        f'Icon={APP_NAME}', 'Terminal=false', 'Categories=Graphics;Science;MedicalSoftware;Viewer;',
        'MimeType=image/tiff;', f'StartupWMClass={APP_NAME}', ''])
    p = shortcut_paths()[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(entry, encoding='utf-8')
    p.chmod(0o755)
    if shutil.which('update-desktop-database'):
        subprocess.run(['update-desktop-database', str(p.parent)], capture_output=True, check=False)
    return [p]


# ------------------------------------------------------------------------------------------------ entry points
def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(prog='wissi-viz desktop', description='wissi-viz in its own window')
    ap.add_argument('files', nargs='*', help='slides, folders, cell tables or annotation files to open')
    ap.add_argument('--session', default=None, help='a session file (slides + layers), e.g. from wissi-viz --demo')
    ap.add_argument('--browser', action='store_true', help='use a browser window instead of the native one')
    ap.add_argument('--port', type=int, default=None, help='local server port (default: a free one)')
    ap.add_argument('--cache-dir', default=None, help=f'tile / layer cache (default: {user_dir("cache")})')
    ap.add_argument('--memory-mb', type=int, default=512, help='in-memory tile cache (MB)')
    ap.add_argument('--cache-gb', type=float, default=10, help='disk tile cache (GB)')
    ap.add_argument('--install-shortcut', action='store_true', help='add a launcher with the app icon, then exit')
    ap.add_argument('--uninstall-shortcut', action='store_true', help='remove the launcher, then exit')
    ap.add_argument('--debug', action='store_true', help='enable the web inspector')
    a = ap.parse_args(argv)
    if a.install_shortcut:
        for p in install_shortcut():
            print(f'installed {p}')
        return
    if a.uninstall_shortcut:
        for p in uninstall_shortcut():
            print(f'removed {p}')
        return

    session = Session(a.cache_dir, memory_mb=a.memory_mb, cache_gb=a.cache_gb)
    if a.session:
        session.load_session(a.session)
    res = session.open(a.files) if a.files else None
    for e in (res or {}).get('errors', []):
        print(f'wissi-viz: {e}', file=sys.stderr)
    server = _Server(session.app(), a.port)
    url = server.start()
    try:
        native, why = (False, 'requested --browser') if a.browser else _has_native()
        if native:
            try:
                run_native(url, session, debug=a.debug)
                return
            except Exception as e:  # noqa: BLE001  (e.g. WebView2 runtime / GTK missing at start)
                why = f'native window failed: {e}'
        if not a.browser:
            print(f'wissi-viz: {why}; using a browser window', file=sys.stderr)
        run_browser(url, server)
    finally:
        server.stop()
        session.shutdown()


def gui_main() -> None:
    """``wissi-viz-app``: the GUI launcher (no console window on Windows)."""
    main(sys.argv[1:])


if __name__ == '__main__':
    main()
