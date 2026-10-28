"""Small cross-platform helpers (Linux, macOS, Windows)."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np


def utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 (with replacement) so reports with →, ≤, µ, ✅ print on any console or pipe.
    (On Windows, redirected output otherwise uses the ANSI code page and raises UnicodeEncodeError.)"""
    for s in (sys.stdout, sys.stderr):
        try:
            if s is not None and (getattr(s, 'encoding', '') or '').lower().replace('-', '') != 'utf8':
                s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:  # noqa: BLE001  (not a TextIOWrapper, e.g. captured by a test runner)
            pass


WINDOWS_RESERVED = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}


def safe_name(text: str, max_len: int = 48) -> str:
    """A file/dir name that is valid on Windows, macOS and Linux: ASCII word chars, short (Windows MAX_PATH),
    never a reserved device name (CON, NUL, …), never ending in a dot or space, plus a hash so it stays unique."""
    base = re.sub(r'[^A-Za-z0-9_.\-]+', '_', text)[:max_len].rstrip('. ') or 'x'
    if base.split('.')[0].lower() in WINDOWS_RESERVED:
        base = '_' + base
    return f'{base}-{hashlib.blake2b(text.encode(), digest_size=4).hexdigest()}'


def user_dir(kind: str, app: str = 'wissi-viz'):
    """Per-user folder for ``kind`` = 'cache' | 'config', following each OS's convention:

    Windows  %LOCALAPPDATA%\\wissi-viz\\Cache  ·  %APPDATA%\\wissi-viz
    macOS    ~/Library/Caches/wissi-viz       ·  ~/Library/Application Support/wissi-viz
    Linux    $XDG_CACHE_HOME/wissi-viz        ·  $XDG_CONFIG_HOME/wissi-viz   (~/.cache, ~/.config)
    """
    home = Path.home()
    if sys.platform == 'win32':
        if kind == 'cache':
            base = Path(os.environ.get('LOCALAPPDATA') or home / 'AppData' / 'Local') / app / 'Cache'
        else:
            base = Path(os.environ.get('APPDATA') or home / 'AppData' / 'Roaming') / app
    elif sys.platform == 'darwin':
        base = home / 'Library' / ('Caches' if kind == 'cache' else 'Application Support') / app
    else:
        env, default = ('XDG_CACHE_HOME', '.cache') if kind == 'cache' else ('XDG_CONFIG_HOME', '.config')
        base = Path(os.environ.get(env) or home / default) / app
    base.mkdir(parents=True, exist_ok=True)
    return base


def otsu(values, default=None):
    """Otsu threshold that never raises: nearly constant data (where numpy cannot make 256 finite histogram bins —
    seen on some platforms with float32 data) returns ``default``."""
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        return default
    lo, hi = float(v.min()), float(v.max())
    if not hi - lo > 1e-9 * max(1.0, abs(hi), abs(lo)):
        return default
    try:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(v))
    except ValueError:
        return default
