"""Build a connection from what a user can type: a JSON service description or a Python factory.

    open_connection('halolink.json')                 # HttpTileConnection (no code; see connectors/http.py)
    open_connection({'base_url': ..., 'tile': ...})  # the same, inline
    open_connection('mypkg.halo:make_connection', url='https://halo.example.org')   # your RemoteConnection

Factories run your code, so only the command line may name one (``allow_code=True``); the app's Connect dialog
accepts JSON only.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

from .base import RemoteConnection
from .http import HttpTileConnection


def open_connection(spec: Any, *, allow_code: bool = False, expand_env: bool = True, **kwargs: Any) -> RemoteConnection:
    """``spec``: a RemoteConnection, a config dict, JSON text, a path to a ``.json`` file, or ``module:factory``
    (a RemoteConnection subclass or a function returning one; ``kwargs`` are passed to it). ``expand_env``: allow
    ``${VAR}`` in inline descriptions (files on disk always may use them)."""
    if isinstance(spec, RemoteConnection):
        return spec
    if isinstance(spec, dict):
        return HttpTileConnection(spec, expand_env=expand_env)
    text = str(spec).strip()
    if text.startswith('{'):
        return HttpTileConnection(json.loads(text), expand_env=expand_env)
    path = Path(os.path.expanduser(text))
    if path.suffix.lower() == '.json' or path.exists():
        if not path.is_file():
            raise FileNotFoundError(f'no such service file: {path}')
        return HttpTileConnection(path)
    if ':' in text:
        if not allow_code:
            raise PermissionError('Python connector factories can only be named on the command line')
        module, _, attr = text.rpartition(':')
        conn = getattr(importlib.import_module(module), attr)(**kwargs)
        if not isinstance(conn, RemoteConnection):
            raise TypeError(f'{text} returned {type(conn).__name__}, not a RemoteConnection')
        return conn
    raise ValueError(f'not a service description: {text!r} (a .json file, JSON text, or module:factory)')


def image_list(conn: RemoteConnection, ids: Any = None) -> list[tuple[str, str]]:
    """``(id, name)`` pairs: the ids given (a list or a comma / newline separated string), else what the
    connection lists."""
    if isinstance(ids, str):
        ids = [s.strip() for s in ids.replace('\n', ',').split(',')]
    if ids:
        return [(str(i), str(i)) for i in ids if str(i).strip()]
    out = []
    for item in conn.images() or []:
        iid, name = (item, item) if isinstance(item, (str, int)) else item
        out.append((str(iid), str(name)))
    return out
