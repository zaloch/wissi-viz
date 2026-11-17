"""Where the running app is, for ``wissi-viz mcp`` (the MCP bridge for Claude Desktop / Claude Code): the app writes
its URL and token to the config folder (readable by this user only) and removes them when it closes."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._compat import user_dir


def running_file() -> Path:
    return user_dir('config') / 'agent.json'


def announce(url: str, token: str) -> Path:
    """Write where the running app is (its URL and token, readable by this user only) for ``wissi-viz mcp``."""
    p = running_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'url': url, 'token': token, 'pid': os.getpid()}), encoding='utf-8')
    try:
        p.chmod(0o600)
    except OSError:
        pass
    return p


def withdraw() -> None:
    try:
        d = json.loads(running_file().read_text(encoding='utf-8'))
        if d.get('pid') == os.getpid():
            running_file().unlink()
    except (OSError, ValueError):
        pass


def app_caller(url: str, token: str) -> Callable[[str, dict], Any]:
    def call(name: str, args: dict) -> Any:
        req = urllib.request.Request(f'{url}/app/agent/call', data=json.dumps({'tool': name, 'args': args}).encode(),
                                     headers={'Content-Type': 'application/json', 'X-Wissi-Token': token})  # fmt: skip
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode('utf-8'))

    return call


def find_app() -> tuple[str, str] | None:
    """The running app's (url, token), if it answers."""
    try:
        d = json.loads(running_file().read_text(encoding='utf-8'))
        req = urllib.request.Request(f'{d["url"]}/app/state', headers={'X-Wissi-Token': d['token']})
        with urllib.request.urlopen(req, timeout=3):
            return d['url'], d['token']
    except Exception:  # noqa: BLE001  (not running, or not this user's)
        return None
