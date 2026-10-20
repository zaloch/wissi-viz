"""CZI metadata helpers shared by both CZI backends.

Accepts the metadata dict from czifile (attributes as plain keys) and from pylibCZIrw
(xmltodict style, attributes prefixed with ``@``).
"""
from __future__ import annotations

from typing import Any

from .base import DEFAULT_CHANNEL_COLORS


def get(d: Any, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def as_list(v):
    return v if isinstance(v, list) else ([] if v is None else [v])


def attr(d: Any, k: str, default=None):
    if not isinstance(d, dict):
        return default
    v = d.get('@' + k)
    return d.get(k, default) if v is None else v


def argb_to_hex(s: Any) -> str | None:
    if not isinstance(s, str) or not s.startswith('#'):
        return None
    s = s[1:]
    if len(s) == 8:
        s = s[2:]
    return '#' + s.lower() if len(s) == 6 else None


def channel_meta(meta: dict, n: int) -> tuple[list[str], list[str]]:
    md = get(meta, 'ImageDocument', 'Metadata') or {}
    disp = as_list(get(md, 'DisplaySetting', 'Channels', 'Channel'))
    info = as_list(get(md, 'Information', 'Image', 'Dimensions', 'Channels', 'Channel'))
    names, colors = [], []
    for c in range(n):
        d = disp[c] if c < len(disp) else {}
        i = info[c] if c < len(info) else {}
        names.append(attr(d, 'Name') or attr(i, 'Name') or f'Channel {c + 1}')
        colors.append(argb_to_hex(attr(d, 'Color') or attr(i, 'Color'))
                      or DEFAULT_CHANNEL_COLORS[c % len(DEFAULT_CHANNEL_COLORS)])
    return names, colors


def mpp(meta: dict) -> tuple[float, float] | None:
    items = as_list(get(meta, 'ImageDocument', 'Metadata', 'Scaling', 'Items', 'Distance'))
    vals = {}
    for it in items:
        try:
            vals[attr(it, 'Id')] = float(attr(it, 'Value')) * 1e6
        except (TypeError, ValueError):
            pass
    if vals.get('X'):
        return (vals['X'], vals.get('Y') or vals['X'])
    return None
