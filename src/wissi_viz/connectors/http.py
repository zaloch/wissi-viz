"""A configurable connector for tile APIs that expose plain HTTP tile URLs — no Python to write.

Describe the service in a small JSON (or dict): where image metadata lives and which fields hold the
size / pyramid / mpp / version, the tile URL template, auth headers (``${ENV_VAR}`` is expanded), and
which HTTP statuses mean "empty tile" or "slow down"::

    {
      "kind": "halolink",                                   # label only: halolink | pathcore | custom
      "base_url": "https://tiles.example.org",
      "headers": {"Authorization": "Bearer ${TILE_API_TOKEN}"},
      "max_concurrency": 8, "timeout": 30, "retries": 3,
      "info": {"path": "/api/images/{image}", "width": "width", "height": "height", "tile_size": "tile_size",
               "mpp": "mpp", "version": "modified", "name": "name",
               "levels": "levels", "level_width": "width", "level_height": "height", "level_id": "id",
               "deepzoom_if": {"field": "format", "equals": "dzi"}, "overlap": "overlap"},
      "tile": {"path": "/api/images/{image}/tiles/{level}/{col}/{row}", "empty_status": [404, 204],
               "retry_status": [429, 503]},
      "annotations": {"path": "/api/images/{image}/annotations", "format": "geojson", "units": "um"}
    }

``info`` alternatives: ``"downsamples": "downsamples"`` (a field with [1, 4, 16, …]); ``"dzi_path":
"/images/{image}.dzi"`` (a DeepZoom descriptor); or a fixed ``"pyramid": {"width": …, "height": …,
"tile_size": 256, "downsamples": [1, 2, 4, …]}`` when there is no metadata endpoint. Field names may be
dotted paths (``"meta.size.width"``).

Services whose tiles are not plain URLs (GraphQL-only, signed requests, SDK clients) should subclass
:class:`~wissi_viz.connectors.RemoteConnection` with the app's existing client instead.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..sources.remote import RemoteImageInfo, RemotePyramid, RetryAfter, TileNotFound
from .base import AnnotationPayload, RemoteConnection

_ENV = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def _expand(v: Any) -> Any:
    if isinstance(v, str):

        def sub(m):
            if m.group(1) not in os.environ:
                raise KeyError(f'environment variable {m.group(1)} is not set (used in the connector config)')
            return os.environ[m.group(1)]

        return _ENV.sub(sub, v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def _flat(d: Any, pre: str = ''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flat(v, f'{pre}{k}.')
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flat(v, f'{pre}{i}.')
    else:
        yield pre.rstrip('.'), d


def _get(d: Any, path: str | None, default: Any = None) -> Any:
    if not path:
        return default
    for part in str(path).split('.'):
        if isinstance(d, dict) and part in d:
            d = d[part]
        elif isinstance(d, list) and part.isdigit() and int(part) < len(d):
            d = d[int(part)]
        else:
            return default
    return d


class HttpTileConnection(RemoteConnection):
    """Tile API described by a config dict / JSON file (see module docstring). Needs ``httpx`` ([remote] extra)."""

    id_prefix = 'http'

    def __init__(self, config: dict | str | os.PathLike):
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text(encoding='utf-8-sig'))
        raw = {k: v for k, v in config.items() if not k.startswith('_')}
        todo = [k for k, v in _flat(raw) if isinstance(v, str) and 'TODO' in v]
        if todo:
            raise ValueError('connector config still has TODO fields: ' + ', '.join(todo))
        self.config = cfg = _expand(raw)
        self.kind = cfg.get('kind', 'custom')
        self.id_prefix = cfg.get('id_prefix', self.kind if self.kind in ('halolink', 'pathcore') else 'http')
        self.max_concurrency = int(cfg.get('max_concurrency', self.max_concurrency))
        self.retries = int(cfg.get('retries', self.retries))
        self.timeout = float(cfg.get('timeout', self.timeout or 30))
        super().__init__()
        import httpx

        self.http = httpx.Client(
            base_url=cfg['base_url'].rstrip('/'),
            headers=cfg.get('headers') or {},
            timeout=self.timeout,
            verify=cfg.get('verify_tls', True),
            follow_redirects=True,
            http2=bool(cfg.get('http2', False)),
            limits=httpx.Limits(
                max_connections=self.max_concurrency + 2, max_keepalive_connections=self.max_concurrency
            ),
        )
        t = cfg.get('tile') or {}
        self._empty = set(t.get('empty_status', [404, 204]))
        self._retry = set(t.get('retry_status', [429, 503]))

    # ------------------------------------------------------------------ plumbing
    def _fmt(self, path: str, image_id: str, **kw) -> str:
        return path.format(image=image_id, **kw)

    def _req(self, path: str, params: dict | None = None):
        r = self.http.get(path, params=params or None)
        if r.status_code in self._empty:
            raise TileNotFound(path)
        if r.status_code in self._retry:
            raise RetryAfter(float(r.headers.get('retry-after', 1.0)), f'HTTP {r.status_code}')
        r.raise_for_status()
        return r

    # ------------------------------------------------------------------ the contract
    def image_info(self, image_id: str) -> RemoteImageInfo:
        i = self.config.get('info') or {}
        fixed = i.get('pyramid') or self.config.get('pyramid')
        if fixed:
            return RemoteImageInfo(
                width=int(fixed['width']),
                height=int(fixed['height']),
                tile_size=int(fixed.get('tile_size', 256)),
                downsamples=fixed.get('downsamples'),
                mpp=fixed.get('mpp'),
                name=str(image_id),
                version=str(fixed.get('version', '0')),
                overlap=int(fixed.get('overlap', 0)),
            )
        if i.get('dzi_path'):
            pyr = RemotePyramid.from_dzi(self._req(self._fmt(i['dzi_path'], image_id)).content)
            return RemoteImageInfo(
                width=pyr.width,
                height=pyr.height,
                pyramid=pyr,
                tile_size=pyr.tile_size,
                name=str(image_id),
                mpp=i.get('mpp'),
                version=str(i.get('version', '0')),
            )
        m = self._req(self._fmt(i['path'], image_id), i.get('query')).json()
        common = {
            'width': int(_get(m, i.get('width', 'width'))),
            'height': int(_get(m, i.get('height', 'height'))),
            'mpp': _get(m, i.get('mpp')),
            'name': str(_get(m, i.get('name'), image_id)),
            'version': str(_get(m, i.get('version'), '0')),
        }
        ts = int(_get(m, i.get('tile_size', 'tile_size'), 256))
        dz = i.get('deepzoom_if')
        if i.get('deepzoom') or (dz and _get(m, dz['field']) == dz['equals']):
            return RemoteImageInfo(**common, deepzoom=True, tile_size=ts, overlap=int(_get(m, i.get('overlap'), 1)))
        if i.get('levels') and _get(m, i['levels']):
            lv = _get(m, i['levels'])
            return RemoteImageInfo(
                **common,
                tile_size=ts,
                overlap=int(_get(m, i.get('overlap'), 0)),
                levels=[
                    (int(_get(x, i.get('level_width', 'width'))), int(_get(x, i.get('level_height', 'height'))))
                    for x in lv
                ],
                level_ids=[_get(x, i['level_id']) for x in lv] if i.get('level_id') else None,
            )
        if i.get('downsamples'):
            return RemoteImageInfo(**common, tile_size=ts, downsamples=[float(d) for d in _get(m, i['downsamples'])])
        return RemoteImageInfo(**common, tile_size=ts)

    def tile(self, image_id: str, level, col: int, row: int, channel=None):
        t = self.config['tile']
        q = {k: str(v).format(channel=channel) for k, v in (t.get('query') or {}).items()}
        return self._req(self._fmt(t['path'], image_id, level=level, col=col, row=row, channel=channel), q).content

    def annotations(self, image_id: str):
        a = self.config.get('annotations')
        if not a:
            return []
        try:
            r = self._req(self._fmt(a['path'], image_id))
        except TileNotFound:
            return []
        data = r.json() if a.get('format', 'geojson') in ('geojson', 'json') else r.text
        if isinstance(data, dict) and data.get('type') == 'FeatureCollection' and not data.get('features'):
            return []
        return [
            AnnotationPayload(
                a.get('name', 'Annotations (API)'), data, format=a.get('format', 'geojson'), units=a.get('units', 'px')
            )
        ]
