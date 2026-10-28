"""App-side connector for the (mock) vendor tile API — the file you'd write for PathCore / HALO Link.

Everything vendor-specific lives here: URLs, auth header, how the service describes its pyramid,
and how it signals "no tile" / "slow down". wissi-viz handles caching, retries, concurrency,
overlap/padding, rendering and annotations.

    conn = DemoApiConnection('http://127.0.0.1:9000', token='secret')
    sids = [conn.register(registry, i['id'], name=i['name']) for i in conn.list_images()]
"""

from __future__ import annotations

import io

import httpx
import numpy as np

from wissi_viz.connectors import AnnotationPayload, RemoteConnection, RemoteImageInfo, RetryAfter, TileNotFound


class DemoApiConnection(RemoteConnection):
    id_prefix = 'api'
    max_concurrency = 10  # stay just under the service's limit (it answers 429 above 12)
    retries = 4
    timeout = 30.0

    def __init__(self, base_url: str, token: str, client: httpx.Client | None = None):
        super().__init__()
        # one pooled client for all slides of this connection (keep-alive, HTTP/1.1)
        self.http = client or httpx.Client(
            base_url=base_url.rstrip('/'),
            timeout=self.timeout,
            headers={'Authorization': f'Bearer {token}'},
            limits=httpx.Limits(
                max_connections=self.max_concurrency + 2, max_keepalive_connections=self.max_concurrency
            ),
        )

    # ------------------------------------------------------------------ plumbing
    def _get(self, path: str, **params) -> httpx.Response:
        r = self.http.get(path, params=params or None)
        if r.status_code == 404:
            raise TileNotFound(path)
        if r.status_code in (429, 503) and 'retry-after' in r.headers:
            raise RetryAfter(float(r.headers['retry-after']))
        r.raise_for_status()  # 5xx etc. → retried with back-off by wissi-viz
        return r

    def list_images(self) -> list[dict]:
        return self._get('/api/images').json()

    # ------------------------------------------------------------------ the contract
    def image_info(self, image_id: str) -> RemoteImageInfo:
        m = self._get(f'/api/images/{image_id}').json()
        common = {
            'width': m['width'],
            'height': m['height'],
            'mpp': m['mpp'],
            'name': m['name'],
            'version': m['modified'],
        }
        if m['format'] == 'dzi':
            return RemoteImageInfo(**common, deepzoom=True, tile_size=m['tile_size'], overlap=m['overlap'])
        info = RemoteImageInfo(
            **common,
            tile_size=m['tile_size'],
            levels=[(lv['width'], lv['height']) for lv in m['levels']],  # any order
            level_ids=[lv['id'] for lv in m['levels']],
        )  # passed back to tile()
        if m.get('channels'):
            info.mode, info.dtype = 'channels', m.get('dtype', 'uint16')
            info.channels = [c['name'] for c in m['channels']]
        return info

    def tile(self, image_id: str, level, col: int, row: int, channel=None):
        params = {'channel': channel} if channel is not None else {}
        r = self._get(f'/api/images/{image_id}/tiles/{level}/{col}/{row}', **params)
        if r.headers.get('content-type', '').startswith('image/png') and channel is not None:
            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(r.content)))  # 16-bit channel tile → array
        return r.content  # JPEG → passed to the browser untouched

    def annotations(self, image_id: str):
        gj = self._get(f'/api/images/{image_id}/annotations').json()
        if not gj.get('features'):
            return []
        return [
            AnnotationPayload('ROIs (API)', gj, units='um', style={'line_width': 2.5, 'fill_opacity': 0.0, 'z': 10})
        ]

    def objects(self, image_id: str):
        try:
            d = self._get(f'/api/images/{image_id}/objects').json()
        except TileNotFound:
            return []
        return [
            AnnotationPayload(
                'Cells (API)',
                d['columns'],
                format='table',
                units=d.get('units', 'px'),
                style={'point_size': 4},
                visible=False,
            )
        ]  # large: off until the user turns it on
