"""PathCore connector — **stub**: your app supplies the actual connection.

Same contract as :mod:`wissi_viz.connectors.halolink`. Async clients work as-is::

    from wissi_viz.connectors import PathCoreConnection, RemoteImageInfo, AnnotationPayload

    class WissiPathCore(PathCoreConnection):
        def __init__(self, client):
            super().__init__()
            self.client = client                       # ← e.g. an authenticated httpx.AsyncClient wrapper

        async def image_info(self, slide_id):
            m = await self.client.slide(slide_id)      # ← your call
            return RemoteImageInfo(width=m['width'], height=m['height'], tile_size=m['tile_size'],
                                   downsamples=m['downsamples'],            # e.g. [1, 4, 16, 32]
                                   mpp=m['mpp'], name=m['label'], version=m['etag'])

        async def tile(self, slide_id, level, col, row, channel=None):
            return await self.client.tile(slide_id, level, col, row)      # ← bytes / ndarray / None

        async def annotations(self, slide_id):
            gj = await self.client.annotations_geojson(slide_id)          # ← GeoJSON FeatureCollection
            return [AnnotationPayload('PathCore annotations', gj, units='um')]  # µm → px uses the slide mpp

    pc = WissiPathCore(client).bind_loop(nicegui_loop)   # only if the client is bound to that loop
    sid = pc.register(registry, 'S-998')                 # → 'pathcore-S-998'
"""
from __future__ import annotations

from .base import RemoteConnection


class PathCoreConnection(RemoteConnection):
    """Stub for PathCore. Implement :meth:`image_info` and :meth:`tile` (see module docstring)."""

    id_prefix = 'pathcore'
    max_concurrency = 8
