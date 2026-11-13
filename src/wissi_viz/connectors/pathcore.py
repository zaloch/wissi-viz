"""PathCore connector — **stub**: your app supplies the actual connection.

Same contract as :mod:`wissi_viz.connectors.halolink`. Async clients work as-is::

    from wissi_viz.connectors import PathCoreConnection, RemoteImageInfo, AnnotationPayload
    from wissi_viz.sources.base import Channel

    class WissiPathCore(PathCoreConnection):
        def __init__(self, client):
            super().__init__()
            self.client = client                       # ← e.g. an authenticated httpx.AsyncClient wrapper

        async def images(self):                        # each scene of a multi-scene file is an image of its own
            out = []
            for slide_id in await self.client.slide_ids():
                scenes = (await self.client.slide(slide_id))['metadata'].get('all_images') or [None]
                out += [(f'{slide_id}#{i}', f'{slide_id} scene {i + 1}') if len(scenes) > 1 else (slide_id, slide_id)
                        for i in range(len(scenes))]
            return out

        async def image_info(self, image_id):
            slide_id, _, scene = image_id.partition('#')
            m = await self.client.slide(slide_id)      # ← your call
            bands = m['metadata'].get('bands') or []   # fluorescence: one band per marker
            return RemoteImageInfo(width=m['width'], height=m['height'], tile_size=m['tile_size'],
                                   downsamples=m['downsamples'],            # e.g. [1, 4, 16, 32]
                                   mpp=m['mpp'], name=m['label'], version=m['etag'],
                                   # without these every image is treated as brightfield RGB (R·G·B histograms)
                                   mode='channels' if bands else 'rgb',
                                   dtype=m.get('dtype', 'uint8') if bands else 'uint8',
                                   channels=[Channel(b['name'], b.get('color', '#ffffff')) for b in bands])

        async def tile(self, image_id, level, col, row, channel=None):
            slide_id, _, scene = image_id.partition('#')
            # pass the channel (its index in ``channels``) and the scene on: ignoring them returns the default
            # composite of the default scene for every channel
            return await self.client.tile(slide_id, level, col, row,
                                          band=channel, image_index=int(scene) if scene else None)  # ← bytes / ndarray

        async def annotations(self, slide_id):
            gj = await self.client.annotations_geojson(slide_id)          # ← GeoJSON FeatureCollection
            return [AnnotationPayload('PathCore annotations', gj, units='um')]  # µm → px uses the slide mpp

    pc = WissiPathCore(client).bind_loop(nicegui_loop)   # only if the client is bound to that loop
    sid = pc.register(registry, 'S-998')                 # → 'pathcore-S-998'

A fluorescence tile is one channel as a 2-D array (uint8 / uint16 / float32, or an encoded single-channel PNG /
TIFF), one call per channel; RGB tiles stay as they are. ``tests/test_remote.py`` has a working example.
"""

from __future__ import annotations

from .base import RemoteConnection


class PathCoreConnection(RemoteConnection):
    """Stub for PathCore. Implement :meth:`image_info` and :meth:`tile` (see module docstring)."""

    id_prefix = 'pathcore'
    max_concurrency = 8
