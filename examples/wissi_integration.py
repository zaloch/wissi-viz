"""wissi integration: local SVS/CZI + PathCore + HALO Link in one NiceGUI viewer.

The two connector classes below are the ONLY wissi-specific code. Fill the marked
lines with calls to the clients wissi already has (auth, sessions, base URLs stay
in wissi). wsiviz never opens a connection itself.

Everything else is handled for you and covered by tests (tests/test_remote.py):
pyramid mapping (explicit / downsample list / DeepZoom, any level order, your own
level ids), tile overlap and padded edge tiles, JPEG/PNG pass-through with GPU-side
cropping, sync or async clients, retries + concurrency limits, memory/disk caching,
and conversion of HALO XML / GeoJSON / object tables (px or µm) to GPU annotations.
"""
from __future__ import annotations

from nicegui import app, ui

from wsiviz.connectors import (AnnotationPayload, HaloLinkConnection, PathCoreConnection, RemoteImageInfo,
                               TileNotFound)
from wsiviz.nicegui import WsiGrid, configure

registry = configure(cache_dir='/cache/wsiviz', memory_mb=2048, disk_gb=100)  # docker volume → survives restarts


# ============================================================================ HALO Link  (stub → fill in)
class WissiHaloLink(HaloLinkConnection):
    max_concurrency = 8          # simultaneous requests to HALO Link from this process

    def __init__(self, client):
        super().__init__()
        self.client = client     # ← wissi's existing, authenticated HALO Link client

    def image_info(self, image_id):
        m = self.client.get_image(image_id)                                   # ← FILL IN
        return RemoteImageInfo(
            width=m['width'], height=m['height'],
            tile_size=m['tile_size'],
            # describe the pyramid exactly as the service does — pick ONE of:
            levels=[(lv['width'], lv['height']) for lv in m['levels']],       #   explicit levels (any order)
            level_ids=[lv['id'] for lv in m['levels']],                       #   …and the ids your tile() expects
            # downsamples=[1, 4, 16, 64],                                     #   or a downsample list
            # deepzoom=True, overlap=1,                                       #   or DeepZoom
            mpp=(m['mpp_x'], m['mpp_y']),
            name=m['name'],
            version=str(m['modified']),        # changes → all cached tiles for this image are invalidated
        )

    def tile(self, image_id, level, col, row, channel=None):
        data = self.client.get_tile(image_id, level, col, row)                # ← FILL IN: bytes / ndarray
        if data is None:
            raise TileNotFound                 # empty background → rendered as blank, cached
        return data

    def annotations(self, image_id):
        xml = self.client.get_annotations_xml(image_id)                       # ← FILL IN (or GeoJSON)
        return [AnnotationPayload('HALO annotations', xml, format='halo_xml', style={'line_width': 2.5})] if xml else []

    def objects(self, image_id):
        table = self.client.get_object_data(image_id)                         # ← FILL IN (DataFrame / dict / rows)
        if table is None:
            return []
        return [AnnotationPayload('HALO objects', table, format='table',     # XMin/XMax/YMin/YMax, Classification…
                                  style={'line_width': 1, 'fill_opacity': 0.0})]


# ============================================================================ PathCore  (stub → fill in)
class WissiPathCore(PathCoreConnection):
    def __init__(self, client):
        super().__init__()
        self.client = client     # ← wissi's existing PathCore client (sync or async both fine)

    async def image_info(self, slide_id):
        m = await self.client.slide_metadata(slide_id)                        # ← FILL IN
        return RemoteImageInfo(width=m['width'], height=m['height'], tile_size=m['tile_size'],
                               downsamples=m['downsamples'], mpp=m['mpp'], name=m['name'], version=m['etag'])

    async def tile(self, slide_id, level, col, row, channel=None):
        return await self.client.tile(slide_id, level, col, row)             # ← FILL IN

    async def annotations(self, slide_id):
        gj = await self.client.annotations(slide_id)                          # ← FILL IN
        return [AnnotationPayload('PathCore annotations', gj, units='um')]   # µm → px via the slide's mpp


# ============================================================================ wiring
halo = WissiHaloLink(client=None)        # ← pass wissi's client objects
pathcore = WissiPathCore(client=None)

# If an async client is bound to NiceGUI's event loop (e.g. one httpx.AsyncClient created at
# startup), run its coroutines there; otherwise wsiviz uses its own background loop.
app.on_startup(lambda: pathcore.bind_loop())


@ui.page('/viewer')
def viewer_page(halo_ids: str = '', pathcore_ids: str = '', files: str = ''):
    slides = [f for f in files.split(',') if f]                                   # local SVS / CZI paths
    slides += [halo.register(registry, i) for i in halo_ids.split(',') if i]       # lazy: no call until shown
    slides += [pathcore.register(registry, i) for i in pathcore_ids.split(',') if i]
    grid = WsiGrid(slides[:16], layout='auto').classes('w-full').style('height: 85vh')

    with ui.row():
        ui.button('All → (1200 µm, 800 µm) @20×', on_click=lambda: grid.go_to('all', 1200, 800, units='um', magnification=20))
        ui.button('Link navigation', on_click=lambda: grid.set_sync('all'))
        ui.button('4×4', on_click=lambda: grid.set_layout('4x4'))
