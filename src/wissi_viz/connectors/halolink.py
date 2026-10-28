"""HALO Link connector — **stub**: your app supplies the actual connection.

wissi-viz never talks to HALO Link itself. Subclass :class:`HaloLinkConnection` in wissi,
call your existing authenticated client inside the methods, and wissi-viz does the
rest (pyramid mapping, overlap/padding, caching, GPU rendering, annotations)::

    from wissi_viz.connectors import HaloLinkConnection, RemoteImageInfo, AnnotationPayload

    class WissiHaloLink(HaloLinkConnection):
        def __init__(self, client):
            super().__init__()
            self.client = client                       # ← your session / GraphQL / REST client

        def image_info(self, image_id):
            m = self.client.image_metadata(image_id)   # ← your call
            return RemoteImageInfo(width=m['width'], height=m['height'], tile_size=m['tileSize'],
                                   levels=[(lv['width'], lv['height']) for lv in m['levels']],
                                   level_ids=[lv['index'] for lv in m['levels']],
                                   mpp=(m['mppX'], m['mppY']), name=m['name'], version=m['modified'])

        def tile(self, image_id, level, col, row, channel=None):
            return self.client.tile(image_id, level, col, row)      # ← bytes / ndarray / None

        def annotations(self, image_id):
            xml = self.client.annotations_xml(image_id)              # ← HALO .annotations XML text
            return [AnnotationPayload('HALO annotations', xml, format='halo_xml')] if xml else []

        def objects(self, image_id):
            df = self.client.object_data(image_id)                   # ← DataFrame: XMin, XMax, YMin, YMax, Classification …
            return [AnnotationPayload('HALO objects', df, format='table',
                                      style={'line_width': 1, 'fill_opacity': 0.0})] if df is not None else []

    halo = WissiHaloLink(wissi_halolink_client)
    sid = halo.register(registry, 'image-123')                       # → 'halo-image-123'

Pyramid variants are all supported through :class:`RemoteImageInfo` — explicit levels in
any order with your own level ids, ``downsamples=[1, 4, 16, …]``, or ``deepzoom=True``
with ``overlap`` — so match whatever the service returns rather than converting.
"""

from __future__ import annotations

from .base import RemoteConnection


class HaloLinkConnection(RemoteConnection):
    """Stub for HALO Link. Implement :meth:`image_info` and :meth:`tile` (see module docstring)."""

    id_prefix = 'halo'
    max_concurrency = 8
