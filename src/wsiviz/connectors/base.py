"""Connector base: the contract between wsiviz and *your* already-authenticated API client.

Subclass :class:`RemoteConnection` (or the PathCore / HALOLink stubs, which only add
naming and defaults), implement the methods marked **required**, and register slides:

    conn = MyHaloLink(session=wissi.halolink_session)      # your auth/session, untouched
    sid = conn.register(registry, image_id)                # lazy: no network call yet
    grid.set_slide(0, sid)

Methods may be plain functions or ``async def`` — both work. Raise
:class:`~wsiviz.sources.remote.TileNotFound` (or return ``None``) for "no tile here".
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..annotations import AnnotationLayer
from ..sources.remote import RemoteImageInfo, RemotePyramid, RemoteSlideSource, TileNotFound, resolve

log = logging.getLogger('wsiviz.connectors')

__all__ = ['RemoteConnection', 'RemoteImageInfo', 'RemotePyramid', 'AnnotationPayload', 'TileNotFound']


@dataclass
class AnnotationPayload:
    """One annotation layer as delivered by the remote API.

    ``data``: GeoJSON (dict/list/str), HALO XML text, Aperio XML text, an object table
    (DataFrame / dict of columns / list of rows), or a ready :class:`AnnotationLayer`.
    Coordinates are converted to level-0 pixels with ``units``/``downsample``/``offset``.
    """
    name: str
    data: Any
    format: str = 'auto'          # 'auto' | 'geojson' | 'halo_xml' | 'aperio_xml' | 'table'
    units: str = 'px'             # 'px' (level-0 pixels) | 'um'
    downsample: float = 1.0       # coordinates given at a pyramid level with this downsample
    offset: tuple[float, float] = (0.0, 0.0)  # added after scaling (level-0 px)
    style: dict[str, Any] = field(default_factory=dict)  # line_width, fill_opacity, point_size, …
    classes: dict[str, str] = field(default_factory=dict)  # class → '#rrggbb' overrides
    visible: bool = True
    table_options: dict[str, Any] = field(default_factory=dict)  # passed to AnnotationLayer.from_table


class RemoteConnection:
    """Base class for API connectors. **Required:** :meth:`image_info` and :meth:`tile`
    (or :meth:`region`). **Optional:** :meth:`annotations`, :meth:`objects`.

    Class attributes tune behaviour per service; override per instance if needed.
    """

    #: prefix for registry slide ids, e.g. 'halo' → 'halo-<image id>'
    id_prefix = 'remote'
    #: max simultaneous remote calls for this connection (shared by all its slides)
    max_concurrency = 8
    #: retries on transient errors, then the tile fails (and the browser retries later)
    retries = 2
    #: seconds, applied to async calls (sync clients should set their own HTTP timeout)
    timeout: float | None = 30.0
    #: event loop that async methods must run on (e.g. NiceGUI's, if your client is bound to it).
    #: None → a private background loop. Set via ``bind_loop()`` at startup.
    loop: asyncio.AbstractEventLoop | None = None

    def __init__(self) -> None:
        self._sem = threading.BoundedSemaphore(self.max_concurrency)

    # ------------------------------------------------------------------ to implement
    def image_info(self, image_id: str) -> RemoteImageInfo:
        """**Required.** Describe the image: size, pyramid, tile size, mpp, version, channels."""
        raise NotImplementedError(f'{type(self).__name__}.image_info() — supply it from your app')

    def tile(self, image_id: str, level: Any, col: int, row: int, channel: int | None = None) -> Any:
        """**Required** (unless :meth:`region` is implemented). Return one tile of the remote
        pyramid: encoded bytes (JPEG/PNG/WebP are passed to the browser untouched), a numpy
        array, or ``None`` for "no data". ``level`` is the remote level id from your pyramid."""
        raise NotImplementedError(f'{type(self).__name__}.tile() — supply it from your app')

    def region(self, image_id: str, level: Any, x: int, y: int, w: int, h: int, channel: int | None = None) -> Any:
        """Optional fallback for services that serve arbitrary regions instead of tiles."""
        raise NotImplementedError

    def annotations(self, image_id: str) -> Iterable[AnnotationPayload | AnnotationLayer]:
        """Optional. Annotation layers (ROIs, drawings) for the image."""
        return []

    def objects(self, image_id: str) -> Iterable[AnnotationPayload | AnnotationLayer]:
        """Optional. Detected objects / cells (often large) — loaded in the background."""
        return []

    # ------------------------------------------------------------------ provided
    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> 'RemoteConnection':
        """Run this connection's ``async`` methods on ``loop`` (default: the running loop)."""
        self.loop = loop or asyncio.get_running_loop()
        return self

    def slide_id(self, image_id: str) -> str:
        return f'{self.id_prefix}-' + re.sub(r'[^A-Za-z0-9_.\-~]+', '_', str(image_id))[:180]

    def source(self, image_id: str, name: str | None = None) -> RemoteSlideSource:
        """Build the slide source (calls :meth:`image_info` once)."""
        info: RemoteImageInfo = resolve(self.image_info(image_id), self.loop, self.timeout)
        pyr = info.build_pyramid()
        has_tile = type(self).tile is not RemoteConnection.tile
        has_region = type(self).region is not RemoteConnection.region
        if not (has_tile or has_region):
            raise NotImplementedError(f'{type(self).__name__} must implement tile() or region()')
        fetch_tile = (lambda lv, c, r, ch: self.tile(image_id, lv, c, r, ch)) if has_tile else None
        fetch_region = (lambda lv, x, y, w, h, ch: self.region(image_id, lv, x, y, w, h, ch)) if has_region else None
        mpp = info.mpp
        return RemoteSlideSource(
            pyr, fetch_tile, fetch_region=fetch_region, mpp=mpp, name=name or info.name or str(image_id),
            mode=info.mode, channels=info.channels, dtype=info.dtype, background=info.background,
            annotations=lambda: self.load_annotations(image_id, mpp), version=info.version,
            semaphore=self._sem, retries=self.retries, timeout=self.timeout, loop=self.loop,
            browser_tile_url=info.browser_tile_url)

    def register(self, registry, image_id: str, name: str | None = None, slide_id: str | None = None,
                 lazy: bool = True) -> str:
        """Add the image to a :class:`~wsiviz.SlideRegistry`. With ``lazy=True`` nothing is fetched
        until a viewer opens it. Returns the slide id to pass to ``WsiGrid``."""
        sid = slide_id or self.slide_id(image_id)
        if lazy:
            registry.add(lambda: self.source(image_id, name), slide_id=sid, name=name or str(image_id))
        else:
            registry.add(self.source(image_id, name), slide_id=sid, name=name)
        return sid

    def load_annotations(self, image_id: str, mpp: Any = None) -> list[AnnotationLayer]:
        """Fetch and convert annotations + objects (runs in a background thread)."""
        out: list[AnnotationLayer] = []
        for getter in (self.annotations, self.objects):
            try:
                items = resolve(getter(image_id), self.loop, None) or []
            except NotImplementedError:
                continue
            except Exception:  # one failing endpoint must not hide the other layers
                log.exception('%s.%s(%r) failed', type(self).__name__, getter.__name__, image_id)
                continue
            for item in items:
                try:
                    out.append(to_layer(item, mpp))
                except Exception:
                    log.exception('could not convert annotation layer %r', getattr(item, 'name', item))
        return out


def to_layer(item: AnnotationPayload | AnnotationLayer, mpp: Any = None) -> AnnotationLayer:
    if isinstance(item, AnnotationLayer):
        return item
    p = item
    from ..annotations import _looks_tabular
    kw = dict(p.table_options) if (p.format == 'table' or _looks_tabular(p.data)) else {}
    layer = AnnotationLayer.parse(p.data, fmt=p.format, name=p.name, **kw)
    layer.name = p.name
    sx = sy = float(p.downsample)
    if p.units == 'um':
        if mpp is None:
            raise ValueError(f'annotation layer {p.name!r} is in µm but the image has no mpp')
        mx, my = (mpp, mpp) if isinstance(mpp, (int, float)) else mpp
        sx, sy = sx / mx, sy / my
    if (sx, sy) != (1.0, 1.0) or tuple(p.offset) != (0.0, 0.0):
        layer.transform((sx, sy), p.offset)
    for k, v in p.style.items():
        setattr(layer, k, v)
    layer.classes.update(p.classes)
    layer.visible = p.visible
    layer.touch()
    return layer


_ = np
