"""Adapters for slides that live behind an API (PathCore, HALOLink, OMERO, …).

You already have authenticated clients for those services; wrap their calls here
and the viewer treats the result exactly like a local file. Example::

    def fetch_tile(level, tx, ty, channel):
        # your HALOLink / PathCore client — return JPEG/PNG/WebP bytes or a numpy array
        return halolink.get_tile(image_id, level=level, col=tx, row=ty)

    info = SlideInfo(width=W, height=H, tile_size=256,       # use the service's own tile size
                     levels=[Level(w, h, ds) for ...],       # and its own pyramid → zero re-tiling
                     mpp=(0.25, 0.25), name='Case 12 / block A')
    src = CallbackSource(info, read_tile=fetch_tile, version=image_modified_timestamp)
    registry.add(src, slide_id='halo-12345')

Matching ``tile_size``/``levels`` to the remote pyramid means every viewer tile
maps to exactly one remote tile, and already-encoded bytes are passed straight
through (no decode, no re-encode) and cached in memory and on disk.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import SlideInfo, SlideSource, _fit

TileFn = Callable[[int, int, int, 'int | None'], 'np.ndarray | bytes']
RegionFn = Callable[[int, int, int, int, int, 'int | None'], np.ndarray]


def sniff_media_type(b: bytes) -> str | None:
    if b[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if b[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if b[:4] == b'RIFF' and b[8:12] == b'WEBP':
        return 'image/webp'
    if b[:12] in (b'\x00\x00\x00\x0cjP  \r\n\x87\n',):
        return None  # JPEG 2000: browsers can't decode → re-encode
    return None


class CallbackSource(SlideSource):
    def __init__(self, info: SlideInfo, *, read_tile: TileFn | None = None, read_region: RegionFn | None = None,
                 annotations: Callable[[], list] | list | None = None, version: str = '0',
                 thread_safe: bool = True) -> None:
        super().__init__()
        if read_tile is None and read_region is None:
            raise ValueError('provide read_tile and/or read_region')
        self.info = info
        self.version = str(version)
        self._tile, self._region, self._ann = read_tile, read_region, annotations
        self._thread_safe = thread_safe

    def _call_tile(self, level, tx, ty, channel):
        if self._thread_safe:
            return self._tile(level, tx, ty, channel)
        with self.lock:
            return self._tile(level, tx, ty, channel)

    def fetch_tile(self, level, tx, ty, channel=None):
        """One call to the remote API per tile; browser-decodable bytes pass straight through."""
        if self._tile is None:
            return super().fetch_tile(level, tx, ty, channel)
        out = self._call_tile(level, tx, ty, channel)
        if isinstance(out, (bytes, bytearray, memoryview)):
            b = bytes(out)
            mt = sniff_media_type(b)
            if mt and self.info.mode == 'rgb':
                return b, mt
            out = _decode(b)
        return self._normalize(np.asarray(out), level, tx, ty)

    def _normalize(self, out, level, tx, ty):
        _, _, w, h = self.tile_rect(level, tx, ty)
        if self.info.mode == 'rgb':
            if out.ndim == 2:
                out = np.repeat(out[..., None], 3, axis=2)
            out = out[..., :3]
        elif out.ndim == 3:
            out = out[..., 0]
        return _fit(out, w, h)

    def read_tile(self, level, tx, ty, channel=None):
        if self._tile is None:
            x, y, w, h = self.tile_rect(level, tx, ty)
            return _fit(np.asarray(self._region(level, x, y, w, h, channel)), w, h)
        out = self._call_tile(level, tx, ty, channel)
        if isinstance(out, (bytes, bytearray, memoryview)):
            out = _decode(bytes(out))
        return self._normalize(np.asarray(out), level, tx, ty)

    def read_region(self, level, x, y, w, h, channel=None):
        if self._region is not None:
            return self._region(level, x, y, w, h, channel)
        # assemble from tiles
        ts = self.info.tile_size
        sample = None
        out = None
        for ty in range(y // ts, (y + h - 1) // ts + 1):
            for tx in range(x // ts, (x + w - 1) // ts + 1):
                t = self.read_tile(level, tx, ty, channel)
                if out is None:
                    out = np.zeros((h, w) + t.shape[2:], t.dtype)
                    sample = t
                ox, oy = tx * ts - x, ty * ts - y
                sx0, sy0 = max(0, -ox), max(0, -oy)
                dx0, dy0 = max(0, ox), max(0, oy)
                ww = min(t.shape[1] - sx0, w - dx0)
                hh = min(t.shape[0] - sy0, h - dy0)
                if ww > 0 and hh > 0:
                    out[dy0:dy0 + hh, dx0:dx0 + ww] = t[sy0:sy0 + hh, sx0:sx0 + ww]
        return out if sample is not None else np.zeros((h, w, 3), np.uint8)

    def annotation_layers(self):
        a = self._ann
        return list(a() if callable(a) else (a or []))


def _decode(b: bytes) -> np.ndarray:
    try:
        import imagecodecs
        return imagecodecs.imread(b)
    except Exception:
        import io

        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(b)).convert('RGB'))
