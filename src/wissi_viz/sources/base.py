"""Slide source abstraction.

Every slide the viewer can show — a local SVS/CZI file, or a slide that lives
behind PathCore / HALOLink / any other API — is a :class:`SlideSource`.

A source only has to describe its pyramid (:class:`SlideInfo`) and return pixels
for a region of one pyramid level. Everything else (tiling, caching, encoding,
HTTP, GPU rendering) is handled by wissi-viz.
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from ..annotations import AnnotationLayer

Mode = Literal['rgb', 'channels']

DEFAULT_CHANNEL_COLORS = ['#0000ff', '#00ff00', '#ff0000', '#ff00ff', '#00ffff', '#ffff00', '#ff8000', '#ffffff']


@dataclass
class Level:
    """One pyramid level. ``downsample`` is relative to level 0."""

    width: int
    height: int
    downsample: float


@dataclass
class Channel:
    name: str
    color: str = '#ffffff'
    #: Initial display window (lo, hi) in raw pixel units; None → auto from statistics.
    window: tuple[float, float] | None = None
    visible: bool = True


@dataclass
class SlideInfo:
    width: int
    height: int
    levels: list[Level]
    mode: Mode = 'rgb'
    dtype: Literal['uint8', 'uint16'] = 'uint8'
    channels: list[Channel] = field(default_factory=list)
    #: microns per pixel at level 0 (x, y)
    mpp: tuple[float, float] | None = None
    name: str = ''
    tile_size: int = 512
    background: str = '#ffffff'
    #: Optional: the browser fetches tiles directly from this URL template instead of
    #: from the wissi-viz server. Placeholders: {level} {x} {y} {c} {z}. Use this when an
    #: external service (e.g. a tile endpoint that already accepts browser auth) can
    #: serve tiles laid out exactly like this pyramid.
    tile_url: str | None = None
    #: Remote tiles that carry extra pixels on interior edges (DeepZoom overlap). Encoded tiles
    #: are passed through and cropped on the GPU; decoded arrays are cropped on the server.
    overlap: int = 0
    #: Remote level identifiers (for ``tile_url`` templates); index-aligned with ``levels``.
    level_ids: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d['levels'] = [asdict(lv) for lv in self.levels]
        return d

    @staticmethod
    def pyramid(width: int, height: int, tile_size: int = 512, factor: int = 2) -> list[Level]:
        """Synthesize a power-of-``factor`` pyramid down to a single tile."""
        levels, ds = [], 1
        while True:
            w, h = max(1, math.ceil(width / ds)), max(1, math.ceil(height / ds))
            levels.append(Level(w, h, float(ds)))
            if w <= tile_size and h <= tile_size:
                return levels
            ds *= factor


class SlideSource(ABC):
    """Base class for anything the viewer can display.

    Subclasses must set ``self.info`` and implement :meth:`read_region`.
    Optionally override :meth:`fetch_tile` to pass through already-encoded
    tiles (e.g. JPEG bytes from a remote API) without a decode/re-encode cycle,
    and :meth:`annotation_layers` to expose annotations that come with the slide.
    """

    info: SlideInfo
    #: Cache-busting token; change it when the underlying data changes.
    version: str = '0'

    def __init__(self) -> None:
        self.lock = threading.RLock()

    @abstractmethod
    def read_region(self, level: int, x: int, y: int, w: int, h: int, channel: int | None = None) -> np.ndarray:
        """Return pixels of ``level`` for the rectangle (x, y, w, h) in *that level's* coordinates.

        * ``mode == 'rgb'``: uint8 array (h, w, 3)
        * ``mode == 'channels'``: array (h, w) of ``info.dtype`` for ``channel``
        """

    def fetch_tile(self, level: int, tx: int, ty: int, channel: int | None = None) -> np.ndarray | tuple[bytes, str]:
        """Return a tile as an array, or as already-encoded ``(bytes, media_type)`` that the
        browser can decode directly (pass-through fast path for remote JPEG/WebP tiles)."""
        return self.read_tile(level, tx, ty, channel)

    def annotation_layers(self) -> list[AnnotationLayer]:
        """Annotation layers that ship with the slide (embedded or sidecar files)."""
        return []

    def close(self) -> None:  # noqa: B027
        pass

    # ------------------------------------------------------------------ helpers
    def tile_rect(self, level: int, tx: int, ty: int) -> tuple[int, int, int, int]:
        lv, ts = self.info.levels[level], self.info.tile_size
        x, y = tx * ts, ty * ts
        if x >= lv.width or y >= lv.height or tx < 0 or ty < 0:
            raise IndexError('tile out of range')
        return x, y, min(ts, lv.width - x), min(ts, lv.height - y)

    def read_tile(self, level: int, tx: int, ty: int, channel: int | None = None) -> np.ndarray:
        x, y, w, h = self.tile_rect(level, tx, ty)
        with self.lock:
            arr = self.read_region(level, x, y, w, h, channel)
        return _fit(arr, w, h)

    def thumbnail(self, max_size: int = 1024, channel: int | None = None) -> np.ndarray:
        """Read the smallest level that is at least ``max_size`` on its long side (or the coarsest)."""
        lvl = len(self.info.levels) - 1
        for i, lv in enumerate(self.info.levels):
            if max(lv.width, lv.height) <= max_size * 2:
                lvl = i
                break
        lv = self.info.levels[lvl]
        if lv.width * lv.height > 4096 * 4096:  # never read huge arrays for a thumbnail
            lvl = len(self.info.levels) - 1
            lv = self.info.levels[lvl]
        with self.lock:
            return self.read_region(lvl, 0, 0, lv.width, lv.height, channel)

    def channel_stats(self) -> list[tuple[float, float]]:
        """Robust display windows (0.5–99.8 percentile) per channel from a thumbnail."""
        out = []
        for c in range(len(self.info.channels)):
            a = self.thumbnail(1024, c).astype(np.float32).ravel()
            a = a[a > 0] if np.count_nonzero(a) > 100 else a
            if a.size == 0:
                out.append((0.0, 1.0))
                continue
            lo, hi = np.percentile(a, [0.5, 99.8])
            out.append((float(lo), float(max(hi, lo + 1))))
        return out


def _fit(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Crop or zero-pad ``arr`` to exactly (h, w)."""
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    out = np.zeros((h, w) + arr.shape[2:], dtype=arr.dtype)
    hh, ww = min(h, arr.shape[0]), min(w, arr.shape[1])
    out[:hh, :ww] = arr[:hh, :ww]
    return out
