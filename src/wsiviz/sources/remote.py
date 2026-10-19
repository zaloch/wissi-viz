"""Remote (API-backed) slides: PathCore, HALOLink, OMERO, DeepZoom servers, …

Real tile APIs differ from local files in ways that break naive viewers. This module
handles them so the rest of wsiviz never has to know a slide is remote:

* **Pyramid layout** — explicit level lists (finest- or coarsest-first), DeepZoom
  (level 0 = 1×1 px, ``maxLevel`` = full resolution), arbitrary downsample lists, and
  the service's own level identifiers (``"0"``, ``12``, ``"x4"`` …).
* **Tile geometry** — DeepZoom-style *overlap* (tiles carry extra pixels on interior
  edges) and *padded edge tiles* (always ``tile_size`` wide, filled past the image).
  Encoded tiles are passed to the browser untouched and cropped on the GPU by
  texture coordinates; decoded arrays are cropped on the server.
* **Clients** — sync or ``async`` callables. Coroutines run on the loop you pass
  (e.g. NiceGUI's, so an ``httpx.AsyncClient`` bound to it keeps working) or on a
  private background loop.
* **Robustness** — per-connection concurrency limit (don't hammer the service),
  retries with back-off on transient errors, a timeout, and "no tile here"
  (``None`` / :class:`TileNotFound`) rendered as background instead of an error.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .base import Channel, Level, SlideInfo, SlideSource, _fit
from .callback import _decode, sniff_media_type


class TileNotFound(Exception):
    """Raise (or return ``None``) from a tile fetch for "no data here" — rendered as background, cached."""


class RemoteError(Exception):
    """A remote call failed after all retries."""


# ----------------------------------------------------------------------------- pyramid layouts
@dataclass
class RemotePyramid:
    """How the remote service lays out its pyramid. Levels are stored finest-first.

    ``level_ids[i]`` is what gets passed back to *your* fetch function for wsiviz level ``i``.
    """
    width: int
    height: int
    tile_size: int
    levels: list[tuple[int, int]]
    level_ids: list[Any]
    overlap: int = 0

    @classmethod
    def explicit(cls, width: int, height: int, tile_size: int, levels: Sequence[Sequence[float]],
                 level_ids: Sequence[Any] | None = None, overlap: int = 0) -> 'RemotePyramid':
        """``levels``: ``(w, h)`` or ``(w, h, downsample)`` per level, in any order.
        ``level_ids`` (same order as ``levels``) default to each level's index in the order given."""
        ids = list(level_ids) if level_ids is not None else list(range(len(levels)))
        if len(ids) != len(levels):
            raise ValueError('level_ids and levels differ in length')
        pairs = sorted(zip(levels, ids), key=lambda t: -int(t[0][0]))
        return cls(int(width), int(height), int(tile_size), [(int(lv[0]), int(lv[1])) for lv, _ in pairs],
                   [i for _, i in pairs], int(overlap))

    @classmethod
    def from_downsamples(cls, width: int, height: int, tile_size: int, downsamples: Sequence[float],
                         level_ids: Sequence[Any] | None = None, overlap: int = 0,
                         rounding: str = 'ceil') -> 'RemotePyramid':
        r = math.ceil if rounding == 'ceil' else (math.floor if rounding == 'floor' else round)
        lv = [(max(1, int(r(width / d))), max(1, int(r(height / d)))) for d in downsamples]
        return cls.explicit(width, height, tile_size, lv, level_ids, overlap)

    @classmethod
    def deepzoom(cls, width: int, height: int, tile_size: int = 254, overlap: int = 1,
                 min_level_tiles: int = 1) -> 'RemotePyramid':
        """Microsoft DeepZoom (DZI): level ``maxLevel = ceil(log2(max(w, h)))`` is full resolution,
        each lower level halves (rounding up). Levels coarser than one tile are skipped."""
        max_level = math.ceil(math.log2(max(width, height))) if max(width, height) > 1 else 0
        levels, ids = [], []
        for i in range(max_level + 1):
            w, h = math.ceil(width / 2 ** i), math.ceil(height / 2 ** i)
            levels.append((w, h))
            ids.append(max_level - i)
            if math.ceil(w / tile_size) * math.ceil(h / tile_size) <= min_level_tiles:
                break
        return cls(int(width), int(height), int(tile_size), levels, ids, int(overlap))

    @classmethod
    def from_dzi(cls, xml_text: str | bytes) -> 'RemotePyramid':
        """Parse a ``.dzi`` descriptor (``<Image TileSize Overlap Format><Size Width Height/></Image>``)."""
        root = ET.fromstring(xml_text)
        size = next(e for e in root.iter() if e.tag.split('}')[-1] == 'Size')
        return cls.deepzoom(int(size.get('Width')), int(size.get('Height')), int(root.get('TileSize', 254)),
                            int(root.get('Overlap', 0)))

    def to_levels(self) -> list[Level]:
        return [Level(w, h, self.width / w) for (w, h) in self.levels]


# ----------------------------------------------------------------------------- async bridge
class _LoopThread:
    """A private asyncio loop in a daemon thread, for async clients without a loop of their own."""
    _inst: '_LoopThread | None' = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name='wsiviz-remote-loop').start()

    @classmethod
    def get(cls) -> asyncio.AbstractEventLoop:
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst.loop


def resolve(value: Any, loop: asyncio.AbstractEventLoop | None, timeout: float | None) -> Any:
    """Return ``value``, awaiting it first if it is awaitable. Called from worker threads."""
    if not inspect.isawaitable(value):
        return value
    target = loop or _LoopThread.get()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is target:
        raise RuntimeError('remote fetch called on its own event loop; wsiviz calls it from worker threads')

    async def _run():
        return await asyncio.wait_for(value, timeout) if timeout else await value
    return asyncio.run_coroutine_threadsafe(_run(), target).result()


# ----------------------------------------------------------------------------- source
FetchTile = Callable[[Any, int, int, 'int | None'], Any]
FetchRegion = Callable[[Any, int, int, int, int, 'int | None'], Any]


class RemoteSlideSource(SlideSource):
    """A slide served by a remote tile API.

    ``fetch_tile(level_id, col, row, channel)`` receives the *remote* level identifier
    (from :class:`RemotePyramid`) and returns JPEG/PNG/WebP bytes, a numpy array, ``None``
    (no data) — or an awaitable of those. ``fetch_region`` is an optional fallback with
    ``(level_id, x, y, w, h, channel)`` in that level's pixel coordinates.
    """

    def __init__(self, pyramid: RemotePyramid, fetch_tile: FetchTile | None = None, *,
                 fetch_region: FetchRegion | None = None, mpp: tuple[float, float] | float | None = None,
                 name: str = '', mode: str = 'rgb', channels: Sequence[Channel | str] | None = None,
                 dtype: str = 'uint8', background: str | None = None,
                 annotations: Callable[[], list] | list | None = None, version: str = '0',
                 max_concurrency: int = 8, semaphore: threading.Semaphore | None = None,
                 retries: int = 2, timeout: float | None = 30.0, loop: asyncio.AbstractEventLoop | None = None,
                 retry_on: tuple[type[BaseException], ...] = (Exception,),
                 not_found: tuple[type[BaseException], ...] = (TileNotFound, FileNotFoundError, KeyError),
                 browser_tile_url: str | None = None) -> None:
        super().__init__()
        if fetch_tile is None and fetch_region is None:
            raise ValueError('provide fetch_tile and/or fetch_region')
        self.pyramid = pyramid
        self._tile, self._region = fetch_tile, fetch_region
        self._ann = annotations
        self.version = str(version)
        self._sem = semaphore or threading.BoundedSemaphore(max_concurrency)
        self.retries, self.timeout, self.loop = retries, timeout, loop
        self.retry_on, self.not_found = retry_on, not_found
        if isinstance(mpp, (int, float)):
            mpp = (float(mpp), float(mpp))
        chans = [c if isinstance(c, Channel) else Channel(str(c)) for c in (channels or [])]
        if mode == 'channels' and not chans:
            chans = [Channel('Channel 1')]
        self.info = SlideInfo(width=pyramid.width, height=pyramid.height, levels=pyramid.to_levels(),
                              mode=mode, dtype=dtype, channels=chans, mpp=mpp, name=name,
                              tile_size=pyramid.tile_size,
                              background=background or ('#ffffff' if mode == 'rgb' else '#000000'),
                              tile_url=browser_tile_url, overlap=pyramid.overlap,
                              level_ids=[str(i) for i in pyramid.level_ids])
        self.stats = {'calls': 0, 'retries': 0, 'not_found': 0, 'errors': 0, 'seconds': 0.0}

    # -------------------------------------------------------------- calling the client
    def _call(self, fn: Callable, *args: Any) -> Any:
        delay = 0.25
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                with self._sem:
                    out = resolve(fn(*args), self.loop, self.timeout)
                self.stats['calls'] += 1
                self.stats['seconds'] += time.perf_counter() - t0
                return out
            except self.not_found:
                self.stats['not_found'] += 1
                return None
            except self.retry_on as e:  # noqa: PERF203
                if attempt >= self.retries:
                    self.stats['errors'] += 1
                    raise RemoteError(f'{fn.__name__ if hasattr(fn, "__name__") else fn}{args}: {e}') from e
                self.stats['retries'] += 1
                time.sleep(delay)
                delay *= 2
        return None

    # -------------------------------------------------------------- geometry
    def _expected(self, level: int, tx: int, ty: int) -> tuple[int, int]:
        _, _, w, h = self.tile_rect(level, tx, ty)
        return w, h

    def _crop(self, a: np.ndarray, level: int, tx: int, ty: int) -> np.ndarray:
        """Remove overlap (interior edges) and padding (past the image) from a raw remote tile."""
        w, h = self._expected(level, tx, ty)
        ov = self.pyramid.overlap
        ox = ov if (tx > 0 and a.shape[1] > w) else 0
        oy = ov if (ty > 0 and a.shape[0] > h) else 0
        return _fit(a[oy:oy + h, ox:ox + w], w, h)

    def _blank(self, level: int, tx: int, ty: int) -> np.ndarray:
        w, h = self._expected(level, tx, ty)
        if self.info.mode == 'rgb':
            bg = self.info.background.lstrip('#')
            rgb = [int(bg[i:i + 2], 16) for i in (0, 2, 4)]
            return np.broadcast_to(np.array(rgb, np.uint8), (h, w, 3)).copy()
        return np.zeros((h, w), np.uint16 if self.info.dtype == 'uint16' else np.uint8)

    def _normalize(self, a: np.ndarray) -> np.ndarray:
        if self.info.mode == 'rgb':
            if a.ndim == 2:
                a = np.repeat(a[..., None], 3, axis=2)
            a = a[..., :3]
            if a.dtype != np.uint8:
                a = (a >> 8).astype(np.uint8) if a.dtype == np.uint16 else np.clip(a, 0, 255).astype(np.uint8)
        else:
            if a.ndim == 3:
                a = a[..., 0]
            want = np.uint16 if self.info.dtype == 'uint16' else np.uint8
            if a.dtype != want:
                a = np.clip(a, 0, np.iinfo(want).max).astype(want)
        return a

    # -------------------------------------------------------------- SlideSource API
    def fetch_tile(self, level, tx, ty, channel=None):
        """One remote call per tile. Browser-decodable bytes are passed through untouched
        (the GPU crops overlap/padding); anything else is decoded, cropped and normalised."""
        if self._tile is None:
            return self.read_tile(level, tx, ty, channel)
        out = self._call(self._tile, self.pyramid.level_ids[level], tx, ty, channel)
        if out is None:
            return self._blank(level, tx, ty)
        if isinstance(out, (bytes, bytearray, memoryview)):
            b = bytes(out)
            mt = sniff_media_type(b)
            if mt and self.info.mode == 'rgb':
                return b, mt
            out = _decode(b)
        return self._crop(self._normalize(np.asarray(out)), level, tx, ty)

    def read_tile(self, level, tx, ty, channel=None):
        if self._tile is not None:
            out = self._call(self._tile, self.pyramid.level_ids[level], tx, ty, channel)
            if out is None:
                return self._blank(level, tx, ty)
            if isinstance(out, (bytes, bytearray, memoryview)):
                out = _decode(bytes(out))
            return self._crop(self._normalize(np.asarray(out)), level, tx, ty)
        x, y, w, h = self.tile_rect(level, tx, ty)
        out = self._call(self._region, self.pyramid.level_ids[level], x, y, w, h, channel)
        if out is None:
            return self._blank(level, tx, ty)
        if isinstance(out, (bytes, bytearray, memoryview)):
            out = _decode(bytes(out))
        return _fit(self._normalize(np.asarray(out)), w, h)

    def read_region(self, level, x, y, w, h, channel=None):
        if self._region is not None:
            out = self._call(self._region, self.pyramid.level_ids[level], x, y, w, h, channel)
            if out is not None:
                if isinstance(out, (bytes, bytearray, memoryview)):
                    out = _decode(bytes(out))
                return _fit(self._normalize(np.asarray(out)), w, h)
        ts = self.info.tile_size
        lv = self.info.levels[level]
        out = None
        for ty in range(y // ts, min((y + h - 1) // ts, math.ceil(lv.height / ts) - 1) + 1):
            for tx in range(x // ts, min((x + w - 1) // ts, math.ceil(lv.width / ts) - 1) + 1):
                t = self.read_tile(level, tx, ty, channel)
                if out is None:
                    out = np.zeros((h, w) + t.shape[2:], t.dtype)
                ox, oy = tx * ts - x, ty * ts - y
                sx0, sy0, dx0, dy0 = max(0, -ox), max(0, -oy), max(0, ox), max(0, oy)
                ww, hh = min(t.shape[1] - sx0, w - dx0), min(t.shape[0] - sy0, h - dy0)
                if ww > 0 and hh > 0:
                    out[dy0:dy0 + hh, dx0:dx0 + ww] = t[sy0:sy0 + hh, sx0:sx0 + ww]
        return out if out is not None else self._blank(level, 0, 0)[:h, :w]

    def annotation_layers(self):
        a = self._ann
        layers = a() if callable(a) else (a or [])
        return list(resolve(layers, self.loop, None))


@dataclass
class RemoteImageInfo:
    """What a connection's ``image_info()`` returns. Give either ``pyramid`` or enough to build one."""
    width: int
    height: int
    pyramid: RemotePyramid | None = None
    tile_size: int = 256
    #: levels as (w, h) or (w, h, downsample); ignored if ``pyramid`` or ``deepzoom`` is given
    levels: Sequence[Sequence[float]] | None = None
    level_ids: Sequence[Any] | None = None
    downsamples: Sequence[float] | None = None
    deepzoom: bool = False
    overlap: int = 0
    mpp: tuple[float, float] | float | None = None
    name: str = ''
    #: change whenever the image changes → invalidates every cached tile (e.g. modified timestamp / etag)
    version: str = '0'
    mode: str = 'rgb'
    dtype: str = 'uint8'
    channels: Sequence[Channel | str] = field(default_factory=list)
    background: str | None = None
    #: optional browser-direct tile URL template ({level} {col} {row} {c}); only if the browser can
    #: authenticate to the service itself. Otherwise tiles are proxied (and cached) by wsiviz.
    browser_tile_url: str | None = None

    def build_pyramid(self) -> RemotePyramid:
        if self.pyramid is not None:
            return self.pyramid
        if self.deepzoom:
            return RemotePyramid.deepzoom(self.width, self.height, self.tile_size, self.overlap)
        if self.levels:
            return RemotePyramid.explicit(self.width, self.height, self.tile_size, self.levels, self.level_ids,
                                          self.overlap)
        if self.downsamples:
            return RemotePyramid.from_downsamples(self.width, self.height, self.tile_size, self.downsamples,
                                                  self.level_ids, self.overlap)
        return RemotePyramid.explicit(self.width, self.height, self.tile_size, [(self.width, self.height)],
                                      self.level_ids, self.overlap)
