"""Tile pipeline: pyramid normalisation, encoding, and the two cache tiers.

Request path for one tile::

    memory LRU  ──hit──▶ bytes
        │ miss
    disk cache  ──hit──▶ bytes (promoted to memory)
        │ miss
    in-flight table ──already being produced──▶ wait for that Future
        │
    worker pool: source.fetch_tile → encode → store in both tiers
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from .sources.base import Level, SlideInfo, SlideSource


# ---------------------------------------------------------------- encoders
def encode_rgb(arr: np.ndarray, fmt: str = 'jpeg', quality: int = 85) -> tuple[bytes, str]:
    arr = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    try:
        import imagecodecs
        if fmt == 'webp':
            return imagecodecs.webp_encode(arr, level=quality), 'image/webp'
        if fmt == 'png':
            return imagecodecs.png_encode(arr, level=1), 'image/png'
        return imagecodecs.jpeg8_encode(arr, level=quality, subsampling='420'), 'image/jpeg'
    except (ImportError, AttributeError):
        from PIL import Image
        buf = io.BytesIO()
        im = Image.fromarray(arr)
        if fmt == 'webp':
            im.save(buf, 'WEBP', quality=quality)
            return buf.getvalue(), 'image/webp'
        if fmt == 'png':
            im.save(buf, 'PNG', compress_level=1)
            return buf.getvalue(), 'image/png'
        im.save(buf, 'JPEG', quality=quality)
        return buf.getvalue(), 'image/jpeg'


def encode_channel(arr: np.ndarray, dtype: str) -> tuple[bytes, str]:
    """Raw little-endian samples, zlib-compressed. The browser inflates with DecompressionStream."""
    a = np.ascontiguousarray(arr, dtype='<u2' if dtype == 'uint16' else 'u1')
    return zlib.compress(a.tobytes(), 1), 'application/x-wissi-raw'


# ---------------------------------------------------------------- pyramid normalisation
class PyramidSlide:
    """Wraps a SlideSource and fills pyramid gaps with *virtual* levels so that
    (a) consecutive levels differ by ≤ 4× and (b) the coarsest level fits in one tile.

    A virtual tile is the 2× box-filtered mosaic of its 2×2 children on the level below
    (native or virtual), memoised in a small array LRU — so even a slide with *no*
    pyramid is zoomed out by touching every pixel once, never once per tile.
    """

    MAX_STEP = 4.0

    def __init__(self, source: SlideSource, array_cache_bytes: int = 256 * 2**20) -> None:
        self.source = source
        info = source.info
        ts = info.tile_size
        native = sorted(enumerate(info.levels), key=lambda t: t[1].downsample)
        out: list[Level] = []
        self.virtual: list[bool] = []
        self.native_index: list[int] = []
        for i, (ni, lv) in enumerate(native):
            out.append(lv)
            self.virtual.append(False)
            self.native_index.append(ni)
            nxt = native[i + 1][1].downsample if i + 1 < len(native) else math.inf
            while True:
                cur = out[-1]
                if nxt != math.inf:
                    if nxt / cur.downsample <= self.MAX_STEP * 1.01 or cur.downsample * 2 >= nxt * 0.99:
                        break
                elif max(cur.width, cur.height) <= ts:
                    break
                out.append(Level(max(1, math.ceil(cur.width / 2)), max(1, math.ceil(cur.height / 2)),
                                 cur.downsample * 2))
                self.virtual.append(True)
                self.native_index.append(-1)
        self.info: SlideInfo = replace(info, levels=out)
        self._arr = OrderedDict()
        self._arr_bytes = 0
        self._arr_max = array_cache_bytes
        self._alock = threading.Lock()

    def _tile_size(self, level, tx, ty):
        lv, ts = self.info.levels[level], self.info.tile_size
        return min(ts, lv.width - tx * ts), min(ts, lv.height - ty * ts)

    def _array(self, level, tx, ty, channel) -> np.ndarray:
        key = (level, tx, ty, channel)
        with self._alock:
            a = self._arr.get(key)
            if a is not None:
                self._arr.move_to_end(key)
                return a
        if not self.virtual[level]:
            a = np.asarray(self.source.read_tile(self.native_index[level], tx, ty, channel))
        else:
            a = self._virtual(level, tx, ty, channel)
        with self._alock:
            self._arr[key] = a
            self._arr_bytes += a.nbytes
            while self._arr_bytes > self._arr_max and self._arr:
                _, old = self._arr.popitem(last=False)
                self._arr_bytes -= old.nbytes
        return a

    def _virtual(self, level, tx, ty, channel) -> np.ndarray:
        w, h = self._tile_size(level, tx, ty)
        ts = self.info.tile_size
        child = self.info.levels[level - 1]
        cw, ch = 2 * w, 2 * h
        mosaic = None
        for dy in range(2):
            for dx in range(2):
                cx, cy = 2 * tx + dx, 2 * ty + dy
                if cx * ts >= child.width or cy * ts >= child.height:
                    continue
                a = self._array(level - 1, cx, cy, channel)
                if mosaic is None:
                    mosaic = np.zeros((ch, cw) + a.shape[2:], a.dtype)
                ys, xs = dy * ts, dx * ts
                hh, ww = min(a.shape[0], ch - ys), min(a.shape[1], cw - xs)
                mosaic[ys:ys + hh, xs:xs + ww] = a[:hh, :ww]
        # replicate edges so partial tiles average correctly
        vh, vw = min(ch, child.height - 2 * ty * ts), min(cw, child.width - 2 * tx * ts)
        if vh < ch:
            mosaic[vh:] = mosaic[vh - 1:vh]
        if vw < cw:
            mosaic[:, vw:] = mosaic[:, vw - 1:vw]
        return _reduce(mosaic, 2, w, h)

    def fetch_tile(self, level: int, tx: int, ty: int, channel: int | None):
        if not self.virtual[level]:
            return self.source.fetch_tile(self.native_index[level], tx, ty, channel)
        return self._array(level, tx, ty, channel)


def _reduce(a: np.ndarray, f: int, w: int, h: int) -> np.ndarray:
    sh = (h, f, w, f) + a.shape[2:]
    return a[:h * f, :w * f].reshape(sh).mean(axis=(1, 3)).astype(a.dtype)


# ---------------------------------------------------------------- caches
class MemoryCache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._d: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str):
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key: str, value: tuple[bytes, str]) -> None:
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._size -= len(old[0])
            self._d[key] = value
            self._size += len(value[0])
            while self._size > self.max_bytes and self._d:
                _, v = self._d.popitem(last=False)
                self._size -= len(v[0])

    def stats(self) -> dict:
        return {'entries': len(self._d), 'bytes': self._size, 'max_bytes': self.max_bytes,
                'hits': self.hits, 'misses': self.misses}


_EXT = {'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/png': 'png', 'application/x-wissi-raw': 'raw'}
_MT = {v: k for k, v in _EXT.items()}


class DiskCache:
    """Persistent tile cache (survives restarts; essential for remote/API-backed slides).
    Trimmed to ``max_bytes`` by least-recently-*written* files."""

    def __init__(self, root: str | os.PathLike, max_bytes: int = 20 * 2**30) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._writes = 0
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        h = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
        return self.root / h[:2] / h[2:4] / h

    def get(self, key: str):
        p = self._path(key)
        for ext, mt in _MT.items():
            q = p.with_suffix('.' + ext)
            try:
                return q.read_bytes(), mt
            except FileNotFoundError:
                continue
        return None

    def put(self, key: str, value: tuple[bytes, str]) -> None:
        data, mt = value
        p = self._path(key).with_suffix('.' + _EXT.get(mt, 'bin'))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
        tmp.write_bytes(data)
        os.replace(tmp, p)
        with self._lock:
            self._writes += 1
            trim = self._writes % 2000 == 0
        if trim:
            threading.Thread(target=self.trim, daemon=True).start()

    def trim(self) -> None:
        files = []
        total = 0
        for p in self.root.rglob('*'):
            if p.is_file():
                st = p.stat()
                files.append((st.st_mtime, st.st_size, p))
                total += st.st_size
        if total <= self.max_bytes:
            return
        files.sort()
        for _, size, p in files:
            if total <= self.max_bytes * 0.9:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                pass


class TileService:
    """Produces encoded tiles with caching, request coalescing and a bounded worker pool."""

    def __init__(self, memory_bytes: int = 1024 * 2**20, disk_dir: str | os.PathLike | None = None,
                 disk_bytes: int = 20 * 2**30, workers: int | None = None, jpeg_quality: int = 85,
                 rgb_format: str = 'jpeg') -> None:
        self.memory = MemoryCache(memory_bytes)
        self.disk = DiskCache(disk_dir, disk_bytes) if disk_dir else None
        self.pool = ThreadPoolExecutor(max_workers=workers or min(32, (os.cpu_count() or 4) * 2),
                                       thread_name_prefix='wissi-viz-tile')
        self.warm_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='wissi-viz-warm')
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()
        self.jpeg_quality = jpeg_quality
        self.rgb_format = rgb_format
        self.produced = 0
        self.produce_seconds = 0.0

    @staticmethod
    def key(slide_id: str, version: str, level: int, tx: int, ty: int, channel: int | None) -> str:
        return f'{slide_id}|{version}|{level}|{tx}|{ty}|{-1 if channel is None else channel}'

    def get(self, slide_id: str, slide: PyramidSlide, version: str, level: int, tx: int, ty: int,
            channel: int | None) -> Future:
        """Return a Future resolving to (bytes, media_type)."""
        k = self.key(slide_id, version, level, tx, ty, channel)
        v = self.memory.get(k)
        if v is not None:
            f: Future = Future()
            f.set_result(v)
            return f
        with self._lock:
            f = self._inflight.get(k)
            if f is not None:
                return f
            f = self.pool.submit(self._produce, k, slide, level, tx, ty, channel)
            self._inflight[k] = f
        f.add_done_callback(lambda _f, k=k: self._done(k))
        return f

    def _done(self, k: str) -> None:
        with self._lock:
            self._inflight.pop(k, None)

    def _produce(self, k, slide: PyramidSlide, level, tx, ty, channel):
        if self.disk is not None:
            v = self.disk.get(k)
            if v is not None:
                self.memory.put(k, v)
                return v
        t0 = time.perf_counter()
        out = slide.fetch_tile(level, tx, ty, channel)
        if isinstance(out, tuple):
            v = out
        elif slide.info.mode == 'rgb':
            v = encode_rgb(out, self.rgb_format, self.jpeg_quality)
        else:
            v = encode_channel(out, slide.info.dtype)
        self.memory.put(k, v)
        if self.disk is not None:
            try:
                self.disk.put(k, v)
            except OSError:
                pass
        self.produced += 1
        self.produce_seconds += time.perf_counter() - t0
        return v

    def warm(self, slide_id: str, slide: PyramidSlide, version: str, max_tiles_per_level: int = 16) -> None:
        """Pre-render coarse levels in the background so the first zoomed-out view is instant.
        Warming yields to interactive requests: it pauses while any foreground tile is in flight."""
        info = slide.info
        chans = [None] if info.mode == 'rgb' else list(range(len(info.channels)))

        def run():
            for lvl in range(len(info.levels) - 1, -1, -1):
                lv = info.levels[lvl]
                nx, ny = math.ceil(lv.width / info.tile_size), math.ceil(lv.height / info.tile_size)
                if nx * ny > max_tiles_per_level:
                    break
                for c in chans:
                    for ty in range(ny):
                        for tx in range(nx):
                            for _ in range(200):  # foreground first (≤ 2 s back-off per tile)
                                if not self._inflight:
                                    break
                                time.sleep(0.01)
                            try:
                                self.get(slide_id, slide, version, lvl, tx, ty, c).result()
                            except Exception:
                                return
        self.warm_pool.submit(run)

    def stats(self) -> dict:
        return {'memory': self.memory.stats(), 'disk': str(self.disk.root) if self.disk else None,
                'produced': self.produced,
                'avg_produce_ms': round(1000 * self.produce_seconds / max(1, self.produced), 2),
                'inflight': len(self._inflight)}
