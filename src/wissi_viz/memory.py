"""Process-wide byte budgets: decoded pixel arrays, encoded overlay tiles, and memory telemetry.

Every open slide used to keep its own caches (256 MB of decoded tiles per slide, plus 512 MB of decoded
subblocks per CZI file). With 16 slides on screen that can reach ~6 GB, and an out-of-memory kill in a
container. All slides now share one bytes-bounded LRU (default 1 GiB). Busy slides get more of it, idle
slides age out, and closing a slide releases its entries straight away.

    import wissi_viz
    wissi_viz.set_decoded_cache_mb(2048)        # or SlideRegistry(decoded_cache_mb=2048)

Object-overlay tiles (pyramid tiles, outline buckets) share a second budget (default 128 MiB) instead of
per-layer caches capped by entry count — those grew with every slide opened.
"""
from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from typing import Any, Callable

import numpy as np


class ByteLRU:
    """Thread-safe LRU bounded by total bytes (numpy arrays by default; ``sizeof=len`` for bytes). Keys are tuples
    whose first item is an owner token."""

    def __init__(self, max_bytes: int, sizeof: Callable[[Any], int] | None = None) -> None:
        self.max_bytes = int(max_bytes)
        self._sizeof = sizeof or (lambda a: int(getattr(a, 'nbytes', 0)))
        self._d: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0

    def get(self, key: tuple) -> Any:
        with self._lock:
            a = self._d.get(key)
            if a is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return a

    def put(self, key: tuple, a: np.ndarray) -> None:
        n = self._sizeof(a)
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._size -= self._sizeof(old)
            self._d[key] = a
            self._size += n
            while self._size > self.max_bytes and len(self._d) > 1:
                _, o = self._d.popitem(last=False)
                self._size -= self._sizeof(o)
                self.evictions += 1

    def purge(self, owner: object) -> int:
        """Drop every entry of ``owner`` (a closed slide)."""
        with self._lock:
            ks = [k for k in self._d if k[0] is owner]
            for k in ks:
                self._size -= self._sizeof(self._d.pop(k))
        return len(ks)

    def resize(self, max_bytes: int) -> None:
        with self._lock:
            self.max_bytes = int(max_bytes)
            while self._size > self.max_bytes and len(self._d) > 1:
                _, o = self._d.popitem(last=False)
                self._size -= self._sizeof(o)
                self.evictions += 1

    def stats(self) -> dict:
        with self._lock:
            return {'entries': len(self._d), 'bytes': self._size, 'max_bytes': self.max_bytes,
                    'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions}


_DECODED = ByteLRU(1024 * 2**20)


def decoded_cache() -> ByteLRU:
    """The shared cache of decoded arrays (virtual pyramid levels, CZI subblocks)."""
    return _DECODED


def set_decoded_cache_mb(mb: float) -> None:
    _DECODED.resize(int(mb * 2**20))


_OVERLAY = ByteLRU(128 * 2**20, sizeof=len)


def overlay_cache() -> ByteLRU:
    """The shared cache of encoded object-overlay tiles (pyramid tiles and outline buckets of every layer)."""
    return _OVERLAY


def set_overlay_cache_mb(mb: float) -> None:
    _OVERLAY.resize(int(mb * 2**20))


# ------------------------------------------------------------------------------------------- telemetry
def rss_bytes() -> int:
    """Resident memory of this process (Linux: current VmRSS; elsewhere: the peak)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == 'darwin' else r * 1024)


def rss_breakdown() -> dict:
    """Linux: resident memory split into anonymous (heap: what budgets control) and file-backed (memory-mapped
    layers and libraries — page cache the OS reclaims under pressure). Empty elsewhere."""
    out = {}
    try:
        with open('/proc/self/status') as f:
            for line in f:
                k = line.split(':')[0]
                if k in ('RssAnon', 'RssFile', 'RssShmem'):
                    out[k] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


_LIBC = None


def trim() -> bool:
    """Hand freed heap memory back to the OS (glibc ``malloc_trim``). Python + numpy free large arrays to the
    allocator, which often keeps the pages; after evicting big layers this returns them. No-op elsewhere."""
    global _LIBC
    if not sys.platform.startswith('linux'):
        return False
    try:
        if _LIBC is None:
            import ctypes
            _LIBC = ctypes.CDLL('libc.so.6')
        return bool(_LIBC.malloc_trim(0))
    except Exception:  # noqa: BLE001  (musl / no glibc)
        _LIBC = False
        return False


def _file_backed(a: np.ndarray) -> bool:
    while a is not None:
        if isinstance(a, np.memmap):
            return True
        a = a.base if isinstance(a.base, np.ndarray) else (a.base if isinstance(a.base, np.memmap) else None)
    return False


def anon_nbytes(obj: Any, _seen: set | None = None, _depth: int = 0) -> int:
    """Estimated anonymous (non file-backed) bytes held in numpy arrays / bytes / lists reachable from a wissi-viz
    object. Memory-mapped arrays are free here: their pages belong to the OS page cache and can be reclaimed."""
    seen = _seen if _seen is not None else set()
    if id(obj) in seen or _depth > 6:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        base = obj
        while isinstance(base.base, np.ndarray):
            base = base.base
        if base is not obj and id(base) in seen:
            return 0
        seen.add(id(base))
        return 0 if _file_backed(obj) else int(base.nbytes)
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, dict):
        return sum(anon_nbytes(v, seen, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        n = 8 * len(obj)
        if len(obj) > 2000:                     # e.g. per-feature lists: estimate from a sample
            k = len(obj) // 200
            return n + int(sum(anon_nbytes(v, seen, _depth + 1) for v in obj[::k]) * k)
        return n + sum(anon_nbytes(v, seen, _depth + 1) for v in obj)
    d = getattr(obj, '__dict__', None)
    if d is not None and type(obj).__module__.startswith('wissi_viz'):
        return anon_nbytes(d, seen, _depth + 1)
    return 0
