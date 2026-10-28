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

import ctypes
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

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
            return {
                'entries': len(self._d),
                'bytes': self._size,
                'max_bytes': self.max_bytes,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
            }


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
    """Current resident memory of this process, on Linux, macOS and Windows (psutil when installed)."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001  (not installed, or access denied: fall through)
        pass
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/self/status', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('VmRSS'):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    elif sys.platform == 'win32':
        try:
            return _win_working_set()
        except Exception:  # noqa: BLE001  (fall through to the next probe)
            pass
    elif sys.platform == 'darwin':
        try:
            out = subprocess.run(
                ['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True, timeout=5
            )
            return int(out.stdout.strip()) * 1024
        except Exception:  # noqa: BLE001  (fall through to the next probe)
            pass
    try:
        import resource  # Unix fallback: the peak, not the current value

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == 'darwin' else r * 1024)
    except Exception:  # noqa: BLE001
        return 0


def _win_working_set() -> int:

    class PMC(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError(ctypes.get_last_error())
    return int(pmc.WorkingSetSize)


def rss_breakdown() -> dict:
    """Linux: resident memory split into anonymous (heap: what budgets control) and file-backed (memory-mapped
    layers and libraries — page cache the OS reclaims under pressure). Empty elsewhere."""
    out = {}
    try:
        with open('/proc/self/status', encoding='utf-8') as f:
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
        if len(obj) > 2000:  # e.g. per-feature lists: estimate from a sample
            k = len(obj) // 200
            return n + int(sum(anon_nbytes(v, seen, _depth + 1) for v in obj[::k]) * k)
        return n + sum(anon_nbytes(v, seen, _depth + 1) for v in obj)
    d = getattr(obj, '__dict__', None)
    if d is not None and type(obj).__module__.startswith('wissi_viz'):
        return anon_nbytes(d, seen, _depth + 1)
    return 0
