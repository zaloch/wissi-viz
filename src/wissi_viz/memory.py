"""One process-wide byte budget for decoded pixel arrays.

Every open slide used to keep its own caches (256 MB of decoded tiles per slide, plus 512 MB of decoded
subblocks per CZI file). With 16 slides on screen that can reach ~6 GB, and an out-of-memory kill in a
container. All slides now share one bytes-bounded LRU (default 1 GiB). Busy slides get more of it, idle
slides age out, and closing a slide releases its entries straight away.

    import wissi_viz
    wissi_viz.set_decoded_cache_mb(2048)        # or SlideRegistry(decoded_cache_mb=2048)
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np


class ByteLRU:
    """Thread-safe LRU of numpy arrays bounded by total bytes. Keys are tuples whose first item is an owner token."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
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
        n = int(getattr(a, 'nbytes', 0))
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._size -= int(old.nbytes)
            self._d[key] = a
            self._size += n
            while self._size > self.max_bytes and len(self._d) > 1:
                _, o = self._d.popitem(last=False)
                self._size -= int(o.nbytes)
                self.evictions += 1

    def purge(self, owner: object) -> int:
        """Drop every entry of ``owner`` (a closed slide)."""
        with self._lock:
            ks = [k for k in self._d if k[0] is owner]
            for k in ks:
                self._size -= int(self._d.pop(k).nbytes)
        return len(ks)

    def resize(self, max_bytes: int) -> None:
        with self._lock:
            self.max_bytes = int(max_bytes)
            while self._size > self.max_bytes and len(self._d) > 1:
                _, o = self._d.popitem(last=False)
                self._size -= int(o.nbytes)
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
