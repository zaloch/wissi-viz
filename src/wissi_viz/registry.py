"""Slide registry: maps public slide ids → sources and annotation layers.

The browser can only ever address slides by id; file paths are never accepted
from the client.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .annotations import AnnotationLayer
from .sources import SlideSource, open_slide
from .tiles import PyramidSlide, TileService

SLIDE_EXTS = ('.svs', '.tif', '.tiff', '.ome.tif', '.ome.tiff', '.btf', '.czi', '.ndpi', '.mrxs', '.scn',
              '.vms', '.bif', '.qptiff', '.vsi', '.zvi', '.dcm', '.afi')
_ID_OK = re.compile(r'^[A-Za-z0-9_.\-~]{1,200}$')


def make_id(text: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9_.\-]+', '_', Path(text).name)[:60].strip('._') or 'slide'
    return f'{stem}-{hashlib.blake2b(text.encode(), digest_size=4).hexdigest()}'


@dataclass
class SlideEntry:
    id: str
    name: str
    path: Path | None = None
    factory: Callable[[], SlideSource] | None = None
    source: SlideSource | None = None
    pyramid: PyramidSlide | None = None
    layers: 'OrderedDict[str, AnnotationLayer]' = field(default_factory=OrderedDict)
    file_layers_loaded: bool = False
    file_layers_future: object = None
    windows: list | None = None
    closeable: bool = True  # sources passed in by the user are never auto-closed
    open_lock: threading.Lock = field(default_factory=threading.Lock)


class SlideRegistry:
    def __init__(self, tiles: TileService | None = None, max_open: int = 64, warm: bool = True,
                 tile_size: int = 512, max_level_step: float = 4.0, decoded_cache_mb: float | None = None) -> None:
        self.tiles = tiles or TileService()
        if decoded_cache_mb is not None:  # process-wide budget shared by every slide (see wissi_viz.memory)
            from .memory import set_decoded_cache_mb
            set_decoded_cache_mb(decoded_cache_mb)
        self.max_level_step = max_level_step  # PyramidSlide: insert virtual levels where native ones are further apart
        self.max_open = max_open
        self.warm = warm
        self.tile_size = tile_size
        self._entries: dict[str, SlideEntry] = {}
        self._open: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, str | None], None]] = []
        from concurrent.futures import ThreadPoolExecutor
        self._ann_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='wissi-viz-ann')

    # ------------------------------------------------------------- registration
    def add(self, src: str | os.PathLike | SlideSource | Callable[[], SlideSource], slide_id: str | None = None,
            name: str | None = None) -> str:
        """Register a slide file path, a ready SlideSource, or a zero-arg factory (opened lazily)."""
        with self._lock:
            if isinstance(src, SlideSource):
                sid = slide_id or make_id(src.info.name or f'source{len(self._entries)}')
                e = SlideEntry(sid, name or src.info.name or sid, source=src, closeable=False)
            elif callable(src):
                sid = slide_id or make_id(f'factory{len(self._entries)}')
                e = SlideEntry(sid, name or sid, factory=src)
            else:
                p = Path(src).resolve()
                if not p.exists():
                    raise FileNotFoundError(p)
                sid = slide_id or make_id(str(p))
                e = SlideEntry(sid, name or p.name, path=p)
            if not _ID_OK.match(sid):
                raise ValueError(f'invalid slide id {sid!r}')
            old = self._entries.get(sid)
            if old is not None:
                e.layers = old.layers
                self._close(old)
            self._entries[sid] = e
        self._notify(sid, None)
        return sid

    def add_directory(self, root: str | os.PathLike, recursive: bool = True) -> list[str]:
        root = Path(root)
        it = root.rglob('*') if recursive else root.glob('*')
        out = []
        for p in sorted(it):
            n = p.name.lower()
            if p.is_file() and n.endswith(SLIDE_EXTS):
                rel = p.relative_to(root).as_posix()
                out.append(self.add(p, slide_id=make_id(rel), name=rel))
        return out

    def remove(self, slide_id: str) -> None:
        with self._lock:
            e = self._entries.pop(slide_id, None)
            if e:
                self._close(e)
        self._notify(slide_id, None)

    def close(self) -> None:
        """Close every slide (releasing file handles and decoded caches) and stop the tile workers."""
        for sid in list(self._entries):
            self.remove(sid)
        self.tiles.close()
        self._ann_pool.shutdown(wait=False, cancel_futures=True)

    def ids(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, sid: str) -> bool:
        return sid in self._entries

    # ------------------------------------------------------------- annotations
    def add_annotations(self, slide_id: str, layer: AnnotationLayer | str | os.PathLike, name: str | None = None) -> AnnotationLayer:
        """Attach a layer (or load one from a .geojson / .annotations / .xml file). Replaces a layer of the same name."""
        if not isinstance(layer, AnnotationLayer):
            layer = AnnotationLayer.load(layer)
        if name:
            layer.name = name
        e = self._entry(slide_id)
        self._load_file_layers(e)
        e.layers[layer.name] = layer
        self._notify(slide_id, layer.name)
        return layer

    def remove_annotations(self, slide_id: str, name: str) -> None:
        e = self._entry(slide_id)
        e.layers.pop(name, None)
        self._notify(slide_id, name)

    def layers(self, slide_id: str) -> 'OrderedDict[str, AnnotationLayer]':
        e = self._entry(slide_id)
        self._load_file_layers(e)
        return e.layers

    def layer(self, slide_id: str, name: str) -> AnnotationLayer:
        layers = self.layers(slide_id)
        if name not in layers:
            raise KeyError(name)
        return layers[name]

    def changed(self, slide_id: str, layer: str | None = None) -> None:
        """Tell connected viewers that a slide or layer changed (after in-place edits)."""
        if layer is not None:
            self.layer(slide_id, layer).touch()
        self._notify(slide_id, layer)

    def subscribe(self, fn: Callable[[str, str | None], None]) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    def _notify(self, sid: str, layer: str | None) -> None:
        for fn in list(self._listeners):
            try:
                fn(sid, layer)
            except Exception:
                pass

    # ------------------------------------------------------------- access
    def _entry(self, sid: str) -> SlideEntry:
        e = self._entries.get(sid)
        if e is None:
            raise KeyError(sid)
        return e

    def get(self, sid: str) -> tuple[SlideEntry, PyramidSlide]:
        e = self._entry(sid)
        if e.pyramid is None:
            # per-slide lock: opening a remote slide (an API round-trip) never blocks other slides
            with e.open_lock:
                if e.pyramid is None:
                    src = e.source
                    if src is None:
                        src = e.factory() if e.factory is not None else open_slide(e.path, tile_size=self.tile_size)
                    pyr = PyramidSlide(src, max_step=self.max_level_step)
                    e.source, e.pyramid = src, pyr
                    if self.warm:
                        self.tiles.warm(sid, pyr, src.version)
        with self._lock:
            self._open[sid] = None
            self._open.move_to_end(sid)
            while len(self._open) > self.max_open:
                old, _ = self._open.popitem(last=False)
                oe = self._entries.get(old)
                if oe is not None and oe.closeable and oe is not e:
                    self._close(oe)
        return e, e.pyramid

    def _close(self, e: SlideEntry) -> None:
        if e.source is not None and e.closeable:
            try:
                e.source.close()
            except Exception:
                pass
            e.source = None
        if e.pyramid is not None:  # decoded arrays are ours to drop, whoever owns the source
            e.pyramid.release()
        e.pyramid = None

    def _load_file_layers(self, e: SlideEntry, block: bool = True) -> bool:
        """Embedded (CZI) and sidecar (.geojson/.annotations/.xml) layers, parsed once in the
        background so opening a slide never waits for a large annotation file. Returns True when loaded."""
        if e.file_layers_loaded:
            return True
        with self._lock:
            if e.file_layers_future is None:
                e.file_layers_future = self._ann_pool.submit(self._parse_file_layers, e)
            fut = e.file_layers_future
        if not block:
            return fut.done()
        try:
            fut.result()
        except Exception:
            pass
        return True

    def _parse_file_layers(self, e: SlideEntry) -> None:
        try:
            _, pyr = self.get(e.id)
            for L in pyr.source.annotation_layers():
                e.layers.setdefault(L.name, L)
                L.compiled()  # bucket now, off the request path
        finally:
            e.file_layers_loaded = True

    def _channel_windows(self, sid: str, e: SlideEntry, pyr: PyramidSlide) -> list[tuple[float, float]]:
        """Auto-contrast from the coarsest pyramid level, fetched *through the tile cache* so the
        same read also serves the first tiles the browser asks for (and is persisted on disk)."""
        import math
        import zlib

        import numpy as np
        info = pyr.info
        L = len(info.levels) - 1
        lv, ts = info.levels[L], info.tile_size
        nx, ny = math.ceil(lv.width / ts), math.ceil(lv.height / ts)
        dt = '<u2' if info.dtype == 'uint16' else 'u1'
        futs = {c: [self.tiles.get(sid, pyr, e.source.version, L, tx, ty, c) for ty in range(ny) for tx in range(nx)]
                for c in range(len(info.channels))}
        out = []
        for c, fs in futs.items():
            vals = np.concatenate([np.frombuffer(zlib.decompress(f.result()[0]), dt) for f in fs]).astype(np.float32)
            nz = vals[vals > 0]
            vals = nz if nz.size > 100 else vals
            lo, hi = np.percentile(vals, [0.5, 99.8]) if vals.size else (0.0, 1.0)
            out.append((float(lo), float(max(hi, lo + 1))))
        return out

    def info(self, sid: str) -> dict:
        e, pyr = self.get(sid)
        info = pyr.info
        d = info.to_json()
        d['id'] = sid
        d['name'] = e.name
        d['version'] = e.source.version
        # per level: virtual (synthesised by wissi-viz) and the remote level id (for browser-direct tile_url)
        src_ids = getattr(e.source.info, 'level_ids', None)
        for i, lv in enumerate(d['levels']):
            lv['virtual'] = bool(pyr.virtual[i])
            ni = pyr.native_index[i]
            lv['remote_level'] = (src_ids[ni] if src_ids and ni >= 0 else (None if ni < 0 else str(ni)))
        if info.mode == 'channels':
            if e.windows is None:
                try:
                    e.windows = self._channel_windows(sid, e, pyr)
                except Exception:
                    e.windows = [(0, 65535 if info.dtype == 'uint16' else 255)] * len(info.channels)
            for c, w in zip(d['channels'], e.windows):
                if c.get('window') is None:
                    c['window'] = list(w)
        loaded = self._load_file_layers(e, block=False)
        d['annotations'] = [L.meta() for L in list(e.layers.values())]
        d['annotations_pending'] = not loaded
        return d
