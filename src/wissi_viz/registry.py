"""Slide registry: maps public slide ids → sources and annotation layers.

The browser can only ever address slides by id; file paths are never accepted
from the client.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .annotations import AnnotationLayer
from .sources import SlideSource, open_slide
from .tiles import LAYER_DIR, PyramidSlide, TileService

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
    auto_layers: set = field(default_factory=set)   # loaded from the source / sidecars: reloadable, so evictable
    last_used: float = 0.0


class SlideRegistry:
    """Memory: every cache has a byte budget, so a long-running server stays flat however many slides it has
    served — encoded tiles (``TileService(memory_bytes)``), decoded arrays (``decoded_cache_mb``), overlay tiles
    (``overlay_cache_mb``) and annotation layers.

    Annotation / object layers that come with a slide (connector ``annotations()`` / ``objects()``, sidecar files)
    are *auto layers*: they can be fetched again, so they are evictable. With a disk cache (``TileService(disk_dir)``,
    or ``layer_cache_dir``) each large object layer is saved once in compiled form and memory-mapped — its pages
    belong to the OS page cache (shared by workers, reclaimable under pressure), and reopening a remote slide later
    costs no API calls. Auto layers held in RAM beyond ``annotation_memory_mb`` are dropped least-recently-used
    first and reload on demand. Layers your app adds (``add_annotations`` / ``show``) are never dropped.
    """

    def __init__(self, tiles: TileService | None = None, max_open: int = 64, warm: bool = True,
                 tile_size: int = 512, max_level_step: float = 4.0, decoded_cache_mb: float | None = None,
                 annotation_memory_mb: float | None = 1024, overlay_cache_mb: float | None = None,
                 layer_cache_dir: str | os.PathLike | None = None, layer_disk_gb: float = 20,
                 gzip_cache_mb: float = 256, layer_mapped_mb: float | None = 2048) -> None:
        self.tiles = tiles or TileService()
        if decoded_cache_mb is not None:  # process-wide budget shared by every slide (see wissi_viz.memory)
            from .memory import set_decoded_cache_mb
            set_decoded_cache_mb(decoded_cache_mb)
        if overlay_cache_mb is not None:
            from .memory import set_overlay_cache_mb
            set_overlay_cache_mb(overlay_cache_mb)
        self.annotation_budget = int(annotation_memory_mb * 2**20) if annotation_memory_mb else None
        # memory-mapped layer files: their touched pages count in this process's RSS (and a container's memory
        # use) while mapped; beyond this, least-recently-used slides are unmapped (reopening costs milliseconds)
        self.mapped_budget = int(layer_mapped_mb * 2**20) if layer_mapped_mb else None
        self._layer_mapped: dict[tuple[str, str], int] = {}
        if layer_cache_dir is None and getattr(self.tiles, 'disk', None) is not None:
            layer_cache_dir = self.tiles.disk.root / LAYER_DIR
        self.layer_cache_dir = Path(layer_cache_dir) if layer_cache_dir else None
        self.layer_disk_bytes = int(layer_disk_gb * 2**30)
        self.gzip_cache_mb = gzip_cache_mb      # the server's cache of gzipped overlay payloads
        self.layer_evictions = 0
        self.layer_disk_hits = 0
        self._layer_bytes: dict[tuple[str, str], int] = {}   # (slide, layer) → anonymous bytes, measured at load
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
        e.last_used = time.monotonic()
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
        e.last_used = time.monotonic()
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
            src = pyr.source
            layers = self._open_saved_layers(e, src)
            if layers is None:
                layers = list(src.annotation_layers())
                for L in layers:
                    L.compiled()  # bucket now, off the request path
                layers = self._save_layers(e, src, layers)
            from .memory import anon_nbytes
            for L in layers:
                if L.name not in e.layers:
                    e.layers[L.name] = L
                    e.auto_layers.add(L.name)
                    self._layer_bytes[(e.id, L.name)] = anon_nbytes(L)
                    self._layer_mapped[(e.id, L.name)] = getattr(L, '_mapped_bytes', 0)
        finally:
            e.file_layers_loaded = True
        self._enforce_layer_budget(keep=e)

    # ------------------------------------------------------------- annotation memory
    def _layer_dir(self, e: SlideEntry, src) -> Path | None:
        if self.layer_cache_dir is None:
            return None
        v = hashlib.blake2b(f'{getattr(src, "version", "0")}|{AnnotationLayer._SAVE_FORMAT}'.encode(), digest_size=6).hexdigest()
        from ._compat import safe_name
        return self.layer_cache_dir / safe_name(e.id, 40) / v     # short + valid on Windows (MAX_PATH, CON, …)

    @staticmethod
    def _reusable(src) -> bool:
        """Saved layers are reused across loads only for remote slides: their version comes from the API and
        changes with the image. (Local sidecar files can change without the slide changing — those are re-read.)"""
        from .sources.remote import RemoteSlideSource
        return isinstance(src, RemoteSlideSource)

    def _open_saved_layers(self, e: SlideEntry, src) -> list | None:
        d = self._layer_dir(e, src)
        if d is None or not self._reusable(src) or not (d / 'index.json').exists():
            return None
        try:
            out = []
            for it in json.loads((d / 'index.json').read_text(encoding='utf-8'))['layers']:
                if it['kind'] == 'bulk':
                    L = AnnotationLayer.open(d / it['dir'], version=it['version'])
                    L._mapped_bytes = sum(f.stat().st_size for f in (d / it['dir']).iterdir())
                    out.append(L)
                else:
                    L = AnnotationLayer.parse(json.loads((d / it['file']).read_text(encoding='utf-8')), fmt='geojson', name=it['name'])
                    for k, v in it.get('style', {}).items():
                        setattr(L, k, v)
                    L.classes.update(it.get('classes') or {})
                    out.append(L)
            os.utime(d)
            self.layer_disk_hits += 1
            return out
        except Exception:  # noqa: BLE001  (corrupt / partial cache: fetch again)
            return None

    def _save_layers(self, e: SlideEntry, src, layers: list) -> list:
        """Large object layers → compiled on disk + memory-mapped (anonymous RAM freed at once). Small layers stay
        in RAM; for remote slides they are written too (GeoJSON) so a later load needs no API call."""
        d = self._layer_dir(e, src)
        if d is None or not layers:
            return layers
        out, index = [], []
        try:
            d.mkdir(parents=True, exist_ok=True)
            for i, L in enumerate(layers):
                if L._block is not None and not len(L._kinds) and L.compiled().n >= 1000:
                    sub = L.save(d / f'L{i}-{os.urandom(3).hex()}')   # fresh folder: old ones may still be mapped
                    M = AnnotationLayer.open(sub, version=L.version)
                    M._mapped_bytes = sum(f.stat().st_size for f in sub.iterdir())
                    c = getattr(L, '_compiled', None)
                    if c is not None:
                        c.release()
                    index.append({'name': L.name, 'kind': 'bulk', 'dir': sub.name, 'version': L.version})
                    out.append(M)
                else:
                    if self._reusable(src):
                        (d / f'L{i}.geojson').write_text(json.dumps(L.to_geojson()), encoding='utf-8')
                        index.append({'name': L.name, 'kind': 'geojson', 'file': f'L{i}.geojson', 'classes': dict(L.classes),
                                      'style': {k: getattr(L, k) for k in ('line_width', 'fill_opacity', 'point_size', 'visible',
                                                                           'detail_min_px', 'z', 'overview', 'pyramid')}})
                    out.append(L)
            if self._reusable(src):
                (d / 'index.json').write_text(json.dumps({'source_version': str(getattr(src, 'version', '')), 'layers': index}), encoding='utf-8')
            # sweep earlier saves no index points to (on Windows a still-mapped one fails quietly; next save retries)
            live = {x.get('dir') for x in index}
            for old in d.glob('L*-*'):
                if old.is_dir() and old.name not in live:
                    shutil.rmtree(old, ignore_errors=True)
        except Exception:  # noqa: BLE001  (disk full / read-only: keep the layers in RAM)
            return layers if len(out) < len(layers) else out
        threading.Thread(target=self._trim_layer_disk, daemon=True).start()
        from .memory import trim
        trim()
        return out

    def _trim_layer_disk(self) -> None:
        root = self.layer_cache_dir
        if root is None or not root.exists():
            return
        dirs = []
        for sd in root.iterdir():
            for vd in (sd.iterdir() if sd.is_dir() else ()):
                size = sum(f.stat().st_size for f in vd.rglob('*') if f.is_file())
                dirs.append((vd.stat().st_mtime, size, vd))
        total = sum(sz for _, sz, _ in dirs)
        for _, sz, vd in sorted(dirs):
            if total <= self.layer_disk_bytes * 0.9:
                break
            shutil.rmtree(vd, ignore_errors=True)   # (open memory maps stay valid until released)
            total -= sz

    def _enforce_layer_budget(self, keep: SlideEntry | None = None) -> int:
        """Drop auto layers of least-recently-used slides while their RAM exceeds the budget. Returns layers dropped."""
        if not self.annotation_budget and not self.mapped_budget:
            return 0
        with self._lock:
            entries = list(self._entries.values())
        use = [(e, sum(self._layer_bytes.get((e.id, n), 0) for n in e.auto_layers),
                sum(self._layer_mapped.get((e.id, n), 0) for n in e.auto_layers)) for e in entries]
        anon, mapped = sum(u[1] for u in use), sum(u[2] for u in use)
        over = lambda: ((self.annotation_budget and anon > self.annotation_budget)  # noqa: E731
                        or (self.mapped_budget and mapped > self.mapped_budget))
        dropped = 0
        for e, a_, m_ in sorted(use, key=lambda t: t[0].last_used):
            if not over():
                break
            if e is keep or not (a_ or m_):
                continue
            dropped += self._evict_layers(e)
            anon -= a_
            mapped -= m_
        if dropped:
            from .memory import trim
            trim()
        return dropped

    def _evict_layers(self, e: SlideEntry) -> int:
        with e.open_lock:
            n = 0
            for name in list(e.auto_layers):
                L = e.layers.pop(name, None)
                self._layer_bytes.pop((e.id, name), None)
                self._layer_mapped.pop((e.id, name), None)
                c = getattr(L, '_compiled', None)
                if c is not None:
                    c.release()
                n += 1
            e.auto_layers.clear()
            e.file_layers_loaded = False
            e.file_layers_future = None
        self.layer_evictions += 1
        return n

    def memory_stats(self) -> dict:
        """Where this process's memory is: every cache, layers (auto vs app-owned, in RAM vs memory-mapped), RSS."""
        from .memory import anon_nbytes, decoded_cache, overlay_cache, rss_breakdown, rss_bytes
        with self._lock:
            entries = list(self._entries.values())
        auto_b = app_b = n_auto = n_app = 0
        for e in entries:
            for name, L in list(e.layers.items()):
                if name in e.auto_layers:
                    auto_b += self._layer_bytes.get((e.id, name), 0)
                    n_auto += 1
                else:
                    app_b += anon_nbytes(L)
                    n_app += 1
        mb = lambda b: round(b / 2**20, 1)  # noqa: E731
        t, dc, oc = self.tiles.memory.stats(), decoded_cache().stats(), overlay_cache().stats()
        rb = rss_breakdown()
        return {'rss_mb': mb(rss_bytes()), 'rss_anon_mb': mb(rb['RssAnon']) if 'RssAnon' in rb else None,
                'rss_file_mb': mb(rb['RssFile']) if 'RssFile' in rb else None,
                'tile_cache_mb': mb(t['bytes']), 'tile_cache_max_mb': mb(t['max_bytes']),
                'decoded_mb': mb(dc['bytes']), 'decoded_max_mb': mb(dc['max_bytes']),
                'overlay_mb': mb(oc['bytes']), 'overlay_max_mb': mb(oc['max_bytes']),
                'auto_layers': n_auto, 'auto_layers_ram_mb': mb(auto_b),
                'auto_layers_mapped_mb': mb(sum(self._layer_mapped.values())),
                'mapped_budget_mb': mb(self.mapped_budget) if self.mapped_budget else None,
                'annotation_budget_mb': mb(self.annotation_budget) if self.annotation_budget else None,
                'app_layers': n_app, 'app_layers_ram_mb': mb(app_b),
                'layer_evictions': self.layer_evictions, 'layer_disk_hits': self.layer_disk_hits,
                'layer_cache_dir': str(self.layer_cache_dir) if self.layer_cache_dir else None,
                'open_slides': len(self._open), 'slides': len(entries)}

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
