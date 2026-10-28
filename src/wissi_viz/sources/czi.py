"""Zeiss CZI reader on czifile + imagecodecs (both BSD-3) — the permissive default.

Region reads decode only the subblocks that intersect the request, from the CZI's
own pyramid subblocks where present (each distinct ``shape / stored_shape`` scale
becomes a native level; wissi-viz synthesises any missing levels). Mosaic tiles are
composited in M-index order like ZEN/libCZI. Decoded subblocks are kept in an LRU,
so the many viewer tiles that fall inside one large subblock decode it once.

The LGPL libCZI backend (``pylibCZIrw``) is optional: ``open_slide(p, czi_backend='libczi')``.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import czifile
import imagecodecs
import numpy as np
import tifffile

from ..annotations import discover_sidecars, from_czi_metadata
from ..memory import ByteLRU, decoded_cache
from . import czi_meta
from .base import Channel, Level, SlideInfo, SlideSource

RGB_TYPES = {'BGR24', 'BGR48', 'BGRA32', 'BGR96FLOAT'}


@dataclass
class _Block:
    index: int  # position in czifile's subblock_directory
    x: int  # level-0 (logical) placement, relative to the image origin
    y: int
    w: int  # logical size (level-0 px)
    h: int
    sw: int  # stored size (px at this block's scale)
    sh: int
    m: int  # mosaic index (draw order)


class _DecodeCache:
    """Decoded subblocks (bytes-bounded LRU) with in-flight de-duplication across threads."""

    def __init__(self, max_bytes: int | None = None) -> None:
        # storage: the process-wide decoded-array budget (shared with every other slide), or a private one
        self.store = ByteLRU(max_bytes) if max_bytes else decoded_cache()
        self.owner = object()
        self.inflight: dict[int, Future] = {}
        self.lock = threading.Lock()

    def get(self, key: int, make):
        a = self.store.get((self.owner, key))
        if a is not None:
            return a
        with self.lock:
            a = self.store.get((self.owner, key))
            if a is not None:
                return a
            f = self.inflight.get(key)
            owner = f is None
            if owner:
                f = self.inflight[key] = Future()
        if not owner:
            return f.result()
        try:
            a = make()
        except BaseException as e:
            with self.lock:
                self.inflight.pop(key, None)
            f.set_exception(e)
            raise
        self.store.put((self.owner, key), a)
        with self.lock:
            self.inflight.pop(key, None)
        f.set_result(a)
        return a

    def clear(self) -> None:
        self.store.purge(self.owner)


def _metadata_dict(f) -> dict:
    """Metadata as a dict on every czifile release: 2026.x has ``metadata(asdict=True)``, 2019.7.2.x (the newest on
    Python 3.10/3.11) only XML. Both return XML from ``metadata()``; parse it with tifffile's identical xml2dict."""
    xml = f.metadata()
    if isinstance(xml, dict):
        return xml
    if not xml:
        return {}
    return tifffile.xml2dict(xml)


def _dims(entry) -> tuple:
    """Subblock dimension names: ``entry.dims`` on czifile 2026.x; ``entry.axes`` (samples as '0') on 2019.7.2.x."""
    d = getattr(entry, 'dims', None)
    return tuple(d) if d is not None else tuple('S' if c == '0' else c for c in entry.axes)


class CziSource(SlideSource):
    extensions = ('.czi',)

    def __init__(
        self,
        path: str | os.PathLike,
        tile_size: int = 512,
        scene: int | None = None,
        z: int = 0,
        t: int = 0,
        decode_cache_mb: int | None = None,
    ) -> None:
        super().__init__()

        self.path = Path(path)
        self.scene, self.z, self.t = scene, z, t
        st = self.path.stat()
        self.version = f'{int(st.st_mtime)}-{st.st_size}'
        self._local = threading.local()
        self._files: list = []
        self._flock = threading.Lock()
        self._cache = _DecodeCache(decode_cache_mb * 2**20 if decode_cache_mb else None)

        f = self._file()
        entries = f.subblock_directory
        meta = _metadata_dict(f)
        dims = _dims(entries[0])
        self._dims = dims
        iy, ix = dims.index('Y'), dims.index('X')
        ic = dims.index('C') if 'C' in dims else None
        self._ic = ic

        def pos(e, d, default=0):
            return e.start[dims.index(d)] if d in dims else default

        def scene_of(e) -> int:  # czifile's scene_index (from the S dimension); -1: no scene
            si = getattr(e, 'scene_index', None)
            return int(si) if si is not None else int(pos(e, 'S', -1))

        # keep entries of the requested scene / Z / T (other non-spatial dims: their first value)
        other = [d for d in dims if d not in ('X', 'Y', 'C', 'S', 'Z', 'T')]
        first = {d: min(pos(e, d) for e in entries) for d in other}
        sel = []
        for i, e in enumerate(entries):
            if scene is not None and scene_of(e) not in (scene, -1):
                continue
            if pos(e, 'Z') != z or pos(e, 'T') != t:
                continue
            if any(pos(e, d) != first[d] for d in other):
                continue
            sel.append((i, e))
        if not sel:
            raise ValueError(f'{self.path.name}: no subblocks for scene={scene} z={z} t={t}')

        full = [(i, e) for i, e in sel if e.shape[ix] == e.stored_shape[ix]]
        ref = full or sel
        x0 = min(e.start[ix] for _, e in ref)
        y0 = min(e.start[iy] for _, e in ref)
        W = max(e.start[ix] + e.shape[ix] for _, e in ref) - x0
        H = max(e.start[iy] + e.shape[iy] for _, e in ref) - y0
        self._origin = (x0, y0)
        c_vals = sorted({pos(e, 'C') for _, e in sel}) if ic is not None else [0]
        self._c_vals = c_vals

        # group by (scale, channel); scale = logical / stored size
        groups: dict[tuple[float, int], list[_Block]] = {}
        for i, e in sel:
            s = e.shape[ix] / max(1, e.stored_shape[ix])
            s = round(s, 2) if abs(s - round(s)) > 0.02 else float(round(s))
            b = _Block(
                i,
                e.start[ix] - x0,
                e.start[iy] - y0,
                e.shape[ix],
                e.shape[iy],
                e.stored_shape[ix],
                e.stored_shape[iy],
                getattr(e, 'mosaic_index', 0) or 0,
            )
            groups.setdefault((s, pos(e, 'C')), []).append(b)
        scales = sorted({s for s, _ in groups})
        self._scales = scales
        self._blocks: dict[tuple[int, int], dict] = {}
        for (s, c), blocks in groups.items():
            blocks.sort(key=lambda b: (b.m, b.index))
            arr = np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in blocks], np.int64)
            self._blocks[(scales.index(s), c)] = {'blocks': blocks, 'bbox': arr}

        ptype = entries[sel[0][0]].pixel_type
        pname = (ptype.name if hasattr(ptype, 'name') else str(ptype)).upper()
        self._rgb = pname in RGB_TYPES
        levels = [Level(max(1, math.ceil(W / s)), max(1, math.ceil(H / s)), float(s)) for s in scales]
        if self._rgb:
            mode, dtype, channels = 'rgb', 'uint8', []
        else:
            mode = 'channels'
            dtype = 'uint8' if pname == 'GRAY8' else 'uint16'
            names, colors = czi_meta.channel_meta(meta, len(c_vals))
            channels = [Channel(names[k], colors[k]) for k in range(len(c_vals))]
        self.info = SlideInfo(
            width=W,
            height=H,
            levels=levels,
            mode=mode,
            dtype=dtype,
            channels=channels,
            mpp=czi_meta.mpp(meta),
            name=self.path.name,
            tile_size=tile_size,
            background='#ffffff' if self._rgb else '#000000',
        )
        # scenes (several tissue pieces / wells in one file): their bounds, for "zoom to scene" and per-scene contrast
        scenes: dict[int, list[int]] = {}
        for _, e in ref:
            k = scene_of(e)
            if k < 0:
                continue
            bx0, by0 = e.start[ix] - x0, e.start[iy] - y0
            bx1, by1 = bx0 + e.shape[ix], by0 + e.shape[iy]
            r = scenes.setdefault(k, [bx0, by0, bx1, by1])
            r[0], r[1], r[2], r[3] = min(r[0], bx0), min(r[1], by0), max(r[2], bx1), max(r[3], by1)
        if len(scenes) > 1:
            self.info.extra['scenes'] = [
                {'index': k, 'name': f'Scene {k + 1}', 'rect': scenes[k]} for k in sorted(scenes)
            ]
        self._meta = meta

    # ---------------------------------------------------------------- per-thread file handles
    def _file(self):
        f = getattr(self._local, 'f', None)
        if f is None:
            f = czifile.CziFile(str(self.path))
            with self._flock:
                self._files.append(f)
            self._local.f = f
        return f

    def _decode(self, b: _Block) -> np.ndarray:
        def make():
            f = self._file()
            entry = f.subblock_directory[b.index]
            seg = entry.read_segment_data(f) if hasattr(entry, 'read_segment_data') else entry.data_segment()
            a = _fast_decode(entry, seg, b)
            if a is None:  # anything unusual → czifile's own (slower) path
                a = np.asarray(seg.data())
            a = a.reshape(a.shape[-3:]) if a.ndim >= 3 else a  # (..., Y, X, S) → (Y, X, S)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]
            if self._rgb:
                a = a[..., 2::-1] if a.shape[-1] >= 3 else np.repeat(a[..., None], 3, axis=2)
                if a.dtype == np.uint16:
                    a = (a >> 8).astype(np.uint8)
                elif a.dtype != np.uint8:
                    a = np.clip(a, 0, 255).astype(np.uint8)
            elif a.dtype not in (np.uint8, np.uint16):
                a = np.clip(a, 0, 65535).astype(np.uint16)
            return np.ascontiguousarray(a)

        return self._cache.get(b.index, make)

    # ---------------------------------------------------------------- reading
    def read_region(self, level, x, y, w, h, channel=None):
        s = self.info.levels[level].downsample
        c = self._c_vals[channel or 0]
        if self._rgb:
            out = np.full((h, w, 3), 255, np.uint8)
        else:
            out = np.zeros((h, w), np.uint16 if self.info.dtype == 'uint16' else np.uint8)
        g = self._blocks.get((level, c))
        if g is None:
            return out
        X0, Y0, X1, Y1 = x * s, y * s, (x + w) * s, (y + h) * s
        bb = g['bbox']
        hit = np.nonzero((bb[:, 0] < X1) & (bb[:, 2] > X0) & (bb[:, 1] < Y1) & (bb[:, 3] > Y0))[0]
        for k in hit:  # ascending M index → later blocks on top
            b = g['blocks'][k]
            a = self._decode(b)
            bx, by = int(round(b.x / s)), int(round(b.y / s))  # block origin in level px
            sx0, sy0 = max(0, x - bx), max(0, y - by)
            dx0, dy0 = max(0, bx - x), max(0, by - y)
            ww = min(a.shape[1] - sx0, w - dx0)
            hh = min(a.shape[0] - sy0, h - dy0)
            if ww > 0 and hh > 0:
                out[dy0 : dy0 + hh, dx0 : dx0 + ww] = a[sy0 : sy0 + hh, sx0 : sx0 + ww]
        return out

    def read_tile(self, level, tx, ty, channel=None):
        x, y, w, h = self.tile_rect(level, tx, ty)
        return self.read_region(level, x, y, w, h, channel)

    def annotation_layers(self):
        layers = discover_sidecars(self.path)
        emb = from_czi_metadata(self._meta, offset=(-self._origin[0], -self._origin[1]))
        if emb is not None:
            layers.insert(0, emb)
        return layers

    def close(self):
        self._cache.clear()
        with self._flock:
            for f in self._files:
                try:
                    f.close()
                except Exception:  # noqa: BLE001  (best-effort close)
                    pass
            self._files.clear()


_PIXEL = {  # pixel type → (numpy dtype, samples)
    'GRAY8': (np.uint8, 1),
    'GRAY16': (np.uint16, 1),
    'GRAY32FLOAT': (np.float32, 1),
    'BGR24': (np.uint8, 3),
    'BGR48': (np.uint16, 3),
    'BGRA32': (np.uint8, 4),
    'BGR96FLOAT': (np.float32, 3),
}


def _fast_decode(entry, seg, b: _Block) -> np.ndarray | None:
    """Decode a subblock straight from its raw bytes with imagecodecs (≈10–20× faster than the generic path)."""
    try:
        pt = entry.pixel_type
        dt, spp = _PIXEL.get((pt.name if hasattr(pt, 'name') else str(pt)).upper(), (None, None))
        if dt is None:
            return None
        comp = int(entry.compression)
        raw = seg.data(raw=True)
        n = b.sh * b.sw * spp
        if comp == 0:
            a = np.frombuffer(raw, dt, count=n)
        elif comp == 6:  # ZSTDHDR: zstd with optional hi/lo byte unpacking
            a = np.frombuffer(imagecodecs.zstd1_decode(bytes(raw), itemsize=np.dtype(dt).itemsize), dt, count=n)
        elif comp == 5:
            a = np.frombuffer(imagecodecs.zstd_decode(bytes(raw)), dt, count=n)
        elif comp == 1:
            return np.asarray(imagecodecs.jpeg8_decode(bytes(raw))).reshape(b.sh, b.sw, -1)
        elif comp == 4:
            return np.asarray(imagecodecs.jpegxr_decode(bytes(raw))).reshape(b.sh, b.sw, -1)
        elif comp == 2:
            a = np.frombuffer(imagecodecs.lzw_decode(bytes(raw)), dt, count=n)
        else:
            return None
        return a.reshape(b.sh, b.sw, spp)
    except Exception:  # noqa: BLE001  (fast path is optional: caller decodes normally)
        return None
