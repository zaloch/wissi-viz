"""SVS / pyramidal TIFF / OME-TIFF reader built on tifffile + zarr.

Only the native tiles that intersect a requested region are decoded. Each worker
thread gets its own file handle so tile decoding runs in parallel across cores.
"""

from __future__ import annotations

import os
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
import zarr

from ..annotations import discover_sidecars
from .base import DEFAULT_CHANNEL_COLORS, Channel, Level, SlideInfo, SlideSource, _fit


class TiffSource(SlideSource):
    extensions = ('.svs', '.tif', '.tiff', '.ome.tif', '.ome.tiff', '.btf', '.qptiff')

    def __init__(self, path: str | os.PathLike, tile_size: int = 512, series: int = 0) -> None:
        super().__init__()

        self.path = Path(path)
        self.series_index = series
        self._local = threading.local()
        self._handles: list = []
        self._fast = True  # direct tile decode (bypasses zarr); disabled automatically on any surprise
        self._hlock = threading.Lock()
        st = self.path.stat()
        self.version = f'{int(st.st_mtime)}-{st.st_size}'

        with tifffile.TiffFile(self.path) as tf:
            s = tf.series[series]
            level_shapes = [lv.shape for lv in s.levels]
            axes = s.axes
            mpp = _mpp_from_tiff(tf)
            ome_channels = _ome_channels(tf) if tf.is_ome else []
            extra_levels = [] if len(level_shapes) > 1 else _loose_levels(tf, s)

        self._rgb = axes.endswith('S') and level_shapes[0][-1] in (3, 4)
        self._axes = axes.replace('S', 'C') if 'C' not in axes else axes
        self._extra_pages = extra_levels  # page indices used as levels when tifffile found none
        if extra_levels:
            level_shapes = level_shapes + [shape for _, shape in extra_levels]

        yi, xi = axes.index('Y'), axes.index('X')
        H, W = level_shapes[0][yi], level_shapes[0][xi]
        levels = [Level(int(sh[xi]), int(sh[yi]), W / sh[xi]) for sh in level_shapes]
        self._yi, self._xi = yi, xi
        self._ci = self._axes.index('C') if 'C' in self._axes else None
        self._dtype = None

        if self._rgb:
            mode, dtype, channels = 'rgb', 'uint8', []
        else:
            n = level_shapes[0][self._ci] if self._ci is not None else 1
            with tifffile.TiffFile(self.path) as tf:
                dt = tf.series[series].dtype
            mode = 'channels'
            dtype = 'uint8' if dt == np.uint8 else 'uint16'
            self._dtype = dt
            channels = []
            for c in range(n):
                name, color = ome_channels[c] if c < len(ome_channels) else (f'Channel {c + 1}', None)
                channels.append(Channel(name, color or DEFAULT_CHANNEL_COLORS[c % len(DEFAULT_CHANNEL_COLORS)]))
        self.info = SlideInfo(
            width=int(W),
            height=int(H),
            levels=levels,
            mode=mode,
            dtype=dtype,
            channels=channels,
            mpp=mpp,
            name=self.path.name,
            tile_size=tile_size,
            background='#ffffff' if mode == 'rgb' else '#000000',
        )

    # one zarr view per thread → parallel decoding
    def _arrays(self):
        arrs = getattr(self._local, 'arrs', None)
        if arrs is None:
            tf = tifffile.TiffFile(self.path)
            self._local.tf = tf
            with self._hlock:
                self._handles.append(tf)
            z = zarr.open(tf.series[self.series_index].aszarr(), mode='r')
            arrs = [z[str(i)] for i in range(len(z))] if hasattr(z, 'keys') else [z]
            for page_idx, _ in self._extra_pages:
                arrs.append(zarr.open(tf.pages[page_idx].aszarr(), mode='r'))
            self._local.arrs = arrs
        return arrs

    # ---------------------------------------------------------------- fast path
    def _page_for(self, level: int, channel: int | None):
        """The TIFF page backing (level, channel) in this thread, or None if the fast path doesn't apply."""
        cache = getattr(self._local, 'pages', None)
        if cache is None:
            cache = self._local.pages = {}
        key = (level, channel)
        if key in cache:
            return cache[key]
        page = None
        try:
            self._arrays()  # opens this thread's TiffFile
            tf = self._handles[-1] if not hasattr(self._local, 'tf') else self._local.tf
            series = tf.series[self.series_index]
            levels = list(series.levels)
            if level < len(levels):
                lv = levels[level]
                axes = lv.axes
                plane_axes = [a for a in axes if a not in 'YXS']
                shape = [lv.shape[axes.index(a)] for a in plane_axes]
                idx = [((channel or 0) if a == 'C' and not self._rgb else 0) for a in plane_axes]
                flat = int(np.ravel_multi_index(idx, shape)) if plane_axes else 0
                pages = lv.pages
                if flat < len(pages):
                    page = pages[flat]
                    page = (
                        getattr(page, 'keyframe', page)
                        if page is not None and not hasattr(page, 'dataoffsets')
                        else page
                    )
            if page is not None and not (
                page.is_tiled
                and (page.samplesperpixel == 1 or page.planarconfig == 1)
                and page.compression in (1, 5, 7, 8, 32946, 50000, 50001, 34887, 34712, 33003, 33005)
            ):
                page = None
        except Exception:  # noqa: BLE001  (unusual page: use the general path)
            page = None
        cache[key] = page
        return page

    def _read_fast(self, page, x, y, w, h) -> np.ndarray:
        tw, th = page.tilewidth, page.tilelength
        W, H = page.imagewidth, page.imagelength
        ntx = (W + tw - 1) // tw
        fh = page.parent.filehandle
        spp = page.samplesperpixel
        out = None
        for ty in range(y // th, min((y + h - 1) // th, (H - 1) // th) + 1):
            for tx in range(x // tw, min((x + w - 1) // tw, (W - 1) // tw) + 1):
                k = ty * ntx + tx
                off, cnt = page.dataoffsets[k], page.databytecounts[k]
                if cnt == 0:
                    continue
                fh.seek(off)
                a = page.decode(fh.read(cnt), k, jpegtables=page.jpegtables)[0]
                a = np.asarray(a).reshape(th, tw, -1) if a is not None else None
                if a is None:
                    continue
                if out is None:
                    out = np.zeros((h, w, a.shape[2]), a.dtype)
                ox, oy = tx * tw - x, ty * th - y
                sx0, sy0, dx0, dy0 = max(0, -ox), max(0, -oy), max(0, ox), max(0, oy)
                ww, hh = min(tw - sx0, w - dx0, W - tx * tw - sx0), min(th - sy0, h - dy0, H - ty * th - sy0)
                if ww > 0 and hh > 0:
                    out[dy0 : dy0 + hh, dx0 : dx0 + ww] = a[sy0 : sy0 + hh, sx0 : sx0 + ww]
        if out is None:
            out = np.zeros((h, w, spp), page.dtype)
        return out

    def read_region(self, level, x, y, w, h, channel=None):
        page = self._page_for(level, channel) if self._fast else None
        if page is not None:
            try:
                out = self._read_fast(page, x, y, w, h)
                if self._rgb:
                    return out[..., :3].astype(np.uint8, copy=False)
                out = out[..., 0]
                return out if out.dtype in (np.uint8, np.uint16) else np.clip(out, 0, 65535).astype(np.uint16)
            except Exception:  # noqa: BLE001
                self._fast = False  # anything unexpected → the general zarr path from now on
        return self._read_zarr(level, x, y, w, h, channel)

    def _read_zarr(self, level, x, y, w, h, channel=None):
        a = self._arrays()[level]
        sl = [slice(None)] * a.ndim
        sl[self._yi] = slice(y, y + h)
        sl[self._xi] = slice(x, x + w)
        for i, ax in enumerate(self._axes):
            if ax not in 'YXC':
                sl[i] = 0  # first Z/T plane
        if not self._rgb and self._ci is not None:
            sl[self._ci] = channel or 0
        out = np.asarray(a[tuple(sl)])
        if self._rgb:
            return out[..., :3].astype(np.uint8, copy=False)
        if out.dtype not in (np.uint8, np.uint16):
            out = np.clip(out, 0, 65535).astype(np.uint16)
        return out

    def read_tile(self, level, tx, ty, channel=None):
        # thread-local handles: no global lock needed
        x, y, w, h = self.tile_rect(level, tx, ty)
        return _fit(self.read_region(level, x, y, w, h, channel), w, h)

    def annotation_layers(self):
        return discover_sidecars(self.path)

    def close(self):
        with self._hlock:
            for tf in self._handles:
                tf.close()
            self._handles.clear()


def _mpp_from_tiff(tf) -> tuple[float, float] | None:
    try:
        if tf.is_svs:
            m = re.search(r'MPP\s*=\s*([0-9.]+)', tf.pages[0].description)
            if m:
                v = float(m.group(1))
                return (v, v)
        if tf.is_ome:
            root = ET.fromstring(tf.ome_metadata)
            px = root.find('.//{*}Pixels')
            if px is not None and px.get('PhysicalSizeX'):
                unit = px.get('PhysicalSizeXUnit', 'µm')
                f = {'nm': 1e-3, 'mm': 1e3, 'm': 1e6}.get(unit, 1.0)
                return (float(px.get('PhysicalSizeX')) * f, float(px.get('PhysicalSizeY', px.get('PhysicalSizeX'))) * f)
        p = tf.pages[0]
        philips = _mpp_philips(p.description or '')
        if philips:
            return philips
        tags = p.tags
        if 'XResolution' in tags and 'ResolutionUnit' in tags:
            num, den = tags['XResolution'].value
            unit = int(tags['ResolutionUnit'].value)
            if num and unit in (2, 3):
                ppu = num / den
                um = (25400.0 if unit == 2 else 10000.0) / ppu
                if 0.01 < um < 100:
                    return (um, um)
    except Exception:  # noqa: BLE001  (malformed resolution tags: no calibration)
        pass
    return None


def _mpp_philips(desc: str) -> tuple[float, float] | None:
    """Philips TIFF (e.g. CAMELYON16): an XML description whose ``DICOM_PIXEL_SPACING`` attributes give each
    level's pixel size in mm, as "row" "column" (y, x); the finest is the full-resolution level."""
    if 'DICOM_PIXEL_SPACING' not in desc:
        return None
    best = None
    for text in re.findall(r'Name="DICOM_PIXEL_SPACING"[^>]*>([^<]*)<', desc):
        v = [float(n) for n in re.findall(r'[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', text)]
        if len(v) >= 2 and v[0] > 0 and v[1] > 0 and (best is None or v[0] < best[0]):
            best = (v[0], v[1])
    if best is None:
        return None
    y_um, x_um = best[0] * 1000, best[1] * 1000
    return (x_um, y_um) if 0.01 < x_um < 100 else None


def _ome_channels(tf) -> list[tuple[str, str | None]]:
    out = []
    try:
        root = ET.fromstring(tf.ome_metadata)
        for ch in root.iter():
            if ch.tag.endswith('}Channel'):
                color = ch.get('Color')
                hexc = None
                if color is not None:
                    v = int(color) & 0xFFFFFFFF  # OME: signed RGBA int
                    hexc = f'#{(v >> 24) & 255:02x}{(v >> 16) & 255:02x}{(v >> 8) & 255:02x}'
                out.append((ch.get('Name') or ch.get('ID', ''), hexc))
    except Exception:  # noqa: BLE001  (malformed OME-XML: default channel names)
        pass
    return out


def _loose_levels(tf, s) -> list[tuple[int, tuple]]:
    """Pyramids written as plain consecutive pages (no SubIFDs / no format-specific layout)."""
    base = s.shape
    out = []
    for i, page in enumerate(tf.pages):
        if i == 0 or not getattr(page, 'is_tiled', False):
            continue
        sh = page.shape
        if len(sh) != len(base) or sh[-1:] != base[-1:] and s.axes.endswith('S'):
            continue
        r1, r2 = base[0] / sh[0], base[1] / sh[1]
        if r1 > 1.5 and abs(r1 - r2) / r1 < 0.02:
            out.append((i, sh))
    out.sort(key=lambda t: -t[1][0])
    return out
