"""SlideIO reader (BSD-3; bundles OpenCV Apache-2.0, DCMTK BSD-3, OpenJPEG BSD-2) — permissive.

Adds formats the other permissive readers don't cover: Olympus VSI (+ .ets), Zeiss ZVI,
DICOM WSI, Hamamatsu NDPI, Leica SCN, AFI, QPTIFF/PHTIFF. Used automatically for those
extensions and as a fallback for SVS/TIFF/CZI. Note: SlideIO's CZI driver cannot decode
zstd-compressed CZI (ZEN 3.x default for some acquisitions); the czifile reader can.

    pip install wissi-viz[slideio]
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .base import DEFAULT_CHANNEL_COLORS, Channel, Level, SlideInfo, SlideSource, _fit

DRIVERS = {'.svs': 'SVS', '.czi': 'CZI', '.vsi': 'VSI', '.zvi': 'ZVI', '.dcm': 'DCM', '.ndpi': 'NDPI',
           '.scn': 'SCN', '.afi': 'AFI', '.qptiff': 'QPTIFF', '.tif': 'GDAL', '.tiff': 'GDAL'}


class SlideioSource(SlideSource):
    extensions = tuple(DRIVERS)

    def __init__(self, path: str | os.PathLike, tile_size: int = 512, scene: int = 0, driver: str | None = None,
                 z: int = 0, t: int = 0, rgb: bool | None = None) -> None:
        super().__init__()
        import slideio

        self.path = Path(path)
        p = self.path.name.lower()
        if driver is None:
            driver = 'OMETIFF' if p.endswith(('.ome.tif', '.ome.tiff')) else DRIVERS.get(self.path.suffix.lower(), 'AUTO')
        self.driver, self.scene_index, self.z, self.t = driver, scene, z, t
        self._local = threading.local()
        st = self.path.stat()
        self.version = f'{int(st.st_mtime)}-{st.st_size}'
        sc = self._scene()
        _, _, W, H = sc.rect
        n = sc.num_channels
        dtypes = [str(sc.get_channel_data_type(i)) for i in range(n)]
        names = [sc.get_channel_name(i) or '' for i in range(n)]
        if rgb is None:  # brightfield: 3 unnamed 8-bit channels
            rgb = n in (3, 4) and all(d == 'uint8' for d in dtypes) and not any(names)
        self._rgb = rgb
        levels = []
        for i in range(sc.num_zoom_levels):
            zl = sc.get_zoom_level_info(i)
            levels.append(Level(int(zl.size.width), int(zl.size.height), W / max(1, zl.size.width)))
        levels.sort(key=lambda lv: lv.downsample)
        res = sc.resolution
        mpp = (res[0] * 1e6, res[1] * 1e6) if res and res[0] > 0 else None
        if rgb:
            mode, dtype, channels = 'rgb', 'uint8', []
        else:
            mode = 'channels'
            dtype = 'uint8' if all(d == 'uint8' for d in dtypes) else 'uint16'
            channels = [Channel(names[i] or f'Channel {i + 1}', DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)])
                        for i in range(n)]
        self.info = SlideInfo(width=int(W), height=int(H), levels=levels or [Level(int(W), int(H), 1.0)], mode=mode,
                              dtype=dtype, channels=channels, mpp=mpp, name=self.path.name, tile_size=tile_size,
                              background='#ffffff' if rgb else '#000000')

    def _scene(self):
        sc = getattr(self._local, 'scene', None)
        if sc is None:
            import slideio
            slide = slideio.open_slide(str(self.path), self.driver)
            sc = slide.get_scene(self.scene_index)
            self._local.slide, self._local.scene = slide, sc  # keep the slide alive with its scene
        return sc

    def read_region(self, level, x, y, w, h, channel=None):
        ds = self.info.levels[level].downsample
        W, H = self.info.width, self.info.height
        X, Y = int(round(x * ds)), int(round(y * ds))
        RW, RH = min(int(round(w * ds)), W - X), min(int(round(h * ds)), H - Y)
        if RW <= 0 or RH <= 0:
            return self._blank(w, h)
        ow, oh = max(1, int(round(RW / ds))), max(1, int(round(RH / ds)))
        sc = self._scene()
        if self._rgb:
            a = sc.read_block((X, Y, RW, RH), size=(ow, oh), channel_indices=[0, 1, 2],
                              slices=(self.z, self.z + 1), frames=(self.t, self.t + 1))
            a = np.asarray(a)[..., :3].astype(np.uint8, copy=False)
        else:
            a = sc.read_block((X, Y, RW, RH), size=(ow, oh), channel_indices=[channel or 0],
                              slices=(self.z, self.z + 1), frames=(self.t, self.t + 1))
            a = np.asarray(a)
            a = a.reshape(a.shape[:2]) if a.ndim == 3 else a
            if a.dtype not in (np.uint8, np.uint16):
                a = np.clip(a, 0, 65535).astype(np.uint16)
            if self.info.dtype == 'uint16' and a.dtype == np.uint8:
                a = a.astype(np.uint16)
        out = self._blank(w, h)
        hh, ww = min(h, a.shape[0]), min(w, a.shape[1])
        out[:hh, :ww] = a[:hh, :ww]
        return out

    def _blank(self, w, h):
        if self._rgb:
            return np.full((h, w, 3), 255, np.uint8)
        return np.zeros((h, w), np.uint16 if self.info.dtype == 'uint16' else np.uint8)

    def read_tile(self, level, tx, ty, channel=None):
        x, y, w, h = self.tile_rect(level, tx, ty)
        return _fit(self.read_region(level, x, y, w, h, channel), w, h)

    def annotation_layers(self):
        from ..annotations import discover_sidecars
        return discover_sidecars(self.path)
