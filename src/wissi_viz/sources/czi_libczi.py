"""Zeiss CZI reader built on pylibCZIrw (libCZI) — **LGPL-3**, optional (``pip install wissi-viz[lgpl]``).

libCZI composes any region at any zoom, using the file's own pyramid subblocks when
present. We expose a power-of-two pyramid and keep one reader per worker thread.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from ..annotations import discover_sidecars, from_czi_metadata
from . import czi_meta
from .base import Channel, SlideInfo, SlideSource, _fit


class LibCziSource(SlideSource):
    extensions = ('.czi',)

    def __init__(
        self, path: str | os.PathLike, tile_size: int = 512, scene: int | None = None, z: int = 0, t: int = 0
    ) -> None:
        super().__init__()
        from pylibCZIrw import czi as pyczi

        self.path = Path(path)
        self.scene, self.z, self.t = scene, z, t
        self._local = threading.local()
        self._readers: list = []
        self._rlock = threading.Lock()
        st = self.path.stat()
        self.version = f'{int(st.st_mtime)}-{st.st_size}'

        with pyczi.open_czi(str(self.path)) as czi:
            if scene is not None:
                rect = czi.scenes_bounding_rectangle[scene]
            else:
                rect = czi.total_bounding_rectangle
            ptypes = czi.pixel_types
            meta = czi.metadata
            bbox = czi.total_bounding_box
        self._x0, self._y0 = rect.x, rect.y
        W, H = rect.w, rect.h
        nC = bbox.get('C', (0, 1))[1] - bbox.get('C', (0, 1))[0]
        self._c0 = bbox.get('C', (0, 1))[0]
        p0 = next(iter(ptypes.values()))
        self._rgb = p0 in ('Bgr24', 'Bgra32', 'Bgr48')
        self._bit16 = p0 in ('Gray16', 'Bgr48')
        names, colors = _channel_meta(meta, nC)
        if self._rgb:
            mode, dtype, channels = 'rgb', 'uint8', []
        else:
            mode = 'channels'
            dtype = 'uint8' if p0 == 'Gray8' else 'uint16'
            channels = [Channel(names[c], colors[c]) for c in range(nC)]
        self.info = SlideInfo(
            width=W,
            height=H,
            levels=SlideInfo.pyramid(W, H, tile_size),
            mode=mode,
            dtype=dtype,
            channels=channels,
            mpp=_mpp(meta),
            name=self.path.name,
            tile_size=tile_size,
            background='#ffffff' if self._rgb else '#000000',
        )
        self._meta = meta

    def _reader(self):
        r = getattr(self._local, 'r', None)
        if r is None:
            from pylibCZIrw import czi as pyczi

            r = pyczi.CziReader(str(self.path))
            with self._rlock:
                self._readers.append(r)
            self._local.r = r
        return r

    def read_region(self, level, x, y, w, h, channel=None):
        ds = self.info.levels[level].downsample
        roi = (
            self._x0 + int(round(x * ds)),
            self._y0 + int(round(y * ds)),
            max(1, int(round(w * ds))),
            max(1, int(round(h * ds))),
        )
        plane = {'C': (channel or 0) + self._c0, 'Z': self.z, 'T': self.t}
        bg = (255, 255, 255) if self._rgb else (0, 0, 0)
        a = self._reader().read(roi=roi, plane=plane, scene=self.scene, zoom=1.0 / ds, background_pixel=bg)
        if self._rgb:
            a = a[..., 2::-1]  # BGR → RGB
            if a.dtype != np.uint8:
                a = (a >> 8).astype(np.uint8)
            return _fit(np.ascontiguousarray(a), w, h)
        a = a[..., 0]
        if a.dtype == np.float32:
            a = np.clip(a, 0, 65535).astype(np.uint16)
        return _fit(a, w, h)

    def read_tile(self, level, tx, ty, channel=None):
        x, y, w, h = self.tile_rect(level, tx, ty)
        return self.read_region(level, x, y, w, h, channel)

    def annotation_layers(self):
        layers = discover_sidecars(self.path)
        emb = from_czi_metadata(self._meta, offset=(-self._x0, -self._y0))
        if emb is not None:
            layers.insert(0, emb)
        return layers

    def close(self):
        with self._rlock:
            for r in self._readers:
                try:
                    r.close()
                except Exception:  # noqa: BLE001  (best-effort close)
                    pass
            self._readers.clear()


def _mpp(meta):
    return czi_meta.mpp(meta)


def _channel_meta(meta, n):
    return czi_meta.channel_meta(meta, n)
