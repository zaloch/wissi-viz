"""OpenSlide reader (MRXS, NDPI, VMS, SCN, BIF, …). Optional: ``pip install wsiviz[openslide]``."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .base import Level, SlideInfo, SlideSource


class OpenSlideSource(SlideSource):
    extensions = ('.svs', '.ndpi', '.mrxs', '.vms', '.vmu', '.scn', '.bif', '.svslide', '.tif', '.tiff')

    def __init__(self, path: str | os.PathLike, tile_size: int = 512) -> None:
        super().__init__()
        import openslide

        self.path = Path(path)
        self._local = threading.local()
        self._all: list = []
        self._alock = threading.Lock()
        st = self.path.stat()
        self.version = f'{int(st.st_mtime)}-{st.st_size}'
        s = openslide.OpenSlide(str(self.path))
        self._all.append(s)
        self._local.s = s
        props = s.properties
        try:
            mpp = (float(props['openslide.mpp-x']), float(props['openslide.mpp-y']))
        except (KeyError, ValueError):
            mpp = None
        # respect the slide's bounds (e.g. MRXS) if present
        self._ox = int(props.get('openslide.bounds-x', 0))
        self._oy = int(props.get('openslide.bounds-y', 0))
        W, H = s.dimensions
        if 'openslide.bounds-width' in props:
            W, H = int(props['openslide.bounds-width']), int(props['openslide.bounds-height'])
        levels = [Level(int(round(W / ds)), int(round(H / ds)), float(ds)) for ds in s.level_downsamples]
        bg = props.get('openslide.background-color')
        self.info = SlideInfo(width=W, height=H, levels=levels, mode='rgb', mpp=mpp, name=self.path.name,
                              tile_size=tile_size, background=f'#{bg.lower()}' if bg else '#ffffff')

    def _slide(self):
        s = getattr(self._local, 's', None)
        if s is None:
            import openslide
            s = openslide.OpenSlide(str(self.path))
            with self._alock:
                self._all.append(s)
            self._local.s = s
        return s

    def read_region(self, level, x, y, w, h, channel=None):
        ds = self.info.levels[level].downsample
        img = self._slide().read_region((self._ox + int(x * ds), self._oy + int(y * ds)), level, (w, h))
        a = np.asarray(img)
        rgb, alpha = a[..., :3].astype(np.float32), a[..., 3:4].astype(np.float32) / 255.0
        bg = np.array([int(self.info.background[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
        return (rgb * alpha + bg * (1 - alpha)).astype(np.uint8) if alpha.min() < 1 else a[..., :3].copy()

    def read_tile(self, level, tx, ty, channel=None):
        x, y, w, h = self.tile_rect(level, tx, ty)
        return self.read_region(level, x, y, w, h, channel)

    def annotation_layers(self):
        from ..annotations import discover_sidecars
        return discover_sidecars(self.path)

    def close(self):
        with self._alock:
            for s in self._all:
                s.close()
            self._all.clear()
