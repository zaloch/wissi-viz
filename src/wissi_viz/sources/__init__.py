from __future__ import annotations

import os
from pathlib import Path

from .base import Channel, Level, SlideInfo, SlideSource
from .callback import CallbackSource

__all__ = ['Channel', 'Level', 'SlideInfo', 'SlideSource', 'CallbackSource', 'open_slide']


SLIDEIO_ONLY = ('.vsi', '.zvi', '.dcm', '.afi')
OPENSLIDE_ONLY = ('.mrxs', '.vms', '.vmu', '.bif', '.svslide')


def open_slide(path: str | os.PathLike, *, backend: str = 'auto', **kwargs) -> SlideSource:
    """Open a slide file with the best available reader.

    Permissive backends (default order):
      * ``.czi`` → czifile (BSD-3) → SlideIO (BSD-3)
      * ``.svs`` / ``.tif(f)`` / OME-TIFF / ``.ndpi`` / ``.qptiff`` / ``.scn`` → tifffile (BSD-3) → SlideIO
      * ``.vsi`` / ``.zvi`` / ``.dcm`` / ``.afi`` → SlideIO          (``pip install wissi-viz[slideio]``)

    Opt-in LGPL backends (``pip install wissi-viz[lgpl]``), used only if installed and needed:
      * ``.mrxs`` / ``.vms`` / ``.bif`` → OpenSlide;  ``backend='libczi'`` → pylibCZIrw;
        ``backend='openslide'`` forces OpenSlide.

    ``backend``: 'auto' | 'tifffile' | 'czifile' | 'slideio' | 'openslide' | 'libczi'.
    """
    p = Path(path)
    name = p.name.lower()
    ext = p.suffix.lower()
    errors: list[str] = []

    def attempt(kind: str):
        try:
            if kind == 'czifile':
                from .czi import CziSource

                return CziSource(p, **_only(kwargs, 'tile_size', 'scene', 'z', 't', 'decode_cache_mb'))
            if kind == 'tifffile':
                from .tiff import TiffSource

                return TiffSource(p, **_only(kwargs, 'tile_size', 'series'))
            if kind == 'slideio':
                from .slideio_source import SlideioSource

                return SlideioSource(p, **_only(kwargs, 'tile_size', 'scene', 'z', 't', 'driver', 'rgb'))
            if kind == 'openslide':
                from .openslide_source import OpenSlideSource

                return OpenSlideSource(p, **_only(kwargs, 'tile_size'))
            if kind == 'libczi':
                from .czi_libczi import LibCziSource

                return LibCziSource(p, **_only(kwargs, 'tile_size', 'scene', 'z', 't'))
        except ImportError as e:
            hint = {'slideio': 'wissi-viz[slideio]', 'openslide': 'wissi-viz[lgpl]', 'libczi': 'wissi-viz[lgpl]'}
            errors.append(f'{kind}: not installed ({hint.get(kind, e)})')
        except Exception as e:  # noqa: BLE001  (collected and reported if no reader works)
            errors.append(f'{kind}: {e}')
        return None

    if backend != 'auto':
        order = [backend]
    elif ext == '.czi':
        order = ['czifile', 'slideio', 'libczi']
    elif name.endswith(SLIDEIO_ONLY):
        order = ['slideio']
    elif name.endswith(OPENSLIDE_ONLY):
        order = ['openslide']
    elif name.endswith(('.svs', '.tif', '.tiff', '.btf', '.qptiff', '.ndpi', '.scn')):
        order = ['tifffile', 'slideio', 'openslide']
    else:
        order = ['tifffile', 'slideio', 'openslide']
    for kind in order:
        src = attempt(kind)
        if src is not None:
            return src
    raise ValueError(f'cannot open {p.name}: ' + '; '.join(errors))


def _only(kw: dict, *keys: str) -> dict:
    return {k: v for k, v in kw.items() if k in keys}
