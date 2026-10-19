from __future__ import annotations

import os
from pathlib import Path

from .base import Channel, Level, SlideInfo, SlideSource
from .callback import CallbackSource

__all__ = ['Channel', 'Level', 'SlideInfo', 'SlideSource', 'CallbackSource', 'open_slide']


def open_slide(path: str | os.PathLike, **kwargs) -> SlideSource:
    """Open a slide file with the best available reader.

    * ``.czi`` → pylibCZIrw
    * ``.svs`` / ``.tif(f)`` / OME-TIFF → tifffile (parallel, no native deps), OpenSlide as fallback
    * ``.ndpi`` / ``.mrxs`` / ``.scn`` / ``.vms`` / ``.bif`` → OpenSlide
    """
    p = Path(path)
    name = p.name.lower()
    if name.endswith('.czi'):
        from .czi import CziSource
        return CziSource(p, **kwargs)
    errors = []
    if name.endswith(('.svs', '.tif', '.tiff', '.btf', '.qptiff')):
        try:
            from .tiff import TiffSource
            return TiffSource(p, **kwargs)
        except Exception as e:  # unusual TIFF flavours → OpenSlide
            errors.append(f'tifffile: {e}')
    try:
        from .openslide_source import OpenSlideSource
        return OpenSlideSource(p, **kwargs)
    except Exception as e:
        errors.append(f'openslide: {e}')
    raise ValueError(f'cannot open {p.name}: ' + '; '.join(errors))
