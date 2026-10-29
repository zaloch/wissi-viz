from __future__ import annotations

import os
import re
from pathlib import Path

from .base import Channel, Level, SlideInfo, SlideSource
from .callback import CallbackSource
from .tiff import DamagedFile

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
    if p.is_file():  # say plainly what is wrong with files no reader can take (rather than each reader's error)
        size = p.stat().st_size
        if size == 0:
            raise ValueError(f'cannot open {p.name}: the file is empty (0 bytes): an interrupted copy or download?')
        with open(p, 'rb') as f:
            head = f.read(16)
        kind = _not_a_slide(head)
        if kind:
            raise ValueError(f'cannot open {p.name}: it is {kind}, not a slide')

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
        except DamagedFile:  # the right reader, a broken file: say that, rather than what other readers make of it
            raise
        except Exception as e:  # noqa: BLE001  (collected and reported if no reader works)
            errors.append(f'{kind}: {_tidy(e)}')
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
    raise ValueError(f'cannot open {p.name}: no slide reader recognises it (' + '; '.join(errors) + ')')


def _only(kw: dict, *keys: str) -> dict:
    return {k: v for k, v in kw.items() if k in keys}


def _not_a_slide(head: bytes) -> str | None:
    """What a file is, from its first bytes, when that is plainly not a slide (else None)."""
    sigs = {
        b'\x89PNG': 'a PNG picture',
        b'\xff\xd8\xff': 'a JPEG picture',
        b'%PDF': 'a PDF document',
        b'PK\x03\x04': 'a ZIP archive (unpack it first)',
        b'\x1f\x8b': 'a gzip archive (unpack it first)',
        b'<?xml': 'an XML file (annotations? open it together with its slide)',
        b'{': 'a JSON file (annotations? open it together with its slide)',
        b'<!DOCTYPE html': 'a web page (a download that returned an error page?)',
        b'<html': 'a web page (a download that returned an error page?)',
    }
    for sig, what in sigs.items():
        if head.startswith(sig):
            return what
    return None


def _tidy(e: object) -> str:
    """A reader's error without its build paths (``/__w/…/tifftools.cpp:352:``) and with a sane length."""
    t = re.sub(r'\S*/[\w./-]+\.(?:cpp|cc|c|h|hpp):\d+:\s*', '', str(e)).strip()
    return t[:240] + ('…' if len(t) > 240 else '')
