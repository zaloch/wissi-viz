"""wissi-viz — GPU-accelerated whole-slide image viewer for the browser and NiceGUI.

* Readers: SVS / pyramidal TIFF / OME-TIFF (tifffile), CZI (pylibCZIrw), anything OpenSlide reads,
  and API-backed slides (PathCore, HALOLink, …) via :class:`CallbackSource`.
* Annotations: GeoJSON, Aperio XML, HALO .annotations, CZI graphics, or built in Python.
* Front end: a dependency-free WebGL2 web component (``<wsi-grid>``) that shows 1–16 slides in one
  GPU context, and a NiceGUI element (:class:`wissi_viz.nicegui.WsiGrid`).
"""
from .annotations import AnnotationLayer
from .registry import SlideRegistry
from .server import create_app, create_router
from .sources import CallbackSource, Channel, Level, SlideInfo, SlideSource, open_slide
from .tiles import TileService

__version__ = '0.2.0'
__all__ = ['AnnotationLayer', 'SlideRegistry', 'create_app', 'create_router', 'CallbackSource', 'Channel', 'Level',
           'SlideInfo', 'SlideSource', 'open_slide', 'TileService']
