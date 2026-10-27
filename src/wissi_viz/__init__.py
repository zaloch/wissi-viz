"""wissi-viz — GPU-accelerated whole-slide image viewer and spatial-analysis toolkit for
brightfield and fluorescence histology, for the browser and NiceGUI.

* Readers (permissive): SVS / pyramidal TIFF / OME-TIFF / NDPI (tifffile), CZI (czifile), VSI / ZVI /
  DICOM / SCN / AFI (SlideIO, optional), PathCore / HALO Link via connector stubs. Opt-in LGPL extras:
  OpenSlide (MRXS, BIF) and libCZI.
* Objects & annotations: QuPath, HALO, CellProfiler, GeoJSON, Aperio/HALO XML, label masks, AnnData.
* Analysis (:mod:`wissi_viz.spatial`): nearest neighbours, proximity, margins, composition, density /
  hotspots, Ripley, co-location, cellular neighbourhoods, contact graphs, Moran's I, IHC scoring,
  tissue detection, stain vectors — each with a slide overlay and an ECharts plot.
* Front end: a dependency-free WebGL2 web component (``<wissi-viz>``) showing 1–16 slides in one GPU
  context, and a NiceGUI element (:class:`wissi_viz.nicegui.WissiViz`).
"""
from .annotations import AnnotationLayer
from .objects import ObjectTable, cached_object_layer, read_objects
from .registry import SlideRegistry
from .server import create_app, create_router
from .sources import CallbackSource, Channel, Level, SlideInfo, SlideSource, open_slide
from .memory import decoded_cache, set_decoded_cache_mb
from .tiles import TileService

__version__ = '0.7.0'
__all__ = ['decoded_cache', 'set_decoded_cache_mb', 'AnnotationLayer', 'ObjectTable', 'read_objects', 'cached_object_layer', 'SlideRegistry', 'create_app', 'create_router',
           'CallbackSource', 'Channel', 'Level', 'SlideInfo', 'SlideSource', 'open_slide', 'TileService']
