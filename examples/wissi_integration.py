"""Dropping wsiviz into wissi (NiceGUI in Docker) with PathCore / HALOLink slides.

Your existing, authenticated PathCore / HALOLink clients stay exactly as they are; you
only adapt two calls — "describe the pyramid" and "give me tile (level, col, row)".
Everything below the adapter (tile cache in memory + on disk, request coalescing,
GPU rendering, 1–16 panes, annotations) is shared with local SVS/CZI files.

Replace the ``halolink.*`` / ``pathcore.*`` calls with your client's methods.
"""
from __future__ import annotations

from nicegui import ui

from wsiviz import AnnotationLayer, CallbackSource, Level, SlideInfo
from wsiviz.nicegui import WsiGrid, configure

# Persistent tile cache → mount as a volume: docker run -v wsiviz-cache:/cache …
registry = configure(cache_dir='/cache/wsiviz', memory_mb=2048, disk_gb=100)


# ----------------------------------------------------------------------------- HALOLink
def halolink_source(halolink, image_id: str) -> CallbackSource:
    meta = halolink.image_metadata(image_id)               # ← your call
    # Mirror the service's own pyramid and tile size → 1 viewer tile == 1 remote tile,
    # JPEG bytes are passed straight through (no decode / re-encode).
    levels = [Level(width=lv['width'], height=lv['height'], downsample=lv['downsample']) for lv in meta['levels']]
    info = SlideInfo(width=levels[0].width, height=levels[0].height, levels=levels,
                     tile_size=meta.get('tile_size', 256), mpp=(meta['mpp_x'], meta['mpp_y']),
                     name=meta.get('name', image_id), mode='rgb')

    def tile(level: int, col: int, row: int, channel):
        return halolink.get_tile(image_id, level, col, row)  # ← returns JPEG/PNG bytes or a numpy array

    def annotations():
        # e.g. HALO annotation layers / object data → AnnotationLayer
        layers = []
        for ann in halolink.annotation_layers(image_id):     # ← your call
            L = AnnotationLayer(name=ann['name'])
            for region in ann['regions']:
                L.add_polygon(region['vertices'], cls=ann['name'], color=ann.get('color'))
            layers.append(L)
        objs = halolink.object_data(image_id)                 # ← cells: XMin/XMax/YMin/YMax/Class
        if objs is not None:
            layers.append(AnnotationLayer.from_boxes(objs['XMin'], objs['YMin'], objs['XMax'], objs['YMax'],
                                                     classes=objs['Classification'], name='HALO objects'))
        return layers

    return CallbackSource(info, read_tile=tile, annotations=annotations,
                          version=str(meta.get('modified', '0')))  # bump → caches invalidate


# ----------------------------------------------------------------------------- PathCore
def pathcore_source(pathcore, slide_id: str) -> CallbackSource:
    m = pathcore.slide_info(slide_id)                         # ← your call
    info = SlideInfo(width=m['width'], height=m['height'], tile_size=m['tile_size'],
                     levels=[Level(w, h, ds) for (w, h, ds) in m['levels']],
                     mpp=(m['mpp'], m['mpp']), name=m['name'])
    return CallbackSource(info, read_tile=lambda L, x, y, c: pathcore.tile(slide_id, L, x, y),
                          version=m.get('etag', '0'))


# ----------------------------------------------------------------------------- page
@ui.page('/viewer')
def viewer_page(case: str = ''):
    # Mix sources freely: local files, HALOLink, PathCore — up to 16 panes.
    slides = [
        '/data/slides/case12_HE.svs',
        '/data/slides/case12_IF.czi',
        # registry.add(halolink_source(halolink_client, 'img-123'), slide_id='halo-img-123'),
        # registry.add(pathcore_source(pathcore_client, 'S-998'), slide_id='pc-S-998'),
    ]
    grid = WsiGrid(slides, layout='auto', sync='none').classes('w-full').style('height: 85vh')

    # Navigate pane 0 to coordinates (level-0 px or µm), at a magnification / FOV:
    ui.button('Pane 0 → hotspot', on_click=lambda: grid.go_to(0, 15200, 9800, magnification=20))
    ui.button('All panes → 1.2 mm, 0.8 mm', on_click=lambda: grid.go_to('all', 1200, 800, units='um', fov_um=500))
    ui.button('4×4', on_click=lambda: grid.set_layout('4x4'))

    # Push annotations from your analysis results at any time; viewers refresh automatically:
    def add_rois():
        L = AnnotationLayer(name='Model ROIs', classes={'tumor': '#ff1744', 'stroma': '#00e676'}, line_width=2)
        L.add_polygon([[1000, 1000], [5000, 1200], [4800, 4000], [900, 3800]], cls='tumor', props={'score': 0.93})
        grid.add_annotations(grid.slides[0], L)
    ui.button('Add model ROIs', on_click=add_rois)
