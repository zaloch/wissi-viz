"""Native NiceGUI element.

    from nicegui import ui
    from wissi_viz.nicegui import WissiViz

    grid = WissiViz(['/data/a.svs', '/data/b.czi'], layout='1x2').classes('w-full h-[80vh]')
    grid.go_to(0, 12000, 8000, magnification=20)

The tile/annotation routes are mounted on NiceGUI's FastAPI app automatically
(under ``/wissi-viz`` by default). Everything runs in the NiceGUI process — no extra
server or port, which is what you want inside a Docker container.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from nicegui import app
from nicegui.element import Element
from nicegui.events import GenericEventArguments

from .annotations import AnnotationLayer
from .registry import SlideRegistry
from .server import create_router
from .sources import SlideSource
from .tiles import TileService

_default: SlideRegistry | None = None
_mounted: dict[str, SlideRegistry] = {}

SlideRef = 'str | os.PathLike | SlideSource | dict'


def configure(*, cache_dir: str | os.PathLike | None = None, memory_mb: int = 1024, disk_gb: float = 20,
              workers: int | None = None, registry: SlideRegistry | None = None, prefix: str = '/wissi-viz',
              allow_edit: bool = True, dependencies: list | None = None) -> SlideRegistry:
    """Set up the shared registry (call once at startup, before creating viewers).

    ``cache_dir`` enables the persistent tile cache — mount it as a Docker volume so it
    survives container restarts. ``dependencies``: FastAPI dependencies applied to every
    tile/annotation route, e.g. ``[Depends(require_login)]`` to protect slides with your app's auth.
    """
    global _default
    if registry is None:
        registry = SlideRegistry(TileService(memory_bytes=memory_mb * 2**20, disk_dir=cache_dir,
                                             disk_bytes=int(disk_gb * 2**30), workers=workers))
    _default = registry
    mount(registry, prefix, allow_edit=allow_edit, dependencies=dependencies)
    return registry


def default_registry() -> SlideRegistry:
    global _default
    if _default is None:
        _default = SlideRegistry()
    return _default


def mount(registry: SlideRegistry, prefix: str = '/wissi-viz', allow_edit: bool = True,
          dependencies: list | None = None) -> None:
    if prefix in _mounted:
        if _mounted[prefix] is not registry:
            raise RuntimeError(f'another registry is already mounted at {prefix}')
        return
    app.include_router(create_router(registry, cors=None, allow_edit=allow_edit), prefix=prefix,
                       dependencies=list(dependencies or []))
    _mounted[prefix] = registry


class WissiViz(Element, component='nicegui_wsi.js', esm={'wissi-viz': 'static'},
              default_style='width:100%;height:600px;'):
    """1–16 whole-slide images in one GPU canvas.

    ``slides``: file paths, :class:`SlideSource` objects (e.g. PathCore/HALOLink adapters),
    registered slide ids, or dicts ``{'slide': ref, 'view': {...}, 'channels': [...], 'layers': {...}}``.
    """

    def __init__(self, slides: Sequence[Any] = (), *, layout: str = 'auto', sync: str = 'none',
                 controls: bool = True, legend: bool = True, tool: str = 'pan', draw_layer: str | None = None, quality: float = 1.0,
                 texture_budget_mb: int | None = None, lod: str | dict | None = None, registry: SlideRegistry | None = None,
                 prefix: str = '/wissi-viz',
                 on_view_change: Callable | None = None, on_click: Callable | None = None,
                 on_object_click: Callable | None = None, on_draw: Callable | None = None,
                 on_slide_load: Callable | None = None) -> None:
        super().__init__()
        self.registry = registry or _mounted.get(prefix) or default_registry()
        mount(self.registry, prefix)
        self.prefix = prefix
        #: latest view per pane (updated from the browser, throttled)
        self.views: dict[int, dict] = {}
        self._groups: dict[str, list[str]] = {}
        self.active_pane = 0
        self._props['api'] = prefix
        self._props['slides'] = [self._ref(s) for s in slides]
        self._props['layout'] = layout
        self._props['sync'] = sync
        self._props['controls'] = controls
        self._props['legend'] = legend
        self._props['tool'] = tool
        self._props['drawLayer'] = draw_layer
        self._props['quality'] = quality
        if texture_budget_mb:
            self._props['textureBudgetMb'] = texture_budget_mb
        if lod is not None:
            self._props['lod'] = lod

        def _view(e: GenericEventArguments) -> None:
            if isinstance(e.args, dict) and 'pane' in e.args:
                self.views[e.args['pane']] = e.args

        def _active(e: GenericEventArguments) -> None:
            self.active_pane = (e.args or {}).get('pane', 0)

        self.on('viewchange', _view, throttle=0.1, leading_events=False)
        self.on('activepane', _active)
        if on_view_change:
            self.on('viewchange', lambda e: on_view_change(e.args), throttle=0.2, leading_events=False)
        if on_click:
            self.on('paneclick', lambda e: on_click(e.args))
        if on_object_click:
            self.on('objectclick', lambda e: on_object_click(e.args))
        if on_draw:
            self.on('draw', lambda e: on_draw(e.args))
        if on_slide_load:
            self.on('slideload', lambda e: on_slide_load(e.args))

        unsubscribe = self.registry.subscribe(self._registry_changed)
        self._unsubscribe = unsubscribe

    # ------------------------------------------------------------ slides & layout
    def _ref(self, s: Any) -> Any:
        if isinstance(s, dict):
            d = dict(s)
            ref = d.pop('slide', None) or d.pop('source', None) or d.pop('path', None)
            d['slide'] = self._ref(ref)
            return d
        if isinstance(s, SlideSource):
            for sid in self.registry.ids():
                if self.registry._entries[sid].source is s:
                    return sid
            return self.registry.add(s)
        if isinstance(s, (os.PathLike,)) or (isinstance(s, str) and s not in self.registry and os.path.exists(s)):
            return self.registry.add(s)
        if isinstance(s, str):
            return s
        if s is None:
            return None
        raise TypeError(f'unsupported slide reference: {s!r}')

    @property
    def slides(self) -> list:
        return list(self._props['slides'])

    def set_slides(self, slides: Sequence[Any]) -> 'WissiViz':
        """Replace all panes (1–16)."""
        self._props['slides'] = [self._ref(s) for s in list(slides)[:16]]
        self.update()
        return self

    def set_slide(self, pane: int, slide: Any) -> 'WissiViz':
        lst = list(self._props['slides'])
        while len(lst) <= pane:
            lst.append(None)
        lst[pane] = self._ref(slide)
        self._props['slides'] = lst
        self.update()
        return self

    def set_layout(self, layout: str) -> 'WissiViz':
        """'auto' or 'ROWSxCOLS', e.g. '4x4', '3x4', '1x2'."""
        self._props['layout'] = layout
        self.update()
        return self

    def set_sync(self, mode: str) -> 'WissiViz':
        """'none' | 'all' | 'pan' | 'zoom' — linked navigation across panes."""
        self._props['sync'] = mode
        self.update()
        return self

    def set_tool(self, tool: str, draw_layer: str | None = None) -> 'WissiViz':
        """'pan' | 'point' | 'rect' | 'ellipse' | 'polygon' | 'freehand' | 'line'."""
        self._props['tool'] = tool
        if draw_layer is not None:
            self._props['drawLayer'] = draw_layer
        self.update()
        return self

    def set_lod(self, preset: str | None = None, **opts: Any) -> 'WissiViz':
        """Level-of-detail policy. ``set_lod('legacy')`` / ``set_lod('default')`` or individual options:
        ``motion_bias`` (texels per device px allowed while moving, default 2), ``minify`` (keep loaded finer tiles
        down to this many device px per texel, 0.5), ``quadtree`` (draw the finest loaded tile per region, True),
        ``background`` (full-res tiles use spare request slots while moving, True), ``prefetch`` (fetch the next
        finer level around the centre at rest, False), ``decoupled`` (start requests between frames, True)."""
        names = {'motion_bias': 'motionBias'}
        arg: Any = preset if preset else {names.get(k, k): v for k, v in opts.items()}
        self._props['lod'] = arg
        self._call('setLod', arg)
        return self

    # ------------------------------------------------------------ navigation
    def _call(self, name: str, *args: Any):
        return self.run_method('call', name, *args)

    def go_to(self, pane: int | str | list[int], x: float, y: float, *, units: str = 'px', zoom: float | None = None,
              magnification: float | None = None, mpp: float | None = None, fov: float | None = None,
              fov_um: float | None = None, animate: bool = True):
        """Centre ``pane`` (index, 'active', 'all' or a list) on (x, y) in level-0 px (or µm with ``units='um'``)."""
        opts: dict[str, Any] = {'units': units, 'animate': animate}
        for k, v in (('zoom', zoom), ('magnification', magnification), ('mpp', mpp), ('fov', fov), ('fovUm', fov_um)):
            if v is not None:
                opts[k] = v
        return self._call('goTo', pane, x, y, opts)

    def go_to_bounds(self, pane, x0: float, y0: float, x1: float, y1: float, padding: float = 0.1, animate: bool = True):
        return self._call('goToBounds', pane, x0, y0, x1, y1, {'padding': padding, 'animate': animate})

    def go_to_object(self, pane: int, slide: str, layer: str, index: int, padding: float = 1.0):
        """Zoom to an annotation object by its index in ``layer``."""
        L = self.registry.layer(slide, layer)
        bb = L.compiled().bb[index]
        return self.go_to_bounds(pane, *map(float, bb), padding=padding)

    def fit(self, pane: int | str = 'all', animate: bool = True):
        return self._call('fit', pane, animate)

    def set_active(self, pane: int):
        return self._call('setActive', pane)

    async def get_views(self) -> list[dict | None]:
        return await self._call('getViews')

    # ------------------------------------------------------------ display
    def set_channel(self, pane, index: int, *, visible: bool | None = None, color: str | None = None,
                    min: float | None = None, max: float | None = None, gamma: float | None = None):  # noqa: A002
        opts = {k: v for k, v in dict(visible=visible, color=color, min=min, max=max, gamma=gamma).items() if v is not None}
        return self._call('setChannel', pane, index, opts)

    def set_adjust(self, pane, *, brightness: float | None = None, contrast: float | None = None,
                   gamma: float | None = None, invert: bool | None = None):
        opts = {k: v for k, v in dict(brightness=brightness, contrast=contrast, gamma=gamma, invert=invert).items()
                if v is not None}
        return self._call('setAdjust', pane, opts)

    def set_stain(self, pane, mode: str = 'off', *, preset: str | None = None, vectors: dict | None = None,
                  scale: float | None = None, colors: list[str] | None = None):
        """Brightfield colour deconvolution on the GPU: ``mode`` = 'off' | 'hematoxylin' | 'eosin' | 'dab' |
        'residual' | 'fluorescence' (pseudo-IF composite). ``preset`` 'HE' / 'HDAB', or ``vectors`` from
        :func:`wissi_viz.spatial.estimate_stain_vectors` (Macenko) for this slide's staining."""
        opts = {k: v for k, v in dict(preset=preset, vectors=vectors, scale=scale, colors=colors).items() if v is not None}
        return self._call('setStain', pane, mode, opts)

    async def auto_stain(self, pane: int, mode: str = 'hematoxylin', slide: str | None = None) -> dict:
        """Estimate this slide's stain vectors (Macenko) and apply them to ``pane``."""
        from nicegui import run

        from .spatial import estimate_stain_vectors
        sid = slide or self._slide_of(pane)
        _, pyr = self.registry.get(sid)
        vec = await run.io_bound(estimate_stain_vectors, pyr.source)
        self.set_stain(pane, mode, vectors=vec)
        return vec

    def show(self, slide: str, item: Any, name: str | None = None, group: str | None = 'analysis',
             replace: bool = True, **layer_kw) -> AnnotationLayer:
        """Overlay an analysis result (anything with ``.layer()``), an ObjectTable, or a layer on ``slide``.
        Every viewer showing that slide updates. Layers shown with the same ``group`` replace each other
        in this viewer (previous ones are hidden, not deleted); ``replace=False`` joins the group without
        hiding the others (e.g. hotspots on top of a density map); ``group=None`` just adds."""
        if hasattr(item, 'to_layer') and not hasattr(item, 'layer'):
            layer = item.to_layer(name=name, **layer_kw)
        elif hasattr(item, 'layer') and callable(item.layer):
            layer = item.layer(**({'name': name} if name else {}), **layer_kw)
        else:
            layer = item
        if name:
            layer.name = name
        out = self.registry.add_annotations(self._ref(slide), layer)
        if group is not None:
            members = self._groups.setdefault(group, [])
            for other in (members if replace else []):
                if other != layer.name:
                    self.set_layer_visible('all', other, False)
            if layer.name in members:
                members.remove(layer.name)
            members.append(layer.name)
            self.set_layer_visible('all', layer.name, True)
        return out

    def _slide_of(self, pane: int) -> str:
        s = self._props['slides'][pane]
        return s.get('slide') if isinstance(s, dict) else s

    def set_layer_visible(self, pane, layer: str, visible: bool = True):
        return self._call('setLayerVisible', pane, layer, visible)

    def set_layer_style(self, pane, layer: str, **style: Any):
        """e.g. ``line_width=2, fill_opacity=0.3, opacity=0.8, point_size=6, detail_min_px=4``."""
        return self._call('setLayerStyle', pane, layer, style)

    def set_class_visible(self, pane, layer: str, cls: str, visible: bool = True):
        return self._call('setClassVisible', pane, layer, cls, visible)

    def select(self, pane: int, layer: str | None, index: int | None):
        return self._call('select', pane, layer, index)

    async def stats(self) -> dict:
        return await self._call('stats')

    # ------------------------------------------------------------ annotations
    def add_annotations(self, slide: str, layer: AnnotationLayer | str | os.PathLike, name: str | None = None) -> AnnotationLayer:
        """Attach a layer to a slide; every viewer showing that slide refreshes automatically."""
        return self.registry.add_annotations(self._ref(slide), layer, name=name)

    def _registry_changed(self, sid: str, layer: str | None) -> None:
        if self.is_deleted:
            return
        if any((s.get('slide') if isinstance(s, dict) else s) == sid for s in self._props['slides']):
            try:
                self._call('refreshAnnotations', sid)
            except Exception:
                pass

    def _handle_delete(self) -> None:
        self._unsubscribe()
        super()._handle_delete()


#: backwards-compatible alias
WsiGrid = WissiViz

__all__ = ['WissiViz', 'WsiGrid', 'configure', 'mount', 'default_registry']
