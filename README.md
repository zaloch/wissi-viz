# wissi-viz

wissi-viz is a GPU-accelerated whole-slide image viewer for the browser.

- **What it shows:** 1–16 slides at once, from SVS, CZI or OME-TIFF files, from anything OpenSlide reads, or from PathCore / HALO Link through connector stubs your app fills in.
- **Annotations:** ROIs, drawings and millions of detected objects, rendered on the GPU.
- **Where it runs:** as a plain HTML element (`<wsi-grid>`), or as a native NiceGUI element (`wissi_viz.nicegui.WsiGrid`) that you can drop straight into wissi.

```
 Browser                                                     Python (NiceGUI / FastAPI process)
┌────────────────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
│ <wsi-grid>  one WebGL2 context, 1–16 panes          │      │ /wissi-viz router (mounted on NiceGUI's app) │
│  ├ scheduler: coarse → centre-out → margin/predict  │ HTTP │  ├ memory LRU  (encoded tiles)            │
│  ├ worker pool: fetch + decode off main thread      │◀────▶│  ├ disk cache  (survives restarts)        │
│  ├ GPU texture LRU (shared by all panes, budgeted)  │      │  ├ in-flight coalescing                    │
│  ├ shaders: RGB adjust / 8-channel uint16 composite │      │  ├ worker pool (per-thread file handles)   │
│  └ annotations: bucket cull + LOD, instanced lines  │      │  └ PyramidSlide (fills pyramid gaps)       │
└────────────────────────────────────────────────────┘      │       └ SlideSource: SVS │ CZI │ OpenSlide │
                                                             │                     │ CallbackSource (APIs)│
                                                             └──────────────────────────────────────────┘
```

## Why Python + WebGL2, not Rust/WASM

The slides sit next to the server, in Docker, and they are multi-GB files.

- **Decoding belongs on the server.** Decoding has to happen where the bytes are: tifffile, libCZI and OpenSlide do it in native code, and the server's caches make each tile a one-time cost.
- **WASM would have to download the files.** A WASM decoder would need the whole file, or HTTP range reads, pulled into every browser.
- **CZI's codecs are hard in WASM.** CZI uses JPEG-XR and zstd subblocks, which are painful to port.
- **The GPU work is small.** It is texture upload, multi-channel compositing, contrast and gamma, and annotation geometry. WebGL2 shaders already do all of that, everywhere, including Safari and iPad.
- **No build step.** The front end is a single dependency-free ES module.
- **WebGPU later is a local change.** A WebGPU backend could replace the renderer without changing the protocol.

## Install

```bash
pip install -e ".[all]"      # czi + openslide + nicegui + uvicorn
# or minimal: pip install -e .   (SVS/TIFF only)  and add extras: [czi] [openslide] [nicegui] [server]
```

## Quick start (standalone server + demo page)

```bash
wissi-viz serve /data/slides --port 8010 --cache-dir /var/cache/wissi-viz
# open http://127.0.0.1:8010/
```

## Embed as an HTML element (any page, any framework)

```html
<script type="module" src="https://your-server/wissi-viz/static/index.js"></script>
<wsi-grid api="https://your-server/wissi-viz" slides="slideA,slideB,slideC,slideD" layout="auto"></wsi-grid>
<script type="module">
  const g = document.querySelector('wsi-grid');
  g.goTo(0, 15200, 9800, { magnification: 20 });            // level-0 px
  g.goTo('all', 1200, 800, { units: 'um', fovUm: 500 });    // microns, all panes
  g.setSync('all');                                         // linked navigation
  g.addEventListener('objectclick', (e) => console.log(e.detail));
</script>
```

## NiceGUI / wissi

```python
from nicegui import ui
from wissi_viz.nicegui import WsiGrid, configure

registry = configure(cache_dir='/cache/wissi-viz')        # once at startup

@ui.page('/viewer')
def page():
    grid = WsiGrid(['/data/a.svs', '/data/b.czi'], on_object_click=print).classes('w-full h-[85vh]')
    ui.button('hotspot', on_click=lambda: grid.go_to(0, 15200, 9800, magnification=20))
```

Routes mount on NiceGUI's own FastAPI app, so there is no extra port or process. The JavaScript is delivered through NiceGUI's ESM import map, and `window.path_prefix` (reverse-proxy subpaths) is respected. See `examples/nicegui_app.py` (a full UI) and `examples/wissi_integration.py` (Docker + HALOLink/PathCore adapters).

**Docker notes**
- Mount a volume for `cache_dir`, so cached tiles survive restarts.
- Put an HTTP/2 reverse proxy in front: Caddy, Traefik, or nginx with `http2 on`. Browsers allow only 6 HTTP/1.1 connections per host, and uvicorn speaks HTTP/1.1. With HTTP/2 the viewer detects multiplexing and raises its parallel tile requests from 5 to 24, which matters a lot with 16 panes.

## PathCore / HALO Link (or any tile API)

wissi-viz never opens a connection itself. `wissi_viz.connectors` has two **stubs**, `HaloLinkConnection` and `PathCoreConnection`. You subclass one and call the client your app already has (auth and sessions stay in your app):

```python
from wissi_viz.connectors import HaloLinkConnection, RemoteImageInfo, AnnotationPayload, TileNotFound

class WissiHaloLink(HaloLinkConnection):
    def __init__(self, client):
        super().__init__(); self.client = client

    def image_info(self, image_id):                       # required
        m = self.client.get_image(image_id)
        return RemoteImageInfo(width=m['width'], height=m['height'], tile_size=m['tile_size'],
                               levels=[(l['width'], l['height']) for l in m['levels']],
                               level_ids=[l['id'] for l in m['levels']],
                               mpp=(m['mpp_x'], m['mpp_y']), name=m['name'], version=str(m['modified']))

    def tile(self, image_id, level, col, row, channel=None):   # required
        return self.client.get_tile(image_id, level, col, row)  # bytes | ndarray | None

    def annotations(self, image_id):                     # optional
        return [AnnotationPayload('HALO annotations', self.client.get_annotations_xml(image_id), format='halo_xml')]

    def objects(self, image_id):                         # optional: cell tables, loaded in the background
        return [AnnotationPayload('HALO objects', self.client.get_object_data(image_id), format='table')]

sid = WissiHaloLink(client).register(registry, 'image-123')   # lazy → 'halo-image-123'
grid.set_slide(0, sid)
```

`examples/wissi_integration.py` has both stubs with the lines to fill in marked. What the connector layer handles:

| Real-world API behaviour | Handling |
|---|---|
| Pyramid described as explicit levels (any order), a downsample list, or DeepZoom (`.dzi`) | `RemoteImageInfo(levels=…, level_ids=…)` / `downsamples=…` / `deepzoom=True`, or `RemotePyramid.from_dzi(xml)`. Your own level ids are passed back to `tile()`. |
| DeepZoom tile overlap | JPEG/PNG bytes pass through untouched and the GPU crops the overlap with texture coordinates. Decoded arrays are cropped on the server. |
| Fixed-size padded edge tiles | Cropped the same way. |
| `async` clients | Coroutines run on a private loop, or on your app's loop via `conn.bind_loop()` (for clients bound to NiceGUI's loop). |
| Flaky upstream | Retries with back-off (`retries`), a `timeout`, and HTTP 502 to the browser if it still fails. Failures are not cached, and the browser retries later. |
| Not overloading the service | `max_concurrency` limits simultaneous calls per connection, shared across all its slides. Duplicate tile requests are merged. |
| Empty regions | Return `None` or raise `TileNotFound`: the tile is drawn as background and cached. |
| Slow metadata calls | `register()` is lazy, and opening one slide never blocks the others. |
| Annotations in µm, at a pyramid level, or offset | `AnnotationPayload(units='um', downsample=…, offset=…)`. |
| HALO XML, GeoJSON, Aperio XML, object tables (XMin/XMax/YMin/YMax, centroids, classes) | `AnnotationPayload(format='auto' \| 'halo_xml' \| 'geojson' \| 'aperio_xml' \| 'table')`. Table measurement columns are kept column-wise, so clicking one of 10⁶ cells shows its values. |
| Cache invalidation | `RemoteImageInfo.version` (a modified time or ETag) goes into every cache key. |

**Verification** (`tests/test_remote.py`, `examples/mock_remote_server.py`). The mock connectors reproduce these awkward cases, backed by a local SVS:
- A HALO-style mock: DeepZoom with 1 px overlap, a sync client, flaky calls, HALO XML and a cell table.
- A PathCore-style mock: an async client, coarsest-first string level ids, padded PNG tiles, and µm GeoJSON.

What was checked:
- Decoded tiles match the source pixel-for-pixel at every level, edges included.
- In the browser, the remote panes render on exactly the same screen pixels as the local file. A ±2 px search finds the best match at offset 0 at every zoom. What remains is JPEG noise: 1.4/255 at 40–80×, and 0.0–0.2/255 for lossless tiles.
- The annotations from both APIs line up with the local ones.

## Annotations

- **Sources:** GeoJSON (QuPath, HALO export), Aperio XML, HALO `.annotations`, and CZI graphics layers. Sidecar files with the same stem as a slide load automatically.
- **Building them in Python:**
  - `AnnotationLayer.from_points / from_polygons / from_boxes` for HALO object data (XMin/XMax/YMin/YMax).
  - `add_polygon / add_rect / add_ellipse / add_polyline / add_point`.
  - Attach with `registry.add_annotations(slide, layer)` or `grid.add_annotations(...)`. Viewers refresh automatically; call `registry.changed(slide, layer)` after in-place edits.

How it stays light with 16 slides × 10⁵–10⁶ objects:

- **Buckets.** Each layer is split into size tiers (cells, regions, ROIs). Each tier is then split into a quadtree of buckets.
- **Overview.** The browser first loads one compact *overview* per layer: a centroid, size, class and index for each feature, about 19 B per feature before gzip. Zoomed out, every feature is drawn as a GPU point. Only buckets that intersect the pane are drawn (frustum culling).
- **Detail on demand.** Exact outlines and fills are fetched per bucket, only when the bucket is in view *and* its objects would be at least `detail_min_px` on screen. They are cached server-side and in a GPU LRU in the browser.
- **Rendering.** Outlines are instanced quads, so any line width works. Fills are earcut-triangulated on the server. Colours come from a per-class texture, so hiding or recolouring a class costs nothing.
- **Clicking.** A click hit-tests the overview on the CPU, fetches that feature's properties, and emits `objectclick`.

**Drawing tools:** `setTool('rect'|'ellipse'|'polygon'|'freehand'|'line'|'point')`.
- Every drawing emits `draw` with GeoJSON in level-0 px.
- With the `draw-layer` attribute set, drawings are also saved to that layer on the server.

## Navigation API

| JS (`<wsi-grid>`) | Python (`WsiGrid`) | |
|---|---|---|
| `setSlides([...])`, `setSlide(i, ref)` | `set_slides`, `set_slide` | 1–16 panes; ref = id / URL / `{slide, view, channels, layers}` |
| `setLayout('auto'\|'4x4'\|'3x4')` | `set_layout` | |
| `goTo(pane, x, y, {units, zoom\|magnification\|mpp\|fov\|fovUm, animate})` | `go_to(...)` | pane = index, `'active'`, `'all'`, `[i, j]` |
| `goToBounds(pane, x0, y0, x1, y1)` | `go_to_bounds`, `go_to_object` | |
| `fit(pane)`, `getViews()` | `fit`, `await get_views()`, `grid.views` | |
| `setSync('none'\|'all'\|'pan'\|'zoom')` | `set_sync` | linked in µm when calibrated |
| `setChannel(pane, i, {visible,color,min,max,gamma})`, `setAdjust` | `set_channel`, `set_adjust` | |
| `setLayerVisible`, `setLayerStyle`, `setClassVisible`, `select` | same, snake_case | |

**Events:** `viewchange` (throttled), `paneclick`, `objectclick`, `draw`, `activepane`, `slideload`.

**Keyboard:** arrows, `+`/`-`, `0` to fit, `Tab` to change pane, `Enter`/`Esc` while drawing.

## Caching and smooth panning

**Server**
- **Two cache tiers.** An in-memory LRU of encoded tiles (default 1 GB), plus a disk cache (default 20 GB, trimmed by age). Tile URLs carry the slide version, so they are sent with `Cache-Control: immutable` and the browser's HTTP cache holds them too.
- **Coalescing.** If many panes or users ask for the same tile, it is decoded once.
- **Parallel reads.** Each worker thread has its own reader (tifffile/zarr, libCZI, OpenSlide), so decoding uses every core.
- **Pyramid gaps.** `PyramidSlide` inserts virtual 2× levels wherever native levels are more than 4× apart, and below the coarsest level. Each virtual tile is a box-filtered 2×2 mosaic of the level beneath, memoised. A slide with *no* pyramid therefore zooms out by touching each pixel once.
- **Warming.** The coarse levels are pre-rendered in the background, pausing whenever an interactive request is in flight.
- **Fluorescence data.** Raw uint16 tiles (zlib) are sent, never 8-bit JPEGs, so contrast, gamma and colour are applied exactly on the GPU.

**Browser**
- **Load order.** The coarsest level loads first, then the parent of the target level, then the target level from the centre outward, then a one-tile margin plus the area along the pan/zoom direction. Off-screen requests are cancelled.
- **No blank areas.** A missing tile is covered by the best coarser level already loaded.
- **No stutter.** Fetching and decoding run in Web Workers (`createImageBitmap`, `DecompressionStream`). GPU uploads are capped at about 6 ms per frame.
- **Memory.** One texture LRU, sized by `navigator.deviceMemory` and changeable with `setTextureBudget(mb)`, is shared across panes. Rendering only happens on demand, so an idle viewer uses no CPU or GPU.

**Measured in this build environment** (2 vCPU, *software* WebGL in headless Chromium, so a real GPU will be much faster):
- Cached tiles are served in about 1 ms.
- Cold CZI tiles were produced at about 17 ms each with 6 requests in parallel.
- With 16 linked panes being panned, a frame took 7 ms median of JavaScript and rasterisation.

## Tests

```bash
pip install -e ".[all,dev]" && pytest -q tests
python tests/make_test_data.py /tmp/wsi --size 16000   # synthetic SVS + 3-channel CZI + 120k cell GeoJSON
```

## Limits and next steps

- **Test data.** Verified on synthetic Aperio-style SVS, CZI (2- and 3-channel uint16) and tiled TIFF/OME-TIFF files, and against mock PathCore/HALO Link connectors. It has not yet been run against real scanner files or live PathCore/HALO Link endpoints. The connectors are stubs, and their method bodies come from your app.
- **Channels.** At most 8 channels are composited per pane at once. CZI Z and T planes are chosen when the source is created (`CziSource(path, z=, t=)`); there is no Z slider yet.
- **Annotation editing.** Drawings are saved as new features. Editing or deleting existing vertices in the browser is not implemented.
- **Possible additions:** a minimap/navigator, a WebGPU renderer, and a density heat-map LOD for more than 5 M objects.
