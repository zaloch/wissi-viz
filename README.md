# wissi-viz

wissi-viz is a GPU-accelerated whole-slide image viewer and spatial-analysis toolkit for brightfield and fluorescence histology. It runs in the browser.

- **What it shows:** 1–16 slides at once, from SVS, CZI, OME-TIFF, NDPI, VSI, ZVI, DICOM, SCN or AFI files, or from PathCore / HALO Link through connector stubs your app fills in.
- **Objects and annotations:** ROIs, drawings and millions of cells, imported from QuPath, HALO, CellProfiler, GeoJSON, label masks or AnnData, and rendered on the GPU. Layers can be coloured by class or by any measurement.
- **Spatial analysis:** nearest neighbours, proximity, invasive margins, ROI composition (part of whole), density and hotspot maps, Ripley's K/L/g, co-location enrichment, cellular neighbourhoods, contact graphs, Moran's I, IHC H-score, tissue detection and GPU stain deconvolution. Each one produces a table, a slide overlay and a chart.
- **Where it runs:** as a plain HTML element (`<wissi-viz>`), or as a native NiceGUI element (`wissi_viz.nicegui.WissiViz`) for wissi.
- **Licences:** the default install is **permissive-only** (BSD / MIT / Apache-2.0 / ISC), and a test enforces it. LGPL components are an opt-in extra; see [Licences](#licences).

```
 Browser                                                     Python (NiceGUI / FastAPI process)
┌────────────────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
│ <wissi-viz> one WebGL2 context, 1–16 panes          │      │ /wissi-viz router (mounted on NiceGUI's app) │
│  ├ scheduler: coarse → centre-out → margin/predict  │ HTTP │  ├ memory LRU  (encoded tiles)            │
│  ├ worker pool: fetch + decode off main thread      │◀────▶│  ├ disk cache  (survives restarts)        │
│  ├ GPU texture LRU (shared by all panes, budgeted)  │      │  ├ in-flight coalescing                    │
│  ├ shaders: RGB/stain unmixing / 8-ch uint16 compose│      │  ├ worker pool (per-thread file handles)   │
│  └ annotations: bucket cull + LOD, colormaps        │      │  └ PyramidSlide (fills pyramid gaps)       │
└────────────────────────────────────────────────────┘      │       └ SlideSource: tifffile │ czifile │      │
                                                             │         SlideIO │ PathCore/HALO connectors │
                                                             │ wissi_viz.objects / .spatial / .plots      │
                                                             └──────────────────────────────────────────┘
```

## Why Python + WebGL2, not Rust/WASM

The slides sit next to the server, in Docker, and they are multi-GB files.

- **Decoding belongs on the server.** Decoding has to happen where the bytes are: tifffile, czifile/imagecodecs and SlideIO do it in native code, and the server's caches make each tile a one-time cost.
- **WASM would have to download the files.** A WASM decoder would need the whole file, or HTTP range reads, pulled into every browser.
- **CZI's codecs are hard in WASM.** CZI uses JPEG-XR and zstd subblocks, which are painful to port.
- **The GPU work is small.** It is texture upload, multi-channel compositing, contrast and gamma, and annotation geometry. WebGL2 shaders already do all of that, everywhere, including Safari and iPad.
- **No build step.** The front end is a single dependency-free ES module.
- **WebGPU later is a local change.** A WebGPU backend could replace the renderer without changing the protocol.

## Install

```bash
pip install -e ".[all]"      # viewer + analysis + extra formats + NiceGUI + server — all permissive
pip install -e .             # minimal: SVS/TIFF/OME-TIFF/NDPI + CZI viewer
# extras: [analysis] scipy/shapely/scikit-image/pandas · [io] pyarrow/anndata · [slideio] VSI/ZVI/DICOM/SCN/AFI
#         [nicegui] · [server] uvicorn · [lgpl] OpenSlide + libCZI (opt-in, LGPL)
```

## Quick start (standalone server + demo page)

```bash
wissi-viz serve /data/slides --port 8010 --cache-dir /var/cache/wissi-viz
# open http://127.0.0.1:8010/
```

## Embed as an HTML element (any page, any framework)

```html
<script type="module" src="https://your-server/wissi-viz/static/index.js"></script>
<wissi-viz api="https://your-server/wissi-viz" slides="slideA,slideB,slideC,slideD" layout="auto"></wissi-viz>
<script type="module">
  const g = document.querySelector('wissi-viz');
  g.goTo(0, 15200, 9800, { magnification: 20 });            // level-0 px
  g.goTo('all', 1200, 800, { units: 'um', fovUm: 500 });    // microns, all panes
  g.setSync('all');                                         // linked navigation
  g.setStain(1, 'hematoxylin');                             // GPU colour deconvolution (brightfield)
  g.addEventListener('objectclick', (e) => console.log(e.detail));
</script>
```

`<wsi-grid>` and `<wsi-viewer>` remain as aliases.

## NiceGUI / wissi

```python
from nicegui import ui
from wissi_viz.nicegui import WissiViz, configure

registry = configure(cache_dir='/cache/wissi-viz')        # once at startup

@ui.page('/viewer')
def page():
    grid = WissiViz(['/data/a.svs', '/data/b.czi'], on_object_click=print).classes('w-full h-[85vh]')
    ui.button('hotspot', on_click=lambda: grid.go_to(0, 15200, 9800, magnification=20))
```

Routes mount on NiceGUI's own FastAPI app, so there is no extra port or process. The JavaScript is delivered through NiceGUI's ESM import map, and `window.path_prefix` (reverse-proxy subpaths) is respected. `WsiGrid` remains as an alias.

Examples:
- `examples/analysis_app.py`: the analysis workbench (viewer plus every analysis).
- `examples/nicegui_app.py`: a viewer UI.
- `examples/wissi_integration.py`: Docker plus HALO Link/PathCore stubs.

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

## Objects (cells, nuclei, detections)

```python
from wissi_viz import read_objects
cells = read_objects('export.tsv', mpp=0.5)       # QuPath TSV/CSV, HALO object data, CellProfiler, Parquet,
                                                  # QuPath GeoJSON, label mask (.tif), AnnData (.h5ad)
cells.phenotype({'CD3': ('Cell: CD3 mean', 400), 'CD8': ('Cell: CD8 mean', 300), 'CK': ('Cell: PanCK mean', 600)},
                rules={'CD8 T cell': {'CD3': True, 'CD8': True}, 'CD4 T cell': {'CD3': True, 'CD8': False},
                       'Tumor': {'CK': True}})
grid.show(slide_id, cells.to_layer(color_by='Nucleus: Ki67 mean', cmap='magma'))   # heat-coloured cells
```

- **Units:** coordinates in µm (detected from column names such as `Centroid X µm`) are converted to pixels with `mpp`. If the file is in µm and no `mpp` is given, it's an error, never a silent mix-up.
- **HALO multiplex flags:** `<Marker> Positive Classification` columns become phenotypes such as `CD3+CD8+`.
- **Measurements:** measurement columns travel with the layer, so clicking any cell shows them.

## Spatial analysis

`wissi_viz.spatial` (install the `[analysis]` extra). Every function works in µm and returns:
- `.table`, a DataFrame;
- `.layer()`, a slide overlay;
- `.plot()`, an ECharts option dict (`ui.echart(r.plot())`).

`grid.show(slide, result)` overlays a result. Results replace each other in a group (`replace=False` stacks them).

| Function | What it measures | Overlay |
|---|---|---|
| `nearest_neighbors(objs, A, B)` | distance from each A to the nearest B | line to the nearest B, coloured by distance |
| `proximity(objs, A, B, r)` | number of B within r µm of each A (infiltration, engagement) | A coloured by count |
| `distance_to_roi`, `margin_bands`, `infiltration_profile` | signed distance to a boundary, invasive-margin bands, density against distance | bands coloured by distance |
| `roi_composition(objs, rois, whole=tissue)` | per ROI: counts, % of ROI, density /mm², **% of whole** (part of whole), area share | ROI outlines or a choropleth; stacked-bar and sunburst charts |
| `density_grid(objs, cls, cell_um, denominator=…, hotspots=True)` | hex/square density or ratio maps, Getis-Ord Gi* hot and cold spots | heat map, hotspot layer |
| `ripley(objs, A, B=None, area=…)` | K, L, H, g(r), with border correction and a CSR simulation envelope | H(r) and g(r) charts |
| `neighborhood_enrichment(objs, r)` | class co-location z-scores (label permutation) | heat map |
| `cellular_neighborhoods(objs, k, n)` | kNN-composition niches (Schürch et al. 2020) | cells coloured by niche |
| `delaunay_graph(objs, max_edge_um)` | cell–cell contact graph, contact matrix, degree | edges coloured by class pair |
| `morans_i(objs, column)` | spatial autocorrelation of a measurement | — |
| `h_score`, `positivity`, `ObjectTable.intensity_bins` | IHC 0/1+/2+/3+, H-score, % positive per ROI | cells by bin |
| `tissue_mask(source)` | tissue detection, brightfield or fluorescence | tissue outline |
| `estimate_stain_vectors(source)` | Macenko H&E or H-DAB vectors | used by `set_stain` |

**Validation** (`tests/test_spatial.py`):
- **Exact checks:**
  - Nearest neighbours and proximity match brute force.
  - Margin-band areas match the closed-form geometry.
  - Composition shares add up to 100%.
  - Moran's I matches the dense-matrix formula.
  - Each object lands in exactly one grid cell.
- **Planted patterns:**
  - Ripley separates CSR, clustered and lattice patterns.
  - Enrichment finds planted pairs, and nothing in a random null.
  - Cellular neighbourhoods recover two planted regions (> 95% agreement).
  - Tissue detection reaches IoU > 0.97.
  - Macenko recovers H&E vectors within 6°.

On 120k cells, each analysis takes 0.01–4 s.

**GPU colour deconvolution.** `set_stain(pane, 'hematoxylin' | 'eosin' | 'dab' | 'residual' | 'fluorescence')` unmixes brightfield pixels in the fragment shader (Ruifrok & Johnston). It uses preset vectors or per-slide Macenko vectors (`await grid.auto_stain(pane)`). It needs no re-tiling and no extra server work. 'fluorescence' shows hematoxylin and the second stain as two pseudo-fluorescence channels.

**Colour by value.** `layer.color_by(values | column, cmap, vmin, vmax, log=…)` works on any layer. The colormaps are viridis, magma, inferno, plasma, cividis, turbo, coolwarm, RdBu and others, embedded without matplotlib. Each pane shows a legend of the visible layers.

## Navigation API

| JS (`<wissi-viz>`) | Python (`WissiViz`) | |
|---|---|---|
| `setSlides([...])`, `setSlide(i, ref)` | `set_slides`, `set_slide` | 1–16 panes; ref = id / URL / `{slide, view, channels, layers}` |
| `setLayout('auto'\|'4x4'\|'3x4')` | `set_layout` | |
| `goTo(pane, x, y, {units, zoom\|magnification\|mpp\|fov\|fovUm, animate})` | `go_to(...)` | pane = index, `'active'`, `'all'`, `[i, j]` |
| `goToBounds(pane, x0, y0, x1, y1)` | `go_to_bounds`, `go_to_object` | |
| `fit(pane)`, `getViews()` | `fit`, `await get_views()`, `grid.views` | |
| `setSync('none'\|'all'\|'pan'\|'zoom')` | `set_sync` | linked in µm when calibrated |
| `setChannel(pane, i, {visible,color,min,max,gamma})`, `setAdjust` | `set_channel`, `set_adjust` | |
| `setLayerVisible`, `setLayerStyle`, `setClassVisible`, `select` | same, snake_case | |
| `setStain(pane, mode, {preset, vectors})` | `set_stain`, `await auto_stain(pane)` | brightfield colour deconvolution |
| — | `show(slide, result_or_layer, group=…, replace=…)` | overlay analysis results |

**Events:** `viewchange` (throttled), `paneclick`, `objectclick`, `draw`, `activepane`, `slideload`.

**Keyboard:** arrows, `+`/`-`, `0` to fit, `Tab` to change pane, `Enter`/`Esc` while drawing.

## Caching and smooth panning

**Server**
- **Two cache tiers.** An in-memory LRU of encoded tiles (default 1 GB), plus a disk cache (default 20 GB, trimmed by age). Tile URLs carry the slide version, so they are sent with `Cache-Control: immutable` and the browser's HTTP cache holds them too.
- **Coalescing.** If many panes or users ask for the same tile, it is decoded once.
- **Parallel reads.** Each worker thread has its own reader (tifffile/zarr, czifile, SlideIO), so decoding uses every core.
- **CZI.** The reader decodes only the subblocks that intersect a tile, straight from their raw bytes through imagecodecs (zstd, JPEG-XR, JPEG, LZW). The CZI's own pyramid subblocks become native levels. Decoded subblocks are kept in an LRU, so the ~16 tiles inside one 2048² subblock decode it once (cold tile ~15–30 ms, cached tile ~0.1 ms). It is pixel-identical to libCZI on every tile tested.
- **Pyramid gaps.** `PyramidSlide` inserts virtual 2× levels wherever native levels are more than 4× apart, and below the coarsest level. Each virtual tile is a box-filtered 2×2 mosaic of the level beneath, memoised. A slide with *no* pyramid therefore zooms out by touching each pixel once.
- **Warming.** The coarse levels are pre-rendered in the background, pausing whenever an interactive request is in flight.
- **Fluorescence data.** Raw uint16 tiles (zlib) are sent, never 8-bit JPEGs, so contrast, gamma and colour are applied exactly on the GPU.

**Browser**
- **Load order.** The coarsest level loads first, then the parent of the target level, then the target level from the centre outward, then a one-tile margin plus the area along the pan/zoom direction. Off-screen requests are cancelled.
- **No blank areas.** A missing tile is covered by the best coarser level already loaded.
- **No stutter.** Fetching and decoding run in Web Workers (`createImageBitmap`, `DecompressionStream`). GPU uploads are capped at about 6 ms per frame.
- **Memory.** One texture LRU, sized by `navigator.deviceMemory` and changeable with `setTextureBudget(mb)`, is shared across panes. Rendering only happens on demand, so an idle viewer uses no CPU or GPU.

**Benchmarks** (`wissi-viz bench`, see [benchmarks/](benchmarks/README.md)). Measured on a 2-vCPU cloud container with no GPU, where Chromium uses SwiftShader software WebGL:

- **Opening:** 8–17 ms for every file, including a 1.65 GB OME-TIFF.
- **Cold full-resolution tile:** 2.2 ms for SVS, 2.6 ms for OME-TIFF, 20 ms for CZI (a whole 2048² zstd subblock is decoded on first touch). A cached tile takes about 5 µs.
- **Panning:** the viewer's own JavaScript costs 0.2–5 ms per frame. The rest of each frame is rasterisation, which in this container is CPU-emulated.
  - At 800×500 a single pane pans at a locked 60 fps.
  - At 1600×1000 the frame rate is bound by software fill rate: a bare WebGL page manages only 20 fps here.
  - No run showed a blank area, at any pane count.

GPU acceleration happens in the viewer's browser. The server, for example wissi in Docker, needs no GPU. Run `wissi-viz bench … --headed` on a desktop to get numbers for your GPU.

## Licences

| Install | Licence | Contents |
|---|---|---|
| core | BSD-3 / MIT / ISC / HPND | numpy, fastapi (starlette, pydantic), tifffile, zarr, imagecodecs, czifile, pillow, mapbox-earcut |
| `[analysis]` | BSD-3 | scipy, shapely, scikit-image, pandas |
| `[io]` | Apache-2.0 / BSD-3 | pyarrow, anndata |
| `[slideio]` | BSD-3 (bundles OpenCV Apache-2.0, DCMTK BSD-3, OpenJPEG BSD-2) | SlideIO |
| `[nicegui]`, `[server]` | MIT / BSD-3 | NiceGUI (bundles ECharts, Apache-2.0), uvicorn |
| **`[lgpl]` (opt-in)** | **LGPL-2.1 / LGPL-3** | OpenSlide (MRXS, BIF), pylibCZIrw (libCZI) |

- **The audit test:** `tests/test_licenses.py` reads the installed metadata and fails if a core or permissive extra pulls in a GPL, LGPL, MPL or similar licence. It also checks that importing the core never loads an LGPL module.
- **The colormaps** are CC0 (viridis family), Apache-2.0 (turbo) or public domain (coolwarm).
- **Formats that need the LGPL extra:** only MRXS (3DHistech) and BIF (Ventana). Everything else has a permissive reader.

## Tests

```bash
pip install -e ".[all,dev]" && pytest -q tests                      # 51 tests
python tests/make_test_data.py /tmp/wsi --size 16000                # synthetic SVS + 3-channel CZI + 120k cell GeoJSON
python tests/make_analysis_data.py /tmp/wsi                         # QuPath-style multiplex cell table + ROIs
python examples/analysis_app.py /tmp/wsi/synthetic_he.svs /tmp/wsi/synthetic_he_cells.tsv /tmp/wsi/synthetic_he_rois.geojson
```

## Limits and next steps

- **Test data.**
  - Verified on synthetic Aperio-style SVS, CZI (mosaic with overlap, pyramid subblocks, RGB and multichannel uint16), tiled TIFF/OME-TIFF, and mock PathCore/HALO Link connectors.
  - The analyses were validated on synthetic cell tables with planted biology.
  - It has not yet been run on real scanner files, real QuPath/HALO exports or live PathCore/HALO Link endpoints.
- **Analysis scale.** Ripley simulations subsample to 20k points, and contact graphs and NN-line overlays cap the number of drawn edges. All the numbers in the tables use every object.
- **Channels.** At most 8 channels are composited per pane at once. CZI Z and T planes are chosen when the source is created (`CziSource(path, z=, t=)`); there is no Z slider yet.
- **Annotation editing.** Drawings are saved as new features. Editing or deleting existing vertices in the browser is not implemented.
- **Possible additions:** a minimap/navigator, a WebGPU renderer, and a density heat-map LOD for more than 5 M objects.
