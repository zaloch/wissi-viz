# wissi-viz — full guide

*The project overview is in the [README](../README.md); integration recipes are in [INTEGRATION.md](INTEGRATION.md).*

wissi-viz is a GPU-accelerated whole-slide image viewer and spatial-analysis toolkit for brightfield and fluorescence histology. It runs in the browser.

- **What it shows:** 1–32 slides at once, from SVS, CZI, OME-TIFF, NDPI, VSI, ZVI, DICOM, SCN or AFI files, or from PathCore / HALO Link through connector stubs your app fills in.
- **Objects and annotations:** ROIs, drawings and millions of cells, imported from QuPath, HALO, CellProfiler, GeoJSON, label masks or AnnData, and rendered on the GPU. Layers can be coloured by class or by any measurement.
- **Spatial analysis:** nearest neighbours, proximity, invasive margins, ROI composition (part of whole), density and hotspot maps, Ripley's K/L/g, co-location enrichment, cellular neighbourhoods, contact graphs, Moran's I, IHC H-score, tissue detection and GPU stain deconvolution. Each one produces a table, a slide overlay and a chart.
- **Where it runs:** as a plain HTML element (`<wissi-viz>`), or as a native NiceGUI element (`wissi_viz.nicegui.WissiViz`) for wissi.
- **Licences:** the default install is **permissive-only** (BSD / MIT / Apache-2.0 / ISC), and a test enforces it. LGPL components are an opt-in extra; see [Licences](#licences).

```
 Browser                                                     Python (NiceGUI / FastAPI process)
┌────────────────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
│ <wissi-viz> one WebGL2 context, 1–32 panes          │      │ /wissi-viz router (mounted on NiceGUI's app) │
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

Windows, macOS (Apple silicon and Intel) and Linux (x86_64, ARM), Python 3.10–3.14. Packaged with [uv](https://docs.astral.sh/uv/); `pip` works too.

```bash
uv add "wissi-viz[all]"               # in your project: viewer + analysis + extra formats + NiceGUI + server — all permissive
uv add wissi-viz                      # minimal: SVS/TIFF/OME-TIFF/NDPI + CZI viewer
uv tool install "wissi-viz[desktop]"  # the standalone desktop app (below)
# extras: [analysis] scipy/shapely/scikit-image/pandas · [io] pyarrow/anndata · [slideio] VSI/ZVI/DICOM/SCN/AFI
#         [nicegui] · [server] uvicorn · [remote] httpx · [desktop] pywebview · [bench] playwright/hypercorn/cryptography
#         [lgpl] OpenSlide + libCZI (opt-in, LGPL)
```

## Desktop app

The viewer in its own native window, with its own icon, as an option next to the embedded element:

```bash
wissi-viz-app                              # File ▸ Open slides… (Ctrl/⌘+O), or drop files on the window
wissi-viz desktop a.svs b.czi cells.csv    # open files from the command line
wissi-viz desktop --install-shortcut       # Start menu + desktop (Windows), ~/Applications (macOS), app menu (Linux)
```

It runs the tile server in-process on 127.0.0.1 and shows it with pywebview: WebView2 on Windows, WKWebView on macOS, WebKitGTK or Qt on Linux (Linux needs GTK or Qt bindings installed separately; without them it opens a Chrome/Edge app window). Details: [docs/INTEGRATION.md §12](INTEGRATION.md#12-desktop-app-path-f).

> **Integrating into an app?** [docs/INTEGRATION.md](INTEGRATION.md) covers every option: the NiceGUI element, a raw `<wissi-viz>` tag, mounting on any ASGI app, a separate tile server behind React, Vue, Streamlit, Dash or Django, and an iframe. It also covers auth, Docker and proxies, and how to wire a vendor tile API. **Before go-live:**
> - `scripts/wissi_probe.py` checks a connector (HALO Link, PathCore, any tile API) or a deployment in one command, under a hard API-call cap. It smoke-tests the API, sweeps concurrency, runs 1–16-pane viewer sessions, and writes recommended settings ([§10](INTEGRATION.md#10-probing-an-integration-one-command)).
> - [docs/LOAD_TESTING.md](LOAD_TESTING.md) and `scripts/wissi_loadtest.py` simulate 10–100 users (≤ 999 requests each) and estimate capacity.

## Quick start (standalone server + demo page)

```bash
wissi-viz serve /data/slides --port 8010 --cache-dir /var/cache/wissi-viz
# open http://127.0.0.1:8010/
wissi-viz serve --allow-open /data/slides --key s3cret   # visitors browse that folder and open slides
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

registry = configure(cache_dir='/cache/wissi-viz')  # once at startup


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

To protect slides with your app's login, pass FastAPI dependencies, which apply to every tile and annotation route: `configure(dependencies=[Depends(require_login)])`.

**Docker notes**
- Mount a volume for `cache_dir`, so cached tiles survive restarts.
- Put an HTTP/2 reverse proxy in front: Caddy, Traefik, or nginx with `http2 on`. Browsers allow only 6 HTTP/1.1 connections per host, and uvicorn speaks HTTP/1.1. With HTTP/2 the viewer detects multiplexing and raises its parallel tile requests from 5 to 24, which matters a lot with 16 panes.

## PathCore / HALO Link (or any tile API)

Two ways in:

- **A JSON description** for services with plain tile URLs: `wissi-viz serve --connect my_api.json`, `wissi-viz desktop --connect my_api.json`, or **Connect…** in the app. See [INTEGRATION.md §7](INTEGRATION.md#7-connecting-a-tile-api-pathcore-halo-link-).
- **Your own connector** for everything else. `wissi_viz.connectors` has two **stubs**, `HaloLinkConnection` and `PathCoreConnection`. You subclass one and call the client your app already has (auth and sessions stay in your app). From the command line: `--connect mypkg.halo:make_connection`.

```python
from wissi_viz.connectors import HaloLinkConnection, RemoteImageInfo, AnnotationPayload, TileNotFound


class WissiHaloLink(HaloLinkConnection):
    def __init__(self, client):
        super().__init__()
        self.client = client

    def image_info(self, image_id):  # required
        m = self.client.get_image(image_id)
        return RemoteImageInfo(
            width=m['width'],
            height=m['height'],
            tile_size=m['tile_size'],
            levels=[(l['width'], l['height']) for l in m['levels']],
            level_ids=[l['id'] for l in m['levels']],
            mpp=(m['mpp_x'], m['mpp_y']),
            name=m['name'],
            version=str(m['modified']),
        )

    def tile(self, image_id, level, col, row, channel=None):  # required
        return self.client.get_tile(image_id, level, col, row)  # bytes | ndarray | None

    def annotations(self, image_id):  # optional
        return [AnnotationPayload('HALO annotations', self.client.get_annotations_xml(image_id), format='halo_xml')]

    def objects(self, image_id):  # optional: cell tables, loaded in the background
        return [AnnotationPayload('HALO objects', self.client.get_object_data(image_id), format='table')]


sid = WissiHaloLink(client).register(registry, 'image-123')  # lazy → 'halo-image-123'
grid.set_slide(0, sid)
```

`examples/wissi_integration.py` has both stubs with the lines to fill in marked. **`examples/tile_api/`** is a full, runnable simulation:
- a mock vendor service (bearer auth, 16 images in 4 pyramid styles, latency, 429/503/404);
- a ~100-line `httpx` connector;
- a NiceGUI app showing 1–32 images;
- a cold/warm browser benchmark. Results are in [benchmarks/results-tile-api.md](../benchmarks/results-tile-api.md).

What the connector layer handles:

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

**No Python needed for plain-URL tile APIs:** `HttpTileConnection('my_api.json')` reads base URL, auth headers (`${ENV_VAR}`), metadata fields and the tile URL template from JSON. See `scripts/probe_configs/`.

**Check a connector before go-live:**
```bash
python scripts/wissi_probe.py yourpkg.halo:make_connection --images IMG1,IMG2 --yes
```
This runs a smoke test, a concurrency sweep at 5 / 8 / 10, and 1–16-pane sessions, at ≤ 1000 API calls by default. `mock:halolink` / `mock:pathcore` run the same probe on the mocks below.

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

cells = read_objects('export.tsv', mpp=0.5)  # QuPath TSV/CSV, HALO object data, CellProfiler, Parquet,
# QuPath GeoJSON, label mask (.tif), AnnData (.h5ad)
cells.phenotype(
    {'CD3': ('Cell: CD3 mean', 400), 'CD8': ('Cell: CD8 mean', 300), 'CK': ('Cell: PanCK mean', 600)},
    rules={'CD8 T cell': {'CD3': True, 'CD8': True}, 'CD4 T cell': {'CD3': True, 'CD8': False}, 'Tumor': {'CK': True}},
)
grid.show(slide_id, cells.to_layer(color_by='Nucleus: Ki67 mean', cmap='magma'))  # heat-coloured cells
```

- **Units:** coordinates in µm (detected from column names such as `Centroid X µm`) are converted to pixels with `mpp`. If the file is in µm and no `mpp` is given, it's an error, never a silent mix-up.
- **HALO multiplex flags:** `<Marker> Positive Classification` columns become phenotypes such as `CD3+CD8+`.
- **Measurements:** measurement columns travel with the layer, so clicking any cell shows them.

### Millions of objects, 1–16 panes, 10–100 users

Layers with ≥ 50,000 objects get an **object pyramid**, built once on the server. It mirrors the image pyramid:

- **Zoomed out:** each pane gets aggregated dots, about one per 4 screen px.
  - Each dot carries a per-class histogram, so hiding a class re-colours dots by the most common *visible* class.
  - Colour-by-value layers carry the bin's mean value instead.
- **Zoomed in:** only the visible cells arrive, as 128 × 128-bin tiles.
- **Outlines:** bounding-box objects (HALO XMin/XMax/YMin/YMax, centroid tables) are stored as columns and drawn as exact ellipses or rectangles by a GPU shader, at about 19 bytes per cell. Before, each cell was a 12-vertex polygon of ~200 bytes, and the server kept 10⁶ small arrays.
- **Clicks:** clicks on aggregated dots are resolved by the server. Zoomed in, they are resolved in the browser.

What one browser downloads now scales with **screen area, not with the object count**.

```python
from wissi_viz import cached_object_layer

layer = cached_object_layer(
    'cells.parquet', '/cache/layers', mpp=0.25, name='cells'
)  # compiled once, then memory-mapped
grid.show(slide_id, layer)
```

`cached_object_layer` keys the compiled layer (buckets plus pyramid) on the file's path, size and modification time and on every option.
- **Reopening is fast:** ~35 ms instead of ~3.5 s of CPU per 1M cells.
- **It survives restarts.**
- **Server workers share one copy** through the OS page cache.

1M cells per slide, same browser and container ([benchmarks/results-overlays.md](../benchmarks/results-overlays.md)):

| | 1 pane, legacy | 1 pane, new | 16 panes, legacy | 16 panes, new |
|---|---|---|---|---|
| until every overlay is drawn | 1.95 s | 1.08 s | 75.7 s | **1.5 s** |
| data sent to the browser (gzip) | 9.1 MB | 0.43 MB | 145 MB | **0.64 MB** |
| GPU memory for overlays | 20 MB | 0.6 MB | 321 MB | **1.1 MB** |
| server memory kept per 1M-cell layer | ≈ 650 MB | 225 MB | | |

**Server- or client-side?** `setLod({overlays: 'server' | 'client' | 'auto'})`, or `WissiViz(lod={'overlays': …})`.
- **`client`:** the browser downloads each layer's objects once (17 B each), builds the same pyramid in a Web Worker, and then navigates and clicks with no overlay requests.
  - Measured per 1M cells: 7.8 MB, 72 MB of browser memory and a 1.9 s build.
  - With 16 slides: 125 MB, **1.1 GB** and 32 s, against 0.66 MB, 5 MB and 1.5 s for the server pyramid.
- **`server`** stays the default.
- **`auto`** goes client-side only for layers ≤ 300k objects, up to 1M per viewer.

**Which file format for object tables?** Measured on 1M cells × 37 columns ([details](../benchmarks/results-overlays.md#storage-formats)):
- **Parquet** for exchange: 0.4 s to load what the viewer needs. CSV/TSV with pandas takes 2.8–3.2 s; wissi-viz now parses CSV/TSV with pyarrow in 0.8 s.
- **Arrow IPC / Feather** for the fastest loads and single-row lookups (0.2 ms).
- **SQLite or PostgreSQL** for *editable, shared* annotations. They are slow for bulk object loads (2.5–14 s, up to 2 GB peak), but single lookups take ~0.1 ms and concurrent writers are safe.
- **DuckDB** for cohort queries over many slides' Parquet files.
- **CSV vs TSV:** the same speed; TSV avoids quoting problems with commas in class names.
- **pandas vs Polars:** Polars parses CSV/TSV 3.8× faster than pandas and holds a table in about half the memory ([results-dataframes.md](../benchmarks/results-dataframes.md)).
  - `read_objects(engine='auto')` uses Polars when installed (`[polars]` extra), else pyarrow, else pandas.
  - Polars `DataFrame`/`LazyFrame` and Arrow tables can be passed in directly.

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
| `setSlides([...])`, `setSlide(i, ref)` | `set_slides`, `set_slide` | 1–32 panes; ref = id / URL / `{slide, view, channels, layers}` |
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
- **Pyramid gaps.** `PyramidSlide` inserts virtual 2× levels wherever native levels are more than 4× apart (`max_level_step`, default 4), and below the coarsest level. Each virtual tile is a box-filtered 2×2 mosaic of the level beneath, memoised. A slide with *no* pyramid therefore zooms out by touching each pixel once.
- **Memory.** Decoded arrays (virtual-level inputs, CZI subblocks) share **one process-wide budget**, `decoded_cache_mb`, default 1 GB. Busy slides get more of it, and closing a slide frees its share. (Per-slide caches let a 16-pane run grow past 5 GB and get killed for running out of memory.)
- **Memory stays flat over time.** Every server cache has a byte budget: tiles, decoded arrays, overlay tiles, gzipped overlays, and annotation layers.
  - Annotation and object layers that come with a slide (e.g. PathCore / HALO `objects()`) are saved once in compiled form under `cache_dir` and memory-mapped. That's ~0.3 MB of RAM per 300k cells instead of 90 MB.
  - Reopening a case costs no API calls.
  - Layers beyond `annotation_memory_mb` / `layer_mapped_mb` are dropped least-recently-used and reload on demand.
  - Before 0.6.1 these layers stayed in RAM for every slide ever opened, which is how a long-running PathCore server could reach ~20 GB.
  - `GET /wissi-viz/stats` → `memory` shows where RAM is. See [benchmarks/results-memory.md](../benchmarks/results-memory.md).
- **Warming.** The coarse levels are pre-rendered in the background, pausing whenever an interactive request is in flight.
- **Fluorescence data.** Raw uint16 tiles (zlib) are sent, never 8-bit JPEGs, so contrast, gamma and colour are applied exactly on the GPU.

**Browser**
- **Load order.** The coarsest level loads first, then the parent of the target level, then the target level from the centre outward, then a one-tile margin plus the area along the pan/zoom direction. Off-screen requests are cancelled.
- **No blank areas.** A missing tile is covered by the best coarser level already loaded.
- **Level of detail across zoom** (`setLod`, see [Zooming through the pyramid](#zooming-through-the-pyramid-level-of-detail)). What is drawn is a 3-D frustum over (x, y, pyramid level). Each region shows the finest tile already loaded, coarse tiles fill only the holes, and each pane pixel is drawn once. While a pane moves, a level one step coarser is fetched first and full resolution follows in spare request slots. Animations fetch for where the pane will land.
- **No stutter.** Fetching and decoding run in Web Workers (`createImageBitmap`, `DecompressionStream`). GPU uploads are capped at about 6 ms per frame.
- **Smooth on weak or no GPU** (`lod.adaptive`, on by default; see [benchmarks/results-smoothness.md](../benchmarks/results-smoothness.md)):
  - **Adaptive resolution.** While something moves *and* frames are slow, the WebGL backing store drops to ¾ or ½ resolution, and full resolution returns on the first frame at rest. Without a GPU, Chrome reads the whole canvas back every frame, so a smaller backing store is the lever that works.
  - **Measured frame cost.** The real per-frame cost (our JS plus the browser's commit) is measured. When it is high, tile arrivals at rest are batched into fewer redraws, so the next drag isn't stuck behind them.
  - **Culled cell outlines.** Cell outlines are binned per bucket, and only the rows in view are drawn: 16 panes at 1:1 went from 905 ms to 57 ms of GPU work per frame in software WebGL.
  - **No per-frame churn.** No garbage from tile keys (the longest GC fell from 127 ms to 8 ms), no per-frame DOM writes, and no per-frame 2-D canvas clears.
  - **Measured result.** In a GPU-less container, 16-pane fling went from 4.6 to 34 fps, and 16-pane wheel-zoom input latency (p95) from 781 to 136 ms.
- **Memory.** One texture LRU, sized by `navigator.deviceMemory` and changeable with `setTextureBudget(mb)`, is shared across panes. Rendering only happens on demand, so an idle viewer uses no CPU or GPU.

**Benchmarks** (`wissi-viz bench`, see [benchmarks/](../benchmarks/README.md)). Measured on a 2-vCPU cloud container with no GPU, where Chromium uses SwiftShader software WebGL:

- **Opening:** 8–17 ms for every file, including a 1.65 GB OME-TIFF.
- **Cold full-resolution tile:** 2.2 ms for SVS, 2.6 ms for OME-TIFF, 20 ms for CZI (a whole 2048² zstd subblock is decoded on first touch). A cached tile takes about 5 µs.
- **Panning:** the viewer's own JavaScript costs 0.2–5 ms per frame. The rest of each frame is rasterisation, which in this container is CPU-emulated.
  - At 800×500 a single pane pans at a locked 60 fps.
  - At 1600×1000 the frame rate is bound by software fill rate: a bare WebGL page manages only 20 fps here.
  - No run showed a blank area, at any pane count.

GPU acceleration happens in the viewer's browser. The server, for example wissi in Docker, needs no GPU. Run `wissi-viz bench … --headed` on a desktop to get numbers for your GPU.

## Zooming through the pyramid (level of detail)

**What gets drawn.** The view is treated as a 3-D frustum over x, y and pyramid level. For every region of every pane, the viewer walks the pyramid as a quadtree, from the coarsest level down to a *floor*:
- It draws the finest tile already loaded.
- A coarser tile is drawn only through the holes its children leave, as sub-rectangles of its texture.
- Each pane pixel is shaded **once**. The legacy painter's algorithm drew a whole fallback level underneath: 1.1–2.1× fill, which matters for the 8-channel fluorescence shader on a weak GPU.
- The floor is one level *finer* than needed when that level is already loaded (`minify: 0.5`). When you zoom out, sharp tiles stay on screen instead of dropping to the coarser level.

**What gets fetched.**

| Situation | Fetched first | Then |
|---|---|---|
| At rest | the full-resolution level (texels ≤ 1 device px) | margin ring |
| Moving (pan, fling, zoom) | the level whose texels are ≤ 2 device px (`motionBias: 2`): **¼ of the tiles** on a 2×-step pyramid | full resolution in *spare* request slots only, ⅓ always kept free for urgent tiles; swapped in as it arrives |
| Animating to a known view (goTo, double-click, wheel target) | the destination's level and its parent | the in-between views, only with spare capacity |

- **Requests between frames.** A finished request starts the next queued one right away, instead of waiting for the next frame. GPU uploads get up to 20% of a slow frame.
- **Prefetch (`prefetch: true`, off by default)** fetches the next finer level around the centre at rest. It was measured as roughly neutral, so it stays off.
- **Server option:** `max_level_step=2` gives 4×-step SVS pyramids virtual 2× levels. Browsers then download less, at the cost of server CPU per virtual tile.

```js
grid.setLod('default');                       // or 'legacy' (pre-0.4 behaviour), or individual options:
grid.setLod({ motionBias: 2, minify: 0.5, quadtree: true, background: true, destination: true, decoupled: true, prefetch: false });
```
```python
WissiViz(slides, lod='default')  # or grid.set_lod('legacy') / grid.set_lod(motion_bias=1.5, prefetch=True)
```
```html
<wissi-viz lod='{"motionBias": 1.5}' ...></wissi-viz>
```

Measured with `wissi-viz bench --zoom` (legacy vs new; 1 and 16 panes; SVS, OME-TIFF and CZI): see [benchmarks/results-zoom.md](../benchmarks/results-zoom.md).

**Z-stacks.** "3-D" here means the scale axis. Focal planes (CZI Z/T) are still chosen when a source is opened (`CziSource(path, z=…)`). Streaming Z as a fourth tile dimension, with neighbouring planes prefetched, is a natural next step on the same scheduler.

## Licences

| Install | Licence | Contents |
|---|---|---|
| core | BSD-3 / MIT / ISC / HPND | numpy, fastapi (starlette, pydantic), tifffile, zarr, imagecodecs, czifile, pillow, mapbox-earcut |
| `[analysis]` | BSD-3 | scipy, shapely, scikit-image, pandas |
| `[io]` | Apache-2.0 / BSD-3 | pyarrow, anndata |
| `[slideio]` | BSD-3 (bundles OpenCV Apache-2.0, DCMTK BSD-3, OpenJPEG BSD-2) | SlideIO |
| `[nicegui]`, `[server]` | MIT / BSD-3 | NiceGUI (bundles ECharts, Apache-2.0), uvicorn |
| `[desktop]` | BSD-3 / MIT | pywebview, pythonnet (Windows), pyobjc (macOS), bottle, proxy-tools, uvicorn. Linux's GTK/Qt bindings (LGPL/GPL) are not dependencies. |
| **`[lgpl]` (opt-in)** | **LGPL-2.1 / LGPL-3** | OpenSlide (MRXS, BIF), pylibCZIrw (libCZI) |

- **The audit test:** `tests/test_licenses.py` reads the installed metadata and fails if a core or permissive extra pulls in a GPL, LGPL, MPL or similar licence. It also checks that importing the core never loads an LGPL module.
- **The colormaps** are CC0 (viridis family), Apache-2.0 (turbo) or public domain (coolwarm).
- **Formats that need the LGPL extra:** only MRXS (3DHistech) and BIF (Ventana). Everything else has a permissive reader.

## Tests

```bash
uv sync --extra analysis --extra io --extra polars --extra server --extra remote --extra bench --extra desktop
uv run playwright install chromium && uv run pytest -q tests         # or: pip install -e ".[all,desktop,dev]" && pytest -q
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
