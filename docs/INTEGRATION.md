# Integrating wissi-viz

This guide covers how to put wissi-viz into an app: NiceGUI (wissi), other Python web frameworks, JavaScript frameworks and notebooks. It also covers how to connect a vendor tile API such as PathCore or HALO Link.

```
 browser (any app)                       your server                                   storage / vendors
┌───────────────────────┐   HTTP(S)   ┌──────────────────────────────────┐        ┌───────────────────────┐
│ <wissi-viz> (WebGL2)  │◀──────────▶│ wissi-viz router (/wissi-viz/…)   │──read──▶│ SVS/CZI/OME-TIFF files │
│ GPU: tiles, channels, │  tiles,     │  memory + disk tile cache         │        └───────────────────────┘
│ stains, annotations   │  overlays   │  SlideRegistry                    │──HTTP──▶┌───────────────────────┐
└───────────────────────┘             │   ├ file sources                  │  token  │ PathCore / HALO Link / │
                                      │   └ RemoteConnection (your code)  │         │ any tile API          │
                                      └──────────────────────────────────┘        └───────────────────────┘
```

- **GPU work happens in the browser.** The server decodes and caches tiles on the CPU and needs no GPU.
- **The browser never talks to the vendor API.** Tiles flow API → wissi-viz cache → browser, so vendor tokens stay on your server and repeat views make no upstream requests.

## 1. Choose an integration path

| Path | Use it when | Python control | Tested here |
|---|---|---|---|
| **A. NiceGUI element** `WissiViz` | your app is NiceGUI (wissi) | full: navigation, layers, analysis, events | ✔ `examples/*app.py`, browser tests |
| **B. Web component in a NiceGUI page** | you prefer to drive it from JavaScript | `ui.run_javascript` | ✔ (same component) |
| **C. Same server, any ASGI app** (FastAPI, Starlette, Litestar) | one Python service | your own endpoints | ✔ `create_router` / `create_app` |
| **D. Separate tile server + `<wissi-viz>` in any page** (React, Vue, Svelte, Django or Flask templates, Streamlit, Dash, static HTML) | non-Python front ends, WSGI apps, or scaling tiles separately | over HTTP | ✔ cross-origin test (`tests/test_browser.py`) |
| **E. iframe** of a viewer page | maximum isolation, minimum integration | `postMessage` | — |

## 2. NiceGUI (path A)

```python
from fastapi import Depends, Request, HTTPException
from nicegui import ui
from wissi_viz.nicegui import WissiViz, configure

def require_login(request: Request):                       # your auth; protects every tile route
    if not request.session.get('user'):
        raise HTTPException(401)

registry = configure(cache_dir='/cache/wissi-viz', memory_mb=2048,
                     dependencies=[Depends(require_login)])

@ui.page('/viewer')
def viewer():
    grid = WissiViz(['/data/case1.svs', '/data/case1_if.czi'], layout='1x2', sync='all',
                    on_object_click=lambda e: ui.notify(e.get('properties')),
                    on_view_change=lambda v: None).classes('w-full h-[85vh]')
    with ui.row():
        ui.button('ROI 1', on_click=lambda: grid.go_to(0, 15200, 9800, magnification=20))
        ui.button('All → 1.2 mm, 0.8 mm', on_click=lambda: grid.go_to('all', 1200, 800, units='um', fov_um=500))
        ui.button('4×4', on_click=lambda: grid.set_layout('4x4'))
```

| Python call | Effect |
|---|---|
| `set_slides([...])`, `set_slide(i, ref)` | 1–16 panes. A ref is a path, a registered slide id, a `SlideSource` or `{slide, view, channels, layers}`. |
| `go_to(pane, x, y, units='px'\|'um', magnification=…\|zoom=…\|fov=…\|fov_um=…)`, `go_to_bounds`, `go_to_object`, `fit` | navigation; `pane` = index, `'active'`, `'all'` or a list |
| `set_layout('auto'\|'RxC')`, `set_sync('none'\|'all'\|'pan'\|'zoom')` | grid layout and linked navigation |
| `set_channel(pane, i, visible=, color=, min=, max=, gamma=)`, `set_adjust(...)`, `set_stain(pane, mode)`, `await auto_stain(pane)` | display |
| `show(slide, result_or_layer, group=…, replace=…)`, `add_annotations(slide, layer)` | overlays; viewers refresh automatically |
| `set_layer_visible`, `set_layer_style`, `set_class_visible`, `select` | layer control |
| `await get_views()`, `grid.views` | where every pane is looking (px and µm) |
| events: `on_view_change`, `on_click`, `on_object_click`, `on_draw`, `on_slide_load` | Python callbacks |

**Notes**
- **Shared state:** the registry and its caches are per process and shared by all users. Use per-user slide ids if users must not see each other's annotation edits.
- **Paths:** routes mount on NiceGUI's own FastAPI app, so there is no extra port. Reverse-proxy subpaths (`window.path_prefix`) work.
- **Analysis:** results go into plain NiceGUI widgets, for example `ui.echart(result.plot())` and `ui.table(rows=plots.table_rows(result.table))`. See `examples/analysis_app.py`.

### Path B: the raw component inside NiceGUI

```python
registry = configure()        # still mounts /wissi-viz
ui.add_head_html('<script type="module" src="/wissi-viz/static/index.js"></script>')
ui.html('<wissi-viz id="v" slides="slideA,slideB" style="display:block;height:80vh"></wissi-viz>')
ui.button('20×', on_click=lambda: ui.run_javascript(
    'document.getElementById("v").goTo("all", 8000, 6000, {magnification: 20})'))
```

## 3. Any ASGI app (path C)

```python
from fastapi import FastAPI
from wissi_viz import SlideRegistry, TileService, create_router

registry = SlideRegistry(TileService(disk_dir='/cache/wissi-viz'))
registry.add('/data/case1.svs', slide_id='case1')
app = FastAPI()
app.include_router(create_router(registry, cors=None), prefix='/wissi-viz', dependencies=[...])
# HTML: <script type="module" src="/wissi-viz/static/index.js"></script> <wissi-viz api="/wissi-viz" slides="case1">
```

## 4. A separate tile server for any front end (path D)

Run the tile server on its own (in a container), with CORS limited to your app's origin:

```python
# tiles.py — uvicorn tiles:app --port 8010   (or hypercorn with TLS for HTTP/2)
from wissi_viz import SlideRegistry, TileService, create_app
registry = SlideRegistry(TileService(disk_dir='/cache/wissi-viz'))
registry.add_directory('/data/slides')                  # and/or connector.register(registry, image_id)
app = create_app(registry, cors='https://app.example.org')
```

Then embed it in any page. This exact setup, a plain page on one origin and tiles from another, is covered by `tests/test_browser.py::test_cross_origin_embed`:

```html
<script type="module" src="https://tiles.example.org/wissi-viz/static/index.js"></script>
<wissi-viz id="v" api="https://tiles.example.org/wissi-viz" slides="case1,case2" layout="1x2"
           style="display:block;height:80vh"></wissi-viz>
<script type="module">
  const v = document.getElementById('v');
  v.addEventListener('objectclick', (e) => console.log(e.detail));     // {layer, index, class, properties, x, y, …}
  v.addEventListener('viewchange', (e) => console.log(e.detail));      // {pane, x, y, zoom, mpp, magnification, bounds}
  v.goTo('all', 1200, 800, { units: 'um', magnification: 20 });
</script>
```

The framework snippets below use the standard custom-element pattern. They were not run in this repository's tests.

**React**
```jsx
import { useEffect, useRef } from 'react';
// load once, e.g. in index.html: <script type="module" src="https://tiles.example.org/wissi-viz/static/index.js">
export function SlideGrid({ slides, onObject }) {
  const ref = useRef(null);
  useEffect(() => { ref.current?.setSlides(slides); }, [slides]);
  useEffect(() => {
    const el = ref.current, h = (e) => onObject?.(e.detail);
    el.addEventListener('objectclick', h);
    return () => el.removeEventListener('objectclick', h);
  }, [onObject]);
  return <wissi-viz ref={ref} api="https://tiles.example.org/wissi-viz" style={{ display: 'block', height: '80vh' }} />;
}
```

**Vue 3**: tell the compiler it's a custom element (`compilerOptions.isCustomElement = (t) => t === 'wissi-viz'`), then:
```vue
<wissi-viz ref="v" api="https://tiles.example.org/wissi-viz" :slides="ids.join(',')" @objectclick="onObject" style="display:block;height:80vh" />
```

**Streamlit**
```python
import streamlit.components.v1 as components
components.html(f'''<script type="module" src="https://tiles.example.org/wissi-viz/static/index.js"></script>
<wissi-viz api="https://tiles.example.org/wissi-viz" slides="{','.join(ids)}" style="display:block;height:760px"></wissi-viz>''',
                height=780)
```

**Dash, Panel, Jupyter:** embed a page served by the tile server, for example `html.Iframe(src='https://tiles.example.org/')` or `IPython.display.IFrame(...)`. You can also render the same `<script>` + `<wissi-viz>` HTML wherever the framework allows raw HTML with scripts.

**Django or Flask (WSGI):** use path D. Put the two tags in your template and run the tile server as a sidecar ASGI service.

## 5. Connecting a tile API (PathCore, HALO Link, …)

You write one class. wissi-viz handles the rest: caching, retries, concurrency limits, pyramid mapping, tile overlap and padding, rendering and annotations. `examples/tile_api/` contains a complete, runnable simulation:

| File | Role |
|---|---|
| `mock_tile_api.py` | a stand-in **vendor service** (a separate process): bearer token, 16 images in four pyramid styles (DeepZoom 254+1 overlap; 512 tiles with coarsest-first opaque level ids; 16-bit PNG channel tiles; server-rendered fluorescence JPEGs), 40–120 ms latency, a 429 limit with `Retry-After`, 1% random 503s, 404 for empty tiles, ROIs and cell tables in µm |
| `connector.py` | **your code**: `DemoApiConnection(RemoteConnection)`, built on httpx (~90 lines) |
| `app.py` | a NiceGUI app showing 1–16 API images with live upstream statistics |
| `bench_api.py` | a cold-vs-warm browser benchmark over 1–16 panes |

```bash
pip install "wissi-viz[nicegui,remote]"          # remote = httpx (BSD-3); add [bench] for the benchmark
python examples/tile_api/mock_tile_api.py slides/*.svs slides/*.czi --port 9000 --token secret   # the "vendor"
python examples/tile_api/app.py --api http://127.0.0.1:9000 --token secret                        # your app → :8100
```

### The contract

```python
class MyVendor(RemoteConnection):
    id_prefix = 'halo'          # slide ids become 'halo-<image id>'
    max_concurrency = 10        # stay under the vendor's rate limit
    retries = 4

    def image_info(self, image_id) -> RemoteImageInfo: ...          # required
    def tile(self, image_id, level, col, row, channel=None): ...    # required: bytes | ndarray | None
    def annotations(self, image_id): ...                            # optional: [AnnotationPayload]
    def objects(self, image_id): ...                                # optional: [AnnotationPayload] (cell tables)

sid = MyVendor(client).register(registry, 'image-123')   # lazy: nothing is fetched until a pane shows it
```

| The vendor… | Put this in the connector |
|---|---|
| lists explicit levels (any order) with its own ids | `RemoteImageInfo(levels=[(w, h), …], level_ids=[…])`. Your ids come back in `tile(level=…)`. |
| gives a downsample list | `RemoteImageInfo(downsamples=[1, 4, 16, …])` |
| speaks DeepZoom (`.dzi`) | `RemoteImageInfo(deepzoom=True, tile_size=254, overlap=1)`, or `RemotePyramid.from_dzi(xml)` |
| pads edge tiles | nothing; they are cropped automatically, on the GPU for JPEG/PNG bytes |
| serves JPEG/PNG/WebP | return the bytes; they reach the browser untouched |
| serves 16-bit channels | `mode='channels'`, `channels=[…]`, `dtype='uint16'`; return an array per channel |
| returns 404 for empty tiles | raise `TileNotFound` or return `None`; the tile is drawn as background and cached |
| answers 429/503 with `Retry-After` | raise `RetryAfter(seconds)`; the call is retried politely |
| fails transiently (5xx, timeouts) | raise anything; retried with back-off, then HTTP 502 to the browser, which retries later |
| has async clients | write `async def`; call `conn.bind_loop()` at startup if the client is tied to NiceGUI's loop |
| changes images | set `version=` (modified time or ETag) so every cache key changes |
| gives coordinates in µm or at a level | `AnnotationPayload(units='um', downsample=…, offset=…)` |
| exports objects (XMin/XMax/YMin/YMax, centroids, phenotypes) | `AnnotationPayload(data, format='table')`; measurements stay clickable |
| lets the browser authenticate itself (cookies) | optional `browser_tile_url=…`: the browser fetches native RGB levels directly. Otherwise tiles are proxied, which is the safe default. |

Metadata and annotation calls follow the same retry policy as tiles. A transient 503 while opening a slide no longer fails it; the simulation found that one.

### Verified by the simulation

- **Pixels** (`tests/test_tile_api.py`):
  - 16-bit channel tiles from the API equal the source file exactly at every level.
  - DeepZoom tiles align to 0 px; the best match in a ±2 px search is at offset 0.
- **Fault handling:**
  - 429s, 503s and 404s all occur and are all absorbed (`errors == 0`).
  - A second pass over the same tiles makes **zero** upstream requests.
  - A bad token fails cleanly.
- **Annotations:** ROIs and cell tables arrive in µm and land at the right pixel positions.
- **Two viewer bugs found and fixed** (regression tests in `tests/test_browser.py`):
  - Over HTTP/2, tiles waiting for GPU upload were requested again: 5,499 requests for 141 textures. Fixed, and the test checks that requests ≈ textures for both protocols.
  - A tile that failed while the viewer was idle was never retried, so one 503 could leave a pane blurry. Fixed with per-tile exponential back-off (0.5 s → 30 s) and a wake-up timed to the back-off. The test fails on the old code.

**Benchmarks:** the browser numbers from `bench_api.py` are in `benchmarks/results-tile-api.md`.

## 6. Security checklist

- **Protect the tile routes:** use `configure(dependencies=[…])` for NiceGUI, or `include_router(..., dependencies=[…])` in your own app. Without this, anyone who can reach the server can fetch tiles by slide id.
- **Keep vendor tokens server-side:** they live in your `RemoteConnection` and never reach the browser.
- **Restrict CORS:** for path D, set it to your app's origin, not `'*'`.
- **Use opaque slide ids:** `registry.add(src, slide_id=…)`. The browser can only address registered ids; file paths are never accepted from the client.
- **Decide on annotation editing:** `allow_edit=False` disables the annotation-append endpoint used by the drawing tools.

## 7. Deployment

```yaml
# docker-compose.yml (sketch)
services:
  wissi:                      # your NiceGUI app with wissi-viz inside
    build: .
    volumes: [ "/data/slides:/data/slides:ro", "wissi-cache:/cache" ]
    environment: [ "TILE_API_TOKEN=${TILE_API_TOKEN}" ]
  proxy:                      # HTTP/2 + TLS in front (Caddy, Traefik or nginx with http2)
    image: caddy:2
    ports: [ "443:443" ]
    command: caddy reverse-proxy --from wissi.example.org --to wissi:8080
volumes:
  wissi-cache: {}
```

- **HTTP/2 matters for many panes.** Browsers open only 6 HTTP/1.1 connections per host. Over HTTP/2 the viewer detects multiplexing and keeps up to 24 tile requests in flight. Measurements are in `benchmarks/results-tile-api.md`.
- **Cache volume:** the disk tile cache (`cache_dir`) survives restarts. That matters most for API-backed slides, where every cache miss is a network round-trip.
- **Scaling:** one NiceGUI process serves the UI and the tiles. If tile decoding saturates the CPU, move it to a separate tile server (path D) and scale that alone. The viewer only needs `api=` pointed at it.
