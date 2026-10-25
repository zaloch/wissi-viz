# Objects and annotations at scale: server- vs client-side, file formats, 1–16 panes, 10–100 users

The test layer is a HALO-style object export: 1M cells × 37 columns (XMin/XMax/YMin/YMax, class, region, 30 measurements) over a 100k × 80k px slide. Every slide gets the same layer. Image tiles are blank, so only overlays are measured.

**Machine:** a 2-vCPU container with **no GPU**. Chromium rasterises on the CPU and shares those cores with the server and the load generator. Absolute times are pessimistic and the comparisons are fair. Bytes, memory and request counts do not depend on the hardware.

## Where the work happens: three designs

| | legacy (≤ 0.4) | **server pyramid** (default) | **client pyramid** (`setLod({overlays: 'client'})`) |
|---|---|---|---|
| **The server** | buckets objects, ships one per-object overview (21 B/object) | buckets objects and builds the **object pyramid** once (cached on disk, memory-mapped) | builds a compact column file once (17 B/object) |
| **The browser downloads** | the whole overview, every object, for every slide | the tiles its panes show: aggregated dots ≈ one per 4 screen px, individual cells only where zoomed in | every object of every open layer, once |
| **The browser computes** | nothing: it draws 10⁶ dots per pane per frame | nothing: it draws a screen-sized number of dots | the same pyramid, in a Web Worker (O(n) counting sorts) |
| **Zoomed-in cells** | 12-vertex polygons, ~200 B each | GPU-drawn ellipses/rects from their bounding box, 19 B each | the same shader, from local data |
| **Clicks** | the browser scans the overview | the browser zoomed in; the server for aggregated dots | the browser, at any zoom |

## Browser A/B: 1 and 16 panes

`benchmarks/overlay_bench.py --mode legacy|new --overlays server|client`

| 1M cells per slide | legacy, 1 pane | server pyramid, 1 pane | client pyramid, 1 pane | legacy, 16 panes | server pyramid, 16 panes | client pyramid, 16 panes |
|---|---|---|---|---|---|---|
| overlays drawn after | 2.1 s | **1.2 s** | 3.3 s | 79.0 s | **1.5 s** | 32.1 s |
| downloaded at open (gzip) | 9.09 MB | **0.43 MB** | 7.82 MB | 145.41 MB | **0.66 MB** | 124.54 MB |
| browser JS heap | 23 MB | **3 MB** | 72 MB | 323 MB | **5 MB** | 1109 MB |
| GPU memory for overlays | 20 MB | 0.6 MB | 0.5 MB | 320.7 MB | 1.1 MB | 1 MB |
| client worker build | — | — | 1.9 s | — | — | 12.2 s |
| frame while panning, p50 / p95 | 579 / 765 ms | 103 / 137 ms | 96 / 116 ms | ~9 s frames | 106 / 136 ms | 104 / 154 ms |
| overlay data for zooms 2.5× → 10× → 40× | 0.24 MB | 0.37 MB | **0** | 3.85 MB | 2.75 MB | **0** |
| click → object details, fit / 40× | 653 / 1143 ms | 117 / 171 ms | 117 / **94** ms | 8215 / — ms | 99 / 1395 ms | 113 / 159 ms |

Both click timings include fetching the cell's measurement columns (`/feature`), which always come from the server. The 1.4 s server-mode click at 40× with 16 panes was the details request waiting behind 16 panes' tile requests on the browser's 6 HTTP/1.1 connections.

**Reading it:**
- **The server pyramid wins whenever there is a lot to show.**
  - Payload, memory and frame cost scale with the **screen**, not with the object count.
  - Overlays for 16 panes × 1M cells appear in 1.5 s, cost 0.66 MB to download and 5 MB of browser memory.
- **The client pyramid's advantage is real but small:** zero overlay requests while navigating, and local clicks (94 ms instead of 171 ms zoomed in).
  - It pays for that by downloading and holding every object of every open layer: 7.8 MB, 72 MB of heap and a 1.9 s build per 1M cells.
  - With 16 slides that becomes 125 MB, **1.1 GB of browser memory** and 32 s before anything is drawn.
  - Zoom steps were not faster in practice: the image tiles a zoom needs take as long as the overlay tiles, which load alongside them.
- **So the default stays the server pyramid.** `overlays: 'auto'` uses client mode only for layers ≤ 300k objects, while the viewer holds < 1M objects client-side (≈ 70 MB). That suits a single slide with a moderate cell table, a slow or unreliable link, or offline review after loading.

## 10–100 users, 4 slides each, 1M cells per slide

`examples/tile_api/load_sim.py --backend synthetic --overlay-table cells.parquet [--overlay-mode client]`

Users fetch image tiles plus overlays exactly as the viewer would (layer index, pyramid tiles, outline buckets, or the whole columns in client mode), with ≤ 300 requests each. The load generator shares the 2 vCPUs, so above ~50 users its own lag (the last column) inflates latencies.

| overlays | users | requests | req/s | MB sent (decoded) | latency p50 / p95 ms | **view p50 / p95 ms** | overlay requests p95 ms | errors | load-gen lag p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| legacy | 10 | 3000 | 77.3 | 869.5 | 9.4 / 195.0 | 31 / 303 | bucket 300, index 10672 | 0.0% | 83.5 |
| legacy | 25 | 7500 | 156.4 | 2028.0 | 15.1 / 576.4 | 44 / 1072 | bucket 659, index 12888 | 0.0% | 148.3 |
| server pyramid | 10 | 3000 | 130.8 | 114.8 | 10.4 / 35.5 | 42 / 138 | bucket 38, index 76, ptile 48 | 0.0% | 4.9 |
| server pyramid | 25 | 7500 | 253.1 | 264.6 | 13.1 / 53.6 | 46 / 213 | bucket 51, index 43, ptile 69 | 0.0% | 9.0 |
| server pyramid | 50 | 15000 | 566.6 | 534.1 | 81.7 / 275.2 | 281 / 1210 | bucket 209, index 192, ptile 260 | 0.0% | 51.0 |
| server pyramid | 100 | 30000 | 659.7 | 1017.7 | 345.9 / 854.1 | 1226 / 3869 | bucket 632, index 1303, ptile 726 | 0.0% | 164.0 |
| client pyramid | 10 | 3000 | 82.6 | 644.4 | 9.7 / 144.3 | 28 / 150 | columns 9422 | 0.0% | 71.5 |
| client pyramid | 25 | 7500 | 179.1 | 1521.9 | 10.2 / 431.7 | 27 / 771 | columns 8444 | 0.0% | 109.2 |
| client pyramid | 50 | 15000 | 278.1 | 3060.4 | 16.6 / 902.2 | 48 / 1429 | columns 21341 | 0.0% | 305.0 |

- **Legacy:** each user first downloads every slide's 20 MB overview (decoded size; 9 MB gzip). Those requests alone took **10.7–12.9 s at p95**. The time for a move to be sharp is 2–5× the server pyramid's, and throughput is ~40% lower.
- **Server pyramid:** no errors at any load. The server streams small, cached tiles: p95 per tile request 48 ms at 10 users, 69 ms at 25 and 260 ms at 50.
- **Read "view" carefully for client mode and legacy:** it covers moves *after* the slides are open. Both front-load a big download before any overlay appears: the "index" (legacy) and "columns" (client) p95 above.
- **Client pyramid:** users see no overlays until their 4 × 7.8 MB of columns arrive, **9.4 s at p95 with 10 users and 21 s with 50**, against 76–190 ms for the server pyramid's first index and tiles. After that, overlays cost the server nothing, and the moves themselves are about as quick as with the server pyramid. At 100 users the *load generator* ran out of memory holding 400 layers' columns. That is a fair preview of the network and memory a fleet of client-mode browsers needs.

## Storage formats

The same 1M-cell table in every format (`benchmarks/object_formats.py`). Each operation runs in a fresh process, and peak memory is that process's peak RSS. "Viewer columns" means geometry + class.

| format | size MB | write s | load all 37 columns, s (peak MB) | load viewer columns, s (peak MB) | viewport query ms | one cell (click) ms |
|---|---|---|---|---|---|---|
| CSV, pandas | 285 | 4.1 | 5.73 (1239) | 3.23 (995) | — | — |
| TSV, pandas | 285 | 4.7 | 5.78 (1239) | 2.80 (995) | — | — |
| CSV, pyarrow (wissi-viz now) | 285 | 4.1 | 1.94 (951) | 0.81 (694) | — | — |
| TSV, pyarrow (wissi-viz now) | 285 | 4.7 | 1.77 (945) | 0.91 (696) | — | — |
| Parquet (zstd) | 274 | 3.4 | 1.04 (841) | **0.40 (260)** | 51.8 | 25.2 |
| Parquet, Hilbert-sorted rows | 208 | 3.7 | 1.17 (778) | 0.47 (260) | **9.2** | 30.0 |
| Arrow IPC / Feather, memory-mapped | 273 | **0.2** | **0.55 (390)** | **0.39 (259)** | 15.3 | **0.18** |
| Feather, lz4 | 197 | 0.6 | 0.75 (531) | 0.38 (211) | — | — |
| SQLite + R*Tree | 353 | 19.5 | 14.56 (2186) | 2.48 (444) | 23.5 | **0.03** |
| DuckDB | **102** | 0.9 | 0.99 (894) | 0.43 (283) | **8.3** | 3.6 |
| PostgreSQL 16 + GiST on box | 393 | 14.5 | 6.33 (1246) | 1.76 (337) | 22.5 | 0.14 |
| **wissi-viz compiled layer cache** | 181 | (3.6 build) | — | **0.035 (memory-mapped)** | — | — |

**What to use:**
- **The viewer:** export **Parquet**, or Arrow/Feather. wissi-viz reads it once into the compiled cache (`cached_object_layer`), and every later open takes ~35 ms with no parsing.
  - CSV and TSV are equally slow. pandas costs 3–7× more than pyarrow, so wissi-viz now parses CSV/TSV with pyarrow.
- **Editable, shared annotations** (drawings, review state, ROIs: thousands, not millions):
  - **SQLite in WAL mode** inside the container: single-row reads in 0.03 ms and safe concurrent writers.
  - **PostgreSQL** when several app replicas or other services write.
  - Neither is a good *bulk* store for 10⁶-object layers: 2.5–14 s and up to 2.2 GB to load one.
- **Cohort analytics** across many slides: **DuckDB** directly over the Parquet files. It is the smallest on disk, does 8 ms window queries, and needs no server.

## Server pipeline per 1M-cell layer

| | legacy | new |
|---|---|---|
| build the layer from the table | 1.1–1.5 s | 0.5 s (columnar shapes) |
| compile (buckets + pyramid) | 0.9 s | 1.5–1.8 s (adds the pyramid) |
| reopen from the compiled cache | — | **0.035 s** |
| memory kept by the server | ≈ 650 MB (460 MB measured + ~190 MB of float64 columns it referenced) | **225 MB** (float32 + categorical columns; columnar shapes) |
| outline data for a 10× view | 2.36 MB (0.66 MB gzip) | 0.13 MB (0.05 MB gzip) |

## Bugs found and fixed along the way

- **16-pane layers stuck in "loading" forever.**
  - The scheduler cancelled a layer's overview request while it was loading, because it was not "in view" yet.
  - A response already on its way was then dropped, and the layer never retried.
  - Now the request stays wanted until it arrives.
- **Stale overlays after a restart.**
  - Layer versions key *immutable* browser caches, but they restarted at 0 in every process.
  - A changed layer could get an old number, and browsers would keep showing the old tiles.
  - Versions now start at a random offset. Cached layers derive theirs from the content key.
- **`GL_INVALID_OPERATION` on aggregated tiles.** Integer vertex inputs left unbound need integer defaults (`vertexAttribI4ui`).
- **Inconsistent "other" colour** for layers with more than 8 classes: the server and the shader picked different colours. Both now use the most common of the rarer classes.
- **Client worker:** a flag overwrote each tile's histogram array, so aggregated tiles had no class data. Found by the GPU-memory counter reading NaN.

Tests (`tests/test_overlays.py`):
- pyramid accounting, i.e. every object counted at every level;
- the 19-byte shape detail;
- hidden-class hit tests;
- byte-identical save/open;
- cache invalidation;
- reader detection;
- 16 panes loading, drawing, class hiding and clicking;
- client mode: the same pyramid built in the browser, local clicks, no overlay requests after the download.

Raw data: `*-container.json` next to this file.
