# Tile-API simulation: 1–16 panes from a remote vendor API

This simulates wissi-viz embedded in an app whose slides live behind a vendor tile API, as PathCore or HALO Link would serve them. The pieces are in `examples/tile_api/`:

```
browser (Chromium, <wissi-viz>)  ──HTTP/2 or HTTP/1.1──▶  wissi-viz server (registry + caches)
                                                              │  DemoApiConnection (httpx, max 10 in flight, 4 retries)
                                                              ▼
                                                   mock vendor API (separate process)
```

**Vendor simulation** (`mock_tile_api.py`):
- Bearer token.
- 40–120 ms random latency per request.
- 12 concurrent requests allowed, then HTTP 429 with `Retry-After`.
- 1% random HTTP 503.
- HTTP 404 for empty tiles.
- 16 images built from 6 source files (SVS 64k, OME-TIFF 40k, CZI 24k, a real Aperio CMU-1 region, synthetic H&E and IF). They are exposed in four pyramid styles in rotation:
  - DeepZoom with 254 px tiles and 1 px overlap;
  - 512 px tiles with coarsest-first opaque level ids;
  - 16-bit PNG per-channel tiles;
  - server-rendered fluorescence JPEGs.

**Scenario** (`bench_api.py`):
- **Open:** open N panes and wait until every pane is sharp.
- **Zoom:** jump to 10× and wait until sharp.
- **Pan:** pan at 900 screen-px/s for 6 s.
- **Revisit:** pan back over the same path.
- **Cold:** first pass; every tile comes from the vendor.
- **Warm:** the same views again; tiles come from the wissi-viz cache.

**Machine:** a 2-vCPU cloud container with **no GPU**. Chromium falls back to SwiftShader software WebGL at a 1600×1000 viewport. The browser, the wissi-viz server and the mock vendor all share those 2 cores. Frame rates here are therefore floors set by software rasterisation, not by the viewer (see "Viewer JS ms/frame"). GPU frame rates were not measured here; in the same container a single 800×500 pane holds 60 fps even on SwiftShader (see [results-container.md](results-container.md)).

Raw JSON: `api_bench_h2.json` and `api_bench_h1.json` from `python examples/tile_api/bench_api.py --panes 1,4,9,16 --seconds 6 --magnification 10 [--http2]`.

## HTTP/2 (TLS via hypercorn, as behind Caddy/Traefik/nginx)

**Cold.** Upstream: 1,870 requests, 23 × 503 absorbed, 0 errors.

| panes | open→sharp ms | 10×→sharp ms | pan fps (p95 ms) | viewer JS ms/frame p50 (p95) | sharp while panning % | blank % | settle ms | revisit fps / sharp % |
|---|---|---|---|---|---|---|---|---|
| 1 | 2939 | 1531 | 8.6 (212) | 0.9 (4.0) | 99.9 | 0.0 | 81 | 10.0 / 100 |
| 4 | 483 | 2272 | 2.3 (558) | 5.8 (12.4) | 85.4 | 0.0 | 1158 | 6.7 / 100 |
| 9 | 3391 | 3639 | 2.9 (490) | 9.9 (17.4) | 62.6 | 0.0 | 1156 | 3.1 / 97.4 |
| 16 | 4978 | 2721 | 2.9 (496) | 8.8 (16.1) | 76.4 | 0.0 | 1560 | 4.0 / 100 |

**Warm.** Upstream: 159 requests.

| panes | open→sharp ms | 10×→sharp ms | pan fps (p95 ms) | viewer JS ms/frame p50 (p95) | sharp while panning % | blank % | settle ms | revisit fps / sharp % |
|---|---|---|---|---|---|---|---|---|
| 1 | 312 | 716 | 10.0 (139) | 0.8 (6.7) | 100.0 | 0.0 | 106 | 10.0 / 100 |
| 4 | 483 | 946 | 5.5 (270) | 3.1 (8.8) | 99.9 | 0.0 | 190 | 7.5 / 100 |
| 9 | 625 | 1310 | 4.3 (364) | 7.8 (13.4) | 92.9 | 0.0 | 264 | 6.0 / 100 |
| 16 | 996 | 1089 | 5.0 (259) | 7.1 (12.6) | 99.0 | 0.0 | 158 | 7.5 / 100 |

## HTTP/1.1 (plain uvicorn, no proxy)

**Cold.** Upstream: 1,444 requests, 20 × 503 absorbed, 0 errors.

| panes | open→sharp ms | 10×→sharp ms | pan fps (p95 ms) | viewer JS ms/frame p50 (p95) | sharp while panning % | blank % | settle ms | revisit fps / sharp % |
|---|---|---|---|---|---|---|---|---|
| 1 | 4508 | 2611 | 3.8 (365) | 2.0 (6.8) | 86.4 | 0.0 | 216 | 10.0 / 100 |
| 4 | 1300 | 2891 | 3.2 (388) | 2.2 (8.2) | 52.5 | 0.0 | 2266 | 3.8 / 93.6 |
| 9 | 4132 | 3795 | 3.5 (357) | 5.6 (9.1) | 54.9 | 0.0 | 2784 | 3.1 / 86.0 |
| 16 | 5486 | 3799 | 3.7 (417) | 4.8 (12.9) | 67.6 | 0.0 | 2221 | 3.5 / 88.4 |

**Warm.** Upstream: 259 requests.

| panes | open→sharp ms | 10×→sharp ms | pan fps (p95 ms) | viewer JS ms/frame p50 (p95) | sharp while panning % | blank % | settle ms | revisit fps / sharp % |
|---|---|---|---|---|---|---|---|---|
| 1 | 1598 | 1672 | 10.0 (150) | 1.0 (4.5) | 100.0 | 0.0 | 115 | 10.0 / 100 |
| 4 | 1452 | 2264 | 3.8 (332) | 2.0 (4.9) | 65.7 | 0.0 | 1219 | 5.5 / 99.2 |
| 9 | 1146 | 2072 | 4.3 (283) | 4.0 (7.9) | 64.5 | 0.0 | 1054 | 4.3 / 98.7 |
| 16 | 986 | 1958 | 5.0 (250) | 3.5 (8.4) | 78.3 | 0.0 | 987 | 5.0 / 98.9 |

## What this shows

- **Never blank.** In every run, 0% of the screen was empty while panning. The coarser cached level is always drawn underneath until the sharp tile arrives.
- **The vendor is hit once per tile.**
  - The warm pass makes 159–259 upstream requests, against 1,444–1,870 cold. The remainder is tiles at the edge of slightly different pan paths and the predictive margin.
  - Every 429 and 503 was absorbed: 0 errors reached the browser.
- **HTTP/2 is worth it for multi-pane grids.** With 16 warm panes, HTTP/2 gives:

  | | HTTP/2 | HTTP/1.1 |
  |---|---|---|
  | 10×→sharp | 1.1 s | 2.0 s |
  | sharp while panning | 99% | 78% |
  | settle | 158 ms | 987 ms |

  Browsers cap HTTP/1.1 at 6 connections per host. Over HTTP/2 the viewer keeps up to 24 requests in flight. Put Caddy, Traefik or nginx (`http2 on`) in front.
- **Cold open is bound by vendor latency, not by wissi-viz.** Sixteen slides × (metadata + coarse tiles) at 40–120 ms each, through a 10-request concurrency cap, takes about 5 s. Raising `max_concurrency` (if the vendor allows it) or pre-registering slides with `warm=True` before the user opens them shortens this.
- **Frame rate here is the software renderer's.** Viewer JavaScript costs 1–10 ms per frame. The 2–10 fps is SwiftShader rasterising 1600×1000 on a CPU that the mock vendor and tile server also share. With a real GPU, rasterisation should stop being the limit, but run `bench_api.py --headed` on your hardware to confirm.

## Bugs this simulation found (fixed, with regression tests)

1. **Duplicate requests under HTTP/2.**
   - Tiles that had arrived but were waiting for their GPU upload slot were not tracked, so they were requested again: 5,499 requests for 141 textures.
   - Fix: a `queued` set plus upload back-pressure.
   - `test_no_duplicate_requests_and_never_blank[h2/http1]` asserts that requests ≈ textures.
2. **Failed tiles never retried when the viewer was idle.**
   - A 503 on a needed tile set a 2 s back-off, but the only wake-up fired at 1.1 s, and an idle viewer never rendered again. The pane stayed blurry: a warm 16-pane open timed out.
   - Fix: per-tile exponential back-off (0.5 s → 30 s) with a wake-up at expiry.
   - `test_failed_tile_is_retried_until_sharp` fails on the old code and passes on the fix.
3. **Connector: metadata and annotation calls were not retried.**
   - One 503 while opening a slide failed the whole pane.
   - Fix: `image_info`, `annotations` and `objects` now go through the same retry/`RetryAfter` policy as tiles.
