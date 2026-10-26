# wissi-viz integration probe: pathcore connector, mock modality (in-process, 40,120 ms latency)

_2026-10-26 16:41 · images: pathcore-1, pathcore-2 · panes [1, 4, 9, 16] · connections [5, 8, 10] · users 1_

**API calls: 881 of 1000 allowed** (annotations 2, info 2, tile 877)

## Recommended settings

| setting | value | why |
|---|---|---|
| `max_concurrency` | **8** | 41.2 tiles/s at 8 concurrent calls, within 10 % of the best clean rate (41.9/s) |
| `timeout_s` | **10** | 5 × the worst p99 call latency (305 ms), between 10 and 60 s |
| `retries` | **2** | no transient failures seen |
| `http2` | **optional** | 4 panes: view p95 3347 → 3477 ms with 5 → 8 connections → little gain from more connections; HTTP/1.1 is adequate (HTTP/2 still recommended behind a proxy) |
| `comfortable_panes` | **1** | no pane count met 1500 ms p95 on first view |
| `memory_mb` | **1024** | a 9-pane session fetched 7.8 MB; × 10 expected concurrent users × 2 for history, rounded to a power of two (≥ 1 GB) |
| `disk_cache` | **True** | every cache miss is an API round trip; a persistent cache_dir keeps tiles across restarts |
| `workers` | **16** | server threads mostly wait on the API: at least 2 × max_concurrency |
| network | | one tile ≈ 123.5 ms (p50, sequential) · 94.9 kB · 0.76 MB/s sequential |
| cache | | cold views p95 3304 ms vs cached 62 ms (medians over runs) |
| open cost | | opening a slide costs ≈ 30 API calls (its coarsest API level is 30 tiles; wissi-viz builds the overview levels from them), once per slide while it stays in the cache |
| upstream per tile | | up to 2.5 API calls per viewer tile: wissi-viz assembles some tiles from several API tiles — typically overview levels below a shallow API pyramid (see "pyramid depth" in the smoke test), or API tiles smaller than wissi-viz tiles. Warm reruns made 0 calls. |

Ready-to-paste: `recommended_settings.py`.

## 1. Smoke test

| | check | detail | ms |
|---|---|---|---|
| ✅ | [pathcore-1] metadata | 24000×18000, tile 256, 3 levels (24000×18000, 6000×4500, 1500×1125), mode rgb | 120.0 |
| ✅ | [pathcore-1] mpp | (0.25, 0.25) |  |
| ⚠️ | [pathcore-1] pyramid depth | coarsest level 1500×1125 = 30 tile(s): the first (zoomed-out) view of each slide needs all 30 — wissi-viz builds the smaller overview levels from them, once per slide (then cached). If the API has a thumbnail / overview endpoint, expose it as an extra coarse level in image_info(); a persistent cache_dir makes this a one-time cost per slide |  |
| ✅ | [pathcore-1] version | pc-etag-1 |  |
| ✅ | [pathcore-1] coarsest tile (level 2, 2,2) | image/png 95 KB, 256×256 for a 256×256 area: exact size | 347.4 |
| ✅ | [pathcore-1] middle-level tile (level 1, 11,8) | image/png 83 KB, 256×256 for a 256×256 area: exact size | 104.3 |
| ✅ | [pathcore-1] full-resolution tile (level 0, 46,35) | image/png 68 KB, 256×256 for a 256×256 area: exact size | 85.0 |
| ℹ️ | [pathcore-1] tile outside the image | returns data (bytes) — fine if intentional | 88.3 |
| ✅ | [pathcore-1] annotations + objects | PathCore annotations (1 features) — wissi-viz fetches these once per slide when a viewer first opens it (2 calls, in the background) | 1.3 |
| ✅ | [pathcore-2] metadata | 24000×18000, tile 256, 3 levels (24000×18000, 6000×4500, 1500×1125), mode rgb | 84.0 |
| ✅ | [pathcore-2] mpp | (0.25, 0.25) |  |
| ⚠️ | [pathcore-2] pyramid depth | coarsest level 1500×1125 = 30 tile(s): the first (zoomed-out) view of each slide needs all 30 — wissi-viz builds the smaller overview levels from them, once per slide (then cached). If the API has a thumbnail / overview endpoint, expose it as an extra coarse level in image_info(); a persistent cache_dir makes this a one-time cost per slide |  |
| ✅ | [pathcore-2] version | pc-etag-1 |  |
| ✅ | [pathcore-2] coarsest tile (level 2, 2,2) | image/png 92 KB, 256×256 for a 256×256 area: exact size | 141.3 |
| ✅ | [pathcore-2] middle-level tile (level 1, 11,8) | image/png 88 KB, 256×256 for a 256×256 area: exact size | 91.0 |
| ✅ | [pathcore-2] full-resolution tile (level 0, 46,35) | image/png 103 KB, 256×256 for a 256×256 area: exact size | 105.2 |
| ℹ️ | [pathcore-2] tile outside the image | returns data (bytes) — fine if intentional | 95.1 |
| ✅ | [pathcore-2] annotations + objects | PathCore annotations (1 features) — wissi-viz fetches these once per slide when a viewer first opens it (2 calls, in the background) | 0.8 |

## 2. API, called directly

Distinct full-resolution tiles, no retries, so the service's own behaviour shows.

| run | concurrency | calls | tiles/s | MB/s | p50 ms | p95 ms | p99 ms | statuses |
|---|---|---|---|---|---|---|---|---|
| sequential | 1 | 8 | 8.2 | 0.76 | 123.5 | 149.5 | 153.4 | ok 8 |
| c5 | 5 | 20 | 34.1 | 3.14 | 129.6 | 201.1 | 201.7 | ok 20 |
| c8 | 8 | 24 | 41.2 | 3.94 | 177.6 | 229.8 | 263.0 | ok 24 |
| c10 | 10 | 30 | 41.9 | 4.09 | 226.1 | 265.6 | 305.2 | ok 30 |

## 3. Viewer sessions end to end

Each pane shows a different image and moves on its own (open, then pan / zoom / jump). *Cold*: fresh cache, everything from the source. *Warm*: the same session again, served from the wissi-viz cache. *View*: time until every tile of a move has arrived.

| panes | conn | cold view p50 / p95 ms | cold tile p95 ms | warm view p95 ms | requests | MB | errors % | API calls | API calls per tile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 2069 / **3260** | 261.2 | 53 | 26 | 2.1 | 0.0 | 54 (+0 warm) | 2.16 |
| 1 | 8 | 2051 / **3136** | 322.4 | 56 | 27 | 2.1 | 0.0 | 55 (+0 warm) | 2.115 |
| 1 | 10 | 2107 / **3226** | 376.4 | 67 | 28 | 2.1 | 0.0 | 56 (+0 warm) | 2.074 |
| 4 | 5 | 254 / **3347** | 3698.3 | 55 | 56 | 1.3 | 0.0 | 132 (+0 warm) | 2.538 |
| 4 | 8 | 521 / **3477** | 1935.9 | 86 | 72 | 2.7 | 0.0 | 148 (+0 warm) | 2.176 |
| 4 | 10 | skipped: needs ≈ 154 API calls (≈ 120 to open 4 slide(s): 30 per slide) and only 126 remain — raise --max-calls | | | | | | | |
| 9 | 5 | 2282 / **9670** | 1933.7 | 140 | 171 | 7.8 | 0.0 | 342 (+0 warm) | 2.111 |
| 9 | 8 | skipped: needs ≈ 334 API calls (≈ 270 to open 9 slide(s): 30 per slide) and only 172 remain — raise --max-calls | | | | | | | |
| 9 | 10 | skipped: needs ≈ 334 API calls (≈ 270 to open 9 slide(s): 30 per slide) and only 126 remain — raise --max-calls | | | | | | | |
| 16 | 5 | skipped: needs ≈ 586 API calls (≈ 480 to open 16 slide(s): 30 per slide) and only 372 remain — raise --max-calls | | | | | | | |
| 16 | 8 | skipped: needs ≈ 586 API calls (≈ 480 to open 16 slide(s): 30 per slide) and only 172 remain — raise --max-calls | | | | | | | |
| 16 | 10 | skipped: needs ≈ 586 API calls (≈ 480 to open 16 slide(s): 30 per slide) and only 126 remain — raise --max-calls | | | | | | | |

## Notes

- The load generator opens up to --conn connections per user over HTTP/1.1, like a browser does behind an HTTP/2 proxy (a browser on plain HTTP/1.1 is limited to 6 per host).
- Request caps per run are small by design (the API-call budget); view latencies from few moves are indicative, not precise.
- Registry warm-up (background pre-fetch of coarse levels) was off, so every API call is user-driven and counted.
