# wissi-viz integration probe: halolink connector, mock modality (in-process, 40,120 ms latency)

_2026-10-26 16:39 · images: halo-1, halo-2 · panes [1, 4, 9, 16] · connections [5, 8, 10] · users 1_

**API calls: 954 of 1000 allowed** (annotations 2, info 2, objects 2, tile 948)

## Recommended settings

| setting | value | why |
|---|---|---|
| `max_concurrency` | **8** | 72.9 tiles/s at 8 concurrent calls, within 10 % of the best clean rate (72.9/s) |
| `timeout_s` | **10** | 5 × the worst p99 call latency (151 ms), between 10 and 60 s |
| `retries` | **3** | 13 transient failures (429 / 5xx / timeouts / retried calls) across all phases |
| `http2` | **optional** | 4 panes: view p95 797 → 827 ms with 5 → 10 connections; 9 panes: view p95 1513 → 1803 ms with 5 → 10 connections; 16 panes: view p95 1616 → 1549 ms with 5 → 10 connections → little gain from more connections; HTTP/1.1 is adequate (HTTP/2 still recommended behind a proxy) |
| `comfortable_panes` | **4** | largest pane count whose first views are sharp within 1500 ms (p95) at every tested connection count |
| `memory_mb` | **1024** | a 16-pane session fetched 2.3 MB; × 10 expected concurrent users × 2 for history, rounded to a power of two (≥ 1 GB) |
| `disk_cache` | **True** | every cache miss is an API round trip; a persistent cache_dir keeps tiles across restarts |
| `workers` | **16** | server threads mostly wait on the API: at least 2 × max_concurrency |
| network | | one tile ≈ 97.7 ms (p50, sequential) · 22.4 kB · 0.22 MB/s sequential |
| cache | | cold views p95 1190 ms vs cached 94 ms (medians over runs) |
| upstream per tile | | 1.03 API calls per viewer tile request at most (≤ 1: each tile fetched once, concurrent requests coalesced); warm reruns made 0 calls |

Ready-to-paste: `recommended_settings.py`.

## 1. Smoke test

| | check | detail | ms |
|---|---|---|---|
| ✅ | [halo-1] metadata | 24000×18000, tile 254 overlap 1, 8 levels (24000×18000, 12000×9000, 6000×4500, 3000×2250, 1500×1125, 750×563 …), mode rgb | 108.2 |
| ✅ | [halo-1] mpp | (0.25, 0.25) |  |
| ✅ | [halo-1] pyramid depth | coarsest level 188×141 = 1 tile(s) |  |
| ✅ | [halo-1] version | halo-v1 |  |
| ✅ | [halo-1] coarsest tile (level 7, 0,0) | image/jpeg 7 KB, 188×141 for a 188×141 area: exact size | 347.4 |
| ✅ | [halo-1] middle-level tile (level 3, 5,4) | image/jpeg 22 KB, 256×256 for a 254×254 area: padded/overlap (cropped by wissi-viz) | 105.1 |
| ✅ | [halo-1] full-resolution tile (level 0, 47,35) | image/jpeg 21 KB, 256×256 for a 254×254 area: padded/overlap (cropped by wissi-viz) | 63.3 |
| ℹ️ | [halo-1] tile outside the image | error: negative dimensions are not allowed (wissi-viz never asks for these) | 81.2 |
| ✅ | [halo-1] annotations + objects | HALO annotations (1 features) — wissi-viz fetches these once per slide when a viewer first opens it (2 calls, in the background) | 0.9 |
| ✅ | [halo-2] metadata | 24000×18000, tile 254 overlap 1, 8 levels (24000×18000, 12000×9000, 6000×4500, 3000×2250, 1500×1125, 750×563 …), mode rgb | 74.9 |
| ✅ | [halo-2] mpp | (0.25, 0.25) |  |
| ✅ | [halo-2] pyramid depth | coarsest level 188×141 = 1 tile(s) |  |
| ✅ | [halo-2] version | halo-v1 |  |
| ✅ | [halo-2] coarsest tile (level 7, 0,0) | image/jpeg 7 KB, 188×141 for a 188×141 area: exact size | 184.2 |
| ✅ | [halo-2] middle-level tile (level 3, 5,4) | image/jpeg 20 KB, 256×256 for a 254×254 area: padded/overlap (cropped by wissi-viz) | 78.4 |
| ✅ | [halo-2] full-resolution tile (level 0, 47,35) | image/jpeg 23 KB, 256×256 for a 254×254 area: padded/overlap (cropped by wissi-viz) | 80.8 |
| ℹ️ | [halo-2] tile outside the image | error: negative dimensions are not allowed (wissi-viz never asks for these) | 87.0 |
| ✅ | [halo-2] annotations + objects | HALO annotations (1 features) — wissi-viz fetches these once per slide when a viewer first opens it (2 calls, in the background) | 0.7 |

## 2. API, called directly

Distinct full-resolution tiles, no retries, so the service's own behaviour shows.

| run | concurrency | calls | tiles/s | MB/s | p50 ms | p95 ms | p99 ms | statuses |
|---|---|---|---|---|---|---|---|---|
| sequential | 1 | 8 | 10.3 | 0.22 | 97.7 | 123.8 | 124.7 | ok 8 |
| c5 | 5 | 20 | 48.6 | 1.06 | 98.3 | 129.0 | 131.9 | ok 20 |
| c8 | 8 | 24 | 72.9 | 1.61 | 88.7 | 131.1 | 135.9 | ok 24 |
| c10 | 10 | 30 | 81.0 | 1.67 | 105.9 | 143.1 | 151.4 | ok 28, connection 2 |

## 3. Viewer sessions end to end

Each pane shows a different image and moves on its own (open, then pan / zoom / jump). *Cold*: fresh cache, everything from the source. *Warm*: the same session again, served from the wissi-viz cache. *View*: time until every tile of a move has arrived.

| panes | conn | cold view p50 / p95 ms | cold tile p95 ms | warm view p95 ms | requests | MB | errors % | API calls | API calls per tile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 285 / **414** | 145.8 | 36 | 19 | 0.4 | 0.0 | 18 (+0 warm) | 1.0 |
| 1 | 8 | 268 / **404** | 218.5 | 31 | 19 | 0.4 | 0.0 | 18 (+0 warm) | 1.0 |
| 1 | 10 | 244 / **354** | 228.4 | 36 | 19 | 0.4 | 0.0 | 18 (+0 warm) | 1.0 |
| 4 | 5 | 543 / **797** | 208.1 | 73 | 41 | 0.4 | 0.0 | 37 (+0 warm) | 1.0 |
| 4 | 8 | 558 / **866** | 237.1 | 73 | 41 | 0.4 | 0.0 | 37 (+0 warm) | 1.0 |
| 4 | 10 | 547 / **827** | 269.5 | 73 | 42 | 0.4 | 0.0 | 38 (+0 warm) | 1.0 |
| 9 | 5 | 987 / **1513** | 191.5 | 120 | 80 | 0.8 | 0.0 | 72 (+0 warm) | 1.014 |
| 9 | 8 | 1104 / **1670** | 342.6 | 136 | 83 | 0.8 | 0.0 | 75 (+0 warm) | 1.014 |
| 9 | 10 | 1155 / **1803** | 345.9 | 153 | 86 | 0.9 | 0.0 | 79 (+0 warm) | 1.026 |
| 16 | 5 | 771 / **1616** | 225.6 | 116 | 147 | 1.5 | 0.0 | 133 (+0 warm) | 1.015 |
| 16 | 8 | 1076 / **1549** | 328.1 | 158 | 163 | 1.7 | 0.0 | 150 (+0 warm) | 1.02 |
| 16 | 10 | 879 / **1549** | 362.8 | 172 | 197 | 2.3 | 0.0 | 183 (+0 warm) | 1.011 |

## Notes

- The load generator opens up to --conn connections per user over HTTP/1.1, like a browser does behind an HTTP/2 proxy (a browser on plain HTTP/1.1 is limited to 6 per host).
- Request caps per run are small by design (the API-call budget); view latencies from few moves are indicative, not precise.
- Registry warm-up (background pre-fetch of coarse levels) was off, so every API call is user-driven and counted.
