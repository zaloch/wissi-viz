# wissi-viz integration probe: mock vendor API (http://127.0.0.1:59305)

_2026-10-26 16:40 · images: img-01, img-04 · panes [1, 4, 9, 16] · connections [5, 8, 10] · users 1_

**API calls: 954 of 1000 allowed** (annotations 2, info 2, objects 2, tile 948); the vendor's own counter: 954 requests

## Recommended settings

| setting | value | why |
|---|---|---|
| `max_concurrency` | **10** | 100.2 tiles/s at 10 concurrent calls, within 10 % of the best clean rate (100.2/s); throughput was still rising at the highest tested value — if the vendor allows it, test higher (e.g. --conn 10,12,16) |
| `timeout_s` | **10** | 5 × the worst p99 call latency (140 ms), between 10 and 60 s |
| `retries` | **3** | 15 transient failures (429 / 5xx / timeouts / retried calls) across all phases |
| `http2` | **optional** | 4 panes: view p95 754 → 851 ms with 5 → 10 connections; 9 panes: view p95 1714 → 1647 ms with 5 → 10 connections; 16 panes: view p95 987 → 1105 ms with 5 → 10 connections → little gain from more connections; HTTP/1.1 is adequate (HTTP/2 still recommended behind a proxy) |
| `comfortable_panes` | **16** | largest pane count whose first views are sharp within 1500 ms (p95) at every tested connection count |
| `memory_mb` | **1024** | a 16-pane session fetched 4.0 MB; × 10 expected concurrent users × 2 for history, rounded to a power of two (≥ 1 GB) |
| `disk_cache` | **True** | every cache miss is an API round trip; a persistent cache_dir keeps tiles across restarts |
| `workers` | **20** | server threads mostly wait on the API: at least 2 × max_concurrency |
| network | | one tile ≈ 109.1 ms (p50, sequential) · 41.8 kB · 0.41 MB/s sequential |
| cache | | cold views p95 919 ms vs cached 72 ms (medians over runs) |
| upstream per tile | | 1.06 API calls per viewer tile request at most (≤ 1: each tile fetched once, concurrent requests coalesced); warm reruns made 0 calls |

Ready-to-paste: `recommended_settings.py`.

## 1. Smoke test

| | check | detail | ms |
|---|---|---|---|
| ✅ | [img-01] metadata | 24000×18000, tile 254 overlap 1, 8 levels (24000×18000, 12000×9000, 6000×4500, 3000×2250, 1500×1125, 750×563 …), mode rgb | 106.3 |
| ✅ | [img-01] mpp | (0.25, 0.25) |  |
| ✅ | [img-01] pyramid depth | coarsest level 188×141 = 1 tile(s) |  |
| ✅ | [img-01] version | 1793046251-dzi |  |
| ✅ | [img-01] coarsest tile (level 7, 0,0) | image/jpeg 5 KB, 188×141 for a 188×141 area: exact size | 334.9 |
| ✅ | [img-01] middle-level tile (level 3, 5,4) | image/jpeg 14 KB, 256×256 for a 254×254 area: padded/overlap (cropped by wissi-viz) | 92.4 |
| ✅ | [img-01] full-resolution tile (level 0, 47,35) | image/jpeg 15 KB, 256×256 for a 254×254 area: padded/overlap (cropped by wissi-viz) | 70.7 |
| ℹ️ | [img-01] tile outside the image | http_400: Client error '400 Bad Request' for url 'http://127.0.0.1:59305/api/images/img-01/tiles/15/98/0' (wissi-viz never asks for these) | 90.9 |
| ✅ | [img-01] annotations + objects | none — wissi-viz fetches these once per slide when a viewer first opens it (2 calls, in the background) | 187.1 |
| ✅ | [img-04] metadata | 24000×18000, tile 512, 4 levels (24000×18000, 6000×4500, 1500×1125, 375×282), mode rgb | 84.3 |
| ✅ | [img-04] mpp | (0.25, 0.25) |  |
| ✅ | [img-04] pyramid depth | coarsest level 375×282 = 1 tile(s) |  |
| ✅ | [img-04] version | 1793046308-tiles |  |
| ✅ | [img-04] coarsest tile (level 3, 0,0) | image/jpeg 16 KB, 512×512 for a 375×282 area: padded/overlap (cropped by wissi-viz) | 289.4 |
| ✅ | [img-04] middle-level tile (level 1, 5,4) | image/jpeg 65 KB, 512×512 for a 512×512 area: exact size | 120.4 |
| ✅ | [img-04] full-resolution tile (level 0, 23,17) | image/jpeg 63 KB, 512×512 for a 512×512 area: exact size | 103.4 |
| ℹ️ | [img-04] tile outside the image | http_400: Client error '400 Bad Request' for url 'http://127.0.0.1:59305/api/images/img-04/tiles/r0-274/50/0' (wissi-viz never asks for these) | 98.5 |
| ✅ | [img-04] annotations + objects | none — wissi-viz fetches these once per slide when a viewer first opens it (2 calls, in the background) | 142.7 |

## 2. API, called directly

Distinct full-resolution tiles, no retries, so the service's own behaviour shows.

| run | concurrency | calls | tiles/s | MB/s | p50 ms | p95 ms | p99 ms | statuses |
|---|---|---|---|---|---|---|---|---|
| sequential | 1 | 8 | 11.6 | 0.41 | 109.1 | 120.7 | 122.7 | ok 7, http_503 1 |
| c5 | 5 | 20 | 50.7 | 1.91 | 93.0 | 121.6 | 139.5 | ok 20 |
| c8 | 8 | 24 | 68.3 | 2.52 | 100.5 | 125.0 | 125.8 | ok 24 |
| c10 | 10 | 30 | 100.2 | 3.8 | 84.6 | 117.4 | 122.5 | ok 30 |

## 3. Viewer sessions end to end

Each pane shows a different image and moves on its own (open, then pan / zoom / jump). *Cold*: fresh cache, everything from the source. *Warm*: the same session again, served from the wissi-viz cache. *View*: time until every tile of a move has arrived.

| panes | conn | cold view p50 / p95 ms | cold tile p95 ms | warm view p95 ms | requests | MB | errors % | API calls | API calls per tile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 425 / **645** | 198.0 | 37 | 19 | 0.3 | 0.0 | 19 (+0 warm) | 1.056 |
| 1 | 8 | 278 / **432** | 234.6 | 33 | 19 | 0.3 | 0.0 | 18 (+0 warm) | 1.0 |
| 1 | 10 | 320 / **478** | 296.9 | 44 | 19 | 0.3 | 0.0 | 19 (+0 warm) | 1.056 |
| 4 | 5 | 550 / **754** | 172.1 | 62 | 41 | 0.9 | 0.0 | 37 (+0 warm) | 1.0 |
| 4 | 8 | 550 / **773** | 266.4 | 77 | 41 | 0.9 | 0.0 | 37 (+0 warm) | 1.0 |
| 4 | 10 | 604 / **851** | 309.7 | 66 | 42 | 0.9 | 0.0 | 38 (+0 warm) | 1.0 |
| 9 | 5 | 1187 / **1714** | 344.0 | 123 | 80 | 1.5 | 0.0 | 74 (+0 warm) | 1.042 |
| 9 | 8 | 1040 / **1454** | 339.5 | 129 | 82 | 1.5 | 0.0 | 73 (+0 warm) | 1.0 |
| 9 | 10 | 1139 / **1647** | 444.4 | 171 | 86 | 1.8 | 0.0 | 78 (+0 warm) | 1.013 |
| 16 | 5 | 744 / **987** | 264.6 | 60 | 147 | 2.9 | 0.0 | 133 (+0 warm) | 1.015 |
| 16 | 8 | 894 / **1302** | 444.3 | 103 | 163 | 4.0 | 0.0 | 151 (+0 warm) | 1.027 |
| 16 | 10 | 969 / **1105** | 446.3 | 105 | 195 | 3.9 | 0.0 | 181 (+0 warm) | 1.011 |

## Notes

- The load generator opens up to --conn connections per user over HTTP/1.1, like a browser does behind an HTTP/2 proxy (a browser on plain HTTP/1.1 is limited to 6 per host).
- Request caps per run are small by design (the API-call budget); view latencies from few moves are indicative, not precise.
- Registry warm-up (background pre-fetch of coarse levels) was off, so every API call is user-driven and counted.
