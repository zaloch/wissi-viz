# wissi-viz integration probe: local files

_2026-10-26 16:46 · images: /root/.cache/wissi-probe/probe_1_24000x18000.svs, /root/.cache/wissi-probe/probe_2_24000x18000.svs · panes [1, 4, 9, 16] · connections [5, 8, 10] · users 1_

## Recommended settings

| setting | value | why |
|---|---|---|
| `http2` | **optional** | 4 panes: view p95 147 → 237 ms with 5 → 10 connections; 9 panes: view p95 357 → 321 ms with 5 → 10 connections; 16 panes: view p95 516 → 523 ms with 5 → 10 connections → little gain from more connections; HTTP/1.1 is adequate (HTTP/2 still recommended behind a proxy) |
| `comfortable_panes` | **16** | largest pane count whose first views are sharp within 1500 ms (p95) at every tested connection count |
| `memory_mb` | **1024** | a 16-pane session fetched 16.5 MB; × 10 expected concurrent users × 2 for history, rounded to a power of two (≥ 1 GB) |
| tile server | | cold tiles: 295.9 tiles/s at 10 concurrent requests (p95 95.4 ms) — the server's own capacity for never-seen tiles |
| cache | | cold views p95 279 ms vs cached 79 ms (medians over runs) |

Ready-to-paste: `recommended_settings.py`.

## 1. Smoke test

| | check | detail | ms |
|---|---|---|---|
| ✅ | [probe_1_24000x18000.svs] open | 24000×18000, 3 native levels, tile 512, mpp (0.25, 0.25), 110 MB | 17.3 |
| ✅ | [probe_2_24000x18000.svs] open | 24000×18000, 3 native levels, tile 512, mpp (0.25, 0.25), 110 MB | 1.6 |

## 2. Tile server, cold tiles

Distinct full-resolution tiles, no retries, so the service's own behaviour shows.

| run | concurrency | calls | tiles/s | MB/s | p50 ms | p95 ms | p99 ms | statuses |
|---|---|---|---|---|---|---|---|---|
| sequential | 1 | 8 | 32.5 | 1.9 | 3.7 | 139.9 | 194.1 | ok 8 |
| c5 | 5 | 20 | 204.4 | 12.02 | 16.6 | 47.3 | 59.6 | ok 20 |
| c8 | 8 | 24 | 269.6 | 16.21 | 18.5 | 63.7 | 77.9 | ok 24 |
| c10 | 10 | 30 | 295.9 | 17.24 | 22.5 | 95.4 | 98.4 | ok 30 |

## 3. Viewer sessions end to end

Each pane shows a different image and moves on its own (open, then pan / zoom / jump). *Cold*: fresh cache, everything from the source. *Warm*: the same session again, served from the wissi-viz cache. *View*: time until every tile of a move has arrived.

| panes | conn | cold view p50 / p95 ms | cold tile p95 ms | warm view p95 ms | requests | MB | errors % | API calls | API calls per tile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 54 / **153** | 29.4 | 78 | 300 | 17.9 | 0.0 | — | — |
| 1 | 8 | 42 / **132** | 48.3 | 99 | 300 | 17.7 | 0.0 | — | — |
| 1 | 10 | 36 / **126** | 57.6 | 75 | 300 | 17.8 | 0.0 | — | — |
| 4 | 5 | 53 / **147** | 39.2 | 66 | 300 | 16.9 | 0.0 | — | — |
| 4 | 8 | 76 / **235** | 55.0 | 71 | 300 | 17.6 | 0.0 | — | — |
| 4 | 10 | 49 / **237** | 68.2 | 99 | 300 | 16.8 | 0.0 | — | — |
| 9 | 5 | 92 / **357** | 49.2 | 93 | 300 | 15.2 | 0.0 | — | — |
| 9 | 8 | 100 / **336** | 73.4 | 95 | 300 | 15.5 | 0.0 | — | — |
| 9 | 10 | 93 / **321** | 108.6 | 80 | 300 | 15.5 | 0.0 | — | — |
| 16 | 5 | 105 / **516** | 259.1 | 66 | 300 | 14.8 | 0.0 | — | — |
| 16 | 8 | 157 / **629** | 231.8 | 121 | 300 | 16.5 | 0.0 | — | — |
| 16 | 10 | 125 / **523** | 254.4 | 78 | 300 | 15.4 | 0.0 | — | — |

## Notes

- The load generator opens up to --conn connections per user over HTTP/1.1, like a browser does behind an HTTP/2 proxy (a browser on plain HTTP/1.1 is limited to 6 per host).
- Request caps per run are small by design (the API-call budget); view latencies from few moves are indicative, not precise.
- Registry warm-up (background pre-fetch of coarse levels) was off, so every API call is user-driven and counted.
- Local target: no API involved; this measures wissi-viz itself (file decoding, encoding, caching).
