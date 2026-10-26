# Integration probe: local SVS, mock API, HALO-Link-style and PathCore-style connectors

`scripts/wissi_probe.py` runs one command per target. It does a smoke test, a direct concurrency sweep (5 / 8 / 10), and viewer sessions with 1 / 4 / 9 / 16 panes (a different image per pane) at 5 / 8 / 10 connections. Each session runs cold (fresh cache) and then warm (cache only). Every API call passes a hard budget guard (`--max-calls`, default 1000).

```bash
python scripts/wissi_probe.py local            # 2 generated SVS, 24000×18000, ~110 MB each
python scripts/wissi_probe.py mock             # mock vendor API over HTTP: 40–120 ms latency, 429 above 12 concurrent, 1 % 503s
python scripts/wissi_probe.py mock:halolink    # HALO-Link-style connector: sync client, DeepZoom 254 + 1 overlap, HALO XML + cells
python scripts/wissi_probe.py mock:pathcore    # PathCore-style connector: async client, coarsest-first ids, padded 256 px PNG, µm GeoJSON
```

Full reports, `recommended_settings.py` and raw JSON for each target are in `benchmarks/probe_reports/`. The setup was a 2-vCPU container with the same two SVS files behind every target. The mock modalities add 40–120 ms of simulated latency per call.

## The budget holds

| target | API calls used / allowed | checked by | runs that fit | runtime |
|---|---|---|---|---|
| mock (HTTP) | **954 / 1000** | the vendor's own request counter: **954** | 12 / 12 | 48 s |
| mock:halolink | 954 / 1000 | the guard | 12 / 12 | 41 s |
| mock:pathcore | 881 / 1000 | the guard | 6 / 12: the rest are skipped up front, with the reason | 47 s |
| local | no API | — | 12 / 12 (300 requests each) | 224 s |

**Calls per tile.**
- In every cold run of the mock API and the HALO-style connector, each viewer tile cost **≤ 1.06 API calls**.
- Every warm rerun made **0** calls.
- Opening a slide costs 1 metadata call plus annotations and objects. These are fetched once per slide in the background (the probe fetches them once per image).

## What each target showed

**Smoke tests:** all checks passed on every target (INFO = how the API answers a tile outside the image). The PathCore-style API got a **WARN**: its pyramid stops at 1500×1125, which is 30 tiles.

**Direct sweep** (distinct full-resolution tiles, no retries):

| target | 1 (sequential) | 5 | 8 | 10 | p95 at 10 | statuses | recommended `max_concurrency` |
|---|---|---|---|---|---|---|---|
| mock (HTTP) | 11.6 tiles/s | 50.7 | 68.3 | **100.2** | 117 ms | one random 503 | 10 (still rising: test higher if the vendor allows) |
| mock:halolink | 10.3 | 48.6 | **72.9** | 81.0 | 143 ms | 2 connection errors at 10 | 8 |
| mock:pathcore | 8.2 | 34.1 | **41.2** | 41.9 | 266 ms | — (latency doubles past 8) | 8 |
| local (cold tiles from SVS) | 32.5 | 204 | 270 | **296** | 95 ms | — | — (the server's own capacity) |

**Viewer sessions: first-view latency p95** (cold → warm), at 5 connections:

| panes | local SVS | mock (HTTP) | mock:halolink | mock:pathcore |
|---|---|---|---|---|
| 1 | 153 → 78 ms | 645 → 37 ms | 414 → 36 ms | **3260** → 53 ms |
| 4 | 147 → 66 ms | 754 → 62 ms | 797 → 73 ms | **3347** → 55 ms |
| 9 | 357 → 93 ms | 1714 → 123 ms | 1513 → 120 ms | **9670** → 140 ms |
| 16 | 516 → 66 ms | 987 → 60 ms | 1616 → 116 ms | skipped: ≈ 586 calls needed, 480 of them just to open 16 slides |

**Errors:** 0% in every cold and warm run, despite the mock API's 503s and the HALO-style connection errors. These were absorbed by retries, and the probe raised `retries` to 3 where it saw them.

**Connections, 5 → 8 → 10:** little difference in any run. The API side is bounded by the connector's `max_concurrency`, not by browser connections. The verdict is "HTTP/2 optional at this latency"; the guide still recommends it behind a proxy.

## The PathCore-style finding: a shallow API pyramid makes every first open expensive

That API's coarsest level is 1500×1125, which is 30 tiles of 256 px. The viewer's first, zoomed-out view needs smaller overview levels, and wissi-viz builds those from all 30 tiles. So:
- Opening one slide costs **≈ 30 API calls** and about 3 s at this latency.
- That is 2.1–2.5 API calls per viewer tile, against ≤ 1.06 for the other APIs.
- Sixteen panes would need ≈ 480 calls before the first move.

After that, cached (warm) views take 53–140 ms, like everything else.

The probe flags this in the smoke test ("pyramid depth"), prices it per slide ("open cost"), and plans around it: only runs that fit go ahead, and the others are listed with the calls they would need. The remedies it suggests:
- Expose the API's thumbnail or overview endpoint as an extra coarse level in `image_info()`.
- Keep a persistent `cache_dir`, so it is a one-time cost per slide.
- Raise `--max-calls` to measure 9–16 panes.

## How the budget is planned

1. **Smoke:** ≤ 7 calls per image.
2. **Sweep:** 8 sequential calls, plus max(20, 3c) at each concurrency c.
3. **Viewer runs, planned after the smoke test.** By then the opening cost per slide is known: the API's coarsest-level tiles.
   - Runs are admitted in priority order: every pane count at the first connection setting, then the other settings. A run is admitted only if its opening cost plus a few moves fits.
   - What's left is split across the admitted runs, with a 15% margin. Request caps are scaled by the measured calls per request.
4. **Hard stop.** If a run still hits the cap, the guard refuses further calls. That run is marked invalid and the rest are skipped, so no number in the report comes from a cut-off run.
