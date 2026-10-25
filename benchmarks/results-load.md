# Load simulation: 10–100 users

`examples/tile_api/load_sim.py` runs three processes, as they would be deployed:
- the **load generator** (`scripts/wissi_loadtest.py`);
- the **wissi-viz server**;
- the **mock vendor API**, behind `DemoApiConnection`.

Each simulated user keeps 4 slides open with linked panes. It pans, zooms and jumps with 0.3–1.5 s pauses, fetches up to 6 tiles at once like a browser, never re-fetches a tile it already has, and stops after 200 requests. All users browse the same 16 vendor images, so they share the server's tile cache.

**Machine:** a 2-vCPU cloud container that runs **all three processes**. From about 75 users the load generator itself lags (event-loop lag > 50 ms, flagged in the tables), so the high-user numbers are lower bounds for the server. A real test should run the generator on another machine, with `--procs`.

**Throughout:** no user ever received a 404 or a 502.

## 1. Vendor API, healthy

The vendor adds 40–120 ms latency per call, answers 429 above 12 concurrent calls and 503 to 1% of calls, and returns 404 for empty tiles. The connector allows at most 10 concurrent vendor calls.

| users | requests | req/s | latency p50 / p95 ms | **view p50 / p95 ms** | errors to users | load-gen lag p95 ms |
|---|---|---|---|---|---|---|
| 10 | 2000 | 73.9 | 163.1 / 819.0 | 706 / 4696 | 0 | 11.9 |
| 25 | 5000 | 114.6 | 100.0 / 2291.5 | 1208 / 11269 | 0 | 16.2 |
| 50 | 10000 | 183.8 | 104.6 / 3769.0 | 1109 / 15219 | 0 | 38.5 |
| 75 | 15000 | 259.5 | 433.1 / 4427.2 | 3375 / 19398 | 0 | 107.0 |
| 100 | 20000 | 339.8 | 620.0 / 4122.2 | 3247 / 18647 | 0 | 124.4 |

What reached the vendor (mean vendor call time not available for this run: it was recorded before the averages fix):

| users | user requests | vendor tile calls (share) | vendor 429 | vendor 503 | vendor 404 (empty tiles) | connector retries | mean vendor call ms | 502 to users |
|---|---|---|---|---|---|---|---|---|
| 10 | 2000 | 1308 (65%) | 0 | 19 | 10 | 19 | — | 0 |
| 25 | 5000 | 1899 (38%) | 0 | 20 | 184 | 20 | — | 0 |
| 50 | 10000 | 2597 (26%) | 0 | 18 | 79 | 18 | — | 0 |
| 75 | 15000 | 3228 (22%) | 0 | 30 | 246 | 30 | — | 0 |
| 100 | 20000 | 3490 (17%) | 0 | 34 | 367 | 34 | — | 0 |

- **No errors reached users.**
  - The vendor's 503s were retried by the connector.
  - Its 404s (empty tiles) were drawn as background tiles and cached.
  - The connector stayed under the 429 limit (10 < 12 concurrent).
- **The shared cache carries the load.**
  - As users are added, a smaller share of their requests reaches the vendor: 65% at 10 users, 17% at 100.
  - That is why throughput keeps rising: 74 → 340 req/s.
- **Waiting is the cost.**
  - The p95 time for a move to become sharp grows from 4.7 s to about 19 s.
  - Cold tiles queue for the connector's **10 concurrent vendor calls**. At ~80 ms each, that is ≈ 125 tiles/s of new data for *all* users together.
  - This cap is the σ (contention) term of the fit, and it is the first thing to raise if your vendor allows it.

USL fit: λ = 4.89 req/s per user, σ = 0.0049, κ = 0.000000, R² = 0.975.
Closed-loop user model: ≈ 25.1 tiles per move, ≈ 4.131 s think.

| users | how | req/s | latency mean / p95 ms | view p95 ms |
|---|---|---|---|---|
| 17 | interpolated | 92.9 | 446.2 / 1506.2 | 7763 |
| 37 | interpolated | 147.8 | 742.6 / 3000.7 | 13165 |
| 150 | EXTRAPOLATED (USL) | 424.3 | 1616.3 / 6006.7 | 27171 |
| 200 | EXTRAPOLATED (USL) | 495.6 | 2044.4 / 7597.6 | 34368 |

## 2. Vendor API, struggling

The vendor now answers 503 to **5%** of calls and 429 above **8** concurrent calls. The connector still sends up to 10, so it is rate-limited constantly.

| users | requests | req/s | latency p50 / p95 ms | **view p50 / p95 ms** | errors to users | load-gen lag p95 ms |
|---|---|---|---|---|---|---|
| 10 | 2000 | 70.4 | 166.2 / 922.9 | 1159 / 4675 | 0 | 8.0 |
| 25 | 5000 | 117.9 | 107.7 / 2220.3 | 1060 / 11819 | 0 | 25.3 |
| 50 | 10000 | 224.4 | 209.9 / 2874.9 | 1637 / 11497 | 0 | 62.1 |
| 100 | 20000 | 312.0 | 652.5 / 5367.3 | 3909 / 22411 | 0 | 176.9 |

| users | user requests | vendor tile calls (share) | vendor 429 | vendor 503 | vendor 404 (empty tiles) | connector retries | mean vendor call ms | 502 to users |
|---|---|---|---|---|---|---|---|---|
| 10 | 2000 | 1308 (65%) | 654 | 94 | 10 | 94 | 146.5 | 0 |
| 25 | 5000 | 1899 (38%) | 1171 | 96 | 184 | 96 | 151.0 | 0 |
| 50 | 10000 | 2597 (26%) | 1318 | 149 | 79 | 149 | 140.0 | 0 |
| 100 | 20000 | 4031 (20%) | 2299 | 233 | 294 | 233 | 136.0 | 0 |

- **Still no user-visible errors: 0 × 502, 0 × 404.**
  - At 100 users, 2,299 × 429 (honouring `Retry-After`) and 233 × 503 were all absorbed by retries.
- **The cost is slower vendor calls and longer waits.**
  - Each vendor call took ~140 ms instead of ~80.
  - View p95 reached 22 s at 100 users.
- **The fit finds a peak:** throughput tops out near 160 users at about 340 req/s and falls beyond that. Retries and rate limiting are what make throughput *fall*: the κ term.

USL fit: λ = 5.99 req/s per user, σ = 0.0054, κ = 0.000039, R² = 0.990; peak ≈ 160 users at ≈ 337 req/s.
Closed-loop user model: ≈ 20.2 tiles per move, ≈ 2.592 s think.

| users | how | req/s | latency mean / p95 ms | view p95 ms |
|---|---|---|---|---|
| 17 | interpolated | 92.6 | 425.8 / 1528.4 | 8009 |
| 37 | interpolated | 169.0 | 618.2 / 2534.5 | 11664 |
| 150 | EXTRAPOLATED (USL) | 336.6 | 2170.8 / 8806.1 | 36769 |
| 200 | EXTRAPOLATED (USL) | 331.4 | 3250.8 / 13187.0 | 55062 |

## 3. Local files, no vendor

The wissi-viz server reads 4 slides itself: 64k SVS, 40k OME-TIFF, 24k CZI and the real CMU-1 region.

| users | requests | req/s | latency p50 / p95 ms | **view p50 / p95 ms** | errors to users | load-gen lag p95 ms |
|---|---|---|---|---|---|---|
| 10 | 2000 | 103.5 | 16.6 / 103.5 | 76 / 734 | 0 | 7.6 |
| 25 | 5000 | 306.9 | 20.6 / 178.2 | 71 / 1020 | 0 | 13.5 |
| 50 | 9848 | 545.5 | 93.6 / 353.6 | 330 / 1830 | 0 | 9.7 |
| 100 | 20000 | 595.3 | 465.7 / 1323.2 | 1923 / 7290 | 0 | 141.5 |

- **Below the server's limit, moves are quick:** view p95 under 2 s up to 50 users.
- **Throughput tops out around 600 req/s at about 85 users.** This limit belongs to this 2-vCPU box, which also runs the load generator (lag 141 ms at 100 users), not to wissi-viz.
- **At 100 users,** view p95 is 7.3 s.

USL fit: λ = 13.81 req/s per user, σ = 0.0000, κ = 0.000131, R² = 0.988; peak ≈ 87 users at ≈ 606 req/s.
Closed-loop user model: ≈ 22.9 tiles per move, ≈ 1.907 s think.

| users | how | req/s | latency mean / p95 ms | view p95 ms |
|---|---|---|---|---|
| 17 | interpolated | 198.4 | 41.7 / 138.4 | 867 |
| 37 | interpolated | 421.4 | 85.0 / 262.4 | 1409 |
| 150 | EXTRAPOLATED (USL) | 526.9 | 1321.9 / 3175.5 | 17495 |
| 200 | EXTRAPOLATED (USL) | 444.1 | 2409.3 / 5788.0 | 31888 |

## What to do with this on your deployment

1. **Run the same script against your server**, from a separate machine (see [docs/LOAD_TESTING.md](../docs/LOAD_TESTING.md)):

   ```
   python scripts/wissi_loadtest.py https://<wissi>/wissi-viz --users 10,25,50,75,100 --requests 300 \
       --stats-url https://<wissi>/wissi-viz/stats --header "Cookie: …" --out load.json --yes
   ```

   **Validated here:**
   - the script recovers a known synthetic system's parameters and its 150-user view time to within 5% (`tests/test_loadtest.py`);
   - a live run against a real server counts statuses and respects the request cap.
2. **If view p95 grows while vendor calls are flat:** you are bound by the connector's `max_concurrency`. Raise it to the vendor's real limit, or let browsers fetch native levels from the vendor directly (`browser_tile_url`).
3. **If vendor calls track user requests:** users rarely share tiles. Add the disk cache (`cache_dir`) so tiles survive restarts. Warm the slides you know people will open.
4. **If the server's CPU saturates (local files):** give the container more cores (`workers`), or run more replicas behind the proxy. They can share the disk cache.
5. **If 502s appear:** the vendor still failed after the retries. Check the connector's `stats()`: high `rate_limited` means lower `max_concurrency`; high `retries` with `errors` means the vendor is unhealthy.

## Problems the simulation found in the load tester (fixed)

- **The scalability fit was linearised, which could imply negative coefficients.** Clipping them produced an R² = 0.09 fit that predicted *less* throughput at 150 users than measured at 100. It is now a bounded least-squares fit (σ, κ ≥ 0). A poor fit falls back to a pessimistic "no more throughput" extrapolation.
- **Latency extrapolation assumed one request at a time per user.** It now models each move as a burst of b tiles after a think time T, fitted from the stages.
- **Server averages were differenced between snapshots like counters,** giving negative mean latencies. Averages are now reported as-is.
