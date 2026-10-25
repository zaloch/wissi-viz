# Load testing a wissi-viz deployment

`scripts/wissi_loadtest.py` is a single file you can copy to any machine: Python 3.9+, `pip install httpx numpy`, plus `h2` for `--http2`. It is also bundled in the package as `wissi-viz loadtest …`, with the same options. It drives many simulated viewer users against your server, reports what each user count did, and estimates what untested user counts would do.

## What a simulated user does

Each user acts like a person at the viewer:

1. **Opens slides:** 1–16 at once (`--panes`), fetching `info.json` for each.
2. **Loads the overview:** the whole coarsest level, then the fit view.
3. **Navigates with linked panes**, pausing `--think` seconds (default 0.3–1.5 s) between moves:
   - **55% pans** of 0.3–1 screen;
   - **20% zoom in** ×2;
   - **15% zoom out** ×2;
   - **10% jumps** to a random place at 2.5–40×.
4. **Fetches only what each move made visible:**
   - It requests the tiles the viewer would request at rest.
   - At most `--conn` requests are in flight per user: 6 is a browser on HTTP/1.1; 24 is wissi-viz over HTTP/2.
   - It never re-fetches a tile it already has, as a browser cache wouldn't.
5. **Stops** after `--requests` requests (default 300, **hard cap 999**) or at the stage time limit.

Users are added in stages (`--users 10,25,50,100`), each stage ramped over `--ramp` seconds. Later stages reuse the server's warm cache for coarse levels. Each stage uses a different random seed, so most fine-resolution tiles are new.

## What it reports, per stage

| Column | Meaning |
|---|---|
| req/s | throughput while all of the stage's users are active (ramp-up and tail excluded) |
| latency p50/p95/p99 | per tile request, successful requests only |
| **view p50/p95** | how long a move takes until every tile it needs has arrived: *what users feel* |
| error %, statuses | every status that isn't 2xx/3xx (404, 429, 500, 502, 503, 504) plus `timeout` and `conn` (connection refused/reset) |
| client lag p95 | the load generator's own event-loop lag. Above ~50 ms the **client** is the bottleneck: use `--procs N` or a bigger machine |
| server | `--stats-url` snapshots (e.g. `…/wissi-viz/stats`), differenced per stage: cache hits and misses, upstream calls |

How to read the statuses on a wissi-viz server:

| Status | Usual cause |
|---|---|
| **404** | unknown slide id or a tile index out of range. It should be 0. Non-zero means stale ids, or a registry that lost slides (container restart, `max_open` eviction of an unregistered path). |
| **502** | the vendor API still failed after the connector's retries. Read it with the connector's `stats()` (retries, rate-limited). |
| **500** | a reader or decoder error. |
| **503/504** | usually your reverse proxy: an upstream timeout, or too many connections. |
| **timeout, conn** | the server or proxy stopped accepting work. This is the capacity edge. |

Against a raw vendor API (template mode), pass `--ok-404` if the API answers 404 for empty tiles.

## Interpolating and extrapolating

With three or more stages, the script fits the **Universal Scalability Law** (Gunther):

    X(N) = λN / (1 + σ(N−1) + κN(N−1))

| Term | Meaning |
|---|---|
| X | throughput |
| N | users |
| λ | throughput of one user |
| σ | **contention**: serialised work such as a single worker, a lock, or an upstream rate limit (`max_concurrency`) |
| κ | **coherency**: crosstalk that makes throughput *fall* when more users are added, such as cache thrash or retry storms |

From the fit it reports:
- **peak users** = √((1−σ)/κ) and the maximum throughput;
- or, with κ ≈ 0, the plateau λ/σ.

Latency at an untested N comes from Little's law for a closed system, `N = X·(R + Z)`. Z, the effective think time per request, is measured from your stages.

- **Inside the tested range:** the numbers are interpolated between stages and are reliable.
- **Outside it:** they are marked **EXTRAPOLATED**. Trust them only to about 2× the largest tested load.
- **Error rates** are extrapolated as a linear trend. Real systems usually fail abruptly once saturated, so treat a rising error trend as a warning, not a forecast.

## Running it on your servers

```bash
# 1. a quick smoke run: 2 users, 50 requests each
python scripts/wissi_loadtest.py https://wissi.example.org/wissi-viz --users 2 --requests 50 \
    --header "Cookie: session=<copied from your browser>" --yes

# 2. the real run: 10 → 100 users, ≤ 300 requests each (≤ 64,500 requests in total), with server counters
python scripts/wissi_loadtest.py https://wissi.example.org/wissi-viz --users 10,25,50,75,100 --requests 300 \
    --panes 4 --conn 6 --header "Cookie: session=…" --stats-url https://wissi.example.org/wissi-viz/stats \
    --out load.json --yes

# 3. what HTTP/2 behind your proxy buys (wissi-viz raises in-flight tiles to 24 per user on h2)
python scripts/wissi_loadtest.py https://wissi.example.org/wissi-viz --http2 --conn 24 ... --out load_h2.json

# 4. the vendor API on its own, to separate their limits from yours
python scripts/wissi_loadtest.py "https://halolink.example.org/tiles/{image}/{level}/{col}/{row}" \
    --images 1234,1235 --pyramid 98000x76000:256:2 --header "Authorization: Bearer …" --ok-404 --users 5,10,20 --yes
```

**Tips:**
- **Run from a separate machine** on the same network path as your users. Running on the server itself measures a different system.
- **Watch client lag.** If it climbs, use `--procs 4`.
- **Warm or cold:** run once to warm the caches, then again for the numbers you care about, or restart the container between runs to measure cold behaviour.
- **Get permission** before load-testing a service you don't operate, especially a vendor API. The script refuses non-local hosts without `--yes`, and the 999-request cap keeps each user small.

## The simulation in this repo

`examples/tile_api/load_sim.py` starts three processes, as in production:
- the mock vendor API (latency, a 429 limit, random 503s, 404s for empty tiles);
- a wissi-viz server backed by it through `DemoApiConnection`;
- the load generator.

It then runs the stages and prints the upstream counters per stage. Results: [benchmarks/results-load.md](../benchmarks/results-load.md).
