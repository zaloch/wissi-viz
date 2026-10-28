# Server memory with API-backed slides (PathCore / HALO Link)

**The symptom:** a wissi server using the PathCore connector grows to about 20 GB of RAM over time.

**The harness** (`mem_harness`, described below) uses a PathCore-style async connector whose `objects()` returns a 300,000-cell table: positions, class, and 20 measurements. It opens one different slide after another, the way users review cases. For each slide it reads the metadata, annotations and 36 image tiles, plus pyramid tiles at 3 zoom levels and 30 outline buckets of the cell layer. After every slide it records process memory and what each cache holds.

## What was growing

| after N slides (0.6.0) | 1 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| process RSS (MB) | 270 | 789 | 1386 | 1969 | 2524 | **3115** |
| annotation / object layers in RAM | 90 | 452 | 904 | 1356 | 1808 | **2260** |
| per-layer overlay caches | 10 | 51 | 101 | 152 | 203 | 253 |

Process memory grew by about **120 MB per slide, without limit**. The causes:

1. **Every slide's annotation and object layers stayed in the registry for good.** That held after the slide itself closed (`max_open`) and after nobody had looked at it for hours. A 300k-cell layer is about 90 MB in RAM. At 1M cells, growth extrapolates to about 400 MB per slide, so 40–50 cases ≈ 20 GB.
2. **Overlay tile caches were capped per layer by entry count** (4,096 pyramid tiles + 2,048 outline buckets), not by bytes, and there was one pair per layer.
3. **A latent bug:** `AnnotationLayer.save()` / `open()` failed on tables with plain text columns (a connector's `Classification` as strings). Column files were numbered across all columns but read back numbered across saved ones. `read_objects()` converts text to categories, which hid it.

The tile cache and the decoded-array cache were already bounded, and stayed small here.

## What changed (0.6.1)

- **Auto layers are memory-mapped from the disk cache.** These are the layers a slide brings with it: connector `annotations()` / `objects()` and sidecar files.
  - With `cache_dir` set, each large object layer is saved once in compiled form under `<cache_dir>/_layers`, keyed by the image's `version`, and reopened memory-mapped.
  - RAM per 300k-cell layer: **90 MB → 0.3 MB**. Centres and sizes are now stored in the files; zero arrays are zero-stride views.
  - The tiles and outline buckets it serves are byte-identical, and so are the cell properties shown on click (tested).
- **Remote slides reopen from that copy.** Reopening a case needs **no `annotations()` / `objects()` calls and no re-parse**. A changed image gets a new `version` and is fetched fresh.
- **Budgets for layers:**
  - `annotation_memory_mb` (default 1 GB) caps layers held in RAM.
  - `layer_mapped_mb` (default 2 GB) caps mapped layer files, whose touched pages count in RSS while mapped.
  - Beyond either, the least-recently-used slides' auto layers are dropped (and their overlay-cache entries purged); they reload on the next request, transparently. Layers your app adds are never dropped.
- **Overlay tiles share one process-wide byte budget** (`overlay_cache_mb`, default 128 MB). The server's gzip cache is configurable too (`gzip_cache_mb`, 256 MB).
- **Heap returned to the OS:** freed memory goes back via `malloc_trim` after evictions (glibc).
- **`GET /wissi-viz/stats` → `memory`** reports RSS split into anonymous and file-backed, every cache's size and cap, layers in RAM and mapped, evictions, and disk reopen hits.
- **The save/open bug is fixed:** files are indexed per column, and text columns are saved as categories. The save format is bumped to 3, so old caches rebuild.

## After

| after N slides | 1 | 5 | 10 | 15 | 20 | 25 | + 10 revisits |
|---|---|---|---|---|---|---|---|
| **with `cache_dir`**: RSS (MB) | 207 | 414 | 642 | 852 | 1004 | 1180 | 1222 |
| · anonymous / file-backed | 157 / 49 | 282 / 132 | 408 / 235 | 515 / 338 | 564 / 440 | 637 / 543 | 679 / 543 |
| · layers in RAM | 0.6 | 2.9 | 5.8 | 8.7 | 11.5 | 14.4 | 14.4 |
| **RAM only**, `annotation_memory_mb=400`: RSS | 270 | 582 | 644 | 721 | 793 | 863 | 929 |
| · layers in RAM (evictions) | 90 (0) | 362 (1) | 362 (6) | 362 (11) | 362 (16) | 362 (21) | 362 (31) |

In both runs the rest of the growth is bounded caches filling toward their caps: decoded arrays (+6 MB per slide, cap 1 GB), overlay tiles (cap 128 MB, reached at slide 13) and the gzip cache. In the disk-cache run, the file-backed half is mapped layer pages, which the OS reclaims and worker processes share.

**Plateau with the caches full:** small caps make them fill within the run: tiles 64 MB, decoded 64, overlay 32, gzip 64, `layer_mapped_mb=256`, over 40 different slides and 10 revisits:

| after N slides | 1 | 6 | 11 | 16 | 21 | 31 | 40 | + 10 revisits |
|---|---|---|---|---|---|---|---|---|
| RSS (MB) | 206 | 303 | 350 | 397 | 409 | 419 | 424 | **433** |
| anonymous / file-backed | 157 / 49 | 213 / 91 | 260 / 91 | 307 / 90 | 319 / 90 | 328 / 91 | 334 / 90 | 343 / 90 |
| layer evictions · reopened from disk | 0 · 0 | 3 · 0 | 8 · 0 | 13 · 0 | 18 · 0 | 28 · 0 | 38 · 1 | 47 · **10** |

Memory is flat once the caches are full. Every revisit of an evicted slide reopened its layers from disk.

## Sizing

- **Anonymous memory ≤ the sum of budgets + ~150 MB of Python:** `memory_mb` + `decoded_cache_mb` + `overlay_cache_mb` + `gzip_cache_mb` + `annotation_memory_mb`. The defaults sum to about 3.6 GB; with `cache_dir` set, the layer part stays far below its cap.
- **File-backed:** at most `layer_mapped_mb` of layer files mapped (2 GB). This is page cache the OS reclaims under pressure. Container memory accounting counts it, but it is reclaimed before an OOM kill.
- **Disk:** `<cache_dir>/_layers` is capped by `layer_disk_gb` (20 GB), least-recently-used first. It is roughly 70 MB per 300k-cell layer with 20 measurements.

A smaller footprint for a PathCore server, e.g. a 4 GB container:
```python
configure(
    cache_dir='/cache/wissi-viz',
    memory_mb=512,
    decoded_cache_mb=512,
    overlay_cache_mb=128,
    gzip_cache_mb=128,
    annotation_memory_mb=512,
    layer_mapped_mb=1024,
)
```

Rerun: `python benchmarks/memory_harness.py` (regression tests: `tests/test_memory.py`). Numbers are from a 2-vCPU container, opening slides through the real server routes.
