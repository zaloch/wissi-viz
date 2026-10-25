# Zooming through the pyramid: legacy vs new level of detail

`wissi-viz bench --zoom`, legacy (pre-0.4) vs the new default `setLod` policy. Each (slide set, policy) pair ran with a **fresh server and a fresh browser**, so both started with cold caches. The scenario has three parts:

1. **Continuous zoom** about the pane centre from fit to 40× and back, at 4× per second. It drives the viewer the way a wheel or trackpad does (`_zoomAt` with animation).
2. **Five animated jumps** (`goTo`, as for "centre on object" or a double-click) to 40×, 10×, 20×, 2.5× and 40× at different places. Each is timed until the pane is at full resolution.
3. **Metrics** throughout:
   - **detail**: 1 = texels ≤ 1 device px, 0.5 = one 2× level coarser, 0 = blank;
   - **full-res %**;
   - **blank %**;
   - data downloaded and requests;
   - **GPU fill**: device pixels shaded ÷ pane pixels;
   - textures held at the end;
   - tiles uploaded but never drawn.

**Machine:** a 2-vCPU cloud container with **no GPU**. Chromium rasterises with SwiftShader on the same 2 cores that run the tile server, so frame rates are 1–10 fps and tile delivery is CPU-starved. Treat the times as relative, legacy vs new on the same machine; the data, fill and texture numbers do not depend on the hardware. Viewport 1600×1000. With 16 panes each pane is 400×250.

## Headline

| slides | panes | | jumps → full res, s (5 jumps) | data, MB (whole run) | jump data, MB | zoom-out data, MB | GPU fill (worst phase) | textures held, MB | uploaded never drawn | detail while zooming in |
|---|---|---|---|---|---|---|---|---|---|---|
| SVS 64k (4x) | 1 | legacy | 7.1 | 16.9 | 8.57 | 2.4 | 1.43× | 346.4 | 40 | 93.9% |
| | | **new** | **6.1** (-14%) | **11.3** (-33%) | **4.38** (-49%) | **1.86** (-22%) | **1.00×** | **225.4** (-35%) | **11** | 93.9% |
| OME-TIFF 40k (2x) | 1 | legacy | 47.0 | 20.9 | 15.32 | 1.78 | 1.98× | 244.8 | 66 | 66.6% |
| | | **new** | **30.2** (-36%) | **20.8** (-0%) | **15.92** (+4%) | **1.03** (-42%) | **1.00×** | **224.8** (-8%) | **31** | 62.9% |
| CZI 24k (2x) | 1 | legacy | 43.8 | 60.6 | 41.64 | 4.4 | 1.65× | 230.4 | 55 | 68.1% |
| | | **new** | **30.3** (-31%) | **58.0** (-4%) | **38.14** (-8%) | **4.57** (+4%) | **1.00×** | **212.9** (-8%) | **35** | 68.1% |
| SVS 64k (4x) | 16 | legacy | 14.1 | 33.9 | 20.15 | 5.22 | 1.42× | 678.7 | 120 | 81.6% |
| | | **new** | **10.5** (-25%) | **28.2** (-17%) | **14.81** (-27%) | **4.87** (-7%) | **1.00×** | **541.9** (-20%) | **52** | 72.9% |
| mixed SVS+OME+CZI | 16 | legacy | 66.8 | 83.4 | 64.03 | 2.47 | 1.59× | 594.0 | 95 | 49.5% |
| | | **new** | **45.3** (-32%) | **81.1** (-3%) | **61.72** (-4%) | **4.16** (+68%) | **1.00×** | **564.4** (-5%) | **63** | 45.3% |

Reading it:

- **Jumps land at full resolution 14–36% sooner, on every format, with 1 and 16 panes.** During an animated jump the viewer fetches for the destination: the destination's level one step coarse first, then full resolution in spare slots. In-between views get only leftover capacity. Legacy spent the flight requesting full-resolution tiles for views that existed for a few frames.
- **Each pane pixel is shaded exactly once** (fill 1.00×, down from 1.4–2.0×). That is a 30–50% cut in GPU fill, which matters most for the multi-channel fluorescence shader on integrated GPUs and laptops.
- **Less data and memory on 4×-step pyramids (SVS):** 17–33% less downloaded, 20–35% fewer texture MB held, and about half the tiles uploaded but never drawn. On 2×-step pyramids (OME-TIFF, CZI) the data is roughly unchanged (−4% to +4%).
- **The trade-off: lower detail while zooming in continuously.** On SVS with 16 panes it drops from 82% to 73%; on mixed slides from 50% to 45%. While moving, the viewer deliberately shows a level one step coarser, and full resolution follows when the motion stops. The settle time after the motion is similar: 114 → 181 ms (SVS, 16 panes) and 2.96 → 3.20 s (mixed). Set `setLod({motionBias: 1})` to disable this.
- **Blank: 0.0% in every phase of every run**, legacy and new.

## Things tried and not adopted

- **Virtual 2× levels for 4×-step SVS (`max_level_step=2`).** In theory browsers download about half the pixels in half of every zoom range. Measured, with 16 panes:
  - **Data:** 33.4 MB vs 28.2 MB at native 4× steps. Re-encoded, box-filtered tiles compress worse than the scanner's own levels, which cancels the gain.
  - **Jumps:** 18.2 s vs 10.5 s. Every virtual tile costs server CPU.
  - **With 1 pane** it saved data (9.1 vs 11.3 MB) but was slower.

  The option stays (`SlideRegistry(max_level_step=2)`, `wissi-viz serve --max-level-step 2`) for thin links with a strong server, but the default remains 4.
- **Idle prefetch of the next finer level** (`prefetch: true`, measured with 1 pane in an earlier run) was within noise on time and cost −1% to +5% data. It is off by default.

## Problems the benchmark itself turned up (fixed)

1. **Memory: out-of-memory kill with 16 panes.**
   - Every open slide kept its own 256 MB decoded-tile cache, plus 512 MB of decoded subblocks per CZI file.
   - A 16-pane run reached 5.4 GB and the kernel killed it.
   - Fix: decoded arrays now share **one process-wide budget** (`decoded_cache_mb`, default 1 GB), and closing a slide frees its share. `SlideRegistry.close()` and `TileService.close()` were added.
   - Test: `test_decoded_arrays_share_one_budget_across_slides`.
2. **Slow frames throttled the network.**
   - New requests and GPU uploads started only once per frame, so a GPU-bound viewer at 3 fps left the connection idle between frames.
   - Now a finished request starts the next queued one immediately, and uploads may use up to 20% of a slow frame.
3. **A first version of "moving → coarser level" downloaded more on channel images during jumps** (CZI 47 MB vs 42 MB legacy). It fetched a coarse level for every in-between view. Destination-first scheduling fixed it (38 MB).

## Correctness

`test_quadtree_draw_matches_legacy_pixels_and_has_no_overdraw` renders the same views with both draw paths:
- views at full resolution, between levels, and with holes from tiles that always fail, which the quadtree fills from coarser levels as sub-rectangles;
- it compares framebuffers pixel by pixel (max difference ≤ 2/255);
- it asserts fill ≤ 1.0.

## All phases

| config | panes | LOD | phase | detail % | full-res % | blank % | settle ms | MB | requests | cancelled | uploaded | overdraw | draws/frame | JS ms p50 (p95) | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SVS 64k (4x) | 1 | legacy | zoom in | 93.9 | 78.7 | 0.0 | 102 | 2.99 | 69 | 0 | 69 | 1.095 | 22.0 | 4.0 (10.1) | 7.5 |
| SVS 64k (4x) | 1 | legacy | zoom out | 99.8 | 97.7 | 0.0 | 63 | 2.4 | 53 | 0 | 53 | 1.113 | 36.4 | 0.6 (9.16) | 10.0 |
| SVS 64k (4x) | 1 | legacy | jumps (872, 1463, 2127, 1461, 1142 ms) | 91.1 | | 0.0 | 7065 | 8.57 | 175 | 0 | 175 | 1.432 | 28.4 | | |
| SVS 64k (4x) | 1 | default | zoom in | 93.9 | 81.0 | 0.0 | 137 | 2.07 | 62 | 16 | 46 | 0.971 | 28.1 | 1.4 (17.45) | 10.0 |
| SVS 64k (4x) | 1 | default | zoom out | 99.0 | 95.1 | 0.0 | 66 | 1.86 | 39 | 4 | 45 | 0.944 | 36.2 | 0.65 (9.45) | 10.0 |
| SVS 64k (4x) | 1 | default | jumps (824, 1057, 2016, 1026, 1172 ms) | 91.3 | | 0.0 | 6095 | 4.38 | 106 | 32 | 85 | 1.0 | 27.0 | | |
| SVS 64k, virtual 2x | 1 | legacy | zoom in | 91.7 | 66.8 | 0.0 | 100 | 3.72 | 85 | 0 | 85 | 1.252 | 12.7 | 6.6 (12.67) | 6.0 |
| SVS 64k, virtual 2x | 1 | legacy | zoom out | 99.2 | 96.9 | 0.0 | 87 | 2.06 | 44 | 10 | 39 | 1.198 | 17.0 | 1.1 (10.1) | 7.5 |
| SVS 64k, virtual 2x | 1 | legacy | jumps (1066, 1950, 3152, 2289, 2246 ms) | 87.0 | | 0.0 | 10703 | 6.79 | 136 | 41 | 114 | 1.52 | 15.3 | | |
| SVS 64k, virtual 2x | 1 | default | zoom in | 73.7 | 35.4 | 0.0 | 818 | 2.18 | 64 | 24 | 47 | 0.971 | 9.6 | 3.1 (16.7) | 5.0 |
| SVS 64k, virtual 2x | 1 | default | zoom out | 98.2 | 91.3 | 0.0 | 106 | 1.77 | 41 | 6 | 43 | 0.918 | 15.9 | 5.3 (12.3) | 6.0 |
| SVS 64k, virtual 2x | 1 | default | jumps (1324, 1859, 3761, 1424, 2624 ms) | 77.8 | | 0.0 | 10992 | 4.16 | 99 | 54 | 67 | 1.0 | 11.8 | | |
| OME-TIFF 40k (2x) | 1 | legacy | zoom in | 66.6 | 51.1 | 0.0 | 332 | 1.42 | 75 | 0 | 70 | 1.98 | 6.8 | 2.15 (8.8) | 1.6 |
| OME-TIFF 40k (2x) | 1 | legacy | zoom out | 89.8 | 80.4 | 0.0 | 308 | 1.78 | 45 | 0 | 50 | 1.322 | 9.9 | 1.05 (6.62) | 2.6 |
| OME-TIFF 40k (2x) | 1 | legacy | jumps (4008, 11684, 14174, 9653, 7507 ms) | 78.4 | | 0.0 | 47026 | 15.32 | 359 | 0 | 354 | 1.68 | 14.3 | | |
| OME-TIFF 40k (2x) | 1 | default | zoom in | 62.9 | 41.8 | 0.0 | 460 | 1.51 | 85 | 18 | 71 | 0.965 | 6.8 | 2.4 (7.7) | 2.9 |
| OME-TIFF 40k (2x) | 1 | default | zoom out | 89.2 | 72.8 | 0.0 | 235 | 1.03 | 50 | 13 | 48 | 0.874 | 12.2 | 1.4 (6.58) | 3.2 |
| OME-TIFF 40k (2x) | 1 | default | jumps (3292, 8261, 8486, 6129, 4060 ms) | 80.7 | | 0.0 | 30228 | 15.92 | 355 | 72 | 314 | 1.0 | 15.8 | | |
| CZI 24k (2x) | 1 | legacy | zoom in | 68.1 | 51.4 | 0.0 | 793 | 11.37 | 65 | 0 | 60 | 1.456 | 7.1 | 1.95 (3.98) | 1.8 |
| CZI 24k (2x) | 1 | legacy | zoom out | 93.4 | 92.6 | 0.0 | 230 | 4.4 | 30 | 0 | 35 | 1.164 | 10.1 | 1.3 (3.3) | 3.5 |
| CZI 24k (2x) | 1 | legacy | jumps (4673, 10539, 14332, 8136, 6150 ms) | 84.7 | | 0.0 | 43830 | 41.64 | 352 | 0 | 347 | 1.65 | 16.6 | | |
| CZI 24k (2x) | 1 | default | zoom in | 68.1 | 47.6 | 0.0 | 386 | 12.06 | 75 | 8 | 66 | 0.963 | 8.1 | 2.85 (14.7) | 2.4 |
| CZI 24k (2x) | 1 | default | zoom out | 89.6 | 85.1 | 0.0 | 237 | 4.57 | 36 | 2 | 40 | 0.868 | 13.5 | 0.9 (3.18) | 3.2 |
| CZI 24k (2x) | 1 | default | jumps (3150, 7056, 10208, 5913, 3946 ms) | 85.7 | | 0.0 | 30273 | 38.14 | 341 | 70 | 301 | 1.0 | 15.8 | | |
| SVS 64k (4x) | 16 | legacy | zoom in | 81.6 | 57.9 | 0.0 | 114 | 7.43 | 135 | 0 | 128 | 1.338 | 33.5 | 8.45 (11.99) | 6.3 |
| SVS 64k (4x) | 16 | legacy | zoom out | 94.4 | 89.0 | 0.0 | 77 | 5.22 | 110 | 0 | 117 | 1.412 | 53.3 | 6.3 (9.06) | 6.7 |
| SVS 64k (4x) | 16 | legacy | jumps (1176, 3654, 4572, 2743, 1933 ms) | 71.7 | | 0.0 | 14078 | 20.15 | 435 | 0 | 430 | 1.417 | 53.5 | | |
| SVS 64k (4x) | 16 | default | zoom in | 72.9 | 47.2 | 0.0 | 181 | 7.46 | 122 | 8 | 112 | 0.976 | 35.6 | 7.8 (23.9) | 6.7 |
| SVS 64k (4x) | 16 | default | zoom out | 95.0 | 89.8 | 0.0 | 95 | 4.87 | 110 | 16 | 108 | 0.941 | 63.6 | 7.5 (23.15) | 7.5 |
| SVS 64k (4x) | 16 | default | jumps (844, 2904, 4034, 1184, 1576 ms) | 70.7 | | 0.0 | 10542 | 14.81 | 368 | 90 | 316 | 0.997 | 63.7 | | |
| SVS 64k, virtual 2x | 16 | default | zoom in | 50.1 | 19.4 | 0.0 | 1537 | 6.92 | 132 | 16 | 119 | 0.976 | 23.8 | 10.15 (18.12) | 6.0 |
| SVS 64k, virtual 2x | 16 | default | zoom out | 87.3 | 80.2 | 0.0 | 137 | 3.27 | 64 | 18 | 60 | 0.937 | 36.9 | 4.45 (11.83) | 6.7 |
| SVS 64k, virtual 2x | 16 | default | jumps (1662, 6605, 4593, 2319, 3067 ms) | 61.7 | | 0.0 | 18246 | 22.21 | 422 | 112 | 360 | 0.994 | 34.4 | | |
| mixed SVS+OME+CZI | 16 | legacy | zoom in | 49.5 | 29.8 | 0.0 | 2961 | 14.57 | 150 | 0 | 144 | 1.051 | 19.7 | 3.85 (12.0) | 3.3 |
| mixed SVS+OME+CZI | 16 | legacy | zoom out | 85.1 | 76.1 | 0.0 | 244 | 2.47 | 70 | 0 | 76 | 1.076 | 26.2 | 3.7 (8.89) | 3.0 |
| mixed SVS+OME+CZI | 16 | legacy | jumps (3779, 15483, 19190, 19208, 9109 ms) | 68.8 | | 0.0 | 66769 | 64.03 | 755 | 0 | 748 | 1.593 | 43.5 | | |
| mixed SVS+OME+CZI | 16 | default | zoom in | 45.3 | 24.0 | 0.0 | 3199 | 12.91 | 157 | 8 | 145 | 0.978 | 19.2 | 5.45 (13.4) | 3.6 |
| mixed SVS+OME+CZI | 16 | default | zoom out | 84.3 | 73.7 | 0.0 | 171 | 4.16 | 69 | 8 | 73 | 0.897 | 33.4 | 4.4 (15.56) | 3.7 |
| mixed SVS+OME+CZI | 16 | default | jumps (2524, 10901, 14976, 9782, 7113 ms) | 70.2 | | 0.0 | 45296 | 61.72 | 763 | 90 | 709 | 0.997 | 43.9 | | |
