# Results: cloud container (2 vCPU, no GPU — SwiftShader software WebGL)

Slides: a real Aperio file (CMU-1-Small-Region.svs, from the libvips test suite) and three full-size slides
built from its tissue by `make_bench_slides.py` (64k×48k SVS, 40k×30k×3 OME-TIFF, 24k×18k×3 CZI).
Server, browser, tile decoding and the software GPU all share the same 2 vCPUs.

## Server: open + tiles

| slide | reader | size | MB | open ms | level (÷) | cold p50/p95 ms | tiles/s (×8) | warm p50 ms | tile KB |
|---|---|---|---|---|---|---|---|---|---|
| CMU-1-Small-Region.svs | TiffSource | 2220x2967 | 1.8 | 16.8 | 0 (÷1) | 3.63 / 94.19 | 279.3 | 0.004 | 27.0 |
|  |  |  |  |  | 2 (÷4) | 94.94 / 270.75 | – | 0.008 | 24.6 |
|  |  |  |  |  | 3 (÷8) | 23.26 / 23.26 | – | 0.032 | 26.0 |
| bench_he_64k.svs | TiffSource | 64000x48000 | 360.7 | 8.3 | 0 (÷1) | 2.2 / 3.71 | 334.6 | 0.004 | 24.5 |
|  |  |  |  |  | 2 (÷16) | 3.38 / 4.06 | 639.6 | 0.006 | 63.0 |
|  |  |  |  |  | 4 (÷128) | 38.63 / 38.63 | – | 0.034 | 67.2 |
| bench_if_40k.ome.tif | TiffSource | 40000x30000 | 1652.4 | 8.0 | 0 (÷1) | 2.55 / 17.47 | 207.2 | 0.005 | 70.4 |
|  |  |  |  |  | 4 (÷16) | 6.98 / 11.53 | 313.8 | 0.005 | 128.0 |
|  |  |  |  |  | 7 (÷128) | 27.0 / 31.0 | – | 0.008 | 58.2 |
| bench_if_24k.czi | CziSource | 24000x18000 | 644.3 | 16.7 | 0 (÷1) | 20.42 / 53.58 | 85.8 | 0.004 | 107.1 |
|  |  |  |  |  | 3 (÷8) | 4.92 / 15.05 | 436.1 | 0.004 | 112.6 |
|  |  |  |  |  | 6 (÷64) | 26.43 / 27.95 | – | 0.006 | 75.4 |


## Browser (1600×1000): ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver) — SOFTWARE rendering (no GPU)

| panes | open→sharp ms | 20×→sharp ms | pan fps (p95 ms) | viewer JS ms/frame p50 (p95) | long frames % | sharp while panning % | blank % | settle ms | revisit fps / sharp % | MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2434 | 1363 | 10.0 (188.3) | 0.7 (7.77) | 100.0 | 99.3 | 0.0 | 89 | 12.0 / 100.0 | 4.7 |
| 4 | 1508 | 2000 | 4.0 (344.2) | 4.4 (7.72) | 100.0 | 81.1 | 0.0 | 565 | 6.7 / 100.0 | 10.4 |
| 9 | 2676 | 2812 | 4.3 (296.7) | 3.3 (10.13) | 100.0 | 79.7 | 0.0 | 168 | 5.5 / 99.9 | 8.2 |
| 16 | 1291 | 3181 | 5.0 (250.0) | 4.85 (7.89) | 100.0 | 78.9 | 0.0 | 377 | 5.0 / 93.5 | 8.4 |


## Browser (800×500): ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver) — SOFTWARE rendering (no GPU)

| panes | open→sharp ms | 20×→sharp ms | pan fps (p95 ms) | viewer JS ms/frame p50 (p95) | long frames % | sharp while panning % | blank % | settle ms | revisit fps / sharp % | MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 513 | 438 | 59.9 (33.4) | 0.2 (3.2) | 7.4 | 100.0 | 0.0 | 23 | 59.9 / 100.0 | 2.9 |
| 4 | 172 | 655 | 29.9 (83.3) | 0.7 (6.2) | 48.7 | 100.0 | 0.0 | 27 | 30.0 / 100.0 | 14.7 |
| 16 | 1402 | 1458 | 15.0 (100.0) | 4.1 (7.8) | 100.0 | 84.8 | 0.0 | 78 | 20.0 / 100.0 | 21.6 |

## Software-WebGL ceiling in the same browser

A bare page drawing 16 textured full-screen quads per frame (no viewer, no loading):
1600×1000 → 20 fps; 800×500 → 60 fps (vsync). The viewer's own JavaScript is 0.2–5 ms per frame
in every run, so the remaining frame time is CPU-emulated rasterisation — a GPU removes it.
