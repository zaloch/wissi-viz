# wissi-viz 0.13.0 benchmark

5 public slides (quick set) · Linux-7.0.0-1017-azure-x86_64-with-glibc2.39 · Python 3.12.3 · 4 CPUs

## Slide opening and tile serving

| slide | open (ms) | cold tile p50 (ms) | warm tile p50 (ms) | tiles/s |
|---|---|---|---|---|
| CMU-1.svs (46000x32914) | 8.3 | 1.09 | 0.004 | 714.9 |
| CMU-2.svs (78000x30462) | 10.4 | 1.21 | 0.004 | 619.6 |
| CMU-3.svs (66000x45402) | 17.3 | 1.08 | 0.003 | 627.9 |
| xenium-breast.ome.tif (5792x3529) | 6.8 | 10.95 | 0.004 | 232.9 |
| xenium-kidney-protein.ome.tif (2963x6915) | 37.7 | 1.08 | 0.003 | 242.4 |

## Panning in the browser

ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver) (software), http/1.1

| panes | open (ms) | zoom to sharp (ms) | fps (median) | frame p95 (ms) | sharp while moving | settle (ms) | GPU textures (MB) |
|---|---|---|---|---|---|---|---|
| 1 | 840 | 262 | 59.9 | 16.8 | 100.0 % | 17 | 66.4 |
| 4 | 1130 | 148 | 59.9 | 33.4 | 99.1 % | 666 | 208.3 |
| 16 | 6950 | 400 | 59.9 | 33.4 | 98.9 % | 1589 | 576.2 |
| 32 | 7167 | 610 | 59.9 | 33.4 | 98.8 % | 1285 | 869.9 |

## SAM click segmentation

backend **onnx** on cpu (model sam2.1-tiny (int8, bundled))

| slide | view | first click (ms) | next click (ms) |
|---|---|---|---|
| CMU-1.svs | overview | 2471.6 | 28.9 |
| CMU-1.svs | 20x field | 1383.2 | 25.4 |
| CMU-2.svs | overview | 1267.3 | 23.7 |
| CMU-2.svs | 20x field | 1248.8 | 25.1 |
| CMU-3.svs | overview | 1265.6 | 29.6 |
| CMU-3.svs | 20x field | 1352.1 | 26.1 |

## Region metrics

| image | cells | measurements | tissue (s) | metrics (s) |
|---|---|---|---|---|
| cmu-1 | 268,373 | 124 | 0.4 | 3.17 |
| cmu-2 | 411,148 | 124 | 2.09 | 16.96 |

## Annotation formats

411,148 cells (cmu-2.cells.parquet)

| | size (MB) | write (s) | read (s) |
|---|---|---|---|
| GeoParquet | 64.1 | 1.02 | 0.49 |
| GeoJSON | 330.7 | 13.61 | 23.26 |
