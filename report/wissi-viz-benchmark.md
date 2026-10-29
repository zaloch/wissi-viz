# wissi-viz 0.12.0 benchmark

5 public slides (quick set) · Linux-7.0.0-1017-azure-x86_64-with-glibc2.39 · Python 3.12.3 · 4 CPUs

## Slide opening and tile serving

| slide | open (ms) | cold tile p50 (ms) | warm tile p50 (ms) | tiles/s |
|---|---|---|---|---|
| CMU-1.svs (46000x32914) | 6.2 | 1.09 | 0.004 | 664.9 |
| CMU-2.svs (78000x30462) | 7.8 | 1.12 | 0.003 | 693.4 |
| CMU-3.svs (66000x45402) | 9.4 | 1.04 | 0.004 | 686.2 |
| xenium-breast.ome.tif (5792x3529) | 4.8 | 10.83 | 0.003 | 234.7 |
| xenium-kidney-protein.ome.tif (2963x6915) | 24.6 | 1.04 | 0.003 | 244.3 |

## Panning in the browser

ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver) (software), http/1.1

| panes | open (ms) | zoom to sharp (ms) | fps (median) | frame p95 (ms) | sharp while moving | settle (ms) | GPU textures (MB) |
|---|---|---|---|---|---|---|---|
| 1 | 816 | 287 | 59.9 | 16.8 | 100.0 % | 16 | 66.4 |
| 4 | 1397 | 108 | 59.9 | 33.3 | 99.0 % | 468 | 208.3 |
| 16 | 6747 | 450 | 59.9 | 33.4 | 99.0 % | 1543 | 570.6 |
| 32 | 6910 | 573 | 59.9 | 33.4 | 98.7 % | 1194 | 1020.2 |

## SAM click segmentation

backend **onnx** on cpu (model sam2.1-tiny (int8, bundled))

| slide | view | first click (ms) | next click (ms) |
|---|---|---|---|
| CMU-1.svs | overview | 2289.3 | 28.9 |
| CMU-1.svs | 20x field | 1376.2 | 30.1 |
| CMU-2.svs | overview | 1243.2 | 26.1 |
| CMU-2.svs | 20x field | 1254.9 | 25.3 |
| CMU-3.svs | overview | 1230.4 | 24.5 |
| CMU-3.svs | 20x field | 1241.6 | 34.9 |

## Region metrics

| image | cells | measurements | tissue (s) | metrics (s) |
|---|---|---|---|---|
| cmu-1 | 268,373 | 124 | 0.39 | 3.21 |
| cmu-2 | 411,148 | 124 | 2.12 | 16.76 |

## Annotation formats

411,148 cells (cmu-2.cells.parquet)

| | size (MB) | write (s) | read (s) |
|---|---|---|---|
| GeoParquet | 64.1 | 1.03 | 0.48 |
| GeoJSON | 330.7 | 13.57 | 22.72 |
