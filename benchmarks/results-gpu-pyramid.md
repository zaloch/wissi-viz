# Building the client-mode object pyramid on the GPU (experimental)

In client overlay mode (`overlays: 'client'`, or `'auto'` for layers of 300k objects or fewer), the browser downloads a layer's columns once and builds the object pyramid itself. That pyramid is the full-detail level for outlines, plus aggregated dots with per-class histograms at every coarser zoom. Today a Web Worker builds it. This experiment builds the same pyramid with **WebGPU compute shaders** and checks that the output is identical.

```bash
python benchmarks/gpu_pyramid_ab.py --headed                 # on your desktop GPU (a real browser window)
python benchmarks/gpu_pyramid_ab.py --headed --table cells.parquet
```

Files:
- `benchmarks/gpu_pyramid/gpu_pyramid.js`: the GPU builder. It is not wired into the viewer yet.
- `tests/test_gpu_pyramid.py`: the parity test.

## Where the Worker spends its time (1M cells)

| phase | 100k | 300k | 1M |
|---|---|---|---|
| aggregated levels: binning (counting sorts, level by level) | 201 ms | 328 ms | **863 ms** |
| aggregated levels: slicing into tiles | 99 ms | 153 ms | 451 ms |
| raw level: tiles | 63 ms | 95 ms | 378 ms |
| parse, transfer | ~70 ms | ~50 ms | ~170 ms |
| **total** | 434 ms | 626 ms | **1,864 ms** |

## The GPU build

Power-of-two bins nest. So a level-k dot, which is the count-weighted mean of the level below, equals the plain mean of all objects in its bin. That means every level can be built **straight from the objects, independently**:

| pass | work |
|---|---|
| `agg_insert` (per object) | hash-insert its bin, count, 8-slot class histogram, and the minimum "quadrant path" |
| `agg_sum` (per object) | fixed-point position sums relative to the bin origin; representative = lowest index on the minimum path |
| `agg_tcount` (per bin) | occupied bins per 128×128-bin tile |
| `scan` | prefix sum over tiles |
| `agg_scatter` (per bin) | mean → tile-relative u16 position, dominant class, normalised histogram, written in tile order |
| `raw_*` (per object) | the full-detail level: the same count, scan and scatter |

The CPU then reads back exactly the bytes used and makes per-tile views; no per-tile copies are needed.

- **Same representative object.** The quadrant path reproduces the Worker's representative object for each dot. That is the cell a click on an aggregated dot resolves to.

## Results (2 vCPUs, no GPU: Chromium's WebGPU runs on SwiftShader, CPU-emulated)

| cells | Worker | GPU build | GPU compute (emulated) | upload | readback | per-tile views | GPU memory |
|---|---|---|---|---|---|---|---|
| 100k | 416 ms | 585 ms | 544 ms | 6 ms | 28 ms | 11 ms | 30 MB |
| 300k | 604 ms | 1,313 ms | 1,250 ms | 3 ms | 54 ms | 10 ms | 69 MB |
| 1M synthetic | 1,293 ms | 3,920 ms | 3,712 ms | 62 ms | 139 ms | 20 ms | 229 MB |
| 1M real (`cells.parquet`) | 1,751 ms | 4,390 ms | 4,087 ms | 60 ms | 200 ms | 28 ms | 231 MB |

**Parity, at every size and on the real table:**
- Identical levels, tiles and points per tile.
- The same representative object for every dot.
- Identical counts, classes, 8-slot histograms, sizes and outline shapes.
- Positions within **1/65535 of a tile**: at most 0.06 px at the finest aggregated level, for 0.2–0.3% of points. This is 32-bit vs 64-bit float rounding.

**Reading the numbers:**
- **Emulated compute is slower here.** On a GPU-less machine, compute emulated by SwiftShader is 2.5–3× *slower* than the Worker. GPU-less clients must keep the Worker; the builder can tell from `adapter.info.architecture === 'swiftshader'` or `isFallbackAdapter`.
- **The floor a real GPU can't remove** is upload + readback + views: about 220–290 ms at 1M here, against a 1.3–1.8 s Worker build. Even with compute at 0 ms, the ceiling in this container is about 6×.
- **Estimate only, not measured:** on a real GPU the compute is about 20 M simple atomic operations, typically tens of ms. Readback over a real memory bus is also faster than SwiftShader's. Expect roughly **0.1–0.3 s instead of 1.3–1.8 s per 1M cells**. Run `--headed` on your hardware to replace this estimate with a measurement.

## If it is wired in

- **Choosing the path.** Use the GPU only on a hardware adapter, and keep the Worker everywhere else. Run the GPU build inside a dedicated worker: WebGPU is available there, and the main thread then never pays the 30–90 ms of CPU-side setup and views.
- **Memory.** The hash table is sized for the largest level: 117 MB at 1M objects, 229 MB in all. Building levels sequentially with a smaller table, or reusing the output buffer, would roughly halve that.
- **Larger gains need a WebGPU renderer.** The viewer renders with WebGL2, and WebGPU buffers can't be shared with a WebGL context. So the result is read back and uploaded again. A WebGPU renderer could draw straight from the compute output (no readback), and could also cull and aggregate per frame.
- **Not covered yet:** `value` mode (layers coloured by a measurement). It needs 64-bit-safe sums.
