# Smoothness: freezes and jank in the main loop

`python benchmarks/smoothness.py --out smooth.json` drives the viewer with **real mouse input** in headless Chromium.

**Scenarios:**
- continuous drag-pan;
- bursts of wheel zoom in and out;
- quick flicks;
- hovering with no buttons pressed.

**Setup:**
- Real slides: a 64k SVS, a 40k OME-TIFF and a 24k CZI.
- A 1M-cell object overlay on every third pane.
- 1 and 16 panes in a 960×600 viewport, starting at 5× with everything loaded.

**Recorded per scenario:**
- An independent `requestAnimationFrame` clock (frame intervals).
- Input → next-frame latency.
- Long tasks over 50 ms.
- A Chromium trace: main-thread busy %, JS, style/layout, and GC with its longest pause.
- A CPU profile (JS hotspots).

The container has **no GPU**: 2 vCPUs, and Chromium falls back to SwiftShader (CPU-emulated WebGL with software compositing). That is the worst case. It is also what a hospital thin client or VDI desktop without a GPU gets.

## Before → after (0.5.1 → 0.5.2)

| panes | scenario | fps | frame p95 (ms) | frames > 33 ms (%) | input → frame p95 (ms) | longest task (ms) | longest GC (ms) | main thread busy |
|---|---|---|---|---|---|---|---|---|
| 1 | drag-pan | 47 → **68** | 50 → **17** | 11 → **1.1** | 16 → **15** | 82 → **61** | 1.4 → 5.1 | 70% → 44% |
| 1 | wheel-zoom | 18 → **58** | 135 → **33** | 44 → **4.3** | 375 → **78** | 156 → **85** | 0.7 → 1.6 | 95% → 93% |
| 1 | fling | 26 → **50** | 117 → **50** | 26 → **10** | 15 → 25 | 155 → **69** | 4.5 → 1.1 | 80% → 76% |
| 1 | hover | 68 → **72** | 17 → 17 | 0 → 0 | 7.5 → 7.2 | 0 → 0 | 0 → 2.8 | 8% → 11% |
| 16 | drag-pan | 14 → **32** | 150 → **83** | 74 → **38** | 109 → **66** | 323 → **150** | 4.3 → 7.1 | 97% → 92% |
| 16 | wheel-zoom | 12 → **35** | 317 → **67** | 92 → **31** | 781 → **136** | 384 → **129** | **127 → 8.4** | 99% → 98% |
| 16 | fling | 4.6 → **34** | 365 → **100** | 91 → **33** | 347 → **60** | 389 → **143** | 1.1 → 11 | 100% → 87% |
| 16 | hover | 59 → **61** | 17 → 17 | 3.0 → 2.4 | 7.8 → 10 | **303 → 123** | 0.4 → 1.1 | 36% → 23% |

Raw data: `smoothness_before-container.json` and `smoothness_after-container.json`. On a machine with a GPU, most of these frames take a few ms either way. The fixes matter most where frames are expensive, and they cost nothing where frames are cheap.

## What was found, and what changed

| Stall | Evidence | Fix |
|---|---|---|
| **Cell outlines submitted for whole buckets.** At ≥ 1:1, every overlapping detail bucket drew all of its cells: 12,755 instanced shapes per pane, though about 200 are on screen. | GPU time per frame measured with a synchronous `readPixels`: **166 ms for 1 pane, 905 ms for 16** with overlays, vs 20 / 42 ms without. Without a GPU, each instance costs about 10 µs even when clipped. | Shapes are binned on a coarse grid per bucket or tile (counting sort, row-major). Only the grid rows that intersect the view are drawn: WebGL2 has no base-instance, so the instance attributes are re-pointed per row. **905 → 57 ms**, pixel-identical (tested). |
| **Browser readback of the whole canvas each frame.** | Trace: `GLES2::ReadPixels` inside `Commit` on every frame (software compositing). It scales with backing-store pixels: half the DPR took 16 panes from **11 → 40 fps**. | **Adaptive resolution** (`lod.adaptive`, on by default). While something moves *and* frames are slow, the WebGL backing store shrinks to ¾ or ½ and CSS stretches it. Full resolution returns on the first frame at rest (tested pixel-identical). Rendering into a small offscreen buffer and blitting it up was tried first; it cut our fill cost but not the browser's readback. |
| **The frame-time signal was blind to input-driven frames.** | During drags, `_frameMs` was only sampled while an animation chained frames, so the controller never saw the slow frames. | Frame cost is now measured directly. A message posted at the end of each frame lands after Chrome's style/paint/commit. The controller uses the worse of that and the frame clock. |
| **Stream of tiles at rest → back-to-back full-res frames.** | After every flick, each arriving tile triggered another 100–300 ms frame, so the next input waited. | When frames are expensive (≥ 40 ms) and nothing moves, tile arrivals are batched into one redraw every ~2 frame costs (≤ 250 ms), with a 40 ms upload budget. Never active on a GPU machine. |
| **Garbage from tile-key strings.** | Profile: `queueLevel` took 212 ms per 4 s, and the longest major GC paused for **127 ms**. | Tile keys are cached by a numeric id, and request URLs are built lazily. queueLevel 212 → 5 ms; longest GC 127 → 8 ms. |
| **Per-pane `setTimeout` for every motion frame.** | 16 timers re-armed per frame. | One grid-wide rest timer (`_wakeAt`). |
| **DOM writes every frame.** | Scale bar, pointer coordinates and status text were rewritten on each frame. Text layout and paint compete with WebGL frames. | Scale bar only when its value changes. Coordinates once per frame, only when changed. The demo page's status line throttled to 4 Hz with a trailing update: **12 → 49 fps** in the 16-pane continuous-redraw test. |
| **2-D overlay canvas cleared every frame.** | `clearRect` on a full-screen canvas made the compositor re-upload it per frame. | Touched only when a selection or drawing is shown, plus one final clear (tested: 0 clears in 30 idle frames). |
| **16 `viewchange` events per frame over the NiceGUI websocket.** | One event per pane per emit. | One batched `views` event holding every pane's snapshot, at most about 8/s, with the trailing snapshot always delivered. NiceGUI listens with `throttle=0.15`. |

## Compositing path

`--compositor gpu` forces GPU-path compositing (`--enable-gpu-compositing --use-angle=swiftshader`). The readback leaves the main thread: it is only 4–20% busy, and input stays responsive. With no real GPU, though, SwiftShader in the GPU process competes for the same 2 cores, and fps was no better. On real hardware Chrome uses GPU compositing by default and there is no readback at all.

## For apps that embed the viewer

- Don't update the DOM on every `viewchange`. Use the `views` event: one snapshot of every pane, throttled. If you must show live coordinates, coalesce updates to a few Hz.
- `grid.stats()` reports `renderScale`, `lowresFrames`, `backingResizes` and `coalesced`, so you can see adaptive behaviour in the field.
- `setLod({adaptive: false})` turns adaptive resolution off, for example for screen recording at full resolution. `setLod('legacy')` restores pre-0.4 behaviour.
