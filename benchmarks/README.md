# Benchmarks

## Run it on your machine (real slides, real GPU)

```bash
pip install "wissi-viz[all,dev]" && playwright install chromium
wissi-viz fetch-samples ~/wsi-samples --set standard        # OpenSlide test data: Aperio SVS, generic TIFF,
                                                            # NDPI, Leica SCN, Zeiss CZI (~1–2 GB)
wissi-viz bench ~/wsi-samples/Aperio/CMU-1.svs ~/wsi-samples/Leica/*.scn ~/wsi-samples/Zeiss/*.czi \
                --headed --panes 1,4,9,16 --out results.json   # → results.json + results.md
```

- `--headed` opens a real browser window, so Chromium uses your desktop GPU. Headless Chromium on a server or in a container falls back to SwiftShader, which is software WebGL running on the CPU.
- The report prints the WebGL renderer string, so you can tell which you got. "SwiftShader" means software; a GPU vendor string means hardware.
- `--server-only` measures just opening the file and serving tiles, with no browser.
- `--set full` adds every OpenSlide vendor folder and a CAMELYON16 slide (CC0, ~0.9 GB).
- `--url` adds any other public URL.

- **Tiles from a remote API** (PathCore, HALO Link, …) instead of files: see [results-tile-api.md](results-tile-api.md) and `examples/tile_api/bench_api.py [--http2]`.
- **Smoothness** (freezes, jank, input latency under real mouse input, 1 and 16 panes): `python benchmarks/smoothness.py --out smooth.json`. See [results-smoothness.md](results-smoothness.md).
- **GPU pyramid build (experimental):** the client-mode object pyramid built with WebGPU compute vs the Web Worker, with a full parity check: `python benchmarks/gpu_pyramid_ab.py --headed`. See [results-gpu-pyramid.md](results-gpu-pyramid.md).

### Free sources

| Source | Formats | Notes |
|---|---|---|
| [OpenSlide test data](https://openslide.cs.cmu.edu/download/openslide-testdata/) | Aperio SVS, Leica SCN (brightfield and fluorescence), Zeiss CZI, Hamamatsu NDPI, generic tiled TIFF, Ventana, MIRAX, DICOM | Licences are per file; see the README in each folder. |
| [OME sample data](https://downloads.openmicroscopy.org/images/) | OME-TIFF (including pyramids), Zeiss CZI, plus many microscopy formats | Bio-Formats test corpus. |
| [CAMELYON16/17](https://registry.opendata.aws/camelyon/) | pyramidal TIFF (H&E lymph nodes) | CC0. `aws s3 ls --no-sign-request s3://camelyon-dataset/`. |
| [TCGA via GDC](https://portal.gdc.cancer.gov/) | Aperio SVS (diagnostic and tissue slides) | Open-access slides; very large. |
| [NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) | DICOM WSI | Available on GCS/AWS. |

## What the numbers mean

**Server**
- **open:** time to parse the file and build the pyramid.
- **cold tile:** read, decode and re-encode one 512² tile that is in no cache.
- **tiles/s:** 8 concurrent requests for distinct, uncached tiles. Panning several panes generates this kind of load.
- **warm:** the memory-cache hit time.

**Browser**
- **open→sharp:** time until every pane is fully sharp at the fit view.
- **20×→sharp:** the same after jumping to 20×.
- **pan:** an 8 s Lissajous pan of every pane at 900 screen px/s. It records frame times from `requestAnimationFrame` (fps, p95, % of frames over 33 ms) and the share of the view that is sharp or blank while moving.
- **settle:** time to become fully sharp again after the pan stops.
- **revisit:** the same pan a second time, with caches warm.
- **viewer JS ms/frame:** the viewer's own JavaScript per frame, including WebGL command submission. Total frame time minus this is time spent rasterising and compositing on the GPU, or, under SwiftShader, on the CPU.

## Synthetic full-size slides

`make_bench_slides.py` builds full-size test slides from any real H&E slide. It mirror-tiles the real tissue, so codec cost and tile sizes are realistic:
- a 64k×48k Aperio SVS;
- a 40k×30k 3-channel uint16 OME-TIFF with a SubIFD pyramid;
- a 24k×18k 3-channel uint16 CZI built from overlapping zstd mosaic tiles with pyramid subblocks.

```bash
python benchmarks/make_bench_slides.py CMU-1-Small-Region.svs ./slides
```
