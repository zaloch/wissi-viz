# pandas vs Polars (vs pyarrow) for object tables

**Data:** the same 1M-cell × 37-column HALO-style export as [results-overlays.md](results-overlays.md).

**Method:** each (library, operation) runs in a fresh process, timed after imports. Times are the median of 3 runs; peak memory is the process's peak RSS. The machine has 2 vCPUs, so Polars' multi-threading has less to work with here than on a larger server.

Versions: pandas 3.0 (Arrow-backed strings by default), Polars 2.1, pyarrow 26. All are permissive: BSD-3, MIT and Apache-2.0.

## The libraries on their own

| operation | pandas | Polars | pyarrow | fastest |
|---|---|---|---|---|
| read CSV (all 37 columns) | 3.29 s (661 MB) | 882 ms (656 MB) | 1.07 s (875 MB) | polars |
| read TSV (all columns) | 3.24 s (661 MB) | 866 ms (656 MB) | 1.06 s (871 MB) | polars |
| read CSV, geometry + class only | 1.65 s (208 MB) | 328 ms (384 MB) | 449 ms (610 MB) | polars |
| read Parquet (all columns) | 557 ms (776 MB) | 386 ms (374 MB) | 592 ms (777 MB) | polars |
| read Parquet, geometry + class only | 70 ms (198 MB) | 45 ms (113 MB) | 222 ms (193 MB) | polars |
| read one viewport from Parquet ×20 (predicate push-down) | 316 ms (159 MB) | 295 ms (100 MB) | 465 ms (170 MB) | polars |
| class counts | 25 ms (772 MB) | 41 ms (406 MB) | 19 ms (778 MB) | pyarrow |
| gate CD3/CD8 → phenotype counts | 159 ms (865 MB) | 45 ms (428 MB) | 44 ms (796 MB) | pyarrow |
| viewport filter ×20 (in memory) | 183 ms (774 MB) | 139 ms (395 MB) | 157 ms (791 MB) | polars |
| mean of 30 measurements per class | 97 ms (793 MB) | 72 ms (430 MB) | 32 ms (776 MB) | pyarrow |
| part of whole: % class per region | 114 ms (848 MB) | 11 ms (386 MB) | 14 ms (779 MB) | polars |
| top 1,000 by CD8 | 31 ms (784 MB) | 122 ms (701 MB) | 41 ms (787 MB) | pandas |
| hand geometry + class to numpy | 49 ms (829 MB) | 70 ms (451 MB) | 68 ms (849 MB) | pandas |

**Reading:**
- **Polars parses CSV/TSV 3.8× faster than pandas,** and 5× faster when only geometry + class are read.
- **It loads Parquet with about half the memory** (374 vs 776 MB).
- **CSV and TSV perform the same.**

**Analysis at 10⁶ rows:** every library takes milliseconds to ~0.2 s.
- **Polars is clearly ahead on expression-style work:** gating 3.5×, part-of-whole 10×.
- **pandas is quicker than Polars at** `value_counts` and `nlargest`.
- **The bigger difference is memory:** a Polars process holding the table peaks at ~400 MB, against ~780 MB for pandas and pyarrow. That matters on a server holding many slides.

## Inside wissi-viz: `read_objects(path, engine=…)` end to end

This runs from the file to an `ObjectTable` (unit detection, labels, columns), then `to_layer()`:

| file | pandas | pyarrow | Polars |
|---|---|---|---|
| CSV | 5.45 s (1259 MB) | 1.85 s (1156 MB) | 2.07 s (991 MB) |
| TSV | 5.25 s (1259 MB) | 1.75 s (1158 MB) | 1.93 s (982 MB) |
| PARQUET | 1.16 s (1020 MB) | 1.01 s (1158 MB) | 1.13 s (942 MB) |

- **pyarrow was consistently 0.1–0.2 s faster than Polars here, end to end.** The parse itself is faster in Polars; the difference is in converting its columns to numpy. Both are ~3× faster than pandas for CSV/TSV.
- **Polars uses 15–20% less peak memory,** so `engine='auto'` picks Polars when installed, then pyarrow, then pandas.
- **Parquet is ~1 s with any library.** It is still the format to prefer, and `cached_object_layer` makes repeat opens 35 ms anyway.

Things this benchmark turned up and fixed:
- **`NaN` in integer columns:** QuPath writes `NaN` into columns Polars infers as integers, which made Polars fail to parse. The reader now retries with a full scan and `NaN`/empty as missing, then falls back to pyarrow.
- **Class labels:** normalised one Python object at a time (0.4 s per 10⁶). They are now dictionary-encoded in C, and Polars text columns are converted through their distinct values instead of 10⁶ string conversions.
- **Inputs:** Polars `DataFrame` and `LazyFrame`, and pyarrow `Table`, are accepted directly by `from_table`, `AnnotationLayer.from_table` and `grid.show(slide, ObjectTable)`.

## Recommendation

- **wissi-viz itself:** install the `[polars]` extra. It is the fastest reader with the lowest memory, and pyarrow is the fallback.
- **Your own analysis code in the NiceGUI app:** Polars for loading, filtering, gating and per-ROI summaries of large tables, in about half the memory. Use `.to_pandas()` only at the edges where a library expects pandas (seaborn, scanpy/anndata, statsmodels). wissi-viz takes either.
- **pandas** remains fine for small tables and for code that already depends on its ecosystem. Read through pyarrow (`pd.read_csv(..., engine='pyarrow')`) when the files are large.

Scripts: `benchmarks/dataframe_bench.py` (raw: `dataframe-container.json`) and `read_objects_engines-container.json`.
