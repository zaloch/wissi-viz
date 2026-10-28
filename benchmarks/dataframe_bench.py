"""pandas vs Polars (vs pyarrow) on a 1M-cell object table: reading and typical cell-table analysis.

Every (library, operation) runs in a fresh process, timed after imports, median of 3 runs, peak RSS from VmHWM.

    python benchmarks/dataframe_bench.py /var/tmp/objbench --out df.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

GEO = ['XMin', 'XMax', 'YMin', 'YMax', 'Classification']
CD3, CD8 = 'CD3 Cell Intensity', 'CD8 Cell Intensity'
MEAS = [
    f'{m} {c} {s}'
    for m in ('DAPI', 'CD3', 'CD8', 'PanCK', 'CD68')
    for c in ('Nucleus', 'Cytoplasm', 'Cell')
    for s in ('Intensity', 'Positive %')
]


def windows(n=20, seed=1):
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        (float(x), float(y), float(x) + 8000, float(y) + 5000) for x, y in rng.uniform([0, 0], [92000, 75000], (n, 2))
    ]


def load(lib, d):
    if lib == 'pandas':
        import pandas as pd

        return pd.read_parquet(d / 'cells.parquet')
    if lib == 'polars':
        import polars as pl

        return pl.read_parquet(d / 'cells.parquet')
    import pyarrow.parquet as pq

    return pq.read_table(d / 'cells.parquet')


def op(lib: str, name: str, d: str):
    d = Path(d)
    if lib == 'pandas':
        import pandas as pd
    elif lib == 'polars':
        import polars as pl
    else:
        import pyarrow as pa  # noqa: F401
        import pyarrow.compute as pc
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    import numpy as np

    pre = None
    if name.startswith('a_'):
        pre = load(lib, d)
    t0 = time.perf_counter()
    out = None
    # ------------------------------------------------------------------ reading
    if name == 'read_csv':
        out = (
            pd.read_csv(d / 'cells.csv')
            if lib == 'pandas'
            else pl.read_csv(d / 'cells.csv')
            if lib == 'polars'
            else pacsv.read_csv(d / 'cells.csv')
        )
    elif name == 'read_tsv':
        out = (
            pd.read_csv(d / 'cells.tsv', sep='\t')
            if lib == 'pandas'
            else pl.read_csv(d / 'cells.tsv', separator='\t')
            if lib == 'polars'
            else pacsv.read_csv(d / 'cells.tsv', parse_options=pacsv.ParseOptions(delimiter='\t'))
        )
    elif name == 'read_csv_viewer_cols':
        out = (
            pd.read_csv(d / 'cells.csv', usecols=GEO)
            if lib == 'pandas'
            else pl.read_csv(d / 'cells.csv', columns=GEO)
            if lib == 'polars'
            else pacsv.read_csv(d / 'cells.csv', convert_options=pacsv.ConvertOptions(include_columns=GEO))
        )
    elif name == 'read_parquet':
        out = load(lib, d)
    elif name == 'read_parquet_viewer_cols':
        out = (
            pd.read_parquet(d / 'cells.parquet', columns=GEO)
            if lib == 'pandas'
            else pl.read_parquet(d / 'cells.parquet', columns=GEO)
            if lib == 'polars'
            else pq.read_table(d / 'cells.parquet', columns=GEO)
        )
    elif name == 'parquet_window_pushdown':  # read only the cells in one viewport (Hilbert-sorted file, 20 windows)
        n = 0
        for x0, y0, x1, y1 in windows():
            if lib == 'pandas':
                n += len(
                    pd.read_parquet(
                        d / 'cells_hilbert.parquet',
                        columns=GEO,
                        filters=[('XMax', '>=', x0), ('XMin', '<=', x1), ('YMax', '>=', y0), ('YMin', '<=', y1)],
                    )
                )
            elif lib == 'polars':
                n += (
                    pl.scan_parquet(d / 'cells_hilbert.parquet')
                    .filter(
                        (pl.col('XMax') >= x0)
                        & (pl.col('XMin') <= x1)
                        & (pl.col('YMax') >= y0)
                        & (pl.col('YMin') <= y1)
                    )
                    .select(GEO)
                    .collect()
                    .height
                )
            else:
                f = (
                    (pc.field('XMax') >= x0)
                    & (pc.field('XMin') <= x1)
                    & (pc.field('YMax') >= y0)
                    & (pc.field('YMin') <= y1)
                )
                n += pq.read_table(d / 'cells_hilbert.parquet', columns=GEO, filters=f).num_rows
        out = n
    # ------------------------------------------------------------------ analysis (table already in memory)
    elif name == 'a_class_counts':
        out = (
            pre['Classification'].value_counts()
            if lib == 'pandas'
            else pre['Classification'].value_counts()
            if lib == 'polars'
            else pc.value_counts(pre['Classification'])
        )
    elif name == 'a_gate_phenotype':  # CD3+/CD8+ gating → phenotype label → counts
        if lib == 'pandas':
            ph = np.where(pre[CD3] > 100, np.where(pre[CD8] > 120, 'CD8 T', 'CD4 T'), 'other')
            out = pd.Series(ph).value_counts()
        elif lib == 'polars':
            out = pre.select(
                pl.when(pl.col(CD3) > 100)
                .then(pl.when(pl.col(CD8) > 120).then(pl.lit('CD8 T')).otherwise(pl.lit('CD4 T')))
                .otherwise(pl.lit('other'))
                .alias('ph')
            )['ph'].value_counts()
        else:
            c3, c8 = pc.greater(pre[CD3], 100), pc.greater(pre[CD8], 120)
            out = pc.value_counts(pc.if_else(c3, pc.if_else(c8, 'CD8 T', 'CD4 T'), 'other'))
    elif name == 'a_viewport_filter':  # 20 viewports, in memory
        n = 0
        for x0, y0, x1, y1 in windows():
            if lib == 'pandas':
                n += len(pre[(pre.XMax >= x0) & (pre.XMin <= x1) & (pre.YMax >= y0) & (pre.YMin <= y1)])
            elif lib == 'polars':
                n += pre.filter(
                    (pl.col('XMax') >= x0) & (pl.col('XMin') <= x1) & (pl.col('YMax') >= y0) & (pl.col('YMin') <= y1)
                ).height
            else:
                m = pc.and_(
                    pc.and_(pc.greater_equal(pre['XMax'], x0), pc.less_equal(pre['XMin'], x1)),
                    pc.and_(pc.greater_equal(pre['YMax'], y0), pc.less_equal(pre['YMin'], y1)),
                )
                n += pre.filter(m).num_rows
        out = n
    elif name == 'a_groupby_means':  # per class: mean of 30 measurements
        if lib == 'pandas':
            out = pre.groupby('Classification')[MEAS].mean()
        elif lib == 'polars':
            out = pre.group_by('Classification').agg([pl.col(c).mean() for c in MEAS])
        else:
            out = pre.group_by('Classification').aggregate([(c, 'mean') for c in MEAS])
    elif name == 'a_region_class_pct':  # part of whole: % of each class per region
        if lib == 'pandas':
            ct = pd.crosstab(pre['Analysis Region'], pre['Classification'])
            out = ct.div(ct.sum(1), axis=0) * 100
        elif lib == 'polars':
            out = (
                pre.group_by(['Analysis Region', 'Classification'])
                .len()
                .with_columns((pl.col('len') / pl.col('len').sum().over('Analysis Region') * 100).alias('pct'))
            )
        else:
            out = pre.group_by(['Analysis Region', 'Classification']).aggregate([('Object Id', 'count')])
    elif name == 'a_top_k':  # 1000 brightest CD8 cells
        out = (
            pre.nlargest(1000, CD8)
            if lib == 'pandas'
            else pre.top_k(1000, by=CD8)
            if lib == 'polars'
            else pre.take(pc.select_k_unstable(pre[CD8], 1000, [(CD8, 'descending')]))
        )
    elif name == 'a_to_numpy_viewer':  # hand geometry + class to wissi-viz
        out = [pre[c].to_numpy() for c in GEO]  # same call for pandas, polars and arrow
    dt = time.perf_counter() - t0
    hwm = [int(x.split()[1]) for x in open('/proc/self/status', encoding='utf-8') if x.startswith('VmHWM')][0] / 1024
    del out  # the result stays alive until the memory high-water mark is read
    return {'s': dt, 'peak_MB': round(hwm)}


OPS = [
    'read_csv',
    'read_tsv',
    'read_csv_viewer_cols',
    'read_parquet',
    'read_parquet_viewer_cols',
    'parquet_window_pushdown',
    'a_class_counts',
    'a_gate_phenotype',
    'a_viewport_filter',
    'a_groupby_means',
    'a_region_class_pct',
    'a_top_k',
    'a_to_numpy_viewer',
]


def run(lib, name, d):
    code = (
        f'import json, sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); import dataframe_bench as m; '
        f'print(json.dumps(m.op({lib!r}, {name!r}, {str(d)!r})))'
    )
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=900)
    if p.returncode:
        return {'error': (p.stderr.strip().splitlines() or ['?'])[-1]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    import statistics

    res = {}
    for name in OPS:
        res[name] = {}
        for lib in ('pandas', 'polars', 'pyarrow'):
            rs = [run(lib, name, a.dir) for _ in range(a.repeats)]
            ok = [r for r in rs if 'error' not in r]
            res[name][lib] = (
                {'s': round(statistics.median(r['s'] for r in ok), 3), 'peak_MB': max(r['peak_MB'] for r in ok)}
                if ok
                else rs[0]
            )
        print(name, json.dumps(res[name]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
