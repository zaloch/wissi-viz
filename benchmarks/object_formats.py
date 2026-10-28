"""How should object/cell tables (10^5–10^7 rows) be stored for the viewer and analysis?

Builds one synthetic HALO-style export:
- 1M cells over a 100k × 80k slide;
- XMin/XMax/YMin/YMax, Classification, Analysis Region and 30 float measurement columns;
- written as CSV, TSV, Parquet (zstd; plus a Hilbert-sorted copy), Feather/Arrow IPC, SQLite (+ R*Tree),
  DuckDB and PostgreSQL (GiST on box).

For each format it measures:
- **full**: load every column into numpy;
- **viewer**: load only geometry + class (what drawing needs);
- **window**: cells in a 8000×5000 px viewport (~5k cells), mean of 20 random windows;
- **click**: one cell's row by id, mean of 50;
- **size** on disk and **peak RSS** (each operation runs in a fresh subprocess).

    python benchmarks/object_formats.py /tmp/objbench --rows 1000000 [--postgres]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

W, H = 100_000, 80_000
CLASSES = ['Tumor', 'Stroma', 'CD8 T cell', 'CD4 T cell', 'Macrophage', 'B cell', 'Other']
MEAS = [
    f'{m} {c} {s}'
    for m in ('DAPI', 'CD3', 'CD8', 'PanCK', 'CD68')
    for c in ('Nucleus', 'Cytoplasm', 'Cell')
    for s in ('Intensity', 'Positive %')
]
GEO = ['XMin', 'XMax', 'YMin', 'YMax', 'Classification']


def make(n: int, seed: int = 0):
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    # clustered tissue: cells around 40 blobs, realistic spatial non-uniformity
    centres = rng.uniform([0, 0], [W, H], (40, 2))
    k = rng.integers(0, 40, n)
    xy = centres[k] + rng.normal(0, 6000, (n, 2))
    xy = np.clip(xy, 0, [W - 30, H - 30])
    r = rng.uniform(4, 12, n)
    t = {
        'Object Id': np.arange(n, dtype=np.int64),
        'XMin': (xy[:, 0] - r).astype(np.int32),
        'XMax': (xy[:, 0] + r).astype(np.int32),
        'YMin': (xy[:, 1] - r).astype(np.int32),
        'YMax': (xy[:, 1] + r).astype(np.int32),
        'Classification': np.asarray(CLASSES)[rng.integers(0, len(CLASSES), n)],
        'Analysis Region': np.asarray(['Layer 1', 'Layer 2', 'Margin'])[rng.integers(0, 3, n)],
    }
    for c in MEAS:
        t[c] = rng.gamma(2.0, 50.0, n).round(4)
    return pa.table(t)


def hilbert_order(x, y, bits=16):
    """Hilbert index of (x, y) on a 2^bits grid: nearby cells end up in the same Parquet row groups."""
    x = (x.astype(np.int64) * ((1 << bits) - 1) // W).astype(np.int64)
    y = (y.astype(np.int64) * ((1 << bits) - 1) // H).astype(np.int64)
    d = np.zeros_like(x)
    s = 1 << (bits - 1)
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        m = ~ry
        swap = m & rx
        x = np.where(swap, s - 1 - x, x)
        y = np.where(swap, s - 1 - y, y)
        x, y = np.where(m, y, x), np.where(m, x, y)
        s >>= 1
    return d


def write_all(d: Path, n: int, postgres: bool) -> dict:
    import duckdb
    import pyarrow.csv as pacsv
    import pyarrow.feather as pf
    import pyarrow.parquet as pq

    d.mkdir(parents=True, exist_ok=True)
    tbl = make(n)
    out = {}

    def timed(name, fn):
        t = time.perf_counter()
        fn()
        out[name] = round(time.perf_counter() - t, 2)

    timed('csv', lambda: pacsv.write_csv(tbl, d / 'cells.csv'))
    timed('tsv', lambda: pacsv.write_csv(tbl, d / 'cells.tsv', pacsv.WriteOptions(delimiter='\t')))
    timed('parquet', lambda: pq.write_table(tbl, d / 'cells.parquet', compression='zstd', row_group_size=65536))
    cx = (tbl['XMin'].to_numpy() + tbl['XMax'].to_numpy()) // 2
    cy = (tbl['YMin'].to_numpy() + tbl['YMax'].to_numpy()) // 2
    o = np.argsort(hilbert_order(cx, cy), kind='stable')
    timed(
        'parquet_hilbert',
        lambda: pq.write_table(
            tbl.take(o), d / 'cells_hilbert.parquet', compression='zstd', row_group_size=16384, write_statistics=True
        ),
    )
    timed('feather', lambda: pf.write_feather(tbl, d / 'cells.arrow', compression='uncompressed'))
    timed('feather_lz4', lambda: pf.write_feather(tbl, d / 'cells_lz4.arrow', compression='lz4'))

    def sqlite():
        import sqlite3

        p = d / 'cells.sqlite'
        p.unlink(missing_ok=True)
        con = sqlite3.connect(p)
        cols = tbl.column_names
        q = ', '.join(
            f'"{c}" {"TEXT" if c in ("Classification", "Analysis Region") else ("INTEGER" if c in ("Object Id", "XMin", "XMax", "YMin", "YMax") else "REAL")}'
            for c in cols
        )
        con.execute(
            f'CREATE TABLE cells ({q.replace(chr(34) + "Object Id" + chr(34) + " INTEGER", chr(34) + "Object Id" + chr(34) + " INTEGER PRIMARY KEY")})'
        )
        rows = zip(*[tbl[c].to_pylist() for c in cols])
        con.executemany(f'INSERT INTO cells VALUES ({",".join("?" * len(cols))})', rows)
        con.execute('CREATE VIRTUAL TABLE cells_rtree USING rtree(id, x0, x1, y0, y1)')
        con.execute('INSERT INTO cells_rtree SELECT "Object Id", XMin, XMax, YMin, YMax FROM cells')
        con.commit()
        con.close()

    timed('sqlite', sqlite)

    def duck():
        p = d / 'cells.duckdb'
        p.unlink(missing_ok=True)
        con = duckdb.connect(str(p))
        con.register('t', tbl)
        con.execute('CREATE TABLE cells AS SELECT * FROM t')
        con.close()

    timed('duckdb', duck)
    if postgres:
        timed('postgres', lambda: pg_load(d, tbl))
    sizes = {}
    for k, f in [
        ('csv', 'cells.csv'),
        ('tsv', 'cells.tsv'),
        ('parquet', 'cells.parquet'),
        ('parquet_hilbert', 'cells_hilbert.parquet'),
        ('feather', 'cells.arrow'),
        ('feather_lz4', 'cells_lz4.arrow'),
        ('sqlite', 'cells.sqlite'),
        ('duckdb', 'cells.duckdb'),
    ]:
        sizes[k] = round((d / f).stat().st_size / 2**20, 1)
    if postgres:
        sizes['postgres'] = pg_size()
    return {'write_s': out, 'size_MB': sizes}


# ------------------------------------------------------------------------------------------------ postgres
PG = {'host': '/tmp/wissi-pg', 'port': 55432, 'dbname': 'postgres', 'user': 'postgres'}


def pg_start(root: Path):
    """Throwaway PostgreSQL 16 cluster (runs as the postgres OS user)."""
    data = root / 'pgdata'
    sock = Path(PG['host'])
    bin_ = Path('/usr/lib/postgresql/16/bin')
    sock.mkdir(exist_ok=True)
    os.system(f'chown postgres {sock} {root}')
    if not data.exists():
        subprocess.run(['su', 'postgres', '-c', f'{bin_}/initdb -D {data} -A trust -U postgres >/dev/null'], check=True)
    subprocess.run(
        [
            'su',
            'postgres',
            '-c',
            f'{bin_}/pg_ctl -D {data} -o "-p {PG["port"]} -k {sock} -c shared_buffers=512MB '
            f'-c max_wal_size=4GB -c fsync=off -c synchronous_commit=off" -l {root}/pg.log -w start >/dev/null',
        ],
        check=True,
    )


def pg_stop(root: Path):
    subprocess.run(
        ['su', 'postgres', '-c', f'/usr/lib/postgresql/16/bin/pg_ctl -D {root / "pgdata"} -m fast stop >/dev/null']
    )


def pg_load(d: Path, tbl):
    import psycopg

    os.system(f'chmod a+r {d / "cells.csv"}; chmod a+x {d}')
    with psycopg.connect(**PG, autocommit=True) as con:
        con.execute('DROP TABLE IF EXISTS cells')
        cols = ', '.join(
            f'"{c}" {"text" if c in ("Classification", "Analysis Region") else ("integer" if c in ("XMin", "XMax", "YMin", "YMax") else ("bigint primary key" if c == "Object Id" else "double precision"))}'
            for c in tbl.column_names
        )
        con.execute(f'CREATE TABLE cells ({cols})')
        with con.cursor().copy('COPY cells FROM STDIN (FORMAT csv, HEADER true)') as cp:
            with open(d / 'cells.csv', 'rb') as f:
                while chunk := f.read(1 << 20):
                    cp.write(chunk)
        con.execute('CREATE INDEX cells_box ON cells USING gist (box(point("XMin", "YMin"), point("XMax", "YMax")))')
        con.execute('ANALYZE cells')


def pg_size():
    import psycopg

    with psycopg.connect(**PG) as con:
        return round(con.execute("select pg_total_relation_size('cells')").fetchone()[0] / 2**20, 1)


# ------------------------------------------------------------------------------------------------ operations
def _windows(n=20, seed=1):
    rng = np.random.default_rng(seed)
    return [
        (float(x), float(y), float(x) + 8000, float(y) + 5000)
        for x, y in rng.uniform([0, 0], [W - 8000, H - 5000], (n, 2))
    ]


def op(fmt: str, what: str, d: str) -> dict:
    """Runs in a fresh process: returns seconds and rows."""
    import pyarrow as pa

    d = Path(d)
    t0 = time.perf_counter()
    rows = 0
    cols = None if what == 'full' else GEO
    if what in ('full', 'viewer'):
        if fmt in ('csv', 'tsv'):
            import pyarrow.csv as pacsv

            ro = pacsv.ConvertOptions(include_columns=cols) if cols else None
            po = pacsv.ParseOptions(delimiter='\t' if fmt == 'tsv' else ',')
            t = pacsv.read_csv(d / f'cells.{fmt}', parse_options=po, convert_options=ro)
        elif fmt in ('csv_pandas', 'tsv_pandas'):
            import pandas as pd

            f = 'tsv' if fmt.startswith('tsv') else 'csv'
            t = pa.Table.from_pandas(
                pd.read_csv(d / f'cells.{f}', sep='\t' if f == 'tsv' else ',', usecols=cols, low_memory=False)
            )
        elif fmt.startswith('parquet'):
            import pyarrow.parquet as pq

            t = pq.read_table(
                d / ('cells_hilbert.parquet' if fmt == 'parquet_hilbert' else 'cells.parquet'), columns=cols
            )
        elif fmt.startswith('feather'):
            import pyarrow.feather as pf

            f = d / ('cells_lz4.arrow' if fmt == 'feather_lz4' else 'cells.arrow')
            t = pf.read_table(f, columns=cols, memory_map=fmt == 'feather')
        elif fmt == 'sqlite':
            import sqlite3

            con = sqlite3.connect(d / 'cells.sqlite')
            q = 'SELECT ' + (', '.join(f'"{c}"' for c in cols) if cols else '*') + ' FROM cells'
            cur = con.execute(q)
            names = [x[0] for x in cur.description]
            data = list(zip(*cur.fetchall()))
            t = pa.table(dict(zip(names, data)))
        elif fmt == 'duckdb':
            import duckdb

            con = duckdb.connect(str(d / 'cells.duckdb'), read_only=True)
            t = con.execute(
                'SELECT ' + (', '.join(f'"{c}"' for c in cols) if cols else '*') + ' FROM cells'
            ).to_arrow_table()
        elif fmt == 'postgres':
            import io

            import psycopg
            import pyarrow.csv as pacsv

            buf = io.BytesIO()
            with psycopg.connect(**PG) as con:
                q = 'SELECT ' + (', '.join(f'"{c}"' for c in cols) if cols else '*') + ' FROM cells'
                with con.cursor().copy(f'COPY ({q}) TO STDOUT (FORMAT csv, HEADER true)') as cp:
                    for chunk in cp:
                        buf.write(chunk)
            buf.seek(0)
            t = pacsv.read_csv(buf)
        # to numpy, as the viewer needs it
        for c in t.column_names:
            t[c].to_numpy(zero_copy_only=False)
        rows = t.num_rows
    elif what == 'window':
        ws = _windows()
        if fmt.startswith('parquet'):
            import pyarrow.dataset as ds

            dset = ds.dataset(d / ('cells_hilbert.parquet' if fmt == 'parquet_hilbert' else 'cells.parquet'))
            t1 = time.perf_counter()
            for x0, y0, x1, y1 in ws:
                f = (
                    (ds.field('XMax') >= x0)
                    & (ds.field('XMin') <= x1)
                    & (ds.field('YMax') >= y0)
                    & (ds.field('YMin') <= y1)
                )
                rows += dset.to_table(columns=GEO, filter=f).num_rows
        elif fmt == 'feather':
            import pyarrow.compute as pc
            import pyarrow.feather as pf

            t = pf.read_table(d / 'cells.arrow', columns=GEO, memory_map=True)
            t1 = time.perf_counter()
            for x0, y0, x1, y1 in ws:
                m = pc.and_(
                    pc.and_(pc.greater_equal(t['XMax'], x0), pc.less_equal(t['XMin'], x1)),
                    pc.and_(pc.greater_equal(t['YMax'], y0), pc.less_equal(t['YMin'], y1)),
                )
                rows += t.filter(m).num_rows
        elif fmt == 'sqlite':
            import sqlite3

            con = sqlite3.connect(d / 'cells.sqlite')
            t1 = time.perf_counter()
            for x0, y0, x1, y1 in ws:
                rows += len(
                    con.execute(
                        'SELECT c.XMin, c.XMax, c.YMin, c.YMax, c.Classification FROM cells_rtree r JOIN cells c '
                        'ON c."Object Id" = r.id WHERE r.x1 >= ? AND r.x0 <= ? AND r.y1 >= ? AND r.y0 <= ?',
                        (x0, x1, y0, y1),
                    ).fetchall()
                )
        elif fmt == 'duckdb':
            import duckdb

            con = duckdb.connect(str(d / 'cells.duckdb'), read_only=True)
            t1 = time.perf_counter()
            for x0, y0, x1, y1 in ws:
                rows += (
                    con.execute(
                        'SELECT XMin, XMax, YMin, YMax, Classification FROM cells WHERE XMax >= ? AND XMin <= ? '
                        'AND YMax >= ? AND YMin <= ?',
                        [x0, x1, y0, y1],
                    )
                    .to_arrow_table()
                    .num_rows
                )
        elif fmt == 'postgres':
            import psycopg

            con = psycopg.connect(**PG)
            t1 = time.perf_counter()
            for x0, y0, x1, y1 in ws:
                rows += len(
                    con.execute(
                        'SELECT "XMin", "XMax", "YMin", "YMax", "Classification" FROM cells WHERE '
                        'box(point("XMin", "YMin"), point("XMax", "YMax")) && box(point(%s, %s), point(%s, %s))',
                        (x0, y0, x1, y1),
                    ).fetchall()
                )
        else:
            return {'skip': True}
        return {'seconds': (time.perf_counter() - t1) / len(ws), 'open_s': t1 - t0, 'rows': rows / len(ws)}
    elif what == 'click':
        ids = np.random.default_rng(2).integers(0, 10**6, 50).tolist()
        if fmt == 'sqlite':
            import sqlite3

            con = sqlite3.connect(d / 'cells.sqlite')
            t1 = time.perf_counter()
            for i in ids:
                rows += len(con.execute('SELECT * FROM cells WHERE "Object Id" = ?', (i,)).fetchall())
        elif fmt == 'duckdb':
            import duckdb

            con = duckdb.connect(str(d / 'cells.duckdb'), read_only=True)
            t1 = time.perf_counter()
            for i in ids:
                rows += len(con.execute('SELECT * FROM cells WHERE "Object Id" = ?', [i]).fetchall())
        elif fmt == 'postgres':
            import psycopg

            con = psycopg.connect(**PG)
            t1 = time.perf_counter()
            for i in ids:
                rows += len(con.execute('SELECT * FROM cells WHERE "Object Id" = %s', (i,)).fetchall())
        elif fmt == 'feather':
            import pyarrow.feather as pf

            t = pf.read_table(d / 'cells.arrow', memory_map=True)  # row i == id i (export order)
            t1 = time.perf_counter()
            for i in ids:
                rows += len(t.slice(i, 1).to_pylist())
        elif fmt.startswith('parquet'):
            import pyarrow.parquet as pq

            f = pq.ParquetFile(d / 'cells.parquet')
            t1 = time.perf_counter()
            for i in ids:
                rg = i // 65536
                rows += len(f.read_row_group(rg).slice(i - rg * 65536, 1).to_pylist())
        else:
            return {'skip': True}
        return {'seconds': (time.perf_counter() - t1) / len(ids), 'open_s': t1 - t0, 'rows': rows / len(ids)}
    return {'seconds': time.perf_counter() - t0, 'rows': rows}


def run_op(fmt, what, d):
    code = (
        f'import json, resource, sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); import object_formats as m; '
        f'r = m.op({fmt!r}, {what!r}, {str(d)!r}); '
        f'r["peak_rss_MB"] = round([int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")][0] / 1024, 1); print(json.dumps(r))'
    )
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=1800)
    if p.returncode:
        return {'error': p.stderr.strip().splitlines()[-1] if p.stderr else 'failed'}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--rows', type=int, default=1_000_000)
    ap.add_argument('--postgres', action='store_true')
    ap.add_argument('--skip-write', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    if a.postgres:
        pg_start(d if d.exists() else (d.mkdir(parents=True) or d))
    try:
        res = {'rows': a.rows}
        if not a.skip_write:
            res.update(write_all(d, a.rows, a.postgres))
            print(json.dumps(res), flush=True)
        fmts = [
            'csv_pandas',
            'tsv_pandas',
            'csv',
            'tsv',
            'parquet',
            'parquet_hilbert',
            'feather',
            'feather_lz4',
            'sqlite',
            'duckdb',
        ]
        if a.postgres:
            fmts.append('postgres')
        res['ops'] = {}
        for f in fmts:
            res['ops'][f] = {}
            for w in ('full', 'viewer', 'window', 'click'):
                r = run_op(f, w, d)
                if r.get('skip'):
                    continue
                res['ops'][f][w] = r
                print(f, w, json.dumps(r), flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    finally:
        if a.postgres:
            pg_stop(d)


if __name__ == '__main__':
    main()
