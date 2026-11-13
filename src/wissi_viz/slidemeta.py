"""Slide metadata for the slide list: fields every slide has (from its path), fields a tile API gives per image, and
metadata tables (CSV / TSV / Parquet) whose key column names the slides. The list sorts, groups and filters by
these fields; nothing here opens a slide."""

from __future__ import annotations

import csv
import datetime as _dt
import math
import os
from pathlib import Path
from typing import Any

TABLE_EXTS = ('.csv', '.tsv', '.txt', '.parquet')
MAX_ROWS = 200_000


def auto_meta(path: str | None, *, remote: bool = False) -> dict[str, Any]:
    """Name, folder, format, size and date modified of a file; the service of a tile-API image ('service:id')."""
    if not path:
        return {}
    if remote:
        service, _, iid = path.partition(':')
        return {'Service': service, 'Image id': iid}
    p = Path(path)
    out: dict[str, Any] = {'Folder': p.parent.name, 'Format': p.suffix.lstrip('.').lower() or 'folder'}
    try:
        st = p.stat()
        if p.is_file():
            out['Size (MB)'] = round(st.st_size / 2**20, 1)
        out['Modified'] = _dt.date.fromtimestamp(st.st_mtime).isoformat()
    except OSError:
        pass
    return out


def clean_meta(meta: Any) -> dict[str, Any]:
    """A connector's metadata as flat {field: str | number | bool | None} (nested values become text)."""
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in list(meta.items())[:200]:
        k = str(k).strip()
        if not k:
            continue
        if v is None or isinstance(v, (bool, int)) or (isinstance(v, float) and math.isfinite(v)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = None
        else:
            out[k] = str(v)[:500]
    return out


def _value(s: Any) -> Any:
    """Table text as a number where it is one (so that sorting is numeric), else stripped text; '' / NaN → None."""
    if s is None:
        return None
    if isinstance(s, (int, float, bool)):
        return None if isinstance(s, float) and not math.isfinite(s) else s
    t = str(s).strip()
    if not t or t.lower() in ('nan', 'na', 'null', 'none'):
        return None
    digits = t.lstrip('+-')
    if len(digits) > 1 and digits[0] == '0' and digits[1].isdigit():  # an id such as 001042 stays text
        return t
    try:
        f = float(t)
    except ValueError:
        return t
    if not math.isfinite(f):
        return t
    return int(f) if f.is_integer() and not any(c in t for c in '.eE') else f


def read_table(path: str | os.PathLike) -> tuple[list[str], list[dict[str, Any]]]:
    """Columns and rows of a CSV / TSV / Parquet table."""
    p = Path(path)
    if p.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq

        t = pq.read_table(p)
        cols = [str(c) for c in t.column_names]
        rows = [{str(k): _value(v) for k, v in r.items()} for r in t.slice(0, MAX_ROWS).to_pylist()]
        return cols, rows
    with open(p, encoding='utf-8-sig', errors='replace', newline='') as f:
        head = f.readline()
        f.seek(0)
        delim = max(('\t', ',', ';'), key=head.count) if any(d in head for d in '\t,;') else ','
        r = csv.DictReader(f, delimiter=delim)
        cols = [c.strip() for c in (r.fieldnames or []) if c is not None]
        rows = []
        for i, row in enumerate(r):
            if i >= MAX_ROWS:
                break
            rows.append({(k or '').strip(): _value(v) for k, v in row.items() if k is not None})
    return cols, rows


def slide_keys(sid: str, name: str, path: str | None) -> set[str]:
    """What a metadata table's key column may hold for a slide: its id, name, file name, stem or path."""
    keys = {sid, name, Path(name).stem}
    if path:
        keys |= {path, Path(path).name, Path(path).stem}
        if ':' in path and not Path(path).exists():  # a tile-API image 'service:id'
            keys.add(path.partition(':')[2])
    return {k.strip().lower() for k in keys if k and k.strip()}


class MetaTable:
    """One metadata table: which column names the slides (the one matching most of them) and its rows by key."""

    def __init__(self, path: str | os.PathLike, key: str | None = None) -> None:
        self.path = str(Path(path).expanduser())
        self.columns, self.rows = read_table(self.path)
        if not self.columns:
            raise ValueError(f'{Path(self.path).name}: no columns')
        self.key = key

    def match(self, slides: list[tuple[str, str, str | None]]) -> dict[str, dict[str, Any]]:
        """{slide id: {field: value}} for the slides a row names (the key column is chosen if not given)."""
        keys = {sid: slide_keys(sid, name, path) for sid, name, path in slides}
        cols = [self.key] if self.key in self.columns else self.columns
        best, best_n = cols[0], -1
        for c in cols:
            vals = {str(r.get(c)).strip().lower() for r in self.rows if r.get(c) is not None}
            n = sum(bool(k & vals) for k in keys.values())
            if n > best_n:
                best, best_n = c, n
        self.key = best
        by = {}
        for r in self.rows:
            v = r.get(best)
            if v is not None:
                by.setdefault(str(v).strip().lower(), r)
        out = {}
        for sid, ks in keys.items():
            row = next((by[k] for k in ks if k in by), None)
            if row is not None:
                out[sid] = {k: v for k, v in row.items() if k != best and k}
        return out

    def describe(self, matched: int) -> dict[str, Any]:
        return {'path': self.path, 'name': Path(self.path).name, 'key': self.key, 'rows': len(self.rows),
                'columns': [c for c in self.columns if c != self.key], 'matched': matched}  # fmt: skip
