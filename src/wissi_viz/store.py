"""The app's own small database (SQLite): saved views — points of interest, a region of a slide at a given detail,
with metadata — and figure recipes. Views come in and go out as tables (CSV / TSV / Parquet) so that a lab's own
database or a script can supply them; the HTTP API (``/app/views``) reads and writes the same rows.

A view's rectangle is in level-0 pixels of its slide. Tables may give it as ``x0, y0, x1, y1``; as a centre ``x, y``
with ``width`` (and ``height``) in pixels; or as a centre with ``width_um`` (needs the slide's pixel size). The
slide is named by its id, file name, file stem or path (``slide``, ``image`` or ``file``)."""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import math
import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .slidemeta import read_table

SCHEMA = 1
_DDL = """
create table if not exists views (
  id integer primary key,
  slide text not null,              -- id, file name, stem or path: matched to the open slides
  name text not null default '',
  x0 real not null, y0 real not null, x1 real not null, y1 real not null,   -- level-0 px
  display text not null default '{}',   -- json: channels, contrast, layers shown
  meta text not null default '{}',      -- json: {field: value}
  created text not null, updated text not null
);
create index if not exists views_slide on views(slide);
create table if not exists figures (
  id integer primary key,
  name text not null,
  recipe text not null default '{}',    -- json: views or a query, layout, labels, charts
  created text not null, updated text not null
);
"""
SLIDE_COLS = ('slide', 'image', 'file', 'slide_id', 'image_id', 'filename')
_FIXED = {'id', 'name', 'x0', 'y0', 'x1', 'y1', 'x', 'y', 'width', 'height', 'width_um', 'height_um', 'display',
          'created', 'updated', *SLIDE_COLS}  # fmt: skip


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class Store:
    """One SQLite file (``:memory:`` for a session that remembers nothing); safe to use from several threads."""

    def __init__(self, path: str | Path = ':memory:') -> None:
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(_DDL)
            self._db.execute(f'pragma user_version = {SCHEMA}')

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---------------------------------------------------------------------------------------------------------- views
    @staticmethod
    def _view(r: sqlite3.Row) -> dict:
        return {'id': r['id'], 'slide': r['slide'], 'name': r['name'], 'bounds': [r['x0'], r['y0'], r['x1'], r['y1']],
                'display': json.loads(r['display'] or '{}'), 'meta': json.loads(r['meta'] or '{}'),
                'created': r['created'], 'updated': r['updated']}  # fmt: skip

    def views(self, slides: Iterable[str] | None = None) -> list[dict]:
        """All views, or those of the slides named (as stored)."""
        with self._lock:
            if slides is None:
                rows = self._db.execute('select * from views order by id').fetchall()
            else:
                s = list(slides)
                rows = self._db.execute(
                    f'select * from views where slide in ({",".join("?" * len(s))}) order by id', s
                ).fetchall()
        return [self._view(r) for r in rows]

    def view(self, vid: int) -> dict:
        with self._lock:
            r = self._db.execute('select * from views where id = ?', (int(vid),)).fetchone()
        if r is None:
            raise KeyError(f'no view {vid}')
        return self._view(r)

    def add_view(
        self, slide: str, bounds, name: str = '', meta: dict | None = None, display: dict | None = None
    ) -> dict:
        x0, y0, x1, y1 = (float(v) for v in bounds)
        if not (x1 > x0 and y1 > y0) or not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            raise ValueError('a view needs a rectangle with width and height (x0 < x1, y0 < y1)')
        if not str(slide).strip():
            raise ValueError('a view needs its slide')
        t = _now()
        with self._lock, self._db:
            cur = self._db.execute(
                'insert into views (slide, name, x0, y0, x1, y1, display, meta, created, updated) '
                'values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (str(slide), str(name or ''), x0, y0, x1, y1, json.dumps(display or {}), json.dumps(meta or {}), t, t),
            )
            vid = cur.lastrowid
        return self.view(vid)

    def update_view(self, vid: int, **fields) -> dict:
        """Change a view: ``name``, ``bounds``, ``meta`` (merged; a None value removes a field), ``display``."""
        v = self.view(vid)
        name = fields.get('name', v['name'])
        x0, y0, x1, y1 = (float(b) for b in fields.get('bounds') or v['bounds'])
        if not (x1 > x0 and y1 > y0):
            raise ValueError('a view needs a rectangle with width and height')
        meta = dict(v['meta'])
        for k, val in (fields.get('meta') or {}).items():
            if val is None:
                meta.pop(k, None)
            else:
                meta[k] = val
        display = fields.get('display', v['display'])
        with self._lock, self._db:
            self._db.execute(
                'update views set name=?, x0=?, y0=?, x1=?, y1=?, display=?, meta=?, updated=? where id=?',
                (str(name), x0, y0, x1, y1, json.dumps(display), json.dumps(meta), _now(), int(vid)),
            )
        return self.view(vid)

    def delete_views(self, ids: Iterable[int]) -> int:
        ids = [int(i) for i in ids]
        with self._lock, self._db:
            cur = self._db.execute(f'delete from views where id in ({",".join("?" * len(ids))})', ids) if ids else None
        return cur.rowcount if cur else 0

    def relink(self, old: str, new: str) -> int:
        """Views of slide ``old`` (as stored) now of ``new``: how many."""
        with self._lock, self._db:
            return self._db.execute('update views set slide=?, updated=? where slide=?', (new, _now(), old)).rowcount

    def import_views(
        self,
        path: str | Path,
        mpp_of: Callable[[str], float | None] = lambda s: None,
        slide_of: Callable[[str], str] = lambda s: s,
    ) -> dict:
        """Add the rows of a table (CSV / TSV / Parquet) as views. ``mpp_of(slide)``: µm per pixel of a slide (for
        sizes given in µm); ``slide_of(slide)``: how to keep it (e.g. an open slide's path for a name, so that the
        view can open it later). Returns {'added': n, 'errors': ['row 3: …', …]} (rows with errors are skipped)."""
        cols, rows = read_table(path)
        lower = {c.lower(): c for c in cols}
        scol = next((lower[c] for c in SLIDE_COLS if c in lower), None)
        if scol is None:
            raise ValueError(f'{Path(path).name}: needs a column naming the slide ({", ".join(SLIDE_COLS[:3])})')
        get = lambda r, k: r.get(lower[k]) if k in lower else None
        added, errors = [], []
        for i, r in enumerate(rows, start=2):  # the header is row 1
            slide = r.get(scol)
            try:
                if slide is None or not str(slide).strip():
                    raise ValueError('no slide')
                b = [_num(get(r, k)) for k in ('x0', 'y0', 'x1', 'y1')]
                if None in b:
                    cx, cy = _num(get(r, 'x')), _num(get(r, 'y'))
                    if cx is None or cy is None:
                        raise ValueError('needs x0, y0, x1, y1 or a centre x, y with a width')
                    w, h = _num(get(r, 'width')), _num(get(r, 'height'))
                    if w is None and _num(get(r, 'width_um')) is not None:
                        mpp = mpp_of(str(slide))
                        if not mpp:
                            raise ValueError('width_um needs the slide’s pixel size: open the slide first')
                        w = _num(get(r, 'width_um')) / mpp
                        hu = _num(get(r, 'height_um'))
                        h = hu / mpp if hu is not None else None
                    if not w:
                        raise ValueError('needs a width (width, or width_um)')
                    h = h or w
                    b = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                meta = {c: v for c, v in r.items() if c and c.lower() not in _FIXED and v is not None}
                display = r.get(lower['display']) if 'display' in lower else None
                display = json.loads(display) if isinstance(display, str) and display.strip() else {}
                key = slide_of(str(slide)) or str(slide)
                added.append(self.add_view(key, b, name=str(get(r, 'name') or ''), meta=meta, display=display))
            except (ValueError, TypeError) as e:
                errors.append(f'row {i}: {e}')
        return {'added': len(added), 'ids': [v['id'] for v in added], 'errors': errors[:50]}

    def export_views(self, fmt: str = 'csv', ids: Iterable[int] | None = None) -> bytes:
        """The views as a table: slide, name, x0, y0, x1, y1, then each metadata field (CSV or Parquet)."""
        vs = self.views() if ids is None else [self.view(i) for i in ids]
        fields: dict[str, None] = {}
        for v in vs:
            fields.update(dict.fromkeys(v['meta']))
        header = ['id', 'slide', 'name', 'x0', 'y0', 'x1', 'y1', *fields, 'display']
        rows = [[v['id'], v['slide'], v['name'], *v['bounds'], *(v['meta'].get(f) for f in fields),
                 json.dumps(v['display']) if v['display'] else None] for v in vs]  # fmt: skip
        if fmt == 'parquet':
            import pyarrow as pa
            import pyarrow.parquet as pq

            cols = list(zip(*rows)) if rows else [[] for _ in header]
            arrays = []
            for c in cols:
                vals = list(c)
                kinds = {type(x) for x in vals if x is not None}
                arrays.append(pa.array(vals if len(kinds) <= 1 else [None if x is None else str(x) for x in vals]))
            buf = io.BytesIO()
            pq.write_table(pa.table(arrays, names=header), buf)
            return buf.getvalue()
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(header)
        w.writerows([['' if x is None else x for x in r] for r in rows])
        return out.getvalue().encode('utf-8')

    # -------------------------------------------------------------------------------------------------------- figures
    @staticmethod
    def _figure(r: sqlite3.Row) -> dict:
        return {'id': r['id'], 'name': r['name'], 'recipe': json.loads(r['recipe'] or '{}'), 'created': r['created'],
                'updated': r['updated']}  # fmt: skip

    def figures(self) -> list[dict]:
        with self._lock:
            return [self._figure(r) for r in self._db.execute('select * from figures order by updated desc')]

    def save_figure(self, name: str, recipe: dict, fid: int | None = None) -> dict:
        """A figure recipe (views or a query, layout, labels, charts): new, or replacing figure ``fid``."""
        t = _now()
        with self._lock, self._db:
            if fid is None:
                fid = self._db.execute(
                    'insert into figures (name, recipe, created, updated) values (?, ?, ?, ?)',
                    (str(name), json.dumps(recipe), t, t),
                ).lastrowid
            elif not self._db.execute(
                'update figures set name=?, recipe=?, updated=? where id=?',
                (str(name), json.dumps(recipe), t, int(fid)),
            ).rowcount:
                raise KeyError(f'no figure {fid}')
            r = self._db.execute('select * from figures where id = ?', (int(fid),)).fetchone()
        return self._figure(r)

    def delete_figure(self, fid: int) -> None:
        with self._lock, self._db:
            self._db.execute('delete from figures where id = ?', (int(fid),))
