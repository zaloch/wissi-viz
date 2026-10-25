"""Object (cell / nucleus / detection) tables — the input to every spatial analysis.

:class:`ObjectTable` is columnar: centroids in level-0 pixels, an optional class /
phenotype per object, optional outlines or bounding boxes, and any number of
measurement columns (marker intensities, areas, OD, scores …).

Importers (``read_objects`` picks one from the file):

* **QuPath** measurement exports (``.tsv``/``.csv``: ``Centroid X µm``, ``Class``, ``Cell: CD3 mean`` …)
  and QuPath detection GeoJSON (outlines + ``measurements``)
* **HALO** object data (``XMin/XMax/YMin/YMax``, ``Classification`` or ``<Marker> Positive Classification``)
* **CellProfiler** (``Location_Center_X``/``AreaShape_Center_X``), **StarDist/Cellpose** label masks
* generic CSV/TSV/Parquet/Feather with x/y columns, **AnnData** ``.h5ad`` (``obsm['spatial']``)

Microns are converted to pixels with the slide's mpp (detected from column names like
``µm``/``um``/``micron`` or set with ``units=``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .annotations import ELLIPSE, POINT, POLYGON, RECT, AnnotationLayer, _columns_of, _find_col, _norm_col, _ring

_UM = re.compile(r'(µm|μm|\bum\b|micron)', re.I)


@dataclass
class ObjectTable:
    """Columnar objects. ``x``/``y`` are level-0 pixel centroids."""
    x: np.ndarray
    y: np.ndarray
    classes: np.ndarray | None = None          # str per object (None/'' = unclassified)
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    bbox: np.ndarray | None = None             # (n, 4) x0 y0 x1 y1, level-0 px
    outlines: list[np.ndarray] | None = None   # per-object (k, 2) polygons, level-0 px
    mpp: float | None = None                   # µm per level-0 pixel
    name: str = 'objects'

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, np.float64)
        self.y = np.asarray(self.y, np.float64)
        if self.classes is not None:
            self.classes = np.asarray([('' if c is None or (isinstance(c, float) and np.isnan(c)) else str(c))
                                       for c in self.classes], dtype=object)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.x)

    @property
    def xy(self) -> np.ndarray:
        return np.stack([self.x, self.y], 1)

    @property
    def xy_um(self) -> np.ndarray:
        return self.xy * self._mpp()

    def _mpp(self) -> float:
        if not self.mpp:
            raise ValueError('this analysis needs microns: set ObjectTable.mpp (µm per pixel)')
        return float(self.mpp)

    def class_names(self) -> list[str]:
        if self.classes is None:
            return []
        u, c = np.unique(self.classes, return_counts=True)
        return [str(k) for k in u[np.argsort(-c)]]

    def mask(self, cls: str | Sequence[str] | None) -> np.ndarray:
        """Boolean mask for one class, several classes, or a boolean/index selector."""
        if cls is None:
            return np.ones(len(self), bool)
        if isinstance(cls, np.ndarray) and cls.dtype == bool:
            return cls
        if self.classes is None:
            raise ValueError('table has no classes')
        if isinstance(cls, str):
            if cls.endswith('*'):  # prefix match, e.g. 'CD8*'
                return np.array([c.startswith(cls[:-1]) for c in self.classes])
            return self.classes == cls
        return np.isin(self.classes, list(cls))

    def subset(self, sel: Any) -> 'ObjectTable':
        m = self.mask(sel) if not (isinstance(sel, np.ndarray) and sel.dtype != bool) else sel
        idx = np.nonzero(m)[0] if np.asarray(m).dtype == bool else np.asarray(m)
        return ObjectTable(self.x[idx], self.y[idx], None if self.classes is None else self.classes[idx],
                           {k: v[idx] for k, v in self.columns.items()},
                           None if self.bbox is None else self.bbox[idx],
                           None if self.outlines is None else [self.outlines[i] for i in idx], self.mpp, self.name)

    def column(self, name: str) -> np.ndarray:
        if name in self.columns:
            return self.columns[name]
        k = _find_col(self.columns, [name])
        if k is None:
            raise KeyError(f'no column {name!r}; have {list(self.columns)[:30]}')
        return self.columns[k]

    def find_columns(self, pattern: str) -> list[str]:
        r = re.compile(pattern, re.I)
        return [c for c in self.columns if r.search(c)]

    # ------------------------------------------------------------------ phenotyping
    def gate(self, column: str, threshold: float) -> np.ndarray:
        """Boolean positivity: ``column > threshold``."""
        return np.asarray(self.column(column), np.float64) > threshold

    def phenotype(self, markers: dict[str, Any], *, negative: str = '', rules: dict[str, Any] | None = None,
                  inplace: bool = True) -> np.ndarray:
        """Combine marker positivities into phenotype labels.

        ``markers``: ``{'CD3': ('Cell: CD3 mean', 400), 'CD8': ('Cell: CD8 mean', 300), 'PanCK': bool_array}``.
        Default labels concatenate the positive markers (``'CD3+CD8+'``; nothing positive → ``negative``).
        ``rules``: ordered ``{'CD8 T cell': {'CD3': True, 'CD8': True}, 'Tumor': {'PanCK': True}, …}`` —
        first match wins, unmatched objects fall back to ``negative`` or 'Other'.
        """
        pos = {}
        for name, spec in markers.items():
            if isinstance(spec, tuple):
                pos[name] = self.gate(*spec)
            else:
                pos[name] = np.asarray(spec, bool)
            self.columns[f'{name}+'] = pos[name].astype(np.uint8)
        n = len(self)
        if rules:
            lab = np.full(n, negative or 'Other', dtype=object)
            done = np.zeros(n, bool)
            for label, cond in rules.items():
                m = ~done
                for mk, want in cond.items():
                    m &= pos[mk] if want else ~pos[mk]
                lab[m] = label
                done |= m
        else:
            lab = np.full(n, '', dtype=object)
            for name in markers:
                lab = np.where(pos[name], lab + name + '+', lab)
            lab = np.where(lab == '', negative or 'Negative', lab)
        if inplace:
            self.classes = lab
        return lab

    def intensity_bins(self, column: str, thresholds: Sequence[float], labels: Sequence[str] = ('0', '1+', '2+', '3+'),
                       inplace: bool = False) -> np.ndarray:
        """IHC scoring bins (e.g. DAB OD → 0 / 1+ / 2+ / 3+)."""
        v = np.asarray(self.column(column), np.float64)
        lab = np.asarray(labels, dtype=object)[np.digitize(v, thresholds)]
        if inplace:
            self.classes = lab
        return lab

    # ------------------------------------------------------------------ to the viewer
    def to_layer(self, name: str | None = None, shape: str = 'auto', color_by: str | np.ndarray | None = None,
                 cmap: str = 'viridis', classes: dict[str, str] | None = None, max_columns: int = 24,
                 **style) -> AnnotationLayer:
        """Viewer layer: outlines if present, else bbox ellipses, else points. Measurement columns
        are attached (click a cell to see them); ``color_by`` a column or array for a heat-coloured layer."""
        name = name or self.name
        cls = list(self.classes) if self.classes is not None else None
        if shape == 'auto':
            shape = 'outline' if self.outlines is not None else ('ellipse' if self.bbox is not None else 'point')
        # classes given here fix both colour and legend order (e.g. 0 / 1+ / 2+ / 3+)
        L = AnnotationLayer(name=name, classes=dict(classes or {}), **{k: v for k, v in style.items() if k != 'radius'})
        n = len(self)
        labels = cls if cls is not None else [None] * n
        if shape == 'outline' and self.outlines is not None:
            L._bulk(POLYGON, [_ring(p, closed=True) for p in self.outlines], labels, None, None)
        elif shape in ('ellipse', 'rect') and self.bbox is not None:
            L._bulk_shapes(ELLIPSE if shape == 'ellipse' else RECT, self.bbox, cls)   # columnar, GPU-drawn shapes
        else:
            xy = self.xy.astype(np.float32)
            r = np.broadcast_to(np.asarray(style.get('radius', 0.0), np.float32), (n,))
            L._bulk_shapes(POINT, np.stack([xy[:, 0] - r, xy[:, 1] - r, xy[:, 0] + r, xy[:, 1] + r], 1), cls)
        # measurement columns: float32 (half of float64) — clicked cells read their values from here
        L._columns = {k: _compact(v) for k, v in list(self.columns.items())[:max_columns]}
        if color_by is not None:
            L.color_by(self.column(color_by) if isinstance(color_by, str) else color_by, cmap=cmap,
                       label=color_by if isinstance(color_by, str) else None)
        return L

    def to_frame(self):
        import pandas as pd
        d = {'x_px': self.x, 'y_px': self.y}
        if self.mpp:
            d['x_um'], d['y_um'] = self.x * self.mpp, self.y * self.mpp
        if self.classes is not None:
            d['class'] = self.classes
        d.update(self.columns)
        return pd.DataFrame(d)


# ============================================================================= readers
def from_table(table: Any, *, mpp: float | None = None, units: str = 'auto', name: str = 'objects',
               x: str | None = None, y: str | None = None, class_column: str | None = None,
               offset: tuple[float, float] = (0.0, 0.0)) -> ObjectTable:
    """Any tabular object export (DataFrame, dict of columns, list of rows)."""
    cols = _columns_of(table)
    if not cols:
        return ObjectTable(np.zeros(0), np.zeros(0), name=name, mpp=mpp)
    pick = lambda given, *c: given if given is not None else _find_col(cols, c)  # noqa: E731
    bx0, by0 = pick(None, 'xmin', 'x min', 'minx', 'bbox_x0', 'Bounding Box X Min'), pick(None, 'ymin', 'y min', 'miny', 'bbox_y0')
    bx1, by1 = pick(None, 'xmax', 'x max', 'maxx', 'bbox_x1'), pick(None, 'ymax', 'y max', 'maxy', 'bbox_y1')
    cx = pick(x, 'Centroid X µm', 'Centroid X um', 'Centroid X px', 'Centroid X', 'centroid_x', 'x_centroid',
              'Location_Center_X', 'AreaShape_Center_X', 'Nuclei_Location_Center_X', 'Cell X Position', 'X Position',
              'x_um', 'x_px', 'x', 'cx', 'center_x', 'X')
    cy = pick(y, 'Centroid Y µm', 'Centroid Y um', 'Centroid Y px', 'Centroid Y', 'centroid_y', 'y_centroid',
              'Location_Center_Y', 'AreaShape_Center_Y', 'Nuclei_Location_Center_Y', 'Cell Y Position', 'Y Position',
              'y_um', 'y_px', 'y', 'cy', 'center_y', 'Y')
    has_box = None not in (bx0, by0, bx1, by1)
    if cx is None and not has_box:
        raise ValueError(f'no centroid or bounding-box columns among {list(cols)[:25]}')
    geo_cols = {c for c in (bx0, by0, bx1, by1, cx, cy) if c}
    um = units == 'um' or (units == 'auto' and any(_UM.search(c) for c in geo_cols if c))
    s = 1.0
    if um:
        if not mpp:
            raise ValueError('coordinates are in µm: pass mpp= (µm per pixel) to convert to pixels')
        s = 1.0 / mpp
    ox, oy = offset
    bbox = None
    if has_box:
        bbox = np.stack([np.asarray(cols[k], np.float64) for k in (bx0, by0, bx1, by1)], 1) * s + [ox, oy, ox, oy]
    if cx is not None:
        X, Y = np.asarray(cols[cx], np.float64) * s + ox, np.asarray(cols[cy], np.float64) * s + oy
    else:
        X, Y = (bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2
    ccol = pick(class_column, 'Classification', 'Class', 'Phenotype', 'cell_type', 'celltype', 'cell type', 'label',
                'Name', 'type', 'cluster', 'leiden', 'louvain')
    classes = None
    if ccol is not None:
        classes = np.asarray(cols[ccol], dtype=object)
    meas = {}
    skip = geo_cols | ({ccol} if ccol else set()) | {'Image', 'Object ID', 'Parent', 'ROI', 'Object type', 'Analysis Region'}
    for k, v in cols.items():
        if k in skip:
            continue
        v = np.asarray(v)
        if v.dtype.kind in 'biuf':
            meas[k] = v
        elif v.dtype.kind in 'OSU':
            try:
                meas[k] = v.astype(np.float64)
            except (TypeError, ValueError):
                if len(set(v[:1000].tolist())) <= 64:  # low-cardinality text → keep as category
                    meas[k] = v.astype(object)
    t = ObjectTable(X, Y, classes, meas, bbox, None, mpp, name)
    # HALO multiplex: '<Marker> Positive Classification' 0/1 columns → phenotype if no class column
    if classes is None:
        pos = {re.sub(r'\s*positive\s*classification\s*$', '', k, flags=re.I).strip(): v
               for k, v in meas.items() if re.search(r'positive\s*classification\s*$', k, re.I)}
        if pos:
            t.phenotype({m: np.asarray(v, np.float64) > 0 for m, v in pos.items()})
    return t


def from_geojson(data: Any, *, mpp: float | None = None, name: str = 'objects', outlines: bool = True) -> ObjectTable:
    """QuPath detection/cell GeoJSON (``properties.classification``, ``properties.measurements``)."""
    if isinstance(data, (str, Path)) and Path(str(data)).exists():
        data = json.loads(Path(data).read_bytes())
    elif isinstance(data, (str, bytes)):
        data = json.loads(data)
    feats = data.get('features', []) if isinstance(data, dict) else data
    xs, ys, cl, polys = [], [], [], []
    meas: dict[str, list] = {}
    for i, f in enumerate(feats):
        g = (f.get('properties', {}) or {}).get('nucleusGeometry') if False else f.get('geometry')
        if not g:
            continue
        t, c = g.get('type'), g.get('coordinates')
        if t == 'Point':
            ring = np.array([c], np.float64)
        elif t == 'Polygon':
            ring = np.asarray(c[0], np.float64)
        elif t == 'MultiPolygon':
            ring = np.asarray(max(c, key=lambda p: len(p[0]))[0], np.float64)
        else:
            continue
        p = f.get('properties') or {}
        xs.append(ring[:, 0].mean())
        ys.append(ring[:, 1].mean())
        polys.append(ring[:-1] if len(ring) > 3 and np.array_equal(ring[0], ring[-1]) else ring)
        c_ = p.get('classification')
        cl.append(c_.get('name') if isinstance(c_, dict) else (c_ if isinstance(c_, str) else p.get('class', '')))
        m = p.get('measurements') or {}
        if isinstance(m, list):  # older QuPath: [{'name':…, 'value':…}]
            m = {d.get('name'): d.get('value') for d in m}
        for k, v in m.items():
            meas.setdefault(k, [np.nan] * (len(xs) - 1)).append(v)
        for k in meas:
            if len(meas[k]) < len(xs):
                meas[k].append(np.nan)
    cols = {k: np.asarray(v, np.float64) for k, v in meas.items()}
    return ObjectTable(np.asarray(xs), np.asarray(ys), np.asarray(cl, dtype=object), cols, None,
                       polys if outlines and any(len(q) > 2 for q in polys) else None, mpp, name)


def from_label_mask(mask: Any, *, mpp: float | None = None, downsample: float = 1.0, offset=(0.0, 0.0),
                    intensities: dict[str, np.ndarray] | None = None, outlines: bool = True,
                    name: str = 'objects') -> ObjectTable:
    """StarDist / Cellpose / any integer label image → objects (centroid, area, bbox, mean intensities,
    and polygon outlines via scikit-image if installed). ``downsample``: mask pixel size in level-0 px."""
    from scipy import ndimage
    if isinstance(mask, (str, Path)):
        import tifffile
        mask = tifffile.imread(mask) if str(mask).lower().endswith(('.tif', '.tiff')) else np.asarray(
            __import__('PIL.Image', fromlist=['Image']).open(mask))
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    labels = np.unique(m)
    labels = labels[labels != 0]
    if not len(labels):
        return ObjectTable(np.zeros(0), np.zeros(0), name=name, mpp=mpp)
    com = np.asarray(ndimage.center_of_mass(np.ones_like(m), m, labels))
    area = np.asarray(ndimage.sum(np.ones_like(m, dtype=np.float64), m, labels))
    sl = ndimage.find_objects(m)
    boxes = np.array([[s[1].start, s[0].start, s[1].stop, s[0].stop] if s else [0, 0, 0, 0]
                      for s in (sl[int(k) - 1] for k in labels)], np.float64)
    ox, oy = offset
    ds = downsample
    cols = {'label': labels, 'area_px': area * ds * ds}
    if mpp:
        cols['area_um2'] = area * (ds * mpp) ** 2
    for k, img in (intensities or {}).items():
        cols[f'{k} mean'] = np.asarray(ndimage.mean(np.asarray(img, np.float64), m, labels))
    polys = None
    if outlines:
        try:
            from skimage import measure
            polys = []
            for k, b in zip(labels, boxes.astype(int)):
                sub = np.pad(m[b[1]:b[3], b[0]:b[2]] == k, 1)
                cs = measure.find_contours(sub.astype(np.uint8), 0.5)
                c = max(cs, key=len) if cs else np.array([[1.0, 1.0]])
                c = c[::max(1, len(c) // 24)]
                polys.append(np.stack([(c[:, 1] - 1 + b[0]) * ds + ox, (c[:, 0] - 1 + b[1]) * ds + oy], 1))
        except ImportError:
            polys = None
    return ObjectTable(com[:, 1] * ds + ox, com[:, 0] * ds + oy, None, cols, boxes * ds + [ox, oy, ox, oy], polys, mpp, name)


def from_anndata(adata: Any, *, mpp: float | None = None, spatial_key: str = 'spatial', class_column: str | None = None,
                 units: str = 'px', layer_genes: Sequence[str] = (), name: str = 'objects') -> ObjectTable:
    """AnnData / ``.h5ad`` (squidpy, scimap, …): coordinates from ``obsm[spatial_key]``, classes and
    measurements from ``obs``; selected ``var_names`` (markers) copied as columns."""
    if isinstance(adata, (str, Path)):
        import anndata
        adata = anndata.read_h5ad(adata)
    xy = np.asarray(adata.obsm[spatial_key], np.float64)[:, :2]
    if units == 'um':
        xy = xy / float(mpp)
    obs = adata.obs
    ccol = class_column or _find_col({c: None for c in obs.columns}, ['phenotype', 'cell_type', 'celltype', 'leiden',
                                                                     'louvain', 'cluster', 'class'])
    cols = {str(c): np.asarray(obs[c]) for c in obs.columns if c != ccol and obs[c].dtype.kind in 'biuf'}
    X = adata.X
    for g in layer_genes:
        j = list(adata.var_names).index(g)
        v = X[:, j]
        cols[g] = np.asarray(v.toarray() if hasattr(v, 'toarray') else v).ravel()
    return ObjectTable(xy[:, 0], xy[:, 1], None if ccol is None else np.asarray(obs[ccol]).astype(str), cols, None,
                       None, mpp, name)


def cached_object_layer(path: str | Path, cache_dir: str | Path, *, mpp: float | None = None, read_kw: dict | None = None,
                        **layer_kw) -> AnnotationLayer:
    """``read_objects(path).to_layer(**layer_kw)``, compiled once and kept in ``cache_dir``.

    The cache key covers the file (absolute path, size, modification time) and every option, so editing the export
    rebuilds it. A cached layer opens in milliseconds (memory-mapped .npy files: no parsing, no compiling, shared
    by every server worker through the OS page cache); a 1M-cell table otherwise costs ~3 s of CPU per process."""
    import hashlib
    p = Path(path).resolve()
    st = p.stat()
    key_src = json.dumps([str(p), st.st_size, st.st_mtime_ns, mpp, read_kw or {}, layer_kw, AnnotationLayer._SAVE_FORMAT],
                         sort_keys=True, default=str)
    key = hashlib.sha1(key_src.encode()).hexdigest()[:20]
    d = Path(cache_dir) / f'{p.stem}-{key}'
    version = int(key[:6], 16) * 1024   # stable per content: browser caches stay valid across restarts
    if (d / 'meta.json').exists():
        try:
            return AnnotationLayer.open(d, version=version)
        except Exception:  # noqa: BLE001  (corrupt / old cache: rebuild)
            pass
    layer = read_objects(p, mpp=mpp, **(read_kw or {})).to_layer(**layer_kw)
    try:
        layer.save(d)
        return AnnotationLayer.open(d, version=version)   # serve the memory-mapped copy (frees the build memory)
    except (OSError, ValueError):
        return layer


def read_objects(path: str | Path, *, mpp: float | None = None, fmt: str = 'auto', **kw) -> ObjectTable:
    """Read an object export by extension / content (see module docstring)."""
    p = Path(path)
    ext = ''.join(p.suffixes[-2:]).lower() if p.suffix.lower() in ('.gz',) else p.suffix.lower()
    name = kw.pop('name', p.stem)
    if fmt == 'geojson' or ext in ('.geojson', '.json'):
        return from_geojson(p, mpp=mpp, name=name, **kw)
    if fmt == 'anndata' or ext == '.h5ad':
        return from_anndata(p, mpp=mpp, name=name, **kw)
    if fmt == 'mask' or ext in ('.tif', '.tiff', '.png') and fmt == 'auto':
        return from_label_mask(p, mpp=mpp, name=name, **kw)
    if ext in ('.parquet', '.feather', '.arrow'):
        import pyarrow.parquet as pq
        import pyarrow.feather as pf
        tbl = pq.read_table(p) if ext == '.parquet' else pf.read_table(p)
        return from_table({c: tbl.column(c).to_numpy(zero_copy_only=False) for c in tbl.column_names},
                          mpp=mpp, name=name, **kw)
    return from_table(_read_delimited(p), mpp=mpp, name=name, **kw)


def _read_delimited(p: Path) -> dict[str, np.ndarray]:
    """CSV / TSV / semicolon exports. pyarrow's multi-threaded parser when available (≈3× faster than pandas on
    10^6 rows), else pandas, else the csv module. The delimiter is detected from the header line."""
    with open(p, 'rb') as f:
        head = f.readline().decode('utf-8-sig', 'replace')
    delim = max(('\t', ',', ';'), key=head.count) if any(d in head for d in '\t,;') else ','
    try:
        import pyarrow.csv as pacsv
        t = pacsv.read_csv(p, parse_options=pacsv.ParseOptions(delimiter=delim),
                           read_options=pacsv.ReadOptions(block_size=1 << 24))
        return {str(c): t[c].to_numpy(zero_copy_only=False) for c in t.column_names}
    except ImportError:
        pass
    try:
        import pandas as pd
        df = pd.read_csv(p, sep=delim, low_memory=False)
        return {str(c): df[c].to_numpy() for c in df.columns}
    except ImportError:
        import csv
        with open(p, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f, delimiter=delim))
        return {k: np.asarray([r[k] for r in rows]) for k in (rows[0] if rows else {})}


class _Categorical:
    """A text column as integer codes + categories (10^6 Python strings would cost ~60 MB per column)."""

    def __init__(self, values) -> None:
        arr = np.asarray(values, dtype=object)
        cats, codes = np.unique(np.where(arr == None, '', arr).astype(str), return_inverse=True)  # noqa: E711
        self.categories = cats
        self.codes = codes.astype(np.uint16 if len(cats) < 65536 else np.uint32)
        self.dtype = np.dtype(object)

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return str(self.categories[self.codes[i]])
        return self.categories[self.codes[i]]

    @property
    def nbytes(self) -> int:
        return int(self.codes.nbytes + self.categories.nbytes)


def _compact(v: np.ndarray):
    """Columns kept on a viewer layer: float32 for floats, categories for text."""
    if v.dtype.kind == 'f':
        return v.astype(np.float32)
    if v.dtype.kind in 'OUS':
        return _Categorical(v)
    return v


_ = _norm_col
