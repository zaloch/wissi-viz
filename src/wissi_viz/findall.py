"""Find all: point at a few examples, or name what to find, and get every such object or area — in the view, a region or
the whole slide.

Four ways, each adding a layer to the slide (``origin='found'``, so the app lists it under *Found*):

* **similar objects** — every object of a layer (detected cells, an imported cell table) is measured (shape, stain or
  channel intensity) and scored against the examples; those that look alike are kept.
* **segment everything** — SAM (or, without a model, the classic colour fill) outlines every object in the region from
  a grid of clicks, skipping what is already outlined; with examples, only those like them are kept.
* **by text** — name it ("nucleus", "gland. vessel"): Grounding DINO finds the boxes and SAM 2.1 outlines them, or
  SAM 3 (opt-in) does both; with examples, only those like them are kept.
* **areas** — a pixel classifier (random forest on multi-scale colour, edge and texture features) is trained on a few
  shapes or strokes per class (tumour, stroma, necrosis…) and maps those areas.

Examples are shapes on the slide — drawn or made with SAM — taken from a layer (optionally only some classes), plus
objects picked on the slide (``picks``: ``[{layer, index, neg}]``; ``neg``: "not like this").

    out = find_objects(registry, 'case1', examples={'layer': 'Annotations', 'classes': ['Tumor']},
                       candidates='cells', scope={'kind': 'slide'})

Similar-object and segment-everything results can be re-thresholded without measuring again
(:func:`rethreshold`): the app's strictness slider.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure
from skimage.color import rgb2lab
from skimage.feature import multiscale_basic_features, peak_local_max

from . import annio, grounding, layerops, sam3, segment
from .annotations import AnnotationLayer
from .jobs import Job
from .metrics import object_metrics
from .sam import _polygons, _rgb
from .sam import service as sam_service
from .spatial import _geom_layer, tissue_mask

ORIGIN = 'found'
#: measurements used to compare objects (whichever the slide has)
SHAPE = ('log_area', 'circularity', 'solidity', 'eccentricity')
BRIGHTFIELD = ('hematoxylin_mean', 'eosin_mean', 'hematoxylin_sd', 'eosin_sd')
DETAIL_UM = {'coarse': 8.0, 'medium': 4.0, 'fine': 2.0}  # areas: µm per pixel of the classification
_FEATURES: OrderedDict = OrderedDict()  # (slide, layer, version, level) → measured candidates
_RESULTS: OrderedDict = OrderedDict()  # result id → what rethreshold needs
_TISSUE: dict = {}  # slide → tissue geometry


# ================================================================================================ inputs
def _source(registry, sid: str):
    return registry.get(sid)[1].source


def _mpp(src) -> float:
    return float((src.info.mpp or (1.0,))[0] or 1.0)


def scope_geometry(registry, sid: str, scope: dict | None, tissue: bool = False):
    """``{'kind': 'view', 'rect': [x0, y0, x1, y1]}`` | ``{'kind': 'region', 'layer': name, 'index': [i…]}`` |
    ``{'kind': 'slide'}`` → (shapely geometry in level-0 px or None for the whole slide, a label). ``tissue``: the
    whole slide becomes its tissue (segmenting or classifying glass is wasted time). ``exclude``: ``{'layer': name,
    'classes': [...]}`` (or a list of them): the regions of that layer are left out of the scope."""
    g, where = _scope_geometry(registry, sid, scope, tissue)
    excl = (scope or {}).get('exclude') or []
    for e in [excl] if isinstance(excl, dict) else excl:
        if not e or not e.get('layer'):
            continue
        try:
            missing = registry.layer(sid, e['layer']) is None
        except KeyError:
            missing = True
        if missing:
            raise ValueError(f'no layer {e["layer"]!r} on this slide to leave out')
        eg = layerops.geometry(registry, sid, e['layer'], e.get('classes') or None)
        if eg is None:
            raise ValueError(f'{e["layer"]} has no regions (polygons) to leave out')
        src = _source(registry, sid)
        base = g if g is not None else shapely.box(0, 0, src.info.width, src.info.height)
        g = shapely.make_valid(base.difference(eg))
        if g.is_empty or g.area < 4:
            raise ValueError(f'nothing of {where} is outside {e["layer"]}')
        where = f'{where}, outside {e["layer"]}'
    return g, where


def _scope_geometry(registry, sid: str, scope: dict | None, tissue: bool = False):
    scope = scope or {'kind': 'slide'}
    src = _source(registry, sid)
    W, H = src.info.width, src.info.height
    kind = scope.get('kind', 'slide')
    if kind == 'geometry':  # internal: a shapely geometry already in level-0 px
        return scope['geom'], scope.get('label', 'the given area')
    if kind == 'view':
        x0, y0, x1, y1 = [float(v) for v in scope['rect']]
        g = shapely.box(max(0, x0), max(0, y0), min(W, x1), min(H, y1))
        if g.is_empty or g.area < 4:
            raise ValueError('the view is outside the slide')
        return g, 'this view'
    if kind == 'region':
        L = registry.layer(sid, scope['layer'])
        F = annio.from_layer(L)
        idx = scope.get('index')
        sel = range(len(F)) if idx is None else [int(i) for i in idx]
        polys = []
        for i in sel:
            if F.kind[i] != annio.GEO_POLYGON:
                continue
            rs = F.rings(i)
            p = shapely.Polygon(rs[0], rs[1:]).buffer(0)
            if not p.is_empty:
                polys.append(p)
        if not polys:
            raise ValueError(f'{scope["layer"]} has no regions (polygons) to search in')
        return shapely.union_all(polys), scope['layer'] if idx is None else f'{scope["layer"]} ({len(polys)})'
    if tissue:
        if sid not in _TISSUE:
            g, _ = tissue_mask(src)
            _TISSUE[sid] = g if g is not None and not g.is_empty else None
        if _TISSUE[sid] is not None:
            return _TISSUE[sid], 'the whole slide (tissue)'
    return None, 'the whole slide'


class Examples:
    """What to look for: outlines (drawn / SAM) and picked objects of existing layers, each positive or negative."""

    def __init__(self, registry, sid: str, ex: dict | None) -> None:
        ex = ex or {}
        self.registry, self.sid = registry, sid
        pos, neg, names, colors = [], [], [], {}
        if ex.get('layer'):
            L = registry.layer(sid, ex['layer'])
            F = annio.from_layer(L)
            want = set(ex.get('classes') or [])
            cls = F.class_of()
            for i in range(len(F)):
                if want and cls[i] not in want:
                    continue
                ring = F.rings(i)[0]
                if F.kind[i] == annio.GEO_POLYGON and len(ring) >= 3:
                    (neg if ex.get('neg') else pos).append(ring)
                    names.append(cls[i])
            colors.update(L.classes)
        for ring in ex.get('polygons') or []:  # outlines given directly (level-0 px), e.g. confident text matches
            ring = np.asarray(ring, float)
            if len(ring) >= 3:
                pos.append(ring)
                names.append(str(ex.get('name') or 'object'))
        self.picks: list[tuple[str, int, bool]] = []
        for p in ex.get('picks') or []:
            L = registry.layer(sid, p['layer'])
            i = int(p['index'])
            if not 0 <= i < len(L):
                continue
            self.picks.append((p['layer'], i, bool(p.get('neg'))))
            if not p.get('neg'):
                names.append(str(L.class_of(i)))
                colors.update(L.classes)
        self.pos, self.neg = _polys_features(pos), _polys_features(neg)
        self.cls = str(ex.get('name') or (max(set(names), key=names.count) if names else 'object'))
        self.color = ex.get('color') or colors.get(self.cls) or '#ffb454'

    @property
    def n_pos(self) -> int:
        return (len(self.pos) if self.pos is not None else 0) + sum(1 for *_, ng in self.picks if not ng)

    def __bool__(self) -> bool:
        return self.n_pos > 0

    def bboxes(self) -> np.ndarray:
        """Level-0 bounding boxes of the positive examples."""
        out = [self.pos.bbox()] if self.pos is not None else []
        for name, i, ng in self.picks:
            if ng:
                continue
            F = annio.from_layer(self.registry.layer(self.sid, name))
            bb = F.bbox()[i : i + 1].astype(float)
            if (bb[0, 2] - bb[0, 0]) * (bb[0, 3] - bb[0, 1]) < 4:  # a point: a box of its area (or ~10 µm)
                mpp = _mpp(_source(self.registry, self.sid))
                area = _num(F.props.get('area_um2'), i) or _num(F.props.get('area'), i) or 78.5
                r = float(np.sqrt(area / np.pi)) / mpp
                cx, cy = bb[0, 0], bb[0, 1]
                bb = np.array([[cx - r, cy - r, cx + r, cy + r]])
            out.append(bb)
        return np.concatenate(out) if out else np.zeros((0, 4))

    def diameter(self) -> float:
        b = self.bboxes()
        return float(np.median(np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1, None)))) if len(b) else 0.0

    def outline_points(self, svc, src) -> None:
        """Picked objects without an outline (a cell table's points): outline them with the segmenter at their
        centre, so they compare on colour and shape too, like the outlines they are matched against."""
        keep, add_pos, add_neg = [], [], []
        for name, i, ng in self.picks:
            L = self.registry.layer(self.sid, name)
            F = annio.from_layer(L)
            if F.kind[i] == annio.GEO_POLYGON:
                keep.append((name, i, ng))
                continue
            x, y = F.centroid()[i]
            r = 4 * max(8.0, self.diameter())
            try:
                polys, _ = svc.segment(src, [x - r, y - r, x + r, y + r], [[x, y]], [1])
            except ValueError:
                polys = []
            if polys:
                (add_neg if ng else add_pos).append(polys[0])
            else:
                keep.append((name, i, ng))
        self.picks = keep
        if add_pos:
            self.pos = _polys_features(
                ([self.pos.rings(k)[0] for k in range(len(self.pos))] if self.pos is not None else []) + add_pos
            )
        if add_neg:
            self.neg = _polys_features(
                ([self.neg.rings(k)[0] for k in range(len(self.neg))] if self.neg is not None else []) + add_neg
            )

    def measures(self, src, level: int) -> tuple[dict, dict | None]:
        """(positives, negatives) as measurement columns: outlines measured on the slide, picked objects with the
        measurements of their own layer (so a picked cell compares exactly like its neighbours)."""

        def gather(shapes, neg):
            parts = [_measure(shapes, src, level)] if shapes is not None else []
            for name, i, ng in self.picks:
                if ng == neg:
                    _, M = _candidate_features(self.registry, self.sid, self.registry.layer(self.sid, name), src, level)
                    parts.append({k: v[i : i + 1] for k, v in M.items()})
            if not parts:
                return None
            keys = set(parts[0]).intersection(*parts[1:])
            return {k: np.concatenate([p[k] for p in parts]).astype(float) for k in keys}

        return gather(self.pos, False), gather(self.neg, True)


def _num(col, i: int) -> float | None:
    try:
        v = float(np.asarray(col)[i])
    except (TypeError, ValueError, IndexError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _polys_features(polys: list, cls: str = 'object', props: dict | None = None) -> annio.Features | None:
    """Single-ring polygons (level-0 px) → Features."""
    if not polys:
        return None
    rings = [np.asarray(p, np.float64)[:, :2] for p in polys]
    nv = np.array([len(r) for r in rings], np.int64)
    F = annio.Features(
        np.full(len(rings), annio.GEO_POLYGON, np.uint8),
        np.concatenate(rings),
        np.concatenate([[0], np.cumsum(nv)]),
        np.arange(len(rings) + 1, dtype=np.int64),
        np.zeros(len(rings), np.int32),
        [cls],
    )
    F.props = dict(props or {})
    return F


_subset = annio.subset  # (moved: layer operations use it too)


# ================================================================================================ measuring
def _level_for(src, diameter_px: float, min_px: float = 8.0) -> int:
    """The coarsest pyramid level where an object of this diameter (level-0 px) is still ``min_px`` across."""
    lv = 0
    for i, L in enumerate(src.info.levels):
        if diameter_px / L.downsample >= min_px:
            lv = i
    return lv


def _measure(F: annio.Features, src, level: int) -> dict[str, np.ndarray]:
    """Comparable measurements per object: log area, circularity, solidity, eccentricity, and stain / channel
    intensity (mean, sd) at ``level``."""
    if not (F.kind == annio.GEO_POLYGON).any():  # points (a cell table without outlines): only their area compares
        area = next((np.asarray(F.props[k], float) for k in ('area_um2', 'area') if k in F.props), None)
        return {} if area is None else {'log_area': np.log(np.clip(area, 1e-3, None))}
    G = copy.copy(F)
    G.props = {}  # measure every outline the same way (a table's own columns may be defined differently)
    df = object_metrics(G, source=src, intensity=True, level=level)
    out = {'log_area': np.log(np.clip(df['area_um2'].to_numpy(float), 1e-3, None))}
    for k in SHAPE[1:]:
        out[k] = df[k].to_numpy(float)
    if src.info.mode == 'rgb':
        for k in BRIGHTFIELD:
            if k in df:
                out[k] = df[k].to_numpy(float)
    else:
        for k in df.columns:
            if k.endswith('_mean') and k != 'grey_mean':
                out[k] = df[k].to_numpy(float)
    return out


def _candidate_features(registry, sid: str, layer: AnnotationLayer, src, level: int):
    key = (sid, layer.name, layer.version, level)
    hit = _FEATURES.get(key)
    if hit is None:
        F = annio.from_layer(layer)
        hit = (F, _measure(F, src, level))
        _FEATURES[key] = hit
        while len(_FEATURES) > 6:
            _FEATURES.popitem(last=False)
    else:
        _FEATURES.move_to_end(key)
    return hit


def _diameter(F: annio.Features) -> float:
    b = F.bbox()
    return float(np.median(np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1, None))))


# ================================================================================================ scoring
def similarity(
    cand: dict[str, np.ndarray], pos: dict[str, np.ndarray], neg: dict[str, np.ndarray] | None = None
) -> tuple[np.ndarray, list[str]]:
    """How alike each candidate is to the nearest example (1 = identical, → 0 = unlike), on robustly standardised
    measurements; a candidate closer to a negative example than to every positive one scores 0."""
    keys = [k for k in pos if k in cand and np.isfinite(pos[k]).all() and np.isfinite(cand[k]).mean() > 0.5]
    if not keys:
        raise ValueError('no measurements to compare')
    X = np.stack([np.nan_to_num(cand[k], nan=np.nanmedian(cand[k])) for k in keys], 1)
    E = np.stack([pos[k] for k in keys], 1)
    allx = np.vstack([X, E])
    med = np.median(allx, 0)
    q1, q3 = np.percentile(allx, [25, 75], 0)
    scale = (q3 - q1) / 1.349
    scale = np.where(scale > 1e-9, scale, np.where(allx.std(0) > 1e-9, allx.std(0), 1.0))
    Z, ZE = (X - med) / scale, (E - med) / scale
    d = len(keys)

    def nearest(A, B):
        out = np.full(len(A), np.inf)
        for i in range(0, len(B)):
            out = np.minimum(out, np.sqrt(((A - B[i]) ** 2).sum(1) / d))
        return out

    dpos = nearest(Z, ZE)
    if len(ZE) >= 2:  # how far apart the examples are: a looser class tolerates more
        spread = np.array([np.sort(np.sqrt(((ZE - e) ** 2).sum(1) / d))[1] for e in ZE])
        r = float(np.clip(np.median(spread), 0.3, 0.8))
    else:
        r = 0.5
    sim = np.exp(-0.5 * (dpos / r) ** 2)
    if neg:
        N = np.stack([np.nan_to_num(neg[k], nan=med[j]) for j, k in enumerate(keys)], 1)
        dneg = nearest(Z, (N - med) / scale)
        sim = np.where(dneg < dpos, 0.0, sim)
    return sim, keys


def threshold(strictness: float) -> float:
    """Strictness 0 (loose) … 1 (strict) → the similarity an object needs."""
    return 0.12 + 0.76 * float(np.clip(strictness, 0, 1))


def _size_range(mpp: float, object_um, object_um_max) -> tuple[float, float, bool]:
    """(smallest, largest object diameter in level-0 px, whether a range was given): ``object_um`` alone is one size."""
    lo = float(object_um or 10.0)
    hi = float(object_um_max) if object_um_max not in (None, '', 0) else None
    if hi is not None and hi < lo:
        lo, hi = hi, lo
    return lo / mpp, (hi if hi is not None else lo) / mpp, hi is not None


def _in_size(polys: list, lo: float, hi: float) -> np.ndarray:
    """Which outlines' equivalent diameter (level-0 px) is within the range, with some slack (0.8× – 1.25×)."""
    if not polys:
        return np.zeros(0, bool)
    a = np.abs(shapely.area(shapely.polygons([shapely.linearrings(np.asarray(p, float)) for p in polys])))
    d = np.sqrt(4 * a / math.pi)
    return (d >= 0.8 * lo) & (d <= 1.25 * hi)


# ================================================================================================ results
def _publish(
    registry,
    sid: str,
    F: annio.Features,
    sim: np.ndarray | None,
    strictness: float,
    name: str,
    cls: str,
    color: str,
    extra: dict | None = None,
    classes: list[tuple[str, str]] | None = None,
    labels: np.ndarray | None = None,
) -> tuple[str, int]:
    """Add the objects scoring ``sim ≥ threshold(strictness)`` as the layer ``name`` (one class ``cls``, or one per
    entry of ``classes`` with ``labels`` the index of each object's)."""
    keep = np.arange(len(F)) if sim is None else np.nonzero(sim >= threshold(strictness))[0]
    G = _subset(F, keep)
    if classes and labels is not None:
        G.cls = np.asarray(labels, np.int32)[keep]
        G.class_names = [c for c, _ in classes]
        G.class_colors = dict(classes)
    else:
        classes = [(cls, color)]
        G.cls = np.zeros(len(G), np.int32)
        G.class_names = [cls]
        G.class_colors = {cls: color}
    G.props = {}
    if sim is not None:
        G.props['similarity'] = np.round(sim[keep], 3)
    for k, v in (extra or {}).items():
        G.props[k] = np.asarray(v)[keep]
    L = annio.to_layer(G, name=name)
    L.classes = dict(classes)
    L.origin = ORIGIN
    L.fill_opacity = 0.25
    L.line_width = 2.0
    L.detail_min_px = 4.0
    if len(G) and not (G.kind == annio.GEO_POLYGON).any():  # points (a table without outlines): bold markers
        L.point_size = 7.0
    registry.add_annotations(sid, L)
    return L.name, int(len(keep))


def _unique(registry, sid: str, base: str) -> str:
    """A new layer name: runs never overwrite each other (``Found · Tumor``, ``Found · Tumor 2``, …)."""
    have = set(registry.layers(sid))
    name, k = base, 1
    while name in have:
        k += 1
        name = f'{base} {k}'
    return name


def _remember(rid: str, **kw) -> None:
    _RESULTS[rid] = kw
    while len(_RESULTS) > 16:
        _RESULTS.popitem(last=False)


def rethreshold(registry, rid: str, strictness: float) -> dict:
    """Apply another strictness to a finished similar-object / segment-everything result (no re-measuring)."""
    r = _RESULTS.get(rid)
    if r is None:
        raise ValueError('that result is gone (the server restarted or it is too old): run it again')
    name, n = _publish(registry, r['slide'], r['F'], r['sim'], strictness, r['name'], r['cls'], r['color'], r['extra'],
                       r.get('classes'), r.get('labels'))  # fmt: skip
    r['strictness'] = strictness
    return {'layer': name, 'count': n, 'strictness': strictness, 'of': int(len(r['F']))}


def _summary(rid, name, n, total, strictness, where, t0, mode, note='', kpis=None) -> dict:
    return {
        'result': rid,
        'mode': mode,
        'layer': name,
        'count': n,
        'of': total,
        'strictness': strictness,
        'where': where,
        'seconds': round(time.perf_counter() - t0, 1),
        'note': note,
        'kpis': kpis or [],
    }


# ================================================================================================ similar objects
def find_objects(
    registry,
    sid: str,
    *,
    examples: dict,
    candidates: str,
    scope: dict | None = None,
    strictness: float = 0.5,
    name: str | None = None,
    job: Job | None = None,
) -> dict:
    """Score every object of the layer ``candidates`` against the examples; keep the similar ones."""
    t0 = time.perf_counter()
    upd = job.update if job else (lambda *a, **k: None)
    src = _source(registry, sid)
    ex = Examples(registry, sid, examples)
    cls, color = ex.cls, ex.color
    if not ex:
        raise ValueError('no examples: draw or SAM a few objects (or pick some on the slide) first')
    layer = registry.layer(sid, candidates)
    upd(0.05, f'measuring the objects of {candidates}…')
    F_all = annio.from_layer(layer)
    lv = _level_for(src, _diameter(F_all) if len(F_all) else ex.diameter())
    F, M = _candidate_features(registry, sid, layer, src, lv)
    geom, where = scope_geometry(registry, sid, scope)
    idx = np.arange(len(F))
    if geom is not None:
        c = F.centroid()
        shapely.prepare(geom)
        idx = np.nonzero(shapely.contains_xy(geom, c[:, 0], c[:, 1]))[0]
    if not len(idx):
        raise ValueError(f'no objects of {candidates} in {where}')
    upd(0.6, f'comparing {len(idx):,} objects with {ex.n_pos} example{"s" if ex.n_pos > 1 else ""}…')
    P, N = ex.measures(src, lv)
    sim, keys = similarity({k: v[idx] for k, v in M.items()}, P, N)
    upd(0.9, 'adding the layer…')
    Fs = _subset(F, idx)
    name = _unique(registry, sid, name or f'Found · {cls}')
    rid = uuid.uuid4().hex[:10]
    extra = {'source_index': idx}
    lname, n = _publish(registry, sid, Fs, sim, strictness, name, cls, color, extra)
    _remember(rid, slide=sid, F=Fs, sim=sim, name=lname, cls=cls, color=color, extra=extra, strictness=strictness)
    return _summary(
        rid, lname, n, len(idx), strictness, where, t0, 'objects',
        note=f'compared on {", ".join(keys)}',
        kpis=[['found', f'{n:,}'], ['searched', f'{len(idx):,}'], ['examples', str(ex.n_pos)]],
    )  # fmt: skip


# ================================================================================================ segment everything
def segment_everything(
    registry,
    sid: str,
    *,
    examples: dict | None = None,
    scope: dict | None = None,
    strictness: float = 0.5,
    object_um: float = 10.0,
    name: str | None = None,
    job: Job | None = None,
    max_tiles: int = 4000,
    object_um_max: float | None = None,
) -> dict:
    """Outline every object in the scope from a grid of single-click prompts (SAM, or the classic colour fill), then
    keep those like the examples (all of them without examples). ``object_um`` – ``object_um_max``: the objects' size
    range in µm (clicks are as dense as the smallest needs; outlines outside the range are dropped)."""
    t0 = time.perf_counter()
    upd = job.update if job else (lambda *a, **k: None)
    src = _source(registry, sid)
    info = src.info
    ex = Examples(registry, sid, examples) if examples else None
    ex = ex if ex else None  # no positive examples: keep every outline
    cls, color = (ex.cls, ex.color) if ex else ('object', '#ffb454')
    d_lo, d_hi, ranged = _size_range(_mpp(src), object_um, object_um_max)
    ranged = ranged and not ex
    d0 = ex.diameter() if ex else d_lo  # object diameter, level-0 px (the smallest: clicks dense enough for it)
    geom, where = scope_geometry(registry, sid, scope, tissue=True)
    svc = sam_service()
    impl = svc._load()
    # the model sees each object ~40 px across (SAM outlines objects of a few px badly: on a 0.5 µm/px H&E, nuclei
    # are ~16 px, so tiles are enlarged up to 4×); the classic fill works at the slide's own pixels or coarser
    f = max(1.0, d0 / 28.0) if impl.name == 'classic' else max(0.25, d0 / 40.0)  # level-0 px per model px
    if ranged:  # the largest still well inside a tile
        f = max(f, d_hi / (0.45 * (1024 if impl.name != 'classic' else 768)))
    lv = 0
    for i, Lv in enumerate(info.levels):
        if Lv.downsample <= f * 1.0001:
            lv = i
    ds = info.levels[lv].downsample
    VW, VH = info.width / f, info.height / f  # the slide in model px
    d_img = d0 / f
    T = 1024 if impl.name != 'classic' else 768
    O = int(min(T // 4, max(16, 2.5 * d_img, 1.1 * d_hi / f if ranged else 0)))  # noqa: E741
    step = T - O
    if geom is not None:
        gx0, gy0, gx1, gy1 = geom.bounds
    else:
        gx0, gy0, gx1, gy1 = 0, 0, info.width, info.height
    tiles = []
    for ty in range(int(gy0 / f // step), int(math.ceil(gy1 / f / step)) + 1):
        for tx in range(int(gx0 / f // step), int(math.ceil(gx1 / f / step)) + 1):
            x, y = tx * step, ty * step
            w, h = int(min(T, VW - x)), int(min(T, VH - y))
            if w < 8 or h < 8:
                continue
            box = shapely.box(x * f, y * f, (x + w) * f, (y + h) * f)
            if geom is None or geom.intersects(box):
                tiles.append((x, y, w, h))
    if not tiles:
        raise ValueError(f'nothing to segment in {where}')
    if len(tiles) > max_tiles:
        raise ValueError(f'{len(tiles):,} tiles at this object size: pick a region or a larger object size')
    if geom is not None:
        shapely.prepare(geom)
    if ex:
        b = ex.bboxes()
        ea = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * (math.pi / 4)  # the ellipse in each box
        amin, amax = float(ea.min()) / 4 / f**2, float(ea.max()) * 4 / f**2
    else:
        amin, amax = math.pi / 4 * (0.3 * d_img) ** 2, math.pi / 4 * (3.0 * d_img) ** 2  # 0.3–3× the diameter
        if ranged:  # the range, with some slack
            amin, amax = math.pi / 4 * (0.8 * d_lo / f) ** 2, math.pi / 4 * (1.25 * d_hi / f) ** 2
        elif impl.name == 'classic':  # its fills can join touching objects: keep those too
            amax *= 4
    amin = max(amin, 6.0)
    polys: list[np.ndarray] = []
    name = _unique(registry, sid, name or (f'Segmented · {cls}' if ex else 'Segmented objects'))
    prov = f'{name} (segmenting…)'
    last_pub = time.perf_counter()
    upd(0.0, f'{impl.name}: {len(tiles)} tile{"s" if len(tiles) > 1 else ""} in {where}')
    for ti, (x, y, w, h) in enumerate(tiles):
        upd(ti / len(tiles), f'{impl.name}: tile {ti + 1} of {len(tiles)} · {len(polys):,} objects')
        rgb = _read_scaled(src, lv, ds, x * f, y * f, w * f, h * f, w, h)
        polys += _tile_objects(svc, impl, rgb, x, y, f, d_img, amin, amax, O, (x == 0, y == 0,
                               x + w >= VW - 1, y + h >= VH - 1), geom, job)  # fmt: skip
        if time.perf_counter() - last_pub > 4 and polys:  # show progress on the slide
            _publish(registry, sid, _polys_features(polys, 'candidate'), None, 0, prov, 'candidate', '#9aa3ad')
            last_pub = time.perf_counter()
    if prov in registry.layers(sid):
        registry.remove_annotations(sid, prov)
    if not polys:
        raise ValueError(f'no objects found in {where}')
    F = _polys_features(polys, cls)
    rid = uuid.uuid4().hex[:10]
    sim = None
    note = f'{impl.name} at {f:.2g} level-0 px per model px' + (f' (level {lv})' if lv else '')
    if ranged:
        note += f'; {d_lo * _mpp(src):.3g}–{d_hi * _mpp(src):.3g} µm across'
    if ex:
        upd(0.95, f'comparing {len(polys):,} outlines with the examples…')
        lvm = _level_for(src, d0)
        ex.outline_points(svc, src)
        P, N = ex.measures(src, lvm)
        sim, keys = similarity(_measure(F, src, lvm), P, N)
        note += f'; compared on {", ".join(keys)}'
    lname, n = _publish(registry, sid, F, sim, strictness, name, cls, color)
    if sim is not None:
        _remember(rid, slide=sid, F=F, sim=sim, name=lname, cls=cls, color=color, extra={}, strictness=strictness)
    return _summary(
        rid if sim is not None else None, lname, n, len(polys), strictness, where, t0, 'segment', note=note,
        kpis=[['found', f'{n:,}'], ['outlined', f'{len(polys):,}'], ['tiles', str(len(tiles))]],
    )  # fmt: skip


def _read_scaled(src, lv: int, ds: float, x0: float, y0: float, w0: float, h0: float, w: int, h: int) -> np.ndarray:
    """The level-0 rectangle (x0, y0, w0, h0), read at level ``lv`` (downsample ``ds``), as ``w`` × ``h`` RGB."""
    L = src.info.levels[lv]
    lx, ly = int(x0 / ds), int(y0 / ds)
    lw = max(1, min(int(round(w0 / ds)), L.width - lx))
    lh = max(1, min(int(round(h0 / ds)), L.height - ly))
    rgb = _rgb(src, lv, lx, ly, lw, lh)
    if (lw, lh) != (w, h):
        rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
    return rgb


def _tile_objects(svc, impl, rgb, x, y, ds, d_img, amin, amax, O, edges, geom, job) -> list[np.ndarray]:  # noqa: E741
    """One tile: grid of clicks → non-overlapping masks → outlines (level-0 px) whose centre is in the tile's core."""
    h, w = rgb.shape[:2]
    sp = max(3.0, 0.8 * d_img)
    while (w / sp) * (h / sp) > 1600:
        sp *= 1.25
    gy, gx = np.mgrid[sp / 2 : h : sp, sp / 2 : w : sp]
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    # clicks where a person would click first: the centres of what stands out from the background (colour distance
    # to the tile's median, smoothed at the object scale), most distinct first; then the grid fills the gaps
    lab = rgb2lab(rgb)
    sal = ndi.gaussian_filter(np.sqrt(((lab - np.median(lab.reshape(-1, 3), 0)) ** 2).sum(-1)), max(1.0, d_img / 4))
    pk = peak_local_max(sal, min_distance=max(2, int(d_img / 2)), threshold_abs=float(np.percentile(sal, 60)),
                        exclude_border=False, num_peaks=4000)  # fmt: skip
    pk = pk[np.argsort(-sal[pk[:, 0], pk[:, 1]])]
    if impl.name != 'classic':  # a model click costs ~35 ms on a CPU: the most distinct seeds, then a sparse grid
        grid = grid[:: max(1, len(grid) // 64)]
        pk = pk[:192]
    pts = np.concatenate([pk[:, ::-1].astype(float), grid]) if len(pk) else grid
    covered = np.zeros((h, w), bool)
    out = []
    classic = impl.name == 'classic'
    R = int(max(8, 3 * d_img))
    with svc._lock:
        emb = impl.embed(rgb)
        for px, py in pts:
            if job is not None and job.cancelled:  # every click: a SAM click can take a while on a slow CPU
                job.update()
            xi, yi = int(px), int(py)
            if covered[yi, xi]:
                continue
            if classic:  # the colour fill only needs the neighbourhood
                cx0, cy0 = max(0, xi - R), max(0, yi - R)
                cx1, cy1 = min(w, xi + R + 1), min(h, yi + R + 1)
                sub = {'rgb': emb['rgb'][cy0:cy1, cx0:cx1], 'lab': emb['lab'][cy0:cy1, cx0:cx1]}
                m, score = impl.predict(sub, [(xi - cx0, yi - cy0)], [1], None)
            else:
                m, score = impl.predict(emb, [(px, py)], [1], None, area=(amin, amax))
                cx0, cy0 = 0, 0
                ys, xs = np.nonzero(m)
                if not len(ys):
                    continue
                cy0, cx0 = int(ys.min()), int(xs.min())
                m = m[cy0 : ys.max() + 1, cx0 : xs.max() + 1]
            if score < 0.6:
                continue
            a = int(m.sum())
            if a < amin or a > amax:
                continue
            if not classic:  # cut by the tile's edge (not the slide's): whole in the next tile, which overlaps
                mh, mw = m.shape
                if (cx0 <= 0 and not edges[0]) or (cy0 <= 0 and not edges[1]):
                    continue
                if (cx0 + mw >= w and not edges[2]) or (cy0 + mh >= h and not edges[3]):
                    continue
            cov = covered[cy0 : cy0 + m.shape[0], cx0 : cx0 + m.shape[1]]
            if (m & cov).sum() > 0.25 * a:
                continue
            m = m & ~cov
            cov |= m
            ps = _polygons(m, amin)
            if not ps:
                continue
            p = ps[0]
            ccx, ccy = p[:, 0].mean() + cx0, p[:, 1].mean() + cy0
            lo, hi = O / 2, O / 2
            if (not edges[0] and ccx < lo) or (not edges[1] and ccy < lo):
                continue
            if (not edges[2] and ccx >= w - hi) or (not edges[3] and ccy >= h - hi):
                continue
            q = np.stack([(p[:, 0] + cx0 + x) * ds, (p[:, 1] + cy0 + y) * ds], 1)
            if geom is not None and not shapely.contains_xy(geom, (ccx + x) * ds, (ccy + y) * ds):
                continue
            out.append(q)
    return out


# ================================================================================================ by text
PHRASE_COLORS = ('#ffb454', '#59c2ff', '#aad94c', '#f07178', '#d2a6ff', '#95e6cb', '#e6b673', '#ff8f40')
TEXT_FLOOR = 0.1  # the lowest detection score kept (strictness 0)


def text_threshold(strictness: float) -> float:
    """Strictness 0 … 1 → the detection score a match needs (0.35 at 0.5: the models' usual cut-off)."""
    return TEXT_FLOOR + 0.5 * float(np.clip(strictness, 0, 1))


def _text_sim(score: np.ndarray) -> np.ndarray:
    """Detection scores on the similarity scale, so that ``sim ≥ threshold(s)`` ⇔ ``score ≥ text_threshold(s)``."""
    return 0.12 + 0.76 * (np.asarray(score, float) - TEXT_FLOOR) / 0.5


def text_detector(model: str = 'grounding', progress=None):
    """The model that finds things by name: ``grounding`` (Grounding DINO + SAM 2.1, Apache-2.0) or ``sam3``
    (SAM 3, opt-in under Meta's SAM License)."""
    if model == 'sam3':
        return sam3.detector(progress)
    return grounding.detector(progress)


def _box_nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.5) -> np.ndarray:
    """Indices kept by non-maximum suppression (xyxy boxes), best first; uses a spatial index (many boxes)."""
    order = np.argsort(-scores, kind='stable')
    tree = shapely.STRtree(shapely.box(*boxes.T))
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    dead = np.zeros(len(boxes), bool)
    keep = []
    for i in order:
        if dead[i]:
            continue
        keep.append(i)
        j = tree.query(shapely.box(*boxes[i]))
        j = j[(j != i) & ~dead[j]]
        if not len(j):
            continue
        b = boxes[j]
        w = np.clip(np.minimum(b[:, 2], boxes[i, 2]) - np.maximum(b[:, 0], boxes[i, 0]), 0, None)
        h = np.clip(np.minimum(b[:, 3], boxes[i, 3]) - np.maximum(b[:, 1], boxes[i, 1]), 0, None)
        inter = w * h
        dead[j[inter / np.maximum(area[i] + area[j] - inter, 1e-9) > iou]] = True
    return np.asarray(keep, np.int64)


class _Sub:
    """A slice of a job's progress bar (one stage of a two-stage run); cancelling still cancels the job."""

    def __init__(self, job: Job, lo: float, hi: float) -> None:
        self.job, self.lo, self.hi = job, lo, hi

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        p = None if progress is None else self.lo + (self.hi - self.lo) * float(np.clip(progress, 0, 1))
        self.job.update(p, message)

    @property
    def cancelled(self) -> bool:
        return self.job.cancelled


def _excluded(boxes: np.ndarray, scores: np.ndarray, ex_boxes: np.ndarray, ex_scores: np.ndarray) -> np.ndarray:
    """Which boxes are mostly (≥ 50 %) covered by a box of an excluded phrase scoring at least 0.8× theirs."""
    out = np.zeros(len(boxes), bool)
    if not len(boxes) or not len(ex_boxes):
        return out
    B = shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    E = shapely.box(ex_boxes[:, 0], ex_boxes[:, 1], ex_boxes[:, 2], ex_boxes[:, 3])
    i, j = shapely.STRtree(E).query(B, predicate='intersects')
    if not len(i):
        return out
    cover = shapely.area(shapely.intersection(B[i], E[j])) / np.maximum(shapely.area(B[i]), 1e-9)
    bad = (cover >= 0.5) & (ex_scores[j] >= 0.8 * scores[i])
    out[i[bad]] = True
    return out


def _expand(registry, sid, polys, boxes, keep, score, tsim, d0, scope, strictness, name, cls, color, label, job,
            size=None, ex_boxes=None):  # fmt: skip
    """Find all like the best text matches: segment everything in the scope (no examples), take the outlines at the
    best matches as examples (outlined the same way as everything they are compared with), keep those like them.

    Examples: outlines containing the centre of a match about the object's size (0.4–2.5× its diameter); when the
    text model only boxed groups of objects (dense nuclei), the outlines of about that size inside its ≤ 3 best boxes.
    None if there is nothing to take examples from."""
    src = _source(registry, sid)
    tmp = _unique(registry, sid, '_expand')
    mpp = _mpp(src)
    lo, hi = size if size else (d0, None)
    seg = segment_everything(registry, sid, scope=scope, object_um=lo * mpp, object_um_max=hi * mpp if hi else None,
                             name=tmp, job=_Sub(job, 0.5, 0.9) if job else None)  # fmt: skip
    F = annio.from_layer(registry.layer(sid, seg['layer']))
    registry.remove_annotations(sid, seg['layer'])
    if not len(F):
        return None
    shapes = shapely.polygons([shapely.linearrings(F.rings(i)[0]) for i in range(len(F))])
    dia = np.sqrt(4 * np.abs(shapely.area(shapes)) / math.pi)
    tree = shapely.STRtree(shapes)
    tdia = np.sqrt(4 * np.abs(shapely.area(shapely.polygons([shapely.linearrings(p) for p in polys]))) / math.pi)
    good = np.nonzero((score >= 0.12) & (tsim >= threshold(0.0)) & (tdia >= 0.4 * d0) & (tdia <= 2.5 * d0))[0]
    good = good[np.argsort(-score[good], kind='stable')][:16]
    seeds: set[int] = set()
    for i in good:  # the outline under each good match's centre
        c = np.asarray(polys[i]).mean(0)
        hit = tree.query(shapely.Point(*c))
        hit = [j for j in hit if shapely.contains_xy(shapes[j], *c)]
        seeds.update(hit[:1])
    how = 'matches'
    if len(seeds) < 5:  # few (or only boxes around groups): add the object-sized outlines inside the best boxes
        b = np.asarray(boxes, float)[keep]
        best = [i for i in np.argsort(-score, kind='stable') if score[i] >= 0.12][:3]
        cand = set()
        for i in best:
            cand.update(int(j) for j in tree.query(shapely.box(*b[i]), predicate='contains'))
        cand = sorted(j for j in cand if 0.5 * d0 <= dia[j] <= 2.0 * d0)
        more = set(sorted(cand, key=lambda j: abs(math.log(dia[j] / d0)))[:16])
        how = (
            'matches and outlines inside the best ones'
            if seeds and more - seeds
            else ('matches' if seeds else 'outlines inside the best matches')
        )
        seeds |= more
    if not seeds:
        return None
    neg: set[int] = set()
    for bx in (ex_boxes if ex_boxes is not None else [])[:32]:  # the outlines at excluded matches: "not like these"
        c = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
        neg.update(j for j in tree.query(shapely.Point(*c)) if shapely.contains_xy(shapes[j], *c))
    neg -= seeds
    lvm = _level_for(src, d0)
    M = _measure(F, src, lvm)
    idx = np.array(sorted(seeds))
    nidx = np.array(sorted(neg), int)
    sim, keys = similarity(M, {k: v[idx] for k, v in M.items()},
                           {k: v[nidx] for k, v in M.items()} if len(nidx) else None)  # fmt: skip
    G = copy.copy(F)
    lname, n = _publish(registry, sid, G, sim, strictness, name, cls, color)
    rid = uuid.uuid4().hex[:10]
    _remember(rid, slide=sid, F=G, sim=sim, name=lname, cls=cls, color=color, extra={}, strictness=strictness)
    return _summary(
        rid, lname, n, len(F), strictness, seg['where'], time.perf_counter(), 'text',
        note=f'{len(idx)} {label} {how} as examples{f", {len(nidx)} excluded ones as counter-examples" if len(nidx) else ""} among {len(F):,} outlines ({seg["note"]}); compared on {", ".join(keys)}',
        kpis=[['found', f'{n:,}'], ['outlined', f'{len(F):,}'], ['examples', str(len(idx))]],
    )  # fmt: skip


def find_by_text(
    registry,
    sid: str,
    *,
    prompt: str,
    model: str = 'grounding',
    examples: dict | None = None,
    scope: dict | None = None,
    strictness: float = 0.5,
    object_um: float | None = None,
    name: str | None = None,
    job: Job | None = None,
    max_tiles: int = 2000,
    detector=None,
    expand: bool = False,
    object_um_max: float | None = None,
    exclude: str = '',
) -> dict:
    """Outline everything in the scope that matches the text ``prompt`` (one class per phrase: "nucleus. gland").

    ``grounding``: Grounding DINO finds boxes, SAM 2.1 outlines each; ``sam3``: SAM 3 finds and outlines in one go.
    The slide is read in tiles at the scale where an object of ``object_um`` (or the examples' size) is ~48 px
    across. With examples, only matches like them are kept too.

    ``expand`` (one phrase): the confident matches of about the object's size become the examples of
    :func:`segment_everything`, which outlines every object in the scope and keeps those like them. For dense small
    objects (nuclei) the text model alone marks only a few; this finds the rest.

    ``object_um`` – ``object_um_max``: a size range (µm): the scale fits the smallest, matches outside it are dropped.
    ``exclude``: phrases not to find ("red blood cell. lymphocyte"): matches mostly covered by a confident match of
    one of them are dropped (and, with ``expand``, outlines like those are not kept)."""
    t0 = time.perf_counter()
    names = grounding.phrases(prompt)
    ex_names = [n for n in grounding.phrases(exclude or '') if n not in names]
    expand = bool(expand) and len(names) == 1
    upd = (_Sub(job, 0.0, 0.5) if expand else job).update if job else (lambda *a, **k: None)
    if not names:
        raise ValueError('type what to find, e.g. "nucleus" or "gland. vessel" (one class per phrase)')
    src = _source(registry, sid)
    info = src.info
    ex = Examples(registry, sid, examples) if examples else None
    ex = ex if ex else None
    mpp = _mpp(src)
    d_lo, d_hi, ranged = _size_range(mpp, object_um, object_um_max)
    ranged = ranged and not ex
    d0 = ex.diameter() if ex else math.sqrt(d_lo * d_hi)  # object diameter, level-0 px (a range: its middle)
    geom, where = scope_geometry(registry, sid, scope, tissue=True)
    if detector is None:
        upd(0.0, 'loading the model…')
        detector = text_detector(model, lambda f, m: upd(0.0, m))
    T = int(getattr(detector, 'tile', 800))
    f = max(0.5, (d_lo if ranged else d0) / 48.0)  # level-0 px per model px (≥ 0.5: at most 2× enlarged)
    if ranged:  # the largest still well inside a tile
        f = max(f, d_hi / (0.45 * T))
    lv = 0
    for i, Lv in enumerate(info.levels):
        if Lv.downsample <= f * 1.0001:
            lv = i
    ds = info.levels[lv].downsample
    S0 = T * f  # a tile's side, level-0 px
    O0 = min(S0 / 4, max(32 * f, 2.5 * d0, 1.1 * d_hi if ranged else 0))  # objects cut by a tile: whole in the next
    step = S0 - O0
    gx0, gy0, gx1, gy1 = geom.bounds if geom is not None else (0, 0, info.width, info.height)
    tiles = []
    for ty in range(int(gy0 // step), int(math.ceil(gy1 / step)) + 1):
        for tx in range(int(gx0 // step), int(math.ceil(gx1 / step)) + 1):
            x0, y0 = tx * step, ty * step
            x1, y1 = min(info.width, x0 + S0), min(info.height, y0 + S0)
            if x1 - x0 < 8 * f or y1 - y0 < 8 * f:
                continue
            if geom is None or geom.intersects(shapely.box(x0, y0, x1, y1)):
                tiles.append((x0, y0, x1, y1))
    if not tiles:
        raise ValueError(f'nothing to search in {where}')
    if len(tiles) > max_tiles:
        raise ValueError(
            f'{len(tiles):,} tiles at this object size: pick a region or the view, or a larger object size'
        )
    if geom is not None:
        shapely.prepare(geom)
    svc = impl = None
    if not getattr(detector, 'masks', False):  # boxes only: SAM 2.1 outlines them
        svc = sam_service()
        impl = svc._load()
    polys, boxes, scores, labels = [], [], [], []
    ex_boxes: list = []
    ex_scores: list = []
    name = _unique(registry, sid, name or f'Text · {", ".join(names)}'[:60])
    prov = f'{name} (searching…)'
    last_pub = time.perf_counter()
    floor = TEXT_FLOOR
    label = getattr(detector, 'label', getattr(detector, 'name', 'model'))
    for ti, (x0, y0, x1, y1) in enumerate(tiles):
        el = time.perf_counter() - t0
        eta = f' · ~{el / ti * (len(tiles) - ti) / 60:.0f} min left' if ti >= 2 and el > 20 else ''
        upd(ti / len(tiles), f'{label}: tile {ti + 1} of {len(tiles)} · {len(polys):,} found{eta}')
        lx, ly = int(x0 / ds), int(y0 / ds)
        lw, lh = max(1, int(round((x1 - x0) / ds))), max(1, int(round((y1 - y0) / ds)))
        lw, lh = min(lw, info.levels[lv].width - lx), min(lh, info.levels[lv].height - ly)
        rgb = _rgb(src, lv, lx, ly, lw, lh)
        tw, th = max(1, int(round((x1 - x0) / f))), max(1, int(round((y1 - y0) / f)))
        if (tw, th) != (rgb.shape[1], rgb.shape[0]):
            rgb = np.asarray(Image.fromarray(rgb).resize((tw, th), Image.BILINEAR))
        out = detector.detect(rgb, names + ex_names, min_score=floor)
        b, sc, lab = out[:3]
        masks = out[3] if len(out) > 3 else None
        if not len(b):
            continue
        big = ((b[:, 2] - b[:, 0]) > 0.8 * T) & ((b[:, 3] - b[:, 1]) > 0.8 * T)  # "the whole picture"
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        core = O0 / 2 / f
        ok = ~big
        ok &= (cx >= core) | (x0 <= 0)
        ok &= (cy >= core) | (y0 <= 0)
        ok &= (cx < tw - core) | (x1 >= info.width)
        ok &= (cy < th - core) | (y1 >= info.height)
        if geom is not None:
            ok &= shapely.contains_xy(geom, x0 + cx * f, y0 + cy * f)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        emb = None
        for k in idx:
            if job is not None and job.cancelled:
                job.update()
            bx = b[k]
            if lab[k] >= len(names):  # an excluded phrase: its box is enough
                ex_boxes.append([x0 + bx[0] * f, y0 + bx[1] * f, x0 + bx[2] * f, y0 + bx[3] * f])
                ex_scores.append(float(sc[k]))
                continue
            if masks is not None:
                m = np.asarray(masks[k], bool)
            else:
                with svc._lock:
                    if emb is None:
                        emb = impl.embed(rgb)
                    m, _ = impl.predict(emb, [], [], [float(v) for v in bx])
                pad = 0.15 * max(bx[2] - bx[0], bx[3] - bx[1]) + 2  # SAM may spill out of the box: not far
                yy0, xx0 = max(0, int(bx[1] - pad)), max(0, int(bx[0] - pad))
                yy1, xx1 = min(th, int(bx[3] + pad) + 1), min(tw, int(bx[2] + pad) + 1)
                clip = np.zeros_like(m)
                clip[yy0:yy1, xx0:xx1] = True
                m = m & clip
            ps = _polygons(m, 4.0)
            if ps:
                p = ps[0]
                q = np.stack([x0 + p[:, 0] * f, y0 + p[:, 1] * f], 1)
            else:  # no mask (tiny object): its box
                q = np.array([[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]) * f + [x0, y0]
            polys.append(q)
            boxes.append([x0 + bx[0] * f, y0 + bx[1] * f, x0 + bx[2] * f, y0 + bx[3] * f])
            scores.append(float(sc[k]))
            labels.append(int(lab[k]))
        if time.perf_counter() - last_pub > 4 and polys:
            _publish(registry, sid, _polys_features(polys, 'candidate'), None, 0, prov, 'candidate', '#9aa3ad')
            last_pub = time.perf_counter()
    if prov in registry.layers(sid):
        registry.remove_annotations(sid, prov)
    if not polys:
        raise ValueError(f'nothing matching "{", ".join(names)}" in {where} (try other words or a larger object size)')
    upd(0.95, f'{len(polys):,} matches: removing duplicates…')
    keep = _box_nms(np.asarray(boxes, float), np.asarray(scores, float))
    ok = np.ones(len(keep), bool)
    n_size = n_excl = 0
    if ranged:
        ok &= _in_size([polys[i] for i in keep], d_lo, d_hi)
        n_size = int((~ok).sum())
    if ex_boxes:
        hit = _excluded(np.asarray(boxes, float)[keep], np.asarray(scores, float)[keep], np.asarray(ex_boxes, float),
                        np.asarray(ex_scores, float))  # fmt: skip
        n_excl = int((hit & ok).sum())
        ok &= ~hit
    keep = np.asarray(keep)[ok]
    if not len(keep):
        why = ' and '.join(w for w, n in ((f'{n_size} outside {d_lo * mpp:.3g}–{d_hi * mpp:.3g} µm', n_size),
                                          (f'{n_excl} like "{", ".join(ex_names)}"', n_excl)) if n)  # fmt: skip
        raise ValueError(f'every match of "{", ".join(names)}" in {where} was left out ({why})')
    polys = [polys[i] for i in keep]
    score = np.asarray(scores, float)[keep]
    lab = np.asarray(labels, np.int64)[keep]
    classes = [(n, PHRASE_COLORS[i % len(PHRASE_COLORS)]) for i, n in enumerate(names)]
    F = _polys_features(polys, names[0])
    sim = _text_sim(score)
    note = f'{label} at {f:.2g} level-0 px per model px'
    if n_size:
        note += f'; {n_size:,} outside {d_lo * mpp:.3g}–{d_hi * mpp:.3g} µm left out'
    if ex_names:
        note += f'; {n_excl:,} like "{", ".join(ex_names)}" left out'
    if ex:
        upd(0.97, f'comparing {len(polys):,} matches with the examples…')
        lvm = _level_for(src, d0)
        if svc is None:
            svc = sam_service()
        ex.outline_points(svc, src)
        P, N = ex.measures(src, lvm)
        sim_ex, keys = similarity(_measure(F, src, lvm), P, N)
        sim = np.minimum(sim, sim_ex)
        note += f'; also like the examples ({", ".join(keys)})'
    if expand:
        out = _expand(registry, sid, polys, boxes, keep, score, sim, d0, scope, strictness, name, names[0],
                      classes[0][1], label, job, size=(d_lo, d_hi) if ranged else None,
                      ex_boxes=np.asarray(ex_boxes, float).reshape(-1, 4))  # fmt: skip
        if out is not None:
            out.update(seconds=round(time.perf_counter() - t0, 1))
            out['kpis'] = [*out['kpis'], ['text matches', f'{len(polys):,}']]
            return out
        note += f'; no match about {d0 * mpp:.0f} µm across to find more like (check the object size)'
    extra = {'score': np.round(score, 3)}
    lname, n = _publish(registry, sid, F, sim, strictness, name, names[0], classes[0][1], extra, classes, lab)
    rid = uuid.uuid4().hex[:10]
    _remember(rid, slide=sid, F=F, sim=sim, name=lname, cls=names[0], color=classes[0][1], extra=extra,
              strictness=strictness, classes=classes, labels=lab)  # fmt: skip
    return _summary(
        rid, lname, n, len(polys), strictness, where, t0, 'text', note=note,
        kpis=[['found', f'{n:,}'], ['matches', f'{len(polys):,}'], ['tiles', str(len(tiles))]],
    )  # fmt: skip


# ================================================================================================ cells
CELL_MODELS = {
    'cellpose': 'Cellpose-SAM',
    'classic': 'classic stain threshold',
    'classpose': 'Classpose',
}


def cell_models() -> dict:
    """The cell models this server can run: {id: {label, available, why}} (the By text model list's third group)."""
    have = set(segment.available())
    why = {
        'cellpose': 'pip install "wissi-viz[cells]" (Cellpose is BSD-3; it brings fill-voids, LGPL-3.0)',
        'classpose': 'install uv (https://docs.astral.sh/uv/): Classpose runs in its own environment',
    }
    return {
        k: {'label': v, 'available': k in have, 'why': '' if k in have else why.get(k, '')}
        for k, v in CELL_MODELS.items()
    }


def find_cells(
    registry,
    sid: str,
    *,
    model: str = 'cellpose',
    scope: dict | None = None,
    name: str | None = None,
    job: Job | None = None,
) -> dict:
    """Outline every cell nucleus in the scope with a cell model: ``cellpose`` (Cellpose-SAM), ``classic`` (stain
    threshold + watershed, no model) or ``classpose`` (whole slide files only: it runs on the slide, then the cells in
    the scope are kept). Adds a ``detected`` layer with classes (by size, or by the brightest marker) and
    measurements (area, perimeter, circularity, stain / channel means), which the analyses use."""
    t0 = time.perf_counter()
    if model not in CELL_MODELS:
        raise ValueError(f'unknown cell model {model!r} ({" | ".join(CELL_MODELS)})')
    caps = cell_models()[model]
    if not caps['available']:
        raise ValueError(f'{CELL_MODELS[model]} is not installed here: {caps["why"]}')
    upd = job.update if job else (lambda *a, **k: None)
    src = _source(registry, sid)
    geom, where = scope_geometry(registry, sid, scope)
    if model == 'classpose':
        e = registry.get(sid)[0]
        path = e.path
        if e.factory is not None or path is None or not Path(path).is_file():
            raise ValueError('Classpose reads slide files, and this slide comes from a tile API: use Cellpose-SAM '
                             'or the classic threshold, which read the tiles')  # fmt: skip
        upd(0.02, 'Classpose: the whole slide in its own environment (the first run installs it)…')
        L = segment.classpose_layer(path, log=lambda m: upd(None, str(m)[:200]))
        F = annio.from_layer(L)
        if geom is not None and len(F):
            c = F.centroid()
            shapely.prepare(geom)
            F = _subset(F, np.nonzero(shapely.contains_xy(geom, c[:, 0], c[:, 1]))[0])
        F.mpp = _mpp(src)
    else:
        F = segment.segment_source(src, model, geom, progress=lambda f, m: upd(0.02 + 0.9 * f, m))
    if not len(F):
        raise ValueError(f'no cells found in {where} by {CELL_MODELS[model]}')
    upd(0.95, f'{len(F):,} cells: adding the layer…')
    lname = _unique(registry, sid, name or f'Cells · {CELL_MODELS[model]}')
    L = annio.to_layer(F, name=lname)
    L.origin = 'detected'
    L.mpp = F.mpp
    L.fill_opacity = 0.25
    L.line_width = 1.2
    L.detail_min_px = 3.0
    registry.add_annotations(sid, L)
    names = F.class_of()
    kinds, counts = np.unique(names, return_counts=True)
    kpis = [['cells', f'{len(F):,}']]
    kpis += [[str(k), f'{c:,}'] for k, c in sorted(zip(kinds, counts), key=lambda kc: -kc[1])[:4]]
    if 'area_um2' in F.props and len(F):
        kpis.append(['median area', f'{float(np.median(F.props["area_um2"])):.0f} µm²'])
    out = _summary(None, lname, len(F), len(F), 0.5, where, t0, 'cells', note=CELL_MODELS[model], kpis=kpis)
    return out


# ================================================================================================ areas
def find_areas(
    registry,
    sid: str,
    *,
    training: str,
    classes: list[str] | None = None,
    scope: dict | None = None,
    detail: str = 'medium',
    name: str | None = None,
    job: Job | None = None,
    max_pixels: float = 60e6,
) -> dict:
    """Train a pixel classifier on the shapes / strokes of ``training`` (two or more classes) and map those areas
    over the scope."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as ex:
        raise ValueError('finding areas needs scikit-learn: pip install "wissi-viz[analysis]"') from ex
    t0 = time.perf_counter()
    upd = job.update if job else (lambda *a, **k: None)
    src = _source(registry, sid)
    info = src.info
    TL = registry.layer(sid, training)
    F = annio.from_layer(TL)
    names = F.class_of()
    want = set(classes or [])
    keep = [i for i in range(len(F)) if (not want or names[i] in want) and F.kind[i] != annio.GEO_POINT]
    labels = sorted({str(names[i]) for i in keep})
    if len(labels) < 2:
        raise ValueError(
            f'{training} has {len(labels)} class{"" if len(labels) == 1 else "es"} of shapes: brush examples of at '
            'least two (e.g. Area 1 and Area 2) — draw each with its class set in the Annotate tab'
        )
    geom, where = scope_geometry(registry, sid, scope, tissue=True)
    if geom is not None and (scope or {}).get('kind', 'slide') == 'slide':  # tissue: keep faint edges (fat, stroma)
        geom = geom.buffer(50 / _mpp(src))
    um = DETAIL_UM.get(detail, 4.0)
    f = max(1.0, um / _mpp(src))  # level-0 px per classification px
    gx0, gy0, gx1, gy1 = geom.bounds if geom is not None else (0, 0, info.width, info.height)
    gx0, gy0, gx1, gy1 = max(0.0, gx0), max(0.0, gy0), min(float(info.width), gx1), min(float(info.height), gy1)
    while (gx1 - gx0) * (gy1 - gy0) / f**2 > max_pixels:
        f *= 1.25
    lv = 0
    for i, Lv in enumerate(info.levels):
        if Lv.downsample <= f * 1.0001:
            lv = i
    ds = info.levels[lv].downsample
    k = f / ds  # level px per classification px
    pad = 24

    def read(x0, y0, x1, y1):
        """Level-0 box → RGB at the classification resolution."""
        lx0, ly0 = int(max(0, x0 / ds)), int(max(0, y0 / ds))
        lx1 = int(min(info.levels[lv].width, math.ceil(x1 / ds)))
        ly1 = int(min(info.levels[lv].height, math.ceil(y1 / ds)))
        img = _rgb(src, lv, lx0, ly0, max(1, lx1 - lx0), max(1, ly1 - ly0))
        if k > 1.01:
            img = np.asarray(
                Image.fromarray(img).resize(
                    (max(1, round(img.shape[1] / k)), max(1, round(img.shape[0] / k))), Image.BILINEAR
                )
            )
        return img, lx0 * ds, ly0 * ds

    def feats(img):
        return multiscale_basic_features(img, channel_axis=-1, sigma_min=1, sigma_max=12, num_sigma=4).astype(
            np.float32
        )

    # ---- training pixels
    upd(0.02, f'reading {len(keep)} training shapes…')
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for j, i in enumerate(keep):
        rs = F.rings(i)
        ring = rs[0]
        bx0, by0 = ring.min(0) - pad * f
        bx1, by1 = ring.max(0) + pad * f
        img, ox, oy = read(bx0, by0, bx1, by1)
        hh, ww = img.shape[:2]
        m = Image.new('1', (ww, hh), 0)
        dr = ImageDraw.Draw(m)
        pts = [((x - ox) / f, (y - oy) / f) for x, y in ring]
        if F.kind[i] == annio.GEO_POLYGON and len(pts) >= 3:
            dr.polygon(pts, fill=1)
            for hole in rs[1:]:
                dr.polygon([((x - ox) / f, (y - oy) / f) for x, y in hole], fill=0)
        else:
            dr.line(pts, fill=1, width=3)
        m = np.asarray(m, bool)
        yy, xx = np.nonzero(m)
        if not len(yy):
            continue
        fe = feats(img)
        sel = rng.choice(len(yy), min(len(yy), 4000), replace=False)
        Xs.append(fe[yy[sel], xx[sel]])
        ys.append(np.full(len(sel), labels.index(str(names[i])), np.int16))
        upd(0.02 + 0.13 * (j + 1) / len(keep))
    if not Xs:
        raise ValueError('the training shapes are too small at this detail: draw larger ones or pick "fine"')
    X, y = np.concatenate(Xs), np.concatenate(ys)
    got = sorted(set(y.tolist()))
    if len(got) < 2:
        raise ValueError('need pixels of at least two classes: draw larger shapes')
    upd(0.16, f'training on {len(X):,} pixels of {len(got)} classes…')
    clf = RandomForestClassifier(n_estimators=60, max_depth=16, min_samples_leaf=2, class_weight='balanced',
                                 n_jobs=-1, random_state=0)  # fmt: skip
    clf.fit(X, y)

    # ---- classify the scope, tile by tile
    W = int(math.ceil((gx1 - gx0) / f))
    H = int(math.ceil((gy1 - gy0) / f))
    out = np.zeros((H, W), np.uint8)  # 0: not classified, 1 + class index
    T = 512
    tiles = [(tx, ty) for ty in range(0, H, T) for tx in range(0, W, T)]
    if geom is not None:
        shapely.prepare(geom)
    for n_t, (tx, ty) in enumerate(tiles):
        upd(0.18 + 0.72 * n_t / len(tiles), f'classifying tile {n_t + 1} of {len(tiles)}…')
        x0, y0 = gx0 + tx * f, gy0 + ty * f
        tw, th = min(T, W - tx), min(T, H - ty)
        box = shapely.box(x0, y0, x0 + tw * f, y0 + th * f)
        if geom is not None and not geom.intersects(box):
            continue
        img, ox, oy = read(x0 - pad * f, y0 - pad * f, x0 + (tw + pad) * f, y0 + (th + pad) * f)
        fe = feats(img)
        sx, sy = int(round((x0 - ox) / f)), int(round((y0 - oy) / f))
        sub = fe[sy : sy + th, sx : sx + tw]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            continue
        pred = clf.predict(sub.reshape(-1, sub.shape[-1])).reshape(sub.shape[:2]).astype(np.uint8) + 1
        if info.mode == 'rgb':  # glass (bright and grey) is no class: it is left unclassified
            px = img[sy : sy + pred.shape[0], sx : sx + pred.shape[1]].astype(np.int16)
            pred[(px.min(-1) > 215) & (px.max(-1) - px.min(-1) < 30)] = 0
        out[ty : ty + sub.shape[0], tx : tx + sub.shape[1]] = pred
    # ---- clean up and outline
    upd(0.92, 'outlining the areas…')
    if geom is not None:
        inside = Image.new('1', (W, H), 0)
        dr = ImageDraw.Draw(inside)
        for g in getattr(geom, 'geoms', [geom]):
            if g.geom_type != 'Polygon':
                continue
            dr.polygon([((x - gx0) / f, (y - gy0) / f) for x, y in g.exterior.coords], fill=1)
            for hole in g.interiors:
                dr.polygon([((x - gx0) / f, (y - gy0) / f) for x, y in hole.coords], fill=0)
        out[~np.asarray(inside, bool)] = 0
    out = ndi.median_filter(out, size=5)
    min_px = max(4, int(2000 / (_mpp(src) * f) ** 2))  # areas under ~2000 µm² are noise
    items, kpis = [], []
    px_um2 = (_mpp(src) * f) ** 2
    total = 0.0
    for ci, lab in enumerate(labels):
        m = out == ci + 1
        lab_, nlab = ndi.label(m)
        if nlab:
            sizes = np.bincount(lab_.ravel())
            m = (sizes >= min_px)[lab_] & m
        if not m.any():
            continue
        g = None
        for c in measure.find_contours(np.pad(m, 1).astype(np.uint8), 0.5):
            if len(c) < 4:
                continue
            p = shapely.Polygon((c[:, ::-1] - 1) * f + [gx0, gy0]).buffer(0)
            g = p if g is None else g.symmetric_difference(p)
        if g is None or g.is_empty:
            continue
        g = g.simplify(f * 0.75)
        area = float(m.sum()) * px_um2 / 1e6
        total += area
        items.append((lab, g))
        kpis.append([lab, f'{area:.3g} mm²'])
    if not items:
        raise ValueError(f'no areas found in {where}')
    name = _unique(registry, sid, name or f'Areas · {training}')
    L = _geom_layer(items, name, fill_opacity=0.3, line_width=1.5, detail_min_px=0)
    for lab in labels:
        if lab in TL.classes:
            L.classes[lab] = TL.classes[lab]
    L.origin = ORIGIN
    registry.add_annotations(sid, L)
    for kp in kpis:
        a = float(kp[1].split()[0])
        kp[1] += f' · {100 * a / total:.0f}%' if total else ''
    return _summary(
        None, L.name, len(items), len(labels), None, where, t0, 'areas',
        note=f'random forest on {len(X):,} pixels; classified at {_mpp(src) * f:.2g} µm/px',
        kpis=kpis,
    )  # fmt: skip


def capabilities() -> dict:
    """What this server can do: SAM backend, whether area classification is available."""
    try:
        import sklearn  # noqa: F401

        areas = True
    except ImportError:
        areas = False
    st = sam_service().status()
    return {
        'areas': areas,
        'sam': st.get('backend') or (st.get('available') or ['classic'])[0],
        'details': list(DETAIL_UM),
        'text': {'grounding': grounding.status(), 'sam3': sam3.status()},
        'cells': cell_models(),
    }


def run(registry, sid: str, body: dict, job: Job | None = None) -> dict:
    """The app's ``POST /slides/{sid}/find``: {mode: objects | segment | areas | text, examples, candidates, scope,
    strictness, object_um, training, classes, detail, prompt, model, name}."""
    mode = body.get('mode') or 'objects'
    if mode == 'objects':
        if not body.get('candidates'):
            raise ValueError('pick the layer to search (e.g. detected cells), or use "Segment everything"')
        return find_objects(registry, sid, examples=body.get('examples') or {}, candidates=body['candidates'],
                            scope=body.get('scope'), strictness=float(body.get('strictness', 0.5)),
                            name=body.get('name'), job=job)  # fmt: skip
    if mode == 'segment':
        ex = body.get('examples') or None
        if ex and not (ex.get('layer') or ex.get('picks')):
            ex = None
        return segment_everything(registry, sid, examples=ex, scope=body.get('scope'),
                                  strictness=float(body.get('strictness', 0.5)),
                                  object_um=float(body.get('object_um') or 10), name=body.get('name'), job=job,
                                  object_um_max=body.get('object_um_max'))  # fmt: skip
    if mode == 'areas':
        if not body.get('training'):
            raise ValueError('pick the layer with the training shapes')
        return find_areas(registry, sid, training=body['training'], classes=body.get('classes'),
                          scope=body.get('scope'), detail=body.get('detail') or 'medium', name=body.get('name'),
                          job=job)  # fmt: skip
    if mode == 'cells' or (mode == 'text' and body.get('model') in CELL_MODELS):
        return find_cells(registry, sid, model=body.get('model') or 'cellpose', scope=body.get('scope'),
                          name=body.get('name'), job=job)  # fmt: skip
    if mode == 'text':
        ex = body.get('examples') or None
        if ex and not (ex.get('layer') or ex.get('picks')):
            ex = None
        return find_by_text(registry, sid, prompt=body.get('prompt') or '', model=body.get('model') or 'grounding',
                            examples=ex, scope=body.get('scope'), strictness=float(body.get('strictness', 0.5)),
                            object_um=float(body.get('object_um') or 10), name=body.get('name'), job=job,
                            expand=bool(body.get('expand')), object_um_max=body.get('object_um_max'),
                            exclude=body.get('exclude') or '')  # fmt: skip
    raise ValueError(f'unknown mode {mode!r} (objects | segment | areas | text | cells)')
