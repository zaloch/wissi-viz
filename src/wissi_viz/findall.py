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

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure
from skimage.color import rgb2lab
from skimage.feature import multiscale_basic_features, peak_local_max

from . import annio, grounding, sam3
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
    whole slide becomes its tissue (segmenting or classifying glass is wasted time)."""
    scope = scope or {'kind': 'slide'}
    src = _source(registry, sid)
    W, H = src.info.width, src.info.height
    kind = scope.get('kind', 'slide')
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


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    lens = np.asarray(lens, np.int64)
    if not len(lens) or lens.sum() == 0:
        return np.zeros(0, np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return np.repeat(np.asarray(starts, np.int64) - offs, lens) + np.arange(int(lens.sum()))


def _subset(F: annio.Features, idx: np.ndarray) -> annio.Features:
    """The features ``idx`` of F (geometry, classes, properties), vectorised."""
    idx = np.asarray(idx, np.int64)
    fr0, nr = F.feat_ring_off[idx], F.feat_ring_off[idx + 1] - F.feat_ring_off[idx]
    rings = _ranges(fr0, nr)
    v0 = F.ring_off[rings]
    nv = F.ring_off[rings + 1] - v0
    G = annio.Features(
        F.kind[idx],
        F.coords[_ranges(v0, nv)],
        np.concatenate([[0], np.cumsum(nv)]).astype(np.int64),
        np.concatenate([[0], np.cumsum(nr)]).astype(np.int64),
        F.cls[idx],
        list(F.class_names),
    )
    G.class_colors = dict(F.class_colors)
    G.props = {k: np.asarray(v)[idx] for k, v in F.props.items()}
    return G


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
) -> dict:
    """Outline every object in the scope from a grid of single-click prompts (SAM, or the classic colour fill), then
    keep those like the examples (all of them without examples)."""
    t0 = time.perf_counter()
    upd = job.update if job else (lambda *a, **k: None)
    src = _source(registry, sid)
    info = src.info
    ex = Examples(registry, sid, examples) if examples else None
    ex = ex if ex else None  # no positive examples: keep every outline
    cls, color = (ex.cls, ex.color) if ex else ('object', '#ffb454')
    d0 = ex.diameter() if ex else object_um / _mpp(src)  # object diameter, level-0 px
    geom, where = scope_geometry(registry, sid, scope, tissue=True)
    svc = sam_service()
    impl = svc._load()
    want_ds = max(1.0, d0 / 28.0)  # objects ~28 px across in the model's input
    lv = 0
    for i, L in enumerate(info.levels):
        if L.downsample <= want_ds * 1.0001:
            lv = i
    L = info.levels[lv]
    ds = L.downsample
    d_img = d0 / ds
    T = 1024 if impl.name != 'classic' else 768
    O = int(min(T // 4, max(16, 2.5 * d_img)))  # noqa: E741
    step = T - O
    if geom is not None:
        gx0, gy0, gx1, gy1 = geom.bounds
    else:
        gx0, gy0, gx1, gy1 = 0, 0, info.width, info.height
    tiles = []
    for ty in range(int(gy0 / ds // step), int(math.ceil(gy1 / ds / step)) + 1):
        for tx in range(int(gx0 / ds // step), int(math.ceil(gx1 / ds / step)) + 1):
            x, y = tx * step, ty * step
            w, h = min(T, L.width - x), min(T, L.height - y)
            if w < 8 or h < 8:
                continue
            box = shapely.box(x * ds, y * ds, (x + w) * ds, (y + h) * ds)
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
        amin, amax = float(ea.min()) / 4 / ds**2, float(ea.max()) * 4 / ds**2
    else:
        amin, amax = (0.08 * d_img) ** 2, (3.0 * d_img) ** 2 * math.pi
    amin = max(amin, 6.0)
    polys: list[np.ndarray] = []
    name = _unique(registry, sid, name or (f'Segmented · {cls}' if ex else 'Segmented objects'))
    prov = f'{name} (segmenting…)'
    last_pub = time.perf_counter()
    upd(0.0, f'{impl.name}: {len(tiles)} tile{"s" if len(tiles) > 1 else ""} in {where}')
    for ti, (x, y, w, h) in enumerate(tiles):
        upd(ti / len(tiles), f'{impl.name}: tile {ti + 1} of {len(tiles)} · {len(polys):,} objects')
        rgb = _rgb(src, lv, x, y, w, h)
        polys += _tile_objects(svc, impl, rgb, x, y, ds, d_img, amin, amax, O, (x == 0, y == 0,
                               x + w >= L.width, y + h >= L.height), geom, job)  # fmt: skip
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
    note = f'{impl.name} at level {lv} ({ds:g}× downsample)'
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
                m, score = impl.predict(emb, [(px, py)], [1], None)
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
) -> dict:
    """Outline everything in the scope that matches the text ``prompt`` (one class per phrase: "nucleus. gland").

    ``grounding``: Grounding DINO finds boxes, SAM 2.1 outlines each; ``sam3``: SAM 3 finds and outlines in one go.
    The slide is read in tiles at the scale where an object of ``object_um`` (or the examples' size) is ~48 px
    across. With examples, only matches like them are kept too."""
    t0 = time.perf_counter()
    upd = job.update if job else (lambda *a, **k: None)
    names = grounding.phrases(prompt)
    if not names:
        raise ValueError('type what to find, e.g. "nucleus" or "gland. vessel" (one class per phrase)')
    src = _source(registry, sid)
    info = src.info
    ex = Examples(registry, sid, examples) if examples else None
    ex = ex if ex else None
    mpp = _mpp(src)
    d0 = ex.diameter() if ex else float(object_um or 10.0) / mpp  # object diameter, level-0 px
    geom, where = scope_geometry(registry, sid, scope, tissue=True)
    if detector is None:
        upd(0.0, 'loading the model…')
        detector = text_detector(model, lambda f, m: upd(0.0, m))
    T = int(getattr(detector, 'tile', 800))
    f = max(0.5, d0 / 48.0)  # level-0 px per model px (≥ 0.5: at most 2× enlarged)
    lv = 0
    for i, Lv in enumerate(info.levels):
        if Lv.downsample <= f * 1.0001:
            lv = i
    ds = info.levels[lv].downsample
    S0 = T * f  # a tile's side, level-0 px
    O0 = min(S0 / 4, max(32 * f, 2.5 * d0))  # overlap: objects cut by one tile are whole in the next
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
        out = detector.detect(rgb, names, min_score=floor)
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
    polys = [polys[i] for i in keep]
    score = np.asarray(scores, float)[keep]
    lab = np.asarray(labels, np.int64)[keep]
    classes = [(n, PHRASE_COLORS[i % len(PHRASE_COLORS)]) for i, n in enumerate(names)]
    F = _polys_features(polys, names[0])
    sim = _text_sim(score)
    note = f'{label} at {f:.2g} level-0 px per model px'
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
    extra = {'score': np.round(score, 3)}
    lname, n = _publish(registry, sid, F, sim, strictness, name, names[0], classes[0][1], extra, classes, lab)
    rid = uuid.uuid4().hex[:10]
    _remember(rid, slide=sid, F=F, sim=sim, name=lname, cls=names[0], color=classes[0][1], extra=extra,
              strictness=strictness, classes=classes, labels=lab)  # fmt: skip
    return _summary(
        rid, lname, n, len(polys), strictness, where, t0, 'text', note=note,
        kpis=[['found', f'{n:,}'], ['matches', f'{len(polys):,}'], ['tiles', str(len(tiles))]],
    )  # fmt: skip


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
            'least two (e.g. Tumor and Stroma) — draw each with its class set in the Annotate tab'
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
                                  object_um=float(body.get('object_um') or 10), name=body.get('name'), job=job)  # fmt: skip
    if mode == 'areas':
        if not body.get('training'):
            raise ValueError('pick the layer with the training shapes')
        return find_areas(registry, sid, training=body['training'], classes=body.get('classes'),
                          scope=body.get('scope'), detail=body.get('detail') or 'medium', name=body.get('name'),
                          job=job)  # fmt: skip
    if mode == 'text':
        ex = body.get('examples') or None
        if ex and not (ex.get('layer') or ex.get('picks')):
            ex = None
        return find_by_text(registry, sid, prompt=body.get('prompt') or '', model=body.get('model') or 'grounding',
                            examples=ex, scope=body.get('scope'), strictness=float(body.get('strictness', 0.5)),
                            object_um=float(body.get('object_um') or 10), name=body.get('name'), job=job)  # fmt: skip
    raise ValueError(f'unknown mode {mode!r} (objects | segment | areas | text)')
