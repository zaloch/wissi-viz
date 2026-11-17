"""A rectangle of a slide as an RGB picture (and layers' outlines drawn on it): read at the coarsest pyramid level that still gives the size asked for,
channels coloured and blended with their display windows (fluorescence), and parts outside the slide filled (black
for fluorescence, white for brightfield). Used by the thumbnails of saved views and by figure exports."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .annio import GEO_POINT, GEO_POLYGON


def is_fluorescence(source) -> bool:
    return source.info.mode != 'rgb'


def region_rgb(
    source, rect, size: int | tuple[int, int], windows=None, channels=None, gray: bool = False
) -> np.ndarray:
    """``rect``: (x0, y0, x1, y1) in level-0 px (may reach outside the slide). ``size``: the longer side in px, or
    (width, height) to resize to exactly. ``windows``: per-channel (lo, hi) display windows (default: the slide's
    channel statistics). ``channels``: the channels to blend (fluorescence; default: all); ``gray``: each in white
    (one channel in grey). Returns an (h, w, 3) uint8 array."""
    info = source.info
    x0, y0, x1, y1 = (float(v) for v in rect)
    rw, rh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    if isinstance(size, tuple):
        out_w, out_h = max(1, int(size[0])), max(1, int(size[1]))
    else:
        k = max(1, int(size)) / max(rw, rh)
        out_w, out_h = max(1, round(rw * k)), max(1, round(rh * k))
    need = max(out_w / rw, out_h / rh)  # output px per level-0 px
    lv = 0
    for i, L in enumerate(info.levels):
        if 1 / L.downsample >= need * 0.999:
            lv = i
    ds = info.levels[lv].downsample
    # the part inside the slide, in level coordinates
    cx0, cy0 = max(0.0, x0), max(0.0, y0)
    cx1, cy1 = min(float(info.width), x1), min(float(info.height), y1)
    fl = is_fluorescence(source)
    bg = 0 if fl else 255
    canvas_w, canvas_h = max(1, round(rw / ds)), max(1, round(rh / ds))
    canvas = np.full((canvas_h, canvas_w, 3), bg, np.uint8)
    if cx1 - cx0 >= 1 and cy1 - cy0 >= 1:
        args = (lv, int(cx0 / ds), int(cy0 / ds), max(1, int((cx1 - cx0) / ds)), max(1, int((cy1 - cy0) / ds)))
        if not fl:
            a = np.asarray(source.read_region(*args))[..., :3]
        else:
            wins = windows or source.channel_stats()
            use = [
                c for c in (range(len(info.channels)) if channels is None else channels) if 0 <= c < len(info.channels)
            ]
            comp = np.zeros((args[4], args[3], 3), np.float32)
            for ci in use:
                ch = info.channels[ci]
                t = np.asarray(source.read_region(*args, ci)).astype(np.float32)
                lo, hi = wins[ci]
                t = np.clip((t - lo) / max(hi - lo, 1), 0, 1)
                col = np.array([255.0] * 3 if gray else [int(ch.color[i : i + 2], 16) for i in (1, 3, 5)], np.float32)
                comp[: t.shape[0], : t.shape[1]] += t[..., None] * col
            a = np.clip(comp, 0, 255).astype(np.uint8)
        ox, oy = round((cx0 - x0) / ds), round((cy0 - y0) / ds)
        h, w = min(a.shape[0], canvas_h - oy), min(a.shape[1], canvas_w - ox)
        if h > 0 and w > 0:
            canvas[oy : oy + h, ox : ox + w] = a[:h, :w]
    im = Image.fromarray(canvas)
    if im.size != (out_w, out_h):
        im = im.resize((out_w, out_h), Image.LANCZOS)
    return np.asarray(im)


def _inside(F, bounds) -> np.ndarray:
    """Indices of features whose box meets the rectangle."""
    if not len(F):
        return np.zeros(0, np.int64)
    b = F.bbox()
    x0, y0, x1, y1 = bounds
    return np.nonzero((b[:, 2] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 1] <= y1))[0]


def _color(F, name: str) -> str:
    c = (F.class_colors or {}).get(name) or (F.style or {}).get('color')
    return c if isinstance(c, str) and c.startswith('#') else '#ffd23f'


def layer_key(features, bounds, limit: int = 8) -> list[dict]:
    """The classes of the features in a rectangle: [{'name', 'color'}] (the key of a window that draws them)."""
    out: dict[str, str] = {}
    for F in features:
        idx = _inside(F, bounds)
        if not len(idx):
            continue
        present = set(F.class_of()[idx].tolist())
        for n in [c for c in [*F.class_names, ''] if c in present]:  # the layer's own order, in every window
            label = n or F.name
            out.setdefault(label, _color(F, n))
            if len(out) >= limit:
                return [{'name': k, 'color': v} for k, v in out.items()]
    return [{'name': k, 'color': v} for k, v in out.items()]


def draw_layers(img: np.ndarray, bounds, features, max_features: int = 50_000) -> np.ndarray:
    """Outlines (polygons, lines) and dots (points) of features (:class:`wissi_viz.annio.Features`) inside
    ``bounds`` (level-0 px) drawn on a picture of that rectangle, in their class colours."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bounds)
    sx, sy = W / max(x1 - x0, 1e-9), H / max(y1 - y0, 1e-9)
    lw = max(1, round(W / 400))
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    for F in features:
        idx = _inside(F, bounds)[:max_features]
        if not len(idx):
            continue
        names = F.class_of()
        cache: dict[str, str] = {}
        for f in idx.tolist():
            n = names[f]
            col = cache.get(n) or cache.setdefault(n, _color(F, n))
            for ring in F.rings(f):
                pts = [((px - x0) * sx, (py - y0) * sy) for px, py in ring]
                if F.kind[f] == GEO_POINT:
                    cx, cy = pts[0]
                    r = max(2.0, lw * 2.0)
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=lw)
                elif len(pts) >= 2:
                    d.line(pts + ([pts[0]] if F.kind[f] == GEO_POLYGON else []), fill=col, width=lw)
    return np.asarray(im)
