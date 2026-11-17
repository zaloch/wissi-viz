"""Figure exports (matplotlib, PSF-based licence): the pages of :func:`wissi_viz.figures.layout` drawn as PNG, TIFF,
PDF or SVG — views as pictures, letters, labels, row heads, titles and charts as text and vector shapes (PDF and SVG
keep text editable) — and the export bundle: the picture(s), the recipe snapshot and the chart's data."""

from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import math
import zipfile
from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .figures import PT, box_stats, chart_csv

FORMATS = ('png', 'tiff', 'pdf', 'svg', 'pptx')
MIME = {'png': 'image/png', 'tiff': 'image/tiff', 'pdf': 'application/pdf', 'svg': 'image/svg+xml',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'zip': 'application/zip'}  # fmt: skip
SANS = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def _rc(fonts: dict | None) -> dict:
    fam = [f for f in [(fonts or {}).get('body')] if f] + SANS
    return {'font.family': 'sans-serif', 'font.sans-serif': fam, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
            'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6}  # fmt: skip


def draw_chart(ax, series: dict, ink: str, font: float) -> None:
    """Box, bar or dot chart of the series (one value per view, by group) in an axes."""
    groups = series.get('groups') or []
    kind = series.get('kind', 'box')
    for i, g in enumerate(groups):
        vals = [x['value'] for x in g['values'] if x['value'] is not None]
        col = g['color']
        pos = i + 1
        st = box_stats(vals)
        if st and kind == 'box':
            ax.add_patch(Rectangle((pos - 0.25, st['q1']), 0.5, max(st['q3'] - st['q1'], 1e-12), facecolor=col + '33',
                                   edgecolor=col, linewidth=1.0, zorder=2))  # fmt: skip
            ax.plot([pos - 0.25, pos + 0.25], [st['median']] * 2, color=col, linewidth=1.6, zorder=3)
            ax.plot([pos, pos], [st['lo'], st['q1']], color=col, linewidth=0.9, zorder=2)
            ax.plot([pos, pos], [st['q3'], st['hi']], color=col, linewidth=0.9, zorder=2)
        elif st and kind == 'bar':
            ax.bar([pos], [st['mean']], width=0.6, color=col + '99', edgecolor=col, linewidth=1.0, zorder=2)
        elif st and kind == 'dots':
            ax.plot([pos - 0.22, pos + 0.22], [st['median']] * 2, color=col, linewidth=1.6, zorder=3)
        n = len(vals)
        k = min(n, 5)
        xs = [pos + ((j % k) - (k - 1) / 2) * 0.07 for j in range(n)]
        ax.scatter(xs, vals, s=max(6, font * 1.6), color=col, zorder=4, linewidths=0)
    ax.set_xlim(0.4, len(groups) + 0.6)
    ax.set_xticks(range(1, len(groups) + 1), [g['name'] for g in groups])
    lo = min([0.0] + [x['value'] for g in groups for x in g['values'] if x['value'] is not None])
    ax.set_ylim(bottom=lo)
    ax.set_ylabel(series.get('ylabel') or '', fontsize=font, color=ink)
    ax.tick_params(colors=ink, labelsize=font * 0.9, length=2.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(ink)
    ax.set_facecolor('none')


def _page_figure(page: dict, result: dict, picture, series, dpi: int, fonts: dict | None) -> Figure:
    W, H = page['w'], page['h']
    bg = '#000000' if page.get('bg') == 'black' else '#ffffff'
    ink = page.get('ink') or '#111111'
    font = float(result.get('font') or 8)
    fig = Figure(figsize=(W / 25.4, H / 25.4), dpi=dpi, facecolor=bg)
    FigureCanvasAgg(fig)
    fx = lambda x: x / W
    fy = lambda y: 1 - y / H
    head_font = (fonts or {}).get('heading')
    t = page.get('title')
    if t and t.get('text') and t.get('rect'):
        x, y, w, h = t['rect']
        fig.text(fx(x), fy(y + h / 2), t['text'], ha='left', va='center', fontsize=min(28, h / PT * 0.6),
                 fontweight='bold', color=ink, **({'family': [head_font, *SANS]} if head_font else {}))  # fmt: skip
    for c in page.get('cols', []):
        x, y, w, h = c['rect']
        fig.text(fx(x + w / 2), fy(y + h - 0.6), c['text'], ha='center', va='bottom', fontsize=font, fontweight='bold',
                 color=c.get('color') or ink)  # fmt: skip
    for hd in page.get('heads', []):
        x, y, w, h = hd['rect']
        fig.text(fx(x + w / 2), fy(y + h / 2), hd['text'], ha='center', va='center', rotation=90, fontsize=font,
                 fontweight='bold', color=ink)  # fmt: skip
    for w in page['windows']:
        x, y, ww, wh = w['rect']
        if w['kind'] == 'view':
            pw, ph = max(8, round(ww / 25.4 * dpi)), max(8, round(wh / 25.4 * dpi))
            img = picture(w, min(pw, 6000), min(ph, 6000))
            ax = fig.add_axes((fx(x), fy(y + wh), ww / W, wh / H))
            ax.imshow(
                img, extent=(0, 1, 0, 1), aspect='auto', interpolation='lanczos' if img.shape[1] > pw else 'antialiased'
            )
            ax.set_axis_off()
            edge = '#ffffff' if w.get('fluorescence') else '#111111'
            ins = w.get('inset')
            if ins:  # the zoom: its own picture in a corner, and a dashed frame round the part it shows
                ix, iy, iw, ih = ins['rect']
                ipw, iph = max(8, round(iw / 25.4 * dpi)), max(8, round(ih / 25.4 * dpi))
                sub = picture({**w, 'bounds': ins['bounds']}, min(ipw, 4000), min(iph, 4000))
                ax2 = fig.add_axes((fx(ix), fy(iy + ih), iw / W, ih / H))
                ax2.imshow(sub, extent=(0, 1, 0, 1), aspect='auto', interpolation='antialiased')
                ax2.set_axis_off()
                fig.patches.append(Rectangle((fx(ix), fy(iy + ih)), iw / W, ih / H, transform=fig.transFigure,
                                             fill=False, edgecolor=edge, linewidth=0.8, zorder=6))  # fmt: skip
                fx0, fy0, fw0, fh0 = ins['frame']
                fig.patches.append(Rectangle((fx(fx0), fy(fy0 + fh0)), fw0 / W, fh0 / H, transform=fig.transFigure,
                                             fill=False, edgecolor=edge, linewidth=0.7, linestyle='--', zorder=6))  # fmt: skip
            for i, k in enumerate(w.get('legend') or []):  # the layers' classes
                ky = y + wh - 1.2 - (len(w['legend']) - i) * font * PT * 1.25
                fig.patches.append(Rectangle((fx(x + 1.2), fy(ky + font * PT * 0.8)), font * PT * 0.8 / W, font * PT * 0.8 / H,
                                             transform=fig.transFigure, facecolor=k['color'], edgecolor='none', zorder=6))  # fmt: skip
                fig.text(fx(x + 1.2 + font * PT * 1.2), fy(ky + font * PT * 0.85), k['name'], ha='left', va='bottom',
                         fontsize=font * 0.8, color=edge, zorder=7)  # fmt: skip
            sb = w.get('scalebar')
            if sb:
                m = 1.5 if ww < 60 else 3.0
                bh = max(0.5, wh * 0.015)
                bx, by = x + ww - m - sb['mm'], y + wh - m - bh
                fig.patches.append(Rectangle((fx(bx), fy(by + bh)), sb['mm'] / W, bh / H, transform=fig.transFigure,
                                             facecolor=sb['color'], edgecolor='none', zorder=5))  # fmt: skip
                fig.text(fx(bx + sb['mm'] / 2), fy(by - 0.4), sb['text'], ha='center', va='bottom',
                         fontsize=font * 0.8, fontweight='bold', color=sb['color'], zorder=6)  # fmt: skip
            if w.get('label') and w.get('label_rect'):
                lx, ly, lw, lh = w['label_rect']
                fig.text(fx(lx), fy(ly), w['label'], ha='left', va='top', fontsize=font, color=ink)
        elif w['kind'] == 'chart' and series:
            ml, mb, mt, mr = font * PT * 5.2, font * PT * 3.2, font * PT * 2.2, 1.0
            ax = fig.add_axes((fx(x + ml), fy(y + wh - mb), max(1, ww - ml - mr) / W, max(1, wh - mb - mt) / H))
            draw_chart(ax, series, ink, font)
        elif w['kind'] == 'text' and w.get('text'):
            fig.text(fx(x), fy(y), w['text'], ha='left', va='top', fontsize=font, color=ink, wrap=True)
        if w.get('letter'):
            col = ('#ffffff' if w.get('fluorescence') else '#111111') if w['kind'] == 'view' else ink
            fig.text(fx(x + 1.0), fy(y + 0.8), w['letter'], ha='left', va='top', fontsize=font * 1.3,
                     fontweight='bold', color=col, zorder=7)  # fmt: skip
    return fig


def render(
    result: dict,
    picture: Callable[[dict, int, int], np.ndarray],
    fmt: str = 'png',
    dpi: int = 300,
    series: dict | None = None,
    fonts: dict | None = None,
) -> list[bytes]:
    """The pages as files of ``fmt`` (png | tiff | pdf | svg): one per page, or one multi-page PDF.
    ``picture(window, w_px, h_px)``: a view window's picture as an (h, w, 3) uint8 array."""
    if fmt not in ('png', 'tiff', 'pdf', 'svg'):
        raise ValueError(f'format: png, tiff, pdf or svg, not {fmt!r}')
    out = []
    with matplotlib.rc_context(_rc(fonts)):
        if fmt == 'pdf':
            buf = io.BytesIO()
            with PdfPages(buf, metadata={'Creator': 'wissi-viz'}) as pdf:
                for page in result['pages']:
                    pdf.savefig(_page_figure(page, result, picture, series, dpi, fonts))
            return [buf.getvalue()]
        for page in result['pages']:
            fig = _page_figure(page, result, picture, series, dpi, fonts)
            buf = io.BytesIO()
            kw: dict[str, Any] = {'format': fmt, 'dpi': dpi, 'facecolor': fig.get_facecolor()}
            if fmt == 'tiff':
                kw['pil_kwargs'] = {'compression': 'tiff_lzw', 'dpi': (dpi, dpi)}
            fig.savefig(buf, **kw)
            out.append(buf.getvalue())
    return out


def chart_png(series: dict, w_px: int, h_px: int, *, dpi: int = 200, ink: str = '#111111', font: float = 12,
              fonts: dict | None = None) -> bytes:  # fmt: skip
    """The chart alone, transparent, as PNG (for a slide)."""
    with matplotlib.rc_context(_rc(fonts)):
        fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
        FigureCanvasAgg(fig)
        W, H = w_px / dpi * 25.4, h_px / dpi * 25.4
        ml, mb, mt, mr = font * PT * 5.2, font * PT * 3.2, font * PT * 2.2, 1.0
        ax = fig.add_axes((ml / W, mb / H, max(1, W - ml - mr) / W, max(1, H - mb - mt) / H))
        draw_chart(ax, series, ink, font)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, transparent=True)
        return buf.getvalue()


def snapshot(name: str, recipe: dict, members: list[dict]) -> dict:
    """What the figure was made of at export: the recipe and each view's slide, frame and metadata, so that the
    figure can be drawn again exactly as exported."""
    return {
        'wissi_viz_figure': 1,
        'name': name,
        'exported': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        'recipe': recipe,
        'views': [
            {
                'id': int(v['id']),
                'slide': v.get('slide'),
                'name': v.get('name'),
                'bounds': v['bounds'],
                'meta': v.get('meta') or {},
                'mpp': v.get('mpp'),
            }
            for v in members
        ],  # fmt: skip
    }


def bundle(stem: str, fmt: str, files: list[bytes], *, snap: dict | None = None, series: dict | None = None) -> tuple[
    bytes, str, str]:  # fmt: skip
    """(data, file name, media type): the one file, or a zip of the pages, the snapshot (``.recipe.json``) and the
    chart's data (``.csv``)."""
    ext = 'tif' if fmt == 'tiff' else fmt
    if len(files) == 1 and snap is None and series is None:
        return files[0], f'{stem}.{ext}', MIME[fmt]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        width = max(1, int(math.log10(max(1, len(files)))) + 1)
        for i, data in enumerate(files, start=1):
            z.writestr(f'{stem}.{ext}' if len(files) == 1 else f'{stem}-{i:0{width}d}.{ext}', data)
        if snap is not None:
            z.writestr(f'{stem}.recipe.json', json.dumps(snap, indent=1, ensure_ascii=False))
        if series is not None:
            z.writestr(f'{stem}.csv', chart_csv(series))
    return buf.getvalue(), f'{stem}.zip', MIME['zip']
