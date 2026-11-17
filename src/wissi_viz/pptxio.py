"""PowerPoint templates for figures (python-pptx, MIT): read one, and write a figure's slides on it.

:func:`read_template` describes a .pptx / .potx: its slide size, theme colours and fonts, a sketch of its master (the
filled rectangles and pictures every slide shows: title bars, rules, logos) and its layouts — the layouts it defines
(their picture and content placeholders become windows) and layouts learned from its slides (where pictures were
placed by hand, e.g. a 2 × 3 grid of images on a 'Title Only' slide). A past talk works as a template too: its masters,
layouts and arrangements are used, not its slides' content.

:func:`write_pptx` writes the figure's pages (:func:`wissi_viz.figures.layout`) as slides on the template's master —
or on a plain 16:9 / 4:3 deck — with windows as pictures and titles, labels, letters and scale bars as shapes and
text, editable in PowerPoint.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_FILL
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.util import Emu, Pt

EMU_MM = 36000.0
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
#: placeholders a picture goes into
PICTURE_SLOTS = (PP_PLACEHOLDER.PICTURE, PP_PLACEHOLDER.OBJECT, PP_PLACEHOLDER.CHART, PP_PLACEHOLDER.TABLE,
                 PP_PLACEHOLDER.MEDIA_CLIP, PP_PLACEHOLDER.BITMAP)  # fmt: skip
TITLES = (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)
SCHEME = ('dk1', 'lt1', 'dk2', 'lt2', 'accent1', 'accent2', 'accent3', 'accent4', 'accent5', 'accent6')
_MAX_SKETCH_PICTURE = 400_000  # bytes: bigger master pictures (photos) are left out of the sketch


def _mm(v) -> float:
    return round(float(v or 0) / EMU_MM, 2)


def _rect(sh) -> list[float] | None:
    try:
        if sh.left is None or sh.width is None:
            return None
        return [_mm(sh.left), _mm(sh.top), _mm(sh.width), _mm(sh.height)]
    except (AttributeError, ValueError, KeyError):
        return None


def _reading_order(rects: list[list[float]]) -> list[list[float]]:
    """Top to bottom in rows (tops within 5 mm are one row), left to right in a row."""
    rs = sorted(rects, key=lambda r: (r[1], r[0]))
    rows: list[list[list[float]]] = []
    for r in rs:
        if rows and abs(rows[-1][0][1] - r[1]) < 5:
            rows[-1].append(r)
        else:
            rows.append([r])
    return [r for row in rows for r in sorted(row, key=lambda r: r[0])]


def _theme(prs) -> tuple[dict, dict]:
    colors, fonts = {}, {}
    try:
        root = etree.fromstring(prs.slide_masters[0].part.part_related_by(RT.THEME).blob)
    except (KeyError, IndexError, etree.XMLSyntaxError):
        return colors, fonts
    cs = root.find(f'.//{A}clrScheme')
    if cs is not None:
        for name in SCHEME:
            el = cs.find(f'{A}{name}')
            if el is None or not len(el):
                continue
            c = el[0]
            v = c.get('val') if c.tag == f'{A}srgbClr' else c.get('lastClr')
            if v and re.fullmatch(r'[0-9A-Fa-f]{6}', v):
                colors[name] = '#' + v.lower()
    for key, tag in (('heading', 'majorFont'), ('body', 'minorFont')):
        el = root.find(f'.//{A}fontScheme/{A}{tag}/{A}latin')
        if el is not None and el.get('typeface'):
            fonts[key] = el.get('typeface')
    return colors, fonts


def _color_of(fill, colors: dict) -> str | None:
    """A solid fill's colour as '#rrggbb' (theme colours through the theme), else None."""
    try:
        if fill.type != MSO_FILL.SOLID:
            return None
        fc = fill.fore_color
        try:
            return '#' + str(fc.rgb).lower()
        except AttributeError:
            pass
        name = str(fc.theme_color).split('.')[-1].split(' ')[0].lower()
        m = {'text_1': 'dk1', 'background_1': 'lt1', 'text_2': 'dk2', 'background_2': 'lt2', 'dark_1': 'dk1',
             'light_1': 'lt1', 'dark_2': 'dk2', 'light_2': 'lt2'}  # fmt: skip
        name = m.get(name, name.replace('_', ''))
        return colors.get(name)
    except Exception:  # noqa: BLE001  (an unusual fill: no colour)
        return None


def _sketch(shapes, colors: dict) -> list[dict]:
    """The decorations of a master or layout: filled rectangles and small pictures (not placeholders)."""
    out = []
    for sh in shapes:
        if getattr(sh, 'is_placeholder', False):
            continue
        r = _rect(sh)
        if r is None:
            continue
        try:
            st = sh.shape_type
        except NotImplementedError:
            continue
        if st == MSO_SHAPE_TYPE.PICTURE:
            try:
                img = sh.image
            except (AttributeError, ValueError, KeyError):
                continue
            if len(img.blob) <= _MAX_SKETCH_PICTURE and img.content_type in ('image/png', 'image/jpeg', 'image/gif'):
                out.append({'kind': 'image', 'rect': r,
                            'src': f'data:{img.content_type};base64,{base64.b64encode(img.blob).decode()}'})  # fmt: skip
        elif st == MSO_SHAPE_TYPE.AUTO_SHAPE:
            c = _color_of(sh.fill, colors)
            if c:
                out.append({'kind': 'rect', 'rect': r, 'color': c})
    return out


def _background(obj, colors: dict) -> str | None:
    try:
        return _color_of(obj.background.fill, colors)
    except Exception:  # noqa: BLE001
        return None


def read_template(path: str | Path) -> dict:
    """{'name', 'size': [w, h] mm, 'colors', 'fonts', 'background', 'master': [...], 'layouts': [{'name', 'source':
    'layout' | 'slides', 'base' (the layout a slide uses), 'slots': [[x, y, w, h] mm], 'title': rect | None,
    'master': [...] (that layout's own decorations)}]}."""
    p = Path(path)
    prs = Presentation(str(p))
    W, H = _mm(prs.slide_width), _mm(prs.slide_height)
    colors, fonts = _theme(prs)
    master = prs.slide_masters[0]
    out: dict[str, Any] = {'name': p.name, 'size': [W, H], 'colors': colors, 'fonts': fonts,
                           'background': _background(master, colors), 'master': _sketch(master.shapes, colors),
                           'layouts': []}  # fmt: skip
    seen: set[tuple] = set()
    for lay in prs.slide_layouts:
        slots, title = [], None
        for ph in lay.placeholders:
            t = ph.placeholder_format.type
            r = _rect(ph)
            if r is None:
                continue
            if t in TITLES and title is None:
                title = r
            elif t in PICTURE_SLOTS:
                slots.append(r)
        slots = _reading_order(slots)
        seen.add(_signature(slots))
        out['layouts'].append({'name': lay.name, 'source': 'layout', 'base': lay.name, 'slots': slots, 'title': title,
                               'master': _sketch(lay.shapes, colors)})  # fmt: skip
    # arrangements learned from the slides: pictures placed by hand (two or more on a slide)
    for i, slide in enumerate(prs.slides, start=1):
        pics, title = [], None
        for sh in slide.shapes:
            r = _rect(sh)
            if r is None:
                continue
            try:
                st = sh.shape_type
            except NotImplementedError:
                continue
            if getattr(sh, 'is_placeholder', False) and sh.placeholder_format.type in TITLES:
                title = r
            elif st == MSO_SHAPE_TYPE.PICTURE or (st == MSO_SHAPE_TYPE.PLACEHOLDER and hasattr(sh, 'image')):
                if r[2] > 5 and r[3] > 5:
                    pics.append(r)
        if len(pics) < 2:
            continue
        pics = _reading_order(pics)
        sig = _signature(pics)
        if sig in seen:
            continue
        seen.add(sig)
        name = f'Slide {i} · {_describe(pics)}'
        base = slide.slide_layout.name
        lay_title = next((x['title'] for x in out['layouts'] if x['name'] == base), None)
        out['layouts'].append({'name': name, 'source': 'slides', 'base': base, 'slots': pics,
                               'title': title or lay_title, 'master': []})  # fmt: skip
    return out


def _signature(rects: list[list[float]]) -> tuple:
    return tuple(tuple(round(v / 2) for v in r) for r in rects)


def _describe(rects: list[list[float]]) -> str:
    rows: list[int] = []
    last = None
    for r in rects:
        if last is None or abs(r[1] - last) >= 5:
            rows.append(0)
            last = r[1]
        rows[-1] += 1
    if len(set(rows)) == 1 and len(rows) > 1:
        return f'{len(rows)} × {rows[0]} pictures'
    return f'{len(rects)} pictures'


# ---------------------------------------------------------------------------------------------------------- write
def _emu(mm: float) -> Emu:
    return Emu(int(round(mm * EMU_MM)))


def _rgb(c: str) -> RGBColor:
    return RGBColor.from_string(c.lstrip('#')[:6].upper())


def _drop_slides(prs) -> None:
    lst = prs.slides._sldIdLst
    for sld in list(lst):
        prs.part.drop_rel(sld.rId)
        lst.remove(sld)


def _text(slide, rect, text: str, size: float, color: str, *, bold=False, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, font: str | None = None, rotation: float = 0.0):  # fmt: skip
    x, y, w, h = rect
    tb = slide.shapes.add_textbox(_emu(x), _emu(y), _emu(max(w, 1)), _emu(max(h, 1)))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    para = tf.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = _rgb(color)
    if font:
        run.font.name = font
    if rotation:
        tb.rotation = rotation
    return tb


def write_pptx(
    pages: dict,
    picture: Callable[[dict, int, int], bytes],
    chart_png: Callable[[dict, int, int], bytes] | None = None,
    *,
    template: str | Path | None = None,
    info: dict | None = None,
    layout_name: str | None = None,
    dpi: int = 200,
) -> bytes:
    """The figure's pages as a .pptx. ``pages``: :func:`wissi_viz.figures.layout`'s result. ``picture(window, w_px,
    h_px)``: a view window's picture (PNG / JPEG bytes); ``chart_png(window, w_px, h_px)``: the chart window's.
    ``template``: the .pptx / .potx to build on (its slides are not kept), else a plain deck of the pages' size."""
    prs = Presentation(str(template)) if template else Presentation()
    first = pages['pages'][0] if pages['pages'] else {'w': 338.667, 'h': 190.5}
    if template:
        _drop_slides(prs)
    else:
        prs.slide_width, prs.slide_height = _emu(first['w']), _emu(first['h'])
    lays = {lay.name: lay for lay in prs.slide_layouts}
    chosen = next((x for x in (info or {}).get('layouts', []) if x['name'] == layout_name), None)
    if template:
        name = chosen['base'] if chosen else layout_name or ''
        base = lays.get(name) or lays.get('Title Only') or lays.get('Blank') or prs.slide_layouts[0]
    else:
        base = prs.slide_layouts[6]  # the default deck's Blank
    fonts = (info or {}).get('fonts', {})
    font = pages.get('font', 14.0)
    px = lambda mm: max(8, int(round(mm / 25.4 * dpi)))
    for page in pages['pages']:
        slide = prs.slides.add_slide(base)
        if page.get('bg') in ('black', 'white'):
            slide.background.fill.solid()
            slide.background.fill.fore_color.rgb = _rgb('#000000' if page['bg'] == 'black' else '#ffffff')
        ink = page.get('ink') or '#111111'
        title_ph = None
        for ph in list(slide.placeholders):
            if ph.placeholder_format.type in TITLES and title_ph is None:
                title_ph = ph
            else:  # windows go where the figure puts them: the template's empty placeholders would show prompts
                ph._element.getparent().remove(ph._element)
        t = page.get('title')
        if t and t.get('text'):
            if title_ph is not None:
                title_ph.text_frame.text = t['text']
            elif t.get('rect'):
                _text(
                    slide, t['rect'], t['text'], 28, ink, bold=True, anchor=MSO_ANCHOR.MIDDLE, font=fonts.get('heading')
                )
        elif title_ph is not None:
            title_ph._element.getparent().remove(title_ph._element)
        for c in page.get('cols', []):
            _text(slide, c['rect'], c['text'], font, c.get('color') or ink, bold=True, align=PP_ALIGN.CENTER,
                  anchor=MSO_ANCHOR.BOTTOM, font=fonts.get('body'))  # fmt: skip
        for hd in page.get('heads', []):
            x, y, w, h = hd['rect']
            cx, cy = x + w / 2, y + h / 2
            _text(slide, [cx - h / 2, cy - w / 2, h, w], hd['text'], font, ink, bold=True, align=PP_ALIGN.CENTER,
                  anchor=MSO_ANCHOR.MIDDLE, font=fonts.get('body'), rotation=270)  # fmt: skip
        for w in page['windows']:
            x, y, ww, wh = w['rect']
            if w['kind'] == 'view':
                data = picture(w, px(ww), px(wh))
                slide.shapes.add_picture(io.BytesIO(data), _emu(x), _emu(y), _emu(ww), _emu(wh))
                sb = w.get('scalebar')
                if sb:
                    m = 2.0
                    bar_h = max(0.8, wh * 0.012)
                    bx, by = x + ww - m - sb['mm'], y + wh - m - bar_h
                    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, _emu(bx), _emu(by), _emu(sb['mm']), _emu(bar_h))
                    r.fill.solid()
                    r.fill.fore_color.rgb = _rgb(sb['color'])
                    r.line.fill.background()
                    r.shadow.inherit = False
                    _text(slide, [bx - 10, by - font * 0.3528 * 0.8 - 1.2, sb['mm'] + 20, font * 0.3528],
                          sb['text'], font * 0.7, sb['color'], bold=True, align=PP_ALIGN.CENTER,
                          anchor=MSO_ANCHOR.BOTTOM)  # fmt: skip
                if w.get('label') and w.get('label_rect'):
                    _text(slide, w['label_rect'], w['label'], font * 0.85, ink, font=fonts.get('body'))
            elif w['kind'] == 'chart' and chart_png is not None:
                data = chart_png(w, px(ww), px(wh))
                slide.shapes.add_picture(io.BytesIO(data), _emu(x), _emu(y), _emu(ww), _emu(wh))
            elif w['kind'] == 'text' and w.get('text'):
                _text(slide, w['rect'], w['text'], font, ink, font=fonts.get('body'))
            if w.get('letter'):
                col = (
                    '#ffffff'
                    if (w['kind'] == 'view' and w.get('fluorescence'))
                    else ('#111111' if w['kind'] == 'view' else ink)
                )
                _text(slide, [x + 1.5, y + 1.0, 12, 10], w['letter'], font * 1.2, col, bold=True)
    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()
