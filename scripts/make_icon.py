"""Draw the wissi-viz app icon (an original design) and write every format the desktop app needs.

    python scripts/make_icon.py          # → src/wissi_viz/assets/icon.{svg,png,ico,icns}

A magnifier over stained tissue on a tile grid: pink stroma, purple nuclei, the pyramid tiles the viewer streams.
PNG/ICO/ICNS come from Pillow (supersampled 2×, then downsampled); the SVG is written from the same geometry.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / 'src' / 'wissi_viz' / 'assets'
N = 1024                           # design canvas
BG_TOP, BG_BOT = (24, 42, 70), (9, 15, 27)
INSET, RADIUS = 56, 210            # rounded-square plate
LENS_C, LENS_R, RING = (452, 452), 300, 46
HANDLE = ((690, 690), (850, 850), 104)
CYAN, CYAN_DARK = (94, 208, 255), (40, 140, 205)
STROMA = [((330, 360, 300, 230), (236, 140, 182)), ((560, 560, 260, 200), (222, 118, 168)),
          ((380, 610, 190, 150), (244, 170, 200)), ((610, 330, 170, 140), (240, 156, 192))]   # (cx, cy, rx, ry), rgb
NUCLEUS = (112, 66, 168)
GRID = 150                         # tile grid pitch inside the lens


def _nuclei():
    rng = random.Random(7)
    out = []
    while len(out) < 46:
        x, y = rng.uniform(170, 740), rng.uniform(170, 740)
        if (x - LENS_C[0]) ** 2 + (y - LENS_C[1]) ** 2 > (LENS_R - 30) ** 2:
            continue
        if any((x - a) ** 2 + (y - b) ** 2 < 46 ** 2 for a, b, *_ in out):
            continue
        out.append((x, y, rng.uniform(13, 22), rng.uniform(10, 17), rng.uniform(0, 180)))
    return out


NUCLEI = _nuclei()


def render(size: int = 1024) -> Image.Image:
    W = 2 * N                                  # draw at 2× then downsample

    def S(v):
        return v * 2

    # plate with vertical gradient
    grad = Image.new('RGB', (1, W))
    for y in range(W):
        t = y / (W - 1)
        grad.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOT)))
    grad = grad.resize((W, W))
    plate = Image.new('L', (W, W), 0)
    ImageDraw.Draw(plate).rounded_rectangle([S(INSET), S(INSET), W - S(INSET), W - S(INSET)], S(RADIUS), fill=255)
    img = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    img.paste(grad, (0, 0), plate)

    # tissue inside the lens
    lens = Image.new('L', (W, W), 0)
    cx, cy = S(LENS_C[0]), S(LENS_C[1])
    ImageDraw.Draw(lens).ellipse([cx - S(LENS_R), cy - S(LENS_R), cx + S(LENS_R), cy + S(LENS_R)], fill=255)
    tissue = Image.new('RGBA', (W, W), (250, 240, 246, 255))
    d = ImageDraw.Draw(tissue)
    for (x, y, rx, ry), c in STROMA:
        d.ellipse([S(x - rx), S(y - ry), S(x + rx), S(y + ry)], fill=c + (255,))
    tissue = tissue.filter(ImageFilter.GaussianBlur(S(10)))
    nuc = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    for x, y, rx, ry, a in NUCLEI:
        e = Image.new('RGBA', (S(int(rx * 2 + 4)), S(int(rx * 2 + 4))), (0, 0, 0, 0))
        ImageDraw.Draw(e).ellipse([S(2 + rx - rx), S(2 + rx - ry), S(2 + rx + rx), S(2 + rx + ry)], fill=NUCLEUS + (255,))
        e = e.rotate(a, resample=Image.BICUBIC)
        nuc.alpha_composite(e, (int(S(x) - e.width / 2), int(S(y) - e.height / 2)))
    tissue.alpha_composite(nuc)
    g = ImageDraw.Draw(tissue)
    for k in range(-3, 4):
        p = S(LENS_C[0] + k * GRID - GRID / 2)
        g.line([p, 0, p, W], fill=(255, 255, 255, 70), width=S(5))
        q = S(LENS_C[1] + k * GRID - GRID / 2)
        g.line([0, q, W, q], fill=(255, 255, 255, 70), width=S(5))
    img.paste(tissue, (0, 0), ImageChops.multiply(lens, plate))

    # handle (drawn first so the ring sits on top), then ring with a soft inner highlight
    d = ImageDraw.Draw(img)
    (x0, y0), (x1, y1), w = HANDLE
    d.line([S(x0), S(y0), S(x1), S(y1)], fill=CYAN_DARK, width=S(w))
    for (x, y) in ((x0, y0), (x1, y1)):
        d.ellipse([S(x - w / 2), S(y - w / 2), S(x + w / 2), S(y + w / 2)], fill=CYAN_DARK)
    r = S(LENS_R + RING / 2)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=CYAN, width=S(RING))
    r2 = S(LENS_R - 4)
    d.arc([cx - r2, cy - r2, cx + r2, cy + r2], 200, 250, fill=(255, 255, 255, 150), width=S(10))
    return img.resize((size, size), Image.LANCZOS)


def svg() -> str:
    (x0, y0), (x1, y1), w = HANDLE
    cx, cy = LENS_C
    blobs = ''.join(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="rgb{c}"/>' for (x, y, rx, ry), c in STROMA)
    nuclei = ''.join(f'<ellipse cx="{x:.0f}" cy="{y:.0f}" rx="{rx:.0f}" ry="{ry:.0f}" transform="rotate({-a:.0f} {x:.0f} {y:.0f})"/>'
                     for x, y, rx, ry, a in NUCLEI)
    lines = ''.join(f'<path d="M{cx + k * GRID - GRID / 2:.0f} 0V1024M0 {cy + k * GRID - GRID / 2:.0f}H1024"/>' for k in range(-3, 4))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="rgb{BG_TOP}"/><stop offset="1" stop-color="rgb{BG_BOT}"/></linearGradient>
<clipPath id="lens"><circle cx="{cx}" cy="{cy}" r="{LENS_R}"/></clipPath>
<filter id="soft"><feGaussianBlur stdDeviation="10"/></filter>
</defs>
<rect x="{INSET}" y="{INSET}" width="{1024 - 2 * INSET}" height="{1024 - 2 * INSET}" rx="{RADIUS}" fill="url(#bg)"/>
<g clip-path="url(#lens)">
<rect width="1024" height="1024" fill="rgb(250,240,246)"/>
<g filter="url(#soft)">{blobs}</g>
<g fill="rgb{NUCLEUS}">{nuclei}</g>
<g stroke="#fff" stroke-opacity=".27" stroke-width="5">{lines}</g>
</g>
<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="rgb{CYAN_DARK}" stroke-width="{w}" stroke-linecap="round"/>
<circle cx="{cx}" cy="{cy}" r="{LENS_R + RING / 2:.0f}" fill="none" stroke="rgb{CYAN}" stroke-width="{RING}"/>
</svg>
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    big = render(1024)
    render(512).save(OUT / 'icon.png', optimize=True)
    big.save(OUT / 'icon.ico', sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    big.save(OUT / 'icon.icns')
    (OUT / 'icon.svg').write_text(svg(), encoding='utf-8')
    for p in sorted(OUT.glob('icon.*')):
        print(p, p.stat().st_size)


if __name__ == '__main__':
    main()
