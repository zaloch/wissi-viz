"""Generate the wissi-viz tiled-W app icon in SVG, PNG, ICO, and ICNS formats."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / 'src' / 'wissi_viz' / 'assets'
SIZE = 1024
BG = (23, 19, 33, 255)
TILE_SIZE = 170
TILE_RADIUS = 42
PLATE = (24, 24, 1000, 1000)
PLATE_RADIUS = 224
TILES = [
    (112, 322, '#FFB84D'),
    (270, 514, '#68D8B4'),
    (427, 322, '#8D7BCE'),
    (584, 514, '#68D8B4'),
    (742, 322, '#FFB84D'),
]


def render(size: int = SIZE) -> Image.Image:
    """Draw at 2× resolution, then downsample for clean rounded edges."""
    scale = 2 * size / SIZE
    large = Image.new('RGBA', (2 * size, 2 * size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(large)

    def scaled(box: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(round(value * scale) for value in box)

    draw.rounded_rectangle(scaled(PLATE), radius=round(PLATE_RADIUS * scale), fill=BG)
    for x, y, color in TILES:
        box = (x, y, x + TILE_SIZE, y + TILE_SIZE)
        draw.rounded_rectangle(scaled(box), radius=round(TILE_RADIUS * scale), fill=color)
    return large.resize((size, size), Image.Resampling.LANCZOS)


def svg() -> str:
    tiles = '\n'.join(
        f'  <rect x="{x}" y="{y}" width="{TILE_SIZE}" height="{TILE_SIZE}" rx="{TILE_RADIUS}" fill="{color}"/>'
        for x, y, color in TILES
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">
  <rect x="{PLATE[0]}" y="{PLATE[1]}" width="{PLATE[2] - PLATE[0]}" height="{PLATE[3] - PLATE[1]}"
        rx="{PLATE_RADIUS}" fill="#171321"/>
  <g>
{tiles}
  </g>
</svg>
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render(512).save(OUT / 'icon.png', optimize=True)
    big = render()
    big.save(OUT / 'icon.ico', sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    big.save(OUT / 'icon.icns')
    (OUT / 'icon.svg').write_text(svg(), encoding='utf-8')
    for path in sorted(OUT.glob('icon.*')):
        print(path.name, path.stat().st_size)


if __name__ == '__main__':
    main()
