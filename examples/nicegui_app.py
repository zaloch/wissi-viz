"""NiceGUI example: 1–16 slides with annotations, navigation and object inspection.

python examples/nicegui_app.py /path/to/slides
"""

import sys
from pathlib import Path

from nicegui import ui

from wissi_viz import AnnotationLayer
from wissi_viz.nicegui import WissiViz, configure

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
registry = configure(cache_dir=ROOT / '.wissi-viz-cache')  # persistent tile cache (mount as a Docker volume)
slide_ids = registry.add_directory(ROOT)  # sidecar .geojson/.annotations/.xml load automatically


@ui.page('/')
def index():
    info = ui.label('Click an object…').classes('text-xs opacity-70')

    def show_object(e: dict):
        info.text = f'{e["layer"]} #{e["index"]} · {e.get("class")} · {e.get("properties", {})}'

    def drawn(e: dict):
        # persist drawings as a layer on that slide (all viewers of the slide refresh)
        layer = registry.layers(e['slide']).get('Drawings') or registry.add_annotations(
            e['slide'], AnnotationLayer(name='Drawings', classes={'Drawing': '#ffeb3b'}, line_width=2.5)
        )
        g = e['geometry']
        if g['type'] == 'Polygon':
            layer.add_polygon(g['coordinates'][0], cls='Drawing')
        elif g['type'] == 'LineString':
            layer.add_polyline(g['coordinates'], cls='Drawing')
        else:
            layer.add_point(*g['coordinates'], cls='Drawing')
        registry.changed(e['slide'], 'Drawings')

    with ui.row().classes('items-center gap-2 w-full'):
        n = ui.select([1, 2, 3, 4, 6, 8, 9, 12, 13, 14, 15, 16], value=min(4, len(slide_ids)), label='Panes').classes(
            'w-24'
        )
        sync = ui.select(['none', 'all', 'pan', 'zoom'], value='none', label='Sync').classes('w-24')
        tool = ui.toggle(['pan', 'rect', 'polygon', 'freehand', 'point'], value='pan')
        x = ui.number('x', value=0).classes('w-28')
        y = ui.number('y', value=0).classes('w-28')
        units = ui.select(['px', 'um'], value='px').classes('w-20')
        mag = ui.select([None, 2.5, 5, 10, 20, 40], value=20, label='mag').classes('w-24')
        target = ui.select({'active': 'active pane', 'all': 'all panes'}, value='active').classes('w-32')

    grid = (
        WissiViz([slide_ids[i % len(slide_ids)] for i in range(n.value)], on_object_click=show_object, on_draw=drawn)
        .classes('w-full')
        .style('height: calc(100vh - 140px)')
    )

    n.on_value_change(lambda e: grid.set_slides([slide_ids[i % len(slide_ids)] for i in range(e.value)]))
    sync.on_value_change(lambda e: grid.set_sync(e.value))
    tool.on_value_change(lambda e: grid.set_tool(e.value))

    with ui.row().classes('items-center gap-2'):
        ui.button(
            'Go',
            on_click=lambda: grid.go_to(
                target.value if target.value == 'all' else 'active',
                x.value,
                y.value,
                units=units.value,
                magnification=mag.value,
            ),
        )
        ui.button('Fit all', on_click=lambda: grid.fit('all'))

        async def where():
            views = await grid.get_views()
            ui.notify(
                ' | '.join(
                    f'{v["pane"]}: {v["x"]:.0f},{v["y"]:.0f} @{v.get("magnification", 0):.1f}x' for v in views if v
                )
            )

        ui.button('Where am I?', on_click=where)
    info.move()


ui.run(title='wissi-viz + NiceGUI', port=8080, reload=False, show=False)
