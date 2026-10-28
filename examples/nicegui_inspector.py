"""NiceGUI: the wissi-viz inspector two ways.

    python examples/nicegui_inspector.py SLIDE CELLS [REGIONS] [--port 8080]

``/``        the whole app (slide list, tool bar, Layers / Analyze / Measure / Draw) embedded with ``WissiApp``.
``/custom``  your own panel next to a ``WissiViz`` element, built from the same pieces in Python: layer switches
             that pulse the layer on hover, a summary table, an analysis with its chart, the object under a
             right-click, and the hovered object's class.

Needs ``wissi-viz[nicegui,analysis]``.
"""

from __future__ import annotations

import argparse

from nicegui import ui

from wissi_viz import analyses
from wissi_viz.nicegui import WissiApp, WissiViz, configure, mount_app

ap = argparse.ArgumentParser()
ap.add_argument('slide')
ap.add_argument('cells')
ap.add_argument('regions', nargs='?')
ap.add_argument('--port', type=int, default=8080)
args = ap.parse_args()

registry = configure()  # tiles and overlays under /wissi-viz on NiceGUI's own server
session = mount_app('/wissi-app')  # the app page, sharing that registry (open_dirs=[...] lets users browse)
SLIDE = session.open([args.slide], trusted=True)['opened'][0]
CELLS = session.add_layer(SLIDE, args.cells)  # cell tables (CSV, Parquet, AnnData…) or annotation files
if args.regions:
    session.add_layer(SLIDE, args.regions)


@ui.page('/')
def full_app():
    ui.query('.nicegui-content').classes('p-0 gap-0')  # edge to edge
    WissiApp([SLIDE], tab='layers').style('height: 100vh')


@ui.page('/custom')
def custom():
    ui.query('.nicegui-content').classes('p-0 gap-0')
    with ui.row().classes('w-full no-wrap gap-0').style('height: 100vh'):
        grid = WissiViz(
            [SLIDE],
            controls=False,
            on_object_click=lambda e: picked.set_text(f'{e.get("class")} #{e.get("index")} · {e.get("layer")}'),
            on_object_hover=lambda e: hover.set_text(f'hover: {e["class"]} ({e["layer"]})' if e else ''),
            on_context=lambda e: context.set_text(
                f'right-click at {e["x"]:.0f}, {e["y"]:.0f}'
                + (f' on {e["object"]["layer"]}' if e.get('object') else '')
            ),
        ).style('flex: 1; height: 100vh')
        with ui.column().classes('p-4 gap-2').style('width: 400px; height: 100vh; overflow: auto'):
            ui.label('Layers').classes('text-lg font-semibold')
            for name in registry.layers(SLIDE):
                row = ui.row().classes('items-center gap-2 layer-row')
                with row:
                    ui.checkbox(name, value=True, on_change=lambda e, n=name: grid.set_layer_visible(0, n, e.value))
                row.on('mouseenter', lambda _, n=name: grid.flash_layer(0, n))
            hover, picked, context = ui.label(), ui.label(), ui.label()

            ui.label(f'Summary of {CELLS}').classes('text-lg font-semibold mt-2')
            table = ui.table(
                columns=[{'name': k, 'label': k, 'field': k} for k in ('class', 'n', '%')], rows=[]
            ).classes('w-full')

            async def summarise():
                sm = await grid.summary(SLIDE, CELLS)
                table.rows = [{'class': c['name'], 'n': c['n'], '%': c['pct']} for c in sm['classes']]
                density.set_text(f'{sm["per_mm2"]:,.0f} cells / mm² of tissue' if sm.get('per_mm2') else '')

            density = ui.label()
            ui.button('Summarise', on_click=summarise)

            ui.label('Density & hotspots').classes('text-lg font-semibold mt-2')
            chart = ui.echart({}).classes('w-full').style('height: 220px')
            result = ui.label()

            async def density_run():
                out = await grid.run_analysis(SLIDE, 'density', CELLS, cls='all', cell_um=100)
                result.set_text(' · '.join(f'{k["label"]} {k["value"]}' for k in out['kpis']))
                chart.options.clear()
                chart.options.update(analyses.echarts_option(out['chart']))
                chart.update()

            ui.button('Run', on_click=density_run).props('id=run-density')


ui.run(port=args.port, show=False, reload=False, title='wissi-viz in NiceGUI')
