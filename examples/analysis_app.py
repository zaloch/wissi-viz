"""wissi-viz analysis workbench (NiceGUI): viewer + spatial analyses side by side.

    python examples/analysis_app.py SLIDE CELLS [ROIS] [--mpp 0.5] [--port 8090]

SLIDE  any slide wissi-viz opens (SVS, CZI, OME-TIFF, …)
CELLS  object export: QuPath TSV/CSV/GeoJSON, HALO CSV, CellProfiler, Parquet, .h5ad, label mask
ROIS   optional ROI GeoJSON / HALO .annotations / Aperio XML (level-0 px)

Every analysis overlays its result on the slide (left pane) and shows its table and plot here.
In wissi, copy the handlers: each one is ~5 lines around a ``wissi_viz.spatial`` call.
"""
from __future__ import annotations

import argparse

from nicegui import run, ui

from wissi_viz import AnnotationLayer, plots
from wissi_viz import spatial as sp
from wissi_viz.nicegui import WissiViz, configure
from wissi_viz.objects import read_objects

ap = argparse.ArgumentParser()
ap.add_argument('slide')
ap.add_argument('cells')
ap.add_argument('rois', nargs='?')
ap.add_argument('--mpp', type=float, default=None, help='µm per pixel (default: from the slide)')
ap.add_argument('--port', type=int, default=8090)
args = ap.parse_args()

registry = configure()
SID = registry.add(args.slide, slide_id='slide')
_, pyr = registry.get(SID)
MPP = args.mpp or (pyr.source.info.mpp or (None,))[0]
OBJS = read_objects(args.cells, mpp=MPP)
ROIS = AnnotationLayer.load(args.rois) if args.rois else None
CLASSES = OBJS.class_names()
STATE: dict = {'tissue': None}


def tissue():
    if STATE['tissue'] is None:
        STATE['tissue'], layer = sp.tissue_mask(pyr.source)
        registry.add_annotations(SID, layer)
    return STATE['tissue']


for L in registry.layers(SID).values():  # sidecar annotations found next to the slide: start hidden
    L.visible = False
    L.touch()
if ROIS is not None:
    ROIS.name, ROIS.line_width, ROIS.fill_opacity, ROIS.z = 'ROIs', 2.5, 0.0, 10
    registry.add_annotations(SID, ROIS)
cells_layer = OBJS.to_layer(name='Cells', point_size=4)
cells_layer.visible = False
registry.add_annotations(SID, cells_layer)


@ui.page('/')
def page():
    ui.add_css('.q-tab{min-height:34px;padding:0 10px} .q-tab__label{font-size:12px}')
    with ui.row().classes('w-full no-wrap gap-2').style('height: calc(100vh - 24px)'):
        with ui.column().classes('h-full').style('flex: 1.35; min-width: 0'):
            grid = WissiViz([SID, SID], layout='1x2', sync='all').classes('w-full h-full')
            grid.set_layer_visible(1, 'Cells', False)
        with ui.column().classes('h-full overflow-auto').style('flex: 1; min-width: 380px'):
            ui.label(f'{len(OBJS):,} objects · {len(CLASSES)} classes · {MPP} µm/px').classes('text-xs opacity-70')
            with ui.tabs().classes('w-full') as tabs:
                names = ['Composition', 'Nearest', 'Proximity', 'Margin', 'Density', 'Ripley', 'Co-location',
                         'Niches', 'Contacts', 'IHC', 'Stain']
                tab = {n: ui.tab(n) for n in names}
            out_table = None

            def show_table(t, container):
                rows = plots.table_rows(t)
                with container:
                    ui.table(rows=rows, columns=[{'name': k, 'label': k, 'field': k, 'sortable': True} for k in rows[0]] if rows else [],
                             ).props('dense flat').classes('w-full text-xs')

            def classes_select(label, value):
                return ui.select(CLASSES, value=value, label=label).classes('w-40')

            with ui.tab_panels(tabs, value=tab['Composition']).classes('w-full'):
                # ------------------------------------------------------------ composition
                with ui.tab_panel(tab['Composition']):
                    box = ui.column().classes('w-full')
                    kind = ui.toggle(['percent', 'density', 'sunburst'], value='percent')

                    async def composition():
                        box.clear()
                        if ROIS is None:
                            ui.notify('load ROIs to compare regions'); return
                        whole = await run.io_bound(tissue)
                        res = await run.io_bound(sp.roi_composition, OBJS, ROIS, None, whole, 'Tissue')
                        grid.show(SID, res.layer(name='ROI choropleth', value=f'density {CLASSES[0]} /mm2'))
                        with box:
                            ui.echart(res.plot(kind.value)).classes('w-full h-80')
                        show_table(res.table[['roi', 'area_mm2', 'area % of whole', 'n_objects', 'density /mm2']
                                             + [f'% {c}' for c in CLASSES]], box)
                    ui.button('Run', on_click=composition)
                # ------------------------------------------------------------ nearest neighbour
                with ui.tab_panel(tab['Nearest']):
                    with ui.row():
                        a, b = classes_select('from', 'Tumor' if 'Tumor' in CLASSES else CLASSES[0]), classes_select('to nearest', CLASSES[min(2, len(CLASSES) - 1)])
                    box2 = ui.column().classes('w-full')

                    async def nearest():
                        box2.clear()
                        r = await run.io_bound(sp.nearest_neighbors, OBJS, a.value, b.value)
                        grid.show(SID, r, name='Nearest neighbour')
                        with box2:
                            ui.echart(r.plot(max_um=200)).classes('w-full h-72')
                        show_table([r.summary()], box2)
                    ui.button('Run', on_click=nearest)
                # ------------------------------------------------------------ proximity
                with ui.tab_panel(tab['Proximity']):
                    with ui.row():
                        pa, pb = classes_select('around', 'Tumor' if 'Tumor' in CLASSES else CLASSES[0]), classes_select('count', CLASSES[min(2, len(CLASSES) - 1)])
                        rad = ui.number('radius µm', value=30, min=1).classes('w-28')
                    box3 = ui.column().classes('w-full')

                    async def prox():
                        box3.clear()
                        r = await run.io_bound(sp.proximity, OBJS, pa.value, pb.value, float(rad.value))
                        grid.show(SID, r, name='Proximity')
                        with box3:
                            ui.echart(r.plot()).classes('w-full h-72')
                        show_table([r.summary()], box3)
                    ui.button('Run', on_click=prox)
                # ------------------------------------------------------------ margin
                with ui.tab_panel(tab['Margin']):
                    roi_names = [n for n, _ in sp.roi_geometries(ROIS)] if ROIS is not None else []
                    with ui.row():
                        mroi = ui.select(roi_names, value=roi_names[0] if roi_names else None, label='boundary').classes('w-40')
                        mw = ui.number('band µm', value=100, min=5).classes('w-24')
                        mc = ui.select(CLASSES, value=CLASSES[:3], multiple=True, label='classes').classes('w-64')
                    box4 = ui.column().classes('w-full')

                    async def margin():
                        box4.clear()
                        g = dict(sp.roi_geometries(ROIS, group=True))[mroi.value]
                        w = float(mw.value)
                        tis = await run.io_bound(tissue)
                        br = await run.io_bound(lambda: sp.margin_bands(g, (-2 * w, -w, 0, w, 2 * w), objs=OBJS, within=tis, classes=mc.value))
                        pr = await run.io_bound(lambda: sp.infiltration_profile(OBJS, g, mc.value, (-4 * w, 6 * w), w / 4, within=tis))
                        grid.show(SID, br, name='Margin bands')
                        with box4:
                            ui.echart(pr.plot()).classes('w-full h-72')
                            ui.echart(br.plot()).classes('w-full h-64')
                        show_table(br.table, box4)
                    ui.button('Run', on_click=margin)
                # ------------------------------------------------------------ density / hotspots
                with ui.tab_panel(tab['Density']):
                    with ui.row():
                        dc = ui.select(CLASSES, value=CLASSES[min(2, len(CLASSES) - 1)], label='class').classes('w-40')
                        dd = ui.select(['(per mm²)', *CLASSES, 'all cells'], value='(per mm²)', label='ratio to').classes('w-40')
                        ds = ui.number('cell µm', value=100, min=10).classes('w-24')
                    box5 = ui.column().classes('w-full')

                    async def density():
                        box5.clear()
                        den = None if dd.value == '(per mm²)' else (None if dd.value == 'all cells' else dd.value)
                        tis = await run.io_bound(tissue)
                        g = await run.io_bound(lambda: sp.density_grid(OBJS, dc.value, float(ds.value), denominator=den if dd.value != 'all cells' else (lambda m: m)(OBJS.mask(None)),
                                                                      within=tis, hotspots=True))
                        grid.show(SID, g.layer(), name='Density')
                        grid.show(SID, g.hotspot_layer(), name='Hotspots', replace=False)  # on top of the density map
                        with box5:
                            ui.echart(g.plot()).classes('w-full h-64')
                            ui.label(f'{(g.gi_z > 1.96).sum()} hot / {(g.gi_z < -1.96).sum()} cold cells of {len(g.cells)} (Getis-Ord Gi*, |z|>1.96)').classes('text-xs')
                    ui.button('Run', on_click=density)
                # ------------------------------------------------------------ Ripley
                with ui.tab_panel(tab['Ripley']):
                    with ui.row():
                        ra, rb = classes_select('A', CLASSES[min(2, len(CLASSES) - 1)]), ui.select(['(univariate)', *CLASSES], value='(univariate)', label='B').classes('w-40')
                    box6 = ui.column().classes('w-full')

                    async def ripley():
                        box6.clear()
                        tis = await run.io_bound(tissue)
                        r = await run.io_bound(lambda: sp.ripley(OBJS, ra.value, None if rb.value == '(univariate)' else rb.value, area=tis, n_sim=19))
                        with box6:
                            ui.echart(r.plot('H')).classes('w-full h-72')
                            ui.echart(r.plot('g')).classes('w-full h-60')
                    ui.button('Run', on_click=ripley)
                # ------------------------------------------------------------ co-location
                with ui.tab_panel(tab['Co-location']):
                    er = ui.number('radius µm', value=25, min=1).classes('w-28')
                    box7 = ui.column().classes('w-full')

                    async def enrich():
                        box7.clear()
                        e = await run.io_bound(lambda: sp.neighborhood_enrichment(OBJS, float(er.value), n_perm=99))
                        with box7:
                            ui.echart(e.plot()).classes('w-full h-96')
                    ui.button('Run', on_click=enrich)
                # ------------------------------------------------------------ cellular neighbourhoods
                with ui.tab_panel(tab['Niches']):
                    with ui.row():
                        nk = ui.number('k neighbours', value=12, min=3).classes('w-28')
                        nn_ = ui.number('# niches', value=6, min=2, max=16).classes('w-24')
                    box8 = ui.column().classes('w-full')

                    async def niches():
                        box8.clear()
                        cn = await run.io_bound(lambda: sp.cellular_neighborhoods(OBJS, int(nk.value), int(nn_.value)))
                        grid.show(SID, cn, name='Cellular neighborhoods')
                        with box8:
                            ui.echart(cn.plot()).classes('w-full h-80')
                    ui.button('Run', on_click=niches)
                # ------------------------------------------------------------ contact graph
                with ui.tab_panel(tab['Contacts']):
                    ce = ui.number('max edge µm', value=20, min=2).classes('w-28')
                    box9 = ui.column().classes('w-full')

                    async def contacts():
                        box9.clear()
                        gr = await run.io_bound(lambda: sp.delaunay_graph(OBJS, float(ce.value)))
                        grid.show(SID, gr, name='Contact graph')
                        with box9:
                            ui.echart(gr.plot()).classes('w-full h-96')
                    ui.button('Run', on_click=contacts)
                # ------------------------------------------------------------ IHC scoring
                with ui.tab_panel(tab['IHC']):
                    num_cols = [c for c, v in OBJS.columns.items() if getattr(v, 'dtype', None) is not None and v.dtype.kind == 'f']
                    with ui.row():
                        hc = ui.select(num_cols, value=next((c for c in num_cols if 'DAB' in c or 'OD' in c), num_cols[0] if num_cols else None),
                                       label='intensity').classes('w-56')
                        hcls = ui.select(['(all)', *CLASSES], value='Tumor' if 'Tumor' in CLASSES else '(all)', label='within').classes('w-32')
                        t1, t2, t3 = (ui.number(f't{i}', value=v).classes('w-20') for i, v in ((1, 0.15), (2, 0.3), (3, 0.5)))
                    box10 = ui.column().classes('w-full')

                    async def ihc():
                        box10.clear()
                        cls = None if hcls.value == '(all)' else hcls.value
                        th = (float(t1.value), float(t2.value), float(t3.value))
                        h = await run.io_bound(lambda: sp.h_score(OBJS, hc.value, th, rois=ROIS, cls=cls))
                        sub = OBJS.subset(OBJS.mask(cls))
                        sub.intensity_bins(hc.value, th, inplace=True)
                        grid.show(SID, sub.to_layer(name='IHC 0/1+/2+/3+', classes={'0': '#3d7bff', '1+': '#ffd400', '2+': '#ff8a00', '3+': '#d7191c'}, point_size=5))
                        with box10:
                            ui.echart(plots.stacked_bars(list(h['roi']), {k: list(h[k]) for k in ('% 0', '% 1+', '% 2+', '% 3+')},
                                                         'Intensity bins per ROI', percent=True)).classes('w-full h-60')
                        show_table(h, box10)
                    ui.button('Run', on_click=ihc)
                # ------------------------------------------------------------ stain deconvolution (GPU)
                with ui.tab_panel(tab['Stain']):
                    ui.label('Right pane: colour deconvolution on the GPU (brightfield).').classes('text-xs')
                    sm = ui.toggle(['off', 'hematoxylin', 'eosin', 'dab', 'fluorescence'], value='off',
                                   on_change=lambda e: grid.set_stain(1, e.value))
                    ui.button('Estimate stain vectors for this slide (Macenko)', on_click=lambda: grid.auto_stain(1, sm.value))
            _ = out_table


ui.run(title='wissi-viz analysis', port=args.port, reload=False, show=False)
