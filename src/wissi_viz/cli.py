"""``wissi-viz serve`` — standalone tile + annotation server with a demo page.

wissi-viz serve /data/slides --port 8010 --cache-dir /var/cache/wissi-viz
wissi-viz desktop slide.svs            # the viewer in its own window ("wissi-viz[desktop]")
wissi-viz info slide.svs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import desktop, pack
from ._compat import utf8_stdio
from .annio import convert
from .bench import fetch_samples, format_zoom_report, zoom_bench
from .bench import main as bench_main
from .pack import main as pack_main
from .registry import SlideRegistry
from .server import create_app
from .sources import open_slide
from .tiles import TileService


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (['--demo'], ['demo'], ['--benchmark'], ['benchmark']):
        _demo_or_benchmark(argv)
        return
    if argv[:1] == ['pack']:  # the public sample pack, step by step
        pack_main(argv[1:])
        return
    if argv[:1] == ['segment']:  # cells of a whole slide → GeoParquet
        from .segment import main as seg_main

        seg_main(argv[1:])
        return
    if argv[:1] == ['desktop']:  # the viewer in its own window (pywebview), see wissi_viz/desktop.py
        desktop.main(argv[1:])
        return
    if argv[:1] == [
        'loadtest'
    ]:  # multi-user load test (the standalone scripts/wissi_loadtest.py, bundled in the wheel)
        try:
            from . import loadtest as lt
        except ImportError:  # editable/source checkout: use the script itself
            sys.path.insert(
                0, str(Path(__file__).resolve().parents[2] / 'scripts')
            )  # importable by --procs children too
            import wissi_loadtest as lt
        lt.main(argv[1:])
        return
    ap = argparse.ArgumentParser(prog='wissi-viz')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve', help='serve slides from files/directories')
    s.add_argument('paths', nargs='+', help='slide files or directories (searched recursively)')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=8010)
    s.add_argument('--prefix', default='/wissi-viz')
    s.add_argument('--cache-dir', default=None, help='persistent tile cache directory')
    s.add_argument('--cache-gb', type=float, default=20, help='disk cache size (GB)')
    s.add_argument('--memory-mb', type=int, default=1024, help='in-memory tile cache (MB)')
    s.add_argument('--workers', type=int, default=None, help='tile decode threads')
    s.add_argument('--cors', default='*')
    s.add_argument(
        '--max-level-step',
        type=float,
        default=4.0,
        help='insert virtual levels where native ones are further apart (2 = leaner browsers for 4x-step SVS)',
    )
    s.add_argument(
        '--decoded-cache-mb', type=float, default=1024, help='decoded-array budget shared by all slides (MB)'
    )
    i = sub.add_parser('info', help='print slide metadata')
    i.add_argument('path')
    b = sub.add_parser('bench', help='benchmark opening, tile serving and in-browser panning')
    b.add_argument('slides', nargs='+')
    b.add_argument('--panes', default='1,4,9,16')
    b.add_argument('--server-only', action='store_true')
    b.add_argument('--browser-only', action='store_true')
    b.add_argument('--headed', action='store_true', help='visible browser window (uses your desktop GPU)')
    b.add_argument('--seconds', type=float, default=8, help='pan duration per run')
    b.add_argument('--speed', type=float, default=900, help='pan speed, screen px/s')
    b.add_argument('--magnification', type=float, default=20)
    b.add_argument('--concurrency', type=int, default=8)
    b.add_argument('--browser-args', default=None, help='extra Chromium flags, e.g. "--use-angle=vulkan"')
    b.add_argument('--viewport', default='1600x1000', help='browser viewport WxH')
    b.add_argument('--http2', action='store_true', help='serve over HTTPS/HTTP2 (hypercorn) like a production proxy')
    b.add_argument('--out', default=None, help='write JSON (+ .md table)')
    b.add_argument(
        '--zoom',
        action='store_true',
        help='zoom benchmark instead: continuous zoom in/out + jump zooms, each slide alone and all together',
    )
    b.add_argument(
        '--lod', default='legacy,default', help='--zoom: viewer LOD policies to compare (or JSON dicts, ";"-separated)'
    )
    b.add_argument('--zoom-rate', type=float, default=4.0, help='--zoom: zoom speed, × per second')
    c = sub.add_parser('convert', help='convert annotations: GeoParquet, GeoJSON/QuPath, HALO, Aperio, CSV, COCO, mask')
    c.add_argument('src')
    c.add_argument('dst')
    c.add_argument('--from', dest='src_fmt', default='auto', help='input format (default: by extension)')
    c.add_argument(
        '--to',
        dest='dst_fmt',
        default='auto',
        help='geoparquet | geojson | qupath | halo | aperio | csv | coco | mask (default: by extension)',
    )
    c.add_argument('--slide', default=None, help='the slide: mask size, µm/px and COCO image name')
    c.add_argument('--downsample', type=float, default=1.0, help='mask: level-0 px per mask px')
    c.add_argument('--by', default='feature', choices=['feature', 'class'], help='mask: label per feature or per class')
    c.add_argument('--wkt', action='store_true', help='csv: add a WKT geometry column')
    f = sub.add_parser('fetch-samples', help='download public whole-slide test images')
    f.add_argument('dest')
    f.add_argument('--set', default='standard', choices=['small', 'standard', 'full'])
    f.add_argument('--max-mb', type=float, default=2000)
    f.add_argument('--url', action='append', default=[], help='additional URL(s) to download')
    a = ap.parse_args(argv)

    if a.cmd == 'bench' and a.zoom:
        n = max(int(x) for x in a.panes.split(','))
        cfg = [(Path(p).name, [p], 1) for p in a.slides] + ([('all slides', list(a.slides), n)] if n > 1 else [])
        modes = a.lod.split(';') if ';' in a.lod else a.lod.split(',')
        rep_ = zoom_bench(
            cfg,
            modes=modes,
            rate=a.zoom_rate,
            magnification=max(a.magnification, 40),
            headed=a.headed,
            viewport=tuple(int(v) for v in a.viewport.lower().split('x')),
            http2=a.http2,
            browser_args=a.browser_args.split() if a.browser_args else None,
            save_to=a.out,
        )
        md = format_zoom_report(rep_)
        print('\n' + md)
        if a.out:
            Path(a.out).with_suffix('.md').write_text(md, encoding='utf-8')
        return
    if a.cmd == 'bench':
        bench_main(a)
        return
    if a.cmd == 'fetch-samples':
        for p in fetch_samples(a.dest, a.set, a.max_mb, a.url):
            print(p)
        return

    if a.cmd == 'convert':
        t0 = time.perf_counter()
        out = convert(
            a.src,
            a.dst,
            src_fmt=a.src_fmt,
            dst_fmt=a.dst_fmt,
            slide=a.slide,
            downsample=a.downsample,
            by=a.by,
            wkt=a.wkt,
        )
        print(f'{a.src} → {out}  ({out.stat().st_size / 2**20:.1f} MB, {time.perf_counter() - t0:.1f} s)')
        return

    if a.cmd == 'info':
        src = open_slide(a.path)
        print(json.dumps(src.info.to_json(), indent=2))
        return

    import uvicorn

    tiles = TileService(
        memory_bytes=a.memory_mb * 2**20, disk_dir=a.cache_dir, disk_bytes=int(a.cache_gb * 2**30), workers=a.workers
    )
    reg = SlideRegistry(tiles, max_level_step=a.max_level_step, decoded_cache_mb=a.decoded_cache_mb)
    for p in a.paths:
        p = Path(p)
        if p.is_dir():
            reg.add_directory(p)
        else:
            reg.add(p)
    if not reg.ids():
        sys.exit('no slides found')
    print(f'wissi-viz: {len(reg.ids())} slide(s) → http://{a.host}:{a.port}/')
    uvicorn.run(create_app(reg, prefix=a.prefix, cors=a.cors or None), host=a.host, port=a.port, log_level='warning')


def _demo_or_benchmark(argv: list[str]) -> None:
    """``wissi-viz --demo [quick|full]`` and ``wissi-viz --benchmark [quick|full]`` on the public sample pack."""
    bench = argv[0].lstrip('-') == 'benchmark'
    ap = argparse.ArgumentParser(prog=f'wissi-viz --{"benchmark" if bench else "demo"}')
    ap.add_argument(
        'set',
        nargs='?',
        default='quick',
        choices=['quick', 'full'],
        help='quick: 5 images (~1 GB); full: 32 images (~7 GB)',
    )
    ap.add_argument('--dir', default=None, help='where the images are kept (default: your user cache)')
    ap.add_argument('--yes', action='store_true', help="don't ask before downloading")
    if bench:
        ap.add_argument('--out', default='wissi-viz-benchmark', help='report path without extension (.md + .json)')
        ap.add_argument('--panes', default='1,4,16,32')
        ap.add_argument('--headed', action='store_true', help='visible browser: measures your real GPU')
        ap.add_argument('--seconds', type=float, default=6, help='pan duration per run')
    else:
        ap.add_argument(
            '--model',
            default='auto',
            choices=['auto', 'cellpose', 'classpose', 'classic'],
            help='cell segmentation (auto: Cellpose-SAM if installed, else classic)',
        )
        ap.add_argument('--browser', action='store_true', help='open in a browser window instead of the desktop app')
        ap.add_argument('--no-open', action='store_true', help='prepare everything, open nothing')
    a = ap.parse_args(argv[1:])
    if bench:
        pack.run_benchmark(a.set, a.dir, a.out, a.panes, a.yes, a.headed, a.seconds)
    else:
        pack.run_demo(a.set, a.dir, a.model, a.yes, open_app=not a.no_open, browser=a.browser)


if __name__ == '__main__':
    main()
