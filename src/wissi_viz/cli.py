"""The ``wissi-viz`` command.

wissi-viz --demo                       # 5 public slides, cells segmented, opened in the app
wissi-viz desktop slide.svs            # the viewer in its own window ("wissi-viz[desktop]")
wissi-viz serve /data/slides --port 8010 --cache-dir /var/cache/wissi-viz
wissi-viz serve --allow-open /data/slides --key s3cret     # visitors open slides from that folder
wissi-viz serve --connect halolink.json                    # images from a tile API
wissi-viz info slide.svs
wissi-viz --demo --system-certs        # on networks that inspect HTTPS (corporate): trust this computer's certificates
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import time
from pathlib import Path

from . import certs, desktop, pack
from ._compat import utf8_stdio
from .annio import convert
from .bench import fetch_samples, format_zoom_report, zoom_bench
from .bench import main as bench_main
from .pack import main as pack_main
from .registry import SlideRegistry
from .session import Session, add_connect_args, apply_connects, create_session_app
from .sources import open_slide
from .tiles import TileService


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    argv = certs.take_flag(list(sys.argv[1:] if argv is None else argv))
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
    ap = argparse.ArgumentParser(
        prog='wissi-viz',
        description='GPU whole-slide image viewer: desktop app, web server, sample pack and tools.',
        epilog='Run "wissi-viz <command> --help" for the options of each command.',
    )
    certs.add_arg(ap)  # works with every command
    sub = ap.add_subparsers(dest='cmd', required=True, metavar='command')
    # handled above, before argparse; listed here so --help shows them
    for name, text in (
        ('--demo', 'download the public sample pack, segment cells and open it all (quick | full)'),
        ('--benchmark', 'benchmark panning, SAM, metrics and formats on the sample pack (quick | full)'),
        ('desktop', 'open slides, cell tables and annotations in the app window'),
        ('pack', 'the sample pack step by step: status, download, prepare, segment, analyze, open'),
        ('segment', 'segment the cells of a slide into GeoParquet (Cellpose-SAM, Classpose or classic)'),
        ('loadtest', 'multi-user load test against a running server'),
    ):
        sub.add_parser(name, help=text, add_help=False)
    s = sub.add_parser('serve', help='serve slides to browsers (the app page at /)')
    s.add_argument('paths', nargs='*', help='slide files or directories (searched recursively)')
    s.add_argument(
        '--allow-open',
        action='append',
        default=[],
        metavar='DIR',
        help='let visitors open slides and add cell / annotation files from this folder (repeatable)',
    )
    s.add_argument(
        '--allow-connect',
        action='store_true',
        help='let visitors connect the server to tile APIs (it fetches from them)',
    )
    s.add_argument(
        '--key',
        default=os.environ.get('WISSI_VIZ_KEY'),
        help='password visitors type before opening or connecting (default: $WISSI_VIZ_KEY)',
    )
    add_connect_args(s)
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
    for d in a.allow_open:
        if not Path(d).expanduser().is_dir():
            sys.exit(f'--allow-open: not a folder: {d}')
    session = Session(a.cache_dir, registry=reg, roots=a.allow_open, connect=a.allow_connect, remember=False)
    for e in session.open(a.paths, trusted=True, eager=False)['errors'] if a.paths else []:
        print(f'wissi-viz: {e}', file=sys.stderr)
    apply_connects(session, a, log=lambda m: print(m, file=sys.stderr))
    if not reg.ids() and not (a.allow_open or a.allow_connect):
        sys.exit('no slides found (give slide paths, --connect, or --allow-open)')
    if (a.allow_open or a.allow_connect) and not a.key and not _loopback(a.host):
        print(
            f'wissi-viz: warning: anyone who can reach {a.host}:{a.port} may open files / connect; set --key',
            file=sys.stderr,
        )
    what = (
        [f'{len(reg.ids())} slide(s)']
        + [f'open from {d}' for d in a.allow_open]
        + (['connect'] if a.allow_connect else [])
    )
    print(f'wissi-viz: {", ".join(what)} → http://{a.host}:{a.port}/')
    app = create_session_app(session, mode='server', prefix=a.prefix, cors=a.cors or None, key=a.key)
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning')


def _loopback(host: str) -> bool:
    if host in ('localhost', ''):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _demo_or_benchmark(argv: list[str]) -> None:
    """``wissi-viz --demo [quick|full]`` and ``wissi-viz --benchmark [quick|full]`` on the public sample pack."""
    bench = argv[0].lstrip('-') == 'benchmark'
    ap = argparse.ArgumentParser(prog=f'wissi-viz --{"benchmark" if bench else "demo"}')
    ap.add_argument(
        'set',
        nargs='?',
        default=None,
        choices=['quick', 'full'],
        help='quick: 5 images (~1.2 GB); full: 32 images (~7 GB). Asked when not given (quick without a terminal)',
    )
    ap.add_argument('--dir', default=None, help='where the images are kept (default: your user cache)')
    ap.add_argument('--yes', action='store_true', help="don't ask before downloading")
    certs.add_arg(ap)
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
    if a.set is None:
        a.set = _ask_set() if sys.stdin and sys.stdin.isatty() and not a.yes else 'quick'
    if bench:
        pack.run_benchmark(a.set, a.dir, a.out, a.panes, a.yes, a.headed, a.seconds)
    else:
        pack.run_demo(a.set, a.dir, a.model, a.yes, open_app=not a.no_open, browser=a.browser)


def _ask_set() -> str:
    print('Which sample set?')
    print('  1  quick:  5 images, ~1.2 GB  (3 H&E, 2 multiplex IF)')
    print('  2  full:  32 images, ~7 GB    (26 H&E: OpenSlide, CAMELYON16 lymph nodes, 18 TCGA tumour types; 6 IF)')
    ans = input('[1/2, default 1] ').strip().lower()
    return 'full' if ans in ('2', 'f', 'full') else 'quick'


if __name__ == '__main__':
    main()
