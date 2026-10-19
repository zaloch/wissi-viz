"""``wsiviz serve`` — standalone tile + annotation server with a demo page.

    wsiviz serve /data/slides --port 8010 --cache-dir /var/cache/wsiviz
    wsiviz info slide.svs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog='wsiviz')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve', help='serve slides from files/directories')
    s.add_argument('paths', nargs='+', help='slide files or directories (searched recursively)')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=8010)
    s.add_argument('--prefix', default='/wsiviz')
    s.add_argument('--cache-dir', default=None, help='persistent tile cache directory')
    s.add_argument('--cache-gb', type=float, default=20, help='disk cache size (GB)')
    s.add_argument('--memory-mb', type=int, default=1024, help='in-memory tile cache (MB)')
    s.add_argument('--workers', type=int, default=None, help='tile decode threads')
    s.add_argument('--cors', default='*')
    i = sub.add_parser('info', help='print slide metadata')
    i.add_argument('path')
    a = ap.parse_args(argv)

    if a.cmd == 'info':
        from .sources import open_slide
        src = open_slide(a.path)
        print(json.dumps(src.info.to_json(), indent=2))
        return

    import uvicorn

    from .registry import SlideRegistry
    from .server import create_app
    from .tiles import TileService

    tiles = TileService(memory_bytes=a.memory_mb * 2**20, disk_dir=a.cache_dir, disk_bytes=int(a.cache_gb * 2**30),
                        workers=a.workers)
    reg = SlideRegistry(tiles)
    for p in a.paths:
        p = Path(p)
        if p.is_dir():
            reg.add_directory(p)
        else:
            reg.add(p)
    if not reg.ids():
        sys.exit('no slides found')
    print(f'wsiviz: {len(reg.ids())} slide(s) → http://{a.host}:{a.port}/')
    uvicorn.run(create_app(reg, prefix=a.prefix, cors=a.cors or None), host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
