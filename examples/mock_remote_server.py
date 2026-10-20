"""Demo/verification server: the same slide served three ways — local file, HALOLink-style
remote (DeepZoom, overlap, JPEG, sync client) and PathCore-style remote (async client, padded PNG
tiles, coarsest-first level ids) — so you can compare them side by side.

    python examples/mock_remote_server.py tests-data/synthetic_he.svs [cells.geojson] --port 8020
"""
import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))
from mock_remote import MockHaloLink, MockPathCore  # noqa: E402

from wissi_viz import SlideRegistry, TileService, create_app  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('slide')
ap.add_argument('geojson', nargs='?')
ap.add_argument('--port', type=int, default=8020)
ap.add_argument('--cache-dir')
a = ap.parse_args()

reg = SlideRegistry(TileService(disk_dir=a.cache_dir))
reg.add(a.slide, slide_id='local', name='local file')
MockHaloLink(a.slide, geojson_path=a.geojson).register(reg, 'demo', name='HALOLink (mock)')
MockPathCore(a.slide).register(reg, 'demo', name='PathCore (mock)')
print('slides:', reg.ids())
uvicorn.run(create_app(reg), host='127.0.0.1', port=a.port, log_level='warning')
