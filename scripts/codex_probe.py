"""A highly multiplexed (CODEX / PhenoCycler, 39 markers) image in wissi-viz: does it open, how fast are its
channels, and what does the app show? Downloads the public cut-out (Domanskyi 2025, Zenodo 15757766, CC BY 4.0).

    uv run python scripts/codex_probe.py --out codex-probe --work codex-work

Prints the record's files (name, size, md5), the image's channels and timings; saves screenshots of the app.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import urllib.request
from pathlib import Path

RECORD = 'https://zenodo.org/api/records/15757766'


def notice(title: str, text: str) -> None:
    print(f'{title}: {text}', flush=True)
    if os.environ.get('GITHUB_ACTIONS'):
        print(f'::notice title={title}::{" ".join(str(text).split())[:4000]}', flush=True)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--out', default='codex-probe')
    p.add_argument('--work', default='codex-work')
    a = p.parse_args(argv)
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(RECORD, timeout=60).read().decode())
    lic = (meta.get('metadata') or {}).get('license', {}).get('id')
    files = [{'key': f['key'], 'size': f['size'], 'checksum': f.get('checksum'), 'url': f['links']['self']}
             for f in meta['files']]  # fmt: skip
    notice('codex record', f'licence {lic}; ' + '; '.join(f'{f["key"]} {f["size"]} {f["checksum"]} {f["url"]}'
                                                           for f in files))  # fmt: skip
    (out / 'record.json').write_text(json.dumps({'licence': lic, 'files': files, 'title': meta['metadata']['title'],
                                                 'creators': meta['metadata'].get('creators')}, indent=1))  # fmt: skip
    for f in files:
        dest = work / f['key']
        if not dest.exists():
            t0 = time.time()
            urllib.request.urlretrieve(f['url'], dest)
            print(f'{f["key"]}: {dest.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s', flush=True)

    from wissi_viz.session import Session
    from wissi_viz.sources import open_slide

    rec: dict = {}
    for name in ('if-cutout.ome.tif', 'he-cutout.ome.tif', 'mask-cutout.ome.tif'):
        t0 = time.time()
        try:
            src = open_slide(work / name)
        except Exception as e:  # noqa: BLE001
            notice(f'codex {name}', f'DOES NOT OPEN: {type(e).__name__}: {e}')
            rec[name] = {'error': str(e)}
            continue
        i = src.info.to_json()
        ch = [c.get('name') for c in (i.get('channels') or [])]
        r = {'open_s': round(time.time() - t0, 2), 'mode': i.get('mode'), 'size': [i.get('width'), i.get('height')],
             'mpp': i.get('mpp'), 'levels': len(i.get('levels') or []), 'dtype': i.get('dtype'), 'channels': ch,
             'tile': i.get('tile_size')}  # fmt: skip
        rec[name] = r
        notice(f'codex {name}', json.dumps({**r, 'channels': f'{len(ch)}: {", ".join(map(str, ch))}'}))
    (out / 'images.json').write_text(json.dumps(rec, indent=1))

    # through the server, as the app reads it: every channel's tile at two levels, and the histograms of all channels
    from starlette.testclient import TestClient

    s = Session(cache_dir=work / 'cache', remember=False)
    try:
        sid = s.open([str(work / 'if-cutout.ome.tif'), str(work / 'he-cutout.ome.tif')], trusted=True)['opened'][0]
        c = TestClient(s.app())
        base = f'/wissi-viz/slides/{sid}'
        info = c.get(f'{base}/info.json').json()
        n = len(info.get('channels') or [])
        top = len(info.get('levels', [])) - 1 if info.get('levels') else 0
        times = {}
        for level, label in ((top, 'overview level'), (0, 'full resolution')):
            ts = []
            for ch_i in range(n):
                t0 = time.time()
                r = c.get(f'{base}/tiles/{level}/0/0', params={'c': ch_i})
                ts.append(time.time() - t0)
                if r.status_code != 200:
                    notice('codex tile', f'channel {ch_i} level {level}: HTTP {r.status_code} {r.text[:200]}')
                    break
            times[label] = {'first_ms': round(ts[0] * 1000), 'median_ms': round(sorted(ts)[len(ts) // 2] * 1000),
                            'max_ms': round(max(ts) * 1000), 'all_channels_s': round(sum(ts), 2)}  # fmt: skip
        t0 = time.time()
        h = c.get(f'{base}/histogram')
        hist_s = round(time.time() - t0, 2)
        notice('codex through the server', f'{n} channels; tiles {json.dumps(times)}; histogram of all channels '
               f'{hist_s} s (HTTP {h.status_code}, {len(h.json().get("channels", [])) if h.status_code == 200 else 0} '
               f'channels); peak memory {rss_mb():.0f} MB; info keys {sorted(info)}')  # fmt: skip
        (out / 'server.json').write_text(json.dumps({'channels': n, 'tiles': times, 'histogram_s': hist_s,
                                                     'peak_mb': rss_mb(), 'info': info}, indent=1))  # fmt: skip
    finally:
        s.shutdown()

    # the demo itself, end to end (download, prepare, tissue + region metrics), then the app on its session
    try:
        from playwright.sync_api import sync_playwright

        from wissi_viz import desktop as D
        from wissi_viz import pack

        t0 = time.time()
        sess = pack.run_demo('codex', str(work / 'pack'), yes=True, open_app=False)
        demo_s = round(time.time() - t0)
        s = Session(cache_dir=work / 'cache2', remember=False)
        s.load_session(sess)
        srv = D._Server(s.app())
        url = srv.start()
        errors = []
        with sync_playwright() as pw:
            b = pw.chromium.launch(args=['--enable-unsafe-swiftshader'])
            pg = b.new_page(viewport={'width': 1600, 'height': 1000})
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.goto(url + '/app')
            g = '() => document.getElementById("grid")'
            pg.wait_for_function(f'() => ({g})().panes.length >= 2 && ({g})().panes.every(p => p.slide && p.slide.info)',
                                 timeout=120000)  # fmt: skip
            pg.wait_for_timeout(8000)
            pg.screenshot(path=str(out / 'app-1-opened.png'))
            pg.keyboard.press('i')  # the Image tab: channels, histograms
            pg.wait_for_timeout(8000)
            pg.screenshot(path=str(out / 'app-2-image-tab.png'))
            shown = pg.evaluate(f"""() => ({g})().panes.map(p => ({{name: p.slide.info.name, mode: p.slide.info.mode,
                mpp: p.slide.info.mpp, n: p.channels?.length || 0,
                visible: (p.channels || []).filter(c => c.visible).map(c => c.name)}}))""")
            # closer: 10× at the middle of what each pane opened on
            pg.evaluate(f"""() => {{ const e = ({g})(); for (const p of e.panes)
                e.goTo(p.index, p.view.cx, p.view.cy, {{magnification: 10, animate: false}}); }}""")
            pg.wait_for_timeout(9000)
            pg.screenshot(path=str(out / 'app-3-10x.png'))
            with pg.expect_download() as dl:
                pg.click('#vtools [data-v=snapgrid]')
            dl.value.save_as(str(out / 'figure.png'))
            b.close()
        srv.stop()
        s.shutdown()
        notice('codex demo in the app', f'demo ready in {demo_s} s; panes: {json.dumps(shown)}; '
               f'page errors: {errors[:5] or "none"}')  # fmt: skip
    except BaseException as e:  # noqa: BLE001
        notice('codex demo in the app', f'failed: {type(e).__name__}: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
