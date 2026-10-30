"""What is in the rat DRG record on Zenodo (10.5281/zenodo.6546069): files, the annotation archive's layout, and
one image's size, channels and pixel size. Prints ``DRG`` lines."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REC = 'https://zenodo.org/api/records/6546069'


def main() -> None:
    w = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/drg')
    w.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(REC, timeout=60).read().decode())
    md = meta.get('metadata') or {}
    print('DRG licence:', (md.get('license') or {}).get('id'), '· title:', md.get('title'), flush=True)
    files = meta.get('files') or []
    for f in files:
        print(f'DRG file: {f.get("key")} {int(f.get("size", 0)) / 2**20:.0f} MB', flush=True)
    rar = next((f for f in files if str(f.get('key', '')).lower().endswith('.rar')), None)
    for f in [x for x in (rar,) if x]:
        dest = w / f['key']
        if not dest.exists():
            urllib.request.urlretrieve(f['links']['self'], dest)
        tool = shutil.which('7z') or shutil.which('7zz') or shutil.which('bsdtar')
        if dest.suffix.lower() == '.zip':
            import zipfile

            names = zipfile.ZipFile(dest).namelist()
        else:
            cmd = [tool, 'l', '-ba', str(dest)] if 'bsdtar' not in tool else [tool, '-tf', str(dest)]
            names = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
        print(f'DRG {f["key"]}: {len(names)} entries', flush=True)
        dirs: dict = {}
        for n in names:
            n = n.strip().split()[-1] if n.strip() else ''
            dirs.setdefault(str(Path(n).parent), []).append(Path(n).name)
        for k, v in sorted(dirs.items()):
            print(f'DRG   {k}: {len(v)} · {", ".join(sorted(v)[:4])}', flush=True)
    if rar:
        out = w / 'gt'
        out.mkdir(exist_ok=True)
        tool = shutil.which('7z') or shutil.which('7zz')
        if tool:
            subprocess.run([tool, 'x', '-y', f'-o{out}', str(w / rar['key'])], capture_output=True)
        else:
            subprocess.run(['bsdtar', '-xf', str(w / rar['key']), '-C', str(out)])
        import numpy as np
        import tifffile
        from PIL import Image

        shown, seen_dirs = 0, set()
        for p in sorted(out.rglob('*')):
            if p.suffix.lower() in ('.tif', '.tiff', '.png') and p.parent not in seen_dirs and shown < 16:
                seen_dirs.add(p.parent)
                try:
                    if p.suffix.lower() == '.png':
                        a = np.asarray(Image.open(p))
                        extra = ''
                    else:
                        with tifffile.TiffFile(p) as tf:
                            a = tf.asarray()
                            pg = tf.pages[0]
                            xr = pg.tags.get('XResolution')
                            unit = pg.tags.get('ResolutionUnit')
                            extra = f' res={xr.value if xr else None} unit={unit.value if unit else None} ij={tf.imagej_metadata}'
                    print(
                        f'DRG img {p.relative_to(out)} shape={a.shape} dtype={a.dtype} max={a.max()} '
                        f'unique={len(np.unique(a)) if a.size < 5e7 else "?"}{extra}',
                        flush=True,
                    )
                    shown += 1
                except Exception as e:  # noqa: BLE001
                    print(f'DRG img {p}: {e}', flush=True)


if __name__ == '__main__':
    main()
