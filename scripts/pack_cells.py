"""One shard of the full sample pack: download, prepare and segment its images (classic, as the demo does without
Cellpose). Run by .github/workflows/pack-cells.yml, which publishes the cells as the release ``pack32-results-v1``.

    python scripts/pack_cells.py --shard 0 --of 4 --dir /mnt/pack
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main() -> None:
    from wissi_viz import pack

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--dir', default=None)
    a = ap.parse_args()
    sel = pack.items('full')[a.shard :: a.of]
    d = pack.pack_dir(a.dir)
    print('images:', [x['id'] for x in sel], flush=True)
    t0 = time.time()
    pack.download(sel, d)
    pack.prepare(sel, d)
    failed = []
    for x in sel:  # one at a time: a slide that fails is reported, the others still get their cells
        try:
            pack.segment([x], d, model='auto')
        except Exception as e:  # noqa: BLE001  (reported as an annotation; the other slides go on)
            failed.append(x['id'])
            tail = ' | '.join(traceback.format_exc().strip().splitlines()[-4:])[:1500]
            print(f'::error::{x["id"]}: {type(e).__name__}: {str(e)[:500]} :: {tail}', flush=True)
    print(
        f'::notice::shard {a.shard}: {len(sel) - len(failed)} of {len(sel)} images in {(time.time() - t0) / 60:.0f} min'
    )


if __name__ == '__main__':  # worker processes (spawn) re-import this file: they must not run it
    main()
