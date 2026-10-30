"""How SAM 3's int8 copies score on this CPU against float32: the current recipe (int8 weights), with 7-bit
``reduce_range`` (no saturation on x86 without VNNI), and uint8 weights. Prints ``PROBE`` lines.

    python scripts/sam3_quant_probe.py --work /tmp/sam3probe
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

CASES = [('chelsea', 'cat'), ('coffee', 'cup'), ('astronaut', 'person'), ('immunohistochemistry', 'nucleus')]


def cpu() -> str:
    try:
        info = Path('/proc/cpuinfo').read_text()
    except OSError:
        return 'unknown'
    model = next((ln.split(':', 1)[1].strip() for ln in info.splitlines() if ln.startswith('model name')), '?')
    flags = next((ln.split(':', 1)[1].split() for ln in info.splitlines() if ln.startswith('flags')), [])
    return f'{model} · ' + ' '.join(f for f in ('avx2', 'avx512f', 'avx512_vnni', 'avx_vnni', 'amx_int8') if f in flags)


def score(folder: Path) -> list[str]:
    from skimage import data

    from wissi_viz import sam3

    m = sam3.Sam3(folder, 'cpu')
    out = []
    for pic, word in CASES:
        img = getattr(data, pic)()
        img = np.stack([img] * 3, -1) if img.ndim == 2 else img[..., :3]
        t0 = time.perf_counter()
        _, s, _, _ = m.detect(np.ascontiguousarray(img), [word], min_score=0.1)
        out.append(
            f'{word}: n={int((s >= 0.5).sum())} max={float(s.max()) if len(s) else 0:.2f} ({time.perf_counter() - t0:.0f}s)'
        )
    return out


def main() -> None:
    from huggingface_hub import hf_hub_download

    from wissi_viz import sam3

    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='/tmp/sam3probe')
    a = ap.parse_args()
    w = Path(a.work)
    fp32 = w / 'fp32'
    fp32.mkdir(parents=True, exist_ok=True)
    print('PROBE cpu:', cpu(), flush=True)
    if not (fp32 / 'sam3_decoder.onnx').exists():
        z = hf_hub_download(sam3.HF_REPO, sam3.HF_FILE, cache_dir=str(w / 'hf'))
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                base = Path(info.filename).name
                if info.is_dir() or not base or base.startswith('.'):
                    continue
                with zf.open(info) as r, open(fp32 / base, 'wb') as f:
                    shutil.copyfileobj(r, f, 1 << 24)
        shutil.rmtree(w / 'hf', ignore_errors=True)
    print('PROBE float32:', ' · '.join(score(fp32)), flush=True)
    for label, kw in (
        ('int8 (now)', {}),
        ('int8 reduce_range', {'reduce_range': True}),
        ('uint8', {'weights8': 'uint8'}),
    ):
        d = w / label.replace(' ', '_').replace('(', '').replace(')', '')
        t0 = time.perf_counter()
        sam3.quantize(fp32, d, **kw)
        print(f'PROBE {label} (quantized in {time.perf_counter() - t0:.0f}s):', ' · '.join(score(d)), flush=True)
        shutil.rmtree(d)


if __name__ == '__main__':
    main()
