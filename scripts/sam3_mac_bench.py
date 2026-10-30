"""Is SAM 3 faster on this Mac's GPU / Neural Engine (Core ML) than on its CPU, and does it find the same things?

    uv run python scripts/sam3_mac_bench.py            # after accepting the SAM License in the app

Downloads the float32 image step once (~3.3 GB download, ~1.7 GB kept) if needed. For each setting it prints how long
the first picture took (Core ML compiles its model then; later launches reuse it), the time per picture after that,
and what SAM 3 found on four test pictures, against the CPU (int8) result. Then a verdict: use
``WISSI_SAM3_DEVICE=coreml`` only if it is clearly faster and finds the same.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

CASES = [('chelsea', 'cat'), ('coffee', 'cup'), ('astronaut', 'person'), ('immunohistochemistry', 'nucleus')]


def pictures():
    from skimage import data

    for pic, word in CASES:
        img = getattr(data, pic)()
        img = np.stack([img] * 3, -1) if img.ndim == 2 else img[..., :3]
        yield pic, word, np.ascontiguousarray(img)


def run(label: str, device: str, units: str | None = None) -> dict:
    from wissi_viz import sam3

    if units:
        os.environ['WISSI_SAM3_COREML_UNITS'] = units
    t0 = time.perf_counter()
    m = sam3.Sam3(sam3.folder(), device)
    load = time.perf_counter() - t0
    res, first, times = {}, None, []
    for _pic, word, img in pictures():
        t1 = time.perf_counter()
        _b, s, _, masks = m.detect(img, [word], min_score=0.3)
        dt = time.perf_counter() - t1
        first = dt if first is None else first
        times.append(dt)
        res[word] = (s, masks)
    t2 = time.perf_counter()
    m.image(next(pictures())[2])  # the image step alone, warm
    enc = time.perf_counter() - t2
    steady = float(np.median(times[1:]))
    print(f'{label:28s} load {load:6.1f}s · first picture {first:6.1f}s · then {steady:5.1f}s per picture '
          f'(image step {enc:4.1f}s) · ' + ' · '.join(f'{w}: {int((s >= 0.5).sum())}' for w, (s, _) in res.items()),
          flush=True)  # fmt: skip
    return {'steady': steady, 'enc': enc, 'res': res}


def same(a: dict, b: dict) -> tuple[bool, str]:
    notes, ok = [], True
    for word, (sa, ma) in a['res'].items():
        sb, mb = b['res'][word]
        ka, kb = ma[sa >= 0.5], mb[sb >= 0.5]
        if len(ka) != len(kb):
            ok = False
        if len(ka) and len(kb):
            iou = max(float((x & y).sum() / max(1, (x | y).sum())) for x in ka for y in kb)
            ok &= iou > 0.8
            notes.append(f'{word} IoU {iou:.2f}')
    return ok, ', '.join(notes)


def main() -> None:
    import onnxruntime as ort

    from wissi_viz import sam3

    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise SystemExit('this compares Core ML with the CPU on an Apple-silicon Mac')
    if 'CoreMLExecutionProvider' not in ort.get_available_providers():
        raise SystemExit('this ONNX Runtime has no Core ML backend: uv sync --extra all (onnxruntime for macOS has it)')
    if not sam3.accepted():
        raise SystemExit('accept the SAM License first (Annotate → Find all → By text → SAM 3 → Licence)')
    if sam3.folder() is None:
        print('downloading SAM 3 (once)…', flush=True)
        sam3.download(lambda f, m: print(m, flush=True))
    sam3.encoder_fp32(fetch=True)
    print(f'ONNX Runtime {ort.__version__} · {platform.platform()}', flush=True)
    cpu = run('CPU (int8, as now)', 'cpu')
    best = None
    for units in ('ALL', 'CPUAndGPU'):
        try:
            r = run(f'Core ML ({units})', 'coreml', units)
        except Exception as e:  # noqa: BLE001
            print(f'Core ML ({units}): failed: {e}', flush=True)
            continue
        ok, note = same(cpu, r)
        speed = cpu['steady'] / r['steady']
        print(f'  → {speed:.1f}× the CPU per picture; {"same finds" if ok else "DIFFERENT finds"} ({note})', flush=True)
        if ok and (best is None or speed > best[1]):
            best = (units, speed)
    if best and best[1] >= 1.5:
        print(f'\nVerdict: Core ML ({best[0]}) is {best[1]:.1f}× faster and finds the same: worth using '
              f'(WISSI_SAM3_DEVICE=coreml WISSI_SAM3_COREML_UNITS={best[0]}).')  # fmt: skip
    else:
        print('\nVerdict: Core ML is not clearly faster here (or finds different things): keep the CPU.')


if __name__ == '__main__':
    main()
