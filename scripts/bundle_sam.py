"""Build the SAM 2.1 model that ships inside wissi-viz: SAM 2.1 tiny (Meta, Apache-2.0), exported to ONNX by
scripts/export_sam_onnx.py (published as the GitHub release ``sam2.1-onnx-v1``), weights quantized to int8.

    pip install onnxruntime onnx
    python scripts/bundle_sam.py                      # → src/wissi_viz/models/sam2.1-tiny-int8/

About 33 MB instead of 126 MB, and faster on CPU. Only the matrix multiplications are quantized (int8 convolutions
need ONNX Runtime ≥ 1.24 on CPU), so it runs on ONNX Runtime 1.20 and later. Masks match the float32 model closely
(median IoU 0.96 on 40 clicked nuclei of CMU-1, 10th percentile 0.91). The larger models (float32 tiny, small) are still downloaded on
demand when ``WISSI_SAM_MODEL`` asks for them.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

RELEASE = 'https://github.com/zaloch/wissi-viz/releases/download/sam2.1-onnx-v1'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src' / 'wissi_viz' / 'models' / 'sam2.1-tiny-int8'


def _get(url: str, dst: Path) -> None:
    with urllib.request.urlopen(url, timeout=600) as r, open(dst, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--src', default=None, help='a folder with sam2.1-tiny-{encoder,decoder}.onnx (default: download)')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = Path(a.src) if a.src else tmp
        if not a.src:
            _get(f'{RELEASE}/SHA256SUMS', tmp / 'SHA256SUMS')
            want = {ln.split()[1]: ln.split()[0] for ln in (tmp / 'SHA256SUMS').read_text().splitlines() if ln.strip()}
            for n in ('sam2.1-tiny-encoder.onnx', 'sam2.1-tiny-decoder.onnx', 'LICENSE-Apache-2.0.txt', 'NOTICE.txt'):
                _get(f'{RELEASE}/{n}', tmp / n)
                if n in want and _sha(tmp / n) != want[n]:
                    raise SystemExit(f'{n}: checksum mismatch')
        for part in ('encoder', 'decoder'):
            quantize_dynamic(
                str(src / f'sam2.1-tiny-{part}.onnx'),
                str(out / f'{part}.onnx'),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=['MatMul', 'Gemm'],  # not Conv: ConvInteger has no CPU kernel in older ORT
            )
        for n in ('LICENSE-Apache-2.0.txt', 'NOTICE.txt'):
            if (src / n).exists():
                shutil.copy(src / n, out / n)
    (out / 'README.md').write_text(
        '# SAM 2.1 tiny, int8 (bundled with wissi-viz)\n\n'
        'SAM 2.1 hiera-tiny by Meta Platforms, Inc. (Apache-2.0, see LICENSE-Apache-2.0.txt and NOTICE.txt), exported to '
        'ONNX by wissi-viz (`scripts/export_sam_onnx.py`, release `sam2.1-onnx-v1`).\n\n'
        'Modified: the weights of both files were quantized to 8-bit integers (ONNX Runtime dynamic quantization, '
        '`scripts/bundle_sam.py`). The masks match the float32 model closely; the float32 files are on the release.\n',
        encoding='utf-8',
    )
    sums = ''.join(f'{_sha(out / n)}  {n}\n' for n in ('encoder.onnx', 'decoder.onnx'))
    (out / 'SHA256SUMS').write_text(sums, encoding='utf-8')
    for p in sorted(out.iterdir()):
        print(f'{p.name:28} {p.stat().st_size / 2**20:8.1f} MB')


if __name__ == '__main__':
    main()
