# SAM 2.1 tiny, int8 (bundled with wissi-viz)

SAM 2.1 hiera-tiny by Meta Platforms, Inc. (Apache-2.0, see LICENSE-Apache-2.0.txt and NOTICE.txt), exported to ONNX by wissi-viz (`scripts/export_sam_onnx.py`, release `sam2.1-onnx-v1`).

Modified: the weights of both files were quantized to 8-bit integers (ONNX Runtime dynamic quantization, `scripts/bundle_sam.py`). The masks match the float32 model closely; the float32 files are on the release.
