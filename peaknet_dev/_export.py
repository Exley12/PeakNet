"""Dev-only: convert the trained PyTorch checkpoint to ONNX.

Run this whenever peaknet.pth changes:

    python -m peaknet_package._export
"""
from pathlib import Path

import torch

from model import Peaknet

PROJECT_DIR = Path(__file__).resolve().parent.parent
PTH_PATH = PROJECT_DIR / "Models" / "checkpoints" / "peaknet_20260815_011311.pth"
ONNX_PATH = Path(PTH_PATH.with_suffix(".onnx").name)


def export() -> Path:
    model = Peaknet()
    model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
    model.eval()

    # Dummy input: (batch=1, channels=1, length). Length is made dynamic below
    # so the exported model accepts any spectrum length.
    dummy = torch.zeros(1, 1, 256, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH.as_posix(),
        input_names=["spectrum"],
        output_names=["prediction"],
        dynamic_axes={
            "spectrum": {0: "batch", 2: "length"},
            "prediction": {0: "batch", 2: "length"},
        },
        opset_version=17,
    )
    print(f"[peaknet] Exported ONNX model -> {ONNX_PATH}")
    return ONNX_PATH


if __name__ == "__main__":
    export()