import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Resolve model path relative to this file so it works after pip install
_MODEL_PATH = Path(__file__).resolve().parent / "peaknet_20260815_011311.onnx"

# Load once at import time so every call shares the same session
_SESSION = ort.InferenceSession(_MODEL_PATH.as_posix())
_INPUT_NAME = _SESSION.get_inputs()[0].name
_OUTPUT_NAME = _SESSION.get_outputs()[0].name

def _predict(spectrum):
    # Force input to float32 numpy array
    spectrum = np.asarray(spectrum, dtype=np.float32)
    # Model only handles 1D input
    if spectrum.ndim != 1:
        raise ValueError("Expected 1D spectrum")
    # Clip to the range the model was trained on
    spectrum = np.clip(spectrum, -0.2, 1.0)
    # Pad with zeros either side for better edge performance
    pad = 100
    padded = np.pad(spectrum, pad, mode='constant', constant_values=0)
    # Model expects shape (batch, channel, length)
    x = padded[None, None, :].astype(np.float32)
    # Run inference
    pred = _SESSION.run([_OUTPUT_NAME], {_INPUT_NAME: x})[0].squeeze()
    # Crop the padded borders — output is already in [0, 1] from Sigmoid
    pred = pred[pad:-pad]
    return pred

# Inherit from the module's metaclass to customize module behavior
class _CallableModule(type(sys.modules[__name__])):
    def __call__(self, spectrum):
        # This runs when you call peaknet(spectrum)
        return _predict(spectrum)

# Replace the module's class so it becomes callable
# Note a __class__ example is:
# x = 5
# print(x.__class__)  # <class 'int'>
sys.modules[__name__].__class__ = _CallableModule
