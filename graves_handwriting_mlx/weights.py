"""Load the bundled MLX weights.

The conversion from the original TensorFlow checkpoint lives in
`scripts/convert_checkpoint.py` and is the only place TensorFlow is touched.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_weights(path: Path | None = None) -> dict[str, mx.array]:
    if path is None:
        path = Path(str(files("graves_handwriting_mlx") / "data" / "weights.npz"))
    raw = np.load(path)
    return {key: mx.array(raw[key]) for key in raw.files}
