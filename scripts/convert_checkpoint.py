"""One-shot: convert the upstream TF1 checkpoint into graves_handwriting_mlx/data/weights.npz.

Usage:
    uv run python scripts/convert_checkpoint.py \
        --checkpoint handwriting-synthesis/checkpoints/model-17900 \
        --output graves_handwriting_mlx/data/weights.npz

TF1 `tf.contrib.rnn.LSTMCell` adds `forget_bias=1.0` to the forget-gate
slice of the LSTM bias at runtime; we bake that into the saved bias here so
the inference loop has one fewer constant add per layer per step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHECKPOINT_VARIABLES = {
    "lstm1_kernel": "rnn/LSTMAttentionCell/lstm_cell/kernel",
    "lstm1_bias": "rnn/LSTMAttentionCell/lstm_cell/bias",
    "lstm2_kernel": "rnn/LSTMAttentionCell/lstm_cell_1/kernel",
    "lstm2_bias": "rnn/LSTMAttentionCell/lstm_cell_1/bias",
    "lstm3_kernel": "rnn/LSTMAttentionCell/lstm_cell_2/kernel",
    "lstm3_bias": "rnn/LSTMAttentionCell/lstm_cell_2/bias",
    "attention_weights": "rnn/LSTMAttentionCell/attention/weights",
    "attention_biases": "rnn/LSTMAttentionCell/attention/biases",
    "gmm_weights": "rnn/gmm/weights",
    "gmm_biases": "rnn/gmm/biases",
}

HIDDEN_SIZE = 400


def bake_forget_bias(bias: np.ndarray) -> np.ndarray:
    out = bias.astype(np.float32).copy()
    out[2 * HIDDEN_SIZE : 3 * HIDDEN_SIZE] += 1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import tensorflow as tf  # local import: TF is a dev-only dependency

    reader = tf.train.load_checkpoint(str(args.checkpoint))
    arrays: dict[str, np.ndarray] = {}
    for friendly_name, tf_name in CHECKPOINT_VARIABLES.items():
        tensor = reader.get_tensor(tf_name).astype(np.float32)
        if friendly_name.startswith("lstm") and friendly_name.endswith("_bias"):
            tensor = bake_forget_bias(tensor)
        arrays[friendly_name] = tensor

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(f"wrote {args.output} ({sum(a.nbytes for a in arrays.values()) / 1e6:.1f} MB raw)")


if __name__ == "__main__":
    main()
