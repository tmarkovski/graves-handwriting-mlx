"""The handwriting cell: 3 LSTMs + Gaussian-window attention + MDN head.

This module is the per-timestep unit. It does not loop, sample multiple
strokes, or check termination — that is the `Generator`'s job.

State tuple layout (matches the upstream `LSTMAttentionCellState`):
    (h1, c1, h2, c2, h3, c3, kappa, w, phi)

Concat orders inside `step` must match the original TF code exactly,
otherwise the saved weight rows map to the wrong inputs:
    lstm1 input          = [w_prev, x]
    attention input      = [w_prev, x, h1]
    lstm2 input          = [x, h1, w]
    lstm3 input          = [x, h2, w]
"""

from __future__ import annotations

from typing import Mapping

import mlx.core as mx

from .modules import GaussianWindowAttention, LSTMCell, MixtureDensityHead

HIDDEN_SIZE = 400
NUM_ATTENTION_MIXTURES = 10
NUM_OUTPUT_MIXTURES = 20
ALPHABET_SIZE = 73

State = tuple[
    mx.array, mx.array,  # h1, c1
    mx.array, mx.array,  # h2, c2
    mx.array, mx.array,  # h3, c3
    mx.array,            # kappa
    mx.array,            # w
    mx.array,            # phi
]


class HandwritingCell:
    def __init__(self, weights: Mapping[str, mx.array]):
        self.lstm_one = LSTMCell(weights["lstm1_kernel"], weights["lstm1_bias"])
        self.lstm_two = LSTMCell(weights["lstm2_kernel"], weights["lstm2_bias"])
        self.lstm_three = LSTMCell(weights["lstm3_kernel"], weights["lstm3_bias"])
        self.attention = GaussianWindowAttention(weights["attention_weights"], weights["attention_biases"])
        self.head = MixtureDensityHead(weights["gmm_weights"], weights["gmm_biases"])

    def initial_state(self, batch_size: int, max_char_length: int) -> State:
        zeros = lambda *shape: mx.zeros(shape, dtype=mx.float32)
        return (
            zeros(batch_size, HIDDEN_SIZE),
            zeros(batch_size, HIDDEN_SIZE),
            zeros(batch_size, HIDDEN_SIZE),
            zeros(batch_size, HIDDEN_SIZE),
            zeros(batch_size, HIDDEN_SIZE),
            zeros(batch_size, HIDDEN_SIZE),
            zeros(batch_size, NUM_ATTENTION_MIXTURES),
            zeros(batch_size, ALPHABET_SIZE),
            zeros(batch_size, max_char_length),
        )

    def step(
        self,
        state: State,
        inputs: mx.array,
        chars_onehot: mx.array,
        char_positions: mx.array,
        char_mask: mx.array,
    ) -> State:
        h1_prev, c1_prev, h2_prev, c2_prev, h3_prev, c3_prev, kappa_prev, w_prev, _ = state

        lstm1_input = mx.concatenate([w_prev, inputs], axis=-1)
        h1, c1 = self.lstm_one(lstm1_input, h1_prev, c1_prev)

        kappa, window, phi = self.attention(
            w_prev, inputs, h1, kappa_prev, char_positions, char_mask, chars_onehot
        )

        lstm2_input = mx.concatenate([inputs, h1, window], axis=-1)
        h2, c2 = self.lstm_two(lstm2_input, h2_prev, c2_prev)

        lstm3_input = mx.concatenate([inputs, h2, window], axis=-1)
        h3, c3 = self.lstm_three(lstm3_input, h3_prev, c3_prev)

        return (h1, c1, h2, c2, h3, c3, kappa, window, phi)
