"""Building blocks of Graves' handwriting model.

Each class is a plain holder for the weights of one architectural piece, with
a `__call__` that runs one forward step on a batch. They deliberately don't
inherit from `mlx.nn.Module` — this code is inference-only and tracing
through `mx.compile` is simpler with bare functions over bare arrays.
"""

from __future__ import annotations

import mlx.core as mx


class LSTMCell:
    """One TF1-style LSTM cell, gate order i, j, f, o.

    The kernel layout matches `tf.contrib.rnn.LSTMCell`: a single
    `[input + hidden, 4 * hidden]` matrix multiplied by `[x, h_prev]`.
    The forget-gate bias of 1.0 that TF added at runtime is baked into
    the saved bias by `weights.load_weights`.
    """

    def __init__(self, kernel: mx.array, bias: mx.array):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, inputs: mx.array, hidden: mx.array, cell: mx.array) -> tuple[mx.array, mx.array]:
        gates = mx.concatenate([inputs, hidden], axis=-1) @ self.kernel + self.bias
        input_gate, candidate, forget_gate, output_gate = mx.split(gates, 4, axis=-1)
        new_cell = mx.sigmoid(forget_gate) * cell + mx.sigmoid(input_gate) * mx.tanh(candidate)
        new_hidden = mx.sigmoid(output_gate) * mx.tanh(new_cell)
        return new_hidden, new_cell


class GaussianWindowAttention:
    """Soft-monotonic attention over the character sequence.

    Reads `[w_prev, x, h1]` and produces a `K`-component mixture of Gaussians
    in character-position space. The mixture's center κ accumulates over
    time (κ_t = κ_{t-1} + softplus(Δκ)/25) so attention can only move
    forward. The output is the convex combination of one-hot character rows
    weighted by φ(u) at integer positions u = 0…L-1.
    """

    def __init__(self, weight: mx.array, bias: mx.array, num_mixtures: int = 10):
        self.weight = weight
        self.bias = bias
        self.num_mixtures = num_mixtures

    def __call__(
        self,
        previous_window: mx.array,
        inputs: mx.array,
        hidden_one: mx.array,
        previous_kappa: mx.array,
        char_positions: mx.array,
        char_mask: mx.array,
        chars_onehot: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        attention_inputs = mx.concatenate([previous_window, inputs, hidden_one], axis=-1)
        raw = attention_inputs @ self.weight + self.bias
        raw = mx.logaddexp(raw, mx.zeros_like(raw))  # softplus
        alpha, beta, kappa_step = mx.split(raw, 3, axis=-1)
        kappa = previous_kappa + kappa_step / 25.0
        beta = mx.maximum(beta, 0.01)

        diff = mx.expand_dims(kappa, 2) - char_positions
        phi = mx.sum(
            mx.expand_dims(alpha, 2) * mx.exp(-(diff * diff) / mx.expand_dims(beta, 2)),
            axis=1,
        )
        window = mx.sum(mx.expand_dims(phi, 2) * char_mask * chars_onehot, axis=1)
        return kappa, window, phi


class MixtureDensityHead:
    """20-component MDN with the Graves bias trick for sharpness control.

    `parse(...)` returns the seven distribution parameters; `sample(...)`
    composes the categorical mixture choice, the chosen 2-D Gaussian and
    the Bernoulli pen-up into a `[B, 3]` stroke `[Δx, Δy, eos]`.
    """

    def __init__(self, weight: mx.array, bias: mx.array, num_mixtures: int = 20):
        self.weight = weight
        self.bias = bias
        self.num_mixtures = num_mixtures

    def parse(
        self, hidden_three: mx.array, sharpness: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        epsilon = 1e-8
        sigma_floor = 1e-4
        raw = hidden_three @ self.weight + self.bias
        pi_logits, sigmas, rhos, mus, eos = mx.split(
            raw,
            [self.num_mixtures, 3 * self.num_mixtures, 4 * self.num_mixtures, 6 * self.num_mixtures],
            axis=-1,
        )
        sharpness_column = mx.expand_dims(sharpness, 1)
        pi_logits = pi_logits * (1.0 + sharpness_column)
        sigmas = sigmas - sharpness_column

        pi = mx.softmax(pi_logits, axis=-1)
        pi = mx.where(pi < 0.01, mx.zeros_like(pi), pi)
        sigmas = mx.maximum(mx.exp(sigmas), sigma_floor)
        rhos = mx.clip(mx.tanh(rhos), epsilon - 1.0, 1.0 - epsilon)
        eos = mx.clip(mx.sigmoid(eos), epsilon, 1.0 - epsilon)
        eos = mx.where(eos < 0.01, mx.zeros_like(eos), eos)

        mu_x, mu_y = mx.split(mus, 2, axis=-1)
        sigma_x, sigma_y = mx.split(sigmas, 2, axis=-1)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rhos, eos

    def sample(self, hidden_three: mx.array, sharpness: mx.array, key: mx.array) -> mx.array:
        pi, mu_x, mu_y, sigma_x, sigma_y, rhos, eos = self.parse(hidden_three, sharpness)
        key_component, key_z1, key_z2, key_eos = mx.random.split(key, 4)

        component = mx.random.categorical(mx.log(pi + 1e-20), key=key_component)

        def take(tensor: mx.array) -> mx.array:
            return mx.take_along_axis(tensor, mx.expand_dims(component, 1), axis=1)[:, 0]

        chosen_mu_x, chosen_mu_y = take(mu_x), take(mu_y)
        chosen_sigma_x, chosen_sigma_y = take(sigma_x), take(sigma_y)
        chosen_rho = take(rhos)
        eos_probability = eos[:, 0]

        z_one = mx.random.normal(chosen_mu_x.shape, key=key_z1)
        z_two = mx.random.normal(chosen_mu_x.shape, key=key_z2)
        delta_x = chosen_mu_x + chosen_sigma_x * z_one
        delta_y = chosen_mu_y + chosen_sigma_y * (
            chosen_rho * z_one + mx.sqrt(mx.maximum(1.0 - chosen_rho * chosen_rho, 0.0)) * z_two
        )
        eos_sample = (mx.random.uniform(shape=eos_probability.shape, key=key_eos) < eos_probability).astype(mx.float32)
        return mx.stack([delta_x, delta_y, eos_sample], axis=1)
