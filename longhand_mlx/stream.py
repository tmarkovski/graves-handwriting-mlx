"""Single-line streaming facade for fragmented generation.

A `HandStream` keeps a live one-sample `Generator` alive across calls.
`advance(until_char=N)` runs forward until the attention's argmax crosses
character index `N`; `advance()` runs to the model's natural end-of-text
termination. Resuming is just calling `advance` again on the same instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .alphabet import encode_ascii
from .generator import Generator
from .hand import MAX_PRIME_LENGTH, STEPS_PER_CHARACTER, load_style

if TYPE_CHECKING:
    from .hand import Hand


class HandStream:
    def __init__(self, hand: "Hand", text: str, *, bias: float = 0.5, style: int | None = None, seed: int = 0):
        self.text = text
        self.bias = float(bias)

        if style is not None:
            stroke_array, primer_text = load_style(style)
            encoded = encode_ascii(primer_text + " " + text)
            x_prime = np.zeros((1, MAX_PRIME_LENGTH, 3), dtype=np.float32)
            x_prime[0, : len(stroke_array)] = stroke_array
            prime_lengths = np.array([len(stroke_array)], dtype=np.int32)
        else:
            encoded = encode_ascii(text)
            x_prime = None
            prime_lengths = None

        chars = np.zeros((1, 120), dtype=np.int32)
        chars[0, : len(encoded)] = encoded
        self._char_length = int(len(encoded))

        self._generator = Generator(
            hand.cell,
            chars=chars,
            char_lengths=np.array([self._char_length], dtype=np.int32),
            biases=np.array([self.bias], dtype=np.float32),
            x_prime=x_prime,
            prime_lengths=prime_lengths,
            seed=seed,
        )

    @property
    def done(self) -> bool:
        return bool(self._generator.done[0])

    def advance(self, until_char: int | None = None, max_steps: int | None = None) -> np.ndarray:
        max_steps = max_steps if max_steps is not None else STEPS_PER_CHARACTER * len(self.text)
        stop_when = None
        if until_char is not None:
            target = min(until_char, self._char_length - 1)
            stop_when = lambda phi_np, _stroke: phi_np.argmax(axis=1) >= target

        strokes = self._generator.advance(max_steps=max_steps, stop_when=stop_when)
        return strokes[0]
