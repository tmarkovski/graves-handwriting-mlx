"""High-level multi-line facade. Mirrors the upstream `demo.py:Hand`."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Sequence

import numpy as np

from .alphabet import MAX_CHAR_LEN, alphabet, encode_ascii
from .draw import write_svg
from .generator import Generator
from .model import HandwritingCell
from .weights import load_weights

MAX_PRIME_LENGTH = 1200
STEPS_PER_CHARACTER = 40


def load_style(style_id: int) -> tuple[np.ndarray, str]:
    """Return `(stroke_array, primer_text)` for one of the bundled styles."""
    styles_dir = files("longhand_mlx") / "data" / "styles"
    strokes = np.load(str(styles_dir / f"style-{style_id}-strokes.npy"))
    primer_chars = np.load(str(styles_dir / f"style-{style_id}-chars.npy")).tobytes().decode("utf-8")
    return strokes.astype(np.float32), primer_chars


class Hand:
    def __init__(self, weights_path: Path | None = None):
        self.weights = load_weights(weights_path)
        self.cell = HandwritingCell(self.weights)
        self._valid_characters = set(alphabet)

    def stream(self, text: str, *, bias: float = 0.5, style: int | None = None, seed: int = 0):
        from .stream import HandStream

        return HandStream(self, text, bias=bias, style=style, seed=seed)

    def write(
        self,
        filename: str | Path,
        lines: Sequence[str],
        *,
        biases: Sequence[float] | None = None,
        styles: Sequence[int] | None = None,
        stroke_colors: Sequence[str] | None = None,
        stroke_widths: Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        self._validate(lines)
        strokes = self._sample(lines, biases=biases, styles=styles, seed=seed)
        write_svg(filename, strokes, lines, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _validate(self, lines: Sequence[str]) -> None:
        for line_index, line in enumerate(lines):
            if len(line) > MAX_CHAR_LEN:
                raise ValueError(f"line {line_index} exceeds {MAX_CHAR_LEN} characters")
            for character in line:
                if character not in self._valid_characters:
                    raise ValueError(f"invalid character {character!r} in line {line_index}")

    def _sample(
        self,
        lines: Sequence[str],
        biases: Sequence[float] | None,
        styles: Sequence[int] | None,
        seed: int,
    ) -> list[np.ndarray]:
        num_samples = len(lines)
        max_steps = STEPS_PER_CHARACTER * max(len(line) for line in lines)
        biases_array = np.array(biases if biases is not None else [0.5] * num_samples, dtype=np.float32)

        chars = np.zeros((num_samples, 120), dtype=np.int32)
        char_lengths = np.zeros((num_samples,), dtype=np.int32)
        x_prime = np.zeros((num_samples, MAX_PRIME_LENGTH, 3), dtype=np.float32)
        prime_lengths = np.zeros((num_samples,), dtype=np.int32)

        if styles is not None:
            for index, (line, style_id) in enumerate(zip(lines, styles)):
                stroke_array, primer_text = load_style(style_id)
                encoded = encode_ascii(primer_text + " " + line)
                x_prime[index, : len(stroke_array)] = stroke_array
                prime_lengths[index] = len(stroke_array)
                chars[index, : len(encoded)] = encoded
                char_lengths[index] = len(encoded)
        else:
            for index, line in enumerate(lines):
                encoded = encode_ascii(line)
                chars[index, : len(encoded)] = encoded
                char_lengths[index] = len(encoded)

        generator = Generator(
            self.cell,
            chars=chars,
            char_lengths=char_lengths,
            biases=biases_array,
            x_prime=x_prime if styles is not None else None,
            prime_lengths=prime_lengths if styles is not None else None,
            seed=seed,
        )
        raw_strokes = generator.advance(max_steps=max_steps)
        return [raw_strokes[i][~np.all(raw_strokes[i] == 0.0, axis=1)] for i in range(num_samples)]
