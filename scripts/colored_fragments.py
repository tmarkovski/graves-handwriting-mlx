"""Render one line of handwriting where each space-separated fragment is its
own color.

Approach: run the model once in a per-step loop, recording which character
the attention was looking at when each stroke was emitted. Then assign every
stroke to a fragment by that character, so color follows the model's actual
intent rather than any external cut points. Layout (denoise + deslant +
center) runs once on the whole line so it stays one coherent visual line.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import svgwrite

from longhand_mlx import Hand
from longhand_mlx.alphabet import encode_ascii
from longhand_mlx.draw import _align, _denoise, offsets_to_coords

TEXT = "Hello world, I'm Graphite!"
FRAGMENTS = ["Hello", "world,", "I'm", "Graphite!"]
COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231"]
SCALE = 1.5
VIEW_WIDTH = 1000
LINE_HEIGHT = 60
SEED = 0
BIAS = 0.75


def fragment_starts(text: str, fragments: list[str]) -> list[int]:
    """Char index at which each fragment begins inside `text`."""
    starts: list[int] = []
    cursor = 0
    for fragment in fragments:
        cursor = text.index(fragment, cursor)
        starts.append(cursor)
        cursor += len(fragment)
    return starts


def fragment_id_for_char(char_index: int, starts: list[int]) -> int:
    """Which fragment owns a given character index."""
    fragment_id = 0
    for index, start in enumerate(starts):
        if char_index >= start:
            fragment_id = index
    return fragment_id


def generate_with_attention_log(hand: Hand, text: str, bias: float, seed: int):
    """Run the model one step at a time, recording the attention argmax at
    each emitted stroke."""
    encoded = encode_ascii(text)
    char_length = len(encoded)
    max_char_len = max(120, char_length)

    chars = np.zeros((1, max_char_len), dtype=np.int32)
    chars[0, : len(encoded)] = encoded
    chars_index = mx.array(chars)
    chars_onehot = mx.take(mx.eye(73, dtype=mx.float32), chars_index, axis=0)
    char_positions = mx.arange(max_char_len, dtype=mx.float32).reshape(1, 1, max_char_len)
    char_mask = (mx.arange(max_char_len, dtype=mx.int32) < char_length).astype(mx.float32).reshape(1, max_char_len, 1)
    bias_array = mx.array([bias], dtype=mx.float32)

    state = hand.cell.initial_state(1, max_char_len)
    last_input = mx.array([[0.0, 0.0, 1.0]], dtype=mx.float32)
    key = mx.random.key(seed)

    strokes: list[np.ndarray] = []
    attended_chars: list[int] = []

    for _ in range(40 * len(text)):
        key, step_key = mx.random.split(key, 2)
        state = hand.cell.step(state, last_input, chars_onehot, char_positions, char_mask)
        last_input = hand.cell.head.sample(state[4], bias_array, step_key)
        mx.eval(state, last_input)

        stroke = np.array(last_input)[0]
        attention_argmax = int(np.array(state[8])[0].argmax())
        strokes.append(stroke)
        attended_chars.append(attention_argmax)

        if attention_argmax >= char_length - 1 and stroke[2] == 1.0:
            break
        if attention_argmax >= char_length:
            break

    return np.array(strokes), np.array(attended_chars), char_length


def main() -> None:
    hand = Hand()
    strokes, attention, _ = generate_with_attention_log(hand, TEXT, bias=BIAS, seed=SEED)

    starts = fragment_starts(TEXT, FRAGMENTS)
    stroke_fragment = np.array([fragment_id_for_char(int(c), starts) for c in attention])
    counts = [int((stroke_fragment == i).sum()) for i in range(len(FRAGMENTS))]
    print(f"strokes per fragment: {dict(zip(FRAGMENTS, counts))}")

    offsets = strokes.astype(np.float64).copy()
    offsets[:, :2] *= SCALE
    coords = offsets_to_coords(offsets)
    coords = _denoise(coords)
    coords[:, :2] = _align(coords[:, :2])
    coords[:, 1] *= -1

    view_height = LINE_HEIGHT * 2
    cursor_origin = np.array([0.0, -(3 * LINE_HEIGHT / 4)])
    coords[:, :2] -= coords[:, :2].min() + cursor_origin
    coords[:, 0] += (VIEW_WIDTH - coords[:, 0].max()) / 2

    drawing = svgwrite.Drawing()
    drawing.viewbox(width=VIEW_WIDTH, height=view_height)
    drawing.add(drawing.rect(insert=(0, 0), size=(VIEW_WIDTH, view_height), fill="white"))

    # Build one path per (run of strokes assigned to the same fragment), so a
    # fragment that the model revisits can still be drawn in its own color
    # without issues.
    run_start = 0
    for i in range(1, len(stroke_fragment) + 1):
        if i == len(stroke_fragment) or stroke_fragment[i] != stroke_fragment[run_start]:
            segment = coords[run_start:i]
            color = COLORS[stroke_fragment[run_start]]
            # Each <path> must start with M so the pen is positioned before
            # any L commands. After the first point, follow the model's
            # pen-up signals.
            previous_eos = 1.0
            path_string = ""
            for index_in_run, (x_value, y_value, eos) in enumerate(segment):
                command = "M" if index_in_run == 0 or previous_eos == 1.0 else "L"
                path_string += f"{command}{x_value},{y_value} "
                previous_eos = eos
            drawing.add(
                svgwrite.path.Path(path_string).stroke(color=color, width=2, linecap="round").fill("none")
            )
            run_start = i

    Path("graphite.svg").write_text(drawing.tostring())
    print("wrote graphite.svg")


if __name__ == "__main__":
    main()
