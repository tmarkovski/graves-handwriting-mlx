"""Render messages from the `Roman1111111/claude-sonnet-4.6-100000X-filtered`
HuggingFace dataset as handwriting and write to a parquet dataset.

Schema matches `graphite/scripts/convert_jsonl_to_parquet.py`:

    strokes : list<list<struct{points: list<struct{x: float32, y: float32}>}>>
    text    : string
    preview : struct{bytes: binary, path: string}    -> cast to HF Image
    file    : string                                  -> sha256(text)

**One row per line.** Each user/assistant message is word-wrapped at
75 characters and every wrapped line is emitted as its own parquet row.
The outer `strokes` list groups by **word** — every space-separated token
in the line gets its own group; the inner list is the pen-down strokes
within that word. Words are identified by tracking which character the
attention's argmax was on at each step.

Messages with characters outside the model's 73-char vocabulary are
skipped, as are non-user/non-assistant messages.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Image as HFImage, load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

from longhand_mlx import Hand
from longhand_mlx.alphabet import alphabet, encode_ascii
from longhand_mlx.draw import _align, _denoise, offsets_to_coords
from longhand_mlx.model import ALPHABET_SIZE

DATASET_NAME = "Roman1111111/claude-sonnet-4.6-100000X-filtered"
NUM_STYLES = 13
LINE_LIMIT = 75
ROLES_TO_RENDER = {"user", "assistant"}
VALID_CHARS = set(alphabet)

SCALE = 1.5
LINE_HEIGHT = 60
VIEW_WIDTH = 1000
PNG_STROKE_WIDTH = 2
MAX_CHAR_BUFFER = 120

POINT_TYPE = pa.struct([("x", pa.float32()), ("y", pa.float32())])
STROKE_TYPE = pa.struct([("points", pa.list_(POINT_TYPE))])
SCHEMA = pa.schema(
    [
        ("strokes", pa.list_(pa.list_(STROKE_TYPE))),
        ("text", pa.string()),
        ("preview", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("file", pa.string()),
    ]
)


def text_is_renderable(text: str) -> bool:
    return all(character in VALID_CHARS for character in text)


def split_into_lines(text: str, limit: int = LINE_LIMIT) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current = ""
        for word in paragraph.split():
            if len(word) > limit:
                word = word[:limit]
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= limit:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def word_id_for_each_char(text: str) -> np.ndarray:
    """For each char index in `text`, the 0-based word number it belongs to.
    Spaces are bucketed with the *preceding* word so they don't create empty
    word groups."""
    word_ids = np.zeros(len(text), dtype=np.int32)
    current_word = -1
    in_word = False
    for index, character in enumerate(text):
        if character == " ":
            in_word = False
        elif not in_word:
            current_word += 1
            in_word = True
        word_ids[index] = max(current_word, 0)
    return word_ids


def generate_with_attention(
    hand: Hand, line: str, style: int, bias: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model one step at a time on a single line, returning
    `(strokes [T, 3], attended_char_index [T])`. Priming with the chosen
    style is done first so the writing inherits its visual idiom."""
    primer_strokes_np, primer_text = _load_style(style)
    combined_text = primer_text + " " + line
    encoded = encode_ascii(combined_text)
    char_length = len(encoded)
    max_char_length = max(MAX_CHAR_BUFFER, char_length)

    chars = np.zeros((1, max_char_length), dtype=np.int32)
    chars[0, : len(encoded)] = encoded
    chars_index = mx.array(chars)
    chars_onehot = mx.take(mx.eye(ALPHABET_SIZE, dtype=mx.float32), chars_index, axis=0)
    char_positions = mx.arange(max_char_length, dtype=mx.float32).reshape(1, 1, max_char_length)
    char_mask = (mx.arange(max_char_length, dtype=mx.int32) < char_length).astype(mx.float32).reshape(
        1, max_char_length, 1
    )
    bias_array = mx.array([bias], dtype=mx.float32)

    state = hand.cell.initial_state(1, max_char_length)

    # Priming: feed the style strokes through the cell with no sampling.
    primer_strokes = mx.array(primer_strokes_np[None, :, :])
    for time_index in range(primer_strokes_np.shape[0]):
        state = hand.cell.step(
            state,
            primer_strokes[:, time_index, :],
            chars_onehot,
            char_positions,
            char_mask,
        )

    key = mx.random.key(int(seed))
    key, sample_key = mx.random.split(key, 2)
    last_input = hand.cell.head.sample(state[4], bias_array, sample_key)
    mx.eval(state, last_input)

    strokes: list[np.ndarray] = []
    attended: list[int] = []
    for _ in range(40 * len(line)):
        key, step_key = mx.random.split(key, 2)
        state = hand.cell.step(state, last_input, chars_onehot, char_positions, char_mask)
        last_input = hand.cell.head.sample(state[4], bias_array, step_key)
        mx.eval(state, last_input)

        stroke = np.array(last_input)[0]
        attention_argmax = int(np.array(state[8])[0].argmax())
        strokes.append(stroke)
        attended.append(attention_argmax)
        if attention_argmax >= char_length - 1 and stroke[2] == 1.0:
            break
        if attention_argmax >= char_length:
            break

    return np.array(strokes), np.array(attended), len(primer_text) + 1


def _load_style(style_id: int) -> tuple[np.ndarray, str]:
    from longhand_mlx.hand import load_style

    return load_style(style_id)


def line_to_absolute_coords(strokes_offsets: np.ndarray, y_offset: float) -> np.ndarray | None:
    """Scale, cumsum to absolute coords, denoise, deslant, and place the
    line at the requested vertical position. Returns `[N, 3]` `(x, y, eos)`."""
    if len(strokes_offsets) == 0:
        return None
    scaled = strokes_offsets.astype(np.float64).copy()
    scaled[:, :2] *= SCALE
    coords = offsets_to_coords(scaled)
    coords = _denoise(coords)
    coords[:, :2] = _align(coords[:, :2])
    coords[:, 1] *= -1
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()
    coords[:, 0] += (VIEW_WIDTH - coords[:, 0].max()) / 2
    coords[:, 1] += y_offset
    return coords


def coords_and_word_assignment_to_groups(
    coords: np.ndarray, word_assignment: np.ndarray
) -> list[list[dict]]:
    """Split coords into one word group per word index. Within each word
    group, further split by `eos == 1` to get individual pen-down strokes."""
    if len(coords) != len(word_assignment):
        raise ValueError(f"length mismatch: {len(coords)} coords vs {len(word_assignment)} attendances")

    groups: list[list[dict]] = []
    run_start = 0
    for index in range(1, len(coords) + 1):
        if index == len(coords) or word_assignment[index] != word_assignment[run_start]:
            segment = coords[run_start:index]
            split_indices = np.where(segment[:, 2] == 1)[0] + 1
            word_group: list[dict] = []
            for stroke_coords in np.split(segment[:, :2], split_indices, axis=0):
                if len(stroke_coords) == 0:
                    continue
                word_group.append(
                    {
                        "points": [
                            {"x": float(x_value), "y": float(y_value)}
                            for x_value, y_value in stroke_coords
                        ]
                    }
                )
            if word_group:
                groups.append(word_group)
            run_start = index
    return groups


def render_preview_png(word_groups: list[list[dict]], canvas_height: int) -> bytes:
    image = Image.new("RGB", (VIEW_WIDTH, canvas_height), "white")
    canvas = ImageDraw.Draw(image)
    for group in word_groups:
        for stroke in group:
            points = [(point["x"], point["y"]) for point in stroke["points"]]
            if len(points) >= 2:
                canvas.line(points, fill="black", width=PNG_STROKE_WIDTH)
            elif len(points) == 1:
                x_value, y_value = points[0]
                canvas.ellipse(
                    [
                        x_value - PNG_STROKE_WIDTH / 2,
                        y_value - PNG_STROKE_WIDTH / 2,
                        x_value + PNG_STROKE_WIDTH / 2,
                        y_value + PNG_STROKE_WIDTH / 2,
                    ],
                    fill="black",
                )
    with io.BytesIO() as png_buffer:
        image.save(png_buffer, format="PNG")
        return png_buffer.getvalue()


def render_line(
    hand: Hand, line: str, style: int, bias: float, seed: int
) -> tuple[list[list[dict]], int] | None:
    """Render one line, returning its word groups and the canvas height."""
    strokes, attended, primer_offset = generate_with_attention(
        hand, line, style=style, bias=bias, seed=seed
    )
    if len(strokes) == 0:
        return None

    attended_in_line = np.maximum(attended - primer_offset, 0)
    attended_in_line = np.minimum(attended_in_line, len(line) - 1)
    word_ids = word_id_for_each_char(line)
    word_assignment = word_ids[attended_in_line]

    coords = line_to_absolute_coords(strokes, y_offset=LINE_HEIGHT - 3 * LINE_HEIGHT / 4)
    if coords is None or len(coords) != len(word_assignment):
        return None
    word_groups = coords_and_word_assignment_to_groups(coords, word_assignment)
    if not word_groups:
        return None
    return word_groups, LINE_HEIGHT * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset_handwritten.parquet"))
    parser.add_argument("--samples", type=int, default=10, help="number of dataset samples to process")
    parser.add_argument("--bias", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hand = Hand()
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    writer = pq.ParquetWriter(args.output, SCHEMA, write_batch_size=64)
    rendered = 0
    skipped_role = 0
    skipped_chars = 0
    skipped_empty = 0

    progress = tqdm(total=args.samples, desc="samples", unit="sample")
    for sample_index, sample in enumerate(dataset):
        if sample_index >= args.samples:
            break
        progress.update(1)

        sample_rng = random.Random(args.seed * 10**9 + sample_index)
        styles_for_roles = {role: sample_rng.randrange(NUM_STYLES) for role in ROLES_TO_RENDER}
        while styles_for_roles["assistant"] == styles_for_roles["user"]:
            styles_for_roles["assistant"] = sample_rng.randrange(NUM_STYLES)

        for message in sample["messages"]:
            role = message.get("role")
            if role not in ROLES_TO_RENDER:
                skipped_role += 1
                continue
            content = message.get("content", "")
            if not text_is_renderable(content):
                skipped_chars += 1
                continue
            lines = split_into_lines(content)
            if not lines:
                skipped_empty += 1
                continue

            for line in lines:
                result = render_line(
                    hand,
                    line,
                    style=styles_for_roles[role],
                    bias=args.bias,
                    seed=sample_rng.randrange(10**9),
                )
                if result is None:
                    skipped_empty += 1
                    continue
                word_groups, canvas_height = result
                preview_bytes = render_preview_png(word_groups, canvas_height)

                row = {
                    "strokes": [word_groups],
                    "text": [line],
                    "preview": [{"bytes": preview_bytes, "path": None}],
                    "file": [hashlib.sha256(line.encode()).hexdigest()],
                }
                writer.write_table(pa.table(row, schema=SCHEMA))
                rendered += 1

    progress.close()
    writer.close()

    converted = Dataset.from_parquet(str(args.output))
    converted = converted.cast_column("preview", HFImage())
    converted.to_parquet(args.output)

    print(
        f"\nwrote {args.output} ({rendered} rows); skipped {skipped_role} non-user/assistant, "
        f"{skipped_chars} with bad chars, {skipped_empty} empty"
    )


if __name__ == "__main__":
    main()
