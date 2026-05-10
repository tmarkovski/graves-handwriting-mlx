"""Render messages from the `Roman1111111/claude-sonnet-4.6-100000X-filtered`
HuggingFace dataset as handwriting and write to a parquet dataset.

Schema matches `graphite/scripts/convert_jsonl_to_parquet.py`:

    strokes : list<list<struct{points: list<struct{x: float32, y: float32}>}>>
    text    : string
    preview : struct{bytes: binary, path: string}    -> cast to HF Image
    file    : string                                  -> sha256(text)

Each row is one rendered message. The outer `strokes` list groups by line;
the inner list is the individual pen-down strokes within that line.
Messages containing characters outside the model's 73-char vocabulary are
skipped, as are non-user/non-assistant messages.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Image as HFImage, load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

from longhand_mlx import Hand
from longhand_mlx.alphabet import alphabet
from longhand_mlx.draw import _align, _denoise, offsets_to_coords

DATASET_NAME = "Roman1111111/claude-sonnet-4.6-100000X-filtered"
NUM_STYLES = 13
LINE_LIMIT = 75
ROLES_TO_RENDER = {"user", "assistant"}
VALID_CHARS = set(alphabet)

SCALE = 1.5
LINE_HEIGHT = 60
VIEW_WIDTH = 1000
PNG_STROKE_WIDTH = 2

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


def line_to_absolute_coords(offsets: np.ndarray, y_offset: float) -> np.ndarray | None:
    """Apply the same denoise/deslant pipeline as `render_svg` to one line and
    place it at the requested vertical position. Returns absolute `[N, 3]`
    `(x, y, eos)` coordinates, or None if the line collapsed to nothing."""
    if len(offsets) == 0:
        return None
    scaled = offsets.astype(np.float64).copy()
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


def coords_to_word_group(coords: np.ndarray) -> list[dict]:
    """Split coords into individual pen-down strokes by `eos == 1` boundaries
    and emit each as a `{points: [{x, y}, ...]}` struct."""
    split_indices = np.where(coords[:, 2] == 1)[0] + 1
    word_group: list[dict] = []
    for stroke_coords in np.split(coords[:, :2], split_indices, axis=0):
        if len(stroke_coords) == 0:
            continue
        word_group.append(
            {"points": [{"x": float(x_value), "y": float(y_value)} for x_value, y_value in stroke_coords]}
        )
    return word_group


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

            style = styles_for_roles[role]
            stroke_offsets_per_line = hand.write(
                lines,
                biases=[args.bias] * len(lines),
                styles=[style] * len(lines),
                seed=sample_rng.randrange(10**9),
            )

            word_groups: list[list[dict]] = []
            for line_index, line_offsets in enumerate(stroke_offsets_per_line):
                y_offset = (line_index + 1) * LINE_HEIGHT - 3 * LINE_HEIGHT / 4
                coords = line_to_absolute_coords(line_offsets, y_offset)
                if coords is None:
                    continue
                word_group = coords_to_word_group(coords)
                if word_group:
                    word_groups.append(word_group)

            if not word_groups:
                skipped_empty += 1
                continue

            canvas_height = (len(stroke_offsets_per_line) + 1) * LINE_HEIGHT
            preview_bytes = render_preview_png(word_groups, canvas_height)
            row = {
                "strokes": [word_groups],
                "text": [content],
                "preview": [{"bytes": preview_bytes, "path": None}],
                "file": [hashlib.sha256(content.encode()).hexdigest()],
            }
            writer.write_table(pa.table(row, schema=SCHEMA))
            rendered += 1

    progress.close()
    writer.close()

    # Re-save through HF datasets to embed the proper Image() metadata on
    # the preview column.
    converted = Dataset.from_parquet(str(args.output))
    converted = converted.cast_column("preview", HFImage())
    converted.to_parquet(args.output)

    print(
        f"\nwrote {args.output} ({rendered} rows); skipped {skipped_role} non-user/assistant, "
        f"{skipped_chars} with bad chars, {skipped_empty} empty"
    )


if __name__ == "__main__":
    main()
