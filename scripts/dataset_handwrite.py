"""Render messages from the `Roman1111111/claude-sonnet-4.6-100000X-filtered`
HuggingFace dataset as handwriting and write to a parquet dataset.

Schema is the one from `graphite/scripts/convert_jsonl_to_parquet.py`
extended with an `author` column identifying which of the 13 bundled
handwriting styles produced the row:

    strokes : list<list<struct{points: list<struct{x: float32, y: float32}>}>>
    text    : string
    preview : struct{bytes: binary, path: string}    -- raw PNG bytes
    file    : string                                  -- sha256(text)
    author  : int32                                   -- style id 0..12

The preview column is plain PNG bytes; consumers wanting HF auto-thumbnails
can do `cast_column("preview", datasets.Image())` after loading. We don't
do it here so the script stays streaming-only and never holds the full
dataset in memory.

**One row per line.** Each user/assistant message is word-wrapped at
6 words / 75 characters (whichever hits first) and every wrapped line
is emitted as its own parquet row.
The outer `strokes` list groups by **word** — every space-separated token
in the line gets its own group; the inner list is the pen-down strokes
within that word. Words are identified by tracking which character the
attention's argmax was on at each step.

Lines are accumulated and rendered in batches of `--batch` (default 64)
to keep the GPU saturated. Messages with characters outside the model's
73-char vocabulary are skipped, as are non-user/non-assistant messages.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json
from dataclasses import asdict

from datasets import DatasetInfo, Features, Image as HFImage, Sequence, Value, load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

from longhand_mlx import Hand
from longhand_mlx.alphabet import alphabet, encode_ascii
from longhand_mlx.draw import _align, _denoise, offsets_to_coords
from longhand_mlx.hand import load_style
from longhand_mlx.model import ALPHABET_SIZE

DATASET_NAME = "Roman1111111/claude-sonnet-4.6-100000X-filtered"
NUM_STYLES = 13
LINE_CHAR_LIMIT = 75
LINE_WORD_LIMIT = 6
ROLES_TO_RENDER = {"user", "assistant"}
VALID_CHARS = set(alphabet)

PNG_STROKE_WIDTH = 2
PNG_PADDING = 8
STEPS_PER_CHAR = 40
EVAL_EVERY = 8

POINT_TYPE = pa.struct([("x", pa.float32()), ("y", pa.float32())])
STROKE_TYPE = pa.struct([("points", pa.list_(POINT_TYPE))])
SCHEMA = pa.schema(
    [
        ("strokes", pa.list_(pa.list_(STROKE_TYPE))),
        ("text", pa.string()),
        ("preview", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("file", pa.string()),
        ("author", pa.int32()),
    ]
)

# HF datasets stores feature metadata inside the parquet schema under the
# "huggingface" key. Embedding it at writer creation (rather than via a
# post-hoc Dataset.cast_column round-trip) lets the script stay streaming —
# we never need to load the file back into memory just to mark `preview` as
# an Image column.
HF_FEATURES = Features(
    {
        "strokes": Sequence(
            Sequence({"points": Sequence({"x": Value("float32"), "y": Value("float32")})})
        ),
        "text": Value("string"),
        "preview": HFImage(),
        "file": Value("string"),
        "author": Value("int32"),
    }
)
SCHEMA = SCHEMA.with_metadata(
    {
        "huggingface": json.dumps(
            {"info": {"features": asdict(DatasetInfo(features=HF_FEATURES))["features"]}}
        )
    }
)


@dataclass
class LineRequest:
    line: str
    style: int


def text_is_renderable(text: str) -> bool:
    return all(character in VALID_CHARS for character in text)


def split_into_lines(
    text: str,
    char_limit: int = LINE_CHAR_LIMIT,
    word_limit: int = LINE_WORD_LIMIT,
) -> list[str]:
    """Greedy word-wrap with both a word-count and a character-count cap.
    Source newlines also force a break."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        chunk: list[str] = []
        chunk_chars = 0
        for word in paragraph.split():
            if len(word) > char_limit:
                word = word[:char_limit]
            projected = chunk_chars + len(word) + (1 if chunk else 0)
            if chunk and (len(chunk) >= word_limit or projected > char_limit):
                lines.append(" ".join(chunk))
                chunk = [word]
                chunk_chars = len(word)
            else:
                chunk.append(word)
                chunk_chars = projected
        if chunk:
            lines.append(" ".join(chunk))
    return lines


def word_id_for_each_char(text: str) -> np.ndarray:
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


def generate_batch(
    hand: Hand,
    requests: list[LineRequest],
    bias: float,
    batch_seed: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Run the model on a batch of lines in lockstep.

    Returns one `(strokes [Ti, 3], attended_chars [Ti], primer_offset)` per
    request, with each row trimmed to its own termination point. Per-row
    primers and char sequences are zero-padded to common lengths; per-row
    masking keeps each sample's state frozen once it terminates.
    """
    batch_size = len(requests)
    primers_and_text = [load_style(request.style) for request in requests]
    primer_offsets = [len(primer_text) + 1 for _, primer_text in primers_and_text]
    encoded_per_row = [
        encode_ascii(primer_text + " " + request.line)
        for (_, primer_text), request in zip(primers_and_text, requests)
    ]
    primers = [primer_strokes for primer_strokes, _ in primers_and_text]

    max_char_length = max(len(encoded) for encoded in encoded_per_row)
    max_prime_length = max(len(primer) for primer in primers)
    max_line_length = max(len(request.line) for request in requests)
    max_steps = STEPS_PER_CHAR * max_line_length

    chars_padded = np.zeros((batch_size, max_char_length), dtype=np.int32)
    char_lengths = np.zeros(batch_size, dtype=np.int32)
    primer_padded = np.zeros((batch_size, max_prime_length, 3), dtype=np.float32)
    prime_lengths = np.zeros(batch_size, dtype=np.int32)
    for index, (encoded, primer) in enumerate(zip(encoded_per_row, primers)):
        chars_padded[index, : len(encoded)] = encoded
        char_lengths[index] = len(encoded)
        primer_padded[index, : len(primer)] = primer
        prime_lengths[index] = len(primer)

    chars_index = mx.array(chars_padded)
    chars_onehot = mx.take(mx.eye(ALPHABET_SIZE, dtype=mx.float32), chars_index, axis=0)
    char_positions = mx.arange(max_char_length, dtype=mx.float32).reshape(1, 1, max_char_length)
    char_mask_2d = (
        mx.arange(max_char_length, dtype=mx.int32).reshape(1, max_char_length)
        < mx.array(char_lengths).reshape(-1, 1)
    ).astype(mx.float32)
    char_mask = mx.expand_dims(char_mask_2d, 2)
    bias_per_sample = mx.full((batch_size,), bias, dtype=mx.float32)

    state = hand.cell.initial_state(batch_size, max_char_length)
    primer_strokes_mx = mx.array(primer_padded)
    prime_lengths_mx = mx.array(prime_lengths)

    @mx.compile
    def prime_step(state, inputs, chars_onehot, char_positions, char_mask):
        return hand.cell.step(state, inputs, chars_onehot, char_positions, char_mask)

    @mx.compile
    def free_step(state, inputs, chars_onehot, char_positions, char_mask, bias_per_sample, key):
        new_state = hand.cell.step(state, inputs, chars_onehot, char_positions, char_mask)
        next_inputs = hand.cell.head.sample(new_state[4], bias_per_sample, key)
        return new_state, next_inputs

    # Priming with per-row "still priming" mask.
    for time_index in range(max_prime_length):
        new_state = prime_step(
            state, primer_strokes_mx[:, time_index, :], chars_onehot, char_positions, char_mask
        )
        active = mx.expand_dims(
            (mx.array(np.array([time_index], dtype=np.int32)) < prime_lengths_mx).astype(mx.float32), 1
        )
        state = tuple(mx.where(active, new, old) for new, old in zip(new_state, state))
        if (time_index + 1) % EVAL_EVERY == 0:
            mx.eval(state)
    mx.eval(state)

    key = mx.random.key(int(batch_seed))
    key, sample_key = mx.random.split(key, 2)
    last_input = hand.cell.head.sample(state[4], bias_per_sample, sample_key)
    mx.eval(last_input)

    strokes_buffer = np.zeros((batch_size, max_steps, 3), dtype=np.float32)
    attention_buffer = np.zeros((batch_size, max_steps), dtype=np.int32)
    done = np.zeros(batch_size, dtype=bool)
    char_lengths_np = char_lengths.copy()
    pending: list[tuple[int, mx.array, mx.array]] = []

    for step_index in range(max_steps):
        key, step_key = mx.random.split(key, 2)
        new_state, next_inputs = free_step(
            state, last_input, chars_onehot, char_positions, char_mask, bias_per_sample, step_key
        )
        active = mx.expand_dims(mx.array((~done).astype(np.float32)), 1)
        state = tuple(mx.where(active, new, old) for new, old in zip(new_state, state))
        next_inputs = next_inputs * active
        last_input = next_inputs
        pending.append((step_index, next_inputs, state[8]))

        if len(pending) >= EVAL_EVERY or step_index == max_steps - 1:
            mx.eval(*(stroke for _, stroke, _ in pending), *(phi for _, _, phi in pending))
            for buffered_index, stroke_array, phi_array in pending:
                stroke_np = np.array(stroke_array)
                phi_np = np.array(phi_array)
                strokes_buffer[:, buffered_index, :] = stroke_np
                attention_buffer[:, buffered_index] = phi_np.argmax(axis=1)
                attention_at_last = attention_buffer[:, buffered_index] >= char_lengths_np - 1
                eos_fired = stroke_np[:, 2] == 1.0
                past_last = attention_buffer[:, buffered_index] >= char_lengths_np
                done = done | (attention_at_last & eos_fired) | past_last
            pending = []
            if done.all():
                break

    results: list[tuple[np.ndarray, np.ndarray, int]] = []
    for row_index in range(batch_size):
        keep = ~np.all(strokes_buffer[row_index] == 0.0, axis=1)
        results.append(
            (
                strokes_buffer[row_index][keep],
                attention_buffer[row_index][keep],
                primer_offsets[row_index],
            )
        )
    return results


STROKE_SCALE = 2.0


def line_to_absolute_coords(strokes_offsets: np.ndarray) -> np.ndarray | None:
    """Cumsum offsets to absolute coords, denoise, deslant, flip Y for screen
    convention, shift so the very first point sits at (0, 0), and apply the
    same 2x scale used by graphite's reference script. Returns `[N, 3]`
    `(x, y, eos)` — no centering, no fixed canvas."""
    if len(strokes_offsets) == 0:
        return None
    coords = offsets_to_coords(strokes_offsets.astype(np.float64))
    coords = _denoise(coords)
    coords[:, :2] = _align(coords[:, :2])
    coords[:, 1] *= -1
    coords[:, 0] -= coords[0, 0]
    coords[:, 1] -= coords[0, 1]
    coords[:, :2] *= STROKE_SCALE
    return coords


def coords_and_word_assignment_to_groups(
    coords: np.ndarray, word_assignment: np.ndarray
) -> list[list[dict]]:
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


DOTTED_CHARS = set("ij?!;:.")
TINY_STROKE_DIAMETER = 3.0


def stroke_diameter(stroke: dict) -> float:
    """Largest extent of the stroke's bounding box. A single point counts as 0."""
    points = stroke["points"]
    if len(points) <= 1:
        return 0.0
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def trim_tiny_extremes(word_groups: list[list[dict]], text: str) -> list[list[dict]]:
    """Drop the very first or very last stroke if it's a tiny artifact dot
    (under `TINY_STROKE_DIAMETER`), unless the text begins/ends with a
    character that legitimately has a dot (i, j, !, ?, :, ;, .).

    After trimming, re-anchor so the first surviving point sits at (0, 0).
    """
    if not word_groups:
        return word_groups
    flat: list[tuple[int, int]] = [
        (word_index, stroke_index)
        for word_index, group in enumerate(word_groups)
        for stroke_index in range(len(group))
    ]
    if not flat:
        return word_groups

    stripped = text.strip()
    text_first = stripped[:1].lower() if stripped else ""
    text_last = stripped[-1:].lower() if stripped else ""

    drop_indices: set[int] = set()
    first_word_index, first_stroke_index = flat[0]
    if (
        text_first not in DOTTED_CHARS
        and stroke_diameter(word_groups[first_word_index][first_stroke_index]) < TINY_STROKE_DIAMETER
    ):
        drop_indices.add(0)
    if len(flat) > 1:
        last_word_index, last_stroke_index = flat[-1]
        if (
            text_last not in DOTTED_CHARS
            and stroke_diameter(word_groups[last_word_index][last_stroke_index]) < TINY_STROKE_DIAMETER
        ):
            drop_indices.add(len(flat) - 1)

    if not drop_indices:
        return word_groups

    new_groups: list[list[dict]] = [[] for _ in word_groups]
    for index, (word_index, stroke_index) in enumerate(flat):
        if index in drop_indices:
            continue
        new_groups[word_index].append(word_groups[word_index][stroke_index])
    new_groups = [group for group in new_groups if group]

    # Re-anchor so the first remaining point is at (0, 0).
    if new_groups:
        anchor = new_groups[0][0]["points"][0]
        anchor_x, anchor_y = anchor["x"], anchor["y"]
        if anchor_x or anchor_y:
            for group in new_groups:
                for stroke in group:
                    for point in stroke["points"]:
                        point["x"] -= anchor_x
                        point["y"] -= anchor_y
    return new_groups


def render_preview_png(word_groups: list[list[dict]]) -> bytes:
    """Render a tight-bbox PNG of the line, just big enough to fit the strokes
    plus a small padding."""
    all_x: list[float] = []
    all_y: list[float] = []
    for group in word_groups:
        for stroke in group:
            for point in stroke["points"]:
                all_x.append(point["x"])
                all_y.append(point["y"])
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max(int(max_x - min_x + 2 * PNG_PADDING), 1)
    height = max(int(max_y - min_y + 2 * PNG_PADDING), 1)
    offset_x = -min_x + PNG_PADDING
    offset_y = -min_y + PNG_PADDING

    image = Image.new("RGB", (width, height), "white")
    canvas = ImageDraw.Draw(image)
    for group in word_groups:
        for stroke in group:
            points = [(point["x"] + offset_x, point["y"] + offset_y) for point in stroke["points"]]
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


def line_strokes_to_word_groups(
    line: str, strokes: np.ndarray, attended: np.ndarray, primer_offset: int
) -> list[list[dict]] | None:
    if len(strokes) == 0:
        return None
    attended_in_line = np.maximum(attended - primer_offset, 0)
    attended_in_line = np.minimum(attended_in_line, len(line) - 1)
    word_ids = word_id_for_each_char(line)
    word_assignment = word_ids[attended_in_line]

    coords = line_to_absolute_coords(strokes)
    if coords is None or len(coords) != len(word_assignment):
        return None
    word_groups = coords_and_word_assignment_to_groups(coords, word_assignment)
    if not word_groups:
        return None
    word_groups = trim_tiny_extremes(word_groups, line)
    return word_groups or None


def flush_batch(
    hand: Hand,
    pending: list[LineRequest],
    bias: float,
    batch_seed: int,
    writer: pq.ParquetWriter,
) -> tuple[int, int]:
    """Generate a batch on the GPU, build the rows, and write them as a single
    parquet row group. Memory peaks at one batch's worth of strokes + previews
    and drops back down on return."""
    if not pending:
        return 0, 0
    results = generate_batch(hand, pending, bias=bias, batch_seed=batch_seed)

    columns: dict[str, list] = {"strokes": [], "text": [], "preview": [], "file": [], "author": []}
    skipped_empty = 0
    for request, (strokes, attended, primer_offset) in zip(pending, results):
        word_groups = line_strokes_to_word_groups(request.line, strokes, attended, primer_offset)
        if word_groups is None:
            skipped_empty += 1
            continue
        columns["strokes"].append(word_groups)
        columns["text"].append(request.line)
        columns["preview"].append({"bytes": render_preview_png(word_groups), "path": None})
        columns["file"].append(hashlib.sha256(request.line.encode()).hexdigest())
        columns["author"].append(request.style)

    rendered = len(columns["text"])
    if rendered:
        writer.write_table(pa.table(columns, schema=SCHEMA))
    return rendered, skipped_empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset_handwritten.parquet"))
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="target number of rows in the output parquet (one row per rendered line)",
    )
    parser.add_argument("--bias", type=float, default=0.9)
    parser.add_argument("--batch", type=int, default=64, help="lines rendered per GPU batch")
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
    rng = random.Random(args.seed)

    pending: list[LineRequest] = []

    # Bar tracks lines committed toward the target rather than lines actually
    # written to parquet. Each accepted line is incremented immediately so the
    # bar moves smoothly during the accumulation phase between flushes (which
    # is most of the wall time when --batch is large). On the rare occasion
    # that a flushed line produces no strokes, we step the bar back to keep
    # the final count honest.
    progress = tqdm(total=args.samples, desc="rows", unit="row")
    target_reached = False

    for sample in dataset:
        if target_reached:
            break
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
                pending.append(LineRequest(line=line, style=rng.randrange(NUM_STYLES)))
                progress.update(1)
                # Cap the buffer at whatever is still needed so we don't
                # overshoot when --batch > remaining target.
                flush_at = min(args.batch, args.samples - rendered)
                if len(pending) >= flush_at:
                    new_rendered, new_empty = flush_batch(
                        hand, pending, bias=args.bias, batch_seed=rng.randrange(10**9), writer=writer
                    )
                    rendered += new_rendered
                    skipped_empty += new_empty
                    if new_empty:
                        progress.update(-new_empty)
                    pending = []
                    if rendered >= args.samples:
                        target_reached = True
                        break
            if target_reached:
                break

    # If the dataset ran dry before we hit the target, flush whatever's left.
    if pending and rendered < args.samples:
        new_rendered, new_empty = flush_batch(
            hand, pending[: args.samples - rendered], bias=args.bias, batch_seed=rng.randrange(10**9), writer=writer
        )
        rendered += new_rendered
        skipped_empty += new_empty
        progress.update(new_rendered)

    progress.close()
    writer.close()

    print(
        f"\nwrote {args.output} ({rendered} rows); skipped {skipped_role} non-user/assistant, "
        f"{skipped_chars} with bad chars, {skipped_empty} empty"
    )


if __name__ == "__main__":
    main()
