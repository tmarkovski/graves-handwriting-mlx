"""Render messages from the `Roman1111111/claude-sonnet-4.6-100000X-filtered`
HuggingFace dataset as handwriting.

For each sample, every user/assistant message is split into ≤75-char lines on
word boundaries and rendered with a randomly chosen style. Messages that
contain any character outside the model's 73-char vocabulary are skipped.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset

from longhand_mlx import Hand
from longhand_mlx.alphabet import alphabet
from longhand_mlx.draw import render_svg

DATASET_NAME = "Roman1111111/claude-sonnet-4.6-100000X-filtered"
NUM_STYLES = 13
LINE_LIMIT = 75
ROLES_TO_RENDER = {"user", "assistant"}
VALID_CHARS = set(alphabet)


def text_is_renderable(text: str) -> bool:
    return all(character in VALID_CHARS for character in text)


def split_into_lines(text: str, limit: int = LINE_LIMIT) -> list[str]:
    """Greedy word-wrap. Newlines in the source split too. Single tokens
    longer than `limit` are truncated rather than overflowing."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset_handwritten"))
    parser.add_argument("--samples", type=int, default=10, help="number of dataset samples to process")
    parser.add_argument("--bias", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    hand = Hand()

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    rendered = 0
    skipped_role = 0
    skipped_chars = 0
    skipped_empty = 0
    for sample_index, sample in enumerate(dataset):
        if sample_index >= args.samples:
            break
        for message_index, message in enumerate(sample["messages"]):
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

            style = rng.randrange(NUM_STYLES)
            strokes = hand.write(
                lines,
                biases=[args.bias] * len(lines),
                styles=[style] * len(lines),
                seed=rng.randrange(10**9),
            )
            svg = render_svg(strokes, lines)
            file_path = args.output / f"sample{sample_index:04d}_msg{message_index:02d}_{role}_style{style:02d}.svg"
            file_path.write_text(svg)
            print(f"wrote {file_path.name}  ({len(lines)} lines, style {style})")
            rendered += 1

    print(
        f"\nrendered {rendered}; skipped {skipped_role} non-user/assistant, "
        f"{skipped_chars} with bad chars, {skipped_empty} empty"
    )


if __name__ == "__main__":
    main()
