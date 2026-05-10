"""Minimal demo: render a few lines of handwriting with style priming."""

from pathlib import Path

from graves_handwriting_mlx import Hand
from graves_handwriting_mlx.draw import render_svg


def main() -> None:
    hand = Hand()
    lines = [
        "Now this is a story all about how",
        "My life got flipped turned upside down",
        "And I had like to take a minute, just sit right there",
        "I will tell you how I became the prince of a town called Bel-Air",
    ]
    strokes = hand.write(
        lines,
        biases=[0.75] * len(lines),
        styles=[9] * len(lines),
        seed=0,
    )
    svg = render_svg(
        strokes,
        lines,
        stroke_colors=["red", "green", "black", "blue"],
        stroke_widths=[1, 2, 1, 2],
    )
    Path("demo.svg").write_text(svg)
    print("wrote demo.svg")


if __name__ == "__main__":
    main()
