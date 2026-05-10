"""Minimal demo: render a few lines of handwriting with style priming."""

from longhand_mlx import Hand


def main():
    hand = Hand()
    lines = [
        "Now this is a story all about how",
        "My life got flipped turned upside down",
        "And I had like to take a minute, just sit right there",
        "I will tell you how I became the prince of a town called Bel-Air",
    ]
    hand.write(
        filename="demo.svg",
        lines=lines,
        biases=[0.75] * len(lines),
        styles=[9] * len(lines),
        stroke_colors=["red", "green", "black", "blue"],
        stroke_widths=[1, 2, 1, 2],
        seed=0,
    )
    print("wrote demo.svg")


if __name__ == "__main__":
    main()
