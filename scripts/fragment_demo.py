"""Fragmented generation: write 'hello world' as two independent calls.

The generator state lives on the `HandStream` instance. Pause is "stop calling
advance"; resume is "call advance again."
"""

from pathlib import Path

import numpy as np

from longhand_mlx import Hand
from longhand_mlx.draw import render_svg

hand = Hand()
stream = hand.stream("hello world", bias=0.75, seed=1)

fragment_one = stream.advance(until_char=5)  # stop after 'hello '
print(f"fragment 1: {fragment_one.shape[0]} strokes")

fragment_two = stream.advance()
print(f"fragment 2: {fragment_two.shape[0]} strokes, done={stream.done}")

combined = np.concatenate([fragment_one, fragment_two], axis=0)
Path("out_fragmented.svg").write_text(render_svg([combined], ["hello world"]))
print("wrote out_fragmented.svg")
