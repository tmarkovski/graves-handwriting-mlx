# graves-handwriting-mlx

Graves-style handwriting synthesis, ported to Apple MLX.

A clean reimplementation of the inference path from
[sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis):
a 3-layer LSTM driven by a soft-monotonic Gaussian-window attention over the
input text, with a 20-component mixture density head producing the next pen
offset and pen-up probability. Pre-converted weights and style examples are
bundled with the package, so there is no TensorFlow dependency at runtime.

---

## Install

```sh
uv sync
```

That's it. The trained weights (~14 MB) and 13 style examples ship inside the
package at `graves_handwriting_mlx/data/`.

## Quick start

```python
from pathlib import Path
from graves_handwriting_mlx import Hand
from graves_handwriting_mlx.draw import render_svg

hand = Hand()
strokes = hand.write(["the quick brown fox jumps over the lazy dog"])
Path("out.svg").write_text(render_svg(strokes, ["the quick brown fox..."]))
```

`Hand.write` returns the model output as a list of NumPy arrays — one
`[T, 3]` array of `(Δx, Δy, eos)` stroke offsets per line. What you do with
those is up to you; rendering to SVG is one option among many. The first
`write` call constructs the model and compiles the per-step function
(~0.3 s warmup); subsequent calls reuse both.

### The stroke format

Each row of the returned array is one timestep of the writing pen:

| column | meaning                                                 |
|-------:|---------------------------------------------------------|
|  `Δx`  | horizontal pen displacement since the previous timestep |
|  `Δy`  | vertical pen displacement (positive = up)               |
|  `eos` | `1.0` if the pen lifts after this stroke, else `0.0`    |

The values are *offsets*, not absolute coordinates. To get the path the pen
actually traces, cumulatively sum the first two columns
(`graves_handwriting_mlx.draw.offsets_to_coords` does this). Each "stroke" in the
visual sense — a continuous pen-down segment — is a run of rows separated
by `eos == 1.0`. The model emits its end-of-text signal by raising `eos` on
its final stroke; trailing all-zero rows (already trimmed by `Hand.write`)
correspond to terminated samples in a batch.

`render_svg` is one consumer of this format; you could just as easily plot
it with matplotlib, animate it with `manim`, drive a pen plotter, or feed it
to a downstream model. Keeping the pipeline open at the stroke level is
intentional.

## Lines, biases, styles

`Hand.write` takes a list of lines and produces one stroke array per line.
Most parameters are per-line lists — pass one entry per line. The model is
deterministic given a `seed`; vary the seed for different renderings of the
same text.

```python
lines = ["Dear Anya,", "I hope this finds you well.", "Yours, Ilya"]
strokes = hand.write(
    lines,
    biases=[0.75, 0.75, 0.5],
    styles=[9, 9, 9],
    seed=42,
)
svg = render_svg(strokes, lines, stroke_colors=["black", "black", "blue"])
Path("letter.svg").write_text(svg)
```

### Bias — neatness vs. wildness

Each line takes a `bias` between `0.0` and roughly `1.5`. It applies the
Graves "sharpening trick": at each timestep it widens the highest-probability
mixture component and narrows its standard deviation. In practice:

| bias  | result                                                |
|------:|-------------------------------------------------------|
|  0.0  | maximally diverse — illegible at the extreme          |
|  0.5  | natural-looking, occasionally messy                   |
|  0.75 | a good default — neat handwriting (used in the demos) |
|  1.0+ | very neat, almost mechanical; loses character         |

Bias is a per-line parameter, so different lines can have different
neatness. Defaults to `0.5` if omitted.

### Styles — handwriting that looks like a real person's

The model can be primed with a short stroke sequence written in some target
person's hand. After priming, the LSTM/attention state is conditioned on
that style, and continuing the generation produces handwriting in the same
visual idiom — same slant, loop shape, letter spacing, baseline drift.

13 style examples are bundled. Each is a real handwriting sample that was
recorded once with the priming text below:

| style | primer text                                     |
|------:|-------------------------------------------------|
|     0 | thought that vengeance                          |
|     1 | So says the Times                               |
|     2 | House was crowded.                              |
|     3 | A stronger finish                               |
|     4 | now eased away                                  |
|     5 | Aurelius Antonius about A.D                     |
|     6 | Byron, had said                                 |
|     7 | I do not know of                                |
|     8 | Holmbridge Vicarage,                            |
|     9 | two-twenty carried, for him,                    |
|    10 | "Just you keep thinking                         |
|    11 | Royal family so often seems to find             |
|    12 | tomorrow. How selfish of me. It                 |

Pick by integer ID; the primer text itself isn't rendered, only its visual
style is inherited. If you omit `styles=`, the model writes in a generic
"average" hand.

```python
# Same text, three different hands
for style_id in [3, 7, 12]:
    strokes = hand.write(["hello world"], biases=[0.75], styles=[style_id])
    Path(f"hello_{style_id}.svg").write_text(render_svg(strokes, ["hello world"]))
```

### Allowed characters

The vocabulary is 73 characters: lowercase a–z, uppercase letters
**except `Q`, `X`, and `Z`**, digits 0–9, and a small punctuation set
(`` ' " # ( ) , - . : ; ! ? ``). Any other character — emoji, accented
letters, tabs — will raise `ValueError`. Each line is capped at 75
characters.

Why no `Q`, `X`, `Z` capitals? They're nearly absent from the IAM On-Line
Handwriting Database the model was trained on, and the vocabulary is baked
into the trained weights (the alphabet size is wired into the LSTM input
dimensions). Adding them would require retraining on a corpus that contains
them.

## Streaming generation

Sometimes you want to generate one piece of a line at a time — for animating
the writing, paginating output, or keeping latency low while you display
progress. The `HandStream` API holds the model state alive between calls:

```python
stream = hand.stream("hello world", bias=0.75, style=9, seed=0)

first  = stream.advance(until_char=5)  # stop after 'hello '
second = stream.advance()              # finish
all_strokes = np.concatenate([first, second], axis=0)
```

`stream.advance(until_char=N)` runs the model until its attention crosses
character index `N` (so it'll have just finished writing the N-th character).
`stream.advance()` with no argument runs to natural end-of-text. The state
lives on the `stream` object; calling `advance` again resumes seamlessly.

Each `advance` returns a NumPy array of shape `[num_strokes, 3]` containing
only the strokes produced by that call. Concatenating them is equivalent to
having generated the whole thing in one shot — same RNG path, same output.

A few useful shapes:

```python
# Generate one character at a time, e.g. for an animation frame
stream = hand.stream("dear anya", bias=0.75, style=9)
for character_index in range(1, len("dear anya") + 1):
    fragment = stream.advance(until_char=character_index)
    # render fragment...

# Cap the work per call to keep UI responsive
stream = hand.stream("a long sentence...", bias=0.75)
while not stream.done:
    chunk = stream.advance(max_steps=64)
    # process chunk...
```

The `done` property is `True` once the model has emitted its end-of-text
signal (attention reached the last character + pen-up). Subsequent
`advance` calls will return an empty array.

## The lower-level Generator

`Hand.write` and `HandStream` are both thin facades over a single primitive,
`graves_handwriting_mlx.generator.Generator`. If you need to drive batched generation
yourself — e.g. to render thousands of lines in parallel, or with custom
stop conditions — instantiate it directly:

```python
import numpy as np
from graves_handwriting_mlx import Hand
from graves_handwriting_mlx.generator import Generator
from graves_handwriting_mlx.alphabet import encode_ascii

hand = Hand()

batch = ["one", "two", "three"]
chars = np.zeros((3, 120), dtype=np.int32)
char_lengths = np.zeros(3, dtype=np.int32)
for index, line in enumerate(batch):
    encoded = encode_ascii(line)
    chars[index, : len(encoded)] = encoded
    char_lengths[index] = len(encoded)

generator = Generator(
    hand.cell,
    chars=chars,
    char_lengths=char_lengths,
    biases=np.full(3, 0.75, dtype=np.float32),
    seed=0,
)
strokes = generator.advance(max_steps=200)  # [3, T, 3]
```

`advance(max_steps, stop_when=None)` runs up to `max_steps` cell steps and
returns the produced strokes. The optional `stop_when(phi, last_stroke)`
callback receives the per-batch attention distribution and last sampled
stroke as NumPy arrays and returns a `[batch]` boolean mask; the loop exits
early when every still-active sample has signaled stop. Default termination
(attention reached the last character + pen-up) is always active.

## Performance

Stroke-steps per second on an M1 Pro (14-core GPU), single warm call:

| batch |  steps/s |
|------:|---------:|
|     1 |    1.5 k |
|    16 |   13.8 k |
|   256 |  105.8 k |
|  4096 |  161.4 k |

Throughput saturates around 160 k steps/s — roughly 4.8 TFLOPS sustained on
a ~5 TFLOPS GPU, near the matmul ceiling. Single-line latency (~0.5 s for 30
characters) is bound by per-step Python/dispatch overhead, not compute, so
the way to go faster is to batch independent lines into one `Hand.write`
call rather than looping in Python.

## Layout

```
graves_handwriting_mlx/
  modules.py      LSTMCell, GaussianWindowAttention, MixtureDensityHead
  model.py        HandwritingCell — composes the three modules
  generator.py    Generator — the basic autoregressive primitive
  hand.py         Hand — multi-line one-shot facade
  stream.py       HandStream — single-line fragmentable facade
  draw.py         SVG output (denoise, deslant, page layout)
  weights.py      load_weights() — finds the bundled npz
  alphabet.py    character vocabulary + encoding
  data/           bundled weights.npz + style examples
scripts/
  convert_checkpoint.py   one-shot TF -> MLX (only file using TensorFlow)
  fragment_demo.py        in-process two-fragment "hello world" demo
```

## Credit

Trained weights and style examples come from
[sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis),
which in turn implements
[Graves 2013, Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850).
This repository is just an MLX inference port of that work.
