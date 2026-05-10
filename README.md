# longhand-mlx

Graves-style handwriting synthesis, ported to Apple MLX.

A clean reimplementation of the inference path from
[sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis):
3-layer LSTM with a Gaussian-window attention over characters and a mixture
density head over pen offsets. Pre-converted weights and style examples are
bundled with the package — no TensorFlow at runtime.

## Install

```sh
uv sync
```

## Use

```python
from longhand_mlx import Hand

hand = Hand()
hand.write(
    filename="out.svg",
    lines=["the quick brown fox", "jumps over the lazy dog"],
    biases=[0.75, 0.75],   # higher = neater, lower = wilder
    styles=[9, 9],          # optional; one of the bundled styles
)
```

For incremental, fragmentable output (e.g. animation, progressive rendering):

```python
stream = hand.stream("hello world", bias=0.75)
first_half = stream.advance(until_char=5)   # stop just past 'hello '
second_half = stream.advance()               # finish
```

The stream object holds the live LSTM/attention state — call `advance` again
later to keep going. Run `uv run python main.py` for the multi-line demo.

## Performance

Stroke-steps per second on an M1 Pro (14-core GPU):

| batch |  steps/s |
|------:|---------:|
|     1 |    1.5 k |
|    16 |   13.8 k |
|   256 |  105.8 k |
|  4096 |  161.4 k |

Throughput saturates around 160 k steps/s — roughly 4.8 TFLOPS sustained
on a ~5 TFLOPS GPU, near the matmul ceiling.

## Layout

```
longhand_mlx/
  modules.py     LSTMCell, GaussianWindowAttention, MixtureDensityHead
  model.py       HandwritingCell — composes the three modules
  generator.py   Generator — the basic autoregressive primitive
  hand.py        Hand — multi-line one-shot facade
  stream.py     HandStream — single-line fragmentable facade
  draw.py        SVG output (denoise, deslant, layout)
  weights.py     load_weights() — finds the bundled npz
  alphabet.py    character vocabulary
  data/          bundled weights.npz + style examples
scripts/
  convert_checkpoint.py   one-shot TF -> MLX (only file using TensorFlow)
  fragment_demo.py        in-process two-fragment "hello world" demo
```
