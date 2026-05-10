"""Stroke post-processing and SVG rendering.

`render_svg` returns the SVG markup as a string — it doesn't touch the
filesystem. The caller decides what to do with the result (write to disk,
serve over HTTP, embed in a notebook, etc.).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import svgwrite
from scipy.signal import savgol_filter


def _align(coords: np.ndarray) -> np.ndarray:
    coords = np.copy(coords)
    x_with_intercept = np.concatenate([np.ones([coords.shape[0], 1]), coords[:, 0:1]], axis=1)
    offset, slope = (
        np.linalg.inv(x_with_intercept.T @ x_with_intercept)
        .dot(x_with_intercept.T)
        .dot(coords[:, 1:2])
        .squeeze()
    )
    theta = np.arctan(slope)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords[:, :2] = coords[:, :2] @ rotation_matrix - offset
    return coords


def _denoise(coords: np.ndarray) -> np.ndarray:
    split_indices = np.where(coords[:, 2] == 1)[0] + 1
    new_strokes = []
    for stroke in np.split(coords, split_indices, axis=0):
        if len(stroke) == 0:
            continue
        smoothed_x = savgol_filter(stroke[:, 0], 7, 3, mode="nearest")
        smoothed_y = savgol_filter(stroke[:, 1], 7, 3, mode="nearest")
        new_strokes.append(
            np.concatenate(
                [smoothed_x.reshape(-1, 1), smoothed_y.reshape(-1, 1), stroke[:, 2:3]],
                axis=1,
            )
        )
    return np.vstack(new_strokes)


def offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    """Cumulatively sum the (Δx, Δy) channels into absolute coordinates,
    keeping the pen-up flag intact."""
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def render_svg(
    strokes: Sequence[np.ndarray],
    lines: Sequence[str],
    *,
    stroke_colors: Sequence[str] | None = None,
    stroke_widths: Sequence[float] | None = None,
    line_height: int = 60,
    view_width: int = 1000,
    scale: float = 1.5,
) -> str:
    """Render a list of per-line stroke offsets into an SVG document string.

    `strokes[i]` is the `[T, 3]` array of `(Δx, Δy, eos)` produced for
    `lines[i]`. Lines are stacked vertically. Pass `lines` to control which
    rows are blank lines (an empty `lines[i]` skips one line of vertical
    space and ignores the corresponding `strokes[i]`).
    """
    stroke_colors = list(stroke_colors) if stroke_colors is not None else ["black"] * len(lines)
    stroke_widths = list(stroke_widths) if stroke_widths is not None else [2] * len(lines)
    view_height = line_height * (len(strokes) + 1)

    drawing = svgwrite.Drawing()
    drawing.viewbox(width=view_width, height=view_height)
    drawing.add(drawing.rect(insert=(0, 0), size=(view_width, view_height), fill="white"))

    cursor = np.array([0.0, -(3 * line_height / 4)])
    for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
        if not line:
            cursor[1] -= line_height
            continue

        offsets = np.array(offsets, dtype=np.float64).copy()
        offsets[:, :2] *= scale
        coords = offsets_to_coords(offsets)
        coords = _denoise(coords)
        coords[:, :2] = _align(coords[:, :2])
        coords[:, 1] *= -1
        coords[:, :2] -= coords[:, :2].min() + cursor
        coords[:, 0] += (view_width - coords[:, 0].max()) / 2

        previous_eos = 1.0
        path_string = "M0,0 "
        for x_value, y_value, eos in coords:
            path_string += "{}{},{} ".format("M" if previous_eos == 1.0 else "L", x_value, y_value)
            previous_eos = eos
        path = svgwrite.path.Path(path_string).stroke(color=color, width=width, linecap="round").fill("none")
        drawing.add(path)
        cursor[1] -= line_height

    return drawing.tostring()
