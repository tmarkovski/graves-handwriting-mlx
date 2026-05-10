"""SVG output. Pure NumPy/SciPy — copied from upstream `drawing.py` and
`demo.py:_draw` so we don't depend on the cloned TF repo at runtime."""

import numpy as np
import svgwrite
from scipy.signal import savgol_filter


def _align(coords):
    coords = np.copy(coords)
    x_with_intercept = np.concatenate([np.ones([coords.shape[0], 1]), coords[:, 0:1]], axis=1)
    offset, slope = np.linalg.inv(x_with_intercept.T @ x_with_intercept).dot(x_with_intercept.T).dot(coords[:, 1:2]).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords[:, :2] = coords[:, :2] @ rotation_matrix - offset
    return coords


def _denoise(coords):
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


def _offsets_to_coords(offsets):
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def write_svg(filename, stroke_offset_lists, lines, stroke_colors=None, stroke_widths=None):
    stroke_colors = stroke_colors or ["black"] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)
    line_height = 60
    view_width = 1000
    view_height = line_height * (len(stroke_offset_lists) + 1)

    drawing = svgwrite.Drawing(filename=filename)
    drawing.viewbox(width=view_width, height=view_height)
    drawing.add(drawing.rect(insert=(0, 0), size=(view_width, view_height), fill="white"))

    initial_coord = np.array([0.0, -(3 * line_height / 4)])
    for offsets, line, color, width in zip(stroke_offset_lists, lines, stroke_colors, stroke_widths):
        if not line:
            initial_coord[1] -= line_height
            continue

        offsets = np.array(offsets, dtype=np.float64).copy()
        offsets[:, :2] *= 1.5
        coords = _offsets_to_coords(offsets)
        coords = _denoise(coords)
        coords[:, :2] = _align(coords[:, :2])

        coords[:, 1] *= -1
        coords[:, :2] -= coords[:, :2].min() + initial_coord
        coords[:, 0] += (view_width - coords[:, 0].max()) / 2

        previous_eos = 1.0
        path_string = "M{},{} ".format(0, 0)
        for x_value, y_value, eos in coords:
            path_string += "{}{},{} ".format("M" if previous_eos == 1.0 else "L", x_value, y_value)
            previous_eos = eos
        path = svgwrite.path.Path(path_string)
        path = path.stroke(color=color, width=width, linecap="round").fill("none")
        drawing.add(path)

        initial_coord[1] -= line_height

    drawing.save()
