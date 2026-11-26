"""Pure, testable pooling of a RAFT optical-flow field into a fixed-length descriptor.

Why a hand-pooled descriptor, not a learned flow branch
---------------------------------------------------------
A learned flow branch (a small CNN/transformer over the raw `[2, H, W]` field)
is a second trainable network, and the project has no budget to train one
against 1,600 videos before the review -- it would need its own capacity
sweep, its own regularisation study, and its own overfitting risk, none of
which this arm can afford. Hand-pooled statistics (per-cell magnitude and a
magnitude-weighted orientation histogram, the same family of descriptor HOF/
motion-boundary-histogram optical-flow features used for two decades) keep
this arm a pure INPUT change: fixed-length, deterministic, requires no
gradient of its own, and is concatenated onto the existing pooled DINOv2
vector so the classification heads under test (`core.p1c_heads`) are
unchanged. This module never imports `torchvision` or any flow model --
`scripts/build_p1c_flow_cache.py` owns the GPU side; this module is pure
NumPy so it is testable without a GPU or a checkpoint.

Descriptor definition
----------------------
`flow_descriptor` takes one `[2, H, W]` float32 field `(u, v)`:

- `u` is horizontal displacement, positive = rightward (increasing column
  index); `v` is vertical displacement, positive = DOWNWARD (increasing row
  index) -- the image/array convention RAFT itself outputs flow in, not the
  math convention where +y is up. This matters only for interpreting which
  orientation bin a given real-world direction lands in; the descriptor
  arithmetic itself is convention-agnostic.
- The frame is split into a `grid = (rows, cols)` grid of cells. Boundaries
  follow the `np.array_split` convention: `np.array_split(np.arange(size),
  n)` when `size` does not divide evenly by `n`, so the first `size % n`
  cells get one extra row/column and the rest are `size // n` -- this module
  reuses that exact convention (`_cell_boundaries`) rather than inventing a
  different remainder rule, so it agrees with every other place in this
  codebase that has to slice something into unequal integer chunks.
- Per cell: mean magnitude, std magnitude, and an `orientation_bins`-bin
  histogram of `arctan2(v, u)` over `[-pi, pi)`, each pixel's vote weighted
  by its own magnitude and the whole histogram L1-normalised within the cell
  (sums to 1, or is all-zero when the cell has zero total magnitude -- never
  NaN: dividing by a zero total is skipped explicitly).
- Bin edges are `np.linspace(-pi, pi, orientation_bins + 1)`, i.e. bin `i`
  covers `[-pi + i * (2*pi/orientation_bins), -pi + (i+1) * (2*pi/orientation_bins))`,
  computed via `bin_index = floor((angle + pi) / bin_width) mod orientation_bins`
  (the modulo folds the closed endpoint `angle == pi`, which `arctan2` can
  return, back onto the same bin as `angle == -pi`, since they are the same
  direction). For the default 8 bins: angle 0 (pure rightward) falls in bin
  `orientation_bins // 2` (4); angle `-pi/2` (pure upward, i.e. `v` negative)
  falls in bin `orientation_bins // 4` (2); angle `+pi/2` (pure downward)
  falls in bin `3 * orientation_bins // 4` (6). Flipping the sign of the
  whole field maps `angle -> angle + pi`, which shifts every bin index by
  exactly `orientation_bins // 2` (mod `orientation_bins`).
- Two trailing global scalars: mean and 95th-percentile magnitude over the
  whole field (not per-cell -- a single frame-level summary of how much
  motion is present at all, ungated by where it is).

Total length is `grid[0] * grid[1] * (2 + orientation_bins) + 2`; the module
constant `DESCRIPTOR_LENGTH` is this value at the defaults `grid=(4, 4)`,
`orientation_bins=8` (162), and `descriptor_length(grid, orientation_bins)`
computes it for any other choice. `descriptor_field_names` returns one
human-readable name per output dimension in the same order, so a cache built
from this module is self-describing without a separate lookup table.
"""

from __future__ import annotations

import numpy as np

DEFAULT_GRID = (4, 4)
DEFAULT_ORIENTATION_BINS = 8


def descriptor_length(
    grid: tuple[int, int] = DEFAULT_GRID, orientation_bins: int = DEFAULT_ORIENTATION_BINS
) -> int:
    """Total descriptor length for a given `grid` and `orientation_bins`."""

    rows, cols = grid
    return rows * cols * (2 + orientation_bins) + 2


#: Descriptor length at the module defaults (`grid=(4, 4)`, `orientation_bins=8`): 162.
DESCRIPTOR_LENGTH = descriptor_length()


def descriptor_field_names(
    grid: tuple[int, int] = DEFAULT_GRID, orientation_bins: int = DEFAULT_ORIENTATION_BINS
) -> list[str]:
    """One human-readable name per `flow_descriptor` output dimension, same order."""

    rows, cols = grid
    names: list[str] = []
    for r in range(rows):
        for c in range(cols):
            names.append(f"cell_r{r}_c{c}_mean_magnitude")
            names.append(f"cell_r{r}_c{c}_std_magnitude")
            for b in range(orientation_bins):
                names.append(f"cell_r{r}_c{c}_orientation_bin{b}")
    names.append("global_mean_magnitude")
    names.append("global_p95_magnitude")
    return names


def _cell_boundaries(size: int, num_cells: int) -> list[tuple[int, int]]:
    """`[(start, stop), ...]` for `num_cells` chunks of `range(size)`, `np.array_split` convention.

    `H`/`W` need not divide evenly by the grid -- `np.array_split` gives the
    first `size % num_cells` chunks one extra element, the rest `size //
    num_cells`, and this reproduces exactly that split rather than inventing
    a different remainder rule.
    """

    if num_cells <= 0:
        raise ValueError(f"num_cells must be positive; got {num_cells}")
    chunks = np.array_split(np.arange(size), num_cells)
    boundaries: list[tuple[int, int]] = []
    for chunk in chunks:
        if chunk.size == 0:
            # size < num_cells: array_split pads with empty trailing chunks.
            # Record a zero-width cell at the end of the array rather than
            # raising -- flow_descriptor handles an empty cell explicitly.
            boundaries.append((size, size))
        else:
            boundaries.append((int(chunk[0]), int(chunk[-1]) + 1))
    return boundaries


def flow_descriptor(
    flow: np.ndarray,
    *,
    grid: tuple[int, int] = DEFAULT_GRID,
    orientation_bins: int = DEFAULT_ORIENTATION_BINS,
) -> np.ndarray:
    """Fixed-length float32 descriptor of one `[2, H, W]` `(u, v)` flow field.

    See the module docstring for the exact per-cell/global statistics, the
    grid-boundary convention, and the orientation-bin convention. An
    all-zero cell (no motion) yields an all-zero orientation histogram for
    that cell, never NaN.
    """

    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"flow must be [2, H, W]; got shape {flow.shape}")
    if orientation_bins <= 0:
        raise ValueError(f"orientation_bins must be positive; got {orientation_bins}")

    u, v = flow[0], flow[1]
    height, width = u.shape
    magnitude = np.sqrt(u * u + v * v)
    angle = np.arctan2(v, u)  # (-pi, pi]

    bin_width = 2.0 * np.pi / orientation_bins
    bin_index = np.floor((angle + np.pi) / bin_width).astype(np.int64) % orientation_bins

    row_bounds = _cell_boundaries(height, grid[0])
    col_bounds = _cell_boundaries(width, grid[1])

    values: list[float] = []
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            cell_magnitude = magnitude[r0:r1, c0:c1]
            cell_bins = bin_index[r0:r1, c0:c1]
            if cell_magnitude.size == 0:
                values.append(0.0)
                values.append(0.0)
                values.extend([0.0] * orientation_bins)
                continue
            values.append(float(cell_magnitude.mean()))
            values.append(float(cell_magnitude.std()))
            histogram = np.zeros(orientation_bins, dtype=np.float64)
            np.add.at(histogram, cell_bins.ravel(), cell_magnitude.ravel())
            total = histogram.sum()
            if total > 0.0:
                histogram = histogram / total
            values.extend(histogram.tolist())

    values.append(float(magnitude.mean()) if magnitude.size else 0.0)
    values.append(float(np.percentile(magnitude, 95)) if magnitude.size else 0.0)

    descriptor = np.asarray(values, dtype=np.float32)
    expected = descriptor_length(grid, orientation_bins)
    if descriptor.shape[0] != expected:
        raise RuntimeError(f"internal error: built {descriptor.shape[0]}-length descriptor, expected {expected}")
    return descriptor


def flow_descriptor_batch(
    flows: np.ndarray,
    *,
    grid: tuple[int, int] = DEFAULT_GRID,
    orientation_bins: int = DEFAULT_ORIENTATION_BINS,
) -> np.ndarray:
    """`flow_descriptor` applied to each field of a `[B, 2, H, W]` batch; `[B, DESCRIPTOR_LENGTH]`."""

    flows = np.asarray(flows, dtype=np.float32)
    if flows.ndim != 4 or flows.shape[1] != 2:
        raise ValueError(f"flows must be [B, 2, H, W]; got shape {flows.shape}")
    return np.stack(
        [flow_descriptor(flow, grid=grid, orientation_bins=orientation_bins) for flow in flows]
    )
