"""Background-tolerant loss weighting from official temporal intervals.

Why this module exists
-----------------------
docs/PROJECT_RULES.md rule 1 (label honesty) says UCF frame labels are video-level,
inherited, noisy supervision -- an annotated video's official (start, end)
event interval is the one piece of frame-level ground truth this dataset
provides, and only for a minority of videos of a handful of labels. Every
frame of an annotated positive video currently gets that video's positive
label whether or not the frame actually shows the anomaly, so a video whose
event occupies (say) 10% of its length trains the head on 90% mislabelled
frames for that class.

This module does not fix that by relabelling anything -- relabelling would
invent ground truth docs/PROJECT_RULES.md does not license (only the interval itself is
evidence; whatever is inside "not the interval" for a positive video is
merely "not known to be positive," never "known to be background"). Instead
it produces a per-row *training weight*: known background frames of the
label under study get weight 0 so their (wrong) positive label stops pushing
the classifier, while every other row -- including every row of every other
label, and every row of unannotated videos of the label under study, where
background cannot be told apart from event -- keeps weight 1, unchanged.

Definitions (used exactly as worded here elsewhere in this codebase):

- **Positive event frame**: a row whose label equals `label`, whose video has
  an official interval, and whose `source_frame_index` falls inside at least
  one (start, end) span. Weight 1, label unchanged.
- **Known background frame**: a row whose label equals `label`, whose video
  has an official interval, and whose `source_frame_index` falls outside
  every span for that video. Weight 0, label unchanged -- the row is still
  present in `rows`, still contributes to `train_row_count`, just excluded
  from the loss.
- **Unannotated video**: a row whose label equals `label` but whose video has
  no official interval. All such rows keep weight 1; there is no way to tell
  background from event in them, so nothing is zeroed.

This is deliberately the simplest background-tolerant loss available: no
lambda to tune, no auxiliary background-vs-event term, no relabelling, no new
class. Its stated limitation is that it provides NO negative signal for
background -- a zero-weighted frame simply stops contributing, it never
teaches the model what background looks like. Whether that is enough to move
a class like Shooting is an empirical question this module does not answer by
itself; `core.p1c_head_training.train_cached_head(..., train_weights=...)` is
where the weights it produces actually get used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_intervals(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Official (start, end) source-frame spans per video; -1 entries are absent.

    Moved here from `scripts/diagnose_p1c_event_separability.py` (rule of
    two): that script now imports this copy rather than keeping its own.
    """

    out: dict[str, list[tuple[int, int]]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        spans = [(int(parts[2]), int(parts[3]))]
        if int(parts[4]) >= 0:
            spans.append((int(parts[4]), int(parts[5])))
        spans = [(a, b) for a, b in spans if a >= 0 and b > a]
        if spans:
            out[parts[0].rsplit(".", 1)[0]] = spans
    return out


def _is_known_background(
    row: dict[str, Any], intervals: dict[str, list[tuple[int, int]]], label: str
) -> bool:
    if row["label"] != label:
        return False
    spans = intervals.get(row["source_video_id"])
    if not spans:
        return False  # unannotated video of this label: cannot tell background from event
    frame = int(row["source_frame_index"])
    return not any(start <= frame <= end for start, end in spans)


def background_weights(
    rows: Sequence[dict[str, Any]], intervals: dict[str, list[tuple[int, int]]], *, label: str
) -> np.ndarray:
    """One weight per row of `rows`: 0.0 for known background frames of
    `label`, 1.0 for everything else (positive event frames of `label`, every
    row of every other label, and every row of an unannotated `label` video).

    An empty `intervals` mapping returns an all-ones array -- the regression
    contract for a caller with no official intervals available.
    """

    return np.array(
        [0.0 if _is_known_background(row, intervals, label) else 1.0 for row in rows],
        dtype=np.float64,
    )


def background_weight_counts(
    rows: Sequence[dict[str, Any]], intervals: dict[str, list[tuple[int, int]]], *, label: str
) -> dict[str, int]:
    """Counts an experiment report needs: annotated videos of `label`, rows
    zeroed (known background), and rows kept (weight 1) for `label`.
    """

    label_rows = [row for row in rows if row["label"] == label]
    annotated_videos = {row["source_video_id"] for row in label_rows if row["source_video_id"] in intervals}
    zeroed = sum(1 for row in label_rows if _is_known_background(row, intervals, label))
    return {
        "annotated_videos": len(annotated_videos),
        "rows_zeroed": zeroed,
        "rows_kept": len(label_rows) - zeroed,
    }
