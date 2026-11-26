"""Causal sliding-window timeline for the presentation showcase, built on the final E3 model (D-76).

E3 scores a video by averaging each encoder's per-frame features over the whole video (V-JEPA 2: over consecutive
16-frame clips) and feeding the concatenations to 36 logistic members. The showcase replays a video and wants an
answer that changes as it plays, so here the same members score *trailing windows*: at playback time t, the window
holds only the frames from the last `width` selected frames up to t, as a live camera feed would.

This is a display aid, not an evaluated quantity. E3 was trained and tested on whole-video averages with
video-inherited labels; a window score says which class the last few seconds resemble, never where an event is.
The whole-video answer remains the reported prediction, and `full_video_parity` checks it against the saved
E3 replay before any window is trusted.
"""

from __future__ import annotations

import numpy as np

CLIP = 16            # V-JEPA 2 clip length in selected frames (E3 registered)
FRAME_KEYS = ("B", "L", "S")


def trailing_windows(n_frames: int, width: int, step: int) -> list[tuple[int, int, list[int]]]:
    """(start, end, clip indices) for windows ending every `step` frames; end is exclusive.

    Clips are the consecutive 16-frame groups from the video start that E3 used; a window only takes clips lying
    wholly inside it. Windows with no complete clip are skipped, because E3 never scores without V-JEPA input.
    """

    if width < CLIP or step < 1:
        raise ValueError("width must hold one clip and step must be positive")
    out = []
    for end in range(min(width, n_frames), n_frames + 1, step):
        start = max(0, end - width)
        clips = [k for k in range(n_frames // CLIP) if k * CLIP >= start and (k + 1) * CLIP <= end]
        if clips:
            out.append((start, end, clips))
    if out and out[-1][1] != n_frames:  # always close on the final frame
        start = max(0, n_frames - width)
        clips = [k for k in range(n_frames // CLIP) if k * CLIP >= start]
        if clips:
            out.append((start, n_frames, clips))
    return out


def pooled(per_frame: dict[str, np.ndarray], per_clip: np.ndarray, frames: slice | None = None,
           clips: list[int] | None = None) -> dict[str, np.ndarray]:
    """E3 pooling: float64 mean of the float16-rounded features, cast to float32 (matches e3_features.extract)."""

    sel = frames if frames is not None else slice(None)
    vec = {k: per_frame[k][sel].astype(np.float64).mean(axis=0).astype(np.float32) for k in FRAME_KEYS}
    c = per_clip if clips is None else per_clip[clips]
    vec["VJ"] = c.astype(np.float64).mean(axis=0).astype(np.float32)
    return vec


def full_video_parity(vectors: dict[str, np.ndarray], saved: dict[str, np.ndarray], probs: np.ndarray,
                      saved_probs: np.ndarray) -> dict:
    """Largest absolute differences against the saved E3 replay vectors and probabilities."""

    return {
        "max_abs_vector_diff": {k: float(np.max(np.abs(vectors[k] - saved[k]))) for k in saved},
        "max_abs_prob_diff": float(np.max(np.abs(probs - saved_probs))),
        "same_top1": int(np.argmax(probs)) == int(np.argmax(saved_probs)),
    }


def first_alert(windows: list[dict], threshold: float, run: int, normal: str = "Normal_Videos_event") -> dict | None:
    """First window end at which the same non-Normal class has led `run` consecutive windows at >= `threshold`.

    A presentation rule, stated on the page; it names a class, not the moment an event happens.
    """

    streak, label = 0, None
    for w in windows:
        top = w["top3"][0]
        ok = top["label"] != normal and top["prob"] >= threshold
        streak = streak + 1 if ok and top["label"] == label else (1 if ok else 0)
        label = top["label"] if ok else None
        if streak >= run:
            return {"label": label, "at_s": w["end_s"]}
    return None
