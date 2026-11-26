"""Pose-clarity measurement: does denoising or a larger input make more of a person's body detectable?

Pure logic for `notebooks/pose_clarity_check.ipynb` (no GPU, no detector import), so it can be unit-tested.

Design (what is measured, and what is not):
- Reference people come from the cached YOLO11m-pose detections (`results/p1c/p1c-roi-cache/detections.jsonl`). Every
  condition, including the original frame, is re-run fresh and each reference person is matched to the nearest box
  of that run (IoU >= `MATCH_IOU`). A person a condition cannot match counts as not usable.
- "Usable body" is the definition used in the 2026-11-05 posture review: shoulder, hip and knee each at keypoint
  confidence >= 0.5 on at least one side. It measures how much body the detector claims to see. It is NOT pose accuracy
  and not ground truth (rule 2): a restorer can raise confidence without adding truth, so the notebook also writes
  side-by-side overlays for a human look.
- Reference people are chosen because the ORIGINAL frame detected them, which favours the original on detection. Extra
  people that other conditions find are therefore reported separately, on frames where the cache found nobody.
- Uncertainty resamples whole videos (source grouping), not frames or people.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

MATCH_IOU = 0.5
KEYPOINT_CONF = 0.5
BUCKETS = ((0, 40, "under40"), (40, 60, "40-60"), (60, 80, "60-80"), (80, 120, "80-120"), (120, 10**9, "over120"))
# COCO-17 indices: shoulders 5/6, hips 11/12, knees 13/14.
SHOULDERS, HIPS, KNEES = (5, 6), (11, 12), (13, 14)
# CHOICE (proposed here, not sourced): a condition only counts as helping if usable-body rate rises by at least this
# many points overall and the paired 95% interval excludes zero.
MIN_GAIN = 0.05


def box_height(box: Sequence[float]) -> float:
    return float(box[3] - box[1])


def height_bucket(height: float) -> str:
    for low, high, name in BUCKETS:
        if low <= height < high:
            return name
    return BUCKETS[-1][2]


def _visible(keypoints: Sequence[Sequence[float]], index: int) -> bool:
    return float(keypoints[index][2]) >= KEYPOINT_CONF


def usable_body(keypoints: Sequence[Sequence[float]]) -> bool:
    """Shoulder, hip and knee each visible on at least one side."""

    return all(any(_visible(keypoints, i) for i in group) for group in (SHOULDERS, HIPS, KNEES))


def visible_joints(keypoints: Sequence[Sequence[float]]) -> int:
    return sum(_visible(keypoints, i) for i in range(len(keypoints)))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_persons(reference: Sequence[Sequence[float]], candidates: Sequence[Sequence[float]]) -> list[int | None]:
    """Greedy one-to-one matching by descending IoU; entry i is the candidate index for reference box i, or None."""

    pairs = sorted(
        ((iou(r, c), i, j) for i, r in enumerate(reference) for j, c in enumerate(candidates)), reverse=True
    )
    out: list[int | None] = [None] * len(reference)
    used: set[int] = set()
    for score, i, j in pairs:
        if score < MATCH_IOU:
            break
        if out[i] is None and j not in used:
            out[i] = j
            used.add(j)
    return out


def plan_frames(
    records: Iterable[dict[str, Any]], per_bucket: int, n_empty: int, seed: int
) -> list[dict[str, Any]]:
    """Stratified sample of cached frames: `per_bucket` per height bucket of the largest person, `n_empty` with no
    detected person. At most one frame per video across the whole plan. Deterministic in `seed`."""

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        persons = rec["persons"]
        if persons:
            largest = max(persons, key=lambda p: box_height(p["box"]))
            pools[height_bucket(box_height(largest["box"]))].append(rec)
        else:
            pools["none"].append(rec)
    rng = random.Random(seed)
    taken: set[str] = set()
    plan: list[dict[str, Any]] = []
    for name in [b[2] for b in BUCKETS] + ["none"]:
        pool = sorted(pools[name], key=lambda r: (r["source_video_id"], r["source_frame_index"]))
        rng.shuffle(pool)
        want = n_empty if name == "none" else per_bucket
        got = 0
        for rec in pool:
            if got == want:
                break
            if rec["source_video_id"] in taken:
                continue
            taken.add(rec["source_video_id"])
            plan.append({"video": rec["source_video_id"], "frame": rec["source_frame_index"], "stratum": name,
                         "ref_persons": rec["persons"]})
            got += 1
    return plan


def score_condition(ref_persons: Sequence[dict[str, Any]], found: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per reference person: matched, usable, joints under one condition. `found` is that condition's detections."""

    matches = match_persons([p["box"] for p in ref_persons], [p["box"] for p in found])
    rows = []
    for person, j in zip(ref_persons, matches):
        kp = found[j]["keypoints"] if j is not None else None
        rows.append({"bucket": height_bucket(box_height(person["box"])), "matched": j is not None,
                     "usable": bool(kp is not None and usable_body(kp)), "joints": visible_joints(kp) if kp else 0})
    return rows


def _rate_diff(rows_a: list[dict], rows_b: list[dict], key: str = "usable") -> float:
    if not rows_a or not rows_b:
        return float("nan")
    return float(np.mean([r[key] for r in rows_a]) - np.mean([r[key] for r in rows_b]))


def paired_video_bootstrap(
    cond_rows: dict[str, list[dict]], base_rows: dict[str, list[dict]], key: str = "usable", n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Difference in mean `key` (condition minus base) with a 95% interval from resampling videos with replacement.
    Both dicts map video -> per-person rows in the same person order."""

    videos = sorted(set(cond_rows) & set(base_rows))
    rng = np.random.default_rng(seed)

    def stat(chosen: Sequence[str]) -> float:
        return _rate_diff([r for v in chosen for r in cond_rows[v]], [r for v in chosen for r in base_rows[v]], key)

    point = stat(videos)
    draws = [stat([videos[i] for i in rng.integers(0, len(videos), len(videos))]) for _ in range(n_boot)]
    draws = [d for d in draws if not np.isnan(d)]
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (float("nan"),) * 2
    return {"diff": point, "lo": lo, "hi": hi, "videos": len(videos)}


def verdict(summary: dict[str, float]) -> str:
    """Pre-declared reading of one overall comparison (MIN_GAIN is a proposed threshold, see module constant)."""

    if summary["lo"] > 0 and summary["diff"] >= MIN_GAIN:
        return "helps"
    if summary["hi"] < 0:
        return "hurts"
    return "inside the noise" if summary["lo"] <= 0 <= summary["hi"] else "small gain below threshold"
