"""Metrics for the live detector (FINAL_PUSH_PLAN_2026-11-11 Step 4c).

All four are computed against the official frame intervals of the 290 test videos, so they are
event-level-honest numbers (docs/PROJECT_RULES.md rule 1), and none of them is comparable with the 14-class
classifier's macro-F1 (rule 3).

- frame AUC: ROC AUC over every source frame of every test video (the standard UCF-Crime number);
- anomaly-video AUC: the same over the 140 anomalous test videos only;
- false alarms per hour on the 150 normal test videos, at a threshold fixed on TRAINING normal frames;
- detection delay: seconds from the start of an annotated event to the first alarm inside it.

A model scores sampled frames (one per `step` source frames). Source frame `t` takes the score of the
latest sampled frame at or before it, which is the causal assignment a live system would make.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def expand_to_source_frames(scores: np.ndarray, source_frames: np.ndarray, total_frames: int) -> np.ndarray:
    """Per-source-frame scores: frame `t` carries the score of the latest sampled frame at or before `t`."""

    if len(scores) != len(source_frames) or len(scores) == 0:
        raise ValueError("scores and source_frames must be equal-length and non-empty")
    position = np.searchsorted(source_frames, np.arange(total_frames), side="right") - 1
    return np.asarray(scores, dtype=np.float32)[np.clip(position, 0, len(scores) - 1)]


def source_frame_labels(total_frames: int, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    labels = np.zeros(total_frames, dtype=np.int8)
    for start, end in spans:
        labels[max(0, start):min(total_frames, end + 1)] = 1
    return labels


def causal_mean(scores: np.ndarray, window: int) -> np.ndarray:
    """Mean of each score with the `window - 1` scores before it (fewer at the start). `window=1` is the identity."""

    if window <= 1:
        return np.asarray(scores, dtype=np.float32)
    padded = np.concatenate([[0.0], np.cumsum(scores, dtype=np.float64)])
    index = np.arange(len(scores))
    start = np.maximum(0, index - window + 1)
    return ((padded[index + 1] - padded[start]) / (index - start + 1)).astype(np.float32)


def centred_mean(scores: np.ndarray, window: int) -> np.ndarray:
    """Mean over a window centred on each score (uses future frames: the offline comparison only)."""

    if window <= 1:
        return np.asarray(scores, dtype=np.float32)
    half = window // 2
    padded = np.concatenate([[0.0], np.cumsum(scores, dtype=np.float64)])
    index = np.arange(len(scores))
    start, stop = np.maximum(0, index - half), np.minimum(len(scores), index + half + 1)
    return ((padded[stop] - padded[start]) / (stop - start)).astype(np.float32)


def alarm_events(above: np.ndarray) -> int:
    """Number of maximal runs of consecutive sampled frames above the threshold."""

    above = np.asarray(above, dtype=bool)
    return int(above[0]) + int(np.sum(above[1:] & ~above[:-1])) if len(above) else 0


def detection_report(videos: Sequence[dict], threshold: float) -> dict:
    """All Step 4c metrics from per-video records.

    Each record: `scores` (sampled frames), `source_frames`, `total_frames`, `fps`, `spans` (official,
    empty for normal videos), `anomaly` (bool), `category`.
    """

    from sklearn.metrics import average_precision_score, roc_auc_score

    all_scores, all_labels, ano_scores, ano_labels = [], [], [], []
    normal_hours, normal_events, normal_videos_alarmed, normal_videos = 0.0, 0, 0, 0
    events, detected, delays = 0, 0, []
    per_category: dict[str, list] = {}
    for video in videos:
        dense = expand_to_source_frames(video["scores"], video["source_frames"], video["total_frames"])
        labels = source_frame_labels(video["total_frames"], video["spans"])
        all_scores.append(dense)
        all_labels.append(labels)
        above = np.asarray(video["scores"]) >= threshold
        if video["anomaly"]:
            ano_scores.append(dense)
            ano_labels.append(labels)
            for start, end in video["spans"]:
                events += 1
                inside = (video["source_frames"] >= start) & (video["source_frames"] <= end) & above
                hit = bool(inside.any())
                detected += hit
                if hit:
                    delays.append((int(video["source_frames"][np.argmax(inside)]) - start) / video["fps"])
                per_category.setdefault(video["category"], []).append(hit)
        else:
            normal_videos += 1
            normal_hours += video["total_frames"] / video["fps"] / 3600.0
            count = alarm_events(above)
            normal_events += count
            normal_videos_alarmed += int(count > 0)
    scores, labels = np.concatenate(all_scores), np.concatenate(all_labels)
    a_scores, a_labels = np.concatenate(ano_scores), np.concatenate(ano_labels)
    return {
        "frame_auc": float(roc_auc_score(labels, scores)), "frame_ap": float(average_precision_score(labels, scores)),
        "anomaly_video_frame_auc": float(roc_auc_score(a_labels, a_scores)),
        "test_frames": int(len(labels)), "anomalous_frame_share": float(labels.mean()),
        "threshold": float(threshold),
        "normal_test_hours": normal_hours, "false_alarms_per_hour": normal_events / normal_hours,
        "normal_videos_with_an_alarm": normal_videos_alarmed / normal_videos,
        "events": events, "event_recall": detected / events,
        "median_delay_s": float(np.median(delays)) if delays else None,
        "mean_delay_s": float(np.mean(delays)) if delays else None,
        "event_recall_by_category": {c: float(np.mean(h)) for c, h in sorted(per_category.items())},
    }
