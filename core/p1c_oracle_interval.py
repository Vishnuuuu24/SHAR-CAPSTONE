"""Oracle-interval probe helpers and UCA interval-recovery helpers (diagnostics only, never headlines)."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

Span = tuple[float, float]

CLASS_LEXICON: dict[str, str] = {
    "Abuse": "abus|beat|hit|slap|kick|punch|hurt|assault|attack",
    "Arrest": "arrest|handcuff|detain|police|officer|cop",
    "Arson": "fire|burn|flame|smoke|arson|ignit|torch|lit",
    "Assault": "assault|attack|beat|hit|punch|kick|push|fight|strike",
    "Burglary": "break|burglar|steal|stole|thief|enter|smash|rob|forc",
    "Explosion": "explo|blast|fire|burst|boom",
    "Fighting": "fight|punch|kick|brawl|hit|strike|clash|scuffl",
    "RoadAccidents": "accident|collid|crash|hit by|knocked|collision|ran into|overturn|rear",
    "Robbery": "rob|gun|threat|snatch|steal|stole|hold up|knife|weapon",
    "Shooting": "shoot|gun|fire|pistol|weapon|firearm",
    "Shoplifting": "steal|stole|shoplift|conceal|hide|put .* in|pocket|bag|without pay",
    "Stealing": "steal|stole|thief|took|take|taken|snatch|grab",
    "Vandalism": "vandal|smash|break|damage|spray|graffiti|destroy|kick|hit",
}


def frames_in_spans(frame_indices: Sequence[int], spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """Boolean mask: True where a source frame index lies inside any inclusive span."""

    idx = np.asarray(frame_indices, dtype=np.int64)
    mask = np.zeros(idx.shape, dtype=bool)
    for start, end in spans:
        mask |= (idx >= start) & (idx <= end)
    return mask


def select_event_positions(frame_indices: Sequence[int], spans: Sequence[tuple[int, int]], inside: bool) -> np.ndarray:
    """Positions (into `frame_indices`) of cached frames inside (or outside every) span, ascending frame order."""

    mask = frames_in_spans(frame_indices, spans)
    if not inside:
        mask = ~mask
    positions = np.flatnonzero(mask)
    return positions[np.argsort(np.asarray(frame_indices)[positions], kind="stable")]


def load_glances(path) -> dict[str, list[int]]:
    """GlanceVAD clicks: video id -> sorted source frame indices."""

    import csv

    out: dict[str, list[int]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["video-id"], []).append(int(row["glance"]))
    return {v: sorted(x) for v, x in out.items()}


def glance_spans(glances: Mapping[str, Sequence[int]], fps: Mapping[str, float], window_s: float) -> dict[str, list[tuple[int, int]]]:
    """Each click becomes an inclusive frame span of +/- `window_s` seconds at that video's own frame rate; overlaps merged."""

    out: dict[str, list[tuple[int, int]]] = {}
    for video, clicks in glances.items():
        half = int(round(window_s * fps[video]))
        spans = sorted((max(0, c - half), c + half) for c in clicks)
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        out[video] = merged
    return out


def sample_evenly(candidates: np.ndarray, k: int, seed: int) -> np.ndarray:
    """`k` candidates at the midpoints of `k` equal segments (repeats when fewer than k), then a seeded order shuffle."""

    n = len(candidates)
    if n == 0:
        raise ValueError("no candidate frames to sample from")
    picks = np.minimum(((np.arange(k) + 0.5) * n / k).astype(np.int64), n - 1)
    chosen = np.asarray(candidates)[picks]
    return chosen[np.random.default_rng(seed).permutation(k)]


def union_spans(spans: Sequence[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_length(spans: Sequence[Span]) -> float:
    return sum(end - start for start, end in union_spans(spans))


def overlap_length(a: Sequence[Span], b: Sequence[Span]) -> float:
    ua, ub = union_spans(a), union_spans(b)
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in ua for s2, e2 in ub)


def interval_iou(a: Sequence[Span], b: Sequence[Span]) -> float:
    """IoU of two span sets (each merged into a union first); 0.0 if both empty."""

    inter = overlap_length(a, b)
    union = total_length(a) + total_length(b) - inter
    return inter / union if union > 0 else 0.0


def spans_overlap(a: Span, spans: Sequence[Span]) -> bool:
    return any(min(a[1], e) - max(a[0], s) > 0 for s, e in spans)


def fps_from_cache(max_cached_source_frame_index: int, duration_seconds: float, stride: int = 5) -> float:
    return (max_cached_source_frame_index + stride) / duration_seconds


def frames_to_seconds(spans: Sequence[tuple[int, int]], fps: float) -> list[Span]:
    return [(start / fps, end / fps) for start, end in spans]


def keyword_hits(sentence: str, class_name: str) -> bool:
    return re.search(CLASS_LEXICON[class_name], sentence, flags=re.IGNORECASE) is not None


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Two-sided exact McNemar p from the two discordant counts (binomial, p=0.5)."""

    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def macro_recall(truth: Sequence[int], pred: Sequence[int]) -> float:
    """Mean per-class recall over classes present in `truth`."""

    truth_a, pred_a = np.asarray(truth), np.asarray(pred)
    recalls = [float((pred_a[truth_a == c] == c).mean()) for c in np.unique(truth_a)]
    return float(np.mean(recalls)) if recalls else float("nan")


def summarise(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "median": float(np.median(arr)), "mean": float(arr.mean())} if arr.size else {"n": 0}


def video_class(video_id: str) -> str:
    return re.sub(r"\d+.*$", "", video_id)


def uca_segment_spans(entry: Mapping[str, Any]) -> list[tuple[Span, str]]:
    return [((float(s), float(e)), sentence) for (s, e), sentence in zip(entry["timestamps"], entry["sentences"])]


def predict_video_probs(head: Any, sequence: np.ndarray, frames: int) -> np.ndarray:
    """Softmax of the chunk-mean logits for one video, exactly the head-training validation rule."""

    import torch
    from core.p1c_head_variants import _chunk_sequence, _pad_sequence

    chunks = [_pad_sequence(chunk, frames) for chunk in _chunk_sequence(sequence, frames)]
    with torch.no_grad():
        logits = head(
            torch.from_numpy(np.stack([c[0] for c in chunks])), key_padding_mask=torch.from_numpy(np.stack([c[1] for c in chunks]))
        )
        return torch.softmax(logits.mean(dim=0), dim=0).numpy()


def arm_row_positions(
    frame_indices: np.ndarray, spans: Sequence[tuple[int, int]] | None, arm: str, frames: int, seed: int
) -> np.ndarray | None:
    """Positions into a video's rows for `oracle_event`/`anti_event`/`uniform8`; None for `uniform` (all frames)."""

    order = np.argsort(frame_indices, kind="stable")
    if arm == "uniform":
        return None
    if arm == "uniform8":
        return sample_evenly(order, frames, seed)
    if spans is None:
        raise ValueError(f"arm {arm} needs the video's official spans")
    pool = select_event_positions(frame_indices, spans, inside=(arm == "oracle_event"))
    return sample_evenly(pool, frames, seed) if len(pool) else None
