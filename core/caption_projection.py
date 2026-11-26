"""Caption-supervised projection for the 14-class classifier (FINAL_PUSH_PLAN_2026-11-11 Step 6).

Why: a bag-of-words classifier on the human UCA captions reaches 0.80 macro-F1 with the same folds and
the same 38 videos per class, so the classes are learnable from a description of what happens; the
visual features are the limit. UCA gives 23,542 time-stamped sentences, far more supervision than 532
video labels.

What is learned: a linear map from the mean visual feature of a sentence's time span to that
sentence's text vector (TF-IDF then truncated SVD; no text-model weights). It is fitted on TRAINING
videos only -- the training side of the inner split plus captioned videos outside the 700-video
protocol -- and then applied to every frame of a video. A video is described by its mean feature plus
the mean and maximum over time of the projected frames. Captions are never an input at prediction
time, and a held-out video's own captions are never used.

`retrieval_gate` is the Step 6a check: on held-out videos, does the projected span feature rank its
own sentence above the other held-out sentences? If not, the features do not carry the information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

UCA_FILES = ("UCFCrime_Train.json", "UCFCrime_Val.json", "UCFCrime_Test.json")
TEXT_DIM = 128


def load_captions(uca_dir: Path) -> dict[str, list[tuple[float, float, str]]]:
    """`{video stem: [(start_s, end_s, sentence)]}` over UCA's three files (its own split is not used)."""

    out: dict[str, list[tuple[float, float, str]]] = {}
    for name in UCA_FILES:
        for stem, entry in json.loads((Path(uca_dir) / name).read_text()).items():
            if stem in out:
                raise ValueError(f"{stem} appears in two UCA files")
            out[stem] = [(float(a), float(b), s.strip()) for (a, b), s in zip(entry["timestamps"], entry["sentences"]) if s.strip()]
    return out


@dataclass
class CaptionData:
    """Frame features and caption spans for every video that may train or be scored."""

    frames: dict[str, np.ndarray]         # stem -> [T, D] float32 frame features (only for videos that are scored)
    times: dict[str, np.ndarray]          # stem -> [T] seconds of each frame
    spans: dict[str, np.ndarray]          # stem -> [n_sentences, D] mean feature of each sentence's time span
    sentences: dict[str, list[str]]       # stem -> sentences aligned with `spans`


def span_features(frames: np.ndarray, times: np.ndarray, sentences: Sequence[tuple[float, float, str]]) -> np.ndarray:
    """Mean frame feature inside each sentence's [start, end] seconds; the nearest frame when the span holds none."""

    out = np.empty((len(sentences), frames.shape[1]), dtype=np.float32)
    for row, (start, end, _text) in enumerate(sentences):
        inside = (times >= min(start, end)) & (times <= max(start, end))
        out[row] = frames[inside].mean(axis=0) if inside.any() else frames[int(np.argmin(np.abs(times - (start + end) / 2)))]
    return out


def build_caption_data(frames_of: Callable[[str], tuple[np.ndarray, np.ndarray]], stems: Sequence[str],
                       captions: dict[str, list[tuple[float, float, str]]], *, keep_frames: bool = True) -> CaptionData:
    """`frames_of(stem) -> (frames [T, D], times [T])`. Videos without captions keep frames but contribute no spans.

    `keep_frames=False` stores only the span features: enough for videos that train the projection but are never scored.
    """

    data = CaptionData({}, {}, {}, {})
    for stem in stems:
        frames, times = frames_of(stem)
        if keep_frames:
            data.frames[stem], data.times[stem] = frames, times
        mine = captions.get(stem, [])
        data.sentences[stem] = [s for _a, _b, s in mine]
        data.spans[stem] = span_features(frames, times, mine) if mine else np.empty((0, frames.shape[1]), dtype=np.float32)
    return data


def merge(*parts: CaptionData) -> CaptionData:
    """One `CaptionData` over several disjoint sets of videos."""

    out = CaptionData({}, {}, {}, {})
    for part in parts:
        for name in ("frames", "times", "spans", "sentences"):
            mine, theirs = getattr(out, name), getattr(part, name)
            if set(mine) & set(theirs):
                raise ValueError("caption data parts overlap")
            mine.update(theirs)
    return out


def fit_text_model(sentences: Sequence[str], *, dim: int = TEXT_DIM, seed: int = 0):
    """TF-IDF (1-2 grams) then truncated SVD, fitted on training sentences only; returns a fitted pipeline."""

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                         TruncatedSVD(n_components=dim, random_state=seed), Normalizer()).fit(list(sentences))


def fit_projection(span_matrix: np.ndarray, text_matrix: np.ndarray, alphas: Sequence[float] = (1e2, 1e3, 1e4, 1e5)):
    """Standardise, then ridge-map span feature -> text vector; the penalty is chosen by leave-one-out on the training pairs.

    The scaler is fitted on the training pairs too, so no statistic of a held-out video enters the map.
    """

    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=list(alphas))).fit(span_matrix, text_matrix)


def stack(data: CaptionData, stems: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Span features and sentences of the given videos, concatenated in order."""

    spans = [data.spans[s] for s in stems if len(data.sentences[s])]
    text = [t for s in stems for t in data.sentences[s]]
    dim = next(iter(data.spans.values())).shape[1]
    return (np.concatenate(spans) if spans else np.empty((0, dim), dtype=np.float32)), text


def retrieval_gate(projection, text_model, data: CaptionData, heldout: Sequence[str]) -> dict[str, float]:
    """On held-out videos: where does each span's own sentence rank among all held-out sentences (cosine)?

    `median_percentile_rank` is 0.5 at chance and 0 when every span retrieves its own sentence first.
    """

    spans, sentences = stack(data, heldout)
    if len(sentences) < 2:
        return {"sentences": len(sentences), "median_percentile_rank": float("nan"), "recall_at_10": float("nan"), "chance_recall_at_10": float("nan")}
    predicted = projection.predict(spans)
    truth = text_model.transform(sentences)
    predicted = predicted / np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-9)
    similarity = predicted @ truth.T
    own = similarity[np.arange(len(sentences)), np.arange(len(sentences))]
    rank = (similarity > own[:, None]).sum(axis=1)
    return {"sentences": len(sentences), "median_percentile_rank": float(np.median(rank / (len(sentences) - 1))),
            "recall_at_10": float((rank < 10).mean()), "chance_recall_at_10": float(min(1.0, 10 / len(sentences)))}


def video_matrix(projection, data: CaptionData, stems: Sequence[str], *, include_mean_feature: bool = True) -> np.ndarray:
    """One row per video: [mean frame feature ++] mean and max over time of the projected frames."""

    rows = []
    for stem in stems:
        projected = projection.predict(data.frames[stem])
        parts = [projected.mean(axis=0), projected.max(axis=0)]
        rows.append(np.concatenate(([data.frames[stem].mean(axis=0)] if include_mean_feature else []) + parts))
    return np.stack(rows).astype(np.float32)


def caption_arm(data: CaptionData, extra_stems: Sequence[str], *, include_mean_feature: bool = True, gate_log: list | None = None):
    """Inner-CV arm (`core.inner_cv.Arm`): fit the text model and the projection on the training side plus
    `extra_stems`, then classify with the reference logistic arm on the projected video matrix."""

    from core import inner_cv

    def arm(pool, train_index: np.ndarray, heldout_index: np.ndarray, seed: int) -> np.ndarray:
        train_stems = [pool.video_ids[i] for i in train_index]
        if set(extra_stems) & set(pool.video_ids):
            raise RuntimeError("an extra caption video is also a pool video")
        spans, sentences = stack(data, train_stems + list(extra_stems))
        text_model = fit_text_model(sentences, seed=seed)
        projection = fit_projection(spans, text_model.transform(sentences))
        if gate_log is not None:
            gate_log.append({"fold": pool.fold, "seed": seed, "alpha": float(projection[-1].alpha_),
                             **retrieval_gate(projection, text_model, data, [pool.video_ids[i] for i in heldout_index])})
        matrix = video_matrix(projection, data, pool.video_ids, include_mean_feature=include_mean_feature)
        return inner_cv.meanpool_logistic_arm(matrix=lambda _pool: matrix)(pool, train_index, heldout_index, seed)

    return arm
