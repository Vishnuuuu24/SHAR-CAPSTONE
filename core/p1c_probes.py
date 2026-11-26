"""B-9: classical probes on the B-1 frozen-feature cache.

Why this module exists
----------------------
Every P1C number so far comes from one training loop -- our head, our AdamW
schedule, our early stopping. That leaves one question unanswered: is 0.3481
what these frozen ResNet50 features are worth, or what our *loop* got out of
them? A logistic regression, a linear SVM and a k-NN classifier answer it,
because they are the standard reference point for "how linearly separable is
this representation" and they share none of our optimizer, schedule or
selection rule. That independence is the whole value; it is why the probes are
scikit-learn's and not a re-implementation here.

Reading the result
------------------
- A probe that *matches* the trained head means the head and its optimizer are
  not where the loss is -- the ceiling is in the features, and the next arm is
  a better backbone (backlog B-2), not a better head.
- A probe clearly *below* the head means the trained head is extracting
  something a convex linear fit cannot, and head design (B-4, B-8) still has
  room.
- k-NN is the non-parametric control: it asks whether the classes are
  locally clustered at all, independent of any decision boundary being linear.

Everything is scored through the same `eval.metrics` and `eval.video_level`
functions the training runner uses, so the numbers are comparable to
`docs/TRAINING_LOG.md` without a translation step. The scope inherits the
cache's scope exactly: AR fold 1, validation split, inherited video-level
labels (D-40), Test untouched.

Feature scaling is an explicit, reported dimension rather than a hidden
default. Pooled C5 activations are ReLU outputs with per-channel scales that
differ by an order of magnitude, and an L2-regularised linear model is not
scale-invariant, so `none` and `standardize` are genuinely different
experiments on the same features -- reporting only one would attribute the
scaler's effect to the classifier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from eval.metrics import classification_metrics, video_clustered_macro_f1_ci
from eval.video_level import aggregate_video_level_predictions
from core.p1c_engine import FramePrediction

#: Feature scalings a probe may be fitted under. `none` feeds the same vectors
#: the trained head consumed; `standardize` is the classical linear-probe
#: convention; `l2` makes a Euclidean k-NN equivalent to cosine similarity.
FEATURE_SCALINGS = ("none", "standardize", "l2")


@dataclass(frozen=True)
class ProbeResult:
    """One fitted probe, scored the same way a training run is scored."""

    probe: str
    scaling: str
    parameters: dict[str, Any]
    seed: int
    train_rows: int
    validation_rows: int
    fit_seconds: float
    predict_seconds: float
    frame_metrics: dict[str, Any]
    video_clustered_macro_f1_ci: dict[str, Any]
    video_level_metrics: dict[str, Any]
    video_count: int
    tie_broken_video_count: int
    converged: bool | None

    def to_json(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        return payload


def build_probe(probe: str, *, seed: int, **overrides: Any) -> tuple[Any, dict[str, Any]]:
    """Return `(estimator, registered_parameters)` for a named probe.

    Imported lazily so that importing this module -- which the tests and the
    readiness checks do -- never requires scikit-learn to be installed.
    """

    if probe == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        params = {"C": 1.0, "max_iter": 200, "tol": 1e-4, **overrides}
        return LogisticRegression(random_state=seed, n_jobs=None, **params), params
    if probe == "linear_svm":
        from sklearn.svm import LinearSVC

        # dual=False solves the 2048x2048 primal rather than a 391,014-variable
        # dual -- the right side when rows vastly outnumber features.
        params = {"C": 1.0, "dual": False, "max_iter": 2000, "tol": 1e-4, **overrides}
        return LinearSVC(random_state=seed, **params), params
    if probe == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        params = {"n_neighbors": 20, "weights": "distance", "algorithm": "brute", **overrides}
        # No random_state: brute-force k-NN is deterministic given the data.
        return KNeighborsClassifier(**params), params
    raise ValueError(f"unknown probe {probe!r}; expected one of logistic_regression, linear_svm, knn")


def apply_scaling(
    train: np.ndarray, validation: np.ndarray, scaling: str
) -> tuple[np.ndarray, np.ndarray]:
    """Scale both splits using statistics fitted on TRAIN only.

    Fitting the scaler on the concatenation would leak validation statistics
    into the training representation -- a small leak, but the kind that makes a
    reported number not mean what it says.
    """

    if scaling not in FEATURE_SCALINGS:
        raise ValueError(f"scaling must be one of {FEATURE_SCALINGS}; got {scaling!r}")
    if scaling == "none":
        return train, validation
    if scaling == "standardize":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(train)
        return scaler.transform(train), scaler.transform(validation)
    from sklearn.preprocessing import Normalizer

    scaler = Normalizer(norm="l2")
    return scaler.transform(train), scaler.transform(validation)


def stride_rows_within_videos(rows: Sequence[dict[str, Any]], stride: int) -> np.ndarray:
    """Indices keeping every `stride`-th frame *of each source video*.

    A uniform random subsample would drop whole short videos by chance and
    change the per-class frame balance; striding inside each video keeps all
    476 source videos represented and keeps each one's share of the training
    set roughly intact. Deterministic, so a subsampled probe is reproducible
    without recording an RNG seed for the subsample itself.
    """

    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    if stride == 1:
        return np.arange(len(rows))
    seen: dict[tuple[str, str], int] = {}
    keep: list[int] = []
    for index, row in enumerate(rows):
        key = (row["source_dataset"], row["source_video_id"])
        position = seen.get(key, 0)
        seen[key] = position + 1
        if position % stride == 0:
            keep.append(index)
    return np.asarray(keep, dtype=np.int64)


def _targets(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["target"]) for row in rows], dtype=np.int64)


def fit_and_score_probe(
    train_features: np.ndarray,
    train_rows: Sequence[dict[str, Any]],
    validation_features: np.ndarray,
    validation_rows: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    *,
    probe: str,
    scaling: str = "standardize",
    seed: int = 0,
    bootstrap_iterations: int = 1000,
    **overrides: Any,
) -> ProbeResult:
    """Fit one probe on the cached train features and score it on validation.

    Scored through exactly the functions the training runner uses, so the
    frame macro-F1, its source-video clustered CI and the video-level majority
    vote are directly comparable to `docs/TRAINING_LOG.md`.
    """

    if len(train_features) != len(train_rows):
        raise ValueError(f"train features/rows disagree: {len(train_features)} vs {len(train_rows)}")
    if len(validation_features) != len(validation_rows):
        raise ValueError(
            f"validation features/rows disagree: {len(validation_features)} vs {len(validation_rows)}"
        )

    estimator, params = build_probe(probe, seed=seed, **overrides)

    # float32, not the cache's float16: scikit-learn upcasts float16 to float64
    # internally, which would quadruple a 1.6 GB matrix for no added precision
    # -- the values only ever carried float16's ~3 decimal digits.
    train_x = np.ascontiguousarray(train_features, dtype=np.float32)
    validation_x = np.ascontiguousarray(validation_features, dtype=np.float32)
    train_x, validation_x = apply_scaling(train_x, validation_x, scaling)

    train_y = _targets(train_rows)

    started = time.perf_counter()
    estimator.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predicted = np.asarray(estimator.predict(validation_x), dtype=np.int64)
    predict_seconds = time.perf_counter() - started

    return score_predictions(
        predicted,
        validation_rows,
        class_names,
        probe=probe,
        scaling=scaling,
        parameters=params,
        seed=seed,
        train_row_count=len(train_rows),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        converged=_converged(estimator),
        bootstrap_iterations=bootstrap_iterations,
    )


def _converged(estimator: Any) -> bool | None:
    """Whether an iterative solver reported reaching its tolerance.

    Returned rather than warned about: a probe that hit `max_iter` is not
    necessarily wrong, but a reader comparing it against the trained head has
    to know the fit was truncated.
    """

    n_iter = getattr(estimator, "n_iter_", None)
    max_iter = getattr(estimator, "max_iter", None)
    if n_iter is None or max_iter is None:
        return None
    return bool(np.all(np.asarray(n_iter) < max_iter))


def score_predictions(
    predicted: np.ndarray,
    validation_rows: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    *,
    probe: str,
    scaling: str,
    parameters: dict[str, Any],
    seed: int,
    train_row_count: int,
    fit_seconds: float,
    predict_seconds: float,
    converged: bool | None,
    bootstrap_iterations: int = 1000,
) -> ProbeResult:
    """Score frame predictions the way `core.p1c_engine` scores an epoch."""

    frame_predictions = tuple(
        FramePrediction(
            source_dataset=row["source_dataset"],
            source_video_id=row["source_video_id"],
            target_index=int(row["target"]),
            predicted_index=int(prediction),
        )
        for row, prediction in zip(validation_rows, predicted, strict=True)
    )
    truth_labels = [class_names[p.target_index] for p in frame_predictions]
    predicted_labels = [class_names[p.predicted_index] for p in frame_predictions]
    # Same composite cluster key `core.p1c_engine.EvaluationResult.source_video_ids`
    # builds, NUL separator included: two datasets could reuse a video id, and a
    # different separator here would silently cluster differently from the runs
    # these numbers are being compared against.
    source_video_ids = [f"{p.source_dataset}\0{p.source_video_id}" for p in frame_predictions]

    frame_metrics = classification_metrics(truth_labels, predicted_labels, class_names)
    clustered = video_clustered_macro_f1_ci(
        truth_labels, predicted_labels, source_video_ids, class_names,
        iterations=bootstrap_iterations, seed=seed,
    )
    video_level = aggregate_video_level_predictions(frame_predictions, class_names)

    return ProbeResult(
        probe=probe,
        scaling=scaling,
        parameters=parameters,
        seed=seed,
        train_rows=train_row_count,
        validation_rows=len(validation_rows),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        frame_metrics=frame_metrics,
        video_clustered_macro_f1_ci=clustered,
        video_level_metrics=video_level.metrics,
        video_count=video_level.video_count,
        tie_broken_video_count=video_level.tie_broken_video_count,
        converged=converged,
    )
