"""Inner cross-validation over one official fold's training pool (FINAL_PUSH_PLAN_2026-11-11 Step 1).

Why this exists: the 56 validation videos per fold (4 per class) cannot resolve a gain below about
0.05 macro-F1, and choosing the training epoch on those same videos inflated earlier figures by
about 0.06 (D-66). Here every classifier comparison is scored on held-out predictions for ALL of a
fold's 532 official training videos instead:

- pool   = the fold's train + validation caches (38 videos per class); the fold's test videos are
           never loaded;
- splits = class-stratified K parts of whole videos, repeated with different seeds;
- rule   = every arm trains for a length fixed in advance; nothing is selected on scored videos.

An arm is a callable `(pool, train_index, heldout_index, seed) -> probs[len(heldout_index), C]`.
Labels are the inherited video-level labels (D-40), never event-only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from core.p1c_video_sampling import sample_rows_per_video

FRAMES = 8
POOL_SPLITS = ("train", "validation")
SOURCE_DATASET = "ucf_crime_official_video_frames"

Arm = Callable[["Pool", np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class Pool:
    """One fold's training pool: every cached frame of its 532 videos plus the registered 8-frame draw."""

    fold: int
    class_names: list[str]
    video_ids: list[str]                 # sorted
    targets: np.ndarray                  # [V] class index per video
    features: list[np.ndarray]           # one [n_split, D] float16 array (memmap) per pool split
    spans: dict[str, tuple[int, int, int]]   # video -> (split position, start, stop), frames ascending
    sampled: dict[str, np.ndarray]       # video -> row indices (within its split) of the 8-frame draw
    cache_sha256: dict[str, str]
    frame_indices: list[np.ndarray] = field(default_factory=list)   # per split, source frame index of every row
    splits: tuple[str, ...] = POOL_SPLITS                           # the cache splits merged into this pool, in order
    _means: np.ndarray | None = field(default=None, repr=False)

    @property
    def dim(self) -> int:
        return int(self.features[0].shape[1])

    def frames_of(self, video: str) -> np.ndarray:
        split, start, stop = self.spans[video]
        return np.asarray(self.features[split][start:stop], dtype=np.float32)

    def sampled_frames_of(self, video: str) -> np.ndarray:
        split, _start, _stop = self.spans[video]
        return np.asarray(self.features[split][self.sampled[video]], dtype=np.float32)

    def frame_indices_of(self, video: str) -> np.ndarray:
        split, start, stop = self.spans[video]
        return self.frame_indices[split][start:stop]

    def videos_of_split(self, split: str) -> list[str]:
        """The videos of one pool split (`train` or `validation`), in the cache's own row order."""

        position = self.splits.index(split)
        return sorted((v for v, span in self.spans.items() if span[0] == position), key=lambda v: self.spans[v][1])

    def light(self) -> "Pool":
        """Copy without feature arrays: safe to send between processes (a memmap pickles by value, K-16)."""

        return Pool(self.fold, self.class_names, self.video_ids, self.targets, [], {}, {}, self.cache_sha256, splits=self.splits)

    def video_means(self) -> np.ndarray:
        """`[V, D]` mean over every cached frame of each video, in `video_ids` order."""

        if self._means is None:
            self._means = np.stack([self.frames_of(v).mean(axis=0) for v in self.video_ids])
        return self._means


def load_pool(cache_root: Path, fold: int, *, frames: int = FRAMES, sampling_seed: int = 0, splits: Sequence[str] = POOL_SPLITS) -> Pool:
    """Read `<cache_root>/{train,validation}/{features.float16.npy,rows.csv,manifest.json}` into a `Pool`.

    Rows must be grouped by video with ascending frame index (the cache builder's order); this is
    checked, because every span below relies on it. `splits` may add `"test"` for the one declared
    final test read (D-73); every inner-cross-validation run uses the default and never loads test.
    """

    features, spans, sampled, targets, digests, class_names, frame_index = [], {}, {}, {}, {}, None, []
    for position, split in enumerate(splits):
        directory = Path(cache_root) / split
        manifest = json.loads((directory / "manifest.json").read_text())
        digests[split] = manifest["cache_sha256"]
        if class_names is None:
            class_names = list(manifest["class_names"])
        elif class_names != list(manifest["class_names"]):
            raise ValueError(f"{directory}: class_names differ between pool splits")
        array = np.load(directory / "features.float16.npy", mmap_mode="r")
        with (directory / "rows.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != array.shape[0]:
            raise ValueError(f"{directory}: {len(rows)} rows vs {array.shape[0]} feature rows")
        features.append(array)
        frame_index.append(np.asarray([int(r["source_frame_index"]) for r in rows], dtype=np.int64))
        start = 0
        for index in range(1, len(rows) + 1):
            if index == len(rows) or rows[index]["source_video_id"] != rows[start]["source_video_id"]:
                video = rows[start]["source_video_id"]
                if video in spans:
                    raise ValueError(f"{directory}: video {video} is not contiguous or appears in two pool splits")
                frame_indices = [int(r["source_frame_index"]) for r in rows[start:index]]
                if frame_indices != sorted(frame_indices):
                    raise ValueError(f"{directory}: frames of {video} are not in ascending order")
                if len({r["target"] for r in rows[start:index]}) != 1:
                    raise ValueError(f"{directory}: frames of {video} disagree on target")
                spans[video] = (position, start, index)
                targets[video] = int(rows[start]["target"])
                start = index
        keep = sample_rows_per_video(rows, frames, seed=sampling_seed)
        for row_index in keep:
            sampled.setdefault(rows[int(row_index)]["source_video_id"], []).append(int(row_index))
    video_ids = sorted(spans)
    return Pool(
        fold=fold, class_names=class_names, video_ids=video_ids,
        targets=np.asarray([targets[v] for v in video_ids], dtype=np.int64), features=features, spans=spans,
        sampled={v: np.asarray(sampled[v], dtype=np.int64) for v in video_ids}, cache_sha256=digests,
        frame_indices=frame_index, splits=tuple(splits),
    )


def inner_splits(targets: np.ndarray, *, parts: int = 5, repeats: int = 3, seed: int = 0) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """`[(repeat, part, train_index, heldout_index)]`: class-stratified, whole videos, every video held out once per repeat."""

    from sklearn.model_selection import StratifiedKFold

    out = []
    for repeat in range(repeats):
        folds = StratifiedKFold(n_splits=parts, shuffle=True, random_state=seed * 1000 + repeat)
        for part, (train_index, heldout_index) in enumerate(folds.split(np.zeros(len(targets)), targets)):
            out.append((repeat, part, train_index, heldout_index))
    return out


def run_arm(pool: Pool, arm: Arm, splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]]) -> dict[int, np.ndarray]:
    """`{repeat: probs[V, C]}`: each video's held-out probabilities, one full pass per repeat."""

    classes = len(pool.class_names)
    out: dict[int, np.ndarray] = {}
    for repeat, part, train_index, heldout_index in splits:
        if set(train_index.tolist()) & set(heldout_index.tolist()):
            raise RuntimeError("inner split leaks a video into both sides")
        probs = np.asarray(arm(pool, train_index, heldout_index, repeat * 100 + part), dtype=np.float64)
        if probs.shape != (len(heldout_index), classes):
            raise ValueError(f"arm returned {probs.shape}, expected {(len(heldout_index), classes)}")
        out.setdefault(repeat, np.full((len(pool.video_ids), classes), np.nan))[heldout_index] = probs
    for repeat, table in out.items():
        if np.isnan(table).any():
            raise RuntimeError(f"repeat {repeat}: some videos were never held out")
    return out


# --------------------------------------------------------------------------------------------- arms


def meanpool_logistic_arm(
    *, c_grid: Sequence[float] = (0.003, 0.01, 0.03, 0.1), nested_parts: int = 4,
    matrix: Callable[[Pool], np.ndarray] | None = None, train_matrix: Callable[[Pool], np.ndarray] | None = None,
    extra: Callable[[Pool], tuple[np.ndarray, np.ndarray]] | None = None, class_weight: str | None = None,
) -> Arm:
    """Reference row: one vector per video (mean over all cached frames), standardised, logistic regression.

    The regularisation strength is chosen by a nested cross-validation inside the training side only.
    Options, each a single-factor change of the same arm:

    - `matrix(pool) -> [V, D]`: another per-video feature matrix (other backbones, fusions);
    - `train_matrix(pool)`: a different matrix for the videos while they are TRAINING rows (frames
      chosen with training-only annotation); held-out videos are always scored on `matrix`;
    - `extra(pool) -> (X, y)`: rows appended to every training side (videos outside the pool);
    - `class_weight="balanced"`: inverse-frequency class weights.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from core.confusion_groups import macro_f1

    def arm(pool: Pool, train_index: np.ndarray, heldout_index: np.ndarray, seed: int) -> np.ndarray:
        x = pool.video_means() if matrix is None else matrix(pool)
        x_train = x if train_matrix is None else train_matrix(pool)
        y, classes = pool.targets, len(pool.class_names)
        extra_x, extra_y = (np.empty((0, x.shape[1]), dtype=x.dtype), np.empty(0, dtype=y.dtype)) if extra is None else extra(pool)

        def fit(c: float, rows: np.ndarray):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight=class_weight))
            return model.fit(np.concatenate([x_train[rows], extra_x]), np.concatenate([y[rows], extra_y]))

        best_c, best_score = c_grid[0], -1.0
        if len(c_grid) > 1:
            nested = list(StratifiedKFold(n_splits=nested_parts, shuffle=True, random_state=seed).split(train_index, y[train_index]))
            for c in c_grid:
                predicted = np.empty(len(train_index), dtype=np.int64)
                for inner_train, inner_held in nested:
                    predicted[inner_held] = fit(c, train_index[inner_train]).predict(x[train_index[inner_held]])
                score = macro_f1(y[train_index], predicted, classes)
                if score > best_score:
                    best_c, best_score = c, score
        fitted = fit(best_c, train_index)
        probs = np.zeros((len(heldout_index), classes))
        probs[:, fitted.classes_] = fitted.predict_proba(x[heldout_index])
        return probs

    return arm


def _rows(pool: Pool, video: str, count: int) -> list[dict[str, Any]]:
    target = int(pool.targets[pool.video_ids.index(video)])
    return [{"source_dataset": SOURCE_DATASET, "source_video_id": video, "source_frame_index": i, "target": target,
             "label": pool.class_names[target]} for i in range(count)]


def linear_vote_arm(*, epochs: int) -> Arm:
    """D-46 pipeline: linear head on the 8 sampled frames per video, every held-out frame votes."""

    import torch

    from core.p1c_head_variants import CHECKPOINT_LAST_EPOCH, train_vector_head
    from core.p1c_heads import GapLinearHead

    def arm(pool: Pool, train_index: np.ndarray, heldout_index: np.ndarray, seed: int) -> np.ndarray:
        classes = len(pool.class_names)
        train_videos = [pool.video_ids[i] for i in train_index]
        held_videos = [pool.video_ids[i] for i in heldout_index]
        train_parts = [pool.sampled_frames_of(v) for v in train_videos]
        held_parts = [pool.frames_of(v) for v in held_videos]
        train_rows = [row for v, part in zip(train_videos, train_parts) for row in _rows(pool, v, len(part))]
        held_rows = [row for v, part in zip(held_videos, held_parts) for row in _rows(pool, v, len(part))]
        torch.manual_seed(seed)
        head = GapLinearHead(in_dim=pool.dim, num_classes=classes)
        train_vector_head(
            head, np.concatenate(train_parts), train_rows, np.concatenate(held_parts), held_rows, pool.class_names,
            seed=seed, epochs=epochs, checkpoint_rule=CHECKPOINT_LAST_EPOCH, bootstrap_iterations=100,
        )
        head.eval()
        probs = np.zeros((len(held_videos), classes))
        with torch.no_grad():
            for position, part in enumerate(held_parts):
                votes = head(torch.from_numpy(part)).argmax(dim=1).numpy()
                probs[position] = np.bincount(votes, minlength=classes) / len(votes)
        return probs

    return arm


def permuted_sequences(pool: Pool, videos: Sequence[str], parts: Sequence[np.ndarray], permute_seed: int) -> dict:
    """`{(dataset, video): (frames in a seeded random order, row)}` -- the `step_permuted` input.

    The permutation generator is consumed video by video in the order of `videos`, so that order is
    part of the seed: pass the cache's own video order to reproduce a registered D-53 run exactly.
    """

    from core.p1c_head_variants import group_pooled_sequences_by_video

    rows = [row for v, part in zip(videos, parts) for row in _rows(pool, v, len(part))]
    return group_pooled_sequences_by_video(np.concatenate(parts), rows, permute_frame_order=True, permute_seed=permute_seed)


def fit_attention_head(pool: Pool, train_videos: Sequence[str], monitor_videos: Sequence[str], seed: int, **trainer_kwargs: Any):
    """Seeded D-53 `StepHead` trained on the 8-frame draw of `train_videos`; returns `(head in eval mode, monitor sequences, result)`.

    `monitor_videos` are scored every epoch with all their cached frames. Under the registered
    default they choose the epoch; with `checkpoint_rule=CHECKPOINT_LAST_EPOCH` they only draw a curve.
    """

    import torch

    from core.p1c_head_variants import train_sequence_head
    from core.p1c_heads import StepHead

    train_seq = permuted_sequences(pool, train_videos, [pool.sampled_frames_of(v) for v in train_videos], 1000 + seed)
    monitor_seq = permuted_sequences(pool, monitor_videos, [pool.frames_of(v) for v in monitor_videos], 2000 + seed)
    torch.manual_seed(seed)  # seeded head init, identical across arms (K-31)
    head = StepHead(token_dim=pool.dim, frames=FRAMES, depth=1, num_classes=len(pool.class_names))
    result = train_sequence_head(head, train_seq, monitor_seq, pool.class_names, frames=FRAMES, seed=seed, **trainer_kwargs)
    head.eval()
    return head, monitor_seq, result


def attention_arm(*, epochs: int) -> Arm:
    """D-53 pipeline: `StepHead` attention pooling with frame order destroyed (`step_permuted`)."""

    from core.p1c_head_variants import CHECKPOINT_LAST_EPOCH
    from core.p1c_oracle_interval import predict_video_probs

    def arm(pool: Pool, train_index: np.ndarray, heldout_index: np.ndarray, seed: int) -> np.ndarray:
        train_videos = [pool.video_ids[i] for i in train_index]
        held_videos = [pool.video_ids[i] for i in heldout_index]
        head, held_seq, _result = fit_attention_head(
            pool, train_videos, held_videos, seed, epochs=epochs, checkpoint_rule=CHECKPOINT_LAST_EPOCH, bootstrap_iterations=100,
        )
        return np.stack([predict_video_probs(head, held_seq[(SOURCE_DATASET, v)][0], FRAMES) for v in held_videos])

    return arm


# ------------------------------------------------------------------------------------------ scoring


def write_probabilities(path: Path, records: Sequence[tuple[str, Pool, dict[int, np.ndarray]]]) -> None:
    """`probs.csv` in the `core.failure_analysis.load_arm_predictions` layout (arm, fold, seed, video, true, p_*).

    The `seed` column holds the inner-CV repeat, so one (fold, repeat) pass is one bootstrap cell.
    """

    class_names = records[0][1].class_names
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "fold", "seed", "video", "true", *[f"p_{c}" for c in class_names]])
        for arm_name, pool, tables in records:
            for repeat in sorted(tables):
                for index, video in enumerate(pool.video_ids):
                    writer.writerow([arm_name, pool.fold, repeat, video, class_names[int(pool.targets[index])],
                                     *[f"{p:.6f}" for p in tables[repeat][index]]])


def cell_scores(pool: Pool, tables: dict[int, np.ndarray]) -> dict[str, list[float]]:
    """Macro-F1 and accuracy of each repeat's full held-out pass over the pool."""

    from core.confusion_groups import macro_f1

    predicted = {r: t.argmax(axis=1) for r, t in tables.items()}
    return {
        "macro_f1": [macro_f1(pool.targets, predicted[r], len(pool.class_names)) for r in sorted(tables)],
        "accuracy": [float((predicted[r] == pool.targets).mean()) for r in sorted(tables)],
    }
