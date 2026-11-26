"""Inner cross-validation harness (FINAL_PUSH_PLAN Step 1): pool loading, splits, arms, last-epoch rule."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import inner_cv

CLASSES = ["A", "B", "C"]
DIM = 12


def _write_split(directory: Path, videos: list[tuple[str, int, int]], rng: np.random.Generator) -> None:
    directory.mkdir(parents=True)
    rows, feats = [], []
    for video, target, count in videos:
        centre = np.zeros(DIM, dtype=np.float32)
        centre[target] = 4.0
        for frame in range(count):
            rows.append([len(rows), inner_cv.SOURCE_DATASET, video, frame * 5, target, CLASSES[target]])
            feats.append(centre + rng.normal(0, 0.3, DIM))
    np.save(directory / "features.float16.npy", np.asarray(feats, dtype=np.float16))
    with (directory / "rows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "source_dataset", "source_video_id", "source_frame_index", "target", "label"])
        writer.writerows(rows)
    (directory / "manifest.json").write_text(json.dumps({"cache_sha256": f"sha-{directory.name}", "class_names": CLASSES}))


def _make_cache(root: Path) -> None:
    rng = np.random.default_rng(0)
    train = [(f"{CLASSES[t]}{i:02d}", t, 10 + i) for t in range(3) for i in range(8)]
    validation = [(f"{CLASSES[t]}{i:02d}", t, 20) for t in range(3) for i in range(8, 10)]
    _write_split(root / "train", train, rng)
    _write_split(root / "validation", validation, rng)


class FinalTestRunnerTest(unittest.TestCase):
    """`scripts.run_final_test.score_fold` on a synthetic three-split cache (the real test folds are read once, D-73)."""

    def test_trains_on_train_plus_validation_and_predicts_only_test_videos(self) -> None:
        from scripts import run_final_test

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_cache(root)
            _write_split(root / "test", [(f"{CLASSES[t]}{i:02d}", t, 16) for t in range(3) for i in range(10, 14)], np.random.default_rng(5))
            default = inner_cv.load_pool(root, fold=1)
            self.assertEqual(len(default.video_ids), 30)                      # the default pool never loads test
            pool = inner_cv.load_pool(root, fold=1, splits=("train", "validation", "test"))
            self.assertEqual((len(pool.video_ids), len(pool.videos_of_split("test"))), (42, 12))
            runs, trained_on = run_final_test.score_fold(pool, [0], {"B_all": inner_cv.meanpool_logistic_arm(c_grid=(1.0,))}, attention=True)
            self.assertEqual(trained_on, 30)
            self.assertEqual(runs[0]["videos"], sorted(pool.videos_of_split("test")))
            probs = np.asarray(runs[0]["probs"])
            self.assertEqual(probs.shape, (12, 3))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)
            self.assertEqual(sorted(runs[0]["member_macro_f1"]), ["B_all", "attention"])
            self.assertGreater(runs[0]["member_macro_f1"]["B_all"], 0.9)

    def test_unknown_stack_member_is_refused(self) -> None:
        from scripts import run_final_test

        with self.assertRaises(ValueError):
            run_final_test._member_arms(["not_a_survivor"], 1, None)
        self.assertEqual(run_final_test._member_arms(["attention"], 1, None), {})
        self.assertEqual(run_final_test.FINAL_MEMBERS, ("L_study", "B+S_study", "B+VJ"))

    def test_clip_features_of_test_videos_fall_back_to_another_folds_cache(self) -> None:
        from core import inner_cv_features as feats

        def write(directory: Path, videos: list[str], value: float) -> None:
            directory.mkdir(parents=True)
            np.save(directory / "features.float16.npy", np.full((2 * len(videos), 4), value, dtype=np.float16))
            with (directory / "rows.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_index", "source_video_id"])
                writer.writerows([i, v] for i, v in enumerate(v for v in videos for _ in range(2)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_cache(root / "pool")
            _write_split(root / "pool" / "test", [("A99", 0, 9)], np.random.default_rng(1))
            pool = inner_cv.load_pool(root / "pool", fold=1, splits=("train", "validation", "test"))
            own = [v for v in pool.video_ids if v != "A99"]
            write(root / "own" / "train", own[:20], 1.0)
            write(root / "own" / "validation", own[20:], 1.0)
            write(root / "other" / "train", ["A99", own[0]], 5.0)
            write(root / "other" / "validation", [own[1]], 5.0)
            with self.assertRaises(ValueError):
                feats.clip_matrix(pool, root / "own")
            matrix = feats.clip_matrix(pool, root / "own", fallback_roots=[root / "other"])
            self.assertEqual(matrix[pool.video_ids.index("A99")].tolist(), [5.0] * 4)      # only the missing video uses the fallback
            self.assertEqual(matrix[pool.video_ids.index(own[0])].tolist(), [1.0] * 4)


class InnerCvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_cache(self.root)
        self.pool = inner_cv.load_pool(self.root, fold=1)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pool_merges_both_splits_by_whole_video(self) -> None:
        self.assertEqual(len(self.pool.video_ids), 30)
        self.assertEqual(np.bincount(self.pool.targets).tolist(), [10, 10, 10])
        self.assertEqual(self.pool.frames_of("A00").shape, (10, DIM))
        self.assertEqual(self.pool.frames_of("C09").shape, (20, DIM))
        self.assertEqual(self.pool.sampled_frames_of("C09").shape, (inner_cv.FRAMES, DIM))
        self.assertLessEqual(len(self.pool.sampled_frames_of("B03")), inner_cv.FRAMES)  # short videos may draw fewer
        self.assertEqual(self.pool.cache_sha256, {"train": "sha-train", "validation": "sha-validation"})
        self.assertEqual(self.pool.video_means().shape, (30, DIM))
        self.assertEqual(self.pool.light().features, [])

    def test_a_video_in_both_splits_is_refused(self) -> None:
        rng = np.random.default_rng(1)
        other = self.root / "dup"
        _write_split(other / "train", [("A00", 0, 9)], rng)
        _write_split(other / "validation", [("A00", 0, 9)], rng)
        with self.assertRaises(ValueError):
            inner_cv.load_pool(other, fold=1)

    def test_every_video_is_held_out_exactly_once_per_repeat(self) -> None:
        splits = inner_cv.inner_splits(self.pool.targets, parts=5, repeats=2, seed=0)
        self.assertEqual(len(splits), 10)
        for repeat in range(2):
            held = np.concatenate([h for r, _p, _t, h in splits if r == repeat])
            self.assertEqual(sorted(held.tolist()), list(range(30)))
        for _r, _p, train, held in splits:
            self.assertFalse(set(train.tolist()) & set(held.tolist()))
        self.assertFalse(np.array_equal(splits[0][3], splits[5][3]))  # repeats differ

    def test_arms_learn_a_separable_problem_from_heldout_predictions(self) -> None:
        splits = inner_cv.inner_splits(self.pool.targets, parts=5, repeats=1, seed=0)
        arms = {
            "meanpool": inner_cv.meanpool_logistic_arm(c_grid=(0.1, 1.0), nested_parts=2),
            "linear": inner_cv.linear_vote_arm(epochs=400),  # 3 steps per epoch at lr 3e-4: needs many epochs on a toy pool
            "attention": inner_cv.attention_arm(epochs=300),  # 1 step per epoch on a toy pool
        }
        for name, arm in arms.items():
            tables = inner_cv.run_arm(self.pool, arm, splits)
            self.assertEqual(tables[0].shape, (30, 3))
            np.testing.assert_allclose(tables[0].sum(axis=1), 1.0, atol=1e-4)
            scores = inner_cv.cell_scores(self.pool, tables)
            self.assertGreater(scores["macro_f1"][0], 0.9, name)

    def test_arm_with_wrong_shape_is_refused(self) -> None:
        splits = inner_cv.inner_splits(self.pool.targets, parts=5, repeats=1, seed=0)
        with self.assertRaises(ValueError):
            inner_cv.run_arm(self.pool, lambda pool, tr, he, seed: np.zeros((len(he), 2)), splits)

    def test_probabilities_round_trip_through_the_bootstrap_loader(self) -> None:
        from core.failure_analysis import load_arm_predictions, paired_cluster_bootstrap

        splits = inner_cv.inner_splits(self.pool.targets, parts=5, repeats=2, seed=0)
        tables = inner_cv.run_arm(self.pool, inner_cv.meanpool_logistic_arm(c_grid=(1.0,)), splits)
        path = self.root / "probs.csv"
        inner_cv.write_probabilities(path, [("ref", self.pool, tables), ("same", self.pool, tables)])
        base = load_arm_predictions(path, "ref")
        self.assertEqual(len(base.video), 60)
        self.assertEqual(sorted(set(base.seed.tolist())), [0, 1])
        result = paired_cluster_bootstrap(base, load_arm_predictions(path, "same"), draws=20)
        self.assertEqual(result["difference"]["observed"], 0.0)


class LastEpochRuleTest(unittest.TestCase):
    def test_last_epoch_rule_keeps_final_weights_and_never_stops_early(self) -> None:
        from core.p1c_head_variants import CHECKPOINT_LAST_EPOCH, _fit_head

        class Model:
            def __init__(self) -> None:
                self.step = 0

            def state_dict(self) -> dict:
                return {"step": self.step}

        model = Model()
        metrics = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]  # best first, then a long plateau

        def train(_epoch: int) -> dict:
            model.step += 1
            return {"train_examples": 1, "train_batches": 1, "train_mean_loss": 0.0, "model_state_dict_source": model}

        def validate() -> tuple[float, np.ndarray]:
            return metrics[model.step - 1], np.array([model.step])

        last = _fit_head(run_train_epoch=train, run_validation=validate, epochs=8, patience=2, min_delta=0.0,
                         checkpoint_rule=CHECKPOINT_LAST_EPOCH)
        self.assertEqual((last.best_epoch, last.best_state_dict, last.stopped_early), (7, {"step": 8}, False))
        self.assertEqual(len(last.epoch_records), 8)

        model.step = 0
        best = _fit_head(run_train_epoch=train, run_validation=validate, epochs=8, patience=2, min_delta=0.0)
        self.assertEqual((best.best_epoch, best.best_state_dict, best.stopped_early), (0, {"step": 1}, True))

        with self.assertRaises(ValueError):
            _fit_head(run_train_epoch=train, run_validation=validate, epochs=1, patience=1, min_delta=0.0, checkpoint_rule="x")


class CandidateFeatureTest(unittest.TestCase):
    """Step 2 feature matrices and the Holm family rule."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_cache(self.root)
        self.pool = inner_cv.load_pool(self.root, fold=1)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_matched_matrix_averages_exactly_the_study_frames(self) -> None:
        from core import inner_cv_features as feats

        row_videos = [v for v in self.pool.video_ids for _ in range(2)]
        row_frames = np.asarray([f for _v in self.pool.video_ids for f in (0, 15)], dtype=np.int64)  # cached rows 0 and 3
        matched = feats.matched_pool_matrix(self.pool, row_videos, row_frames)
        expected = self.pool.frames_of("A00")[[0, 3]].mean(axis=0)
        np.testing.assert_allclose(matched[self.pool.video_ids.index("A00")], expected, rtol=1e-6)
        with self.assertRaises(ValueError):
            feats.matched_pool_matrix(self.pool, row_videos, row_frames + 1)  # frame 1 is not a cached stride-5 frame

    def test_study_matrix_requires_every_pool_video(self) -> None:
        from core import inner_cv_features as feats

        rows = [v for v in self.pool.video_ids for _ in range(3)]
        features = np.arange(len(rows) * 2, dtype=np.float32).reshape(len(rows), 2)
        matrix = feats.study_matrix(self.pool, rows, features)
        np.testing.assert_allclose(matrix[0], features[:3].mean(axis=0))
        with self.assertRaises(ValueError):
            feats.study_matrix(self.pool, rows[3:], features[3:])

    def test_span_matrix_uses_only_frames_inside_the_span(self) -> None:
        from core import inner_cv_features as feats

        matrix, used = feats.span_matrix(self.pool, {"A00": [(0, 10)], "B00": [(10_000, 10_010)]})  # frames 0, 5, 10 of A00
        self.assertEqual(used, 1)  # B00's span holds no cached frame, so it keeps its full mean
        index = self.pool.video_ids.index("A00")
        np.testing.assert_allclose(matrix[index], self.pool.frames_of("A00")[:3].mean(axis=0), rtol=1e-6)
        other = self.pool.video_ids.index("B00")
        np.testing.assert_allclose(matrix[other], self.pool.video_means()[other])

    def test_training_matrix_and_extra_rows_change_only_the_training_side(self) -> None:
        splits = inner_cv.inner_splits(self.pool.targets, parts=5, repeats=1, seed=0)
        plain = inner_cv.run_arm(self.pool, inner_cv.meanpool_logistic_arm(c_grid=(1.0,)), splits)
        zeros = np.zeros_like(self.pool.video_means())
        broken = inner_cv.run_arm(self.pool, inner_cv.meanpool_logistic_arm(c_grid=(1.0,), train_matrix=lambda _p: zeros), splits)
        self.assertGreater(inner_cv.cell_scores(self.pool, plain)["macro_f1"][0], 0.9)
        self.assertLess(inner_cv.cell_scores(self.pool, broken)["macro_f1"][0], 0.5)  # trained on zeros, scored on real vectors
        extra = (self.pool.video_means()[:6], self.pool.targets[:6])
        with_extra = inner_cv.run_arm(
            self.pool, inner_cv.meanpool_logistic_arm(c_grid=(0.1, 1.0), nested_parts=2, extra=lambda _p: extra, class_weight="balanced"), splits)
        self.assertGreater(inner_cv.cell_scores(self.pool, with_extra)["macro_f1"][0], 0.9)

    def test_holm_stops_at_the_first_failure_and_needs_the_minimum_gain(self) -> None:
        from core.inner_cv_report import holm_family

        family = [("a", "x"), ("b", "x"), ("c", "x")]
        contrasts = [
            {"candidate": "a", "base": "x", "difference": 0.05, "one_sided_p": 0.001},
            {"candidate": "b", "base": "x", "difference": 0.01, "one_sided_p": 0.002},   # significant, but below +0.02
            {"candidate": "c", "base": "x", "difference": 0.04, "one_sided_p": 0.020},
            {"candidate": "d", "base": "x", "difference": 0.09, "one_sided_p": 0.400},   # outside the family
        ]
        rows = {r["candidate"]: r for r in holm_family(contrasts, family)}
        self.assertEqual(sorted(rows), ["a", "b", "c"])
        self.assertEqual([rows[k]["passes"] for k in "abc"], [True, False, True])
        self.assertAlmostEqual(rows["a"]["holm_threshold"], 0.025 / 3)
        contrasts[1]["one_sided_p"] = 0.015  # b now misses 0.025/2, so c (the larger p) cannot pass either
        rows = {r["candidate"]: r for r in holm_family(contrasts, family)}
        self.assertEqual([rows[k]["significant"] for k in "abc"], [True, False, False])


if __name__ == "__main__":
    unittest.main()
