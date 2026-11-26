from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

import psutil

from core.precision import resolve_precision_policy  # noqa: E402
from core.throughput import resolve_throughput_policy  # noqa: E402
from core.p1c_engine import (
    FrameClassifierEngine,
    capture_resume_state,
    load_training_checkpoint,
    save_training_checkpoint,
)
from scripts.run_p1c_frame_baseline import _PeakMemory


class TinyFrameClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images.mean(dim=(2, 3)))


def batches() -> list[dict[str, object]]:
    return [
        {
            "image": torch.tensor(
                [
                    [[[1.0]], [[0.0]], [[0.0]]],
                    [[[0.0]], [[1.0]], [[0.0]]],
                ]
            ),
            "target": torch.tensor([0, 1]),
            "source_dataset": ["fixture-a", "fixture-a"],
            "source_video_id": ["video-a", "video-b"],
        },
        {
            "image": torch.tensor(
                [
                    [[[1.0]], [[0.0]], [[0.0]]],
                    [[[0.0]], [[1.0]], [[0.0]]],
                ]
            ),
            "target": torch.tensor([0, 1]),
            "source_dataset": ["fixture-b", "fixture-b"],
            "source_video_id": ["video-a", "video-b"],
        },
    ]


class P1CEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.model = TinyFrameClassifier()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.engine = FrameClassifierEngine(
            self.model,
            optimizer=self.optimizer,
            loss=nn.CrossEntropyLoss(),
            loss_reduction="mean",
            device="cpu",
            class_names=("Abuse", "Normal"),
            precision=resolve_precision_policy("float32", "cpu"),
            throughput=resolve_throughput_policy(None, "cpu"),
        )

    def test_train_and_evaluate_preserve_source_video_traceability(self) -> None:
        before = self.model.classifier.weight.detach().clone()
        trained = self.engine.train_epoch(batches())
        self.assertEqual((trained.examples, trained.batches), (4, 2))
        self.assertTrue(np.isfinite(trained.mean_loss))
        self.assertFalse(torch.equal(before, self.model.classifier.weight))

        first = self.engine.evaluate(batches(), bootstrap_iterations=100, bootstrap_seed=19)
        second = self.engine.evaluate(batches(), bootstrap_iterations=100, bootstrap_seed=19)
        self.assertEqual(first.predictions, second.predictions)
        self.assertEqual(first.source_video_ids, ("fixture-a\0video-a", "fixture-a\0video-b", "fixture-b\0video-a", "fixture-b\0video-b"))
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.video_clustered_macro_f1_ci, second.video_clustered_macro_f1_ci)
        self.assertEqual(first.video_clustered_macro_f1_ci["resampling_unit"], "source_video_id")

    def test_refuses_missing_source_traceability_and_implicit_bootstrap(self) -> None:
        invalid = batches()
        invalid[0] = {key: value for key, value in invalid[0].items() if key != "source_video_id"}
        with self.assertRaisesRegex(ValueError, "source_video_id"):
            self.engine.train_epoch(invalid)
        with self.assertRaisesRegex(ValueError, "both be supplied"):
            self.engine.evaluate(batches(), bootstrap_iterations=100, bootstrap_seed=None)

    def test_checkpoint_round_trip_restores_model_optimizer_progress_and_rng(self) -> None:
        self.engine.train_epoch(batches())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            random.seed(7)
            np.random.seed(8)
            torch.manual_seed(9)
            state = capture_resume_state(next_epoch=3, global_step=12, extra_state={"selection": "validation"})
            expected = (random.random(), float(np.random.random()), float(torch.rand(())))
            saved_weight = self.model.classifier.weight.detach().clone()
            save_training_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                state=state,
                allow_overwrite=False,
            )
            with self.assertRaises(FileExistsError):
                save_training_checkpoint(
                    path,
                    model=self.model,
                    optimizer=self.optimizer,
                    state=state,
                    allow_overwrite=False,
                )
            with torch.no_grad():
                self.model.classifier.weight.zero_()
            random.seed(70)
            np.random.seed(80)
            torch.manual_seed(90)
            restored = load_training_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                device="cpu",
                restore_rng=True,
            )
            self.assertEqual((restored.next_epoch, restored.global_step), (3, 12))
            self.assertEqual(restored.extra_state, {"selection": "validation"})
            self.assertTrue(torch.equal(self.model.classifier.weight, saved_weight))
            self.assertEqual((random.random(), float(np.random.random()), float(torch.rand(()))), expected)

    def test_resume_state_captures_every_present_accelerator_rng(self) -> None:
        # A checkpoint that omits the RNG of the device it was produced on cannot
        # claim exact resume, and MPS is the device these runs execute on.
        state = capture_resume_state(next_epoch=0, global_step=0, extra_state={})
        self.assertLessEqual({"python", "numpy", "torch_cpu"}, set(state.rng_state))
        self.assertEqual("torch_mps" in state.rng_state, torch.backends.mps.is_available())
        self.assertEqual("torch_cuda" in state.rng_state, torch.cuda.is_available())

    def test_mps_rng_survives_a_checkpoint_round_trip(self) -> None:
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable on this host")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.mps.manual_seed(4242)
            state = capture_resume_state(next_epoch=1, global_step=1, extra_state={})
            expected = torch.rand(4, device="mps").cpu()
            save_training_checkpoint(
                path, model=self.model, optimizer=self.optimizer, state=state, allow_overwrite=False
            )
            torch.mps.manual_seed(1)
            torch.rand(4, device="mps")
            load_training_checkpoint(
                path, model=self.model, optimizer=self.optimizer, device="cpu", restore_rng=True
            )
            self.assertTrue(torch.equal(torch.rand(4, device="mps").cpu(), expected))


class PeakMemoryTelemetryTests(unittest.TestCase):
    """COMPUTE_POLICY section 3 requires a real peak, not a literal zero."""

    def test_peak_is_a_non_negative_int_the_lifecycle_will_accept(self) -> None:
        for backend in ({"cpu", "mps"} if torch.backends.mps.is_available() else {"cpu"}):
            with self.subTest(backend=backend):
                tracker = _PeakMemory(torch.device(backend), 1_000)
                tracker.sample(psutil.Process())
                value = tracker.lifecycle_peak_bytes()
                self.assertIsInstance(value, int)
                self.assertNotIsInstance(value, bool)
                self.assertGreaterEqual(value, 0)

    def test_snapshot_reports_both_device_and_host_memory(self) -> None:
        tracker = _PeakMemory(torch.device("cpu"), 1_000)
        tracker.sample(psutil.Process())
        snapshot = tracker.snapshot()
        self.assertEqual(
            set(snapshot),
            {
                "peak_device_allocated_bytes",
                "peak_process_rss_bytes",
                "process_rss_growth_bytes",
                "device_memory_report",
            },
        )

    def test_cpu_runs_fall_back_to_host_growth_rather_than_reporting_nothing(self) -> None:
        # CPU has no device allocator, so a zero here would be indistinguishable
        # from the hardcoded zero this replaced.
        tracker = _PeakMemory(torch.device("cpu"), 0)
        tracker.sample(psutil.Process())
        self.assertGreater(tracker.lifecycle_peak_bytes(), 0)
        self.assertIsNone(tracker.snapshot()["peak_device_allocated_bytes"])

    def test_peak_never_decreases_across_samples(self) -> None:
        tracker = _PeakMemory(torch.device("cpu"), 1_000)
        process = psutil.Process()
        tracker.sample(process)
        first = tracker.lifecycle_peak_bytes()
        tracker.sample(process)
        self.assertGreaterEqual(tracker.lifecycle_peak_bytes(), first)


if __name__ == "__main__":
    unittest.main()


class PrecisionWiringTests(unittest.TestCase):
    """The engine must honour the policy it is given, not choose one.

    These run on CPU with float32, which is the only policy CPU allows. That is
    enough to pin the two properties that had actually broken: the engine
    refuses to invent a precision, and float32 still means no scaler and no
    skipped steps.
    """

    def _engine(self, precision):  # noqa: ANN001
        torch.manual_seed(31)
        model = TinyFrameClassifier()
        return FrameClassifierEngine(
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss=nn.CrossEntropyLoss(),
            loss_reduction="mean",
            device="cpu",
            class_names=("Abuse", "Normal"),
            precision=precision,
            throughput=resolve_throughput_policy(None, "cpu"),
        )

    def _batch(self):
        return {
            "image": torch.randn(4, 3, 8, 8),
            "target": torch.tensor([0, 1, 0, 1]),
            "source_dataset": ("d",) * 4,
            "source_video_id": ("v1", "v2", "v3", "v4"),
        }

    def test_engine_refuses_a_precision_it_did_not_resolve(self) -> None:
        # A bare string is the tempting mistake: it looks like the config value.
        for bad in ("float32", None, torch.float32):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                self._engine(bad)

    def test_float32_reports_no_skipped_steps(self) -> None:
        engine = self._engine(resolve_precision_policy("float32", "cpu"))
        result = engine.train_epoch([self._batch()])
        self.assertEqual(result.skipped_optimizer_steps, 0)
        self.assertEqual(result.batches, 1)

    def test_float32_still_updates_the_weights(self) -> None:
        # Guards against the autocast wiring accidentally disabling the step.
        engine = self._engine(resolve_precision_policy("float32", "cpu"))
        before = engine.model.classifier.weight.detach().clone()
        engine.train_epoch([self._batch()])
        self.assertFalse(torch.equal(before, engine.model.classifier.weight))

    def test_evaluation_runs_under_the_same_policy(self) -> None:
        engine = self._engine(resolve_precision_policy("float32", "cpu"))
        result = engine.evaluate([self._batch()], bootstrap_iterations=None, bootstrap_seed=None)
        self.assertEqual(len(result.predictions), 4)


class ThroughputWiringTests(unittest.TestCase):
    """The engine must honour the throughput policy it is given, not choose one."""

    def _engine(self, throughput):  # noqa: ANN001
        torch.manual_seed(31)
        model = TinyFrameClassifier()
        return FrameClassifierEngine(
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss=nn.CrossEntropyLoss(),
            loss_reduction="mean",
            device="cpu",
            class_names=("Abuse", "Normal"),
            precision=resolve_precision_policy("float32", "cpu"),
            throughput=throughput,
        )

    def test_engine_refuses_a_throughput_it_did_not_resolve(self) -> None:
        for bad in (True, None, "none", {"channels_last": False, "compile": "none", "fused_optimizer": False}):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                self._engine(bad)

    def test_disabled_throughput_leaves_model_identity_as_checkpoint_module(self) -> None:
        engine = self._engine(resolve_throughput_policy(None, "cpu"))
        self.assertIs(engine.checkpoint_module, engine.model)
