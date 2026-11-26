"""Tests for `core.throughput`.

Most of these run on CPU, where every knob is refused, so they pin the
*decision* logic rather than kernel behaviour: which blocks are legal, which
combinations are refused, and that the disabled policy is exactly what an
absent `training.throughput` block resolves to.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.throughput import (  # noqa: E402
    COMPILE_MODES,
    ThroughputPolicy,
    resolve_throughput_policy,
)

CUDA = torch.cuda.is_available()


class ResolutionTests(unittest.TestCase):
    def test_absent_block_is_the_disabled_policy(self) -> None:
        policy = resolve_throughput_policy(None, "cpu")
        self.assertEqual(policy, ThroughputPolicy.disabled("cpu"))
        self.assertFalse(policy.channels_last)
        self.assertEqual(policy.compile_mode, "none")
        self.assertFalse(policy.fused_optimizer)
        self.assertEqual(policy.device_type, "cpu")

    def test_non_mapping_block_is_refused(self) -> None:
        for bad in (True, "channels_last", 1, ["channels_last"]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                resolve_throughput_policy(bad, "cpu")

    def test_unknown_key_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            resolve_throughput_policy(
                {"channels_first": True, "compile": "none", "fused_optimizer": False},
                "cpu",
            )

    def test_missing_key_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            resolve_throughput_policy({"channels_last": True, "compile": "none"}, "cpu")

    def test_wrong_types_are_refused(self) -> None:
        with self.assertRaises(TypeError):
            resolve_throughput_policy(
                {"channels_last": "true", "compile": "none", "fused_optimizer": False}, "cpu"
            )
        with self.assertRaises(TypeError):
            resolve_throughput_policy(
                {"channels_last": False, "compile": "none", "fused_optimizer": "false"}, "cpu"
            )

    def test_bad_compile_mode_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            resolve_throughput_policy(
                {"channels_last": False, "compile": "turbo", "fused_optimizer": False}, "cpu"
            )

    def test_every_registered_compile_mode_resolves_on_cpu_when_everything_else_is_off(self) -> None:
        for mode in COMPILE_MODES:
            spec = {"channels_last": False, "compile": mode, "fused_optimizer": False}
            if mode == "none":
                policy = resolve_throughput_policy(spec, "cpu")
                self.assertEqual(policy.compile_mode, "none")
            else:
                with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                    resolve_throughput_policy(spec, "cpu")

    def test_each_knob_enabled_on_cpu_refuses_rather_than_degrading(self) -> None:
        cases = [
            {"channels_last": True, "compile": "none", "fused_optimizer": False},
            {"channels_last": False, "compile": "default", "fused_optimizer": False},
            {"channels_last": False, "compile": "none", "fused_optimizer": True},
        ]
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(RuntimeError) as caught:
                resolve_throughput_policy(spec, "cpu")
            self.assertIn("CUDA", str(caught.exception))

    def test_all_off_is_legal_on_cpu(self) -> None:
        policy = resolve_throughput_policy(
            {"channels_last": False, "compile": "none", "fused_optimizer": False}, "cpu"
        )
        self.assertEqual(policy, ThroughputPolicy.disabled("cpu"))


class OptimizerKwargsTests(unittest.TestCase):
    def test_fused_optimizer_adds_the_kwarg(self) -> None:
        policy = ThroughputPolicy(channels_last=False, compile_mode="none", fused_optimizer=True, device_type="cuda")
        self.assertEqual(policy.optimizer_kwargs(), {"fused": True})

    def test_disabled_fused_optimizer_adds_nothing(self) -> None:
        policy = ThroughputPolicy.disabled("cpu")
        self.assertEqual(policy.optimizer_kwargs(), {})


class MemoryFormatTests(unittest.TestCase):
    def test_channels_last_reports_channels_last_memory_format(self) -> None:
        policy = ThroughputPolicy(channels_last=True, compile_mode="none", fused_optimizer=False, device_type="cuda")
        self.assertEqual(policy.memory_format, torch.channels_last)

    def test_disabled_reports_contiguous_memory_format(self) -> None:
        policy = ThroughputPolicy.disabled("cpu")
        self.assertEqual(policy.memory_format, torch.contiguous_format)

    def test_prepare_input_returns_channels_last_strided_tensor_when_enabled(self) -> None:
        policy = ThroughputPolicy(channels_last=True, compile_mode="none", fused_optimizer=False, device_type="cpu")
        images = torch.randn(2, 3, 8, 8)
        prepared = policy.prepare_input(images)
        self.assertTrue(prepared.is_contiguous(memory_format=torch.channels_last))

    def test_prepare_input_is_a_no_op_when_disabled(self) -> None:
        policy = ThroughputPolicy.disabled("cpu")
        images = torch.randn(2, 3, 8, 8)
        prepared = policy.prepare_input(images)
        self.assertTrue(prepared.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(torch.equal(prepared, images))


class PrepareModelTests(unittest.TestCase):
    def test_disabled_policy_returns_the_same_module(self) -> None:
        policy = ThroughputPolicy.disabled("cpu")
        model = torch.nn.Linear(3, 2)
        self.assertIs(policy.prepare_model(model), model)

    def test_channels_last_returns_the_same_module_reformatted(self) -> None:
        policy = ThroughputPolicy(channels_last=True, compile_mode="none", fused_optimizer=False, device_type="cpu")
        model = torch.nn.Conv2d(3, 4, kernel_size=3)
        prepared = policy.prepare_model(model)
        self.assertIs(prepared, model)
        self.assertTrue(prepared.weight.is_contiguous(memory_format=torch.channels_last))

    @unittest.skipUnless(CUDA, "torch.compile validated only against CUDA in this project")
    def test_compile_wraps_the_module(self) -> None:
        policy = ThroughputPolicy(channels_last=False, compile_mode="default", fused_optimizer=False, device_type="cuda")
        model = torch.nn.Linear(3, 2)
        prepared = policy.prepare_model(model)
        self.assertIsNot(prepared, model)


class ReportTests(unittest.TestCase):
    def test_report_states_what_actually_ran(self) -> None:
        report = ThroughputPolicy.disabled("cpu").report()
        self.assertEqual(
            report,
            {
                "channels_last": False,
                "compile_mode": "none",
                "fused_optimizer": False,
                "device_type": "cpu",
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
