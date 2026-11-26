"""Tests for `core.precision`.

Most of these run on CPU, where mixed precision is refused outright, so they
pin the *decision* logic rather than kernel behaviour: which names are legal,
which combinations are refused, and -- the one that actually caused a defect --
that float16 never comes back without a gradient scaler while bfloat16 never
comes back with one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.precision import (  # noqa: E402
    AMP_BFLOAT16, AMP_FLOAT16, FLOAT32, PRECISIONS, PrecisionPolicy,
    resolve_precision_policy,
)

CUDA = torch.cuda.is_available()


class ResolutionTests(unittest.TestCase):
    def test_float32_needs_no_accelerator(self) -> None:
        policy = resolve_precision_policy(FLOAT32, "cpu")
        self.assertFalse(policy.is_mixed)
        self.assertIsNone(policy.autocast_dtype)
        self.assertFalse(policy.uses_grad_scaler)
        self.assertIsNone(policy.make_grad_scaler())

    def test_unregistered_names_are_refused(self) -> None:
        # Including plausible-looking aliases: the vocabulary is exact.
        for name in ("bf16", "fp16", "mixed", "amp", "AMP_FLOAT16", "", None, 16):
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_precision_policy(name, "cpu")

    def test_mixed_precision_refuses_cpu_rather_than_degrading(self) -> None:
        # Silent fallback would make a run slower than its registration claims
        # and report numerics it never used.
        for name in (AMP_BFLOAT16, AMP_FLOAT16):
            with self.subTest(name=name), self.assertRaises(RuntimeError) as caught:
                resolve_precision_policy(name, "cpu")
            self.assertIn("CUDA", str(caught.exception))

    def test_every_registered_name_is_resolvable_or_explicitly_refused(self) -> None:
        for name in PRECISIONS:
            try:
                policy = resolve_precision_policy(name, "cuda" if CUDA else "cpu")
            except RuntimeError:
                continue  # hardware refusal is a legal outcome, silence is not
            self.assertIsInstance(policy, PrecisionPolicy)
            self.assertEqual(policy.name, name)


class ScalerPairingTests(unittest.TestCase):
    """float16 must be scaled; bfloat16 must not.

    Getting this backwards is silent: float16 without a scaler underflows small
    gradients to zero and simply trains worse, and bfloat16 with one adds
    pointless work. Neither raises.
    """

    def _policy(self, name: str) -> PrecisionPolicy:
        if not CUDA:
            self.skipTest("mixed precision requires CUDA")
        try:
            return resolve_precision_policy(name, "cuda")
        except RuntimeError as exc:  # pragma: no cover - hardware dependent
            self.skipTest(str(exc))

    def test_float16_always_carries_a_scaler(self) -> None:
        policy = self._policy(AMP_FLOAT16)
        self.assertTrue(policy.uses_grad_scaler)
        self.assertIsNotNone(policy.make_grad_scaler())
        self.assertEqual(policy.autocast_dtype, torch.float16)

    def test_bfloat16_never_carries_a_scaler(self) -> None:
        policy = self._policy(AMP_BFLOAT16)
        self.assertFalse(policy.uses_grad_scaler)
        self.assertIsNone(policy.make_grad_scaler())
        self.assertEqual(policy.autocast_dtype, torch.bfloat16)

    def test_each_engine_gets_its_own_scaler(self) -> None:
        # A shared scaler would couple two runs' training through its scale.
        policy = self._policy(AMP_FLOAT16)
        self.assertIsNot(policy.make_grad_scaler(), policy.make_grad_scaler())


class ReportTests(unittest.TestCase):
    def test_report_states_what_actually_ran(self) -> None:
        report = resolve_precision_policy(FLOAT32, "cpu").report()
        self.assertEqual(report["precision"], FLOAT32)
        self.assertFalse(report["autocast"])
        self.assertIsNone(report["autocast_dtype"])
        self.assertFalse(report["grad_scaler"])
        self.assertEqual(report["device_type"], "cpu")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
