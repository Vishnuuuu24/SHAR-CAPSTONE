from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from core.device import (
    accelerator_rng_state,
    memory_report,
    resolve_device_policy,
    restore_accelerator_rng_state,
    select_device,
    supports_pin_memory,
    synchronize,
)
from core.provenance import REQUIRED_PROVENANCE_FIELDS, finalize_provenance, validate_provenance
from core.reproducibility import deterministic_noise_fixture, seed_everything, stable_sample_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
FILE_DIGEST = hashlib.sha256(b"fixture-image").hexdigest()


class ReproducibilityTests(unittest.TestCase):
    def test_stable_seed_and_noise_repeat_in_process(self) -> None:
        args = ("Train/Abuse/frame.png", FILE_DIGEST, 7, "gaussian-fixture")
        self.assertEqual(stable_sample_seed(*args), stable_sample_seed(*args))
        self.assertTrue(np.array_equal(deterministic_noise_fixture(*args, (3, 8, 8)), deterministic_noise_fixture(*args, (3, 8, 8))))

    def test_noise_is_bitwise_stable_across_fresh_process(self) -> None:
        code = (
            "import hashlib; from core.reproducibility import deterministic_noise_fixture; "
            f"d={FILE_DIGEST!r}; a=deterministic_noise_fixture('Train/Abuse/frame.png',d,7,'gaussian-fixture',(3,8,8)); "
            "print(hashlib.sha256(a.tobytes()).hexdigest())"
        )
        outputs = [
            subprocess.check_output([sys.executable, "-c", code], cwd=REPO_ROOT, text=True).strip()
            for _ in range(2)
        ]
        self.assertEqual(outputs[0], outputs[1])

    def test_noise_is_stable_across_worker_counts(self) -> None:
        def make(index: int) -> bytes:
            return deterministic_noise_fixture(
                f"Train/Abuse/frame-{index}.png", FILE_DIGEST, 7, "gaussian-fixture", (3, 8, 8)
            ).tobytes()

        serial = [make(index) for index in range(8)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel = list(executor.map(make, range(8)))
        self.assertEqual(serial, parallel)

    def test_global_seed_repeats_python_numpy_and_torch(self) -> None:
        seed_everything(11)
        first_numpy = np.random.rand(4)
        first_torch = torch.rand(4)
        seed_everything(11)
        self.assertTrue(np.array_equal(first_numpy, np.random.rand(4)))
        self.assertTrue(torch.equal(first_torch, torch.rand(4)))


class DeviceTests(unittest.TestCase):
    def test_no_silent_cpu_fallback(self) -> None:
        if torch.backends.mps.is_available():
            device, report = select_device(prefer_mps=True, allow_cpu_fallback=False)
            self.assertEqual(device.type, "mps")
            self.assertEqual(report.selected, "mps")
        else:
            with self.assertRaisesRegex(RuntimeError, "refusing silent CPU fallback"):
                select_device(prefer_mps=True, allow_cpu_fallback=False)

    def test_unknown_policy_is_refused(self) -> None:
        for policy in ("gpu", "auto", "mps", "cpu", None, 1):
            with self.assertRaisesRegex(ValueError, "device_policy must be one of"):
                resolve_device_policy(policy)

    def test_explicit_cpu_policy_requires_a_thread_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive explicit cpu_threads"):
            resolve_device_policy("cpu_strict_deterministic")
        with self.assertRaisesRegex(ValueError, "positive explicit cpu_threads"):
            resolve_device_policy("cpu_strict_deterministic", cpu_threads=0)

    def test_explicit_cpu_policy_pins_threads_and_reports_them(self) -> None:
        before = torch.get_num_threads()
        try:
            device, report = resolve_device_policy("cpu_strict_deterministic", cpu_threads=2)
            self.assertEqual(device.type, "cpu")
            self.assertEqual((report.selected, report.policy, report.torch_threads), ("cpu", "cpu_strict_deterministic", 2))
            self.assertFalse(report.cpu_fallback_allowed)
        finally:
            torch.set_num_threads(before)

    def test_accelerator_policies_refuse_when_the_backend_is_absent(self) -> None:
        for policy, present in (
            ("cuda_strict", torch.cuda.is_available()),
            ("mps_strict", torch.backends.mps.is_available()),
        ):
            with self.subTest(policy=policy, present=present):
                if present:
                    device, report = resolve_device_policy(policy)
                    self.assertEqual(report.policy, policy)
                    self.assertEqual(device.type, policy.split("_")[0])
                else:
                    with self.assertRaisesRegex(RuntimeError, "refusing silent"):
                        resolve_device_policy(policy)

    def test_only_the_named_warn_only_policy_downgrades_determinism(self) -> None:
        # B-006 exists because one specific policy tolerates a non-deterministic
        # operator. No other policy may quietly acquire that tolerance.
        if not torch.backends.mps.is_available():
            self.skipTest("MPS policies are not resolvable on this host")
        self.assertFalse(resolve_device_policy("mps_strict")[1].deterministic_warn_only)
        self.assertTrue(resolve_device_policy("mps_warn_only_repeatability")[1].deterministic_warn_only)

    def test_resolving_a_policy_actually_enforces_matching_determinism(self) -> None:
        # Execution-based, not metadata-based: the P1C revalidation found that
        # DeviceReport.deterministic_warn_only was computed but never applied,
        # so torch.use_deterministic_algorithms stayed untouched regardless of
        # which policy a config named. This asserts torch's real global state,
        # not just the report field, so a future refactor cannot silently
        # reintroduce the gap without breaking this test.
        self.addCleanup(torch.use_deterministic_algorithms, False)
        before = torch.get_num_threads()
        try:
            torch.use_deterministic_algorithms(False)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
            device, report = resolve_device_policy("cpu_strict_deterministic", cpu_threads=1)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(report.deterministic_warn_only)
        finally:
            torch.set_num_threads(before)

        if torch.backends.mps.is_available():
            torch.use_deterministic_algorithms(False)
            resolve_device_policy("mps_strict")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

            torch.use_deterministic_algorithms(False)
            resolve_device_policy("mps_warn_only_repeatability")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_pin_memory_support_is_backend_specific(self) -> None:
        self.assertTrue(supports_pin_memory("cuda"))
        self.assertFalse(supports_pin_memory("mps"))
        self.assertFalse(supports_pin_memory("cpu"))

    def test_synchronize_and_memory_report_are_defined_for_every_backend(self) -> None:
        for backend in ("cpu", "mps", "cuda"):
            with self.subTest(backend=backend):
                report = memory_report(backend)
                self.assertEqual(
                    set(report),
                    {"current_allocated_memory", "driver_allocated_memory", "recommended_max_memory", "peak_allocated_memory"},
                )
        synchronize("cpu")  # must not raise on a backend with nothing to await

    def test_accelerator_rng_round_trips_present_backends(self) -> None:
        state = accelerator_rng_state()
        self.assertEqual("torch_mps" in state, torch.backends.mps.is_available())
        self.assertEqual("torch_cuda" in state, torch.cuda.is_available())
        restore_accelerator_rng_state(state)

    def test_restoring_rng_for_an_absent_backend_is_refused(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is present on this host")
        with self.assertRaisesRegex(RuntimeError, "CUDA RNG state but CUDA is unavailable"):
            restore_accelerator_rng_state({"torch_cuda": [torch.zeros(1)]})


class ProvenanceTests(unittest.TestCase):
    def good_record(self) -> dict:
        return {
            "run_id": "fixture-run",
            "run_kind": "smoke",
            "config_digest": "a" * 64,
            "code_revision": "fixture-revision",
            "seed": 0,
            "package_versions": {"torch": torch.__version__},
            "dataset_manifest_digest": "b" * 64,
            "annotation_version": "fixture-v1",
            "environment_digest": "c" * 64,
            "hardware": {"device": "fixture"},
            "metric_artifact_paths": ["metrics.json"],
        }

    def test_required_fields_match_contract(self) -> None:
        self.assertEqual(set(self.good_record()), REQUIRED_PROVENANCE_FIELDS)
        self.assertEqual(validate_provenance(self.good_record()), [])

    def test_missing_field_refuses_closure(self) -> None:
        record = self.good_record()
        record.pop("annotation_version")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "annotation_version"):
                finalize_provenance(record, Path(temporary) / "provenance.json")

    def test_completed_provenance_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.json"
            digest = finalize_provenance(self.good_record(), path)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                finalize_provenance(self.good_record(), path)
            self.assertEqual(json.loads(path.read_text()), self.good_record())


if __name__ == "__main__":
    unittest.main()
