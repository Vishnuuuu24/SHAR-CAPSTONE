"""Tests for `core.feature_cache_verify` -- the recompute-and-compare check.

The point of these tests is that verification must FAIL on a cache that is
internally consistent but numerically or provenance-wrong; a verifier that only
ever passes is worse than none, because it manufactures confidence.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import FeatureCacheMismatch, build_feature_cache
from core.feature_cache_verify import _ordered_int, verify_feature_cache
from test_feature_cache import (  # reuse the fixtures, per docs/PROJECT_RULES.md's rule of two
    _COMMON_KWARGS,
    _FakeFrameDataset,
    _FakeSpatialBackbone,
)


class OrderedIntTests(unittest.TestCase):
    """The ULP metric itself, which was wrong once and cost a pod run."""

    def test_adjacent_float16_values_are_exactly_one_ulp_apart(self) -> None:
        for value in (0.0, 0.5, 1.0, 4.0, 64.0, -1.0, -8.0):
            with self.subTest(value=value):
                a = np.float16(value)
                b = np.nextafter(a, np.float16(np.inf))
                distance = abs(
                    _ordered_int(np.array([a])) - _ordered_int(np.array([b]))
                )
                self.assertEqual(int(distance[0]), 1)

    def test_identical_values_are_zero_ulps_apart(self) -> None:
        values = np.array([0.0, -0.0, 1.0, -3.5], dtype=np.float16)
        distance = np.abs(_ordered_int(values.copy()) - _ordered_int(values.copy()))
        self.assertTrue(np.all(distance == 0))

    def test_near_zero_does_not_explode(self) -> None:
        """The old `delta / np.spacing(value)` ratio reported 1369 ULPs here."""

        a = np.array([0.0], dtype=np.float16)
        b = np.nextafter(a, np.float16(np.inf))
        self.assertEqual(int(np.abs(_ordered_int(a.copy()) - _ordered_int(b.copy()))[0]), 1)

    def test_distance_grows_with_separation(self) -> None:
        a = np.float16(1.0)
        b = np.nextafter(np.nextafter(a, np.float16(np.inf)), np.float16(np.inf))
        distance = np.abs(_ordered_int(np.array([a])) - _ordered_int(np.array([b])))
        self.assertEqual(int(distance[0]), 2)


class VerifyFeatureCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_dir = Path(self._tmp.name) / "cache"
        self.dataset = _FakeFrameDataset(12)
        self.backbone = _FakeSpatialBackbone()
        build_feature_cache(
            self.dataset, self.backbone, output_dir=self.cache_dir,
            device="cpu", **_COMMON_KWARGS,
        )

    def test_a_freshly_built_cache_verifies_bit_identically(self) -> None:
        report = verify_feature_cache(
            self.dataset, self.backbone, self.cache_dir,
            device="cpu", sample_size=12, **_COMMON_KWARGS,
        )
        self.assertEqual(report["rows_sampled"], 12)
        self.assertEqual(report["bit_identical_rows"], 12)
        self.assertEqual(report["max_abs_delta"], 0.0)

    def test_sample_size_is_capped_at_the_row_count(self) -> None:
        report = verify_feature_cache(
            self.dataset, self.backbone, self.cache_dir,
            device="cpu", sample_size=10_000, **_COMMON_KWARGS,
        )
        self.assertEqual(report["rows_sampled"], 12)

    def test_sampling_is_deterministic_for_a_given_seed(self) -> None:
        kwargs = dict(device="cpu", sample_size=4, **_COMMON_KWARGS)
        a = verify_feature_cache(self.dataset, self.backbone, self.cache_dir, seed=7, **kwargs)
        b = verify_feature_cache(self.dataset, self.backbone, self.cache_dir, seed=7, **kwargs)
        self.assertEqual(a["bit_identical_rows"], b["bit_identical_rows"])
        self.assertEqual(a["cache_sha256"], b["cache_sha256"])

    def test_verification_fails_when_the_features_are_wrong(self) -> None:
        """The case a manifest check cannot catch: self-consistent, numerically false."""

        features_path = next(self.cache_dir.glob("features.*.npy"))
        features = np.load(features_path)
        features[3] += np.float16(1.0)
        np.save(features_path, features)
        # Rebuild the manifest digest so the cache is INTERNALLY consistent --
        # otherwise load_feature_cache's own hash check would catch it first and
        # this test would not exercise the recompute path at all.
        from core.feature_cache import _MANIFEST_FILENAME  # noqa: PLC0415
        import json
        from core.p1c_readiness import sha256_file  # noqa: PLC0415

        manifest_path = self.cache_dir / _MANIFEST_FILENAME
        payload = json.loads(manifest_path.read_text())
        payload["cache_sha256"] = sha256_file(features_path)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        with self.assertRaises(FeatureCacheMismatch) as caught:
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=12, **_COMMON_KWARGS,
            )
        self.assertIn("does not reproduce the live backbone", str(caught.exception))

    def test_verification_fails_against_a_different_backbone(self) -> None:
        other = _FakeSpatialBackbone(offset=1.0)
        with self.assertRaises(FeatureCacheMismatch):
            verify_feature_cache(
                self.dataset, other, self.cache_dir,
                device="cpu", sample_size=12, **_COMMON_KWARGS,
            )

    def test_rejects_a_negative_max_ulps(self) -> None:
        with self.assertRaises(ValueError):
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=4, max_ulps=-1, **_COMMON_KWARGS,
            )

    def test_rejects_a_negative_ulp_floor_magnitude(self) -> None:
        with self.assertRaises(ValueError):
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=4, ulp_floor_magnitude=-1.0, **_COMMON_KWARGS,
            )

    def test_near_zero_jitter_is_tolerated_but_a_real_shift_is_not(self) -> None:
        """The exact case that failed on real data: tiny values, hundreds of ULPs apart."""

        import json
        from core.feature_cache import _MANIFEST_FILENAME
        from core.p1c_readiness import sha256_file

        features_path = next(self.cache_dir.glob("features.*.npy"))
        features = np.load(features_path)
        features[2][0] = np.float16(5.5e-5)  # ~900 float16 ULPs from zero, numerically nil
        np.save(features_path, features)
        manifest_path = self.cache_dir / _MANIFEST_FILENAME
        payload = json.loads(manifest_path.read_text())
        payload["cache_sha256"] = sha256_file(features_path)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        report = verify_feature_cache(
            self.dataset, self.backbone, self.cache_dir,
            device="cpu", sample_size=12, **_COMMON_KWARGS,
        )
        self.assertGreater(report["max_ulps"], 100)
        self.assertLess(report["max_abs_delta"], 1e-3)

    def test_rejects_a_nonpositive_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=4, batch_size=0, **_COMMON_KWARGS,
            )

    def test_a_wrong_feature_fails_even_at_a_generous_tolerance(self) -> None:
        """The tolerance bounds float16 jitter; it must not excuse a real error."""

        import json
        from core.feature_cache import _MANIFEST_FILENAME
        from core.p1c_readiness import sha256_file

        features_path = next(self.cache_dir.glob("features.*.npy"))
        features = np.load(features_path)
        features[5] += np.float16(0.5)
        np.save(features_path, features)
        manifest_path = self.cache_dir / _MANIFEST_FILENAME
        payload = json.loads(manifest_path.read_text())
        payload["cache_sha256"] = sha256_file(features_path)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        with self.assertRaises(FeatureCacheMismatch):
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=12, max_ulps=100, ulp_floor_magnitude=1.0, **_COMMON_KWARGS,
            )

    def test_exactly_max_ulps_passes_and_one_more_fails(self) -> None:
        """The gate is `max_ulps` steps at the floored magnitude -- an exact boundary check.

        `_FakeSpatialBackbone` always recomputes element 1 of every row as
        exactly 1.0 (channel index 1, offset 0), which sits exactly at the
        default `ulp_floor_magnitude`, so the allowed step there is known
        precisely: `numpy.spacing(float16(1.0))`.
        """

        import json
        from core.feature_cache import _MANIFEST_FILENAME
        from core.p1c_readiness import sha256_file

        max_ulps = 8
        floor_magnitude = 1.0
        step = np.spacing(np.float16(floor_magnitude))
        manifest_path = self.cache_dir / _MANIFEST_FILENAME

        def _set_row7_element1(value: np.float16) -> None:
            features_path = next(self.cache_dir.glob("features.*.npy"))
            features = np.load(features_path)
            features[7][1] = value
            np.save(features_path, features)
            payload = json.loads(manifest_path.read_text())
            payload["cache_sha256"] = sha256_file(features_path)
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        at_boundary = np.float16(1.0)
        for _ in range(max_ulps):
            at_boundary = np.nextafter(at_boundary, np.float16(np.inf))
        _set_row7_element1(at_boundary)

        report = verify_feature_cache(
            self.dataset, self.backbone, self.cache_dir,
            device="cpu", sample_size=12, max_ulps=max_ulps, ulp_floor_magnitude=floor_magnitude,
            **_COMMON_KWARGS,
        )
        self.assertEqual(report["rows_sampled"], 12)
        self.assertAlmostEqual(report["max_abs_delta"], max_ulps * step, delta=step * 1e-3)

        past_boundary = np.nextafter(at_boundary, np.float16(np.inf))
        _set_row7_element1(past_boundary)

        with self.assertRaises(FeatureCacheMismatch):
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=12, max_ulps=max_ulps, ulp_floor_magnitude=floor_magnitude,
                **_COMMON_KWARGS,
            )

    def test_max_significant_ulps_excludes_near_zero_elements(self) -> None:
        """Near-zero jitter must not pollute the diagnostic meant for real magnitudes."""

        import json
        from core.feature_cache import _MANIFEST_FILENAME
        from core.p1c_readiness import sha256_file

        features_path = next(self.cache_dir.glob("features.*.npy"))
        features = np.load(features_path)
        features[2][0] = np.float16(5.5e-5)  # hundreds of raw ULPs, well under the floor
        np.save(features_path, features)
        manifest_path = self.cache_dir / _MANIFEST_FILENAME
        payload = json.loads(manifest_path.read_text())
        payload["cache_sha256"] = sha256_file(features_path)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        report = verify_feature_cache(
            self.dataset, self.backbone, self.cache_dir,
            device="cpu", sample_size=12, **_COMMON_KWARGS,
        )
        self.assertGreater(report["max_ulps"], 100)
        self.assertEqual(report["max_significant_ulps"], 0)

    def test_provenance_mismatch_fails_regardless_of_feature_tolerance(self) -> None:
        """Provenance keeps zero tolerance no matter how generous the ULP gate is.

        `rows.csv` is not covered by the manifest's `cache_sha256` (only the
        features array is, per `core.feature_cache.load_feature_cache`), so
        editing it directly exercises the recompute-time provenance check
        rather than the manifest's own hash guard.
        """

        import csv

        rows_path = self.cache_dir / "rows.csv"
        with rows_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[4]["source_video_id"] = "not-the-real-video"
        with rows_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaises(FeatureCacheMismatch):
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=12, max_ulps=1_000_000, ulp_floor_magnitude=1.0,
                **_COMMON_KWARGS,
            )

    def test_batched_replay_matches_single_row_replay_on_cpu(self) -> None:
        """On a deterministic CPU backbone both batch shapes must agree exactly."""

        for batch_size in (1, 5, 12):
            with self.subTest(batch_size=batch_size):
                report = verify_feature_cache(
                    self.dataset, self.backbone, self.cache_dir,
                    device="cpu", sample_size=12, batch_size=batch_size, **_COMMON_KWARGS,
                )
                self.assertEqual(report["bit_identical_rows"], 12)

    def test_rejects_a_nonpositive_sample_size(self) -> None:
        with self.assertRaises(ValueError):
            verify_feature_cache(
                self.dataset, self.backbone, self.cache_dir,
                device="cpu", sample_size=0, **_COMMON_KWARGS,
            )


if __name__ == "__main__":
    unittest.main()
