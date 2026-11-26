"""Tests for `data.pixel_source`, the shared field-parsing module.

Before this module existed, `scripts/calibrate_p1c_frame_loader.py` and
`scripts/run_p1c_frame_baseline.py` each carried their own copy of
`data.packed_frames` parsing (and only the calibration script knew about
`data.video_frames` at all), so a fix or a stricter check applied to one
runner silently did not apply to the other. These tests exercise the field
parsing, the both-registered refusal, and the `verify_digests` gate directly
against the shared module, independent of either runner, so a regression here
is caught once instead of twice (or missed entirely in the runner that was
not updated).

Dataset construction and end-to-end blob-SHA256 provenance are covered by
`tests/test_p1c_frame_runner_wiring.py`, which exercises this module through
both real runners.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from data import pixel_source
from data.contracts import MANIFEST_FIELDS
from data.video_manifests import SOURCE_DATASET


class PackedFramesFieldTests(unittest.TestCase):
    def test_absent_field_returns_none(self) -> None:
        self.assertIsNone(pixel_source.packed_frames_field({}))
        self.assertIsNone(pixel_source.pack_closure_path({}))

    def test_valid_field_round_trips_to_a_path(self) -> None:
        data = {"packed_frames": {"pack_closure_path": "results/p1c/packed/pack_closure.json"}}
        self.assertEqual(pixel_source.packed_frames_field(data), data["packed_frames"])
        self.assertEqual(
            pixel_source.pack_closure_path(data), Path("results/p1c/packed/pack_closure.json")
        )

    def test_empty_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pack_closure_path"):
            pixel_source.packed_frames_field({"packed_frames": {"pack_closure_path": ""}})

    def test_extra_key_is_rejected(self) -> None:
        # A config must say exactly what it means; a stray key next to the
        # real field is a typo, not a signal to tolerate the mapping anyway.
        with self.assertRaisesRegex(ValueError, "pack_closure_path"):
            pixel_source.packed_frames_field(
                {"packed_frames": {"pack_closure_path": "x", "extra": 1}}
            )

    def test_wrong_key_name_is_rejected_rather_than_silently_ignored(self) -> None:
        # A typo'd field name (e.g. "closure_path" instead of
        # "pack_closure_path") must fail loudly, not fall back to per-file
        # loading as though nothing had been registered.
        with self.assertRaisesRegex(ValueError, "pack_closure_path"):
            pixel_source.packed_frames_field({"packed_frames": {"closure_path": "x"}})

    def test_non_mapping_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pack_closure_path"):
            pixel_source.packed_frames_field({"packed_frames": "results/p1c/packed/pack_closure.json"})


class VideoFramesFieldTests(unittest.TestCase):
    def test_absent_field_returns_none(self) -> None:
        self.assertIsNone(pixel_source.video_frames_field({}))
        self.assertIsNone(pixel_source.extract_closure_path({}))

    def test_valid_field_round_trips_to_a_path(self) -> None:
        data = {"video_frames": {"extract_closure_path": "results/p1c/p1c-video-frames/extract_closure.json"}}
        self.assertEqual(pixel_source.video_frames_field(data), data["video_frames"])
        self.assertIsNotNone(pixel_source.extract_closure_path(data))

    def test_empty_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "extract_closure_path"):
            pixel_source.video_frames_field({"video_frames": {"extract_closure_path": ""}})

    def test_extra_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "extract_closure_path"):
            pixel_source.video_frames_field(
                {"video_frames": {"extract_closure_path": "x", "extra": 1}}
            )

    def test_wrong_key_name_is_rejected_rather_than_silently_ignored(self) -> None:
        with self.assertRaisesRegex(ValueError, "extract_closure_path"):
            pixel_source.video_frames_field({"video_frames": {"closure_path": "x"}})


class RefuseBothRegisteredTests(unittest.TestCase):
    def test_neither_registered_is_fine(self) -> None:
        pixel_source.refuse_if_both_registered({})

    def test_only_packed_is_fine(self) -> None:
        pixel_source.refuse_if_both_registered(
            {"packed_frames": {"pack_closure_path": "x"}}
        )

    def test_only_video_is_fine(self) -> None:
        pixel_source.refuse_if_both_registered(
            {"video_frames": {"extract_closure_path": "x"}}
        )

    def test_both_registered_is_refused(self) -> None:
        # These are two different pixel sources (D-33 Kaggle-sourced versus
        # D-34/D-35 official-video-sourced); a config must say which one it
        # means rather than leaving the runner to guess or silently prefer
        # one.
        with self.assertRaisesRegex(ValueError, "different pixel sources"):
            pixel_source.refuse_if_both_registered(
                {
                    "packed_frames": {"pack_closure_path": "x"},
                    "video_frames": {"extract_closure_path": "y"},
                }
            )


class VerifyDigestsGateTests(unittest.TestCase):
    def _protocol(self, verify_digests: object) -> dict:
        return {"training": {"loader": {"verify_digests": verify_digests}}}

    def test_true_is_accepted_for_both_fields(self) -> None:
        pixel_source.require_verify_digests_true(self._protocol(True), field="packed_frames")
        pixel_source.require_verify_digests_true(self._protocol(True), field="video_frames")

    def test_false_is_refused_for_packed_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "data.packed_frames requires training.loader.verify_digests=true"):
            pixel_source.require_verify_digests_true(self._protocol(False), field="packed_frames")

    def test_false_is_refused_for_video_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "data.video_frames requires training.loader.verify_digests=true"):
            pixel_source.require_verify_digests_true(self._protocol(False), field="video_frames")

    def test_a_falsy_non_bool_value_is_also_refused(self) -> None:
        # `is not True` is deliberate: a config that wrote a truthy-but-not-
        # literally-true value (e.g. the string "true") must not sneak past
        # this gate either.
        with self.assertRaisesRegex(ValueError, "verify_digests=true"):
            pixel_source.require_verify_digests_true(self._protocol("true"), field="packed_frames")


class _FakeGroupedDataset:
    def __init__(self, groups: frozenset[tuple[str, str]]) -> None:
        self._groups = groups

    def source_video_groups(self) -> frozenset[tuple[str, str]]:
        return self._groups


class SourceVideoLeakageTests(unittest.TestCase):
    def test_disjoint_groups_pass(self) -> None:
        train = _FakeGroupedDataset(frozenset({("fixture", "a")}))
        validation = _FakeGroupedDataset(frozenset({("fixture", "b")}))
        pixel_source.assert_no_source_video_leakage(train, validation, message="unused")

    def test_overlapping_groups_are_refused_with_the_given_message(self) -> None:
        train = _FakeGroupedDataset(frozenset({("fixture", "a")}))
        validation = _FakeGroupedDataset(frozenset({("fixture", "a")}))
        with self.assertRaisesRegex(ValueError, "custom leakage message"):
            pixel_source.assert_no_source_video_leakage(train, validation, message="custom leakage message")


def _write_manifest_row(path: Path, *, source_dataset: str) -> None:
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "filepath": "fixture/a.mp4",
            "source_dataset": source_dataset,
            "source_video_id": "a",
            "label": "Normal",
            "label_scope": "video_inherited",
            "label_source": "fixture",
            "annotation_version": "fixture-v1",
            "split": "train",
            "file_digest": "0" * 64,
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)


class RefuseBlobManifestLacksPixelSourceTests(unittest.TestCase):
    """K-21: a blob-derived manifest with no registered pixel source used to
    fall through to per-file loading and re-hash the source MP4 once per
    manifest row (391,014 rows against a 97.7 GiB corpus, 7h25m of GPU-slice
    time). These prove the refusal fires exactly when it should, and only
    then."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.manifest_path = Path(self.temporary.name) / "manifest.csv"

    def test_blob_derived_with_no_pixel_source_is_refused(self) -> None:
        _write_manifest_row(self.manifest_path, source_dataset=SOURCE_DATASET)
        with self.assertRaisesRegex(ValueError, "data.video_frames.extract_closure_path"):
            pixel_source.refuse_if_blob_manifest_lacks_pixel_source(self.manifest_path, {}, label="train manifest")

    def test_blob_derived_with_video_frames_registered_passes(self) -> None:
        _write_manifest_row(self.manifest_path, source_dataset=SOURCE_DATASET)
        data = {"video_frames": {"extract_closure_path": "x"}}
        pixel_source.refuse_if_blob_manifest_lacks_pixel_source(self.manifest_path, data, label="train manifest")

    def test_blob_derived_with_packed_frames_registered_passes(self) -> None:
        _write_manifest_row(self.manifest_path, source_dataset=SOURCE_DATASET)
        data = {"packed_frames": {"pack_closure_path": "x"}}
        pixel_source.refuse_if_blob_manifest_lacks_pixel_source(self.manifest_path, data, label="train manifest")

    def test_non_blob_derived_with_no_pixel_source_passes(self) -> None:
        # A genuinely per-file manifest (e.g. the fixture/D-33 source_dataset
        # values) with no pixel source registered is the original, still-legal
        # per-file path -- this check must not touch it.
        _write_manifest_row(self.manifest_path, source_dataset="fixture")
        pixel_source.refuse_if_blob_manifest_lacks_pixel_source(self.manifest_path, {}, label="train manifest")

    def test_missing_manifest_is_a_clear_error(self) -> None:
        missing = Path(self.temporary.name) / "does_not_exist.csv"
        with self.assertRaisesRegex(ValueError, "manifest file not found"):
            pixel_source.refuse_if_blob_manifest_lacks_pixel_source(missing, {}, label="train manifest")

    def test_empty_manifest_is_a_clear_error(self) -> None:
        self.manifest_path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest is empty"):
            pixel_source.refuse_if_blob_manifest_lacks_pixel_source(self.manifest_path, {}, label="train manifest")

    def test_manifest_without_source_dataset_column_is_a_clear_error(self) -> None:
        self.manifest_path.write_text("filepath,label\na.mp4,Normal\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no source_dataset column"):
            pixel_source.refuse_if_blob_manifest_lacks_pixel_source(self.manifest_path, {}, label="train manifest")

    def test_manifest_with_header_but_no_rows_is_a_clear_error(self) -> None:
        self.manifest_path.write_text(",".join(MANIFEST_FIELDS) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no data rows"):
            pixel_source.refuse_if_blob_manifest_lacks_pixel_source(self.manifest_path, {}, label="train manifest")


class LoaderModeTests(unittest.TestCase):
    def test_video_blob_wins_when_both_present(self) -> None:
        # Cannot actually happen once refuse_if_both_registered has run, but
        # the precedence should still match: video before packed before
        # per-file, exactly as every log line and report field computes it.
        self.assertEqual(
            pixel_source.loader_mode(packed_blob_sha256="p", video_blob_sha256="v"), "video_blob"
        )

    def test_packed_blob_when_only_packed_present(self) -> None:
        self.assertEqual(
            pixel_source.loader_mode(packed_blob_sha256="p", video_blob_sha256=None), "packed_blob"
        )

    def test_per_file_when_neither_present(self) -> None:
        self.assertEqual(
            pixel_source.loader_mode(packed_blob_sha256=None, video_blob_sha256=None), "per_file"
        )


if __name__ == "__main__":
    unittest.main()
