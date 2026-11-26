"""Proves the P1C runners actually read ``data.packed_frames`` (D-33) and
``data.video_frames`` (D-34/D-35).

``scripts/calibrate_p1c_frame_loader.py`` and ``scripts/run_p1c_frame_baseline.py``
each build their Train/validation datasets from a ``fixed_protocol`` mapping.
Before this wiring existed, both scripts unconditionally built
``data.p1c_loader.P1CFrameDataset`` (the per-file PNG path) regardless of a
``data.packed_frames.pack_closure_path`` registration, and only the
calibration script knew about ``data.video_frames`` at all -- the baseline
runner, the one that actually trains, could not read the D-34/D-35 blob.
These tests exercise the real dataset-construction helpers --
``calibrate_p1c_frame_loader._datasets`` and
``run_p1c_frame_baseline._make_datasets`` -- against small synthetic
fixtures, proving: the packed and video branches are each selected only when
their field is present; the per-file branch is unchanged when neither is
present; per-file digest verification never runs under either blob path;
registering both pixel sources at once is refused; the source-video leakage
check still fires under all three sources; and each blob's own SHA-256
reaches the run manifest under its own (never a mislabeled) field name.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import cv2
import numpy as np
from PIL import Image

import scripts.calibrate_p1c_frame_loader as calibrate_mod
import scripts.run_p1c_frame_baseline as baseline_mod
from core.run_lifecycle import RunLifecycle
from data.contracts import MANIFEST_FIELDS
from data.p1c_loader import P1CFrameDataset, P1CPackedFrameDataset
from data.video_dataset import P1CVideoFrameDataset
from scripts.extract_p1c_video_frames import extract_videos
from scripts.pack_p1c_frames import pack_frames


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frame_64(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


class _RunnerWiringFixture(unittest.TestCase):
    """A synthetic REPO_ROOT, manifests, weights, and protocol both runners accept."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary.name)
        self.dataset_root = self.repo_root / "fixture-dataset"
        (self.dataset_root / "frames").mkdir(parents=True)
        self.train_manifest = self.repo_root / "train_manifest.csv"
        self.validation_manifest = self.repo_root / "validation_manifest.csv"
        self._write_split_manifest(
            self.train_manifest,
            "train",
            [("train-a.png", "Abuse", "abuse-1", 0, 11), ("train-b.png", "Normal", "normal-1", 1, 22)],
        )
        self._write_split_manifest(
            self.validation_manifest,
            "validation",
            [("val-a.png", "Normal", "normal-2", 0, 33)],
        )
        self.pack_dir = self.repo_root / "results/p1c/packed"
        weights_dir = self.repo_root / ".cache/weights"
        weights_dir.mkdir(parents=True)
        self.weights_path = weights_dir / "dummy.pt"
        self.weights_path.write_bytes(b"not-a-real-checkpoint; only its sha256 is ever checked here")
        self.protocol = {
            "data": {
                "dataset_roots": {"fixture": "fixture-dataset"},
                "train": {"path": "train_manifest.csv", "sha256": _digest(self.train_manifest)},
                "validation": {"path": "validation_manifest.csv", "sha256": _digest(self.validation_manifest)},
            },
            "backbone": {
                "weights": {
                    "path": ".cache/weights/dummy.pt",
                    "sha256": _digest(self.weights_path),
                    "enum": "IMAGENET1K_V2",
                }
            },
            "preprocessing": {
                "resize_height": 64,
                "resize_width": 64,
                "interpolation": "bilinear",
                "color_mode": "RGB",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {"class_names": ["Abuse", "Normal"]},
            "training": {"loader": {"verify_digests": True}},
            "augmentation": {"name": "none", "rationale": "wiring fixture: pixel-source plumbing, not augmentation"},
        }
        for target in (calibrate_mod, baseline_mod):
            patcher = mock.patch.object(target, "REPO_ROOT", self.repo_root)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_split_manifest(
        self, manifest_path: Path, split: str, rows: list[tuple[str, str, str, int, int]]
    ) -> None:
        written = []
        for name, label, video, frame_index, seed in rows:
            path = self.dataset_root / "frames" / name
            _write_frame_64(path, seed)
            written.append(
                {
                    "filepath": f"frames/{name}",
                    "source_dataset": "fixture",
                    "source_video_id": video,
                    "source_frame_index": frame_index,
                    "label": label,
                    "label_scope": "video_inherited",
                    "label_source": "fixture-folder",
                    "annotation_version": "fixture-v1",
                    "inside_official_interval": "",
                    "split": split,
                    "file_digest": _digest(path),
                }
            )
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(written)

    def _pack(self) -> dict:
        return pack_frames(
            train_manifest=self.train_manifest,
            validation_manifest=self.validation_manifest,
            dataset_root=self.dataset_root,
            dataset_key="fixture",
            output_dir=self.pack_dir,
        )

    def _packed_protocol(self) -> dict:
        protocol = copy.deepcopy(self.protocol)
        protocol["data"]["packed_frames"] = {"pack_closure_path": str(self.pack_dir / "pack_closure.json")}
        return protocol

    def _write_video(self, path: Path, color: tuple[int, int, int]) -> None:
        """Write one real, tiny, single-frame MP4 for `extract_videos` to decode."""

        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (8, 8))
        if not writer.isOpened():  # pragma: no cover - environment without an mp4v codec
            raise unittest.SkipTest("cv2.VideoWriter (mp4v) is unavailable in this environment")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :] = color
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    def _extract(self) -> Path:
        """Extract a real, tiny D-34/D-35-shaped video blob; return its closure path."""

        videos_root = self.repo_root / "video-src"
        self._write_video(videos_root / "fixture" / "train-a.mp4", (255, 0, 0))
        self._write_video(videos_root / "fixture" / "train-b.mp4", (0, 255, 0))
        self._write_video(videos_root / "fixture" / "val-a.mp4", (0, 0, 255))
        blob_dir = self.repo_root / "video-blob"
        extract_videos(
            source_root=videos_root, directories=None, output_dir=blob_dir,
            stride=1, expected_height=8, expected_width=8,
        )
        return blob_dir / "extract_closure.json"

    def _write_video_manifest(
        self, manifest_path: Path, split: str, rows: list[tuple[str, str]]
    ) -> None:
        """Write a D-34/D-35-shaped manifest: one row per kept frame, addressed
        by ``(source_video_id, source_frame_index)`` rather than by filepath."""

        written = []
        for stem, label in rows:
            written.append(
                {
                    "filepath": f"fixture/{stem}.mp4",
                    "source_dataset": "fixture",
                    "source_video_id": stem,
                    "source_frame_index": 0,
                    "label": label,
                    "label_scope": "video_inherited",
                    "label_source": "fixture-video",
                    "annotation_version": "fixture-v1",
                    "inside_official_interval": "",
                    "split": split,
                    "file_digest": "0" * 64,
                }
            )
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(written)

    def _video_protocol(self, extract_closure_path: Path, *, leak: bool = False) -> dict:
        """A protocol registering ``data.video_frames`` against a real, tiny blob.

        Uses its own Train/validation manifests (not ``self.train_manifest``):
        a video-sourced manifest is addressed by ``(source_video_id,
        source_frame_index)``, not the per-file ``filepath`` the D-33/per-file
        fixtures use, so the two pixel sources cannot share manifest content.
        """

        video_train_manifest = self.repo_root / "video_train_manifest.csv"
        video_validation_manifest = self.repo_root / "video_validation_manifest.csv"
        self._write_video_manifest(
            video_train_manifest, "train", [("train-a", "Abuse"), ("train-b", "Normal")]
        )
        # A leaking validation manifest reuses "train-a" as its source video.
        validation_video = "train-a" if leak else "val-a"
        self._write_video_manifest(video_validation_manifest, "validation", [(validation_video, "Normal")])
        protocol = copy.deepcopy(self.protocol)
        protocol["data"]["train"] = {"path": "video_train_manifest.csv", "sha256": _digest(video_train_manifest)}
        protocol["data"]["validation"] = {
            "path": "video_validation_manifest.csv", "sha256": _digest(video_validation_manifest)
        }
        protocol["data"]["video_frames"] = {"extract_closure_path": str(extract_closure_path)}
        return protocol


class CalibrateDatasetWiringTests(_RunnerWiringFixture):
    def test_per_file_branch_when_packed_frames_absent(self) -> None:
        train, validation, elapsed = calibrate_mod._datasets(self.protocol)
        self.assertIsInstance(train, P1CFrameDataset)
        self.assertNotIsInstance(train, P1CPackedFrameDataset)
        self.assertIsInstance(validation, P1CFrameDataset)
        self.assertNotIsInstance(validation, P1CPackedFrameDataset)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_packed_branch_selected_when_field_present(self) -> None:
        self._pack()
        train, validation, _ = calibrate_mod._datasets(self._packed_protocol())
        self.assertIsInstance(train, P1CPackedFrameDataset)
        self.assertIsInstance(validation, P1CPackedFrameDataset)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)

    def test_packed_branch_requires_verify_digests_true(self) -> None:
        self._pack()
        protocol = self._packed_protocol()
        protocol["training"]["loader"]["verify_digests"] = False
        with self.assertRaisesRegex(ValueError, "verify_digests=true"):
            calibrate_mod._datasets(protocol)

    def test_packed_branch_skips_per_file_digest_verification(self) -> None:
        self._pack()
        # Corrupt the source PNG *after* packing: the blob already holds the
        # original verified bytes, so a fresh packed-dataset build must not
        # notice, while a fresh per-file build (never yet indexed) must.
        (self.dataset_root / "frames" / "train-a.png").write_bytes(b"corrupted, not a real PNG")
        train, validation, _ = calibrate_mod._datasets(self._packed_protocol())
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            calibrate_mod._datasets(self.protocol)

    def test_leakage_check_still_fires_per_file_and_packed(self) -> None:
        # Make the validation row's source_video_id collide with a train row.
        self._write_split_manifest(
            self.validation_manifest, "validation", [("val-a.png", "Normal", "abuse-1", 0, 33)]
        )
        self.protocol["data"]["validation"]["sha256"] = _digest(self.validation_manifest)
        with self.assertRaisesRegex(ValueError, "source-video leakage"):
            calibrate_mod._datasets(self.protocol)
        self._pack()
        packed_protocol = self._packed_protocol()
        with self.assertRaisesRegex(ValueError, "source-video leakage"):
            calibrate_mod._datasets(packed_protocol)

    def test_packed_blob_sha256_matches_closure_and_is_none_when_absent(self) -> None:
        self.assertIsNone(calibrate_mod._packed_blob_sha256(self.protocol))
        closure = self._pack()
        self.assertEqual(calibrate_mod._packed_blob_sha256(self._packed_protocol()), closure["blob_sha256"])


class RunBaselineDatasetWiringTests(_RunnerWiringFixture):
    def test_per_file_branch_when_packed_frames_absent(self) -> None:
        train, validation = baseline_mod._make_datasets(self.protocol, seed=0)
        self.assertIsInstance(train, P1CFrameDataset)
        self.assertNotIsInstance(train, P1CPackedFrameDataset)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)

    def test_packed_branch_selected_when_field_present(self) -> None:
        self._pack()
        train, validation = baseline_mod._make_datasets(self._packed_protocol(), seed=0)
        self.assertIsInstance(train, P1CPackedFrameDataset)
        self.assertIsInstance(validation, P1CPackedFrameDataset)

    def test_packed_branch_requires_verify_digests_true(self) -> None:
        self._pack()
        protocol = self._packed_protocol()
        protocol["training"]["loader"]["verify_digests"] = False
        with self.assertRaisesRegex(ValueError, "verify_digests=true"):
            baseline_mod._make_datasets(protocol, seed=0)

    def test_packed_branch_skips_per_file_digest_verification(self) -> None:
        self._pack()
        (self.dataset_root / "frames" / "train-a.png").write_bytes(b"corrupted, not a real PNG")
        train, validation = baseline_mod._make_datasets(self._packed_protocol(), seed=0)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            baseline_mod._make_datasets(self.protocol, seed=0)

    def test_packed_blob_sha256_matches_closure_and_is_none_when_absent(self) -> None:
        self.assertIsNone(baseline_mod._packed_blob_sha256(self.protocol))
        closure = self._pack()
        self.assertEqual(baseline_mod._packed_blob_sha256(self._packed_protocol()), closure["blob_sha256"])


class RunManifestPackedDigestTests(_RunnerWiringFixture):
    """Proves the blob SHA-256 actually reaches an immutable run manifest."""

    def _provenance(self, packed_frames_blob_sha256: str | None) -> dict:
        return {
            "run_id": "wiring-test-run",
            "run_kind": "p1c_real_loader_calibration",
            "config_digest": "a" * 64,
            "code_revision": "local working tree; commit is intentionally not assumed",
            "seed": 0,
            "package_versions": {"python": "3.12.0"},
            "dataset_manifest_digest": self.protocol["data"]["train"]["sha256"],
            "packed_frames_blob_sha256": packed_frames_blob_sha256,
            "annotation_version": "P1B frozen grouped Train/validation manifests; Test prohibited",
            "environment_digest": "b" * 64,
            "hardware": {"selected": "cpu"},
            "metric_artifact_paths": ["calibration.json"],
        }

    def test_manifest_records_packed_blob_sha256_when_registered(self) -> None:
        closure = self._pack()
        blob_sha256 = calibrate_mod._packed_blob_sha256(self._packed_protocol())
        self.assertEqual(blob_sha256, closure["blob_sha256"])
        lifecycle = RunLifecycle.create(
            self.repo_root / "results/p1c", "wiring-test-run", self._provenance(blob_sha256)
        )
        import json

        manifest = json.loads(lifecycle.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["packed_frames_blob_sha256"], closure["blob_sha256"])
        self.assertEqual(manifest["dataset_manifest_digest"], self.protocol["data"]["train"]["sha256"])

    def test_manifest_records_none_when_unpacked(self) -> None:
        lifecycle = RunLifecycle.create(
            self.repo_root / "results/p1c", "wiring-test-run", self._provenance(None)
        )
        import json

        manifest = json.loads(lifecycle.manifest_path.read_text(encoding="utf-8"))
        self.assertIsNone(manifest["packed_frames_blob_sha256"])


class PackedFramesFieldValidationTests(_RunnerWiringFixture):
    def test_malformed_packed_frames_field_is_rejected(self) -> None:
        for target in (calibrate_mod, baseline_mod):
            protocol = copy.deepcopy(self.protocol)
            protocol["data"]["packed_frames"] = {"pack_closure_path": ""}
            with self.assertRaisesRegex(ValueError, "pack_closure_path"):
                target._pack_closure_path(protocol["data"])
            protocol["data"]["packed_frames"] = {"pack_closure_path": "x", "extra": 1}
            with self.assertRaisesRegex(ValueError, "pack_closure_path"):
                target._pack_closure_path(protocol["data"])


class VideoFramesFieldValidationTests(_RunnerWiringFixture):
    def test_malformed_video_frames_field_is_rejected(self) -> None:
        for target in (calibrate_mod, baseline_mod):
            protocol = copy.deepcopy(self.protocol)
            protocol["data"]["video_frames"] = {"extract_closure_path": ""}
            with self.assertRaisesRegex(ValueError, "extract_closure_path"):
                target._extract_closure_path(protocol["data"])
            protocol["data"]["video_frames"] = {"extract_closure_path": "x", "extra": 1}
            with self.assertRaisesRegex(ValueError, "extract_closure_path"):
                target._extract_closure_path(protocol["data"])


class _VideoDatasetWiringTestsMixin:
    """Shared assertions run against both real runners' dataset builders.

    Subclasses set ``self._datasets`` (the target's ``_datasets`` or
    ``_make_datasets``) in ``setUp``.
    """

    _datasets: Any

    def test_video_branch_builds_both_datasets_when_field_present(self) -> None:
        closure_path = self._extract()
        result = self._datasets(self._video_protocol(closure_path))
        train, validation = result[0], result[1]
        self.assertIsInstance(train, P1CVideoFrameDataset)
        self.assertIsInstance(validation, P1CVideoFrameDataset)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)

    def test_registering_both_packed_and_video_frames_is_refused(self) -> None:
        closure_path = self._extract()
        self._pack()
        protocol = self._video_protocol(closure_path)
        protocol["data"]["packed_frames"] = {"pack_closure_path": str(self.pack_dir / "pack_closure.json")}
        with self.assertRaisesRegex(ValueError, "different pixel sources"):
            self._datasets(protocol)

    def test_video_branch_requires_verify_digests_true(self) -> None:
        closure_path = self._extract()
        protocol = self._video_protocol(closure_path)
        protocol["training"]["loader"]["verify_digests"] = False
        with self.assertRaisesRegex(ValueError, "data.video_frames requires training.loader.verify_digests=true"):
            self._datasets(protocol)

    def test_leakage_check_fires_for_video_source(self) -> None:
        closure_path = self._extract()
        with self.assertRaisesRegex(ValueError, "source-video leakage"):
            self._datasets(self._video_protocol(closure_path, leak=True))


class CalibrateVideoDatasetWiringTests(_VideoDatasetWiringTestsMixin, _RunnerWiringFixture):
    def setUp(self) -> None:
        super().setUp()
        self._datasets = lambda protocol: calibrate_mod._datasets(protocol)


class RunBaselineVideoDatasetWiringTests(_VideoDatasetWiringTestsMixin, _RunnerWiringFixture):
    def setUp(self) -> None:
        super().setUp()
        self._datasets = lambda protocol: baseline_mod._make_datasets(protocol, seed=0)


class RunManifestVideoDigestTests(_RunnerWiringFixture):
    """Proves the video blob's SHA-256 reaches the run manifest under its own
    field name, never mislabeled as ``packed_frames_blob_sha256`` (D-33)."""

    def test_video_blob_sha256_matches_closure_and_is_none_when_absent(self) -> None:
        self.assertIsNone(calibrate_mod._video_blob_sha256(self.protocol))
        self.assertIsNone(baseline_mod._video_blob_sha256(self.protocol))
        closure_path = self._extract()
        import json

        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        protocol = self._video_protocol(closure_path)
        self.assertEqual(calibrate_mod._video_blob_sha256(protocol), closure["blob_sha256"])
        self.assertEqual(baseline_mod._video_blob_sha256(protocol), closure["blob_sha256"])
        # And it is never accidentally recorded as a packed-blob digest.
        self.assertIsNone(calibrate_mod._packed_blob_sha256(protocol))
        self.assertIsNone(baseline_mod._packed_blob_sha256(protocol))


if __name__ == "__main__":
    unittest.main()
