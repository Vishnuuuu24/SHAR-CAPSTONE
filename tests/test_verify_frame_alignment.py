from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from data.frame_alignment import read_source_video_frame, resize_to_match
from data.ucf_intervals import CANONICAL_UCF_LABELS
from scripts.verify_frame_alignment import load_config, main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _video_id(label: str) -> str:
    return "Normal_Videos_001_x264" if label == "Normal" else f"{label}001_x264"


def _folder(label: str) -> str:
    return "NormalVideos" if label == "Normal" else label


def _write_synthetic_video(path: Path, *, frame_count: int, height: int, width: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("test fixture could not open a synthetic VideoWriter")
    salt = int(hashlib.sha256(path.stem.encode("utf-8")).hexdigest()[:8], 16)
    for index in range(frame_count):
        rng = np.random.default_rng(index * 97 + salt)
        writer.write(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8))
    writer.release()


def _write_matching_kaggle_png(png_path: Path, video_path: Path, frame_index: int) -> None:
    decoded = read_source_video_frame(video_path, frame_index)
    resized = resize_to_match(decoded, (64, 64), policy="stretch_to_target", interpolation="area")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), cv2.cvtColor((resized * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))


class VerifyFrameAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".test-frame-alignment-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.frame_root = self.root / "frames"
        self.video_root = self.root / "videos"
        self.results_root = self.root / "results"
        for label in sorted(CANONICAL_UCF_LABELS):
            video_path = self.video_root / "Videos" / _folder(label) / f"{_video_id(label)}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            _write_synthetic_video(video_path, frame_count=20, height=48, width=64)
            for frame_index in (2, 10, 18):
                _write_matching_kaggle_png(
                    self.frame_root / "Test" / _folder(label) / f"{_video_id(label)}_{frame_index}.png",
                    video_path,
                    frame_index,
                )
        self.config_path = self.root / "config.yaml"
        self._write_config()

    def _write_config(self, **overrides: object) -> None:
        config = {
            "schema_version": "1.0.0",
            "status": "VERIFICATION_TOOLING_NO_MANIFEST_OR_RESEARCH",
            "frame_root": self.frame_root.relative_to(REPO_ROOT).as_posix(),
            "video_root": self.video_root.relative_to(REPO_ROOT).as_posix(),
            "results_root": self.results_root.relative_to(REPO_ROOT).as_posix(),
            "split": "Test",
            "required_labels": sorted(CANONICAL_UCF_LABELS),
            "sample": {"videos_per_label": 1, "frames_per_video": 3, "global_seed": 1},
            "interpolation": "area",
            "ssim_pass_threshold": 0.6,
            "index_hypotheses": {"offsets": [-1, 1], "scale_factors": [0.1, 10.0], "minimum_declared_margin": 0.05},
        }
        config.update(overrides)
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def _run(self, run_id: str) -> int:
        with patch("sys.argv", ["verify_frame_alignment.py", "--config", str(self.config_path), "--run-id", run_id]):
            return main()

    def _evidence(self, run_id: str) -> dict:
        return json.loads((self.results_root / run_id / "evidence.json").read_text(encoding="utf-8"))

    def test_end_to_end_run_passes_and_closes_an_immutable_evidence_run(self) -> None:
        self.assertEqual(self._run("fixture-pass"), 0)
        evidence = self._evidence("fixture-pass")
        run_directory = self.results_root / "fixture-pass"
        self.assertTrue(evidence["all_selected_comparisons_pass"])
        self.assertEqual(evidence["resolved_comparison_count"], 42)
        self.assertEqual(evidence["unresolved_count"], 0)
        self.assertEqual(evidence["discovery_quarantine_count"], 0)
        self.assertTrue((run_directory / ".complete").is_file())
        self.assertEqual(len(evidence["sampled_input_manifest_sha256"]), 64)
        with self.assertRaises(FileExistsError):
            self._run("fixture-pass")

    def test_missing_source_video_invalidates_and_closes_the_run(self) -> None:
        (self.video_root / "Videos" / "Abuse" / "Abuse001_x264.mp4").unlink()
        with self.assertRaisesRegex(RuntimeError, "evidence gate failed"):
            self._run("fixture-unresolved")
        evidence = self._evidence("fixture-unresolved")
        self.assertGreaterEqual(evidence["unresolved_count"], 1)
        self.assertFalse(evidence["all_selected_comparisons_pass"])
        self.assertTrue((self.results_root / "fixture-unresolved" / ".complete").is_file())

    def test_misaligned_frame_index_invalidates_the_run(self) -> None:
        video_path = self.video_root / "Videos" / "Abuse" / "Abuse001_x264.mp4"
        _write_matching_kaggle_png(self.frame_root / "Test" / "Abuse" / "Abuse001_x264_2.png", video_path, 15)
        with self.assertRaisesRegex(RuntimeError, "evidence gate failed"):
            self._run("fixture-misaligned")
        evidence = self._evidence("fixture-misaligned")
        self.assertFalse(evidence["all_selected_comparisons_pass"])
        self.assertGreaterEqual(evidence["failing_count"], 1)

    def test_invalid_configuration_is_rejected_before_any_scan(self) -> None:
        self._write_config(ssim_pass_threshold=-1.0)
        with self.assertRaisesRegex(ValueError, "ssim_pass_threshold"):
            load_config(self.config_path)
        self._write_config(sample={"videos_per_label": 1, "frames_per_video": 99, "global_seed": 1})
        with self.assertRaisesRegex(ValueError, "frames_per_video"):
            load_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
