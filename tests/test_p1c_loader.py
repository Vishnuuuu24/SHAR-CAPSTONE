from __future__ import annotations

import csv
import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.contracts import MANIFEST_FIELDS
from data.p1c_loader import (
    FrameLoaderOptions,
    FramePreprocessing,
    P1CFrameDataset,
    P1CPackedFrameDataset,
    build_p1c_frame_dataloader,
    seed_dataloader_worker,
)
from scripts.pack_p1c_frames import pack_frames


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frame_64(path: Path, seed: int) -> None:
    """Write a deterministic, non-uniform 64x64x3 fixture PNG.

    Pack/unpack correctness bugs (byte order, channel swap, transposition,
    off-by-one offsets) are invisible on a single flat colour; a per-pixel
    pseudo-random pattern makes them visible.
    """

    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


class P1CFrameLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "fixture-dataset"
        (self.dataset_root / "frames").mkdir(parents=True)
        self._write_image("train-red.png", (255, 0, 0))
        self._write_image("train-green.png", (0, 255, 0))
        self._write_image("validation-blue.png", (0, 0, 255))
        self._write_image("test-gray.png", (128, 128, 128))
        self.manifest = self.root / "manifest.csv"
        self._write_manifest()
        self.preprocessing = FramePreprocessing.from_config(
            {
                "resize_height": 3,
                "resize_width": 4,
                "interpolation": "nearest",
                "color_mode": "RGB",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_image(self, name: str, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (2, 2), color).save(self.dataset_root / "frames" / name)

    def _write_manifest(self) -> None:
        rows = []
        for name, label, video, split in (
            ("train-red.png", "Abuse", "abuse-1", "train"),
            ("train-green.png", "Normal", "normal-1", "train"),
            ("validation-blue.png", "Abuse", "abuse-2", "validation"),
            ("test-gray.png", "Normal", "normal-2", "test"),
        ):
            path = self.dataset_root / "frames" / name
            rows.append(
                {
                    "filepath": f"frames/{name}",
                    "source_dataset": "fixture",
                    "source_video_id": video,
                    "source_frame_index": 0,
                    "label": label,
                    "label_scope": "video_inherited",
                    "label_source": "fixture-folder",
                    "annotation_version": "fixture-v1",
                    "inside_official_interval": "",
                    "split": split,
                    "file_digest": _digest(path),
                }
            )
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _dataset(self, split: str) -> P1CFrameDataset:
        return P1CFrameDataset(
            self.manifest,
            {"fixture": self.dataset_root},
            split=split,  # type: ignore[arg-type]
            class_names=("Abuse", "Normal"),
            preprocessing=self.preprocessing,
            verify_digests=True,
        )

    def test_dataset_uses_selected_manifest_split_and_explicit_normalization(self) -> None:
        dataset = self._dataset("train")
        self.assertEqual(len(dataset), 2)
        self.assertEqual([row.split for row in dataset.rows], ["train", "train"])
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (3, 3, 4))
        self.assertEqual(sample["target"], 0)
        self.assertEqual(sample["source_video_id"], "abuse-1")
        self.assertTrue(torch.equal(sample["image"][0], torch.ones(3, 4)))
        self.assertTrue(torch.equal(sample["image"][1], torch.zeros(3, 4)))
        self.assertTrue(torch.equal(sample["image"][2], torch.zeros(3, 4)))

    def test_dataset_refuses_test_split_unknown_labels_and_implicit_preprocessing(self) -> None:
        with self.assertRaisesRegex(PermissionError, "allow_held_out_test=True"):
            self._dataset("test")
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            FramePreprocessing.from_config(
                {
                    "resize_height": 3,
                    "resize_width": 4,
                    "interpolation": "nearest",
                    "color_mode": "RGB",
                    "mean": [0.0, 0.0, 0.0],
                }
            )
        with self.assertRaisesRegex(ValueError, "absent from class_names"):
            P1CFrameDataset(
                self.manifest,
                {"fixture": self.dataset_root},
                split="train",
                class_names=("Abuse",),
                preprocessing=self.preprocessing,
            )

    def test_loader_shuffle_and_worker_seeding_are_reproducible(self) -> None:
        dataset = self._dataset("train")
        options = FrameLoaderOptions(
            batch_size=1,
            shuffle=True,
            num_workers=0,
            seed=47,
            epoch=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

        def collect() -> list[str]:
            loader = build_p1c_frame_dataloader(dataset, options=options)
            return [name for batch in loader for name in batch["filepath"]]

        self.assertEqual(collect(), collect())
        next_epoch = FrameLoaderOptions(
            batch_size=1,
            shuffle=True,
            num_workers=0,
            seed=47,
            epoch=1,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )
        next_epoch_first = [name for batch in build_p1c_frame_dataloader(dataset, options=next_epoch) for name in batch["filepath"]]
        next_epoch_second = [name for batch in build_p1c_frame_dataloader(dataset, options=next_epoch) for name in batch["filepath"]]
        self.assertEqual(next_epoch_first, next_epoch_second)
        loader = build_p1c_frame_dataloader(dataset, options=options)
        self.assertIs(loader.worker_init_fn, seed_dataloader_worker)

        torch.manual_seed(913)
        seed_dataloader_worker(0)
        first = (random.random(), float(np.random.random()), float(torch.rand(())))
        torch.manual_seed(913)
        seed_dataloader_worker(0)
        second = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertEqual(first, second)

    def test_loader_options_reject_ambiguous_worker_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "persistent_workers"):
            FrameLoaderOptions(
                batch_size=1,
                shuffle=False,
                num_workers=0,
                seed=0,
                epoch=0,
                pin_memory=False,
                drop_last=False,
                persistent_workers=True,
            )

    def test_disk_backed_index_reuses_small_metadata_but_rechecks_file_digest(self) -> None:
        first = self._dataset("train")
        self.assertEqual(first.source_video_groups(), frozenset({("fixture", "abuse-1"), ("fixture", "normal-1")}))
        index_directory = self.manifest.parent / ".p1c_index"
        self.assertTrue(index_directory.is_dir())
        self._write_image("train-red.png", (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "cached P1C manifest digest mismatch"):
            self._dataset("train")

    def test_cached_index_skips_the_row_walk_when_no_integrity_gate_is_requested(self) -> None:
        # With both gates off, the walk resolved every row's path only to discard
        # it. Pointing at a root that does not exist proves the walk is skipped:
        # it would raise if it still ran, and the digest gate below proves that
        # turning verification back on is still authoritative.
        self._dataset("train")
        unverified = P1CFrameDataset(
            self.manifest,
            {"fixture": self.dataset_root / "absent"},
            split="train",
            class_names=("Abuse", "Normal"),
            preprocessing=self.preprocessing,
            verify_files=False,
            verify_digests=False,
        )
        self.assertEqual(len(unverified), 2)

    def test_missing_files_are_still_caught_when_the_file_gate_is_requested(self) -> None:
        self._dataset("train")
        with self.assertRaisesRegex(FileNotFoundError, "cached P1C manifest file is missing"):
            P1CFrameDataset(
                self.manifest,
                {"fixture": self.dataset_root / "absent"},
                split="train",
                class_names=("Abuse", "Normal"),
                preprocessing=self.preprocessing,
                verify_files=True,
                verify_digests=False,
            )


class P1CPackedFrameLoaderTests(unittest.TestCase):
    """Proves the packed-blob path is bit-identical to the per-file path."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "fixture-dataset"
        (self.dataset_root / "frames").mkdir(parents=True)
        self.train_manifest = self.root / "train_manifest.csv"
        self.validation_manifest = self.root / "validation_manifest.csv"
        self._train_rows = [
            ("train-a.png", "Abuse", "abuse-1", 0, 11),
            ("train-b.png", "Normal", "normal-1", 1, 22),
            ("train-c.png", "Abuse", "abuse-2", 0, 33),
        ]
        self._validation_rows = [
            ("val-a.png", "Normal", "normal-2", 0, 44),
            ("val-b.png", "Abuse", "abuse-3", 3, 55),
        ]
        self._write_split_manifest(self.train_manifest, "train", self._train_rows)
        self._write_split_manifest(self.validation_manifest, "validation", self._validation_rows)
        self.preprocessing = FramePreprocessing.from_config(
            {
                "resize_height": 64,
                "resize_width": 64,
                "interpolation": "bilinear",
                "color_mode": "RGB",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        )
        self.pack_dir = self.root / "packed"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_split_manifest(self, manifest_path: Path, split: str, rows: list[tuple[str, str, str, int, int]]) -> None:
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

    def _per_file_dataset(self, manifest_path: Path, split: str) -> P1CFrameDataset:
        return P1CFrameDataset(
            manifest_path,
            {"fixture": self.dataset_root},
            split=split,  # type: ignore[arg-type]
            class_names=("Abuse", "Normal"),
            preprocessing=self.preprocessing,
            verify_digests=True,
        )

    def _packed_dataset(self, manifest_path: Path, split: str) -> P1CPackedFrameDataset:
        return P1CPackedFrameDataset(
            manifest_path,
            {"fixture": self.dataset_root},
            split=split,  # type: ignore[arg-type]
            class_names=("Abuse", "Normal"),
            preprocessing=self.preprocessing,
            pack_closure_path=self.pack_dir / "pack_closure.json",
        )

    def test_closure_records_expected_provenance(self) -> None:
        closure = self._pack()
        self.assertEqual(closure["row_count"], 5)
        self.assertEqual(closure["per_file_verification_pass_count"], 5)
        self.assertEqual(closure["item_bytes"], 64 * 64 * 3)
        self.assertEqual(closure["image_shape"], [64, 64, 3])
        self.assertEqual(closure["splits"]["train"]["row_count"], 3)
        self.assertEqual(closure["splits"]["train"]["blob_row_offset"], 0)
        self.assertEqual(closure["splits"]["validation"]["row_count"], 2)
        self.assertEqual(closure["splits"]["validation"]["blob_row_offset"], 3)
        blob_path = self.pack_dir / "frames.bin"
        self.assertEqual(blob_path.stat().st_size, 5 * 64 * 64 * 3)
        self.assertEqual(_digest(blob_path), closure["blob_sha256"])
        # metadata.csv / labels.npy are the parallel, order-aligned provenance
        # arrays: source_video_id must survive end to end.
        with (self.pack_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            metadata_rows = list(csv.DictReader(handle))
        self.assertEqual([row["source_video_id"] for row in metadata_rows], ["abuse-1", "normal-1", "abuse-2", "normal-2", "abuse-3"])
        labels = np.load(self.pack_dir / "labels.npy")
        self.assertEqual(list(labels), ["Abuse", "Normal", "Abuse", "Normal", "Abuse"])

    def test_packed_dataset_is_bit_identical_to_per_file_dataset(self) -> None:
        self._pack()
        for manifest_path, split, count in (
            (self.train_manifest, "train", 3),
            (self.validation_manifest, "validation", 2),
        ):
            per_file = self._per_file_dataset(manifest_path, split)
            packed = self._packed_dataset(manifest_path, split)
            self.assertEqual(len(per_file), count)
            self.assertEqual(len(packed), count)
            for index in range(count):
                expected = per_file[index]
                actual = packed[index]
                self.assertTrue(
                    torch.equal(expected["image"], actual["image"]),
                    f"packed tensor differs from per-file tensor at {split}[{index}]",
                )
                self.assertEqual(expected["image"].dtype, actual["image"].dtype)
                self.assertEqual(expected["image"].shape, actual["image"].shape)
                for key in ("target", "source_dataset", "source_video_id", "source_frame_index", "filepath", "file_digest", "split"):
                    self.assertEqual(expected[key], actual[key], f"{key} differs at {split}[{index}]")

    def test_packed_dataset_preserves_source_video_groups(self) -> None:
        self._pack()
        per_file = self._per_file_dataset(self.train_manifest, "train")
        packed = self._packed_dataset(self.train_manifest, "train")
        self.assertEqual(packed.source_video_groups(), per_file.source_video_groups())
        self.assertEqual(packed.source_video_groups(), frozenset({("fixture", "abuse-1"), ("fixture", "normal-1"), ("fixture", "abuse-2")}))

    def test_pack_is_fail_closed_on_digest_mismatch(self) -> None:
        # Corrupt one source file after its correct digest was already written
        # into the manifest, exactly like an on-disk bit-rot or truncation.
        (self.dataset_root / "frames" / "train-b.png").write_bytes(b"not a png")
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            self._pack()
        self.assertFalse((self.pack_dir / "pack_closure.json").is_file(), "a failed pack must not produce a closure artifact")

    def test_pack_is_fail_closed_on_wrong_image_shape(self) -> None:
        # Overwrite one fixture file with a correctly-digested but wrongly
        # shaped image (32x32 instead of 64x64) to prove the shape assertion
        # actually fires rather than silently packing malformed rows.
        bad_path = self.dataset_root / "frames" / "train-a.png"
        small = np.zeros((32, 32, 3), dtype=np.uint8)
        Image.fromarray(small, mode="RGB").save(bad_path)
        rows = list(csv.DictReader((self.train_manifest).open(newline="", encoding="utf-8")))
        rows[0]["file_digest"] = _digest(bad_path)
        with self.train_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "not \\(64, 64, 3\\) uint8 RGB"):
            self._pack()
        self.assertFalse((self.pack_dir / "pack_closure.json").is_file())

    def test_pack_resumes_without_retouching_completed_rows(self) -> None:
        closure = self._pack()
        state_path = self.pack_dir / "pack_state.json"
        self.assertTrue(state_path.is_file())
        closure_path = self.pack_dir / "pack_closure.json"
        # Simulate "the closure step never ran" (e.g. the process was killed
        # right after the last row was written) by removing only the closure
        # marker; blob/state/metadata all remain from the completed pass.
        closure_path.unlink()
        # Corrupt a source file that is already fully packed. If resuming
        # re-verified completed rows, this would raise; it must not.
        (self.dataset_root / "frames" / "train-a.png").write_bytes(b"corrupted-after-pack")
        second_closure = self._pack()
        self.assertEqual(second_closure["row_count"], closure["row_count"])
        self.assertEqual(second_closure["blob_sha256"], closure["blob_sha256"])
        self.assertEqual(second_closure["per_file_verification_pass_count"], 5)

    def test_repacking_a_closed_directory_is_refused(self) -> None:
        self._pack()
        with self.assertRaisesRegex(FileExistsError, "already closed"):
            self._pack()

    def test_packed_dataset_refuses_a_closure_from_a_different_manifest(self) -> None:
        self._pack()
        other_manifest = self.root / "other_train_manifest.csv"
        self._write_split_manifest(other_manifest, "train", self._train_rows[:2])
        with self.assertRaisesRegex(ValueError, "different train manifest"):
            self._packed_dataset(other_manifest, "train")

    def test_packed_dataset_detects_blob_tampering(self) -> None:
        self._pack()
        blob_path = self.pack_dir / "frames.bin"
        data = bytearray(blob_path.read_bytes())
        data[0] ^= 0xFF
        blob_path.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "failed SHA-256 verification"):
            self._packed_dataset(self.train_manifest, "train")


if __name__ == "__main__":
    unittest.main()
