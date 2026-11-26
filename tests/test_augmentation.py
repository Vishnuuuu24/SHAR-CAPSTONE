"""Tests for `data.augmentation`.

K-22: `fixed_protocol.augmentation` used to be registered in every config but
validated only as a fail-closed no-op and read by no loader code. These tests
pin the fix: `none` still resolves to the exact byte-identical resize every
P1C dataset already did; `standard_frame` is validated strictly (every
refusal path) and, once resolved, actually changes pixels; augmentation is
reproducible from (run seed, epoch, sample identity) with no dependency on
worker id or call order; and a validation/test dataset structurally cannot be
handed a real augmentation policy.
"""

from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.augmentation import (  # noqa: E402
    NONE_AUGMENTATION,
    AugmentationPolicy,
    augmentation_seed,
    resolve_augmentation_policy,
)
from data.contracts import MANIFEST_FIELDS  # noqa: E402
from data.p1c_loader import FramePreprocessing, P1CFrameDataset, build_p1c_frame_dataloader, FrameLoaderOptions  # noqa: E402

_STANDARD_FRAME_SPEC = {
    "name": "standard_frame",
    "rationale": "Attack the near-duplicate-frame memorisation seen at epoch 0-1 of the full fine-tune.",
    "random_resized_crop": {"scale_min": 0.7, "scale_max": 1.0, "ratio_min": 0.75, "ratio_max": 1.333},
    "horizontal_flip_probability": 0.5,
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_noisy_image(path: Path, seed: int, size: int = 64) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


class ResolveNoneTests(unittest.TestCase):
    def test_none_is_accepted(self) -> None:
        policy = resolve_augmentation_policy({"name": "none", "rationale": "conventional auditable control"})
        self.assertEqual(policy.name, "none")
        self.assertTrue(policy.is_none)

    def test_none_apply_is_byte_identical_to_a_plain_resize(self) -> None:
        image = Image.new("RGB", (40, 30))
        rng = np.random.default_rng(0)
        image.putdata([tuple(pixel) for pixel in rng.integers(0, 256, size=(40 * 30, 3))])
        policy = resolve_augmentation_policy({"name": "none", "rationale": "conventional auditable control"})
        augmented = policy.apply(image, resize_width=20, resize_height=15, resample=Image.Resampling.BILINEAR, seed=123)
        expected = image.resize((20, 15), resample=Image.Resampling.BILINEAR)
        self.assertEqual(list(augmented.getdata()), list(expected.getdata()))

    def test_none_report(self) -> None:
        policy = resolve_augmentation_policy({"name": "none", "rationale": "x"})
        self.assertEqual(policy.report(), {"name": "none", "rationale": "x"})

    def test_none_rejects_unexpected_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "accepts only"):
            resolve_augmentation_policy({"name": "none", "rationale": "x", "horizontal_flip_probability": 0.5})

    def test_none_augmentation_constant_is_none(self) -> None:
        self.assertTrue(NONE_AUGMENTATION.is_none)


class ResolveRefusalTests(unittest.TestCase):
    def test_non_mapping_is_refused(self) -> None:
        for bad in (None, "none", 1, ["name"]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                resolve_augmentation_policy(bad)

    def test_unregistered_name_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            resolve_augmentation_policy({"name": "random_crop", "rationale": "x" * 40})

    def test_missing_rationale_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rationale"):
            resolve_augmentation_policy({"name": "none"})

    def test_empty_rationale_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rationale"):
            resolve_augmentation_policy({"name": "none", "rationale": "   "})

    def test_standard_frame_short_rationale_is_refused(self) -> None:
        spec = dict(_STANDARD_FRAME_SPEC, rationale="too short")
        with self.assertRaisesRegex(ValueError, "at least"):
            resolve_augmentation_policy(spec)

    def test_standard_frame_missing_key_is_refused(self) -> None:
        spec = {k: v for k, v in _STANDARD_FRAME_SPEC.items() if k != "color_jitter"}
        with self.assertRaisesRegex(ValueError, "missing"):
            resolve_augmentation_policy(spec)

    def test_standard_frame_unknown_key_is_refused(self) -> None:
        spec = dict(_STANDARD_FRAME_SPEC, unexpected=1)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            resolve_augmentation_policy(spec)

    def test_crop_missing_or_unknown_keys_are_refused(self) -> None:
        for crop in ({"scale_min": 0.7}, {**_STANDARD_FRAME_SPEC["random_resized_crop"], "extra": 1}):
            spec = dict(_STANDARD_FRAME_SPEC, random_resized_crop=crop)
            with self.subTest(crop=crop), self.assertRaisesRegex(ValueError, "random_resized_crop"):
                resolve_augmentation_policy(spec)

    def test_crop_scale_min_greater_than_scale_max_is_refused(self) -> None:
        crop = dict(_STANDARD_FRAME_SPEC["random_resized_crop"], scale_min=0.9, scale_max=0.5)
        with self.assertRaisesRegex(ValueError, "scale_min"):
            resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, random_resized_crop=crop))

    def test_crop_scale_out_of_unit_range_is_refused(self) -> None:
        crop = dict(_STANDARD_FRAME_SPEC["random_resized_crop"], scale_max=1.5)
        with self.assertRaisesRegex(ValueError, "scale_min"):
            resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, random_resized_crop=crop))

    def test_crop_ratio_min_greater_than_ratio_max_is_refused(self) -> None:
        crop = dict(_STANDARD_FRAME_SPEC["random_resized_crop"], ratio_min=2.0, ratio_max=1.0)
        with self.assertRaisesRegex(ValueError, "ratio_min"):
            resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, random_resized_crop=crop))

    def test_flip_probability_out_of_range_is_refused(self) -> None:
        for bad in (-0.1, 1.1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "within \\[0,1\\]"):
                resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, horizontal_flip_probability=bad))

    def test_jitter_missing_or_unknown_keys_are_refused(self) -> None:
        for jitter in ({"brightness": 0.1}, {**_STANDARD_FRAME_SPEC["color_jitter"], "extra": 1}):
            spec = dict(_STANDARD_FRAME_SPEC, color_jitter=jitter)
            with self.subTest(jitter=jitter), self.assertRaisesRegex(ValueError, "color_jitter"):
                resolve_augmentation_policy(spec)

    def test_negative_jitter_strengths_are_refused(self) -> None:
        for field in ("brightness", "contrast", "saturation"):
            jitter = dict(_STANDARD_FRAME_SPEC["color_jitter"], **{field: -0.1})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"{field} must be non-negative"):
                resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, color_jitter=jitter))

    def test_hue_out_of_range_is_refused(self) -> None:
        for bad in (-0.1, 0.6):
            jitter = dict(_STANDARD_FRAME_SPEC["color_jitter"], hue=bad)
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "hue"):
                resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, color_jitter=jitter))

    def test_wrong_types_are_refused(self) -> None:
        with self.assertRaises(TypeError):
            resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, horizontal_flip_probability="0.5"))
        crop = dict(_STANDARD_FRAME_SPEC["random_resized_crop"], scale_min=True)
        with self.assertRaises(TypeError):
            resolve_augmentation_policy(dict(_STANDARD_FRAME_SPEC, random_resized_crop=crop))


class StandardFrameApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = resolve_augmentation_policy(_STANDARD_FRAME_SPEC)
        rng = np.random.default_rng(7)
        array = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        self.image = Image.fromarray(array, mode="RGB")

    def test_resolves_to_registered_shape(self) -> None:
        out = self.policy.apply(self.image, resize_width=64, resize_height=64, resample=Image.Resampling.BILINEAR, seed=1)
        self.assertEqual(out.size, (64, 64))

    def test_changes_pixels_relative_to_a_plain_resize(self) -> None:
        plain = self.image.resize((64, 64), resample=Image.Resampling.BILINEAR)
        augmented = self.policy.apply(self.image, resize_width=64, resize_height=64, resample=Image.Resampling.BILINEAR, seed=1)
        self.assertNotEqual(list(plain.getdata()), list(augmented.getdata()))

    def test_same_seed_reproduces_the_same_output(self) -> None:
        first = self.policy.apply(self.image, resize_width=64, resize_height=64, resample=Image.Resampling.BILINEAR, seed=42)
        second = self.policy.apply(self.image, resize_width=64, resize_height=64, resample=Image.Resampling.BILINEAR, seed=42)
        self.assertEqual(list(first.getdata()), list(second.getdata()))

    def test_different_seeds_usually_differ(self) -> None:
        first = self.policy.apply(self.image, resize_width=64, resize_height=64, resample=Image.Resampling.BILINEAR, seed=1)
        second = self.policy.apply(self.image, resize_width=64, resize_height=64, resample=Image.Resampling.BILINEAR, seed=2)
        self.assertNotEqual(list(first.getdata()), list(second.getdata()))

    def test_report_reflects_every_registered_parameter(self) -> None:
        self.assertEqual(self.policy.report(), _STANDARD_FRAME_SPEC)


class AugmentationSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = resolve_augmentation_policy(_STANDARD_FRAME_SPEC)
        self.digest = "a" * 64

    def test_same_inputs_reproduce_the_same_seed(self) -> None:
        first = augmentation_seed("frames/a.png", self.digest, run_seed=3, epoch=0, policy=self.policy)
        second = augmentation_seed("frames/a.png", self.digest, run_seed=3, epoch=0, policy=self.policy)
        self.assertEqual(first, second)

    def test_different_samples_get_different_seeds(self) -> None:
        # This is the property that makes the duplicated-per-worker-stream bug
        # structurally impossible here: two samples' seeds never depend on
        # which worker or what order processed them, only on sample identity.
        first = augmentation_seed("frames/a.png", self.digest, run_seed=3, epoch=0, policy=self.policy)
        second = augmentation_seed("frames/b.png", self.digest, run_seed=3, epoch=0, policy=self.policy)
        self.assertNotEqual(first, second)

    def test_different_epochs_get_different_seeds(self) -> None:
        first = augmentation_seed("frames/a.png", self.digest, run_seed=3, epoch=0, policy=self.policy)
        second = augmentation_seed("frames/a.png", self.digest, run_seed=3, epoch=1, policy=self.policy)
        self.assertNotEqual(first, second)

    def test_different_run_seeds_get_different_seeds(self) -> None:
        first = augmentation_seed("frames/a.png", self.digest, run_seed=3, epoch=0, policy=self.policy)
        second = augmentation_seed("frames/a.png", self.digest, run_seed=4, epoch=0, policy=self.policy)
        self.assertNotEqual(first, second)

    def test_negative_epoch_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            augmentation_seed("frames/a.png", self.digest, run_seed=0, epoch=-1, policy=self.policy)


class DatasetTrainOnlyGuardTests(unittest.TestCase):
    """`P1CFrameDataset` must refuse a real policy on anything but split=train."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "fixture-dataset"
        (self.dataset_root / "frames").mkdir(parents=True)
        for name, seed in (("train.png", 1), ("validation.png", 2)):
            _write_noisy_image(self.dataset_root / "frames" / name, seed)
        self.manifest = self.root / "manifest.csv"
        rows = []
        for name, split, video in (("train.png", "train", "v1"), ("validation.png", "validation", "v2")):
            path = self.dataset_root / "frames" / name
            rows.append(
                {
                    "filepath": f"frames/{name}", "source_dataset": "fixture", "source_video_id": video,
                    "source_frame_index": 0, "label": "Abuse", "label_scope": "video_inherited",
                    "label_source": "fixture-folder", "annotation_version": "fixture-v1",
                    "inside_official_interval": "", "split": split, "file_digest": _digest(path),
                }
            )
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.preprocessing = FramePreprocessing.from_config(
            {"resize_height": 32, "resize_width": 32, "interpolation": "bilinear", "color_mode": "RGB",
             "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        )
        self.policy = resolve_augmentation_policy(_STANDARD_FRAME_SPEC)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _dataset(self, split: str, **kwargs) -> P1CFrameDataset:
        return P1CFrameDataset(
            self.manifest, {"fixture": self.dataset_root}, split=split, class_names=("Abuse",),
            preprocessing=self.preprocessing, verify_digests=True, **kwargs,
        )

    def test_train_split_accepts_a_real_policy(self) -> None:
        dataset = self._dataset("train", augmentation=self.policy, augmentation_run_seed=1)
        self.assertEqual(dataset.augmentation, self.policy)

    def test_validation_split_refuses_a_real_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "may only be registered for split=train"):
            self._dataset("validation", augmentation=self.policy)

    def test_no_augmentation_argument_defaults_to_none_on_every_split(self) -> None:
        self.assertTrue(self._dataset("train").augmentation.is_none)
        self.assertTrue(self._dataset("validation").augmentation.is_none)

    def test_wrong_type_is_refused(self) -> None:
        with self.assertRaises(TypeError):
            self._dataset("train", augmentation={"name": "standard_frame"})

    def test_validation_is_never_augmented_end_to_end(self) -> None:
        """The explicit, whole-pipeline version of the guard above.

        Even with a `standard_frame` policy in hand, a validation dataset's
        actual pixels never move: construction itself already refused the
        combination, so this proves there is no code path -- present or
        future -- that could apply it anyway.
        """

        with self.assertRaises(ValueError):
            self._dataset("validation", augmentation=self.policy)
        plain_validation = self._dataset("validation")
        sample = plain_validation[0]
        expected = plain_validation._preprocessing
        with Image.open(self.dataset_root / "frames" / "validation.png") as opened:
            resized = opened.convert("RGB").resize(
                (expected.resize_width, expected.resize_height), resample=Image.Resampling.BILINEAR
            )
        expected_tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
        self.assertTrue(torch.equal(sample["image"], expected_tensor))


class DatasetWorkerDeterminismTests(unittest.TestCase):
    """The point of this module's seeding: no dependency on worker assignment.

    A duplicated-per-worker RNG stream would make every worker hand back the
    same augmentation for whatever sample it happens to process. Our seed is
    a pure function of (sample identity, run seed, epoch, policy), so the
    same sample must augment identically whether it is read single-process
    or by one of several persistent DataLoader workers, and two different
    samples must never coincide.
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "fixture-dataset"
        (self.dataset_root / "frames").mkdir(parents=True)
        self.names = [f"train-{i}.png" for i in range(6)]
        for index, name in enumerate(self.names):
            _write_noisy_image(self.dataset_root / "frames" / name, seed=100 + index)
        self.manifest = self.root / "manifest.csv"
        rows = []
        for index, name in enumerate(self.names):
            path = self.dataset_root / "frames" / name
            rows.append(
                {
                    "filepath": f"frames/{name}", "source_dataset": "fixture", "source_video_id": f"v{index}",
                    "source_frame_index": 0, "label": "Abuse", "label_scope": "video_inherited",
                    "label_source": "fixture-folder", "annotation_version": "fixture-v1",
                    "inside_official_interval": "", "split": "train", "file_digest": _digest(path),
                }
            )
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.preprocessing = FramePreprocessing.from_config(
            {"resize_height": 32, "resize_width": 32, "interpolation": "bilinear", "color_mode": "RGB",
             "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        )
        self.policy = resolve_augmentation_policy(_STANDARD_FRAME_SPEC)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _dataset(self) -> P1CFrameDataset:
        return P1CFrameDataset(
            self.manifest, {"fixture": self.dataset_root}, split="train", class_names=("Abuse",),
            preprocessing=self.preprocessing, verify_digests=True,
            augmentation=self.policy, augmentation_run_seed=11,
        )

    def _read_by_filepath(self, num_workers: int) -> dict[str, torch.Tensor]:
        dataset = self._dataset()
        dataset.set_epoch(0)
        loader = build_p1c_frame_dataloader(
            dataset,
            options=FrameLoaderOptions(
                batch_size=1, shuffle=False, num_workers=num_workers, seed=0, epoch=0,
                pin_memory=False, drop_last=False, persistent_workers=num_workers > 0,
            ),
        )
        by_path = {}
        for batch in loader:
            by_path[batch["filepath"][0]] = batch["image"][0]
        return by_path

    def test_single_process_and_multi_worker_agree_per_sample(self) -> None:
        single = self._read_by_filepath(num_workers=0)
        multi = self._read_by_filepath(num_workers=2)
        self.assertEqual(set(single), set(multi))
        for path in single:
            self.assertTrue(torch.equal(single[path], multi[path]), path)

    def test_distinct_samples_are_not_augmented_identically(self) -> None:
        samples = self._read_by_filepath(num_workers=0)
        tensors = list(samples.values())
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                self.assertFalse(torch.equal(tensors[i], tensors[j]), (i, j))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
