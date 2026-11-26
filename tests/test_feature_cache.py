"""Tests for `core.feature_cache` (B-1).

Uses a tiny synthetic backbone and dataset rather than a real ResNet50 and
video blob: the property under test is the cache's build/load contract and
its refusals, not backbone numerics, and a fake spatial backbone makes the
expected feature vector for each row computable by hand.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import (  # noqa: E402
    FEATURE_CACHE_POOLING_BY_FAMILY,
    FeatureCacheMismatch,
    build_feature_cache,
    load_feature_cache,
)
from data.augmentation import NONE_AUGMENTATION, resolve_augmentation_policy  # noqa: E402
from data.p1c_loader import FramePreprocessing  # noqa: E402

CLASS_NAMES = ["cat", "dog", "bird"]

PREPROCESSING = FramePreprocessing(
    resize_height=8,
    resize_width=8,
    interpolation="bilinear",
    color_mode="RGB",
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
)

OTHER_PREPROCESSING = FramePreprocessing(
    resize_height=16,
    resize_width=16,
    interpolation="bilinear",
    color_mode="RGB",
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
)

_STANDARD_FRAME = {
    "name": "standard_frame",
    "rationale": "Non-`none` policy used only to prove the cache refuses it, not to actually augment anything.",
    "random_resized_crop": {"scale_min": 0.7, "scale_max": 1.0, "ratio_min": 0.75, "ratio_max": 1.333},
    "horizontal_flip_probability": 0.5,
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
}


class _FakeSpatialBackbone(nn.Module):
    """Returns a constant-per-channel [B, 4, 2, 2] map: value = channel index + offset."""

    out_channels = 4

    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        base = torch.arange(self.out_channels, dtype=torch.float32) + self.offset
        return base.view(1, self.out_channels, 1, 1).expand(batch, self.out_channels, 2, 2).clone()

    def pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        """Same global-average-pool every real backbone's `pooled_features` performs (B-2)."""

        return torch.flatten(nn.functional.adaptive_avg_pool2d(self.forward(images), (1, 1)), 1)


class _FakeFrameDataset(Dataset):
    """Yields `n` rows shaped like `P1CVideoFrameDataset.__getitem__`."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> dict:
        target = index % len(CLASS_NAMES)
        return {
            "image": torch.zeros(3, 8, 8, dtype=torch.float32),
            "target": target,
            "source_dataset": "unit_test_dataset",
            "source_video_id": f"video{index // 2:03d}",
            "source_frame_index": index % 2,
        }


_COMMON_KWARGS = dict(
    class_names=CLASS_NAMES,
    split="train",
    # Must be a family `core.feature_cache` recognizes (B-2 widened pooling
    # from one constant to a per-family registry) -- reuses the real
    # ResNet50 family id even though `_FakeSpatialBackbone` is not one.
    backbone_family="torchvision_resnet50",
    backbone_feature="layer4_C5_pre_GAP",
    backbone_trainable="frozen",
    backbone_weight_sha256="a" * 64,
    model_control="gap_linear",
    preprocessing=PREPROCESSING,
    augmentation=NONE_AUGMENTATION,
    frame_blob_sha256="b" * 64,
    source_manifest_path="results/p1b/fake/train_manifest.csv",
    source_manifest_sha256="c" * 64,
)


class BuildAndLoadRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_dir = Path(self._tmp.name) / "cache"

    def test_round_trip_preserves_row_order_and_label_provenance(self) -> None:
        dataset = _FakeFrameDataset(9)
        manifest = build_feature_cache(
            dataset, _FakeSpatialBackbone(), output_dir=self.cache_dir, device="cpu",
            batch_size=4, **_COMMON_KWARGS,
        )
        self.assertEqual(manifest.row_count, 9)
        self.assertEqual(manifest.feature_dim, 4)
        self.assertEqual(manifest.dtype, "float16")
        self.assertEqual(manifest.pooling, FEATURE_CACHE_POOLING_BY_FAMILY["torchvision_resnet50"])

        features, rows, loaded_manifest = load_feature_cache(self.cache_dir, **_COMMON_KWARGS)
        self.assertEqual(loaded_manifest, manifest)
        self.assertEqual(features.shape, (9, 4))
        self.assertEqual(len(rows), 9)

        expected_feature = np.arange(4, dtype=np.float32)
        for i in range(9):
            np.testing.assert_allclose(features[i], expected_feature, rtol=1e-3)
            self.assertEqual(rows[i]["row_index"], str(i))
            self.assertEqual(rows[i]["source_video_id"], f"video{i // 2:03d}")
            self.assertEqual(int(rows[i]["source_frame_index"]), i % 2)
            self.assertEqual(int(rows[i]["target"]), i % len(CLASS_NAMES))
            self.assertEqual(rows[i]["label"], CLASS_NAMES[i % len(CLASS_NAMES)])

    def test_float32_dtype_is_selectable_and_more_precise(self) -> None:
        dataset = _FakeFrameDataset(3)
        manifest = build_feature_cache(
            dataset, _FakeSpatialBackbone(), output_dir=self.cache_dir, device="cpu",
            dtype="float32", **_COMMON_KWARGS,
        )
        self.assertEqual(manifest.dtype, "float32")
        features, _rows, _manifest = load_feature_cache(self.cache_dir, **_COMMON_KWARGS)
        self.assertEqual(features.dtype, np.float32)


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_dir = Path(self._tmp.name) / "cache"
        build_feature_cache(
            _FakeFrameDataset(6), _FakeSpatialBackbone(), output_dir=self.cache_dir, device="cpu",
            **_COMMON_KWARGS,
        )

    def test_build_refuses_non_none_augmentation(self) -> None:
        augmented = resolve_augmentation_policy(_STANDARD_FRAME)
        kwargs = dict(_COMMON_KWARGS)
        kwargs["augmentation"] = augmented
        with self.assertRaises(ValueError) as caught:
            build_feature_cache(
                _FakeFrameDataset(6), _FakeSpatialBackbone(), output_dir=Path(self.cache_dir.parent) / "other",
                device="cpu", **kwargs,
            )
        self.assertIn("augmentation.name == 'none'", str(caught.exception))

    def test_build_refuses_a_head_that_does_not_pool(self) -> None:
        """A pre-GAP spatial head must never be served pooled vectors (backlog B-4)."""

        kwargs = dict(_COMMON_KWARGS)
        kwargs["model_control"] = "semscnn_multiscale_se"
        with self.assertRaises(ValueError) as caught:
            build_feature_cache(
                _FakeFrameDataset(6), _FakeSpatialBackbone(),
                output_dir=Path(self.cache_dir.parent) / "spatial_head", device="cpu", **kwargs,
            )
        self.assertIn("spatial", str(caught.exception))

    def test_load_refuses_a_different_head_control(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["model_control"] = "same_c5_mlp"
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        self.assertIn("model_control", str(caught.exception))

    def test_build_refuses_non_frozen_backbone(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["backbone_trainable"] = "full"
        with self.assertRaises(ValueError) as caught:
            build_feature_cache(
                _FakeFrameDataset(6), _FakeSpatialBackbone(), output_dir=Path(self.cache_dir.parent) / "other",
                device="cpu", **kwargs,
            )
        self.assertIn("backbone_trainable == 'frozen'", str(caught.exception))

    def test_load_refuses_non_none_augmentation(self) -> None:
        augmented = resolve_augmentation_policy(_STANDARD_FRAME)
        kwargs = dict(_COMMON_KWARGS)
        kwargs["augmentation"] = augmented
        with self.assertRaises(FeatureCacheMismatch):
            load_feature_cache(self.cache_dir, **kwargs)

    def test_load_refuses_backbone_weight_digest_mismatch(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["backbone_weight_sha256"] = "f" * 64
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        self.assertIn("backbone_weight_sha256", str(caught.exception))

    def test_load_refuses_preprocessing_mismatch(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["preprocessing"] = OTHER_PREPROCESSING
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        self.assertIn("preprocessing", str(caught.exception))

    def test_load_refuses_source_manifest_mismatch(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["source_manifest_sha256"] = "9" * 64
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        self.assertIn("source_manifest_sha256", str(caught.exception))

    def test_load_refuses_frame_blob_mismatch(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["frame_blob_sha256"] = "9" * 64
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        self.assertIn("frame_blob_sha256", str(caught.exception))

    def test_load_refuses_class_names_mismatch(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["class_names"] = ["cat", "dog"]
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        self.assertIn("class_names", str(caught.exception))

    def test_load_refuses_split_mismatch(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["split"] = "validation"
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        self.assertIn("split", str(caught.exception))

    def test_load_refuses_corrupted_feature_file(self) -> None:
        features_path = self.cache_dir / "features.float16.npy"
        raw = bytearray(features_path.read_bytes())
        raw[-1] ^= 0xFF
        features_path.write_bytes(bytes(raw))
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **_COMMON_KWARGS)
        self.assertIn("sha256", str(caught.exception))

    def test_load_refuses_missing_manifest(self) -> None:
        empty_dir = Path(self.cache_dir.parent) / "missing"
        empty_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            load_feature_cache(empty_dir, **_COMMON_KWARGS)

    def test_load_refuses_a_cache_built_under_the_other_backbones_pooling(self) -> None:
        """A ResNet50 (global_average) cache must never serve a DINOv2 (cls_plus_mean_patch) load, or vice versa (B-2)."""

        kwargs = dict(_COMMON_KWARGS)
        kwargs["backbone_family"] = "timm_dinov2_vitb14_reg4"
        with self.assertRaises(FeatureCacheMismatch) as caught:
            load_feature_cache(self.cache_dir, **kwargs)
        message = str(caught.exception)
        self.assertIn("backbone_family", message)
        self.assertIn("pooling", message)

    def test_build_refuses_an_unregistered_backbone_family(self) -> None:
        kwargs = dict(_COMMON_KWARGS)
        kwargs["backbone_family"] = "some_other_backbone"
        with self.assertRaises(ValueError) as caught:
            build_feature_cache(
                _FakeFrameDataset(6), _FakeSpatialBackbone(), output_dir=Path(self.cache_dir.parent) / "other",
                device="cpu", **kwargs,
            )
        self.assertIn("unregistered backbone_family", str(caught.exception))


class EmptyDatasetTests(unittest.TestCase):
    def test_build_refuses_an_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_feature_cache(
                    _FakeFrameDataset(0), _FakeSpatialBackbone(), output_dir=Path(tmp) / "cache",
                    device="cpu", **_COMMON_KWARGS,
                )


if __name__ == "__main__":
    unittest.main()
