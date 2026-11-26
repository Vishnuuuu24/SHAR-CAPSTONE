#!/usr/bin/env python3
"""Build a B-1 frozen-backbone feature cache for a registered P1C config.

Thin CLI: parses arguments, resolves the same config/dataset/backbone objects
`scripts/run_p1c_frame_baseline.py` already resolves for training (reused
directly, per docs/PROJECT_RULES.md's rule of two, rather than re-parsed here), and hands
them to `core.feature_cache.build_feature_cache` -- the module that does the
actual forward pass and refusal checks.

Only meaningful for a config whose backbone is registered `frozen` and whose
augmentation is registered `none`; `core.feature_cache.build_feature_cache`
refuses anything else itself, so this script does not duplicate that check.

Only the D-34/D-35 video-blob and D-33 packed-blob pixel sources are
supported: both register one blob-level SHA-256 the cache manifest can pin
against. A per-file config has no single such digest and is out of scope for
this first cache implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import FEATURE_DTYPES, build_feature_cache
from core.feature_cache_verify import verify_feature_cache
from data import pixel_source
from data.augmentation import NONE_AUGMENTATION, AugmentationPolicy, resolve_augmentation_policy
from data.p1c_loader import FramePreprocessing
from data.video_dataset import P1CVideoFrameDataset
from models.classifiers import apply_backbone_trainable_mode
from models.dinov2_backbone import DINOv2ViTB14Reg
from models.siglip_backbone import SigLIPBaseP16
from models.videomae_backbone import VideoMAEBaseK400
from models.vjepa2_backbone import VJEPA2LargeEncoder
from safetensors.torch import load_file as _load_safetensors
from scripts.run_p1c_frame_baseline import _load_config, _local_resnet50, _make_datasets

_SPLITS = ("train", "validation", "test")
#: The test split is the config's inherited-label `noisy_proxy_test` view (D-40), spent once on a final configuration.
_DATA_KEY = {"train": "train", "validation": "validation", "test": "noisy_proxy_test"}


def _local_dinov2(weights: dict) -> DINOv2ViTB14Reg:
    """Load the registered DINOv2 ViT-B/14-reg4 checkpoint (B-2's second backbone).

    Mirrors `scripts.run_p1c_frame_baseline._local_resnet50`: the digest is
    already verified by `core.p1c_readiness` before this runs, so this only
    loads the local file -- never a network fetch -- and hands it to
    `DINOv2ViTB14Reg.from_local_weights` for the exact-state-dict check.
    """

    path = REPO_ROOT / str(weights["path"])
    payload = _load_safetensors(str(path))
    return DINOv2ViTB14Reg.from_local_weights(payload, weights_name=str(weights["enum"]))


def _local_videomae(weights: dict) -> VideoMAEBaseK400:
    """Load the registered VideoMAE-B/16 Kinetics-400 checkpoint (clip-level backbone).

    Mirrors `_local_dinov2`: the digest is already verified by
    `core.p1c_readiness` before this runs, so this only loads the local file
    -- never a network fetch -- and hands it to
    `VideoMAEBaseK400.from_local_weights` for the q_bias/v_bias remap and
    exact-state-dict check. NOTE: the existing builder in this script is
    frame-level; a clip-level cache builder that actually groups frames into
    16-frame clips is a separate script, not attempted here.
    """

    path = REPO_ROOT / str(weights["path"])
    payload = _load_safetensors(str(path))
    return VideoMAEBaseK400.from_local_weights(payload, weights_name=str(weights["enum"]))


def _local_siglip(weights: dict) -> SigLIPBaseP16:
    """Load the registered SigLIP ViT-B/16-224 vision tower (flat state dict, strict)."""

    path = REPO_ROOT / str(weights["path"])
    payload = _load_safetensors(str(path))
    return SigLIPBaseP16.from_local_weights(payload, weights_name=str(weights["enum"]))


def _local_vjepa2(weights: dict) -> VJEPA2LargeEncoder:
    """Load the registered V-JEPA 2 ViT-L encoder (Meta's original checkpoint, `encoder` branch, strict).

    The digest is verified by `core.p1c_readiness` before this runs. `mmap=True` keeps the 5 GB file (encoder, predictor,
    EMA target encoder, optimizer) from being read into memory just to use one branch.
    """

    path = REPO_ROOT / str(weights["path"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return VJEPA2LargeEncoder.from_local_weights(checkpoint["encoder"], weights_name=str(weights["enum"]))


#: `fixed_protocol.backbone.family` -> the loader that builds that backbone from
#: its registered local weight file. One entry per family in
#: `core.p1c_readiness.BACKBONE_FAMILIES`, so a config naming a family this
#: script does not know how to build fails loudly rather than silently
#: falling back to ResNet50.
_BACKBONE_LOADERS = {
    "torchvision_resnet50": _local_resnet50,
    "timm_dinov2_vitb14_reg4": _local_dinov2,
    "transformers_videomae_base_k400": _local_videomae,
    "transformers_siglip_base_p16_224": _local_siglip,
    "transformers_vjepa2_vitl": _local_vjepa2,
}


def _load_backbone(backbone_block: dict):
    family = backbone_block["family"]
    try:
        loader = _BACKBONE_LOADERS[family]
    except KeyError:
        raise ValueError(
            f"no feature-cache backbone loader registered for family {family!r}; "
            f"expected one of {sorted(_BACKBONE_LOADERS)}"
        ) from None
    return loader(backbone_block["weights"])


def _blob_sha256(protocol: dict) -> str:
    """The one registered blob-level SHA-256 this config's pixel source names."""

    video_sha256 = pixel_source.video_blob_sha256(protocol)
    if video_sha256 is not None:
        return video_sha256
    packed_sha256 = pixel_source.packed_blob_sha256(protocol)
    if packed_sha256 is not None:
        return packed_sha256
    raise ValueError(
        "the feature cache CLI supports only the D-34/D-35 video blob and D-33 packed blob "
        "pixel sources -- this config registers neither data.video_frames nor data.packed_frames"
    )


def _augmentation_for_split(protocol: dict, split: str) -> AugmentationPolicy:
    """The augmentation policy actually applied to `split` by `_make_datasets`.

    Validation is never augmented regardless of what `fixed_protocol.augmentation`
    registers (`_make_datasets` never passes an `augmentation=` argument for it);
    the train split's registered policy is read for real so a config that
    registers `standard_frame` reaches `build_feature_cache`'s refusal honestly,
    instead of this CLI silently reporting `none` for a cache that would be
    wrong from the next epoch.
    """

    if split in ("validation", "test"):
        return NONE_AUGMENTATION
    return resolve_augmentation_policy(protocol["augmentation"])


def _load_kwargs_for_config(protocol: dict, split: str, trainable: str) -> dict:
    """The identity a cache is built under and must later be loaded under.

    One definition, used by the builder, the verifier and
    `scripts/verify_p1c_feature_cache.py` alike -- if these three ever derived
    the cache identity separately, a cache could be built under one reading of
    the config and verified under another, and the mismatch check would be
    checking itself.
    """

    backbone_block = protocol["backbone"]
    return dict(
        class_names=protocol["model"]["class_names"],
        split=split,
        backbone_family=backbone_block["family"],
        backbone_feature=backbone_block["feature"],
        backbone_trainable=trainable,
        backbone_weight_sha256=backbone_block["weights"]["sha256"],
        model_control=protocol["model"]["control"],
        preprocessing=FramePreprocessing.from_config(protocol["preprocessing"]),
        augmentation=_augmentation_for_split(protocol, split),
        frame_blob_sha256=_blob_sha256(protocol),
        source_manifest_path=protocol["data"][_DATA_KEY[split]]["path"],
        source_manifest_sha256=protocol["data"][_DATA_KEY[split]]["sha256"],
    )


def _make_test_dataset(protocol: dict) -> P1CVideoFrameDataset:
    """The `noisy_proxy_test` view through the same video-blob loader `_make_datasets` uses for validation."""

    data = protocol["data"]
    closure = pixel_source.extract_closure_path(data)
    if closure is None:
        raise ValueError("the test split is only supported for the D-34/D-35 video-blob pixel source")
    pixel_source.require_verify_digests_true(protocol, field="video_frames")
    return P1CVideoFrameDataset(
        REPO_ROOT / data["noisy_proxy_test"]["path"], closure_path=closure,
        preprocessing=FramePreprocessing.from_config(protocol["preprocessing"]),
        class_names=protocol["model"]["class_names"], split="test",
    )


def build_cache_for_config(
    config_path: Path,
    *,
    split: str,
    output_dir: Path,
    device: str,
    dtype: str,
    batch_size: int,
    num_workers: int,
    verify_sample: int,
) -> tuple[Path, dict]:
    """Build one split's cache, verify a sample against the live backbone, and return both."""

    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}; got {split!r}")
    _config, protocol, _run_class = _load_config(config_path)

    backbone_block = protocol["backbone"]
    trainable = backbone_block.get("trainable", "full")
    backbone = apply_backbone_trainable_mode(_load_backbone(backbone_block), trainable)

    if split == "test":
        dataset = _make_test_dataset(protocol)
    else:
        train, validation = _make_datasets(protocol, seed=0)
        dataset = train if split == "train" else validation

    load_kwargs = _load_kwargs_for_config(protocol, split, trainable)

    build_feature_cache(
        dataset,
        backbone,
        output_dir=output_dir,
        device=torch.device(device),
        dtype=dtype,
        **load_kwargs,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Never return an unverified cache. A manifest proves the cache was built
    # under this config; only recomputation proves its numbers are the ones the
    # backbone emits, and every later experiment inherits that assumption.
    report = verify_feature_cache(
        dataset, backbone, output_dir,
        device=torch.device(device), sample_size=verify_sample, seed=0,
        batch_size=batch_size,  # replay at the build's batch size: same kernels, exact comparison
        **load_kwargs,
    )
    return output_dir / "manifest.json", report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="registered P1C experiment YAML")
    parser.add_argument("--split", choices=_SPLITS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="directory to write the cache into")
    parser.add_argument("--device", default="cpu", help="torch device string, e.g. cpu, cuda:0")
    parser.add_argument("--dtype", choices=FEATURE_DTYPES, default="float16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--verify-sample", type=int, default=256,
        help="rows to recompute against the live backbone after building (minimum 1; "
             "the build refuses to report success without this check)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path, report = build_cache_for_config(
        args.config,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verify_sample=args.verify_sample,
    )
    print(f"wrote {manifest_path}")
    print(
        f"verified {report['bit_identical_rows']}/{report['rows_sampled']} sampled rows "
        f"bit-identical against the live backbone (worst |delta| "
        f"{report['max_abs_delta']:.3e}, {report['max_ulps']} ULPs, "
        f"{report['max_significant_ulps']} at or above magnitude "
        f"{report['ulp_floor_magnitude']:g}; gate {report['max_ulps_allowed']} ULPs) "
        f"({report['rows_total']} rows total, {report['dtype']}, dim {report['feature_dim']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
