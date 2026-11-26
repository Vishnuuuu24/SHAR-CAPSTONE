"""Cache a frozen backbone's pooled features so training skips recomputing them.

## Why this module exists (B-1)

`FrozenBackbone` (`models/classifiers.py`) forces its wrapped backbone into
`eval()` no matter what mode the surrounding model requests, and its weights
never receive a gradient. Under D-42's confirmed configuration that means the
backbone emits a bit-identical 2048-D pre-GAP-pooled vector for a given frame
on every epoch -- verified directly from checkpoints: 23,455,040 backbone
weights, 53,173 BN running stats and 52,992 BN affine parameters were all
bit-identical across five epochs. P1C-VIDEO-FROZEN-CONFIRM-015 therefore spent
391,014 forward passes recomputing the same 391,014 vectors on every one of up
to 30 epochs. This module runs the backbone once per split, writes the pooled
vectors to disk, and gives every later head-design experiment (B-4, B-5, B-8,
B-9, B-10) a ~216s/epoch -> seconds/epoch speedup with the backbone contract
unchanged.

## Why the cache can go stale, and how this module refuses that

A cache is only valid for the exact backbone weights, preprocessing block,
frame source and manifest it was built from. Any of the following makes a
cached vector wrong for a new run, silently, unless refused:

- a different backbone weight checkpoint (even the same architecture);
- a different `trainable` mode -- a non-frozen backbone's output changes
  every epoch and was never eligible for caching in the first place;
- a different resize/interpolation/normalisation -- a different pixel tensor
  reaches the backbone;
- a different frame blob -- same manifest, different pixels;
- a different source manifest -- same blob, different rows/labels/order;
- ANY registered augmentation other than `none` -- an augmented input
  differs every epoch, so a cache built or served under `standard_frame`
  (or loaded to serve one) would either be wrong from the first epoch or
  would silently discard the augmentation it was supposed to apply.

This mirrors the fail-closed pattern already used by `core.precision`,
`core.throughput` and `data.augmentation`: `load_feature_cache` compares every
one of these fields against the manifest recorded at build time and raises
`FeatureCacheMismatch` on the first disagreement rather than degrading into
serving a wrong (but shaped-correctly) answer. A stale feature cache silently
answering for a different backbone is exactly the class of undetected-wrong-
result docs/PROJECT_RULES.md refuses.

## Storage layout

A cache lives in its own directory holding three files:

- ``features.<dtype>.npy`` -- an ``(row_count, feature_dim)`` array, in
  manifest row order.
- ``rows.csv`` -- one row per feature vector, in the same order, carrying
  ``source_dataset``, ``source_video_id``, ``source_frame_index``, ``target``
  and ``label`` -- the label provenance the rest of this project requires
  end to end (docs/PROJECT_RULES.md: "preserve source-video grouping and label
  provenance end to end").
- ``manifest.json`` -- the `FeatureCacheManifest` below, including the
  feature file's own SHA-256 so a cache that was truncated or edited after
  it was built is caught too, not just a cache built from the wrong inputs.

## dtype: float16 by default

`float16` halves the 3.2 GB fp32 footprint the backlog measured (391,014 x
2048 x 4 B) to 1.6 GB. This is safe here specifically because the cached
vector never sits on a gradient path -- it is a frozen backbone's *output*,
consumed only as a linear/MLP head's *input* -- and every run already trains
its forward pass under `amp_bfloat16` (`core/precision.py`), so a downstream
head has never seen full float32 backbone activations to begin with.
float16's ~3 significant decimal digits is far below the 0.02-0.06 macro-F1
seed-to-seed spread this project already treats as noise (see
P1C-VIDEO-FROZEN-CONFIRM-015). `float32` remains a registered opt-out for
anyone who wants to rule reduced precision out as a factor.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.p1c_readiness import sha256_file
from data.augmentation import AugmentationPolicy
from data.p1c_loader import FramePreprocessing

#: Registered on-disk feature dtypes. Not `int8`/quantised anything -- this is
#: a straight precision choice, not a lossy compression scheme.
FEATURE_DTYPES = ("float16", "float32")
_NUMPY_DTYPE = {"float16": np.float16, "float32": np.float32}

_FEATURES_FILENAME_TEMPLATE = "features.{dtype}.npy"
_ROWS_FILENAME = "rows.csv"
_MANIFEST_FILENAME = "manifest.json"
_ROW_FIELDS = ("row_index", "source_dataset", "source_video_id", "source_frame_index", "target", "label")


#: How a backbone's raw output is reduced to the vector this module caches,
#: keyed by `backbone.family`. Recorded in every manifest and checked on load,
#: because a head that consumes the pre-GAP spatial map (the SEMSCNN-inspired
#: 2-D multiscale-SE head in `docs/ARCHITECTURE.md`, backlog item B-4) must NOT
#: be served a pooled vector, and a cache built under one backbone's pooling
#: must never be silently served to the other (B-2). Root `docs/PROJECT_RULES.md` is
#: explicit that a spatial head must never be handed a reshaped GAP vector, so
#: serving either cache to the wrong consumer silently would produce exactly
#: the undetected-wrong-result this project rejects. A spatial cache is a
#: different artifact with a different pooling id.
FEATURE_CACHE_POOLING_BY_FAMILY = {
    "torchvision_resnet50": "global_average",
    "timm_dinov2_vitb14_reg4": "cls_plus_mean_patch",
    "transformers_videomae_base_k400": "videomae_mean_fcnorm",
    "transformers_siglip_base_p16_224": "pooler_plus_mean_patch",
}

#: Head controls whose forward pass begins by global-average-pooling the C5 map,
#: so that a cached pooled vector is byte-for-byte what they would have consumed
#: (`GlobalAverageLinearHead` and `SameC5MLPHead` in `models/classifiers.py`).
#: Any control absent from this set must build its own cache, not reuse this one.
POOLED_FEATURE_CONTROLS = frozenset({"gap_linear", "same_c5_mlp"})


class FeatureCacheMismatch(ValueError):
    """Raised when a cache's manifest does not match what a caller asked it to serve."""


def _pooling_for_family(backbone_family: str) -> str:
    """The registered pooling id for `backbone_family`, or raise if unregistered.

    Every backbone this module caches must own its pooling identity here --
    an unregistered family building or loading a cache would otherwise write
    (or accept) a cache under no declared pooling contract at all.
    """

    try:
        return FEATURE_CACHE_POOLING_BY_FAMILY[backbone_family]
    except KeyError:
        raise ValueError(
            f"unregistered backbone_family for feature caching: {backbone_family!r}; "
            f"expected one of {sorted(FEATURE_CACHE_POOLING_BY_FAMILY)}"
        ) from None


def _preprocessing_payload(preprocessing: FramePreprocessing) -> dict[str, Any]:
    """A JSON-stable dict for `preprocessing`, used identically at build and load time.

    Routed through a JSON round trip so both sides compare the same shapes --
    `dataclasses.asdict` keeps `mean`/`std` as tuples, but a manifest read back
    from disk always holds JSON lists; comparing a tuple to a list would flag
    every honest cache as a mismatch.
    """

    return json.loads(json.dumps(asdict(preprocessing)))


@dataclass(frozen=True)
class FeatureCacheManifest:
    """Sidecar record of everything a stale cache could get wrong."""

    backbone_family: str
    backbone_feature: str
    backbone_trainable: str
    backbone_weight_sha256: str
    model_control: str
    pooling: str
    preprocessing: dict[str, Any]
    augmentation_name: str
    augmentation_rationale: str
    frame_blob_sha256: str
    source_manifest_path: str
    source_manifest_sha256: str
    class_names: tuple[str, ...]
    split: str
    row_count: int
    feature_dim: int
    dtype: str
    cache_sha256: str
    created_at: str

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class_names"] = list(self.class_names)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FeatureCacheManifest":
        required = set(cls.__dataclass_fields__)
        keys = set(payload)
        if keys != required:
            raise ValueError(
                f"feature cache manifest.json keys must be exactly {sorted(required)}; "
                f"missing={sorted(required - keys)}; unexpected={sorted(keys - required)}"
            )
        return cls(
            backbone_family=str(payload["backbone_family"]),
            backbone_feature=str(payload["backbone_feature"]),
            backbone_trainable=str(payload["backbone_trainable"]),
            backbone_weight_sha256=str(payload["backbone_weight_sha256"]),
            model_control=str(payload["model_control"]),
            pooling=str(payload["pooling"]),
            preprocessing=dict(payload["preprocessing"]),
            augmentation_name=str(payload["augmentation_name"]),
            augmentation_rationale=str(payload["augmentation_rationale"]),
            frame_blob_sha256=str(payload["frame_blob_sha256"]),
            source_manifest_path=str(payload["source_manifest_path"]),
            source_manifest_sha256=str(payload["source_manifest_sha256"]),
            class_names=tuple(payload["class_names"]),
            split=str(payload["split"]),
            row_count=int(payload["row_count"]),
            feature_dim=int(payload["feature_dim"]),
            dtype=str(payload["dtype"]),
            cache_sha256=str(payload["cache_sha256"]),
            created_at=str(payload["created_at"]),
        )


def build_feature_cache(
    dataset: Dataset,
    backbone: nn.Module,
    *,
    output_dir: Path | str,
    device: torch.device | str,
    class_names: Sequence[str],
    split: str,
    backbone_family: str,
    backbone_feature: str,
    backbone_trainable: str,
    backbone_weight_sha256: str,
    model_control: str,
    preprocessing: FramePreprocessing,
    augmentation: AugmentationPolicy,
    frame_blob_sha256: str,
    source_manifest_path: Path | str,
    source_manifest_sha256: str,
    dtype: str = "float16",
    batch_size: int = 64,
    num_workers: int = 0,
) -> FeatureCacheManifest:
    """Run `backbone` once over `dataset`, in order, and write pooled features to `output_dir`.

    `dataset` must yield dicts shaped like `data.video_dataset.P1CVideoFrameDataset`
    (and the sibling P1C frame datasets): at minimum `image` and `target`, plus
    `source_video_id`, `source_frame_index` and `source_dataset` when available
    for provenance. `backbone` must implement `pooled_features(images) ->
    [B, D]` (e.g. `models.classifiers.ResNet50C5` or
    `models.dinov2_backbone.DINOv2ViTB14Reg`, optionally `FrozenBackbone`-
    wrapped) -- the backbone owns its own pooling (B-2), so this function
    caches exactly the vector that backbone's pooled-feature contract defines,
    whatever that is for its family.

    Refuses (raises `ValueError`) rather than silently building an invalid
    cache when augmentation is not `none` or the backbone is not registered
    `frozen` -- both make the cached vectors wrong from the very next epoch.
    """

    if dtype not in FEATURE_DTYPES:
        raise ValueError(f"dtype must be one of {FEATURE_DTYPES}; got {dtype!r}")
    pooling = _pooling_for_family(backbone_family)
    if augmentation.name != "none":
        raise ValueError(
            "feature cache requires augmentation.name == 'none': a cache built under any "
            f"other policy would answer differently every epoch. got {augmentation.name!r}"
        )
    if backbone_trainable != "frozen":
        raise ValueError(
            "feature cache requires backbone_trainable == 'frozen': a trainable backbone's "
            f"output changes every epoch and cannot be cached. got {backbone_trainable!r}"
        )
    if model_control not in POOLED_FEATURE_CONTROLS:
        raise ValueError(
            f"feature cache serves {pooling}-pooled vectors, which only "
            f"{sorted(POOLED_FEATURE_CONTROLS)} consume; got model.control={model_control!r}. A head that "
            "reads the pre-GAP spatial map needs its own spatial cache -- a pooled vector must never be "
            "reshaped back into one (see docs/PROJECT_RULES.md and docs/ARCHITECTURE.md)."
        )
    n_rows = len(dataset)  # type: ignore[arg-type]
    if n_rows == 0:
        raise ValueError("dataset is empty; refusing to build an empty cache")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = torch.device(device)
    backbone = backbone.to(resolved_device)
    backbone.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    numpy_dtype = _NUMPY_DTYPE[dtype]

    features: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(resolved_device)
            pooled = backbone.pooled_features(images)
            if pooled.ndim != 2:
                raise RuntimeError(
                    f"expected a pooled [B,D] backbone output; got shape {tuple(pooled.shape)}"
                )
            batch_n = pooled.shape[0]
            if features is None:
                features = np.empty((n_rows, pooled.shape[1]), dtype=numpy_dtype)
            elif pooled.shape[1] != features.shape[1]:
                raise RuntimeError("backbone feature dimension changed mid-run")
            features[offset : offset + batch_n] = pooled.to(torch.float32).cpu().numpy().astype(numpy_dtype)

            targets = batch["target"]
            video_ids = batch.get("source_video_id")
            frame_indices = batch.get("source_frame_index")
            source_datasets = batch.get("source_dataset")
            for i in range(batch_n):
                target = int(targets[i])
                rows.append(
                    {
                        "row_index": offset + i,
                        "source_dataset": str(source_datasets[i]) if source_datasets is not None else "",
                        "source_video_id": str(video_ids[i]) if video_ids is not None else "",
                        "source_frame_index": int(frame_indices[i]) if frame_indices is not None else -1,
                        "target": target,
                        "label": class_names[target],
                    }
                )
            offset += batch_n

    if features is None or offset != n_rows:
        raise RuntimeError(f"expected {n_rows} rows from the dataset, wrote {offset}")

    features_path = output_dir / _FEATURES_FILENAME_TEMPLATE.format(dtype=dtype)
    np.save(features_path, features)
    cache_sha256 = sha256_file(features_path)

    rows_path = output_dir / _ROWS_FILENAME
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = FeatureCacheManifest(
        backbone_family=backbone_family,
        backbone_feature=backbone_feature,
        backbone_trainable=backbone_trainable,
        backbone_weight_sha256=backbone_weight_sha256,
        model_control=model_control,
        pooling=pooling,
        preprocessing=_preprocessing_payload(preprocessing),
        augmentation_name=augmentation.name,
        augmentation_rationale=augmentation.rationale,
        frame_blob_sha256=frame_blob_sha256,
        source_manifest_path=str(source_manifest_path),
        source_manifest_sha256=source_manifest_sha256,
        class_names=tuple(class_names),
        split=split,
        row_count=n_rows,
        feature_dim=int(features.shape[1]),
        dtype=dtype,
        cache_sha256=cache_sha256,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / _MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_feature_cache(
    cache_dir: Path | str,
    *,
    class_names: Sequence[str],
    split: str,
    backbone_family: str,
    backbone_feature: str,
    backbone_trainable: str,
    backbone_weight_sha256: str,
    model_control: str,
    preprocessing: FramePreprocessing,
    augmentation: AugmentationPolicy,
    frame_blob_sha256: str,
    source_manifest_path: Path | str,
    source_manifest_sha256: str,
) -> tuple[np.ndarray, list[dict[str, Any]], FeatureCacheManifest]:
    """Load a cache built by `build_feature_cache`, refusing on any mismatch.

    Every keyword argument here is what the CALLER's current, registered
    config actually names; each is compared against what the cache's own
    manifest recorded when it was built, and the first mismatch refuses the
    whole load with `FeatureCacheMismatch` naming exactly what disagreed --
    never a partial or best-effort load. Returns `(features, rows, manifest)`
    where `rows[i]` is the label/provenance record for `features[i]`, in the
    same order the cache was built in.
    """

    if augmentation.name != "none":
        raise FeatureCacheMismatch(
            f"a feature cache may only be loaded under augmentation.name == 'none'; got {augmentation.name!r}"
        )

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no feature cache manifest at {manifest_path}")
    manifest = FeatureCacheManifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))

    mismatches: list[str] = []

    def _check(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            mismatches.append(f"{name}: cache has {actual!r}, config expects {expected!r}")

    _check("backbone_family", manifest.backbone_family, backbone_family)
    _check("backbone_feature", manifest.backbone_feature, backbone_feature)
    _check("backbone_trainable", manifest.backbone_trainable, backbone_trainable)
    _check("backbone_weight_sha256", manifest.backbone_weight_sha256, backbone_weight_sha256)
    _check("model_control", manifest.model_control, model_control)
    _check("pooling", manifest.pooling, _pooling_for_family(backbone_family))
    _check("preprocessing", manifest.preprocessing, _preprocessing_payload(preprocessing))
    _check("augmentation_name", manifest.augmentation_name, augmentation.name)
    _check("frame_blob_sha256", manifest.frame_blob_sha256, frame_blob_sha256)
    _check("source_manifest_path", manifest.source_manifest_path, str(source_manifest_path))
    _check("source_manifest_sha256", manifest.source_manifest_sha256, source_manifest_sha256)
    _check("class_names", list(manifest.class_names), list(class_names))
    _check("split", manifest.split, split)
    if mismatches:
        raise FeatureCacheMismatch(
            f"feature cache at {cache_dir} does not match the registered config:\n- " + "\n- ".join(mismatches)
        )

    features_path = cache_dir / _FEATURES_FILENAME_TEMPLATE.format(dtype=manifest.dtype)
    if not features_path.exists():
        raise FileNotFoundError(f"manifest names dtype {manifest.dtype!r} but {features_path} is missing")
    actual_sha256 = sha256_file(features_path)
    if actual_sha256 != manifest.cache_sha256:
        raise FeatureCacheMismatch(
            f"{features_path} sha256 {actual_sha256} does not match the manifest's "
            f"{manifest.cache_sha256}; the cache file was modified or corrupted after it was built"
        )
    features = np.load(features_path)
    if features.shape != (manifest.row_count, manifest.feature_dim):
        raise FeatureCacheMismatch(
            f"{features_path} has shape {features.shape}, manifest declares "
            f"({manifest.row_count}, {manifest.feature_dim})"
        )

    rows_path = cache_dir / _ROWS_FILENAME
    if not rows_path.exists():
        raise FileNotFoundError(f"feature cache manifest exists but {rows_path} is missing")
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != manifest.row_count:
        raise FeatureCacheMismatch(f"{rows_path} has {len(rows)} rows, manifest declares row_count={manifest.row_count}")

    return features, rows, manifest
