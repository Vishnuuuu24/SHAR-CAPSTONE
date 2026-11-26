"""Resolve a registered augmentation block into an executable, train-only transform.

Mirrors `core.throughput` and `core.precision`: a config names an augmentation,
this module turns that name into a concrete PIL-image transform, or refuses to
run. Nothing here turns augmentation on by itself -- an absent block is a
readiness error (see `core.p1c_readiness`), never a silent default.

## Why this module exists (K-22)

`fixed_protocol.augmentation` was registered in every P1C config (`name`,
`rationale`) but, before this module, was validated only as a fail-closed
no-op and read by no loader code: a config naming `random_crop` would have
been silently ignored while the run reported "augmentation made no
difference" having applied none. This module is the fix -- `name` is
validated AND actually read AND recorded in the run artifact via `.report()`.

## Registered names

`none`
    The existing no-op: a deterministic resize, byte-identical to what every
    P1C dataset already does. This stays the default so every config that
    predates this module keeps its exact current behaviour with no edits.

`standard_frame`
    Random-resized-crop + horizontal flip + colour jitter, with every
    parameter explicitly registered (see `resolve_augmentation_policy`) --
    no hidden defaults, matching this project's preprocessing contract
    (`data.p1c_loader.FramePreprocessing`).

## Train-only is structural, not a call-site convention

`P1CFrameDataset` and `P1CVideoFrameDataset` both refuse construction if an
`AugmentationPolicy` other than the implicit `NONE_AUGMENTATION` default is
paired with anything but `split="train"`. A validation or test dataset
therefore cannot be augmented even by an editing mistake at the call site --
the dataset itself refuses to accept the policy. See both classes'
`__init__` for the exact guard.

## Determinism (per-sample, not per-worker)

Every call to `apply` takes an explicit `seed`, and `augmentation_seed` below
derives it the same way `data/degradations.py` derives its per-sample seed:
as a pure function of (sample identity, the run's registered seed, the
current epoch, the exact registered policy). There is no dependency on
worker id, iteration order, or any mutable RNG state, so:

- the same run seed + epoch + sample always reproduces the same augmentation,
  regardless of `num_workers`;
- two different samples always get independent seeds, so `num_workers > 1`
  with `persistent_workers=True` cannot silently hand every worker the same
  augmented output -- the classic duplicated-per-worker-stream bug this
  project's DataLoader seeding (`data.p1c_loader.seed_dataloader_worker`)
  already had to guard against for shuffling, now sidestepped for
  augmentation by construction rather than by careful worker seeding.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image
from torchvision.transforms import functional as TF

from core.reproducibility import stable_sample_seed

AUGMENTATION_NAMES = ("none", "standard_frame")

_NONE_KEYS = frozenset({"name", "rationale"})
_STANDARD_FRAME_KEYS = frozenset(
    {"name", "rationale", "random_resized_crop", "horizontal_flip_probability", "color_jitter"}
)
_CROP_KEYS = frozenset({"scale_min", "scale_max", "ratio_min", "ratio_max"})
_JITTER_KEYS = frozenset({"brightness", "contrast", "saturation", "hue"})

#: `standard_frame` changes what a run actually trains on, unlike `none`'s
#: purely cosmetic rationale, so it is held to a real explanation rather than
#: a one-line placeholder.
_STANDARD_FRAME_MIN_RATIONALE_CHARS = 40

#: A random-resized-crop attempt is retried this many times before falling
#: back to a centred crop matching the image's own aspect ratio -- the same
#: retry budget torchvision's `RandomResizedCrop.get_params` uses.
_CROP_ATTEMPTS = 10


@dataclass(frozen=True)
class RandomResizedCropSpec:
    scale_min: float
    scale_max: float
    ratio_min: float
    ratio_max: float


@dataclass(frozen=True)
class ColorJitterSpec:
    brightness: float
    contrast: float
    saturation: float
    hue: float


@dataclass(frozen=True)
class AugmentationPolicy:
    """An executable augmentation decision, resolved from a registered block."""

    name: str
    rationale: str
    random_resized_crop: RandomResizedCropSpec | None = None
    horizontal_flip_probability: float | None = None
    color_jitter: ColorJitterSpec | None = None

    @property
    def is_none(self) -> bool:
        return self.name == "none"

    @property
    def transform_id(self) -> str:
        """A stable identity for this exact policy, folded into every seed.

        Two differently-parameterised `standard_frame` policies must not
        collide on the same seed, so every registered number is part of the
        identity -- not just the name.
        """

        if self.is_none:
            return "shar:p1c-augmentation-v1:none"
        payload = (
            f"{asdict(self.random_resized_crop)}"
            f":{self.horizontal_flip_probability}"
            f":{asdict(self.color_jitter)}"
        )
        return f"shar:p1c-augmentation-v1:standard_frame:{payload}"

    def apply(self, image: Image.Image, *, resize_width: int, resize_height: int, resample: int, seed: int) -> Image.Image:
        """Return the augmented image at the registered output size.

        `resample` is an already-resolved PIL resample constant (the caller
        looks it up from `FramePreprocessing.interpolation` via
        `data.p1c_loader._INTERPOLATION`) so this module never needs to know
        about that mapping or import the loader module that owns it.
        """

        if self.is_none:
            return image.resize((resize_width, resize_height), resample=resample)

        rng = random.Random(seed)
        width, height = image.size
        crop = self.random_resized_crop
        assert crop is not None  # only None on the `none` policy, handled above
        top, left, crop_height, crop_width = _sample_crop_box(
            rng, width=width, height=height,
            scale_min=crop.scale_min, scale_max=crop.scale_max,
            ratio_min=crop.ratio_min, ratio_max=crop.ratio_max,
        )
        image = image.crop((left, top, left + crop_width, top + crop_height))
        image = image.resize((resize_width, resize_height), resample=resample)

        assert self.horizontal_flip_probability is not None
        if rng.random() < self.horizontal_flip_probability:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        jitter = self.color_jitter
        assert jitter is not None
        # Fixed order (brightness, contrast, saturation, hue) rather than
        # torchvision ColorJitter's randomly permuted order: this module owns
        # its own semantics end to end, and a fixed order keeps `apply`
        # reproducible from `seed` alone without an extra permutation draw.
        if jitter.brightness > 0:
            factor = rng.uniform(max(0.0, 1.0 - jitter.brightness), 1.0 + jitter.brightness)
            image = TF.adjust_brightness(image, factor)
        if jitter.contrast > 0:
            factor = rng.uniform(max(0.0, 1.0 - jitter.contrast), 1.0 + jitter.contrast)
            image = TF.adjust_contrast(image, factor)
        if jitter.saturation > 0:
            factor = rng.uniform(max(0.0, 1.0 - jitter.saturation), 1.0 + jitter.saturation)
            image = TF.adjust_saturation(image, factor)
        if jitter.hue > 0:
            factor = rng.uniform(-jitter.hue, jitter.hue)
            image = TF.adjust_hue(image, factor)
        return image

    def report(self) -> dict[str, Any]:
        """Provenance for the run manifest -- what was actually applied."""

        if self.is_none:
            return {"name": "none", "rationale": self.rationale}
        return {
            "name": "standard_frame",
            "rationale": self.rationale,
            "random_resized_crop": asdict(self.random_resized_crop),
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "color_jitter": asdict(self.color_jitter),
        }


#: The implicit default every dataset uses when no policy is registered --
#: e.g. every validation/test dataset, and every existing caller that never
#: passes `augmentation=`. Behaviour is byte-identical to pre-augmentation
#: code: a plain resize, nothing else.
NONE_AUGMENTATION = AugmentationPolicy(
    name="none",
    rationale="internal default: no augmentation policy was registered for this dataset",
)


def _sample_crop_box(
    rng: random.Random, *, width: int, height: int, scale_min: float, scale_max: float, ratio_min: float, ratio_max: float
) -> tuple[int, int, int, int]:
    """Return `(top, left, crop_height, crop_width)`, torchvision's own algorithm.

    Reimplemented over a local `random.Random` instead of calling into
    `torchvision.transforms.RandomResizedCrop.get_params` (which draws from
    torch's *global* RNG): a local generator keeps this fully determined by
    `seed` with no side effect on any other RNG state, which matters because
    `apply` must be safe to call from any worker process in any order.
    """

    area = float(width * height)
    log_ratio = (math.log(ratio_min), math.log(ratio_max))
    for _ in range(_CROP_ATTEMPTS):
        target_area = area * rng.uniform(scale_min, scale_max)
        aspect_ratio = math.exp(rng.uniform(*log_ratio))
        crop_width = round(math.sqrt(target_area * aspect_ratio))
        crop_height = round(math.sqrt(target_area / aspect_ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = rng.randint(0, height - crop_height)
            left = rng.randint(0, width - crop_width)
            return top, left, crop_height, crop_width

    # Fallback: a centred crop matching the image's own aspect ratio, clamped
    # into the registered ratio range -- the same fallback torchvision uses
    # when ten attempts cannot satisfy both scale and ratio simultaneously.
    in_ratio = width / height
    if in_ratio < ratio_min:
        crop_width = width
        crop_height = round(crop_width / ratio_min)
    elif in_ratio > ratio_max:
        crop_height = height
        crop_width = round(crop_height * ratio_max)
    else:
        crop_width, crop_height = width, height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return top, left, crop_height, crop_width


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0,1]")
    return result


def resolve_augmentation_policy(spec: object) -> AugmentationPolicy:
    """Return the policy a registered `fixed_protocol.augmentation` block names, or refuse.

    Refuses rather than degrading: an unknown key, a missing key, an
    out-of-range value, or an unregistered `name` must never be silently
    reinterpreted as `none` or as "close enough".
    """

    if not isinstance(spec, dict):
        raise TypeError(f"fixed_protocol.augmentation must be a mapping; got {type(spec).__name__}")
    name = spec.get("name")
    if name not in AUGMENTATION_NAMES:
        raise ValueError(f"fixed_protocol.augmentation.name must be one of {sorted(AUGMENTATION_NAMES)}; got {name!r}")
    rationale = spec.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("fixed_protocol.augmentation.rationale must be a non-empty string")

    if name == "none":
        unknown = set(spec) - _NONE_KEYS
        if unknown:
            raise ValueError(f"augmentation name=none accepts only {sorted(_NONE_KEYS)}; unexpected keys: {sorted(unknown)}")
        return AugmentationPolicy(name="none", rationale=rationale)

    if len(rationale.strip()) < _STANDARD_FRAME_MIN_RATIONALE_CHARS:
        raise ValueError(
            "augmentation name=standard_frame requires a rationale of at least "
            f"{_STANDARD_FRAME_MIN_RATIONALE_CHARS} characters; this changes what the run trains "
            "on and is not a one-line placeholder"
        )
    unknown = set(spec) - _STANDARD_FRAME_KEYS
    missing = _STANDARD_FRAME_KEYS - set(spec)
    if unknown or missing:
        raise ValueError(
            f"augmentation name=standard_frame requires exactly {sorted(_STANDARD_FRAME_KEYS)}; "
            f"missing={sorted(missing)}; unexpected={sorted(unknown)}"
        )

    crop_raw = spec["random_resized_crop"]
    if not isinstance(crop_raw, dict) or set(crop_raw) != _CROP_KEYS:
        raise ValueError(f"augmentation.random_resized_crop must register exactly {sorted(_CROP_KEYS)}")
    scale_min = _finite_float(crop_raw["scale_min"], "random_resized_crop.scale_min")
    scale_max = _finite_float(crop_raw["scale_max"], "random_resized_crop.scale_max")
    ratio_min = _finite_float(crop_raw["ratio_min"], "random_resized_crop.ratio_min")
    ratio_max = _finite_float(crop_raw["ratio_max"], "random_resized_crop.ratio_max")
    if not (0.0 < scale_min <= scale_max <= 1.0):
        raise ValueError("random_resized_crop requires 0 < scale_min <= scale_max <= 1.0")
    if not (0.0 < ratio_min <= ratio_max):
        raise ValueError("random_resized_crop requires 0 < ratio_min <= ratio_max")

    flip_probability = _unit_interval(spec["horizontal_flip_probability"], "horizontal_flip_probability")

    jitter_raw = spec["color_jitter"]
    if not isinstance(jitter_raw, dict) or set(jitter_raw) != _JITTER_KEYS:
        raise ValueError(f"augmentation.color_jitter must register exactly {sorted(_JITTER_KEYS)}")
    brightness = _finite_float(jitter_raw["brightness"], "color_jitter.brightness")
    contrast = _finite_float(jitter_raw["contrast"], "color_jitter.contrast")
    saturation = _finite_float(jitter_raw["saturation"], "color_jitter.saturation")
    hue = _finite_float(jitter_raw["hue"], "color_jitter.hue")
    for value, field_name in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation)):
        if field_name < 0.0:
            raise ValueError(f"color_jitter.{value} must be non-negative")
    if not 0.0 <= hue <= 0.5:
        raise ValueError("color_jitter.hue must be within [0, 0.5]")

    return AugmentationPolicy(
        name="standard_frame",
        rationale=rationale,
        random_resized_crop=RandomResizedCropSpec(scale_min, scale_max, ratio_min, ratio_max),
        horizontal_flip_probability=flip_probability,
        color_jitter=ColorJitterSpec(brightness, contrast, saturation, hue),
    )


def augmentation_seed(relative_identity: str, file_digest: str, run_seed: int, epoch: int, policy: AugmentationPolicy) -> int:
    """Derive a reproducible per-sample, per-epoch augmentation seed.

    Follows `data/degradations.py`'s `SampleKey.seed_for` pattern exactly
    (rule of two: reuses `core.reproducibility.stable_sample_seed` rather
    than a second hashing scheme). `relative_identity` need not be a real
    file path -- callers whose manifest row does not uniquely identify one
    physical file per sample (e.g. many frames sharing one source video) must
    fold the per-frame discriminator into it themselves; see
    `data.video_dataset.P1CVideoFrameDataset` for why that matters there.
    """

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    return stable_sample_seed(relative_identity, file_digest, run_seed, f"{policy.transform_id}:epoch:{epoch}")
