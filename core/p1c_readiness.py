"""Fail-closed validation for a registered P1C frame-baseline configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.device import DEVICE_POLICIES, policy_backend, supports_pin_memory
from data.augmentation import resolve_augmentation_policy
from data.manifest import iter_manifest
from data.p1c_loader import FramePreprocessing
from core.precision import PRECISIONS
from core.throughput import resolve_throughput_policy
from eval.video_level import VIDEO_LEVEL_AGGREGATION_SETTINGS
from models.classifiers import BACKBONE_TRAINABLE_MODES
from core.p1c_screening import (
    SCREENING_DATA_SPLITS,
    forbid_screening_test_access,
    validate_screening_block,
    validate_split_unavailability,
)


#: Registered policy -> whether ``pin_memory=True`` is honoured on its backend.
#: Every policy name is ``<backend>_<qualifier>``, so the backend is its prefix.
_POLICY_SUPPORTS_PIN_MEMORY = {
    policy: supports_pin_memory(policy.split("_")[0]) for policy in DEVICE_POLICIES
}

PLACEHOLDER_MARKERS = ("OWNER_", "owner_required")
SHA256_HEX = set("0123456789abcdef")

#: Registered `fixed_protocol.backbone.family` values (B-2 widened this from a
#: single hardcoded ResNet50 family), each with its own registered set of
#: weight enums so a config cannot register one family's identifier under the
#: other -- a ResNet50 config claiming `LVD142M_REG4`, or a DINOv2 config
#: claiming `IMAGENET1K_V2`, describes weights it did not actually load.
CANONICAL_WEIGHT_ENUMS_BY_FAMILY = {
    "torchvision_resnet50": {"IMAGENET1K_V1", "IMAGENET1K_V2"},
    "timm_dinov2_vitb14_reg4": {"LVD142M_REG4"},
    "transformers_videomae_base_k400": {"KINETICS400_FINETUNED"},
    "transformers_vjepa2_vitl": {"VJEPA2_VITL_FPC64_256_ENCODER"},
    "transformers_siglip_base_p16_224": {"WEBLI_SIGLIP_BASE_P16_224_VISION"},
}
BACKBONE_FAMILIES = tuple(CANONICAL_WEIGHT_ENUMS_BY_FAMILY)
REQUIRED_TRAINING_FIELDS = {
    "seeds", "batch_size", "loader", "epochs", "optimizer", "scheduler", "early_stopping",
    "selection_rule", "device_policy", "precision",
}
REQUIRED_DATA_SPLITS = ("train", "validation", "event_only_test", "noisy_proxy_test")

#: Registered `experiment.status` a run class must carry to close its launch gate.
_RUN_CLASS_STATUS = {
    "full": "REGISTERED_FULL_RUN_READY",
    "screening": "REGISTERED_SCREENING_RUN_READY",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_HEX


def _placeholder_paths(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, str) and any(marker.lower() in value.lower() for marker in PLACEHOLDER_MARKERS):
        return [prefix]
    if isinstance(value, dict):
        return [path for key, item in value.items() for path in _placeholder_paths(item, f"{prefix}.{key}")]
    if isinstance(value, list):
        return [path for index, item in enumerate(value) for path in _placeholder_paths(item, f"{prefix}[{index}]")]
    return []


def _resolve_registered_file(raw: object, base_dir: Path, field: str, errors: list[str]) -> Path | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("path"), str) or not _is_sha256(raw.get("sha256")):
        errors.append(f"{field} must register a relative path and lowercase SHA-256 digest")
        return None
    candidate = (base_dir / raw["path"]).resolve()
    try:
        candidate.relative_to(base_dir.resolve())
    except ValueError:
        errors.append(f"{field}.path escapes the registered project root")
        return None
    if not candidate.is_file():
        errors.append(f"{field}.path does not exist: {raw['path']}")
        return None
    if sha256_file(candidate) != raw["sha256"]:
        errors.append(f"{field}.sha256 does not match its registered file")
        return None
    return candidate


def event_only_test_registration(data: dict[str, Any]) -> dict[str, Any]:
    """Parse `data.event_only_test` into the shape a run manifest records.

    Shared by the readiness validator's own check and the frame-baseline
    runner (rule of two), so the two places that need to understand this
    shape do not grow two independent parsers of it.  Callers should only
    rely on this after `validate_p1c_full_run_config` returned no errors;
    it raises on anything that validator would already have rejected.
    """

    raw = data.get("event_only_test") if isinstance(data, dict) else None
    if isinstance(raw, dict) and "unavailable" in raw:
        return {"available": False, "reason": raw.get("reason")}
    if isinstance(raw, dict) and isinstance(raw.get("path"), str) and _is_sha256(raw.get("sha256")):
        return {"available": True, "path": raw["path"], "sha256": raw["sha256"]}
    raise ValueError("data.event_only_test is not a registered {path, sha256} or a validated unavailability declaration")


def _positive_int(value: object, field: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field} must be a positive integer")


def _protocol_from_experiment_config(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Normalize the canonical experiment wrapper for the strict launch gate.

    P1C experiment files separate immutable experiment metadata from the
    frozen protocol.  Readiness validates the latter while binding the status
    from the former; it intentionally does not accept a silently flattened
    hybrid that would let a runner and log disagree about the same run.
    """

    experiment = config.get("experiment")
    protocol = config.get("fixed_protocol")
    if not isinstance(experiment, dict) or not isinstance(protocol, dict):
        errors.append("config must contain experiment metadata and fixed_protocol")
        return {}
    if not isinstance(experiment.get("id"), str) or not experiment["id"]:
        errors.append("experiment.id must be a non-empty string")
    if not isinstance(experiment.get("status"), str):
        errors.append("experiment.status must be an explicit string")
    normalized = dict(protocol)
    normalized["status"] = experiment.get("status")
    return normalized


def _validate_p1c_protocol(config: dict[str, Any], *, base_dir: Path, run_class: str) -> list[str]:
    """Shared launch-gate body for both the full-run and screening entry points.

    `run_class` selects exactly two things: which `experiment.status` closes
    the gate, and how `training.seeds` / Test access are checked (see the two
    call sites below). Every other check -- digests, leakage, throughput,
    augmentation, class_balance, primary_metric, video_level_aggregation -- is
    identical for both, because a screening run is a cheap comparison against
    the same protocol, not a license to relax it.
    """

    errors: list[str] = []
    if not isinstance(base_dir, Path) or not base_dir.is_dir():
        return ["base_dir must be an existing project directory"]
    if not isinstance(config, dict):
        return ["config must be a mapping"]
    protocol = _protocol_from_experiment_config(config, errors)
    expected_status = _RUN_CLASS_STATUS[run_class]
    status_ready = protocol.get("status") == expected_status
    if not status_ready:
        errors.append(f"status must be {expected_status}")
    for path in _placeholder_paths(protocol):
        errors.append(f"owner placeholder remains at {path}")
    model = protocol.get("model")
    if not isinstance(model, dict) or model.get("control") != "gap_linear":
        errors.append("model.control must be the registered frame baseline gap_linear")
    elif model.get("num_classes") != 14:
        errors.append("model.num_classes must be the registered 14-label output dimension")
    backbone = protocol.get("backbone")
    weights_file: Path | None = None
    if not isinstance(backbone, dict) or backbone.get("family") not in BACKBONE_FAMILIES:
        errors.append(f"backbone.family must be one of {sorted(BACKBONE_FAMILIES)}")
    elif not isinstance(backbone.get("weights"), dict):
        errors.append("backbone.weights must register exact enum/path/SHA-256")
    else:
        weights = backbone["weights"]
        family = backbone["family"]
        if weights.get("enum") not in CANONICAL_WEIGHT_ENUMS_BY_FAMILY[family]:
            errors.append(
                f"backbone.weights.enum is not a registered {family} enum "
                f"(one of {sorted(CANONICAL_WEIGHT_ENUMS_BY_FAMILY[family])})"
            )
        weights_file = _resolve_registered_file(weights, base_dir, "backbone.weights", errors)
    # `trainable` is optional and absent resolves to "full" -- the exact
    # current behaviour for every existing config, the same contract an
    # absent `training.throughput` already carries. Only a *present* value
    # outside the registered two is refused.
    if isinstance(backbone, dict) and "trainable" in backbone and backbone["trainable"] not in BACKBONE_TRAINABLE_MODES:
        errors.append(
            f"backbone.trainable must be one of {sorted(BACKBONE_TRAINABLE_MODES)}; got {backbone['trainable']!r}"
        )

    class_names = model.get("class_names") if isinstance(model, dict) else None
    if not isinstance(class_names, list) or len(class_names) != 14 or len(set(class_names)) != 14:
        errors.append("model.class_names must be the explicit 14-label ordered task mapping")
    preprocessing = protocol.get("preprocessing")
    if not isinstance(preprocessing, dict):
        errors.append("explicit preprocessing registration is required")
    else:
        try:
            FramePreprocessing.from_config(preprocessing)
        except (TypeError, ValueError) as exc:
            errors.append(f"preprocessing is invalid: {exc}")
    loss = protocol.get("loss")
    if not isinstance(loss, dict) or loss.get("name") != "cross_entropy" or loss.get("reduction") != "mean":
        errors.append("loss must explicitly register mean-reduced cross_entropy")
    # Deliberately pinned, not merely defaulted: the runner hardcodes
    # nn.CrossEntropyLoss with no `weight=`, so a config that registered any
    # value other than "none" here would describe a weighting the run never
    # applies. Implementing class-balance weighting requires changing this
    # check AND scripts/run_p1c_frame_baseline.py's CrossEntropyLoss
    # construction together -- loosening only this validator would let a
    # config claim a weighting that silently still does nothing (owner
    # decision 2026-08-27: the first research run is the unweighted control).
    if not isinstance(loss, dict) or loss.get("class_balance") != "none":
        errors.append("baseline loss must register class_balance=none; imbalance is a later one-change comparison")

    # `augmentation` used to be registered (name + rationale) but validated only
    # as a fail-closed no-op, and read by no loader code: a config registering
    # e.g. `augmentation: {name: random_crop}` was silently ignored, and the run
    # would report "augmentation made no difference" having never applied any --
    # the same registered-but-unread-field failure as the pixel-source guard in
    # data/pixel_source.py (K-21). `data.augmentation.resolve_augmentation_policy`
    # is the fix: `name=none` still validates exactly as before (every existing
    # config keeps passing), and `name=standard_frame` is now both validated
    # here AND actually applied by the P1C datasets AND recorded in the run
    # report -- see data/augmentation.py's module docstring.
    try:
        resolve_augmentation_policy(protocol.get("augmentation"))
    except (TypeError, ValueError) as exc:
        errors.append(f"fixed_protocol.augmentation is invalid: {exc}")

    training = protocol.get("training")
    if not isinstance(training, dict):
        errors.append("training registration is required")
    else:
        missing = sorted(REQUIRED_TRAINING_FIELDS - set(training))
        if missing:
            errors.append(f"training registration missing fields: {missing}")
        if run_class == "full":
            if training.get("seeds") != [0, 1, 2]:
                errors.append("headline P1C seeds must be exactly [0, 1, 2]")
        else:
            seeds = training.get("seeds")
            if (
                not isinstance(seeds, list)
                or len(seeds) != 1
                or isinstance(seeds[0], bool)
                or not isinstance(seeds[0], int)
            ):
                errors.append("screening P1C training.seeds must be a list containing exactly one integer seed")
        _positive_int(training.get("batch_size"), "training.batch_size", errors)
        _positive_int(training.get("epochs"), "training.epochs", errors)
        loader = training.get("loader")
        if not isinstance(loader, dict):
            errors.append("training.loader must register worker, memory, drop, and digest behavior")
        else:
            expected_loader_fields = {"num_workers", "pin_memory", "persistent_workers", "drop_last", "verify_digests"}
            if set(loader) != expected_loader_fields:
                errors.append(f"training.loader keys must be exactly {sorted(expected_loader_fields)}")
            if loader.get("verify_digests") is not True:
                errors.append("training.loader must require full manifest file-digest verification")
            if isinstance(loader.get("num_workers"), bool) or not isinstance(loader.get("num_workers"), int) or loader["num_workers"] < 0:
                errors.append("training.loader.num_workers must be a non-negative integer")
            for name in ("pin_memory", "persistent_workers", "drop_last"):
                if not isinstance(loader.get(name), bool):
                    errors.append(f"training.loader.{name} must be bool")
            if loader.get("persistent_workers") is True and loader.get("num_workers") == 0:
                errors.append("training.loader.persistent_workers requires num_workers greater than zero")
        if not isinstance(training.get("optimizer"), dict) or not isinstance(training["optimizer"].get("name"), str):
            errors.append("training.optimizer must name exact optimizer and parameters")
        if not isinstance(training.get("scheduler"), dict) or not isinstance(training["scheduler"].get("name"), str):
            errors.append("training.scheduler must name an exact scheduler or explicit none")
        stopping = training.get("early_stopping")
        if not isinstance(stopping, dict) or stopping.get("monitor") != "macro_f1" or stopping.get("mode") != "max":
            errors.append("training.early_stopping must register validation macro_f1/max with parameters")
        if training.get("selection_rule") != "best_validation_macro_f1":
            errors.append("training.selection_rule must be best_validation_macro_f1")
        policy = training.get("device_policy")
        if policy not in DEVICE_POLICIES:
            errors.append(
                "training.device_policy must explicitly resolve device/determinism behavior; "
                f"expected one of {sorted(DEVICE_POLICIES)}"
            )
        elif isinstance(loader, dict) and loader.get("pin_memory") is True and not _POLICY_SUPPORTS_PIN_MEMORY[policy]:
            # A registration that asks for pinned staging buffers on a backend
            # that ignores them records an expectation the run cannot meet.
            errors.append(
                f"training.loader.pin_memory=true is not honoured under device_policy={policy}; "
                "register false or name a backend that supports pinned memory"
            )
        if policy == "cpu_strict_deterministic":
            threads = training.get("cpu_threads")
            if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
                errors.append("cpu_strict_deterministic requires a positive training.cpu_threads")
        if training.get("precision") not in set(PRECISIONS):
            errors.append(
                f"training.precision must be explicitly registered as one of {sorted(PRECISIONS)}"
            )
        # training.throughput itself stays optional here; resolve_throughput_policy
        # is the one place its keys/types/rules live, so this only forwards its
        # refusal rather than re-checking them. But `resolve_throughput_policy(None, ...)`
        # legitimately returns the disabled policy -- that default exists so
        # pre-existing calibration configs keep working -- which means an absent
        # block is legal right up until a config claims REGISTERED_FULL_RUN_READY.
        # Past that point, an omitted throughput block is the pixel-source
        # failure again: one missing key silently selects a materially slower
        # path (measured 1.745x) with nothing refusing or warning. So a full
        # run must say something explicit, even if that something is "off".
        if status_ready and training.get("throughput") is None:
            errors.append(
                "training.throughput must be explicitly registered for a REGISTERED_FULL_RUN_READY "
                "config; an absent block silently selects the disabled throughput policy (measured "
                "1.745x slower) instead of refusing"
            )
        # Backend comes from core.device rather than from splitting the policy
        # name, so a future policy whose name does not start with its backend
        # cannot silently validate throughput against the wrong device.
        throughput_backend = policy_backend(policy) if policy in DEVICE_POLICIES else "cpu"
        try:
            resolve_throughput_policy(training.get("throughput"), throughput_backend)
        except (TypeError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))

    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append("evaluation registration is required")
    else:
        if evaluation.get("video_level_aggregation") not in VIDEO_LEVEL_AGGREGATION_SETTINGS:
            # Explicit like every other field here (docs/PROJECT_RULES.md): a config that never
            # says whether it reports the imbalance-immune per-video score must not
            # silently default to skipping it, or to running it.
            errors.append(
                "evaluation.video_level_aggregation must be explicitly registered as one of "
                f"{sorted(VIDEO_LEVEL_AGGREGATION_SETTINGS)}"
            )
        # training.selection_rule is what actually drives checkpoint selection
        # (FrameClassifierEngine never reads evaluation.primary_metric); both
        # currently say macro-F1, but nothing before this checked they agree,
        # so a future edit to one could drift from the other and a config
        # would claim it optimizes one metric while selecting on another with
        # nothing to notice. Refuse the inconsistent combination rather than
        # letting it stand.
        selection_rule = training.get("selection_rule") if isinstance(training, dict) else None
        if selection_rule == "best_validation_macro_f1" and evaluation.get("primary_metric") != "macro_f1":
            errors.append(
                "evaluation.primary_metric must be macro_f1 when training.selection_rule is "
                "best_validation_macro_f1; a config must not claim one metric while selecting on another"
            )

    if run_class == "screening":
        experiment_raw = config.get("experiment") if isinstance(config.get("experiment"), dict) else {}
        if experiment_raw.get("run_class") != "screening":
            errors.append("experiment.run_class must be screening for a REGISTERED_SCREENING_RUN_READY config")
        validate_screening_block(experiment_raw, errors)

    data = protocol.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("dataset_roots"), dict):
        errors.append("data.dataset_roots and immutable manifest registrations are required")
    else:
        if run_class == "screening":
            # A screening run is train/validation only -- see
            # `core.p1c_screening.forbid_screening_test_access`. This is a structural check on
            # the raw config and does not need files to exist, so it runs
            # unconditionally rather than gated on status_ready below.
            forbid_screening_test_access(data, errors)
            required_splits = SCREENING_DATA_SPLITS
        else:
            required_splits = REQUIRED_DATA_SPLITS
        # A planned configuration often has a deliberately absent local
        # weight cache. Do not read manifests until the status and exact
        # weight identity close; the eventual ready check streams them.
        if status_ready and weights_file is not None:
            roots = {name: (base_dir / value) for name, value in data["dataset_roots"].items() if isinstance(value, str)}
            train_groups: set[tuple[str, str]] = set()
            validation_groups: set[tuple[str, str]] = set()
            for split in required_splits:
                raw_split = data.get(split)
                if split == "event_only_test" and isinstance(raw_split, dict) and "unavailable" in raw_split:
                    # The declared-unavailable escape hatch: no file to resolve
                    # or manifest to walk, so validate its own shape and move on.
                    validate_split_unavailability(f"data.{split}", raw_split, errors)
                    continue
                path = _resolve_registered_file(raw_split, base_dir, f"data.{split}", errors)
                if path is None:
                    continue
                try:
                    for row in iter_manifest(path, roots, verify_files=False, verify_digests=False):
                        if row.split != ("test" if split in {"event_only_test", "noisy_proxy_test"} else split):
                            errors.append(f"data.{split} must retain its exact registered split")
                            break
                        if split == "train":
                            train_groups.add((row.source_dataset, row.source_video_id))
                        elif split == "validation":
                            validation_groups.add((row.source_dataset, row.source_video_id))
                        elif split == "event_only_test" and row.label != "Normal" and row.inside_official_interval is not True:
                            errors.append("event_only_test contains anomalous rows outside official intervals")
                            break
                except (OSError, ValueError) as exc:
                    errors.append(f"data.{split} is not a valid manifest: {exc}")
            if train_groups & validation_groups:
                errors.append("Train/validation manifests have source-video leakage")
    return errors


def validate_p1c_full_run_config(config: dict[str, Any], *, base_dir: Path) -> list[str]:
    """Return launch-gate errors; an empty list means artifacts and policy bind.

    It validates only the frame-baseline launch shape. The temporal row needs a
    separately registered clip configuration and B-006 decision.
    """

    return _validate_p1c_protocol(config, base_dir=base_dir, run_class="full")


def validate_p1c_screening_run_config(config: dict[str, Any], *, base_dir: Path) -> list[str]:
    """Return launch-gate errors for a single-seed screening run.

    A screening run exists to search cheaply for a better config before
    spending three seeds confirming it (the first research run's validation
    macro-F1 peaked at epoch 0-1 and declined -- see docs/TRAINING_LOG.md).
    It trades the three-seed headline commitment for exactly one seed plus a
    required `experiment.screening` block naming the `parent_id` it is
    compared against and a `decision_threshold` with its `threshold_rationale`
    (see `core.p1c_screening.validate_screening_block`), and it is structurally forbidden from
    touching either Test split (see `core.p1c_screening.forbid_screening_test_access`) so it
    cannot be mistaken for a Test-evaluation event. Every other check is
    identical to `validate_p1c_full_run_config`; see `_validate_p1c_protocol`.
    """

    return _validate_p1c_protocol(config, base_dir=base_dir, run_class="screening")
