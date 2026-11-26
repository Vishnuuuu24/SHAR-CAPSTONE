from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml

from core.device import DEVICE_POLICIES
from core.p1c_readiness import (
    event_only_test_registration,
    validate_p1c_full_run_config,
    validate_p1c_screening_run_config,
)
from data.contracts import MANIFEST_FIELDS
from data.views import CANONICAL_UCF_LABELS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _P1CReadinessFixture:
    """Shared fixture (manifests, weights, `registered_config`) for the
    full-run and screening readiness test cases. Not itself a TestCase --
    subclasses mix this in with `unittest.TestCase` -- so that the screening
    suite reuses the fixture without unittest also re-running every full-run
    test method under the screening class name.
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "dataset").mkdir()
        self.weights = self.root / "weights.pt"
        self.weights.write_bytes(b"fixture weights")
        self.manifests: dict[str, Path] = {}
        for name, split, interval in (
            ("train", "train", ""),
            ("validation", "validation", ""),
            ("event_only_test", "test", "true"),
            ("noisy_proxy_test", "test", "true"),
        ):
            path = self.root / f"{name}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "filepath": f"frames/{name}.png",
                        "source_dataset": "fixture",
                        "source_video_id": f"{name}-video",
                        "source_frame_index": 0,
                        "label": "Abuse",
                        "label_scope": "temporal_interval" if split == "test" else "video_inherited",
                        "label_source": "fixture",
                        "annotation_version": "fixture-v1",
                        "inside_official_interval": interval,
                        "split": split,
                        "file_digest": "a" * 64,
                    }
                )
            self.manifests[name] = path

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def registered_config(self) -> dict:
        protocol = {
            "model": {"control": "gap_linear", "num_classes": 14, "class_names": list(CANONICAL_UCF_LABELS)},
            "backbone": {
                "family": "torchvision_resnet50",
                "weights": {"enum": "IMAGENET1K_V2", "path": "weights.pt", "sha256": sha256(self.weights)},
            },
            "preprocessing": {
                "resize_height": 64, "resize_width": 64, "interpolation": "bilinear", "color_mode": "RGB",
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            },
            "loss": {"name": "cross_entropy", "reduction": "mean", "class_balance": "none"},
            "augmentation": {"name": "none", "rationale": "fixture control; augmentation is a future one-change experiment"},
            "training": {
                "seeds": [0, 1, 2], "batch_size": 32,
                "loader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "drop_last": False, "verify_digests": True},
                "epochs": 20,
                "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.0001},
                "scheduler": {"name": "none"},
                "early_stopping": {"monitor": "macro_f1", "mode": "max", "patience": 5, "min_delta": 0.001},
                "selection_rule": "best_validation_macro_f1", "device_policy": "mps_strict", "precision": "float32",
                "throughput": {"channels_last": False, "compile": "none", "fused_optimizer": False},
            },
            "evaluation": {"primary_metric": "macro_f1", "video_level_aggregation": "majority_vote"},
            "data": {
                "dataset_roots": {"fixture": "dataset"},
                **{name: {"path": path.name, "sha256": sha256(path)} for name, path in self.manifests.items()},
            },
        }
        return {
            "experiment": {"id": "P1C-FIXTURE-001", "status": "REGISTERED_FULL_RUN_READY"},
            "fixed_protocol": protocol,
        }


class P1CReadinessTests(_P1CReadinessFixture, unittest.TestCase):
    """Combines the fixture with the original full-run test methods."""

    def test_registered_config_binds_real_artifacts(self) -> None:
        self.assertEqual(validate_p1c_full_run_config(self.registered_config(), base_dir=self.root), [])

    def test_every_registered_device_policy_is_accepted(self) -> None:
        for policy in DEVICE_POLICIES:
            with self.subTest(policy=policy):
                config = self.registered_config()
                config["fixed_protocol"]["training"]["device_policy"] = policy
                if policy == "cpu_strict_deterministic":
                    config["fixed_protocol"]["training"]["cpu_threads"] = 18
                self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_unregistered_device_policy_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["training"]["device_policy"] = "auto"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("device_policy must explicitly resolve" in error for error in errors))

    def test_cpu_policy_without_thread_count_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["training"]["device_policy"] = "cpu_strict_deterministic"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("positive training.cpu_threads" in error for error in errors))

    def test_pin_memory_is_rejected_on_a_backend_that_ignores_it(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["training"]["loader"]["pin_memory"] = True
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("pin_memory=true is not honoured" in error for error in errors))

    def test_pin_memory_is_accepted_on_a_backend_that_honours_it(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["training"]["device_policy"] = "cuda_strict"
        config["fixed_protocol"]["training"]["loader"]["pin_memory"] = True
        self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_missing_protocol_or_owner_placeholders_are_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["backbone"]["weights"] = "OWNER_SELECTION_REQUIRED"
        del config["fixed_protocol"]["training"]["scheduler"]
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("owner placeholder" in error for error in errors))
        self.assertTrue(any("training registration missing" in error for error in errors))

    def test_tampered_artifact_split_leakage_and_bool_batch_are_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["train"]["sha256"] = "b" * 64
        config["fixed_protocol"]["training"]["batch_size"] = True
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("data.train.sha256" in error for error in errors))
        self.assertIn("training.batch_size must be a positive integer", errors)

    def test_event_only_unavailability_declaration_is_accepted(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["event_only_test"] = {
            "unavailable": True,
            "reason": "Fixture reason string long enough to pass the forty-character floor.",
        }
        self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_event_only_unavailable_false_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["event_only_test"] = {
            "unavailable": False,
            "reason": "Fixture reason string long enough to pass the forty-character floor.",
        }
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("unavailable must be literally true" in error for error in errors))

    def test_event_only_short_reason_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["event_only_test"] = {"unavailable": True, "reason": "too short"}
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("reason must be a concrete explanation" in error for error in errors))

    def test_event_only_missing_reason_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["event_only_test"] = {"unavailable": True}
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("reason must be a concrete explanation" in error for error in errors))

    def test_event_only_unknown_sibling_keys_are_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["event_only_test"] = {
            "unavailable": True,
            "reason": "Fixture reason string long enough to pass the forty-character floor.",
            "surprise": "value",
        }
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("unknown keys" in error and "surprise" in error for error in errors))

    def test_event_only_path_and_unavailable_together_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["event_only_test"] = {
            "unavailable": True,
            "reason": "Fixture reason string long enough to pass the forty-character floor.",
            "path": self.manifests["event_only_test"].name,
        }
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("cannot register both a path and an unavailable declaration" in error for error in errors))

    def test_noisy_proxy_test_rejects_the_event_only_escape_hatch(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["data"]["noisy_proxy_test"] = {
            "unavailable": True,
            "reason": "Fixture reason string long enough to pass the forty-character floor.",
        }
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("data.noisy_proxy_test must register a relative path" in error for error in errors))

    def test_event_only_normal_registration_still_validates(self) -> None:
        # Belt-and-braces: the escape hatch must not have disturbed the
        # ordinary {path, sha256} shape that most configs still use.
        self.assertEqual(validate_p1c_full_run_config(self.registered_config(), base_dir=self.root), [])

    def test_event_only_test_registration_helper(self) -> None:
        path = self.manifests["event_only_test"]
        available = event_only_test_registration({"event_only_test": {"path": path.name, "sha256": sha256(path)}})
        self.assertEqual(available, {"available": True, "path": path.name, "sha256": sha256(path)})

        unavailable = event_only_test_registration(
            {"event_only_test": {"unavailable": True, "reason": "because reasons long enough to matter here"}}
        )
        self.assertEqual(
            unavailable,
            {"available": False, "reason": "because reasons long enough to matter here"},
        )

        with self.assertRaises(ValueError):
            event_only_test_registration({"event_only_test": "not a mapping"})

    def test_video_level_aggregation_majority_vote_is_accepted(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["evaluation"]["video_level_aggregation"] = "majority_vote"
        self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_video_level_aggregation_none_is_accepted(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["evaluation"]["video_level_aggregation"] = "none"
        self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_video_level_aggregation_unregistered_value_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["evaluation"]["video_level_aggregation"] = "mean_probability"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("evaluation.video_level_aggregation must be explicitly registered" in error for error in errors))

    def test_video_level_aggregation_absent_is_rejected(self) -> None:
        config = self.registered_config()
        del config["fixed_protocol"]["evaluation"]["video_level_aggregation"]
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("evaluation.video_level_aggregation must be explicitly registered" in error for error in errors))

    def test_evaluation_block_absent_is_rejected(self) -> None:
        config = self.registered_config()
        del config["fixed_protocol"]["evaluation"]
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertIn("evaluation registration is required", errors)

    def test_augmentation_absent_is_rejected(self) -> None:
        # K-22 finding 1: augmentation was registered in every config but read
        # by no code. Absent must still fail closed rather than being silently
        # ignored -- see data/augmentation.py for the now-implemented policy.
        config = self.registered_config()
        del config["fixed_protocol"]["augmentation"]
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("fixed_protocol.augmentation is invalid" in error for error in errors))

    def test_augmentation_unregistered_name_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["augmentation"] = {"name": "random_crop", "rationale": "not a registered name"}
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("fixed_protocol.augmentation is invalid" in error for error in errors))

    def test_augmentation_none_with_rationale_is_accepted(self) -> None:
        config = self.registered_config()
        self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_augmentation_standard_frame_is_accepted(self) -> None:
        # K-22 finding 1's fix: `standard_frame` is now implemented (see
        # data/augmentation.py) and readiness must accept a fully-registered
        # instance of it, not just the `none` no-op.
        config = self.registered_config()
        config["fixed_protocol"]["augmentation"] = {
            "name": "standard_frame",
            "rationale": "Attack the near-duplicate-frame memorisation this baseline showed at epoch 0-1.",
            "random_resized_crop": {"scale_min": 0.7, "scale_max": 1.0, "ratio_min": 0.75, "ratio_max": 1.333},
            "horizontal_flip_probability": 0.5,
            "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.0},
        }
        self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_throughput_absent_is_rejected_for_full_run_ready(self) -> None:
        # K-22 finding 2: an absent training.throughput silently selects the
        # disabled (measured 1.745x slower) policy instead of refusing, once
        # a config claims REGISTERED_FULL_RUN_READY.
        config = self.registered_config()
        del config["fixed_protocol"]["training"]["throughput"]
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("training.throughput must be explicitly registered" in error for error in errors))

    def test_throughput_absent_is_legal_when_not_full_run_ready(self) -> None:
        # Pre-existing calibration configs (run_class=calibration) never
        # register training.throughput and must keep working: the new
        # requirement is gated on REGISTERED_FULL_RUN_READY only.
        config = self.registered_config()
        del config["fixed_protocol"]["training"]["throughput"]
        config["experiment"]["status"] = "PLANNED"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertFalse(any("training.throughput must be explicitly registered" in error for error in errors))

    def test_primary_metric_inconsistent_with_selection_rule_is_rejected(self) -> None:
        # K-22 finding 4: nothing checked evaluation.primary_metric agreed
        # with the field that actually drives checkpoint selection.
        config = self.registered_config()
        config["fixed_protocol"]["evaluation"]["primary_metric"] = "accuracy"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("evaluation.primary_metric must be macro_f1" in error for error in errors))

    def test_primary_metric_matching_selection_rule_is_accepted(self) -> None:
        self.assertEqual(validate_p1c_full_run_config(self.registered_config(), base_dir=self.root), [])

    def test_planned_baseline_uses_the_canonical_nested_schema_and_fails_fast(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (repository / "configs/experiments/p1c/P1C-FRAME-GAP-BASELINE-001.yaml").read_text(encoding="utf-8")
        )
        errors = validate_p1c_full_run_config(config, base_dir=repository)
        self.assertIn("status must be REGISTERED_FULL_RUN_READY", errors)
        self.assertTrue(any(error.startswith("backbone.weights") for error in errors))
        self.assertFalse(any("class_names" in error for error in errors))

    # -- backbone.trainable ---------------------------------------------

    def test_backbone_trainable_absent_defaults_and_is_accepted(self) -> None:
        # Absent resolves to "full" -- the exact current behaviour for every
        # existing config, so the closed baseline (which never registers this
        # key) keeps validating with zero errors.
        self.assertNotIn("trainable", self.registered_config()["fixed_protocol"]["backbone"])
        self.assertEqual(validate_p1c_full_run_config(self.registered_config(), base_dir=self.root), [])

    def test_backbone_trainable_full_and_frozen_are_accepted(self) -> None:
        for value in ("full", "frozen"):
            with self.subTest(value=value):
                config = self.registered_config()
                config["fixed_protocol"]["backbone"]["trainable"] = value
                self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_backbone_trainable_unregistered_value_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["backbone"]["trainable"] = "partial"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("backbone.trainable must be one of" in error for error in errors))

    # -- backbone.family (B-2: widened to two registered families) ------

    def test_dinov2_family_with_its_own_enum_is_accepted(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["backbone"] = {
            "family": "timm_dinov2_vitb14_reg4",
            "weights": {"enum": "LVD142M_REG4", "path": "weights.pt", "sha256": sha256(self.weights)},
        }
        self.assertEqual(validate_p1c_full_run_config(config, base_dir=self.root), [])

    def test_unregistered_backbone_family_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["backbone"]["family"] = "some_other_vit"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("backbone.family must be one of" in error for error in errors))

    def test_dinov2_enum_on_a_resnet_config_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["backbone"]["weights"]["enum"] = "LVD142M_REG4"
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("backbone.weights.enum is not a registered" in error for error in errors))

    def test_resnet_enum_on_a_dinov2_config_is_rejected(self) -> None:
        config = self.registered_config()
        config["fixed_protocol"]["backbone"] = {
            "family": "timm_dinov2_vitb14_reg4",
            "weights": {"enum": "IMAGENET1K_V2", "path": "weights.pt", "sha256": sha256(self.weights)},
        }
        errors = validate_p1c_full_run_config(config, base_dir=self.root)
        self.assertTrue(any("backbone.weights.enum is not a registered" in error for error in errors))


class P1CScreeningReadinessTests(_P1CReadinessFixture, unittest.TestCase):
    """Screening-specific checks, built on the same fixtures as the full-run tests."""

    def screening_config(self) -> dict:
        config = self.registered_config()
        config["experiment"]["status"] = "REGISTERED_SCREENING_RUN_READY"
        config["experiment"]["run_class"] = "screening"
        config["experiment"]["screening"] = {
            "parent_id": "P1C-VIDEO-FRAME-BASELINE-011-AR-FOLD1",
            "decision_threshold": 0.32,
            "threshold_rationale": "Must clear the closed baseline's peak validation macro-F1 by 0.03.",
        }
        config["fixed_protocol"]["training"]["seeds"] = [7]
        del config["fixed_protocol"]["data"]["event_only_test"]
        del config["fixed_protocol"]["data"]["noisy_proxy_test"]
        return config

    def test_screening_config_is_accepted(self) -> None:
        self.assertEqual(validate_p1c_screening_run_config(self.screening_config(), base_dir=self.root), [])

    def test_full_run_config_is_rejected_by_the_screening_validator(self) -> None:
        errors = validate_p1c_screening_run_config(self.registered_config(), base_dir=self.root)
        self.assertTrue(any("status must be REGISTERED_SCREENING_RUN_READY" in error for error in errors))

    def test_screening_status_alone_is_rejected_by_the_full_run_validator(self) -> None:
        errors = validate_p1c_full_run_config(self.screening_config(), base_dir=self.root)
        self.assertTrue(any("status must be REGISTERED_FULL_RUN_READY" in error for error in errors))

    def test_seeds_must_be_a_single_integer(self) -> None:
        for bad_seeds in ([0, 1, 2], [], [True], ["0"], 0, None):
            with self.subTest(bad_seeds=bad_seeds):
                config = self.screening_config()
                config["fixed_protocol"]["training"]["seeds"] = bad_seeds
                errors = validate_p1c_screening_run_config(config, base_dir=self.root)
                self.assertTrue(any("exactly one integer seed" in error for error in errors))

    def test_run_class_must_be_screening(self) -> None:
        config = self.screening_config()
        config["experiment"]["run_class"] = "full"
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("experiment.run_class must be screening" in error for error in errors))

    def test_screening_block_missing_is_rejected(self) -> None:
        config = self.screening_config()
        del config["experiment"]["screening"]
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("experiment.screening is required" in error for error in errors))

    def test_screening_block_unknown_key_is_rejected(self) -> None:
        config = self.screening_config()
        config["experiment"]["screening"]["surprise"] = "value"
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("experiment.screening has unknown keys" in error and "surprise" in error for error in errors))

    def test_screening_block_missing_key_is_rejected(self) -> None:
        config = self.screening_config()
        del config["experiment"]["screening"]["parent_id"]
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("experiment.screening is missing required keys" in error and "parent_id" in error for error in errors))

    def test_screening_block_wrong_types_are_rejected(self) -> None:
        config = self.screening_config()
        config["experiment"]["screening"]["parent_id"] = ""
        config["experiment"]["screening"]["decision_threshold"] = 1
        config["experiment"]["screening"]["threshold_rationale"] = "too short"
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("screening.parent_id must be a non-empty string" in error for error in errors))
        self.assertTrue(any("screening.decision_threshold must be a float" in error for error in errors))
        self.assertTrue(any("screening.threshold_rationale must be a concrete explanation" in error for error in errors))

    def test_decision_threshold_rejects_bool(self) -> None:
        config = self.screening_config()
        config["experiment"]["screening"]["decision_threshold"] = True
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("screening.decision_threshold must be a float" in error for error in errors))

    def test_test_splits_may_be_omitted(self) -> None:
        self.assertEqual(validate_p1c_screening_run_config(self.screening_config(), base_dir=self.root), [])

    def test_test_splits_may_be_an_unavailability_declaration(self) -> None:
        config = self.screening_config()
        for field in ("event_only_test", "noisy_proxy_test"):
            config["fixed_protocol"]["data"][field] = {
                "unavailable": True,
                "reason": "Screening runs never evaluate Test; this is not a real absence claim.",
            }
        self.assertEqual(validate_p1c_screening_run_config(config, base_dir=self.root), [])

    def test_a_real_test_registration_is_forbidden_for_event_only_test(self) -> None:
        config = self.screening_config()
        config["fixed_protocol"]["data"]["event_only_test"] = {
            "path": self.manifests["event_only_test"].name,
            "sha256": sha256(self.manifests["event_only_test"]),
        }
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("data.event_only_test must be omitted or registered as an explicit" in error for error in errors))

    def test_a_real_test_registration_is_forbidden_for_noisy_proxy_test(self) -> None:
        config = self.screening_config()
        config["fixed_protocol"]["data"]["noisy_proxy_test"] = {
            "path": self.manifests["noisy_proxy_test"].name,
            "sha256": sha256(self.manifests["noisy_proxy_test"]),
        }
        errors = validate_p1c_screening_run_config(config, base_dir=self.root)
        self.assertTrue(any("data.noisy_proxy_test must be omitted or registered as an explicit" in error for error in errors))
