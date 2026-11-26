"""Screening-run-specific validation helpers used by `core.p1c_readiness`.

Split out of that module by responsibility (docs/PROJECT_RULES.md's ~400-line soft
ceiling): everything here concerns the two things unique to a screening run
-- the `{unavailable: true, reason}` escape hatch a split can carry instead of
a real registration, and the `experiment.screening` block itself -- while
`core.p1c_readiness` keeps the shared full-protocol validation both run
classes go through unchanged.
"""

from __future__ import annotations

from typing import Any

#: Splits a screening run is permitted to touch. Screening exists to find a
#: config cheaply, not to produce a citable Test number -- see
#: `forbid_screening_test_access`, which enforces that the two Test splits
#: below are either absent or an explicit unavailability declaration.
SCREENING_DATA_SPLITS = ("train", "validation")
_SCREENING_FORBIDDEN_TEST_SPLITS = ("event_only_test", "noisy_proxy_test")

#: Keys `experiment.screening` must carry, exactly, for a screening run.
_SCREENING_BLOCK_KEYS = frozenset({"parent_id", "decision_threshold", "threshold_rationale"})
_SCREENING_RATIONALE_MIN_LENGTH = 40

#: Keys an unavailability declaration may carry, beyond the ``path`` key it
#: must NOT carry alongside ``unavailable`` (that combination gets its own,
#: more specific error rather than being folded into "unknown").
_UNAVAILABILITY_KEYS = {"unavailable", "reason"}
_UNAVAILABILITY_MIN_REASON_LENGTH = 40


def validate_split_unavailability(field: str, raw: dict[str, Any], errors: list[str]) -> None:
    """Validate an explicit ``{unavailable: true, reason: ...}`` declaration
    for the named data split, and ONLY that shape -- shared by the one narrow
    escape hatch a full run gets from REQUIRED_DATA_SPLITS (``event_only_test``,
    see its call site in `core.p1c_readiness`) and the wider one a screening
    run gets from both Test splits (`forbid_screening_test_access`).

    docs/PROJECT_RULES.md rule 1 / D-03 make the official-interval-derived event-only view
    the mandated headline evaluation for a full run. But some registered folds
    genuinely lack one -- D-40 measured that Action-Recognition fold 1 has
    official temporal intervals for only 48 of its 168 test videos, and three
    classes have zero, because the official intervals cover the
    Anomaly-Detection test split, not the AR folds. Pointing event_only_test at
    the inherited-label file instead would make the run manifest assert an
    evaluation the run never performed; dropping the key silently would let a
    future config lose rule 1's headline by accident and nothing would notice.
    This escape hatch is deliberately narrow: it records an honest absence, not
    a substitute for the real thing, and the `reason` string is what makes that
    absence auditable rather than a bare admission. It does not generalise to
    train/validation for a full run -- those splits are never optional there.
    """

    if "path" in raw:
        errors.append(f"{field} cannot register both a path and an unavailable declaration")
    unknown = sorted(set(raw) - _UNAVAILABILITY_KEYS - {"path"})
    if unknown:
        errors.append(f"{field} unavailability declaration has unknown keys: {unknown}")
    if raw.get("unavailable") is not True:
        errors.append(
            f"{field}.unavailable must be literally true; "
            "a config that has this view must register it normally as {path, sha256}"
        )
    reason = raw.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < _UNAVAILABILITY_MIN_REASON_LENGTH:
        errors.append(
            f"{field}.reason must be a concrete explanation of at least "
            f"{_UNAVAILABILITY_MIN_REASON_LENGTH} characters, not a placeholder word"
        )


def forbid_screening_test_access(data: dict[str, Any], errors: list[str]) -> None:
    """Structurally block a screening run from registering a real Test split.

    A screening run trades three seeds for one to search cheaply; it must not
    also become a Test-evaluation event by accident. Each of `event_only_test`
    / `noisy_proxy_test` must be omitted entirely, or registered as the same
    `{unavailable: true, reason}` declaration `validate_split_unavailability`
    validates for a full run's `event_only_test` -- never a real
    `{path, sha256}` registration.
    """

    for field in _SCREENING_FORBIDDEN_TEST_SPLITS:
        raw_split = data.get(field)
        if raw_split is None:
            continue
        if isinstance(raw_split, dict) and "unavailable" in raw_split:
            validate_split_unavailability(f"data.{field}", raw_split, errors)
            continue
        errors.append(
            f"data.{field} must be omitted or registered as an explicit {{unavailable: true, reason}} "
            "declaration for a screening run; a screening run may not evaluate Test"
        )


def validate_screening_block(experiment: dict[str, Any], errors: list[str]) -> None:
    """Validate `experiment.screening`, required only for a screening run.

    Mirrors `core.throughput.resolve_throughput_policy`'s refusal style:
    unknown keys, missing keys, and wrong types are all refused rather than
    partially accepted.
    """

    screening = experiment.get("screening") if isinstance(experiment, dict) else None
    if not isinstance(screening, dict):
        errors.append("experiment.screening is required for a screening run and must be a mapping")
        return
    unknown = sorted(set(screening) - _SCREENING_BLOCK_KEYS)
    if unknown:
        errors.append(f"experiment.screening has unknown keys: {unknown}")
    missing = sorted(_SCREENING_BLOCK_KEYS - set(screening))
    if missing:
        errors.append(f"experiment.screening is missing required keys: {missing}")
    if "parent_id" in screening and (not isinstance(screening["parent_id"], str) or not screening["parent_id"]):
        errors.append("experiment.screening.parent_id must be a non-empty string")
    if "decision_threshold" in screening:
        threshold = screening["decision_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, float):
            errors.append("experiment.screening.decision_threshold must be a float")
    if "threshold_rationale" in screening:
        rationale = screening["threshold_rationale"]
        if not isinstance(rationale, str) or len(rationale.strip()) < _SCREENING_RATIONALE_MIN_LENGTH:
            errors.append(
                "experiment.screening.threshold_rationale must be a concrete explanation of at least "
                f"{_SCREENING_RATIONALE_MIN_LENGTH} characters, not a placeholder word"
            )
