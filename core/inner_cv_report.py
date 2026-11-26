"""Scoring, paired contrasts and output files shared by the `core.inner_cv` runners (rule of two)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core import inner_cv
from core.failure_analysis import load_arm_predictions, paired_cluster_bootstrap

MIN_GAIN = 0.02
ALPHA_ONE_SIDED = 0.025


def arm_summaries(probs_path: Path, results: Sequence[dict], arm_names: Sequence[str], draws: int) -> dict[str, dict]:
    """Per arm: mean-of-cell macro-F1 with its source-video bootstrap interval, per-fold means, accuracy."""

    out = {}
    for name in arm_names:
        per_fold = {str(r["fold"]): float(np.mean(r["arms"][name]["macro_f1"])) for r in results}
        boot = paired_cluster_bootstrap(load_arm_predictions(probs_path, name), draws=draws)
        out[name] = {
            "macro_f1": float(np.mean(list(per_fold.values()))), "ci95": boot["base"]["ci95"], "per_fold_macro_f1": per_fold,
            "accuracy": float(np.mean([np.mean(r["arms"][name]["accuracy"]) for r in results])),
            "seconds": float(sum(r["arms"][name]["seconds"] for r in results)),
        }
        print(json.dumps({"arm": name, **{k: out[name][k] for k in ("macro_f1", "ci95", "accuracy")}}), flush=True)
    return out


def paired_contrasts(probs_path: Path, contrasts: Sequence[tuple[str, str]], arm_names: Sequence[str], draws: int) -> list[dict]:
    """Candidate-minus-base macro-F1 with a paired interval; `passes` is the single-contrast rule (interval > 0, gain >= MIN_GAIN)."""

    out = []
    for cand, base in contrasts:
        if cand not in arm_names or base not in arm_names:
            continue
        boot = paired_cluster_bootstrap(load_arm_predictions(probs_path, base), load_arm_predictions(probs_path, cand), draws=draws)
        diff = boot["difference"]
        out.append({"candidate": cand, "base": base, "difference": diff["observed"], "ci95": diff["ci95"],
                    "share_of_draws_above_zero": diff["share_of_draws_above_zero"],
                    "one_sided_p": 1.0 - diff["share_of_draws_above_zero"],
                    "passes": bool(diff["ci95"][0] > 0 and diff["observed"] >= MIN_GAIN)})
        print(json.dumps(out[-1]), flush=True)
    return out


def holm_family(contrasts: Sequence[dict], family: Sequence[tuple[str, str]]) -> list[dict]:
    """Holm step-down over a registered family of contrasts (D-67's rule): one-sided p <= ALPHA/(m - rank), and gain >= MIN_GAIN.

    Once one member fails its threshold every later (larger-p) member fails too.
    """

    members = sorted((c for c in contrasts if (c["candidate"], c["base"]) in set(family)), key=lambda c: c["one_sided_p"])
    out, still_rejecting = [], True
    for rank, c in enumerate(members):
        threshold = ALPHA_ONE_SIDED / (len(members) - rank)
        still_rejecting = still_rejecting and c["one_sided_p"] <= threshold
        out.append({"candidate": c["candidate"], "base": c["base"], "difference": c["difference"], "one_sided_p": c["one_sided_p"],
                    "holm_threshold": threshold, "significant": bool(still_rejecting),
                    "passes": bool(still_rejecting and c["difference"] >= MIN_GAIN)})
    return out


def write_outputs(output: Path, provenance: dict[str, Any], results: Sequence[dict], arms: dict, contrasts: Sequence[dict],
                  extra: dict[str, Any] | None = None) -> None:
    runs = [{"fold": r["fold"], "videos": r["videos"], "arms": r["arms"]} for r in results]
    (output / "runs.json").write_text(json.dumps({"provenance": provenance, "runs": runs}, indent=1) + "\n")
    (output / "summary.json").write_text(json.dumps({"provenance": provenance, "arms": arms, "contrasts": list(contrasts), **(extra or {})}, indent=1) + "\n")


def write_probabilities(output: Path, results: Sequence[dict], arm_names: Sequence[str]) -> Path:
    path = output / "probs.csv"
    inner_cv.write_probabilities(path, [(name, r["pool"], r["tables"][name]) for name in arm_names for r in results])
    return path
