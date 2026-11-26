"""The D-65 claim rule, shared by every D-65 experiment runner.

A contrast passes when its paired source-video bootstrap gain is >= +0.02 macro-F1 and its one-sided bootstrap
p-value (share of draws with gain <= 0) clears Holm's step-down threshold over the registered contrasts at a
family-wise two-sided 95% level (one-sided 0.025). With one contrast this is exactly "95% CI lower bound > 0".
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

VIOLENCE = ("Abuse", "Assault", "Fighting", "Shooting")
MIN_GAIN = 0.02
ONE_SIDED_ALPHA = 0.025


def holm(pvalues: Sequence[float], alpha: float = ONE_SIDED_ALPHA) -> list[bool]:
    """Holm step-down: reject in increasing-p order while p_(i) <= alpha / (m - i)."""

    order = np.argsort(pvalues, kind="stable")
    reject = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] > alpha / (len(pvalues) - rank):
            break
        reject[i] = True
    return reject


def contrast(probs_path: Path, candidate: str, control: str, draws: int = 2000) -> dict:
    """Paired bootstrap, per-class F1 and violence mean for one candidate-vs-control contrast (no verdict)."""

    from core import failure_analysis as fa
    from core.confusion_groups import per_class_f1

    base, cand = fa.load_arm_predictions(probs_path, control), fa.load_arm_predictions(probs_path, candidate)
    boot = fa.paired_cluster_bootstrap(base, cand, draws=draws)

    def class_f1(p):
        cells = sorted({(f, s) for f, s in zip(p.fold, p.seed)})
        return np.mean([per_class_f1(p.truth[m], p.pred[m], len(p.classes)) for f, s in cells for m in [(p.fold == f) & (p.seed == s)]], axis=0)

    fb, fc = class_f1(base), class_f1(cand)
    vi = [base.classes.index(c) for c in VIOLENCE]
    return {"contrast": f"{candidate} vs {control}", "paired_bootstrap": boot,
            "one_sided_p": 1.0 - boot["difference"]["share_of_draws_above_zero"],
            "per_class_f1": {c: {"control": float(fb[i]), "candidate": float(fc[i]), "change": float(fc[i] - fb[i])} for i, c in enumerate(base.classes)},
            "violence_mean_f1": {"control": float(fb[vi].mean()), "candidate": float(fc[vi].mean())}}


def claims(probs_path: Path, contrasts: Sequence[tuple[str, str]], draws: int = 2000) -> list[dict]:
    """Every registered contrast with its Holm-adjusted D-65 verdict."""

    from core.confusion_groups import GATE_DELTA

    out = [contrast(probs_path, cand, ctrl, draws) for cand, ctrl in contrasts]
    significant = holm([c["one_sided_p"] for c in out])
    for c, sig in zip(out, significant):
        c["holm_significant"] = bool(sig)
        c["claim_passed"] = bool(sig and c["paired_bootstrap"]["difference"]["observed"] >= MIN_GAIN)
        c["rule"] = (f"Holm over {len(out)} registered contrast(s), one-sided bootstrap p <= {ONE_SIDED_ALPHA}/(m-rank), "
                     f"and point gain >= +{MIN_GAIN}")
        c["historical_gate_threshold"] = GATE_DELTA
    return out
