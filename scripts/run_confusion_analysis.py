#!/usr/bin/env python3
"""E0-E2 of the divide-and-conquer study: baseline, confusion analysis, group selection, oracle ceilings.

Reads the registered DINOv2 attention-pooling probabilities; trains nothing. Validation folds only (test untouched).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import confusion_groups as cg  # noqa: E402

MANUAL_GROUPS = {
    "violent_trio": ["Shooting", "Assault", "Fighting"],
    "violent_quad": ["Shooting", "Assault", "Fighting", "Abuse"],
    "fight_assault": ["Fighting", "Assault"],
    "property": ["Burglary", "Robbery", "Shoplifting", "Stealing"],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", type=Path, default=Path("results/p1c/p1c-late-fusion/dinov2_probs.csv"))
    ap.add_argument("--reference-summary", type=Path, default=Path("results/p1c/p1c-late-fusion/summary.json"))
    ap.add_argument("--output", type=Path, default=Path("results/divide-and-conquer/e0-e2"))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    classes, runs = cg.load_runs(args.probs)
    n = len(classes)
    base = cg.baseline_report(classes, runs)
    ref = json.loads(args.reference_summary.read_text())["systems"]["A"]
    base["reference_mean"] = ref["macro_f1_mean"]
    base["reproduces_reference"] = abs(base["mean_over_runs"] - ref["macro_f1_mean"]) < 1e-9
    fold_cms = cg.fold_confusions(classes, runs)
    pooled = sum(fold_cms.values())
    f1 = {c: float(np.mean([cg.per_class_f1(r["truth"], r["pred"], n)[i] for r in runs.values()])) for i, c in enumerate(classes)}
    confusion = {"classes": classes, "per_class_f1_mean": f1,
                 "pooled_confusion": pooled.tolist(), "fold_confusions": {str(f): cm.tolist() for f, cm in fold_cms.items()},
                 "pooled_breakdown": cg.error_breakdown(pooled, classes),
                 "fold_breakdown": {str(f): cg.error_breakdown(cm, classes) for f, cm in fold_cms.items()},
                 "top_pairs_with_fold_consistency": cg.pair_consistency(fold_cms, classes),
                 "note": "counts are summed over 3 seeds per (fold, video); validation videos recur across folds, so pooled rows are not independent"}
    # groups: manual (frozen in code above) and data-driven, selected per scored fold from the other folds only
    selected = {f: cg.select_groups(cg.selection_confusion(classes, runs, f), classes) for f in sorted(fold_cms)}
    global_groups = cg.select_groups(pooled, classes)
    ceilings = []
    for name, g in MANUAL_GROUPS.items():
        for mode in ("oracle_route", "hard_route"):
            ceilings.append({"strategy": "manual", "name": name, **cg.ceiling(classes, runs, g, mode)})
    for size in (2, 3, 4):
        for mode in ("oracle_route", "hard_route"):
            per_fold = {}
            for f in sorted(fold_cms):
                if size not in selected[f]:
                    continue
                sub = {k: r for k, r in runs.items() if k[0] == f}
                c = cg.ceiling(classes, sub, selected[f][size]["classes"], mode)
                per_fold[f] = {"group": c["group"], "delta": c["per_fold_delta"][f]}
            if per_fold:
                d = [v["delta"] for v in per_fold.values()]
                ceilings.append({"strategy": "data_driven_lofo", "name": f"size{size}", "mode": mode, "per_fold": per_fold,
                                 "mean_delta": float(np.mean(d)), "folds_meeting_gate": sum(x >= cg.GATE_DELTA for x in d),
                                 "min_fold_delta": min(d), "n_folds": len(d)})
    for size, s in global_groups.items():
        for mode in ("oracle_route", "hard_route"):
            ceilings.append({"strategy": "data_driven_global_BIASED", "name": f"size{size}", **cg.ceiling(classes, runs, s["classes"], mode)})
    sha = hashlib.sha256(args.probs.read_bytes()).hexdigest()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    meta = {"probs": str(args.probs), "probs_sha256": sha, "git_revision": rev, "gate_delta": cg.GATE_DELTA, "device": "cpu",
            "selection_rule": "best coverage per size in {2,3,4}, anomaly classes only, closure>=0.5, internal>=6; LOFO excludes videos shared with the scored fold"}
    for name, obj in (("baseline", {**base, **meta}), ("confusion", confusion), ("selected_groups", {"lofo": {str(f): v for f, v in selected.items()},
                      "global_biased": global_groups}), ("ceilings", ceilings)):
        (args.output / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"baseline": base["fold_mean_macro_f1"], "mean": base["mean_over_runs"], "reproduces": base["reproduces_reference"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
