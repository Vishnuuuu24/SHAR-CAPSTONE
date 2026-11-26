#!/usr/bin/env python3
"""Pre-registered equal-weight late fusion of DINOv2 head (A) with the frozen-SigLIP head (S), same head path.

Validation only (test untouched). Weight fixed at 0.5; nothing is selected on these numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_p1c_late_fusion import _sha, dinov2_probs  # noqa: E402  (sets the 4-thread limits on import)
from core import p1c_late_fusion as lf  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dinov2-csv", type=Path, default=Path("results/p1c/p1c-late-fusion/dinov2_probs.csv"))
    ap.add_argument("--siglip-cache-root", type=Path, default=Path("results/p1c/p1c-siglip-patch-cache"))
    ap.add_argument("--reference-summary", type=Path, default=Path("results/p1c/p1c-siglip-probe/summary.json"))
    ap.add_argument("--output", type=Path, default=Path("results/p1c/p1c-late-fusion-siglip"))
    ap.add_argument("--reuse-siglip", action="store_true", help="read saved siglip_probs.csv instead of re-training")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    s_csv = args.output / "siglip_probs.csv"
    if args.reuse_siglip:
        s = lf.load_dinov2_probs(s_csv)
    else:
        s = dinov2_probs(args.folds, args.seeds, args.siglip_cache_root)
        lf.write_probs_csv(s_csv, *s)
    a = lf.load_dinov2_probs(args.dinov2_csv)
    keys = lf.align(a, s)
    classes = a[0]
    systems = {"A": a, "S": (classes, {k: s[1][k] for k in keys}), "A+S": (classes, lf.fused_probs(a, s, keys))}
    ev = lf.evaluate(systems, "A", ["S", "A+S"])
    ev["comparisons"].update({k: v for k, v in lf.evaluate(systems, "S", ["A+S"])["comparisons"].items()})
    ev["correctness_overlap_A_vs_S"] = lf.correctness_overlap(a, s, keys)
    ref = json.loads(args.reference_summary.read_text())["arms"]["baseline"]
    ref_fm = {int(f): v for f, v in ref["fold_mean_macro_f1"].items()}
    ev["sanity_vs_siglip_probe"] = {
        "reference_overall_mean": ref["macro_f1_over_runs"]["mean"], "rerun_overall_mean": ev["systems"]["S"]["macro_f1_mean"],
        "per_fold_abs_diff": {f: abs(ev["systems"]["S"]["fold_mean_macro_f1"][f] - ref_fm[f]) for f in ref_fm}}
    ev["purpose"] = "validation-only pre-registered equal-weight late fusion A+S; no tuning; test untouched"
    (args.output / "summary.json").write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n")
    with (args.output / "per_video.csv").open("w") as out:
        for i, name in enumerate(("S", "A+S")):
            tmp = args.output / f"_tmp_{name}.csv"
            lf.write_probs_csv(tmp, classes, systems[name][1], system=name)
            lines = tmp.read_text().splitlines(keepends=True)
            out.writelines(lines if i == 0 else lines[1:])
            tmp.unlink()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    config = {"folds": args.folds, "seeds": args.seeds, "fusion_weight": lf.FUSION_WEIGHT, "git_revision": rev,
              "gate": "+0.04 fold-mean macro-F1 on 4/4 folds vs A (and vs S)",
              "dinov2_probs_sha256": _sha(args.dinov2_csv), "siglip_probs_sha256": _sha(s_csv),
              "reference_summary_sha256": _sha(args.reference_summary), "siglip_cache_root": str(args.siglip_cache_root),
              "siglip_cache_manifest_sha256": _sha(args.siglip_cache_root / "manifest.json"),
              "device": "cpu", "num_threads": 4, "runtime_s": round(time.time() - started), "reused_siglip_csv": args.reuse_siglip}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"systems": {n: {"mean": x["macro_f1_mean"], "folds": x["fold_mean_macro_f1"]} for n, x in ev["systems"].items()},
                      "sanity": ev["sanity_vs_siglip_probe"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
