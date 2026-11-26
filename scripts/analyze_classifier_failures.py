#!/usr/bin/env python3
"""Validation-only failure analysis of the D-53 headline classifier (stored probabilities; no retraining, no test data).

Reads results/p1c/p1c-late-fusion/dinov2_probs.csv (4 folds x 3 seeds x 56 validation videos) and writes
results/p1c/classifier-failure-analysis/analysis.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import failure_analysis as fa  # noqa: E402

PROBS = "results/p1c/p1c-late-fusion/dinov2_probs.csv"
FRAMES = "results/p1c/error-analysis-per-video.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", type=Path, default=REPO_ROOT / PROBS)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "results/p1c/classifier-failure-analysis")
    args = ap.parse_args(argv)
    p = fa.load_predictions(args.probs)
    frames = {r["video"]: r["frames"] for rows in json.loads((REPO_ROOT / FRAMES).read_text()).values() for r in rows}
    out = {"source": str(args.probs.relative_to(REPO_ROOT)), "source_sha256": hashlib.sha256(args.probs.read_bytes()).hexdigest(),
           "evaluations": int(len(p.truth)), "unique_validation_videos": int(len(set(p.video.tolist()))),
           "headline": fa.headline(p), "bootstrap": fa.cluster_bootstrap(p), "per_class": fa.per_class(p),
           "error_structure": fa.error_structure(p), "video_reliability": fa.video_reliability(p),
           "topk": fa.topk_accuracy(p), "calibration": fa.calibration(p), "normal_vs_crime_auc": fa.normal_vs_crime_auc(p),
           "length_effect": fa.length_effect(p, frames), "test_data_used": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("evaluations", "unique_validation_videos", "headline", "bootstrap", "topk", "normal_vs_crime_auc")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
