#!/usr/bin/env python3
"""Re-score RESULT 1 (denoising) from the stored condition features without selecting on scored videos (plan Step 3).

1. check   re-score under the registered rule (best epoch on the clean validation videos) and require every
           condition's registered mean macro-F1 to be reproduced within --tolerance (cell differences reported);
2. rescore the same stored features under a fixed training length (final-epoch weights, nothing selected);
3. table   `result1_denoise_v2.csv` in the layout of `results/p2/result1_denoise.csv`.

The registered table is not touched. CPU only; the pod still claims the MIG slice (Kyverno).
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path.cwd(), help="checkout holding results/p2/*/features.float16.npy")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8, help="fixed training length for the re-score (plan Step 1: 8)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=0.005, help="largest allowed difference of a condition mean from the registered table")
    ap.add_argument("--skip-check", action="store_true", help="skip the reproduction of the registered numbers")
    ap.add_argument("--code-revision", default="unrecorded")
    args = ap.parse_args(argv)

    from core.p1c_head_variants import CHECKPOINT_BEST_VALIDATION, CHECKPOINT_LAST_EPOCH
    from core.p2_rescore import RUN_DIRS, difference_from_stored, rescore, write_run
    from scripts.build_result1_table import rows_for

    run_dirs = [args.data_root / d for d in RUN_DIRS]
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    common = dict(folds=args.folds, seeds=args.seeds, threads=args.threads)

    check = None
    if not args.skip_check:
        registered = rescore(args.data_root, run_dirs, rule=CHECKPOINT_BEST_VALIDATION, epochs=None, **common)
        check = {d.name: difference_from_stored(d, registered[d.name]) for d in run_dirs}
        print(json.dumps({"reproduction_of_registered_macro_f1": check}), flush=True)
        if max(c["max_condition_mean"] for c in check.values()) > args.tolerance:
            raise SystemExit(f"the registered numbers are not reproduced (tolerance {args.tolerance}): {check}")

    fixed = rescore(args.data_root, run_dirs, rule=CHECKPOINT_LAST_EPOCH, epochs=args.epochs, **common)
    rows = []
    for d in run_dirs:
        write_run(d, args.output / d.name, fixed[d.name], args.folds)
        rows.extend(rows_for(args.output / d.name))
    order = list(dict.fromkeys(r["degradation"] for r in rows))
    rows.sort(key=lambda r: order.index(r["degradation"]))
    with (args.output / "result1_denoise_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "provenance.json").write_text(json.dumps({
        "plan": "docs/FINAL_PUSH_PLAN_2026-11-11.md Step 3", "code_revision": args.code_revision, "host": socket.gethostname(),
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "rule": f"D-53 head trained {args.epochs} epochs on the fold's training videos, final-epoch weights, no selection",
        "folds": args.folds, "seeds": args.seeds, "source_runs": {d.name: json.loads((d / "config.json").read_text()) for d in run_dirs},
        "reproduction_of_registered_macro_f1": check, "test_split": "untouched",
        "label_view": "inherited video-level labels (D-40), not event-only",
    }, indent=1) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output / "result1_denoise_v2.csv")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
