#!/usr/bin/env python3
"""Write the B-9 probe sweep's single comparable artifact.

    ./.venv/bin/python scripts/summarize_p1c_probes.py \
        --reports results/p1c/p1c-probes-b9-ar-fold1 \
        --reference results/p1c/p1c-video-frozen-confirm-015-ar-fold1-summary.json \
        --experiment-id P1C-PROBES-B9-AR-FOLD1

Reads every probe report in `--reports`, ranks them, and states the gap to the
trained head by reading that run's own aggregate rather than by restating its
number here.
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

from core.p1c_probe_summary import load_probe_reports, summarize_probe_reports  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True, help="directory of probe report JSONs")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="run summary the probes are compared against, e.g. the CONFIRM-015 aggregate",
    )
    parser.add_argument("--output", type=Path, default=None, help="default: <reports>/summary.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reports = load_probe_reports(args.reports)
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    payload = summarize_probe_reports(reports, experiment_id=args.experiment_id, reference=reference)

    output = args.output or args.reports / "summary.json"
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")

    best = payload["best"]
    print(f"{payload['probe_count']} probes; best {best['probe']}/{best['scaling']} "
          f"{best['parameters']} -> frame macro-F1 {best['frame_macro_f1']:.4f} "
          f"(clustered 95% CI {best['clustered_ci_lower_95']:.4f}-{best['clustered_ci_upper_95']:.4f})")
    if reference is not None:
        print(f"vs {payload['reference']['experiment_id']} "
              f"{payload['reference']['frame_macro_f1_mean']:.4f} "
              f"+/- {payload['reference']['frame_macro_f1_sd']:.4f}: "
              f"{payload['best_probe_minus_reference']:+.4f}")
    print(f"wrote {output} (SHA-256 {hashlib.sha256(text.encode('utf-8')).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
