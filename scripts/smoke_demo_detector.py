#!/usr/bin/env python3
"""GPU check of the demo's prototype overlay: live-path detector scores and E3 probabilities versus the saved runs,
with and without the K-33 JPEG step. Official detector TEST videos only, so saved test scores exist; nothing is
tuned on the result (the detector's threshold and weights are the registered ones).

  .venv/bin/python scripts/smoke_demo_detector.py --out results/demo/e3-live-demo-20261124/detector_smoke.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
E3 = REPO / "results/demo/e3-comparison-20261116"
SAVED = REPO / "results/detect/followup-20261112/ucf-002/test-control-seed0.npz"
VIDEOS = ["Explosion/Explosion021_x264.mp4", "Shoplifting/Shoplifting033_x264.mp4", "Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from core.e3_inference import E3DemoModel

    model = E3DemoModel(REPO)
    saved = np.load(SAVED)
    rows = {}
    for r in model.support.CONFIG["ucf"]:
        for alias in r["aliases"]:
            rows[alias] = r
    report = {"git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "threshold": model.detector.threshold, "videos": []}
    for rel in VIDEOS:
        key = next((k for k in saved.files if k.endswith(Path(rel).name)), None)
        row = rows.get(rel) or next(r for a, r in rows.items() if a.endswith(Path(rel).name))
        want = saved[key]
        answer = json.loads((E3 / "answers/ucf" / f"{row['key']}.json").read_text())
        for jpeg in (False, True):
            model.encoders.jpeg = jpeg
            got = model.predict(Path(row["path"]), Path(row["path"]).name)
            sc = np.array([s["score"] for s in got["detector"]["steps"]])
            n = min(len(sc), len(want))
            th = model.detector.threshold
            rec = {"video": rel, "jpeg": jpeg, "steps": [int(len(sc)), int(len(want))],
                   "max_abs_score_diff": float(np.max(np.abs(sc[:n] - want[:n]))),
                   "alarm_agreement": float(np.mean((sc[:n] >= th) == (want[:n] >= th))),
                   "alarm_segments": got["detector"]["segments"], "exposure": got["detector"]["exposure"],
                   "boxes_in_alarms": sum(1 for s in got["detector"]["steps"] if s["alarm"] and s["box"]),
                   "alarm_steps": sum(1 for s in got["detector"]["steps"] if s["alarm"]),
                   "e3_prediction": got["prediction"], "e3_saved": answer["prediction"],
                   "e3_max_abs_prob_diff": max(abs(got["probs"][c] - answer["probs"][c]) for c in model.classes),
                   "seconds": got["timing_s"]["total"]}
            report["videos"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "alarm_segments"}), flush=True)
    for jpeg in (False, True):
        v = [r for r in report["videos"] if r["jpeg"] == jpeg]
        report[f"summary_jpeg_{jpeg}"] = {
            "max_abs_score_diff": max(r["max_abs_score_diff"] for r in v),
            "min_alarm_agreement": min(r["alarm_agreement"] for r in v),
            "e3_max_abs_prob_diff": max(r["e3_max_abs_prob_diff"] for r in v),
            "e3_same_top1": all(r["e3_prediction"] == r["e3_saved"] for r in v)}
        print(json.dumps({f"jpeg={jpeg}": report[f"summary_jpeg_{jpeg}"]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
