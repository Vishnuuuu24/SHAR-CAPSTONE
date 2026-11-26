#!/usr/bin/env python3
"""GPU-pod check that the live demo's E3 path agrees with the saved E3 replay, and that the server answers.

1. Upload-path predictions (MP4 decoded with OpenCV) for a few UCF and external videos versus the saved replay.
   External videos were replayed through the same OpenCV path, so they should agree to rounding. UCF replays read
   the registered JPEG frame blob, so small differences are expected and are reported, not hidden.
2. A renamed copy of a UCF video must still get its dataset label and held-out members (content, not name).
3. The HTTP server (ui/demo/server.py handler) answers /health and /predict.

  .venv/bin/python scripts/smoke_demo_e3.py --out results/demo/e3-live-demo-20261124/smoke.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
E3 = REPO / "results/demo/e3-comparison-20261116"
UCF_KEYS = ["Explosion021_x264", "RoadAccidents086_x264", "Robbery044_x264", "Abuse001_x264"]


def compare(model, row: dict, group: str) -> dict:
    saved = json.loads((E3 / "answers" / group / f"{row['key']}.json").read_text())
    got = model.predict(Path(row["path"]), Path(row["path"]).name)
    diff = max(abs(got["probs"][c] - saved["probs"][c]) for c in model.classes)
    return {"key": row["key"], "group": group, "saved": saved["prediction"], "got": got["prediction"],
            "same_top1": saved["prediction"] == got["prediction"], "max_abs_prob_diff": diff,
            "confidence": got["confidence"], "members": len(got["heads_used"]), "mode": got["mode"],
            "frames": got["frames_used"], "duration_s": got["duration_s"], "timing_s": got["timing_s"],
            "analysed_frames_per_s": got["frames_used"] / max(got["timing_s"]["total"], 1e-9)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from core.e3_inference import E3DemoModel
    from ui.demo.server import make_handler

    model = E3DemoModel(REPO)
    cfg = model.support.CONFIG
    rows = {r["key"]: r for r in cfg["ucf"]}
    report = {"git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "health": model.health(), "videos": []}
    for k in UCF_KEYS:
        report["videos"].append(compare(model, rows[k], "ucf"))
        print(json.dumps(report["videos"][-1]), flush=True)
    for ds in ("RWF-2000", "XD-Violence"):
        row = next(r for r in cfg["external"] if r["dataset"] == ds
                   and json.loads((E3 / "answers/external" / f"{r['key']}.json").read_text())["status"] == "OK")
        report["videos"].append(compare(model, row, "external"))
        print(json.dumps(report["videos"][-1]), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        renamed = Path(tmp) / "upload.mp4"
        shutil.copyfile(rows["Robbery044_x264"]["path"], renamed)
        r = model.predict(renamed, "upload.mp4")
        report["renamed"] = {"expected": r["expected"], "mode": r["mode"], "prediction": r["prediction"]}
        print(json.dumps(report["renamed"]), flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(model, threading.Lock()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    health = json.loads(urllib.request.urlopen(base + "/health").read())
    page = urllib.request.urlopen(base + "/").read()
    body = Path(rows["Explosion021_x264"]["path"]).read_bytes()
    req = urllib.request.Request(base + "/predict", data=body, method="POST", headers={"X-Filename": "Explosion021_x264.mp4"})
    pred = json.loads(urllib.request.urlopen(req).read())
    server.shutdown()
    report["http"] = {"health_model": health["model"], "health_members": health["members"], "page_bytes": len(page),
                      "predict": pred["prediction"], "confidence": pred["confidence"], "windows": len(pred["windows"]),
                      "thumbnails": len(pred["thumbnails"]), "expected": pred["expected"]}
    print(json.dumps(report["http"]), flush=True)

    v = report["videos"]
    report["summary"] = {
        "all_same_top1": all(x["same_top1"] for x in v),
        "external_max_abs_prob_diff": max(x["max_abs_prob_diff"] for x in v if x["group"] == "external"),
        "ucf_max_abs_prob_diff": max(x["max_abs_prob_diff"] for x in v if x["group"] == "ucf"),
        "analysed_frames_per_s_median": float(np.median([x["analysed_frames_per_s"] for x in v])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
