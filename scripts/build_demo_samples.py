#!/usr/bin/env python3
"""Copy the demo page's example clips into results/demo/samples/ and write samples.json (CPU only).

Featured clips are the presentation showcase (results/demo/showcase-20261124); the rest are short backups from the
same checked candidate list. The facts shown on the page (label, E3 prediction and confidence, length) come from
the saved E3 replay answers. The page itself re-runs E3 live, so nothing displayed is taken on trust.

  .venv/bin/python scripts/build_demo_samples.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.demo_samples import MANIFEST  # noqa: E402

E3 = REPO / "results/demo/e3-comparison-20261116"
FEATURED = ["Explosion021_x264", "RoadAccidents086_x264", "Robbery044_x264"]
MORE = ["Explosion014_x264", "Shoplifting033_x264", "RoadAccidents089_x264", "Stealing081_x264"]
TITLES = {
    "Explosion021_x264": "Petrol-station explosion",
    "RoadAccidents086_x264": "Motorbike falls at a junction",
    "Robbery044_x264": "Shop counter robbery",
    "Explosion014_x264": "Explosion",
    "Shoplifting033_x264": "Shoplifting",
    "RoadAccidents089_x264": "Road accident",
    "Stealing081_x264": "Stealing",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    out = REPO / MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = {r["key"]: r for r in json.loads((E3 / "config.json").read_text())["ucf"]}
    samples = []
    for key in FEATURED + MORE:
        a = json.loads((E3 / "answers/ucf" / f"{key}.json").read_text())
        if not a["ucf_correct"] or a["exposure"] != "held_out":
            raise SystemExit(f"{key}: not a held-out correct E3 answer; refusing to feature it")
        shutil.copyfile(rows[key]["path"], out.parent / f"{key}.mp4")
        samples.append({
            "file": f"{key}.mp4", "key": key, "title": TITLES[key], "label": a["ucf_label"],
            "featured": key in FEATURED, "e3_prediction": a["prediction"], "e3_confidence": round(a["confidence"], 3),
            "duration_s": round(rows[key]["kept_frames"] * 5 / rows[key]["fps"], 1),
            "held_out_folds": a["selected_folds"],
        })
    out.write_text(json.dumps({"note": "Licensed UCF-Crime clips: local only, served to token holders only.",
                               "samples": samples}, indent=1) + "\n")
    print(json.dumps({"manifest": str(out), "clips": [s["file"] for s in samples]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
