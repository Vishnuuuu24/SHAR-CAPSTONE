#!/usr/bin/env python3
"""Build ui/index.html: one page showing the final recorded result of every SHAR model, read from results/."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
J = lambda p: json.loads((R / p).read_text())

head = J("p1c/headline-14class-test/test.json")
roi = J("p1c/p1c-roi-study/summary.json")["arms"]
late = J("p1c/p1c-late-fusion-siglip/summary.json")["systems"]
fus = J("p1c/p1c-exp-P2-fusion/fusion-results.json")["results"]
sub = J("subset7-search/test.json")["summary"]
xd = J("xd/xd-probe/test.json")
mean = lambda v: sum(v) / len(v)
fus_m = {a: mean([f[k]["video_macro_f1_mean"] for k in f]) for a, f in fus.items()}
s7 = {v["novelty"]: v["test_mean"] for v in sub.values()}
xr = xd["unweighted"]["runs"]

ucf = [
    dict(name="ResNet50 baseline (frame, majority vote)", f1=roi["full_frame"]["video_level_macro_f1_mean"] - 0.1056, note="derived: DINOv2 0.4068 minus the registered +0.1056 gain", split="validation", derived=True),
    dict(name="DINOv2 + 8 frames + linear head + majority vote", f1=roi["full_frame"]["video_level_macro_f1_mean"], sd=roi["full_frame"]["video_level_macro_f1_sd"], split="validation"),
    dict(name="VideoMAE-B (Kinetics, clip features)", f1=fus_m["videomae"], split="validation"),
    dict(name="DINOv2 + VideoMAE fusion", f1=fus_m["fusion"], split="validation", note="same experiment's own DINOv2 arm: %.4f" % fus_m["dinov2"]),
    dict(name="SigLIP probe", f1=late["S"]["macro_f1_mean"], split="validation"),
    dict(name="DINOv2 attention-pooling head (D-53, headline)", f1=late["A"]["macro_f1_mean"], split="validation", test=head["summary"]["test_macro_f1"], test_acc=head["summary"]["test_accuracy"], acc=head["summary"]["val_accuracy"]),
    dict(name="DINOv2 + SigLIP late fusion", f1=late["A+S"]["macro_f1_mean"], split="validation", note="failed the +0.04 on 4/4 folds gate; test untouched"),
]
data = dict(
    ucf=ucf,
    headline_classes=head["test_per_class_f1"],
    headline_folds=head["per_fold"],
    roi=dict(
        full=roi["full_frame"]["video_level_macro_f1_mean"], roi=roi["roi"]["video_level_macro_f1_mean"],
        paired=J("p1c/p1c-roi-study/summary.json")["paired_diff_mean"], sd=J("p1c/p1c-roi-study/summary.json")["paired_diff_sd"],
        per_fold=J("p1c/p1c-roi-study/summary.json")["paired_diff_roi_minus_full_frame_per_fold"],
        fallback=J("p1c/p1c-roi-study/summary.json")["fallback_fraction_per_fold"],
        cls_full=roi["full_frame"]["per_class_video_level_f1_mean_over_seeds"], cls_roi=roi["roi"]["per_class_video_level_f1_mean_over_seeds"],
        det=J("p1c/p1c-roi-detector-check/summary.json"),
    ),
    sub7=s7,
    sub7_classes=next(v for v in sub.values() if v["novelty"] == 6)["test_per_class_f1"],
    xd=dict(
        map=mean([r["map"] for r in xr]), acc=mean([r["single_label_accuracy"] for r in xr]), f1=mean([r["single_label_macro_f1"] for r in xr]),
        ap={k: mean([r["ap"][k] for r in xr]) for k in xr[0]["ap"]},
    ),
)
tpl = (Path(__file__).parent / "template.html").read_text()
(Path(__file__).parent / "index.html").write_text(tpl.replace("__DATA__", json.dumps(data)))
print("wrote ui/index.html")
