#!/usr/bin/env python3
"""CPU diagnostic: is XD-Violence Shooting/Fighting/Abuse close to UCF's in DINOv2 space? Writes results/p1c/p1c-xd-domain-gap/."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import p1c_xd_domain_gap as g  # noqa: E402
from core.p1c_head_variants import load_pooled_features  # noqa: E402
from core.p1c_video_sampling import sample_rows_per_video  # noqa: E402
from core.p1c_xd_extra import load_cache  # noqa: E402
from scripts.run_p1c_head_comparison import _load_fold, _load_patch_cache, _patch_indices_for  # noqa: E402

THREE = ["Shooting", "Fighting", "Abuse"]
SEEDS = list(range(10))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_ucf(patch_root: Path):
    """Video-level UCF vectors for all 700 videos, 8 sampled frames each (sampling_seed 0, as the headline path).

    Returns dict with vids, labels, vecs, val_fold (fold where the video is validation), fold1_train mask.
    """

    cls, feats, k2i = _load_patch_cache(patch_root)
    args = types.SimpleNamespace(frames=8, sampling_seed=0,
                                 extract_closure_path=Path("results/p1c/p1c-video-frames/extract_closure.json"))
    per_video, val_fold, fold1_train = {}, {}, set()
    for fold in (1, 2, 3, 4):
        fd = _load_fold(fold, args)
        val_rows = [fd["validation_rows"][i] for i in sample_rows_per_video(fd["validation_rows"], 8, seed=0)]
        for r in val_rows:
            val_fold[r["source_video_id"]] = fold
        for rows in (val_rows, fd["train_rows"]):
            for r in rows:
                if per_video.get(r["source_video_id"], {}).get("_done"):
                    continue
                per_video.setdefault(r["source_video_id"], {})[int(r["source_frame_index"])] = r
            for v in {r["source_video_id"] for r in rows}:
                per_video[v]["_done"] = True
        if fold == 1:
            fold1_train = {r["source_video_id"] for r in fd["train_rows"]}
            fold1_val = {r["source_video_id"] for r in val_rows}
    vids = sorted(per_video)
    for v in vids:
        per_video[v].pop("_done")
    rows = [r for v in vids for r in per_video[v].values()]
    pooled = load_pooled_features(cls, feats, _patch_indices_for(rows, k2i))
    vid_of = [r["source_video_id"] for r in rows]
    _, vecs = g.video_vectors(pooled, vid_of)
    label = {r["source_video_id"]: r["label"] for r in rows}
    return {"vids": vids, "labels": np.array([label[v] for v in vids]), "vecs": vecs,
            "val_fold": np.array([val_fold.get(v, 0) for v in vids]), "fold1_train": np.array([v in fold1_train for v in vids]),
            "fold1_val": np.array([v in fold1_val for v in vids]), "frames_per_video": {v: len(per_video[v]) for v in vids}}


def summarise(o: dict) -> str:
    """Numbers-only paragraph assembled from the results; no verdict beyond the stated comparisons."""

    s, t3, t2 = o["4_summary_inputs"], o["3_transfer"]["3way"], o["3_transfer"]["2way_Shooting_vs_Fighting"]
    geo = o["1_centroid_geometry"]["ucf_all_700"]["xd_centroid_vs_ucf_centroids"]
    rk = {c: geo[c]["same_name_rank"] for c in THREE}
    r = lambda d, c: round(d["recall"][c]["mean"], 3)
    return (
        f"(i) XD->UCF transfer: 3-way accuracy {s['xd_to_ucf_acc']:.3f} (chance 0.333); UCF recall Shooting {r(t3['a_XD_to_UCF'], 'Shooting')}, "
        f"Fighting {r(t3['a_XD_to_UCF'], 'Fighting')}, Abuse {r(t3['a_XD_to_UCF'], 'Abuse')}; 2-way Shooting/Fighting {t2['a_XD_to_UCF']['accuracy']['mean']:.3f} "
        f"(Shooting recall {r(t2['a_XD_to_UCF'], 'Shooting')}). Same-named UCF centroid rank of 14 for XD classes (1=nearest): {rk}. "
        f"(ii) Domain gap: XD-vs-UCF same-class ROC-AUC Shooting {s['domain_auc_by_class']['Shooting']:.3f}, Fighting {s['domain_auc_by_class']['Fighting']:.3f}, "
        f"Abuse {s['domain_auc_by_class']['Abuse']:.3f}, versus {s['domain_auc_reference_real_classes']:.3f} for UCF Shooting-vs-Fighting; UCF->XD balanced 3-way accuracy "
        f"{s['ucf_to_xd_acc_balanced']:.3f}. (iii) In-domain 3-way accuracy: UCF leave-one-fold-out {s['ucf_in_domain_acc']:.3f}, XD 5-fold CV {s['xd_in_domain_acc']:.3f} "
        f"(chance 0.333; UCF n=150 so one s.e. is about 0.04, XD CV used balanced 31/class subsamples). Interpretation is left to the reader: the numbers do not by "
        f"themselves identify whether low cross-domain accuracy is caused by the domain gap or by weak class signal, since in-domain UCF separation is also low.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xd-cache", type=Path, default=Path("results/p1c/p1c-xd-extra-cache"))
    ap.add_argument("--patch-cache-root", type=Path, default=Path("results/p1c/p1c-patch-cache"))
    ap.add_argument("--output", type=Path, default=Path("results/p1c/p1c-xd-domain-gap"))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    xf, xrows = load_cache(args.xd_cache)
    xids, xvec = g.video_vectors(xf, [r["video_id"] for r in xrows])
    xlab_of = {r["video_id"]: r["class_name"] for r in xrows}
    xlab = np.array([xlab_of[v] for v in xids])
    u = load_ucf(args.patch_cache_root)
    classes = sorted(set(u["labels"]))
    out = {"n_ucf_videos": len(u["vids"]), "n_xd_videos": len(xids),
           "ucf_per_class": {c: int((u["labels"] == c).sum()) for c in classes},
           "xd_per_class": {c: int((xlab == c).sum()) for c in sorted(set(xlab))}}

    # 1. centroid geometry
    geo = {}
    for name, mask in (("ucf_fold1_train", u["fold1_train"]), ("ucf_all_700", np.ones(len(u["vids"]), bool))):
        ucents = g.centroids(u["vecs"][mask], u["labels"][mask])
        xcents = g.centroids(xvec, xlab)
        geo[name] = {"n_videos": int(mask.sum()),
                     "xd_centroid_vs_ucf_centroids": {c: g.centroid_rank(xcents[c], ucents, c) for c in THREE},
                     "ucf_shooting_vs_other_ucf_centroids": g.centroid_rank(ucents["Shooting"], {k: v for k, v in ucents.items() if k != "Shooting"}, "Fighting")["cosine"],
                     "ucf_within_class_mean_video_to_centroid_cos": g.within_class_cos(u["vecs"][mask], u["labels"][mask]),
                     "xd_centroid_vs_xd_other_centroids_cos": {c: {k: float(xcents[c] @ xcents[k]) for k in THREE if k != c} for c in THREE}}
    geo["xd_within_class_mean_video_to_centroid_cos"] = g.within_class_cos(xvec, xlab)
    out["1_centroid_geometry"] = geo

    # 2. domain separability (all 700 UCF videos, 50/class)
    sep = {c: g.domain_separability(xvec[xlab == c], u["vecs"][u["labels"] == c], seed=0) for c in THREE}
    sep["reference_UCF_Shooting_vs_UCF_Fighting"] = g.domain_separability(
        u["vecs"][u["labels"] == "Shooting"], u["vecs"][u["labels"] == "Fighting"], seed=0)
    out["2_domain_separability"] = {"setup": f"LR C={g.LR_C}, standardised on train folds, 5-fold stratified CV seed 0, A=XD(label 0) vs B=UCF(label 1); "
                                              "reference: A=UCF Shooting vs B=UCF Fighting", "results": sep}

    # 3. transfer
    tr = {}
    for tag, cl in (("3way", THREE), ("2way_Shooting_vs_Fighting", ["Shooting", "Fighting"])):
        um = np.isin(u["labels"], cl); ux, uy, ug = u["vecs"][um], u["labels"][um], u["val_fold"][um]
        xm = np.isin(xlab, cl); xx, xy = xvec[xm], xlab[xm]
        per = int(min((xy == c).sum() for c in cl))
        xd_to_ucf, ucf_to_xd_bal, xd_cv = [], [], []
        for s in SEEDS:
            idx = g.balanced_subsample(xy, per, np.random.default_rng(s))
            xd_to_ucf.append(g.transfer(xx[idx], xy[idx], ux, uy))
            xd_cv.append(g.kfold_cv(xx[idx], xy[idx], seed=s))
            ucf_to_xd_bal.append(g.transfer(ux, uy, xx[idx], xy[idx]))
        tr[tag] = {"classes": cl, "chance": 1 / len(cl), "xd_balanced_per_class": per,
                   "a_XD_to_UCF": g.mean_reports(xd_to_ucf),
                   "b_UCF_to_XD_all_xd_videos": g.transfer(ux, uy, xx, xy),
                   "b_UCF_to_XD_balanced_subsamples": g.mean_reports(ucf_to_xd_bal),
                   "c_UCF_to_UCF_leave_one_fold_out": g.leave_one_group_out(ux, uy, ug),
                   "c_XD_to_XD_5fold_cv_balanced_subsamples": g.mean_reports(xd_cv)}
    out["3_transfer"] = tr

    t3 = tr["3way"]
    out["4_summary_inputs"] = {
        "xd_to_ucf_acc": t3["a_XD_to_UCF"]["accuracy"]["mean"], "ucf_to_xd_acc_all": t3["b_UCF_to_XD_all_xd_videos"]["accuracy"],
        "ucf_to_xd_acc_balanced": t3["b_UCF_to_XD_balanced_subsamples"]["accuracy"]["mean"],
        "ucf_in_domain_acc": t3["c_UCF_to_UCF_leave_one_fold_out"]["accuracy"],
        "xd_in_domain_acc": t3["c_XD_to_XD_5fold_cv_balanced_subsamples"]["accuracy"]["mean"],
        "domain_auc_by_class": {c: sep[c]["roc_auc"] for c in THREE},
        "domain_auc_reference_real_classes": sep["reference_UCF_Shooting_vs_UCF_Fighting"]["roc_auc"]}
    out["4_summary_text"] = summarise(out)
    (args.output / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    patch_manifest = json.loads((args.patch_cache_root / "manifest.json").read_text())
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    cfg = {"git_revision": rev, "seeds_subsamples": SEEDS, "cv_seed": 0, "lr_C": g.LR_C, "threads": 4,
           "video_vector": "mean of 8 pooled frame vectors (CLS++mean patch), then L2-normalised",
           "ucf_frames": "8/video: fold train rows as cached (sampling_seed 0); validation rows via sample_rows_per_video(8, seed 0)",
           "xd_features_sha256_actual": _sha(args.xd_cache / "features.npy"), "xd_manifest_sha256_actual": _sha(args.xd_cache / "manifest.json"),
           "xd_features_sha256_in_manifest": json.loads((args.xd_cache / "manifest.json").read_text())["features_sha256"],
           "ucf_patch_cache_features_sha256": patch_manifest["features_sha256"], "ucf_patch_cache_cls_sha256": patch_manifest["cls_sha256"],
           "ucf_videos": [{"video": v, "label": str(l), "val_fold": int(f), "fold1_train": bool(t)} for v, l, f, t in
                          zip(u["vids"], u["labels"], u["val_fold"], u["fold1_train"])],
           "ucf_frames_per_video_counts": {str(k): int(v) for k, v in zip(*np.unique(list(u["frames_per_video"].values()), return_counts=True))}}
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["4_summary_inputs"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
