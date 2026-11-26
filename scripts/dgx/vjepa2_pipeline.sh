#!/usr/bin/env bash
# V-JEPA 2 ViT-L clip-feature caches and the probe that compares them with DINOv2 and VideoMAE. Runs INSIDE a pod.
#
#   STAGE_FROM=$PWD/results/p1c/p1c-video-frames \
#     scripts/dgx/start_pod_job.sh vjepa2 'bash scripts/dgx/vjepa2_pipeline.sh <stage>'
#
# Stages (one pod each, or `all`): smoke | natural | shuffled | all
#   smoke     LIMIT (default 64) validation clips of fold 1; prints throughput so the real ETA can be quoted before committing the slice
#   natural   frames in order, 4 folds x (validation, train); then the probe without the shuffled arm
#   shuffled  the same with a fixed random frame order per clip; then the full probe
# Resumable: a split whose manifest.json exists is skipped.
set -euo pipefail

PY=./.venv/bin/python
stage="${1:?usage: vjepa2_pipeline.sh smoke|natural|shuffled|all}"
stamp() { date -u +%H:%M:%SZ; }

build() {  # build <fold> <split> <frame-order> <out-root>
    local fold=$1 split=$2 order=$3 root=$4
    local out="${root}/${split}"
    if [[ -f "${out}/manifest.json" ]]; then echo "skip ${out} (done)"; return; fi
    echo "########## fold${fold} ${split} ${order} $(stamp) ##########"
    $PY scripts/build_p1c_vjepa2_clip_cache.py --frame-order "${order}" \
        --config "configs/experiments/p1c/P1C-VIDEO-VJEPA2-AR-FOLD${fold}.yaml" --split "${split}" --output-dir "${out}" \
        --batch-size 4 --num-workers 6 \
        --extract-closure-path "${SCRATCH_MOUNT:-/scratch-emptydir}/p1c-video-frames/extract_closure.json"
}

caches() {  # caches <frame-order> <dir-stem>
    for fold in 1 2 3 4; do
        for split in validation train; do build "${fold}" "${split}" "$1" "results/p1c/$2-ar-fold${fold}"; done
    done
}

echo "=== started $(stamp) ==="
case "${stage}" in
smoke)
    t0=$(date +%s)
    $PY scripts/build_p1c_vjepa2_clip_cache.py --frame-order natural \
        --config configs/experiments/p1c/P1C-VIDEO-VJEPA2-AR-FOLD1.yaml --split validation \
        --output-dir "${SCRATCH_MOUNT:-/scratch-emptydir}/vjepa2-smoke" --batch-size 4 --num-workers 6 --limit "${LIMIT:-64}" --verify-sample 16 \
        --extract-closure-path "${SCRATCH_MOUNT:-/scratch-emptydir}/p1c-video-frames/extract_closure.json"
    echo "SMOKE seconds (includes weight load and sha256 of the 5 GB checkpoint): $(( $(date +%s) - t0 ))"
    ;;
natural|all)
    caches natural p1c-vjepa2-clip-cache
    $PY scripts/run_p1c_vjepa2_probe.py --arms dinov2 videomae vjepa2 fusion --output results/p1c/p1c-exp-vjepa2-probe
    [[ "${stage}" == natural ]] || { caches shuffled p1c-vjepa2-shuffled-clip-cache
        $PY scripts/run_p1c_vjepa2_probe.py --output results/p1c/p1c-exp-vjepa2-probe-with-shuffled; }
    ;;
shuffled)
    caches shuffled p1c-vjepa2-shuffled-clip-cache
    $PY scripts/run_p1c_vjepa2_probe.py --output results/p1c/p1c-exp-vjepa2-probe-with-shuffled
    ;;
*) echo "unknown stage ${stage}" >&2; exit 2 ;;
esac
echo "=== ALL DONE $(stamp) ==="
