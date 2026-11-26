#!/usr/bin/env bash
# Run one SHAR command inside an ephemeral DGX pod and always delete the pod.
#
# Usage:
#   scripts/dgx/run_pod_job.sh <job-name> '<command to run in the repo>'
#
# Example:
#   scripts/dgx/run_pod_job.sh cal004 \
#     '"$REPO_DIR/.venv/bin/python3.12" scripts/calibrate_p1c_frame_loader.py \
#        --config configs/experiments/p1c/P1C-FRAME-GAP-CALIBRATION-004-CUDA.yaml'
#
# The pod is deleted on success, on failure, and on Ctrl-C, via an EXIT trap.
# The script's exit status is the container's exit status, so a failed research
# command fails this script rather than being masked by a successful cleanup.
#
# This is the right tool for a short calibration, where "delete on Ctrl-C" and
# "delete if the log stream ever drops" are the behaviour you want. It is the
# WRONG tool for a multi-day training run: a transient API-server blip,
# network hiccup, or kubelet log rotation can end `kubectl logs -f` while the
# pod is still Running for hours, and this script now reconnects rather than
# deleting on that (see follow_logs_with_reconnect in _pod_common.sh) -- but
# Ctrl-C still deletes the pod here, which is right for "abandon this
# calibration" and wrong for "I just want to stop watching a long run". For a
# long run, use scripts/dgx/start_pod_job.sh, then
# scripts/dgx/watch_pod_job.sh / scripts/dgx/stop_pod_job.sh.
#
# Configure the cluster-specific values below by environment variable, or edit
# the defaults once they are confirmed:
#   NFS_SERVER  headnode address exporting /home
#   NFS_PATH    exported path
#   HOME_MOUNT  where that export is mounted inside the pod
#   REPO_DIR    repository location inside the pod (under HOME_MOUNT)

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <job-name> '<command>'" >&2
    exit 2
fi

JOB_SUFFIX="$1"
JOB_CMD="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_pod_common.sh
source "${SCRIPT_DIR}/_pod_common.sh"

render_pod_manifest "${JOB_SUFFIX}" "${JOB_CMD}"

cleanup() {
    local status=$?
    echo
    echo "--- cleaning up ${JOB_NAME} ---"
    kubectl delete pod "${JOB_NAME}" --ignore-not-found --wait=false || true
    rm -f "${MANIFEST}"
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "--- applying ${JOB_NAME} ---"
kubectl apply -f "${MANIFEST}"

echo "--- waiting for scheduling ---"
wait_for_scheduling "${JOB_NAME}"

echo "--- logs (${JOB_NAME}) ---"
follow_logs_with_reconnect "${JOB_NAME}"
phase="${FOLLOW_FINAL_PHASE}"

if [[ "${phase}" != "Succeeded" && "${phase}" != "Failed" ]]; then
    echo "--- pod ${JOB_NAME} disappeared before reaching a terminal phase ---" >&2
    exit 1
fi

diagnose_terminal_pod "${JOB_NAME}" "${phase}"

if [[ "${EXIT_CODE:-}" != "0" ]]; then
    exit "${EXIT_CODE:-1}"
fi
