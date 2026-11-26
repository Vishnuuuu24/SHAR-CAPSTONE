#!/usr/bin/env bash
# Launch one SHAR command inside a DGX pod and return as soon as it is
# running -- do not use this for anything you want auto-deleted; use
# watch_pod_job.sh to follow it and stop_pod_job.sh to tear it down.
#
# Usage:
#   scripts/dgx/start_pod_job.sh <job-name> '<command to run in the repo>'
#
# This exists because run_pod_job.sh's "launch, follow, delete on exit" shape
# is wrong for a multi-day run: `kubectl logs -f` can return on a transient
# API-server blip, network hiccup, or kubelet log rotation while the pod is
# still Running, and run_pod_job.sh's EXIT trap would then delete a job that
# has 20 hours left. This script never watches logs and never deletes the
# pod, so there is no trap for a dropped stream (or a Ctrl-C) to mistakenly
# fire.
#
# Same cluster-configuration environment variables as run_pod_job.sh (see
# _pod_common.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_pod_common.sh
source "${SCRIPT_DIR}/_pod_common.sh"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <job-name> '<command>'" >&2
    exit 2
fi

JOB_SUFFIX="$1"
JOB_CMD="$2"

render_pod_manifest "${JOB_SUFFIX}" "${JOB_CMD}"

# The only thing worth cleaning up on any exit from here on is the local
# manifest temp file -- never the pod. That is the entire difference from
# run_pod_job.sh's trap.
trap 'rm -f "${MANIFEST}"' EXIT

echo "--- applying ${JOB_NAME} ---"
kubectl apply -f "${MANIFEST}"

echo "--- waiting for scheduling ---"
if ! wait_for_scheduling "${JOB_NAME}"; then
    echo "pod ${JOB_NAME} failed to schedule; it has been left in place for inspection" >&2
    echo "(scripts/dgx/stop_pod_job.sh ${JOB_NAME} to remove it)" >&2
    exit 1
fi

echo
echo "=== ${JOB_NAME} is ${SCHEDULING_PHASE} ==="
echo "watch it:  scripts/dgx/watch_pod_job.sh ${JOB_NAME}"
echo "stop it:   scripts/dgx/stop_pod_job.sh ${JOB_NAME}"
echo
