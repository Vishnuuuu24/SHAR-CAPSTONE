#!/usr/bin/env bash
# Explicitly delete a SHAR job pod. This is the only one of the pod-job
# scripts that destroys anything -- start_pod_job.sh and watch_pod_job.sh
# both leave the pod alone on purpose (see their headers).
#
# Usage:
#   scripts/dgx/stop_pod_job.sh [pod-name]
#
# With no pod name, resolves the one Pending/Running pod named shar-*; if
# there is more than one it lists them and exits 2 rather than guessing which
# one you mean (see _pod_common.sh's resolve_pod_name).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_pod_common.sh
source "${SCRIPT_DIR}/_pod_common.sh"

POD_NAME="$(resolve_pod_name "${1:-}")" || exit $?

echo "--- deleting ${POD_NAME} ---"
kubectl delete pod "${POD_NAME}" --ignore-not-found
