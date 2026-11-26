#!/usr/bin/env bash
# Follow a SHAR job pod's logs. Safe to Ctrl-C: this script only watches, it
# never deletes the pod, on interrupt or otherwise. Use stop_pod_job.sh when
# you actually want it gone.
#
# Usage:
#   scripts/dgx/watch_pod_job.sh [pod-name]
#
# With no pod name, resolves the one Pending/Running pod named shar-*; if
# there is more than one it lists them and exits 2 rather than guessing which
# one you mean (see _pod_common.sh's resolve_pod_name).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_pod_common.sh
source "${SCRIPT_DIR}/_pod_common.sh"

POD_NAME="$(resolve_pod_name "${1:-}")" || exit $?

# Ctrl-C here means "stop watching", not "stop the job" -- the opposite of
# run_pod_job.sh's trap, and the whole reason this script exists separately
# from it. No EXIT trap at all: there is nothing to clean up.
trap 'echo; echo "--- stopped watching ${POD_NAME}; it is still running (scripts/dgx/stop_pod_job.sh ${POD_NAME} to delete it) ---"; exit 130' INT

echo "--- logs (${POD_NAME}) ---"
follow_logs_with_reconnect "${POD_NAME}"
phase="${FOLLOW_FINAL_PHASE}"

if [[ "${phase}" != "Succeeded" && "${phase}" != "Failed" ]]; then
    echo "--- ${POD_NAME} is gone (no terminal phase observed) ---" >&2
    exit 1
fi

diagnose_terminal_pod "${POD_NAME}" "${phase}"

if [[ "${EXIT_CODE:-}" != "0" ]]; then
    exit "${EXIT_CODE:-1}"
fi
