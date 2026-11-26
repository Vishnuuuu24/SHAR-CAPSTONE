#!/usr/bin/env bash
# Run scripts/test_video.py inside a pod and delete the pod when it finishes.
#
# Usage (same arguments as scripts/test_video.py):
#   scripts/dgx/test_video.sh my_videos/clip.mp4 --label Fighting
#   scripts/dgx/test_video.sh my_videos/
#   scripts/dgx/test_video.sh --self-check
#
# Videos must be under the home directory, the only place a pod can read. The GPU slice is shared by every session and
# job, so if anything is already running or waiting this stops and says so; it never touches another pod.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${USER_HOME:=<home>}"
export USER_HOME
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${SHAR_PYTHON:-./.venv/bin/python}"

busy="$(kubectl get pods --no-headers 2>/dev/null | awk '$3 == "Running" || $3 == "Pending" || $3 == "ContainerCreating" {print $1 " (" $3 ")"}')"
if [[ -n "${busy}" ]]; then
    echo "The GPU slice is in use, so nothing was started. Try again when this finishes:" >&2
    echo "${busy}" >&2
    exit 3
fi

args=()
for arg in "$@"; do
    if [[ -e "${arg}" ]]; then
        arg="$(realpath "${arg}")"
        if [[ "${arg}" != "${USER_HOME}"/* ]]; then
            echo "A pod cannot read ${arg}. Copy it under ${USER_HOME}, for example ${REPO_DIR}/my_videos/." >&2
            exit 2
        fi
    fi
    args+=("${arg}")
done

command="$(printf '%q ' "${PYTHON}" scripts/test_video.py "${args[@]}")"
"${SCRIPT_DIR}/run_pod_job.sh" testvideo "${command}"
