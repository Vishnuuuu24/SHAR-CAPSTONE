#!/usr/bin/env bash
# Open notebooks/test_video.ipynb in JupyterLab inside a pod on the GPU slice.
#
# Usage: scripts/dgx/notebook.sh
#
# The pod holds the account's only GPU slice while it is up. Jupyter shuts itself down after IDLE_SECONDS with no
# activity, so a forgotten notebook frees the slice. For a single quick test, scripts/dgx/test_video.sh is simpler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${USER_HOME:=<home>}"
export USER_HOME
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PORT="${PORT:-8888}"
IDLE_SECONDS="${IDLE_SECONDS:-1800}"

busy="$(kubectl get pods --no-headers 2>/dev/null | awk '$3 == "Running" || $3 == "Pending" || $3 == "ContainerCreating" {print $1 " (" $3 ")"}')"
if [[ -n "${busy}" ]]; then
    echo "The GPU slice is in use, so nothing was started. Try again when this finishes:" >&2
    echo "${busy}" >&2
    exit 3
fi

"${SCRIPT_DIR}/start_pod_job.sh" notebook "jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --allow-root --ServerApp.root_dir=\"\$REPO_DIR\" --ServerApp.shutdown_no_activity_timeout=${IDLE_SECONDS} --MappingKernelManager.cull_idle_timeout=${IDLE_SECONDS} --MappingKernelManager.cull_interval=60"

pod="$(kubectl get pods --no-headers --sort-by=.metadata.creationTimestamp -o custom-columns=NAME:.metadata.name | grep -- '-notebook-' | tail -1)"
cat <<EOF

Notebook pod: ${pod}. It holds the GPU slice until you stop it, or after ${IDLE_SECONDS}s with no activity.

1. Get the login link (wait until it prints a URL with ?token=, then press Ctrl-C):
     kubectl logs -f ${pod}
2. On the head node, leave this running:
     kubectl port-forward pod/${pod} ${PORT}:${PORT}
3. On your laptop, leave this running:
     ssh -N -L ${PORT}:localhost:${PORT} $(whoami)@<internal-ip>
4. Open the link from step 1 in your browser, then open notebooks/test_video.ipynb.
5. When you finish, free the slice:
     scripts/dgx/stop_pod_job.sh ${pod}
EOF
