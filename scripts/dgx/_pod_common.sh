# Shared machinery for the DGX pod-job scripts (run_pod_job.sh,
# start_pod_job.sh, watch_pod_job.sh, stop_pod_job.sh).
#
# Sourced, never executed: it only defines defaults and functions, and relies
# on the caller having already done `set -euo pipefail` (or having sourced it
# from a script that has).  Nothing below spawns a long-running process, so
# there is nothing here that a caller's `set -e` can trip over.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "_pod_common.sh is a library; source it from another script, don't run it directly" >&2
    exit 1
fi

# --- cluster configuration -------------------------------------------------
# /home is a 66 TB local XFS disk on <head-node>, exported read-write to
# the Bright cluster fabric <internal-ip>/16, and the worker nodes mount it.  The
# pod therefore takes it as a hostPath.  If the workers ever stop mounting it,
# the pod fails to start (type: Directory) rather than running against an empty
# directory -- the failure is visible, not silent.
#
# A Kyverno admission policy confines hostPath volumes to an explicit
# allowlist for this account:
#   <home>
#   /workspace/fast-dgx-storage/<user>
#   /shared-ro
#   /shared-workspace
# The bare /home export is not on that list and is rejected at admission, so
# HOST_PATH must be the scoped per-account directory, mounted to itself.
: "${USER_HOME:=<home>}"
: "${HOST_PATH:=${USER_HOME}}"
: "${HOME_MOUNT:=${USER_HOME}}"
: "${REPO_DIR:=${USER_HOME}/SHAR}"
: "${IMAGE:=<head-node>/nvcr.io/nvidia/pytorch:24.10-py3}"
: "${CPU:=8}"
: "${MEM:=16Gi}"
: "${RUN_CLASS:=calibration}"
# Node-local scratch (D-33/D-34).  SCRATCH_SIZE_LIMIT must exceed the
# staged blob; the probed emptyDir had 1.81 TB free.  STAGE_FROM empty
# means no staging, the right default for jobs that touch no bulk data.
: "${SCRATCH_MOUNT:=/scratch-emptydir}"
: "${SCRATCH_SIZE_LIMIT:=200Gi}"
: "${STAGE_FROM:=}"
# /dev/shm for DataLoader worker IPC. The Kubernetes default of 64Mi is too
# small for any large batch or resolution (K-18). 4Gi covers the largest point
# in the 009 sweep (224x224 batch 256: a 154 MiB batch, ~1.85 GiB in flight).
# This is tmpfs, so it is real RAM inside the 16Gi limit -- raise it deliberately, not reflexively.
: "${DSHM_SIZE_LIMIT:=4Gi}"

_POD_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${_POD_COMMON_DIR}/pod-job.template.yaml"

# envsubst is restricted to exactly these names.  Unrestricted, it would
# also rewrite the shell expansions inside the template's args block
# (e.g. "$STAGE_FROM", "$src_bytes") to empty strings before the container
# ever ran.  The template avoids the ${...} brace form for the same reason;
# both layers would have to fail for that to regress.
TEMPLATE_VARS='$JOB_NAME $JOB_CMD_B64 $HOST_PATH $HOME_MOUNT $USER_HOME $REPO_DIR $IMAGE $CPU $MEM $RUN_CLASS $SCRATCH_MOUNT_SUBST $SCRATCH_SIZE_LIMIT $STAGE_FROM_SUBST $DSHM_SIZE_LIMIT'

# Renders the pod manifest for one job into a fresh temp file and sets three
# globals for the caller to use: JOB_NAME, JOB_CMD_B64, MANIFEST.  Pod names
# must be DNS-1123 labels and unique per attempt, so a re-run never collides
# with a pod still terminating from the previous attempt -- hence the
# timestamp suffix.
render_pod_manifest() {
    local job_suffix="$1"
    local job_cmd="$2"

    # The command is passed to the pod base64-encoded (single line, tr -d '\n'
    # strips wrap newlines from either GNU or BSD base64) rather than
    # interpolated into the YAML template directly. A raw multi-line command,
    # or one containing quotes/colons, corrupts the template's YAML block
    # scalar; base64 makes the substitution content-agnostic regardless of
    # what the command contains.
    JOB_CMD_B64="$(printf '%s' "${job_cmd}" | base64 | tr -d '\n')"
    JOB_NAME="shar-${job_suffix}-$(date -u +%Y%m%d%H%M%S)"
    MANIFEST="$(mktemp -t shar-pod-XXXXXX.yaml)"

    export JOB_NAME JOB_CMD_B64 HOST_PATH HOME_MOUNT USER_HOME REPO_DIR IMAGE CPU MEM RUN_CLASS DSHM_SIZE_LIMIT
    SCRATCH_MOUNT_SUBST="${SCRATCH_MOUNT}"
    STAGE_FROM_SUBST="${STAGE_FROM}"
    export SCRATCH_MOUNT SCRATCH_SIZE_LIMIT STAGE_FROM SCRATCH_MOUNT_SUBST STAGE_FROM_SUBST

    envsubst "${TEMPLATE_VARS}" < "${TEMPLATE}" > "${MANIFEST}"
}

# Waits for a pod to leave Pending, printing progress every 5s for up to 10
# minutes.  kubectl wait --for=condition=Ready is not usable here: a short job
# can run to completion before Ready is ever observed, and a Succeeded pod
# never satisfies that condition.  Sets SCHEDULING_PHASE on success; on
# failure to schedule it describes the pod and returns non-zero without
# deleting anything -- deleting a pod that failed to schedule is the caller's
# decision, not this function's.
wait_for_scheduling() {
    local pod="$1"
    local phase=""
    for i in $(seq 1 120); do
        phase="$(kubectl get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
        case "${phase}" in
            Running | Succeeded | Failed) break ;;
        esac
        echo "  ... still ${phase:-unknown} after $((i * 5))s"
        sleep 5
    done
    if [[ "${phase:-}" == "" || "${phase}" == "Pending" ]]; then
        echo "pod did not start; describing and aborting:" >&2
        kubectl describe pod "${pod}" >&2
        return 1
    fi
    SCHEDULING_PHASE="${phase}"
}

# Follows a pod's logs until it reaches a genuinely terminal phase
# (Succeeded/Failed) or disappears, reconnecting automatically in between.
#
# `kubectl logs -f` returning is NOT the same as the pod being done: a
# transient API-server blip, network hiccup, or kubelet log rotation ends the
# stream while the pod keeps running for hours. Treating that return as
# completion is exactly the bug this function exists to close (see
# run_pod_job.sh's header). So every time the stream ends, this polls the
# pod's actual phase and only stops if that phase is terminal or the pod is
# gone; otherwise it reconnects.
#
# --timestamps lets a reconnect resume with --since-time instead of replaying
# the whole log. The resume point is the last timestamp printed, so the one
# line at that exact timestamp may be re-emitted after a reconnect; a rare
# duplicated line is a far smaller cost than either replaying the whole log
# or silently swallowing whatever was said during the gap.
#
# Sets FOLLOW_FINAL_PHASE on return: Succeeded, Failed, or "" (pod gone).
# Reconnect backoff for follow_logs_with_reconnect. A dropped log stream over a
# multi-day run is expected (API-server restart, network blip, log rotation);
# an immediate retry storm is not.
: "${FOLLOW_BACKOFF_START_SECONDS:=2}"
: "${FOLLOW_BACKOFF_MAX_SECONDS:=30}"
# How long to let the pod phase settle after the log stream ends before
# treating the end as a dropped connection rather than a finished container.
: "${FOLLOW_PHASE_SETTLE_TRIES:=5}"

follow_logs_with_reconnect() {
    local pod="$1"
    local last_ts=""
    local backoff="${FOLLOW_BACKOFF_START_SECONDS}"
    FOLLOW_FINAL_PHASE=""

    while true; do
        if [[ -n "${last_ts}" ]]; then
            echo "--- reconnecting to ${pod} logs, resuming after ${last_ts} (stream dropped; the pod itself is untouched) ---" >&2
        fi
        # Process substitution, not a pipe: a pipe would run the while loop in
        # a subshell and last_ts would not survive past this iteration.
        if [[ -n "${last_ts}" ]]; then
            local resume_ts="${last_ts}"
            while IFS= read -r line; do
                # --since-time is inclusive, so the first line back is the one
                # we already printed. Suppress exactly that repeat.
                if [[ "${line%% *}" != "${resume_ts}" ]]; then
                    printf '%s\n' "${line}"
                fi
                if [[ -n "${line}" ]]; then last_ts="${line%% *}"; fi
            done < <(kubectl logs -f --timestamps --since-time="${last_ts}" "${pod}" 2>/dev/null)
        else
            while IFS= read -r line; do
                printf '%s\n' "${line}"
                if [[ -n "${line}" ]]; then last_ts="${line%% *}"; fi
            done < <(kubectl logs -f --timestamps "${pod}" 2>/dev/null)
        fi

        # A container that has just exited closes its log stream a beat before
        # the pod phase flips to Succeeded/Failed. Polling once here would read
        # "Running", announce a dropped stream, and reconnect -- printing an
        # alarming false alarm at the end of every successful run. Give the
        # phase a moment to settle before concluding the stream really dropped.
        for _ in $(seq 1 "${FOLLOW_PHASE_SETTLE_TRIES}"); do
            FOLLOW_FINAL_PHASE="$(kubectl get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
            case "${FOLLOW_FINAL_PHASE}" in
                Succeeded | Failed | "") return 0 ;;
            esac
            sleep 1
        done
        # Still Running (or a brief Pending during a kubelet hiccup): the
        # stream just dropped out from under us. Loop back and reconnect --
        # but back off first. Without this, a drop mode where `kubectl logs -f`
        # returns immediately becomes a tight loop hammering the API server for
        # however many hours the run has left, which is exactly the kind of
        # shared-infrastructure abuse D-39/K-10 exist to prevent. Backoff
        # doubles to a ceiling rather than growing without bound, so a watcher
        # that recovers after a long outage still resumes promptly.
        echo "--- log stream dropped; ${pod} still ${FOLLOW_FINAL_PHASE}, reconnecting in ${backoff}s ---" >&2
        sleep "${backoff}"
        backoff=$(( backoff * 2 ))
        # Written as an `if`, not `[[ ... ]] && ...`: the latter evaluates to
        # non-zero when the condition is false, and as the last command in this
        # loop body that would trip `set -e` and kill the watcher on the very
        # first reconnect that did not need clamping.
        if [[ ${backoff} -gt ${FOLLOW_BACKOFF_MAX_SECONDS} ]]; then
            backoff="${FOLLOW_BACKOFF_MAX_SECONDS}"
        fi
    done
}

# Prints the exitCode/reason diagnosis block for a pod already in a terminal
# phase. Sets EXIT_CODE and TERM_REASON for the caller.
#
# The exit code alone is ambiguous for the failure this cluster produces most:
# 137 is SIGKILL from *any* source, and only .reason distinguishes an
# out-of-memory kill (OOMKilled) from an ordinary one (Error).  Diagnosing a
# 137 without it cost a full debugging cycle on 2026-08-25; see K-15.
diagnose_terminal_pod() {
    local pod="$1"
    local phase="$2"

    EXIT_CODE="$(kubectl get pod "${pod}" \
        -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)"
    TERM_REASON="$(kubectl get pod "${pod}" \
        -o jsonpath='{.status.containerStatuses[0].state.terminated.reason}' 2>/dev/null || true)"

    echo "--- pod phase=${phase:-unknown} exitCode=${EXIT_CODE:-unknown} reason=${TERM_REASON:-unknown} ---"
    if [[ "${TERM_REASON:-}" == "OOMKilled" ]]; then
        echo "    the container exceeded its memory limit (MEM=${MEM}); the account's" >&2
        echo "    ResourceQuota caps limits.memory at 16Gi (K-13), so the fix is to" >&2
        echo "    reduce the job's footprint, not to raise MEM." >&2
    fi
}

# Resolves the pod name for the watcher and stopper: an explicit argument
# wins outright; with none, the only sensible default is "the job I just
# started", so this looks only at pods still Pending/Running (a Succeeded or
# Failed pod from an earlier run is not that, and would otherwise poison a
# plain "most recent by name" pick). Exactly one such pod resolves silently;
# zero or several is refused rather than guessed -- printed to stderr, exit
# code 1 or 2 respectively, echoed name on stdout only on success.
resolve_pod_name() {
    local given="${1:-}"
    if [[ -n "${given}" ]]; then
        printf '%s\n' "${given}"
        return 0
    fi

    local -a candidates=()
    local line
    while IFS= read -r line; do
        [[ -n "${line}" ]] && candidates+=("${line}")
    done < <(kubectl get pods --no-headers \
        -o custom-columns='NAME:.metadata.name,PHASE:.status.phase' 2>/dev/null \
        | awk '$1 ~ /^shar-/ && ($2 == "Running" || $2 == "Pending") {print $1}')

    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "no Pending/Running pod named shar-* found" >&2
        return 1
    fi
    if [[ ${#candidates[@]} -gt 1 ]]; then
        echo "more than one active shar-* pod; pass the one you mean:" >&2
        printf '  %s\n' "${candidates[@]}" >&2
        return 2
    fi
    printf '%s\n' "${candidates[0]}"
}
