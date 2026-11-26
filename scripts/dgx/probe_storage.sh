#!/usr/bin/env bash
# Measure real DGX storage facts before P1C uses any of them for the packed
# frame blob. See docs/DECISIONS.md D-33 and the P1C execution work item.
#
# This is a read-mostly measurement pod, not a training or calibration job:
# it never launches kubectl against a GPU workload and it writes only its own
# small test files (never anything under Datasets/).
#
# Usage:
#   scripts/dgx/probe_storage.sh [job-name-suffix]
#
# What it measures, all as real numbers written to one JSON artifact under
# results/p1c/ on the /home NFS export (so the file survives pod deletion):
#   - does /workspace/fast-dgx-storage/<account> mount, and is it writable?
#   - filesystem type of that path vs. /home vs. an emptyDir (the practical
#     node-local-vs-shared signal available to a single pod: a network
#     filesystem such as nfs/nfs4/cifs is shared by construction; a local
#     block filesystem such as xfs/ext4 on a hostPath is node-local unless
#     the cluster additionally guarantees the same backing device on both
#     workers, which this probe cannot see from inside one pod)
#   - a marker file with this pod's node name, so that running this script a
#     second time (ideally scheduled onto the other of the two workers) lets
#     a human diff the two resulting JSON artifacts' node_name and
#     previous_marker_found fields for a direct cross-node confirmation
#   - sequential write/read throughput on all three targets
#   - whether Kyverno admits an emptyDir volume at all (this pod would never
#     have started otherwise) and what capacity `df` reports for it
#   - open+read latency for a random sample of real 64x64 PNG files under
#     the /home-mounted `Datasets/UCF Crime Dataset Kaggle/Train` tree
#
# THIS SCRIPT IS WRITE-ONLY DELIVERABLE FOR NOW: per the work item that
# produced it, it must not be executed as part of that task. Run it only
# once the owner has reviewed it and kubectl/cluster access is confirmed.
#
# Cluster configuration mirrors scripts/dgx/run_pod_job.sh; see that script
# and scripts/dgx/README.md for NFS_SERVER/HOST_PATH background.

set -euo pipefail

JOB_SUFFIX="${1:-probe-storage}"

# --- cluster configuration (same defaults as run_pod_job.sh) ---------------
: "${USER_HOME:=<home>}"
: "${HOST_PATH:=${USER_HOME}}"
: "${HOME_MOUNT:=${USER_HOME}}"
: "${REPO_DIR:=${USER_HOME}/SHAR}"
: "${IMAGE:=<head-node>/nvcr.io/nvidia/pytorch:24.10-py3}"
: "${CPU:=2}"
: "${MEM:=4Gi}"

# --- probe-specific configuration -------------------------------------------
# The untested Kyverno-allowlisted path (see scripts/dgx/README.md's
# allowlist and this work item's background). Mounted to itself, matching
# the constraint documented for HOST_PATH in run_pod_job.sh.
: "${FAST_STORAGE_PATH:=/workspace/fast-dgx-storage/<user>}"
: "${SCRATCH_MOUNT:=/scratch-emptydir}"
: "${SCRATCH_SIZE_LIMIT:=4Gi}"
: "${IO_TEST_MEBIBYTES:=256}"
: "${LATENCY_SAMPLE_SIZE:=300}"

RUN_ID="probe-storage-$(date -u +%Y%m%dT%H%M%SZ)"
# OUT_PATH is a REPO_DIR-relative-turned-absolute path inside the pod's
# container filesystem; REPO_DIR is bind-mounted from /home, so this file
# exists on durable NFS storage under the repository's own results/ tree
# once the pod finishes, independent of the pod's own lifetime.
OUT_PATH="${REPO_DIR}/results/p1c/${RUN_ID}/probe_storage.json"

# --- the in-pod probe program ------------------------------------------------
# A single Python heredoc rather than several small shell utilities: it needs
# percentile arithmetic (statistics module) and structured JSON output, and
# keeping every measurement in one script means one process, one consistent
# clock, and one place that decides what "writable"/"mounts" mean.
read -r -d '' PROBE_PY <<'PYEOF' || true
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

REPO_DIR = Path(os.environ["REPO_DIR"])
FAST_STORAGE = Path(os.environ["FAST_STORAGE_PATH"])
HOME_DIR = Path(os.environ["USER_HOME"])
SCRATCH = Path(os.environ["SCRATCH_MOUNT"])
NODE_NAME = os.environ.get("NODE_NAME", "unknown")
POD_NAME = os.environ.get("POD_NAME", "unknown")
IO_TEST_MEBIBYTES = int(os.environ.get("IO_TEST_MEBIBYTES", "256"))
LATENCY_SAMPLE_SIZE = int(os.environ.get("LATENCY_SAMPLE_SIZE", "300"))
DATASET_TRAIN_ROOT = REPO_DIR / "Datasets" / "UCF Crime Dataset Kaggle" / "Train"
OUT_PATH = Path(os.environ["OUT_PATH"])


def sh(cmd: str) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except Exception as exc:  # noqa: BLE001 - a probe records failures, it does not raise them
        return -1, "", str(exc)


def filesystem_type(path: Path) -> str:
    rc, out, _ = sh(f"findmnt -T {str(path)!r} -no FSTYPE,SOURCE,TARGET 2>/dev/null")
    if rc == 0 and out:
        return out
    rc, out, err = sh(f"stat -f -c '%T' {str(path)!r}")
    return out or f"unknown ({err})"


def probe_target(path: Path, *, create_if_missing: bool) -> dict:
    result: dict = {"path": str(path)}
    existed_before = path.is_dir()
    result["existed_before_probe"] = existed_before
    exists = existed_before
    if not exists and create_if_missing:
        try:
            path.mkdir(parents=True, exist_ok=True)
            exists = path.is_dir()
            result["created_by_probe"] = True
        except Exception as exc:  # noqa: BLE001
            result["mkdir_error"] = str(exc)
    result["mounts"] = exists
    if not exists:
        result["writable"] = False
        return result
    result["filesystem"] = filesystem_type(path)

    marker_path = path / "shar_probe_node_marker.json"
    previous = None
    if marker_path.is_file():
        try:
            previous = json.loads(marker_path.read_text())
        except Exception as exc:  # noqa: BLE001
            previous = {"unreadable": str(exc)}
    result["previous_marker_found"] = previous
    if previous is None:
        result["node_locality_signal"] = (
            "no previous marker at this path; rerun this probe (ideally scheduled onto "
            "the other worker) and compare node_name against this run's marker"
        )
    elif previous.get("node_name") == NODE_NAME:
        result["node_locality_signal"] = (
            "same node_name as the previous marker; consistent with node-local storage "
            "but not conclusive from a single node alone -- confirm with a run on the other worker"
        )
    else:
        result["node_locality_signal"] = (
            f"previous marker was written from node_name={previous.get('node_name')!r}, this run is "
            f"on {NODE_NAME!r}; this path IS shared across at least these two workers"
        )

    test_file = path / f"shar_probe_iotest_{POD_NAME}.bin"
    try:
        block = os.urandom(1024 * 1024)
        started = time.perf_counter()
        with open(test_file, "wb") as handle:
            for _ in range(IO_TEST_MEBIBYTES):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        write_seconds = time.perf_counter() - started
        result["writable"] = True
        result["sequential_write_mib_per_s"] = round(IO_TEST_MEBIBYTES / write_seconds, 2)

        reads = []
        for _ in range(2):
            started = time.perf_counter()
            with open(test_file, "rb") as handle:
                while handle.read(4 * 1024 * 1024):
                    pass
            reads.append(IO_TEST_MEBIBYTES / (time.perf_counter() - started))
        result["sequential_read_mib_per_s_pass1"] = round(reads[0], 2)
        result["sequential_read_mib_per_s_pass2_page_cache_warm"] = round(reads[1], 2)
        result["read_caveat"] = (
            "measured from an unprivileged container with no ability to drop the page cache; "
            "pass1 is only 'likely cold' for a freshly written file, not guaranteed cold, and both "
            "passes may be inflated relative to a truly cold read on a busy shared filesystem"
        )
    except Exception as exc:  # noqa: BLE001
        result["writable"] = False
        result["io_error"] = str(exc)
    finally:
        try:
            test_file.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    try:
        marker_path.write_text(json.dumps({
            "node_name": NODE_NAME,
            "pod_name": POD_NAME,
            "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        result["marker_written"] = True
    except Exception as exc:  # noqa: BLE001
        result["marker_written"] = False
        result["marker_error"] = str(exc)
    return result


report: dict = {
    "kind": "P1C DGX storage probe (D-33 background); measurement only, no training/calibration",
    "probe_id": "RUN_ID_PLACEHOLDER",
    "node_name": NODE_NAME,
    "pod_name": POD_NAME,
    "kernel": platform.platform(),
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "io_test_mebibytes": IO_TEST_MEBIBYTES,
}

report["targets"] = {
    "fast_dgx_storage": probe_target(FAST_STORAGE, create_if_missing=True),
    "home_nfs": probe_target(HOME_DIR, create_if_missing=False),
}

empty_result: dict = {
    "path": str(SCRATCH),
    "kyverno_permitted": SCRATCH.is_dir(),
    "note": (
        "kyverno_permitted is inferred from pod admission having succeeded with this emptyDir "
        "volume already declared in the pod spec; if Kyverno rejected emptyDir outright the pod "
        "would never have been admitted and this JSON artifact would not exist at all"
    ),
}
if SCRATCH.is_dir():
    _, df_out, _ = sh(f"df -Pk {str(SCRATCH)!r}")
    empty_result["df_-Pk_output"] = df_out
    empty_result.update(probe_target(SCRATCH, create_if_missing=False))
report["targets"]["emptydir"] = empty_result

latency: dict = {"dataset_root": str(DATASET_TRAIN_ROOT), "requested_sample_size": LATENCY_SAMPLE_SIZE}
try:
    # --random-source=/dev/zero makes the sample deterministic across runs.
    rc, out, err = sh(
        f"find {str(DATASET_TRAIN_ROOT)!r} -type f -name '*.png' | "
        f"shuf -n {LATENCY_SAMPLE_SIZE} --random-source=/dev/zero"
    )
    files = [line for line in out.splitlines() if line]
    latency["actual_sample_size"] = len(files)
    if rc != 0 and not files:
        latency["find_or_shuf_error"] = err
    samples_ms = []
    for filepath in files:
        started = time.perf_counter()
        try:
            with open(filepath, "rb") as handle:
                handle.read()
        except Exception:  # noqa: BLE001
            continue
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    if samples_ms:
        samples_ms.sort()

        def pct(p: float) -> float:
            index = min(len(samples_ms) - 1, int(round(p / 100 * (len(samples_ms) - 1))))
            return round(samples_ms[index], 4)

        latency["open_read_latency_ms"] = {
            "count": len(samples_ms),
            "min": round(min(samples_ms), 4),
            "mean": round(statistics.mean(samples_ms), 4),
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
            "max": round(max(samples_ms), 4),
        }
    else:
        latency["error"] = "no sampled files could be opened and read"
except Exception as exc:  # noqa: BLE001
    latency["error"] = str(exc)
report["small_file_latency_over_home_nfs"] = latency

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(f"WROTE {OUT_PATH}")
PYEOF

# Substitute the run id into the probe_id field without touching the Python
# source's own quoting (the placeholder is plain text, not an f-string).
PROBE_PY="${PROBE_PY//RUN_ID_PLACEHOLDER/${RUN_ID}}"

JOB_CMD="$(cat <<BASHEOF
set -uo pipefail
python3 - <<'PYEOF'
${PROBE_PY}
PYEOF
BASHEOF
)"

# Same content-agnostic base64 pattern as run_pod_job.sh: the command above
# is multi-line and contains quotes, so interpolating it directly into the
# pod template's YAML block scalar would corrupt it.
JOB_CMD_B64="$(printf '%s' "${JOB_CMD}" | base64 | tr -d '\n')"

JOB_NAME="shar-${JOB_SUFFIX}-$(date -u +%Y%m%d%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/probe-storage-pod.template.yaml"
MANIFEST="$(mktemp -t shar-probe-pod-XXXXXX.yaml)"

export JOB_NAME JOB_CMD_B64 HOST_PATH HOME_MOUNT USER_HOME REPO_DIR IMAGE CPU MEM
export FAST_STORAGE_PATH SCRATCH_MOUNT SCRATCH_SIZE_LIMIT OUT_PATH

cleanup() {
    local status=$?
    echo
    echo "--- cleaning up ${JOB_NAME} ---"
    kubectl delete pod "${JOB_NAME}" --ignore-not-found --wait=false || true
    rm -f "${MANIFEST}"
    exit "${status}"
}
trap cleanup EXIT INT TERM

envsubst < "${TEMPLATE}" > "${MANIFEST}"

echo "--- applying ${JOB_NAME} ---"
kubectl apply -f "${MANIFEST}"

echo "--- waiting for scheduling ---"
for i in $(seq 1 120); do
    phase="$(kubectl get pod "${JOB_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    case "${phase}" in
        Running | Succeeded | Failed) break ;;
    esac
    echo "  ... still ${phase:-unknown} after $((i * 5))s"
    sleep 5
done

if [[ "${phase:-}" == "" || "${phase}" == "Pending" ]]; then
    echo "pod did not start; describing and aborting:" >&2
    kubectl describe pod "${JOB_NAME}" >&2
    exit 1
fi

echo "--- logs (${JOB_NAME}) ---"
kubectl logs -f "${JOB_NAME}" || true

echo "--- waiting for terminal pod phase ---"
for i in $(seq 1 60); do
    phase="$(kubectl get pod "${JOB_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    case "${phase}" in
        Succeeded | Failed) break ;;
    esac
    echo "  ... still ${phase:-unknown} after $((i * 5))s"
    sleep 5
done

EXIT_CODE="$(kubectl get pod "${JOB_NAME}" \
    -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)"

echo "--- pod phase=${phase:-unknown} exitCode=${EXIT_CODE:-unknown} ---"
echo "--- result artifact (on /home, survives pod deletion): ${OUT_PATH} ---"

if [[ "${EXIT_CODE:-}" != "0" ]]; then
    exit "${EXIT_CODE:-1}"
fi
