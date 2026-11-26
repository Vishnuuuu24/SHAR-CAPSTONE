#!/usr/bin/env bash
# Run in a frozen snapshot, inside the account's single GPU pod.
# Usage: bash scripts/dgx/run_detector_followup.sh <main-data-root> <snapshot-sha256>
set -euo pipefail
DATA_ROOT="$1"
SOURCE_ID="$2"
RUN_ROOT="$DATA_ROOT/results/detect/followup-20261112"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec > >(tee -a "$RUN_ROOT/full-002.log") 2>&1
PYTHON="$DATA_ROOT/.venv/bin/python"
"$PYTHON" -m unittest discover -s tests -p test_detector_followup.py
"$PYTHON" -m unittest discover -s tests -p test_detect.py
if [[ ! -e "$RUN_ROOT/rwf-002/features.json" ]]; then
  "$PYTHON" scripts/run_rwf2000.py features --data-root "$DATA_ROOT" \
    --intake "$RUN_ROOT/intake-004/intake.json" --output "$RUN_ROOT/rwf-002" --code-revision "$SOURCE_ID"
fi
if [[ ! -e "$RUN_ROOT/rwf-002/test_summary.json" ]]; then
  "$PYTHON" scripts/run_rwf2000.py evaluate --data-root "$DATA_ROOT" \
    --intake "$RUN_ROOT/intake-004/intake.json" --output "$RUN_ROOT/rwf-002" --code-revision "$SOURCE_ID"
fi
"$PYTHON" scripts/run_detector_followup.py train --data-root "$DATA_ROOT" \
  --output "$RUN_ROOT/ucf-002" --code-revision "$SOURCE_ID"
"$PYTHON" scripts/run_detector_followup.py evaluate --data-root "$DATA_ROOT" \
  --output "$RUN_ROOT/ucf-002" --code-revision "$SOURCE_ID"
echo 'All D-77 declared rows finished. No additional training or test reads are queued.'
