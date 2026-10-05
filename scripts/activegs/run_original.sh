#!/usr/bin/env bash
# Execute the author's pipeline; this verifies execution, not the paper's table.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${ACTIVEGS_ROOT:?Set ACTIVEGS_ROOT to the pinned upstream checkout}"
: "${ACTIVEGS_PYTHON:?Set ACTIVEGS_PYTHON to the isolated Python executable}"
: "${RUN_DIR:?Set RUN_DIR to a new absolute experiment directory}"
GPU=${GPU:-0}
BUDGET=${BUDGET:-300}
RECORD_INTERVAL=${RECORD_INTERVAL:-60}
if [[ "$RUN_DIR" != /* || -e "$RUN_DIR" ]]; then
    printf 'RUN_DIR must be a new absolute directory\n' >&2
    exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/evidence"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export TORCH_HOME="$RUN_DIR/../torch-cache"
cd "$ACTIVEGS_ROOT"
"$ACTIVEGS_PYTHON" -m pip freeze > "$RUN_DIR/evidence/pip-freeze.txt"
"$ACTIVEGS_PYTHON" "$SCRIPT_DIR/preflight.py" "$ACTIVEGS_ROOT" "$RUN_DIR/evidence" > "$RUN_DIR/logs/preflight.log" 2>&1
"$ACTIVEGS_PYTHON" data_generation.py scene=replica/office0 use_gui=false \
    dataset_path="$RUN_DIR/dataset" > "$RUN_DIR/logs/data_generation.log" 2>&1
"$ACTIVEGS_PYTHON" main.py planner=confidence scene=replica/office0 use_gui=false \
    experiment.output_dir="$RUN_DIR/experiments" experiment.exp_id=original experiment.run_id=0 \
    experiment.budget="$BUDGET" experiment.record_interval="$RECORD_INTERVAL" \
    > "$RUN_DIR/logs/main.log" 2>&1
"$ACTIVEGS_PYTHON" mesh_generation.py planner=confidence scene=replica/office0 \
    experiment.output_dir="$RUN_DIR/experiments" experiment.exp_id=original experiment.run_id=0 \
    > "$RUN_DIR/logs/mesh_generation.log" 2>&1
"$ACTIVEGS_PYTHON" eval.py planner=confidence scene=replica/office0 eval_mode=mesh \
    experiment.output_dir="$RUN_DIR/experiments" experiment.exp_id=original experiment.run_id=0 \
    test_folder="$RUN_DIR/dataset/replica/office0" > "$RUN_DIR/logs/eval.log" 2>&1
"$ACTIVEGS_PYTHON" "$SCRIPT_DIR/check_artifacts.py" \
    "$RUN_DIR/experiments/original/replica/office0/confidence/0" \
    --output "$RUN_DIR/evidence/artifact-check.json"
