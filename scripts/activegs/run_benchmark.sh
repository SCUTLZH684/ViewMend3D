#!/usr/bin/env bash
# Controlled protocol; the author's original entry point remains separate.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${ACTIVEGS_ROOT:?Set ACTIVEGS_ROOT to the pinned upstream checkout}"
: "${ACTIVEGS_PYTHON:?Set ACTIVEGS_PYTHON to the isolated Python executable}"
: "${RUN_DIR:?Set RUN_DIR to a new absolute experiment directory}"
GPU=${GPU:-0}
IFS=',' read -r -a BENCHMARK_METHOD_ARRAY <<< "${BENCHMARK_METHODS:-confidence_nooracle,random_matched,defect}"
IFS=',' read -r -a BENCHMARK_SEED_ARRAY <<< "${BENCHMARK_SEEDS:-0,1,2}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export MPLBACKEND=Agg PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TORCH_HOME="$(dirname -- "$RUN_DIR")/torch-cache"
exec "$ACTIVEGS_PYTHON" "$SCRIPT_DIR/run_benchmark.py" \
    --upstream "$ACTIVEGS_ROOT" --run-dir "$RUN_DIR" --gpu "$GPU" \
    --methods "${BENCHMARK_METHOD_ARRAY[@]}" --seeds "${BENCHMARK_SEED_ARRAY[@]}" \
    --frames "${BENCHMARK_FRAMES:-60}" --prefix-frames "${BENCHMARK_PREFIX:-20}" \
    --protocol "${BENCHMARK_PROTOCOL:-observations}" --budget "${BENCHMARK_BUDGET:-180}" "$@"
