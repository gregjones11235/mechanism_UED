#!/bin/bash
# ==============================================================================
# Sequential Sweep Launcher
#
# Runs aggregation experiments one at a time on GPU 1.
# Each run waits for the previous to complete before starting.
#
# Usage:
#   bash scripts/run_sequential_sweep.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/root/experiments/dicode_runs/aggregation/logs/sweep"
mkdir -p "$LOG_DIR"

# Conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dicode310

# Fixed parameters
GPU=1
SEED=1
TOTAL_TIMESTEPS=500000
NUM_ENVS=64
NUM_STEPS=64
EVAL_ENVS=64
EVAL_STEPS=512
MAX_UPDATES=2
SAMPLE_SIZE=4
CAPACITY=16

# Sweep configurations: mode trigger run_name
CONFIGS=(
    # Mode                          Trigger  RunName
    "robust_weighted                0.15     s1_robust_weighted_t0.15"
    "soft_copeland                  0.15     s2_soft_copeland_t0.15"
    "budgeted_retention_trigger     0.15     s3_budgeted_retention_t0.15"
    "budgeted_retention_trigger     0.10     s4_budgeted_retention_t0.10"
    "entropy_regularized            0.15     s5_entropy_regularized_t0.15"
    "raw_weighted                   0.15     s6_raw_weighted_t0.15"
)

echo "=============================================="
echo "Sequential Aggregation Sweep"
echo "=============================================="
echo "Total runs: ${#CONFIGS[@]}"
echo "Start time: $(date)"
echo "=============================================="

RUN_NUM=0
for CONFIG in "${CONFIGS[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))
    read -r MODE TRIGGER RUN_NAME <<< "$CONFIG"

    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

    echo ""
    echo "=============================================="
    echo "[${RUN_NUM}/${#CONFIGS[@]}] Starting: ${RUN_NAME}"
    echo "  Mode: ${MODE}, Trigger: ${TRIGGER}"
    echo "  Log: ${LOG_FILE}"
    echo "  Time: $(date)"
    echo "=============================================="

    # Clean up previous graph
    rm -f "${REPO_ROOT}/task_graph.graphml"

    # Build override string
    OVERRIDES="training.total_timesteps=${TOTAL_TIMESTEPS}"
    OVERRIDES="${OVERRIDES} training.num_envs=${NUM_ENVS}"
    OVERRIDES="${OVERRIDES} training.num_steps=${NUM_STEPS}"
    OVERRIDES="${OVERRIDES} evaluation.num_envs=${EVAL_ENVS}"
    OVERRIDES="${OVERRIDES} evaluation.num_steps=${EVAL_STEPS}"
    OVERRIDES="${OVERRIDES} dicode_manager.max_updates_per_session=${MAX_UPDATES}"
    OVERRIDES="${OVERRIDES} dicode_manager.training_sample_size_n=${SAMPLE_SIZE}"
    OVERRIDES="${OVERRIDES} dicode_manager.active_task_capacity=${CAPACITY}"
    OVERRIDES="${OVERRIDES} aggregation.enabled=true"
    OVERRIDES="${OVERRIDES} aggregation.mode=${MODE}"
    OVERRIDES="${OVERRIDES} aggregation.retention_trigger=${TRIGGER}"
    OVERRIDES="${OVERRIDES} seed=${SEED}"
    OVERRIDES="${OVERRIDES} dicode_manager.num_generation_tasks=0"

    if [ "${MODE}" = "entropy_regularized" ]; then
        OVERRIDES="${OVERRIDES} aggregation.entropy_regularization=0.1"
    fi

    # Run experiment (foreground, sequential)
    START_TIME=$(date +%s)

    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES=${GPU} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_MODE=disabled \
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
    python experiments/training/run_dicode.py ${OVERRIDES} 2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "${EXIT_CODE}" > "${LOG_DIR}/${RUN_NAME}.exit"
    echo "[${RUN_NUM}/${#CONFIGS[@]}] Finished: ${RUN_NAME} (exit=${EXIT_CODE}, duration=${DURATION}s)"
done

echo ""
echo "=============================================="
echo "Sweep complete."
echo "End time: $(date)"
echo "=============================================="
