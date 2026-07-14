#!/bin/bash
# Fast 3-mode sweep: ~200K steps each, no LLM generation
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dicode310

LOG_DIR="/root/experiments/dicode_runs/aggregation/logs/sweep"
mkdir -p "$LOG_DIR"
REPO="/root/experiments/dreaming-in-code-coop"

MODES=("robust_weighted" "soft_copeland" "budgeted_retention_trigger")
TRIGGER="0.15"
STEPS=200000

for MODE in "${MODES[@]}"; do
    RUN_NAME="fast_${MODE}_t${TRIGGER}"
    LOG="${LOG_DIR}/${RUN_NAME}.log"
    rm -f "${REPO}/task_graph.graphml"

    echo "=== $(date) Starting ${RUN_NAME} ===" | tee -a "$LOG"

    cd "$REPO"
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_MODE=disabled \
    PYTHONPATH="${REPO}/src:${REPO}" \
    python experiments/training/run_dicode.py \
        training.total_timesteps=${STEPS} \
        training.num_envs=64 \
        training.num_steps=64 \
        evaluation.num_envs=64 \
        evaluation.num_steps=512 \
        dicode_manager.max_updates_per_session=2 \
        dicode_manager.training_sample_size_n=4 \
        dicode_manager.active_task_capacity=16 \
        dicode_manager.num_generation_tasks=0 \
        dicode_manager.additional_num_parents=0 \
        aggregation.enabled=true \
        aggregation.mode=${MODE} \
        aggregation.retention_trigger=${TRIGGER} \
        seed=1 \
        2>&1 | tee -a "${LOG}"

    RC=${PIPESTATUS[0]}
    echo "=== $(date) Finished ${RUN_NAME} exit=${RC} ===" | tee -a "$LOG"
done

echo "=== $(date) Sweep complete ==="
