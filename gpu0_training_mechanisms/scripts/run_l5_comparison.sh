#!/bin/bash
# Stage L5: Short cost-aware comparison
# Compares rule-based vs LLM-enhanced aggregation
# 500K steps each, no API calls during training, sequential execution
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dicode310

LOG_DIR="/root/experiments/dicode_runs/aggregation/logs/l5_comparison"
mkdir -p "$LOG_DIR"
REPO="/root/experiments/dreaming-in-code-coop"
STEPS=500000

# B0 and B1 already done in fast sweep — we'll reference those results
# Running B4 and B5 with LLM cache enabled

declare -A RUNS
RUNS=(
    ["l5_B4_llm_soft_copeland"]="aggregation.enabled=true aggregation.mode=soft_copeland aggregation.use_llm_cache=true aggregation.llm_weight_progression=0.5 aggregation.llm_weight_novelty=0.5 aggregation.llm_weight_critic=0.5 aggregation.llm_weight_retention=0.2 aggregation.retention_trigger=0.15"
    ["l5_B5_llm_budgeted_soft_copeland"]="aggregation.enabled=true aggregation.mode=budgeted_soft_copeland aggregation.use_llm_cache=true aggregation.llm_weight_progression=0.5 aggregation.llm_weight_novelty=0.5 aggregation.llm_weight_critic=0.5 aggregation.llm_weight_retention=0.2 aggregation.retention_trigger=0.15"
    ["l5_B5b_llm_entropy_regularized"]="aggregation.enabled=true aggregation.mode=entropy_regularized aggregation.use_llm_cache=true aggregation.llm_weight_progression=0.5 aggregation.llm_weight_novelty=0.5 aggregation.llm_weight_critic=0.5 aggregation.llm_weight_retention=0.2 aggregation.retention_trigger=0.15 aggregation.entropy_regularization=0.1"
)

COMMON="training.total_timesteps=${STEPS} training.num_envs=64 training.num_steps=64 evaluation.num_envs=64 evaluation.num_steps=512 dicode_manager.max_updates_per_session=2 dicode_manager.training_sample_size_n=4 dicode_manager.active_task_capacity=16 dicode_manager.num_generation_tasks=0 dicode_manager.additional_num_parents=0 seed=1"

echo "=== L5 Comparison Sweep ==="
echo "Start: $(date)"
echo "Runs: ${!RUNS[@]}"

for RUN_NAME in "${!RUNS[@]}"; do
    LOG="${LOG_DIR}/${RUN_NAME}.log"
    rm -f "${REPO}/task_graph.graphml"

    echo ""
    echo "=== $(date) Starting ${RUN_NAME} ===" | tee -a "$LOG"

    cd "$REPO"
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_MODE=disabled \
    PYTHONPATH="${REPO}/src:${REPO}" \
    python experiments/training/run_dicode.py \
        ${COMMON} \
        ${RUNS[$RUN_NAME]} \
        2>&1 | tee -a "${LOG}"

    RC=${PIPESTATUS[0]}
    echo "=== $(date) Finished ${RUN_NAME} exit=${RC} ===" | tee -a "$LOG"
done

echo ""
echo "=== L5 Complete: $(date) ==="
