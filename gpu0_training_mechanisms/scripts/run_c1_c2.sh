#!/bin/bash
# C1/C2: soft_copeland vs budgeted_soft_copeland head-to-head
# Synthetic diverse pool, LLM cache enabled
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dicode310

LOG_DIR="/root/experiments/dicode_runs/aggregation/logs/c1_c2"
mkdir -p "$LOG_DIR"
REPO="/root/experiments/dreaming-in-code-coop"

COMMON="training.total_timesteps=5000000 training.num_envs=64 training.num_steps=64 evaluation.num_envs=64 evaluation.num_steps=512 dicode_manager.max_updates_per_session=2 dicode_manager.training_sample_size_n=8 dicode_manager.active_task_capacity=16 dicode_manager.num_generation_tasks=0 dicode_manager.additional_num_parents=0 seed=1 aggregation.enabled=true aggregation.use_llm_cache=true aggregation.llm_weight_progression=0.5 aggregation.llm_weight_novelty=0.5 aggregation.llm_weight_critic=0.5 aggregation.llm_weight_retention=0.2 aggregation.retention_trigger=0.15"

echo "=== C1/C2 Comparison ==="
echo "Start: $(date)"

# C1: soft_copeland (no caps)
echo ""
echo "=== $(date) Starting C1: soft_copeland ==="
rm -f "${REPO}/task_graph.graphml"
# Create synthetic pool for this run
cd "$REPO"
PYTHONPATH=src:$PYTHONPATH python scripts/create_synthetic_pool.py 2>&1 | tail -3

CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=disabled \
PYTHONPATH="${REPO}/src:${REPO}" \
python experiments/training/run_dicode.py \
    ${COMMON} \
    aggregation.mode=soft_copeland \
    2>&1 | tee "${LOG_DIR}/C1_soft_copeland.log"
echo "C1 exit: ${PIPESTATUS[0]}" | tee -a "${LOG_DIR}/C1_soft_copeland.log"

# C2: budgeted_soft_copeland
echo ""
echo "=== $(date) Starting C2: budgeted_soft_copeland ==="
rm -f "${REPO}/task_graph.graphml"
cd "$REPO"
PYTHONPATH=src:$PYTHONPATH python scripts/create_synthetic_pool.py 2>&1 | tail -3

CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=disabled \
PYTHONPATH="${REPO}/src:${REPO}" \
python experiments/training/run_dicode.py \
    ${COMMON} \
    aggregation.mode=budgeted_soft_copeland \
    aggregation.max_source_share=0.4 \
    aggregation.max_signal_share=0.5 \
    2>&1 | tee "${LOG_DIR}/C2_budgeted_soft_copeland.log"
echo "C2 exit: ${PIPESTATUS[0]}" | tee -a "${LOG_DIR}/C2_budgeted_soft_copeland.log"

echo ""
echo "=== C1/C2 Complete: $(date) ==="
