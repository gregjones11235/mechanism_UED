#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dicode310
cd /root/experiments/dreaming-in-code-coop
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export PYTHONPATH=/root/experiments/dreaming-in-code-coop/src:/root/experiments/dreaming-in-code-coop

SYNTH_GRAPH="/root/experiments/dreaming-in-code-coop/task_graph_synthetic.graphml"
COMMON="training.total_timesteps=5000000 training.num_envs=64 training.num_steps=64 evaluation.num_envs=64 evaluation.num_steps=512 dicode_manager.max_updates_per_session=2 dicode_manager.training_sample_size_n=8 dicode_manager.active_task_capacity=16 dicode_manager.num_generation_tasks=0 dicode_manager.additional_num_parents=0 seed=1 aggregation.enabled=true aggregation.use_llm_cache=true aggregation.llm_weight_progression=0.5 aggregation.llm_weight_novelty=0.5 aggregation.llm_weight_critic=0.5 aggregation.llm_weight_retention=0.2 aggregation.retention_trigger=0.15 gen_manager.graph_path=${SYNTH_GRAPH}"

LOG_DIR="/root/experiments/dicode_runs/aggregation/logs/c1_c2"
mkdir -p "$LOG_DIR"

if [ ! -f "$SYNTH_GRAPH" ]; then
    echo "ERROR: $SYNTH_GRAPH not found"
    exit 1
fi

echo "=== C1: soft_copeland (no caps) ==="
echo "Start: $(date)"

python experiments/training/run_dicode.py \
    ${COMMON} \
    aggregation.mode=soft_copeland \
    2>&1 | tee "${LOG_DIR}/C1_soft_copeland.log"

echo "C1 done: $(date)"

echo "=== C2: budgeted_soft_copeland (source caps) ==="
echo "Start: $(date)"

python experiments/training/run_dicode.py \
    ${COMMON} \
    aggregation.mode=budgeted_soft_copeland \
    aggregation.max_source_share=0.4 \
    2>&1 | tee "${LOG_DIR}/C2_budgeted_soft_copeland.log"

echo "C2 done: $(date)"
echo "=== All done ==="
