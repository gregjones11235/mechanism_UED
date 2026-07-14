#!/bin/bash
# L5 B2/B3: Test which LLM roles add value (tutor-only vs tutor+critic)
# Uses diverse pool + fixed cache. Runs on GPU 1.
# Launch AFTER C1/C2 finishes.

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dicode310
cd /root/experiments/dreaming-in-code-coop
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export PYTHONPATH=/root/experiments/dreaming-in-code-coop/src:/root/experiments/dreaming-in-code-coop

GRAPH=/root/experiments/dreaming-in-code-coop/task_graph_synthetic.graphml
LOG_DIR=/root/experiments/dicode_runs/aggregation/logs/l5_remaining
mkdir -p "$LOG_DIR"

COMMON="training.total_timesteps=500000 training.num_envs=64 training.num_steps=64 evaluation.num_envs=64 evaluation.num_steps=512 dicode_manager.max_updates_per_session=2 dicode_manager.training_sample_size_n=8 dicode_manager.active_task_capacity=16 dicode_manager.num_generation_tasks=0 dicode_manager.additional_num_parents=0 seed=1 aggregation.enabled=true aggregation.use_llm_cache=true aggregation.retention_trigger=0.15 gen_manager.graph_path=${GRAPH}"

echo "=== L5 B2/B3: Role ablation ==="
echo "Start: $(date)"

# B2: tutor-only
echo "=== B2 tutor-only (Qwen progression, no critic/explorer) ==="
python experiments/training/run_dicode.py ${COMMON} \
    aggregation.mode=soft_copeland \
    aggregation.llm_weight_novelty=0.0 \
    aggregation.llm_weight_critic=0.0 \
    aggregation.llm_weight_retention=0.0 \
    2>&1 | tee "${LOG_DIR}/B2_tutor_only.log"
echo "B2 done: $(date)"

# B3: tutor + critic
echo "=== B3 tutor+critic (Qwen + DeepSeek, no GLM explorer) ==="
python experiments/training/run_dicode.py ${COMMON} \
    aggregation.mode=soft_copeland \
    aggregation.llm_weight_novelty=0.0 \
    aggregation.llm_weight_critic=0.5 \
    aggregation.llm_weight_retention=0.2 \
    2>&1 | tee "${LOG_DIR}/B3_tutor_critic.log"
echo "B3 done: $(date)"

# Analysis
cd /root/experiments/dreaming-in-code-coop
PYTHONPATH=src:$PYTHONPATH python scripts/summarize_aggregation_runs.py \
    --log-dir /root/experiments/dicode_runs/aggregation/logs \
    --output-dir /root/experiments/dicode_runs/aggregation
PYTHONPATH=src:$PYTHONPATH python scripts/summarize_llm_collaboration_runs.py
echo "=== L5 complete: $(date) ==="
