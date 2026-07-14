#!/bin/bash
# Stage L5 remaining: B2 (tutor-only) and B3 (tutor+critic)
# Uses diverse pool (4 real Craftax tasks) + fixed LLM cache
# Run AFTER C1/C2 completes on GPU 1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dicode310
cd /root/experiments/dreaming-in-code-coop
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export PYTHONPATH=/root/experiments/dreaming-in-code-coop/src:/root/experiments/dreaming-in-code-coop

SYNTH_GRAPH="/root/experiments/dreaming-in-code-coop/task_graph_synthetic.graphml"
LOG_DIR="/root/experiments/dicode_runs/aggregation/logs/l5_remaining"
mkdir -p "$LOG_DIR"

COMMON="training.total_timesteps=500000 training.num_envs=64 training.num_steps=64 evaluation.num_envs=64 evaluation.num_steps=512 dicode_manager.max_updates_per_session=2 dicode_manager.training_sample_size_n=8 dicode_manager.active_task_capacity=16 dicode_manager.num_generation_tasks=0 dicode_manager.additional_num_parents=0 seed=1 aggregation.enabled=true aggregation.use_llm_cache=true aggregation.retention_trigger=0.15 gen_manager.graph_path=${SYNTH_GRAPH}"

echo "=== Stage L5 remaining ==="
echo "Start: $(date)"

# B2: Tutor-only (Qwen progression only, no critic or explorer)
echo ""
echo "=== B2: llm_tutor_only (Qwen progression, no critic, no explorer) ==="
echo "Start: $(date)"
/usr/bin/tmux new-session -d -s l5_b2 "bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate dicode310 && cd /root/experiments/dreaming-in-code-coop && CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled PYTHONPATH=/root/experiments/dreaming-in-code-coop/src:/root/experiments/dreaming-in-code-coop python experiments/training/run_dicode.py ${COMMON} aggregation.mode=soft_copeland aggregation.llm_weight_novelty=0.0 aggregation.llm_weight_critic=0.0 aggregation.llm_weight_retention=0.0 2>&1 | tee ${LOG_DIR}/B2_tutor_only.log'"
echo "B2 launched in tmux l5_b2"

# Wait for B2
echo "Waiting for B2 (~1h)..."
sleep 3600

# B3: Tutor + Critic (no explorer)
echo ""
echo "=== B3: llm_tutor_critic (Qwen + DeepSeek, no GLM explorer) ==="
echo "Start: $(date)"
/usr/bin/tmux new-session -d -s l5_b3 "bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate dicode310 && cd /root/experiments/dreaming-in-code-coop && CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled PYTHONPATH=/root/experiments/dreaming-in-code-coop/src:/root/experiments/dreaming-in-code-coop python experiments/training/run_dicode.py ${COMMON} aggregation.mode=soft_copeland aggregation.llm_weight_novelty=0.0 aggregation.llm_weight_critic=0.5 aggregation.llm_weight_retention=0.2 2>&1 | tee ${LOG_DIR}/B3_tutor_critic.log'"
echo "B3 launched in tmux l5_b3"

echo "Waiting for B3 (~1h)..."
sleep 3600

# Run analysis
echo ""
echo "=== Running L5 comparison ==="
cd /root/experiments/dreaming-in-code-coop
PYTHONPATH=src:$PYTHONPATH python scripts/summarize_aggregation_runs.py \
    --log-dir /root/experiments/dicode_runs/aggregation/logs \
    --output-dir /root/experiments/dicode_runs/aggregation
PYTHONPATH=src:$PYTHONPATH python scripts/summarize_llm_collaboration_runs.py
echo ""
echo "=== L5 remaining complete: $(date) ==="
