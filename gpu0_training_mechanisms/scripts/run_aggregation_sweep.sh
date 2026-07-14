#!/bin/bash
# ==============================================================================
# Aggregation Sweep Launcher
# ==============================================================================
#
# Runs a sweep of aggregation experiments across modes and retention triggers.
# Logs output to timestamped directories under:
#   /root/experiments/dicode_runs/aggregation/logs/
#
# Usage:
#   bash scripts/run_aggregation_sweep.sh
#
# Constraints:
#   - CUDA_VISIBLE_DEVICES=1
#   - XLA_PYTHON_CLIENT_PREALLOCATE=false
#   - WANDB_MODE=disabled
#   - PYTHONPATH=$PWD/src:$PWD
#   - training.num_steps=64
#   - training.total_timesteps=5000000
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

LOG_BASE="/root/experiments/dicode_runs/aggregation/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Experiment parameters
GPU=1
SEED=1
TOTAL_TIMESTEPS=5000000
NUM_ENVS=64
NUM_STEPS=64
MAX_UPDATES_PER_SESSION=2
TRAINING_SAMPLE_SIZE_N=4
ACTIVE_TASK_CAPACITY=16

# Aggregation modes to sweep
MODES=(
    "raw_weighted"
    "robust_weighted"
    "soft_copeland"
    "budgeted_soft_copeland"
    "budgeted_retention_trigger"
    "entropy_regularized"
)

# Retention trigger values to sweep
RETENTION_TRIGGERS=(
    "0.05"
    "0.10"
    "0.15"
    "0.20"
)

# Common Hydra overrides
COMMON_OVERRIDES=(
    "training.total_timesteps=${TOTAL_TIMESTEPS}"
    "training.num_envs=${NUM_ENVS}"
    "training.num_steps=${NUM_STEPS}"
    "dicode_manager.max_updates_per_session=${MAX_UPDATES_PER_SESSION}"
    "dicode_manager.training_sample_size_n=${TRAINING_SAMPLE_SIZE_N}"
    "dicode_manager.active_task_capacity=${ACTIVE_TASK_CAPACITY}"
    "aggregation.enabled=true"
    "seed=${SEED}"
)

echo "=============================================="
echo "Aggregation Sweep Launcher"
echo "=============================================="
echo "Timestamp:         ${TIMESTAMP}"
echo "GPU:               ${GPU}"
echo "Seed:              ${SEED}"
echo "Total timesteps:   ${TOTAL_TIMESTEPS}"
echo "Modes:             ${MODES[*]}"
echo "Retention triggers: ${RETENTION_TRIGGERS[*]}"
echo "Log base:          ${LOG_BASE}"
echo "=============================================="
echo ""

# Create log directory
mkdir -p "${LOG_BASE}"

TOTAL_RUNS=$((${#MODES[@]} * ${#RETENTION_TRIGGERS[@]}))
CURRENT_RUN=0

for MODE in "${MODES[@]}"; do
    for TRIGGER in "${RETENTION_TRIGGERS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        RUN_NAME="${MODE}_trigger${TRIGGER}"
        LOG_DIR="${LOG_BASE}/${TIMESTAMP}_${RUN_NAME}"
        mkdir -p "${LOG_DIR}"

        echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Starting: ${RUN_NAME}"

        # Build overrides
        OVERRIDES=(
            "${COMMON_OVERRIDES[@]}"
            "+aggregation.mode=${MODE}"
            "+aggregation.retention_trigger=${TRIGGER}"
        )

        # Mode-specific overrides
        if [ "${MODE}" = "entropy_regularized" ]; then
            OVERRIDES+=("+aggregation.entropy_regularization=0.1")
        fi

        OVERRIDE_STR=""
        for ov in "${OVERRIDES[@]}"; do
            OVERRIDE_STR="${OVERRIDE_STR} ${ov}"
        done

        # Launch via tmux (long-running)
        SESSION_NAME="agg_${MODE}_t${TRIGGER}_${TIMESTAMP}"

        echo "  tmux session: ${SESSION_NAME}"
        echo "  log: ${LOG_DIR}/run.log"

        # Create the run script for this experiment
        RUN_SCRIPT="${LOG_DIR}/run_cmd.sh"
        cat > "${RUN_SCRIPT}" << EOF
#!/bin/bash
set -euo pipefail
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=${GPU}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

echo "Starting run: ${RUN_NAME}"
echo "Mode: ${MODE}"
echo "Trigger: ${TRIGGER}"
echo "Overrides: ${OVERRIDE_STR}"

python experiments/training/run_dicode.py ${OVERRIDE_STR} 2>&1 | tee "${LOG_DIR}/run.log"

EXIT_CODE=\${PIPESTATUS[0]}
echo "Run finished with exit code: \${EXIT_CODE}"
echo "\${EXIT_CODE}" > "${LOG_DIR}/exit_code.txt"
EOF
        chmod +x "${RUN_SCRIPT}"

        # Launch in tmux
        tmux new-session -d -s "${SESSION_NAME}" "bash ${RUN_SCRIPT}"

        echo "  Launched. tmux attach -t ${SESSION_NAME} to view."
        echo ""

        # Brief pause between launches to avoid resource contention
        sleep 2
    done
done

echo ""
echo "=============================================="
echo "All ${TOTAL_RUNS} runs launched."
echo "Timestamp: ${TIMESTAMP}"
echo ""
echo "To monitor:"
echo "  tmux ls | grep agg_"
echo ""
echo "To kill all:"
echo "  tmux ls | grep agg_ | awk -F: '{print \$1}' | xargs -I{} tmux kill-session -t {}"
echo "=============================================="
