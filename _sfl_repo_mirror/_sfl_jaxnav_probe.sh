#!/bin/bash
#SBATCH -J sfljnP
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o sfljnP-%j.out
#SBATCH -e sfljnP-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 36:00:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiayu_zhu@brown.edu
#================================================================================
# SFL JaxNav（单 agent）+ V1 正交性探针（边训边验四信号正交性）。
#
# 在 _sfl_jaxnav.sh 基础上 +探针：每个 eval epoch host 侧拟 GMM 算四信号(difficulty/
# learnability/PVL/CENIE)两两相关，写 orthogonality_trace.csv，避开末期饱和 ckpt 退化相关。
# 见 mechanism_UED/_sfl_repo_mirror/INTEGRATION.md §4-§5。
#
# 用法： sbatch _sfl_jaxnav_probe.sh <seed>
# config: jaxnav-sfl.yaml (BATCH_SIZE=1000 x NUM_BATCHES=5 = 5000 海选, ROLLOUT_STEPS=1000,
#         num_agents=1, TOTAL_TIMESTEPS=3e8)
#
# 产物：
#   checkpoints/multi_robot_ued/<run>/model.safetensors          末期(向后兼容)
#   checkpoints/multi_robot_ued/<run>/model_step<N>.safetensors  多 checkpoint(不覆盖,挑非饱和阶段)
#   checkpoints/multi_robot_ued/<run>/probe/orthogonality_trace.csv  相关随训练轨迹(每 epoch 一行)
#   checkpoints/multi_robot_ued/<run>/probe/verdict_summary.txt      训练末总裁决(只采非饱和 epoch)
#   wandb: probe/rho/*, probe/max_abs_offdiag, probe/saturated 曲线
#
# 墙钟 36h：3e8 steps + 每 eval epoch 多一次探针 rollout(5x1000 步)的额外开销，留余量。
# 邮件通知 + GPU 自检：符合 mechanism_UED memory「SBATCH脚本两条硬规约」+「Stage1失败真因」教训。
#================================================================================
SEED=${1:?用法: sbatch _sfl_jaxnav_probe.sh <seed>}
echo "Job START: $(date)  seed=$SEED  node=$SLURM_JOB_NODELIST"
module load cuda
source /oscar/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate sfl || { echo "FATAL: conda activate sfl 失败"; exit 1; }

cd ~/sampling-for-learnability

# --- GPU 自检：JAX 认不出 GPU 就秒退，绝不空烧 36h 墙钟（Stage1 教训）---
python -c "import jax,sys; d=jax.devices(); print('JAX devices:', d); sys.exit(0 if any(x.platform in ('gpu','cuda') for x in d) else 1)" \
    || { echo "FATAL: JAX 没用上 GPU，立即终止作业"; exit 1; }

# --- 探针依赖自检：三件套 + cenie_density import 通才开训（错了几秒退，不空烧）---
python -c "import sys; sys.path.insert(0,'sfl/train'); import train_probe; print('probe import OK')" \
    || { echo "FATAL: 探针模块 import 失败，检查 sfl/train/{estimators,probe_orthogonality,train_probe,cenie_density}.py"; exit 1; }

# PROBE_ORTHOGONALITY 默认已在 jaxnav-sfl.yaml 设 true（探针开），无需命令行 override。
# 如需临时关探针对照：加 PROBE_ORTHOGONALITY=false。
python sfl/train/jaxnav_sfl.py \
    SEED=$SEED \
    GROUP_NAME=probe-jaxnav-sfl \
    WANDB_MODE=online

echo "Job END: $(date)"
