#!/bin/bash
#SBATCH -J s4p2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o s4p2-%x-%A_%a.out
#SBATCH -e s4p2-%x-%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiayu_zhu@brown.edu
#================================================================================
# STAGE4 阶段2: 真实规模 SOTA 对标 (方案B generator 注入路径)
#   - TOTAL_TIMESTEPS=3e8 跑满 (=约45 eval epoch)
#   - 10 seed (array 0-9), 对齐 baseline rliable 口径
#   - PROBE_ORTHOGONALITY=true (注入档也拿 p_std 判饱和; 阶段1是 false)
#   - 代码已修: none档z-score归一 + probe解耦SAVE_PATH (2026-06-23)
#
# 用法 (单档手动提交; 整套请用 _submit_stage4p2.sh):
#   sbatch --array=0-9 _sfl_stage4_phase2.sh base   # 基线 GENERATOR_INJECTION=false
#   sbatch --array=0-9 _sfl_stage4_phase2.sh 1.0    # 甜区 fractional
#   sbatch --array=0-9 _sfl_stage4_phase2.sh inf    # exploit single-winner
#   sbatch --array=0-9 _sfl_stage4_phase2.sh none   # 去auction消融 (已修z-score)
#================================================================================
ARG=${1:?用法: sbatch --array=0-9 _sfl_stage4_phase2.sh <base|1.0|inf|none>}
SEED=${SLURM_ARRAY_TASK_ID:?必须以 array 提交}

if [ "$ARG" = "base" ]; then
    INJECT_FLAGS="GENERATOR_INJECTION=false"
    GROUP="s4p2-base"
else
    LTAG=$(echo "$ARG" | tr . _)
    INJECT_FLAGS="GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=$ARG"
    GROUP="s4p2-lambda-${LTAG}"
fi

echo "Job START: $(date)  seed=$SEED  arg=$ARG  group=$GROUP  node=$SLURM_JOB_NODELIST  jobid=$SLURM_JOB_ID"
module load cuda
source /oscar/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate sfl || { echo "FATAL: conda activate sfl failed"; exit 1; }
cd ~/sampling-for-learnability

# GPU 自检: JAX 认不到 GPU 秒退,不空烧墙钟
python -c "import jax,sys; d=jax.devices(); print('JAX devices:', d); sys.exit(0 if any(x.platform=='gpu' for x in d) else 1)" \
    || { echo "FATAL: JAX no GPU"; exit 1; }
# import 自检 (含修复后的模块)
python -c "import sys; sys.path.insert(0,'sfl/train'); import pcgrl_generator, auction_bid, cenie_density, train_probe; print('import OK')" \
    || { echo "FATAL: import failed"; exit 1; }

python -u sfl/train/jaxnav_sfl.py \
    SEED=$SEED \
    GROUP_NAME=$GROUP \
    WANDB_MODE=online \
    GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" \
    AUCTION_USE_CENIE=true \
    PROBE_ORTHOGONALITY=true \
    learning.TOTAL_TIMESTEPS=300000000 \
    $INJECT_FLAGS

echo "Job END: $(date)  seed=$SEED  group=$GROUP"
