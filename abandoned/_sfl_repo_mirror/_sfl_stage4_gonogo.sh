#!/bin/bash
#SBATCH -J s4gng
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o s4gng-%A_%a.out
#SBATCH -e s4gng-%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiayu_zhu@brown.edu
#================================================================================
# STAGE4 阶段1 go/no-go: learnability 方差复活 + lambda 扫描 (方案B generator 注入路径)
# 短跑 TOTAL_TIMESTEPS=1e8 (满跑3e8的1/3 = 约15 eval epoch)
# array 范围由提交命令的 --array 控制:
#   sbatch --array=0-6 _sfl_stage4_gonogo.sh base   # 基线 GENERATOR_INJECTION=false
#   sbatch --array=0-6 _sfl_stage4_gonogo.sh inf    # exploit 端 single-winner
#   sbatch --array=0-3 _sfl_stage4_gonogo.sh 1.0    # 甜区候选 seed0-3 (Alec半)
#================================================================================
ARG=${1:?用法: sbatch --array=0-N _sfl_stage4_gonogo.sh <base|inf|1.0|5.0|none>}
SEED=${SLURM_ARRAY_TASK_ID:?必须以 array 提交}

if [ "$ARG" = "base" ]; then
    INJECT_FLAGS="GENERATOR_INJECTION=false"
    GROUP="s4-base"
else
    LTAG=$(echo "$ARG" | tr . _)
    INJECT_FLAGS="GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=$ARG"
    GROUP="s4-lambda-${LTAG}"
fi

echo "Job START: $(date)  seed=$SEED  arg=$ARG  group=$GROUP  node=$SLURM_JOB_NODELIST"
module load cuda
source /oscar/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate sfl || { echo "FATAL: conda activate sfl failed"; exit 1; }
cd ~/sampling-for-learnability

# GPU 自检: JAX 认不到 GPU 秒退,不空烧墙钟
python -c "import jax,sys; d=jax.devices(); print('JAX devices:', d); sys.exit(0 if any(x.platform=='gpu' for x in d) else 1)" \
    || { echo "FATAL: JAX no GPU"; exit 1; }
# import 自检
python -c "import sys; sys.path.insert(0,'sfl/train'); import pcgrl_generator, auction_bid, cenie_density; print('import OK')" \
    || { echo "FATAL: import failed"; exit 1; }

python sfl/train/jaxnav_sfl.py \
    SEED=$SEED \
    GROUP_NAME=$GROUP \
    WANDB_MODE=online \
    GEN_ESTIMATOR_IDS=[difficulty,pvl,cenie] \
    AUCTION_USE_CENIE=true \
    learning.TOTAL_TIMESTEPS=100000000 \
    $INJECT_FLAGS

echo "Job END: $(date)"
