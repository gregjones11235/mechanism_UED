#!/bin/bash
#SBATCH -J sfljnA
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o sfljnA-%j.out
#SBATCH -e sfljnA-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 36:00:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiayu_zhu@brown.edu
#================================================================================
# SFL JaxNav + auction 海选打分（N=3: difficulty+PVL+CENIE → bid → auction → 混合分 top-K）。
# estimator 层接进课程层（方案B §2 打分层）。CENIE 用官方 GMM（gmm_params 穿 jit + 每 epoch
# host 重拟）。探针仍开（对照混合分 vs 单 p(1-p) 的 top-K 选择）。
#
# 用法： sbatch _sfl_jaxnav_auction.sh <seed> [auction_lambda]
#   auction_lambda: inf=argmax(single-winner)、1.0=fractional 混合(默认)、大=uniform。消融用。
#
# 验证目标（首要）：N=3 auction 不崩——GMM 求值不 OOM(已 subsample)、auction 权重含 3 维、
#   每 epoch host 重拟 GMM 成功。第一个 eval epoch 过了即证接线通。
#================================================================================
SEED=${1:?用法: sbatch _sfl_jaxnav_auction.sh <seed> [auction_lambda]}
ALAMBDA=${2:-1.0}
echo "Job START: $(date)  seed=$SEED  auction_lambda=$ALAMBDA  node=$SLURM_JOB_NODELIST"
module load cuda
source /oscar/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate sfl || { echo "FATAL: conda activate sfl 失败"; exit 1; }

cd ~/sampling-for-learnability

python -c "import jax,sys; d=jax.devices(); print('JAX devices:', d); sys.exit(0 if any(x.platform in ('gpu','cuda') for x in d) else 1)" \
    || { echo "FATAL: JAX 没用上 GPU"; exit 1; }
python -c "import sys; sys.path.insert(0,'sfl/train'); import train_probe, auction_bid, cenie_density; print('auction+cenie import OK')" \
    || { echo "FATAL: auction/cenie 模块 import 失败"; exit 1; }

python sfl/train/jaxnav_sfl.py \
    SEED=$SEED \
    GROUP_NAME=auction-jaxnav-sfl \
    WANDB_MODE=online \
    AUCTION_SCORING=true \
    AUCTION_USE_CENIE=true \
    AUCTION_LAMBDA=$ALAMBDA

echo "Job END: $(date)"
