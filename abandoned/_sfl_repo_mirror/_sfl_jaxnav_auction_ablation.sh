#!/bin/bash
#SBATCH -J sfljnAbl
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o sfljnAbl-%A_%a.out
#SBATCH -e sfljnAbl-%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 36:00:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiayu_zhu@brown.edu
#SBATCH --array=0-9
#================================================================================
# auction 必要性消融（方案B §6.1 / STAGE3 w 轴）—— N=3 estimator(difficulty+PVL+CENIE)
# 打分层，扫 auction_lambda 三档比下游性能，看「混合 vs single-winner」是否有实质增益。
#
#   λ=inf → argmax (single-winner)        ：只回放 bid 最高那个 estimator 的 top-K
#   λ=1.0 → fractional 三方混合 [.25,.28,.47]：三 estimator 加权混合分 top-K（方案B §200 标定）
#   λ=3.0 → 近 uniform                     ：权重摊平，退化对照
#
# 判据（答 §6.1「auction 必要性待验证」）：
#   三档下游 win rate 的 10-seed CI 分得开（混合 > argmax）→ auction 必要性坐实；
#   CI 重叠（如旧 maze future_routes §1.1 那样）→ 诚实砍掉，用简单 argmax。
#
# 用法（三档各提交一次，每次铺 10 seed array）：
#   sbatch _sfl_jaxnav_auction_ablation.sh inf
#   sbatch _sfl_jaxnav_auction_ablation.sh 1.0
#   sbatch _sfl_jaxnav_auction_ablation.sh 3.0
#   → 共 30 run，2 GPU 并发、单 run 实测≈52min（3360442 跑满 52min），约 13h 出齐。
#
# 与正在跑/已跑的 N=3 run（3360442, λ=1.0）的关系：λ=1.0 这档与之同配置；本消融把它扩到
#   10 seed 并补 argmax/uniform 两端做对照。N=3 健康跑通已验接线（45 epoch, GMM 重拟 OK,
#   PVL~CENIE≈-0.21 正交），故本消融关探针只测下游性能（见下 PROBE_ORTHOGONALITY=false）。
#
# 墙钟 36h：实测单 run≈52min，36h 远超余量（memory「SBATCH脚本两条硬规约」: 墙钟≥24h 宁多勿少）。
# 邮件通知 + GPU/import 自检：符合两条硬规约 +「Stage1 失败真因=空烧墙钟」教训。
#================================================================================
ALAMBDA=${1:?用法: sbatch _sfl_jaxnav_auction_ablation.sh <inf|1.0|3.0>}
SEED=${SLURM_ARRAY_TASK_ID:?本脚本必须以 array 提交（--array=0-9 已在头部声明）}

# λ 写进 GROUP_NAME 让 wandb 把三档分组对比；点号换下划线避免路径/标签歧义。
LTAG=$(echo "$ALAMBDA" | tr '.' '_')
GROUP="auction-abl-l${LTAG}"

echo "Job START: $(date)  seed=$SEED  auction_lambda=$ALAMBDA  group=$GROUP  node=$SLURM_JOB_NODELIST"
module load cuda
source /oscar/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate sfl || { echo "FATAL: conda activate sfl 失败"; exit 1; }

cd ~/sampling-for-learnability

# --- GPU 自检：JAX 认不出 GPU 就秒退，绝不空烧墙钟（Stage1 教训）---
python -c "import jax,sys; d=jax.devices(); print('JAX devices:', d); sys.exit(0 if any(x.platform in ('gpu','cuda') for x in d) else 1)" \
    || { echo "FATAL: JAX 没用上 GPU，立即终止作业"; exit 1; }

# --- auction/cenie import 自检：错了几秒退，不空烧 ---
python -c "import sys; sys.path.insert(0,'sfl/train'); import train_probe, auction_bid, cenie_density; print('auction+cenie import OK')" \
    || { echo "FATAL: auction/cenie 模块 import 失败"; exit 1; }

# AUCTION_SCORING=true 开 auction 混合分 top-K；AUCTION_USE_CENIE=true → N=3([difficulty,PVL,CENIE])。
# PROBE_ORTHOGONALITY=false：消融只看下游 win rate，正交性已由 N=3 run(3360442) 单独验过；
#   关探针省掉每 epoch 拟 GMM + 画图开销，让 30 run 更快出齐。
python sfl/train/jaxnav_sfl.py \
    SEED=$SEED \
    GROUP_NAME=$GROUP \
    WANDB_MODE=online \
    AUCTION_SCORING=true \
    AUCTION_USE_CENIE=true \
    AUCTION_LAMBDA=$ALAMBDA \
    PROBE_ORTHOGONALITY=false

echo "Job END: $(date)"
