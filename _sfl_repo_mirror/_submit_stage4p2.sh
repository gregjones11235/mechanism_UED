#!/bin/bash
#================================================================================
# STAGE4 阶段2 一键提交 orchestrator
#   - Alec 批 (base + λ=1.0, 各10seed=20run) 先跑, 默认优先级
#   - Henry 批 (λ=inf + λ=none, 各10seed=20run) --dependency=afterany 硬依赖 Alec 整批结束才开始
#   - 共 40 run, 全部进 jzhu223 的 Oscar 队列, 抢同一 2-GPU 配额, Alec 优先跑满后 Henry 才动
#
# 用法 (在 ~/sampling-for-learnability 下, 登录节点直接跑此脚本; 它只 sbatch 不算计算):
#   bash _submit_stage4p2.sh
#
# Henry 若自己做完某档/某seed: scancel 对应排队 job (脚本末尾打印命令), 不浪费配额。
#================================================================================
set -e
cd ~/sampling-for-learnability

echo "=== 提交 Alec 批 (base + λ=1.0, 各 10 seed) ==="
JOB_BASE=$(sbatch --parsable -J s4p2A_base --array=0-9 _sfl_stage4_phase2.sh base)
echo "  s4p2-base       array job: $JOB_BASE"
JOB_L10=$(sbatch --parsable -J s4p2A_l1_0 --array=0-9 _sfl_stage4_phase2.sh 1.0)
echo "  s4p2-lambda-1_0 array job: $JOB_L10"

# afterany: 依赖 job 全部结束(成败均可)后才放行。Henry 批同时依赖 Alec 两批。
ALEC_DEP="afterany:${JOB_BASE}:${JOB_L10}"

echo "=== 提交 Henry 批 (λ=inf + λ=none, 各 10 seed), 硬依赖 Alec 整批结束 ==="
JOB_INF=$(sbatch --parsable -J s4p2H_inf  --dependency=$ALEC_DEP --array=0-9 _sfl_stage4_phase2.sh inf)
echo "  s4p2-lambda-inf  array job: $JOB_INF  (dependency: $ALEC_DEP)"
JOB_NONE=$(sbatch --parsable -J s4p2H_none --dependency=$ALEC_DEP --array=0-9 _sfl_stage4_phase2.sh none)
echo "  s4p2-lambda-none array job: $JOB_NONE (dependency: $ALEC_DEP)"

echo
echo "================ 提交完成: 40 run (Alec 20 优先 → Henry 20 依赖) ================"
echo "Alec 批 jobid: base=$JOB_BASE  l1.0=$JOB_L10"
echo "Henry 批 jobid: inf=$JOB_INF  none=$JOB_NONE  (afterany Alec)"
echo
echo "查队列:        squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.20E'"
echo "Henry 做完撤销: scancel $JOB_INF $JOB_NONE         # 撤整批"
echo "  或撤某几seed:  scancel ${JOB_INF}_4,5,6              # array 元素级"
echo "改 Henry 优先级(让它在Alec空隙也能跑,不死等整批): scontrol update jobid=$JOB_INF dependency="
