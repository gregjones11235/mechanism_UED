# 安慰剂臂(rarity bounty)runbook —— bounty 效应的机制判别

> flag = `+training.rarity_bounty=2.0`(与 combat_bounty 互斥,断言钉死)。同一个
> wrapper 类(活体测试已背书)、同额 +2.0、**不相交的成就集**:diamond 三件套 +
> 宝石二 + enchant 二(非战斗、稀有)。回答周五必问:"+3 分是战斗的功劳,还是随便给
> 稀有成就加钱都行?"

## 预注册判据(点火前钉死)

三臂同总位对表(A 臂 2oyy46uv / bounty 臂 aacjo8fp / 安慰剂臂):
- 安慰剂 ret/perf **≈ A 臂** → 战斗特异,bounty 设计故事完整;
- **≈ bounty 臂** → 通用稀有加权/价值尺度效应 → 2e9 提案换配置(全稀有加权);
- 居中 → 记剂量注记(两组触发率不同:战斗组由 orc ~80% 主导,安慰剂组由 diamond
  ~25%/宝石 ~20% 主导——本判别的已知混杂,居中结果部分归因于剂量差)。

## 命令(rep2 剧本,目录/flag/日志三换)

FORK=/workspace/mechanism_UED/dicode_src/outputs/forkPlacebo
rm -rf $FORK && mkdir -p $FORK/rl_checkpoints
cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/rl_checkpoints/6900 $FORK/rl_checkpoints/0
cp /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/task_graph.graphml $FORK/
cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/runtime_analysis $FORK/ 2>/dev/null

tmux new-session -d -s train "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && python experiments/training/run_dicode.py hydra.run.dir=/workspace/mechanism_UED/dicode_src/outputs/forkPlacebo seed=1 use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED training.total_timesteps=2000000000 +training.rarity_bounty=2.0 gen_manager/llm@gen_manager.task_generator=local_qwen14b gen_manager/llm@gen_manager.env_generator=local_qwen14b gen_manager.embedding_model.model=nomic-embed-text gen_manager.task_generator.max_tokens=8192 gen_manager.env_generator.max_tokens=8192 +skill_preflight.use_scheduler=true +skill_preflight.use_preflight=true +skill_preflight.mastery_threshold=0.2 +skill_preflight.frontier_mode=prereq +skill_preflight.prereq_threshold=0.3 +skill_preflight.use_scaffold_gate=true > /workspace/run_placebo.log 2>&1"

sed -i 's|LOG=.*|LOG=/workspace/run_placebo.log|; s|CUTOFF=.*|CUTOFF=17|' /workspace/mon.sh
sleep 600 && grep -aE "Restoring step|RarityBounty|Found [0-9]+ frontier" /workspace/run_placebo.log | tail -4
# 三行判定:Restoring step: 0 + [RarityBounty/PLACEBO] ACTIVE + Found 72;
# ~40min 后 ls $FORK/rl_checkpoints/ 见 100 = 尸体可保。

## 周日 9:00 收数
# mon.sh >=17 -> kill -> wandb dump 三臂(脚本同款,安慰剂 run id 从日志取,s0=0,base=13700)
# -> STOP POD -> 出门。裁决贴 Claude(周一亦可)。
