# 稠密战斗奖励(shot-2)fork 测试 Runbook

> flag = `+training.combat_bounty=2.0`(缺省/0 = 不构造 = v1 逐位同)。深层 DEFEAT_*
> 首杀(除 zombie/skeleton)每次 +2.0 ≈ 该成就总价 3 倍;故意非势能(φ 裁决的直接推论:
> 保最优的整形扳不动理性撤退,本发**以改变最优策略为目的**,把战斗尝试的期望值抬过
> 死亡风险)。诚实定位:成就位锁存 → 本质是"首杀加权"而非逐杀稠密;逐杀版(怪物数组
> 差分)为升级候选。防刷分内置(每 episode 每型一次)。

## 预注册判据(点火前钉死)

对照 = A 臂(2oyy46uv,同起点同 seed 无 flag)。同总位(13700-15400):
- **主判据**:defeat_orc_solider/mage 系统性高于 A 臂,或 gnome_warrior+archer 多点
  持续 >1.5(A 臂噪声顶 0.5);
- 次级:gnomish 抬升;**守门**:ret 相对 A 持续低 3+ = 红灯(非势能可致行为畸变);
- 不过线 → shot-2 归档("首杀加权不足以买通战斗链"),逐杀稠密版与退火组合升为
  周五提案主体;过线 → 周五带数据提案 2e9 组合方案。

## 命令(pod,forkPhi 剧本换目录换 flag)

FORK=/workspace/mechanism_UED/dicode_src/outputs/forkBounty_from13700
rm -rf $FORK && mkdir -p $FORK/rl_checkpoints
cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/rl_checkpoints/6900 $FORK/rl_checkpoints/
cp /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/task_graph.graphml $FORK/
cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/runtime_analysis $FORK/ 2>/dev/null

tmux new-session -d -s train "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && python experiments/training/run_dicode.py hydra.run.dir=/workspace/mechanism_UED/dicode_src/outputs/forkBounty_from13700 seed=1 use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED training.total_timesteps=2000000000 +training.combat_bounty=2.0 gen_manager/llm@gen_manager.task_generator=local_qwen14b gen_manager/llm@gen_manager.env_generator=local_qwen14b gen_manager.embedding_model.model=nomic-embed-text gen_manager.task_generator.max_tokens=8192 gen_manager.env_generator.max_tokens=8192 +skill_preflight.use_scheduler=true +skill_preflight.use_preflight=true +skill_preflight.mastery_threshold=0.2 +skill_preflight.frontier_mode=prereq +skill_preflight.prereq_threshold=0.3 +skill_preflight.use_scaffold_gate=true > /workspace/run_forkBounty.log 2>&1"

sed -i 's|LOG=.*|LOG=/workspace/run_forkBounty.log|; s|CUTOFF=.*|CUTOFF=172|' /workspace/mon.sh
sleep 600 && grep -aE "Finished restoring|CombatBounty|Found [0-9]+ frontier" /workspace/run_forkBounty.log | tail -4

## 收数

mon.sh ≥172 → kill → wandb dump 老脚本换 run id(SK 列表加 defeat_orc_solider/mage 两键)
→ 对表裁决。progressive budget:排在本发裁决之后,周五作"胜者搭档"提案(验尸账:
997 死 vs 27 超时,先买命再买钟)。
