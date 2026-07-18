# bounty 复跑 runbook(n=2 + 存档抢救)

> 与首炉唯一差别:**拷入的 checkpoint 重命名为 0**——orbax restore 取最新号(0)照常
> 恢复权重(Optimizer Reset 不依赖目录名),内部计数器从 0 起数后 save(100)>0 可落盘,
> 绕过计数器 bug,终点尸体得以保留。CUTOFF 相应改 17(从 0 起数)。

FORK=/workspace/mechanism_UED/dicode_src/outputs/forkBounty_rep2
rm -rf $FORK && mkdir -p $FORK/rl_checkpoints
cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/rl_checkpoints/6900 $FORK/rl_checkpoints/0
cp /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/task_graph.graphml $FORK/
cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/runtime_analysis $FORK/ 2>/dev/null

tmux new-session -d -s train "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && python experiments/training/run_dicode.py hydra.run.dir=/workspace/mechanism_UED/dicode_src/outputs/forkBounty_rep2 seed=1 use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED training.total_timesteps=2000000000 +training.combat_bounty=2.0 gen_manager/llm@gen_manager.task_generator=local_qwen14b gen_manager/llm@gen_manager.env_generator=local_qwen14b gen_manager.embedding_model.model=nomic-embed-text gen_manager.task_generator.max_tokens=8192 gen_manager.env_generator.max_tokens=8192 +skill_preflight.use_scheduler=true +skill_preflight.use_preflight=true +skill_preflight.mastery_threshold=0.2 +skill_preflight.frontier_mode=prereq +skill_preflight.prereq_threshold=0.3 +skill_preflight.use_scaffold_gate=true > /workspace/run_bounty_rep2.log 2>&1"

sed -i 's|LOG=.*|LOG=/workspace/run_bounty_rep2.log|; s|CUTOFF=.*|CUTOFF=17|' /workspace/mon.sh
sleep 600 && grep -aE "Finished restoring|CombatBounty|Found [0-9]+ frontier" /workspace/run_bounty_rep2.log | tail -4
# 起跑 ~40min 后关键新验证:ls $FORK/rl_checkpoints/ 应出现 100(存档抢救成功的铁证)

## 收数(~8h 后)
# mon.sh ≥17 → kill → ls rl_checkpoints 取末号 N →
# ① wandb dump 对表(run id 从日志 View run 取,s0=0,base=13700)——复现性裁决;
# ② 离线 eval steps=[N] +eval.details=true tag=BOUNTYREP —— 终点分数 + 验尸对比。
