# 崩溃测定台 Runbook:两臂 A/B(缩放 S / 防火墙 F),对照白嫖 A 臂

> 病灶:2e9 schedule 末端(总 ~15500)六次同位复现的崩溃,死亡链
> value_loss 1.5e10 → grad_norm 2.5e10 → entropy→0 → ret −0.90 躺平。
> 测定台:从主跑 **ckpt 8600(总 15400,崩溃前最后健康档)**分叉,带单一 flag 冲过
> 崩溃区;**对照臂 = A 臂既有数据(15500 处 ret 2.85,免费)**。
> 两臂 flag:S = +training.adaptive_value_scale=true(值损失批自适应归一,诚实命名:
> 非完整 PopArt);F = +training.critic_grad_firewall=true(critic 梯度断骨干)。
> 判据(预注册):臂活过 **总 15800+**(4+ session)且干净通道 evaluation/mean_return
> ≥41、train/value_loss 全程 <1e6 = PASS;15500 同位崩 = FAIL;**双 FAIL 亦有价值**
> (病因不在值尺度/传播 → 指向 LR/任务 regime)。
> 单变量纪律:两臂均为纯 v1 + 单 flag,无任何整形。门牌:主跑档案续 ~155,5 个
> session 到 160 ≈ 总 15900;**串联过夜:timeout 7500 自动接力**,无需人守。

## ① 目录(两臂各建,ckpt 拷 0 抢救存档)

for ARM in S F; do
  FORK=/workspace/mechanism_UED/dicode_src/outputs/crashfix_$ARM
  rm -rf $FORK && mkdir -p $FORK/rl_checkpoints
  cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/rl_checkpoints/8600 $FORK/rl_checkpoints/0
  cp /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/task_graph.graphml $FORK/
  cp -r /workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/runtime_analysis $FORK/ 2>/dev/null
done
ls /workspace/mechanism_UED/dicode_src/outputs/crashfix_S/rl_checkpoints/

## ② 串联点火(一条 tmux,S 跑 125 分钟自动切 F;两臂共 ~4-5h,过夜正好)

tmux new-session -d -s train "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && timeout 7500 python experiments/training/run_dicode.py hydra.run.dir=/workspace/mechanism_UED/dicode_src/outputs/crashfix_S seed=1 use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED training.total_timesteps=2000000000 +training.adaptive_value_scale=true gen_manager/llm@gen_manager.task_generator=local_qwen14b gen_manager/llm@gen_manager.env_generator=local_qwen14b gen_manager.embedding_model.model=nomic-embed-text gen_manager.task_generator.max_tokens=8192 gen_manager.env_generator.max_tokens=8192 +skill_preflight.use_scheduler=true +skill_preflight.use_preflight=true +skill_preflight.mastery_threshold=0.2 +skill_preflight.frontier_mode=prereq +skill_preflight.prereq_threshold=0.3 +skill_preflight.use_scaffold_gate=true > /workspace/run_crashfix_S.log 2>&1 ; timeout 7500 python experiments/training/run_dicode.py hydra.run.dir=/workspace/mechanism_UED/dicode_src/outputs/crashfix_F seed=1 use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED training.total_timesteps=2000000000 +training.critic_grad_firewall=true gen_manager/llm@gen_manager.task_generator=local_qwen14b gen_manager/llm@gen_manager.env_generator=local_qwen14b gen_manager.embedding_model.model=nomic-embed-text gen_manager.task_generator.max_tokens=8192 gen_manager.env_generator.max_tokens=8192 +skill_preflight.use_scheduler=true +skill_preflight.use_preflight=true +skill_preflight.mastery_threshold=0.2 +skill_preflight.frontier_mode=prereq +skill_preflight.prereq_threshold=0.3 +skill_preflight.use_scaffold_gate=true > /workspace/run_crashfix_F.log 2>&1"

sed -i 's|LOG=.*|LOG=/workspace/run_crashfix_S.log|; s|CUTOFF=.*|CUTOFF=160|' /workspace/mon.sh

## ③ 十分钟确认(S 臂):Restoring step: 0 + Found frontier;无 shaping ACTIVE(纯 v1+flag)
sleep 600 && grep -aE "Restoring step|Found [0-9]+ frontier|ACTIVE" /workspace/run_crashfix_S.log | tail -3

## ④ 明早裁决(wandb 或日志):两臂各自 15400→15900 的干净通道 eval 序列 +
##    train/value_loss 峰值;对照 = A 臂 15500 的 2.85。三种结局全部预注册在案。
