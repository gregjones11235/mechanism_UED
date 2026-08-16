# φ-深度势能 shaping:烟测 → 3e8 消融 → 2e9 Runbook

> flag = `+training.depth_potential_c`(缺省/0 = wrapper 不构造 = v1 逐位同)。
> 势能只挂训练 env(ppo_tr.make_train);官方 eval 与 preflight 准入路径结构测试钉死不受染。
> 判据预注册于《阶段总结》§阶段二;本文件只放命令。**所有命令单行,防 markdown 渲染吞字符。**

## 0. Windows:打 patch 推送;Pod:环境 SOP + git pull + pytest(期望 65 passed——含 pod 才跑的 φ 数学测试)

## 1. 烟测(~2-3h,3-4 个 session)

tmux new-session -d -s train "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && python experiments/training/run_dicode.py seed=1 use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED training.total_timesteps=2000000000 +training.depth_potential_c=0.5 gen_manager/llm@gen_manager.task_generator=local_qwen14b gen_manager/llm@gen_manager.env_generator=local_qwen14b gen_manager.embedding_model.model=nomic-embed-text gen_manager.task_generator.max_tokens=8192 gen_manager.env_generator.max_tokens=8192 +skill_preflight.use_scheduler=true +skill_preflight.use_preflight=true +skill_preflight.mastery_threshold=0.2 +skill_preflight.frontier_mode=prereq +skill_preflight.prereq_threshold=0.3 +skill_preflight.use_scaffold_gate=true > /workspace/run_phi_smoke.log 2>&1"

烟测三判据(session 3-4 后 kill):
1. grep "DepthPotential" 日志 → `ACTIVE c=0.5` 打印在;
2. 无 NaN(grep -iE "nan|overflow");
3. **eval 隔离**:wandb `evaluation/mean_return` 前几点落在 v1 同期带位(±1.5)——
   势能项没有渗进官方在线 eval(train/* 指标带势能属预期,忽略)。
三过 → kill → 直接进 §2(同配置,总量本来就是 2e9,只是烟测手动停)。

## 2. 3e8 消融(~10-12h)

命令与 §1 完全相同(新开日志 run_phi_3e8.log);mon.sh 改 LOG + CUTOFF=23。
收线 → kill → 官方离线 eval(steps=[300,900,1500,2100,2300],tag=PHI3E8)。
判据(预注册):对表 v1 主跑同位(12.49/25.43/31.02/32.33);
**过线 = 裸态 enter_gnomish 显著抬升(±1.5)且 gnome 战斗非零化**(mean_return 平也算过——
势能治的是深层,浅层持平 + 深层开门就是胜利);
不过线 → c 调 1.0 重跑一轮 → 再不过归档("组合墙对深度势能稳健")。

## 3. 2e9(~55-65h,过线且组内报备后点)

同命令,日志 run_phi_2e9.log,mon.sh CUTOFF=154。
已知风险:schedule 末端崩溃(总 ~15500,第 60+ 小时)——收数取崩溃前存档,老规矩。
收线 eval 加密末端(参照主跑收数 steps 清单),终点对表 43.5 / 46.74。

## 备忘

- 烟测/消融/2e9 是同一配置的三段观察,不是三个配置——中途不改任何参数;
- v2 双 flag(r3 豁免/api_repair)**不带**——单变量纪律,φ 独测;
- 溜溜球复查:2e9 收线后对终点 ckpt 跑 +eval.details=true,151/168 是否消解
  = shaping 机制成立与否的直接证据(比 mean_return 更早、更机制)。
