# 方向三 E3 审核前阻断清单（ONE_REAL_FRONTIER_WINDOW_READY_FOR_AUDIT）

状态：生产链路代码已完成并冻结，**尚未执行任何真实 frontier window**。
`REAL_ACTUAL_N_EXECUTED / REAL_TWO_LLM_EXECUTED / REAL_ONE_UPDATE_EXECUTED / CHECKPOINT_RELOAD`
本轮全部为 **false**，只反映实际执行；任何阻断路径不得升级这些旗标。

完整 13 步窗口编排：`simulator_frontier/e3_window.py`（`one_window_pipeline`），
入口 `scripts/run_e3_real_one_window.py`（退出码 0=PASS / 4=FAIL / 5=BLOCKED），
长跑入口 `scripts/run_e3_longrun.py`（total_env_steps=98304，只冻结配置、不启动）。

## 阻断项与解除条件

1. **BLOCKED_TRAINING_SURFACE_PENDING_R9**
   - 现状：RMT16 `save_full_state` / `restore_full_state` 抛 `NotImplementedError`（本轮只读挂载）。
   - 影响：「恰好一次真实 optimizer update」与 checkpoint round trip（STEP10/11）不可执行。
   - 解除条件：R9 训练面落地（同一 Student 全状态保存/恢复：params+optimizer+step+rng+memory），且通过复验。

2. **BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE**
   - 现状：无总控签名的 `ProductionRegistryBundle`；生产联合恢复入口拒绝 synthetic 签名。
   - 影响：STEP03 单 fresh-process 联合恢复无法执行，actual-N 搜索的 restore context 无法成立。
   - 解除条件：总控签发真实 registry bundle（含验证材料）并绑定；`run_fresh_process_restore_production` 真实跑绿（单一 child PID + 逐组件 digest + production_joint_pass）。

3. **BLOCKED_SHARED_ANCHOR_MANIFEST**
   - 现状：无总控签名的共享冻结 anchor manifest；`compose_12_plus_4` 拒绝自拟锚点科学。
   - 影响：STEP07「12 动态分布 + 4 标准 reset 锚」组合无法放行。
   - 解除条件：总控下发签名 manifest（恰 4 锚、STANDARD_RESET、hash 绑定），`bind_anchor_manifest` 绑定成功。

4. **BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY**
   - 现状：生产注入槽未获总控注入 PRODUCTION 级 discovery registry。
   - 影响：STEP02 生产 Archive 写入口的 capture provenance 验证 fail-closed。
   - 解除条件：总控注入真实冻结正式资产身份集 registry（usage=PRODUCTION）。

5. **SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT / NO_BURN_IN_EXECUTOR**
   - 现状：审计过的 CC2 pkl `contains_memory=False`，无真实 saved-policy-memory artifact；亦无 burn-in executor。
   - 影响：正式记忆模式（SAVED_POLICY_MEMORY / HISTORY_BURN_IN）的分支搜索与 mixed-start 记忆准备被阻断；ZERO_MEMORY 仅消融、永不作生产模式。
   - 解除条件：产出真实记忆 artifact（sha256 + spec/identity hash 绑定）或授权 burn-in executor。

6. **REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT**
   - 现状：无总控授权的真实 LLM client factory；生产双 LLM 路径绝不回退 fake client。
   - 影响：STEP05 两次类型化 LLM 调用无法真实执行（0-call 复用路径亦需显式 reuse_plan_ref + 类型化旧计划）。
   - 解除条件：总控授权真实 Diagnostician/Planner client 并注入 factory。

7. **BLOCKED_REFERENCE_IDENTITY_PENDING_CONTROLLER_DESIGNATION**
   - 现状：更强的 Reference 策略身份由总控指定，本轮未指定，不得自拟。
   - 解除条件：总控指定 Reference 候选并完成只读挂载。

8. **BLOCKED_AUDIT_APPROVAL_NOT_GRANTED（长跑）**
   - 现状：`run_e3_longrun.py` 只冻结运行配置（98304 步、actual-N 预算、horizon、身份、记忆模式、seed、Git SHA、config hash），从不自行启动。
   - 解除条件：外部审核逐项放行以上 1–7 并显式批准长跑参数。

## 不受阻断项影响、已冻结的生产面

- Archive 生产写入口守卫链 + tmp→flush→fsync→os.replace 原子持久化 + 加载全量复验（`add_production_entry` / `save_production` / `load_production`）。
- 真实 actual-N runner：`actual_N == 实际执行分支数`；不记录 action sequence / 成功路线 / logits / hidden state / Reference memory。
- 类型化双 LLM 合同：严格 schema、hash 重算、禁词复检；无 fake 回退。
- 证据权威 Selector：`evidence_based_select` 为正式最终权威，正式路径无 `priority_score` 输入面；`deterministic_select` 冻结为消融咨询面。
- 预检门：`run_e3_preflight` 全部 fail-closed，缺口一律实名 blocker。

回归基线：`gpu1_aggregation_siege/tests/simulator_frontier` 312 passed（JAX_PLATFORMS=cpu）。
