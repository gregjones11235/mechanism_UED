# P9-AUTHENTIC-RESET — 最终归档报告（GPU3）

## 最终标签
**P9_NO_POSITIVE_SIGNAL**

## 假设与结论
- 假设：Student 失败是因为很少练习 floor2→3 关键相位；用真实健康轨迹的"到达态"重置（50/50 自然/真实）增加该相位练习可提升 dark-search。
- 结论：仅带来短暂中期提升（24576–73728 SR/floor3 领先 Control），到 98304 被 Control 反超。**真实到达态重置不产生持续收益**。

## 98304 正向门槛（vs 同 step Control，256 世界自然起点）
- SR_delta = -4.30pp（需 ≥ +8）→ FAIL
- floor3_delta = -4.69pp（需 ≥ 0）→ FAIL
- death +3.12pp / timeout +1.17pp / sewers -4.69pp（需 ≥1 项改善）→ FAIL
- finite_pass = True
→ 三门未过 → 非 EXPLORATORY_POSITIVE_SIGNAL；工程健全 → 非 ENGINEERING_FAIL → **NO_POSITIVE_SIGNAL**

## 逐节点（P9 vs Control，256 世界，DK ever-set，max_steps=4096）
| step | P9_SR | CTL_SR | dSR | dFloor3 | dDeath | dSewers | condKill P9/CTL |
|------|-------|--------|-----|---------|--------|---------|-----------------|
| 24576 | 41.80 | 36.33 | +5.47 | +4.69 | -4.69 | +4.69 | 86.99/83.78 |
| 49152 | 39.45 | 37.11 | +2.34 | +5.08 | -2.34 | +5.08 | 79.53/83.33 |
| 73728 | 32.03 | 31.25 | +0.78 | +4.69 | -1.17 | +4.69 | 71.93/78.43 |
| 98304 | 33.20 | 37.50 | -4.30 | -4.69 | +3.12 | -4.69 | 76.58/78.05 |

## 工程门（全过）
- 采集三门 P9_VALIDATE_PASS：位可恢复 40/40、经同 wrapper 单步转移 8/8（max_obs_diff=0.0）、无泄漏/有限。
- 4096 smoke：数值有限、熵健康。
- EXACT_RESUME_PASS：连续 0→8192 与 0→4096→恢复→8192 的 params_sha 逐位一致（9ba3f2b9…）；full_state roundtrip 覆盖 params/opt_state/rng/env_state/memories/mask/midx/obs/done。
- 训练 P9_TRAIN_OK，6 标准节点 roundtrip_ok=True，全程熵 0.51–1.52（末值 1.37）无塌缩。

## 状态库
- 40 快照：saw_stair_lost=32（满额），near_floor3_failed=8；floor2_reached/mid_clear/gate_unlocked=0（S4_dark 出生即清场，三类结构性为空，符合设计预期）。
- 库 sha256 = 6ac4decbd6776b7fbc4cc4b870832ff6654104b15b8a10cce7772f950389d272

## 路径与 SHA
- 库：/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/lib/p9_library.pkl
- 验证：/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/lib/p9_validate_summary.json
- 训练 ckpt：/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/train_98304/ckpt/{0,4096,24576,49152,73728,98304}/full_state.pkl
- 98304 ckpt sha = 1c8918c5477c404a；最终 params_sha = f5b928d1128d48ce…
- 终评：/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/eval/p9_final_gate.json（evaluator_sha=eff1d77cae6d229a…）
- teacher 初始化 sha = d4e85af58b7f87d6…
- Control 锚点：/home/oseasy/experiments/exploratory_delayed_onset_20260724/control_RUN2/ckpt

## 冻结配置
seed=42 LR=2e-5 adam_eps=1e-5 γ=0.999 λ=0.8 num_envs=16 rollout=128 clip=0.2 vf=0.5 ent=0.002
gradnorm=1.0 anneal_lr=False window_mem=128 window_grad=64 heads=8 layers=2 embed/qkv=256
gating=True gating_bias=2.0 optimistic_reset_ratio=16 total=98304；重置混合 50/50（FROZEN，不按结果调）。
终评 100% 自然 Stage4 起点，Authentic Reset 不进入评估。
