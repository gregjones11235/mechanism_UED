# SlowGRU-Reset128 持续性与增益来源 — Phase3 最终报告

**范围**：CC2/总监B，仅 GPU2 与 GPU3。生成时间(UTC)：2026-07-25T04:44:35Z

> 本轮唯一目标：(1) 验证 SlowGRU-Reset128 在 24576 步的 +8.20pp 信号是否持续到 98304 步；(2) 定位收益来源（共享 CNN/GTrXL 梯度塑形 / 128 步内部递归时间建模 / 额外参数·深度·残差分支正则化）。本轮不是长期记忆实验。

---

## 1. Phase2 冻结结论（接受，不得修改）

**强制措辞（verbatim）**：已测试的SlowGRU和EventMemory32跨128步carry机制，未产生正向因果收益。

> 禁用措辞：禁止写：「128步窗口已经被证明不是瓶颈。」

| 冻结标签 | 值 |
|---|---|
| SLOWGRU_LONG_MEMORY_CAUSAL_SIGNAL | False |
| EVENTMEM_LONG_MEMORY_CAUSAL_SIGNAL | False |
| SLOWGRU_TRAINING_REGULARIZATION_ONLY | True |
| EVENTMEM_TRAINING_REGULARIZATION_ONLY | True |
| LONG_MEMORY_WINNER_B_SIDE | NONE |
| P2_FULL_B_AUTHORIZED | False |
| TWO_BY_TWO_ABLATION_AUTHORIZED | False |

Phase2 报告与 checkpoint 保持冻结，未被修改。

## 2. GPU2 四节点学习曲线（SlowGRU-Reset128 vs canonical Control，256 世界配对）

**锚点复现**：BASELINE(teacher17500) = 39.45%（ok=True）；Control@24576 = 36.33%（ok=True）；anchor_pass=True。

| step | SlowGRU DK SR | Control DK SR | ΔSR(pp) | Δworlds | SlowGRU floor3 | Control floor3 | SlowGRU death | Control death | McNemar p | 95% CI(pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| 24576 | 44.53% (114/256) | 36.33% (93/256) | +8.20 | +21 | 49.61% | 43.36% | 53.52% | 60.94% | 0.038088 | [+0.78,+15.23] |
| 49152 | 40.62% (104/256) | 37.11% (95/256) | +3.52 | +9 | 48.44% | 44.53% | 58.98% | 61.72% | 0.406787 | [-3.91,+10.94] |
| 73728 | 37.11% (95/256) | 31.25% (80/256) | +5.86 | +15 | 42.19% | 39.84% | 60.94% | 67.97% | 0.110612 | [-0.78,+12.50] |
| 98304 | 28.91% (74/256) | 37.50% (96/256) | -8.59 | -22 | 41.41% | 48.05% | 71.09% | 61.33% | 0.016002 | [-15.23,-1.95] |

## 3. SlowGRU-Reset128 是否持续有效

**持续性裁决 SUSTAINABILITY_VERDICT = `TRANSIENT_SIGNAL`**

理由：early/mid lead (leads={24576: True, 49152: True, 73728: True, 98304: False}, dSR={24576: 8.2, 49152: 3.52, 73728: 5.86, 98304: -8.59}) but 98304 REVERSED to significantly WORSE (dSR@98304=-8.59pp, death@98304=+9.77pp) -> signal vanished and turned harmful by the final node

SUSTAINED 建议门：c1(49152/73728/98304 至少两节点领先)=True [领先数 2/3]；c2(98304 ≥ Control +5pp)=False；c3(98304 floor3 不低于 Control)=False；c4(98304 death 不恶化)=False；c5(无数值/熵/恢复问题)=True；ALL_PASS=False。

## 4. Full / Detach / MatchedMLP / Control 统一表（@24576，256 世界配对）

| 臂 | DK SR | n_success | floor3 | ENTER_SEWERS | conditional kill | death | timeout | eplen |
|---|---|---|---|---|---|---|---|---|
| Control | 36.33% | 93/256 | 43.36% | 43.36% | 83.8% | 60.94% | 2.73% | 862 |
| Full (SlowGRU-Reset128) | 44.53% | 114/256 | 49.61% | 49.61% | 89.8% | 53.52% | 1.95% | 921 |
| Detach (stop_gradient) | 42.97% | 110/256 | 47.66% | 47.66% | 90.2% | 56.25% | 0.78% | 848 |
| MatchedMLP (non-recurrent) | 37.11% | 95/256 | 45.31% | 45.31% | 81.9% | 60.94% | 1.95% | 896 |

完整性：Full@24576 sha 前缀 = `2ffdd269b94e1e6b`（期望 2ffdd269，ok=True）；四臂参数全 finite = True；Control@24576 锚点 ok = True。

## 5. 增益来源效应分解（配对，+ = 前者更优）

| 效应 | 比较 | ΔSR(pp) | Δworlds | Δfloor3 | Δdeath | McNemar p | 95% CI(pp) | 是否成立 |
|---|---|---|---|---|---|---|---|---|
| BACKBONE_GRADIENT_SHAPING = Full - Detach | Full_Reset128_24576 − Detach_24576 | +1.56 | +4 | +1.95 | -2.73 | 0.727282 | [-5.08,+8.20] | 否 |
| WITHIN_ROLLOUT_RECURRENCE = Full - MatchedMLP | Full_Reset128_24576 − MatchedMLP_24576 | +7.42 | +19 | +4.30 | -7.42 | 0.048182 | [+0.39,+14.06] | 是 |
| CAPACITY_OR_RESIDUAL_REGULARIZATION = MatchedMLP - Control | MatchedMLP_24576 − Control_24576 | +0.78 | +2 | +1.95 | +0.00 | 0.912067 | [-5.86,+7.42] | 否 |
| TOTAL = Full - Control (sanity) | Full_Reset128_24576 − Control_24576 | +8.20 | +21 | +6.25 | -7.42 | 0.038088 | [+0.78,+15.23] | — |

判定阈值「明显优于」：ΔSR ≥ 2pp 且（McNemar p < 0.10 或 配对 95% CI 下界 > 0）。成立效应数 = 1。

## 6. 最终标签

- **持续性 SUSTAINABILITY_VERDICT = `TRANSIENT_SIGNAL`**
- **增益来源 GAIN_SOURCE = `WITHIN_ROLLOUT_RECURRENCE`**

增益来源理由：Full clearly > MatchedMLP (dSR=+7.42pp, p=0.048182) while backbone=False capacity=False -> gain from within-rollout recurrence

## 7. checkpoint / 代码 / 评估器路径与 SHA

- 冻结评估器：`/home/oseasy/experiments/student_upgrade_wave1_4gpu/eval_phase2_unified.py`（sha 前缀 22451402）；GPU2 复用 sha=224514026aefd273，GPU3 复用 sha=224514026aefd273。
- 冻结 SlowGRU 网络 sha 前缀：b265210597d00321（longrun/detach/mlp 公共结构同源）。
- Longrun 四节点 ckpt：`/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_slowgru_reset128_longrun/train/ckpt/{24576,49152,73728,98304}/full_state.pkl`
  - 24576: params_sha=2ffdd269b94e1e6b finite=True
  - 49152: params_sha=5a4f65ac65a3f8e1 finite=True
  - 73728: params_sha=bf82b0f6ed0031e6 finite=True
  - 98304: params_sha=9d92c5b9e2e2148b finite=True
- canonical Control 根目录：`/home/oseasy/experiments/exploratory_delayed_onset_20260724/control_RUN2/ckpt`（24576/49152/73728/98304）。
- Full@24576 sha=2ffdd269b94e1e6b；Detach@24576 sha=0890aa1fc5993875；MatchedMLP@24576 sha=07a49467ca9841c8；Control@24576 sha=ece6fa9962e81512。
- 训练器：longrun=`run_slowgru_reset128_longrun.py`；detach=`run_slowgru_detach_24576.py`；mlp=`run_slowgru_mlp_24576.py`。
- 网络：detach=`slowgru_detach_network.py`；mlp=`slowgru_mlp_network.py`。
- 工程门：LONGRUN_GATES_PASS=True；ATTRIBUTION_GATES_PASS=True。
- 配置 diff：detach=True；mlp=True。

```
# CC2 GPU UUID map (written once at SLOWGRU_RESET128_SUSTAINABILITY_AND_ATTRIBUTION launch)
# date_utc: 2026-07-25T03:15:58Z
0, GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6, NVIDIA RTX A6000, 34534 MiB, 46068 MiB
1, GPU-3c7a2864-755b-7045-b293-6f80e748283f, NVIDIA RTX A6000, 1 MiB, 46068 MiB
2, GPU-8df11537-ab79-722d-606f-411966196c4c, NVIDIA RTX A6000, 1 MiB, 46068 MiB
3, GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd, NVIDIA RTX A6000, 1 MiB, 46068 MiB
# CC2 owns PHYSICAL GPU2 and GPU3 ONLY. GPU0/GPU1 forbidden.
CC2_GPU2_UUID=GPU-8df11537-ab79-722d-606f-411966196c4c
CC2_GPU3_UUID=GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd
```

## 8. 明确声明：未启动 P2-Full-B 与长期记忆 2×2

本轮明确未启动 P2-Full-B，未启动长期记忆 2×2 消融；未训练 SlowGRU-Persistent / EventMemory，未继续长期记忆 Carry 路线，未跑 Replay/V-trace/hindsight，未跑第二 seed，未跑 512-world，未跑 Official FULL。

| 项目 | 状态 |
|---|---|
| P2-Full-B | 未启动 |
| 长期记忆 2×2 消融 | 未启动 |
| SlowGRU-Persistent / EventMemory 重新训练 | 未执行 |
| Replay / V-trace / hindsight | 未执行 |
| 第二训练 seed / 512-world / Official FULL | 未执行 |

---

*评估协议：冻结 256 世界 Stage4 随机策略 seed42（复用 eval_phase2_unified.py，sha 前缀 22451402），所有臂按世界索引配对。本报告由 build_phase3_report.py 自动生成。*
