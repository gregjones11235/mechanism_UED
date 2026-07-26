# LONG_MEMORY_CAUSAL_CARRY_ABLATION_PHASE2 — 总监B 最终一次性汇报（GPU2/GPU3）

日期：2026-07-25 ｜ 仅 GPU2/3。Original PPO，无 Replay/hindsight/V-trace/AWR。γ/λ 固定 0.999/0.8。
TRAINING_TO_98304_AUTHORIZED=false；P2_FULL_B_AUTHORIZED=false；TWO_BY_TWO_ABLATION_AUTHORIZED=false。未训至 98304，未改冻结网络结构。

**唯一目标**：判断 SlowGRU / EventMemory32 的性能变化是否真正来自跨128步长期状态(carry)，而非额外参数、共训练、优化扰动或短期表示变化。
**方法**：训练期匹配消融。Persistent（长期状态跨rollout携带）vs Reset128（每128步rollout边界清空长期状态）。除「长期状态是否跨rollout携带」外**完全一致**（网络/参数/init/优化器/损失/配置/seed 逐位相同）。

---

## 1. 工程一致性（13 门全过 + 配置 diff）
- **配置/代码 diff**：Persistent vs Reset128 训练器仅差「rollout 边界 longstate 清空块 + gate5 记录/汇报 + 命名/路由/manifest carry_mode」。`_env_step`/memories/mask/GAE/`_loss_fn`/优化器/配置/init-merge **逐字未改**；两臂网络文件 sha 冻结不变（slowgru `b265210597d0`、eventmem `6a5cd6955ac3`）。
- **门1 schema同 / 门2 step0逐位同 / 门3 参数量同**：Reset128 init 与 Persistent ckpt@0 **逐位相同**（SlowGRU=`5ae94ed0`、EventMem=`2a9a0f8b`），叶子数/元素数完全一致（SG 102叶/5695020元；EM 98叶/5367086元）。
- **门4 rollout内真实读写 / 门7 env不串线 / 门8 true-done重置 / 门13 长模块有限非零梯度**：全过（SG 残差门grad=23.4；EM=124.6）。
- **门5 rollout边界清空**（正式训练12 rollout）：起始 longstate 哈希恒为 init（SG=`229bcd09`、EM=`2919d3ce`，唯一值），携带进入哈希逐rollout变化 → 状态在rollout内累积、边界被清空。boundary_clear=True、clear_nontrivial=True。
- **门6 GTrXL未改**：网络sha同 + diff仅carry块。**门9 ckpt roundtrip_ok / 门10 EXACT_RESUME_PASS（SG A@8192==B2@8192=`1a4232e6`；EM=`67ee581c`）/ 门11 无NaN / 门12 熵不坍塌**（SG 0.44–1.35；EM 0.44–1.69）。
- **两臂 ALL_13_PASS=True**。

## 2. 五臂统一评估表（@24576，256世界配对，seed42）
锚点精确复现：BASELINE teacher17500=**39.45%**；Control@24576=**36.33%**（control_matches_canonical=True）。

| 臂 | DK SR | floor3 | ENTER_SEWERS | 条件击杀 | death | timeout | eplen |
|---|---|---|---|---|---|---|---|
| Control@24576 | 36.33% | 43.36% | 43.36% | 83.8% | 156 | 7 | 862 |
| SlowGRU-Persistent | 42.97% | 50.78% | 50.78% | 84.6% | 142 | 4 | 796 |
| **SlowGRU-Reset128** | **44.53%** | 49.61% | 49.61% | 89.8% | 137 | 5 | 921 |
| EventMem-Persistent | 37.50% | 43.75% | 43.75% | 85.7% | 156 | 4 | 960 |
| **EventMem-Reset128** | **40.23%** | 46.48% | 46.48% | 86.5% | 148 | 5 | 887 |

**关键观察：两臂 Reset128 均 ≥ Persistent**（SlowGRU 44.53>42.97；EventMem 40.23>37.50）——去掉跨rollout携带不掉点反而略升。

## 3. Persistent − Reset128 因果效应（carry，配对）
| 臂 | ΔSR | 世界 | floor3 | death | McNemar p | 配对CI95 |
|---|---|---|---|---|---|---|
| SlowGRU carry | **−1.56pp** | −4 | +1.17 | +1.95 | 0.737 | [−8.20,+5.08] |
| EventMem carry | **−2.73pp** | −7 | −2.73 | +3.12 | 0.510 | [−9.77,+4.30] |

**两臂 carry 均为负**：跨rollout长期状态携带对 DK SR 无正向因果贡献（Reset128 反而略高）。

## 4. Reset128 − Control 共训练效应（配对）
| 臂 | ΔSR | 世界 | floor3 | death | McNemar p | 配对CI95 |
|---|---|---|---|---|---|---|
| SlowGRU 共训练 | **+8.20pp** | +21 | +6.25 | −7.42 | **0.038** | [+0.78,+15.23] |
| EventMem 共训练 | +3.91pp | +10 | +3.12 | −3.12 | 0.363 | [−3.91,+11.33] |

**增益全部来自共训练/正则**（含额外参数与rollout内计算，但无跨rollout携带）。SlowGRU 共训练效应大且显著。
⚠️ 不得把 Persistent−Control 当作长期记忆收益：SG=+6.64pp、EM=+1.17pp，其中 carry 贡献 ≤0。

## 5. 行为与策略消融
- **推理期关闭长期通道**（因果门5，要求≥2pp）：SlowGRU-Persistent on−off=**−0.78pp**（关闭反升，失败）；EventMem-Persistent on−off=**+1.17pp**（<2pp，失败）。
  （Reset128 模型 on−off：SG=+7.03pp、EM=+3.12pp——这是 rollout 内依赖，非跨rollout carry。）
- **Actor 是否真实读取长期态**（因果门6，populated 态动作KL/top翻转）：
  SlowGRU KL=6.43e-5、top翻转=0.0（几乎不读，失败）；EventMem KL=8.10e-4、top翻转=0.0（KL>1e-4 通过，但 carry 仍为负）。
- **行为指标**：两臂 Reset128 的 death 均低于 Control（SG 137、EM 148 vs 156），floor3/sewers 上升——这些方向性收益保留在无跨rollout携带的 Reset128 中，进一步证明它们源于共训练而非 carry。

## 6. 最终标签（6 条件因果资格门）
| 臂 | c1 carry≥+4pp | c2 floor3 | c3 death | c4 显著性 | c5 推理关闭≥2pp | c6 Actor读取 | 结论 |
|---|---|---|---|---|---|---|---|
| SlowGRU | ✗(−1.56) | ✓ | ✓ | ✗ | ✗ | ✗ | **SLOWGRU_TRAINING_REGULARIZATION_ONLY** |
| EventMem | ✗(−2.73) | ✗ | ✗ | ✗ | ✗ | ✓ | **EVENTMEM_TRAINING_REGULARIZATION_ONLY** |

- LONG_MEMORY_CAUSAL_SIGNAL：SlowGRU=**false**，EventMem=**false**。
- TRAINING_REGULARIZATION_SIGNAL：SlowGRU=**true**，EventMem=**true**（Reset128≥Persistent 且均优于 Control）。
- **LONG_MEMORY_WINNER = NONE**（两臂均未通过 carry 因果资格门；不因 Persistent 本身 SR 较高而选 SlowGRU）。

**结论**：在训练期匹配消融下，**没有任何长期记忆结构被证明因果有益**。SlowGRU/EventMemory32 相对 Control 的增益完全来自共训练/正则效应，跨128步长期状态携带本身贡献≤0。这与 Phase1 的消融签名（SlowGRU D_off≥A_on；EventMem 仅+1.17pp）及 P8 归因一致，并以更严格的训练期匹配设计确证「128步窗口非瓶颈、长期记忆 carry 非赢因」。

## 7. 路径与 SHA
| 项 | 值 |
|---|---|
| teacher ckpt17500 sha | d4e85af58b7f87d6 |
| Control@24576 sha | ece6fa9962e815123ce947577a93040057bc9df0b1e686dd28424cb2bbdabf55 |
| slowgru_network.py / eventmem_network.py sha | b265210597d00321 / 6a5cd6955ac34cd8 |
| 评估脚本 sha | 224514026aefd273 (eval_phase2_unified.py) |
| SlowGRU init / Persistent@24576 / Reset128@24576 | 5ae94ed0 / 1bd4fbfe91ab4da4 / 2ffdd269b94e1e6b |
| EventMem init / Persistent@24576 / Reset128@24576 | 2a9a0f8b / 11307081315f8059 / a3030f387c2e8cbb |
| exact-resume@8192 (SG/EM) | 1a4232e6fb816987 / 67ee581c7ddbca20 |
| gate5 init_ls_hash (SG/EM) | 229bcd099658fc1b / 2919d3ce66d6406e |
| SlowGRU-Reset128 目录 | gpu2_lc_slowgru_reset128/{src, train_24576/ckpt/{0,4096,24576}} |
| EventMem-Reset128 目录 | gpu3_lc_eventmem32_reset128/{src, train_24576/ckpt/{0,4096,24576}} |
| 只读参照 | gpu2_lc_slowgru / gpu2_lc_eventmem32（Persistent@24576）；control_RUN2（Control@24576） |
| 统一评估原始JSON | reports/phase2_unified_eval.json |

GPU 使用：GPU2=SlowGRU-Reset128，GPU3=EventMem-Reset128（并行训练）；统一5臂评估在 GPU2。未触碰 GPU0/1、P0/P1、D052、P2 POSTHOC。
