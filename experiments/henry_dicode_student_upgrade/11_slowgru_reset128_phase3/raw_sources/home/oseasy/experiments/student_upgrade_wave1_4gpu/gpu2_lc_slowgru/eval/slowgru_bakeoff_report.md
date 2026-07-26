# LC-SLOWGRU-PPO 候选B 烘焙赛马归档报告（GPU2）

**结论：SLOWGRU = QUALIFIED（门通过，且明显强于 EventMemory32；但消融显示增益来自共训练短期策略，非长期慢状态）**

## 1. 设计与冻结协议
- 结构：原CNN → 原128步GTrXL 快速状态 → SlowGRU 长期状态（每 SLOW_INTERVAL=32 步对 GTrXL hidden 做
  注意力池化→投影 SLOW_DIM=256→普通 GRU 更新；慢状态跨 rollout 持续，true_done 清除）→ 经**零初始化残差门**
  `slow_to_actor` 注入 actor；critic 只读 x。**无 S5/SSM**（避免与队友 P0 重叠）。
- 协议：teacher ckpt17500 起点、Original PPO、deterministic ops、seed=42、LR=2e-5、Adam eps=1e-5、
  γ=0.999、λ=0.8、num_envs=16、rollout=128、Stage4-native、total=24576（12 updates）、存 0/4096/24576。
- 独立代码/配置/输出目录/检查点结构；无 Replay/V-trace/AWR/hindsight/novelty/NavAux/EgoMap。

## 2. 工程门（全部通过，ALL_PASS=True）
| 门 | 结果 | 门 | 结果 |
|---|---|---|---|
| 1 feature-off+init零门==teacher逐位 | PASS | 6 精确续训 A@8192==B2@8192=3192516c | PASS |
| 2 环境隔离 | PASS | 7 4096 smoke确定性 A@4096==B1@4096=00df4765 | PASS |
| 3 rollout连续性 | PASS | 8 长路径梯度有限非零（残差门 grad 2.343e+01） | PASS |
| 4 true-done重置 | PASS | 9 无NaN/熵不崩溃（熵 0.78–1.20） | PASS |
| 5 roundtrip逐位 | PASS | 10 清状态动作KL=6.43e-5>0 | PASS |

关键哈希：teacher_sha=d4e85af58b7f87d6；init_sha=5ae94ed0257f50fa；24576 params_sha=1bd4fbfe91ab4da4。

## 3. 256世界终评（随机策略，seed42，max4096，S4_dark 自然起点）
锚点精确复现：BASELINE(teacher17500) SR=39.45%；CTL@24576 SR=36.33%（与权威 Control 一致）。

| 变体 | DK SR | floor3 | 条件击杀 | death | timeout | sewers |
|---|---|---|---|---|---|---|
| BASELINE teacher17500 | 39.45% | 43.36% | 91.0% | 147 | 8 | 111 |
| CTL@24576 | 36.33% | 43.36% | 83.8% | 156 | 7 | 111 |
| **SG A_on（长开）** | **42.97%** | 50.78% | 84.6% | 142 | 4 | 130 |
| SG B_clear（每步清空h） | 42.19% | 46.09% | 91.5% | 145 | 3 | 118 |
| SG C_hdim_perm（h维置换） | 41.41% | 48.44% | 85.5% | 146 | 4 | 124 |
| SG D_off（仅短期） | 43.75% | 48.05% | 91.1% | 141 | 3 | 123 |

SG vs CTL：dSR=+6.64pp(+17世界)、dFloor3=+7.42pp、dDeath=-5.47pp、dSewers=+7.42pp。
McNemar p=0.0718（边际）；配对 bootstrap SR差 CI95=[+0.00,+13.28]pp。

## 4. 长期记忆因果消融（A/B/C/D）— 关键发现
- **D_off (43.75%) ≥ A_on (42.97%)**：关闭慢通道反而略升 SR。
- B_clear (42.19%)：每步清空 h→h 恒为 0→慢通道贡献为 0，却仍 ≈ A_on。
- 结论：**+6.64pp 增益全部来自共训练改善的短期策略**；累积慢状态贡献≈0（A_on−B_clear=+0.78pp），
  且中性偏负（A_on−D_off=−0.78pp）。这与 P8 归因一致（共训练提升 + 长期路径无益/轻微有害）。
- 记忆参与：slow_to_actor 范数 0→0.0257；gate10 KL=6.43e-5（极小，技术上>0）。

## 5. 资格门
| 判据 | 值 | 通过 |
|---|---|---|
| SR 相对 Control 下降 ≤5pp | -6.64pp（更高） | ✓ |
| floor3 保持 ≥90%（相对） | 保持率 1.171 | ✓ |
| death 无明显恶化 | -5.47pp（改善） | ✓ |
| ≥1 方向性信号 | SR/floor3/sewers↑、death/timeout↓ | ✓ |
| 记忆真实使用 | gate10 KL>0、slow_to_actor>0 | ✓ |
| 工程门全过 | 1–10 | ✓ |
| **QUALIFIED** | | **是** |

**重要限定**：(1) 信号边际显著（McNemar p=0.072，CI 下界恰为0）；(2) 因果消融表明长期慢状态非增益来源。
作为「长期记忆结构」筛选，SlowGRU 训练臂强但**其长期结构本身未被证明有益**——是否推进需总监裁决。

## 6. 产物路径
- 代码：`gpu2_lc_slowgru/src/{slowgru_network.py, run_slowgru_24576.py, gates_slowgru.py, eval_slowgru_final.py}`
- 训练：`gpu2_lc_slowgru/train_24576/{ckpt/{0,4096,24576}, LC_SLOWGRU_train_summary.json}`
- 评估：`gpu2_lc_slowgru/eval/{slowgru_final_gate.json, slowgru_qualification.json}`
- 门：`gpu2_lc_slowgru/src/LC_SLOWGRU_gates.json`；smoke/resume：`gpu2_lc_slowgru/smoke_resume.log`
