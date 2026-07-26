# P7-EGOMAP 核验报告 (2026-07-25)

## 1. 训练锚点核验

| 锚点 | 期望 SHA | P7 Control base-only SHA | 结果 |
|------|----------|--------------------------|------|
| step0 = ckpt17500 | d4e85af5... | d4e85af5... (80 leaves) | ✅ PASS |
| step24576 | ece6fa99... | 5ab64c27... (80 leaves) | ❌ FAIL |
| step98304 | 25accab4... | f29daf5d... (80 leaves) | ❌ FAIL |

### 根因：超参数不匹配

| 参数 | Canonical Control | P7 launcher_p7.py |
|------|-------------------|-------------------|
| update_epochs | **1** | **4** |
| num_minibatches | **2** | **8** |
| 每 rollout 梯度步数 | 2 | 32 (16×) |
| mode | score | task |
| XLA deterministic_ops | true | 未设置 |
| lr | 2e-5 constant | 2e-5 constant ✅ |
| num_envs | 16 | 16 ✅ |
| num_steps | 128 | 128 ✅ |

P7 Control 每 rollout 做 32 次梯度更新（canonical 仅 2 次），导致策略严重偏离。

## 2. 评估锚点核验

| 锚点 | 期望 | 实际 | 结果 |
|------|------|------|------|
| Baseline (ckpt17500) | ~98/256 | 98/256 (38.28%) | ✅ PASS |
| Canonical Control@98304 | ~88/256 | 88/256 (34.38%) | ✅ PASS |
| Evaluator SHA | 51c37c27... | 51c37c27... | ✅ PASS |
| 每 run 正好 256 条 | 256 | 256 | ✅ PASS |

评估协议：stochastic, seed_base=100000, Stage4-native, spawn_floor=2, max_steps=4096, DEFEAT_KOBOLD ever-set。

## 3. 修正后两臂结果 (256-world, stochastic, seed_base=100000)

| 臂 | SR | n_success | floor3 | cond_kill|floor3 | died | timeout |
|----|-----|-----------|--------|-------------------|------|---------|
| Canonical Control@98304 | 34.38% | 88/256 | 47.27% | 0.727 | 163 | 5 |
| **P7 Control@98304** | **17.97%** | **46/256** | 39.06% | 0.460 | 201 | 9 |
| **P7 EgoMap@98304** | **16.41%** | **42/256** | 36.33% | 0.452 | 203 | 11 |
| Baseline (ckpt17500) | 38.28% | 98/256 | 44.53% | 0.860 | 149 | 9 |

P7 EgoMap vs P7 Control: ΔSR = -1.56% (42 vs 46, -4 worlds)
P7 Control vs Canonical Control: ΔSR = -16.41% (46 vs 88, -42 worlds) ← 超参错误导致

## 4. 最终 P7 裁决



**P7 实验无效**：Control 臂因 update_epochs/num_minibatches 超参错误（4/8 vs 1/2）
严重偏离 canonical Control（46/256 vs 88/256），无法作为 EgoMap 的有效对照。
EgoMap 臂（42/256）甚至略低于已损坏的 P7 Control（46/256），
但此比较无意义——两臂均因超参错误而严重退化。

## 5. SHA 清单

| 文件 | SHA256 |
|------|--------|
| eval_paired_256.py (canonical evaluator) | 51c37c2759fe6a0ebc990c916c29b59f01126f4d2ef6c2ae1ac1bf4c1d99f88b |
| eval_p7_egomap_paired_256.py (P7 evaluator) | c082db8b82e86b971d8943bd9275ba8b709ffdc0da198fb236c52ccd56c08325 |
| ckpt17500 base params | d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5 |
| canonical Control@24576 | ece6fa9962e815123ce947577a93040057bc9df0b1e686dd28424cb2bbdabf55 |
| canonical Control@98304 | 25accab4ffd71061ea66acc167404fb743ad79799217e26407e3f075828cd64d |
| P7 Control@0 base-only | d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5 ✅ |
| P7 Control@24576 base-only | 5ab64c2708637b753bc11deb0d3e5fde503456743f75acdd7fdea5d0660e7088 ❌ |
| P7 Control@98304 base-only | f29daf5d399e58c9d4863350b6c0949fba735250499030c6d893089631bad8a7 ❌ |
| P7 Control@98304 full (87) | b47caac34f43580c63f29762ebdf0bb7cbc284d8f1dcaa34f77fb1aa2178b099 |

## 6. 产物路径

- P7 Control eval: gpu1_p7_egomap/eval_paired_out/p7_control_eval/results/
- P7 EgoMap eval: gpu1_p7_egomap/eval_paired_out/p7_egomap_eval/results/
- Canonical eval: exploratory_delayed_onset_20260724/eval_prep/eval_out/results/
- P7 Control orbax: gpu1_p7_egomap/eval_paired_out/p7_control_orbax/98304/
