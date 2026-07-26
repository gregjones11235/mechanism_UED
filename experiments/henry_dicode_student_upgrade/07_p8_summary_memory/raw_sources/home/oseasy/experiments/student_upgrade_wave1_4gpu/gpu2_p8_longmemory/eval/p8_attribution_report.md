# P8-LONGMEM-SUMMARY — 最小只读失败归因（GPU2）

最终标签（已定）：**NO_POSITIVE_SIGNAL**。本归因只读分析其失败机理（不训练、不改码）。

## Q1 长期读出是否被启用（参数范数曲线）
长期摘要进入 actor 的唯一通道 summary_to_actor（零初始化）kernel 范数随训练单调增长：
4096=0.0395 → 24576=0.0451 → 49152=0.0500 → 73728=0.0569 → 98304=0.0601。
→ 长期路径确实被启用，且越训越强。

## Q2 因果：评估期关闭长期路径（long-OFF）vs 开启（long-ON）vs 同step Control（256世界，自然起点）
| step | long-ON SR | long-OFF SR | Control SR | on-vs-ctl | off-vs-on(关掉长期带来的恢复) |
|------|-----------|-------------|-----------|-----------|------------------------------|
| 24576 | 34.77 | 35.16 | 36.33 | -1.56 | +0.39 |
| 49152 | 27.73 | 32.03 | 37.11 | -9.38 | +4.30 |
| 73728 | 23.83 | 26.56 | 31.25 | -7.42 | +2.73 |
| 98304 | 23.44 | 27.34 | 37.50 | -14.06 | +3.91 |

## 结论（两条机理）
1. **主因——长期读出直接有害**：每个节点关闭长期路径都把 SR 拉回（off-vs-on = +0.39/+4.30/+2.73/+3.91pp）。
   零初始化门一旦打开，就把有害上下文注入 actor；门单调打开（Q1）与 long-ON SR 单调下滑同步，
   off-vs-on 的恢复直接证明因果。
2. **次因——共训练中毒**：即使评估期关掉长期路径，P8 已训练的短期策略仍低于 Control，且差距随训练扩大
   （off-vs-ctl = -1.17/-5.08/-4.69/-10.16pp）。短期 GTrXL/encoder/actor 头在与有害长期路径共同训练时被
   其梯度带偏，自身已偏离 Control。

→ 128步窗口的长期摘要并非有益记忆，而是**直接有害 + 污染共训练**双重负面。这稳健解释了 P8 单调退化的
NO_POSITIVE_SIGNAL，并支持"128步窗口不是瓶颈"的判断。

## 产物
- 数据：/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/eval/p8_attribution.json
- 依赖：/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/eval/p8_final_gate.json（long-ON 与 Control 原始 256 世界结果）
- 检查点：/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/train_98304/ckpt/{4096,24576,49152,73728,98304}/full_state.pkl
