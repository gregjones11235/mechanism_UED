# ★C Stage 1 结果：离线修复级联基准（判据通过，Stage 2 待批）

> 2026-07-08。工具 = `dicode_src/stage1_repair_bench.py`（可复现，seed=0）。
> 设定 = 30 件真实 14B 生成任务代码（取自 baseline task_graph.graphml）× 注入取自**有机幻觉分布**的故障
> → C-0 静态 lint 检测 → C-1 qwen2.5-coder:14b 修复（≤2 次尝试，temp 0.2）→ 三级验证（ast / exec / Env 实例化，镜像 gen_manager.load_env）。
> 原始数据：`stage1_report.json`。

## 1. 判据结果：全过

| 指标 | 结果 | 判据 |
|---|---|---|
| C-0 lint 检测率 | **1.000**（30/30） | 预期 ~100% ✓ |
| C-1 修复成功率（检出条件下） | **1.000** | **>0.7 放行线 → 大幅通过** ✓ |
| 每次修复平均 token（入+出） | **2,127.8** | 远低于一次完整重生成；且为"输出整文件"的保守上界（patch 式输出可再降） |

**结论：Stage 2（接入管线为第四臂 +A+B+C）的离线可行性判据通过。** 待 PI 批准后实施（flag-gated `use_repair`，默认关）。

## 2. 有机幻觉清单（三个 run 日志全量收集）

| 类 | 幻觉成员（次数） |
|---|---|
| Achievement | DEFEAT_ORC_SOLDIER (9)、DESCEND (5)、DEFEND_SKELETON (2)、DEFEAT_BAT (2)、DESCENT (1)、COLLECT_BAT (1) |
| BlockType | **LADDER_DOWN (17)**、**BOW (10)**、TORCH (2)、BAT (2)、FLOOR (1)、ITEM_BOW (1)、NONE (1)、DIRT (1) |

## 3. 两个发现（比"小模型会瞎编"深一层）

### 3.1 幻觉 = 跨游戏常识污染，不是随机噪声
高频幻觉几乎全是 **Minecraft/roguelike 通用概念**：LADDER_DOWN、BOW、TORCH、FLOOR、DIRT——在同类游戏里真实存在，只是 Craftax 的枚举里没有。最典型的一例：**`DEFEAT_ORC_SOLDIER`（9 次）**——craftax 源码里真身拼写是 `DEFEAT_ORC_SOLIDER`（源码自身的拼写错误，见 wandb 指标名 `skill_defeat_orc_solider`），**14B 按正确英语"纠正"了源码的 typo，反而触发 AttributeError**。
→ 定性：小模型的失败模式是**把通用游戏先验与规范拼写投射到具体 API 上**。这解释了为何幻觉集中在语义层而语法为零，也说明该失败随任务深度增加（更长的 API 引用链）只会更频繁——机制修复的必要性随深度上升。

### 3.2 prompt 清单是干净的 → 幻觉是模型内生的
交叉参照：真身成员 **0 个**缺失于 prompt；幻觉成员 **0 个**出现在 prompt 里。
→ 排除"补全 prompt 就能修"的便宜解释：**幻觉源于模型先验而非信息缺失，机制级修复（★C）是必要路径而非可绕开项。**
（`prompt_extra_not_real` 里的 5 项为正则误捕获的 MAP/常量名，非枚举成员，无害。）

## 4. 局限（诚实清单）

1. **注入式故障**（从有机分布采样注入真实代码），非端到端有机失败——测不到幻觉伴生的其他语义问题；Stage 2 的在线运行即是有机验证。
2. 验证到 **Env 实例化级**，未跑 `--deep`（reset/world-gen 级，BlockType.BAT 类深层故障未覆盖）——Stage 2 前可补一轮 --deep 抽查。
3. 修复器 = 生成器本身（qwen 自修）；**Q2（异构修复器，如 deepseek-coder）未测**——留待 Stage 2 消融。
4. 修复正确性 = "编译+实例化通过"，未人工审语义保真度——建议抽查 10 件（§5）。

## 5. 下一步

1. 周五组会：以本结果为 ★C 提案的实证支撑，请 PI 裁定 Stage 2 优先级（vs seeds / 阈值敏感性）。
2. 批准后 Stage 2：`use_repair` hook 接入 `check_compilation` 失败分支 → +A+B+C 臂跑 session 10 → 官方协议 eval 同表。
3. 补充：--deep 抽查；10 件修复件语义人工抽查；（可选）拉一个 deepseek-coder 小模型测 Q2 异构修复。
4. 给 craftax 上游提个 issue/PR 修 `DEFEAT_ORC_SOLIDER` 拼写（顺手的社区贡献，也让这个发现有出处）。
