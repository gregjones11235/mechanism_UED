# Frontier 锁死分析与修复（★A mastery 阈值 × 评测口径错配）

> 2026-07-09。第二轮实验候选之一（与 seeds / ★C Stage 2 并列，优先级待 PI 排序）。
> 关联：`official_eval_v2.md` §5.3。

## 1. 现象

- 全部机制臂（85qid2ev / u1gjqror / z8jygtyw）的 `[SkillGraph] frontier tier` **从 session ~2 起全程钉死 tier 2**（铁器族 targets），300M+ 步未推进到 tier 3；
- 机制臂 mean_return 在 ~30-31 出现疑似平台（v2 主表）；tier-3/4 技能（钻石装、附魔、深层怪）三臂全程为 0。

## 2. 根因链（全部实证，非推测）

1. hook 调用：`run_dicode.py` L180 `pick_target(evaluation_metrics)` —— 传入的是 **`run_session_evaluation` 的输出 = held-out 裸评估口径**（L87 处生成）；
2. 判定逻辑：`skill_scheduler.py` —— tier 计为"已掌握"需 **该 tier 平均 SR ≥ threshold（默认 0.60）**；`reachable_ceiling` 返回第一个未掌握 tier 作为 frontier（L61-74）；
3. 数据事实：iron 族 held-out 裸 SR 全程 **0-1% 量级**（eval JSON）→ tier 2 平均 SR 距 0.60 差两个数量级；
4. **结论：frontier 数学上不可能推进到 tier 3。** 非代码 bug（逻辑按设计运行），是**阈值语义（按训练任务 SR 直觉设定）与实际口径（苛刻的裸 held-out SR）的错配**。

含义：平台 ~30 的成因至少有二——(a) 14B 生成 tier-3 级复杂任务的能力上限（生成端）；(b) **调度器被口径锁死，根本未让生成端尝试 tier 3**（调度端，本文档，可修）。(b) 修复后才能分离测量 (a)。

## 3. 阈值两难（设计分析）

| 阈值 | 失败模式 | 状态 |
|---|---|---|
| 过高（0.60 on 裸 SR） | frontier 永久锁死 → 课程边际收益递减 → 平台 | **已观测** |
| 过低（如 0.05） | iron 裸 SR 刚过 5% 即推 tier 3 → 前置依赖未备 → 候选大面积不可学 | 风险（Mason 提出） |

**两个缓冲**使风险可控：
1. **preflight 兜底**（仅 +A+B 臂）：不可学候选被 B 闸拦截（`sr<0.05` 拒），系统自动回退用 archive 训练——**A 试探边界，B 封顶下行**。若 probe 中 preflight 拒绝率从 3% 飙升，本身即"student 未就绪"的量化证据，数据仍有效；
2. **逐 session 重算**：`pick_target` 无棘轮，误推进下个 session 自动回落，代价 ≤ 1-2 个 session 的生成预算。

## 4. 修复

### 方案一（已实施 flag，待标定数值）：阈值校准至 onset 区间
- **Patch 已打**（pod，2026-07-09；待同步 Windows push）：`run_dicode.py` L180 改为
  `pick_target(evaluation_metrics, threshold=config.get("skill_preflight",{}).get("mastery_threshold", 0.60))`
  —— **默认 0.60 行为不变**（全部已有 run 有效），新增命令行开关 `+skill_preflight.mastery_threshold=X`。
- 数值原则：对齐"学习曲线进入自持增长段"（onset）而非"完全掌握"——UED 的 frontier 语义本就该在 ZPD（会一点未熟）处。初判 **0.15-0.25**，按 eval JSON 标定：

```bash
# 标定：看各 tier 技能在末点 checkpoint 的裸 SR 分布，取 tier1(已通)/tier2(未通)自然分界
python3 - <<'EOF'
import json
for tag, f in [("BASEEXT","eval_BASEEXT14B_seed0.json"), ("ARMAEXT","eval_ARMAEXT14B_seed0.json"), ("ARMAB","eval_ARMAB14B_seed0.json")]:
    d = json.load(open(f"/workspace/eval_out/archive/{f}")) if False else json.load(open(f))
    # 结构自适应:找最后一个 step 的 skill_ 键
    last = d[max(d, key=lambda k: int(k))] if isinstance(d, dict) and all(k.isdigit() for k in d) else d
    # 打印全部 skill_ 值,人工看分界(结构不明时先 print(list(d)) 摸 schema)
    print(tag, {k: round(v,3) for k,v in (last.items() if isinstance(last,dict) else []) if str(k).startswith("skill")})
EOF
```
（JSON schema 若与上不符，先 `python3 -c "import json;d=json.load(open('...'));print(type(d), list(d)[:5])"` 摸结构再取数。）

### 方案二（Stage 2 工程项）：mastery 加权混合 target
frontier 推进后按 mastery 加权混合生成（如 70% tier2 + 30% tier3），消除悬崖式切换。需改 `format_target_for_prompt` 注入逻辑，非纯参数，归入 ★C Stage 2 同批工程。

## 5. 建议 probe（待 PI 排序）

- **臂**：+A+B + `mastery_threshold=0.2`（B 兜底下行风险）；对照 = 现成 u1gjqror（0.60），同 seed 同环境单 flag 差；
- **成本**：一条 run（~7h 到 session 23）+ 8 点 eval；
- **读数**：frontier 是否解锁 tier 3（日志 `[SkillGraph] frontier tier 3` 出现）；preflight 拒绝率变化；mean_return 平台是否抬升；tier-3 技能是否首次非零；
- **判据**：任一为真即有信息量——解锁+抬升 = 修复有效；解锁+拒绝率飙升 = 生成端上限实证（分离出成因 (a)），转 ★C。
