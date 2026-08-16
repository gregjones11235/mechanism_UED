# C2lite 短跑验证结果:三判据全过 → 2e9 主跑放行

> 2026-07-12。run = **`p2pyfhb9`**(C2lite_shortrun_s10,hydra `outputs/2026-07-12_005155_302279`,
> 140M steps / 9 sessions 善终,checkpoint 至 1100,archive 64 nodes / 25 trainable,
> activation_success_ratio 收 0.81 且全程走高)。配置 = 实现记录 §5(修正:**去掉
> `+validation=default`** —— config defaults 已内置,重复声明 hydra 即死,首次点火因此夭折)。
> 对比基准 = probe(t=0.2,oun2yfm6),与本臂仅差三个 C2lite flag,同 seed 同环境。

## 判据结果一览

| 判据 | 结果 | 关键数字 |
|---|---|---|
| 1. 泄漏语义 ↓ | ✅ | S2 预标通道实质关闭:26 生成件中 3 件有预标、均数 1.0(302 基线 = 14 连预标常态);逐件时序:早期 6 件 R1 残留(闸未获快照窗口),**task_17 起 24/24 连续零预标** |
| 2. tier-3 裸指标动(核心) | ✅ | iron_sword / iron_pickaxe 学习信号**提前 ~50M 出现**(C2lite ~110-140M 抬到 2-2.5%,probe 同期为 0、~150M+ 才起步);对照组 collect_iron 两臂重合(排除全局 rng 好运);iron_armour 双零(预期,链更长,2e9 观察项) |
| 3. 拒绝率可控 | ✅ | preflight 拒绝率 19%(kept 21/26,随 session 走低);闸 dropped 14%(5/36) |

## 机制画像(全程累计)

- **ScaffoldGate:checked 36 / violations 33(92%)/ repaired 28(85% 救回)/ dropped 5(14%)。**
  一句话:**prompt 教不会(裸服从率 8%),罪证修得回(85%)** —— 闸+修复回路是必需品而非保险;
  判据 1 的干净是"修出来的",不是模型自发服从。
- **前沿逐节点爬链实录**:石器层 → make_stone_pickaxe 毕业 → collect_iron 顶上 → iron 三件套
  集体入池 → 末段摸到 tier 3(enter_gnomish_mines / defeat_orc_mage)。**回放测试的 cap 担忧
  销案**:真实 SR 分布下 iron_pickaxe 稳居目标池,cap=6 不动。
- **S1 之谜**:25/26 件带库存,但几乎全是 `sword:1`(floor-1 任务防身最低配)+ 少量原料,
  无焦点工具授予 —— 契约允许形态,非泄漏。

## 幻觉线新账(C-1 弹药,不动本期队列)

- 幻觉致弃 **17 件**,= 闸 dropped 的 3.4 倍,≈ 入闸候选量的 47%,是第一大丢件来源;
- 分布跨 run 稳定:LADDER_DOWN 8 / DESCEND 2 / FIND_LADDER_UP、DEFEND_ZOMBIE、FLOOR、PATH 各 1
  ——"下楼"类执念正撞在前沿所需的深度任务上,**C-1 救的是刀刃上的吞吐**;
- ★实施要点(本轮最值钱发现):`BlockType.PATH` 上游真实、wrapper 未暴露 → **C-1 白名单必须
  从生成代码实际 import 的 minicraftax 枚举构建,不能用 craftax 原版**;
- 决策:C-1 升格为"2e9 后第一优先接入项",本期不上(改动纪律:短跑验证过的配置原样上主跑)。

## 规则边界记录(不改,观察)

1. **floor=1 + rel=enter_dungeon 不违规**(enter_dungeon 直接前置为空集,R3 无从触发);若 2e9
   出现"白给分"迹象再收紧为 R2'。
2. **已掌握原料授予的规则间隙**(task_48:焦点 make_stone_pickaxe,inv 给 stone:20,授予了
   已掌握的直接前置 collect_stone)—— R3 只护未掌握焦点、prompt 契约第 3 条无闸背书。伤害趋零
   (该前置 96% 已掌握),不改闸(改则误杀合法补给),记录待 2e9 数据复判。

## 2e9 主跑定稿

配置 = 短跑原样,唯一改动 `training.total_timesteps=2000000000`;C-1 不上、cap 不动、阈值不动。
预计 ~55-65h。监控项新增:幻觉致弃累计数(C-1 提案数据)。收数后:官方 eval
(steps=[300..15300] 量级)→ gap 报告 vs 44.58 / 48.33。
