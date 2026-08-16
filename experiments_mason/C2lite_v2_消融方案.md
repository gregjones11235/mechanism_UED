# C2lite-v2:R3 已掌握豁免 + C-1 幻觉修复 —— 3e8 消融方案

> 2026-07-14。两个 flag,默认全关(v1 逐位复现);消融 = 2e9 定稿配置 + 两个新 flag,3e8(~10h)。
> 靶心 = gap 报告 §3 的结论:剩余 gap 几乎全在 iron/gnomish 巩固度 + 幻觉致弃 115+ 件的供给损耗。

## 改动清单

**① R3 已掌握豁免**(`+skill_preflight.r3_mastered_exemption=true`)
- `scaffold_gate.check_code`:新参 `mastered_prereq_exemption`——R3 检查跳过 SR≥mastered_cut(70%)
  的 immediate prereq;未掌握前置照旧保裸。三条新测试钉死(默认关=v1 逐位同、开=豁免、
  开但前置未掌握=仍拦)。
- one-step 契约 prompt + coder 规则块:同 flag 换措辞("未掌握的 immediate prereq 须裸做;
  已掌握的可提供/跳过——鼓励用起始层跳过已掌握的下楼前缀")。
- 合法性边界:泄漏定义=压缩未掌握步骤(裸复验 0.65→0.01 全部发生在此);压缩已掌握前缀
  =脚手架的本来用途。Henry v6 设计 §4 独立推导出同一规则(团队级双重确认)。

**② C-1 API 幻觉修复**(`+skill_preflight.use_api_repair=true`,`api_repair_retries` 默认 2)
- 新模块 `skill_preflight/api_lint.py`:错误驱动诊断——解析编译错误(H1 枚举成员 / H2 builder
  方法 / H3 ctor kwarg 三类,2e9 台账全覆盖),AST 扫任务代码的 import 定位符号真实来源,
  importlib 动态解析**真实成员表**(自动用 wrapper 的枚举,PATH 案例的教训),difflib 近似
  匹配给出罪证("X.Y 不存在,最接近的合法项:…")。非幻觉类错误返回 None → 走旧丢弃路径。
- hook 在 evolution_efficient 编译循环后、脚手架闸前;复用 repair_scaffold_violations 同一套
  修复回路;打印 [ApiRepair] compile-failures/hallucination-class/repaired 三个数。

## 消融配置(2026-07-14 修正:total_timesteps 不砍)

2e9 定稿命令**逐位原样**(含 `training.total_timesteps=2000000000`)+ 两个新 flag,seed 1;
**跑到 session ≈23(≈3e8)手动 kill**(mon.sh CUTOFF=23)。

**为什么 total_timesteps 必须保持 2e9**(baseline_v2 记录 §4.2 的家规):LR schedule 按
total_timesteps 归一化——砍到 3e8 会把 schedule 压 6.7 倍,①训练动力学变形(当年 5e8→3e6
压 167 倍直接训炸);②本消融的判据是与 2e9 主跑**同期**比巩固斜率,主跑前 3e8 走的是
2e9 归一化 schedule,消融必须同 schedule 才是单变量对比。时长一律靠手动停止控制
(五臂消融的既有标准)。诚实注记:短跑当时用了 140M 作 total,偏离此纪律——未出事是因为
判据效应量大(iron 提前 ~50M)+ 两条判据为机制行为而非斜率精比;v2 消融是斜率对比,
不再占这个便宜。

对照基线:主跑前 3e8 官方离线点(update 300/900/1500/2100 = 12.5/25.4/31.0/32.3,
eval_C2LITE2E9_seed0.json)+ wandb 在线段。

## 判据(止损线写死)

- **过线**:3e8 内 iron_sword/iron_armour 巩固速率相对 2e9 主跑同期肉眼可辨抬升,或
  enter_gnomish_mines 突破 20%;[ApiRepair] repaired 占 hallucination-class 的 ≥60%;
- **不过线**:teacher 侧正式收摊("43.5 附近即本框架 teacher 侧极限"本身即论文结论),
  剩余预算转 seeds。
- 过线后:与 PI/Alec 谈 v2 干净 2e9(同时治愈热修叙事 + 计数器序列化 bug)。

## 归因预案

打包消融(两 flag 同开)先回答"v2 值不值";若过线且需归因,再补单 flag 臂
(R3-only / C1-only)各 3e8——flag 全独立可拆,随时可做。

## 不在本次范围

Henry 的 chain_order_log(break-link mining)接入调度器——纯 python 已在他分支写好,
集成价值高,但等周五对稿后再动(跨线代码,先谈后拿)。
