# 交接增补(2026-07-21 夜):φ 重测全卷 + 存档双陷阱军规 + 周五资产更新

> 追加于《交接状态_崩溃侦破与周五备牌.md》之后,周三合并进主文档。

## 一、φ 重测案全卷(已结,7/21)

**背景**:原 φ 双 regime FAIL(366fe91)判决无效——判决出自 shaping 被
step/step_env 静默 bypass 的实验(84d7386 勘误)。重测前提 7/21 全部到位:
bypass 修复 + liveness 3 passed + 在线通道修复 + clamp(8221f3d)。

**设置**:`+training.depth_potential_c=0.5`(历史原值,wandb 存档抄录),
13700 起 fork,素净对照组同款 overrides,shaping 活体印章
`[DepthPotential] ACTIVE c=0.5` 两段日志均在案。

**主判据(总分):null 维持。** PHI1300(总15000 离线官方)= 43.04;
PHIEND2(总15200)= 43.11;在线 s172(总15400)= 43.84。对照:A 收官
43.21、rep2 尸检 42.61——全部 ±0.8 带内。"三 null" 落定为真·三 null
(第三 null 首次在 shaping 确认生效的条件下测得)。

**副判据(行为三针,预注册):第一针历史性大动。** 对 43.5 基线地形图:

| 针 | 基线 | PHI1300 | PHIEND2 | 读法 |
|---|---|---|---|---|
| 2 层步数/env | 1.0 | 19.2 | 15.9 | **19×,随后饱和/微回吐** |
| 2 层死亡 | 1 (0.1%) | 41 (4.0%) | 40 (3.9%) | 稳定 |
| 2 层击杀(gnome t1)| 1 | 151 | 136 | 深层战斗从全体~0 到实打 |
| 2 层交战比 | — | 1.32 | 1.20 | 楼下经济学持续劣势 |
| ep 均长 | 1670 | 1274 | 1278 | 用寿命换深度 |
| 渴死率 | 30.6% | 29.1% | 29.4% | 不动(φ 不教管水,如预测)|

**结论(周五口径)**:此前三套干预验尸"完全同形";**φ 是第一个改变
agent 死法的干预**——它把 agent 真的送下 2 层、真的打了 gnome,然后死于
1.20-1.32 的交换比与缩短的寿命。安慰剂金句修订:**激励能改"去哪",
改不了"活着";分数卡在转化环节**。且两点连线显示效应即时饱和、
不自我复利——**"找"的一半被 φ 一个 flag 演示成功,"lock"的一半
(SIL)被证明确实缺席**。find-and-lock 由此获得实证的腿;
φ 与 frontier-spawn 并列"找"位候选(φ 带数据)。

**诚实条款**:单 seed;严格 15400 同位离线尸体经两次存档事故不可得,
判决基于两具中期尸体(15000/15200)+ 在线全表(155-174),位置注记在案;
在线 s172 的 43.84 为带内上沿,不构成信号,如后续复测可留意。

## 二、新军规:存档双陷阱(7/21 双双踩实)

1. **orbax 原地续跑陷阱**:同 hydra.run.dir 续跑时,训练内部 update 计数
   分段重起(100,200,…),而 CheckpointManager 的 latest_step 仍是上段高位
   (如 1300)→ 低步号保存被**静默跳过**(异步、零报错,"Checkpointing
   agent state" 照打)。forkPhi_retest 的 s169-172 权重因此丢失。
   **军规:一切续跑 = 新 hydra.run.dir + 最新 ckpt 拷 rl_checkpoints/0
   (+ 被续 run 自己的 task_graph 与 runtime_analysis)**——拷 0 剧本的
   存在理由即此。
2. **异步存档 × SIGKILL 陷阱**:orbax enable_async_checkpointing=true,
   保存排队/写入中;收线 `tmux kill-session` 的 SIGKILL 会带走在途存档
   (tail 段 6 session 仅落 100/200 两档)。**军规:收线 kill 前,确认
   最后一个 "Checkpointing" 对应的步号目录已实际出现再动手**;必要时
   多等一个 save 周期(~10-15 分钟)。
3. mon.sh 修正:"resume(同 hydra.run.dir)"的老提示词违反军规 1,改为
   "resume = 新 dir + 拷 0 剧本";会话名提示 `train` 改为按当前会话名。

## 三、周五资产与提案栈(7/21 夜终版)

资产新增:**φ 重测章**(score-null × 行为 19× × 饱和曲线,三 null 的
第三席位首次带机制显微镜)| **死因地形图两张**(43.5 基线 + φ 臂对照,
仪表 fc2517a)| 崩溃案完整章(已入主文档)。

提案栈(死因地形图 + φ 实证共同排序):
**SIL(主攻,lock 半场,设计卡上桌)→ 找位候选二选一:φ(有数据)/
frontier-spawn(依赖 SIL 库)→ budget(条件搭档,扳机 = SIL 臂顶 cap
比例 >8%)→ RND(第三阶段)→ 逐杀稠密(毙,地形图三连理由)**。

## 四、run 对照表增补

| run | wandb id | 备注 |
|---|---|---|
| forkPhi_retest 首段 | (View run 取)| s0=155, base=13700;s169-172 权重丢失(陷阱 1)|
| forkPhi_retest_tail | gejgawhc | 拷 0 重长段,门牌 169 起 base=15100;ckpt 100/200 在,300+ 被 SIGKILL(陷阱 2)|
| PHI1300 / PHIEND2 尸检 | — | /tmp/phi_mid、/tmp/phi_end2 的 details JSON,判读脚本 experiments/analysis/necro_verdict.py |
