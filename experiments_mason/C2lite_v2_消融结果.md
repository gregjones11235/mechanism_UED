# C2lite-v2 3e8 消融结果:不过线,teacher 侧按止损线收摊

> 2026-07-15。run 目录 2026-07-15_055441_283639(24 sessions,~3e8,total_timesteps=2e9
> schedule 对齐主跑),eval = eval_C2LITEV2ABL_seed0.json(5 点官方离线)。

## 裁决

| update | v1 主跑(离线) | v2 消融 | Δ |
|---|---|---|---|
| 300 | 12.49 | 13.66 | +1.2 |
| 900 | 25.43 | 18.26 | −7.2(后程追平,单 seed 方差带内) |
| 1500 | 31.02 | 29.80 | −1.2 |
| 2100 | 32.33 | 32.58 | +0.3 |
| 2400 | 在线 ~33-35 | 33.14 | ≈平 |

2400 技能:iron_sword 3.6 / pickaxe 4.1 / armour 0.5 / **gnomish 0**。
预注册判据(iron 巩固斜率肉眼抬升,或 gnomish >20%)**双双未过 → 按止损线,teacher 侧
正式收摊**:"43.5 附近即本框架 teacher 侧极限"进入论文结论;剩余预算转 seeds(周五定)。

## 两条诚实注记

1. **C-1 全场哑火(事后发现)**:check_compilation 返回 str(e) 无异常类前缀,分类器正则
   要求前缀 → hallucination-class 0/45。已修(prefix-optional + H4 缺 import 类,62 tests)。
   因此本消融实际 = **R3 豁免单变量测试**(归因反而更干净);C-1 维持"已实现 + 离线验证"
   定位,在体疗效未测。零成本补验:拿本 run 日志 45 条错误串离线跑 diagnose 看分类率。
2. **Regime 混杂**:R3 豁免只对已掌握前置生效,0-3e8 冷启动期 mastered 集小
   (enter_dungeon 到末段才 84%),豁免几乎无开火机会——本消融把它放在最弱 regime 测。
   干净测法 = 晚期 checkpoint resume-A/B(dungeon 96% 的 regime)。按纪律不单方面加跑,
   作为周五预算会议的备选项记录。

## ScaffoldGate 全程(v2)

checked 99 / violations 88(89%)/ repaired 74(84%)/ dropped 14(14%)——与 v1
(92/85/14)持平;豁免未降低违规率(早期 mastered 集小 + R1/R2 不在豁免管辖)。

## 收摊后的线

teacher 侧关账;seeds vs resume-A/B vs student 侧立项 → 周五 RA 会三选。
