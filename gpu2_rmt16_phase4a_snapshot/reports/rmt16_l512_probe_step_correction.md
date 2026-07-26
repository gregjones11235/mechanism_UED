# RMT16 L512 探针 — resolved-step 出处精化（§二，前后对照）

**任务**：`RMT16_PHASE4A_V2_ORIGINAL_GOAL_VTRACE_IMPLEMENTATION` §二
**性质**：**离线**重算既有 episode 记录的精确出处；**不**重跑 16384 探针，**不**改变可达性判定。
**冻结结论（不变）**：`L512_REACHABILITY_BOTH = PASS`（Persistent 6/20，Reset128 5/21；Replay=OFF，Hindsight=OFF；探针**不**做 Carry/性能断言）。
**重算执行环境**：服务器 CPU，`/home/oseasy/miniconda3/envs/dicode310/bin/python`（py3.10.20）；脚本 `tests/recompute_probe_step.py`。
**数据源**（冻结探针运行输出，**不在** Git 仓库内）：
- Persistent：`runs/RMT16-PERSISTENT-PROBE-L512-16384/out/RMT16-Persistent-PPO_probe_episodes.jsonl`
- Reset128：`runs/RMT16-RESET128-PROBE-L512-16384/out/RMT16-Reset128-PPO_probe_episodes.jsonl`
- 探针常量：`num_envs=16, rollout_steps=128`（每条记录含 `update_index, rollout_step, env_id, length, episode_id, completion_global_step`）。

---

## 1. 被修正的旧字段

旧（探针沿用）：
```
completion_global_step = update_index*(num_envs*rollout_steps) + rollout_step
```
缺陷：`rollout_step` 项**漏乘 `num_envs`**、**漏掉每 env 的 `env_id` 偏移**、**漏掉 `+1`**。
它既少计，又把同一 rollout_step 的全部 16 个并行 env 折叠到同一整数，故**不是**精确解析 env step。
本轮将其保留为对照字段，并标记 `completion_global_step_deprecated = True`。

## 2. 新的权威字段

```
completion_resolved_env_step =
    outer_update_index * num_envs * rollout_steps
  + rollout_step * num_envs
  + env_id
  + 1
```
语义：每个 outer update 消耗 `num_envs*rollout_steps = 16*128 = 2048` 个解析 env step；update 内第
`rollout_step` 个步进同时推进全部 16 个 env，env `env_id` 是该 update 的第 `(rollout_step*16 + env_id)`
个解析 step；`+1` 使计数 1-索引（整条 run 的第一个解析 step == 1）。

## 3. 两臂首条 ≥512 episode 的精确 step（离线重算实测）

| 臂 | 完成 episode 数 | count_ge512 | reachable | first episode_id | length | update_index | rollout_step | env_id | **resolved（新）** | deprecated（旧） | delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistent | 20 | **6** | **true** | 2 | 562 | 4 | 49 | 2 | **8979** | 8241 | **738** |
| Reset128   | 21 | **5** | **true** | 2 | 562 | 4 | 49 | 2 | **8979** | 8241 | **738** |

验算（Persistent / Reset128 首条相同）：
```
resolved = 4*16*128 + 49*16 + 2 + 1 = 8192 + 784 + 3 = 8979
deprecated = 4*2048 + 49 = 8241
delta = 8979 - 8241 = 738  (= rollout_step*15 + env_id + 1 = 49*15 + 2 + 1)
```
`count_ge512` 与冻结证据 `rmt16_l512_probe_final.json` 完全一致（Persistent 6、Reset128 5；两臂
`first_ge512_update=4 / first_ge512_global_step(deprecated)=8241 / episode_id=2 / length=562`）。

## 4. 结论

- **可达性判定不变**：两臂均 `reachable=true`，`L512_REACHABILITY_BOTH = PASS` 维持。
- **出处被精化**：首条 ≥512 episode 的精确解析 env step 为 **8979**（旧非精确字段为 8241，少计 738）。
- 本修正**仅**改变 step 的"出处/数值标注"，**不**触及任何训练数值、探针重跑、Carry/performance 断言。
- GATE 1（"旧 L512 可达性可由原始 episode 记录重算且结论仍 BOTH"）：方法自检（合成记录）本地 PASS；
  两臂真实 jsonl 重算服务器 PASS（即上表）。
