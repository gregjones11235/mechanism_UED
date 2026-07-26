# gpu2_rmt16_phase4a_snapshot — RMT16 L512 可达性探针运行源码快照

本目录是 **固定 16384 预算 L512 可达性探针**（Persistent / Reset128 两臂）的**实际运行源码逐字节快照与审计入口**。它**不是**第二套正式 package，也不进入运行时 `sys.path`；运行时仍从下方两棵原始工作树加载代码。本快照仅用于 Git 固化、字节一致性核验与可复现审计。

## 1. 这是什么
- 探针在两台 4×RTX-A6000 服务器的 GPU2/GPU3 上运行，固定 `total_updates=8`（8×2048=**16384 resolved env steps**/臂），在线 PPO，`--replay off --probe --early_stop_len 0 --seed 42`，从预冻结 step0（ckpt17500 基座）全新启动。
- 本快照 = 运行代码的逐字节副本 + before_probe 原件 + diff + 门禁证据 + manifest/import-map。

## 2. 三个修改文件来自两棵非 Git 工作树
运行目录本身**不是 Git 仓库**。三个被改文件分别位于两棵独立的非 Git 树：
- **wrapper 来源树**：`/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src/dicode/wrappers_cl.py`
- **experiment 来源树**：`/home/oseasy/experiments/rmt16_replay_phase4a/src/{rmt_collect.py, train_rmt16_p2replay.py}`

## 3. 为什么不直接改 mechanism_UED 现有 src
- 本仓库（gregjones11235/mechanism_UED）现有 `src/`/`dicode_src/` 等是**他人/历史版本**，其 `wrappers_cl.py` 基线（sha `cbbf5865…`，LF）与本轮运行所用副本（基线 `2ded41d8…`，CRLF/TAB）**不是同一版本**，补丁锚点无法匹配；且仓库工作树带有在途修改。
- 为避免污染/覆盖现有 RMT/D052/CC2/CC3 文件，本轮代码以**独立快照目录**固化，不混入现有 `src/`。

## 4. 来源树（运行时真实加载路径，import 溯源实测）
运行时 `sys.path` 顺序 = `[SRC, V7/src, V7]`，其中 `SRC=/home/oseasy/experiments/rmt16_replay_phase4a/src`，`V7=/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB`。
- `dicode.wrappers_cl` → `V7/src/dicode/wrappers_cl.py`
- `dicode.task_utils`、`minicraftax.*`、`craftax.*` → `V7/src/...`
- 15 个本地实验模块（含 collector、8 冻结、7 依赖）→ `SRC/<module>.py`
- **无同名遮蔽**：每个本地模块在 sys.path 中唯一解析（详见 `provenance/manifests/runtime_import_map.json`）。

## 5. 三个修改分别是什么（仅探针日志/聚合路径；reset/step/done/reward/obs/optimistic-reset 语义不变）
1. **wrappers_cl.py**（`runtime/wrapper_src/`，sha `b2f6b43a…`，before `2ded41d8…`）：新增构造参数 `probe_term`（默认 False）；`probe_term=True` 时在 `step` 的 info 中**增量**写入 5 个 `_term_*` JAX 数组（player_health/player_level/timestep/is_dead/done_steps，shape `[num_envs]`、固定 dtype、无字符串），取自 wrapper reset **前** 的终端 env_state；`probe_max_timesteps` 在 `__init__` 读取**全部** `tasks[i].params.max_timesteps`，要求 ≥1 且全一致，否则 `raise`（无硬编码 4096、无 silent fallback）。`probe_term=False` 时 info 与原版**逐位一致**。
2. **rmt_collect.py**（`runtime/experiment_src/`，sha `5be7e32b…`，before `0ef0a167…`）：只读终止仪表 R1–R7——每条完整 episode 记录 `episode_records`（length/terminated/truncated/无推断 done_reason/achievements/max_floor/final_health/term_* 等）。done_reason 规则：候选∈{time_limit,task_success,player_death}，**恰一个**→该原因，0 或 >1→`unknown`（保留 candidates+ambiguous）。
3. **train_rmt16_p2replay.py**（`runtime/experiment_src/`，sha `b8fd96a9…`，before `a21da58d…`）：D1–D10——`--probe/--early_stop_len(默认0)/--equiv_dump`；probe 下断言 `--replay off`；逐 episode/逐 update 探针输出（**仅记录，first≥512 不停止**）；probe summary + `exit(0)`；A/B 等价哈希 dump；**D10 将 `probe_term=PROBE` 接入 wrapper 构造**。`PROBE=False` 时逐位一致。

## 6. 八个冻结模块未修改（SHA 与冻结基线逐字吻合）
`runtime/frozen_modules/`：network_rmt16=`b5c37d7a`、rmt_replay_buffer=`21e31565`、rmt_memory_anchor=`92c56b63`、rmt16_memory=`17e1a614`、rmt_ppo=`2d2d943e`、rmt_hindsight=`2447a798`、replay_buffer=`c36c95b4`、rmt_replay_learner=`a575019f`。另含 7 个未改直接/传递本地依赖：rng_utils、full_p2_learner、hindsight、pending_episodes、awr、memory_anchor、vtrace（manifest 标 `unmodified_dependency`）。

## 7. 门禁状态
- **门禁4（wrapper schema + 四场景 pre-reset provenance，CPU）= PASS**：probe_term OFF info 73 键无 `_term_*`（bit-exact）；ON 恰 5 个 `_term_*`、shape `(16,)`、键集不变、无字符串；time_limit/player_death/task_success 均证明终端值来自 reset 前状态；`_term_done_steps` 阈值缺陷已修复。
- **门禁5（step0 SHA）= PASS**：A=B=冻结=`2f8cd875993ae10385dbb5dae530a557a0eb1008541b98de416cc7ae7ba2d93b`。
- **门禁6（A/B 训练无扰动）= PASS**：`PROBE_INSTRUMENTATION_TRAINING_EQUIVALENCE=PASS`，A(probe OFF) vs B(probe ON) 2 updates 全字段逐位相同（actions/rewards/dones/ard hash、params/ppo_opt/rmt_state/memories/mem_mask/mem_idx SHA、ppo_actor/entropy/value）。B 臂端到端复现冻结 episode 长度 `[85,128,155,255]`，`done_reason={player_death:4}`。
- 证据：`reports/gate4_wrapper_schema_v2.log`、`reports/gate6_equiv_compare.{log,json}`。

## 8. 探针只回答 L512 可达性，不回答 Carry 性能
本探针**不**输出：Persistent 更优 / Reset128 更优 / Carry 有效 / Carry 无效 / RMT 提升性能。本轮只回答：**固定 16384 预算内，完整 episode 能否达到 512 步，以及终止原因是什么。**

## 9. replay = off
Replay learner 与采样更新关闭；buffer 对完整 done episode 的收集仍运行（用于 episode 长度统计），但不进行 Replay 学习。

## 10. hindsight = off
hindsight relabel / AWR 更新关闭。

## 11. online PPO only
仅在线 PPO 原始配置更新；无 Replay、无 hindsight、无变长序列。

## 12. 正式运行命令（已批准；输出目录全新、不续跑任何 4096 ckpt）
```bash
cd /home/oseasy/experiments/rmt16_replay_phase4a
PY=/home/oseasy/miniconda3/envs/dicode310/bin/python
CKPT=/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500
mkdir -p logs reports/pids

# Persistent 臂
nohup "$PY" src/train_rmt16_p2replay.py --carry_mode persistent --replay off --probe \
  --early_stop_len 0 --ckpt17500 "$CKPT" --out runs/RMT16-PERSISTENT-PROBE-L512-16384 \
  --gpu_uuid GPU-8df11537-ab79-722d-606f-411966196c4c --total_updates 8 --seed 42 \
  > logs/RMT16-PERSISTENT-PROBE-L512-16384.log 2>&1 &
echo $! > reports/pids/RMT16-PERSISTENT-PROBE-L512-16384.pid

# Reset128 臂
nohup "$PY" src/train_rmt16_p2replay.py --carry_mode reset128 --replay off --probe \
  --early_stop_len 0 --ckpt17500 "$CKPT" --out runs/RMT16-RESET128-PROBE-L512-16384 \
  --gpu_uuid GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd --total_updates 8 --seed 42 \
  > logs/RMT16-RESET128-PROBE-L512-16384.log 2>&1 &
echo $! > reports/pids/RMT16-RESET128-PROBE-L512-16384.pid
```

## 13. GPU UUID
- Persistent：`GPU-8df11537-ab79-722d-606f-411966196c4c`（GPU2）
- Reset128：`GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd`（GPU3）
- GPU0/GPU1 严禁触碰。

## 14. 输出目录不进入 Git
`runs/`、`logs/`、checkpoint/OCDBT/JAX cache/权重/大张量/Replay 数据**均不提交**。本仓库仅存源码快照 + 小型证据。

## 15. 科学限制
- `probe="REACHABILITY_ONLY"`，`not_for_formal_science=True`：探针 checkpoint **不**作为正式科学结论。
- 单 seed（42）、单基座 step0、固定 16384 预算；不代表总体推断。
- 未授权：Phase4A-v2 训练、Replay、hindsight、512 变长 Replay、RMT hindsight 重算 memory 修复、多 seed、24576、98304、Base GTrXL matched Replay。
- 裁定标签仅限：`L512_REACHABILITY_BOTH / _PERSISTENT_ONLY / _RESET128_ONLY / _NOT_REACHED / PROBE_BLOCKED`。

## 字节一致性与 provenance
- `provenance/manifests/runtime_source_manifest.json`：每文件 snapshot_path/original_absolute_path/role/modified_or_frozen/size/sha256/line_ending/exec_mode/source_tree/before_probe_sha256/gate_verified/notes。
- `provenance/manifests/runtime_source_SHA256SUMS`：全部快照文件 SHA256。
- `provenance/manifests/runtime_import_map.json`：运行时 `inspect.getfile` 实测加载路径 + 遮蔽检查。
- `provenance/before_probe/`：三文件修改前原件。`provenance/diffs/`：三份 diff。
- 运行时代码身份字段：`source_snapshot_commit=<git commit>`、`runtime_source_manifest_sha`、`runtime_file_sha256=<各文件>`。**不存在** `runtime_git_head`（运行目录非 Git 仓库）。
