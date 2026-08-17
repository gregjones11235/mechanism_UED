# E3 Mechanism Completion Report

Date: 2026-08-18 · Branch: `Henry-branch` · Commit: `ddc5996`

把 E3 从「可运行 vertical slice」补全为真正闭环的 `Probe → Frontier → Data → PPO →
Reprobe → New Frontier`，并定位了真实 SlowGRU 的上线阻塞点。

---

## 1. 三个首要问题：全部修复

| 问题 | 结论 | 修复 |
|---|---|---|
| ① Adaptive Frontier 未闭环 | ✅ 已闭环 | `e3_loop.run()` 每轮 `probe_k → locate_frontier → capsule/bank → data → PPO → reprobe → frontier_after`，下一轮 `frontier_used = 上一轮 frontier_after`；新增 `frontier_after.json` + `reprobe_it{it}.json` |
| ② `slice_student` 硬编码 | ✅ 已移除 | 新增 `StudentIdentity`（student_id/version/architecture_family/params_hash/checkpoint_step），后端暴露真实 `student_id`（SlowGRU → `SLOWGRU_PERSISTENT_CANONICAL_98304`）；5 类 fail-closed |
| ③ `G9_VERTICAL_SLICE=True` 无条件 | ✅ 已证据化 | ≥2 PPO update、≥2 reprobe、student version 推进、`frontier_after` 存在、无 stale data、无 binding mismatch、G1–G7 全过 |

## 2. 核心机制补全

- **A. State Bank**：`BankEntry` 增加 `student_version/params_hash/tier/architecture_family`；build/validate 返回 simulator transition 计数；recurrent state 可恢复（不清零）。
- **B/C. Strict on-policy + PPO Bridge**：G5 逐 batch 校验 `policy_hash == 当前 params`（fail-closed）；G6 校验 `update_count>0 / loss finite / grad finite / params finite / params changed`。
- **D. Accounting**：新增 `state_bank_build / state_bank_validation` 类别 + `conservation_ok`；`sum(category) == total_simulator_transitions`。
- **键别名**：`recurrent_state.canonicalize_memory` 归一 `mem_mask/mem_idx ↔ memories_mask/memories_mask_idx`，并校验 fast-window 键。
- **NoCausal / CF**：`E3LoopConfig.mode` 切分，共用同一套机制；NoCausal 下 `llm_calls==0`、`evidence=None`。

## 3. 修改文件

- `src/dicode/e3_litesim/`：`orchestration/e3_loop.py`、`runtime/student_binding.py`、`runtime/recurrent_state.py`、`runtime/slice_student.py`、`data/state_bank.py`、`data/data_engine.py`、`diagnostics/accounting.py`、`learning/ppo_bridge.py`
- `src/dicode/training_backend_slowgru.py`（student_id / checkpoint_step）
- `tests/e3_litesim/`：`test_e3_loop_vertical_slice.py`（新增 adaptive frontier A→B、闭环链）、`test_student_binding_guard.py`（5 类）、`test_accounting.py`（守恒）、`test_recurrent_alignment.py`（键别名）
- `tools/`：`run_e3_litesim_slice.py` / `run_e3_litesim_slice_slowgru.py`（`--mode`）、新增 `benchmark_e3_litesim_grid_slowgru.py`（真实 SlowGRU GPU throughput）

## 4. 测试与 artifact

- `tests/e3_litesim`：**28 passed**（CPU，`sim_frontier_venv`）。
- 本地闭环 artifact：`gpu1_aggregation_siege/artifacts/e3_litesim/e3_closed_loop_ddc5996/`，`STATUS=PASS`，G1–G9 全 true。
  闭环证据（summary）：
  ```
  it0: version=0  frontier_used=1ae1ccbd74 → frontier_after=2774eeb074
  it1: version=1  frontier_used=2774eeb074 → frontier_after=434c21835a   # 第二轮用的是新 frontier
  accounting: conservation_ok=True  total=712 = probe624+build12+valid4+train72  llm_calls=0
  ```

## 5. 真实 SlowGRU：上线阻塞（未修复）

服务器（`oseasy@172.25.14.221`，GPU0 空闲卡）跑 `run_e3_litesim_slice_slowgru.py`，第一步 probe 即崩：

```
flax.errors.ScopeParamShapeError: encoder kernel expected (8335,256) got (8268,256)
```

根因（已实测确认）：

| | obs_dim |
|---|---|
| e3_litesim tier env（`MiniCraftaxTrain`，base） | **8268** |
| canonical SlowGRU 98304（`MultiTaskMiniCraftaxEnv` + `condition_on_task=True, embedding_size=67`） | **8335** = 8268 + 67（achievement multi-hot embedding） |

`run_e3_litesim_slice_slowgru.py` 是从未在真实后端跑过的工具（写它时服务器断连），env 与真实
SlowGRU 的观测空间未对齐。修法需让 litesim 数据面产出 8335 obs（`tier_registry` /
`lightweight_rollout` 接入 task-conditioned env，每 step 传 67 维 task_embedding），属动核心数据面。

## 6. 服务器状态（未动任何进程）

4× NVIDIA RTX A6000：GPU0/3 idle（ollama 常驻 0% util）；GPU1 100%（Mason `seed=2` 训练）；
GPU2 0%（`seed=3` 驻留）。GPU0/3 全程未被影响。

## 7. 未完成 / 下一步

- [ ] real SlowGRU obs 对齐（8268 → 8335，接入 task-conditioned env）
- [ ] real SlowGRU vertical slice PASS（G1–G9）
- [ ] real SlowGRU GPU throughput artifact
- [ ] 设计正式 scientific experiment（Mason runtime 同配置对比）
