# Phase 3：Hook 集成方案（skill_scheduler + preflight 接入主循环）

> 基于 pod 上真实代码定位（run_dicode.py 主循环 + evolution_efficient.py）。
> 两处 hook 都在**主循环、单线程**，不碰后台 worker 线程（关键：避免 policy 并发问题）。
> baseline 停止后再 apply + 跑验证。

---

## 架构结论（重要，记入方案文档）

DiCode 的生成+编译（`evolve_and_validate_tasks` → `run_session_validation`）是 `executor.submit(...)`
提交到 **ThreadPoolExecutor 后台线程** 跑的。而 preflight 的 cold rollout 需要读 policy（`rl_train_state`），
policy 在主循环训练中不断更新。**在后台线程读边训边变的 policy 会造成并发/JAX 状态问题**，因此：

- preflight **必须放主循环**（policy 手边、单线程），**不能放 validation（worker 线程）**。
- 正确落点：主循环 Step 3，`new_task_ids` 拿到后、进训练前，用当前 `rl_train_state` 过滤。

（这解释了 preflight.py 当初把 cold_preflight 与纯函数分开、标注"集成时定位置"的设计——集成时果然定在主循环。）

---

## ★A：Skill Graph Scheduler（注入生成 target）

### A-1. run_dicode.py —— Step 2 `dispatch_evolution_worker` 调用**之前**插入

在这一段之前：
```python
        # --- Step 2: Dispatch new evolution worker if needed ---
        if evolve_future is None:
            worker_start_time = time.time()
            evolve_future = dispatch_evolution_worker(
                executor, evolve_future, gen_manager, config, evaluation_metrics
            )
```

插入：
```python
        # ★A: Skill Graph Scheduler — 用上一轮评估的 per-achievement SR 定位学习前沿，
        #     存到 gen_manager 供生成 prompt 读取（不改 dispatch 链签名，改动最小）。
        try:
            from dicode.skill_preflight.skill_scheduler import (
                pick_target, format_target_for_prompt,
            )
            if evaluation_metrics:                      # 第一轮可能还没 eval，跳过
                _sched = pick_target(evaluation_metrics)
                gen_manager.current_skill_target = format_target_for_prompt(_sched)
                print(f"  [SkillGraph] frontier tier {_sched.tier}, "
                      f"targets: {_sched.target_achievements}")
            else:
                gen_manager.current_skill_target = None
        except Exception as e:
            print(f"  [SkillGraph] scheduler skipped: {e}")
            gen_manager.current_skill_target = None
```

### A-2. gen_manager.py —— 生成 user_prompt 处，把 target 拼进去

在 `EnvGenerator.generate`（或 `evolve_tasks` 构造 user_prompt 的地方），找到
`user_prompt.format(CODE_EXAMPLES=..., TASK_DESCRIPTION=...)` 那处，改成把 skill target 附加进
TASK_DESCRIPTION（或 prompt 末尾）：

```python
        # 读取 skill graph 目标（可能为 None）
        _skill_hint = getattr(self, "current_skill_target", None)
        _task_desc = state["task_info"]["description"]
        if _skill_hint:
            _task_desc = _task_desc + "\n\n[Curriculum focus]\n" + _skill_hint
        # 用 _task_desc 替换原来的 TASK_DESCRIPTION 入参
        ... user_prompt.format(CODE_EXAMPLES=example_str, TASK_DESCRIPTION=_task_desc) ...
```

> ⚠️ apply 时对着 gen_manager.py 里真实的 user_prompt.format 调用微调（占位符名以实际为准）。
> `EnvGenerator.__init__` 里加一行 `self.current_skill_target = None` 作为默认，防止属性不存在。

---

## ★B：Preflight Gate（编译通过后、进训练前过滤）

### B-1. run_dicode.py —— Step 3，`sampled_task_ids = new_task_ids + ...` **之前**插入

原代码：
```python
        # --- Step 3: Sample tasks for training ---
        ...
        sampled_from_archive = sample_tasks_for_training(
            gen_manager, config, num_to_sample_from_archive
        )
        sampled_task_ids = new_task_ids + sampled_from_archive
```

在 `sampled_task_ids = new_task_ids + sampled_from_archive` 之前插入 preflight 过滤：
```python
        # ★B: Preflight Gate — 用当前 policy 对新任务 cold rollout，只保留"可学习"的。
        #     policy(rl_train_state) 在此作用域现成、单线程，无并发问题。
        try:
            from dicode.skill_preflight.preflight import cold_preflight
            if new_task_ids:
                _kept = []
                for _tid in new_task_ids:
                    _env, _env_params = gen_manager.archive.get_task_env(_tid)   # ← 见注1
                    _target_ach = getattr(_sched, "target_achievements", []) \
                        if 'evaluation_metrics' in dir() and evaluation_metrics else []
                    _res = cold_preflight(
                        _env, _env_params, rl_train_state, rng, config, _target_ach,
                    )
                    if _res.action == "accept":
                        _kept.append(_tid)
                    else:
                        gen_manager.archive.update_node_status(_tid, f"preflight_{_res.reason}")
                        print(f"  [Preflight] reject {_tid}: {_res.reason} (sr={_res.sr:.2f})")
                print(f"  [Preflight] kept {len(_kept)}/{len(new_task_ids)} new tasks")
                new_task_ids = _kept
        except Exception as e:
            print(f"  [Preflight] skipped (kept all): {e}")   # 出错则不过滤，保守放行
```

> **注1**：`gen_manager.archive.get_task_env(_tid)` 是**占位**——apply 时要确认 archive 怎么从
> task_id 拿到可跑的 env + env_params（可能是 `archive.get_task_code` 后 `Task(...)`，或已有方法）。
> 这是 ★B 落地时唯一需要对着真实 archive API 敲定的点。
>
> **注2**：`cold_preflight` 内部 `make_evaluate(config, env, env_params)(rl_train_state, rng)` 的调用，
> 需对照 `run_session_training` 里 make_evaluate 的真实用法微调（train_state/rng 传法）。
>
> **注3**：MVP 阶段若 cold_preflight 接起来复杂，可先只跑 ★A（skill graph）+ preflight 的
> 纯静态过滤（去重/编译，已有），cold rollout 版作为 Phase 3b。preflight.py 的 route/partial_progress
> 逻辑已单测通过，只等 env+policy 接上。

---

## Apply 顺序（baseline 停止后）

1. `EnvGenerator.__init__` 加 `self.current_skill_target = None`。
2. 应用 A-1（run_dicode Step 2 前）+ A-2（gen_manager user_prompt）。
3. 先只测 ★A：短 run 看日志有没有 `[SkillGraph] frontier tier ...` + 生成是否受 target 影响、能否编译。
4. ★A 通过后，应用 B-1，敲定注1（get_task_env）+ 注2（make_evaluate 调用）。
5. 短 run 看 `[Preflight] kept X/Y`，确认不崩、过滤合理。
6. 都通过后，跑三组消融（纯baseline / +A / +A+B）。

## 验证要点
- ★A：日志出现 frontier tier 打印；生成的关卡围绕 target tier。
- ★B：日志出现 kept X/Y；被 reject 的任务状态标 `preflight_*`。
- 整条 run 不崩、能落 WandB、能出 checkpoint。
