# Phase 3：Hook 集成方案（skill_scheduler + preflight 接入主循环）

> 基于 pod 上真实代码逐行核对（run_dicode.py 主循环 + gen_manager.py + craftax_evaluation.py）。
> 两处 hook 都在**主循环、单线程**，不碰后台 worker 线程（关键：避免 policy 并发问题）。
> **本版已把原占位符替换为核对过的真实接口**；仅剩 1 处需在 pod 上确认（见文末）。
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

## 接口核对结论（已确认无误，直接能接）

| 调用 | 真实签名/返回 | 状态 |
| --- | --- | --- |
| `skill_scheduler.pick_target(evaluation_metrics)` | 返回 `SchedulerTarget(tier, target_achievements, tier_mastery, frontier_mastery, gap_type)` | ✅ 与方案一致 |
| `skill_scheduler.format_target_for_prompt(target)` | 返回 str | ✅ |
| `preflight.cold_preflight(env, env_params, train_state, rng, config, target_achievements)` | 返回 `PreflightResult(action, reason, sr, any_partial_progress, n_episodes, extra)` | ✅ 位置参数与返回字段一致 |
| `archive.update_node_status(task_path, status)` | 存在（gen_manager.py:240） | ✅ |

**核对中发现并已修正的 4 处**（旧版方案里是错的/缺的）：

1. **对象错配**：★A 的 target 必须存到 **`gen_manager.task_generator`**（GenManager 持有 `self.task_generator`=TaskGenerator、`self.env_generator`=EnvGenerator，是两个子对象）。描述生成在 `TaskGenerator.evolve_mastered`。旧版设在 `gen_manager` 上、又在 `EnvGenerator.self` 上读 → 永远读不到。
2. **注入点**：`evolve_mastered` 的 prompt **没有 `TASK_DESCRIPTION` 占位符**，用的是 `MASTERED_TASK / TASK_PERFORMANCE_CONTEXT / GLOBAL_AGENT_PROFILE`。最小改动是把 skill hint **追加到 `global_profile_str`**（两个分支都经过它，不改 prompt 模板）。
3. **取 env**：archive **没有** `get_task_env`。真实路径是 `Task(smart_absolute_path(tid)).env` + `BatchEnvWrapper` + `default_params.replace(...)`（照 craftax_evaluation.py:273 的 make_evaluate 调用点）。
4. **make_evaluate 要 jit**：真实调用点是 `jax.jit(make_evaluate(...))`。cold_preflight 里未 jit，每候选跑未编译 eval 会极慢——加 jit。

---

## ★0：Config flag（消融必需 —— 旧版遗漏）

三组消融（纯 baseline / +A / +A+B）**靠 config 开关切**，不靠 try/except。
新建 `conf/skill_preflight/default.yaml`（或加进主 config，确保 `skill_preflight` 节点存在）：

```yaml
skill_preflight:
  use_scheduler: false   # ★A  baseline=false, +A=true, +A+B=true
  use_preflight: false   # ★B  baseline=false, +A=false, +A+B=true
```

> flag 默认 false ⇒ 不翻开关时行为与原版 DiCode 逐字节一致（= 干净 baseline）。
> 所以这份 hook 可以**现在就 push**，待验证的代码在翻开关前处于休眠。

---

## ★A：Skill Graph Scheduler（注入生成 target）

### A-0. `TaskGenerator.__init__`（gen_manager.py，L626 那个 __init__）末尾加默认

```python
        self.current_skill_target = None
```

### A-1. run_dicode.py —— Step 2 `dispatch_evolution_worker` **之前**插入

在这一段之前：
```python
        # --- Step 2: Dispatch new evolution worker if needed ---
        if evolve_future is None:
            worker_start_time = time.time()
            evolve_future = dispatch_evolution_worker(
                executor, evolve_future, gen_manager, config, evaluation_metrics
            )
```

插入（注意 `_sched` 提到循环作用域，★B 要复用；存到 **task_generator**）：
```python
        # ★A: Skill Graph Scheduler — 用上一轮评估的 per-achievement SR 定位学习前沿，
        #     存到 task_generator 供 design prompt 读取。
        _sched = None
        gen_manager.task_generator.current_skill_target = None
        if config.skill_preflight.get("use_scheduler", False) and evaluation_metrics:
            try:
                from dicode.skill_preflight.skill_scheduler import (
                    pick_target, format_target_for_prompt,
                )
                _sched = pick_target(evaluation_metrics)
                gen_manager.task_generator.current_skill_target = format_target_for_prompt(_sched)
                print(f"  [SkillGraph] frontier tier {_sched.tier}, "
                      f"targets: {_sched.target_achievements}")
            except Exception as e:
                print(f"  [SkillGraph] scheduler skipped: {e}")
```

### A-2. gen_manager.py —— `TaskGenerator.evolve_mastered` 里，`global_profile_str` 构造**之后**

找到（L786 附近）：
```python
        global_profile_str = self._format_global_agent_profile(global_agent_profile)
```
紧跟其后插入（追加 hint，两个 `.format()` 分支都经过 `GLOBAL_AGENT_PROFILE`，无需改模板）：
```python
        # ★A-2: 把 skill graph 目标追加进 profile（可能为 None）
        _skill_hint = getattr(self, "current_skill_target", None)
        if _skill_hint:
            global_profile_str = global_profile_str + "\n\n[Curriculum focus]\n" + _skill_hint
```

> 若想让 target 成为独立字段而非塞进 profile：在 `evolve_mastered_prompt` 模板加 `{CURRICULUM_FOCUS}`
> 占位符并在两处 `.format(...)` 传入——更显式但要动模板文件。MVP 用上面的追加法即可。

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

在 `sampled_task_ids = new_task_ids + sampled_from_archive` 之前插入（`jax` 已 import）：
```python
        # ★B: Preflight Gate — 当前 policy 对新任务 cold rollout，只保留"可学习"的。
        #     policy(rl_train_state) 在此作用域现成、单线程，无并发问题。
        if config.skill_preflight.get("use_preflight", False) and new_task_ids:
            try:
                from dicode.dreaming.gen_manager import Task, smart_absolute_path
                from dicode.wrappers import BatchEnvWrapper
                from dicode.skill_preflight.preflight import cold_preflight
                _target_ach = _sched.target_achievements if _sched is not None else []
                _kept = []
                for _tid in new_task_ids:
                    rng, _pf_rng = jax.random.split(rng)          # 别复用训练的 rng
                    _raw = Task(smart_absolute_path(_tid)).env    # ← Task 加载器（archive 无 get_task_env）
                    _eparams = _raw.default_params.replace(       # ← ⚠ pod 上确认 default_params（见文末）
                        max_timesteps=config.evaluation.get("max_timesteps", 8192))
                    _env = BatchEnvWrapper(_raw, num_envs=config.evaluation.num_envs)
                    _res = cold_preflight(_env, _eparams, rl_train_state, _pf_rng, config, _target_ach)
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

### B-2. preflight.py —— `cold_preflight` 里给 make_evaluate 加 jit

把：
```python
    evaluate = make_evaluate(config, env, env_params)
    metrics = evaluate(train_state, rng)
```
改成：
```python
    import jax
    evaluate = jax.jit(make_evaluate(config, env, env_params))
    metrics = evaluate(train_state, rng)
```

> 效率备注（呼应旧版注3）：B-1 现在每候选跑**完整** `config.evaluation` 规模 eval。先接起来验证；
> 若 12 候选太慢，给 preflight 单独配小 `num_envs`/`num_steps`，或接 `staged_preflight` 的 L2/L3 漏斗（Phase 3b）。

---

## ⚠ 唯一需在 pod 上确认的一点

`MiniCraftaxTrain.default_params` 是否存在——canonical eval 路径（craftax_evaluation.py:267）用的是
`CraftaxAugObsTrain().default_params`，而 Task 的 env 是 `MiniCraftaxTrain(task=...)`。gymnax 风格 env 一般都有，
但翻 ★B 开关前，先跑：
```python
python -c "from dicode.dreaming.gen_manager import Task, smart_absolute_path; \
t=Task(smart_absolute_path('<某个已有task路径>')); print(hasattr(t.env,'default_params'))"
```
若为 False，就照 `run_session_training` 里给 sampled task 造 env_params 的方式复用（B-1 只这一处 env_params 取法要换）。

---

## Apply 顺序（baseline 停止后）

1. 加 ★0 config + A-0（TaskGenerator.__init__ 默认）。
2. 应用 A-1 + A-2。
3. **只开 `use_scheduler`** 短 run：看日志有无 `[SkillGraph] frontier tier ...`，生成是否受 target 影响、能否编译。
4. ★A 通过后，应用 B-1 + B-2，并按文末确认 `default_params`。
5. **再开 `use_preflight`** 短 run：看 `[Preflight] kept X/Y`，被 reject 的任务状态标 `preflight_*`，确认不崩。
6. 都通过 → 跑三组消融（纯 baseline=都关 / +A=开 scheduler / +A+B=都开）。

## 验证要点
- ★A：日志出现 frontier tier 打印；生成的关卡围绕 target tier。
- ★B：日志出现 `kept X/Y`；被 reject 的任务状态标 `preflight_*`。
- 整条 run 不崩、能落 WandB、能出 checkpoint。
