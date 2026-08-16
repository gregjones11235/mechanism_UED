# MEMORY_STUDY_CODE_MAP (G0 / Step 1)

- 项目: gregjones11235/mechanism_UED — Long-Horizon Navigation Memory Study
- 阶段: Step 1 只读代码审查（G0 Code Map）
- 日期: 2026-08-16 (Asia/Shanghai)
- 工作树: mechanism_UED_Henry_worktree @ Henry-branch
- HEAD: 0dd9de5bf9cb4a6c540cd915b9f90c07bfa7ead2（与 origin/Henry-branch 完全同步，git ls-remote 经代理核实）
- 方法: 全部结论来自对当前工作树源码的只读检索（rg/直接读文件），逐条给出 file:line。未依赖分支名或旧设计文档推断。

## 0. 审查范围排除声明

工作树中存在 5 个未提交的他人改动（先前 E3 阶段评估器修复），本审查未触碰、未纳入结论主干：

- M gpu1_aggregation_siege/src/dicode/student_adapters/slowgru_adapter.py
- M gpu1_aggregation_siege/tests/simulator_frontier/test_slowgru_adapter.py
- ?? gpu1_aggregation_siege/scripts/evaluate_e3_slowgru_original_task.py
- ?? gpu1_aggregation_siege/src/dicode/simulator_frontier/e3_slowgru_original_eval.py
- ?? gpu1_aggregation_siege/tests/simulator_frontier/test_e3_slowgru_original_eval.py

创建说明: 本文档由总控（director）直接落盘。原因: 本轮会话中委派通道（spawn_agent/followup_task 消息投递）连续 3 次故障（子代理收不到任务消息），G0 为协议 Step 1 且仅新增本文档一个文件、不触碰任何实验逻辑，故按受控偏离执行并在此披露。

## 1. G0 PASS 判据对照（结论先行）

| G0 判据 | 结论 | 依据 |
| --- | --- | --- |
| 三种 Student 真实代码路径确认 | CONFIRMED | 见 3.1 GTrXL / 3.2 RMT16 / 3.3 SlowGRU（全部 file:line） |
| HO 路径确认 | NOT_FOUND | 现役源码树中不存在 Historical Observation reinjection 实现，见 3.4 与搜索证据 |
| probe 路径确认 | NOT_FOUND（无 Floor2→Floor3 probe；存在其它 probe 基建） | 见 3.5 与搜索证据 |
| checkpoint provenance 初步确认 | CONFIRMED（本地 manifest 级；checkpoint 实体仅在服务器） | 见 4 |

G0 总体判定: CONDITIONAL —— 三种 Student 与 checkpoint provenance 成立；但协议预设的 HO reinjection 框架与 Floor2→Floor3 probe 在当前仓库中并不存在，需要 Step 3/4 新建（协议允许在后续步骤写 probe 代码，但 G0 字面判据"H0路径确认"目前无法满足）。这是需要用户决策的缺口，见 6。

## 2. 组件总表（协议要求格式）

| Component | File | Function/Class | Runtime State | Checkpoint State | Planned Modification |
| --- | --- | --- | --- | --- | --- |
| GTrXL 主 Student 网络 | gpu1_aggregation_siege/src/dicode/network.py | ActorCriticTransformer (:115), model_forward_eval (:177), model_forward_train (:196) | window memories/memories_mask/memories_mask_idx（ppo_tr.py 内滚动） | params only（orbax 目录，CC1 协议） | 无（G0 禁改） |
| GTrXL PPO 训练回路 | gpu1_aggregation_siege/src/dicode/ppo_tr.py | _env_step_original (:312 起), reconstruction (:1034-1062) | memories 滚动、done 清零、session 计数 | 不入 checkpoint | 无 |
| RMT16 记忆状态机 | gpu1_aggregation_siege/src/dicode/student_adapters/architectures/rmt16_memory.py | rmt16_step (:83-105), rmt16_store_h (:64), rmt16_update_tokens (:71-81), rmt16_reset_envs (:57) | mem_tokens (16,D) / seg_buf (128,D) / seg_count | 不入 params；经 anchor burn-in replay 重建 | 无 |
| RMT16 carry 语义（persistent vs reset128） | gpu1_aggregation_siege/src/dicode/student_adapters/architectures/rmt16_anchor.py | rmt_advance_tokens (:66-90) | carry_mode 分支 (:80-85) | checkpoint 带 carry_mode manifest（adapter 校验 :229-231） | 无 |
| RMT16 训练后端 | gpu1_aggregation_siege/src/dicode/training_backend_rmt16.py | StudentTrainingBackend(RMT16)（carry_mode :72/:90/:106, rmt_advance_tokens 调用 :275） | rmt.mem_tokens/seg_buf/seg_count (:11-13) | params（cc2_params_sha256 规约） | 无 |
| RMT16 候选适配器 | gpu1_aggregation_siege/src/dicode/student_adapters/rmt16_adapter.py | memory_mode→carry_mode 映射 (:53-54), manifest 校验 (:229-231) | — | profile+manifest 绑定 | 无 |
| SlowGRU 训练后端 | gpu1_aggregation_siege/src/dicode/training_backend_slowgru.py | policy_forward_eval (:262-322), reset_runner_memory (:215-254), _ensure_loaded (:133-167) | longstate.h/buf/count + GTrXL fast memory + true_done (:11-14) | full_state.pkl（params+runtime memory） | 无 |
| SlowGRU 外部运行时 | student_pool_v1/cc3/slowgru_runtime/slowgru_runtime.py | load_candidate / forward_eval / init_longstate（经 training_backend_slowgru.py:146-167 装载） | SLOW_INTERVAL=32, SLOW_DIM=256 (:123, :296) | 与 full_state.pkl 绑定 | 无 |
| SlowGRU 候选适配器 | gpu1_aggregation_siege/src/dicode/student_adapters/slowgru_adapter.py | checkpoint 状态规格 (:415-420, :500-505) | memories(128,2,256)/mask/idx + longstate | full_state.pkl 规格 | 有未提交修复（见 0，排除） |
| probe 基建（非 HO） | gpu1_aggregation_siege/src/dicode/shared_runtime/probe_runner.py | RealProbeRunner (:39), run_probes (:58), _run_candidate_rollout (:108), _run_episode (:162) | — | — | Step 3/4 需另建 HO probe（见 6） |
| Student profiles（provenance 绑定） | gpu1_aggregation_siege/conf/student_profiles/*.yaml | 6 个 profile（见 4） | — | source_commit/params_sha/ckpt sha | 无 |
| HO reinjection | NOT_FOUND | — | — | — | Step 3/4 新建（协议预期产物） |
| Floor2→Floor3 probe | NOT_FOUND | — | — | — | Step 3/4 新建（协议预期产物） |

## 3. 五条真实调用链

### 3.1 GTrXL 主 Student（window_mem=128）

1. 网络定义: `ActorCriticTransformer` gpu1_aggregation_siege/src/dicode/network.py:115；setup():126 组装 Transformer + actor/critic 头；前向入口 `__call__(self, memories, obs, mask)`:159；评估前向 model_forward_eval:177；训练前向 model_forward_train:196。
2. 默认窗口: `window_mem: 128` gpu1_aggregation_siege/conf/training/default.yaml:26（源码消费点为 config.window_mem / config.training.window_mem，见下）。
3. rollout 初始化: memories/mask/idx 零初始化 gpu1_aggregation_siege/src/dicode/ppo_tr.py:192-193（backend 路径 init_runner_memory:221）与 :224-229；restore 时同样零初始化 gpu1_aggregation_siege/src/dicode/utils/general/train_state_utils.py:51-52/:176-177。
4. rollout 每步更新（_env_step_original，ppo_tr.py:312 起）: done 时 memories_mask_idx 复位为 window_mem、mask 清零 ppo_tr.py:319-324；one_hot 并入 mask :331；memory 滚动写入由 memory_indices 选择 :363（reconstruction 对应 :444/:450）。
5. PPO reconstruction 使用 memory: ppo_tr.py:1034-1062（config.training.window_mem 系列）。
6. episode reset 行为: 由 done 驱动（上面第 4 条 :319-324）——episode 真 done 即清空该 env 的窗口记忆。
7. session 边界行为: `max_updates_per_session` ppo_tr.py:153-156；初始 runner state 注释明确"可来自上一 session 的 RunState" ppo_tr.py:214；session 内步计数 step_env_currentloop :235/:312/:387。结论: session 边界本身不清 memory，memory 随 RunState 移交（Step 2 checkpoint audit 需实证移交内容）。

结论: main Student = GTrXL/Transformer-XL 风格 Actor-Critic，默认 window_mem=128 —— 与协议 1.1 预期一致，CONFIRMED。

### 3.2 RMT16 canonical

1. 状态机: rmt16_memory.py:11-13（mem_tokens (16,D) / seg_buf (128,D) / seg_count），:16-18（段内 actor 读 mem_tokens；每 128 步段末以 cross-attention(query=mem_tokens, kv=seg_buf) 更新）；初始化 :48-49；done 复位 :57；存 h_t :64；段末更新 :71-81；单步推进 rmt16_step :83-105（边界条件选择：仅边界 env 取更新结果）。
2. carry 语义（persistent vs reset128 的唯一区别）: rmt16_anchor.py:13-16 与 :66-90 —— 128 步段边界处: persistent → mem_tokens ← 残差 cross-attention 更新值并跨段携带；reset128 → mem_tokens ← 0（边界清零，单窗读写）。seg_buf/seg_count 两种模式都清零。这是 bakeoff 单变量推导（config diff 只有 carry/reset）。
3. 训练后端接线: training_backend_rmt16.py:11-13（rmt.* 三个 runtime key）、:69-72（rmt_num_tokens=16、num_steps=128=每次 update 的 rollout 步、carry_mode "persistent"|"reset128"）、:90/:106（carry_mode 传入）、:136（segment_len=num_steps）、:275（每步调用 rmt_advance_tokens，carry_mode 生效）、:355（注释: 若无 persistent RMT state，PPO loss 重算会失真）、:417-423（BLOCKER-5: RunState 必须携带真实 memory 值）。
4. 是否过 Transformer: 是 —— 读路径经 network.model_forward_eval 带 mem_tokens（rmt16_anchor.py:32-37 make_apply_eval_rmt 包装 network.apply(..., mem_tokens=..., method=model_forward_eval)）；写路径经 network.update_rmt_tokens（rmt16_anchor.py:55-60 make_update_fn）。
5. RMT state 是否真实进入 rollout 与 PPO training: 是 —— Transition 数据类携带 rmt_entering_tokens（network.py:106-108，注释 BUG-E3-01: RMT16 entering tokens per step for training window）；reconstruction 使用相同 transition（rmt16_anchor.py:9-12 注释: collect 与 rebuild 跑同一 transition，bit-exact by construction）。
6. checkpoint 中 params vs runtime memory: checkpoint 只含 params（params_sha256 规约 cc2_params_sha256 = jax tree_leaves 顺序叶拼接，见 4）；mem_tokens/seg_buf/seg_count 是 runtime memory，不入 checkpoint，恢复时经 anchor bit-exact burn-in replay（≤128 步）重建（rmt16_anchor.py:129；anchor 与段边界对齐说明 :17-21）。真实 CC2 checkpoint manifest["config"]=={} 的判别见 rmt16_provenance.py:145-147。

结论: RMT16 canonical = LC-RMT16 16-token/128-段，persistent 与 reset128 唯一区别为 carry_mode —— CONFIRMED。

### 3.3 SlowGRU

1. 状态规格: training_backend_slowgru.py:11-14 —— longstate.h (256,) 慢 GRU 隐态；longstate.buf (32,256) 当前周期 GTrXL hiddens；longstate.count 周期内步数；true_done 慢态复位信号。fast memory（GTrXL 窗口）仍然存在: memories/mem_mask/mem_idx 与 longstate 并存（policy_forward_eval :272-286 同时推进 GTrXL mask 与 slow 前向；reset 注释 :221 "GTrXL: mem_idx -> window_mem, mem_mask -> zeros"）。
2. 32-step aggregation: slowgru_runtime.py:123（assert SLOW_INTERVAL==32 and SLOW_DIM==256）、:296（slow_interval=32）；buf 形状 (32,256) 亦见 slowgru_adapter.py:419/:504。即每 32 步把 GTrXL hidden 聚合进慢 GRU。
3. SlowGRU 输入: 当前周期的 GTrXL hiddens（buf），经 slowgru_runtime 的 forward_eval 更新 h（training_backend_slowgru.py:303-304 调用 self._forward_eval(params, memories, obs, mask, longstate, reset) → (pi, value, mem_out, ls_new)）。
4. h 参与 policy/value: forward_eval 直接返回 pi 与 value（:303-304），h 经 slowgru_runtime 网络内部参与 actor/critic 头（外部运行时，源码在 student_pool_v1/cc3/slowgru_runtime/slowgru_runtime.py）。
5. 运行时装载: _ensure_loaded :133-167 —— 校验 slowgru_runtime_path → sys.path 注入 :143-144 → `import slowgru_runtime as sr` :146 → 读 checkpoint_contract.json :149-153 → sr.load_candidate(contract) :156 → sr.seed_policy_rng(handle, 42) :161 → 取 network/params/forward_eval/init_longstate :163-167。
6. done/reset 语义（:16-20）: true episode done → 该 env 清 fast memory + longstate（reset_runner_memory :215-254: GTrXL 部分 :229-231，longstate 清零 :242-250，true_done 作为下一步复位信号 :253-254）；128-step segment 边界 → Persistent 契约保留 longstate（:19）；true reset → longstate 按契约 zero-init（:20）。
7. Persistent vs Reset128 版本: 本地现役后端代码承载 Persistent 契约（上述 :19）；RESET128 变体以独立 canonical 候选存在（student_pool_v1/cc3/SLOWGRU_RESET128_CANONICAL_98304/，READY.json/candidate_manifest.json），其段边界清零语义在训练驱动源码（source_commit 绑定，实体在服务器）。本地代码树未包含 reset128 SlowGRU 的分支实现 —— 如实记录，见 6 决策项。
8. checkpoint 状态规格: slowgru_adapter.py:415-420/:500-505 —— memories (128,2,256)、memories_mask、memories_mask_idx、longstate.h (256,)、longstate.buf (32,256)、longstate.count —— 即 canonical full_state.pkl 同时包含 params 与 runtime memory（READY.json determinism_gate: fresh reload + identical seeds → first 32 actions + step-32 memory hash bit-exact PASS）。

结论: SlowGRU = GTrXL fast window + 256 维慢 GRU（每 32 步聚合，跨段 persistent）—— CONFIRMED。

### 3.4 Historical Observation reinjection —— NOT_FOUND

搜索证据（全部在 mechanism_UED_Henry_worktree 执行）:

- rg -i "historical|hist_obs|ho_zero|ho_real|reinjection|reinject" gpu1_aggregation_siege/src → 0 命中。
- rg -i "replay|history|ho_view|obs_history" gpu1_aggregation_siege/src → 命中均为无关语义: ppo_tr.py:418（PPO reconstruction 重放 reset flags 的注释）、dreaming/gen_manager.py 与 logging_utils.py（日志/生成历史）、rmt16_anchor.py（anchor burn-in replay，见下）等；无"把历史 observation 注入 Student 影响其决策"的路径。

最接近的既有管道（供 Step 3/4 复用参考，均不是 HO reinjection）:

- rmt16_anchor.py:129 bit-exact burn-in replay —— 把 anchor 之后 ≤128 步 obs 重放以重建 RMT runtime state（服务于 checkpoint 恢复，不改变策略评估语义）。
- shared_runtime/probe_runner.py:39 RealProbeRunner —— 候选环境 rollout 探针（E3 模拟器方向），与 HO 无关。
- simulator_frontier/e3_window.py 与 probes/student_compatibility.py —— 窗口化执行与兼容性探针，无历史 obs 注入接口。

协议 1.4 的确认项（历史 observation 来源 / 注入接口 / 是否改变 env state、RNG、task embedding、params、timestep）: 无对象可确认。结论: HO reinjection 在当前仓库不存在，属 Step 3/4 需要新建的组件。

### 3.5 Floor2→Floor3 probe —— NOT_FOUND

搜索证据:

- rg -i "floor2|floor3|floor_2|floor_3" 全树（排除 .git）→ 现役源码命中仅: gpu1_aggregation_siege/d052/bagr_ued/{constants.py, behavior_taxonomy.py, report_writer.py} 与 d052/ba_cwm_ued/vocabularies.py（行为分类词表，属 UED 行为分析模块，与导航楼层 probe 无关）；dicode_v6/auction/modeler.py:618（prompt 示例文本 "reach floor2"）；其余命中全部在 experiments/**/raw_sources 旧实验存档。
- rg -i "dark_area|dark area" gpu1_aggregation_siege/src → 0 命中。
- Craftax 环境本身含楼层推进机制（环境原生），但仓库内没有围绕 Floor2→Floor3 transition 的 dark-area/transition/capture/probe 评测 harness。

最接近的既有基建: shared_runtime/probe_runner.py（RealProbeRunner 候选 rollout）、gpu1_aggregation_siege/src/dicode/craftax_evaluation.py（craftax 评测入口）、未提交的 e3 阶段评估器（见 0 排除项）。

结论: Floor2→Floor3 probe 不存在，属 Step 3/4 需要新建的组件（协议 G1-G3 门禁的载体）。

## 4. Checkpoint provenance（初步确认）

事实基线（experiments/henry_dicode_student_upgrade/student_candidate_registry_v1.json，2026-07-30）: "所有 checkpoint 实体均被归档策略排除、仅在服务器 oseasy@172.25.14.221；本地仅存 manifest/report/源码镜像"。本地 SHA 为报告级转录，不可本地复算。

| Student | candidate_id | 本地绑定文件 | 关键 SHA / provenance | 实体位置 |
| --- | --- | --- | --- | --- |
| GTrXL 主 Student（control） | CONTROL_CONTINUOUS_98304 | conf/student_profiles/gtrxl128_control_98304.yaml:6 | params_sha256=4c313c58…（:11）; checkpoint_dir_sha256=34819d77…（:27, CC1 orbax 目录协议 :28）; source_commit=src-sha256:d3d4e552…（:14, owner-bound runtime source sha :30） | 服务器（owner CC1） |
| GTrXL 教师参照 | BASELINE_TEACHER_CKPT17500 | conf/student_profiles/baseline_teacher_17500_smoke.yaml | registry secondary_artifact_recovery | 服务器 |
| RMT16 persistent | PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 | conf/student_profiles/rmt16_persistent_98304.yaml:9 | params_sha256=aa6ba440…（:14）; ckpt sha=2866b5de…（:35）; contract sha=7dda2bc7…（:36）; source_commit=src-sha256:453bd1ec…（:17, 冻结训练驱动源码 sha :38） | 服务器（CC2→CC4 direct98304 交接，EXTERNAL_VERIFIED_ARTIFACT） |
| RMT16 reset128 | RESET128_RMT16_ORIGINAL_VTRACE_98304 | conf/student_profiles/rmt16_reset128_98304.yaml:7 | params_sha256=78a14cc6…（:12）; ckpt sha=de3a159f…（:33） | 服务器（同上） |
| SlowGRU persistent | SLOWGRU_PERSISTENT_CANONICAL_98304 | student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304/（READY.json, candidate_manifest.json, training_contract.json, checkpoint_contract.json, SHA256SUMS）+ conf/student_profiles/slowgru_persistent_98304.yaml | canonical_checkpoint=ckpt/98304/full_state.pkl; file sha256=0bc92c9e…; params sha=99d734b4…; source_commit=57b6925e…（profile :15）; exact_resume_proof_sha28672=27bf5249…; bit_exact_determinism=true; budget_class=MATCHED_98304; owner CC3 | 服务器（GPU3 训练） |
| SlowGRU reset128 | SLOWGRU_RESET128_CANONICAL_98304 | student_pool_v1/cc3/SLOWGRU_RESET128_CANONICAL_98304/（READY.json, candidate_manifest.json, identity_verification.json, SHA256SUMS） | 同上证据包结构 | 服务器 |
| SlowGRU 长跑/中间 | SLOWGRU_RESET128_LONGRUN_98304, SLOWGRU_PERSISTENT_24576 | student_pool_v1/cc3/ 对应目录 | registry round_1 名单 | 服务器 |

Canonical 常量锚点（registry canonical_constants）: student_obs_dim=8335（=8268 base + 67 achievements multi-hot，源 d052/legacy/canonical_constants.py，冻结标签 D052_STUDENT_OBS_DIM_8335=PASS）；action_dim=43（运行时导出，run_p9_authentic_98304.py:198 assert 固化）。

Step 2 需做: 在服务器端对上述实体做 fresh-reload 可比性审计（协议 CHECKPOINT_COMPARABILITY.md）。

## 5. 协议 §1 确认清单逐项回答

1.1 GTrXL: main Student = GTrXL/Transformer-XL Actor-Critic，window_mem=128（default.yaml:26，源码消费 config.window_mem）——确认。memory 更新/使用/reset/session 行为见 3.1 第 4-7 条。
1.2 RMT16: 初始化 jnp.zeros (rmt16_memory.py:48-49)；更新=段末 cross-attention（:71-81 经 network.update_rmt_tokens）；过 Transformer（读 model_forward_eval 带 mem_tokens，写 update_rmt_tokens，rmt16_anchor.py:32-60）；persistent 跨段保留、reset128 段边界清零（rmt16_anchor.py:80-85）；两版本网络参数结构完全相同（唯一差异是 carry_mode 标志，rmt16_anchor.py:13-16 "single-change derivation"）；RMT state 真实进入 rollout 与 PPO training（Transition.rmt_entering_tokens, network.py:106-108）；checkpoint 只含 params，runtime memory 经 anchor replay 重建（rmt16_anchor.py:129）。canonical 实现定位在 student_adapters/architectures/rmt16_*.py + training_backend_rmt16.py；未依赖 henry/rmt16-l512-reachability-probe 等分支名（该分支仅存在于远端 refs，未检出其独立实现）。
1.3 SlowGRU: 见 3.3。fast memory 存在；输入=周期内 GTrXL hiddens；32-step aggregation（SLOW_INTERVAL=32）；h 参与 policy/value（forward_eval 直接产出 pi/value）；persistent 跨 rollout 保持（:19，且 checkpoint full_state.pkl 携带 longstate）；reset 边界: true episode done 清（:17/:242-254），reset128 变体的段边界清零语义在服务器端训练驱动（本地无分支实现，如实记录）。
1.4 Floor2→Floor3 / HO: 无对象（NOT_FOUND，见 3.4/3.5 及搜索证据）。

## 6. G0 结论与决策项

- 三种 Student 真实代码路径: CONFIRMED（file:line 全链）。
- checkpoint provenance: CONFIRMED（本地 manifest/profile 级；实体在服务器，Step 2 服务器审计）。
- HO 路径与 Floor2→Floor3 probe: NOT_FOUND —— 协议文本预设"当前已有 HO reinjection 框架"，但本仓库现役树没有该实现。G1（BASE vs HO_ZERO 等价）、G2（干预隔离）、G3（HO_REAL 阳性对照）三个门禁的载体都不存在。
- 决策项（需用户/总监定夺，不在 G0 内解决）:
  1) 确认 HO reinjection 与 Floor2→Floor3 probe 作为 Step 3/4 的新建工程（协议允许后续写 probe 代码），并批准其设计边界（只影响 Student 输入，不改 env state/RNG/task embedding/params）；或
  2) 指明 HO 框架的其它真实位置（若存在于服务器端未同步树中），由 Step 2 服务器审计时一并取证。
- 本文档不修改任何实验逻辑，不触碰任何既有代码（G0 纪律）。

## 7. 复现命令（审查证据可复核）

```text
git -C mechanism_UED_Henry_worktree rev-parse HEAD                      # 0dd9de5b...
rg -n "window_mem" gpu1_aggregation_siege/conf/training/default.yaml    # :26 -> 128
rg -n "class ActorCriticTransformer" gpu1_aggregation_siege/src/dicode/network.py   # :115
rg -n "window_mem" gpu1_aggregation_siege/src/dicode/ppo_tr.py          # :192/:224/:319/:1034...
rg -n "carry_mode" gpu1_aggregation_siege/src/dicode/student_adapters/architectures/rmt16_anchor.py  # :66-90
rg -n "longstate|true_done" gpu1_aggregation_siege/src/dicode/training_backend_slowgru.py            # :11-20/:215-254/:262-322
rg -n "SLOW_INTERVAL|slow_interval" student_pool_v1/cc3/slowgru_runtime/slowgru_runtime.py           # :123/:296
rg -i "reinjection|reinject|ho_zero|ho_real" gpu1_aggregation_siege/src # 0 命中
rg -i "floor2|floor3|dark_area" gpu1_aggregation_siege/src              # 0 命中
```

## 8. 并行工作流声明（落盘后补充）

本审查期间检测到同一工作树内有另一并行会话的在制品（非本任务产物，未触碰、未纳入结论）:

- gpu1_aggregation_siege/src/dicode/e3_litesim/（data/diagnostics/learning/measurement/orchestration/runtime/scheduler 子包，E3-litesim 重构脚手架，含 capability_probe/frontier_locator/counterfactual_runner 等测量组件）
- gpu1_aggregation_siege/docs/e3_litesim/E3_CURRENT_ARCHITECTURE_AUDIT.md（该工作流的 P0 只读审计，2026-08-16）

声明: 上述组件属"E3 课程训练/模拟器重构"工作流，与 Memory Study 的 Floor2→Floor3 HO probe 无已验证关联；3.4/3.5 的 NOT_FOUND 结论针对的是本审查时点已提交/已定型的现役代码。若后续 Step 3/4 决定复用 e3_litesim 的测量组件，需先对其做独立的只读审查与接口冻结。
