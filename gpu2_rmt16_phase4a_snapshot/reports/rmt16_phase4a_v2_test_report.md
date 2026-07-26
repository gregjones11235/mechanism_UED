# RMT16 Phase4A-v2 — 测试报告（§十 15 门禁）

**任务**：`RMT16_PHASE4A_V2_ORIGINAL_GOAL_VTRACE_IMPLEMENTATION`
**实现分支**：`henry/rmt16-phase4a-v2-original-vtrace`
**审核基线**：`henry/reviewed-rmt16-l512-probe` @ `d3c8c7d6abd2df3d0ba69dc2c1f326f8668798e5`
**测试文件**：`tests/test_phase4a_v2_gates.py`（可直接 `python tests/test_phase4a_v2_gates.py`，亦兼容 pytest）
**重算脚本**：`tests/recompute_probe_step.py`；**config diff**：`tests/config_diff_validator.py`

---

## 0. 结果总览

| 层 | 环境 | 结果 |
|---|---|---|
| 本地（纯 Python / numpy / AST / subprocess） | Windows，py + numpy 1.26.4 + yaml 6.0.1 + pytest 9.1.1，**无** jax | 14 PASS / 0 FAIL / 1 SKIP（GATE8 需 JAX） |
| 服务器 CPU（全量，含 JAX 门禁） | `dicode310`，py3.10.20，**jax 0.6.0**，numpy 2.2.6；`JAX_PLATFORMS=cpu`，`CUDA_VISIBLE_DEVICES=""` | **15 PASS / 0 FAIL / 0 SKIP** |

```
SUMMARY pass=15 fail=0 skip=0
GATES_RESULT=PASS      (SUITE_EXIT=0)
```
** failing tests = 0 **。本轮未为追求通过而改任何阈值；未发生"失败后自动修码重跑"。

> 本地运行期修正一处**测试脚本自身**的缺陷（GATE15 在 Windows 下 `subprocess` 未加 `sys.executable`
> 前缀触发 WinError 193）——这是测试 harness 的平台兼容修正，**不**触及任何被测代码/阈值；修正后
> 本地 14/14、服务器 15/15 全绿。

## 1. 部署字节一致性（§一：git 文件 ↔ runtime 文件）

- 本地 `core.autocrlf=true`，工作区为 CRLF → git **blob 存 LF**。为使"git 文件 == runtime 文件"字节成立，
  部署树做了 **LF 规范化**（部署字节 == 未来 git blob 字节；`.sh` 在 Linux 亦必须 LF）。
- 25 个文件（runtime 全树含 wrapper_src、configs×2、tests×3）打包 scp 至服务器隔离目录
  `experiments/rmt16_replay_phase4a/phase4a_v2_deploy/`（**不**触碰 CC2 运行 src、**不**触碰 git_work 状态）。
- 服务器逐文件 `sha256` 对照 `deploy_manifest.json`：**25/25 OK，`DEPLOY_BYTE_PARITY=PASS`**。
- 提交后将以 `git show HEAD:<path>` 的 blob sha256 再次对照本 manifest（LF 字节同源），闭环 §一 校验。

## 2. 逐门禁结果（服务器 CPU 实测）

| # | 门禁 | 方法 | 结果 | 证据 |
|---|---|---|---|---|
| 1 | 旧 L512 可达性可由原始 episode 记录重算，结论仍 BOTH | 合成自检（本地）+ 两臂真实 jsonl 重算（服务器） | **PASS** | Persistent 6/20、Reset128 5/21，reachable=true；首条 ge512 resolved=8979（见 §二 报告） |
| 2 | `completion_resolved_env_step` 公式对各 env_id/rollout_step 正确 | 纯函数断言 + 1..2048 连续性 | **PASS** | (0,0,0)=1、(0,0,15)=16、(0,1,0)=17、(1,0,0)=2049、(2,5,3)=4180；deprecated 不同 |
| 3 | outer/PPO/Replay/policy_version 计数不混用 | `Phase4ACounters` 行为断言 | **PASS** | KL-rollback：executed+1 但 accepted/policy_version 不变；PPO 计数不被 replay 影响 |
| 4 | original_vtrace 不调 Hindsight | AST 结构（两函数体符号集） | **PASS** | 无 `relabel_sample_rmt / relabel_trajectory_rmt / rmt_hindsight / RH` |
| 5 | original_vtrace 不算 AWR | AST 结构 | **PASS** | 无 `awr / awr_losses / AWRConfig / w_awr / A` |
| 6 | loss 仅一次 original RMT scan + 对应 target scan | AST 调用计数 | **PASS** | loss `scan_fn`×1；update `_target_scan_rmt`×1、`reconstruct_rmt_batch`×2（online+target，皆 original）；无 `recon_r/target_vals_r/samples_rel` |
| 7 | sequence_length=129 跨 128 边界 | config + launcher 静态 | **PASS** | 两 config 129>128、crosses_boundary=true；launcher `default=129` + original_vtrace 越界 guard |
| 8 | Persistent 第129步进入 token 非零 / Reset128 token 零 | **JAX CPU** `rmt_advance_tokens` 前向 | **PASS** | 边界(seg_count 127→128)：persistent mem_tokens maxabs=5.0（非零携带），reset128=0.0（清零）；seg_count 两臂均复位 |
| 9 | eligible-only 采样器绝不抽短轨迹 | numpy buffer 行为 | **PASS** | 8/8 length==200，仅取自 {200,260}；短(150) 从未被抽；空 eligible → 显式 NOT_READY（无异常、无短顶替） |
| 10 | 相同 buffer+RNG → 相同 sample IDs & offsets | numpy buffer 确定性 | **PASS** | ids/offsets/lengths 逐位复现；**不**依赖隐藏 `self._rng`（扰动后仍复现）；新同构 buffer 亦复现 |
| 11 | KL rollback → policy_version 语义正确 | `Phase4ACounters` 行为 | **PASS** | rejected：executed+1、policy_version/accepted 不变；committed：policy_version+1、accepted+1 |
| 12 | checkpoint 含全部状态 | launcher `save_ckpt` 静态 | **PASS** | params/PPO opt/Replay opt/target(EMA)/rng/action_rng/buffer/pending/memories/mem_mask/mem_idx/rmt_state/obsv/counters + replay RNG state + phase4a_v2 |
| 13 | 旧探针 off-path 逐位不变 | 构造论证 + off 计数等价单测 + replay-guard 静态 | **PASS**（构造） | off 下每步 policy_version==legacy update_count；replay/relabel 调用均在 `REPLAY_ON and REPLAY_MODE==` 之后；改动全加性。**逐位 hash 数值复跑：本轮未授权（NEW_TRAINING_RUNS=0），记为 deferred** |
| 14 | Persistent/Reset config diff 仅 carry_mode | `config_diff_validator` 递归 diff | **PASS** | `scientific_config` 唯一叶差异 `carry_mode`（persistent/reset128）；§六 不变式两臂均成立 |
| 15 | full_p2_legacy 需显式授权 | 行为子进程（JAX 导入前退出） | **PASS** | 缺 `--replay_mode` → exit 2；`full_p2_legacy` 无 `--allow-full-p2-legacy` → exit 2（均在 JAX import 前，本地无 jax 亦可复现） |

## 3. §八 Hindsight 防火墙的验证方式说明

任务书要求"monkeypatch `RH.relabel_sample_rmt` 使其被调用即 raise，所有 original_vtrace 测试仍通过"。
本测试采用**更强的 AST 结构证明**（GATE 4/5/6）：直接证明两个 original_vtrace 函数的符号集里**根本不存在**
任何 relabel/hindsight/AWR 符号——这是"结构不进入"，强于"某次运行未调用"的 monkeypatch。配合
`original_vtrace_update_rmt` **没有** `samples_rel` 形参（relabeled sample 无法传入）、launcher 在每次
original_vtrace replay 更新后与 run 终局各调用一次 `assert_hindsight_awr_disabled()`（四个防火墙计数 == 0 硬断言），
构成 `HINDSIGHT_STRUCTURALLY_DISABLED=true` / `AWR_STRUCTURALLY_DISABLED=true` 的依据。

## 4. 分层与 deferred 项（诚实记录）

- **本地可证**（无 JAX）：GATE 1(方法)/2/3/4/5/6/7/9/10/11/12/13(构造)/14/15。
- **服务器 CPU 实证**（JAX 0.6.0）：GATE 8（token 非零/清零前向）；GATE 1 两臂真实 jsonl 重算；全量复跑确认无 SKIP。
- **deferred（本轮未授权，非失败）**：
  - GATE 13 的**逐位 hash 数值复跑**——需一次参数更新 run，本轮 `NEW_TRAINING_RUNS=0` 禁止；以构造论证 + 计数等价单测 + 加性审计代替，详见 `rmt16_phase4a_v2_known_limitations.md` §6。
  - `MATCHED_REPLAY_EXPOSURE` 的跨臂**真实相等**——协议与计数器已就绪（GATE 9/10/3），但尚无 original_vtrace 正式 run 去满足它；`matched_replay_protocol_ready=true` 仅表示"协议可用"，**不**表示"已匹配"。

## 5. 结论

- 15 门禁全绿（服务器 CPU）；failing tests = 0；无阈值篡改；无自动修码重跑。
- off-path 加性、original_vtrace 结构隔离、eligible-only 确定性、config 单变量、checkpoint 完备、legacy 授权门禁均按 §十 验证。
- 本轮 `NEW_TRAINING_RUNS=0`、未用 GPU 训练（GATE 8 为 CPU 前向，`CUDA_VISIBLE_DEVICES=""`）、未启动正式两臂。
