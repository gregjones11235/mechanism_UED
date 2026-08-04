# E1 round-3 当前阻断清单（PRODUCTION_PATH_READY_FOR_AUDIT）

> 本清单由 `scripts/run_e1_real_one_update.py` /
> `scripts/run_e1_longrun.py` / `scripts/e1_formal_readiness.py`
> 的实跑阻断码汇总（本机，审计 venv，JAX_PLATFORMS=cpu）。
> 每一项都给出证据路径；阻断解除前生产路径恒 fail-closed，
> 三个 `REAL_*` 标志保持 false。

## 1. 共享运行时未注入（8 个合同全部未绑定）

- 码：`BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>` × 8
  （StudentIdentity / StudentAdapter / ReferenceIdentity /
  ReferenceAdapter / AnchorManifest / FormalAssetRegistry /
  CandidateProbeResult / FullStateCheckpoint）。
- 证据：`src/dicode/teachers/e1_formal/shared_runtime_seam.py`
  （`SHARED_RUNTIME_MODULE = "dicode.shared_runtime"` 在本 worktree
  不存在；接缝只解析、不构造、不铸造、不伪装）；
  `reports/e1_formal_ued/real_smoke_readiness.json` 的
  `shared_*_bound` 三字段均 false。
- 影响：统一 probe 入口 `evaluate_candidate` 恒返回
  `BLOCKED_WAITING_SHARED_RUNTIME`（`evaluated=False`，无
  `CANDIDATE_EVALUATION` 戳）；单更新入口与长跑 manifest 的
  checkpoint 字段被阻断。解除方=CC4 共享运行时。

## 2. Reference 身份合同未冻结（G1）

- 码：`REFERENCE_CONTRACT_UNFROZEN`。
- 证据：`conf/teacher/e1_formal.yaml` 的
  `teacher.reference_contract.frozen: false`（全部身份字段 null，
  无默认）；`src/dicode/teachers/e1_formal/reference_contract.py`
  fail-closed 消费。
- 影响：评价 seam 阻断；长跑 manifest 的 Reference 字段判为未冻结
  ⇒ 拒绝。解除方=总控冻结 G1 身份值。

## 3. 共享 anchor manifest 为 DRAFT（G3）

- 码：`BLOCKED_SHARED_ANCHOR_MANIFEST` /
  `ANCHOR_MANIFEST_NOT_FROZEN`。
- 证据：`configs/e1_formal_ued_anchor_manifest.DRAFT.json`
  （`status: DRAFT_UNFROZEN`，各身份字段
  `DRAFT_UNFROZEN_PENDING_SUPERVISOR`）。
- 影响：retention 评价与 REUSE 认证保持 BLOCKED；长跑 manifest 的
  anchor 字段判为未冻结 ⇒ 拒绝。解除方=总控冻结 G3 manifest。

## 4. 真实 EnvCoder 后端未授权

- 码：`ENVCODER_BACKEND_BLOCKED`。
- 证据：`src/dicode/teachers/e1_formal/envcoder_backends.py`
  （`RealBackendAdapter.validate` 与 `make_backend("real")` 恒抛
  阻断码；八阶段 SYNTAX→…→TERMINAL_AUTORESET 中后五阶段需要真实
  craftax 运行时，本审计 venv 无 craftax）。
- 影响：真实 reset/step 验证不可执行；生产 EnvCoder 停留在 replay
  后端（SYNTAX+GUARDS+STRUCTURE，诚实标注不执行 craftax import）。
  解除方=craftax 运行时 + 总控授权 real 后端。

## 5. 无授权真实 LLM provider

- 码：`E1_REAL_LLM_NOT_AUTHORIZED`。
- 证据：`scripts/e1_production_runtime.py`
  （`AUTHORIZED_REAL_LLM_PROVIDERS = ()`，白名单由总控持有且本轮
  为空；六角色 board 仅在显式授权下可用真实 provider，envcoder/
  probe 永不回退）。
- 影响：真实六角色开窗不可执行。解除方=总控下发授权 provider。

## 6. 本机无 craftax/jax 真实训练运行时

- 证据：`src/dicode/training.py` 模块级依赖 jax/craftax/optax/flax
  （审计 venv 不可用）；套件 5 个 skip 全部因 craftax 缺失。
- 影响：`run_session_training` 单更新路径只能在具备真实运行时的
  环境执行；本轮 `REAL_TRAINING_UPDATE_EXECUTED=false`。

## 7. 网络不可达（如推送失败，如实记录）

- 证据：step 0 `git fetch` 曾失败（网络不可达），以本地 HEAD 为
  基线继续；`git push origin henry/static-llm-ued-v1` 若失败 ⇒
  `REMOTE_PUSHED=false`，由既有定时任务（BA-BAGR 同款复查机制）
  在网络恢复后非强制推送并复核远端 SHA。

---

**汇总**：单更新真实门实跑 12 条阻断（第 1 节 ×8 + 第 2/3/4/5 节
各 1）；长跑 manifest 3 个字段未冻结（第 2/3 节 + 第 1 节的
FullStateCheckpoint）。全部阻断解除后，
`real_smoke_readiness.json::e1_real_smoke_ready` 才会由脚本计算为
true——布尔绝不手写。
