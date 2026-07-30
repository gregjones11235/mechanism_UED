# Candidate Runtime ABI — mechanism_UED.candidate_runtime_abi/v1

收口合同 §5 的公共 runtime 边界规范。实现:
`tools/tier3_scaffolded_evaluation/tier3_candidate_runtime.py`(自检
`--self-test`,纯门任意解释器可跑;JAX 宿主部分在 venv/服务器跑)。

## 1. 边界原则(全部 fail closed)

1. **公共 evaluator 独占科学语义**(收口 §3)。候选 runtime 不得定义:
   FRONT/BACK state-bank 选择、FULL canonical world/profile、terminal labels、
   FRONT transition 判定、FRONT graph-distance progress、BACK/FULL DEFEAT_KOBOLD
   判定、metric aggregation、evaluation certificate。
   `candidate_metadata()` 固定报告 `scientific_predicates_defined_here=false`。
2. **runtime 族注册制**。runner 按 `runtime_family` 分派,`RUNTIME_FAMILIES =
   ("rmt16_gtrxl_cc2",)`;未知/缺失族 fail closed。common runner 不硬编码 RMT16 —
   Base GTrXL / Control / SlowGRU / Teacher 由各自 owner 注册(本轮不实现)。
3. **内存语义只复用、不重实现**。RMT16 的每一步走 CC2 自己的
   `rmt_step_forward`(tier3_cc2_policy_adapter 驱动,审计源码绑定
   `cc2_policy_source_sha256`)。`memory_state` 是
   `(memories, mem_mask, mem_idx, rmt_st)` 的不透明快照;evaluator 只原样回传。
4. **batch-1**。CC2 policy N=1,`init_memory(batch_size)` 仅在 `batch_size==1`
   通过;批量会静默改变 CC2 每步动力学,直接 fail closed。
5. **done_mask 契约**。Tier3 episode 在 terminal 即停,terminal 之后不再有 step,
   因此 CC2 的 terminal-reset 分支在 episode 内永不进入:`policy_step` 要求
   `done_mask` 为 None 或全 False。
6. **checkpoint 验证唯一路径**。`load_candidate` 对真实 pkl 执行与
   tier3_evaluator 完全相同的合同核验链(文件 SHA / params SHA / manifest /
   driver-source SHA / policy-source SHA vs 冻结 final98304 合同),不存在第二套门。
7. **params 只读**。无优化器、无训练、无采样;greedy_argmax(冻结)。

## 2. ABI 表面

```
load_candidate(checkpoint_contract: dict) -> CandidateRuntime
runtime.init_memory(batch_size: int) -> memory_state
runtime.policy_step(observation, memory_state, done_mask=None)
    -> {"action": int, "memory_state": memory_state}
runtime.reset_memory(memory_state, reset_mask=None) -> memory_state
runtime.candidate_metadata() -> dict
```

`checkpoint_contract` 字段:`runtime_family`(必)、`arm`(必,
persistent|reset128)、`checkpoint_path`(必);可选
`checkpoint_contract_path` / `cc2_snapshot_root` / `driver_source_path` /
`observation_shape`(必须 (8335,)) / `action_dim`(必须 43)。

`reset_memory`:`reset_mask` None/False → 原样返回并校验键集;True → 重新初始化
(与 `init_memory(1)` 叶值相等)。

## 3. 内存快照的正确性依据

快照是四个状态字段的**引用捕获**,不做拷贝。安全性来自 CC2
`rmt_step_forward`(审计源码 `rmt_memory_anchor.py`)的纯函数性:每步返回全新
数组与全新 `rmt_st` 字典,不原位修改、不做 buffer donation。因此任意早先快照
在 policy 继续运行后仍然逐位有效 — 自检以"从第 3 步快照续跑必须复现原序列
第 4-6 步动作"验证此不变量(memory_snapshot_restore_continues)。

## 4. candidate_metadata 身份字段

`runtime_family` / `arm` / `carry_mode` / `checkpoint_step` /
`checkpoint_file_sha256` / `params_sha256` /
`base_checkpoint_params_sha256` / `driver_source_sha256` /
`cc2_policy_source_sha256` / `checkpoint_contract_sha256`(全部 FULL SHA)+
`action_mode=greedy_argmax` / `action_dim=43` / `observation_shape=[8335]` /
`trainable=false` / `batch_size_supported=1`。capsule 的
checkpoint_contract.json / READY.json 直接消费这些值。

## 5. 冻结恒等式(与合同 §6 逐字节一致)

| 候选 | checkpoint_file_sha256 | params_sha256 |
|---|---|---|
| Persistent RMT16 @98304 | `2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723` | `aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d` |
| Reset128 RMT16 @98304 | `de3a159f58f904c4ed0bce17bcb87e4b39b21b4ffd0cea557ce61b860727b638` | `78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2` |

Base-compatible params SHA(两臂共同基座):
`d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5`。
以上五个值冻结于 `tier3_checkpoint_contract.py`,服务器正式加载时由
`load_candidate` 重新计算并比对 — 不一致即 fail closed,不得静默替换。

## 6. 自检覆盖

纯门(任意解释器):未知族/缺族/坏 arm/缺 checkpoint_path/非 dict 合同拒绝;
batch!=1 拒绝(含 bool);done_mask 含 True 拒绝;reset_mask size!=1 拒绝;
ABI 表面与注册族冻结值检查。
JAX 宿主(venv/服务器):合成 random-init policy 上的 ABI 确定性(两个 runtime
同参同观测 → 同动作序列)、快照续跑不变量、reset True/False 语义、params
只读(逐叶前后相等)、persistent+reset128 双 carry_mode 加载。
真实 pkl(服务器,`CC4_REAL_PKL_PERSISTENT` 或 binding smoke 驱动):完整合同
核验链 + 单步 ABI。
