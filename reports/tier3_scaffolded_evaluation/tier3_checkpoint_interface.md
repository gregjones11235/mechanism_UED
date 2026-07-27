# Tier3 Checkpoint 接口 (checkpoint interface)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1;模块 `tier3_checkpoint_adapter.py`
- 状态: **PASS**(`tier3_checkpoint_adapter.py --self-test` exit 0;NEG21/22/23 守卫 live)

## 1. 职责边界

- **CC2** 训练 Student / 写 checkpoint;**CC3** 消费 CC4 的 StudentProfile。
- 本适配器是 CC4 对 Student checkpoint 的**只读身份层**:只绑定“哪份 params + 哪种 observation/action 接口”,**从不训练、从不写参数**。

## 2. Checkpoint 身份记录(只读)

`make_checkpoint_record(params, observation_shape, action_space_id, ...)` 产出:
`params_sha256(seed-free 规范字节哈希)`、`observation_shape`、`action_space_id`、`observation_schema=canonical_craftax_symbolic`、`trainable=False`、`writable=False`、`trained_by=CC2_TRAINING_RUN`。params 不以可变引用保存,仅存哈希与接口描述。

## 3. 守卫(负向测试)

| 守卫 | 负向测试 | 语义 |
|---|---|---|
| `assert_params_identity` | NEG21 | checkpoint params SHA 必须等于声明值,否则 fail-closed |
| `assert_observation_shape` | NEG22 | observation 接口必须与评测器一致,否则 fail-closed |
| `assert_evaluation_does_not_update_params` | NEG23 | 评测前后 params SHA 必须相同且 trainable/writable=False;任何改变 → fail-closed |

`params_sha256`:纯 JSON 化 params 用 canonical JSON;含数组/pytree 的真实 params 回退到 V3 canonical encoder(加法式复用,不重新实现)。

## 4. 本机状态

真实 JAX checkpoint 不在本机 → 以合成 params dict 演练(TESTED_SYNTHETIC);`load_checkpoint_readonly` 的只读 SHA 身份对任意文件可行(已用 world_builder 源文件验证,SHA 与审计值一致)。
