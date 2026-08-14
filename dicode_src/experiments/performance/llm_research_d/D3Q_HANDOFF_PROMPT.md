# D3Q 交接提示词（给 Codex 的详细执行说明）

你接管 DiCode D3Q 实验的**后续实现**。前序已打通 metadata gate，你现在按下方"压缩计划"继续实现并跑完实验，产出最终报告。除安全/语义/API/GPU/证据完整性阻断外持续推进，不要停下来等确认。

## 一、工作位置与当前状态（已核实）

- worktree：`C:\Users\Lenovo\Desktop\dicode-codex-director\skill_preflight_ued_d3_worktree`
- branch：`perf/llm-small-large-quality-cost-d3`
- base：`62b7d115b6de6506cb955733beaf1f5b8e79d521`，当前 HEAD `58af146`（本 session 已提交 6 个修复，勿 amend/rebase）。
- 服务器：`oseasy@172.25.14.221`；SSH key `D:\Projects\dicode-codex-director\orchestration\control\ssh_oseasy_172_25_14_221_ed25519`。
- 远端 env：`/home/oseasy/.config/dicode/experiment_llm.env`，DeepSeek credential 变量 `EXP_DEEPSEEK_API_KEY`。
- 远端 python：`/home/oseasy/venvs/skill_preflight_e0e1/bin/python`。

## 二、已完成的里程碑（不要重做）

1. **metadata gate 已 PASS**：`GET https://api.deepseek.com/models` → HTTP 200，`deepseek-v4-flash` advertised=true，request_count=1，completion=0，embedding=0。证据在 `d3_artifacts/deepseek_flash_metadata_gate_rerun_20260814T230252Z/`。
2. **env 已修复**（用户授权）：已追加 `EXP_DEEPSEEK_MODEL=deepseek-v4-flash` + `EXP_DEEPSEEK_PROVIDER=deepseek`（未动既有 `EXP_GENERATOR_*`）。
3. **旧代码已改**：`d3_runner.py` `LARGE_MODEL`、`d3_metadata_gate.py` `DEEPSEEK_MODEL` 已从 `deepseek-v4-pro` 改为 `deepseek-v4-flash`。测试文件里故意保留的 `v4-pro` 反例**不要改**。
4. **launcher 已修**：OpenSSH 10.x post-quantum stderr 警告已容忍；request_count 状态机（pre_request_blocked=0 / request_attempted_blocked=1 / PASS=1+200+model）已正确。

## 三、环境陷阱（务必遵守）

1. **Git Bash 路径转换**：本地跑 launcher/脚本传 `/home/...` 远端路径时，必须 `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'`，否则 `/home/...` 被转成 Windows 路径。SSH key 用 Windows 风格 `D:/Projects/...`。
2. **SSH 必须带** `-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes`。
3. **禁止**：读取/输出/记录/哈希 `EXP_DEEPSEEK_API_KEY` 值；把凭据放入 argv/stdout/stderr/artifact；`shell=True`；SDK 隐式 retry；并发正式请求。
4. **GPU**：GPU0=Ollama 14B（不动）；GPU2=`GPU-8df11537-ab79-722d-606f-411966196c4c`（唯一 Preflight 卡，UUID 门禁、无外部 compute PID）；GPU1/GPU3 只读不触碰。

## 四、压缩计划（本 pilot 规模，偏离原 72-slot 协议，已获用户同意）

| 项 | 值 |
|---|---|
| prompts | 6 = early `task_1/task_2`、mid `task_276/task_277`、late `task_551/task_552`（从 `FROZEN_MANIFEST.json` 确定性选取，顺序保持冻结） |
| repeats | 2（顺序 repeat0 A→B、repeat1 B→A） |
| models | 2（`qwen2.5-coder:14b` vs `deepseek-v4-flash`） |
| slots | 24 |
| 每 slot POST 预算 | 3（初始/transport/empty retry/semantic repair 共用，硬上限） |
| 总 POST 上限 | 72/模型 |
| Preflight arms | 4（2 models × 2 repeats），交替 A→B→B→A |

**固定不变**：temperature=0.6、top_p=0.95、max_tokens=8192、max_in_flight=1；DeepSeek thinking enabled、Ollama 非 thinking；禁止改 prompt/人工修代码/删失败样本/换模型/隐藏 warmup；第一条正式请求计入统计。

## 五、每个候选的流水线（固定顺序）

1. 记录请求元数据（model/provider/prompt SHA/slot ID/attempt/request ID/start-end monotonic ns/duration/HTTP status/token usage/finish reason）。
2. 提取代码（与生产同源 extraction，禁人工修）。
3. 静态检查（Python syntax、非法 import、Craftax 枚举、Inventory 参数、文件/网络/subprocess 等危险能力）。
4. CPU-JAX validation（`CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`，隔离进程，记录 compile/execute/error class，超时）。
5. repair（仅未通过候选；同模型修自己；统一冻结 repair 模板；准确脱敏错误；每 slot 总 POST 仍 ≤3）。
6. 最终冻结（保存最终代码 + code SHA + attempt chain + 初始/最终 validation 状态，不覆盖原始输出）。

逐 slot 必须输出：initial_valid / final_valid / attempts / repair_requests / repair_success / empty_response / timeout / connection_error / http_4xx / http_5xx / invalid_json / extract_error / syntax_error / api_enum_error / cpu_jax_error / duplicate_code / prompt_tokens / completion_tokens / cached_tokens / generation_wall_s / repair_wall_s / cpu_validation_wall_s / final_code_sha256。

## 六、GPU2 真实 Preflight（4 arms）

- 每 arm = 该 model/repeat 汇总出的最终**唯一有效候选集合**，注入 archive 副本（不改原 archive），跑真实链路 `TaskArchive→load_tasks_from_env_codes→evaluate_new_tasks→calculate_scores_from_snapshot→preflight_route`。
- 冻结：Mason attempt06 step 2100 checkpoint、40 updates、1024 envs、128 steps、RNG/config/archive/conditioning/task order、score function、route threshold、reset indices、GPU UUID。
- 每 arm 记录 code load、evaluate build/lower/compile/execute、scoring transfer、scoring CPU、route、archive update、total wall、scores、kept/rejected、accepted code SHA、RNG after、GPU peak/min free、OOM/Xid/traceback。
- 禁止并发 Preflight；GPU3/GPU1 不触碰；确定性 XLA。

## 七、统计与结论

按 model/repeat 报告：初始/最终有效率、平均 repair、CPU-JAX pass rate、Preflight accepted rate、各阶段 wall、总 wall、每最终有效任务 LLM 秒数、每 accepted task 端到端秒数、tokens、DeepSeek 成本快照（官方价格页 + 抓取日期 + 单价 + 公式；拿不到价格就只报 tokens，不编费用）。给出 repeat-level paired difference、mean、median、range，并明确 n=2 证据有限。

结论七选一（见 `D3Q_HANDOFF_GOAL.md`）。**只有满足"小模型生成更快 + 有效率更低/repair 更多 + 每 accepted task 秒数更高 + ≥2 repeats 同方向 + 无外部解释"才答"小模型因无效任务反而更慢"，否则写证据不足/无明确赢家。**

## 八、Git 与审核纪律

- 每阶段独立 commit：smoke、生成矩阵、GPU2 Preflight、最终报告。
- 每个代码阶段另开只读审核 agent，审核状态只能 PASS / PASS_WITH_CONCERNS / REJECT，REJECT 修复后复审。
- 禁止 push/merge；禁止 amend 历史；旧 D1/D2/D3Q/Qwen/D4-D6/B-C/Mason evidence 必须 byte-identical。
- 每个 artifact：canonical internal hash + raw file SHA + SHA256SUMS + 封闭 schema + 原子发布 + 外部 tool/source SHA 绑定 + no-secret scan。

## 九、停止条件（任一命中即 fail-closed 停，保留脱敏证据，不重试不伪造）

密钥泄漏、request 超预算、模型非 exact `deepseek-v4-flash`、prompt/参数/repair 预算不一致、GPU UUID 不匹配、GPU 外部进程、OOM、CUDA Xid、损坏 checkpoint、未解释 traceback、artifact hash/schema 不一致、远端 tmp 无法清理、旧 evidence 被修改、API 余额/权限阻断。

## 十、服务器连接模板

```
ssh -i "D:\Projects\dicode-codex-director\orchestration\control\ssh_oseasy_172_25_14_221_ed25519" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes oseasy@172.25.14.221 '<command>'
```

本地跑 launcher 需 `MSYS_NO_PATHCONV=1` + Windows 风格 key 路径（见第三节）。
