# D1 审计 Addendum（结论修订）

分类：`D1_AUDIT_ADDENDUM`（Sol 独立审计结果：`PASS_WITH_CONCERNS`）

## 1. 原结论 → 修订结论

| 项 | 原 | 修订 |
|---|---|---|
| D1 结论 | `LLM_RESEARCH_NO_IMPROVEMENT` | `D1_CHAT_ONLY_SINGLE_SLOT_NO_SIGNIFICANT_SPEEDUP` |
| 阶段 D 总状态 | （隐含"已完成"） | `LLM_RESEARCH_PASS_WITH_CONCERNS` |

## 2. 哪些数据仍有效

- 六组汇总数字真实存在且可复算（六个 `result_sha256` 均验证通过）。
- `wall_clock_s / llm_union_s / llm_sum_s / retry_count / empty_response_count / valid_task_rate` 等汇总字段。
- Mason 原始日志统计：HTTP 200=2201，chat=1626，embedding=575，retry=863。
- GPU0 Ollama 服务 UUID 与 PID 证据。
- 本地测试结果：17 passed / py_compile PASS / git diff --check PASS。
- manifest SHA `2066515499…` 与 system_prompt SHA `5cdc991c…`。
- 12 个合成重建 code-generation prompts 的**受控调度比较**（合成输入仍可用于调度比较，只是不能用于生产等价判断）。

## 3. 哪些因果推断被撤回

1. ~~有界并发消除了 Mason 的 863 次 retry~~ → 撤回。D1 未含 embedding（574/863 retry 来自 embedding）。
2. ~~避免了连接风暴~~ → 撤回。
3. ~~queue 从客户端搬到服务端~~ → 撤回。`queue_wait_sum_s` 是客户端 semaphore wait 累加，不是服务端排队。
4. ~~GPU0 全程显存恒定~~ → 撤回。只有每组前后快照。
5. ~~最低剩余显存精确为 2510 MiB~~ → 撤回。快照值非连续峰值。
6. ~~Mason 固定 prompt replay~~ → 撤回。实际为合成重建。
7. ~~阶段 D 已全部完成~~ → 撤回。embedding 与无界对照尚未做。

## 4. 原始事件已删除的事实

六次运行的原始 `events.jsonl / events.csv / critical_path.json / RESULT.json` 已随 `/tmp` 沙箱删除，未持久保存，**无法完成请求级独立复核**。此事实不可补造、不可恢复。

## 5. 无法恢复的证据

- 六次运行 `events.jsonl`（请求级时序）。
- 六次运行 `events.csv / critical_path.json`。
- 六次运行 `RESULT.json` 原始文件（汇总字段已保留，`result_sha256` 可复算，但原始文件已删）。
- GPU0 连续采样数据（从未采集）。

## 6. 允许保留的结论

在当前单槽 Ollama 14B、12 个合成重建 code-generation prompts、每档 2 repeat 的条件下，`max_in_flight=2/4` 相对 `1` 没有显示稳定显著的生成 union 或总 replay 墙钟收益。因此暂不将客户端 chat 并发作为 48 小时主线加速项。

## 7. 后续 D1b 实验定义

- **目标**：判断 Mason 的 embedding retry 是否与客户端并发相关。
- **冻结**：从 Mason attempt06 archive/task graph 冻结实际存在的任务描述文本；同一 manifest 用于所有并发配置。
- **规模**：至少 16 文本批次（与真实后期规模相当）。
- **矩阵**：`max_in_flight ∈ {1, 2, 4, production_like_unbounded(=25)}`，每档 ≥2 repeat，顺序交替（1→2→4→unbounded 与 unbounded→4→2→1）。
- **安全门禁**：OOM / GPU0 剩余显存过低 / Ollama 异常退出 / 持续 timeout / 大量 connection reset / 队列无法恢复 → 立即停止后续高并发臂。

## 8. chat 无界对照定义（D1b 安全后执行）

- 标签：`PRODUCTION_STYLE_CLIENT_UNBOUNDED_FOR_12_REQUESTS`，`max_in_flight=12`。
- 2 repeat，顺序 1→12→12→1。
- 仅判断 12 请求 workload 下无界并发是否增加 retry / 影响 union wall / 影响有效率；不等于 Mason 25 候选或 embedding workload。
