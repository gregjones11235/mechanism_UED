# D1c 根因假设（实验前，未定论）

分类：`LLM_RESEARCH`。实验前记录，禁止在实验前宣称根因已确定。

## 一、8 个关键问题

| # | 问题 | 回答 | 证据等级 |
|---|---|---|---|
| 1 | Mason 每次 embedding 是 batched 还是每文本一 request | **一个 batched request**（`get_embedding([pivot]+valid_contents, instruction)` 单请求多文本） | DIRECT_EVIDENCE（`gen_manager._order_similar_tasks:692`） |
| 2 | embedding client 是否跨 session 长期复用 | **是**。`GenManager.__init__` 创建一次 `embedding_model=LLM(...)`，`TaskSelector` 长期持有 | DIRECT_EVIDENCE（`gen_manager:2384/2431`） |
| 3 | embedding client 在 chat 期间是否长期空闲 | **是**。evolution worker 内 embedding（`select_similar_desc_tasks`）先于 chat（`_query_and_parse_responses`+`generate`），chat 占分钟级 | INFERENCE（代码顺序 + Mason evolution ~1488s） |
| 4 | 574/575 retry 是 SDK 内部还是应用外层 | **SDK 内部**。`openai._base_client` "Retrying request" 日志；`_query_local_embed` 无应用层 retry | DIRECT_EVIDENCE（日志 + `llm.py`） |
| 5 | 日志能否识别 retry 前异常类型 | **不能**。日志只有 "200 OK"+"Retrying request"，无 exception class | NOT_OBSERVED |
| 6 | embedding server 与 chat server 是否不同 PID | **是**。nomic `3156249` vs qwen `3154045` | DIRECT_EVIDENCE（process_manifest） |
| 7 | retry 是否来自 stale keep-alive | **待验证**（核心假设，见下） | INFERENCE |
| 8 | 是否存在代理/连接池/HTTP2/timeout/重启证据 | 无代理（localhost）、无显式 pool/timeout 配置（SDK 默认）、无重启（PID 稳定）、HTTP/1.1 默认 | NOT_OBSERVED / INFERENCE |

## 二、假设清单

### H1：stale keep-alive connection（主假设）

**机制**：embedding client 跨 session 长期存活；chat generation 期间空闲数分钟；下次 embedding 首次复用旧 keep-alive 连接时，服务器已关闭连接，触发 transport reset → SDK retry。

- 证据等级：INFERENCE
- 对应实验臂：C（PERSISTENT_IDLE_GAP）

### H2：asyncio.run 每次新 event loop 导致连接池跨 loop 复用失败

**机制**：`get_embedding` 用 `asyncio.run(...)`（每次新 event loop），但 AsyncOpenAI client 只创建一次；httpx 连接池绑定首个 loop，后续 loop 复用旧连接失败 → retry。

- 证据等级：INFERENCE
- 对应实验臂：A（persistent contiguous，若触发）vs B（fresh client per request，重建 client 规避）

### H3：连接池并发争用（已被 D1b 削弱）

**机制**：高并发请求耗尽连接池。D1b 在 1/2/4/25 并发下均 retry=0，此假设已被削弱。

- 证据等级：DISPROVED（相对 D1b 证据；但不排除其他触发）

### H4：特定时间窗口服务端状态

**机制**：Mason 运行期间的特定 nomic server 状态/负载导致连接关闭。

- 证据等级：NOT_OBSERVED（无法从当前环境复现/证实）

### H5：response body 读取中途连接关闭（remote protocol error）

**机制**：请求拿到 200，但响应体读取中途连接被关闭（`httpx.RemoteProtocolError`），SDK 重试。与 "200 OK 后紧跟 Retrying" 的日志模式一致。

- 证据等级：INFERENCE（与日志模式吻合，但无 exception 证据）
- 对应实验臂：D（若复现，max_retries=0 捕获异常）

## 三、结论边界

实验前不宣称根因确定。D1c 结论只能从（`D1C_RETRY_REPRODUCED_STALE_CONNECTION` / `D1C_RETRY_REPRODUCED_OTHER_TRANSPORT_CAUSE` / `D1C_RETRY_NOT_REPRODUCED` / `D1C_INCONCLUSIVE` / `D1C_REJECTED_RUNTIME_FAILURE`）五者之一选择。
