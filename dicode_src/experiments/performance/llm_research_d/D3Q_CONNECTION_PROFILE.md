# D3Q Phase-2 Connection Profile

classification: D3Q_CONNECTION_PROFILE
schema_version: 1
recorded_utc: 2026-08-16T03:52:59Z
decided_by: codex-primary (direct execution authorized by user goal)

## 用户指令时间线

1. 原始指令：连接服务器注意使用代理端口（117.50.183.232:23）。
2. 2026-08-16 用户通知：本地端口映射已更换，旧映射（端口 23）作废，新映射端口 39467。
3. 两条代理映射均不可达后，用户授权直接执行，按冻结的 Probe 顺序降级到直连路由。

## Probe 结果（按冻结简报顺序）

### Probe A — 代理端口 117.50.183.232:23（oseasy + 项目密钥）

- TCP: Test-NetConnection 117.50.183.232:23 -> TcpTestSucceeded=False；ICMP 可达（RTT 52ms，源 172.20.0.193/WLAN）。
- SSH: ssh -p 23 退出码 1，错误 banner exchange: Connection to UNKNOWN port -1: Connection refused。
- 结论：端口 23 TCP 层不可达，SSH 未建立，未发送任何远端命令。与旧文档 legacy sync host intermittent refused 记录一致；用户随后确认该映射已被更换。

### Probe A2 — 用户新映射 117.50.183.232:39467

- TCP: Test-NetConnection :39467 -> TcpTestSucceeded=False（超时）；同主机 :22 -> False。
- SSH: ssh -p 39467 退出码 1，connect to host 117.50.183.232 port 39467: Connection timed out。
- 结论：新映射同样不可达（未建立连接，无远端副作用）。

### Probe B — root@117.50.183.232（个人密钥）

- 未执行：冻结简报规定 Probe B 仅当 Probe A 的 SSH 建立但 marker 缺失时使用；A/A2 均在 TCP 层失败，同一主机任何端口/用户均不可能建立 SSH，故按序跳过并在此记录理由。

### Probe C — 直连 oseasy@172.25.14.221:22（项目密钥）

- TCP: Test-NetConnection 172.25.14.221:22 -> TcpTestSucceeded=True。
- SSH marker 探测原始输出（2026-08-16 约 03:35Z）：

      hostname: i-00000226
      id: 用户id=1000(oseasy) 组id=1000(oseasy) 组=1000(oseasy),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),120(lpadmin),131(lxd),132(sambashare)
      WORKTREE_OK（/home/oseasy/git_work/wt_d3q_mason_91a75e5 存在）
      Python 3.10.20（/home/oseasy/venvs/skill_preflight_e0e1/bin/python）
      ENV_OK（/home/oseasy/.config/dicode/experiment_llm.env 存在）
      GPU 快照：GPU0 15573 MiB（Ollama llama-server x2，保持不动）；GPU1 1 MiB；GPU2 1 MiB（UUID GPU-8df11537-ab79-722d-606f-411966196c4c，与冻结契约一致，无外部 compute 进程）；GPU3 1 MiB。

- 结论：全部 marker 通过。

## 胜出路由（Phase-2 全部远程操作使用）

    ssh -i D:/Projects/dicode-codex-director/orchestration/control/ssh_oseasy_172_25_14_221_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes oseasy@172.25.14.221 <command>

- 与 d3q_slot_launcher.py 默认常量完全一致（SSH_TARGET=oseasy@172.25.14.221，SSH_KEY=项目密钥），driver 调用 launcher 时无需覆盖默认值。
- known_hosts 已有 172.25.14.221 主机密钥；未改动用户 ssh config（其中 117.50.183.232 Port 23 条目已过时，按纪律不修改）。

## 安全边界确认

- EXP_DEEPSEEK_API_KEY 未读取、未输出、未哈希；仅确认 env 文件存在（ENV_OK）。
- 未触碰 GPU0/1/3；GPU2 通过 UUID 门禁核对。
- 失败 probe 均未建立 SSH 会话，无远端副作用。

## Re-probe 2026-08-16T04:06:44Z（用户再次指示使用代理端口）

- 指令：用户在 phase-2 恢复时要求"注意使用代理端口连接服务器"，故按冻结 Probe 顺序复测代理映射。
- Probe A 复测：Test-NetConnection 117.50.183.232:39467 -> TcpTestSucceeded=False；117.50.183.232:23 -> TcpTestSucceeded=False。两条代理映射仍在 TCP 层不可达（未建立 SSH，无远端副作用）。
- 处置：按冻结 Probe 纪律与 2026-08-16T03:52:59Z 记录的用户授权降级，继续使用直连路由 oseasy@172.25.14.221:22；若用户提供新的代理端口，将按序重新 probe 并切换。

## Re-probe 2026-08-16T06:27:35Z（preflight 启动前；用户再次指示使用代理端口）

- Probe A 复测：117.50.183.232:23 TCP ERROR（connection refused）；117.50.183.232:39467 TCP TIMEOUT。两条代理映射仍在 TCP 层不可达（SSH 未建立，无远端副作用）。
- 处置：按冻结 Probe 纪律与 2026-08-16T03:52:59Z 记录的用户授权降级，继续使用直连路由 oseasy@172.25.14.221:22 执行 preflight orchestrator；若用户提供新代理端口，将按序重新 probe 并切换。
- 直连 preflight 前置只读核查（2026-08-16T06:27:35Z）：GPU2 1 MiB / 0%（UUID GPU-8df11537-ab79-722d-606f-411966196c4c 与冻结契约一致，无外部 compute 进程）；GPU3 34606 MiB / 100%（perf48_async_pipeline_harness 组件 B，另一工作线，不触碰）；GPU0 Ollama llama-server x2（qwen2.5-coder:14b loaded，digest 9ec8897f747e246e970bc5cfdda85d22f1123dc2e3d34978a010a75968716849，保持不动）；WORKTREE_OK；frozen 输入六项齐全（archive_snapshot/candidate_codes/checkpoint/conditioning.npy/config.yaml/rng.npy）。
