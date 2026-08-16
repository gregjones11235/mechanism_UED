# D3Q Phase-2 Incident 03 — root disk full blocked preflight deploy

classification: D3Q_PHASE2_INCIDENT_03
schema_version: 1
recorded_utc: 2026-08-16T06:33:00Z
severity: blocker_resolved

## Symptom

Preflight orchestrator attempt `d3q_preflight_20260816T062840Z` (local out dir) failed within ~3s at the remote deploy gate:

```json
{"cmd": ["test ! -e /tmp/d3q_preflight_20260816T062840Z && mkdir -p /tmp/d3q_preflight_20260816T062840Z"], "rc": 1, "stderr_tail": "mkdir: 无法创建目录“/tmp/d3q_preflight_20260816T062840Z”: 设备上已经没有空间", "status": "BLOCKED"}
```

## Root cause

Server root partition `/dev/sda2` (196G) was 100% used, 0 bytes available. `/tmp` resides on `/`, so the mkdir failed. This is a system-wide disk-full condition, not a /tmp-specific leak:

- /home/oseasy = 152G (experiments 64G, 下载 34G, miniconda3 18G, cc4_tier3_eval 8.6G, venvs 5.9G, 桌面 5.5G, .vscode-server 5.3G, .cache 3.8G, ...)
- No d3q leftovers in /tmp (previous orchestrator cleanups worked).
- Frozen inputs live under /home/oseasy/e2_data_disk2 -> /media/数据磁盘2 (separate disk, 221G free) — unaffected.

## Impact

- One preflight launch blocked at deploy gate; no remote side effects beyond the failed mkdir; no candidate, ledger, or generation evidence touched.
- Disk-full is a standing risk to all concurrent workstreams (any write on / can fail).

## Mitigation (executed)

- `python -m pip cache purge` (dicode310 env): removed 994 files / 3765.3 MB from `~/.cache/pip` — a purely regenerable download cache; no installed packages, checkpoints, evidence, or other workstream data touched.
- Root partition available space after purge: 3.5G.
- Larger directories (experiments/*, 下载, etc.) belong to other workstreams/user data and were deliberately NOT touched; user follow-up recommended for durable cleanup.

## Disposition

- Relaunch preflight orchestrator with a fresh timestamped out dir (never reuse `20260816T062840Z`).
- Failed-attempt out dir (if created locally) retained as evidence, untracked.
