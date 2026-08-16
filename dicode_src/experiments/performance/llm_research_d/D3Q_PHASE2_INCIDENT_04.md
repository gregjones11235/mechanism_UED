# D3Q Phase-2 Incident 04 — ssh drop orphaned driver; cleanup destroyed live exec root

classification: D3Q_PHASE2_INCIDENT_04
schema_version: 1
recorded_utc: 2026-08-16T06:58:00Z
severity: blocker_resolved

## Symptom

Preflight attempt `d3q_preflight_20260816T063657Z` returned BLOCKED
`remote_summary_missing` after only ~12 minutes. Evidence collected locally:
`remote_run_rc.txt` = 4294967295 (rc -1), `remote_run_stderr.txt` =
"Connection to 172.25.14.221 closed by remote host.", empty stdout.

## Root cause chain

1. Orchestrator ran the whole remote driver under ONE blocking ssh exec.
2. At ~06:47Z the sshd connection was closed server-side (transient; auth.log
   shows only normal session open/close, no OOM/restart/sshd crash; uptime 34
   days; root cause of the drop not determinable from available logs).
3. ssh exited rc -1. The remote driver (PID 3394632) and its replay child
   (PID 3394678, GPU2 34.5 GiB) SURVIVED the disconnect as orphans and kept
   running.
4. Orchestrator proceeded: collected a PARTIAL tar (large_r1 spec/manifest/
   partial run only), failed the summary check, then its unconditional
   `finally: rm -rf /tmp/d3q_preflight_20260816T063657Z` deleted the exec
   root WHILE THE ORPHANED DRIVER WAS STILL RUNNING.

Design flaw: long-running work over a single blocking ssh + unconditional
cleanup.

## Impact

- One preflight attempt lost (~12 min GPU2 time). No evidence corruption:
  generation ledger, staging, frozen inputs untouched. Partial collection
  retained in `d3q_artifacts/d3q_preflight_20260816T063657Z` (untracked).
- Orphaned processes killed with SIGTERM (verified gone; GPU2 back to 1 MiB).

## Fix (implemented, tested)

`d3q_preflight_orchestrator.py` hardened:

1. DETACHED launch: `setsid bash -c '<driver> > stdout 2> stderr; echo $? >
   driver.rc'` — returns immediately; ssh drops can no longer kill or blind
   the run.
2. POLLING: every 60s a short ssh probe reads `driver.rc` (DONE rc=N), or
   pgrep-liveness (RUNNING / DEAD); DEAD collects forensics then fails
   closed; >=10 consecutive probe failures fail closed; all probes logged to
   `poll_log.txt`; 6h overall budget.
3. GUARDED cleanup: before any `rm -rf`, pgrep (bracket-trick patterns) must
   show zero live driver/replay processes for this exec root; otherwise
   `cleanup_skipped_live_processes` is raised and the exec root preserved.
4. pgrep patterns use `[.]` so probe shells never self-match.

Tests: `test_classify_poll_states`, `test_poll_probe_self_match_guard`
(17 passed in the preflight+finalize suites).

## Disposition

Relaunch preflight with hardened orchestrator; fresh timestamped out dir.
