# GPU0 Final Training Review

Timestamp: 2026-07-13T22:48:00+08:00

## Outcome

The approved GPU0 DeepSeek-V4-Pro substitution run reached 15,300 global PPO updates and 2,005,401,600 environment steps. Orbax finalized checkpoint `15300` with 16 files and approximately 46.5 MB of checkpoint data. The process then completed a final code-generation batch and exited while the log ended at `Validating 12 generated tasks...`; there is no explicit overall-completion marker. The training target and recoverable checkpoint are complete, while terminal post-training validation status is incomplete.

## Identity and protocol

- Run: `dspro-gpu0-g3-seed0-20260712T041233Z`
- Branch/commit: `exp/dicode-dspro` / `88bec3aad1bd7343bf36c9ccced17d050281ba0c`
- Seed: 0
- Requested/configured model: `deepseek-v4-pro`
- Label: DiCode DeepSeek-V4-Pro substitution baseline
- Training: 1,024 workers x 128 steps x 15,300 updates = 2,005,401,600 environment steps
- Evaluation: 1,024 environments
- Observed curriculum sessions: 151
- Successful DeepSeek chat-completion HTTP responses: 1,824

This is not an exact original-model reproduction: the DiCode paper used Qwen3-235B and reported five-seed results, whereas this run uses DeepSeek-V4-Pro and one seed.

## Final target-environment metrics

| Metric | GPU0 seed 0 |
|---|---:|
| Mean episode return | 46.3288 |
| Mean return percentage | 20.4995% |
| Average episode length | 1,806.98 |
| Make iron sword | 63.61% |
| Make iron armour | 25.85% |
| Make diamond sword | 6.80% |
| Defeat gnome warrior | 0.00% |
| Defeat gnome archer | 0.00% |
| Enter dungeon | 96.43% |
| Enter gnomish mines | 26.02% |
| Collect iron | 82.65% |
| Collect diamond | 30.95% |
| Collect ruby | 26.19% |
| Collect sapphire | 26.19% |

The logged task-level `sr` is 0.0 and must not be confused with mean return or per-achievement success rates.

## Paper correspondence

The paper trains for 2 billion environment steps across five seeds with Qwen3-235B and evaluates on a fixed held-out set of 1,024 worlds. It reports DiCode mean return 48.33, PPO-GTrXL 41.54, Make Iron Armour 45%, Make Diamond Sword 6%, Defeat Gnome Warrior 11%, and Defeat Gnome Archer 9%.

| Paper-comparable metric | GPU0 | Paper DiCode | Difference |
|---|---:|---:|---:|
| Mean episode return | 46.33 | 48.33 | -2.00 (-4.14%) |
| Make iron armour | 25.85% | 45% | -19.15 pp |
| Make diamond sword | 6.80% | 6% | +0.80 pp |
| Defeat gnome warrior | 0% | 11% | -11 pp |
| Defeat gnome archer | 0% | 9% | -9 pp |

The GPU0 point estimate is 4.79 return above the paper's PPO-GTrXL point estimate (about 11.5%), but a single substituted seed cannot establish statistical superiority or reproduce the paper's 48.33 +/- 0.63 five-seed result.

## Integrity and limitations

- Final checkpoint save and finalize are explicitly logged.
- No OOM, killed-process, or residual training process was observed.
- No traceback/error appears after the final 15,300-update record.
- Two `nan` tokens after that record are `lp` diagnostic values, not evidence of a NaN training crash.
- The long-lived log contains historical recovered exceptions, so it is not a clean zero-error transcript.
- The original manifest remained marked `RUNNING` and reports stale API counters; final counts above are derived from the completed log.
- The process lacks a final overall-completion marker and appears to exit during post-target generated-task validation.
- Exported text artifacts passed a potential-secret scan with zero matches.

## Export

Export directory: `orchestration/manifests/gpu0_final_export_20260713T224800+0800`

The export contains the final checkpoint, run/config manifests, training logs, task graph, runtime timing artifacts, SHA256 inventory, and machine-readable `final_result.json`.

PASS_WITH_MINOR_ISSUES
