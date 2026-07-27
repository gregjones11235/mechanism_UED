# Tier3 失败分类 (failure taxonomy)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1;模块 `tier3_failure_taxonomy.py`
- `failure_rule_version = tier3_failure_rules/v1`(记录在每次分类上)
- 状态: **PASS**(`tier3_failure_taxonomy.py --self-test` exit 0;NEG20 fail-closed)

## 1. 原则

- 每条 episode 恰好一个 terminal label;规则集显式、版本化。
- **歧义/矛盾终止信号 → fail-closed(NEG20)**,绝不静默选一个标签(静默误标会污染机制诊断)。

## 2. 场景标签集

| 场景 | 标签 |
|---|---|
| FULL | SUCCESS_DEFEAT_KOBOLD / DIED_BEFORE_KOBOLD / TIMEOUT_NO_KOBOLD / INVALID_START |
| FRONT | EXIT_REACHED / DIED_IN_CORRIDOR / TIMEOUT_EXIT_NOT_FOUND / INVALID_START |
| BACK | SUCCESS_DEFEAT_KOBOLD / DIED_AFTER_ENGAGEMENT / DIED_BEFORE_ENGAGEMENT / TIMEOUT_IN_BOSS_AREA / TIMEOUT_KOBOLD_NOT_FOUND / INVALID_START |

## 3. 矛盾检测(NEG20)

以下组合被判为不可静默调和,分类器直接 FailClosed:
- `defeat_kobold` 与 `player_died` 同为终止;
- `defeat_kobold` 与 `timed_out` 同为终止;
- `player_died` 与 `timed_out` 同为终止;
- FRONT 场景 `corridor_exit_reached` 与 `defeat_kobold` 同时出现(前段止于出口);
- 无任何终止信号(未 defeat/died/timed_out/exit)→ 同样 FailClosed。

`INVALID_START` 为排他性终止(优先于其它规则)。
