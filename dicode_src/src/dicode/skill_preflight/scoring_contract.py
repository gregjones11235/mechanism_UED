"""B3: preflight scoring-payload compaction contract (pure logic, no JAX).

The preflight scoring path transfers ``scoring_window_data`` from GPU to CPU and
then runs the numpy "smart calculator". Not every score function reads every
field, so ``make_eval`` (ppo_tr.py) can trim the payload at its return site
(keeping a single outer scan) when ``performance.compact_preflight_payload`` is
on.

Audited access contract (scoring.py):

- ``_calculate_scores_from_snapshot_impl`` reads, for ALL score functions:
    traj_batch.info["task_id"], ["returned_episode"], ["is_success"],
    ["returned_episode_lengths"], ["returned_episode_returns"],
    67 x info["Achievements/<name>"], traj_batch.reward, traj_batch.value
  (reward/value are read but only dereferenced by the max_mc branch).
- ``_calculate_priority_scores``:
    learnability -> only sr from base metrics (info above);
    pvl          -> returned_episode + advantages + task_id;
    max_mc       -> returned_episode + reward + value + task_id.
- ``traj_batch.done`` is never read by any scoring path.

Hence per score function we may drop:
    learnability: reward, value, done, advantages  (info kept in full, with
                  Achievements/* kept conservatively)
    pvl:          reward, value, done              (keep advantages)
    max_mc:       done, advantages                 (keep reward, value)
Unknown score functions raise immediately (no silent fallback).
"""

from __future__ import annotations

from typing import Any, Iterable

from dicode.skill_preflight.contract import PreflightOptimizationContractError

# info keys read unconditionally by _calculate_base_metrics / snapshot impl.
SCORING_INFO_REQUIRED_KEYS = frozenset({
    "task_id", "returned_episode", "is_success",
    "returned_episode_lengths", "returned_episode_returns",
})

# Transition fields that scoring may drop per score function.
_TRANSITION_FIELDS = ("reward", "value", "done")


def scoring_info_keep_keys(info_keys: Iterable[str]) -> list[str]:
    """Subset of info keys to keep: the required keys + all Achievements/*.

    Achievements/* are kept conservatively (the base-metric achievement loop
    still reads all 67). Returns a deterministic (sorted) list.
    """
    return sorted(
        k for k in info_keys
        if k in SCORING_INFO_REQUIRED_KEYS or k.startswith("Achievements/")
    )


def compact_field_decisions(score_function: str) -> dict[str, Any]:
    """Per-score-function compaction decisions.

    Returns a dict with keys: keep_advantages, keep_reward, keep_value,
    keep_done, trim_info. Raises ValueError for unknown score functions.
    """
    if score_function == "learnability":
        return {"keep_advantages": False, "keep_reward": False,
                "keep_value": False, "keep_done": False, "trim_info": True}
    if score_function == "pvl":
        return {"keep_advantages": True, "keep_reward": False,
                "keep_value": False, "keep_done": False, "trim_info": True}
    if score_function == "max_mc":
        return {"keep_advantages": False, "keep_reward": True,
                "keep_value": True, "keep_done": False, "trim_info": True}
    raise PreflightOptimizationContractError(
        f"Unknown score_function for compact_preflight_payload: {score_function}")
