"""B3 local tests: compact_preflight_payload scoring contract (pure logic).

The decision helper (skill_preflight.scoring_contract) has no JAX dependency and
is fully tested here. A server-only test verifies that scoring a compacted
payload (keeping exactly the audited fields) produces output identical to the
full payload (# requires-jax-server, importorskip'd on the CPU box).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

DICODE = Path(__file__).parents[2]
PPO_TR = DICODE / "ppo_tr.py"
CONF = Path(__file__).parents[4] / "conf" / "config.yaml"

REQUIRED = {"task_id", "returned_episode", "is_success",
            "returned_episode_lengths", "returned_episode_returns"}


def _contract():
    from dicode.skill_preflight.scoring_contract import (
        compact_field_decisions, scoring_info_keep_keys, SCORING_INFO_REQUIRED_KEYS)
    return compact_field_decisions, scoring_info_keep_keys, SCORING_INFO_REQUIRED_KEYS


def test_required_keys_contract():
    _, _, required = _contract()
    assert required == REQUIRED


def test_scoring_info_keep_keys():
    _, keep_keys, _ = _contract()
    keys = ["task_id", "returned_episode", "is_success", "returned_episode_lengths",
            "returned_episode_returns", "Achievements/wood", "Achievements/stone",
            "discount", "other"]
    kept = keep_keys(keys)
    assert set(kept) == REQUIRED | {"Achievements/wood", "Achievements/stone"}
    assert "discount" not in kept and "other" not in kept
    assert kept == sorted(kept)


def test_compact_field_decisions_learnability():
    decisions, _, _ = _contract()
    d = decisions("learnability")
    assert d == {"keep_advantages": False, "keep_reward": False,
                 "keep_value": False, "keep_done": False, "trim_info": True}


def test_compact_field_decisions_pvl():
    decisions, _, _ = _contract()
    d = decisions("pvl")
    assert d["keep_advantages"] is True
    assert d["keep_reward"] is False and d["keep_value"] is False


def test_compact_field_decisions_max_mc():
    decisions, _, _ = _contract()
    d = decisions("max_mc")
    assert d["keep_reward"] is True and d["keep_value"] is True
    assert d["keep_advantages"] is False


def test_compact_field_decisions_unknown_raises():
    decisions, _, _ = _contract()
    with pytest.raises(ValueError):
        decisions("bogus")


def test_b3_config_flag_default_off():
    text = CONF.read_text(encoding="utf-8")
    assert "compact_preflight_payload: false" in text


def test_b3_wiring_source_audit():
    ppo = PPO_TR.read_text(encoding="utf-8")
    assert "_compact_eval_scoring_data(" in ppo
    assert "compact_preflight_payload" in ppo
    assert 'config.dicode_manager.score_function' in ppo
    # flag-off path leaves the payload untouched (historical)
    assert "if _perf_b3.get(\"compact_preflight_payload\", False):" in ppo


# ---------------------------------------------------------------------------
# requires-jax-server: field-access contract with pure-numpy payloads.
# ---------------------------------------------------------------------------
def _build_payload(seed=0, T=6, B=2, num_tasks=2):
    import numpy as np
    rng = np.random.default_rng(seed)
    info = {
        "task_id": rng.integers(0, num_tasks, (T, B)),
        "returned_episode": rng.integers(0, 2, (T, B)).astype(bool),
        "is_success": rng.integers(0, 2, (T, B)).astype(bool),
        "returned_episode_lengths": rng.integers(1, 100, (T, B)),
        "returned_episode_returns": rng.random((T, B)),
        "discount": rng.random((T, B)),          # NOT read by scoring -> dropped
    }
    # guarantee every task has >= 1 finished episode (scoring divides by count)
    for t in range(T):
        info["task_id"][t, 0] = t % num_tasks
        info["returned_episode"][t, 0] = True
    traj = SimpleNamespace(info=info, reward=rng.random((T, B)),
                           value=rng.random((T, B)),
                           done=rng.integers(0, 2, (T, B)).astype(bool))
    return {"traj_batch": traj, "advantages": rng.random((T, B))}


@pytest.mark.parametrize("score_function", ["learnability", "pvl", "max_mc"])
def test_compacted_payload_scoring_output_identical(score_function):
    scoring = pytest.importorskip("dicode.scoring")
    from dicode.skill_preflight.scoring_contract import (
        compact_field_decisions, scoring_info_keep_keys)
    import numpy as np

    T, B, num_tasks = 6, 2, 2
    full = _build_payload(T=T, B=B, num_tasks=num_tasks)
    # add the 67 Achievements/* keys scoring reads
    for name in scoring.get_achievement_names():
        full["traj_batch"].info[f"Achievements/{name}"] = np.zeros((T, B), dtype=np.float32)

    cfg = SimpleNamespace(dicode_manager=SimpleNamespace(
        score_function=score_function, mode="reward"))
    ach_mask = np.zeros((num_tasks, 67), dtype=bool)
    completed = np.zeros((num_tasks, 67), dtype=bool)

    full_out = scoring._calculate_scores_from_snapshot_impl(
        full, num_tasks, ach_mask, completed, cfg)

    # build the compacted payload using the same contract decisions the
    # ppo_tr make_eval return site applies.
    decisions = compact_field_decisions(score_function)
    info = full["traj_batch"].info
    keep = scoring_info_keep_keys(info.keys())
    traj = full["traj_batch"]
    compacted = {
        "traj_batch": SimpleNamespace(
            info={k: info[k] for k in keep},
            reward=traj.reward if decisions["keep_reward"] else None,
            value=traj.value if decisions["keep_value"] else None,
            done=traj.done if decisions["keep_done"] else None,
        ),
        "advantages": full["advantages"] if decisions["keep_advantages"] else None,
    }
    compact_out = scoring._calculate_scores_from_snapshot_impl(
        compacted, num_tasks, ach_mask, completed, cfg)

    assert compact_out == full_out
