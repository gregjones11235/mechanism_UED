# -*- coding: utf-8 -*-
"""E3 actual-N TASK-BASED success/progress predicate tests (audit 2026-08-10).

Covers the sole-controller directive item 1:
  * NO reading of non-existent EnvState fields (gate_progress, floor_number,
    health, max_health) — only the real fields player_level / player_health /
    achievements / timestep.
  * success / progress derive from the concrete Task class's
    is_success() / relevant_achievements (all relevant achievements done).
  * predicate applicability verified with one positive + one negative example.
  * missing interface / missing fields FAIL CLOSED (never a silent 0).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from minicraftax.tasks.seed_tasks import survive

import e3_capsule_actualn as cap


def _real_state():
    sp = StaticEnvParams()
    ep = EnvParams(max_timesteps=100)
    return survive.Env(sp, ep).generate_world(jax.random.PRNGKey(0))


def _set_relevant(state, task, all_done):
    idx = [int(a.value) for a in task.relevant_achievements]
    ach = np.asarray(state.achievements).copy()
    for i in idx:
        ach[i] = bool(all_done)
    return dataclasses.replace(state, achievements=jnp.asarray(ach))


def test_success_predicate_all_relevant_achievements():
    """success == all relevant achievements done (task.is_success)."""
    task = survive.Env(StaticEnvParams(), EnvParams(max_timesteps=100))
    pred, meta = cap.build_task_success_predicate(task)
    assert meta["predicate_kind"] == "TASK_ACHIEVEMENT_ALL_RELEVANT_DONE"
    assert meta["achievement_indices"] == [int(a.value) for a in task.relevant_achievements]
    state = _real_state()
    assert bool(pred(state)) is False  # nothing done at start
    assert bool(pred(_set_relevant(state, task, True))) is True
    assert bool(pred(_set_relevant(state, task, False))) is False


def test_progress_is_fraction_of_relevant_achievements():
    task = survive.Env(StaticEnvParams(), EnvParams(max_timesteps=100))
    prog = cap.build_task_progress_fn(task)
    idx = [int(a.value) for a in task.relevant_achievements]
    n = float(len(idx))
    state = _real_state()
    assert prog(state) == 0.0
    # start from all-relevant-False, then set exactly ONE relevant done.
    done_one = _set_relevant(state, task, False)
    ach = np.asarray(done_one.achievements).copy()
    ach[idx[0]] = True
    done_one = dataclasses.replace(done_one, achievements=jnp.asarray(ach))
    assert prog(done_one) == pytest.approx(1.0 / n)


def test_predicate_applicability_positive_and_negative():
    """Applicability: construct ONE positive and ONE negative example and
    require the predicate to distinguish them."""
    task = survive.Env(StaticEnvParams(), EnvParams(max_timesteps=100))
    pred, meta = cap.build_task_success_predicate(task)
    state = _real_state()
    app = cap.verify_predicate_applicability(
        state, pred, meta["achievement_indices"],
        getattr(task, "__name__", "survive"))
    assert app["applicable"] is True
    assert app["positive_example_success"] is True
    assert app["negative_example_not_success"] is True


def test_no_fake_gate_progress_field():
    """The real EnvState has NO gate_progress / floor_number / health /
    max_health.  Reading them must fail closed (never a silent 0)."""
    state = _real_state()
    for fake in ("gate_progress", "floor_number", "health", "max_health"):
        assert not hasattr(state, fake), \
            f"real EnvState unexpectedly exposes {fake}"
    # build_state_facts must read only real fields and raise on missing ones.
    task = survive.Env(StaticEnvParams(), EnvParams(max_timesteps=100))
    idx = [int(a.value) for a in task.relevant_achievements]
    facts = cap.build_state_facts(state, idx, "survive")
    assert "player_level" in facts and "player_health" in facts
    assert "timestep" in facts
    assert "gate_progress" not in facts and "floor_number" not in facts


def test_missing_interface_fails_closed():
    """A task without is_success / relevant_achievements cannot build a
    predicate — hard error, no default."""
    class FakeTask:
        pass
    with pytest.raises(RuntimeError):
        cap.build_task_success_predicate(FakeTask)


def test_empty_relevant_achievements_fails_closed():
    class NoRelevant:
        def is_success(self, state):
            return True
        relevant_achievements = []
    with pytest.raises(RuntimeError):
        cap.build_task_success_predicate(NoRelevant)
