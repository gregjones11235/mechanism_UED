"""Production worker-to-main integration coverage for validation_cache."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import threading

import pytest

jax = pytest.importorskip("jax")

from dicode.evolution_efficient import (
    evolve_and_validate_tasks,
    run_validation_on_main_thread,
)


class _Archive:
    def update_node_reasoning(self, *args):
        pass

    def update_node_status(self, *args):
        pass

    def set_task_active_status(self, *args):
        pass

    def update_node_priority_score(self, *args):
        pass


def _env_generator(cache_enabled):
    mod = pytest.importorskip("dicode.dreaming.gen_manager")
    obj = mod.EnvGenerator.__new__(mod.EnvGenerator)
    obj.performance = {
        "validation_cache": cache_enabled,
        "validation_cache_max_entries": 8,
        "validation_static_lint": False,
    }
    obj._validation_cache = OrderedDict()
    obj._validation_cache_lock = threading.RLock()
    obj._validation_inflight = {}
    obj._validation_source_sha = "source-v1"
    calls = []

    def compile_once(code):
        calls.append(code)
        return (False, "bad") if code == "bad-code" else (True, "")

    obj._check_compilation_uncached = compile_once
    return obj, calls


def _run_worker_then_main(cache_enabled):
    env_generator, calls = _env_generator(cache_enabled)
    generated = [
        {"generated_task_id": "ok", "code_string": "ok-code", "reasoning": None},
        {"generated_task_id": "bad", "code_string": "bad-code", "reasoning": None},
    ]
    manager = SimpleNamespace(
        evolve_tasks=lambda tasks, metrics: generated,
        env_generator=env_generator,
        config_={},
        archive=_Archive(),
    )
    worker_results = evolve_and_validate_tasks(manager, {}, {}, 1)
    run_validation_on_main_thread(
        SimpleNamespace(), jax.random.PRNGKey(0), None, manager, worker_results
    )
    return env_generator, calls


def test_cache_on_validates_each_unique_success_and_failure_once_across_worker_main():
    _env, calls = _run_worker_then_main(True)
    assert calls.count("ok-code") == 1
    assert calls.count("bad-code") == 1


def test_cache_off_preserves_two_uncached_validations_per_code():
    _env, calls = _run_worker_then_main(False)
    assert calls.count("ok-code") == 2
    assert calls.count("bad-code") == 2


def test_validation_key_change_misses_after_cached_success_and_failure():
    env, calls = _run_worker_then_main(True)
    env._validation_source_sha = "source-v2"
    assert env.check_compilation("ok-code") == (True, "")
    assert env.check_compilation("bad-code") == (False, "bad")
    assert calls.count("ok-code") == 2
    assert calls.count("bad-code") == 2
