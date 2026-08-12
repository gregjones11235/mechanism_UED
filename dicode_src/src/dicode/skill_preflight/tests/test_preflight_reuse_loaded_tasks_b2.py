"""B2 local tests: preflight_reuse_loaded_tasks contract (pure logic).

The decision helper (skill_preflight.reuse_loaded_tasks) has no JAX dependency
and is fully tested here on the CPU-only box. The wiring tests against
evaluate_new_tasks need jax/craftax and are marked # requires-jax-server.
"""
from pathlib import Path

import pytest
from dicode.skill_preflight.contract import PreflightOptimizationContractError

DICODE = Path(__file__).parents[2]
RUN_DICODE = Path(__file__).parents[4] / "experiments" / "training" / "run_dicode.py"
CONF = Path(__file__).parents[4] / "conf" / "config.yaml"


class _Cfg:
    def __init__(self, reuse):
        self._reuse = reuse

    def get(self, key, default=None):
        if key == "performance":
            return {"preflight_reuse_loaded_tasks": self._reuse}
        return default


def _helper():
    from dicode.skill_preflight.reuse_loaded_tasks import resolve_preloaded_tasks
    return resolve_preloaded_tasks


def test_reuse_off_ignores_preloaded_and_returns_none():
    resolve = _helper()
    classes, reuse = resolve(_Cfg(False), ["a", "b"], ["X"], ["a", "b"])
    assert reuse is False and classes is None


def test_reuse_on_reuses_matching_preloaded():
    resolve = _helper()
    classes, reuse = resolve(_Cfg(True), ["a", "b"], ["cls_a", "cls_b"], ["a", "b"])
    assert reuse is True and classes == ["cls_a", "cls_b"]


@pytest.mark.parametrize("ids", [["b", "a"], ["a"], ["a", "b", "c"], ["A", "b"]])
def test_reuse_on_mismatched_ids_raise(ids):
    resolve = _helper()
    with pytest.raises(PreflightOptimizationContractError):
        resolve(_Cfg(True), ["a", "b"], ["cls_a", "cls_b"], ids)


def test_reuse_on_count_mismatch_raise():
    resolve = _helper()
    with pytest.raises(PreflightOptimizationContractError):
        resolve(_Cfg(True), ["a", "b"], ["cls_a"], ["a", "b"])


def test_reuse_on_missing_preloaded_raises():
    resolve = _helper()
    with pytest.raises(PreflightOptimizationContractError):
        resolve(_Cfg(True), ["a", "b"], None, None)
    with pytest.raises(PreflightOptimizationContractError):
        resolve(_Cfg(True), ["a", "b"], ["cls_a"], None)
    with pytest.raises(PreflightOptimizationContractError):
        resolve(_Cfg(True), ["a", "b"], None, ["a", "b"])


def test_b2_config_flag_default_off():
    text = CONF.read_text(encoding="utf-8")
    assert "preflight_reuse_loaded_tasks: false" in text


def test_b2_wiring_source_audit():
    online = (DICODE / "evaluation" / "online_evaluation.py").read_text(encoding="utf-8")
    run = RUN_DICODE.read_text(encoding="utf-8")
    # evaluate_new_tasks exposes the two all-or-nothing params
    assert "preloaded_task_classes: list = None" in online
    assert "preloaded_task_ids: list[str] = None" in online
    # and gates the second load on the helper
    assert "resolve_preloaded_tasks" in online
    assert "preflight_task_reload" in online
    # run_dicode passes the first-load objects only when the flag is on
    assert 'preloaded_task_classes=(_pf_classes if _reuse_tasks else None)' in run
    assert 'preloaded_task_ids=(_pf_ok_ids if _reuse_tasks else None)' in run
    assert "preflight_reuse_loaded_tasks" in run


def test_b2_no_silent_fallback_semantics():
    """Flag on + objects provided means the second load is SKIPPED entirely."""
    online = (DICODE / "evaluation" / "online_evaluation.py").read_text(encoding="utf-8")
    # When _use_preload, task_classes comes from preloaded and the span-wrapped
    # load_tasks_from_env_codes call is in the else branch only.
    assert "if _use_preload:" in online
    assert "task_classes = preloaded" in online
    assert "load_tasks_from_env_codes(archive, new_task_ids)" in online
