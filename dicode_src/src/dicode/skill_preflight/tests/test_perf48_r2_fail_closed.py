"""R2: B2/B3 enabled preflight optimizations must fail closed.

The preflight gate in run_dicode.py historically degraded ANY exception to
"kept all, gate inactive". R2 makes an enabled preflight-optimization contract
violation (PreflightOptimizationContractError) propagate and terminate the run,
while ordinary preflight errors keep the historical degradation.

The classification lives in ``skill_preflight.contract.handle_preflight_gate_error``
(no jax dependency) and is the exact function run_dicode's gate catch calls; a
source-audit test proves the wiring. These tests run on the CPU-only box.
"""
import importlib.util
from pathlib import Path

import pytest

from dicode.skill_preflight.contract import (
    PreflightOptimizationContractError,
    handle_preflight_gate_error,
)
from dicode.skill_preflight.reuse_loaded_tasks import resolve_preloaded_tasks
from dicode.skill_preflight.scoring_contract import (
    compact_field_decisions,
    scoring_info_keep_keys,
)

RUN_DICODE = Path(__file__).parents[4] / "experiments" / "training" / "run_dicode.py"


class _Cfg:
    def __init__(self, reuse=False, compact=False):
        self._reuse = reuse
        self._compact = compact

    def get(self, key, default=None):
        if key == "performance":
            return {"preflight_reuse_loaded_tasks": self._reuse,
                    "compact_preflight_payload": self._compact}
        return default


# ---- Outer gate classification (the production handler) ----

def test_gate_handler_propagates_contract_error():
    with pytest.raises(PreflightOptimizationContractError):
        handle_preflight_gate_error(PreflightOptimizationContractError("b2 ids mismatch"))


def test_gate_handler_degrades_normal_error(capsys):
    r = handle_preflight_gate_error(RuntimeError("some ordinary preflight error"))
    assert r is None
    out = capsys.readouterr().out
    assert "kept all, gate inactive" in out


def test_gate_handler_does_not_string_match():
    # a normal RuntimeError whose message mentions "ids" must NOT be re-raised
    r = handle_preflight_gate_error(RuntimeError("ids order mismatch inside old code"))
    assert r is None


# ---- B2 contract ----

def test_b2_flag_off_returns_none_no_exception():
    classes, reuse = resolve_preloaded_tasks(_Cfg(reuse=False), ["t1"], None, None)
    assert reuse is False and classes is None


def test_b2_missing_preloaded_raises_contract():
    with pytest.raises(PreflightOptimizationContractError):
        resolve_preloaded_tasks(_Cfg(reuse=True), ["t1"], None, None)
    with pytest.raises(PreflightOptimizationContractError):
        resolve_preloaded_tasks(_Cfg(reuse=True), ["t1"], [object()], None)
    with pytest.raises(PreflightOptimizationContractError):
        resolve_preloaded_tasks(_Cfg(reuse=True), ["t1"], None, ["t1"])


def test_b2_id_order_mismatch_raises_contract():
    with pytest.raises(PreflightOptimizationContractError):
        resolve_preloaded_tasks(_Cfg(reuse=True), ["t1", "t2"], [object(), object()], ["t2", "t1"])


def test_b2_count_mismatch_raises_contract():
    with pytest.raises(PreflightOptimizationContractError):
        resolve_preloaded_tasks(_Cfg(reuse=True), ["t1", "t2"], [object()], ["t1", "t2"])


def test_b2_valid_preload_reused():
    cls = [object(), object()]
    classes, reuse = resolve_preloaded_tasks(_Cfg(reuse=True), ["t1", "t2"], cls, ["t1", "t2"])
    assert reuse is True and classes is cls


# ---- B3 contract ----

def test_b3_known_score_functions_ok():
    assert compact_field_decisions("learnability")["keep_advantages"] is False
    assert compact_field_decisions("pvl")["keep_advantages"] is True
    assert compact_field_decisions("max_mc")["keep_reward"] is True


def test_b3_unknown_score_function_raises_contract():
    with pytest.raises(PreflightOptimizationContractError):
        compact_field_decisions("unknown_sf")


def test_b3_scoring_info_keep_keys():
    info = {"task_id": 1, "returned_episode": 2, "is_success": 3,
            "returned_episode_lengths": 4, "returned_episode_returns": 5,
            "Achievements/foo": 6, "other": 7}
    keep = scoring_info_keep_keys(info.keys())
    assert set(keep) == {"task_id", "returned_episode", "is_success",
                         "returned_episode_lengths", "returned_episode_returns",
                         "Achievements/foo"}


# ---- Production wiring (source audit) ----

def test_run_dicode_gate_uses_fail_closed_handler():
    src = RUN_DICODE.read_text(encoding="utf-8")
    assert "handle_preflight_gate_error(e)" in src
    # the preflight gate's outer catch is the LAST `except Exception as e:` in the file
    tail = src.rsplit("except Exception as e:", 1)[1]
    assert "handle_preflight_gate_error(e)" in tail


def test_run_dicode_prevalidates_compact_before_rollout():
    src = RUN_DICODE.read_text(encoding="utf-8")
    # compact pre-validation must appear before the evaluate_new_tasks call
    idx_preval = src.index("compact_field_decisions(config.dicode_manager.score_function)")
    idx_eval = src.index("_pf_raw = evaluate_new_tasks(")
    assert idx_preval < idx_eval


def test_run_dicode_imports_contract_error():
    src = RUN_DICODE.read_text(encoding="utf-8")
    assert "PreflightOptimizationContractError" in src
