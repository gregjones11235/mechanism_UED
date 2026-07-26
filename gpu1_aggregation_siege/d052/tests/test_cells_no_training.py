"""GATE 12 — NO_UNAUTHORIZED_TRAINING and D052_LONG_TRAINING_RUNS=0.

A no-training authorization is STRUCTURALLY incapable of running timesteps: the
registry forces the no-op runner regardless of what is supplied, and any non-zero
timesteps would FAIL the cell. validate/prepare/status never launch. An absolute
invariant is asserted across the whole registry: every launch runs 0 timesteps this
phase. A training-scope authorization (which this phase never issues) is shown to
select the supplied runner -- proving the gate is about authorization, while the
test suite itself still runs 0 timesteps everywhere.
"""
import pytest

from d052.cells import (
    SCOPE_NO_TRAINING,
    SCOPE_TRAINING,
    CellError,
    CellRegistry,
    CellSpec,
    CellState,
    make_authorization,
)
from d052.schemas.selector import SelectorConfig, SelectorType

POOL_HASH = "a" * 64
SEL_HASH = "b" * 64


def _spec(cell_id, scope_steps=4096):
    return CellSpec(
        cell_id=cell_id, protocol_version="canonical_v2", hypothesis="h",
        pool_id="p", pool_hash=POOL_HASH,
        selector=SelectorConfig(selector=SelectorType.S1_THREE_ROLE, k=2, seed=7,
                                roles=["tutor", "critic", "explorer"]),
        candidate_ids=["t1", "t2"], selection_hash=SEL_HASH,
        intended_total_timesteps=scope_steps, output_dir=f"runs/{cell_id}",
        created_by="CC3")


def _to_authorized(reg, spec, scope, actor="CC3"):
    reg.register(spec, actor=actor)
    reg.validate_cell(spec.cell_id, actor=actor)
    reg.prepare(spec.cell_id, actor=actor)
    auth = make_authorization(spec.cell_id, spec.identity_hash(), "human", scope,
                              spec.intended_total_timesteps)
    reg.authorize(spec.cell_id, auth, actor=actor)


def test_no_training_scope_forces_no_op_even_with_supplied_runner(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec("c1")
    _to_authorized(reg, spec, SCOPE_NO_TRAINING)
    # a "malicious" runner that claims to run steps is IGNORED under no-training
    rec = reg.launch(spec.cell_id, actor="CC3",
                     runner=lambda r: {"timesteps_run": 100000, "trained": True})
    assert rec.state is CellState.COMPLETE
    assert rec.launch_manifest["timesteps_run"] == 0
    assert rec.launch_manifest["runner_artifact"]["trained"] is False
    assert "no-training phase" in rec.launch_manifest["runner_artifact"]["reason"]


def test_training_scope_selects_supplied_runner_but_zero_this_phase(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec("c2")
    _to_authorized(reg, spec, SCOPE_TRAINING)
    rec = reg.launch(spec.cell_id, actor="CC3",
                     runner=lambda r: {"timesteps_run": 0, "trained": False,
                                       "marker": "custom-runner-used"})
    # proves a training-scope auth uses the supplied runner (not forced no-op)
    assert rec.launch_manifest["runner_artifact"]["marker"] == "custom-runner-used"
    assert rec.launch_manifest["timesteps_run"] == 0   # still 0 this phase


def test_validate_prepare_status_never_launch(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec("c3")
    reg.register(spec, actor="CC3")
    reg.validate_cell(spec.cell_id, actor="CC3")
    reg.prepare(spec.cell_id, actor="CC3")
    reg.status(spec.cell_id)
    st = reg.status(spec.cell_id)
    assert st["state"] == "READY"
    assert st["launched"] is False
    assert st["timesteps_run"] == 0


def test_unauthorized_launch_refused_and_state_preserved(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec("c4")
    reg.register(spec, actor="CC3")
    reg.validate_cell(spec.cell_id, actor="CC3")
    reg.prepare(spec.cell_id, actor="CC3")   # READY, no authorization
    with pytest.raises(CellError) as ei:
        reg.launch(spec.cell_id, actor="CC3")
    assert ei.value.code == CellError.UNAUTHORIZED_LAUNCH
    assert reg.status(spec.cell_id)["state"] == "READY"


def test_absolute_zero_training_invariant_across_registry(tmp_path):
    reg = CellRegistry(str(tmp_path))
    # run several cells to completion (all no-training) + leave some unlaunched
    for cid in ("x1", "x2", "x3"):
        _to_authorized(reg, _spec(cid), SCOPE_NO_TRAINING)
        reg.launch(cid, actor="CC3")
    reg.register(_spec("x4"), actor="CC3")          # DRAFT, never launched
    total = 0
    for cid in reg.list_cells():
        total += reg.status(cid)["timesteps_run"]
    assert total == 0, "D052_LONG_TRAINING_RUNS must be 0 this phase"


def test_no_training_violation_would_fail_the_cell(tmp_path, monkeypatch):
    # Defence-in-depth: if the forced no-op runner were ever subverted to report
    # timesteps under a no-training authorization, the registry FAILs the cell
    # rather than complete it. Exercise by patching no_op_runner itself.
    from d052.cells import registry as reg_mod
    reg = CellRegistry(str(tmp_path))
    spec = _spec("x5")
    _to_authorized(reg, spec, SCOPE_NO_TRAINING)
    monkeypatch.setattr(reg_mod, "no_op_runner",
                        lambda r: {"timesteps_run": 5, "trained": True})
    with pytest.raises(CellError) as ei:
        reg.launch(spec.cell_id, actor="CC3")
    assert ei.value.code == CellError.NO_TRAINING_VIOLATION
    assert reg.status(spec.cell_id)["state"] == "FAILED"
