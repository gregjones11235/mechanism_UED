"""GATE 11 — cell lifecycle: DRAFT->VALIDATED->READY->AUTHORIZED->RUNNING->COMPLETE,
content-addressed identity, fail-closed illegal transitions, no-overwrite register,
validation->BLOCKED, and authorization binding."""
import pytest

from d052.cells import (
    SCOPE_NO_TRAINING,
    CellError,
    CellRegistry,
    CellSpec,
    CellState,
    assert_transition,
    can_transition,
    make_authorization,
)
from d052.schemas.selector import SelectorConfig, SelectorType

POOL_HASH = "a" * 64
SEL_HASH = "b" * 64


def _spec(cell_id="c1", **over):
    base = dict(
        cell_id=cell_id, protocol_version="canonical_v2", hypothesis="h",
        pool_id="p", pool_hash=POOL_HASH,
        selector=SelectorConfig(selector=SelectorType.S1_THREE_ROLE, k=2, seed=7,
                                roles=["tutor", "critic", "explorer"]),
        candidate_ids=["t1", "t2"], selection_hash=SEL_HASH,
        intended_total_timesteps=4096, output_dir="runs/c1", created_by="CC3")
    base.update(over)
    return CellSpec(**base)


def _advance_to_ready(reg, spec, actor="CC3"):
    reg.register(spec, actor=actor)
    reg.validate_cell(spec.cell_id, actor=actor)
    reg.prepare(spec.cell_id, actor=actor)


# --- happy path -------------------------------------------------------------

def test_full_lifecycle_to_complete(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec()
    _advance_to_ready(reg, spec)
    auth = make_authorization(spec.cell_id, spec.identity_hash(), "human",
                              SCOPE_NO_TRAINING, spec.intended_total_timesteps)
    reg.authorize(spec.cell_id, auth, actor="CC3")
    rec = reg.launch(spec.cell_id, actor="CC3")
    assert rec.state is CellState.COMPLETE
    states = [h.to_state for h in rec.history]
    assert states == [CellState.VALIDATED, CellState.READY, CellState.AUTHORIZED,
                      CellState.RUNNING, CellState.COMPLETE]


def test_status_is_read_only_and_reflects_state(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec()
    _advance_to_ready(reg, spec)
    s1 = reg.status(spec.cell_id)
    s2 = reg.status(spec.cell_id)
    assert s1 == s2
    assert s1["state"] == "READY"
    assert s1["launched"] is False


# --- content-addressed identity --------------------------------------------

def test_identity_hash_is_deterministic_and_content_bound():
    assert _spec().identity_hash() == _spec().identity_hash()
    other_k = _spec(selector=SelectorConfig(
        selector=SelectorType.S1_THREE_ROLE, k=3, seed=7,
        roles=["tutor", "critic", "explorer"]))
    assert _spec().identity_hash() != other_k.identity_hash()


def test_identity_changes_when_candidate_set_changes():
    assert _spec(candidate_ids=["t1", "t2"]).identity_hash() != \
        _spec(candidate_ids=["t1", "t3"]).identity_hash()


# --- no-overwrite register --------------------------------------------------

def test_register_refuses_existing_cell(tmp_path):
    reg = CellRegistry(str(tmp_path))
    reg.register(_spec(), actor="CC3")
    with pytest.raises(CellError) as ei:
        reg.register(_spec(), actor="CC3")
    assert ei.value.code == CellError.EXISTS_NO_OVERWRITE


def test_register_rejects_unsafe_cell_id(tmp_path):
    reg = CellRegistry(str(tmp_path))
    with pytest.raises(CellError) as ei:
        reg.register(_spec(cell_id="../evil"), actor="CC3")
    assert ei.value.code == CellError.INVALID_CELL_ID


# --- validation -> BLOCKED --------------------------------------------------

def test_validation_blocks_legacy_output_dir(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec(output_dir="audit_outputs/c1")
    reg.register(spec, actor="CC3")
    rec = reg.validate_cell(spec.cell_id, actor="CC3")
    assert rec.state is CellState.BLOCKED
    assert "NO_LEGACY_ARTIFACT_OVERWRITE" in rec.block_reason


def test_validation_blocks_wrong_environment(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec(environment_version="craftax==1.4.0")
    reg.register(spec, actor="CC3")
    rec = reg.validate_cell(spec.cell_id, actor="CC3")
    assert rec.state is CellState.BLOCKED


def test_blocked_can_return_to_draft():
    assert can_transition(CellState.BLOCKED, CellState.DRAFT) is True


# --- illegal transitions fail-closed ---------------------------------------

def test_terminals_have_no_outgoing_transitions():
    for term in (CellState.COMPLETE, CellState.FAILED):
        for dst in CellState:
            assert can_transition(term, dst) is False
    with pytest.raises(ValueError):
        assert_transition(CellState.COMPLETE, CellState.RUNNING)


def test_cannot_skip_states(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec()
    reg.register(spec, actor="CC3")
    # prepare requires VALIDATED; cell is still DRAFT
    with pytest.raises(CellError) as ei:
        reg.prepare(spec.cell_id, actor="CC3")
    assert ei.value.code == CellError.NOT_READY


def test_launch_requires_authorized(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec()
    _advance_to_ready(reg, spec)   # READY, not AUTHORIZED
    with pytest.raises(CellError) as ei:
        reg.launch(spec.cell_id, actor="CC3")
    assert ei.value.code == CellError.UNAUTHORIZED_LAUNCH
    assert reg.status(spec.cell_id)["state"] == "READY"  # state preserved


# --- authorization binding --------------------------------------------------

def test_authorization_must_match_current_identity(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec()
    _advance_to_ready(reg, spec)
    # authorization bound to a DIFFERENT identity (e.g. spec edited after grant)
    bad = make_authorization(spec.cell_id, "f" * 64, "human", SCOPE_NO_TRAINING,
                             spec.intended_total_timesteps)
    with pytest.raises(CellError) as ei:
        reg.authorize(spec.cell_id, bad, actor="CC3")
    assert ei.value.code == CellError.AUTHORIZATION_MISMATCH


def test_authorization_timestep_grant_must_match(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec()
    _advance_to_ready(reg, spec)
    bad = make_authorization(spec.cell_id, spec.identity_hash(), "human",
                             SCOPE_NO_TRAINING, 99999)  # != intended 4096
    with pytest.raises(CellError) as ei:
        reg.authorize(spec.cell_id, bad, actor="CC3")
    assert ei.value.code == CellError.AUTHORIZATION_MISMATCH


def test_revoked_authorization_rejected(tmp_path):
    reg = CellRegistry(str(tmp_path))
    spec = _spec()
    _advance_to_ready(reg, spec)
    auth = make_authorization(spec.cell_id, spec.identity_hash(), "human",
                              SCOPE_NO_TRAINING, spec.intended_total_timesteps)
    revoked = auth.model_copy(update={"revoked": True})
    with pytest.raises(CellError) as ei:
        reg.authorize(spec.cell_id, revoked, actor="CC3")
    assert ei.value.code == CellError.REVOKED_AUTHORIZATION
