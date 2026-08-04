"""CC3 E2 (from a2e1bc5): REAL SLOWGRU_PERSISTENT Student ABI compatibility.

The compatibility baseline is the real ``SLOWGRU_PERSISTENT_CANONICAL_98304``
capsule in this worktree — NOT fixtures. These tests:

* bind the real capsule read-only and assert every published ABI fact
  (params/optimizer identity, global step, RNG, wrapper, literal task_params,
  action ABI) equals the literal value carried by the SHA-verified documents;
* drive the bind's fail-closed gates with tmp capsule mirrors (missing root /
  missing artifact / byte tamper / missing ledger entry / malformed ledger /
  cross-document schema inconsistency / wrapper-literal drift / pkl-layout
  drift). The mirror helper copies the raw bytes of the real capsule, and an
  untampered mirror binds to an IDENTICAL baseline_hash (positive control);
* push EnvCoder output, all three FeedbackViews, the shared Soft Copeland
  ranking and the four-anchor manifest through the SAME
  ``SlowgruStudentEvaluator`` — positive and negative cases in normal, static
  and shuffled mode; missing/wrong schema fails closed;
* re-assert the exact k-1 feedback lag and the static/shuffled isolation at
  the evaluator surface (numeric side channel included);
* keep Soft Copeland the single ranking owner (the evaluator RECONSUMES
  ``soft_copeland_rank`` output and rejects any fork).

No training, no network, no GPU, no checkpoint load: the pkl body is a
server-only artifact and ``assert_checkpoint_consumable_locally`` must refuse
(``REAL_CHECKPOINT_LOADED`` stays False).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from d052.bagr_ued.soft_copeland import (
    EnvironmentScoreBundle,
    soft_copeland_rank,
)
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import student_abi_baseline as B
from d052.feedback_llm_ued.anchor_manifest import (
    AnchorManifestBlocked,
    AnchorManifestSource,
    SharedAnchorManifest,
)
from d052.feedback_llm_ued.axis_directive import AxisDirective
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.env_coder import (
    CodedDirective,
    EnvCoderOutput,
    run_env_coder,
)
from d052.feedback_llm_ued.feedback_view import (
    MASKED_IDENTITY,
    NormalFeedbackView,
    NullFeedbackView,
    PermutedFeedbackView,
)
from d052.feedback_llm_ued.llm_backend import DeterministicMockFeedbackBackend
from d052.feedback_llm_ued.student_abi_baseline import (
    SlowgruStudentEvaluator,
    bind_slowgru_persistent_baseline,
)
from d052.feedback_llm_ued.student_binding import local_symbolic_binding
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

REPO_ROOT = B.default_repo_root()

#: measured on 2026-08-04 by binding the real capsule in this worktree;
#: pinned so any silent fact drift in the capsule or the binder fails loudly
EXPECTED_BASELINE_HASH = (
    "8c279faaa2400707b0294fd2f16b4596727f3f0a99a0121f2b8e4f0b9a0efa8f")
EXPECTED_PARAMS_SHA = (
    "99d734b48acfd3499e5b836c7f632a52b1d17a732c3764a24c1935fd82a77ecc")
EXPECTED_FILE_SHA = (
    "0bc92c9ee28684ba507d6d6d728110000f11d7115126fbaf9137b1f8390a9c47")


@pytest.fixture(scope="module")
def baseline() -> B.StudentAbiBaseline:
    return bind_slowgru_persistent_baseline(REPO_ROOT)


@pytest.fixture(scope="module")
def evaluator(baseline) -> SlowgruStudentEvaluator:
    return SlowgruStudentEvaluator(baseline)


# ---------------------------------------------------------------------------
# builders (all values are schema-legal; nothing guesses capsule facts)
# ---------------------------------------------------------------------------
def _directive() -> AxisDirective:
    return AxisDirective(
        directive_id="dir-e2-threat-threat_distance_grading",
        source_window=0,
        environment_family="threat_distance_family",
        axis="threat_distance_grading",
        old_level="low",
        new_level="high",
        direction="increase",
        experiment_control_role="treatment",
        held_constant_axes={"threat_count": "medium"},
        expected_next_signature={"student_success_rate": 0.5})


def _window_records(window: int = 0):
    """Two threat-distance records with DIFFERENT rates (so a per-record
    leak is distinguishable from the family aggregate) + one resource
    record."""
    records = []
    for idx, rate in ((1, 0.3), (2, 0.7)):
        cand = synthetic_candidate(
            candidate_id=f"e2-syn-w{window}-{idx}",
            family="threat_distance_family",
            axes=["threat_distance_grading"])
        records.append(synthetic_feedback_record(
            feedback_id=f"fb-e2-w{window}-{idx:03d}",
            candidate=cand, plan_id=f"plan-e2-w{window}", window=window,
            student_success_rate=rate,
            expected_signature={"student_success_rate": round(rate + 0.1, 6)}))
    cand3 = synthetic_candidate(
        candidate_id=f"e2-syn-w{window}-3",
        family="resource_pressure_family",
        axes=["resource_pressure"])
    records.append(synthetic_feedback_record(
        feedback_id=f"fb-e2-w{window}-003",
        candidate=cand3, plan_id=f"plan-e2-w{window}", window=window,
        student_success_rate=0.5,
        expected_signature={"student_success_rate": 0.6}))
    return records


def _bundles():
    def _make(env_id: str, front_regret: float, learning_progress: float):
        return EnvironmentScoreBundle(
            environment_id=env_id, front_regret=front_regret,
            global_regret=0.4, behavioral_gap=0.3,
            learning_progress=learning_progress, learnability=0.5,
            diversity=0.7, global_retention=0.8, critic_penalty=0.2,
            alpha_front=0.5)
    return [_make("e2-env-a", 0.5, 0.6), _make("e2-env-b", 0.2, 0.9)]


def _envcoder_output(window: int = 0) -> EnvCoderOutput:
    output, _envelope = run_env_coder(
        window=window, directives=[_directive()],
        backend=DeterministicMockFeedbackBackend(), sequence=0)
    return output


def _frozen_anchor_source() -> AnchorManifestSource:
    return AnchorManifestSource(SharedAnchorManifest(
        manifest_id="e2-shared-manifest",
        anchors=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
        frozen=True))


def _evaluate_all(evaluator, *, mode: str, board_window: int, view):
    return evaluator.evaluate(
        mode=mode, board_window=board_window,
        env_coder_output=_envcoder_output(window=board_window - 1),
        feedback_view=view,
        copeland_bundles=_bundles(),
        copeland_ranking=soft_copeland_rank(_bundles()),
        anchor_source=_frozen_anchor_source())


# ---------------------------------------------------------------------------
# A. real-capsule bind: positive facts
# ---------------------------------------------------------------------------
def test_bind_real_capsule_extracts_literal_facts(baseline):
    assert baseline.candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304"
    assert baseline.owner == "CC3"
    assert baseline.network_family == "SlowGRU"
    assert baseline.carry_mode == "PERSISTENT"
    assert baseline.budget_class == "MATCHED_98304"
    assert baseline.formal_eval_binding == "WAITING_CC4_COMMON_CONTRACT"

    abi = baseline.action_abi
    assert (abi.obs_dim, abi.action_dim) == (8335, 43)
    assert (abi.legal_action_min, abi.legal_action_max) == (0, 42)
    assert abi.conditioning_dim == 67
    assert tuple(abi.observation_shape) == (8335,)

    ckpt = baseline.checkpoint
    assert ckpt.global_step == 98304
    assert ckpt.update_step == 48
    assert ckpt.opt_step == 96
    assert ckpt.params_sha256 == EXPECTED_PARAMS_SHA
    assert ckpt.checkpoint_file_sha256 == EXPECTED_FILE_SHA
    assert ckpt.canonical_base_params_sha256 == (
        "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5")
    assert ckpt.resume_source_file_sha256 == (
        "d4e7008da7d2a78a7765b32379704719165288015d031856850ad7c8a0e7495e")
    assert ckpt.resume_source_params_sha256 == (
        "1bd4fbfe91ab4da44c274ef20f372e04bf6a7e8367869e39e0b65f044c85e9f2")

    rng = baseline.rng_policy
    assert (rng.training_seed, rng.smoke_seed) == (42, 777)
    assert rng.full_smoke_seed_base == 200000
    assert rng.full_smoke_seed_count == 64
    assert rng.cc3_created_full_seeds is False
    assert "200000..200063" in rng.full_seed_source

    wrapper = baseline.wrapper
    assert wrapper.runtime_name == "THIN_GTRXL128_SLOWGRU_RUNTIME"
    assert wrapper.abi_version == "cc3_runtime_abi/v1"
    assert tuple(wrapper.abi_surface) == B.RUNTIME_ABI_SURFACE
    assert wrapper.boundary_action == "FULL_CARRY_NO_CLEAR"
    assert (wrapper.window_mem, wrapper.num_layers, wrapper.embed_size,
            wrapper.num_heads, wrapper.slow_interval, wrapper.slow_dim) == (
                128, 2, 256, 8, 32, 256)

    task = baseline.task_params
    assert task.task == "DEFEAT_KOBOLD (S4_dark native)"
    assert task.goal == "DEFEAT_KOBOLD"
    assert task.stage == "S4_dark native"
    assert task.mode == "score"
    assert task.bonus_type == "none"
    assert task.condition_on_task is True
    assert (task.replay, task.vtrace) == ("OFF", "OFF")
    assert (task.hindsight, task.awr) == (False, False)
    assert (task.egomap, task.nav_aux, task.novelty) == ("OFF", "OFF", "OFF")
    assert task.total_env_steps == 98304
    assert task.xla_flags == "--xla_gpu_deterministic_ops=true"

    assert set(baseline.gpu_policy.gpu_allowed) == {
        "GPU-8df11537-ab79-722d-606f-411966196c4c",
        "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"}
    assert set(baseline.gpu_policy.gpu_forbidden) == {
        "GPU-e8c08612", "GPU-3c7a2864"}

    lock = baseline.environment_lock
    assert lock.conda_env == "dicode310"
    assert lock.python == "3.10.20"
    assert lock.jax == "0.6.0"
    assert lock.numpy == "2.2.6"
    assert lock.local_jax_craftax_forbidden is True

    assert baseline.constructor["action_dim"] == 43
    assert baseline.constructor["gating"] is True
    assert baseline.constructor["gating_bias"] == 2.0
    assert baseline.constructor["use_longmem"] is True
    assert baseline.memory_layout == {
        "window_mem": 128, "num_layers": 2, "num_heads": 8,
        "embed_size": 256, "slow_interval": 32, "slow_dim": 256}

    layout = baseline.state_layout
    assert frozenset(layout.required_pkl_keys) == B.CC3_PKL_REQUIRED_KEYS
    assert tuple(layout.packed_pytree_keys) == ("params", "opt_state",
                                                "longstate")

    # determinism: rebinding the same real capsule is bit-identical
    again = bind_slowgru_persistent_baseline(REPO_ROOT)
    assert again.baseline_hash == baseline.baseline_hash
    assert baseline.baseline_hash == EXPECTED_BASELINE_HASH


def test_document_shas_match_ledger_and_crlf_views_are_recorded(baseline):
    capsule = REPO_ROOT / B.CAPSULE_POSIX
    ledger = B.parse_sha256sums(
        (capsule / B.SHA256SUMS_NAME).read_text(encoding="utf-8"))
    # the 12 byte-bound documents (ledger + READY.json are not byte-bound)
    assert len(baseline.capsule_document_shas) == 12
    # crlf_view_documents also covers the two SHA-verified tooling sources,
    # recorded under their capsule-relative LEDGER keys
    tooling = {"../slowgru_runtime/slowgru_runtime.py",
               "../cc3_common/recovery_probe.py"}
    crlf = set(baseline.crlf_view_documents)
    assert crlf <= (set(baseline.capsule_document_shas) | tooling)

    def _check_one(rel: str, doc_sha: str, resolved: Path) -> None:
        assert doc_sha == ledger[rel], rel
        raw = resolved.read_bytes()
        if rel in crlf:
            # the ONLY admitted non-byte-identical view: exactly-invertible
            # CRLF inflation of the LF bytes the ledger hashed
            lf_view = raw.replace(b"\r\n", b"\n")
            assert hashlib.sha256(lf_view).hexdigest() == doc_sha
            assert raw == lf_view.replace(b"\n", b"\r\n")
        else:
            assert hashlib.sha256(raw).hexdigest() == doc_sha

    for rel, doc_sha in baseline.capsule_document_shas.items():
        _check_one(rel, doc_sha, capsule / rel)
    # the wrapper source is ledger-bound under its capsule-relative key
    _check_one("../slowgru_runtime/slowgru_runtime.py",
               baseline.wrapper.shared_runtime_src_sha256,
               REPO_ROOT / B.RUNTIME_SOURCE_POSIX)


def test_mirror_capsule_binds_to_identical_baseline(tmp_path):
    """Positive control for the negative tests: an untampered mirror of the
    real capsule binds to the IDENTICAL baseline (same facts, same hash)."""
    _mirror_capsule(tmp_path)
    mirrored = bind_slowgru_persistent_baseline(tmp_path)
    real = bind_slowgru_persistent_baseline(REPO_ROOT)
    assert mirrored.baseline_hash == real.baseline_hash
    assert mirrored.model_dump() == real.model_dump()


# ---------------------------------------------------------------------------
# B. bind negatives: missing/wrong schema fails closed (no guessing)
# ---------------------------------------------------------------------------
def test_missing_capsule_root_fails_closed(tmp_path):
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="CAPSULE_ROOT_MISSING"):
        bind_slowgru_persistent_baseline(tmp_path)


def test_missing_required_artifact_fails_closed(tmp_path):
    root = _mirror_capsule(tmp_path)
    (root / B.CAPSULE_POSIX / "training_contract.json").unlink()
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="REQUIRED_CAPSULE_ARTIFACT_MISSING"):
        bind_slowgru_persistent_baseline(root)


def test_tampered_document_bytes_fail_closed(tmp_path):
    root = _mirror_capsule(tmp_path)
    manifest = root / B.CAPSULE_POSIX / "candidate_manifest.json"
    # same-length byte tamper: still parseable JSON, wrong bytes
    _tamper_all(
        manifest,
        b"SLOWGRU_PERSISTENT_CANONICAL_98304",
        b"SLOWGRU_PERSISTENT_CANONICAL_98305")
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="CAPSULE_DOC_SHA_MISMATCH"):
        bind_slowgru_persistent_baseline(root)


def test_missing_ledger_entry_fails_closed(tmp_path):
    root = _mirror_capsule(tmp_path)
    ledger_path = root / B.CAPSULE_POSIX / B.SHA256SUMS_NAME
    lines = ledger_path.read_bytes().split(b"\n")
    kept = [ln for ln in lines
            if not ln.rstrip(b"\r").endswith(b"  checkpoint_contract.json")]
    assert len(kept) == len(lines) - 1
    ledger_path.write_bytes(b"\n".join(kept))
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="SHA256SUMS_ENTRY_MISSING"):
        bind_slowgru_persistent_baseline(root)


def test_malformed_ledger_fails_closed(tmp_path):
    root = _mirror_capsule(tmp_path)
    ledger_path = root / B.CAPSULE_POSIX / B.SHA256SUMS_NAME
    raw = ledger_path.read_bytes()
    # corrupt the first hex character of the first entry (kept same length)
    fixed = b"z" + raw[1:]
    assert fixed != raw
    ledger_path.write_bytes(fixed)
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="SHA256SUMS_MALFORMED"):
        bind_slowgru_persistent_baseline(root)


def test_cross_document_step_consistency_gate(tmp_path):
    """The byte gate can be satisfied while the DOCUMENTS disagree: tamper
    the train_summary chunk params sha at global_step=98304 and re-hash that
    one ledger entry with the tampered bytes. The SHA gate passes; the
    triple-consistency gate must fail closed."""
    root = _mirror_capsule(tmp_path)
    capsule = root / B.CAPSULE_POSIX
    summary_path = capsule / (
        "out/CC3_SLOWGRU_CANONICAL_PERSISTENT_train_summary.json")
    old = EXPECTED_PARAMS_SHA.encode("ascii")
    new = b"8" + old[1:]                       # valid-looking, different
    raw = summary_path.read_bytes()
    assert raw.count(old) == 1, "chunk 98304 must be the only occurrence"
    summary_path.write_bytes(raw.replace(old, new))
    _replace_ledger_sha(
        capsule, "out/CC3_SLOWGRU_CANONICAL_PERSISTENT_train_summary.json",
        hashlib.sha256(summary_path.read_bytes()).hexdigest())
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="CAPSULE_STEP_CONSISTENCY_MISMATCH"):
        bind_slowgru_persistent_baseline(root)


def test_wrapper_literal_gate_fails_closed(tmp_path):
    """A re-hashed but ABI-drifted wrapper source must be refused: the
    literals are part of the baseline, not just the source hash."""
    root = _mirror_capsule(tmp_path)
    runtime = root / B.RUNTIME_SOURCE_POSIX
    _replace_bytes_once(runtime, b"OBS_DIM = 8335", b"OBS_DIM = 8336")
    _replace_ledger_sha(
        root / B.CAPSULE_POSIX, "../slowgru_runtime/slowgru_runtime.py",
        hashlib.sha256(runtime.read_bytes()).hexdigest())
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="RUNTIME_ABI_LITERAL_MISSING"):
        bind_slowgru_persistent_baseline(root)


def test_pkl_layout_literal_gate_fails_closed(tmp_path):
    root = _mirror_capsule(tmp_path)
    probe = root / B.RECOVERY_PROBE_POSIX
    # the key literal recurs (REQUIRED_PKL_KEYS + accessors) — the gate
    # requires the literal, so EVERY occurrence must be drifted
    assert _tamper_all(probe, b'"longstate"', b'"longstatX"') >= 1
    _replace_ledger_sha(
        root / B.CAPSULE_POSIX, "../cc3_common/recovery_probe.py",
        hashlib.sha256(probe.read_bytes()).hexdigest())
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="PKL_LAYOUT_LITERAL_MISSING"):
        bind_slowgru_persistent_baseline(root)


# ---------------------------------------------------------------------------
# C. checkpoint locality honesty
# ---------------------------------------------------------------------------
def test_checkpoint_is_server_only_and_local_load_refused(baseline,
                                                          evaluator):
    assert baseline.checkpoint.checkpoint_location_class == (
        B.CHECKPOINT_LOCATION_SERVER_ONLY)
    local_pkl = (REPO_ROOT / B.CAPSULE_POSIX
                 / "ckpt/98304/full_state.pkl")
    assert not local_pkl.is_file()
    with pytest.raises(B.StudentAbiBaselineBlocked,
                       match="LOCAL_CHECKPOINT_LOAD_REFUSED"):
        evaluator.assert_checkpoint_consumable_locally()
    # binding the ABI baseline loads nothing
    assert C.REAL_CHECKPOINT_LOADED is False


# ---------------------------------------------------------------------------
# D. evaluator — normal mode (all four surfaces through ONE evaluator)
# ---------------------------------------------------------------------------
def test_normal_mode_all_four_surfaces_compatible(evaluator):
    view = NormalFeedbackView(_window_records(0), window_scope=0)
    report = _evaluate_all(evaluator, mode=C.MODE_NORMAL_FEEDBACK,
                           board_window=1, view=view)
    assert report.overall_status == B.STATUS_COMPATIBLE
    assert report.baseline_hash == evaluator.baseline.baseline_hash
    assert report.candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304"
    assert set(report.surfaces) == {
        B.SURFACE_ENV_CODER, B.SURFACE_FEEDBACK_VIEW,
        B.SURFACE_SOFT_COPELAND, B.SURFACE_ANCHOR_MANIFEST}
    for surface in report.surfaces.values():
        assert surface.status == B.STATUS_COMPATIBLE
        assert surface.baseline_candidate_id == report.candidate_id
    assert report.exact_feedback_lag_verified is True
    anchors = report.surfaces[B.SURFACE_ANCHOR_MANIFEST].detail
    for anchor_id in C.GLOBAL_CANONICAL_ANCHOR_IDS:
        assert anchor_id in anchors
    # determinism: same inputs -> bit-identical report
    report2 = _evaluate_all(evaluator, mode=C.MODE_NORMAL_FEEDBACK,
                            board_window=1, view=view)
    assert report.report_hash == report2.report_hash
    assert report.model_dump() == report2.model_dump()


def test_normal_identity_stamps_match_baseline(evaluator, baseline):
    stamped = [r.model_copy(update=dict(
        student_parameter_tree_hash=baseline.checkpoint.params_sha256,
        student_checkpoint_step=baseline.checkpoint.global_step))
        for r in _window_records(0)]
    view = NormalFeedbackView(stamped, window_scope=0)
    surface = evaluator.check_feedback_view(
        view, mode=C.MODE_NORMAL_FEEDBACK, board_window=1)
    assert surface.status == B.STATUS_COMPATIBLE
    assert "student_identity_stamps_consistent" in surface.checks_passed


def test_symbolic_binding_stamp_is_the_honest_no_checkpoint_state(evaluator):
    """While the CC4 adapter is absent, controller records carry the local
    symbolic-binding hash (NOT_LOADED_LOCAL). That declared state is legal at
    the evaluator surface — and is the ONLY non-baseline stamp that is."""
    assert C.REAL_CHECKPOINT_LOADED is False
    symbolic = local_symbolic_binding()
    stamped = [r.model_copy(update=dict(
        student_parameter_tree_hash=symbolic.parameter_tree_hash,
        student_checkpoint_step=symbolic.checkpoint_global_step))
        for r in _window_records(0)]
    view = NormalFeedbackView(stamped, window_scope=0)
    surface = evaluator.check_feedback_view(
        view, mode=C.MODE_NORMAL_FEEDBACK, board_window=1)
    assert surface.status == B.STATUS_COMPATIBLE


def test_symbolic_stamp_with_nonzero_step_fails_closed(evaluator):
    symbolic = local_symbolic_binding()
    bad = _window_records(0)[0].model_copy(update=dict(
        student_parameter_tree_hash=symbolic.parameter_tree_hash,
        student_checkpoint_step=98304))
    view = NormalFeedbackView([bad], window_scope=0)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="STUDENT_CHECKPOINT_STEP_MISMATCH"):
        evaluator.check_feedback_view(
            view, mode=C.MODE_NORMAL_FEEDBACK, board_window=1)


def test_wrong_parameter_tree_stamp_fails_closed(evaluator):
    bad = _window_records(0)[0].model_copy(update=dict(
        student_parameter_tree_hash="0" * 64))
    view = NormalFeedbackView([bad], window_scope=0)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="STUDENT_PARAMETER_TREE_MISMATCH"):
        evaluator.check_feedback_view(
            view, mode=C.MODE_NORMAL_FEEDBACK, board_window=1)


def test_wrong_checkpoint_step_stamp_fails_closed(evaluator, baseline):
    bad = _window_records(0)[0].model_copy(update=dict(
        student_parameter_tree_hash=baseline.checkpoint.params_sha256,
        student_checkpoint_step=24576))
    view = NormalFeedbackView([bad], window_scope=0)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="STUDENT_CHECKPOINT_STEP_MISMATCH"):
        evaluator.check_feedback_view(
            view, mode=C.MODE_NORMAL_FEEDBACK, board_window=1)


def test_unknown_mode_rejected(evaluator):
    view = NormalFeedbackView(_window_records(0), window_scope=0)
    with pytest.raises(B.StudentCompatibilityBlocked, match="UNKNOWN_MODE"):
        evaluator.check_feedback_view(view, mode="bogus_mode",
                                      board_window=1)


# ---------------------------------------------------------------------------
# E. EnvCoder surface
# ---------------------------------------------------------------------------
def test_envcoder_symbolic_pipeline_output_compatible(evaluator):
    surface = evaluator.check_env_coder(_envcoder_output(window=0))
    assert surface.status == B.STATUS_COMPATIBLE
    assert "action_abi_untouched" in surface.checks_passed
    assert "reset_step_contracts_family_bound" in surface.checks_passed


def test_envcoder_wrong_output_type_fails_closed(evaluator):
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="ENVCODER_OUTPUT_SCHEMA_MISMATCH"):
        evaluator.check_env_coder({"window": 0, "coded": []})


def test_envcoder_directive_hash_not_sha256_fails_closed(evaluator):
    fam = "threat_distance_family"
    coded = CodedDirective(
        directive_id="dir-bad-hash", directive_hash="not-a-sha",
        environment_family=fam, axis="threat_distance_grading",
        new_level="high", experiment_control_role="treatment",
        code_symbol="ENVCODE_SYMBOLIC_V1::x",
        reset_contract=f"reset(seed)->state::{fam}",
        step_contract=f"step(action)->(state,reward,terminal,info)::{fam}")
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="ENVCODER_DIRECTIVE_HASH_NOT_SHA256"):
        evaluator.check_env_coder(EnvCoderOutput(window=0, coded=[coded]))


def test_envcoder_action_abi_override_forbidden(evaluator):
    """The EnvCoder charter: realize environment axes, NEVER touch the
    observation/action ABI of the real baseline."""
    fam = "threat_distance_family"
    coded = CodedDirective(
        directive_id="dir-abi-override", directive_hash="a" * 64,
        environment_family=fam, axis="threat_distance_grading",
        new_level="high", experiment_control_role="treatment",
        code_symbol="ENVCODE_SYMBOLIC_V1::y",
        reset_contract=f"reset(seed,action_dim=44)->state::{fam}",
        step_contract=f"step(action)->(state,reward,terminal,info)::{fam}")
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="ACTION_ABI_OVERRIDE_FORBIDDEN"):
        evaluator.check_env_coder(EnvCoderOutput(window=0, coded=[coded]))


def test_envcoder_unknown_family_schema_fails_closed():
    with pytest.raises(ValidationError, match="UNKNOWN_ENVIRONMENT_FAMILY"):
        CodedDirective(
            directive_id="dir-bad-family", directive_hash="a" * 64,
            environment_family="bogus_family", axis="threat_count",
            new_level="high", experiment_control_role="treatment",
            code_symbol="ENVCODE_SYMBOLIC_V1::z",
            reset_contract="reset(seed)->state::bogus_family",
            step_contract="step(action)->(state,reward,terminal,info)"
                          "::bogus_family")


def test_envcoder_wrong_reset_contract_fails_closed(evaluator):
    coded = CodedDirective(
        directive_id="dir-bad-contract", directive_hash="a" * 64,
        environment_family="threat_distance_family",
        axis="threat_distance_grading", new_level="high",
        experiment_control_role="treatment",
        code_symbol="ENVCODE_SYMBOLIC_V1::w",
        reset_contract="reset(seed)->state::resource_pressure_family",
        step_contract="step(action)->(state,reward,terminal,info)"
                      "::threat_distance_family")
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="ENVCODER_RESET_CONTRACT_INCOMPATIBLE"):
        evaluator.check_env_coder(EnvCoderOutput(window=0, coded=[coded]))


# ---------------------------------------------------------------------------
# F. static mode (structural null view)
# ---------------------------------------------------------------------------
def test_static_null_view_compatible(evaluator):
    surface = evaluator.check_feedback_view(
        NullFeedbackView(), mode=C.MODE_STATIC_LLM, board_window=3)
    assert surface.status == B.STATUS_COMPATIBLE
    assert "static_structurally_null" in surface.checks_passed
    assert "zero_feedback_payload" in surface.checks_passed
    report = _evaluate_all(evaluator, mode=C.MODE_STATIC_LLM,
                           board_window=1, view=NullFeedbackView())
    assert report.overall_status == B.STATUS_COMPATIBLE
    # static has no fed feedback: the lag is verified structurally
    assert report.exact_feedback_lag_verified is True


def test_static_rejects_fed_view(evaluator):
    view = NormalFeedbackView(_window_records(0), window_scope=0)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="STATIC_VIEW_MUST_BE_STRUCTURALLY_NULL"):
        evaluator.check_feedback_view(
            view, mode=C.MODE_STATIC_LLM, board_window=1)


# ---------------------------------------------------------------------------
# G. shuffled mode (frozen permutation, isolation at the evaluator surface)
# ---------------------------------------------------------------------------
def _permuted_view(window: int = 0, board_window: int = 1):
    return PermutedFeedbackView(
        _window_records(window), window_scope=window,
        board_window=board_window, mode=C.MODE_SHUFFLED_FEEDBACK,
        seed_schedule_hash=C.SEED_SCHEDULE_HASH)


def test_shuffled_permuted_view_compatible(evaluator):
    view = _permuted_view()
    surface = evaluator.check_feedback_view(
        view, mode=C.MODE_SHUFFLED_FEEDBACK, board_window=1)
    assert surface.status == B.STATUS_COMPATIBLE
    assert set(surface.checks_passed) == {
        "exact_window_scope_k_minus_1", "candidate_ids_masked",
        "family_level_aggregates_only", "axes_and_signatures_masked"}
    for payload in view.to_prompt_payload():
        assert payload["candidate_id"] == MASKED_IDENTITY
    report = _evaluate_all(evaluator, mode=C.MODE_SHUFFLED_FEEDBACK,
                           board_window=1, view=view)
    assert report.overall_status == B.STATUS_COMPATIBLE
    assert report.exact_feedback_lag_verified is True


def test_shuffled_rejects_normal_view(evaluator):
    view = NormalFeedbackView(_window_records(0), window_scope=0)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="SHUFFLED_VIEW_MUST_BE_PERMUTED"):
        evaluator.check_feedback_view(
            view, mode=C.MODE_SHUFFLED_FEEDBACK, board_window=1)


def test_shuffled_numeric_side_channel_fails_closed(evaluator):
    """A malicious view subclass that smuggles per-record exact rates into
    the payload (a re-identification fingerprint) must fail closed: the
    evaluator recomputes the public family-level aggregates itself."""

    class _LeakyPermutedView(PermutedFeedbackView):
        def to_prompt_payload(self):
            payloads = super().to_prompt_payload()
            out = []
            for payload, record in zip(payloads, self.records()):
                leaked = dict(payload)
                leaked["student_success_rate"] = (
                    record.stage2_metrics.student_success_rate)
                out.append(leaked)
            return out

    leaky = _LeakyPermutedView(
        _window_records(0), window_scope=0, board_window=1,
        mode=C.MODE_SHUFFLED_FEEDBACK,
        seed_schedule_hash=C.SEED_SCHEDULE_HASH)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="NUMERIC_SIDE_CHANNEL_IN_VIEW"):
        evaluator.check_feedback_view(
            leaky, mode=C.MODE_SHUFFLED_FEEDBACK, board_window=1)


# ---------------------------------------------------------------------------
# H. exact k-1 feedback lag (re-asserted at the evaluator surface)
# ---------------------------------------------------------------------------
def test_exact_lag_violations_fail_closed(evaluator):
    view = NormalFeedbackView(_window_records(0), window_scope=0)
    for board_window in (0, 2, 5):          # stale (0<board-1) and future-ish
        with pytest.raises(B.StudentCompatibilityBlocked,
                           match="EXACT_FEEDBACK_LAG_VIOLATED"):
            evaluator.check_feedback_view(
                view, mode=C.MODE_NORMAL_FEEDBACK,
                board_window=board_window)
    # positive control: scope 1 consumed at board window 2 is exactly k-1
    view1 = NormalFeedbackView(_window_records(1), window_scope=1)
    surface = evaluator.check_feedback_view(
        view1, mode=C.MODE_NORMAL_FEEDBACK, board_window=2)
    assert surface.status == B.STATUS_COMPATIBLE
    assert "exact_window_scope_k_minus_1" in surface.checks_passed


def test_mixed_window_view_construction_fails_closed():
    mixed = _window_records(0) + _window_records(1)
    with pytest.raises(ValueError, match="STALE_FEEDBACK_ID"):
        NormalFeedbackView(mixed, window_scope=0)


# ---------------------------------------------------------------------------
# I. Soft Copeland: single owner, reconsume never fork
# ---------------------------------------------------------------------------
def test_copeland_reconsumption_compatible(evaluator):
    bundles = _bundles()
    surface = evaluator.check_soft_copeland(
        bundles, soft_copeland_rank(bundles))
    assert surface.status == B.STATUS_COMPATIBLE
    assert "shared_soft_copeland_single_owner" in surface.checks_passed
    assert "ranking_hash_reproduced" in surface.checks_passed


def test_copeland_fork_and_schema_fail_closed(evaluator):
    bundles = _bundles()
    ranking = soft_copeland_rank(bundles)
    forked = ranking.model_copy(update=dict(
        ranking_hash="f" * 64))
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="SOFT_COPELAND_RANKING_FORKED"):
        evaluator.check_soft_copeland(bundles, forked)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="EMPTY_COPELAND_BUNDLES"):
        evaluator.check_soft_copeland([], ranking)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="COPELAND_BUNDLE_SCHEMA_MISMATCH"):
        evaluator.check_soft_copeland([{"environment_id": "x"}], ranking)
    with pytest.raises(B.StudentCompatibilityBlocked,
                       match="COPELAND_RANKING_SCHEMA_MISMATCH"):
        evaluator.check_soft_copeland(bundles, {"ranking_hash": "f" * 64})


# ---------------------------------------------------------------------------
# J. the four-anchor manifest seam
# ---------------------------------------------------------------------------
def test_anchor_manifest_missing_or_unfrozen_fail_closed(evaluator):
    with pytest.raises(AnchorManifestBlocked,
                       match="BLOCKED_SHARED_ANCHOR_MANIFEST"):
        evaluator.check_anchor_manifest(AnchorManifestSource(None))
    unfrozen = SharedAnchorManifest(
        manifest_id="e2-unfrozen",
        anchors=list(C.GLOBAL_CANONICAL_ANCHOR_IDS), frozen=False)
    with pytest.raises(AnchorManifestBlocked,
                       match="BLOCKED_SHARED_ANCHOR_MANIFEST"):
        evaluator.check_anchor_manifest(AnchorManifestSource(unfrozen))


def test_anchor_manifest_hash_mismatch_fails_closed(evaluator):
    tampered = SharedAnchorManifest(
        manifest_id="e2-tampered",
        anchors=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
        frozen=True, manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="ANCHOR_MANIFEST_HASH_MISMATCH"):
        evaluator.check_anchor_manifest(AnchorManifestSource(tampered))


def test_anchor_slot_count_schema_fails_closed():
    with pytest.raises(ValidationError, match="ILLEGAL_ANCHOR_SLOT_COUNT"):
        SharedAnchorManifest(
            manifest_id="e2-three-slots",
            anchors=list(C.GLOBAL_CANONICAL_ANCHOR_IDS)[:3], frozen=True)


# ---------------------------------------------------------------------------
# K. controller integration: the SAME evaluator consumes all three modes
# ---------------------------------------------------------------------------
def test_controller_three_modes_same_evaluator(evaluator):
    for mode in (C.MODE_STATIC_LLM, C.MODE_NORMAL_FEEDBACK,
                 C.MODE_SHUFFLED_FEEDBACK):
        controller = FeedbackUEDController(mode)
        summary = controller.run(max_windows=2)
        assert summary.n_windows == 2
        # the controller stamps its records with the honest local symbolic
        # binding (no checkpoint in this worktree) — never with a made-up
        # real parameter tree hash
        symbolic = local_symbolic_binding()
        window0 = controller.store.for_window(0)
        assert window0
        for record in window0:
            assert record.student_parameter_tree_hash == \
                symbolic.parameter_tree_hash
            assert record.student_checkpoint_step == 0
        if mode == C.MODE_STATIC_LLM:
            view = NullFeedbackView()
        elif mode == C.MODE_NORMAL_FEEDBACK:
            view = NormalFeedbackView.from_store(
                controller.store, evidence_window=0)
        else:
            view = PermutedFeedbackView(
                controller.store.for_window(0), window_scope=0,
                board_window=1, mode=C.MODE_SHUFFLED_FEEDBACK,
                seed_schedule_hash=C.SEED_SCHEDULE_HASH)
        surface = evaluator.check_feedback_view(
            view, mode=mode, board_window=1)
        assert surface.status == B.STATUS_COMPATIBLE, mode
        assert surface.baseline_candidate_id == (
            "SLOWGRU_PERSISTENT_CANONICAL_98304")


# ---------------------------------------------------------------------------
# mirror helpers (raw-byte copy of the real capsule; negative tests mutate
# the mirror only — the real capsule is never written)
# ---------------------------------------------------------------------------
def _mirror_capsule(root: Path) -> Path:
    """Raw-byte mirror of the real capsule. REQUIRED_LOCAL_FILES are
    capsule-relative; the three tooling files are repo-relative."""
    capsule_src = REPO_ROOT / B.CAPSULE_POSIX
    capsule_dst = root / B.CAPSULE_POSIX
    for rel in B.REQUIRED_LOCAL_FILES:
        dst = capsule_dst / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((capsule_src / rel).read_bytes())
    for rel in (B.RUNTIME_SOURCE_POSIX, B.RECOVERY_PROBE_POSIX,
                B.BINDING_CONTRACT_POSIX):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO_ROOT / rel).read_bytes())
    return root


def _replace_bytes_once(path: Path, old: bytes, new: bytes) -> None:
    raw = path.read_bytes()
    assert raw.count(old) == 1, f"{old!r} not unique in {path.name}"
    path.write_bytes(raw.replace(old, new))


def _tamper_all(path: Path, old: bytes, new: bytes) -> int:
    """Replace EVERY occurrence (some literals recur); returns the count."""
    raw = path.read_bytes()
    count = raw.count(old)
    assert count >= 1, f"{old!r} absent from {path.name}"
    path.write_bytes(raw.replace(old, new))
    return count


def _replace_ledger_sha(capsule: Path, rel: str, new_sha: str) -> None:
    """Re-hash ONE ledger entry with the (tampered) raw bytes so the byte
    gate passes and the deeper consistency/literal gate is exercised."""
    ledger_path = capsule / B.SHA256SUMS_NAME
    lines = ledger_path.read_bytes().split(b"\n")
    suffix = ("  " + rel).encode("ascii")
    hits = 0
    out = []
    for line in lines:
        body = line.rstrip(b"\r")
        crlf = line[len(body):]
        if body.endswith(suffix):
            line = new_sha.encode("ascii") + suffix + crlf
            hits += 1
        out.append(line)
    assert hits == 1, f"ledger entry for {rel!r} not found"
    ledger_path.write_bytes(b"\n".join(out))
