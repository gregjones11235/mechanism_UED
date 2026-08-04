"""Mechanical-enforcement tests for the R4c production fresh-process driver.

Independent audit closure (2026-08-04): the callback-based contract driver
(combined_restore_contract.run_combined_restore) runs in the CURRENT process
with self-asserted ComponentResult statuses.  These tests pin the mechanical
replacements: exactly ONE spawned child process, atomic PID/PPID/argv/
timestamp/exit-code evidence bound to the actually spawned child,
authoritative per-component leaf count/order/shape/dtype/value hashes,
checkpoint-leaf optimizer binding (tx.init substitution rejected), and
split-process / crash / torn-report rejection.

The single positive test spawns a REAL subprocess but restores labelled
synthetic artifacts only (SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT): it
proves the enforcement contract, NOT a real combined restore —
COMBINED_FRESH_PROCESS_RESTORE_EXECUTED stays False this round.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os

import pytest

from dicode.simulator_frontier import (
    CALLBACK_DRIVER_IS_CONTRACT_ONLY,
    COMBINED_FRESH_PROCESS_RESTORE_EXECUTED,
    FRESH_PROCESS_DRIVER_CONTRACT_READY,
    OPTIMIZER_ORIGIN_CHECKPOINT,
    OPTIMIZER_ORIGIN_TX_INIT,
    SYNTHETIC_FIXTURE_LABEL,
    REQUIRED_COMPONENTS,
    CROSS_CHECKS,
    CombinedRestoreRequest,
    ComponentResult,
    ComponentStatus,
    ComponentArtifactSpec,
    FreshProcessRestoreRequest,
    ProcessEvidence,
    clear_production_registrations,
    evaluate_verdict,
    leaves_digest_of,
    load_evidence_payload,
    production_joint_pass,
    production_registrations_status,
    register_production_component_loader,
    register_production_replay,
    run_fresh_process_restore,
    run_fresh_process_restore_production,
    synthetic_replay_digest,
    tree_leaf_records,
    verify_fresh_process_evidence,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.fresh_process_restore import (
    ARTIFACT_SCHEMA,
    EVIDENCE_SCHEMA,
    ORIGIN_SOURCE_ARTIFACT,
    WORKER_MODULE,
)

CHECKPOINT_SHA = "1" * 64
ABI_SHA = "2" * 64
REGISTRY_SHA = "3" * 64
MANIFEST_SHA = "4" * 64
NEXT_STEP_SHA = "5" * 64

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True  # labeling discipline


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _tree_for(component: str) -> dict:
    """Deterministic multi-leaf synthetic tree (never real scientific data)."""
    return {
        "component": component,
        "values": [1, 2, 3],
        "nested": {"scale": 0.25, "flag": True},
        "step": 98304,
    }


def _write_artifacts(root) -> dict:
    """Write one hash-bound artifact file per required component."""
    out = {}
    for name in REQUIRED_COMPONENTS:
        tree = _tree_for(name)
        blob = _canonical({"schema": ARTIFACT_SCHEMA, "tree": tree}).encode("utf-8")
        path = root / f"artifact_{name}.json"
        path.write_bytes(blob)
        out[name] = {"path": str(path), "sha256": hashlib.sha256(blob).hexdigest(),
                     "tree": tree}
    return out


def _digests(artifacts: dict) -> dict:
    return {name: leaves_digest_of(tree_leaf_records(info["tree"]))
            for name, info in artifacts.items()}


def _fixture_request(root, *, fixture_label=SYNTHETIC_FIXTURE_LABEL,
                     digest_overrides=None, **overrides) -> FreshProcessRestoreRequest:
    artifacts = _write_artifacts(root)
    digests = _digests(artifacts)
    digests.update(digest_overrides or {})
    specs = {name: ComponentArtifactSpec(
                 path=info["path"], sha256=info["sha256"],
                 expected_leaves_digest=digests[name])
             for name, info in artifacts.items()}
    replay = synthetic_replay_digest(digests)
    base = dict(
        checkpoint_path="/synthetic/contract_checkpoint.pkl",
        checkpoint_sha256=CHECKPOINT_SHA,
        student_abi_identity_hash=ABI_SHA,
        registry_hash=REGISTRY_SHA,
        manifest_hash=MANIFEST_SHA,
        expected_global_step=98304,
        expected_next_step_digest=replay,
        component_artifacts=specs,
        fixture_label=fixture_label,
    )
    base.update(overrides)
    return FreshProcessRestoreRequest(**base)


def _component_row(name: str, tree: dict, source_path: str, pid: int,
                   origin: str | None = None) -> dict:
    records = tree_leaf_records(tree)
    if origin is None:
        origin = OPTIMIZER_ORIGIN_CHECKPOINT if name == "optimizer" else ORIGIN_SOURCE_ARTIFACT
    return {
        "component": name,
        "status": "RESTORED_HASH_BOUND",
        "origin": origin,
        "source_path": source_path,
        "pid": pid,
        "treedef": "dict{...}",  # structural string: content checked via leaves
        "leaf_count": len(records),
        "leaves": [rec.to_row() for rec in records],
        "leaves_digest": leaves_digest_of(records),
    }


def _valid_payload(request: FreshProcessRestoreRequest, *,
                   child_pid: int = 424242, parent_pid: int = 414141) -> dict:
    """Forge-free evidence payload shaped exactly like the worker's output
    (pure builder — no subprocess), for directed tampering tests."""
    components = [_component_row(name, _tree_for(name),
                                 request.component_artifacts[name].path, child_pid)
                  for name in REQUIRED_COMPONENTS]
    return {
        "schema": EVIDENCE_SCHEMA,
        "fixture_label": SYNTHETIC_FIXTURE_LABEL,
        "error": "",
        "process": {
            "child_pid": child_pid, "parent_pid": parent_pid,
            "argv": ["python", "-m", WORKER_MODULE, "--request", "r.json",
                     "--evidence", "e.json"],
            "started_at": "2026-08-04T00:00:00+00:00",
            "ended_at": "2026-08-04T00:00:05+00:00",
            "exit_code": 0, "worker_module": WORKER_MODULE,
        },
        "request_echo": {
            "checkpoint_path": request.checkpoint_path,
            "checkpoint_sha256": request.checkpoint_sha256,
            "student_abi_identity_hash": request.student_abi_identity_hash,
            "registry_hash": request.registry_hash,
            "manifest_hash": request.manifest_hash,
            "expected_global_step": request.expected_global_step,
            "optimizer_source": request.optimizer_source,
        },
        "components": components,
        "cross_checks": [{"name": CROSS_CHECKS[0], "status": "RESTORED_CROSS_VERIFIED",
                          "digest": request.expected_next_step_digest, "pid": child_pid}],
    }


def _verify(payload, request, *, allow_synthetic_fixture=True,
            launched_pid=424242, parent_pid=414141):
    return verify_fresh_process_evidence(
        payload, launched_pid=launched_pid, expected_parent_pid=parent_pid,
        request=request, allow_synthetic_fixture=allow_synthetic_fixture)


class TestRequestValidation:
    def test_valid_request_constructs(self, tmp_path):
        request = _fixture_request(tmp_path)
        assert request.optimizer_source == "checkpoint"
        assert request.fixture_label == SYNTHETIC_FIXTURE_LABEL
        payload = request.to_payload()
        assert payload["checkpoint_sha256"] == CHECKPOINT_SHA

    @pytest.mark.parametrize("field", [
        "checkpoint_sha256", "student_abi_identity_hash", "registry_hash",
        "manifest_hash", "expected_next_step_digest"])
    def test_missing_or_malformed_identity_hash_raises(self, tmp_path, field):
        with pytest.raises(InvalidEvidenceError):
            _fixture_request(tmp_path, **{field: "not-a-sha"})
        with pytest.raises(InvalidEvidenceError):
            _fixture_request(tmp_path, **{field: ""})

    def test_incomplete_component_artifacts_raise(self, tmp_path):
        artifacts = _write_artifacts(tmp_path)
        del artifacts["policy_memory"]
        digests = _digests(artifacts)
        specs = {name: ComponentArtifactSpec(path=info["path"], sha256=info["sha256"],
                                             expected_leaves_digest=digests[name])
                 for name, info in artifacts.items()}
        with pytest.raises(InvalidEvidenceError, match="policy_memory"):
            FreshProcessRestoreRequest(
                checkpoint_path="/x.pkl", checkpoint_sha256=CHECKPOINT_SHA,
                student_abi_identity_hash=ABI_SHA, registry_hash=REGISTRY_SHA,
                manifest_hash=MANIFEST_SHA, expected_global_step=1,
                expected_next_step_digest=NEXT_STEP_SHA, component_artifacts=specs)

    def test_tx_init_optimizer_source_rejected_at_construction(self, tmp_path):
        with pytest.raises(InvalidEvidenceError, match="checkpoint leaves"):
            _fixture_request(tmp_path, optimizer_source="tx_init")

    def test_unknown_fixture_label_raises(self, tmp_path):
        with pytest.raises(InvalidEvidenceError):
            _fixture_request(tmp_path, fixture_label="FIXTURE")

    def test_bad_expected_leaves_digest_raises(self, tmp_path):
        with pytest.raises(InvalidEvidenceError):
            ComponentArtifactSpec(path="/a.json", sha256="a" * 64,
                                  expected_leaves_digest="zzz")


class TestMechanicalVerification:
    """Directed tampering tests against verify_fresh_process_evidence."""

    def test_valid_payload_verifies(self, tmp_path):
        request = _fixture_request(tmp_path)
        evidence = _verify(_valid_payload(request), request)
        assert isinstance(evidence, ProcessEvidence)
        assert tuple(c.component for c in evidence.components) == REQUIRED_COMPONENTS
        assert evidence.exit_code == 0
        assert evidence.child_pid != evidence.parent_pid

    def test_parent_process_execution_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["process"]["child_pid"] = 414141  # == parent PID
        with pytest.raises(InvalidEvidenceError, match="parent-process execution rejected"):
            _verify(payload, request)

    def test_foreign_process_chain_rejected(self, tmp_path):
        # evidence whose parent PID matches NEITHER the driver PID nor the
        # launched PID comes from a process outside this invocation
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)  # evidence parent_pid=414141
        with pytest.raises(InvalidEvidenceError, match="split/forged process chain rejected"):
            _verify(payload, request, parent_pid=777777, launched_pid=999999)

    def test_evidence_parent_may_be_launched_pid(self, tmp_path):
        # Windows venv launcher chain: driver -> launcher(Popen.pid) ->
        # real interpreter; the child's parent is the LAUNCHED pid
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["process"]["parent_pid"] = 424242  # == launched_pid
        evidence = _verify(payload, request)
        assert evidence.parent_pid == 424242

    def test_empty_argv_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["process"]["argv"] = []
        with pytest.raises(InvalidEvidenceError, match="argv"):
            _verify(payload, request)

    def test_nonzero_exit_code_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["process"]["exit_code"] = 1
        with pytest.raises(InvalidEvidenceError, match="exit_code"):
            _verify(payload, request)

    def test_torn_timestamps_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["process"]["ended_at"] = "2026-08-03T23:59:00+00:00"
        with pytest.raises(InvalidEvidenceError, match="ended_at precedes started_at"):
            _verify(payload, request)
        payload2 = _valid_payload(request)
        payload2["process"]["started_at"] = "not-a-time"
        with pytest.raises(InvalidEvidenceError, match="started_at invalid"):
            _verify(payload2, request)

    def test_request_echo_mismatch_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["request_echo"]["checkpoint_sha256"] = "9" * 64
        with pytest.raises(InvalidEvidenceError, match="checkpoint_sha256"):
            _verify(payload, request)

    @pytest.mark.parametrize("fake_status", [
        "RESTORED", "PASSED", "OK", "NOT_EXECUTED", ""])
    def test_self_asserted_restored_status_rejected(self, tmp_path, fake_status):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["components"][0]["status"] = fake_status
        with pytest.raises(InvalidEvidenceError, match="self-asserted status"):
            _verify(payload, request)

    def test_missing_component_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["components"] = [c for c in payload["components"]
                                 if c["component"] != "history"]
        with pytest.raises(InvalidEvidenceError, match="missing required components"):
            _verify(payload, request)

    def test_split_process_composition_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["components"][0]["pid"] = 31337  # env/params from ANOTHER process
        with pytest.raises(InvalidEvidenceError, match="split-process composition rejected"):
            _verify(payload, request)

    def test_split_cross_check_pid_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["cross_checks"][0]["pid"] = 31337
        with pytest.raises(InvalidEvidenceError, match="split-process composition rejected"):
            _verify(payload, request)

    def test_missing_leaf_rows_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["components"][0]["leaves"] = payload["components"][0]["leaves"][:-1]
        with pytest.raises(InvalidEvidenceError, match="leaf_count"):
            _verify(payload, request)

    def test_reordered_leaves_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        leaves = payload["components"][0]["leaves"]
        leaves[1], leaves[2] = leaves[2], leaves[1]  # order values no longer 0..n-1
        with pytest.raises(InvalidEvidenceError, match="order gap/reorder"):
            _verify(payload, request)

    @pytest.mark.parametrize("tamper", ["shape", "dtype", "value"])
    def test_shape_dtype_value_tampering_rejected(self, tmp_path, tamper):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        row = payload["components"][0]["leaves"][0]
        if tamper == "shape":
            row[1] = [1]
        elif tamper == "dtype":
            row[2] = "ndarray:float32"
        else:
            row[3] = "f" * 64
        with pytest.raises(InvalidEvidenceError, match="tampering rejected"):
            _verify(payload, request)

    def test_digest_self_consistent_but_wrong_vs_checkpoint_rejected(self, tmp_path):
        # attacker recomputes leaves_digest after tampering: the recorded
        # digest stays self-consistent, but the REQUEST expectation (bound to
        # the checkpoint) still rejects it
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        comp = payload["components"][0]
        comp["leaves"][0][3] = "f" * 64
        rows = comp["leaves"]
        from dicode.simulator_frontier.fresh_process_restore import LeafRecord
        rebuilt = tuple(LeafRecord(order=r[0], shape=tuple(r[1]), dtype=r[2],
                                   value_sha256=r[3]) for r in rows)
        comp["leaves_digest"] = leaves_digest_of(rebuilt)
        comp["leaf_count"] = len(rows)
        with pytest.raises(InvalidEvidenceError, match="request expectation"):
            _verify(payload, request)

    def test_tx_init_substitution_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        optimizer = next(c for c in payload["components"] if c["component"] == "optimizer")
        optimizer["origin"] = OPTIMIZER_ORIGIN_TX_INIT
        with pytest.raises(InvalidEvidenceError, match="tx.init"):
            _verify(payload, request)

    def test_wrong_source_path_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["components"][0]["source_path"] = "/somewhere/else.json"
        with pytest.raises(InvalidEvidenceError, match="authoritative"):
            _verify(payload, request)

    def test_replay_digest_mismatch_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["cross_checks"][0]["digest"] = "e" * 64
        with pytest.raises(InvalidEvidenceError, match="replay diverged"):
            _verify(payload, request)

    def test_missing_cross_check_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["cross_checks"] = []
        with pytest.raises(InvalidEvidenceError, match="missing from evidence"):
            _verify(payload, request)

    def test_production_rejects_synthetic_evidence(self, tmp_path):
        request = _fixture_request(tmp_path, fixture_label="")
        payload = _valid_payload(request)  # carries the SYNTHETIC fixture label
        with pytest.raises(InvalidEvidenceError, match="production path rejects synthetic"):
            _verify(payload, request, allow_synthetic_fixture=False)

    def test_wrong_schema_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["schema"] = "something/else"
        with pytest.raises(InvalidEvidenceError, match="schema"):
            _verify(payload, request)


class TestSpawnedDriver:
    """End-to-end tests that spawn the real child process."""

    def test_labelled_synthetic_subprocess_positive_contract(self, tmp_path):
        # THE positive contract test: one REAL subprocess restores labelled
        # synthetic artifacts and its atomic evidence verifies mechanically.
        request = _fixture_request(tmp_path)
        outcome = run_fresh_process_restore(
            request, allow_synthetic_fixture=True, scratch_dir=tmp_path, timeout_s=90.0)
        assert outcome.accepted is True
        assert outcome.child_returncode == 0
        evidence = outcome.evidence
        assert evidence is not None
        assert evidence.child_pid != os.getpid(), "must run in a NEW process"
        # chain anchored to this invocation: the child's parent is either the
        # driver PID (direct spawn) or the launched PID (on this Windows host
        # Popen.pid is the venv launcher and the interpreter is its child)
        assert evidence.parent_pid in (os.getpid(), outcome.child_pid)
        assert any("restore_worker" in part for part in evidence.child_argv)
        assert evidence.exit_code == 0
        assert evidence.fixture_label == SYNTHETIC_FIXTURE_LABEL
        assert tuple(c.component for c in evidence.components) == REQUIRED_COMPONENTS
        assert evidence.component_map()["optimizer"].origin == OPTIMIZER_ORIGIN_CHECKPOINT
        assert evidence.cross_checks[0].digest == request.expected_next_step_digest
        assert evidence.request_echo["checkpoint_sha256"] == CHECKPOINT_SHA
        assert evidence.request_echo["registry_hash"] == REGISTRY_SHA
        assert evidence.request_echo["manifest_hash"] == MANIFEST_SHA
        # honest semantics: a synthetic contract pass is NOT a joint proof
        assert "FRESH_PROCESS_DRIVER_CONTRACT_PASS" in outcome.joint_proof_status
        assert "COMBINED_FRESH_PROCESS_RESTORE=true" not in outcome.joint_proof_status

    def test_production_driver_rejects_synthetic_request(self, tmp_path):
        request = _fixture_request(tmp_path)  # fixture_label=SYNTHETIC
        with pytest.raises(InvalidEvidenceError, match="synthetic"):
            run_fresh_process_restore_production(request, scratch_dir=tmp_path)

    def test_production_driver_has_no_callback_surface(self):
        # mechanically: there is no restorers/cross_checkers/callbacks knob
        params = set(inspect.signature(run_fresh_process_restore_production).parameters)
        assert params == {"request", "scratch_dir", "timeout_s"}
        assert CALLBACK_DRIVER_IS_CONTRACT_ONLY is True

    def test_crashed_child_report_rejected(self, tmp_path):
        request = _fixture_request(tmp_path)
        # corrupt one authoritative artifact AFTER the hash was bound: the
        # child must fail closed (non-zero exit), never emit green evidence
        first = next(iter(request.component_artifacts.values()))
        with open(first.path, "ab") as handle:
            handle.write(b"CORRUPTED")
        outcome = run_fresh_process_restore(
            request, allow_synthetic_fixture=True, scratch_dir=tmp_path, timeout_s=90.0)
        assert outcome.accepted is False
        assert outcome.child_returncode != 0
        assert outcome.evidence is None
        assert "COMBINED_FRESH_PROCESS_RESTORE=false" in outcome.joint_proof_status

    def test_production_without_registered_loaders_is_blocked(self, tmp_path):
        # honest this-round state: real production path has no controller
        # loaders/replay bound -> the child exits non-zero (BLOCKED)
        clear_production_registrations()
        request = _fixture_request(tmp_path, fixture_label="")
        outcome = run_fresh_process_restore_production(request, scratch_dir=tmp_path,
                                                       timeout_s=90.0)
        assert outcome.accepted is False
        assert outcome.child_returncode != 0
        assert "COMBINED_FRESH_PROCESS_RESTORE=false" in outcome.joint_proof_status

    def test_torn_evidence_file_rejected(self, tmp_path):
        torn = tmp_path / "torn_evidence.json"
        torn.write_text('{"schema": "simulator_frontier.fresh_process_restore_',
                        encoding="utf-8")
        with pytest.raises(InvalidEvidenceError, match="torn"):
            load_evidence_payload(torn)

    def test_missing_evidence_file_rejected(self, tmp_path):
        with pytest.raises(InvalidEvidenceError, match="missing"):
            load_evidence_payload(tmp_path / "no_such_evidence.json")


class TestHonestyAndComposition:
    def test_round_flags_are_honest(self):
        assert FRESH_PROCESS_DRIVER_CONTRACT_READY is True
        assert COMBINED_FRESH_PROCESS_RESTORE_EXECUTED is False
        status = production_registrations_status()
        assert status["loaders_bound"] == []
        assert status["replay_bound"] is False
        assert status["combined_fresh_process_restore_executed"] is False

    def test_production_registration_surface_fail_closed(self):
        clear_production_registrations()
        with pytest.raises(InvalidEvidenceError):
            register_production_component_loader("not_a_component", lambda spec: {})
        with pytest.raises(InvalidEvidenceError):
            register_production_component_loader("params", "not callable")
        register_production_component_loader("params", lambda spec: {})
        assert production_registrations_status()["loaders_bound"] == ["params"]
        with pytest.raises(InvalidEvidenceError):
            register_production_component_loader("params", lambda spec: {})
        register_production_replay(lambda digests, echo: "0" * 64)
        assert production_registrations_status()["replay_bound"] is True
        with pytest.raises(InvalidEvidenceError):
            register_production_replay(lambda digests, echo: "0" * 64)
        clear_production_registrations()
        assert production_registrations_status()["loaders_bound"] == []
        assert production_registrations_status()["replay_bound"] is False

    def _all_restored_verdict(self):
        components = {name: ComponentResult(name, ComponentStatus.RESTORED_HASH_BOUND, "ok")
                      for name in REQUIRED_COMPONENTS}
        cross = {name: ComponentResult(name, ComponentStatus.RESTORED_CROSS_VERIFIED, "ok")
                 for name in CROSS_CHECKS}
        return evaluate_verdict(components, cross)

    def _evidence(self, *, fixture_label: str) -> ProcessEvidence:
        return ProcessEvidence(
            schema=EVIDENCE_SCHEMA, fixture_label=fixture_label,
            child_pid=424242, parent_pid=414141,
            child_argv=("python", "-m", WORKER_MODULE),
            started_at="2026-08-04T00:00:00+00:00",
            ended_at="2026-08-04T00:00:05+00:00", exit_code=0,
            worker_module=WORKER_MODULE, request_echo={}, components=(),
            cross_checks=())

    def test_synthetic_evidence_can_never_upgrade_the_joint_flag(self):
        verdict = self._all_restored_verdict()
        assert verdict.combined_pass is True
        assert production_joint_pass(verdict, self._evidence(
            fixture_label=SYNTHETIC_FIXTURE_LABEL)) is False

    def test_production_evidence_composes_with_verdict(self):
        verdict = self._all_restored_verdict()
        assert production_joint_pass(verdict, self._evidence(fixture_label="")) is True
        # a verdict with a FAILED component never composes, evidence or not
        components = {name: ComponentResult(name, ComponentStatus.RESTORED_HASH_BOUND, "ok")
                      for name in REQUIRED_COMPONENTS}
        components["history"] = ComponentResult("history", ComponentStatus.FAILED, "boom")
        cross = {name: ComponentResult(name, ComponentStatus.RESTORED_CROSS_VERIFIED, "ok")
                 for name in CROSS_CHECKS}
        failed = evaluate_verdict(components, cross)
        assert production_joint_pass(failed, self._evidence(fixture_label="")) is False

    def test_unverified_evidence_argument_raises(self):
        verdict = self._all_restored_verdict()
        with pytest.raises(InvalidEvidenceError, match="verified ProcessEvidence"):
            production_joint_pass(verdict, {"self_asserted": True})
