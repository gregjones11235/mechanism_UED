"""Mechanical-enforcement tests for the R4c production fresh-process driver.

Independent audit closure (2026-08-04): the callback-based contract driver
(combined_restore_contract.run_combined_restore) runs in the CURRENT process
with self-asserted ComponentResult statuses.  These tests pin the mechanical
replacements: exactly ONE spawned child process, atomic PID/PPID/argv/
timestamp/exit-code evidence bound to the actually spawned child,
authoritative per-component leaf count/order/shape/dtype/value hashes,
checkpoint-leaf optimizer binding (tx.init substitution rejected), and
split-process / crash / torn-report rejection.

Audit follow-up (2026-08-04, round 2): the parent-global loader/replay
registry is GONE (a spawned child imports a fresh module where parent
globals never exist).  The child now resolves loaders/replay ONLY through
the immutable controller-signed ProductionRegistryBundle nested in — and
hash-bound to — the request.  These tests prove: parent-process callback
state can never reach the child; explicit child binding succeeds only with
exact registry/ABI/manifest hashes; request/evidence binding includes the
bundle hash; and production_joint_pass refuses any verdict that does not
correspond to the SAME verified ProcessEvidence.

The positive tests spawn REAL subprocesses but restore labelled synthetic
artifacts only (SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT): they prove the
enforcement contract, NOT a real combined restore —
COMBINED_FRESH_PROCESS_RESTORE_EXECUTED stays False this round.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os

import pytest

from dicode.simulator_frontier import (
    BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE,
    BUNDLE_SCHEMA,
    CALLBACK_DRIVER_IS_CONTRACT_ONLY,
    COMBINED_FRESH_PROCESS_RESTORE_EXECUTED,
    CONTROLLER_SIGNED_REGISTRY_BUNDLE_BOUND,
    FRESH_PROCESS_DRIVER_CONTRACT_READY,
    OPTIMIZER_ORIGIN_CHECKPOINT,
    OPTIMIZER_ORIGIN_TX_INIT,
    PRODUCTION_REGISTRY_BUNDLE_CONTRACT_READY,
    SYNTHETIC_FIXTURE_LABEL,
    SYNTHETIC_SIGNATURE_PREFIX,
    REQUIRED_COMPONENTS,
    CROSS_CHECKS,
    CombinedRestoreVerdict,
    ComponentResult,
    ComponentStatus,
    ComponentArtifactSpec,
    FreshProcessRestoreRequest,
    LoaderEntryPoint,
    ProcessEvidence,
    ProductionRegistryBundle,
    evaluate_verdict,
    leaves_digest_of,
    load_evidence_payload,
    production_joint_pass,
    run_fresh_process_restore,
    run_fresh_process_restore_production,
    synthetic_replay_digest,
    tree_leaf_records,
    verdict_from_evidence,
    verify_controller_signature,
    verify_fresh_process_evidence,
)
from dicode.simulator_frontier import fresh_process_restore as fpr_module
from dicode.simulator_frontier import restore_fixture_loaders as fixture_loaders
from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.fresh_process_restore import (
    ARTIFACT_SCHEMA,
    EVIDENCE_SCHEMA,
    ORIGIN_SOURCE_ARTIFACT,
    WORKER_MODULE,
    restore_worker_main,
)

CHECKPOINT_SHA = "1" * 64
ABI_SHA = "2" * 64
MANIFEST_SHA = "4" * 64
NEXT_STEP_SHA = "5" * 64

FIXTURE_LOADER_MODULE = "dicode.simulator_frontier.restore_fixture_loaders"

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


def _registry_bundle(fixture_label: str = SYNTHETIC_FIXTURE_LABEL, *,
                     abi_hash: str = ABI_SHA, manifest_hash: str = MANIFEST_SHA,
                     signature: str | None = None, drop_entry: tuple = (),
                     extra_entry: tuple | None = None,
                     replay_point: LoaderEntryPoint | None = None,
                     registry_id: str = "SYNTHETIC_CONTRACT_REGISTRY",
                     ) -> ProductionRegistryBundle:
    """Bundle whose entry points name the LABELLED synthetic fixture loaders.

    SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT: this bundle (and the modules it
    points at) exists only to exercise the child's bundle-driven resolution.
    """
    entry = LoaderEntryPoint(entry_module=FIXTURE_LOADER_MODULE,
                             entry_attr="load_synthetic_component")
    points = {name: entry for name in REQUIRED_COMPONENTS if name not in drop_entry}
    if extra_entry is not None:
        points[extra_entry[0]] = extra_entry[1]
    if signature is None:
        signature = (SYNTHETIC_SIGNATURE_PREFIX + "NOT_CONTROLLER_ISSUED"
                     if fixture_label == SYNTHETIC_FIXTURE_LABEL
                     else "CONTROLLER_SIGNATURE_PENDING_VERIFICATION")
    replay = replay_point or LoaderEntryPoint(entry_module=FIXTURE_LOADER_MODULE,
                                              entry_attr="synthetic_fixture_replay")
    return ProductionRegistryBundle(
        registry_id=registry_id, controller_signature_ref=signature,
        student_abi_identity_hash=abi_hash, manifest_hash=manifest_hash,
        loader_entry_points=points, replay_entry_point=replay)


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
    bundle = _registry_bundle(fixture_label)
    base = dict(
        checkpoint_path="/synthetic/contract_checkpoint.pkl",
        checkpoint_sha256=CHECKPOINT_SHA,
        student_abi_identity_hash=ABI_SHA,
        registry_hash=bundle.bundle_sha256(),
        manifest_hash=MANIFEST_SHA,
        expected_global_step=98304,
        expected_next_step_digest=replay,
        component_artifacts=specs,
        registry_bundle=bundle,
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
                   child_pid: int = 424242, parent_pid: int = 414141,
                   fixture_label: str = SYNTHETIC_FIXTURE_LABEL) -> dict:
    """Forge-free evidence payload shaped exactly like the worker's output
    (pure builder — no subprocess), for directed tampering tests."""
    components = [_component_row(name, _tree_for(name),
                                 request.component_artifacts[name].path, child_pid)
                  for name in REQUIRED_COMPONENTS]
    return {
        "schema": EVIDENCE_SCHEMA,
        "fixture_label": fixture_label,
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
            "registry_bundle_sha256": request.registry_bundle.bundle_sha256(),
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
        # hash-bound: registry_hash IS the bundle hash
        assert request.registry_hash == request.registry_bundle.bundle_sha256()
        payload = request.to_payload()
        assert payload["checkpoint_sha256"] == CHECKPOINT_SHA
        assert payload["registry_bundle"]["schema"] == BUNDLE_SCHEMA

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
        bundle = _registry_bundle(SYNTHETIC_FIXTURE_LABEL)
        with pytest.raises(InvalidEvidenceError, match="policy_memory"):
            FreshProcessRestoreRequest(
                checkpoint_path="/x.pkl", checkpoint_sha256=CHECKPOINT_SHA,
                student_abi_identity_hash=ABI_SHA,
                registry_hash=bundle.bundle_sha256(),
                manifest_hash=MANIFEST_SHA, expected_global_step=1,
                expected_next_step_digest=NEXT_STEP_SHA, component_artifacts=specs,
                registry_bundle=bundle)

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


class TestRegistryBundleBinding:
    """The child's ONLY resolution surface: hash-bound, exact-hash, signed."""

    def test_bundle_must_cover_every_component_entry_point(self):
        with pytest.raises(InvalidEvidenceError, match="history"):
            _registry_bundle(drop_entry=("history",))

    def test_bundle_rejects_unknown_component_entry(self):
        entry = LoaderEntryPoint(entry_module=FIXTURE_LOADER_MODULE,
                                 entry_attr="load_synthetic_component")
        with pytest.raises(InvalidEvidenceError, match="unknown components"):
            _registry_bundle(extra_entry=("bogus_component", entry))

    def test_bundle_rejects_forbidden_entry_module_namespace(self):
        with pytest.raises(InvalidEvidenceError, match="forbidden"):
            LoaderEntryPoint(entry_module="subprocess", entry_attr="run")
        with pytest.raises(InvalidEvidenceError, match="forbidden"):
            LoaderEntryPoint(entry_module="os.path", entry_attr="join")

    def test_bundle_rejects_non_identifier_entry_attr(self):
        with pytest.raises(InvalidEvidenceError, match="identifier"):
            LoaderEntryPoint(entry_module=FIXTURE_LOADER_MODULE,
                             entry_attr="not an attr")

    def test_bundle_requires_controller_signature_reference(self):
        with pytest.raises(InvalidEvidenceError, match="controller_signature_ref"):
            _registry_bundle(signature="")

    def test_bundle_hash_deterministic_roundtrip_and_schema(self):
        bundle_a = _registry_bundle()
        bundle_b = _registry_bundle()
        assert bundle_a.bundle_sha256() == bundle_b.bundle_sha256()
        payload = bundle_a.to_payload()
        restored = ProductionRegistryBundle.from_payload(payload)
        assert restored.bundle_sha256() == bundle_a.bundle_sha256()
        tampered = json.loads(json.dumps(payload))
        tampered["registry_id"] = "OTHER_REGISTRY"
        assert ProductionRegistryBundle.from_payload(tampered).bundle_sha256() \
            != bundle_a.bundle_sha256()
        bad_schema = json.loads(json.dumps(payload))
        bad_schema["schema"] = "something/else"
        with pytest.raises(InvalidEvidenceError, match="schema"):
            ProductionRegistryBundle.from_payload(bad_schema)

    def test_request_must_be_hash_bound_to_the_bundle(self, tmp_path):
        # valid 64-hex but NOT the bundle hash -> rejected fail closed
        with pytest.raises(InvalidEvidenceError, match="hash-bound"):
            _fixture_request(tmp_path, registry_hash="a" * 64)

    def test_request_rejects_bundle_abi_mismatch(self, tmp_path):
        artifacts = _write_artifacts(tmp_path)
        digests = _digests(artifacts)
        specs = {name: ComponentArtifactSpec(path=info["path"], sha256=info["sha256"],
                                             expected_leaves_digest=digests[name])
                 for name, info in artifacts.items()}
        bundle = _registry_bundle(abi_hash="9" * 64)
        with pytest.raises(InvalidEvidenceError, match="student_abi_identity_hash"):
            FreshProcessRestoreRequest(
                checkpoint_path="/x.pkl", checkpoint_sha256=CHECKPOINT_SHA,
                student_abi_identity_hash=ABI_SHA,
                registry_hash=bundle.bundle_sha256(),
                manifest_hash=MANIFEST_SHA, expected_global_step=1,
                expected_next_step_digest=NEXT_STEP_SHA, component_artifacts=specs,
                registry_bundle=bundle)

    def test_request_rejects_bundle_manifest_mismatch(self, tmp_path):
        artifacts = _write_artifacts(tmp_path)
        digests = _digests(artifacts)
        specs = {name: ComponentArtifactSpec(path=info["path"], sha256=info["sha256"],
                                             expected_leaves_digest=digests[name])
                 for name, info in artifacts.items()}
        bundle = _registry_bundle(manifest_hash="9" * 64)
        with pytest.raises(InvalidEvidenceError, match="manifest_hash"):
            FreshProcessRestoreRequest(
                checkpoint_path="/x.pkl", checkpoint_sha256=CHECKPOINT_SHA,
                student_abi_identity_hash=ABI_SHA,
                registry_hash=bundle.bundle_sha256(),
                manifest_hash=MANIFEST_SHA, expected_global_step=1,
                expected_next_step_digest=NEXT_STEP_SHA, component_artifacts=specs,
                registry_bundle=bundle)

    def test_labelled_request_requires_synthetic_signature(self, tmp_path):
        bundle = _registry_bundle(fixture_label=SYNTHETIC_FIXTURE_LABEL,
                                  signature="CONTROLLER_SIGNATURE_REAL_LOOKING")
        with pytest.raises(InvalidEvidenceError, match="cross-binding"):
            _fixture_request(tmp_path, registry_bundle=bundle,
                             registry_hash=bundle.bundle_sha256())

    def test_production_intent_rejects_synthetic_signature(self, tmp_path):
        bundle = _registry_bundle(fixture_label="")  # synthetic-signature bundle
        bundle = ProductionRegistryBundle(
            registry_id=bundle.registry_id,
            controller_signature_ref=SYNTHETIC_SIGNATURE_PREFIX + "FORGED",
            student_abi_identity_hash=ABI_SHA, manifest_hash=MANIFEST_SHA,
            loader_entry_points=bundle.loader_entry_points,
            replay_entry_point=bundle.replay_entry_point)
        with pytest.raises(InvalidEvidenceError, match="synthetic bundle signatures"):
            _fixture_request(tmp_path, fixture_label="", registry_bundle=bundle,
                             registry_hash=bundle.bundle_sha256())

    def test_verify_controller_signature_fail_closed_this_round(self):
        pending = _registry_bundle(fixture_label="")
        with pytest.raises(InvalidEvidenceError,
                           match=BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE):
            verify_controller_signature(pending)
        synthetic = _registry_bundle(fixture_label=SYNTHETIC_FIXTURE_LABEL)
        with pytest.raises(InvalidEvidenceError, match="synthetic controller signature"):
            verify_controller_signature(synthetic)

    @pytest.mark.parametrize("tamper, recompute, expected_error", [
        ("registry_id", False, "hash-bound"),
        ("student_abi_identity_hash", True, "student_abi_identity_hash"),
        ("manifest_hash", True, "manifest_hash"),
    ])
    def test_child_re_verifies_bundle_binding(self, tmp_path, tamper, recompute,
                                              expected_error):
        """Child-side re-binding: a bundle tampered after hash binding is
        rejected inside the child (in-process worker unit; the fresh-process
        discipline itself is proven by the spawned-driver tests)."""
        request = _fixture_request(tmp_path)
        payload = request.to_payload()
        payload["registry_bundle"][tamper] = "9" * 64 if tamper != "registry_id" \
            else "TAMPERED_REGISTRY"
        if recompute:
            tampered_bundle = ProductionRegistryBundle.from_payload(
                payload["registry_bundle"])
            payload["registry_hash"] = tampered_bundle.bundle_sha256()
        req_path = tmp_path / "tampered_request.json"
        ev_path = tmp_path / "tampered_evidence.json"
        req_path.write_text(_canonical(payload), encoding="utf-8")
        exit_code = restore_worker_main(
            ["restore_worker", "--request", str(req_path), "--evidence", str(ev_path)])
        assert exit_code == 4
        evidence = json.loads(ev_path.read_text(encoding="utf-8"))
        assert expected_error in evidence["error"]

    def test_child_blocks_unsigned_production_bundle(self, tmp_path):
        """Honest this-round state: an unlabelled bundle with a pending
        controller signature is blocked by the child's signature gate."""
        request = _fixture_request(tmp_path, fixture_label="")
        req_path = tmp_path / "production_request.json"
        ev_path = tmp_path / "production_evidence.json"
        req_path.write_text(_canonical(request.to_payload()), encoding="utf-8")
        exit_code = restore_worker_main(
            ["restore_worker", "--request", str(req_path), "--evidence", str(ev_path)])
        assert exit_code == 4
        evidence = json.loads(ev_path.read_text(encoding="utf-8"))
        assert BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE in evidence["error"]


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

    def test_registry_bundle_hash_echo_mismatch_rejected(self, tmp_path):
        # evidence must echo the SAME bundle hash the request is bound to
        request = _fixture_request(tmp_path)
        payload = _valid_payload(request)
        payload["request_echo"]["registry_bundle_sha256"] = "c" * 64
        with pytest.raises(InvalidEvidenceError, match="registry_bundle_sha256"):
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
        # synthetic artifacts via BUNDLE-DRIVEN entry points and its atomic
        # evidence verifies mechanically.
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
        assert evidence.request_echo["manifest_hash"] == MANIFEST_SHA
        # REQUEST/EVIDENCE BINDING through the bundle hash:
        assert evidence.request_echo["registry_hash"] == request.registry_hash
        assert evidence.request_echo["registry_bundle_sha256"] \
            == request.registry_bundle.bundle_sha256() == request.registry_hash
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

    def test_parent_process_callback_state_never_reaches_child(self, tmp_path):
        """PROOF OF NO PARENT CALLBACK REUSE: swapping the loader/replay
        callables in the PARENT process has zero effect on the run — the
        child imports the entry modules fresh and resolves them through the
        bundle only.  Were the child reusing parent-process callbacks or
        module state, the forged loader/replay below would change the
        digests (or diverge the replay) and the run would fail."""
        request = _fixture_request(tmp_path)
        original_loader = fixture_loaders.load_synthetic_component
        original_replay = fixture_loaders.synthetic_fixture_replay
        fixture_loaders.load_synthetic_component = \
            lambda context: {"forged": ["parent", "callback", "state"]}
        fixture_loaders.synthetic_fixture_replay = lambda context: "0" * 64
        try:
            outcome = run_fresh_process_restore(
                request, allow_synthetic_fixture=True, scratch_dir=tmp_path,
                timeout_s=90.0)
        finally:
            fixture_loaders.load_synthetic_component = original_loader
            fixture_loaders.synthetic_fixture_replay = original_replay
        assert outcome.accepted is True, outcome.joint_proof_status
        expected = {name: leaves_digest_of(tree_leaf_records(_tree_for(name)))
                    for name in REQUIRED_COMPONENTS}
        for comp in outcome.evidence.components:
            assert comp.leaves_digest == expected[comp.component], \
                f"component {comp.component} was restored from parent state!"
        assert outcome.evidence.cross_checks[0].digest == \
            synthetic_replay_digest(expected)

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

    def test_production_without_controller_signed_bundle_is_blocked(self, tmp_path):
        """Honest this-round state: the production entry point runs, the
        child re-verifies the bundle, and the controller-signature gate
        fails closed (no verification material bound) -> BLOCKED, never a
        green joint flag."""
        request = _fixture_request(tmp_path, fixture_label="")
        outcome = run_fresh_process_restore_production(request, scratch_dir=tmp_path,
                                                       timeout_s=90.0)
        assert outcome.accepted is False
        assert outcome.child_returncode == 4
        assert "COMBINED_FRESH_PROCESS_RESTORE=false" in outcome.joint_proof_status
        assert BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE in "; ".join(
            outcome.violations)

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
        assert PRODUCTION_REGISTRY_BUNDLE_CONTRACT_READY is True
        assert CONTROLLER_SIGNED_REGISTRY_BUNDLE_BOUND is False
        assert CALLBACK_DRIVER_IS_CONTRACT_ONLY is True

    def test_no_parent_global_loader_registry_surface_exists(self):
        """Structural proof: the dead parent-global loader/replay registry
        (which a spawned child could never see) is gone from the module —
        there is no parent callback reuse surface left to bypass."""
        for removed in ("_PRODUCTION_LOADERS", "_PRODUCTION_REPLAY",
                        "register_production_component_loader",
                        "register_production_replay",
                        "clear_production_registrations",
                        "production_registrations_status"):
            assert not hasattr(fpr_module, removed), \
                f"parent-global surface {removed} must stay removed"

    def _verified_evidence(self, tmp_path, *, fixture_label: str):
        request = _fixture_request(tmp_path, fixture_label=fixture_label)
        payload = _valid_payload(request, fixture_label=fixture_label)
        return _verify(payload, request,
                       allow_synthetic_fixture=(fixture_label != ""))

    def test_synthetic_evidence_can_never_upgrade_the_joint_flag(self, tmp_path):
        evidence = self._verified_evidence(tmp_path,
                                           fixture_label=SYNTHETIC_FIXTURE_LABEL)
        verdict = verdict_from_evidence(evidence)
        assert verdict.combined_pass is True
        assert production_joint_pass(verdict, evidence) is False

    def test_production_evidence_composes_with_its_own_verdict(self, tmp_path):
        evidence = self._verified_evidence(tmp_path, fixture_label="")
        verdict = verdict_from_evidence(evidence)
        assert production_joint_pass(verdict, evidence) is True

    def test_fabricated_verdicts_cannot_compose(self, tmp_path):
        """An independently fabricated CombinedRestoreVerdict — self-asserted
        statuses with no digest binding, wrong digests, or missing components
        — never passes production_joint_pass against real evidence."""
        evidence = self._verified_evidence(tmp_path, fixture_label="")
        # (a) old-style verdict: statuses asserted, no bound digests
        asserted = evaluate_verdict(
            {name: ComponentResult(name, ComponentStatus.RESTORED_HASH_BOUND, "ok")
             for name in REQUIRED_COMPONENTS},
            {name: ComponentResult(name, ComponentStatus.RESTORED_CROSS_VERIFIED, "ok")
             for name in CROSS_CHECKS})
        assert asserted.combined_pass is True
        assert production_joint_pass(asserted, evidence) is False
        # (b) verdict with one component bound to the WRONG digest
        wrong = evaluate_verdict(
            {name: ComponentResult(name, ComponentStatus.RESTORED_HASH_BOUND, "ok",
                                   bound_digest=evidence.component_map()[name].leaves_digest
                                   if name != "history" else "f" * 64)
             for name in REQUIRED_COMPONENTS},
            {name: ComponentResult(name, ComponentStatus.RESTORED_CROSS_VERIFIED, "ok",
                                   bound_digest=evidence.cross_checks[0].digest)
             for name in CROSS_CHECKS})
        assert production_joint_pass(wrong, evidence) is False
        # (c) verdict missing one component entirely
        short = dict(verdict_from_evidence(evidence).components)
        del short["history"]
        incomplete = CombinedRestoreVerdict(
            components=short, cross_checks=verdict_from_evidence(evidence).cross_checks,
            combined_pass=True, env_only_pass=True, checkpoint_only_pass=True,
            joint_proof_status="fabricated")
        assert production_joint_pass(incomplete, evidence) is False
        # (d) verdict with a FAILED component never composes
        failed_components = {
            name: ComponentResult(name, ComponentStatus.RESTORED_HASH_BOUND, "ok",
                                  bound_digest=evidence.component_map()[name].leaves_digest)
            for name in REQUIRED_COMPONENTS}
        failed_components["history"] = ComponentResult("history", ComponentStatus.FAILED,
                                                       "boom")
        failed = evaluate_verdict(
            failed_components,
            {name: ComponentResult(name, ComponentStatus.RESTORED_CROSS_VERIFIED, "ok",
                                   bound_digest=evidence.cross_checks[0].digest)
             for name in CROSS_CHECKS})
        assert production_joint_pass(failed, evidence) is False

    def test_verdict_from_evidence_is_the_canonical_builder(self, tmp_path):
        evidence = self._verified_evidence(tmp_path, fixture_label="")
        verdict = verdict_from_evidence(evidence)
        assert verdict.combined_pass is True
        for name in REQUIRED_COMPONENTS:
            result = verdict.components[name]
            assert result.status.value == "RESTORED_HASH_BOUND"
            assert result.bound_digest == evidence.component_map()[name].leaves_digest
        replay_result = verdict.cross_checks[CROSS_CHECKS[0]]
        assert replay_result.bound_digest == evidence.cross_checks[0].digest

    def test_unverified_evidence_argument_raises(self):
        bundle = _registry_bundle(fixture_label="")
        verdict = CombinedRestoreVerdict(
            components={}, cross_checks={}, combined_pass=True,
            env_only_pass=True, checkpoint_only_pass=True, joint_proof_status="x")
        with pytest.raises(InvalidEvidenceError, match="verified ProcessEvidence"):
            production_joint_pass(verdict, {"self_asserted": True})
        assert bundle.registry_id  # bundle builder sanity
