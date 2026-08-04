"""P0-7 (§19 seam coverage): the shared-runtime bindings hash folds REAL
ASSET IDENTITIES — never status strings.

Contract under test:

* ``bindings_hash`` folds ``asset_identities()``: the registry-issued
  Student / Reference / ProbeRunner / AnchorManifest / Training identities
  plus the formal asset registry's own identity — two bundles bound to
  DIFFERENT real assets can never collide, even when their
  ``status_report()`` is byte-identical (the pre-P0-7 hash folded status
  strings and could not distinguish such bundles);
* registry-issued ONLY: the ProbeRunner and Training slots refuse assets
  that do not expose a sha256 ``registry_identity``; the formal registry
  identity refuses non-sha256 values; direction two never derives an
  identity locally;
* absence stays blocked: every missing asset folds the explicit
  ``ABSENT_NOT_REGISTRY_ISSUED`` sentinel into the identity map, and
  ``resolve_shared_runtime`` still fails closed with
  ``BLOCKED_WAITING_SHARED_RUNTIME`` and the full missing list;
* a slot smuggled into BOUND without an identity fails closed
  (SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: injected
contracts/runners are scripted SimpleNamespaces, NO shared runtime asset
actually exists in this worktree, NO real LLM call happens, and NO
passing test flips a REAL_* flag. Retry/repair exhaustion and
snapshot/restore angles are scoped out here — this seam has no retry
budget and the bundle is never snapshotted; rebuild determinism and
stale-slot immutability are covered instead.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.anchor_manifest import SharedAnchorManifest
from d052.feedback_llm_ued.shared_runtime_binding import (
    SLOT_ABSENT_IDENTITY,
    SLOT_ASSET_DESCRIPTIONS,
    STATUS_BOUND,
    SharedAnchorManifestSlot,
    SharedBindingRejected,
    SharedProbeRunnerSlot,
    SharedReferenceSlot,
    SharedRuntimeBlocked,
    SharedRuntimeBundle,
    SharedStudentSlot,
    SharedTrainingSlot,
    resolve_reference_binding,
    resolve_shared_runtime,
)
from d052.feedback_llm_ued.student_binding import StudentBindingBlocked

#: TEST_ONLY / SYNTHETIC registry-issued identities (NOT_REAL_EXECUTION)
RUNNER_REGISTRY_ID_A = text_sha256("TEST_ONLY_RUNNER_REGISTRY_IDENTITY_A")
RUNNER_REGISTRY_ID_B = text_sha256("TEST_ONLY_RUNNER_REGISTRY_IDENTITY_B")
TRAINING_REGISTRY_ID_A = text_sha256(
    "TEST_ONLY_TRAINING_REGISTRY_IDENTITY_A")
TRAINING_REGISTRY_ID_B = text_sha256(
    "TEST_ONLY_TRAINING_REGISTRY_IDENTITY_B")
FORMAL_REGISTRY_ID = text_sha256("TEST_ONLY_FORMAL_ASSET_REGISTRY_IDENTITY")
STUDENT_PARAM_HASH_A = text_sha256("TEST_ONLY_STUDENT_PARAM_TREE_A")
STUDENT_PARAM_HASH_B = text_sha256("TEST_ONLY_STUDENT_PARAM_TREE_B")
REFERENCE_PARAM_HASH_A = text_sha256("TEST_ONLY_REFERENCE_PARAM_TREE_A")
REFERENCE_PARAM_HASH_B = text_sha256("TEST_ONLY_REFERENCE_PARAM_TREE_B")

IDENTITY_SLOTS = ("student", "reference", "probe_runner", "anchor_manifest",
                  "training", "formal_registry")


def student_contract(parameter_tree_hash=STUDENT_PARAM_HASH_A):
    return SimpleNamespace(
        candidate_id=C.STRONG_STUDENT_CANDIDATE_ID,
        architecture_family="RMT16",
        memory_family="RMT16_ORIGINAL",
        carry_mode="PERSISTENT",
        parameter_tree_hash=parameter_tree_hash,
        checkpoint_global_step=98304)


def reference_contract(parameter_tree_hash=REFERENCE_PARAM_HASH_A):
    return SimpleNamespace(
        candidate_id="TEST_ONLY_REFERENCE_CANDIDATE",
        parameter_tree_hash=parameter_tree_hash,
        checkpoint_global_step=1024)


def probe_runner(*, registry_identity=RUNNER_REGISTRY_ID_A,
                 runner_id="TEST_ONLY_RUNNER_01", real_simulator=True):
    return SimpleNamespace(real_simulator=real_simulator,
                           runner_id=runner_id,
                           registry_identity=registry_identity)


def training_contract(registry_identity=TRAINING_REGISTRY_ID_A):
    return SimpleNamespace(
        run_one_optimizer_update=lambda **kw: None,
        save_checkpoint=lambda **kw: "TEST_ONLY_CHECKPOINT_HASH",
        load_checkpoint=lambda **kw: None,
        verify_full_state_round_trip=lambda **kw: None,
        registry_identity=registry_identity)


def anchor_manifest():
    return SharedAnchorManifest(
        manifest_id="TEST_ONLY_ANCHOR_MANIFEST",
        anchors=list(C.GLOBAL_CANONICAL_ANCHOR_IDS), frozen=True)


def alternate_anchor_manifest():
    return SharedAnchorManifest(
        manifest_id="TEST_ONLY_ANCHOR_MANIFEST_ALT",
        anchors=[f"TEST_ONLY_ALT_ANCHOR_{i}" for i in range(4)],
        frozen=True)


def fully_bound_bundle(*, student=None, reference=None, runner=None,
                       manifest=None, training=None) -> SharedRuntimeBundle:
    return SharedRuntimeBundle(
        student=SharedStudentSlot().bind(student or student_contract()),
        reference=SharedReferenceSlot().bind(
            reference or reference_contract()),
        probe_runner=SharedProbeRunnerSlot().bind(runner or probe_runner()),
        anchor_manifest=SharedAnchorManifestSlot().bind(
            manifest or anchor_manifest()),
        training=SharedTrainingSlot().bind(training or training_contract()))


class TestBindingsHashFoldsIdentitiesNotStatuses:
    def test_empty_bundle_folds_absent_sentinels(self):
        bundle = SharedRuntimeBundle()
        identities = bundle.asset_identities()
        assert sorted(identities) == sorted(IDENTITY_SLOTS)
        for name in IDENTITY_SLOTS:
            assert identities[name] == SLOT_ABSENT_IDENTITY, name
        #: deterministic: two fresh bundles hash identically
        assert bundle.bindings_hash() == SharedRuntimeBundle().bindings_hash()

    def test_fully_bound_bundle_folds_real_identities(self):
        manifest = anchor_manifest()
        bundle = fully_bound_bundle(manifest=manifest,
                                    runner=probe_runner(),
                                    training=training_contract())
        identities = bundle.asset_identities()
        assert identities["student"] == bundle.student.binding.identity_hash
        assert (identities["reference"]
                == bundle.reference.binding.identity_hash)
        assert identities["probe_runner"] == RUNNER_REGISTRY_ID_A
        assert identities["anchor_manifest"] == manifest.manifest_hash
        assert identities["training"] == TRAINING_REGISTRY_ID_A
        #: formal registry absent here -> explicit sentinel, never omitted
        assert identities["formal_registry"] == SLOT_ABSENT_IDENTITY
        #: no entry is a status string
        assert "BOUND" not in identities.values()
        assert SLOT_ABSENT_IDENTITY not in (
            identities["student"], identities["reference"],
            identities["probe_runner"], identities["anchor_manifest"],
            identities["training"])

    def test_identity_substitution_same_statuses_different_hash(self):
        #: THE P0-7 regression: two bundles, BOTH fully BOUND with a
        #: byte-identical status_report() (same runner_id, same constant
        #: slot details) but DIFFERENT registry-issued identities must
        #: produce DIFFERENT bindings hashes — the pre-P0-7 hash folded
        #: status_report() and collided here
        bundle_a = fully_bound_bundle(runner=probe_runner(),
                                      training=training_contract())
        bundle_b = fully_bound_bundle(
            runner=probe_runner(registry_identity=RUNNER_REGISTRY_ID_B),
            training=training_contract(
                registry_identity=TRAINING_REGISTRY_ID_B))
        assert bundle_a.status_report() == bundle_b.status_report()
        assert bundle_a.bindings_hash() != bundle_b.bindings_hash()

    def test_student_identity_substitution_changes_hash(self):
        bundle_a = fully_bound_bundle()
        bundle_b = fully_bound_bundle(
            student=student_contract(STUDENT_PARAM_HASH_B))
        #: same provenance-label detail string, different identity
        assert bundle_a.status_report() == bundle_b.status_report()
        assert bundle_a.bindings_hash() != bundle_b.bindings_hash()

    def test_reference_identity_substitution_changes_hash(self):
        bundle_a = fully_bound_bundle()
        bundle_b = fully_bound_bundle(
            reference=reference_contract(REFERENCE_PARAM_HASH_B))
        assert bundle_a.status_report() == bundle_b.status_report()
        assert bundle_a.bindings_hash() != bundle_b.bindings_hash()

    def test_anchor_manifest_identity_change_changes_hash(self):
        bundle_a = fully_bound_bundle()
        bundle_b = fully_bound_bundle(manifest=alternate_anchor_manifest())
        assert bundle_a.status_report() == bundle_b.status_report()
        assert bundle_a.bindings_hash() != bundle_b.bindings_hash()

    def test_formal_registry_identity_folded(self):
        bundle = fully_bound_bundle()
        registered = bundle.with_formal_registry_identity(FORMAL_REGISTRY_ID)
        assert (registered.asset_identities()["formal_registry"]
                == FORMAL_REGISTRY_ID)
        assert registered.bindings_hash() != bundle.bindings_hash()
        #: every other slot identity is untouched by the registry binding
        for name in ("student", "reference", "probe_runner",
                     "anchor_manifest", "training"):
            assert (registered.asset_identities()[name]
                    == bundle.asset_identities()[name])

    def test_rebuild_determinism(self):
        #: duplicate/rebuild angle: the SAME assets always reproduce the
        #: SAME bindings hash — the identity map is content-addressed
        assert (fully_bound_bundle().bindings_hash()
                == fully_bound_bundle().bindings_hash())

    def test_stale_slot_keeps_old_identity(self):
        #: stale angle: rebinding a slot produces a NEW frozen slot; the
        #: old bundle keeps the old registry identity (immutability) and
        #: the two bundles hash differently
        old_bundle = fully_bound_bundle()
        new_slot = old_bundle.probe_runner.bind(
            probe_runner(registry_identity=RUNNER_REGISTRY_ID_B))
        stale_bundle = replace(old_bundle, probe_runner=new_slot)
        assert (old_bundle.probe_runner.registry_identity
                == RUNNER_REGISTRY_ID_A)
        assert (stale_bundle.probe_runner.registry_identity
                == RUNNER_REGISTRY_ID_B)
        assert old_bundle.bindings_hash() != stale_bundle.bindings_hash()


class TestRegistryIssuedOnly:
    @pytest.mark.parametrize("bad_identity",
                             ["", "short", "z" * 64, "AB" * 32, None, 123])
    def test_probe_runner_registry_identity_refused(self, bad_identity):
        runner = probe_runner(registry_identity=bad_identity)
        with pytest.raises(SharedBindingRejected,
                           match="PROBE_RUNNER_REGISTRY_IDENTITY_MISSING"):
            SharedProbeRunnerSlot().bind(runner)

    def test_probe_runner_without_registry_identity_attr(self):
        runner = SimpleNamespace(real_simulator=True,
                                 runner_id="TEST_ONLY_RUNNER_NO_ATTR")
        #: no registry_identity attribute at all -> missing field angle
        with pytest.raises(SharedBindingRejected,
                           match="PROBE_RUNNER_REGISTRY_IDENTITY_MISSING"):
            SharedProbeRunnerSlot().bind(runner)

    @pytest.mark.parametrize("bad_identity",
                             ["", "short", "z" * 64, "AB" * 32, None, 123])
    def test_training_registry_identity_refused(self, bad_identity):
        contract = training_contract(registry_identity=bad_identity)
        with pytest.raises(
                SharedBindingRejected,
                match="SHARED_TRAINING_REGISTRY_IDENTITY_MISSING"):
            SharedTrainingSlot().bind(contract)

    @pytest.mark.parametrize("bad_identity",
                             ["", "short", "z" * 64, "AB" * 32, None, 123])
    def test_formal_registry_identity_refused(self, bad_identity):
        with pytest.raises(SharedBindingRejected,
                           match="FORMAL_REGISTRY_IDENTITY_INVALID"):
            SharedRuntimeBundle().with_formal_registry_identity(bad_identity)

    def test_training_contract_missing_callable_still_refused_first(self):
        contract = SimpleNamespace(registry_identity=TRAINING_REGISTRY_ID_A)
        with pytest.raises(SharedBindingRejected,
                           match="SHARED_TRAINING_CONTRACT_INCOMPLETE"):
            SharedTrainingSlot().bind(contract)

    @pytest.mark.parametrize("fake_real", [False, "True", 1, None])
    def test_probe_runner_mock_impersonating_real(self, fake_real):
        #: mock-impersonating-real angle: only the literal True passes
        runner = probe_runner(real_simulator=fake_real)
        with pytest.raises(SharedBindingRejected,
                           match="PROBE_RUNNER_NOT_REAL"):
            SharedProbeRunnerSlot().bind(runner)

    def test_probe_runner_missing_runner_id(self):
        runner = SimpleNamespace(real_simulator=True, runner_id="",
                                 registry_identity=RUNNER_REGISTRY_ID_A)
        with pytest.raises(SharedBindingRejected,
                           match="PROBE_RUNNER_ID_MISSING"):
            SharedProbeRunnerSlot().bind(runner)


class TestTamperAndWrongAssets:
    def test_anchor_manifest_hash_tamper_rejected(self):
        genuine = anchor_manifest()
        #: carries the GENUINE hash but different anchor content ->
        #: recomputation mismatch -> typed rejection (the bind wrapper
        #: catches the seam's ValueError tamper code too)
        tampered = SharedAnchorManifest(
            manifest_id="TEST_ONLY_ANCHOR_MANIFEST",
            anchors=[f"TEST_ONLY_ALT_ANCHOR_{i}" for i in range(4)],
            frozen=True, manifest_hash=genuine.manifest_hash)
        with pytest.raises(SharedBindingRejected,
                           match="ANCHOR_MANIFEST_BINDING_REJECTED.*"
                                 "ANCHOR_MANIFEST_HASH_MISMATCH"):
            SharedAnchorManifestSlot().bind(tampered)

    def test_anchor_manifest_unfrozen_rejected(self):
        unfrozen = SharedAnchorManifest(
            manifest_id="TEST_ONLY_UNFROZEN_MANIFEST",
            anchors=list(C.GLOBAL_CANONICAL_ANCHOR_IDS), frozen=False)
        with pytest.raises(SharedBindingRejected,
                           match="ANCHOR_MANIFEST_BINDING_REJECTED"):
            SharedAnchorManifestSlot().bind(unfrozen)

    def test_anchor_manifest_mapping_injection_tamper(self):
        genuine = anchor_manifest()
        tampered_dict = dict(manifest_id="TEST_ONLY_ANCHOR_MANIFEST",
                             anchors=[f"TEST_ONLY_ALT_ANCHOR_{i}"
                                      for i in range(4)],
                             frozen=True,
                             manifest_hash=genuine.manifest_hash)
        with pytest.raises(SharedBindingRejected,
                           match="ANCHOR_MANIFEST_BINDING_REJECTED"):
            SharedAnchorManifestSlot().bind(tampered_dict)

    def test_wrong_student_candidate_rejected(self):
        #: wrong-Student angle: identity must be the registry-issued
        #: PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 candidate
        wrong = student_contract()
        wrong.candidate_id = "TEST_ONLY_WRONG_STUDENT_CANDIDATE"
        with pytest.raises(StudentBindingBlocked,
                           match="STUDENT_IDENTITY_MISMATCH"):
            SharedStudentSlot().bind(wrong)

    def test_reference_incomplete_identity_rejected(self):
        bad = reference_contract(parameter_tree_hash="not-a-sha256")
        with pytest.raises(SharedBindingRejected,
                           match="REFERENCE_IDENTITY_INCOMPLETE"):
            SharedReferenceSlot().bind(bad)

    def test_reference_missing_contract_rejected(self):
        with pytest.raises(SharedBindingRejected,
                           match="REFERENCE_INIT_CONTRACT_MISSING"):
            resolve_reference_binding(None)

    @pytest.mark.parametrize("smuggled_slot", [
        SharedStudentSlot(status=STATUS_BOUND, detail="smuggled",
                          binding=None),
        SharedReferenceSlot(status=STATUS_BOUND, detail="smuggled",
                            binding=None),
        SharedProbeRunnerSlot(status=STATUS_BOUND, detail="smuggled",
                              runner=object(), registry_identity=""),
        SharedAnchorManifestSlot(status=STATUS_BOUND, detail="smuggled",
                                 anchor_ids=("a",), manifest_hash=""),
        SharedTrainingSlot(status=STATUS_BOUND, detail="smuggled",
                           contract=object(), registry_identity=""),
    ], ids=["student", "reference", "probe_runner", "anchor_manifest",
            "training"])
    def test_bound_slot_without_identity_is_smuggled_stand_in(
            self, smuggled_slot):
        #: direct construction bypasses bind(); the identity accessor
        #: still fails closed — a bound slot without an identity can never
        #: reach a bindings hash
        with pytest.raises(SharedBindingRejected,
                           match="SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY"):
            smuggled_slot.slot_identity()
        bundle = SharedRuntimeBundle(**{
            "student": smuggled_slot
            if isinstance(smuggled_slot, SharedStudentSlot)
            else SharedStudentSlot(),
            "probe_runner": smuggled_slot
            if isinstance(smuggled_slot, SharedProbeRunnerSlot)
            else SharedProbeRunnerSlot(),
            "anchor_manifest": smuggled_slot
            if isinstance(smuggled_slot, SharedAnchorManifestSlot)
            else SharedAnchorManifestSlot(),
            "training": smuggled_slot
            if isinstance(smuggled_slot, SharedTrainingSlot)
            else SharedTrainingSlot(),
            "reference": smuggled_slot
            if isinstance(smuggled_slot, SharedReferenceSlot)
            else SharedReferenceSlot()})
        with pytest.raises(SharedBindingRejected,
                           match="SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY"):
            bundle.bindings_hash()


class TestAbsenceStaysBlocked:
    def test_default_bundle_resolve_blocked(self):
        with pytest.raises(SharedRuntimeBlocked,
                           match=C.BLOCKED_WAITING_SHARED_RUNTIME) as exc:
            resolve_shared_runtime()
        #: the full missing list names every one of the five assets
        for description in SLOT_ASSET_DESCRIPTIONS.values():
            assert description in str(exc.value)

    def test_partial_binding_stays_blocked(self):
        bundle = SharedRuntimeBundle(
            student=SharedStudentSlot().bind(student_contract()))
        with pytest.raises(SharedRuntimeBlocked,
                           match=C.BLOCKED_WAITING_SHARED_RUNTIME):
            resolve_shared_runtime(bundle)
        missing = bundle.missing_assets()
        assert len(missing) == 4
        assert SLOT_ASSET_DESCRIPTIONS["student"] not in missing
        #: absent slots fold the sentinel; the bound student folds its
        #: real identity — the hash is still computable and honest
        identities = bundle.asset_identities()
        assert identities["student"] == bundle.student.binding.identity_hash
        for name in ("reference", "probe_runner", "anchor_manifest",
                     "training", "formal_registry"):
            assert identities[name] == SLOT_ABSENT_IDENTITY
        assert len(bundle.bindings_hash()) == 64

    def test_fully_bound_bundle_resolves(self):
        bundle = fully_bound_bundle()
        assert resolve_shared_runtime(bundle) is bundle
        assert bundle.missing_assets() == []

    def test_status_report_contract_unchanged(self):
        #: the entrypoint blocker surface is a frozen contract: exactly
        #: five slots, each with asset/status/detail, and missing_assets
        #: counts only unbound slots (this keeps the 5x
        #: BLOCKED_WAITING_SHARED_RUNTIME blocker reporting unchanged)
        bundle = fully_bound_bundle()
        report = bundle.status_report()
        assert sorted(report) == sorted(SLOT_ASSET_DESCRIPTIONS)
        for name, entry in report.items():
            assert sorted(entry) == ["asset", "detail", "status"]
            assert entry["status"] == STATUS_BOUND
            assert entry["asset"] == SLOT_ASSET_DESCRIPTIONS[name]
        assert bundle.missing_assets() == []
        empty_report = SharedRuntimeBundle().status_report()
        for entry in empty_report.values():
            assert entry["status"] == C.BLOCKED_WAITING_SHARED_RUNTIME
        assert (len(SharedRuntimeBundle().missing_assets())
                == len(SLOT_ASSET_DESCRIPTIONS))


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name

    def test_bindings_never_enable_anything(self):
        #: binding every slot with TEST_ONLY assets keeps the blocked
        #: posture of the constants — the bundle is consume-only metadata
        fully_bound_bundle()
        assert C.E2_PILOT_AUTHORIZED is False
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
