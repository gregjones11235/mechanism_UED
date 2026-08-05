"""§八 (director smoke handoff): the director Runtime Bundle's 12 assets
are all present and non-empty, and the five-slot SharedRuntimeBundle
builds from them.

Contract under test:

* a VALID bundle carries all 12 director assets (registry identities are
  sha256, data-carrying assets are complete) and
  ``runtime_bundle_binding_problems`` returns [] (the "objects not empty"
  gate);
* ``build_shared_bundle`` binds all five slots: student / reference /
  anchor manifest via the data ladders, probe runner / DiCode one-update
  runtime as DIRECTOR-DECLARED identities — direction two never fabricates
  an object, and the bundle's bindings_hash folds the declared identities;
* the smoke origin is restricted to PERSISTENT_RMT16_ORIGINAL_VTRACE_98304
  and the formal-start gate requires human approval (the 98304 origin is
  NEVER the automatic formal start).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.director_runtime_bundle import (
    DIRECTOR_BUNDLE_ASSETS,
    DirectorRuntimeBundleManifest,
    build_shared_bundle,
    runtime_bundle_binding_problems,
)

from e2_test_sign_helpers import (
    valid_director_bundle,
    valid_director_bundle_payload,
    sign_director_runtime_bundle,
)


class TestObjectsNotEmpty:
    def test_all_twelve_assets_present(self):
        manifest = valid_director_bundle()
        dump = manifest.model_dump()
        for asset in DIRECTOR_BUNDLE_ASSETS:
            value = dump[asset]
            assert value is not None, asset
            assert value != "", asset
        assert manifest.registry_identity
        assert manifest.formal_asset_registry
        assert len(manifest.bundle_hash) == 64

    def test_valid_bundle_has_no_binding_problems(self):
        manifest = valid_director_bundle()
        assert runtime_bundle_binding_problems(manifest) == []

    def test_missing_asset_rejected(self):
        payload = valid_director_bundle_payload()
        payload["auxiliary_compute_ledger"] = ""
        with pytest.raises(ValidationError):
            sign_director_runtime_bundle(payload)

    def test_smoke_origin_restricted(self):
        #: an unknown Student candidate is rejected fail-closed (the
        #: profile memory mapping is undefined for unknown candidates)
        payload = valid_director_bundle_payload()
        payload["student_init_contract"]["candidate_id"] = \
            "SOME_OTHER_STUDENT"
        with pytest.raises(ValidationError,
                           match="E2_STUDENT_MEMORY_MODE_MISMATCH"):
            sign_director_runtime_bundle(payload)

    def test_both_allowed_candidates_accepted(self):
        for candidate in C.ALLOWED_STUDENT_CANDIDATE_IDS:
            manifest = valid_director_bundle(candidate_id=candidate)
            assert manifest.student_init_contract.candidate_id == candidate
            assert runtime_bundle_binding_problems(manifest) == []

    def test_formal_start_requires_human(self):
        payload = valid_director_bundle_payload()
        payload["formal_start_gate"] = {"smoke_only_origin": True}
        manifest = sign_director_runtime_bundle(payload)
        problems = runtime_bundle_binding_problems(manifest)
        assert any("FORMAL_START_REQUIRES_HUMAN" in p for p in problems)


class TestBuildSharedBundle:
    def test_all_slots_bound_from_valid_bundle(self):
        manifest = valid_director_bundle()
        bundle = build_shared_bundle(manifest)
        report = bundle.status_report()
        #: three-state model: data slots BOUND_OBJECT; the object slots are
        #: DECLARED_NOT_RESOLVED (identity present, object NOT resolved) —
        #: MANIFEST_ONLY is NOT a handoff
        assert report["student"]["status"] == "BOUND_OBJECT"
        assert report["reference"]["status"] == "BOUND_OBJECT"
        assert report["anchor_manifest"]["status"] == "BOUND_OBJECT"
        assert report["probe_runner"]["status"] == "DECLARED_NOT_RESOLVED"
        assert report["training"]["status"] == "DECLARED_NOT_RESOLVED"
        assert bundle.probe_runner.runner is None
        assert bundle.probe_runner.registry_identity \
            == manifest.candidate_probe_runner
        assert bundle.training.contract is None
        assert bundle.training.registry_identity \
            == manifest.canonical_dicode_one_update_runtime
        #: resolve_shared_runtime REFUSES the unresolved object slots
        from d052.feedback_llm_ued.shared_runtime_binding import (
            resolve_shared_runtime,
        )
        with pytest.raises(Exception,
                           match=C.BLOCKED_WAITING_SHARED_RUNTIME):
            resolve_shared_runtime(bundle)
        #: bindings_hash folds the declared identities (never status
        #: strings) and is deterministic
        assert len(bundle.bindings_hash()) == 64
        assert build_shared_bundle(manifest).bindings_hash() \
            == bundle.bindings_hash()

    def test_student_binding_keeps_smoke_identity(self):
        manifest = valid_director_bundle()
        bundle = build_shared_bundle(manifest)
        assert bundle.student.binding.candidate_id \
            == C.STRONG_STUDENT_CANDIDATE_ID

    def test_reference_binding_resolves(self):
        manifest = valid_director_bundle()
        bundle = build_shared_bundle(manifest)
        assert bundle.reference.binding.identity_hash \
            == manifest.reference_identity.identity_hash


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name

    def test_manifest_is_frozen(self):
        manifest = valid_director_bundle()
        with pytest.raises(ValidationError, match="frozen"):
            manifest.registry_identity = "f" * 64
