"""Signed E3 production runtime bundle: the ONE real asset injection channel.

CC4 follow-up (P0-16): before this contract existed, the one-window driver
had no production channel for its dependencies — checkpoint path and profile
arrived as ad-hoc argv overrides while every other asset (memory artifact,
restore request, anchor manifest, training runtime, capability descriptor,
injected surfaces) was simply None.  Environment-variable guessing and ad-hoc
overrides are never an injection channel.

A runtime bundle is a single SIGNED manifest (controller signature
reference, never synthetic) that names and binds EVERY production asset:

* the Student mount (profile, checkpoint path + sha256, ABI identity hash);
* the original training runtime (descriptors + importable entry points);
* the signed training-surface capability descriptor;
* the saved-policy-memory artifact (path + sha256 + loader entry point);
* the fresh-process restore request payload (typed, hash-bound);
* the shared anchor manifest payload + retention contract;
* the taskparam application and feasibility predicate entry points;
* the search/rollout budget and the output paths.

Everything is validated fail-closed by ``validate_runtime_bundle_manifest``
(exact key set, types, 64-hex digests, non-synthetic signatures, positive
budgets) and ``resolve_bundle_asset_files`` (every referenced file exists
and recomputes to its declared sha256).  Missing or null channels stay
missing and surface as named blockers — never guessed, never defaulted.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .anchor_manifest import (
    ANCHOR_SLOT_COUNT,
    DYNAMIC_DISTRIBUTION_COUNT,
    AnchorDefinition,
    AnchorManifest,
    RetentionContract,
)
from .discovery_provenance import (
    REGISTRY_USAGE_PRODUCTION,
    CaptureProvenance,
    DiscoveryAssetRecord,
    DiscoveryProvenance,
    DiscoveryProvenanceRegistry,
    FormalAssetIdentity,
    registry_hash_of,
)
from .errors import InvalidEvidenceError
from .fresh_process_restore import (
    BUNDLE_SCHEMA,
    REQUEST_SCHEMA,
    ComponentArtifactSpec,
    FreshProcessRestoreRequest,
    ProductionRegistryBundle,
)
from .surface_capability import (
    TrainingSurfaceCapability,
    mint_training_surface_capability,
)
from .two_llm_descriptor import (
    DESCRIPTOR_BUNDLE_KEYS,
    AuthorizedTwoLLMRuntimeDescriptor,
    mint_two_llm_runtime_descriptor,
    verify_two_llm_runtime_descriptor,
)

RUNTIME_BUNDLE_SCHEMA = "simulator_frontier.e3-runtime-bundle/v1"
RUNTIME_BUNDLE_VERSION = "e3-runtime-bundle/v1"

_BLOCKER_PREFIX = "RUNTIME_BUNDLE"

REQUIRED_TOP_KEYS = frozenset({
    "schema", "bundle_id", "run_id", "controller_signature_ref",
    "student", "reference", "training_runtime", "training_surface_capability",
    "memory", "capture_provenance", "formal_asset_registry_payload_path",
    "restore_request_payload_path", "anchor_manifest_payload_path",
    "retention", "taskparam_apply_entrypoint", "predicates",
    "two_llm_runtime", "search", "paths",
})

REFERENCE_SECTION_KEYS = frozenset({
    "profile", "checkpoint_path", "checkpoint_sha256", "abi_identity_hash",
    "adapter_entrypoint", "adapter_hash", "memory_mode",
    "memory_artifact_path", "memory_artifact_sha256", "memory_spec_hash",
    "memory_loader_entrypoint", "burn_in_executor_entrypoint",
    "history_artifact_ref", "reset_protocol_hash",
})

_SYNTHETIC_SIGNATURE_PREFIX = "SYNTHETIC_SIGNATURE_"


def _fail(message: str) -> None:
    raise InvalidEvidenceError(f"{_BLOCKER_PREFIX}: {message}")


def _require_sha256(where: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        _fail(f"{where} must be a lowercase sha256 hex digest, got {text[:24]!r}…")
    return text


def _require_nonempty_str(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where} must be a non-empty string, got {value!r}")
    return value


def _require_entrypoint(where: str, value: Any) -> str:
    text = _require_nonempty_str(where, value)
    if text.count(":") != 1 or not all(part.strip() for part in text.split(":")):
        _fail(f"{where} must be 'module:attr', got {text!r}")
    return text


def _require_positive_int(where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{where} must be a positive int, got {value!r}")
    return int(value)


def _require_nonneg_int(where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{where} must be a non-negative int, got {value!r}")
    return int(value)


def _require_section(where: str, value: Any, keys: frozenset) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be a mapping, got {type(value).__name__}")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing:
        _fail(f"{where} is missing keys {missing}")
    if extra:
        _fail(f"{where} carries unknown keys {extra} (exact schema, fail closed)")
    return value


def load_runtime_bundle_manifest(path: str) -> dict[str, Any]:
    """Read + parse the bundle manifest; fail closed on any I/O or JSON gap."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        _fail(f"runtime bundle manifest not found: {path}")
        raise AssertionError("unreachable") from exc
    except (OSError, ValueError) as exc:
        _fail(f"runtime bundle manifest unreadable: {exc!r}")
        raise AssertionError("unreachable") from exc
    if not isinstance(payload, dict):
        _fail("runtime bundle manifest must be a JSON object")
    return payload


def validate_runtime_bundle_manifest(manifest: Mapping[str, Any]) -> None:
    """Strict, fail-closed validation of the whole signed manifest."""
    if not isinstance(manifest, Mapping):
        _fail(f"manifest must be a mapping, got {type(manifest).__name__}")
    missing = sorted(REQUIRED_TOP_KEYS - set(manifest))
    extra = sorted(set(manifest) - REQUIRED_TOP_KEYS)
    if missing:
        _fail(f"manifest is missing keys {missing}")
    if extra:
        _fail(f"manifest carries unknown keys {extra} (exact schema, fail closed)")

    if manifest["schema"] != RUNTIME_BUNDLE_SCHEMA:
        _fail(f"manifest schema must be {RUNTIME_BUNDLE_SCHEMA!r}, "
              f"got {manifest['schema']!r}")
    _require_nonempty_str("bundle_id", manifest["bundle_id"])
    _require_nonempty_str("run_id", manifest["run_id"])
    signature = _require_nonempty_str(
        "controller_signature_ref", manifest["controller_signature_ref"])
    if signature.startswith(_SYNTHETIC_SIGNATURE_PREFIX):
        _fail(f"controller_signature_ref {signature!r} is synthetic — a self-signed "
              "runtime bundle is never production evidence (fail closed)")

    student = _require_section("student", manifest["student"], frozenset({
        "profile", "checkpoint_path", "checkpoint_sha256", "abi_identity_hash"}))
    _require_nonempty_str("student.profile", student["profile"])
    _require_nonempty_str("student.checkpoint_path", student["checkpoint_path"])
    _require_sha256("student.checkpoint_sha256", student["checkpoint_sha256"])
    _require_sha256("student.abi_identity_hash", student["abi_identity_hash"])

    # Director handoff (P0-b2): the bundle names a COMPLETE Reference runtime.
    # If the reference ABI identity equals the Student's, the run is refused
    # (a Reference is never the Student under another name).
    reference = _require_section("reference", manifest["reference"],
                                 REFERENCE_SECTION_KEYS)
    _require_nonempty_str("reference.profile", reference["profile"])
    _require_nonempty_str("reference.checkpoint_path", reference["checkpoint_path"])
    _require_sha256("reference.checkpoint_sha256", reference["checkpoint_sha256"])
    _require_sha256("reference.abi_identity_hash", reference["abi_identity_hash"])
    _require_entrypoint("reference.adapter_entrypoint",
                        reference["adapter_entrypoint"])
    _require_sha256("reference.adapter_hash", reference["adapter_hash"])
    if reference["memory_mode"] not in ("SAVED_POLICY_MEMORY", "HISTORY_BURN_IN"):
        _fail(f"reference.memory_mode must be SAVED_POLICY_MEMORY or "
              f"HISTORY_BURN_IN, got {reference['memory_mode']!r}")
    _require_nonempty_str("reference.memory_artifact_path",
                          reference["memory_artifact_path"])
    _require_sha256("reference.memory_artifact_sha256",
                    reference["memory_artifact_sha256"])
    _require_sha256("reference.memory_spec_hash", reference["memory_spec_hash"])
    _require_entrypoint("reference.memory_loader_entrypoint",
                        reference["memory_loader_entrypoint"])
    _require_entrypoint("reference.burn_in_executor_entrypoint",
                        reference["burn_in_executor_entrypoint"])
    _require_nonempty_str("reference.history_artifact_ref",
                          reference["history_artifact_ref"])
    _require_sha256("reference.reset_protocol_hash",
                    reference["reset_protocol_hash"])
    if str(reference["abi_identity_hash"]) == str(student["abi_identity_hash"]):
        _fail("reference.abi_identity_hash must differ from "
              "student.abi_identity_hash — a Reference is never the Student "
              "under another name (fail closed)")

    runtime = _require_section("training_runtime", manifest["training_runtime"],
                               frozenset({
                                   "runtime_id", "loss_name", "optimizer_name",
                                   "contract_ref", "loss_entrypoint",
                                   "update_entrypoint"}))
    for key in ("runtime_id", "loss_name", "optimizer_name", "contract_ref"):
        _require_nonempty_str(f"training_runtime.{key}", runtime[key])
    _require_entrypoint("training_runtime.loss_entrypoint",
                        runtime["loss_entrypoint"])
    _require_entrypoint("training_runtime.update_entrypoint",
                        runtime["update_entrypoint"])

    capability = _require_section(
        "training_surface_capability", manifest["training_surface_capability"],
        frozenset({"descriptor_id", "verifier_id", "signature_ref",
                   "save_full_state_capable", "restore_full_state_capable"}))
    for key in ("descriptor_id", "verifier_id", "signature_ref"):
        _require_nonempty_str(f"training_surface_capability.{key}", capability[key])
    if str(capability["signature_ref"]).startswith(_SYNTHETIC_SIGNATURE_PREFIX):
        _fail("training_surface_capability.signature_ref is synthetic — a "
              "self-signed capability descriptor is never production evidence")
    for key in ("save_full_state_capable", "restore_full_state_capable"):
        if not isinstance(capability[key], bool):
            _fail(f"training_surface_capability.{key} must be a genuine bool, "
                  f"got {capability[key]!r}")

    memory = _require_section("memory", manifest["memory"], frozenset({
        "mode", "artifact_path", "artifact_sha256", "memory_spec_hash",
        "student_identity_hash", "loader_entrypoint"}))
    if memory["mode"] not in ("SAVED_POLICY_MEMORY", "HISTORY_BURN_IN"):
        _fail(f"memory.mode must be SAVED_POLICY_MEMORY or HISTORY_BURN_IN, "
              f"got {memory['mode']!r} (ZERO_MEMORY is ablation-only)")
    _require_nonempty_str("memory.artifact_path", memory["artifact_path"])
    _require_sha256("memory.artifact_sha256", memory["artifact_sha256"])
    _require_sha256("memory.memory_spec_hash", memory["memory_spec_hash"])
    _require_sha256("memory.student_identity_hash", memory["student_identity_hash"])
    _require_entrypoint("memory.loader_entrypoint", memory["loader_entrypoint"])

    capture = _require_section("capture_provenance", manifest["capture_provenance"],
                               frozenset({
                                   "provenance", "rollout_protocol_id",
                                   "world_set_hash", "world_set_id",
                                   "bank_refs"}))
    if capture["provenance"] != DiscoveryProvenance.TRAINING_DISCOVERY.value:
        _fail("capture_provenance.provenance must be "
              f"{DiscoveryProvenance.TRAINING_DISCOVERY.value!r} on the production "
              f"path, got {capture['provenance']!r}")
    _require_nonempty_str("capture_provenance.rollout_protocol_id",
                          capture["rollout_protocol_id"])
    _require_sha256("capture_provenance.world_set_hash", capture["world_set_hash"])
    _require_nonempty_str("capture_provenance.world_set_id", capture["world_set_id"])
    if not isinstance(capture["bank_refs"], list) or any(
            not isinstance(ref, str) or not ref.strip()
            for ref in capture["bank_refs"]):
        _fail("capture_provenance.bank_refs must be a list of non-empty strings")

    _require_nonempty_str("formal_asset_registry_payload_path",
                          manifest["formal_asset_registry_payload_path"])
    _require_nonempty_str("restore_request_payload_path",
                          manifest["restore_request_payload_path"])
    _require_nonempty_str("anchor_manifest_payload_path",
                          manifest["anchor_manifest_payload_path"])

    retention = _require_section("retention", manifest["retention"], frozenset({
        "dynamic_distribution_count", "anchor_slot_count", "anchor_ratio",
        "formal_banks_in_online_curriculum"}))
    if retention["dynamic_distribution_count"] != DYNAMIC_DISTRIBUTION_COUNT:
        _fail("retention.dynamic_distribution_count must be "
              f"{DYNAMIC_DISTRIBUTION_COUNT}")
    if retention["anchor_slot_count"] != ANCHOR_SLOT_COUNT:
        _fail(f"retention.anchor_slot_count must be {ANCHOR_SLOT_COUNT}")
    ratio = retention["anchor_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) \
            or not (0.0 < float(ratio) <= 1.0):
        _fail(f"retention.anchor_ratio must be in (0, 1], got {ratio!r}")
    if retention["formal_banks_in_online_curriculum"] is not False:
        _fail("retention.formal_banks_in_online_curriculum must be false — "
              "formal evaluation banks never enter the online curriculum")

    _require_entrypoint("taskparam_apply_entrypoint",
                        manifest["taskparam_apply_entrypoint"])
    predicates = _require_section("predicates", manifest["predicates"], frozenset({
        "success_entrypoint", "progress_entrypoint"}))
    _require_entrypoint("predicates.success_entrypoint",
                        predicates["success_entrypoint"])
    _require_entrypoint("predicates.progress_entrypoint",
                        predicates["progress_entrypoint"])

    # Director handoff (P0-b1): the channel carries a REAL authorized
    # two-LLM runtime descriptor — the previous "must be null" contradiction
    # is gone.  The section is validated strictly (exact keys, non-empty
    # fields, a 64-hex implementation hash, non-synthetic trusted signer);
    # unknown or malformed content is rejected fail closed.
    two_llm = _require_section("two_llm_runtime", manifest["two_llm_runtime"],
                               DESCRIPTOR_BUNDLE_KEYS)
    for key in ("descriptor_id", "authorization_id", "provider", "model",
                "client_factory_entrypoint", "journal_sink", "trusted_signer"):
        _require_nonempty_str(f"two_llm_runtime.{key}", two_llm[key])
    _require_entrypoint("two_llm_runtime.client_factory_entrypoint",
                        two_llm["client_factory_entrypoint"])
    _require_sha256("two_llm_runtime.client_factory_implementation_hash",
                    two_llm["client_factory_implementation_hash"])
    if str(two_llm["trusted_signer"]).startswith(_SYNTHETIC_SIGNATURE_PREFIX):
        _fail("two_llm_runtime.trusted_signer is synthetic — a self-signed "
              "LLM authorization is never production evidence")
    _require_nonneg_int("two_llm_runtime.token_cap", two_llm["token_cap"])
    _require_nonneg_int("two_llm_runtime.retry_cap", two_llm["retry_cap"])

    search = _require_section("search", manifest["search"], frozenset({
        "requested_n", "horizon", "seed_base", "mixed_episodes",
        "episode_horizon", "max_timesteps", "reset_seed", "capture_at_step"}))
    for key in ("requested_n", "horizon", "mixed_episodes", "episode_horizon",
                "max_timesteps", "capture_at_step"):
        _require_positive_int(f"search.{key}", search[key])
    for key in ("seed_base", "reset_seed"):
        _require_nonneg_int(f"search.{key}", search[key])

    paths = _require_section("paths", manifest["paths"], frozenset({
        "archive_path", "checkpoint_dir", "scratch_dir"}))
    for key in ("archive_path", "checkpoint_dir", "scratch_dir"):
        _require_nonempty_str(f"paths.{key}", paths[key])


def resolve_bundle_asset_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Exist + sha256-recompute every file asset named by the manifest.

    Returns the map of logical name -> resolved path.  Any missing file or
    digest mismatch fails closed (nothing is accepted on faith).
    """
    import os
    resolved: dict[str, str] = {}
    student = manifest["student"]
    for name, path_key, sha_key in (
            ("student.checkpoint", "checkpoint_path", "checkpoint_sha256"),):
        path = str(student[path_key])
        if not os.path.isfile(path):
            _fail(f"{name} file does not exist: {path}")
        _recompute_file_sha256(name, path, str(student[sha_key]))
        resolved[name] = path
    memory = manifest["memory"]
    path = str(memory["artifact_path"])
    if not os.path.isfile(path):
        _fail(f"memory.artifact file does not exist: {path}")
    _recompute_file_sha256("memory.artifact", path, str(memory["artifact_sha256"]))
    resolved["memory.artifact"] = path
    reference = manifest["reference"]
    path = str(reference["checkpoint_path"])
    if not os.path.isfile(path):
        _fail(f"reference.checkpoint file does not exist: {path}")
    _recompute_file_sha256("reference.checkpoint", path,
                           str(reference["checkpoint_sha256"]))
    resolved["reference.checkpoint"] = path
    path = str(reference["memory_artifact_path"])
    if not os.path.isfile(path):
        _fail(f"reference.memory_artifact file does not exist: {path}")
    _recompute_file_sha256("reference.memory_artifact", path,
                           str(reference["memory_artifact_sha256"]))
    resolved["reference.memory_artifact"] = path
    for name, key in (
            ("formal_asset_registry_payload", "formal_asset_registry_payload_path"),
            ("restore_request_payload", "restore_request_payload_path"),
            ("anchor_manifest_payload", "anchor_manifest_payload_path")):
        path = str(manifest[key])
        if not os.path.isfile(path):
            _fail(f"{name} file does not exist: {path}")
        resolved[name] = path
    return resolved


def _recompute_file_sha256(name: str, path: str, expected: str) -> None:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected:
        _fail(f"{name} sha256 mismatch: file recomputes to {actual[:16]}…, "
              f"manifest declares {expected[:16]}… (fail closed)")


def callable_source_sha256(name: str, fn: Any) -> str:
    """sha256 of a callable's source file + text (EOL-normalized), fail-closed.

    Used to bind adapter/factory entry points declared in the bundle to their
    implementation — a substituted or drifted implementation never binds.
    """
    import inspect
    if isinstance(fn, Mapping) or not callable(fn):
        _fail(f"{name}: expected a callable, got {type(fn).__name__}")
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        _fail(f"{name}: cannot bind the callable — its source text is "
              f"unavailable ({exc!r}); fail closed")
    try:
        source_file = str(inspect.getsourcefile(fn) or "<unknown>")
    except TypeError:
        source_file = "<unknown>"
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(
        f"{source_file}\n::\n{normalized}".encode("utf-8")).hexdigest()


def import_entrypoint(entrypoint: str, purpose: str) -> Any:
    """Import 'module:attr' fail-closed (no guessing, no fallbacks)."""
    import importlib
    module_name, attr_name = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        _fail(f"cannot import {purpose} entry point module {module_name!r}: {exc!r}")
        raise AssertionError("unreachable") from exc
    try:
        target = getattr(module, attr_name)
    except AttributeError as exc:
        _fail(f"{purpose} entry point attribute {attr_name!r} not found in "
              f"{module_name!r}")
        raise AssertionError("unreachable") from exc
    return target


def restore_request_from_payload(payload: Any) -> FreshProcessRestoreRequest:
    """Rebuild the typed, hash-bound restore request from its JSON payload."""
    if not isinstance(payload, Mapping):
        _fail("restore request payload must be a JSON object")
    if payload.get("schema") != REQUEST_SCHEMA:
        _fail(f"restore request schema must be {REQUEST_SCHEMA!r}, "
              f"got {payload.get('schema')!r}")
    bundle_payload = payload.get("registry_bundle")
    bundle = ProductionRegistryBundle.from_payload(bundle_payload)
    if bundle_payload.get("schema") != BUNDLE_SCHEMA:
        _fail("registry bundle payload schema mismatch")
    raw_artifacts = payload.get("component_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        _fail("restore request component_artifacts must be a mapping")
    artifacts = {
        str(name): ComponentArtifactSpec(
            path=str(spec.get("path", "")),
            sha256=str(spec.get("sha256", "")),
            expected_leaves_digest=str(spec.get("expected_leaves_digest", "")),
        )
        for name, spec in raw_artifacts.items()
        if isinstance(spec, Mapping)
    }
    if len(artifacts) != len(raw_artifacts):
        _fail("restore request component_artifacts entries must all be mappings")
    return FreshProcessRestoreRequest(
        checkpoint_path=str(payload.get("checkpoint_path", "")),
        checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
        student_abi_identity_hash=str(payload.get("student_abi_identity_hash", "")),
        registry_hash=str(payload.get("registry_hash", "")),
        manifest_hash=str(payload.get("manifest_hash", "")),
        expected_global_step=int(payload.get("expected_global_step", -1)),
        expected_next_step_digest=str(payload.get("expected_next_step_digest", "")),
        component_artifacts=artifacts,
        registry_bundle=bundle,
        optimizer_source=str(payload.get("optimizer_source", "checkpoint")),
        fixture_label="",  # production intent only — never a labelled fixture
    )


def anchor_manifest_from_payload(payload: Any) -> AnchorManifest:
    """Rebuild the typed anchor manifest (hash verified by the validator)."""
    if not isinstance(payload, Mapping):
        _fail("anchor manifest payload must be a JSON object")
    raw_anchors = payload.get("anchors")
    if not isinstance(raw_anchors, list):
        _fail("anchor manifest payload must carry an 'anchors' list")
    anchors = tuple(
        AnchorDefinition(
            anchor_id=str(a.get("anchor_id", "")),
            world_set_ref=str(a.get("world_set_ref", "")),
            reset_protocol=str(a.get("reset_protocol", "")),
            seed_policy_ref=str(a.get("seed_policy_ref", "")),
        )
        for a in raw_anchors
        if isinstance(a, Mapping)
    )
    if len(anchors) != len(raw_anchors):
        _fail("anchor manifest anchor entries must all be mappings")
    return AnchorManifest(
        manifest_id=str(payload.get("manifest_id", "")),
        controller_signature_ref=str(payload.get("controller_signature_ref", "")),
        frozen=bool(payload.get("frozen", False)),
        anchors=anchors,
        manifest_hash=str(payload.get("manifest_hash", "")),
    )


def retention_from_payload(payload: Mapping[str, Any]) -> RetentionContract:
    return RetentionContract(
        dynamic_distribution_count=int(payload["dynamic_distribution_count"]),
        anchor_slot_count=int(payload["anchor_slot_count"]),
        anchor_ratio=float(payload["anchor_ratio"]),
        formal_banks_in_online_curriculum=bool(
            payload["formal_banks_in_online_curriculum"]),
    )


def two_llm_descriptor_from_bundle(
        section: Mapping[str, Any]) -> AuthorizedTwoLLMRuntimeDescriptor:
    """Mint + verify the two-LLM runtime descriptor from the bundle section.

    The descriptor's ``client_factory_implementation_hash`` is recomputed
    from the RESOLVED entry point callable and must equal the bundle-declared
    value — a drifted factory never binds.  Building does not call any LLM.
    """
    descriptor = mint_two_llm_runtime_descriptor(
        descriptor_id=str(section["descriptor_id"]),
        authorization_id=str(section["authorization_id"]),
        provider=str(section["provider"]),
        model=str(section["model"]),
        client_factory_entrypoint=str(section["client_factory_entrypoint"]),
        client_factory_implementation_hash=str(
            section["client_factory_implementation_hash"]),
        token_cap=int(section["token_cap"]),
        retry_cap=int(section["retry_cap"]),
        journal_sink=str(section["journal_sink"]),
        trusted_signer=str(section["trusted_signer"]),
    )
    verify_two_llm_runtime_descriptor(descriptor)
    return descriptor


def capability_from_payload(payload: Mapping[str, Any], *,
                            adapter_identity_hash: str
                            ) -> TrainingSurfaceCapability:
    """Mint the capability descriptor BOUND to the mounted adapter identity.

    The descriptor can never be reused for another adapter: the identity hash
    comes from the mount itself, not from the manifest text.
    """
    return mint_training_surface_capability(
        descriptor_id=str(payload["descriptor_id"]),
        adapter_identity_hash=adapter_identity_hash,
        save_full_state_capable=bool(payload["save_full_state_capable"]),
        restore_full_state_capable=bool(payload["restore_full_state_capable"]),
        verifier_id=str(payload["verifier_id"]),
        signature_ref=str(payload["signature_ref"]),
    )


def capture_provenance_from_payload(payload: Mapping[str, Any]) -> CaptureProvenance:
    """Rebuild the typed capture provenance (production mode only)."""
    if str(payload.get("provenance")) != DiscoveryProvenance.TRAINING_DISCOVERY.value:
        _fail("capture provenance payload must be TRAINING_DISCOVERY on the "
              "production path")
    return CaptureProvenance(
        provenance=DiscoveryProvenance.TRAINING_DISCOVERY,
        rollout_protocol_id=str(payload.get("rollout_protocol_id", "")),
        world_set_hash=str(payload.get("world_set_hash", "")),
        world_set_id=str(payload.get("world_set_id", "")),
        bank_refs=tuple(str(ref) for ref in payload.get("bank_refs", ())),
    )


def discovery_registry_from_payload(payload: Any) -> DiscoveryProvenanceRegistry:
    """Rebuild the controller-signed frozen formal asset registry.

    The rebuilt registry must carry usage=PRODUCTION (the only usage the
    injection slot accepts) and its registry_hash must equal the recomputed
    canonical hash — a drifted or TEST_ONLY registry is rejected here, and
    ``inject_frozen_formal_asset_registry`` re-checks everything again.
    """
    if not isinstance(payload, Mapping):
        _fail("formal asset registry payload must be a JSON object")
    if str(payload.get("usage", "")) != REGISTRY_USAGE_PRODUCTION:
        _fail(f"formal asset registry usage must be {REGISTRY_USAGE_PRODUCTION!r} "
              f"to enter the production slot, got {payload.get('usage')!r}")
    raw_forbidden = payload.get("forbidden_formal_identities")
    raw_allowed = payload.get("allowed_discovery_assets")
    if not isinstance(raw_forbidden, list) or not raw_forbidden:
        _fail("formal asset registry must carry a non-empty "
              "forbidden_formal_identities list")
    if not isinstance(raw_allowed, list):
        _fail("formal asset registry allowed_discovery_assets must be a list")
    try:
        forbidden = tuple(
            FormalAssetIdentity(
                asset_kind=str(item.get("asset_kind", "")),
                canonical_id=str(item.get("canonical_id", "")),
                sha256=str(item.get("sha256", "")),
            )
            for item in raw_forbidden
            if isinstance(item, Mapping)
        )
        allowed = tuple(
            DiscoveryAssetRecord(
                asset_id=str(item.get("asset_id", "")),
                asset_kind=str(item.get("asset_kind", "")),
                world_set_hash=str(item.get("world_set_hash", "")),
                content_sha256=str(item.get("content_sha256", "")),
            )
            for item in raw_allowed
            if isinstance(item, Mapping)
        )
    except Exception as exc:
        _fail(f"formal asset registry entries invalid: {exc!r}")
        raise AssertionError("unreachable")
    if len(forbidden) != len(raw_forbidden) or len(allowed) != len(raw_allowed):
        _fail("formal asset registry entries must all be mappings")
    registry_id = str(payload.get("registry_id", ""))
    signature_ref = str(payload.get("controller_signature_ref", ""))
    expected_hash = registry_hash_of(registry_id, signature_ref, forbidden,
                                     allowed, usage=REGISTRY_USAGE_PRODUCTION)
    if str(payload.get("registry_hash", "")) != expected_hash:
        _fail("formal asset registry_hash mismatch: payload does not recompute "
              "to its declared hash (fail closed)")
    return DiscoveryProvenanceRegistry(
        registry_id=registry_id,
        controller_signature_ref=signature_ref,
        frozen=bool(payload.get("frozen", False)),
        forbidden_formal_identities=forbidden,
        allowed_discovery_assets=allowed,
        registry_hash=expected_hash,
        usage=REGISTRY_USAGE_PRODUCTION,
    )
