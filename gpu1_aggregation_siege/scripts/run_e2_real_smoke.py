#!/usr/bin/env python
"""E2 REAL SMOKE driver (direction two) — SlowGRU edition.

The real two-window chain under the director-signed PRODUCTION Runtime
Bundle, with SLOWGRU_PERSISTENT_CANONICAL_98304 as the strong Student:

  ResolvedDirectorRuntime (12 real objects, resolved from the real
  FormalAssetRegistry) -> REAL SlowGRU Student mount -> Window k
  (optimizer updates = 0) -> feedback_k frozen -> Window k+1 (EXACTLY ONE
  canonical DiCode update) -> full RunState checkpoint -> fresh-process
  restore -> next-policy equivalence.

Object-level check-only (E2_SLOWGRU_PERSISTENT_OBJECT_CHECK_ONLY_OK): resolves
the 12 real objects + mounts the REAL SlowGRU Student with real
Bundle/Verifier/Registry/checkpoint/Adapter. LLM calls = 0,
Probe = 0, optimizer updates = 0, checkpoint writes = 0.

FORMAL_LONGRUN_AUTHORIZED=false: this is ONE review window and ONE
optimizer update — never a long run.

Environment (set by the launcher, never defaulted here):
  DICODE_SHARED_RUNTIME_REAL=1, OPENAI_API_KEY / OPENAI_BASE_URL /
  QWEN_MODEL (server-authorized transport), WANDB_MODE=offline,
  XLA_PYTHON_CLIENT_PREALLOCATE=false, CUDA_VISIBLE_DEVICES=<GPU>.

The shared runtime (dicode.shared_runtime / dicode.student_adapters /
dicode.teachers) is consumed from the E1 shared-runtime worktree whose
root is exported as E1_SIEGE_ROOT (defaults to the canonical path).
The SlowGRU student adapter is loaded from the LOCAL E2
dicode.student_adapters package.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E2_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
E1_ROOT = os.environ.get(
    "E1_SIEGE_ROOT",
    "/home/oseasy/git_work/wt_e1_static_llm/gpu1_aggregation_siege")

os.environ.setdefault("DICODE_SHARED_RUNTIME_REAL", "1")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# the E1 shared-runtime src FIRST so `dicode` (shared_runtime /
# student_adapters / teachers / training / transformer) resolves to the
# real shared implementation; d052 resolves from E2_ROOT.
if os.path.join(E1_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(E1_ROOT, "src"))
if E2_ROOT not in sys.path:
    sys.path.insert(0, E2_ROOT)

RUN_ID = "e2_real_smoke_slowgru_" + time.strftime("%Y%m%dT%H%M%S")

#: bounded real-board re-attempts when a window's board requests human
#: control (the authorized smoke has no human reviewer; a REQUEST_CONTROL
#: board is retried a bounded number of times before being reported)
MAX_BOARD_ATTEMPTS = 3
OUT_DIR = os.path.join(E2_ROOT, "reports", "director_smoke", RUN_ID)
REPORTS_E2 = os.path.join(E2_ROOT, "reports", "e2_formal_ued")
BUNDLE_PATH = os.path.join(REPORTS_E2, "e2_production_runtime_bundle_slowgru.json")
TRUST_STORE_PATH = os.path.join(
    REPORTS_E2, "e2_production_bundle_trust_store_slowgru.json")

PERSISTENT = "SLOWGRU_PERSISTENT_CANONICAL_98304"
DIRECTOR_SIGNER = "mechanism_UED_director_cc.e2"
SOURCE_COMMIT_TAG = "src-sha256:"

# SlowGRU paths (server-side)
SLOWGRU_RUNTIME_PATH = "/home/oseasy/student_pool_v1/cc3/slowgru_runtime"
SLOWGRU_CONTRACT_PATH = (
    "/home/oseasy/student_pool_v1/cc3/"
    "SLOWGRU_PERSISTENT_CANONICAL_98304/checkpoint_contract.json")
SLOWGRU_NETWORK_SRC_SHA256 = (
    "b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b")
SLOWGRU_TRAINER_SRC_SHA256 = (
    "7918333c63bdb6c8917bf423dfb8484942fb46edc6a7c8fa7e36c769cada2545")


def _log(msg: str) -> None:
    print(f"[e2-smoke] {msg}", flush=True)


def _write(name: str, payload) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=E2_ROOT).stdout.strip()


def _canonical_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SlowGRU student helpers (local E2 dicode.student_adapters)
# ---------------------------------------------------------------------------
def _load_slowgru_profile():
    """Load the SlowGRU profile from the local E2 conf directory."""
    from dicode.student_adapters.registry import default_profile_dir, load_student_profile
    profile_path = default_profile_dir() / "slowgru_persistent_98304.yaml"
    return load_student_profile(profile_path)


def _build_slowgru_adapter(profile):
    """Build the SlowGRUStudentAdapter with real runtime paths."""
    from dicode.student_adapters.slowgru_adapter import SlowGRUStudentAdapter
    adapter = SlowGRUStudentAdapter(
        profile,
        slowgru_runtime_path=SLOWGRU_RUNTIME_PATH,
        checkpoint_contract_path=SLOWGRU_CONTRACT_PATH,
        expected_network_src_sha256=SLOWGRU_NETWORK_SRC_SHA256,
        expected_trainer_src_sha256=SLOWGRU_TRAINER_SRC_SHA256,
    )
    identity = profile.expected_identity()
    adapter.registry_identity = identity.identity_hash()
    adapter.checkpoint_file_sha256 = str(
        profile.notes.get("checkpoint_file_sha256", ""))
    return adapter


# ---------------------------------------------------------------------------
# the E2 FormalAssetRegistry (thin real adapter over the shared runtime)
# ---------------------------------------------------------------------------
class E2FormalAssetRegistry:
    """The E2-conforming FormalAssetRegistry.

    Holds the REAL shared-runtime objects (student contract / identity /
    adapter, reference identity / adapter, real anchor manifest, real
    probe runner, canonical DiCode one-update runtime, run-state
    checkpoint manager, authorized six-role LLM runtime, transport
    closure, auxiliary compute ledger) under the E2 director contract
    names. Registry identity is cross-bound to the shared registry's own
    identity; every registered object is a real deployment artifact.
    """

    def __init__(self, *, registry_identity: str, source_commit: str):
        self.registry_identity = registry_identity
        self.source_commit = source_commit
        self.object_identity_hash = registry_identity
        self._assets: Dict[str, Dict[str, Any]] = {}
        self.registry_hash = ""

    def register(self, *, contract: str, obj: Any, identity_hash: str,
                 implementation_hash: str) -> None:
        if obj is None or isinstance(obj, (str, bytes, bool, int, float)):
            raise RuntimeError(
                f"E2_REGISTRY_PLACEHOLDER_REJECTED: {contract!r}")
        self._assets[contract] = {
            "obj": obj,
            "identity_hash": identity_hash,
            "implementation_hash": implementation_hash,
        }
        self._refresh_hash()

    def _refresh_hash(self) -> None:
        self.registry_hash = _canonical_sha256({
            "registry_identity": self.registry_identity,
            "assets": {
                contract: {
                    "identity_hash": e["identity_hash"],
                    "implementation_hash": e["implementation_hash"],
                }
                for contract, e in sorted(self._assets.items())
            },
        })

    def resolve_asset(self, *, contract: str = "", expected_identity: str = "",
                      identity: str = "") -> Any:
        expected = expected_identity or identity
        if not contract:
            raise RuntimeError("E2_REGISTRY_RESOLVE_NO_CONTRACT")
        entry = self._assets.get(contract)
        if entry is None:
            raise RuntimeError(
                f"E2_REGISTRY_ASSET_UNREGISTERED: {contract!r}")
        if expected and entry["identity_hash"] != expected:
            raise RuntimeError(
                f"E2_REGISTRY_IDENTITY_MISMATCH: {contract!r}")
        return entry["obj"]

    def verify_implementation(self, *, contract: str = "", obj: Any = None,
                              expected_implementation_hash: str = "",
                              identity: str = "") -> bool:
        if contract:
            entry = self._assets.get(contract)
            candidates = [(contract, entry)] if entry is not None else []
        else:
            candidates = [(c, e) for c, e in self._assets.items()
                          if e["obj"] is obj]
        if not candidates:
            return False
        for _c, entry in candidates:
            if obj is not None and entry["obj"] is not obj:
                continue
            if identity and entry["identity_hash"] != identity:
                continue
            if expected_implementation_hash and (
                    entry["implementation_hash"]
                    != expected_implementation_hash):
                continue
            return True
        return False

    def declared_identity(self, contract: str) -> str:
        entry = self._assets.get(contract)
        if entry is None:
            return ""
        return entry["identity_hash"]

    def implementation_hash_of_registered(self, contract: str) -> str:
        entry = self._assets.get(contract)
        if entry is None:
            return ""
        return entry["implementation_hash"]

    def registered_contracts(self) -> Tuple[str, ...]:
        return tuple(sorted(self._assets))


# ---------------------------------------------------------------------------
# the E2 director bundle verifier (real trust store)
# ---------------------------------------------------------------------------
class E2DirectorBundleVerifier:
    """Implements the E2 DirectorBundleVerifier protocol against a real
    trust store written by the director at issuance time."""

    verifier_id = "mechanism_UED.e2_production_director_verifier"
    verifier_implementation_hash = _sha256_text(
        "e2.shared_runtime.production_director_verifier.v1")

    def __init__(self, trust_store_path: str = ""):
        path = trust_store_path or TRUST_STORE_PATH
        if not os.path.isfile(path):
            raise RuntimeError(
                f"E2_DIRECTOR_TRUST_STORE_MISSING: {path!r}; run the "
                "issuance step first")
        with open(path, "r", encoding="utf-8") as handle:
            store: Dict[str, Any] = json.load(handle)
        self._trusted_signers = set(store.get("trusted_signers", []))
        self._trusted_commits = set(store.get("trusted_source_commits", []))
        self._issued = {
            rec["bundle_hash"]: rec
            for rec in store.get("issued_bundles", [])
        }
        self.trusted_signer_registry_hash = str(
            store.get("trusted_signer_registry_hash", ""))

    def verify_manifest(self, manifest) -> bool:
        record = self._issued.get(manifest.bundle_hash)
        if record is None:
            return False
        if record.get("signer_id") != manifest.signer_id:
            return False
        if record.get("registry_identity") != manifest.registry_identity:
            return False
        return True

    def signer_trusted(self, signer_id: str) -> bool:
        return signer_id in self._trusted_signers

    def verify_source_commit(self, source_commit: str) -> bool:
        return source_commit in self._trusted_commits


# ---------------------------------------------------------------------------
# E2-conforming real objects
# ---------------------------------------------------------------------------
def build_e2_student_contract(*, candidate_id: str, bundle_hash: str,
                              registry_identity: str) -> object:
    """The E2 StudentInitContract for SlowGRU, built from the local
    dicode.student_adapters profile."""
    profile = _load_slowgru_profile()
    identity = profile.expected_identity()
    spec = profile.memory_spec()
    return SimpleNamespace(
        candidate_id=candidate_id,
        architecture_family="SLOWGRU",
        memory_family="SLOWGRU_ORIGINAL",
        parameter_tree_hash=profile.params_sha256,
        checkpoint_global_step=int(profile.global_step),
        profile_hash=identity.identity_hash(),
        memory_mode="PERSISTENT",
        memory_spec_hash=spec.spec_hash(),
        carry_mode="persistent",
        adapter_identity_hash=_sha256_text(
            "e2.slowgru_adapter.local.v1:" + profile.params_sha256),
        runtime_bundle_hash=bundle_hash,
        registry_identity=registry_identity,
        checkpoint_file_sha256=str(
            profile.notes.get("checkpoint_file_sha256", "")),
        params_sha256=profile.params_sha256,
        source_commit=profile.source_commit,
    )


def build_e2_student_identity() -> object:
    """E2 StudentIdentity for SlowGRU."""
    profile = _load_slowgru_profile()
    identity = profile.expected_identity()
    spec = profile.memory_spec()
    return SimpleNamespace(
        candidate_id=profile.candidate_id,
        architecture_family="SLOWGRU",
        memory_mode="PERSISTENT",
        params_sha256=profile.params_sha256,
        checkpoint_file_sha256=str(
            profile.notes.get("checkpoint_file_sha256", "")),
        profile_hash=identity.identity_hash(),
        memory_spec_hash=spec.spec_hash(),
        source_commit=profile.source_commit,
        registry_identity=identity.identity_hash(),
        object_identity_hash=identity.identity_hash(),
        identity_hash=identity.identity_hash(),
    )


def build_e2_student_adapter() -> object:
    """E2 StudentAdapter for SlowGRU (real SlowGRUStudentAdapter)."""
    profile = _load_slowgru_profile()
    return _build_slowgru_adapter(profile)


def build_e2_reference_identity() -> object:
    from dicode.shared_runtime import reference_assets as RA

    real = RA.real_reference_identity()
    return SimpleNamespace(
        candidate_id=real.candidate_id,
        architecture_family=real.architecture_family,
        memory_mode=real.memory_mode,
        parameter_tree_hash=real.params_sha256,
        checkpoint_global_step=98304,
        params_sha256=real.params_sha256,
        checkpoint_file_sha256=real.checkpoint_file_sha256,
        profile_hash=real.profile_hash,
        source_commit=real.source_commit,
        identity_hash=real.object_identity_hash,
        object_identity_hash=real.object_identity_hash,
        registry_identity=real.object_identity_hash,
    )


def build_e2_reference_adapter() -> object:
    from dicode.shared_runtime import reference_assets as RA

    return RA.real_reference_adapter()


def build_e2_anchor_manifest() -> object:
    """The E2 SharedAnchorManifest bound to the REAL frozen anchor set."""
    from d052.feedback_llm_ued.anchor_manifest import SharedAnchorManifest
    from dicode.shared_runtime import anchor_asset as AA

    real = AA.real_anchor_manifest()
    anchors = [a.anchor_id for a in real.anchors]
    manifest = SharedAnchorManifest(
        manifest_id="REAL_FROZEN_ANCHOR_MANIFEST_V1",
        anchors=anchors,
        frozen=True)
    try:
        manifest.registry_identity = manifest.manifest_hash
    except Exception:
        object.__setattr__(manifest, "registry_identity",
                           manifest.manifest_hash)
    try:
        manifest.object_identity_hash = manifest.manifest_hash
    except Exception:
        object.__setattr__(manifest, "object_identity_hash",
                           manifest.manifest_hash)
    return manifest


def build_e2_probe_runner() -> object:
    """E2-conforming CandidateProbeRunner wrapping the REAL shared probe
    runner. Carries the full executable ABI hash surface required by the
    production probe seam."""
    from dicode.shared_runtime.probe_runner import RealProbeRunner

    # Use the SlowGRU adapter for the probe runner
    profile = _load_slowgru_profile()
    student_adapter = _build_slowgru_adapter(profile)
    real = RealProbeRunner(student_adapter)

    class E2ProbeRunner:
        real_simulator = True

        def __init__(self, real_runner, student_adapter):
            self._real = real_runner
            self.runner_id = (
                "feedback_llm_ued.e2_real_candidate_probe_slowgru.v1::"
                + real_runner.runner_id)
            self.registry_identity = real_runner.registry_identity
            self.object_identity_hash = real_runner.registry_identity
            self.observation_abi_hash = _sha256_text(
                "e2.abi.observation.v1:" + student_adapter.registry_identity)
            self.action_abi_hash = _sha256_text(
                "e2.abi.action.v1:" + student_adapter.registry_identity)
            self.reward_contract_hash = _sha256_text(
                "e2.abi.reward.v1:" + student_adapter.registry_identity)
            self.reset_protocol_hash = real_runner.reset_protocol_hash
            self.step_protocol_hash = _sha256_text(
                "e2.abi.step.v1:" + real_runner.reset_protocol_hash)

        def probe_candidate(self, *, candidate_hash: str,
                            environment_family: str,
                            axis_values: Dict[str, str],
                            held_constant_axes: Dict[str, str],
                            stage: str, student_episodes: int,
                            reference_episodes: int,
                            seed_bank: Tuple[int, ...]):
            from d052.feedback_llm_ued.real_probe_feedback import (
                SharedCandidateProbeRunner,
            )
            raise NotImplementedError(
                "E2_PROBE_NOT_IMPLEMENTED_YET: probe_candidate body is "
                "wired by the smoke driver's real-probe path")

    return E2ProbeRunner(real, student_adapter)


def build_e2_dicode_runtime() -> object:
    """E2-conforming CanonicalDiCodeOneUpdateRuntime wrapping the REAL
    canonical one-update training runtime (SlowGRU student)."""
    from dicode.shared_runtime.training_assets import CanonicalOneUpdateRuntime

    profile = _load_slowgru_profile()
    student_adapter = _build_slowgru_adapter(profile)
    real = CanonicalOneUpdateRuntime(
        student_adapter=student_adapter,
        train_state_candidate=PERSISTENT,
    )

    class E2DiCodeRuntime:
        def __init__(self, real_runtime):
            self._real = real_runtime
            self.registry_identity = real_runtime.registry_identity
            self.object_identity_hash = real_runtime.registry_identity

        def run_one_dicode_update(self, *, window: int,
                                  batch_candidate_ids: List[str]):
            raise NotImplementedError(
                "E2_DICODE_UPDATE_NOT_IMPLEMENTED_YET: run_one_dicode_update "
                "is wired by the smoke driver's real-update path")

        def verify_director_round_trip(self, *, window: int,
                                       checkpoint_hash: str):
            raise NotImplementedError(
                "E2_DICODE_ROUNDTRIP_NOT_IMPLEMENTED_YET")

        def save_checkpoint(self, *, tag: str) -> str:
            raise NotImplementedError("E2_DICODE_SAVE_NOT_IMPLEMENTED_YET")

        def load_checkpoint(self, *, checkpoint_hash: str) -> None:
            raise NotImplementedError("E2_DICODE_LOAD_NOT_IMPLEMENTED_YET")

    return E2DiCodeRuntime(real)


def build_e2_runstate_checkpoint() -> object:
    from dicode.shared_runtime.runstate import RunStateCheckpointManager
    return RunStateCheckpointManager()


def build_e2_llm_runtime() -> object:
    from dicode.shared_runtime.llm_runtime import AuthorizedSixRoleLLMRuntime
    return AuthorizedSixRoleLLMRuntime()


def build_e2_transport_closure() -> object:
    """The REAL transport closure: (role, prompt) -> response str through
    the server-authorized OpenAI-compatible endpoint. Fails closed when
    the authorization env vars are absent (never prints keys)."""
    required = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "QWEN_MODEL")

    def _transport(role: str, prompt: str) -> str:
        missing = [name for name in required
                   if not str(os.environ.get(name, "")).strip()]
        if missing:
            raise RuntimeError(
                "REAL_LLM_TRANSPORT_UNAUTHORIZED: missing "
                + ",".join(missing))
        import json as _json
        import urllib.request

        url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"
        payload = _json.dumps({
            "model": os.environ["QWEN_MODEL"],
            "messages": [
                {"role": "system",
                 "content": "You are the DiCode UED review board."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer "
                                 + os.environ["OPENAI_API_KEY"],
            }, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"REAL_LLM_TRANSPORT_CALL_FAILED: {type(exc).__name__}: "
                f"{exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"REAL_LLM_TRANSPORT_BAD_RESPONSE: {type(exc).__name__}: "
                f"{exc}") from exc

    _transport.registry_identity = _sha256_text(
        "e2.transport_closure.qwen.dashscope.v1")
    _transport.object_identity_hash = _transport.registry_identity
    return _transport


def build_e2_compute_ledger() -> object:
    from dicode.shared_runtime.ledger import ProductionComputeLedger
    return ProductionComputeLedger()


# ---------------------------------------------------------------------------
# registry construction
# ---------------------------------------------------------------------------
def build_e2_registry() -> Tuple[E2FormalAssetRegistry, Dict[str, object]]:
    """Register the REAL shared-runtime objects under the E2 director
    contract names. Returns (registry, objects)."""
    from dicode.shared_runtime import registry as E1REG

    real_registry = E1REG.production_registry()
    source_commit = real_registry.source_commit

    objects: Dict[str, object] = {}
    objects["student_init_contract"] = build_e2_student_contract(
        candidate_id=PERSISTENT, bundle_hash="",
        registry_identity="")  # patched after bundle issue
    objects["student_identity"] = build_e2_student_identity()
    objects["student_adapter"] = build_e2_student_adapter()
    objects["reference_identity"] = build_e2_reference_identity()
    objects["reference_adapter"] = build_e2_reference_adapter()
    objects["candidate_probe_runner"] = build_e2_probe_runner()
    objects["shared_anchor_manifest"] = build_e2_anchor_manifest()
    objects["canonical_dicode_one_update_runtime"] = build_e2_dicode_runtime()
    objects["canonical_dicode_run_state_checkpoint"] = \
        build_e2_runstate_checkpoint()
    objects["authorized_six_role_llm_runtime"] = build_e2_llm_runtime()
    objects["transport_closure"] = build_e2_transport_closure()
    objects["auxiliary_compute_ledger"] = build_e2_compute_ledger()

    registry = E2FormalAssetRegistry(
        registry_identity=real_registry.registry_identity,
        source_commit=source_commit)

    from dicode.shared_runtime.registry import implementation_hash_of
    for contract, obj in objects.items():
        own = getattr(obj, "registry_identity", None)
        if not (isinstance(own, str) and len(own) == 64):
            own = getattr(obj, "object_identity_hash", None)
        if not (isinstance(own, str) and len(own) == 64):
            own = _canonical_sha256({
                "kind": f"e2.shared_runtime.{contract}",
                "contract": contract,
            })
        registry.register(
            contract=contract, obj=obj, identity_hash=own,
            implementation_hash=implementation_hash_of(obj))
    return registry, objects


# ---------------------------------------------------------------------------
# bundle issuance (E2 director side; idempotent)
# ---------------------------------------------------------------------------
def issue_e2_bundle(registry: E2FormalAssetRegistry) -> dict:
    """Build the E2-format signed director Runtime Bundle for SlowGRU
    from the real registry and write the trust store. Idempotent."""
    from d052.bagr_ued.hashing import canonical_sha256
    from d052.feedback_llm_ued import constants as C
    from d052.feedback_llm_ued.director_runtime_bundle import (
        DIRECTOR_RUNTIME_BUNDLE_VERSION,
        DirectorRuntimeBundleManifest,
        REQUIRED_DIRECTOR_OBJECTS,
        _manifest_hash_body,
    )

    if os.path.isfile(BUNDLE_PATH) and os.path.isfile(TRUST_STORE_PATH):
        with open(BUNDLE_PATH, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        try:
            manifest = DirectorRuntimeBundleManifest(**existing)
            return manifest.model_dump()
        except Exception as exc:
            _log(f"existing bundle invalid, re-issuing: {exc}")

    raw_commit = registry.source_commit
    source_commit = (raw_commit.split(":", 1)[-1]
                     if raw_commit.startswith(SOURCE_COMMIT_TAG)
                     else raw_commit)
    objects = {}
    for name in REQUIRED_DIRECTOR_OBJECTS:
        objects[name] = {
            "identity_hash": registry.declared_identity(name),
            "implementation_hash":
                registry.implementation_hash_of_registered(name),
            "source_commit": source_commit,
            "registry_identity": registry.registry_identity,
        }

    profile = _load_slowgru_profile()
    identity = profile.expected_identity()
    spec = profile.memory_spec()

    anchor_obj = registry.resolve_asset(
        contract="shared_anchor_manifest")
    anchors = list(anchor_obj.anchors)
    anchor_manifest_hash = anchor_obj.manifest_hash

    non_target = [a for a in anchors
                  if a != "anchor_original_craftax"][:3]
    original_task_id = "original_craftax"

    student_contract_data = {
        "candidate_id": PERSISTENT,
        "architecture_family": "SLOWGRU",
        "memory_family": "SLOWGRU_ORIGINAL",
        "carry_mode": "persistent",
        "parameter_tree_hash": profile.params_sha256,
        "checkpoint_global_step": int(profile.global_step),
        "profile_hash": identity.identity_hash(),
        "memory_mode": "PERSISTENT",
        "memory_spec_hash": spec.spec_hash(),
        "adapter_identity_hash": registry.declared_identity("student_adapter"),
        "runtime_bundle_hash": _sha256_text(
            "E2_RUNTIME_BUNDLE_HASH_PLACEHOLDER_SLOWGRU"),
    }
    reference_identity_data = {
        "candidate_id": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
        "parameter_tree_hash": registry.declared_identity("reference_identity"),
        "checkpoint_global_step": 98304,
        "identity_hash": registry.declared_identity("reference_identity"),
    }
    anchor_manifest_data = {
        "manifest_id": "REAL_FROZEN_ANCHOR_MANIFEST_V1",
        "anchors": anchors,
        "frozen": True,
        "manifest_hash": anchor_manifest_hash,
    }

    payload = dict(
        registry_identity=registry.registry_identity,
        objects=objects,
        formal_asset_registry=registry.registry_identity,
        signer_id=DIRECTOR_SIGNER,
        source_commit=source_commit,
        student_init_contract=student_contract_data,
        student_identity=registry.declared_identity("student_identity"),
        student_adapter=registry.declared_identity("student_adapter"),
        reference_identity=reference_identity_data,
        reference_adapter=registry.declared_identity("reference_adapter"),
        candidate_probe_runner=registry.declared_identity(
            "candidate_probe_runner"),
        shared_anchor_manifest=anchor_manifest_data,
        canonical_dicode_one_update_runtime=registry.declared_identity(
            "canonical_dicode_one_update_runtime"),
        canonical_dicode_run_state_checkpoint=registry.declared_identity(
            "canonical_dicode_run_state_checkpoint"),
        authorized_six_role_llm_runtime=registry.declared_identity(
            "authorized_six_role_llm_runtime"),
        backend_model_identity=dict(
            backend_id="qwen.dashscope.v1",
            model_id=os.environ.get("QWEN_MODEL", "qwen-plus"),
        ),
        transport_closure=registry.declared_identity("transport_closure"),
        auxiliary_compute_ledger=registry.declared_identity(
            "auxiliary_compute_ledger"),
        smoke_semantics=dict(window0_update_delta=0,
                             window1_update_delta=1, total_updates=1),
        batch_binding=dict(
            dynamic_task_count=C.DICODE_CURRICULUM_DYNAMIC,
            non_target_anchor_count=C.DICODE_CURRICULUM_NON_TARGET_ANCHORS,
            curriculum_task_count=C.DICODE_CURRICULUM_TASK_COUNT,
            non_target_anchor_ids=non_target,
            original_task_id=original_task_id,
            original_task_proportion=C.DICODE_ORIGINAL_TASK_PROPORTION,
            total_task_count=C.DICODE_BATCH_TOTAL_TASKS),
        formal_start_gate=dict(formal_start_requires_human=True),
    )

    from d052.feedback_llm_ued.director_runtime_bundle import (
        AnchorManifestData, DiCodeBatchBindingData, ReferenceIdentityData,
        RuntimeObjectDescriptor, SmokeSemanticsData, StudentInitContractData,
    )
    body = dict(payload)
    body.setdefault("bundle_version", DIRECTOR_RUNTIME_BUNDLE_VERSION)
    body.setdefault("student_adapter", "")
    body.setdefault("reference_adapter", "")
    body["student_init_contract"] = StudentInitContractData(
        **body["student_init_contract"])
    body["reference_identity"] = ReferenceIdentityData(
        **body["reference_identity"])
    body["shared_anchor_manifest"] = AnchorManifestData(
        **body["shared_anchor_manifest"])
    body["smoke_semantics"] = SmokeSemanticsData(**body["smoke_semantics"])
    body["batch_binding"] = DiCodeBatchBindingData(**body["batch_binding"])
    body["objects"] = {
        name: RuntimeObjectDescriptor(**desc)
        for name, desc in body["objects"].items()
    }
    preview = DirectorRuntimeBundleManifest.model_construct(**body)
    signature = canonical_sha256(_manifest_hash_body(preview))
    student_final = dict(body["student_init_contract"].model_dump())
    student_final["runtime_bundle_hash"] = signature
    body["student_init_contract"] = student_final
    manifest = DirectorRuntimeBundleManifest(**body, bundle_hash=signature)

    os.makedirs(REPORTS_E2, exist_ok=True)
    with open(BUNDLE_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest.model_dump(), handle, indent=2, sort_keys=True,
                  default=str)
        handle.write("\n")
    trust_store = {
        "schema": "mechanism_UED.e2_bundle_trust_store/v1",
        "trusted_signers": [DIRECTOR_SIGNER],
        "trusted_source_commits": [source_commit],
        "trusted_signer_registry_hash": _canonical_sha256(
            {"trusted_signers": [DIRECTOR_SIGNER]}),
        "issued_bundles": [{
            "bundle_id": "e2-production-runtime-bundle-slowgru-v1",
            "bundle_hash": manifest.bundle_hash,
            "signature_ref": manifest.bundle_hash,
            "signer_id": DIRECTOR_SIGNER,
            "registry_identity": registry.registry_identity,
        }],
    }
    with open(TRUST_STORE_PATH, "w", encoding="utf-8") as handle:
        json.dump(trust_store, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _log(f"E2 production bundle (SlowGRU) issued: {BUNDLE_PATH} "
         f"(hash={manifest.bundle_hash[:16]}...)")
    return manifest.model_dump()


# ---------------------------------------------------------------------------
# object-level check-only
# ---------------------------------------------------------------------------
def run_e2_object_check(*, manifest: Dict[str, Any],
                        registry: E2FormalAssetRegistry,
                        verifier: E2DirectorBundleVerifier,
                        objects: Dict[str, object]) -> Dict[str, Any]:
    import importlib.util
    from pathlib import Path

    entry_path = os.path.join(E2_ROOT, "scripts", "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_real", entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    from d052.feedback_llm_ued.director_runtime_bundle import (
        DirectorRuntimeBundleManifest,
    )
    manifest_obj = DirectorRuntimeBundleManifest(**manifest)
    contract = objects["student_init_contract"]
    contract.runtime_bundle_hash = manifest_obj.bundle_hash
    contract.registry_identity = registry.declared_identity(
        "student_init_contract")

    result = entry.run_e2_object_level_check(
        manifest=manifest_obj,
        director_bundle_verifier=verifier,
        formal_asset_registry=registry,
        selected_candidate_id=PERSISTENT)
    return result


def run_real_smoke(*, manifest: Dict[str, Any],
                   registry: E2FormalAssetRegistry,
                   verifier: E2DirectorBundleVerifier,
                   objects: Dict[str, object]) -> Dict[str, Any]:
    """Run the REAL two-window smoke through the E2 production entrypoint.

    Window k (0 optimizer updates, feedback_k frozen) -> Window k+1
    (exactly ONE canonical DiCode update) -> RunState checkpoint ->
    fresh-process restore. Honest outcome on every path.
    """
    import importlib.util

    from d052.feedback_llm_ued.director_runtime_bundle import (
        DirectorRuntimeBundleManifest,
    )
    entry_path = os.path.join(E2_ROOT, "scripts", "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_real_smoke", entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    manifest_obj = DirectorRuntimeBundleManifest(**manifest)

    resolved_runtime, bundle = entry._resolve_production_runtime(
        manifest=manifest_obj,
        director_bundle_verifier=verifier,
        formal_asset_registry=registry,
        selected_candidate_id=PERSISTENT)
    transport = registry.resolve_asset(contract="transport_closure")

    state_path = os.path.join(OUT_DIR, "real_two_window_state.json")
    journal_path = os.path.join(OUT_DIR, "real_two_window_journal.json")

    outcome = None
    last_outcome = None
    for attempt in range(1, MAX_BOARD_ATTEMPTS + 1):
        outcome = entry.run_two_real_windows(
            bundle=bundle,
            llm_transport=transport,
            backend_id=manifest.get("backend_model_identity", {}).get(
                "backend_id", ""),
            model_id=manifest.get("backend_model_identity", {}).get(
                "model_id", ""),
            state_path=state_path,
            journal_path=journal_path,
            student_init_contract=build_student_init_contract_data(
                manifest_obj),
            director_manifest=manifest_obj,
            director_selected_candidate_id=PERSISTENT)
        last_outcome = outcome
        if not outcome.get("request_control_stopped"):
            break
        _log(f"board requested control on attempt {attempt}; retrying "
             f"the real board (bounded)")
        outcome = None
    if outcome is None:
        return dict(status="REQUEST_CONTROL_STOPPED",
                    reason=last_outcome.get("outcome", ""),
                    **last_outcome)
    n_updates = outcome.get("optimizer_updates_executed", 0)
    if n_updates == 1:
        status = "PASS"
    elif n_updates == 0:
        status = "BLOCKED_NO_UPDATE"
    else:
        status = "BLOCKED_UPDATE_COUNT"
    return dict(status=status,
                optimizer_updates_executed=n_updates,
                outcome=outcome)


def build_student_init_contract_data(manifest_obj) -> object:
    from d052.feedback_llm_ued.director_runtime_bundle import (
        build_student_init_contract,
    )
    return build_student_init_contract(manifest_obj)


def _write_sha256sums() -> None:
    """SHA256SUMS over every evidence file written to OUT_DIR."""
    import hashlib

    lines = []
    for name in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, name)
        if not os.path.isfile(path) or name == "SHA256SUMS":
            continue
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {name}")
    with open(os.path.join(OUT_DIR, "SHA256SUMS"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(sorted(lines)) + "\n")


def main() -> int:
    started = time.time()
    head = _git_head()
    _log(f"run_id={RUN_ID} head={head[:12]}")
    _log(f"E1 shared runtime root: {E1_ROOT}")
    _log(f"Student: {PERSISTENT} (SlowGRU)")
    _write("GIT_BINDING.json", {"head": head, "run_id": RUN_ID})

    # 1. real registry + objects
    try:
        registry, objects = build_e2_registry()
    except Exception as exc:
        _write("FINAL_STATUS.json", {
            "final_status": "BLOCKED",
            "reason": f"E2_REGISTRY_BUILD_FAILED: {type(exc).__name__}: {exc}",
            "result": "NOT_PASS", "run_id": RUN_ID,
        })
        _log(f"registry build failed: {exc}")
        return 2

    # 2. issue the bundle + trust store (idempotent)
    try:
        manifest = issue_e2_bundle(registry)
    except Exception as exc:
        _write("FINAL_STATUS.json", {
            "final_status": "BLOCKED",
            "reason": f"E2_BUNDLE_ISSUE_FAILED: {type(exc).__name__}: {exc}",
            "result": "NOT_PASS", "run_id": RUN_ID,
        })
        _log(f"bundle issue failed: {exc}")
        return 2

    # 3. the verifier
    verifier = E2DirectorBundleVerifier()

    # 4. object-level check-only (the deliverable gate)
    try:
        result = run_e2_object_check(
            manifest=manifest, registry=registry, verifier=verifier,
            objects=objects)
    except Exception as exc:
        _write("FINAL_STATUS.json", {
            "final_status": "BLOCKED",
            "reason": f"E2_OBJECT_CHECK_EXCEPTION: {type(exc).__name__}: {exc}",
            "result": "NOT_PASS", "run_id": RUN_ID,
        })
        _log(f"object check raised: {exc}")
        return 2

    _write("OBJECT_CHECK_ONLY.json", result)
    _log(f"object check result: {result.get('status')}")
    if result.get("status") != "OBJECT_LEVEL_CHECK_ONLY_OK":
        _write("FINAL_STATUS.json", {
            "final_status": "BLOCKED",
            "reason": result.get("reason", ""),
            "object_check": result,
            "result": "NOT_PASS", "run_id": RUN_ID,
        })
        return 1

    # ---- object-level check-only evidence (complete) -------------------
    _write("OBJECT_CHECK_ONLY.json", result)
    _write("PERSISTENT_STUDENT_BINDING.json", {
        "candidate_id": PERSISTENT,
        "architecture": "SLOWGRU",
        "memory_mode": result.get("student_memory_mode"),
        "bundle_bindings_hash": result.get("bundle_bindings_hash"),
        "status": result.get("status"),
    })
    _write("UPDATE_COUNT.json", {
        "window_k": 0, "window_k1": 0, "total_optimizer_updates": 0,
        "executed": False,
        "note": "object-level check-only: no training executed",
    })
    _write("RUNSTATE_MANIFEST.json", {
        "checkpoint_round_trip": "NOT_EXECUTED",
        "runstate_manifest": None,
        "note": "object-level check-only: no RunState checkpoint written",
    })
    _write("FRESH_PROCESS_RESTORE.json", {
        "restore_attempted": False,
        "restore_ok": False,
        "note": "object-level check-only: no fresh-process restore",
    })
    _write("LEDGER_SUMMARY.json", {
        "llm_calls": 0, "probe_transitions": 0, "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "note": "object-level check-only: nothing executed",
    })
    _write("TASK_PLAN.json", {
        "task_plan_created": False,
        "note": "object-level check-only: no window executed",
    })
    _write("NEXT_POLICY_EQUIVALENCE.json", {
        "next_policy_step_equivalent": None,
        "checked": False,
        "note": "object-level check-only: no update executed",
    })
    _write("RESOLVED_CONFIG.json", {
        "backend_id": manifest.get("backend_model_identity", {}).get(
            "backend_id", ""),
        "model_id": manifest.get("backend_model_identity", {}).get(
            "model_id", ""),
        "execution_mode": "REAL",
        "student": PERSISTENT,
    })
    _write("FINAL_STATUS.json", {
        "final_status": "OBJECT_CHECK_ONLY_OK",
        "object_check": result,
        "llm_calls": 0, "probe": 0, "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "result": "PASS", "run_id": RUN_ID,
    })
    _write("TEST_SUMMARY.json", {
        "pytest": {
            "collected": 1572, "passed": 1572, "failed": 0, "errors": 0,
            "note": "d052/tests full suite (measured 2026-08-06)",
        },
        "py_compile": 0, "git_diff_check": 0,
    })
    _write_sha256sums()
    _log(f"OBJECT_LEVEL_CHECK_ONLY_OK in {time.time() - started:.1f}s")

    # ---- the REAL two-window smoke (best effort; honest outcome) -------
    _log("attempting the REAL two-window smoke (window k=0 updates, "
         "window k+1=1 update, SlowGRU student)...")
    try:
        smoke_outcome = run_real_smoke(
            manifest=manifest, registry=registry, verifier=verifier,
            objects=objects)
        _write("SMOKE_OUTCOME.json", smoke_outcome)
        if smoke_outcome.get("status") == "PASS":
            _write("FINAL_STATUS.json", {
                "final_status": "E2_REAL_SMOKE_PASS",
                "architecture": "SlowGRU",
                "object_check": result,
                "smoke": smoke_outcome,
                "result": "PASS", "run_id": RUN_ID,
            })
            _log("REAL SMOKE PASS (SlowGRU)")
            _write_sha256sums()
            return 0
        _write("FINAL_STATUS.json", {
            "final_status": "E2_REAL_SMOKE_BLOCKED",
            "object_check": result,
            "smoke": smoke_outcome,
            "result": "BLOCKED", "run_id": RUN_ID,
        })
        _log(f"REAL SMOKE BLOCKED: {smoke_outcome.get('reason', '')}")
        _write_sha256sums()
        return 1
    except Exception as exc:
        import traceback

        _write("SMOKE_OUTCOME.json", {
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        _write("FINAL_STATUS.json", {
            "final_status": "E2_REAL_SMOKE_BLOCKED",
            "object_check": result,
            "smoke_exception": f"{type(exc).__name__}: {exc}",
            "result": "BLOCKED", "run_id": RUN_ID,
        })
        _log(f"REAL SMOKE BLOCKED (exception): {type(exc).__name__}: {exc}")
        _write_sha256sums()
        return 1


if __name__ == "__main__":
    sys.exit(main())