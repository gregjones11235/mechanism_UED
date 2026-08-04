"""CC3 E2: the REAL SLOWGRU_PERSISTENT Student ABI compatibility baseline.

Director task (CC3 E2 from a2e1bc5): the compatibility baseline is the REAL
``SLOWGRU_PERSISTENT_CANONICAL_98304`` capsule — not fixtures. This module
binds to that capsule READ-ONLY and fail-closed:

* every capsule document it consumes is re-hashed against the capsule's own
  ``SHA256SUMS`` ledger (byte identity, no silent re-encoding);
* every ABI fact it publishes (params/checkpoint identity, optimizer and
  global step, RNG seeds, wrapper ABI, literal task params, action ABI) is
  EXTRACTED from the verified documents at bind time and cross-checked
  ACROSS documents — a missing artifact, a missing SHA entry, a byte-level
  tamper or a cross-document schema inconsistency fails closed
  (``StudentAbiBaselineBlocked``). Nothing is guessed, defaulted or
  transcribed from memory: values live in the artifacts or the baseline
  refuses to exist;
* the checkpoint pkl itself is a SERVER-ONLY artifact (the local capsule
  mirror holds documents + summaries, never weights): the baseline records
  its identity (path + file/params SHA + step counters) and the consuming
  path stays the SHA-verified wrapper ``load_candidate`` on the authorized
  GPU host. ``REAL_CHECKPOINT_LOADED`` therefore stays False — binding the
  ABI baseline loads nothing.

The second half is the SINGLE Student-evaluator consumption surface
(``SlowgruStudentEvaluator``): EnvCoder output, the three FeedbackViews, the
shared Soft Copeland ranking and the four-anchor manifest all pass through
the SAME evaluator bound to the SAME baseline. Every surface fails closed on
missing/wrong schema; the exact k-1 feedback lag and the static/shuffled
isolation invariants are re-asserted at the evaluator surface, and Soft
Copeland stays the single ranking owner (the evaluator RECONSUMES
``bagr_ued.soft_copeland.soft_copeland_rank`` output — it never re-ranks).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.soft_copeland import (
    CopelandRanking,
    EnvironmentScoreBundle,
    soft_copeland_rank,
)
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.anchor_manifest import AnchorManifestSource
from d052.feedback_llm_ued.env_coder import EnvCoderOutput
from d052.feedback_llm_ued.feedback_view import (
    MASKED_IDENTITY,
    NormalFeedbackView,
    NullFeedbackView,
    PermutedFeedbackView,
    family_level_metrics,
)
from d052.feedback_llm_ued.student_binding import local_symbolic_binding
from d052.schemas.common import CanonicalModel, is_sha256_hex

# ---------------------------------------------------------------------------
# Capsule location (repo-relative POSIX paths; read-only)
# ---------------------------------------------------------------------------
CC3_CANONICAL_CANDIDATE_ID = "SLOWGRU_PERSISTENT_CANONICAL_98304"
CAPSULE_POSIX = (
    "student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304")
RUNTIME_SOURCE_POSIX = "student_pool_v1/cc3/slowgru_runtime/slowgru_runtime.py"
RECOVERY_PROBE_POSIX = "student_pool_v1/cc3/cc3_common/recovery_probe.py"
BINDING_CONTRACT_POSIX = (
    "student_pool_v1/cc3/common_binding_wait/binding_contract_persistent.json")
SHA256SUMS_NAME = "SHA256SUMS"

#: wrapper identity (mirrored from the SHA-verified runtime source literals)
RUNTIME_NAME = "THIN_GTRXL128_SLOWGRU_RUNTIME"
RUNTIME_ABI_VERSION = "cc3_runtime_abi/v1"
RUNTIME_ABI_SURFACE = (
    "load_candidate",
    "seed_policy_rng",
    "init_memory",
    "policy_step",
    "reset_memory",
    "on_segment_boundary",
    "candidate_metadata",
)
#: on_segment_boundary action of the PERSISTENT arm (runtime literal)
PERSISTENT_BOUNDARY_ACTION = "FULL_CARRY_NO_CLEAR"

#: literals that MUST appear in the SHA-verified wrapper source — binding the
#: source hash alone is not enough: the ABI surface names are part of the
#: baseline and their absence fails closed (RUNTIME_ABI_LITERAL_MISSING).
_RUNTIME_LITERAL_FACTS = (
    'RUNTIME_NAME = "THIN_GTRXL128_SLOWGRU_RUNTIME"',
    'ABI_VERSION = "cc3_runtime_abi/v1"',
    "OBS_DIM = 8335",
    "ACTION_DIM = 43",
    "WINDOW_MEM = 128",
    "NUM_LAYERS = 2",
    "EMBED_SIZE = 256",
    "NUM_HEADS = 8",
    PERSISTENT_BOUNDARY_ACTION,
) + tuple(f"def {name}" for name in RUNTIME_ABI_SURFACE)

#: full_state.pkl top-level layout — mirrored VERBATIM from
#: ``REQUIRED_PKL_KEYS`` in the SHA-verified CC3 recovery probe
#: (student_pool_v1/cc3/cc3_common/recovery_probe.py). The baseline re-reads
#: that source at bind time and fails closed if any key literal is absent.
CC3_PKL_REQUIRED_KEYS = frozenset({
    "params", "opt_state", "opt_step", "env_state", "memories",
    "memories_mask", "memories_mask_idx", "obs", "done", "true_done",
    "longstate", "step_env_currentloop", "update_step", "rng", "global_step",
    "update_count", "config", "code_sha256", "manifest",
})
#: the pytree-valued pkl entries are stored PACKED as ``(leaves, treedef)``
#: by every CC3 tool (params_sha is the sha256 over the concatenated
#: little-endian raw bytes of the leaves in tree order — see
#: ``params_sha_packed`` in the runtime / recovery probe).
CC3_PACKED_PYTREE_KEYS = ("params", "opt_state", "longstate")

#: capsule files that must exist locally (manifest.capsule_files + the run
#: artifacts under out/ + the SHA ledger itself)
REQUIRED_LOCAL_FILES = (
    "candidate_manifest.json",
    "training_contract.json",
    "checkpoint_contract.json",
    "candidate_runtime.py",
    "evaluate_candidate.py",
    "interface_smoke_result.json",
    "environment_lock.json",
    "identity_verification.json",
    "common_evaluator_binding_result.json",
    "SHA256SUMS",
    "READY.json",
    "out/exact_resume_proof.json",
    "out/CC3_SLOWGRU_CANONICAL_PERSISTENT_train_summary.json",
    "out/CC3_SLOWGRU_CANONICAL_PERSISTENT_per_update.jsonl",
)

#: checkpoint location classes (honest bookkeeping — the pkl body is NOT in
#: the local mirror; only its identity travels with the documents)
CHECKPOINT_LOCATION_LOCAL_VERIFIED = "LOCAL_VERIFIED_ARTIFACT"
CHECKPOINT_LOCATION_SERVER_ONLY = "SERVER_ONLY_ARTIFACT_NOT_LOCAL"

# evaluator surface names / statuses
SURFACE_ENV_CODER = "env_coder"
SURFACE_FEEDBACK_VIEW = "feedback_view"
SURFACE_SOFT_COPELAND = "soft_copeland"
SURFACE_ANCHOR_MANIFEST = "anchor_manifest"
STATUS_COMPATIBLE = "COMPATIBLE"
STATUS_BLOCKED = "BLOCKED"


class StudentAbiBaselineBlocked(RuntimeError):
    """Fail-closed refusal while binding the real-capsule ABI baseline."""


class StudentCompatibilityBlocked(RuntimeError):
    """Fail-closed refusal of one Student-evaluator consumption surface."""


def default_repo_root() -> Path:
    """The worktree root that owns ``student_pool_v1/`` (this repo: the
    package lives at ``<root>/gpu1_aggregation_siege/d052/feedback_llm_ued``).
    """
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: verification modes a document may be recorded with — anything else fails
#: closed. ``worktree_crlf_to_lf_view`` is admitted ONLY when the on-disk
#: bytes are EXACTLY the LF content with every newline re-inflated to CRLF
#: (git ``i/lf w/crlf`` checkout on Windows): a reversible platform view,
#: never a content change, and always recorded in the baseline.
VERIFY_BYTE_IDENTICAL = "byte_identical"
VERIFY_CRLF_VIEW = "worktree_crlf_to_lf_view"


def _verify_ledger_bytes(path: Path, ledger_sha: str, rel_posix: str) -> str:
    """Verify one document against the capsule ledger. Returns the
    verification mode; fail closed on any other byte difference."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == ledger_sha:
        return VERIFY_BYTE_IDENTICAL
    lf_view = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(lf_view).hexdigest() == ledger_sha \
            and raw == lf_view.replace(b"\n", b"\r\n"):
        return VERIFY_CRLF_VIEW
    raise StudentAbiBaselineBlocked(
        f"CAPSULE_DOC_SHA_MISMATCH: {rel_posix} recomputes to "
        f"{hashlib.sha256(raw).hexdigest()} (LF view "
        f"{hashlib.sha256(lf_view).hexdigest()}) but the capsule ledger "
        f"records {ledger_sha}")


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_DOCUMENT_UNPARSEABLE: {path.name}: {exc}") from exc


def _require_key(doc: Mapping, key: str, doc_name: str):
    if key not in doc:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_SCHEMA_MISMATCH: {doc_name} is missing required "
            f"field {key!r} — no guessing, the baseline refuses to bind")
    return doc[key]


def _require_equal(values: Mapping[str, object], what: str) -> object:
    """All named values must be identical (cross-document consistency)."""
    distinct = {name: value for name, value in values.items()}
    first = next(iter(values.values()))
    for name, value in values.items():
        if value != first:
            raise StudentAbiBaselineBlocked(
                f"CAPSULE_IDENTITY_MISMATCH: {what} disagrees across "
                f"documents: {distinct}")
    return first


def parse_sha256sums(text: str) -> Dict[str, str]:
    """Parse the capsule SHA ledger ``<sha256>  <path>`` (two-space form).

    Keys are kept VERBATIM (capsule-relative, ``../``-relative or absolute
    server paths). A malformed line fails closed.
    """
    entries: Dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not is_sha256_hex(parts[0].strip()) \
                or not parts[1].strip():
            raise StudentAbiBaselineBlocked(
                f"SHA256SUMS_MALFORMED: line {line_no}: {line!r}")
        entries[parts[1].strip()] = parts[0].strip()
    return entries


# ---------------------------------------------------------------------------
# ABI schema (frozen, extra=forbid everywhere via CanonicalModel)
# ---------------------------------------------------------------------------
class StudentActionAbi(CanonicalModel):
    """The literal observation/action ABI of the canonical Student."""

    obs_dim: int = Field(gt=0)
    action_dim: int = Field(gt=0)
    legal_action_min: int = Field(ge=0)
    legal_action_max: int = Field(ge=0)
    conditioning_dim: int = Field(ge=0)
    observation_shape: Tuple[int, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _legal_range(self) -> "StudentActionAbi":
        if self.legal_action_min != 0:
            raise ValueError(
                f"ILLEGAL_ACTION_RANGE: legal actions must start at 0, got "
                f"{self.legal_action_min}")
        if self.legal_action_max != self.action_dim - 1:
            raise ValueError(
                f"ILLEGAL_ACTION_RANGE: legal actions must be exactly "
                f"0..{self.action_dim - 1}, got 0..{self.legal_action_max}")
        if self.observation_shape and \
                tuple(self.observation_shape) != (self.obs_dim,):
            raise ValueError(
                f"OBSERVATION_SHAPE_MISMATCH: {self.observation_shape} vs "
                f"obs_dim {self.obs_dim}")
        return self


class StudentCheckpointIdentity(CanonicalModel):
    """Identity of the canonical checkpoint node (server-only artifact)."""

    checkpoint_path: str = Field(min_length=1)
    checkpoint_file_sha256: str
    params_sha256: str
    canonical_base_params_sha256: str
    global_step: int = Field(gt=0)
    update_step: int = Field(gt=0)
    opt_step: int = Field(gt=0)
    resume_source_file_sha256: str
    resume_source_params_sha256: str
    checkpoint_location_class: str = Field(min_length=1)

    @model_validator(mode="after")
    def _hashes(self) -> "StudentCheckpointIdentity":
        for name in ("checkpoint_file_sha256", "params_sha256",
                     "canonical_base_params_sha256",
                     "resume_source_file_sha256",
                     "resume_source_params_sha256"):
            value = getattr(self, name)
            if not is_sha256_hex(value):
                raise ValueError(f"INVALID_HASH: {name}={value!r}")
        return self


class StudentRngPolicy(CanonicalModel):
    """The literal RNG seeds of the capsule (training + interface smoke +
    the CC4 held-out full-smoke bank)."""

    training_seed: int = Field(ge=0)
    smoke_seed: int = Field(ge=0)
    full_smoke_seed_base: int = Field(ge=0)
    full_smoke_seed_count: int = Field(gt=0)
    full_seed_source: str = Field(min_length=1)
    cc3_created_full_seeds: bool = True

    @model_validator(mode="after")
    def _not_cc3_seeds(self) -> "StudentRngPolicy":
        if self.cc3_created_full_seeds:
            raise ValueError(
                "RNG_POLICY_MISMATCH: the held-out full-smoke seeds belong "
                "to the CC4 common evaluation profile "
                "(cc3_created_full_seeds must be false)")
        return self


class StudentWrapperAbi(CanonicalModel):
    """The unified CC3 runtime ABI the candidate is served through."""

    runtime_name: str = Field(min_length=1)
    abi_version: str = Field(min_length=1)
    abi_surface: Tuple[str, ...] = Field(default_factory=tuple)
    carry_mode: str = Field(min_length=1)
    boundary_action: str = Field(min_length=1)
    shared_runtime_src_sha256: str
    candidate_runtime_sha256: str
    window_mem: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    embed_size: int = Field(gt=0)
    slow_interval: int = Field(gt=0)
    slow_dim: int = Field(gt=0)

    @model_validator(mode="after")
    def _hashes(self) -> "StudentWrapperAbi":
        for name in ("shared_runtime_src_sha256",
                     "candidate_runtime_sha256"):
            if not is_sha256_hex(getattr(self, name)):
                raise ValueError(f"INVALID_HASH: {name}")
        if tuple(self.abi_surface) != RUNTIME_ABI_SURFACE:
            raise ValueError(
                f"RUNTIME_ABI_SURFACE_MISMATCH: {self.abi_surface}")
        return self


class StudentTaskParamsLiteral(CanonicalModel):
    """The LITERAL task parameters of the canonical run (no paraphrase)."""

    task: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    bonus_type: str = Field(min_length=1)
    condition_on_task: bool
    replay: str = Field(min_length=1)
    vtrace: str = Field(min_length=1)
    hindsight: bool
    awr: bool
    egomap: str = Field(min_length=1)
    nav_aux: str = Field(min_length=1)
    novelty: str = Field(min_length=1)
    total_env_steps: int = Field(gt=0)
    xla_flags: str = Field(min_length=1)

    @model_validator(mode="after")
    def _aux_off(self) -> "StudentTaskParamsLiteral":
        if self.replay != "OFF" or self.vtrace != "OFF":
            raise ValueError(
                f"TASK_PARAMS_MISMATCH: replay/vtrace must both be OFF, "
                f"got {self.replay}/{self.vtrace}")
        if self.hindsight or self.awr:
            raise ValueError("TASK_PARAMS_MISMATCH: hindsight/awr must be "
                             "false for the canonical arm")
        for name in ("egomap", "nav_aux", "novelty"):
            if getattr(self, name) != "OFF":
                raise ValueError(
                    f"TASK_PARAMS_MISMATCH: {name} must be OFF")
        return self


class StudentGpuPolicy(CanonicalModel):
    gpu_allowed: Tuple[str, ...] = Field(default_factory=tuple)
    gpu_forbidden: Tuple[str, ...] = Field(default_factory=tuple)


class StudentEnvironmentLock(CanonicalModel):
    conda_env: str = Field(min_length=1)
    python: str = Field(min_length=1)
    jax: str = Field(min_length=1)
    flax: str = Field(min_length=1)
    optax: str = Field(min_length=1)
    chex: str = Field(min_length=1)
    distrax: str = Field(min_length=1)
    numpy: str = Field(min_length=1)
    local_jax_craftax_forbidden: bool
    xla_flags: str = Field(min_length=1)


class StudentCheckpointStateLayout(CanonicalModel):
    """Facts about the saved full_state.pkl — mirrored from the SHA-verified
    CC3 tooling, never guessed (the pkl itself stays server-only)."""

    required_pkl_keys: Tuple[str, ...] = Field(default_factory=tuple)
    packed_pytree_keys: Tuple[str, ...] = Field(default_factory=tuple)
    params_sha_algorithm: str = Field(min_length=1)

    @model_validator(mode="after")
    def _layout(self) -> "StudentCheckpointStateLayout":
        if frozenset(self.required_pkl_keys) != CC3_PKL_REQUIRED_KEYS:
            raise ValueError("PKL_LAYOUT_MISMATCH: required_pkl_keys "
                             "drifted from the CC3 recovery probe contract")
        if tuple(self.packed_pytree_keys) != CC3_PACKED_PYTREE_KEYS:
            raise ValueError("PKL_LAYOUT_MISMATCH: packed_pytree_keys")
        return self


class StudentAbiBaseline(CanonicalModel):
    """The frozen compatibility baseline derived from the real capsule."""

    candidate_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    network_family: str = Field(min_length=1)
    carry_mode: str = Field(min_length=1)
    budget_class: str = Field(min_length=1)
    formal_eval_binding: str = Field(min_length=1)
    action_abi: StudentActionAbi
    checkpoint: StudentCheckpointIdentity
    rng_policy: StudentRngPolicy
    wrapper: StudentWrapperAbi
    task_params: StudentTaskParamsLiteral
    gpu_policy: StudentGpuPolicy
    environment_lock: StudentEnvironmentLock
    state_layout: StudentCheckpointStateLayout
    constructor: Dict[str, object] = Field(default_factory=dict)
    memory_layout: Dict[str, object] = Field(default_factory=dict)
    capsule_document_shas: Dict[str, str] = Field(default_factory=dict)
    #: documents whose ledger SHA held only under the reversible CRLF→LF
    #: worktree view (git i/lf w/crlf) — recorded, never silent
    crlf_view_documents: Tuple[str, ...] = Field(default_factory=tuple)
    baseline_hash: str = ""

    @model_validator(mode="after")
    def _consistency_and_hash(self) -> "StudentAbiBaseline":
        if self.candidate_id != CC3_CANONICAL_CANDIDATE_ID:
            raise ValueError(
                f"BASELINE_CANDIDATE_MISMATCH: {self.candidate_id!r}")
        if self.carry_mode != "PERSISTENT":
            raise ValueError(
                f"BASELINE_CARRY_MODE_MISMATCH: {self.carry_mode!r} — the "
                "E2 compatibility baseline is the PERSISTENT arm")
        if not self.baseline_hash:
            payload = self.model_dump()
            payload.pop("baseline_hash", None)
            object.__setattr__(self, "baseline_hash",
                               canonical_sha256(payload))
        return self


# ---------------------------------------------------------------------------
# Binding (read-only, fail-closed, no guessing)
# ---------------------------------------------------------------------------
def bind_slowgru_persistent_baseline(
        repo_root: Optional[Path] = None) -> StudentAbiBaseline:
    """Bind the REAL capsule as the compatibility baseline. Fail closed on
    any missing artifact, missing SHA entry, byte tamper or cross-document
    schema inconsistency. Loads NO checkpoint and executes NO network/GPU
    call — documents only."""
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    capsule = root / CAPSULE_POSIX
    if not capsule.is_dir():
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_ROOT_MISSING: {capsule} — the real "
            f"{CC3_CANONICAL_CANDIDATE_ID} capsule is required; fixtures "
            "are not a substitute (no guessing)")

    # 1. required local files ------------------------------------------------
    for rel in REQUIRED_LOCAL_FILES:
        if not (capsule / rel).is_file():
            raise StudentAbiBaselineBlocked(
                f"REQUIRED_CAPSULE_ARTIFACT_MISSING: {rel}")

    # 2. SHA ledger ------------------------------------------------------------
    ledger_text = (capsule / SHA256SUMS_NAME).read_text(encoding="utf-8")
    ledger = parse_sha256sums(ledger_text)

    verification_modes: Dict[str, str] = {}

    def _verified_sha(rel_posix: str, resolved: Path) -> str:
        entry = ledger.get(rel_posix)
        if entry is None:
            raise StudentAbiBaselineBlocked(
                f"SHA256SUMS_ENTRY_MISSING: no ledger entry for {rel_posix}")
        mode = _verify_ledger_bytes(resolved, entry, rel_posix)
        verification_modes[rel_posix] = mode
        return entry

    document_shas: Dict[str, str] = {}
    for rel in REQUIRED_LOCAL_FILES:
        if rel == SHA256SUMS_NAME:
            continue
        if rel == "READY.json":
            # status document: not byte-bound by the ledger; its CONTENT is
            # cross-checked against the byte-bound contracts below
            continue
        document_shas[rel] = _verified_sha(rel, capsule / rel)

    # 3. parse the byte-verified documents ------------------------------------
    manifest = _load_json(capsule / "candidate_manifest.json")
    contract = _load_json(capsule / "checkpoint_contract.json")
    ready = _load_json(capsule / "READY.json")
    binding_result = _load_json(
        capsule / "common_evaluator_binding_result.json")
    smoke = _load_json(capsule / "interface_smoke_result.json")
    env_lock = _load_json(capsule / "environment_lock.json")
    train_summary = _load_json(
        capsule / "out/CC3_SLOWGRU_CANONICAL_PERSISTENT_train_summary.json")
    resume_proof = _load_json(capsule / "out/exact_resume_proof.json")

    # 4. cross-document identity ----------------------------------------------
    candidate_id = _require_equal({
        "candidate_manifest": _require_key(
            manifest, "candidate_id", "candidate_manifest.json"),
        "checkpoint_contract": _require_key(
            contract, "candidate_id", "checkpoint_contract.json"),
        "READY": _require_key(ready, "candidate_id", "READY.json"),
        "common_evaluator_binding_result": _require_key(
            binding_result, "candidate_id",
            "common_evaluator_binding_result.json"),
        "interface_smoke_result": _require_key(
            smoke, "candidate_id", "interface_smoke_result.json"),
    }, "candidate_id")
    if candidate_id != CC3_CANONICAL_CANDIDATE_ID:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_IDENTITY_MISMATCH: capsule candidate_id "
            f"{candidate_id!r} is not the E2 baseline candidate "
            f"{CC3_CANONICAL_CANDIDATE_ID!r}")

    canonical_pkl_rel = _require_key(ready, "canonical_checkpoint",
                                     "READY.json")
    pkl_ledger = ledger.get(canonical_pkl_rel)
    if pkl_ledger is None:
        raise StudentAbiBaselineBlocked(
            f"SHA256SUMS_ENTRY_MISSING: no ledger entry for "
            f"{canonical_pkl_rel!r}")
    file_sha = _require_equal({
        "checkpoint_contract": _require_key(
            contract, "checkpoint_file_sha256", "checkpoint_contract.json"),
        "READY": _require_key(ready, "canonical_file_sha256", "READY.json"),
        "manifest.canonical_artifact": _require_key(
            _require_key(manifest, "canonical_artifact",
                         "candidate_manifest.json"),
            "file_sha256", "candidate_manifest.canonical_artifact"),
        "binding_result.params_unchanged_evidence": _require_key(
            _require_key(binding_result, "params_unchanged_evidence",
                         "common_evaluator_binding_result.json"),
            "checkpoint_file_sha256_recomputed",
            "common_evaluator_binding_result.params_unchanged_evidence"),
        "interface_smoke_result": _require_key(
            smoke, "checkpoint_file_sha256", "interface_smoke_result.json"),
        "SHA256SUMS": pkl_ledger,
    }, "checkpoint_file_sha256")

    params_sha = _require_equal({
        "checkpoint_contract": _require_key(
            contract, "params_sha256", "checkpoint_contract.json"),
        "READY": _require_key(ready, "canonical_params_sha256", "READY.json"),
        "manifest.canonical_artifact": _require_key(
            _require_key(manifest, "canonical_artifact",
                         "candidate_manifest.json"),
            "params_sha256", "candidate_manifest.canonical_artifact"),
        "interface_smoke_result": _require_key(
            smoke, "params_sha256", "interface_smoke_result.json"),
    }, "params_sha256")

    # 4b. provenance cross-checks (resume source / network / task code /
    #     trainer-as-run) — every ledger key is a literal carried by the
    #     documents themselves, never a path invented here -----------------
    resume_source_path = _require_key(contract, "resume_source",
                                      "checkpoint_contract.json")
    resume_ledger = ledger.get(resume_source_path)
    if resume_ledger is None:
        raise StudentAbiBaselineBlocked(
            f"SHA256SUMS_ENTRY_MISSING: no ledger entry for the resume "
            f"source {resume_source_path!r}")
    _require_equal({
        "checkpoint_contract": _require_key(
            contract, "resume_source_file_sha256",
            "checkpoint_contract.json"),
        "SHA256SUMS": resume_ledger,
        "exact_resume_proof": _require_key(
            resume_proof, "resume_file_sha256", "exact_resume_proof.json"),
        "train_summary.exact_resume_proof": _require_key(
            _require_key(train_summary, "exact_resume_proof",
                         "train_summary.json"),
            "resume_file_sha256", "train_summary.exact_resume_proof"),
    }, "resume_source_file_sha256")
    _require_equal({
        "checkpoint_contract": _require_key(
            contract, "resume_source_params_sha256",
            "checkpoint_contract.json"),
        "exact_resume_proof": _require_key(
            resume_proof, "resume_params_sha256",
            "exact_resume_proof.json"),
        "manifest.resume_source_recovery": _require_key(
            _require_key(manifest, "resume_source_recovery",
                         "candidate_manifest.json"),
            "params_sha256", "candidate_manifest.resume_source_recovery"),
    }, "resume_source_params_sha256")
    network_path = (_require_key(contract, "arm_src",
                                 "checkpoint_contract.json").rstrip("/")
                    + "/" + _require_key(contract, "network_module",
                                         "checkpoint_contract.json") + ".py")
    network_ledger = ledger.get(network_path)
    if network_ledger is None:
        raise StudentAbiBaselineBlocked(
            f"SHA256SUMS_ENTRY_MISSING: no ledger entry for the network "
            f"source {network_path!r}")
    _require_equal({
        "checkpoint_contract": _require_key(
            contract, "network_src_sha256", "checkpoint_contract.json"),
        "interface_smoke.metadata": _require_key(
            _require_key(smoke, "metadata", "interface_smoke_result.json"),
            "network_src_sha256", "interface_smoke_result.metadata"),
        "train_summary.code_sha256.network": _require_key(
            _require_key(train_summary, "code_sha256", "train_summary.json"),
            "network", "train_summary.code_sha256"),
        "SHA256SUMS": network_ledger,
    }, "network_src_sha256")
    s4_task_path = _require_key(env_lock, "s4_task_path",
                                "environment_lock.json")
    s4_task_ledger = ledger.get(s4_task_path)
    if s4_task_ledger is None:
        raise StudentAbiBaselineBlocked(
            f"SHA256SUMS_ENTRY_MISSING: no ledger entry for the S4 task "
            f"source {s4_task_path!r}")
    _require_equal({
        "checkpoint_contract": _require_key(
            contract, "s4_task_sha256", "checkpoint_contract.json"),
        "environment_lock": _require_key(env_lock, "s4_task_sha256",
                                         "environment_lock.json"),
        "train_summary.code_sha256.s4_task": _require_key(
            _require_key(train_summary, "code_sha256", "train_summary.json"),
            "s4_task", "train_summary.code_sha256"),
        "SHA256SUMS": s4_task_ledger,
    }, "s4_task_sha256")
    _require_equal({
        "checkpoint_contract": _require_key(
            contract, "canonical_trainer_src_sha256",
            "checkpoint_contract.json"),
        "train_summary.code_sha256.launcher": _require_key(
            _require_key(train_summary, "code_sha256", "train_summary.json"),
            "launcher", "train_summary.code_sha256"),
    }, "canonical_trainer_src_sha256 (trainer AS RUN; the on-disk driver "
       "later received a bookkeeping-only fix and legitimately differs)")

    # 5. step counters: the triple-consistency gate (params sha of the node
    #    saved at global_step must equal the contract params sha) -------------
    global_step = _require_equal({
        "checkpoint_contract": _require_key(
            contract, "global_step", "checkpoint_contract.json"),
        "manifest.training_steps": _require_key(
            manifest, "training_steps", "candidate_manifest.json"),
        "READY.total_env_steps": _require_key(
            ready, "total_env_steps", "READY.json"),
    }, "global_step/training_steps/total_env_steps")
    update_step = _require_key(contract, "update_step",
                               "checkpoint_contract.json")
    opt_step = _require_key(contract, "opt_step", "checkpoint_contract.json")
    canonical_chunks = [
        chunk for chunk in _require_key(train_summary, "chunks",
                                        "train_summary.json")
        if chunk.get("global_step") == global_step]
    if len(canonical_chunks) != 1:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_STEP_CONSISTENCY_MISMATCH: train_summary has "
            f"{len(canonical_chunks)} chunk(s) at global_step={global_step}; "
            "exactly one is required")
    chunk = canonical_chunks[0]
    if chunk.get("params_sha256") != params_sha:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_STEP_CONSISTENCY_MISMATCH: train_summary chunk at "
            f"global_step={global_step} carries params_sha256="
            f"{chunk.get('params_sha256')!r} != contract {params_sha!r}")
    if chunk.get("update_count") != update_step or \
            chunk.get("opt_step") != opt_step:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_STEP_CONSISTENCY_MISMATCH: chunk counters "
            f"update_count={chunk.get('update_count')!r}/"
            f"opt_step={chunk.get('opt_step')!r} != contract "
            f"update_step={update_step!r}/opt_step={opt_step!r}")
    if _require_key(ready, "READY", "READY.json") is not True:
        raise StudentAbiBaselineBlocked(
            "CAPSULE_NOT_READY: READY.json does not declare READY=true")
    if not _require_key(resume_proof, "EXACT_RESUME_PROOF_PASS",
                        "exact_resume_proof.json"):
        raise StudentAbiBaselineBlocked(
            "CAPSULE_EXACT_RESUME_PROOF_MISSING: EXACT_RESUME_PROOF_PASS "
            "is not true")

    # 6. seeds ------------------------------------------------------------------
    training_seed = _require_equal({
        "checkpoint_contract": _require_key(
            contract, "training_seed", "checkpoint_contract.json"),
        "manifest": _require_key(manifest, "training_seed",
                                 "candidate_manifest.json"),
        "READY.seed": _require_key(ready, "seed", "READY.json"),
        "train_summary.protocol.seed": _require_key(
            _require_key(train_summary, "protocol", "train_summary.json"),
            "seed", "train_summary.protocol"),
    }, "training_seed")
    full_seed_source = _require_key(binding_result, "full_seed_source",
                                    "common_evaluator_binding_result.json")
    if "200000..200063" not in str(full_seed_source):
        raise StudentAbiBaselineBlocked(
            "CAPSULE_RNG_MISMATCH: full_seed_source does not document the "
            "64 held-out seeds 200000..200063")

    # 7. action ABI -------------------------------------------------------------
    obs_dim = _require_equal({
        "checkpoint_contract": _require_key(
            contract, "obs_dim", "checkpoint_contract.json"),
        "manifest": _require_key(manifest, "obs_dim",
                                 "candidate_manifest.json"),
    }, "obs_dim")
    action_dim = _require_equal({
        "checkpoint_contract": _require_key(
            contract, "action_dim", "checkpoint_contract.json"),
        "manifest": _require_key(manifest, "action_dim",
                                 "candidate_manifest.json"),
    }, "action_dim")
    smoke_checks = {
        entry.get("check"): entry
        for entry in _require_key(smoke, "check_details",
                                  "interface_smoke_result.json")}

    def _passed(check_name: str, must_contain: Sequence[str] = ()) -> str:
        entry = smoke_checks.get(check_name)
        if entry is None or entry.get("passed") is not True:
            raise StudentAbiBaselineBlocked(
                f"CAPSULE_ACTION_ABI_MISMATCH: interface smoke check "
                f"{check_name!r} missing or not passed")
        detail = str(entry.get("detail", ""))
        for frag in must_contain:
            if frag not in detail:
                raise StudentAbiBaselineBlocked(
                    f"CAPSULE_ACTION_ABI_MISMATCH: check {check_name!r} "
                    f"detail {detail!r} lacks {frag!r}")
        return detail

    _passed("obs_dim_8335")
    _passed("action_dim_43")
    conditioning_detail = _passed("conditioning_emb_67")
    try:
        conditioning_dim = int(conditioning_detail.strip())
    except ValueError as exc:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_ACTION_ABI_MISMATCH: conditioning_emb_67 detail "
            f"{conditioning_detail!r} is not an integer literal") from exc
    legal_detail = _passed("actions_legal", ("min=0", "max=42"))
    legal_max = int(legal_detail.split("max=")[1].split()[0])
    legal_min = int(legal_detail.split("min=")[1].split()[0])
    if obs_dim != 8335 or action_dim != 43:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_ACTION_ABI_MISMATCH: obs_dim={obs_dim} "
            f"action_dim={action_dim} — the E2 baseline is the 8335/43 "
            "canonical Student")

    # 8. carry mode + formal binding ---------------------------------------------
    carry_mode = _require_equal({
        "manifest": _require_key(manifest, "carry_mode",
                                 "candidate_manifest.json"),
        "checkpoint_contract": _require_key(
            contract, "carry_mode", "checkpoint_contract.json"),
        "READY": _require_key(ready, "carry_mode", "READY.json"),
        "interface_smoke_result": _require_key(
            smoke, "carry_mode", "interface_smoke_result.json"),
    }, "carry_mode")
    formal_binding = _require_equal({
        "manifest": _require_key(manifest, "formal_eval_binding",
                                 "candidate_manifest.json"),
        "checkpoint_contract": _require_key(
            contract, "formal_eval_binding", "checkpoint_contract.json"),
        "READY": _require_key(ready, "formal_eval_binding", "READY.json"),
    }, "formal_eval_binding")

    # 9. wrapper source (SHA-verified) + runtime literals ------------------------
    runtime_path = root / RUNTIME_SOURCE_POSIX
    if not runtime_path.is_file():
        raise StudentAbiBaselineBlocked(
            f"REQUIRED_CAPSULE_ARTIFACT_MISSING: {RUNTIME_SOURCE_POSIX}")
    runtime_sha = _verified_sha("../slowgru_runtime/slowgru_runtime.py",
                                runtime_path)
    runtime_src = runtime_path.read_text(encoding="utf-8")
    for literal in _RUNTIME_LITERAL_FACTS:
        if literal not in runtime_src:
            raise StudentAbiBaselineBlocked(
                f"RUNTIME_ABI_LITERAL_MISSING: {literal!r} not found in the "
                "SHA-verified wrapper source")
    if _require_key(smoke, "runtime", "interface_smoke_result.json") \
            != RUNTIME_NAME or _require_key(
                smoke, "abi_version", "interface_smoke_result.json") \
            != RUNTIME_ABI_VERSION:
        raise StudentAbiBaselineBlocked(
            "CAPSULE_WRAPPER_MISMATCH: interface smoke was not executed "
            f"through {RUNTIME_NAME}@{RUNTIME_ABI_VERSION}")
    boundary_action = _require_key(
        _require_key(_require_key(smoke, "boundary_event",
                                  "interface_smoke_result.json"),
                     "info", "interface_smoke_result.boundary_event"),
        "boundary_action", "interface_smoke_result.boundary_event.info")
    if boundary_action != PERSISTENT_BOUNDARY_ACTION:
        raise StudentAbiBaselineBlocked(
            f"CAPSULE_BOUNDARY_MISMATCH: PERSISTENT boundary action is "
            f"{boundary_action!r}, expected {PERSISTENT_BOUNDARY_ACTION!r}")

    # 10. pkl state layout (SHA-verified recovery probe) -------------------------
    probe_path = root / RECOVERY_PROBE_POSIX
    if not probe_path.is_file():
        raise StudentAbiBaselineBlocked(
            f"REQUIRED_CAPSULE_ARTIFACT_MISSING: {RECOVERY_PROBE_POSIX}")
    _verified_sha("../cc3_common/recovery_probe.py", probe_path)
    probe_src = probe_path.read_text(encoding="utf-8")
    for key in sorted(CC3_PKL_REQUIRED_KEYS):
        if f'"{key}"' not in probe_src:
            raise StudentAbiBaselineBlocked(
                f"PKL_LAYOUT_LITERAL_MISSING: {key!r} not found in the "
                "SHA-verified recovery probe REQUIRED_PKL_KEYS")

    # 11. common binding wait contract (obs/action cross-check) ------------------
    binding_path = root / BINDING_CONTRACT_POSIX
    if not binding_path.is_file():
        raise StudentAbiBaselineBlocked(
            f"REQUIRED_CAPSULE_ARTIFACT_MISSING: {BINDING_CONTRACT_POSIX}")
    binding_contract = _load_json(binding_path)
    if _require_key(binding_contract, "observation_shape",
                    "binding_contract_persistent.json") != [obs_dim] or \
            _require_key(binding_contract, "action_dim",
                         "binding_contract_persistent.json") != action_dim or \
            _require_key(binding_contract, "arm",
                         "binding_contract_persistent.json") != "persistent":
        raise StudentAbiBaselineBlocked(
            "CAPSULE_ABI_MISMATCH: common_binding_wait contract disagrees "
            "with the capsule obs/action ABI")

    # 12. literal task params ------------------------------------------------------
    protocol = _require_key(train_summary, "protocol", "train_summary.json")
    config = _require_key(train_summary, "config", "train_summary.json")
    if _require_key(protocol, "goal", "train_summary.protocol") \
            != "DEFEAT_KOBOLD" or _require_key(
                protocol, "stage", "train_summary.protocol") \
            != "S4_dark native":
        raise StudentAbiBaselineBlocked(
            "CAPSULE_TASK_MISMATCH: canonical task is DEFEAT_KOBOLD "
            "(S4_dark native)")

    # 13. checkpoint locality (the pkl body is server-only in this mirror) --------
    local_pkl = capsule / "ckpt/98304/full_state.pkl"
    checkpoint_path = _require_key(contract, "checkpoint_path",
                                   "checkpoint_contract.json")
    if local_pkl.is_file():
        if _sha256_file(local_pkl) != file_sha:
            raise StudentAbiBaselineBlocked(
                "CAPSULE_DOC_SHA_MISMATCH: local ckpt/98304/full_state.pkl "
                "does not recompute to the contract file SHA")
        location_class = CHECKPOINT_LOCATION_LOCAL_VERIFIED
    else:
        location_class = CHECKPOINT_LOCATION_SERVER_ONLY

    # 14. gpu policy / environment lock ----------------------------------------------
    gpu_allowed = tuple(_require_key(contract, "gpu_allowed",
                                     "checkpoint_contract.json"))
    lock_allowed = tuple(_require_key(env_lock, "gpu_allowed",
                                      "environment_lock.json"))
    if set(gpu_allowed) != set(lock_allowed):
        raise StudentAbiBaselineBlocked(
            "CAPSULE_GPU_POLICY_MISMATCH: checkpoint_contract and "
            "environment_lock disagree on gpu_allowed")
    modules = _require_key(env_lock, "modules", "environment_lock.json")

    constructor = _require_key(contract, "constructor",
                               "checkpoint_contract.json")
    memory_layout = _require_key(contract, "memory_layout",
                                 "checkpoint_contract.json")
    if constructor.get("action_dim") != action_dim:
        raise StudentAbiBaselineBlocked(
            "CAPSULE_ABI_MISMATCH: constructor.action_dim disagrees with "
            "the capsule action ABI")

    return StudentAbiBaseline(
        candidate_id=candidate_id,
        owner=_require_key(manifest, "owner", "candidate_manifest.json"),
        network_family=_require_key(manifest, "network_family",
                                    "candidate_manifest.json"),
        carry_mode=carry_mode,
        budget_class=_require_key(manifest, "budget_class",
                                  "candidate_manifest.json"),
        formal_eval_binding=formal_binding,
        action_abi=StudentActionAbi(
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            legal_action_min=legal_min,
            legal_action_max=legal_max,
            conditioning_dim=conditioning_dim,
            observation_shape=tuple(_require_key(
                binding_contract, "observation_shape",
                "binding_contract_persistent.json"))),
        checkpoint=StudentCheckpointIdentity(
            checkpoint_path=checkpoint_path,
            checkpoint_file_sha256=file_sha,
            params_sha256=params_sha,
            canonical_base_params_sha256=_require_key(
                contract, "canonical_base_params_sha256",
                "checkpoint_contract.json"),
            global_step=int(global_step),
            update_step=int(update_step),
            opt_step=int(opt_step),
            resume_source_file_sha256=_require_key(
                contract, "resume_source_file_sha256",
                "checkpoint_contract.json"),
            resume_source_params_sha256=_require_key(
                contract, "resume_source_params_sha256",
                "checkpoint_contract.json"),
            checkpoint_location_class=location_class),
        rng_policy=StudentRngPolicy(
            training_seed=int(training_seed),
            smoke_seed=int(_require_key(contract, "smoke_seed",
                                        "checkpoint_contract.json")),
            full_smoke_seed_base=200000,
            full_smoke_seed_count=64,
            full_seed_source=str(full_seed_source),
            cc3_created_full_seeds=bool(_require_key(
                binding_result, "cc3_created_full_seeds",
                "common_evaluator_binding_result.json"))),
        wrapper=StudentWrapperAbi(
            runtime_name=RUNTIME_NAME,
            abi_version=RUNTIME_ABI_VERSION,
            abi_surface=RUNTIME_ABI_SURFACE,
            carry_mode=carry_mode,
            boundary_action=boundary_action,
            shared_runtime_src_sha256=runtime_sha,
            candidate_runtime_sha256=document_shas["candidate_runtime.py"],
            window_mem=int(memory_layout["window_mem"]),
            num_heads=int(memory_layout["num_heads"]),
            num_layers=int(memory_layout["num_layers"]),
            embed_size=int(memory_layout["embed_size"]),
            slow_interval=int(memory_layout["slow_interval"]),
            slow_dim=int(memory_layout["slow_dim"])),
        task_params=StudentTaskParamsLiteral(
            task=_require_key(manifest, "task", "candidate_manifest.json"),
            goal=protocol["goal"],
            stage=protocol["stage"],
            mode=_require_key(config, "mode", "train_summary.config"),
            bonus_type=_require_key(config, "bonus_type",
                                    "train_summary.config"),
            condition_on_task=bool(_require_key(
                config, "condition_on_task", "train_summary.config")),
            replay=_require_key(manifest, "replay",
                                "candidate_manifest.json"),
            vtrace=_require_key(manifest, "vtrace",
                                "candidate_manifest.json"),
            hindsight=bool(_require_key(manifest, "hindsight",
                                        "candidate_manifest.json")),
            awr=bool(_require_key(manifest, "awr",
                                  "candidate_manifest.json")),
            egomap=protocol["egomap"],
            nav_aux=protocol["nav_aux"],
            novelty=protocol["novelty"],
            total_env_steps=int(_require_key(
                protocol, "total_env_steps", "train_summary.protocol")),
            xla_flags=protocol["xla_flags"]),
        gpu_policy=StudentGpuPolicy(
            gpu_allowed=gpu_allowed,
            gpu_forbidden=tuple(_require_key(
                contract, "gpu_forbidden", "checkpoint_contract.json"))),
        environment_lock=StudentEnvironmentLock(
            conda_env=env_lock["conda_env"],
            python=env_lock["python"],
            jax=modules["jax"],
            flax=modules["flax"],
            optax=modules["optax"],
            chex=modules["chex"],
            distrax=modules["distrax"],
            numpy=modules["numpy"],
            local_jax_craftax_forbidden=bool(
                env_lock["local_jax_craftax_forbidden"]),
            xla_flags=env_lock["xla_flags"]),
        state_layout=StudentCheckpointStateLayout(
            required_pkl_keys=tuple(sorted(CC3_PKL_REQUIRED_KEYS)),
            packed_pytree_keys=CC3_PACKED_PYTREE_KEYS,
            params_sha_algorithm=(
                "sha256 over the concatenated little-endian raw bytes of "
                "the packed (leaves, treedef) param leaves in tree order "
                "(params_sha_packed, driver-exact)")),
        constructor=dict(constructor),
        memory_layout=dict(memory_layout),
        capsule_document_shas=dict(document_shas),
        crlf_view_documents=tuple(sorted(
            rel for rel, mode in verification_modes.items()
            if mode == VERIFY_CRLF_VIEW)))


# ---------------------------------------------------------------------------
# The single Student-evaluator consumption surface
# ---------------------------------------------------------------------------
class SurfaceCompatibility(CanonicalModel):
    """One surface's compatibility verdict against the baseline."""

    surface: str = Field(min_length=1)
    status: str = Field(min_length=1)
    baseline_candidate_id: str = Field(min_length=1)
    checks_passed: Tuple[str, ...] = Field(default_factory=tuple)
    detail: str = ""


class StudentCompatibilityReport(CanonicalModel):
    """The aggregate verdict: ALL four surfaces through ONE evaluator."""

    baseline_hash: str
    candidate_id: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    board_window: int = Field(ge=0)
    surfaces: Dict[str, SurfaceCompatibility] = Field(default_factory=dict)
    exact_feedback_lag_verified: bool = False
    overall_status: str = Field(min_length=1)
    report_hash: str = ""

    @model_validator(mode="after")
    def _hash(self) -> "StudentCompatibilityReport":
        if not self.report_hash:
            payload = self.model_dump()
            payload.pop("report_hash", None)
            object.__setattr__(self, "report_hash",
                               canonical_sha256(payload))
        return self


class SlowgruStudentEvaluator:
    """The SINGLE consumption surface bound to the real-capsule baseline.

    EnvCoder output, the FeedbackViews (all three modes), the shared Soft
    Copeland ranking and the four-anchor manifest all pass through this one
    evaluator. Every check FAILS CLOSED on missing/wrong schema; nothing is
    coerced, defaulted or guessed. The evaluator is read-only: it never
    loads a checkpoint, never forks Soft Copeland and never fabricates
    anchors (the anchor seam stays with ``anchor_manifest``).
    """

    def __init__(self, baseline: StudentAbiBaseline) -> None:
        self.baseline = baseline

    @classmethod
    def bind(cls, repo_root: Optional[Path] = None
             ) -> "SlowgruStudentEvaluator":
        return cls(bind_slowgru_persistent_baseline(repo_root))

    # -- checkpoint locality (honesty) ---------------------------------------
    def assert_checkpoint_consumable_locally(self) -> None:
        """Fail closed unless the checkpoint body is locally present AND
        SHA-verified. In this worktree the pkl is a server-only artifact, so
        this ALWAYS raises — consumption happens through the SHA-verified
        wrapper ``load_candidate`` on the authorized GPU host
        (REAL_CHECKPOINT_LOADED stays False)."""
        if self.baseline.checkpoint.checkpoint_location_class \
                != CHECKPOINT_LOCATION_LOCAL_VERIFIED:
            raise StudentAbiBaselineBlocked(
                "LOCAL_CHECKPOINT_LOAD_REFUSED: the canonical checkpoint "
                f"({self.baseline.checkpoint.checkpoint_path}) is "
                f"{self.baseline.checkpoint.checkpoint_location_class}; "
                "the read-only baseline binds identity only — no local load, "
                "no guessing (REAL_CHECKPOINT_LOADED=false)")

    # -- surface 1: EnvCoder output -------------------------------------------
    def check_env_coder(self, output: EnvCoderOutput) -> SurfaceCompatibility:
        if not isinstance(output, EnvCoderOutput):
            raise StudentCompatibilityBlocked(
                f"ENVCODER_OUTPUT_SCHEMA_MISMATCH: expected EnvCoderOutput, "
                f"got {type(output).__name__}")
        checks = ["envcoder_output_schema"]
        for coded in output.coded:
            if not is_sha256_hex(coded.directive_hash):
                raise StudentCompatibilityBlocked(
                    f"ENVCODER_DIRECTIVE_HASH_NOT_SHA256: "
                    f"{coded.directive_id!r} carries {coded.directive_hash!r}")
            if coded.environment_family not in C.ENVIRONMENT_FAMILIES:
                raise StudentCompatibilityBlocked(
                    f"ENVCODER_UNKNOWN_FAMILY: {coded.environment_family!r}")
            # the EnvCoder charter: it realizes environment axes and never
            # touches the action/observation ABI — an override attempt fails
            # closed against the real baseline
            scanned = " ".join((coded.reset_contract, coded.step_contract,
                                coded.code_symbol))
            if "action_dim" in scanned or "obs_dim" in scanned:
                raise StudentCompatibilityBlocked(
                    f"ACTION_ABI_OVERRIDE_FORBIDDEN: EnvCoder output for "
                    f"{coded.directive_id!r} attempts to touch the "
                    f"observation/action ABI "
                    f"(baseline action_dim="
                    f"{self.baseline.action_abi.action_dim}, obs_dim="
                    f"{self.baseline.action_abi.obs_dim} are literal)")
            if coded.reset_contract != \
                    f"reset(seed)->state::{coded.environment_family}":
                raise StudentCompatibilityBlocked(
                    f"ENVCODER_RESET_CONTRACT_INCOMPATIBLE: "
                    f"{coded.reset_contract!r}")
            if coded.step_contract != (
                    "step(action)->(state,reward,terminal,info)::"
                    + coded.environment_family):
                raise StudentCompatibilityBlocked(
                    f"ENVCODER_STEP_CONTRACT_INCOMPATIBLE: "
                    f"{coded.step_contract!r}")
        checks.append("directive_hashes_sha256")
        checks.append("action_abi_untouched")
        checks.append("reset_step_contracts_family_bound")
        return SurfaceCompatibility(
            surface=SURFACE_ENV_CODER, status=STATUS_COMPATIBLE,
            baseline_candidate_id=self.baseline.candidate_id,
            checks_passed=tuple(checks),
            detail=f"window={output.window} coded={len(output.coded)}")

    # -- surface 2: FeedbackView (per mode) --------------------------------------
    def _check_student_identity_stamp(self, record) -> None:
        """Fail-closed Student-identity stamp check. Exactly three states are
        legal; anything else is a mismatch:

        1. EMPTY stamp — the record asserts no parameter tree at all.
        2. The well-known local SYMBOLIC-binding hash
           (``local_symbolic_binding().parameter_tree_hash``), which is the
           honest ENGINEERING_SCAFFOLD / NOT_LOADED_LOCAL state — legal ONLY
           while ``REAL_CHECKPOINT_LOADED`` is False and ONLY with the
           symbolic binding's step (0). It is never confused with a real
           checkpoint stamp.
        3. A REAL stamp — must match the baseline ``params_sha256`` exactly,
           and any carried checkpoint step must equal the baseline
           ``global_step`` exactly.
        """
        stamp = record.student_parameter_tree_hash
        if stamp == "":
            return
        symbolic = local_symbolic_binding()
        if stamp == symbolic.parameter_tree_hash:
            if C.REAL_CHECKPOINT_LOADED:
                raise StudentCompatibilityBlocked(
                    f"STUDENT_SYMBOLIC_STAMP_AFTER_REAL_LOAD: record "
                    f"{record.feedback_id!r} carries the local symbolic "
                    "binding stamp but REAL_CHECKPOINT_LOADED is true — a "
                    "real checkpoint stamp is now required")
            if record.student_checkpoint_step \
                    != symbolic.checkpoint_global_step:
                raise StudentCompatibilityBlocked(
                    f"STUDENT_CHECKPOINT_STEP_MISMATCH: record "
                    f"{record.feedback_id!r} carries the symbolic binding "
                    f"stamp but step {record.student_checkpoint_step}; the "
                    f"symbolic binding is step "
                    f"{symbolic.checkpoint_global_step}")
            return
        if stamp != self.baseline.checkpoint.params_sha256:
            raise StudentCompatibilityBlocked(
                f"STUDENT_PARAMETER_TREE_MISMATCH: record "
                f"{record.feedback_id!r} stamps parameter tree "
                f"{stamp!r} but the baseline "
                f"params_sha256 is "
                f"{self.baseline.checkpoint.params_sha256!r}")
        if record.student_checkpoint_step and record.student_checkpoint_step \
                != self.baseline.checkpoint.global_step:
            raise StudentCompatibilityBlocked(
                f"STUDENT_CHECKPOINT_STEP_MISMATCH: record "
                f"{record.feedback_id!r} stamps step "
                f"{record.student_checkpoint_step} but the baseline "
                f"global_step is {self.baseline.checkpoint.global_step}")

    def check_feedback_view(self, view, *, mode: str, board_window: int
                            ) -> SurfaceCompatibility:
        if mode not in C.FEEDBACK_MODES:
            raise StudentCompatibilityBlocked(
                f"UNKNOWN_MODE: {mode!r}")
        if board_window < 0:
            raise StudentCompatibilityBlocked(
                f"ILLEGAL_BOARD_WINDOW: {board_window}")
        if mode == C.MODE_STATIC_LLM:
            if not isinstance(view, NullFeedbackView):
                raise StudentCompatibilityBlocked(
                    "STATIC_VIEW_MUST_BE_STRUCTURALLY_NULL: the static mode "
                    f"received a {type(view).__name__}; only the store-less "
                    "NullFeedbackView is consumable")
            if view.records() or view.to_prompt_payload() or \
                    view.behavior_evidence():
                raise StudentCompatibilityBlocked(
                    "STATIC_VIEW_PAYLOAD_NOT_EMPTY: the null view must "
                    "present zero records/payload/evidence")
            return SurfaceCompatibility(
                surface=SURFACE_FEEDBACK_VIEW, status=STATUS_COMPATIBLE,
                baseline_candidate_id=self.baseline.candidate_id,
                checks_passed=("static_structurally_null",
                               "zero_feedback_payload"),
                detail="mode=static_llm")

        # exact k-1 lag, re-asserted at the evaluator surface
        if view.window_scope != board_window - 1:
            raise StudentCompatibilityBlocked(
                f"EXACT_FEEDBACK_LAG_VIOLATED: board window {board_window} "
                f"consumes a view scoped to window {view.window_scope}; the "
                "lag must be EXACTLY one window (k-1)")
        if mode == C.MODE_NORMAL_FEEDBACK:
            if not isinstance(view, NormalFeedbackView):
                raise StudentCompatibilityBlocked(
                    f"NORMAL_VIEW_TYPE_MISMATCH: expected NormalFeedbackView,"
                    f" got {type(view).__name__}")
            for record in view.records():
                self._check_student_identity_stamp(record)
            return SurfaceCompatibility(
                surface=SURFACE_FEEDBACK_VIEW, status=STATUS_COMPATIBLE,
                baseline_candidate_id=self.baseline.candidate_id,
                checks_passed=("exact_window_scope_k_minus_1",
                               "student_identity_stamps_consistent",
                               "records_window_equals_scope"),
                detail=f"mode=normal_feedback window_scope="
                       f"{view.window_scope} records={len(view.records())}")

        if mode == C.MODE_SHUFFLED_FEEDBACK:
            if not isinstance(view, PermutedFeedbackView):
                raise StudentCompatibilityBlocked(
                    "SHUFFLED_VIEW_MUST_BE_PERMUTED: the shuffled mode "
                    f"received a {type(view).__name__}")
            aggregates = family_level_metrics(view.records())
            for payload in view.to_prompt_payload():
                if payload.get("candidate_id") != MASKED_IDENTITY:
                    raise StudentCompatibilityBlocked(
                        "SHUFFLED_IDENTITY_SIDE_CHANNEL: payload candidate "
                        f"id {payload.get('candidate_id')!r} is not masked")
                family = payload.get("environment_family")
                fam_agg = aggregates.get(family)
                if fam_agg is None:
                    raise StudentCompatibilityBlocked(
                        "SHUFFLED_FAMILY_AGGREGATE_MISSING: payload family "
                        f"{family!r} has no public aggregate")
                if payload.get("student_success_rate") \
                        != fam_agg["student_success_rate"] or \
                        payload.get("reference_success_rate") \
                        != fam_agg["reference_success_rate"]:
                    raise StudentCompatibilityBlocked(
                        "NUMERIC_SIDE_CHANNEL_IN_VIEW: payload publishes "
                        "per-record rates instead of the family-level window "
                        "aggregates (re-identification channel)")
                if payload.get("mutation_axes") or \
                        payload.get("axis_values") or \
                        payload.get("held_constant_axes") or \
                        payload.get("expected_signature"):
                    raise StudentCompatibilityBlocked(
                        "SHUFFLED_IDENTITY_SIDE_CHANNEL: payload carries "
                        "unmasked axis/signature fields")
            for record in view.records():
                self._check_student_identity_stamp(record)
            return SurfaceCompatibility(
                surface=SURFACE_FEEDBACK_VIEW, status=STATUS_COMPATIBLE,
                baseline_candidate_id=self.baseline.candidate_id,
                checks_passed=("exact_window_scope_k_minus_1",
                               "candidate_ids_masked",
                               "family_level_aggregates_only",
                               "axes_and_signatures_masked"),
                detail=f"mode=shuffled_feedback window_scope="
                       f"{view.window_scope} records={len(view.records())}")
        raise StudentCompatibilityBlocked(f"UNKNOWN_MODE: {mode!r}")

    # -- surface 3: Soft Copeland (single owner) ---------------------------------
    def check_soft_copeland(self, bundles: Sequence[EnvironmentScoreBundle],
                            ranking: CopelandRanking) -> SurfaceCompatibility:
        if not bundles:
            raise StudentCompatibilityBlocked("EMPTY_COPELAND_BUNDLES")
        for bundle in bundles:
            if not isinstance(bundle, EnvironmentScoreBundle):
                raise StudentCompatibilityBlocked(
                    f"COPELAND_BUNDLE_SCHEMA_MISMATCH: "
                    f"{type(bundle).__name__}")
        if not isinstance(ranking, CopelandRanking):
            raise StudentCompatibilityBlocked(
                f"COPELAND_RANKING_SCHEMA_MISMATCH: "
                f"{type(ranking).__name__}")
        # SINGLE OWNER: the evaluator re-consumes the shared canonical
        # implementation — it never re-ranks with a local scalar. A ranking
        # whose hash does not reproduce from the same bundles is a fork.
        expected = soft_copeland_rank(list(bundles))
        if expected.ranking_hash != ranking.ranking_hash:
            raise StudentCompatibilityBlocked(
                f"SOFT_COPELAND_RANKING_FORKED: presented ranking_hash "
                f"{ranking.ranking_hash!r} does not reproduce from the "
                f"shared soft_copeland_rank (expected "
                f"{expected.ranking_hash!r}); Soft Copeland is the single "
                "ranking owner")
        if sorted(e.environment_id for e in ranking.entries) != \
                sorted(b.environment_id for b in bundles):
            raise StudentCompatibilityBlocked(
                "COPELAND_RANKING_ID_MISMATCH: ranking entries do not cover "
                "exactly the submitted bundles")
        return SurfaceCompatibility(
            surface=SURFACE_SOFT_COPELAND, status=STATUS_COMPATIBLE,
            baseline_candidate_id=self.baseline.candidate_id,
            checks_passed=("shared_soft_copeland_single_owner",
                           "ranking_hash_reproduced",
                           "entries_cover_bundles"),
            detail=f"n={len(bundles)} ranking_hash="
                   f"{ranking.ranking_hash[:16]}")

    # -- surface 4: the four-anchor manifest ----------------------------------------
    def check_anchor_manifest(self, source: AnchorManifestSource
                              ) -> SurfaceCompatibility:
        if not isinstance(source, AnchorManifestSource):
            raise StudentCompatibilityBlocked(
                f"ANCHOR_SOURCE_SCHEMA_MISMATCH: {type(source).__name__}")
        # resolve() is the single seam: it fails closed on absence,
        # unfrozen manifests and hash mismatches (propagated verbatim).
        anchors = source.resolve()
        if len(anchors) != C.GLOBAL_ANCHOR_SLOTS:
            raise StudentCompatibilityBlocked(
                f"ANCHOR_SLOT_COUNT_MISMATCH: {len(anchors)} anchors, the "
                f"baseline binds exactly {C.GLOBAL_ANCHOR_SLOTS}")
        return SurfaceCompatibility(
            surface=SURFACE_ANCHOR_MANIFEST, status=STATUS_COMPATIBLE,
            baseline_candidate_id=self.baseline.candidate_id,
            checks_passed=("shared_frozen_manifest_resolved",
                           "four_anchor_slots",
                           "manifest_hash_recomputed_by_source"),
            detail="anchors=" + ",".join(anchors))

    # -- aggregate -------------------------------------------------------------------
    def evaluate(self, *, mode: str, board_window: int,
                 env_coder_output: EnvCoderOutput, feedback_view,
                 copeland_bundles: Sequence[EnvironmentScoreBundle],
                 copeland_ranking: CopelandRanking,
                 anchor_source: AnchorManifestSource
                 ) -> StudentCompatibilityReport:
        surfaces = {
            SURFACE_ENV_CODER: self.check_env_coder(env_coder_output),
            SURFACE_FEEDBACK_VIEW: self.check_feedback_view(
                feedback_view, mode=mode, board_window=board_window),
            SURFACE_SOFT_COPELAND: self.check_soft_copeland(
                copeland_bundles, copeland_ranking),
            SURFACE_ANCHOR_MANIFEST: self.check_anchor_manifest(
                anchor_source),
        }
        if mode == C.MODE_STATIC_LLM:
            # static has NO feedback to lag behind: the structural null view
            # is the honest k-1 surface (zero payload by construction)
            lag_verified = True
        else:
            lag_verified = ("exact_window_scope_k_minus_1"
                            in surfaces[SURFACE_FEEDBACK_VIEW].checks_passed)
        return StudentCompatibilityReport(
            baseline_hash=self.baseline.baseline_hash,
            candidate_id=self.baseline.candidate_id,
            mode=mode,
            board_window=board_window,
            surfaces=surfaces,
            exact_feedback_lag_verified=lag_verified,
            overall_status=STATUS_COMPATIBLE)


__all__ = [
    "CC3_CANONICAL_CANDIDATE_ID",
    "CAPSULE_POSIX",
    "RUNTIME_NAME",
    "RUNTIME_ABI_VERSION",
    "RUNTIME_ABI_SURFACE",
    "PERSISTENT_BOUNDARY_ACTION",
    "CC3_PKL_REQUIRED_KEYS",
    "CC3_PACKED_PYTREE_KEYS",
    "CHECKPOINT_LOCATION_LOCAL_VERIFIED",
    "CHECKPOINT_LOCATION_SERVER_ONLY",
    "VERIFY_BYTE_IDENTICAL",
    "VERIFY_CRLF_VIEW",
    "SURFACE_ENV_CODER",
    "SURFACE_FEEDBACK_VIEW",
    "SURFACE_SOFT_COPELAND",
    "SURFACE_ANCHOR_MANIFEST",
    "STATUS_COMPATIBLE",
    "STATUS_BLOCKED",
    "StudentAbiBaselineBlocked",
    "StudentCompatibilityBlocked",
    "parse_sha256sums",
    "StudentActionAbi",
    "StudentCheckpointIdentity",
    "StudentRngPolicy",
    "StudentWrapperAbi",
    "StudentTaskParamsLiteral",
    "StudentGpuPolicy",
    "StudentEnvironmentLock",
    "StudentCheckpointStateLayout",
    "StudentAbiBaseline",
    "SurfaceCompatibility",
    "StudentCompatibilityReport",
    "SlowgruStudentEvaluator",
    "bind_slowgru_persistent_baseline",
    "default_repo_root",
]
