"""E1FormalGenManager — the edge teacher object of the formal
Behavior-Aware Regret-Guided LLM-UED direction (plan C10).

Duck-compatible with the legacy ``dicode.dreaming.gen_manager.
GenManager`` surface consumed by ``setup.py`` / ``run_dicode.py`` /
``evolution_efficient.py``:

* ``.session_idx``            mutable int, OWNED by the training loop;
* ``.archive``                provenance-admissible ``ArchiveView``;
* ``.env_generator.check_compilation(code) -> (bool, str)``;
* ``.evolve_tasks(dict_of_tasks, global_agent_profile) -> list[dict]``
  returns the window's compiled worker pool (>= 12 real compiled
  workers, or the honest 12-entry reuse batch while blocked/refused);
  BOTH arguments are IGNORED entirely (provenance rule: evolve-side
  metrics never enter the teacher, its prompts, or its ledger);
* ``.select_context_tasks(...)`` -> ``[]`` (honest: no admissible
  context tasks exist without real probes);
* ``.observe_session_feedback(session_idx, metrics)`` re-verifies
  provenance BEFORE storing anything;
* ``.build_training_batch(...)`` / ``.build_training_layout(...)``;
* ``.consume_worker_results(worker_results) -> ([], compiled_count)``
  (C11: E1 consumes its own worker dicts itself; promotion happens
  only through E1 selection, never the legacy compare-and-swap);
* ``.anchor_task_ids``.

Worker dicts carry the E1 keys (``task_id`` / ``code``) AND the legacy
aliases (``generated_task_id`` / ``code_string``) spelling the same
values, so every legacy consumer reads consistent data (C11).

Degradation chain (plan D5 — every step honest, nothing fabricated)::

    REFERENCE_CONTRACT_UNFROZEN
      => EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER
      => LEARNABILITY_UNAVAILABLE
      => SELECTION_BLOCKED_NO_REAL_EVIDENCE
      => batch trains NOTHING (zero PPO updates, zero env-step
         progress). An anchors-only batch is a sneak path and is never
         emitted as trainable (C13). REUSE is legitimate ONLY as the
         FULL 12 dynamic + 4 frozen shared anchors of the last fully
         verified window, bound to source/window/hash evidence
         (``record_verified_batch``); without such a snapshot the batch
         is BLOCKED and the training gate refuses run_session_training.

C14 evidence binding: a verified snapshot carries STRUCTURED dual-probe
evidence — Student/Reference probe ids and sha256 hashes, the pinned
strong-Student candidate id, the Reference identity hash, the window
hash and the candidate-set hash — and every entry's artifact_id must
equal the teacher's internal registry. A provenance string alone (even
the exact CANDIDATE_EVALUATION value) NEVER certifies REUSE.

C15 attestation binding (REQUEST_CHANGES fix): caller-supplied probe
strings alone NEVER suffice either — probe evidence is accepted ONLY
as an immutable ``eval_adapter.DualProbeResult`` ISSUED by this
teacher's internal candidate-evaluation adapter registry
(``consume_candidate_eval_result``). A caller-shaped mapping (even
with all-valid fields), a directly constructed result, or a result
citing an unregistered/fake adapter is refused. Every issued result
binds Student/Reference probe ids + sha256 hashes, Student/Reference
CHECKPOINT hashes, the review window id/hash, the ordered
candidate-set hash and the episode reset protocol to the pinned
strong Student and the CURRENT frozen Reference candidate, and every
stored binding (registry artifact/spec/code hashes, window id/hash,
candidate-set hash, Reference identity hash, manifest sha, probes) is
RE-VALIDATED against the current internal state on EVERY reuse
(``_snapshot_still_valid``): a stale window, a re-frozen Reference or
protocol, a changed manifest, a reordered candidate set or a tampered
stored snapshot invalidates REUSE fail-closed.

This module imports NO jax/craftax, performs NO network I/O and NO
file I/O (the anchor manifest and frozen manifest are injected as
mappings by the caller). The LEGACY duck ``check_compilation`` stays a
stdlib syntax-only compile plus deterministic output guards. The
round-3 evolve path validates EnvCoder artifacts through the staged
``envcoder_backends`` surface (ReplayBackend: SYNTAX + GUARDS +
STRUCTURE, a stdlib-AST entry-surface check) with a BOUNDED repair
loop (``envcoder.run_envcoder_with_repair``; F1 counts the repair
calls) — craftax is still NOT installed in the audit venv, so
import/reset/step semantics are NOT validated; ``status_report`` says
so explicitly.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..static_llm.guards import raise_if_forbidden, scan_text
from . import anchor_manifest as AM
from . import envcoder_backends as EB
from . import eval_adapter as EA
from . import layout
from . import metrics as MT
from . import selector
from . import shared_runtime_seam as SRS
from .accounting import LLMCallLedger
from .archive_view import ArchiveView, consume_archive_snapshot, empty_archive_view
from .board import WINDOW_STATUS_COMPLETE
from .canonical import canonical_sha256
from .controller import run_review_cycle
from .envcoder import (
    ENVCODER_REPAIR_PROMPT_VERSION,
    MAX_ENVCODER_REPAIRS,
    EnvCoderError,
    run_envcoder_with_repair,
)
from .evidence import build_evidence_snapshot
from .flags import parse_flags
from .gate_signals import (
    INVOCATION_THRESHOLD_MISSING,
    GateSignalError,
    compute_gate_signals,
    consume_gate_thresholds,
)
from .invocation_gate import build_gate_state
from .llm_client import ReplayLLMClient
from .reference_contract import (
    ReferenceContractError,
    ReferenceIdentityContract,
    consume_reference_identity_contract,
    reference_identity_sha256,
)
from .schemas import E1Code, E1SchemaError, assert_llm_role_admissible
from .student_contract import PINNED_STUDENT_CANDIDATE_ID
from .task_specs import compile_task_specs
from .training_gate import TRAINING_BLOCKED_NO_VERIFIED_BATCH

#: replay provider identity pinned by the frozen manifest (plan D11)
REPLAY_MODEL_ID = "e1-replay-mock-v1"
REPLAY_PROVIDER = "replay"

#: teacher identity
TEACHER_TYPE = "e1_formal"

# fail-closed codes
GEN_MANAGER_BAD_TYPE = "GEN_MANAGER_BAD_TYPE"
GEN_MANAGER_MISSING_FIELD = "GEN_MANAGER_MISSING_FIELD"
GEN_MANAGER_UNKNOWN_FIELD = "GEN_MANAGER_UNKNOWN_FIELD"
GEN_MANAGER_OUT_OF_RANGE = "GEN_MANAGER_OUT_OF_RANGE"
GEN_MANAGER_BAD_TEACHER_TYPE = "GEN_MANAGER_BAD_TEACHER_TYPE"
GEN_MANAGER_MANIFEST_MISMATCH = "GEN_MANAGER_MANIFEST_MISMATCH"
GEN_MANAGER_BAD_DYNAMIC_SET = "GEN_MANAGER_BAD_DYNAMIC_SET"
GEN_MANAGER_FEEDBACK_BAD_FACTS = "GEN_MANAGER_FEEDBACK_BAD_FACTS"
GEN_MANAGER_NO_ADMISSIBLE_EVIDENCE = "GEN_MANAGER_NO_ADMISSIBLE_EVIDENCE"
#: C13: promotion attempted while hard gates still block (contract
#: violation — real selection is impossible while blocked)
GEN_MANAGER_PROMOTION_BLOCKED = "GEN_MANAGER_PROMOTION_BLOCKED"
#: C13: verified-batch snapshot validation failures (REUSE evidence)
GEN_MANAGER_SNAPSHOT_BAD_TYPE = "GEN_MANAGER_SNAPSHOT_BAD_TYPE"
GEN_MANAGER_SNAPSHOT_MISSING_FIELD = "GEN_MANAGER_SNAPSHOT_MISSING_FIELD"
GEN_MANAGER_SNAPSHOT_MISMATCH = "GEN_MANAGER_SNAPSHOT_MISMATCH"
GEN_MANAGER_SNAPSHOT_BLOCKED = "GEN_MANAGER_SNAPSHOT_BLOCKED"

#: provenance a verified REUSE snapshot must carry: only the candidate
#: evaluation path (real Student/Reference dual probes) may certify a
#: window as REUSE-admissible. Necessary but NEVER sufficient (C14):
#: certification additionally requires every structured evidence field
#: below, so a provenance string alone can never certify REUSE.
_VERIFIED_SNAPSHOT_PROVENANCE = "CANDIDATE_EVALUATION"
_VERIFIED_SNAPSHOT_FIELDS = (
    "window_id",
    # C14: canonical hash of the review window; must equal the
    # window_hash recorded in the artifact registry for every one of
    # the 12 dynamic tasks
    "window_hash",
    "provenance",
    "reference_candidate_id",
    # C14: canonical sha256 over the frozen Reference identity (all
    # contracted fields), computed from the CURRENT frozen contract
    "reference_identity_hash",
    "anchor_task_ids",
    "anchor_manifest_sha256",
    # C14: canonical sha256 over the ordered candidate set (the 12
    # dynamic task ids exactly as certified)
    "candidate_set_hash",
    # C14: structured Student/Reference dual-probe evidence block
    "dual_probe",
    "dynamic_tasks",
)

#: C14: structured dual-probe evidence fields (all required; a
#: provenance string alone never certifies REUSE). The probes must
#: have run on the pinned strong Student and the frozen Reference.
_DUAL_PROBE_FIELDS = (
    "student_candidate_id",
    "student_probe_id",
    "student_probe_hash",
    "reference_probe_id",
    "reference_probe_hash",
)

_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256_hex(value: Any) -> bool:
    """True iff ``value`` is a 64-char lowercase sha256 hex string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX_DIGITS for c in value)
    )


def _validate_dual_probe(raw: Any, ctx: str) -> Dict[str, str]:
    """Fail-closed validation of the structured dual-probe block (C14).

    Required evidence: the pinned strong-Student candidate id, a
    Student probe id + sha256 hash and a Reference probe id + sha256
    hash. Raises GenManagerError with a greppable code on ANY
    violation; returns the cleaned block.
    """
    if not isinstance(raw, Mapping):
        raise GenManagerError(
            GEN_MANAGER_SNAPSHOT_BAD_TYPE,
            f"{ctx}: dual_probe must be a mapping, got "
            f"{type(raw).__name__}",
        )
    unknown = sorted(k for k in raw if k not in _DUAL_PROBE_FIELDS)
    if unknown:
        raise GenManagerError(
            GEN_MANAGER_SNAPSHOT_BAD_TYPE,
            f"{ctx}: unknown dual_probe field(s) {unknown}",
        )
    for name in _DUAL_PROBE_FIELDS:
        if name not in raw:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISSING_FIELD,
                f"{ctx}: dual_probe missing field {name!r}",
            )
    student_candidate_id = raw["student_candidate_id"]
    if student_candidate_id != PINNED_STUDENT_CANDIDATE_ID:
        raise GenManagerError(
            GEN_MANAGER_SNAPSHOT_MISMATCH,
            f"{ctx}: dual probes must run on the pinned strong Student "
            f"{PINNED_STUDENT_CANDIDATE_ID!r}, got "
            f"{student_candidate_id!r}",
        )
    for name in ("student_probe_id", "reference_probe_id"):
        value = raw[name]
        if not isinstance(value, str) or not value.strip():
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISSING_FIELD,
                f"{ctx}: dual_probe needs non-empty {name!r}",
            )
    for name in ("student_probe_hash", "reference_probe_hash"):
        value = raw[name]
        if not _is_sha256_hex(value):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: dual_probe field {name!r} must be lowercase "
                f"sha256 hex (64 chars), got {value!r}",
            )
    return {name: raw[name] for name in _DUAL_PROBE_FIELDS}

#: honest environment-compilation note for the LEGACY duck surface
ENVCODER_CHECK_NOTE = (
    "stdlib-syntax-only compile + deterministic guards; craftax "
    "import/reset/step semantics NOT validated this round (craftax "
    "absent from the audit venv)"
)

#: round-3 P0-4: bounded EnvCoder repair — ``teacher.envcoder.max_repairs``
#: absent => 2 (the supervisor-sanctioned <= 2 hard validations per
#: template); present values are validated within [0, MAX_ENVCODER_REPAIRS]
DEFAULT_MAX_REPAIRS = 2


class GenManagerError(E1SchemaError):
    """Fail-closed teacher violation; ``code`` is greppable."""


def _require_mapping(obj: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise GenManagerError(
            GEN_MANAGER_BAD_TYPE,
            f"{ctx}: expected a mapping, got {type(obj).__name__}",
        )
    return obj


def _require_block(container: Mapping, key: str, ctx: str) -> Mapping[str, Any]:
    if key not in container:
        raise GenManagerError(
            GEN_MANAGER_MISSING_FIELD, f"{ctx}: missing block {key!r}"
        )
    return _require_mapping(container[key], f"{ctx}.{key}")


@dataclass(frozen=True)
class _E1EnvGenerator:
    """Duck for ``dreaming.gen_manager.EnvGenerator.check_compilation``.

    THIS ROUND: stdlib syntax-only ``compile`` plus the deterministic
    output guards. See ``ENVCODER_CHECK_NOTE`` for the honest scope.
    Compilation results are NEVER fed back to any LLM (plan stage 5).
    """

    def check_compilation(self, code: Any) -> Tuple[bool, str]:
        if not isinstance(code, str):
            return (
                False,
                f"{GEN_MANAGER_BAD_TYPE}: env code must be str, got "
                f"{type(code).__name__}",
            )
        if not code.strip():
            return (False, f"{GEN_MANAGER_BAD_TYPE}: env code is empty")
        decision = scan_text(code, "env_code")
        if not decision.allowed:
            return (False, f"{decision.code}: {decision.detail}")
        try:
            compile(code, "<e1-artifact>", "exec")
        except SyntaxError as e:
            return (False, f"SYNTAX_ERROR: {e.msg} (line {e.lineno})")
        except ValueError as e:  # e.g. null bytes
            return (False, f"SYNTAX_ERROR: {e}")
        return (True, "")


class E1FormalGenManager:
    """Behavior-Aware Regret-Guided LLM-UED teacher (formal E1)."""

    def __init__(
        self,
        config: Any,
        *,
        frozen_manifest: Any,
        anchor_manifest_mapping: Any,
        replay_store: Optional[Mapping[str, str]] = None,
        llm_client: Any = None,
        archive_snapshot: Any = None,
    ) -> None:
        ctx = "e1_formal.gen_manager"
        config = _require_mapping(config, ctx)
        teacher = _require_block(config, "teacher", ctx)
        teacher_type = teacher.get("teacher_type")
        if teacher_type != TEACHER_TYPE:
            raise GenManagerError(
                GEN_MANAGER_BAD_TEACHER_TYPE,
                f"{ctx}: teacher_type must be {TEACHER_TYPE!r}, got "
                f"{teacher_type!r}",
            )

        # ---- flags vs frozen manifest (D7) -----------------------------
        self._flags = parse_flags(
            _require_block(teacher, "flags", ctx), f"{ctx}.flags"
        )
        from .flags import assert_flags_match_manifest

        assert_flags_match_manifest(
            self._flags, _require_mapping(frozen_manifest, f"{ctx}.manifest"), ctx
        )
        self._frozen_manifest = _require_mapping(frozen_manifest, f"{ctx}.manifest")
        self._verify_frozen_manifest(ctx)

        # ---- G1: Reference identity contract (may stay unfrozen) -------
        self._reference_contract: Optional[ReferenceIdentityContract] = None
        self._init_blocked_codes: List[str] = []
        rc_block = _require_block(teacher, "reference_contract", ctx)
        try:
            self._reference_contract = consume_reference_identity_contract(
                rc_block, f"{ctx}.reference_contract"
            )
        except ReferenceContractError as e:
            if e.code == "REFERENCE_CONTRACT_UNFROZEN":
                # honest degradation, NOT an error: the evaluation seam
                # stays blocked until the supervisor freezes the identity
                self._init_blocked_codes.append(e.code)
            else:
                raise

        # ---- G2: learnability thresholds (no defaults) ------------------
        self._thresholds: Optional[MT.LearnabilityThresholds] = None
        try:
            self._thresholds = MT.consume_learnability_thresholds(
                _require_block(teacher, "learnability", ctx),
                f"{ctx}.learnability",
            )
        except MT.MetricsError as e:
            if e.code == MT.LEARNABILITY_THRESHOLD_MISSING:
                self._init_blocked_codes.append(e.code)
            else:
                raise

        # ---- P0-3: invocation thresholds (no defaults; may stay unfrozen)
        self._invocation_thresholds = None
        self._invocation_threshold_version = None
        self._invocation_degradation = ""
        self.invocation_thresholds_present = False
        invocation = _require_block(teacher, "invocation", ctx)
        version = invocation.get("threshold_version")
        if not isinstance(version, str) or not version.strip():
            raise GenManagerError(
                GEN_MANAGER_MISSING_FIELD,
                f"{ctx}.invocation: missing non-empty threshold_version",
            )
        self._invocation_threshold_version = version.strip()
        thresholds_block = invocation.get("thresholds")
        if thresholds_block is None:
            # honest degradation, NOT an error and NOT a training-gate
            # blocker: the gate signals are genuinely computed and the
            # threshold-driven ones stay False with
            # INVOCATION_THRESHOLD_MISSING reasons until the supervisor
            # freezes the values. The invocation regime governs when a
            # review WINDOW opens; it never invalidates the verified
            # dual-probe/anchor-manifest evidence chain that the C13
            # training gate audits, so it stays out of
            # current_blocked_codes() by design.
            self._invocation_degradation = INVOCATION_THRESHOLD_MISSING
        else:
            try:
                self._invocation_thresholds = consume_gate_thresholds(
                    thresholds_block, f"{ctx}.invocation"
                )
            except GateSignalError as e:
                if e.code == INVOCATION_THRESHOLD_MISSING:
                    self._invocation_degradation = e.code
                else:
                    raise
        if self._invocation_thresholds is not None:
            self.invocation_thresholds_present = True
        # pin: the teacher's threshold version must equal the frozen
        # manifest's — the gate-signal regime is not self-updatable
        manifest_invocation = _require_block(
            self._frozen_manifest, "invocation", f"{ctx}.manifest"
        )
        if (
            manifest_invocation.get("threshold_version")
            != self._invocation_threshold_version
        ):
            raise GenManagerError(
                GEN_MANAGER_MANIFEST_MISMATCH,
                f"{ctx}.manifest.invocation: threshold_version "
                f"{manifest_invocation.get('threshold_version')!r} != "
                f"teacher {self._invocation_threshold_version!r}",
            )

        # ---- selection knobs (mechanical; pinned, not defaulted) --------
        selection = _require_block(teacher, "selection", ctx)
        critic_policy = selection.get("critic_policy")
        if critic_policy not in (
            selector.CRITIC_HARD_VETO,
            selector.CRITIC_SOFT_PENALTY,
            selector.CRITIC_SCORE_ONLY,
        ):
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.selection: critic_policy {critic_policy!r} not in "
                "the canonical policy set",
            )
        k = selection.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or k != layout.NUM_DYNAMIC_SLOTS:
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.selection: k must be exactly "
                f"{layout.NUM_DYNAMIC_SLOTS} (the dynamic slot count), got "
                f"{k!r}",
            )
        seed = selection.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise GenManagerError(
                GEN_MANAGER_BAD_TYPE,
                f"{ctx}.selection: seed must be an int, got {seed!r}",
            )
        # round-3 P0-5: per-family cap for the criterion-wise selector.
        # Optional at teacher init (the formal selector itself REQUIRES
        # it); when present it must be an int within [1, k].
        family_cap = selection.get("family_cap")
        if family_cap is not None and (
            isinstance(family_cap, bool)
            or not isinstance(family_cap, int)
            or family_cap < 1
            or family_cap > k
        ):
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.selection: family_cap must be an int within "
                f"[1, k={k}], got {family_cap!r}",
            )
        self._family_cap = family_cap
        self._critic_policy = critic_policy
        self._selection_k = k
        self._selection_seed = seed

        # ---- anchors (must equal the canonical layout anchors) ----------
        anchors_block = _require_block(teacher, "anchors", ctx)
        anchor_ids = anchors_block.get("task_ids")
        if not isinstance(anchor_ids, (list, tuple)) or tuple(
            anchor_ids
        ) != layout.ANCHOR_TASK_IDS:
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.anchors: task_ids must equal "
                f"{list(layout.ANCHOR_TASK_IDS)}, got {anchor_ids!r}",
            )

        # ---- envcoder seed examples (whitelist prompt input) ------------
        envcoder_block = _require_block(teacher, "envcoder", ctx)
        self._seed_examples = _consume_seed_examples(
            envcoder_block.get("seed_examples"), f"{ctx}.envcoder"
        )

        # ---- round-3 P0-4: bounded repair scope + validation backend ----
        # max_repairs absent => 2 (<= 2 hard validations per template);
        # present values must be ints within [0, MAX_ENVCODER_REPAIRS]
        # (bools rejected). The F1 ledger counter is bounded by this.
        max_repairs = envcoder_block.get("max_repairs")
        if max_repairs is None:
            max_repairs = DEFAULT_MAX_REPAIRS
        if (
            isinstance(max_repairs, bool)
            or not isinstance(max_repairs, int)
            or max_repairs < 0
            or max_repairs > MAX_ENVCODER_REPAIRS
        ):
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.envcoder: max_repairs must be an int within "
                f"[0, {MAX_ENVCODER_REPAIRS}], got {max_repairs!r}",
            )
        self._max_repairs = max_repairs
        # backend: replay (SYNTAX+GUARDS+STRUCTURE) is the honest
        # production default; mock is an explicitly-authorized
        # ablation opt-in; real stays unauthorized this round
        # (fail-closed, never a silent downgrade)
        backend_name = envcoder_block.get("backend")
        if backend_name is None:
            backend_name = EB.BACKEND_REPLAY
        if not isinstance(backend_name, str) or backend_name not in (
            EB.BACKEND_MOCK,
            EB.BACKEND_REPLAY,
            EB.BACKEND_REAL,
        ):
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.envcoder: backend must be one of "
                f"{[EB.BACKEND_MOCK, EB.BACKEND_REPLAY, EB.BACKEND_REAL]}, "
                f"got {backend_name!r}",
            )
        if backend_name == EB.BACKEND_REAL:
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.envcoder: backend {EB.BACKEND_REAL!r} is "
                "unauthorized this round (craftax runtime absent); the "
                "real backend never degrades silently",
            )
        self._envcoder_backend_name = backend_name
        self._envcoder_backend = (
            EB.MockBackend()
            if backend_name == EB.BACKEND_MOCK
            else EB.ReplayBackend()
        )

        # ---- replay identity --------------------------------------------
        replay = _require_block(teacher, "replay", ctx)
        if replay.get("provider") != REPLAY_PROVIDER:
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.replay: provider must be {REPLAY_PROVIDER!r} this "
                f"round, got {replay.get('provider')!r}",
            )
        if replay.get("model_id") != REPLAY_MODEL_ID:
            raise GenManagerError(
                GEN_MANAGER_MANIFEST_MISMATCH,
                f"{ctx}.replay: model_id must be {REPLAY_MODEL_ID!r}, got "
                f"{replay.get('model_id')!r}",
            )
        if replay.get("record") != "disabled":
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}.replay: record must be 'disabled' this round, got "
                f"{replay.get('record')!r}",
            )

        # ---- G3: shared anchor manifest (DRAFT accepted, BLOCKED used) --
        self._anchor_manifest = AM.consume_anchor_manifest(
            anchor_manifest_mapping, f"{ctx}.anchor_manifest"
        )

        # ---- collaborators ------------------------------------------------
        self._ledger = LLMCallLedger()
        if llm_client is not None:
            self._llm = llm_client
        else:
            self._llm = ReplayLLMClient(
                replay_store or {}, f"{ctx}.replay-client"
            )
        if archive_snapshot is None:
            self._archive_view = empty_archive_view()
        else:
            self._archive_view = consume_archive_snapshot(
                archive_snapshot, f"{ctx}.archive"
            )

        # ---- mutable loop-owned state -------------------------------------
        self.session_idx = 1
        self._env_generator = _E1EnvGenerator()
        self._cycles_run = 0
        self._pending_feedback: List[Dict[str, Any]] = []
        self._real_selection_completed = False
        # P0-3: previous review window (prev_window_hash source for the
        # gate signals) and the count of consecutive review cycles that
        # produced no usable window since the last COMPLETE one
        self._last_window = None
        self._consecutive_reuses = 0
        # compiled E1 artifacts recorded by consume_worker_results (C11);
        # promotion happens ONLY via E1 selection, never legacy activation
        self._artifact_registry: Dict[str, Dict[str, Any]] = {}
        # C13: the last FULLY VERIFIED window batch (12 dynamic + 4
        # frozen shared anchors with source/window/hash evidence) — the
        # ONLY legitimate REUSE source. None until real dual-probe
        # selection is certified via record_verified_batch; while None,
        # every batch trains nothing.
        self._verified_batch_snapshot: Optional[Dict[str, Any]] = None
        # C15 (REQUEST_CHANGES fix): the INTERNAL candidate-evaluation
        # adapter registry — the ONLY mint path for dual-probe
        # evidence. Adapters register fail-closed; results are issued
        # ONLY inside the registry as immutable DualProbeResult
        # objects. No caller-shaped mapping is ever accepted.
        self._eval_adapter_registry = EA.CandidateEvalAdapterRegistry()
        # C15: consumed registry-issued dual-probe results (stored as
        # plain dicts of the immutable result's fields, with the full
        # evidence chain: probes, checkpoints, window id/hash,
        # candidate-set hash, reset protocol). A snapshot's probe block
        # certifies REUSE only if it matches one of these records in
        # scope (window + candidate set + CURRENT Reference).
        self._probe_attestations: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # init-time manifest verification (fail-closed, greppable)
    # ------------------------------------------------------------------
    def _verify_frozen_manifest(self, ctx: str) -> None:
        manifest = self._frozen_manifest

        copeland = _require_block(manifest, "copeland", f"{ctx}.manifest")
        pins = {
            "protocol_version": selector.COPELAND_PROTOCOL_VERSION,
            "source_sha256": selector.COPELAND_SOURCE_SHA256,
            "constants_sha256": selector.COPELAND_CONSTANTS_SHA256,
            "base_sha256": selector.COPELAND_BASE_SHA256,
        }
        for name, pinned in pins.items():
            if copeland.get(name) != pinned:
                raise GenManagerError(
                    GEN_MANAGER_MANIFEST_MISMATCH,
                    f"{ctx}.manifest.copeland: {name} "
                    f"{copeland.get(name)!r} != pinned {pinned!r} "
                    "(G4 pins are not self-updatable)",
                )

        replay = _require_block(manifest, "replay", f"{ctx}.manifest")
        if replay.get("model_id") != REPLAY_MODEL_ID:
            raise GenManagerError(
                GEN_MANAGER_MANIFEST_MISMATCH,
                f"{ctx}.manifest.replay: model_id must be "
                f"{REPLAY_MODEL_ID!r}, got {replay.get('model_id')!r}",
            )
        if replay.get("provider") != REPLAY_PROVIDER:
            raise GenManagerError(
                GEN_MANAGER_MANIFEST_MISMATCH,
                f"{ctx}.manifest.replay: provider must be "
                f"{REPLAY_PROVIDER!r}, got {replay.get('provider')!r}",
            )

        anchors = _require_block(manifest, "anchors", f"{ctx}.manifest")
        if tuple(anchors.get("task_ids") or ()) != layout.ANCHOR_TASK_IDS:
            raise GenManagerError(
                GEN_MANAGER_MANIFEST_MISMATCH,
                f"{ctx}.manifest.anchors: task_ids must equal "
                f"{list(layout.ANCHOR_TASK_IDS)}, got "
                f"{anchors.get('task_ids')!r}",
            )

        student = _require_block(manifest, "strong_student", f"{ctx}.manifest")
        if student.get("candidate_id") != PINNED_STUDENT_CANDIDATE_ID:
            raise GenManagerError(
                GEN_MANAGER_MANIFEST_MISMATCH,
                f"{ctx}.manifest.strong_student: candidate_id must be "
                f"{PINNED_STUDENT_CANDIDATE_ID!r}, got "
                f"{student.get('candidate_id')!r}",
            )

    # ------------------------------------------------------------------
    # duck surface
    # ------------------------------------------------------------------
    @property
    def archive(self) -> ArchiveView:
        return self._archive_view

    @property
    def env_generator(self) -> _E1EnvGenerator:
        return self._env_generator

    @property
    def anchor_task_ids(self) -> Tuple[str, ...]:
        return layout.ANCHOR_TASK_IDS

    @property
    def flags(self):
        return self._flags

    @property
    def reference_contract(self) -> Optional[ReferenceIdentityContract]:
        return self._reference_contract

    @property
    def anchor_manifest(self) -> AM.SharedAnchorManifest:
        return self._anchor_manifest

    @property
    def thresholds(self) -> Optional[MT.LearnabilityThresholds]:
        return self._thresholds

    @property
    def ledger(self) -> LLMCallLedger:
        return self._ledger

    # ------------------------------------------------------------------
    # stage 1 -> stage 9: the honest review/evolution cycle
    # ------------------------------------------------------------------
    def evolve_tasks(
        self,
        dict_of_tasks: Any = None,
        global_agent_profile: Any = None,
    ) -> List[Dict[str, Any]]:
        """Return the window's compiled worker pool (plan D9; round-3).

        P0-2 semantics: a COMPLETE window yields one EnvCoder call per
        UNIQUE template and one worker dict per (template, variant)
        spec. If >= 12 workers compiled, the FULL pool is returned
        (never truncated to 12 here — selection happens downstream).
        If fewer than 12 compiled, the WHOLE window is refused with
        INSUFFICIENT_DYNAMIC_ARTIFACTS: no stub/placeholder slot
        padding exists anywhere in this path. REUSE/void/no-evidence
        paths still return the honest 12-entry reuse batch.

        BOTH arguments are ignored entirely — the provenance rule says
        evolve-side metrics/status never enter the E1 teacher. Nothing
        here reads them, hashes them, or logs them.
        """
        del dict_of_tasks, global_agent_profile  # ignored by contract
        ctx = f"e1_formal.evolve.s{self.session_idx}"

        raw_items = self._archive_view.evidence_items() + list(
            self._pending_feedback
        )
        if len(raw_items) == 0:
            # NO admissible evidence => the gate must not open a window;
            # zero ledger calls. The session batch trains nothing while
            # blocked (C13): zero updates, no anchors-only sneak.
            return self._reuse_batch(
                [GEN_MANAGER_NO_ADMISSIBLE_EVIDENCE]
                + self.current_blocked_codes(),
                "no admissible evidence; review window not opened; zero "
                "LLM calls",
            )

        evidence = build_evidence_snapshot(raw_items, ctx)
        # feedback facts are consumed exactly once into the evidence
        # snapshot (hash-bound there); they never accumulate unboundedly
        self._pending_feedback.clear()

        # P0-3: the eight gate signals are COMPUTED from the real
        # training-window facts, the previous window, the teacher's
        # session counters and the frozen thresholds — never hardcoded.
        signals = compute_gate_signals(
            session_idx=self.session_idx,
            cycles_run=self._cycles_run,
            evidence=evidence,
            raw_items=raw_items,
            prev_window=self._last_window,
            thresholds=self._invocation_thresholds,
            threshold_version=self._invocation_threshold_version,
            consecutive_reuses=self._consecutive_reuses,
        )
        gate_state = build_gate_state(
            {
                "session_idx": self.session_idx,
                **{name: value for name, value in signals.signals},
                "signals_binding_hash": signals.binding_hash,
            },
            ctx,
        )
        window_id = f"e1-w{self.session_idx:06d}"
        outcome = run_review_cycle(
            self._llm,
            window_id=window_id,
            gate_state=gate_state,
            evidence=evidence,
            ledger=self._ledger,
        )
        self._cycles_run += 1
        if outcome.window is not None:
            self._last_window = outcome.window

        if outcome.reuse:
            self._consecutive_reuses += 1
            reason = (
                outcome.void_code
                if outcome.void_code
                else f"gate decision {outcome.decision.code} (REUSE)"
            )
            return self._reuse_batch(
                self.current_blocked_codes(),
                f"review cycle produced no usable window: {reason}",
            )

        # COMPLETE window -> canonical templates/specs -> EnvCoder
        assert outcome.window is not None
        assert outcome.window.status == WINDOW_STATUS_COMPLETE
        compile_result = compile_task_specs(outcome.window)

        # P0-2: ONE EnvCoder call per UNIQUE template (K1 counts
        # templates); every variant of a template shares its artifact.
        # Round-3 P0-4: the primary call plus a BOUNDED repair loop run
        # through the staged validation backend (ReplayBackend:
        # SYNTAX + GUARDS + STRUCTURE); F1 counts the repair calls per
        # template. Exhaustion or input violations fail the whole
        # template closed (every variant becomes a failed worker).
        template_artifacts: Dict[str, Any] = {}
        template_outcomes: Dict[str, Tuple[bool, str]] = {}
        template_repairs: Dict[str, Tuple[Any, ...]] = {}
        for template in compile_result.templates:
            representative = next(
                spec
                for spec in compile_result.specs
                if spec.template_hash == template.template_hash
            )
            try:
                artifact, repairs = run_envcoder_with_repair(
                    self._llm,
                    spec=representative,
                    seed_examples=self._seed_examples,
                    backend=self._envcoder_backend,
                    max_repairs=self._max_repairs,
                    ledger=self._ledger,
                    window_id=window_id,
                )
            except EnvCoderError as e:
                # parse/guard/repair-exhaustion failure: the whole
                # template (all its variants) is non-compiled; honest
                # per-template record. The bounded repair records ride
                # on the exhaustion exception (``e.records``) when any
                # repair attempt was made.
                template_outcomes[template.template_hash] = (False, e.code)
                template_repairs[template.template_hash] = getattr(
                    e, "records", ()
                )
                continue
            # the staged backend already validated the artifact
            # (passed => every capability stage ran clean); the success
            # record carries the honest stages_run/stages_blocked scope
            template_artifacts[template.template_hash] = artifact
            template_outcomes[template.template_hash] = (True, "")
            template_repairs[template.template_hash] = repairs

        workers: List[Dict[str, Any]] = []
        for spec in compile_result.specs:
            if spec.template_hash not in template_artifacts:
                _ok, failure_note = template_outcomes[spec.template_hash]
                workers.append(
                    self._failed_worker(
                        window_id,
                        spec,
                        failure_note,
                        repairs_count=len(
                            template_repairs.get(spec.template_hash, ())
                        ),
                    )
                )
                continue
            artifact = template_artifacts[spec.template_hash]
            ok, note = template_outcomes[spec.template_hash]
            workers.append(
                {
                    "task_id": spec.spec_id,
                    # legacy key aliases (C11): every legacy consumer
                    # reads generated_task_id / code_string; both spell
                    # the SAME values as task_id / code
                    "generated_task_id": spec.spec_id,
                    "compiled": ok,
                    "code": artifact.env_code,
                    "code_string": artifact.env_code,
                    "reasoning": "",
                    "e1_status": {
                        "reuse": False,
                        "artifact_id": spec.artifact_id,
                        "spec_hash": spec.spec_hash,
                        "template_hash": spec.template_hash,
                        "template_artifact_id": spec.template_artifact_id,
                        "window_id": window_id,
                        "window_hash": spec.window_hash,
                        "compiled": ok,
                        "compile_note": note,
                        "envcoder_check": ENVCODER_CHECK_NOTE,
                        # round-3 P0-4: the bounded repair record (F1)
                        # and the staged validation scope that actually
                        # ran (honest about the blocked stages)
                        "repairs": [
                            asdict(record)
                            for record in template_repairs[
                                spec.template_hash
                            ]
                        ],
                        "validation": {
                            "backend": self._envcoder_backend.name,
                            "stages_run": list(
                                self._envcoder_backend.capabilities
                            ),
                            "stages_blocked": [
                                list(pair)
                                for pair in (
                                    self._envcoder_backend.stages_blocked
                                )
                            ],
                        },
                    },
                }
            )

        # P0-2: the 12 dynamic slots are filled ONLY by real compiled
        # artifacts; fewer than 12 refuses the whole window.
        compiled_count = sum(1 for w in workers if w["compiled"])
        if compiled_count < layout.NUM_DYNAMIC_SLOTS:
            self._consecutive_reuses += 1
            return self._reuse_batch(
                [E1Code.INSUFFICIENT_DYNAMIC_ARTIFACTS]
                + self.current_blocked_codes(),
                f"window {window_id} produced {compiled_count} compiled "
                f"dynamic artifact(s) < {layout.NUM_DYNAMIC_SLOTS}; the "
                "whole window is refused (no stub/placeholder padding)",
            )
        self._consecutive_reuses = 0
        return workers

    def select_context_tasks(self, config: Any = None, num_tasks: Any = None) -> List[str]:
        """Honest answer this round: NO admissible context tasks.

        Without real probes no archive task may pose as evaluation
        context; the empty list lets the legacy evolution path continue
        unchanged (wired at C11). Both arguments are ignored.
        """
        del config, num_tasks
        return []

    # ------------------------------------------------------------------
    # worker-dict consumption (C11 duck hook for run_dicode)
    # ------------------------------------------------------------------
    @property
    def artifact_registry(self) -> Dict[str, Dict[str, Any]]:
        """Read-only copy of the compiled-artifact registry (audit)."""
        return {
            task_id: dict(record)
            for task_id, record in self._artifact_registry.items()
        }

    def consume_worker_results(
        self, worker_results: Any
    ) -> Tuple[List[str], int]:
        """Duck hook replacing ``run_dicode._process_worker_results``.

        E1 worker dicts are consumed by the teacher itself: every
        compiled artifact is recorded in the E1 artifact registry and
        is NEVER promoted through the legacy compare-and-swap
        activation path — promotion happens ONLY through E1 selection
        (``build_training_batch`` with selector-promoted ids). The
        returned ``new_task_ids`` is therefore ALWAYS empty (legacy
        activation bypassed), and the second element is the honest
        compiled count. Fail-closed on any malformed dict.
        """
        ctx = f"e1_formal.consume_worker_results.s{self.session_idx}"
        if not isinstance(worker_results, (list, tuple)):
            raise GenManagerError(
                GEN_MANAGER_BAD_TYPE,
                f"{ctx}: worker_results must be a list, got "
                f"{type(worker_results).__name__}",
            )
        compiled_count = 0
        for i, res in enumerate(worker_results):
            res_ctx = f"{ctx}[{i}]"
            res = _require_mapping(res, res_ctx)
            task_id = res.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise GenManagerError(
                    GEN_MANAGER_MISSING_FIELD,
                    f"{res_ctx}: worker dict needs a non-empty task_id",
                )
            compiled = res.get("compiled")
            if not isinstance(compiled, bool):
                raise GenManagerError(
                    GEN_MANAGER_BAD_TYPE,
                    f"{res_ctx}: compiled must be bool, got {compiled!r}",
                )
            if not compiled:
                continue
            code = res.get("code")
            if not isinstance(code, str) or not code.strip():
                raise GenManagerError(
                    GEN_MANAGER_MISSING_FIELD,
                    f"{res_ctx}: compiled worker dict needs non-empty code",
                )
            status = res.get("e1_status") or {}
            self._artifact_registry[task_id] = {
                "code": code,
                "window_id": status.get("window_id", ""),
                # C14: REUSE certification binds every dynamic task to
                # the window hash recorded here (empty => the task can
                # never be part of a verified window; fail-closed)
                "window_hash": status.get("window_hash", ""),
                "artifact_id": status.get("artifact_id", ""),
                "spec_hash": status.get("spec_hash", ""),
                "envcoder_check": status.get(
                    "envcoder_check", ENVCODER_CHECK_NOTE
                ),
            }
            compiled_count += 1
        return ([], compiled_count)

    # ------------------------------------------------------------------
    # session feedback (D12: provenance re-verified INSIDE the teacher)
    # ------------------------------------------------------------------
    def observe_session_feedback(self, session_idx: Any, metrics: Any) -> None:
        ctx = f"e1_formal.feedback.s{self.session_idx}"
        if isinstance(session_idx, bool) or not isinstance(session_idx, int):
            raise GenManagerError(
                GEN_MANAGER_BAD_TYPE,
                f"{ctx}: session_idx must be int, got {session_idx!r}",
            )
        if session_idx < 0:
            raise GenManagerError(
                GEN_MANAGER_OUT_OF_RANGE,
                f"{ctx}: session_idx must be >= 0, got {session_idx}",
            )
        metrics = _require_mapping(metrics, ctx)
        if "provenance" not in metrics:
            raise GenManagerError(
                GEN_MANAGER_MISSING_FIELD,
                f"{ctx}: feedback metrics must carry 'provenance'",
            )
        provenance = assert_llm_role_admissible(
            metrics["provenance"], ctx
        )  # re-verified here; FORMAL_* / CANDIDATE_EVALUATION rejected
        facts = {k: v for k, v in metrics.items() if k != "provenance"}
        if len(facts) == 0:
            raise GenManagerError(
                GEN_MANAGER_FEEDBACK_BAD_FACTS,
                f"{ctx}: feedback carries no facts beyond provenance",
            )
        try:
            canonical_sha256(facts)
        except E1SchemaError as e:
            raise GenManagerError(
                GEN_MANAGER_FEEDBACK_BAD_FACTS,
                f"{ctx}: facts not canonical-encodable ({e.code})",
            ) from e
        raise_if_forbidden(facts, ctx)  # guards BEFORE any storage
        self._pending_feedback.append(
            {
                "source": "training_window.session_metrics",
                "session_idx": session_idx,
                "provenance": provenance,
                "facts": facts,
            }
        )

    # ------------------------------------------------------------------
    # batch + layout (G2/G3 degradation lands here; C13 fail-closed:
    # blocked => ZERO trainable tasks, never an anchors-only sneak)
    # ------------------------------------------------------------------
    def build_training_batch(
        self,
        promoted_dynamic_ids: Optional[Sequence[str]] = None,
        dual_probe: Any = None,
    ) -> Dict[str, Any]:
        """Build the session batch; ``training_permitted`` gates training.

        C13 contract (supervisor REQUEST_CHANGES fix):

        * ANY applicable hard gate blocked => ``training_permitted`` is
          False and ``task_ids`` is EMPTY — zero PPO updates, zero
          global/env-step progress. An anchors-only batch is a sneak
          path and is never emitted as trainable.
        * no promoted ids and gates clear => REUSE is legitimate ONLY
          as the FULL 12 dynamic + 4 frozen shared anchors of the last
          fully verified window (``record_verified_batch`` evidence);
          without such a snapshot the batch is BLOCKED
          (``TRAINING_BLOCKED_NO_VERIFIED_BATCH``).
        * 12 promoted ids are accepted ONLY while every gate is clear
          (real selection is impossible otherwise) and every id carries
          compiled-artifact evidence in the registry.

        C14: promotion additionally requires ``dual_probe`` — the
        structured Student/Reference dual-probe evidence block (probe
        ids + sha256 hashes on the pinned strong Student). Ids plus a
        provenance string alone NEVER certify a trainable batch.

        C15: the probe block must match a dual-probe result ISSUED by
        the internal candidate-evaluation adapter registry, bound to
        this exact window, candidate set and the CURRENT frozen
        Reference; caller-supplied probe strings (or mappings) alone
        never certify promotion.
        """
        ids = list(promoted_dynamic_ids or ())
        blocked = self.current_blocked_codes()

        if len(ids) == 0:
            snapshot = self._verified_batch_snapshot
            if not blocked and snapshot is not None and (
                self._snapshot_still_valid(snapshot)
            ):
                dynamic_ids = [
                    entry["task_id"] for entry in snapshot["dynamic_tasks"]
                ]
                layout_map = layout.build_training_layout(dynamic_ids)
                return {
                    "task_ids": dynamic_ids + list(layout.ANCHOR_TASK_IDS),
                    "training_permitted": True,
                    "provenance": "REUSE_VERIFIED_WINDOW",
                    "layout": layout_map,
                    "dynamic_promoted": 0,
                    "reuse_only": True,
                    "reuse_evidence": dict(snapshot),
                    "blocked_codes": [],
                    "notes": [
                        "REUSE: the previous window's fully verified "
                        "12 dynamic + 4 frozen shared anchors, bound to "
                        f"window {snapshot['window_id']} and anchor "
                        f"manifest sha {snapshot['anchor_manifest_sha256']}",
                        "C14 structured evidence: dual probes "
                        "student="
                        f"{snapshot['dual_probe']['student_probe_id']}"
                        " / reference="
                        f"{snapshot['dual_probe']['reference_probe_id']}"
                        "; reference identity hash "
                        f"{snapshot['reference_identity_hash']}; "
                        f"candidate-set hash "
                        f"{snapshot['candidate_set_hash']}",
                    ],
                }
            codes = list(blocked)
            if not codes:
                # gates clear but NO legitimate previous-window batch
                codes = [TRAINING_BLOCKED_NO_VERIFIED_BATCH]
            return {
                "task_ids": [],
                "training_permitted": False,
                "provenance": "BLOCKED",
                "layout": None,
                "dynamic_promoted": 0,
                "reuse_only": True,
                "reuse_evidence": None,
                "blocked_codes": codes,
                "notes": [
                    "hard gate(s) blocked: ZERO training updates this "
                    "session — selection requires real dual probes (G2) "
                    "and a frozen shared anchor manifest (G3)",
                    "an anchors-only batch is a sneak path and is never "
                    "emitted as trainable; REUSE requires the previous "
                    "window's verified 12+4 with source/window/hash "
                    "evidence, which does not exist yet",
                ],
            }

        # --- 12 promoted dynamic ids -----------------------------------
        if len(ids) != layout.NUM_DYNAMIC_SLOTS:
            raise GenManagerError(
                GEN_MANAGER_BAD_DYNAMIC_SET,
                f"e1_formal.batch: promoted dynamic set must have exactly "
                f"{layout.NUM_DYNAMIC_SLOTS} ids, got {len(ids)}",
            )
        if blocked:
            raise GenManagerError(
                GEN_MANAGER_PROMOTION_BLOCKED,
                "e1_formal.batch: promotion is impossible while hard "
                f"gates block ({blocked}); real selection requires real "
                "dual probes (G2) and a frozen anchor manifest (G3). "
                "Refusing to build a trainable batch from unverified ids.",
            )
        missing = [t for t in ids if t not in self._artifact_registry]
        if missing:
            raise GenManagerError(
                GEN_MANAGER_MISSING_FIELD,
                "e1_formal.batch: promoted ids without compiled-artifact "
                f"evidence in the registry: {missing}",
            )
        window_ids = {
            self._artifact_registry[t]["window_id"] for t in ids
        }
        if len(window_ids) != 1:
            raise GenManagerError(
                GEN_MANAGER_BAD_DYNAMIC_SET,
                "e1_formal.batch: a promoted batch must come from ONE "
                f"window, got ids from {sorted(window_ids)}",
            )
        # C14: promotion without structured dual-probe evidence is a
        # provenance-only certification attempt — refused fail-closed
        if dual_probe is None:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISSING_FIELD,
                "e1_formal.batch.promotion: promoting 12 ids requires "
                "the structured Student/Reference dual-probe evidence "
                "(probe ids + sha256 hashes); ids and a provenance "
                "string alone never certify a trainable batch",
            )
        # certified: this window becomes the REUSE source for the next
        # session. C15: _certify_dynamic_window itself validates the
        # probe block AND requires a dual-probe result ISSUED by the
        # internal candidate-evaluation adapter registry (bound to this
        # window + candidate set), so the choke point stays fail-closed
        # even for direct callers.
        self._certify_dynamic_window(
            window_ids.pop(), ids, _VERIFIED_SNAPSHOT_PROVENANCE,
            "e1_formal.batch.promotion", dual_probe=dual_probe,
        )
        layout_map = layout.build_training_layout(ids)
        return {
            "task_ids": list(ids) + list(layout.ANCHOR_TASK_IDS),
            "training_permitted": True,
            "provenance": "PROMOTED_SELECTION",
            "layout": layout_map,
            "dynamic_promoted": len(ids),
            "reuse_only": False,
            "reuse_evidence": dict(self._verified_batch_snapshot),
            "blocked_codes": [],
            "notes": [],
        }

    # ------------------------------------------------------------------
    # C13 verified REUSE snapshot (source/window/hash evidence)
    # ------------------------------------------------------------------
    @property
    def verified_batch_snapshot(self) -> Optional[Dict[str, Any]]:
        """Read-only copy of the last verified window batch (audit)."""
        snapshot = self._verified_batch_snapshot
        if snapshot is None:
            return None
        copy = dict(snapshot)
        copy["dynamic_tasks"] = [dict(e) for e in snapshot["dynamic_tasks"]]
        copy["anchor_task_ids"] = list(snapshot["anchor_task_ids"])
        copy["dual_probe"] = dict(snapshot["dual_probe"])
        return copy

    # ------------------------------------------------------------------
    # C15 (REQUEST_CHANGES fix): dual-probe evidence is minted ONLY by
    # the internal candidate-evaluation adapter registry and consumed
    # ONLY as immutable registry-issued results. Caller-shaped mappings
    # — even with all-valid fields — are NEVER accepted.
    # ------------------------------------------------------------------
    @property
    def probe_attestations(self) -> List[Dict[str, str]]:
        """Read-only copy of the consumed dual-probe results (audit)."""
        return [dict(att) for att in self._probe_attestations]

    @property
    def eval_adapter_registry(self) -> EA.CandidateEvalAdapterRegistry:
        """The internal candidate-evaluation adapter registry.

        The ONLY mint path for dual-probe evidence: adapters register
        fail-closed, results are issued inside the registry as
        immutable ``DualProbeResult`` objects, and the teacher consumes
        only registry-issued instances (``consume_candidate_eval_result``).
        """
        return self._eval_adapter_registry

    def register_candidate_eval_adapter(
        self, spec: Any
    ) -> EA.RegisteredAdapter:
        """Register ONE candidate-evaluation adapter (fail-closed).

        Delegates to the internal registry: exact spec field set,
        non-empty non-placeholder adapter id/version, a sha256-hex
        adapter hash, and the pinned dual-probe capability. A fake or
        malformed adapter never enters the registry, so it can never
        issue probe evidence.
        """
        return self._eval_adapter_registry.register_adapter(spec)

    def consume_candidate_eval_result(
        self, result: Any
    ) -> Dict[str, str]:
        """Consume ONE registry-issued dual-probe result (C15 fix).

        The CC4 evaluation seam (StudentAdapter + frozen Reference) is
        the only producer of real probe evidence; this is the ONLY
        consumer entry point. Fail-closed:

        * ONLY an immutable ``eval_adapter.DualProbeResult`` is
          accepted — a caller-shaped mapping (even with ALL valid
          fields), ``None``, a list or any other type is refused
          outright;
        * the result must have been ISSUED by this teacher's own
          candidate-evaluation adapter registry — a directly
          constructed (or mutated) result with identical fields is an
          UNKNOWN result and never certifies anything;
        * the Reference identity contract must be frozen RIGHT NOW —
          probes run against the frozen Reference or not at all;
        * ``student_candidate_id`` must be exactly the pinned strong
          Student; ``reference_candidate_id`` must equal the CURRENT
          frozen contract's candidate id, ``reference_checkpoint_hash``
          the contracted params sha256, and the episode reset protocol
          id/hash the contracted protocol (a result minted under a
          re-frozen Reference never certifies again);
        * the window id/hash and candidate-set hash carried by the
          result become the SCOPE of any later REUSE certification.

        Returns the stored record (all result fields); an exact
        duplicate is consumed only once. Raises GenManagerError with a
        greppable code on ANY violation. No file/network I/O, no
        training.
        """
        ctx = "e1_formal.consume_candidate_eval_result"
        if not isinstance(result, EA.DualProbeResult):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: caller-shaped mappings are NEVER accepted — "
                "expected an immutable DualProbeResult issued by the "
                "candidate-evaluation adapter registry, got "
                f"{type(result).__name__}",
            )
        if self._reference_contract is None:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BLOCKED,
                f"{ctx}: cannot consume dual-probe evidence while the "
                "Reference identity is unfrozen "
                "(REFERENCE_CONTRACT_UNFROZEN)",
            )
        if not self._eval_adapter_registry.lookup_result(result):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: this DualProbeResult was never ISSUED by this "
                "teacher's candidate-evaluation adapter registry — a "
                "directly constructed or forged result (unknown result) "
                "never certifies REUSE",
            )
        contracted = self._reference_contract
        if result.student_candidate_id != PINNED_STUDENT_CANDIDATE_ID:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: dual probes must run on the pinned strong "
                f"Student {PINNED_STUDENT_CANDIDATE_ID!r}, got "
                f"{result.student_candidate_id!r}",
            )
        if result.reference_candidate_id != contracted.candidate_id:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: reference_candidate_id "
                f"{result.reference_candidate_id!r} != the CURRENT "
                f"frozen Reference candidate {contracted.candidate_id!r}",
            )
        if result.reference_checkpoint_hash != contracted.params_sha256:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: reference_checkpoint_hash "
                f"{result.reference_checkpoint_hash!r} != the CURRENT "
                f"contracted Reference params sha256 "
                f"{contracted.params_sha256!r}",
            )
        if (
            result.episode_reset_protocol_id
            != contracted.episode_reset_protocol_id
        ):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: episode_reset_protocol_id "
                f"{result.episode_reset_protocol_id!r} != the CURRENT "
                "contracted episode reset protocol "
                f"{contracted.episode_reset_protocol_id!r}",
            )
        if (
            result.episode_reset_protocol_hash
            != contracted.episode_reset_protocol_hash
        ):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: episode_reset_protocol_hash does not match the "
                "CURRENT contracted episode reset protocol hash",
            )
        record = asdict(result)
        if record not in self._probe_attestations:
            self._probe_attestations.append(record)
        return dict(record)

    def _dual_probe_attested(
        self,
        probe: Mapping[str, str],
        window_id: str,
        window_hash: str,
        candidate_set_hash: str,
    ) -> bool:
        """True iff ``probe`` matches a consumed registry-issued result
        bound to this exact window, candidate set and the CURRENT
        frozen Reference candidate (C15)."""
        if self._reference_contract is None:
            return False
        contracted = self._reference_contract.candidate_id
        for att in self._probe_attestations:
            if att["reference_candidate_id"] != contracted:
                continue
            if att["window_id"] != window_id:
                continue
            if att["window_hash"] != window_hash:
                continue
            if att["candidate_set_hash"] != candidate_set_hash:
                continue
            if all(att[name] == probe[name] for name in _DUAL_PROBE_FIELDS):
                return True
        return False

    def _require_attested_dual_probe(
        self,
        probe: Mapping[str, str],
        ctx: str,
        window_id: str,
        window_hash: str,
        candidate_set_hash: str,
    ) -> None:
        """Fail closed unless ``probe`` matches a registry-issued
        dual-probe result bound to this window/candidate set."""
        if not self._dual_probe_attested(
            probe, window_id, window_hash, candidate_set_hash
        ):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: dual_probe matches no adapter-issued dual-probe "
                "attestation bound to this window, candidate set and "
                "the CURRENT frozen Reference — caller-supplied probe "
                "ids/hashes alone never certify REUSE",
            )

    def record_verified_batch(self, snapshot: Any) -> Dict[str, Any]:
        """Certify the previous window's FULLY VERIFIED 12+4 batch (C13).

        The ONLY legitimate REUSE source. Fail-closed requirements:

        * the Reference contract is frozen (G1) and the shared anchor
          manifest is FROZEN (G3) RIGHT NOW, with the snapshot bound to
          the current manifest sha and Reference candidate id;
        * learnability thresholds are frozen (a real dual-probe verdict
          is impossible without them);
        * ``provenance`` is exactly CANDIDATE_EVALUATION — necessary
          but NEVER sufficient (C14): certification additionally
          requires every structured evidence field below, so a
          provenance string alone can never certify REUSE;
        * C14 structured dual-probe evidence: a ``dual_probe`` block
          whose probes ran on the pinned strong Student (probe ids +
          sha256 hashes for both Student and Reference), the
          ``reference_identity_hash`` of the CURRENT frozen Reference
          identity, the ``window_hash`` recorded in the registry for
          every one of the 12 tasks, and the ``candidate_set_hash``
          over the ordered certified candidate set;
        * C15 attestation binding: the ``dual_probe`` block must match
          a dual-probe result ISSUED by the internal candidate-
          evaluation adapter registry and consumed via
          ``consume_candidate_eval_result``, bound to this exact
          window, candidate set and the CURRENT frozen Reference —
          caller-supplied probe ids/hashes (or mappings) alone never
          suffice;
        * EXACTLY 12 unique dynamic entries, each bound to a
          compiled artifact in this teacher's own registry with
          matching artifact_id, spec_hash and code sha256 (no entry
          may be certified the teacher never compiled);
        * the four anchors are exactly the canonical shared anchors.

        Returns the stored (cleaned) snapshot. Raises GenManagerError
        with a greppable code on ANY violation — a fake/empty/stale
        snapshot never becomes a REUSE source.
        """
        ctx = "e1_formal.record_verified_batch"
        blocked = self.current_gate_blockers_for_certification(ctx)
        if blocked:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BLOCKED,
                f"{ctx}: cannot certify a REUSE batch while hard gates "
                f"block: {blocked}",
            )
        if not isinstance(snapshot, Mapping):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: snapshot must be a mapping, got "
                f"{type(snapshot).__name__}",
            )
        unknown = sorted(k for k in snapshot if k not in _VERIFIED_SNAPSHOT_FIELDS)
        if unknown:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: unknown snapshot field(s) {unknown}",
            )
        for name in _VERIFIED_SNAPSHOT_FIELDS:
            if name not in snapshot:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISSING_FIELD,
                    f"{ctx}: missing field {name!r}",
                )
        window_id = snapshot["window_id"]
        if not isinstance(window_id, str) or not window_id.strip():
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISSING_FIELD,
                f"{ctx}: window_id must be a non-empty str",
            )
        # C14: window hash — format first, registry binding below
        window_hash = snapshot["window_hash"]
        if not _is_sha256_hex(window_hash):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: window_hash must be lowercase sha256 hex "
                f"(64 chars), got {window_hash!r}",
            )
        # provenance: necessary but NEVER sufficient (C14) — the
        # structured evidence fields around it are what certify REUSE
        provenance = snapshot["provenance"]
        if provenance != _VERIFIED_SNAPSHOT_PROVENANCE:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: provenance must be exactly "
                f"{_VERIFIED_SNAPSHOT_PROVENANCE!r} (real dual probes), "
                f"got {provenance!r}",
            )
        candidate_id = snapshot["reference_candidate_id"]
        if candidate_id != self._reference_contract.candidate_id:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: reference_candidate_id {candidate_id!r} != "
                f"frozen contract {self._reference_contract.candidate_id!r}",
            )
        # C14: bind the snapshot to the CURRENT frozen Reference
        # identity as a whole, not just the candidate-id string
        identity_hash = snapshot["reference_identity_hash"]
        if not _is_sha256_hex(identity_hash):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: reference_identity_hash must be lowercase "
                f"sha256 hex (64 chars), got {identity_hash!r}",
            )
        contracted_identity = reference_identity_sha256(
            self._reference_contract
        )
        if identity_hash != contracted_identity:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: reference_identity_hash {identity_hash!r} != "
                f"the CURRENT frozen Reference identity "
                f"{contracted_identity!r}",
            )
        manifest_sha = snapshot["anchor_manifest_sha256"]
        if manifest_sha != self._anchor_manifest.manifest_sha256:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: anchor_manifest_sha256 {manifest_sha!r} != "
                f"current frozen manifest "
                f"{self._anchor_manifest.manifest_sha256!r}",
            )
        # C14: structured Student/Reference dual-probe evidence
        probe = _validate_dual_probe(
            snapshot["dual_probe"], f"{ctx}.dual_probe"
        )
        anchor_ids = snapshot["anchor_task_ids"]
        if not isinstance(anchor_ids, (list, tuple)) or tuple(
            anchor_ids
        ) != layout.ANCHOR_TASK_IDS:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: anchor_task_ids must equal "
                f"{list(layout.ANCHOR_TASK_IDS)}, got {anchor_ids!r}",
            )
        raw_tasks = snapshot["dynamic_tasks"]
        if not isinstance(raw_tasks, (list, tuple)):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: dynamic_tasks must be a sequence",
            )
        if len(raw_tasks) != layout.NUM_DYNAMIC_SLOTS:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: dynamic_tasks must have exactly "
                f"{layout.NUM_DYNAMIC_SLOTS} entries, got {len(raw_tasks)}",
            )
        entries: List[Dict[str, str]] = []
        seen = set()
        for i, raw in enumerate(raw_tasks):
            entry_ctx = f"{ctx}.dynamic_tasks[{i}]"
            if not isinstance(raw, Mapping):
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                    f"{entry_ctx}: must be a mapping",
                )
            unknown = sorted(
                k
                for k in raw
                if k not in ("task_id", "artifact_id", "spec_hash",
                             "code_sha256")
            )
            if unknown:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                    f"{entry_ctx}: unknown field(s) {unknown}",
                )
            task_id = raw.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISSING_FIELD,
                    f"{entry_ctx}: needs non-empty task_id",
                )
            if task_id in seen:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: duplicate task_id {task_id!r}",
                )
            if task_id in layout.ANCHOR_TASK_IDS:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: dynamic id {task_id!r} collides with "
                    "a shared anchor",
                )
            seen.add(task_id)
            record = self._artifact_registry.get(task_id)
            if record is None:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: task {task_id!r} was never compiled "
                    "and recorded by this teacher — it cannot be part "
                    "of a verified window",
                )
            for field in ("artifact_id", "spec_hash", "code_sha256"):
                value = raw.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise GenManagerError(
                        GEN_MANAGER_SNAPSHOT_MISSING_FIELD,
                        f"{entry_ctx}: needs non-empty {field!r}",
                    )
            # C14: artifact_id must equal the INTERNAL registry record
            # — a snapshot quoting any other artifact id is a forgery
            if raw["artifact_id"] != record["artifact_id"]:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: artifact_id {raw['artifact_id']!r} "
                    f"!= the internal registry artifact id "
                    f"{record['artifact_id']!r} for {task_id!r}",
                )
            if raw["spec_hash"] != record["spec_hash"]:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: spec_hash does not match the "
                    f"registry record for {task_id!r}",
                )
            code_sha = hashlib.sha256(
                record["code"].encode("utf-8")
            ).hexdigest()
            if raw["code_sha256"] != code_sha:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: code_sha256 does not match the "
                    f"registry code for {task_id!r}",
                )
            # C14: every certified task is bound to the window hash
            # recorded in the registry (empty => never certifiable)
            record_window_hash = record.get("window_hash", "")
            if not record_window_hash:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: the registry record for {task_id!r} "
                    "carries no window_hash — the task can never be "
                    "part of a verified window",
                )
            if record_window_hash != window_hash:
                raise GenManagerError(
                    GEN_MANAGER_SNAPSHOT_MISMATCH,
                    f"{entry_ctx}: window_hash does not match the "
                    f"registry-recorded window hash for {task_id!r}",
                )
            entries.append(
                {
                    "task_id": task_id,
                    "artifact_id": raw["artifact_id"],
                    "spec_hash": raw["spec_hash"],
                    "code_sha256": raw["code_sha256"],
                }
            )
        window_ids = {
            self._artifact_registry[entry["task_id"]]["window_id"]
            for entry in entries
        }
        if len(window_ids) != 1:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: a verified batch must come from ONE window, "
                f"got ids from {sorted(window_ids)}",
            )
        if window_ids.pop() != window_id.strip():
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: window_id {window_id!r} does not match the "
                "registry-recorded window of the dynamic tasks",
            )
        # C14: candidate-set hash — canonical sha256 over the ordered
        # certified candidate set (any swapped/added/omitted id or a
        # reordered set fails closed)
        candidate_set_hash = snapshot["candidate_set_hash"]
        if not _is_sha256_hex(candidate_set_hash):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_BAD_TYPE,
                f"{ctx}: candidate_set_hash must be lowercase sha256 "
                f"hex (64 chars), got {candidate_set_hash!r}",
            )
        contracted_set_hash = canonical_sha256(
            [entry["task_id"] for entry in entries]
        )
        if candidate_set_hash != contracted_set_hash:
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: candidate_set_hash {candidate_set_hash!r} != "
                f"canonical hash over the certified candidate set "
                f"{contracted_set_hash!r}",
            )
        # C15: well-formed probe strings alone never suffice — they must
        # match a dual-probe result ISSUED by the internal adapter
        # registry, bound to this exact window, candidate set and the
        # CURRENT frozen Reference candidate
        self._require_attested_dual_probe(
            probe,
            f"{ctx}.dual_probe",
            window_id.strip(),
            window_hash,
            contracted_set_hash,
        )
        stored = {
            "window_id": window_id.strip(),
            "window_hash": window_hash,
            "provenance": provenance,
            "reference_candidate_id": candidate_id,
            "reference_identity_hash": identity_hash,
            "anchor_task_ids": list(layout.ANCHOR_TASK_IDS),
            "anchor_manifest_sha256": manifest_sha,
            "candidate_set_hash": candidate_set_hash,
            "dual_probe": dict(probe),
            "dynamic_tasks": entries,
        }
        self._verified_batch_snapshot = stored
        self._real_selection_completed = True
        return dict(stored)

    def current_gate_blockers_for_certification(self, ctx: str) -> List[str]:
        """Hard gates that block REUSE certification (G1/G2-config/G3)."""
        blocked: List[str] = []
        if self._reference_contract is None:
            blocked.append("REFERENCE_CONTRACT_UNFROZEN")
        if not self._anchor_manifest.is_frozen:
            blocked.append(AM.BLOCKED_SHARED_ANCHOR_MANIFEST)
        if self._thresholds is None:
            blocked.append(MT.LEARNABILITY_THRESHOLD_MISSING)
        return blocked

    def _snapshot_still_valid(self, snapshot: Any) -> bool:
        """Re-validate EVERY binding of a stored snapshot before REUSE.

        C15: caller-supplied strings alone never suffice — before any
        reuse the WHOLE snapshot is re-checked against the CURRENT
        internal state: gate blockers, provenance, canonical anchors,
        the current frozen anchor manifest sha, the current frozen
        Reference candidate and identity hash, the adapter-registry-
        ISSUED dual-probe results (matched in scope: window id/hash +
        candidate-set hash + CURRENT Reference candidate), and — for
        every one of the 12 dynamic tasks — the registry's artifact_id,
        spec_hash, code sha256, window id and window hash, plus the
        candidate-set hash over the ORDERED certified ids. ANY
        inconsistency (a stale window, a re-frozen Reference or reset
        protocol, a changed manifest, a reordered candidate set, a
        tampered stored snapshot, an unissued/out-of-scope probe)
        invalidates REUSE: the batch trains nothing. Never raises —
        any surprise is False (fail-closed).
        """
        if not isinstance(snapshot, Mapping):
            return False
        if self.current_gate_blockers_for_certification("e1_formal.reuse"):
            return False
        for name in _VERIFIED_SNAPSHOT_FIELDS:
            if name not in snapshot:
                return False
        if snapshot["provenance"] != _VERIFIED_SNAPSHOT_PROVENANCE:
            return False
        anchor_ids = snapshot["anchor_task_ids"]
        if not isinstance(anchor_ids, (list, tuple)) or tuple(
            anchor_ids
        ) != layout.ANCHOR_TASK_IDS:
            return False
        if (
            snapshot["anchor_manifest_sha256"]
            != self._anchor_manifest.manifest_sha256
        ):
            return False
        # the snapshot stays bound to the CURRENT frozen Reference
        # identity as a whole — a re-frozen identity (ANY contracted
        # field, including the episode reset protocol) invalidates it
        if (
            snapshot["reference_candidate_id"]
            != self._reference_contract.candidate_id
        ):
            return False
        identity_hash = snapshot["reference_identity_hash"]
        if not _is_sha256_hex(identity_hash):
            return False
        if identity_hash != reference_identity_sha256(
            self._reference_contract
        ):
            return False
        window_id = snapshot["window_id"]
        if not isinstance(window_id, str) or not window_id.strip():
            return False
        window_hash = snapshot["window_hash"]
        if not _is_sha256_hex(window_hash):
            return False
        candidate_set_hash = snapshot["candidate_set_hash"]
        if not _is_sha256_hex(candidate_set_hash):
            return False
        # C15: dual probes must still be structurally valid AND match a
        # registry-ISSUED result bound to this window, candidate set
        # and the CURRENT Reference candidate
        try:
            probe = _validate_dual_probe(
                snapshot["dual_probe"], "e1_formal.reuse.dual_probe"
            )
        except GenManagerError:
            return False
        if not self._dual_probe_attested(
            probe, window_id.strip(), window_hash, candidate_set_hash
        ):
            return False
        raw_tasks = snapshot["dynamic_tasks"]
        if not isinstance(raw_tasks, (list, tuple)):
            return False
        if len(raw_tasks) != layout.NUM_DYNAMIC_SLOTS:
            return False
        seen = set()
        task_ids: List[str] = []
        for raw in raw_tasks:
            if not isinstance(raw, Mapping):
                return False
            task_id = raw.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                return False
            if task_id in seen or task_id in layout.ANCHOR_TASK_IDS:
                return False
            seen.add(task_id)
            record = self._artifact_registry.get(task_id)
            if record is None:
                return False
            if raw.get("artifact_id") != record["artifact_id"]:
                return False
            if raw.get("spec_hash") != record["spec_hash"]:
                return False
            code_sha = hashlib.sha256(
                record["code"].encode("utf-8")
            ).hexdigest()
            if raw.get("code_sha256") != code_sha:
                return False
            # stale-window guard: the registry record must still carry
            # the SAME window id and window hash as the snapshot
            if record.get("window_id", "") != window_id.strip():
                return False
            record_window_hash = record.get("window_hash", "")
            if not record_window_hash or record_window_hash != window_hash:
                return False
            task_ids.append(task_id)
        # candidate-set hash over the ORDERED certified candidate set
        if snapshot["candidate_set_hash"] != canonical_sha256(task_ids):
            return False
        return True

    def _certify_dynamic_window(
        self,
        window_id: str,
        dynamic_ids: Sequence[str],
        provenance: str,
        ctx: str,
        *,
        dual_probe: Any,
    ) -> None:
        """Store registry-bound evidence as the verified snapshot.

        C14: the stored snapshot carries the SAME structured evidence
        as ``record_verified_batch`` — window hash (from the registry
        records), Reference identity hash (from the CURRENT frozen
        contract), candidate-set hash (canonical sha256 over the
        ordered certified ids) and the validated dual-probe block. A
        window whose registry records carry no window hash can never
        be certified (fail-closed).

        C15: the dual-probe block is validated HERE and must match a
        dual-probe result ISSUED by the internal candidate-evaluation
        adapter registry, bound to this exact window, candidate set
        and the CURRENT frozen Reference — this method is the
        certification choke point, so even a direct call can never
        store unissued probe strings.
        """
        entries = []
        window_hashes = set()
        for task_id in dynamic_ids:
            record = self._artifact_registry[task_id]
            window_hashes.add(record.get("window_hash", ""))
            entries.append(
                {
                    "task_id": task_id,
                    "artifact_id": record["artifact_id"],
                    "spec_hash": record["spec_hash"],
                    "code_sha256": hashlib.sha256(
                        record["code"].encode("utf-8")
                    ).hexdigest(),
                }
            )
        if len(window_hashes) != 1 or not next(iter(window_hashes)):
            raise GenManagerError(
                GEN_MANAGER_SNAPSHOT_MISMATCH,
                f"{ctx}: cannot certify a window whose registry records "
                f"carry no single non-empty window_hash "
                f"(got {sorted(window_hashes)})",
            )
        window_hash = next(iter(window_hashes))
        probe = _validate_dual_probe(dual_probe, f"{ctx}.dual_probe")
        self._require_attested_dual_probe(
            probe,
            f"{ctx}.dual_probe",
            window_id,
            window_hash,
            canonical_sha256(list(dynamic_ids)),
        )
        self._verified_batch_snapshot = {
            "window_id": window_id,
            "window_hash": window_hash,
            "provenance": provenance,
            "reference_candidate_id": (
                self._reference_contract.candidate_id
            ),
            "reference_identity_hash": reference_identity_sha256(
                self._reference_contract
            ),
            "anchor_task_ids": list(layout.ANCHOR_TASK_IDS),
            "anchor_manifest_sha256": (
                self._anchor_manifest.manifest_sha256
            ),
            "candidate_set_hash": canonical_sha256(list(dynamic_ids)),
            "dual_probe": dict(probe),
            "dynamic_tasks": entries,
        }
        self._real_selection_completed = True
        del ctx  # accepted for symmetric error context by callers

    def build_training_layout(
        self, dynamic_task_ids: Optional[Sequence[str]] = None
    ) -> Optional[Dict[str, float]]:
        """Layout hook: empty => None (legacy path), else fail-closed."""
        ids = list(dynamic_task_ids or ())
        if len(ids) == 0:
            return None
        return layout.build_training_layout(ids)

    # ------------------------------------------------------------------
    # honest state reporting
    # ------------------------------------------------------------------
    def current_blocked_codes(self) -> List[str]:
        """Every gate currently blocking real dynamic selection."""
        codes: List[str] = list(self._init_blocked_codes)
        if self._reference_contract is None:
            if "REFERENCE_CONTRACT_UNFROZEN" not in codes:
                codes.append("REFERENCE_CONTRACT_UNFROZEN")
        if not self._anchor_manifest.is_frozen:
            codes.append(AM.BLOCKED_SHARED_ANCHOR_MANIFEST)
        if not self._real_selection_completed:
            codes.append(E1Code.SELECTION_BLOCKED_NO_REAL_EVIDENCE)
        return codes

    def status_report(self) -> Dict[str, Any]:
        """Audit-grade honest state (used by the C11 smoke test)."""
        counts = self._ledger.counts()
        return {
            "teacher_type": TEACHER_TYPE,
            "flags": {
                "real_envcoder_used": self._flags.real_envcoder_used,
                "real_student_reference_eval": (
                    self._flags.real_student_reference_eval
                ),
                "real_training_update_executed": (
                    self._flags.real_training_update_executed
                ),
            },
            "reference_contract_frozen": self._reference_contract is not None,
            "anchor_manifest_status": self._anchor_manifest.status,
            "anchor_manifest_sha256": self._anchor_manifest.manifest_sha256,
            "learnability_thresholds_present": self._thresholds is not None,
            # round-3 P0-3: the invocation regime's honest state. When
            # False, threshold-driven gate signals compute to False with
            # INVOCATION_THRESHOLD_MISSING reasons (window invocation
            # degrades; the training gate is untouched by design)
            "invocation_thresholds_present": self.invocation_thresholds_present,
            "invocation_degradation": self._invocation_degradation,
            "copeland": {
                "protocol_version": selector.COPELAND_PROTOCOL_VERSION,
                "source_sha256": selector.COPELAND_SOURCE_SHA256,
                "constants_sha256": selector.COPELAND_CONSTANTS_SHA256,
                "base_sha256": selector.COPELAND_BASE_SHA256,
            },
            "llm_accounting": counts,
            "blocked_codes": self.current_blocked_codes(),
            "envcoder_check_scope": ENVCODER_CHECK_NOTE,
            # round-3 P0-4: the bounded EnvCoder repair scope in force
            # on the evolve path (legacy duck scope stays above)
            "envcoder_repair_scope": {
                "backend": self._envcoder_backend_name,
                "max_repairs": self._max_repairs,
                "max_repairs_bound": MAX_ENVCODER_REPAIRS,
                "repair_prompt_version": ENVCODER_REPAIR_PROMPT_VERSION,
                "stages_run": list(self._envcoder_backend.capabilities),
                "stages_blocked": [
                    list(pair)
                    for pair in self._envcoder_backend.stages_blocked
                ],
            },
            # round-3 P0-5: the shared runtime seam's honest
            # per-contract resolution state (this round: every contract
            # unbound — the seam only resolves; it never constructs,
            # mints or disguises any shared identity)
            "shared_runtime": {
                contract: {
                    "bound": resolution.bound,
                    "code": resolution.code,
                    "detail": resolution.detail,
                }
                for contract, resolution in (
                    SRS.resolve_all_shared_runtime().items()
                )
            },
            "disclaimers": [
                "E1_FORMAL_PLAN_ALIGNED means engineering alignment only, "
                "not a real closed loop",
                "E1S_STATIC_ABLATION_PRESERVED means artifacts preserved "
                "only, not that E1-S is runnable",
                "all REAL_* flags are false this round: replay EnvCoder, "
                "no real Student/Reference evaluation, no real training "
                "update",
                "no dynamic task is ever promoted without real probes; "
                "while hard gates block, the batch trains NOTHING (zero "
                "updates, no anchors-only sneak); REUSE is only the "
                "previous window's verified 12+4 with dual-probe "
                "results ISSUED by the internal candidate-evaluation "
                "adapter registry, Reference identity, checkpoint, "
                "window, candidate-set and reset-protocol hash "
                "evidence, re-validated against the current internal "
                "state on every reuse — a provenance string, a "
                "caller-shaped mapping or caller-supplied probe "
                "strings alone never certify it",
            ],
        }

    # ------------------------------------------------------------------
    # worker-dict helpers (compatible with run_dicode's
    # _process_worker_results: task_id / compiled / code / reasoning,
    # plus the legacy aliases generated_task_id / code_string since C11)
    # ------------------------------------------------------------------
    def _failed_worker(
        self,
        window_id: str,
        spec: Any,
        code: str,
        repairs_count: int = 0,
    ) -> Dict[str, Any]:
        # round-3 P0-4: the note states the BOUNDED repair outcome
        # honestly (F1 = the actual repair-call count for the template)
        if repairs_count > 0:
            note = f"EnvCoder failed after bounded repair (F1={repairs_count})"
        else:
            note = "EnvCoder failed before any repair attempt (F1=0)"
        return {
            "task_id": spec.spec_id,
            "generated_task_id": spec.spec_id,
            "compiled": False,
            "code": None,
            "code_string": None,
            "reasoning": "",
            "e1_status": {
                "reuse": False,
                "artifact_id": spec.artifact_id,
                "spec_hash": spec.spec_hash,
                "template_hash": spec.template_hash,
                "template_artifact_id": spec.template_artifact_id,
                "window_id": window_id,
                "compiled": False,
                "envcoder_failed_code": code,
                "repairs_count": repairs_count,
                "note": note,
            },
        }

    def _reuse_stub(
        self, tag: str, index: int, blocked_codes: Sequence[str], note: str
    ) -> Dict[str, Any]:
        stub_id = f"e1-reuse::{tag}::{index:02d}"
        return {
            "task_id": stub_id,
            "generated_task_id": stub_id,
            "compiled": False,
            "code": None,
            "code_string": None,
            "reasoning": "",
            "e1_status": {
                "reuse": True,
                "blocked_codes": list(blocked_codes),
                "note": note,
            },
        }

    def _reuse_batch(
        self, blocked_codes: Sequence[str], note: str
    ) -> List[Dict[str, Any]]:
        tag = f"s{self.session_idx}"
        return [
            self._reuse_stub(tag, index, blocked_codes, note)
            for index in range(layout.NUM_DYNAMIC_SLOTS)
        ]


def _consume_seed_examples(raw: Any, ctx: str) -> Tuple[Dict[str, str], ...]:
    """Validate the envcoder seed-example block fail-closed."""
    if not isinstance(raw, (list, tuple)):
        raise GenManagerError(
            GEN_MANAGER_BAD_TYPE,
            f"{ctx}: seed_examples must be a sequence, got "
            f"{type(raw).__name__}",
        )
    cleaned: List[Dict[str, str]] = []
    for i, example in enumerate(raw):
        example_ctx = f"{ctx}.seed_examples[{i}]"
        example = _require_mapping(example, example_ctx)
        unknown = sorted(
            k for k in example if k not in ("task_id", "description")
        )
        if unknown:
            raise GenManagerError(
                GEN_MANAGER_UNKNOWN_FIELD,
                f"{example_ctx}: unknown field(s) {unknown}",
            )
        task_id = example.get("task_id")
        description = example.get("description")
        if not isinstance(task_id, str) or not task_id.strip():
            raise GenManagerError(
                GEN_MANAGER_MISSING_FIELD,
                f"{example_ctx}: needs non-empty task_id",
            )
        if not isinstance(description, str) or not description.strip():
            raise GenManagerError(
                GEN_MANAGER_MISSING_FIELD,
                f"{example_ctx}: needs non-empty description",
            )
        cleaned.append(
            {"task_id": task_id.strip(), "description": description.strip()}
        )
    return tuple(cleaned)
