"""E1FormalGenManager — the edge teacher object of the formal
Behavior-Aware Regret-Guided LLM-UED direction (plan C10).

Duck-compatible with the legacy ``dicode.dreaming.gen_manager.
GenManager`` surface consumed by ``setup.py`` / ``run_dicode.py`` /
``evolution_efficient.py``:

* ``.session_idx``            mutable int, OWNED by the training loop;
* ``.archive``                provenance-admissible ``ArchiveView``;
* ``.env_generator.check_compilation(code) -> (bool, str)``;
* ``.evolve_tasks(dict_of_tasks, global_agent_profile) -> list[dict]``
  returns EXACTLY 12 worker dicts; BOTH arguments are IGNORED entirely
  (provenance rule: evolve-side metrics never enter the teacher, its
  prompts, or its ledger);
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
      => batch = 4 anchors + REUSE only

This module imports NO jax/craftax, performs NO network I/O and NO
file I/O (the anchor manifest and frozen manifest are injected as
mappings by the caller). ``check_compilation`` is a stdlib syntax-only
compile plus deterministic output guards this round — craftax is not
installed in the audit venv, so import/reset/step semantics are NOT
validated; ``status_report`` says so explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..static_llm.guards import raise_if_forbidden, scan_text
from . import anchor_manifest as AM
from . import layout
from . import metrics as MT
from . import selector
from .accounting import LLMCallLedger
from .archive_view import ArchiveView, consume_archive_snapshot, empty_archive_view
from .board import WINDOW_STATUS_COMPLETE
from .canonical import canonical_sha256
from .controller import run_review_cycle
from .envcoder import EnvCoderError, run_envcoder
from .evidence import build_evidence_snapshot
from .flags import parse_flags
from .invocation_gate import build_gate_state
from .llm_client import ReplayLLMClient
from .reference_contract import (
    ReferenceContractError,
    ReferenceIdentityContract,
    consume_reference_identity_contract,
)
from .schemas import E1Code, E1SchemaError, assert_llm_role_admissible
from .student_contract import PINNED_STUDENT_CANDIDATE_ID
from .task_specs import compile_task_specs

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

#: honest environment-compilation note for this round
ENVCODER_CHECK_NOTE = (
    "stdlib-syntax-only compile + deterministic guards; craftax "
    "import/reset/step semantics NOT validated this round (craftax "
    "absent from the audit venv)"
)


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
        # compiled E1 artifacts recorded by consume_worker_results (C11);
        # promotion happens ONLY via E1 selection, never legacy activation
        self._artifact_registry: Dict[str, Dict[str, Any]] = {}

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
        """Return EXACTLY 12 worker dicts (plan D9).

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
            # zero ledger calls; anchors + REUSE only.
            return self._reuse_batch(
                [GEN_MANAGER_NO_ADMISSIBLE_EVIDENCE]
                + self.current_blocked_codes(),
                "no admissible evidence; review window not opened; zero "
                "LLM calls",
            )

        evidence = build_evidence_snapshot(raw_items, ctx)
        gate_state = build_gate_state(
            {
                "session_idx": self.session_idx,
                # honest computation only: a window is first-window iff
                # no review cycle has run before. The other seven
                # triggers have NO honest signal source this round and
                # stay False (fail-closed; never fabricated).
                "is_first_window": self._cycles_run == 0,
                "capability_shift": False,
                "new_failure_pattern": False,
                "interventions_exhausted": False,
                "stagnation": False,
                "forgetting_regression": False,
                "exploration_slot_available": False,
                "curriculum_drift": False,
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

        if outcome.reuse:
            reason = (
                outcome.void_code
                if outcome.void_code
                else f"gate decision {outcome.decision.code} (REUSE)"
            )
            return self._reuse_batch(
                self.current_blocked_codes(),
                f"review cycle produced no usable window: {reason}",
            )

        # COMPLETE window -> canonical specs -> independent EnvCoder
        assert outcome.window is not None
        assert outcome.window.status == WINDOW_STATUS_COMPLETE
        compile_result = compile_task_specs(outcome.window)
        workers: List[Dict[str, Any]] = []
        for spec in compile_result.specs[: layout.NUM_DYNAMIC_SLOTS]:
            try:
                artifact = run_envcoder(
                    self._llm,
                    spec=spec,
                    seed_examples=self._seed_examples,
                    ledger=self._ledger,
                    window_id=window_id,
                )
            except EnvCoderError as e:
                # single-pass: parse/guard failure produces a
                # non-compiled worker; NO repair loop (F1 stays 0)
                workers.append(self._failed_worker(window_id, spec, e.code))
                continue
            ok, note = self._env_generator.check_compilation(artifact.env_code)
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
                        "window_id": window_id,
                        "window_hash": spec.window_hash,
                        "compiled": ok,
                        "compile_note": note,
                        "envcoder_check": ENVCODER_CHECK_NOTE,
                    },
                }
            )
        # pad to EXACTLY 12 with honest REUSE stubs
        index = 0
        while len(workers) < layout.NUM_DYNAMIC_SLOTS:
            workers.append(
                self._reuse_stub(
                    window_id,
                    index,
                    self.current_blocked_codes(),
                    "fewer than 12 admissible artifacts this window",
                )
            )
            index += 1
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
    # batch + layout (G2/G3 degradation lands here)
    # ------------------------------------------------------------------
    def build_training_batch(
        self, promoted_dynamic_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Blocked => 4 anchors + REUSE only; 12 promoted ids => 16."""
        ids = list(promoted_dynamic_ids or ())
        blocked = self.current_blocked_codes()
        if len(ids) == 0:
            return {
                "task_ids": list(layout.ANCHOR_TASK_IDS),
                "layout": None,
                "dynamic_promoted": 0,
                "reuse_only": True,
                "blocked_codes": blocked,
                "notes": [
                    "no dynamically promoted tasks: selection requires "
                    "real dual probes (G2); anchors + REUSE only (D5)",
                    "anchor retention evaluation stays blocked until the "
                    "shared manifest is frozen (G3)",
                ],
            }
        if len(ids) != layout.NUM_DYNAMIC_SLOTS:
            raise GenManagerError(
                GEN_MANAGER_BAD_DYNAMIC_SET,
                f"e1_formal.batch: promoted dynamic set must have exactly "
                f"{layout.NUM_DYNAMIC_SLOTS} ids, got {len(ids)}",
            )
        layout_map = layout.build_training_layout(ids)
        return {
            "task_ids": list(ids) + list(layout.ANCHOR_TASK_IDS),
            "layout": layout_map,
            "dynamic_promoted": len(ids),
            "reuse_only": False,
            "blocked_codes": blocked,
            "notes": [],
        }

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
            "copeland": {
                "protocol_version": selector.COPELAND_PROTOCOL_VERSION,
                "source_sha256": selector.COPELAND_SOURCE_SHA256,
                "constants_sha256": selector.COPELAND_CONSTANTS_SHA256,
                "base_sha256": selector.COPELAND_BASE_SHA256,
            },
            "llm_accounting": counts,
            "blocked_codes": self.current_blocked_codes(),
            "envcoder_check_scope": ENVCODER_CHECK_NOTE,
            "disclaimers": [
                "E1_FORMAL_PLAN_ALIGNED means engineering alignment only, "
                "not a real closed loop",
                "E1S_STATIC_ABLATION_PRESERVED means artifacts preserved "
                "only, not that E1-S is runnable",
                "all REAL_* flags are false this round: replay EnvCoder, "
                "no real Student/Reference evaluation, no real training "
                "update",
                "no dynamic task is ever promoted without real probes; "
                "selection degrades to anchors + REUSE",
            ],
        }

    # ------------------------------------------------------------------
    # worker-dict helpers (compatible with run_dicode's
    # _process_worker_results: task_id / compiled / code / reasoning,
    # plus the legacy aliases generated_task_id / code_string since C11)
    # ------------------------------------------------------------------
    def _failed_worker(self, window_id: str, spec: Any, code: str) -> Dict[str, Any]:
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
                "window_id": window_id,
                "compiled": False,
                "envcoder_failed_code": code,
                "note": "single-pass EnvCoder failure; NO repair (F1=0)",
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
