"""FeedbackUEDController — double-window state machine (C8 rewrite).

Per window k (all k >= 0; window 0's k-1 feedback view is empty, the board
still runs its complete six roles):

    A. EVIDENCE  — behavior-failure evidence of window k-1 probes +
                   FeedbackView over EXACTLY the frozen feedback of window
                   k-1 (CC3 C9 gate: the lag is exactly one window —
                   older/current/future records fail closed as
                   STALE_FEEDBACK_ID; the BoardContext is assembled from
                   the view ONLY, never from the raw store). Static
                   (no-feedback control) mode gets the P0-12 shape-matched
                   MaskedFeedbackView — the SAME window k-1 record set,
                   every value a controlled NULL/MASK — so the board runs
                   the same computation with no feedback content;
                   shuffled mode the frozen permuted + anonymized
                   PermutedFeedbackView;
    B. BOARD     — six-role Review Board, always all six calls: verdicts on
                   window k-1 feedback (explicit feedback_id /
                   hypothesis_id / prediction-signature citations) + new
                   PENDING hypotheses + AxisDirectives + per-family
                   proposals;
    C. REVISION  — verdict application to the ledger (P0-6 binding guard) +
                   Reconciler -> plan_k + training-seam no-op bookkeeping;
    D. PROBING   — EnvCoder (7th call) -> compile/reset/step gates ->
                   directive-driven candidate expansion -> staged funnel
                   probe -> expected-vs-observed grading -> STAGED
                   feedback_k (not yet visible to anything);
    E. FREEZE    — feedback_k records are written atomically to the store
                   and graded, the board's new hypotheses are registered,
                   and the window is marked FROZEN.

After phase E, ANY verdict application or plan change for window k raises
``SAME_WINDOW_REVISION_FORBIDDEN`` (fail closed). Revision driven by
feedback_k is only possible at window k+1, through the complete six-role
board citing feedback_k's feedback ids — revisions always lag feedback by
exactly one window (``NEXT_WINDOW_REVISION_ONLY`` /
``SAME_WINDOW_REVISION_REJECTED``).

Three comparison modes (§5) share the SAME six roles / EnvCoder / probe /
training seam / seeds / budget:

* ``static_llm``        — the board reads the P0-12 shape-matched
                          MaskedFeedbackView (same item count / field set,
                          every value a controlled NULL/MASK); every
                          revision is EXPLORATION;
* ``normal_feedback``   — honest candidate<->feedback binding;
* ``shuffled_feedback`` — the FeedbackStore stays HONEST in every mode; the
                          isolation happens at view time: the board reads a
                          PermutedFeedbackView — a frozen, recomputable
                          permutation of the frozen records presented under
                          anonymized ids with all identity side channels
                          masked, so the real candidate<->feedback pairing
                          is unrecoverable from the board context. The board
                          cites anonymized ids; only this controller (the
                          honest bookkeeper) resolves them back to store ids
                          when applying verdicts / closing proposals. This
                          proves the plan really is a function of the
                          feedback and not of window order.

C10 RETIRE lifecycle: the controller is the sole keeper of the retirement
registry (family -> retirement window). A retired family is in COOLDOWN for
the next ``RETIRE_COOLDOWN_WINDOWS`` windows; the board context carries the
blocked lists so the six roles skip those families by construction, and the
Reconciler re-checks fail closed (FAMILY_IN_COOLDOWN / FAMILY_NOT_REOPENED).
Past the cooldown a family STAYS retired until reopened — either through the
explicit ``human_reopen_families`` constructor authorization or because ALL
distinguishing probe evidence postdates the retirement window. A STALE
verdict can therefore never resurrect a retired family.

C11 REQUEST_CONTROL blocking: when a window's board requests human control
(critic escalation and/or tutor REQUEST_CONTROL proposals), the loop HALTS
immediately after phase B: no verdicts are applied, no plan is produced, no
probe runs, nothing is frozen — the stopped window produced NO execution
batch. The halt is recorded as a deterministic HumanDecisionArtifact (bound
to the escalated window's ``board_hash``, tutor citations resolved to real
store ids through the window's view), the WindowRecord documents the stop
(phase stays BOARD, revision_label = REQUEST_CONTROL_STOPPED, training
NOT_EXECUTED_REQUEST_CONTROL), and the RunSummary carries the artifact plus
the LaunchGate ``final_batch`` verdict, whose ``final`` is ALWAYS False for
a stopped loop (and for this whole round, since training is unauthorized).
The stopped window is closed to revision exactly like a frozen one: its
phase is not REVISION, so apply_board_verdicts / revise_plan raise
SAME_WINDOW_REVISION_FORBIDDEN.

C13 shared anchor manifest: the four standard-reset anchor slots are filled
ONLY through the ``anchor_manifest.AnchorManifestSource`` seam — explicit
injection of a cross-direction shared FROZEN manifest (manifest_hash
recomputed and compared) or fail closed. Verified fact for this worktree:
no such frozen manifest exists, so every window falls back to the scaffold
placeholder (the four canonical anchor ids, explicitly labeled
``SCAFFOLD_PLACEHOLDER_NOT_SHARED`` — NOT a shared binding) with the budget
unchanged (12 dynamic + 4 anchors) in all three modes, and
``SHARED_ANCHOR_MANIFEST_BOUND`` stays False. Every WindowRecord carries its
``anchor_binding`` label; formal retention may only count manifest-bound
anchor probes, and this round's placeholder anchors are reported as exactly
that.

C15 persistence: ``persistence.snapshot_controller`` captures the COMPLETE
frozen state of this machine (phase map, mode, next window via the completed
WindowRecords, ledger/store dumps, revisions, plans, retirement registry,
human-reopen authorization, anchor binding, sequence, runner + backend usage
counters, board/envelope metadata) under one recomputable ``snapshot_hash``;
``restore_controller`` rebuilds a controller whose ``run()`` CONTINUES the
same logical run — a freeze-point snapshot -> restore -> continue summary is
byte-identical to the uninterrupted run's summary, and any tamper with the
snapshot fails closed (HASH_CHAIN_BROKEN).

Honesty posture re-asserted at construction: every real-world authorization
flag must be False this round; the loop runs on the deterministic mock LLM
backend + the deterministic symbolic probe runner and says so in every
record (ENGINEERING_SCAFFOLD).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.anchor_manifest import (
    SCAFFOLD_PLACEHOLDER_NOT_SHARED,
    SHARED_MANIFEST_BOUND_LABEL,
    AnchorManifestBlocked,
    AnchorManifestSource,
    SharedAnchorManifest,
)
from d052.feedback_llm_ued.behavior_failure import assemble_board_context
from d052.feedback_llm_ued.deterministic_reconciler import (
    DeterministicReconciler,
)
from d052.feedback_llm_ued.env_coder import run_env_coder
from d052.feedback_llm_ued.env_coder_gate import EnvCoderGate
from d052.feedback_llm_ued.environment_generator import (
    generate_candidates_from_directives,
)
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_MOCK_DRY_RUN,
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
)
from d052.feedback_llm_ued.expected_observed import ExpectedObservedComparator
from d052.feedback_llm_ued.feedback_contracts import (
    CurriculumPlan,
    ProbeMetrics,
    plan_signature_hash,
)
from d052.feedback_llm_ued.causal_failure_analyst import BoardHypothesisVerdict
from d052.feedback_llm_ued.feedback_view import (
    MaskedFeedbackView,
    NormalFeedbackView,
    NullFeedbackView,
    PermutedFeedbackView,
)
from d052.feedback_llm_ued.intervention_tutor import FamilyProposal
from d052.feedback_llm_ued.formal_isolation import FormalSourceIsolationGuard
from d052.feedback_llm_ued.human_decision import HumanDecisionArtifact
from d052.feedback_llm_ued.hypothesis_ledger import (
    HypothesisLedger,
    HypothesisRecord,
)
from d052.feedback_llm_ued.llm_backend import (
    DeterministicMockFeedbackBackend,
    assert_no_real_llm_usage,
)
from d052.feedback_llm_ued.plan_revision import (
    FEEDBACK_DRIVEN_LABEL,
    PlanModification,
    PlanRevisionRecord,
    assert_feedback_ids_known,
)
from d052.feedback_llm_ued.review_board import BoardOutput, run_review_board
from d052.feedback_llm_ued.runtime_authorization import (
    empty_authorization,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.simulator_probe import (
    DeterministicSymbolicProbeRunner,
    run_staged_funnel,
)
from d052.feedback_llm_ued.student_binding import (
    EXECUTED_ONE_UPDATE_STATUS,
    RealTwoWindowSmokePolicy,
    StudentTrainingSeam,
    TrainingStepRecord,
    local_symbolic_binding,
    resolve_student_binding,
)

#: deterministic bootstrap: first four families carry the seeded hypotheses
_BOOTSTRAP_FAMILIES = C.ENVIRONMENT_FAMILIES[:4]

#: slots requested per funded board proposal (the Reconciler caps/tops up;
#: RETIRE and REQUEST_CONTROL proposals always request zero budget)
PROPOSAL_DEFAULT_SLOTS = 4

# ---------------------------------------------------------------------------
# double-window phase machine (fixed order, monotone per window)
# ---------------------------------------------------------------------------
PHASE_EVIDENCE = "EVIDENCE"
PHASE_BOARD = "BOARD"
PHASE_REVISION = "REVISION"
PHASE_PROBING = "PROBING"
PHASE_FROZEN = "FROZEN"
_PHASE_ORDER = (PHASE_EVIDENCE, PHASE_BOARD, PHASE_REVISION, PHASE_PROBING,
                PHASE_FROZEN)


class SameWindowRevisionForbidden(RuntimeError):
    """A verdict / plan change was attempted outside the window's REVISION
    phase (i.e. after feedback_k was staged or frozen) — the double-window
    state machine refuses it, fail closed."""


class StateMachineViolation(RuntimeError):
    """The phase machine was asked to move backwards — an internal bug."""


@dataclass
class WindowRecord:
    window: int
    mode: str
    phase: str
    evidence_window: int
    feedback_view_label: str
    board_call_count: int
    env_coder_call_count: int
    n_llm_calls: int
    request_control: bool
    global_risk: str
    plan_id: str
    plan_signature_hash: str
    revision_label: str
    n_directives: int = 0
    n_coded: int = 0
    gate_passed: bool = False
    n_candidates: int = 0
    n_feedback_records: int = 0
    funnel_stats: Dict[str, int] = field(default_factory=dict)
    window_aggregates: Dict[str, float] = field(default_factory=dict)
    training_step_status: str = ""
    #: C13: provenance label of this window's four anchor slots —
    #: SCAFFOLD_PLACEHOLDER_NOT_SHARED (this round, always) or
    #: SHARED_ANCHOR_MANIFEST_BOUND (only with an injected frozen manifest)
    anchor_binding: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunSummary:
    mode: str
    n_windows: int
    n_llm_calls: int
    revision_rate: float
    decision_distribution: Dict[str, int]
    feedback_citation_coverage: float
    supported_retention_rate: float
    refuted_retirement_rate: float
    total_simulator_transitions: int
    transitions_per_useful_environment: float
    plan_signature_hashes: List[str]
    windows: List[dict]
    # -- C11 REQUEST_CONTROL blocking audit trail --------------------------
    request_control_stopped: bool = False
    stopped_window: Optional[int] = None
    human_decision_artifact: Optional[dict] = None
    #: FeedbackLaunchGate.evaluate_final_batch verdict (FinalBatchDecision
    #: as a dict); ``final`` is always False for a stopped loop
    final_batch: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _assert_authorization_posture() -> None:
    for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
        if getattr(C, name):
            raise RuntimeError(
                f"AUTHORIZATION_POSTURE_VIOLATED: {name}=True, but this "
                "package is written for a round where every real-world "
                "capability flag is False")


class FeedbackUEDController:
    """Deterministic, replayable driver of the double-window loop."""

    def __init__(self, mode: str, *, backend=None, probe_runner=None,
                 human_reopen_families=(), anchor_manifest=None,
                 runtime_authorization=None, student_init_contract=None,
                 training_contract=None, real_env_coder_callable=None,
                 probe_feedback_builder=None,
                 reference_identity_hash: str = "",
                 two_window_smoke_policy:
                 Optional[RealTwoWindowSmokePolicy] = None,
                 dicode_batch_binding=None,
                 dicode_runtime_identity: str = "",
                 runtime_bundle_hash: str = "") -> None:
        """The five trailing kwargs are the PRODUCTION-PATH seams (P0-1..4).
        All default to None, which reproduces the historical mock/symbolic
        behavior byte for byte. With any real capability granted through
        ``runtime_authorization`` the controller rejects every stand-in:
        a non-real backend / probe runner, the local symbolic Student
        binding, and a missing shared training contract all fail closed.
        """
        if mode not in C.FEEDBACK_MODES:
            raise ValueError(f"UNKNOWN_MODE: {mode!r}")
        _assert_authorization_posture()
        for fam in human_reopen_families:
            if fam not in C.ENVIRONMENT_FAMILIES:
                raise ValueError(
                    f"UNKNOWN_ENVIRONMENT_FAMILY: human_reopen_families "
                    f"entry {fam!r}")
        #: C10: explicit human authorization to reopen retired families once
        #: their cooldown has passed (the only non-evidence reopen path)
        self.human_reopen_families = frozenset(human_reopen_families)
        self.mode = mode
        #: the round constants stay False; production authorization flows
        #: ONLY through the injected runtime grants, and those take effect
        #: only in EXECUTION_MODE_REAL (default: byte-identical gate)
        self.runtime_authorization = (runtime_authorization
                                      or empty_authorization())
        if runtime_authorization is None:
            self.launch_gate = FeedbackLaunchGate(EXECUTION_MODE_MOCK_DRY_RUN)
        else:
            self.launch_gate = FeedbackLaunchGate(
                EXECUTION_MODE_REAL, runtime_grants=runtime_authorization)
        self.launch_decision = self.launch_gate.evaluate()
        self.backend = backend or DeterministicMockFeedbackBackend()
        self.launch_gate.assert_backend_allowed(self.backend.kind)
        if self.launch_decision.real_llm_calls_allowed \
                and self.backend.kind != C.BACKEND_KIND_REAL:
            raise RuntimeError(
                "REAL_RUN_BACKEND_NOT_REAL: real LLM calls are authorized "
                f"but the backend kind is {self.backend.kind!r}; running on "
                "a mock/replay backend while claiming real is forbidden "
                "(NO_SILENT_FALLBACK)")
        self.runner = probe_runner or DeterministicSymbolicProbeRunner()
        if self.launch_decision.real_simulator_probe_allowed \
                and getattr(self.runner, "real_simulator", None) is not True:
            raise RuntimeError(
                "REAL_RUN_PROBE_RUNNER_NOT_REAL: real simulator probes are "
                "authorized but the runner is not real_simulator=True; "
                "candidate-hash-derived symbolic metrics may not serve a "
                "real run (NO_SILENT_FALLBACK)")
        self.ledger = HypothesisLedger()
        self.store = SimulatorFeedbackStore()
        self.comparator = ExpectedObservedComparator()
        self.reconciler = DeterministicReconciler()
        self.env_coder_gate = EnvCoderGate()
        self.isolation = FormalSourceIsolationGuard()
        # CC4 Student binding seam: the CC4 shared StudentAdapter is absent
        # from this worktree (verified), so the loop carries the honest local
        # symbolic binding (NOT_LOADED_LOCAL / ENGINEERING_SCAFFOLD) and the
        # training seam only records SKIPPED_UNAUTHORIZED this round.
        # PRODUCTION PATH: any granted real capability rejects the local
        # symbolic binding — the shared StudentInitContract must be injected
        # explicitly (resolve_student_binding fails closed with
        # STUDENT_INIT_CONTRACT_MISSING otherwise).
        if self.launch_decision.real_simulator_probe_allowed \
                or self.launch_decision.training_allowed:
            self.student_binding = resolve_student_binding(
                student_init_contract)
        else:
            self.student_binding = local_symbolic_binding()
        self.training_seam = StudentTrainingSeam(
            self.launch_gate, self.student_binding,
            training_contract=training_contract,
            runtime_bundle_hash=runtime_bundle_hash)
        #: P0-10: the two-window smoke update-count contract (exactly one
        #: optimizer update, in the window that consumes feedback_k).
        #: None = historical behavior (every training-allowed window
        #: updates); the two-window real entrypoint injects the policy.
        self.two_window_smoke_policy = two_window_smoke_policy
        #: P0-16 (DiCode 15+1): the director-declared batch binding
        #: (non-target anchors + OriginalTask) and the CanonicalDiCodeOne-
        #: UpdateRuntime identity. When present, the window k+1 update is
        #: executed through the director's DiCode runtime over the 15
        #: curriculum ids; the OriginalTask is appended internally once.
        self.dicode_batch_binding = dicode_batch_binding
        self.dicode_runtime_identity = dicode_runtime_identity
        #: P0-2 seam: the real EnvCoder execution chain (unique template +
        #: four-link verification + bounded repair). Injectable only when
        #: the runtime grants authorize it; the default symbolic path stays
        #: untouched.
        if real_env_coder_callable is not None \
                and not self.runtime_authorization.real_envcoder:
            raise RuntimeError(
                "REAL_ENVCODER_SEAM_NOT_AUTHORIZED: a real EnvCoder "
                "callable was injected but the runtime grant real_envcoder "
                "is false")
        self._real_env_coder_callable = real_env_coder_callable
        #: P0-3 seam: production-path feedback record builder (provenance-
        #: bound SimulatorFeedbackRecords). Injectable only when real
        #: simulator probes are authorized.
        if probe_feedback_builder is not None \
                and not self.launch_decision.real_simulator_probe_allowed:
            raise RuntimeError(
                "PROBE_FEEDBACK_BUILDER_NOT_AUTHORIZED: a production "
                "feedback builder was injected but real simulator probes "
                "are not authorized")
        self._probe_feedback_builder = probe_feedback_builder
        #: P0-1 (CC3 follow-up audit): the Reference identity hash bound
        #: into every board envelope's context binding — populated by the
        #: production entrypoint ONLY when the shared ReferenceAdapter slot
        #: is actually bound; empty string = unbound (never derived
        #: silently). Default reproduces the historical behavior.
        self.reference_identity_hash = reference_identity_hash
        self.revisions: List[PlanRevisionRecord] = []
        self.envelopes: List[object] = []
        self.plans: Dict[str, CurriculumPlan] = {}
        self.boards: Dict[int, BoardOutput] = {}
        self.training_log: List[object] = []
        self._plans_by_window: Dict[int, CurriculumPlan] = {}
        self._window_feedback: Dict[int, List[str]] = {}
        self._phases: Dict[int, str] = {}
        self._sequence = 0
        self._summary: Optional[RunSummary] = None
        #: C15 persistence: seeding happens exactly once per logical run
        #: (restore re-injects the seeded ledger instead of re-seeding), and
        #: the completed WindowRecords survive across snapshot/restore so a
        #: resumed run() continues — and summarizes — the SAME logical run.
        self._seeded = False
        self._completed_records: List[WindowRecord] = []
        #: C15: board audit metadata surviving restore (window -> board_hash);
        #: the live BoardOutput objects only exist within their own window
        self.board_hashes: Dict[int, str] = {}
        #: C10 RETIRE lifecycle registry: family -> window it was retired at.
        #: Sole keeper is this controller; the board reads blocked lists via
        #: the board context, the Reconciler re-checks fail closed.
        self._retired_at: Dict[str, int] = {}
        #: C11: one artifact per REQUEST_CONTROL stop (the loop halts at the
        #: first one, so at most one entry in practice — the list keeps the
        #: type honest for future resume-after-human-decision work)
        self.human_decision_artifacts: List[HumanDecisionArtifact] = []
        # -- C13 shared anchor manifest seam --------------------------------
        # Explicit injection only. A frozen manifest with a matching
        # recomputed hash binds the anchor slots; anything else (absent,
        # unfrozen) fails closed into the LABELED scaffold placeholder — the
        # budget (12 dynamic + 4 anchors) is identical in both cases, and
        # every WindowRecord records which of the two states applied.
        if isinstance(anchor_manifest, Mapping):
            anchor_manifest = SharedAnchorManifest(**anchor_manifest)
        self.anchor_source = AnchorManifestSource(manifest=anchor_manifest)
        try:
            self.anchor_ids = self.anchor_source.resolve()
            self.anchor_binding = SHARED_MANIFEST_BOUND_LABEL
        except AnchorManifestBlocked:
            self.anchor_ids = self.anchor_source.scaffold_placeholder()
            self.anchor_binding = SCAFFOLD_PLACEHOLDER_NOT_SHARED

    # ------------------------------------------------------------------ seeds
    def _seed(self) -> None:
        """Deterministic hypothesis seeds. There is NO bootstrap plan: plan_0
        is produced by window 0's board + Reconciler exactly like every
        other window's plan (all-exploration, since no feedback exists)."""
        for i, family in enumerate(_BOOTSTRAP_FAMILIES):
            self.ledger.register(HypothesisRecord(
                hypothesis_id=f"hyp-{i:02d}", source_window=0,
                target_behavior=f"baseline_probe_{family}",
                evidence_ids=[f"seed-evidence-{family}"],
                #: prediction under test: "the Student operates in the
                #: middle-difficulty regime on this family"
                predicted_signature={"student_success_rate": 0.47,
                                     "student_behavior_activation": 0.55,
                                     "student_front_progress": 0.48},
                environment_family=family, confidence=0.5))

    # ------------------------------------------------------------------ loop
    def run(self, max_windows: int = C.MAX_WINDOWS) -> RunSummary:
        if not self._seeded:
            self._seed()
            self._seeded = True
        records = self._completed_records
        stopped_window: Optional[int] = None
        if records and records[-1].request_control:
            # C11+C15: a restored run whose last record already stopped the
            # loop stays stopped — nothing else is applied autonomously.
            stopped_window = records[-1].window
        else:
            for window in range(len(records), max_windows):
                record = self._run_window(window)
                records.append(record)
                if record.request_control:
                    # C11: the board requested human control — the loop HALTS
                    # right after the escalated board. No execution batch, no
                    # probe, nothing else is applied autonomously.
                    stopped_window = window
                    break
        self._summary = self._build_summary(records,
                                            stopped_window=stopped_window)
        self._assert_two_window_smoke_update_count(
            stopped_window=stopped_window)
        if self.launch_decision.real_llm_calls_allowed:
            #: the inverse honesty check: a real-authorized run MUST have
            #: consumed real calls and may not have served a single mock
            #: completion (silently falling back to the mock backend and
            #: still claiming "real" is forbidden — NO_SILENT_FALLBACK)
            usage = self.backend.usage
            if usage.real_calls <= 0:
                raise RuntimeError(
                    "REAL_LLM_USAGE_MISSING: real LLM calls are authorized "
                    "but the backend served zero real calls — a real run "
                    "that never called the real backend is not a real run")
            if usage.mock_calls != 0:
                raise RuntimeError(
                    f"MOCK_CALLS_IN_REAL_RUN: {usage.mock_calls} mock "
                    "completion(s) were served inside a real-authorized "
                    "run — silent mock fallback is forbidden")
        else:
            assert_no_real_llm_usage(self.backend.usage)
        return self._summary

    def _assert_two_window_smoke_update_count(self, *, stopped_window
                                              ) -> None:
        """P0-10 end-of-run enforcement: a COMPLETED smoke run must have
        executed EXACTLY the policy's update count — no more, no less.

        Not enforced when no policy is injected, when training is not
        authorized, or when the loop halted on REQUEST_CONTROL (nothing
        past phase B executed there — the halt itself is the honest
        outcome and is reported as such).
        """
        policy = self.two_window_smoke_policy
        if policy is None or not self.launch_decision.training_allowed:
            return
        if stopped_window is not None:
            return
        executed = sum(1 for t in self.training_log
                       if t.status == EXECUTED_ONE_UPDATE_STATUS)
        if executed != policy.updates_expected_total:
            raise RuntimeError(
                "TWO_WINDOW_SMOKE_UPDATE_COUNT_MISMATCH: the smoke policy "
                f"expects exactly {policy.updates_expected_total} optimizer "
                f"update(s) for the completed run but {executed} executed "
                "(the single update belongs to the window that consumes "
                "feedback_k — NO_SILENT_FALLBACK)")

    def _run_window(self, window: int) -> WindowRecord:
        n_calls_before = self.backend.usage.total_calls

        # -- A. evidence assembly: the board reads EXACTLY the frozen
        #    window k-1 feedback, and the BoardContext is built ONLY from
        #    the FeedbackView — never from the raw store (CC3 C9 gate) ----
        self._set_phase(window, PHASE_EVIDENCE)
        evidence_window = max(0, window - 1)
        view = self._feedback_view(window)
        #: P0-12: EVERY mode runs the same lifecycle query — the static
        #: (no-feedback control) mode's masked view resolves no citation,
        #: so its retirement registry can never become non-empty and the
        #: query returns the empty partition by construction (re-checked
        #: fail-closed inside _retirement_state).
        in_cooldown, blocked_retired, _reopened = \
            self._retirement_state(window)
        board_context = assemble_board_context(
            view, window=evidence_window, mode=self.mode,
            families_in_cooldown=in_cooldown,
            retired_families=blocked_retired)

        # -- B. six-role Review Board (always all six calls) ----------------
        self._set_phase(window, PHASE_BOARD)
        #: P0-1: bind the canonical context identities into every role
        #: envelope — Student identity (always held), Reference identity
        #: (only when the shared slot is bound) and the PREVIOUS window's
        #: plan hash (window 0 has no predecessor -> unbound)
        previous_plan = self._plans_by_window.get(window - 1)
        board = run_review_board(
            window=window, mode=self.mode, board_context=board_context,
            view=view, hypotheses=self.ledger.all(), backend=self.backend,
            sequence_start=self._sequence,
            student_identity_hash=self.student_binding.identity_hash,
            reference_identity_hash=self.reference_identity_hash,
            previous_plan_hash=(previous_plan.plan_hash
                                if previous_plan is not None else ""))
        self._sequence += C.BOARD_CALLS_PER_WINDOW
        self.envelopes.extend(board.envelopes)
        self.boards[window] = board
        self.board_hashes[window] = board.board_hash

        # -- C11 REQUEST_CONTROL blocking: the board asked a human to take
        #    over. Halt IMMEDIATELY after phase B: no verdict application,
        #    no plan, no probe, no freeze — this window produced NO
        #    execution batch. The stopped WindowRecord keeps phase BOARD,
        #    which also closes it to revision (SAME_WINDOW_REVISION_-
        #    FORBIDDEN on any later apply_board_verdicts / revise_plan).
        if board.request_control:
            artifact = self._escalation_artifact(window, board, view)
            self.human_decision_artifacts.append(artifact)
            return WindowRecord(
                window=window, mode=self.mode, phase=self._phases[window],
                evidence_window=evidence_window,
                feedback_view_label=view.label,
                board_call_count=C.BOARD_CALLS_PER_WINDOW,
                env_coder_call_count=0,
                n_llm_calls=self.backend.usage.total_calls - n_calls_before,
                request_control=True,
                global_risk=board.critic.global_risk,
                plan_id="",
                plan_signature_hash="",
                revision_label=C.REVISION_LABEL_REQUEST_CONTROL_STOPPED,
                n_directives=len(board.directives),
                n_coded=0,
                gate_passed=False,
                n_candidates=0,
                n_feedback_records=0,
                funnel_stats={},
                window_aggregates={},
                training_step_status="NOT_EXECUTED_REQUEST_CONTROL",
                anchor_binding=self.anchor_binding)

        # -- C. REVISION phase: verdicts -> ledger; proposals -> plan_k -----
        self._set_phase(window, PHASE_REVISION)
        self.apply_board_verdicts(window, board.verdicts, view=view)
        plan, revision = self.revise_plan(window, board, view=view)
        #: production path defers the single real optimizer update until the
        #: probe funnel has selected the final batch (directive ordering:
        #: probe -> select -> exactly one update); the default path keeps the
        #: historical bookkeeping call
        training = self.training_seam.execute_training_step(
            window, defer_real_update=self.launch_decision.training_allowed)
        self.training_log.append(training)

        # -- D. PROBING phase: EnvCoder -> gates -> probe -> staged fb_k ----
        self._set_phase(window, PHASE_PROBING)
        if self._real_env_coder_callable is not None:
            #: P0-2 production chain: unique template, one call per window
            #: batch, four-link verification, bounded repair — the callable
            #: itself raises RealEnvCoderBlocked on budget exhaustion (no
            #: fallback to the symbolic coder exists)
            env_artifact = self._real_env_coder_callable(
                window=window, plan_id=plan.plan_id,
                directives=list(board.directives),
                sequence=self._sequence)
            #: P0-6: a real EnvCoder window consumes EXACTLY n_calls
            #: sequence slots — the unique-template call PLUS every bounded
            #: repair re-call. The global call sequence must stay strictly
            #: monotonic: under-counting a repair sequence would silently
            #: re-use sequence numbers, so a missing/zero count fails
            #: closed BEFORE anything else consumes the sequence. The calls
            #: were made even if the artifact later fails the PASSED check,
            #: so the consumption precedes that check.
            env_n_calls = int(getattr(env_artifact, "n_calls", 0))
            if env_n_calls < 1:
                raise RuntimeError(
                    "REAL_ENVCODER_CALL_COUNT_MISSING: window="
                    f"{window} — the real EnvCoder artifact must declare "
                    "n_calls >= 1 (the exact number of LLM-family calls "
                    f"consumed), got {env_n_calls!r}")
            self._sequence += env_n_calls
            env_coder_call_count = env_n_calls
            if getattr(env_artifact, "overall_status", "") != "PASSED":
                raise RuntimeError(
                    "REAL_ENVCODER_NOT_PASSED: window="
                    f"{window} overall_status="
                    f"{getattr(env_artifact, 'overall_status', None)!r} — "
                    "a real window may not probe against an unverified "
                    "environment artifact")
            n_coded = len(getattr(env_artifact, "directive_artifacts", []))
            gate_passed = True
        else:
            env_out, env_envelope = run_env_coder(
                window=window, directives=list(board.directives),
                backend=self.backend, sequence=self._sequence)
            self._sequence += 1
            env_coder_call_count = 1      # symbolic coder: one call/window
            self.envelopes.append(env_envelope)
            gate_report = self.env_coder_gate.evaluate(
                window=window, directives=list(board.directives),
                output=env_out)
            self.env_coder_gate.assert_passed(gate_report)
            n_coded = len(env_out.coded)
            gate_passed = gate_report.passed
        staged, batch = self._probe_and_stage(window, plan, board.directives)
        if self.launch_decision.training_allowed:
            policy = self.two_window_smoke_policy
            if policy is not None and window != policy.update_window_index:
                #: P0-10: every non-update window (in particular window k,
                #: which has NO prior feedback to consume) trains nothing
                #: — Δ=0. plan_{k+1} is built from feedback_k, so the
                #: single optimizer update belongs to window k+1 only.
                training = TrainingStepRecord(
                    status="SKIPPED_SMOKE_POLICY_UPDATE_WINDOW",
                    student_training_transitions=0,
                    reason=(f"window={window}: the two-window smoke policy "
                            "allows the single real optimizer update only "
                            f"in window {policy.update_window_index} (the "
                            "window that consumes feedback_k); this window "
                            "trains nothing (delta=0)"))
            else:
                #: P0-4: window k+1's single real optimizer update over the
                #: probe-selected final batch (12 dynamic + 4 anchors),
                #: wrapped in the checkpoint save/load round-trip
                batch_plan = None
                if self.dicode_batch_binding is not None:
                    #: P0-16 (DiCode 15+1): convert the final selection into
                    #: the canonical 15+1 batch — the update consumes the 15
                    #: curriculum ids ONLY; the OriginalTask is appended once
                    #: internally by the director's DiCode runtime
                    from d052.feedback_llm_ued.dicode_batch_plan import (
                        build_dicode_batch_plan,
                    )
                    batch_plan = build_dicode_batch_plan(
                        window=window,
                        final_batch_ids=batch.final_batch,
                        anchor_ids=list(self.anchor_ids),
                        non_target_anchor_ids=(
                            self.dicode_batch_binding.non_target_anchor_ids),
                        original_task_id=(
                            self.dicode_batch_binding.original_task_id),
                        original_appended_by=(
                            self.dicode_runtime_identity
                            or self.dicode_batch_binding.original_task_id))
                training = self.training_seam.execute_real_window_update(
                    window,
                    batch_candidate_ids=(batch_plan.batch_candidate_ids
                                         if batch_plan is not None
                                         else batch.final_batch),
                    batch_plan=batch_plan)
            self.training_log.append(training)

        # -- E. ATOMIC FREEZE: feedback_k + new hypotheses + FROZEN ---------
        self._freeze_window(window, staged, board.new_hypotheses)

        agg = self._window_aggregates(window)
        return WindowRecord(
            window=window, mode=self.mode, phase=self._phases[window],
            evidence_window=evidence_window,
            feedback_view_label=view.label,
            board_call_count=C.BOARD_CALLS_PER_WINDOW,
            env_coder_call_count=env_coder_call_count,
            n_llm_calls=self.backend.usage.total_calls - n_calls_before,
            request_control=board.request_control,
            global_risk=board.critic.global_risk,
            plan_id=plan.plan_id,
            plan_signature_hash=plan_signature_hash(plan),
            revision_label=revision.label,
            n_directives=len(board.directives),
            n_coded=n_coded,
            gate_passed=gate_passed,
            n_candidates=C.RAW_CANDIDATES,
            n_feedback_records=len(staged),
            funnel_stats=dict(batch.funnel_stats),
            window_aggregates=agg,
            training_step_status=training.status,
            anchor_binding=self.anchor_binding)

    # ------------------------------------------------------- phase machine
    def phase_of(self, window: int) -> Optional[str]:
        return self._phases.get(window)

    def _set_phase(self, window: int, phase: str) -> None:
        if phase not in _PHASE_ORDER:
            raise StateMachineViolation(
                f"UNKNOWN_PHASE: {phase!r}")
        current = self._phases.get(window)
        if current is not None and \
                _PHASE_ORDER.index(current) >= _PHASE_ORDER.index(phase):
            raise StateMachineViolation(
                f"STATE_MACHINE_PHASE_REGRESSION: window {window} phase "
                f"{current!r} -> {phase!r} (phases are monotone)")
        self._phases[window] = phase

    def _assert_revision_allowed(self, window: int) -> None:
        """Revision is ONLY legal during window k's REVISION phase — before
        feedback_k is staged. After staging/freezing, the window is closed
        and only window k+1's six-role board may act on feedback_k."""
        phase = self._phases.get(window)
        if phase != PHASE_REVISION:
            raise SameWindowRevisionForbidden(
                f"SAME_WINDOW_REVISION_FORBIDDEN: window {window} is in "
                f"phase {phase!r}; verdict application / plan revision is "
                f"only legal during the {PHASE_REVISION!r} phase, before "
                f"feedback_{window} is staged. Revisions based on "
                f"feedback_{window} may only happen at window {window + 1} "
                f"through the complete six-role board explicitly citing its "
                f"feedback ids (NEXT_WINDOW_REVISION_ONLY)")

    # ------------------------------------------------------- feedback view
    def _feedback_view(self, window: int):
        """The ONLY surface through which the board touches feedback.

        CC3 C9 gate: the lag is EXACTLY one window — a window-k view
        presents the frozen records of window k-1 ONLY (older records are
        stale, current/future records do not exist yet from the board's
        point of view; all three fail closed as STALE_FEEDBACK_ID if they
        ever reach a view or a citation validator).

        static:   MaskedFeedbackView (P0-12 shape-matched no-feedback
                  control — the SAME window k-1 record set the normal mode
                  sees, every value replaced by its controlled NULL/MASK
                  value; the board runs the same computation with no
                  feedback content).
        normal:   read-only snapshot of EXACTLY window k-1.
        shuffled: PermutedFeedbackView over the SAME honest records — a
                  frozen, recomputable permutation presented under
                  anonymized ids with the identity side channels masked
                  (the store itself is never permuted).
        """
        records = [r for r in self.store.all() if r.window == window - 1]
        scope = max(0, window - 1)
        if self.mode == C.MODE_STATIC_LLM:
            return MaskedFeedbackView(records, window_scope=scope,
                                      board_window=window)
        if self.mode == C.MODE_SHUFFLED_FEEDBACK:
            return PermutedFeedbackView(
                records, window_scope=scope, board_window=window,
                mode=self.mode, seed_schedule_hash=C.SEED_SCHEDULE_HASH)
        return NormalFeedbackView(records, window_scope=scope)

    # ------------------------------------------- C10 RETIRE lifecycle state
    def _reopen_eligible(self, window: int, retired_windows: Mapping[str, int]
                         ) -> Tuple[str, ...]:
        """Families whose cooldown is over AND that are authorized to come
        back this window — either through the explicit human authorization
        (``human_reopen_families``) or because ALL distinguishing probe
        evidence postdates the retirement window (genuinely new evidence;
        an empty evidence set authorizes nothing)."""
        reopened = set()
        for fam, ret_window in retired_windows.items():
            if window - ret_window <= C.RETIRE_COOLDOWN_WINDOWS:
                continue                       # still inside the cooldown
            if fam in self.human_reopen_families:
                reopened.add(fam)
                continue
            hids = {h.hypothesis_id for h in self.ledger.all()
                    if h.environment_family == fam}
            recs = [r for r in self.store.all()
                    if hids and
                    set(r.distinguishes_hypothesis_ids) & hids]
            if recs and all(r.window > ret_window for r in recs):
                reopened.add(fam)
        return tuple(sorted(reopened))

    def _retirement_state(self, window: int
                          ) -> Tuple[List[str], List[str], Tuple[str, ...]]:
        """The window's RETIRE lifecycle partition, derived ONLY from the
        registry + store + human authorization (pure, recomputable):

        * ``families_in_cooldown`` — retired within the last
          RETIRE_COOLDOWN_WINDOWS windows: hard block;
        * ``retired_families``     — cooldown over but NOT reopened: still
          blocked (a retired family stays retired until reopened);
        * ``reopened``             — cooldown over AND authorized: behaves
          like a normal family this window.

        P0-12: every mode runs this SAME query (the compute-matched
        lifecycle contract). The static (no-feedback control) mode's
        masked view resolves NO citation, so a RETIRE verdict — which
        cites feedback — can never occur there and the retirement
        registry can never become non-empty: a non-empty registry in the
        static mode is an impossible state and fails closed; the query
        then returns the empty partition by construction (never reads the
        store).
        """
        if self.mode == C.MODE_STATIC_LLM:
            if self._retired_at:
                raise ValueError(
                    "STATIC_MODE_RETIREMENT_REGISTRY_NOT_EMPTY: a "
                    "retirement cites feedback, and the static mode's "
                    "masked view resolves no citation — the registry can "
                    "never become non-empty in mode "
                    f"{self.mode!r} (impossible state)")
            return [], [], ()
        retired_windows: Mapping[str, int] = dict(self._retired_at)
        reopened = self._reopen_eligible(window, retired_windows)
        in_cooldown = sorted(
            fam for fam, w in retired_windows.items()
            if 1 <= window - w <= C.RETIRE_COOLDOWN_WINDOWS)
        blocked_retired = sorted(
            fam for fam, w in retired_windows.items()
            if window - w > C.RETIRE_COOLDOWN_WINDOWS
            and fam not in reopened)
        return in_cooldown, blocked_retired, reopened

    # --------------------------------------- C11 REQUEST_CONTROL blocking
    def _escalation_artifact(self, window: int, board: BoardOutput, view
                             ) -> HumanDecisionArtifact:
        """Build the audit-grade HumanDecisionArtifact for a REQUEST_CONTROL
        stop.

        Tutor-proposal citations are resolved through the window's view
        FIRST (de-anonymization under the shuffled mode), so the artifact
        only ever carries REAL store ids — a human reviewer reads the same
        ids the SimulatorFeedbackStore holds. The ``board_hash`` binds the
        complete BoardOutput of the escalated window (all six role outputs,
        verdicts, directives, proposals), so every detail of the stop is
        recomputable from the frozen record.
        """
        sources: set = set()
        if board.critic.request_control:
            sources.add(C.ROLE_CRITIC_SKEPTIC)
        rc_proposals = [p for p in board.family_proposals
                        if p.decision == C.DECISION_REQUEST_CONTROL]
        if rc_proposals:
            sources.add(C.ROLE_INTERVENTION_TUTOR)
        cited = sorted({view.resolve_citation(fid)
                        for p in rc_proposals
                        for fid in p.based_on_feedback_ids})
        return HumanDecisionArtifact(
            artifact_id=f"hda-w{window:02d}-{board.board_hash[:16]}",
            window=window, mode=self.mode,
            trigger_sources=sorted(sources),
            global_risk=board.critic.global_risk,
            critic_objections=list(board.critic.objections),
            request_control_families=sorted(
                {p.environment_family for p in rc_proposals}),
            cited_feedback_ids=cited,
            board_hash=board.board_hash,
            reason=(f"window {window}: the six-role board requested human "
                    f"control (sources: {sorted(sources)}); the loop stops, "
                    f"no execution batch is produced, and nothing else is "
                    f"applied autonomously"))

    # ------------------------------------------------- revision (phase C)
    def validate_verdict_citations(self, window: int, verdicts) -> None:
        """P0-6 binding guard (defense-in-depth over the board's own
        citation validation). Pure validator — phase-independent, so the
        negative tests can exercise it directly. Fail closed on:

        * more than one verdict per hypothesis per window;
        * unknown hypothesis / unknown feedback id;
        * feedback from ANY window other than EXACTLY window k-1
          (STALE_FEEDBACK_ID — older, current and future records all fail
          closed; CC3 C9 gate);
        * duplicate citation inside one verdict;
        * record not distinguishing the verdicted hypothesis;
        * record family != hypothesis family;
        * record produced by a plan this run never generated.
        """
        if window < 0:
            raise ValueError(f"ILLEGAL_REVISION_WINDOW: {window}")
        seen_hypotheses: set = set()
        for v in verdicts:
            if v.hypothesis_id in seen_hypotheses:
                raise ValueError(
                    f"DUPLICATE_HYPOTHESIS_VERDICT: {v.hypothesis_id!r} is "
                    f"verdicted more than once in window {window}")
            seen_hypotheses.add(v.hypothesis_id)
            try:
                hyp = self.ledger.get(v.hypothesis_id)
            except KeyError:
                raise ValueError(
                    f"UNKNOWN_HYPOTHESIS_ID: verdict targets "
                    f"{v.hypothesis_id!r} which the ledger does not hold"
                ) from None
            seen_fids: set = set()
            for fid in v.cited_feedback_ids:
                if fid in seen_fids:
                    raise ValueError(
                        f"DUPLICATE_FEEDBACK_CITATION: {fid!r} cited twice "
                        f"by verdict for {v.hypothesis_id!r}")
                seen_fids.add(fid)
                try:
                    rec = self.store.get(fid)
                except KeyError:
                    raise ValueError(
                        f"UNKNOWN_FEEDBACK_ID: {fid!r} cited by verdict for "
                        f"{v.hypothesis_id!r} does not exist in the "
                        f"SimulatorFeedbackStore") from None
                if rec.window != window - 1:
                    raise ValueError(
                        f"STALE_FEEDBACK_ID: {fid!r} comes from window "
                        f"{rec.window}; a window-{window} revision may only "
                        f"cite feedback from EXACTLY window {window - 1} — "
                        f"older, current and future records all fail closed "
                        f"(CC3 C9 gate)")
                if v.hypothesis_id not in rec.distinguishes_hypothesis_ids:
                    raise ValueError(
                        f"FEEDBACK_BINDING_MISMATCH: {fid!r} does not "
                        f"distinguish hypothesis {v.hypothesis_id!r}")
                if rec.environment_family != hyp.environment_family:
                    raise ValueError(
                        f"FEEDBACK_BINDING_MISMATCH: {fid!r} family "
                        f"{rec.environment_family!r} != hypothesis family "
                        f"{hyp.environment_family!r}")
                if rec.source_plan_id not in self.plans:
                    raise ValueError(
                        f"UNKNOWN_SOURCE_PLAN: {fid!r} was produced by plan "
                        f"{rec.source_plan_id!r} which this run never "
                        f"generated")

    def _resolved_verdicts(self, window: int, verdicts, view):
        """De-anonymize the board's citations through the window's view.

        Under the shuffled mode the six roles cite ANONYMIZED feedback ids
        (that is what the PermutedFeedbackView showed them); only this
        controller-side path may map them back to store ids — fail closed on
        anything the view did not present. Under normal/static modes this is
        an identity check (the view still must hold every citation).
        """
        resolved = []
        for v in verdicts:
            dump = v.model_dump()
            dump["cited_feedback_ids"] = [
                view.resolve_citation(fid) for fid in v.cited_feedback_ids]
            resolved.append(BoardHypothesisVerdict(**dump))
        return resolved

    def apply_board_verdicts(self, window: int, verdicts, view=None) -> None:
        """Apply the board's verdicts to the ledger — ONLY in the REVISION
        phase (double-window state machine). Board citations are resolved
        through the window's FeedbackView first (de-anonymization under the
        shuffled mode); the ledger only ever records real store ids."""
        self._assert_revision_allowed(window)
        view = view if view is not None else self._feedback_view(window)
        resolved = self._resolved_verdicts(window, verdicts, view)
        self.validate_verdict_citations(window, resolved)
        for v in resolved:
            for fid in v.cited_feedback_ids:
                rec = self.store.get(fid)
                agrees = rec.expected_observed_match == \
                    C.MATCH_DIRECTION_AGREE
                self.ledger.bind_feedback(v.hypothesis_id, fid, agrees=agrees)
            self.ledger.apply_verdict(
                v.hypothesis_id, status=v.verdict, window=window,
                reason=v.reason, feedback_ids=list(v.cited_feedback_ids),
                confidence=v.new_confidence)

    def revise_plan(self, window: int, board: BoardOutput, view=None
                    ) -> Tuple[CurriculumPlan, PlanRevisionRecord]:
        """Close the board's family proposals into plan_k — ONLY in the
        REVISION phase (double-window state machine). Proposal citations are
        resolved through the window's FeedbackView first, so the Reconciler
        and the PlanRevisionRecord only ever carry real store ids."""
        self._assert_revision_allowed(window)
        view = view if view is not None else self._feedback_view(window)
        proposals = []
        for p in board.family_proposals:
            dump = p.model_dump()
            dump["based_on_feedback_ids"] = [
                view.resolve_citation(fid)
                for fid in p.based_on_feedback_ids]
            proposals.append(FamilyProposal(**dump))
        allocations = [self._proposal_to_allocation(p) for p in proposals]
        previous = self._plans_by_window.get(window - 1)
        previous_plan_id = previous.plan_id if previous else ""
        previous_slots = ({a.environment_family: a.slots
                           for a in previous.allocations} if previous else {})
        retired_windows: Mapping[str, int] = dict(self._retired_at)
        reopened = self._reopen_eligible(window, retired_windows)
        rc = self.reconciler.reconcile(
            window=window, mode=self.mode, proposals=allocations,
            known_feedback_ids=set(self.store.ids()),
            previous_plan_id=previous_plan_id,
            previous_slots=previous_slots,
            retired_windows=retired_windows,
            reopened_families=reopened)
        plan = rc.plan
        self.plans[plan.plan_id] = plan
        self._plans_by_window[window] = plan
        # C10 registry update (after the plan is fixed): fresh retirements
        # stamp the current window and WIN over a same-window reopen; a
        # reopened family that was actually funded and not re-retired leaves
        # the registry (the reopen was consumed); a reopened family that got
        # no budget stays retired (the authorization is per-window).
        for fam in plan.retired_families:
            self._retired_at[fam] = window
        funded = {a.environment_family for a in plan.allocations}
        for fam in reopened:
            if fam in funded and fam not in plan.retired_families:
                self._retired_at.pop(fam, None)
        cited = sorted({fid for m in rc.modifications
                        for fid in m["based_on_feedback_ids"]})
        revision = PlanRevisionRecord(
            revision_id=f"rev-w{window:02d}", window=window, mode=self.mode,
            previous_plan_id=previous_plan_id, new_plan_id=plan.plan_id,
            based_on_feedback_ids=cited,
            modifications=[PlanModification(**m) for m in rc.modifications],
            label=(FEEDBACK_DRIVEN_LABEL if cited else C.EXPLORATION_LABEL))
        assert_feedback_ids_known(revision, set(self.store.ids()))
        self.revisions.append(revision)
        return plan, revision

    @staticmethod
    def _proposal_to_allocation(proposal) -> dict:
        """FamilyProposal -> Reconciler allocation dict.

        The board proposes actions, the Reconciler disposes budget: core
        proposals request PROPOSAL_DEFAULT_SLOTS (the Reconciler caps and
        tops up); RETIRE / REQUEST_CONTROL request zero budget.
        """
        slots = 0 if proposal.decision in (C.DECISION_RETIRE,
                                           C.DECISION_REQUEST_CONTROL) \
            else PROPOSAL_DEFAULT_SLOTS
        reason = proposal.reason or (
            f"{proposal.decision} proposal for "
            f"{proposal.environment_family}")
        return dict(environment_family=proposal.environment_family,
                    decision=proposal.decision, slots=slots,
                    based_on_feedback_ids=list(proposal.based_on_feedback_ids),
                    reason=reason,
                    is_exploration=proposal.is_exploration)

    # -------------------------------------------- probing + staging (D)
    def _probe_and_stage(self, window: int, plan: CurriculumPlan, directives
                         ) -> Tuple[List[SimulatorFeedbackRecord], object]:
        """Probe the directive-driven candidates and STAGE feedback_k.

        Staged records are NOT visible to anything yet: they enter the store
        only in the atomic freeze (phase E). This is what makes window k's
        feedback unreadable to window k's own revision path.

        The store is HONEST in every mode (binding="normal" always): the
        shuffled-mode isolation happens at view time (PermutedFeedbackView),
        never by mutating what the probe observed.
        """
        hyp_by_family: Dict[str, List[str]] = {}
        for h in self.ledger.all():
            hyp_by_family.setdefault(h.environment_family, []).append(
                h.hypothesis_id)
        candidates = generate_candidates_from_directives(
            plan, directives=list(directives),
            hypothesis_families=hyp_by_family)
        #: P0-2 (CC3 follow-up audit): on the production path every
        #: candidate enters the funnel as a BOUND copy of its family's
        #: executable environment artifact (new candidate_hash; a family
        #: without a bound artifact fails closed as
        #: EXECUTABLE_ARTIFACT_MISSING). The symbolic runner exposes no
        #: such seam, so the default path stays byte-identical.
        bind_artifacts = getattr(
            self.runner, "bind_candidates_to_executable_artifacts", None)
        if bind_artifacts is not None:
            candidates = bind_artifacts(candidates)
        # C13: the anchor ids come from the shared-manifest seam (this
        # round: the labeled scaffold placeholder), never from a hardcode
        # inside the funnel
        batch = run_staged_funnel(candidates, self.runner, window=window,
                                  anchor_ids=self.anchor_ids)

        if self._probe_feedback_builder is not None:
            #: P0-3 production path: provenance-bound records (source
            #: window/plan, hypothesis ids, candidate hash, changed + held
            #: axes, predicted vs observed, residual, CI-sample count,
            #: Student/Reference identity hashes, seed bank, transitions);
            #: the isolation guard still re-scans every record
            staged = self._probe_feedback_builder(
                window=window, plan=plan, candidates=candidates,
                batch=batch, controller=self)
            for record in staged:
                self.isolation.assert_record_clean(
                    record.model_dump(),
                    label=f"feedback:{record.feedback_id}")
            return staged, batch

        cand_by_id = {c.candidate_id: c for c in candidates}
        fast_obs = {r["candidate_id"]: r["metrics"]
                    for r in batch.stage1_results}
        full_obs = {r["candidate_id"]: r["metrics"]
                    for r in batch.stage2_results}

        staged: List[SimulatorFeedbackRecord] = []
        for cid in sorted(fast_obs):
            cand = cand_by_id[cid]
            fid = f"fb-w{window:02d}-{cid}"
            record = SimulatorFeedbackRecord(
                feedback_id=fid, candidate_id=cid,
                candidate_hash=cand.candidate_hash,
                source_plan_id=plan.plan_id, window=window,
                environment_family=cand.environment_family,
                mutation_axes=list(cand.mutation_axes),
                axis_values=dict(cand.axis_values),
                held_constant_axes=dict(cand.held_constant_axes),
                distinguishes_hypothesis_ids=list(
                    cand.distinguishes_hypothesis_ids),
                stage1_metrics=ProbeMetrics(**fast_obs[cid]),
                stage2_metrics=(ProbeMetrics(**full_obs[cid])
                                if cid in full_obs else None),
                expected_signature=self._expected_signature(cand),
                provenance=dict(
                    source=C.SOURCE_CANDIDATE_PROBE,
                    plan_id=plan.plan_id, window=window,
                    runner_id=self.runner.runner_id,
                    real_adapter_status=C.REAL_SIMULATOR_PROBE_STATUS,
                    #: the store always records the HONEST binding; the
                    #: shuffled mode permutes only the board's VIEW of these
                    #: records (PermutedFeedbackView), never the records
                    binding="normal"),
                student_identity_hash=self.student_binding.identity_hash,
                student_parameter_tree_hash=(
                    self.student_binding.parameter_tree_hash),
                student_checkpoint_step=(
                    self.student_binding.checkpoint_global_step),
                student_roles=(C.STUDENT_ROLE_SEARCH,))
            self.isolation.assert_record_clean(
                record.model_dump(), label=f"feedback:{fid}")
            staged.append(record)
        return staged, batch

    def _expected_signature(self, cand) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for hid in sorted(cand.distinguishes_hypothesis_ids):
            try:
                hyp = self.ledger.get(hid)
            except KeyError:
                continue
            for k, v in hyp.predicted_signature.items():
                merged.setdefault(k, float(v))
        return merged

    # -------------------------------------------------- atomic freeze (E)
    def _freeze_window(self, window: int,
                       staged: List[SimulatorFeedbackRecord],
                       new_hypotheses) -> None:
        """Atomic write + FREEZE of feedback_k.

        Order inside the freeze: (1) every staged record enters the store,
        (2) every record is graded by the comparator (bind_match re-stamps
        the record hash), (3) the board's new PENDING hypotheses are
        registered in the ledger, (4) the window is marked FROZEN. New
        hypotheses are registered AFTER the feedback write because
        feedback_k cannot distinguish them (the candidates were generated
        before they existed). After this call the window is closed: any
        verdict application or plan change raises
        SAME_WINDOW_REVISION_FORBIDDEN.
        """
        for record in staged:
            self.store.add(record)
        for record in staged:
            self.comparator.grade_record(self.store, record.feedback_id)
        for proposal in new_hypotheses:
            self.ledger.register(HypothesisRecord(
                hypothesis_id=proposal.hypothesis_id,
                source_window=window,
                target_behavior=proposal.target_behavior,
                evidence_ids=[],
                predicted_signature=dict(proposal.predicted_signature),
                environment_family=proposal.environment_family,
                confidence=proposal.initial_confidence,
                status=C.HYPOTHESIS_PENDING))
        self._window_feedback[window] = [r.feedback_id for r in staged]
        self._set_phase(window, PHASE_FROZEN)

    # ------------------------------------------------------------- metrics
    def _window_aggregates(self, window: int) -> Dict[str, float]:
        fids = self._window_feedback.get(window, [])
        if not fids:
            return {}
        fronts, rets, acts, learns = [], [], [], []
        for fid in fids:
            rec = self.store.get(fid)
            m = rec.stage2_metrics or rec.stage1_metrics
            if m is None:
                continue
            fronts.append(m.student_front_progress)
            rets.append(m.global_retention)
            acts.append(m.student_behavior_activation)
            learns.append(m.learnability)
        if not fronts:
            return {}
        return dict(
            front_progress=round(sum(fronts) / len(fronts), 6),
            global_retention=round(sum(rets) / len(rets), 6),
            behavior_activation=round(sum(acts) / len(acts), 6),
            learnability=round(sum(learns) / len(learns), 6))

    def _build_summary(self, records: List[WindowRecord],
                       stopped_window: Optional[int] = None) -> RunSummary:
        n_windows = len(records)
        n_revisions = len(self.revisions)
        decision_dist: Dict[str, int] = {}
        cited_records: set = set()
        for rev in self.revisions:
            for m in rev.modifications:
                decision_dist[m.decision] = decision_dist.get(m.decision, 0) + 1
                cited_records.update(m.based_on_feedback_ids)
        all_feedback = set(self.store.ids())
        citation_coverage = (len(cited_records & all_feedback)
                             / len(all_feedback)) if all_feedback else 0.0

        # supported retention / refuted retirement rates
        supported = refuted = supported_retained = refuted_retired = 0
        for rev in self.revisions:
            keep = {m.environment_family: m.decision
                    for m in rev.modifications}
            window_verdicts = [v for v in
                               self._verdicts_by_window().get(rev.window, [])]
            for hid, verdict in window_verdicts:
                try:
                    fam = self.ledger.get(hid).environment_family
                except KeyError:
                    continue
                decision = keep.get(fam)
                if verdict == C.HYPOTHESIS_SUPPORTED:
                    supported += 1
                    if decision in (C.DECISION_RETAIN,
                                    C.DECISION_EXPAND_BUDGET):
                        supported_retained += 1
                elif verdict == C.HYPOTHESIS_REFUTED:
                    refuted += 1
                    if decision == C.DECISION_RETIRE:
                        refuted_retired += 1

        transitions = self.runner.total_transitions
        useful = sum(int(rec.funnel_stats.get("dynamic_selected", 0))
                     for rec in records)
        # C11: the final-batch verdict — a REQUEST_CONTROL-stopped loop can
        # NEVER ship a final batch (and this round never trains anyway).
        final_batch = self.launch_gate.evaluate_final_batch(
            loop_completed=(stopped_window is None),
            request_control_stopped=(stopped_window is not None))
        return RunSummary(
            mode=self.mode,
            n_windows=n_windows,
            n_llm_calls=self.backend.usage.total_calls,
            revision_rate=round(n_revisions / n_windows, 4) if n_windows else 0.0,
            decision_distribution=decision_dist,
            feedback_citation_coverage=round(citation_coverage, 4),
            supported_retention_rate=(round(supported_retained / supported, 4)
                                      if supported else 0.0),
            refuted_retirement_rate=(round(refuted_retired / refuted, 4)
                                     if refuted else 0.0),
            total_simulator_transitions=transitions,
            transitions_per_useful_environment=(
                round(transitions / useful, 2) if useful else 0.0),
            plan_signature_hashes=[r.plan_signature_hash for r in records],
            windows=[r.to_dict() for r in records],
            request_control_stopped=stopped_window is not None,
            stopped_window=stopped_window,
            human_decision_artifact=(
                self.human_decision_artifacts[-1].model_dump()
                if self.human_decision_artifacts else None),
            final_batch=asdict(final_batch))

    def _verdicts_by_window(self) -> Dict[int, List[Tuple[str, str]]]:
        out: Dict[int, List[Tuple[str, str]]] = {}
        for h in self.ledger.all():
            for entry in h.revision_history:
                out.setdefault(int(entry["window"]), []).append(
                    (h.hypothesis_id, str(entry["new_status"])))
        return out

    # ------------------------------------------------------------ comparison
    @staticmethod
    def compare_summaries(normal: RunSummary, shuffled: RunSummary,
                          static: Optional[RunSummary] = None) -> dict:
        n = min(normal.n_windows, shuffled.n_windows)
        diffs = sum(1 for i in range(n)
                    if normal.plan_signature_hashes[i]
                    != shuffled.plan_signature_hashes[i])
        result = dict(
            mode_pair=("normal_feedback", "shuffled_feedback"),
            windows_compared=n,
            plan_difference_windows=diffs,
            plan_identical_windows=n - diffs,
            normal_revision_rate=normal.revision_rate,
            shuffled_revision_rate=shuffled.revision_rate,
            normal_llm_calls=normal.n_llm_calls,
            shuffled_llm_calls=shuffled.n_llm_calls,
            normal_decision_distribution=normal.decision_distribution,
            shuffled_decision_distribution=shuffled.decision_distribution,
            feedback_binding_matters=(diffs > 0))
        if static is not None:
            result["static_llm_calls"] = static.n_llm_calls
            result["static_revision_rate"] = static.revision_rate
            result["static_plan_difference_vs_normal"] = sum(
                1 for i in range(min(n, static.n_windows))
                if static.plan_signature_hashes[i]
                != normal.plan_signature_hashes[i])
        return result
