"""FeedbackUEDController — closes the scientific loop (task §1/§5).

Per window k:

    plan_k -> candidate generation -> simulator probe -> expected-vs-observed
    comparison -> (gate) LLM feedback diagnosis -> RETAIN/MUTATE/RETIRE/
    REQUEST_CONTROL -> plan_{k+1}

Three comparison modes (§5):

* ``static_llm``       — the bootstrap initial plan is kept forever; feedback
                         is never read (the generate-then-accept baseline);
* ``normal_feedback``  — the full loop with CORRECT candidate<->feedback
                         binding;
* ``shuffled_feedback``— identical, except the candidate<->feedback binding is
                         deterministically rotated (candidate identities and
                         hypothesis bindings stay; observed metric payloads
                         rotate), proving the plan really is a function of the
                         feedback and not of window order.

All three modes share the SAME deterministic bootstrap plan and hypothesis
seeds, so any plan divergence is attributable solely to feedback use.

Honesty posture re-asserted at construction: every real-world authorization
flag must be False this round; the loop runs on the deterministic mock LLM
backend + the deterministic symbolic probe runner and says so in every record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import (
    adaptive_designer,
    adversarial_reviewer,
    feedback_diagnostician,
)
from d052.feedback_llm_ued.deterministic_reconciler import (
    DeterministicReconciler,
)
from d052.feedback_llm_ued.environment_generator import generate_candidates
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_MOCK_DRY_RUN,
    FeedbackLaunchGate,
)
from d052.feedback_llm_ued.expected_observed import ExpectedObservedComparator
from d052.feedback_llm_ued.feedback_contracts import (
    CurriculumPlan,
    FamilyAllocation,
    ProbeMetrics,
    plan_signature_hash,
)
from d052.feedback_llm_ued.feedback_invocation_gate import (
    GateInput,
    evaluate_gate,
)
from d052.feedback_llm_ued.formal_isolation import FormalSourceIsolationGuard
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
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.simulator_probe import (
    DeterministicSymbolicProbeRunner,
    run_staged_funnel,
)

#: deterministic bootstrap: first four families, seeded hypotheses + plan
_BOOTSTRAP_FAMILIES = C.ENVIRONMENT_FAMILIES[:4]
BOOTSTRAP_PLAN_ID = "plan-0000-bootstrap-v1"


@dataclass
class WindowRecord:
    window: int
    mode: str
    gate_conditions: List[str]
    invoked_llm: bool
    n_llm_calls: int
    reviewer_invoked: bool
    risk_triggers: List[str]
    reused_previous_plan: bool
    plan_id: str
    plan_signature_hash: str
    revision_label: str
    n_candidates: int = 0
    n_feedback_records: int = 0
    funnel_stats: Dict[str, int] = field(default_factory=dict)
    window_aggregates: Dict[str, float] = field(default_factory=dict)

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
    """Deterministic, replayable driver of the feedback-adaptive loop."""

    def __init__(self, mode: str, *, backend=None, probe_runner=None) -> None:
        if mode not in C.FEEDBACK_MODES:
            raise ValueError(f"UNKNOWN_MODE: {mode!r}")
        _assert_authorization_posture()
        self.mode = mode
        self.launch_gate = FeedbackLaunchGate(EXECUTION_MODE_MOCK_DRY_RUN)
        self.launch_decision = self.launch_gate.evaluate()
        self.backend = backend or DeterministicMockFeedbackBackend()
        self.launch_gate.assert_backend_allowed(self.backend.kind)
        self.runner = probe_runner or DeterministicSymbolicProbeRunner()
        self.ledger = HypothesisLedger()
        self.store = SimulatorFeedbackStore()
        self.comparator = ExpectedObservedComparator()
        self.reconciler = DeterministicReconciler()
        self.isolation = FormalSourceIsolationGuard()
        self.revisions: List[PlanRevisionRecord] = []
        self.envelopes: List[object] = []
        self.plans: Dict[str, CurriculumPlan] = {}
        self._window_feedback: Dict[int, List[str]] = {}
        self._sequence = 0
        self._summary: Optional[RunSummary] = None

    # ------------------------------------------------------------------ seeds
    def _seed(self) -> None:
        """Deterministic hypothesis seeds + the shared bootstrap plan."""
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
        allocations = [FamilyAllocation(
            environment_family=family, slots=3, decision=C.DECISION_MUTATE,
            reason="bootstrap exploration (no feedback exists yet)",
            is_exploration=True) for family in _BOOTSTRAP_FAMILIES]
        plan = CurriculumPlan(
            plan_id=BOOTSTRAP_PLAN_ID, window=0, mode=self.mode,
            allocations=allocations, explored_families=list(_BOOTSTRAP_FAMILIES))
        self.plans[plan.plan_id] = plan
        self.revisions.append(PlanRevisionRecord(
            revision_id="rev-w00-bootstrap", window=0, mode=self.mode,
            new_plan_id=plan.plan_id, modifications=[],
            label=C.EXPLORATION_LABEL))

    # ------------------------------------------------------------------ loop
    def run(self, max_windows: int = C.MAX_WINDOWS) -> RunSummary:
        self._seed()
        records: List[WindowRecord] = []
        prev_plan = self.plans[BOOTSTRAP_PLAN_ID]
        state = dict(front_stalled=0, windows_without_improvement=0,
                     prev_front=None, prev_retention=None, prev_activation=None,
                     prev_learnability=None, plan_age=0,
                     prev_dynamic_selected=C.DYNAMIC_UED_SLOTS)
        for window in range(max_windows):
            if window == 0:
                wrec = self._window_bootstrap(window, prev_plan, state)
            else:
                wrec, prev_plan = self._window_adaptive(window, prev_plan, state)
            records.append(wrec)
        self._summary = self._build_summary(records)
        assert_no_real_llm_usage(self.backend.usage)
        return self._summary

    # ------------------------------------------------------------- window 0
    def _window_bootstrap(self, window: int, plan: CurriculumPlan,
                          state: dict) -> WindowRecord:
        n_fb = self._generate_probe_bind(window, plan)
        agg = self._window_aggregates(window)
        state.update(prev_front=agg.get("front_progress"),
                     prev_retention=agg.get("global_retention"),
                     prev_activation=agg.get("behavior_activation"),
                     prev_learnability=agg.get("learnability"),
                     plan_age=0)
        return WindowRecord(
            window=window, mode=self.mode, gate_conditions=[],
            invoked_llm=False, n_llm_calls=0, reviewer_invoked=False,
            risk_triggers=[], reused_previous_plan=False, plan_id=plan.plan_id,
            plan_signature_hash=plan_signature_hash(plan),
            revision_label=C.EXPLORATION_LABEL,
            n_candidates=C.RAW_CANDIDATES, n_feedback_records=n_fb,
            funnel_stats=dict(self._last_funnel_stats),
            window_aggregates=agg)

    # ----------------------------------------------------------- window >=1
    def _window_adaptive(self, window: int, prev_plan: CurriculumPlan,
                         state: dict) -> Tuple[WindowRecord, CurriculumPlan]:
        agg_prev = self._window_aggregates(window - 1)
        front = agg_prev.get("front_progress")
        if state["prev_front"] is not None and front is not None:
            if abs(front - state["prev_front"]) < 0.02:
                state["front_stalled"] += 1
            else:
                state["front_stalled"] = 0
        retention_delta = 0.0
        if state["prev_retention"] is not None and \
                agg_prev.get("global_retention") is not None:
            retention_delta = (agg_prev["global_retention"]
                               - state["prev_retention"])
        activation_change = 0.0
        if state["prev_activation"] is not None and \
                agg_prev.get("behavior_activation") is not None:
            activation_change = abs(agg_prev["behavior_activation"]
                                    - state["prev_activation"])
        learn = agg_prev.get("learnability")
        if state["prev_learnability"] is not None and learn is not None:
            if learn <= state["prev_learnability"] + 0.01:
                state["windows_without_improvement"] += 1
            else:
                state["windows_without_improvement"] = 0

        has_prior_diagnosis = any(
            e.role == feedback_diagnostician.ROLE for e in self.envelopes)
        gate_input = GateInput(
            window=window,
            has_prior_diagnosis=has_prior_diagnosis,
            core_behavior_rate_change=activation_change,
            front_stalled_windows=state["front_stalled"],
            global_retention_delta=retention_delta,
            previous_plan_exhausted=(
                state["prev_dynamic_selected"] < C.STAGE2_KEEP),
            valid_candidate_count=self._last_stage1_survivors,
            required_candidate_count=C.STAGE1_KEEP,
            cached_plan_age_windows=state["plan_age"])
        gate = evaluate_gate(gate_input)

        invoke = bool(gate["invoke_llm"])
        if self.mode == C.MODE_STATIC_LLM:
            invoke = False                     # baseline never reads feedback

        n_calls_before = self.backend.usage.total_calls
        if not invoke:
            state["plan_age"] += 1
            agg = self._window_aggregates(window - 1)
            wrec = WindowRecord(
                window=window, mode=self.mode,
                gate_conditions=list(gate["conditions"]), invoked_llm=False,
                n_llm_calls=0, reviewer_invoked=False, risk_triggers=[],
                reused_previous_plan=True, plan_id=prev_plan.plan_id,
                plan_signature_hash=plan_signature_hash(prev_plan),
                revision_label="REUSED",
                window_aggregates=agg)
            return wrec, prev_plan

        # ---- LLM path: diagnose -> design -> (review) -> reconcile --------
        diagnosis = self._diagnose(window)
        self._apply_verdicts(window, diagnosis)
        allocations, designer_out = self._design(window, diagnosis)
        risk_ctx = self._risk_context(window, diagnosis, allocations, state)
        triggers = adversarial_reviewer.evaluate_risk_triggers(risk_ctx)
        reviewer_invoked = bool(triggers)
        if reviewer_invoked:
            allocations = self._review(window, allocations, risk_ctx, triggers)
        plan, revision = self._reconcile(window, prev_plan, allocations)
        state["plan_age"] = 0

        n_fb = self._generate_probe_bind(window, plan)
        agg = self._window_aggregates(window)
        state.update(prev_front=agg.get("front_progress"),
                     prev_retention=agg.get("global_retention"),
                     prev_activation=agg.get("behavior_activation"),
                     prev_learnability=agg.get("learnability"),
                     prev_dynamic_selected=self._last_dynamic_selected)
        wrec = WindowRecord(
            window=window, mode=self.mode,
            gate_conditions=list(gate["conditions"]), invoked_llm=True,
            n_llm_calls=self.backend.usage.total_calls - n_calls_before,
            reviewer_invoked=reviewer_invoked, risk_triggers=triggers,
            reused_previous_plan=False, plan_id=plan.plan_id,
            plan_signature_hash=plan_signature_hash(plan),
            revision_label=revision.label,
            n_candidates=C.RAW_CANDIDATES, n_feedback_records=n_fb,
            funnel_stats=dict(self._last_funnel_stats),
            window_aggregates=agg)
        return wrec, plan

    # ------------------------------------------------------------ LLM steps
    def _diagnose(self, window: int):
        fb_ids = self._window_feedback.get(window - 1, [])
        feedback_ctx = []
        for fid in fb_ids:
            rec = self.store.get(fid)
            feedback_ctx.append(dict(
                feedback_id=rec.feedback_id,
                candidate_id=rec.candidate_id,
                distinguishes_hypothesis_ids=rec.distinguishes_hypothesis_ids,
                expected_observed_match=rec.expected_observed_match))
        context = dict(
            window=window,
            hypotheses=[dict(hypothesis_id=h.hypothesis_id,
                             target_behavior=h.target_behavior,
                             environment_family=h.environment_family,
                             confidence=h.confidence, status=h.status)
                        for h in self.ledger.all()],
            feedback=feedback_ctx)
        env = feedback_diagnostician.run(context, self.backend, window=window,
                                         sequence=self._sequence)
        self._sequence += 1
        self.envelopes.append(env)
        return feedback_diagnostician.DiagnosisOutput(**env.parsed_json)

    def _apply_verdicts(self, window: int, diagnosis) -> None:
        for v in diagnosis.hypothesis_verdicts:
            try:
                self.ledger.get(v.hypothesis_id)
            except KeyError:
                continue                       # unknown hypothesis: fail-safe
            for fid in v.feedback_ids:
                rec = self.store.get(fid)
                agrees = rec.expected_observed_match == C.MATCH_DIRECTION_AGREE
                self.ledger.bind_feedback(v.hypothesis_id, fid, agrees=agrees)
            self.ledger.apply_verdict(
                v.hypothesis_id, status=v.verdict, window=window,
                reason=v.reason, feedback_ids=list(v.feedback_ids),
                confidence=v.new_confidence)

    def _design(self, window: int, diagnosis):
        context = dict(
            window=window,
            verdicts=[v.model_dump() for v in diagnosis.hypothesis_verdicts],
            hypotheses=[dict(hypothesis_id=h.hypothesis_id,
                             environment_family=h.environment_family)
                        for h in self.ledger.all()],
            budget=C.DYNAMIC_UED_SLOTS,
            global_risk=diagnosis.global_risk)
        env = adaptive_designer.run(context, self.backend, window=window,
                                    sequence=self._sequence)
        self._sequence += 1
        self.envelopes.append(env)
        out = adaptive_designer.DesignerOutput(**env.parsed_json)
        allocations = [a.model_dump() for a in out.allocations]
        if out.request_control:
            cited = [fid for v in diagnosis.hypothesis_verdicts
                     for fid in v.feedback_ids]
            if cited:
                allocations.append(dict(
                    environment_family=C.ENVIRONMENT_FAMILIES[0],
                    decision=C.DECISION_REQUEST_CONTROL, slots=0,
                    based_on_feedback_ids=sorted(set(cited))[:1],
                    reason="escalation: global risk HIGH",
                    is_exploration=False))
        return allocations, out

    def _risk_context(self, window: int, diagnosis, allocations,
                      state: dict) -> dict:
        prev_ids = self._window_feedback.get(window - 1, [])
        opposite = sum(1 for fid in prev_ids
                       if self.store.get(fid).expected_observed_match
                       == C.MATCH_DIRECTION_OPPOSITE)
        return dict(
            window=window,
            overall_confidence=diagnosis.overall_confidence,
            global_risk=diagnosis.global_risk,
            allocations=allocations,
            windows_without_improvement=state["windows_without_improvement"],
            opposite_probe_count=opposite,
            reject_rate=self._last_reject_rate,
            preparing_formal_run=False)

    def _review(self, window: int, allocations: List[dict], risk_ctx: dict,
                triggers: List[str]) -> List[dict]:
        context = dict(risk_ctx, triggered_by=triggers,
                       verdicts=[], hypotheses=[])
        env = adversarial_reviewer.run(context, self.backend, window=window,
                                       sequence=self._sequence)
        self._sequence += 1
        self.envelopes.append(env)
        out = adversarial_reviewer.ReviewerOutput(**env.parsed_json)
        if not out.forced_retire_families:
            return allocations
        forced = set(out.forced_retire_families)
        fixed: List[dict] = []
        for a in allocations:
            if a["environment_family"] in forced and \
                    a["decision"] != C.DECISION_RETIRE:
                if a["based_on_feedback_ids"]:
                    fixed.append(dict(
                        a, decision=C.DECISION_RETIRE, slots=0,
                        reason="forced retirement by adversarial reviewer: "
                               + a["reason"]))
                # uncited (exploration) allocations are simply dropped
                continue
            fixed.append(a)
        return fixed

    def _reconcile(self, window: int, prev_plan: CurriculumPlan,
                   allocations: List[dict]):
        previous_slots = {a.environment_family: a.slots
                          for a in prev_plan.allocations}
        rc = self.reconciler.reconcile(
            window=window, mode=self.mode, proposals=allocations,
            known_feedback_ids=set(self.store.ids()),
            previous_plan_id=prev_plan.plan_id,
            previous_slots=previous_slots)
        plan = rc.plan
        self.plans[plan.plan_id] = plan
        cited = sorted({fid for m in rc.modifications
                        for fid in m["based_on_feedback_ids"]})
        revision = PlanRevisionRecord(
            revision_id=f"rev-w{window:02d}", window=window, mode=self.mode,
            previous_plan_id=prev_plan.plan_id, new_plan_id=plan.plan_id,
            based_on_feedback_ids=cited,
            modifications=[PlanModification(**m) for m in rc.modifications],
            label=(FEEDBACK_DRIVEN_LABEL if cited else C.EXPLORATION_LABEL))
        assert_feedback_ids_known(revision, set(self.store.ids()))
        self.revisions.append(revision)
        return plan, revision

    # ------------------------------------------------- generation + probing
    def _generate_probe_bind(self, window: int,
                             plan: CurriculumPlan) -> int:
        hyp_by_family: Dict[str, List[str]] = {}
        for h in self.ledger.all():
            hyp_by_family.setdefault(h.environment_family, []).append(
                h.hypothesis_id)
        for fam in hyp_by_family:
            hyp_by_family[fam].sort()
        candidates = generate_candidates(plan, hypothesis_families=hyp_by_family)
        batch = run_staged_funnel(candidates, self.runner, window=window)
        self._last_funnel_stats = batch.funnel_stats
        self._last_stage1_survivors = len(batch.stage1_survivors)
        self._last_dynamic_selected = len(batch.dynamic_selected)
        n_probed = len(batch.stage1_results)
        n_rejected_static = len(batch.static_rejects)
        self._last_reject_rate = (
            (n_probed - sum(1 for r in batch.stage1_results
                            if r["action"] == "accept")
             + n_rejected_static) / max(1, n_probed + n_rejected_static))

        cand_by_id = {c.candidate_id: c for c in candidates}
        fast_obs = {r["candidate_id"]: r["metrics"]
                    for r in batch.stage1_results}
        full_obs = {r["candidate_id"]: r["metrics"]
                    for r in batch.stage2_results}
        if self.mode == C.MODE_SHUFFLED_FEEDBACK:
            fast_obs = self._rotate_binding(fast_obs)
            full_obs = self._rotate_binding(full_obs)

        created = 0
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
                    binding=("shuffled" if self.mode
                             == C.MODE_SHUFFLED_FEEDBACK else "normal")))
            self.isolation.assert_record_clean(
                record.model_dump(), label=f"feedback:{fid}")
            self.store.add(record)
            self.comparator.grade_record(self.store, fid)
            created += 1
        self._window_feedback[window] = [f"fb-w{window:02d}-{cid}"
                                         for cid in sorted(fast_obs)]
        return created

    @staticmethod
    def _rotate_binding(obs: Dict[str, dict]) -> Dict[str, dict]:
        """Deterministic candidate<->feedback rotation for shuffled mode."""
        ids = sorted(obs)
        n = len(ids)
        if n <= 1:
            return dict(obs)
        shift = n // 2
        values = [obs[cid] for cid in ids]
        rotated = values[shift:] + values[:shift]
        return {cid: rotated[i] for i, cid in enumerate(ids)}

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

    def _build_summary(self, records: List[WindowRecord]) -> RunSummary:
        n_windows = len(records)
        n_revisions = sum(1 for r in self.revisions
                          if r.revision_id != "rev-w00-bootstrap")
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
            if rev.revision_id == "rev-w00-bootstrap":
                continue
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
            windows=[r.to_dict() for r in records])

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
