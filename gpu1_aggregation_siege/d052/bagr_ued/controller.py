"""BAGRUEdController: the full closed loop, dry run (task sections 0/1/16).

Wires the whole chain for ONE review window:

    raw rollout -> TrainingTrajectoryEvidenceAdapter (leakage-guarded)
                -> DeterministicEventExtractor (8 plugin detectors)
                -> BehaviorClipSelector
                -> ReviewBoard (StudentModeler -> BehaviorAuditor ->
                   CausalFailureAnalyst -> InterventionTutor -> Explorer ->
                   Critic/Skeptic; supervision-guarded per role)
                -> ReviewBoardReconciler (rule-based, bound provenance)
                -> CounterfactualEnvironmentBuilder (control + single-axis
                   + capped factorial)
                -> GlobalTaskParamsProposer (MOCK adapter; real = BLOCKED)
                -> LegalityGate
                -> front_regret + global_regret + behavioral_gap (SEPARATE)
                -> learnability + learning_progress + diversity
                -> Soft Copeland (>=8 inputs, alpha split visible)
                -> BudgetAllocator (12 UED + 4 global anchors)
                -> ProposalArchive refresh (DRY RUN only)

Authorization hard-asserted at construction:
    TRAINING_AUTHORIZED=false / FORMAL_EVALUATION_AUTHORIZED=false /
    REAL_LLM_CALLS_AUTHORIZED=false / REAL_TASKPARAMS_ADAPTER=BLOCKED.
The controller ends by asserting real_llm_calls==0 and re-running the
supervision guard over the WHOLE result (fail-closed).
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from d052.bagr_ued import constants as C
from d052.bagr_ued.archive import ProposalArchive
from d052.bagr_ued.batch_planner import BatchPlan, BatchPlanner
from d052.bagr_ued.behavior_clip_selector import BehaviorClipSelector
from d052.bagr_ued.behavioral_gap import (
    BehavioralGapScore,
    behavior_failure_score,
    compute_behavioral_gaps,
)
from d052.bagr_ued.budget_allocator import BudgetAllocator, BudgetPlan
from d052.bagr_ued.counterfactual_environment import (
    CounterfactualEnvironmentBuilder,
    CounterfactualPlan,
)
from d052.bagr_ued.diversity import compute_diversity
from d052.bagr_ued.environment_proposer import GlobalTaskParamsProposer
from d052.bagr_ued.event_extractor import DeterministicEventExtractor
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.legality_gate import LegalityGate
from d052.bagr_ued.learnability import compute_learnability
from d052.bagr_ued.learning_progress import compute_learning_progress
from d052.bagr_ued.mock_llm_backend import DeterministicMockBackend
from d052.bagr_ued.proposal_distribution import ProposalDistribution
from d052.bagr_ued.regret_scorer import RegretEvidence, combined_regret_scores
from d052.bagr_ued.review_board import ReviewBoard
from d052.bagr_ued.review_reconciler import ReviewBoardReconciler
from d052.bagr_ued.soft_copeland import (
    EnvironmentScoreBundle,
    soft_copeland_rank,
)
from d052.bagr_ued.synthetic_traces import (
    CF_ENV_ID,
    TEST_VOCABULARY,
    build_mock_failure_history,
    build_mock_global_retention,
    build_mock_reference_failure_scores,
    build_mock_regret_evidences,
    build_mock_student_success_rates,
)
from d052.bagr_ued.training_trace_adapter import TrainingTrajectoryEvidenceAdapter
from d052.bagr_ued.trajectory_evidence import (
    EvidenceSource,
    MockSymbolicAdapter,
    TrajectoryEvidenceBundle,
)
from d052.bagr_ued.trajectory_supervision_guard import TrajectorySupervisionGuard
from d052.schemas.common import CanonicalModel

ALPHA_FRONT = 0.5   # documented dry-run weighting; visible inside Soft Copeland


def assert_round_authorization() -> None:
    """Hard refusal to operate under any flipped authorization flag."""
    assert C.TRAINING_AUTHORIZED is False, "TRAINING_AUTHORIZED must be false"
    assert C.FORMAL_EVALUATION_AUTHORIZED is False, \
        "FORMAL_EVALUATION_AUTHORIZED must be false"
    assert C.REAL_LLM_CALLS_AUTHORIZED is False, \
        "REAL_LLM_CALLS_AUTHORIZED must be false"
    assert C.REAL_TASKPARAMS_ADAPTER == "BLOCKED_EXTERNAL_DEPENDENCY", \
        "REAL_TASKPARAMS_ADAPTER must stay BLOCKED_EXTERNAL_DEPENDENCY"
    assert C.TIER3_ONLY_TRAINING is False, "TIER3_ONLY_TRAINING must be false"
    assert C.TRAINING_SCOPE == "GLOBAL", "TRAINING_SCOPE must be GLOBAL"
    assert C.REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE == "PENDING"
    assert C.REAL_CANONICAL_CRITIC_SELECTION_POLICY == "PENDING"


def assert_alpha_front_bounds(alpha_front: float = ALPHA_FRONT,
                              alpha_min: float = C.ALPHA_FRONT_MIN,
                              alpha_max: float = C.ALPHA_FRONT_MAX) -> None:
    """CC1 audit fix1 (§8): alpha_front MUST be structurally < 1.

    Three runtime invariants, checked on every dry run before scoring:
      1. 0 <= alpha_front < 1            (strict upper bound),
      2. 0 <= alpha_min <= alpha_max < 1 (auditable bound window),
      3. 1 - alpha_front > 0             (global component always nonzero).
    The schema (EnvironmentScoreBundle) already refuses alpha_front == 1;
    this assert is the second, independent layer.
    """
    assert 0.0 <= alpha_front < 1.0, (
        f"ALPHA_FRONT_OUT_OF_BOUNDS: alpha_front={alpha_front!r} must "
        f"satisfy 0 <= alpha_front < 1 (global component weight "
        f"1-alpha_front must be strictly positive)")
    assert 0.0 <= alpha_min <= alpha_max < 1.0, (
        f"ALPHA_WINDOW_OUT_OF_BOUNDS: require 0 <= alpha_min ({alpha_min}) "
        f"<= alpha_max ({alpha_max}) < 1")
    assert (1.0 - alpha_front) > 0.0, (
        "GLOBAL_COMPONENT_ZERO: 1 - alpha_front must be > 0 so the global "
        "regret component can never vanish")


#: fixed global anchor signatures (mock, deterministic) for diversity baseline
ANCHOR_SIGNATURES = [
    {axis: ("low" if (i + j) % 2 == 0 else "high")
     for j, axis in enumerate(sorted(C.MUTATION_AXES))}
    for i in range(C.GLOBAL_CANONICAL_ANCHORS)
]


class DryRunResult(CanonicalModel):
    bundle: TrajectoryEvidenceBundle
    detector_manifest: List[dict] = Field(default_factory=list)
    anomalies: List[dict] = Field(default_factory=list)
    clips: List[dict] = Field(default_factory=list)
    clips_dropped: int = 0
    board: dict = Field(default_factory=dict)
    reconciliation: dict = Field(default_factory=dict)
    counterfactual_plan: dict = Field(default_factory=dict)
    descriptors: List[dict] = Field(default_factory=list)
    rejected_descriptors: List[dict] = Field(default_factory=list)
    proposal_distribution_hash: str = ""
    regret_scores: dict = Field(default_factory=dict)
    behavioral_gaps: List[dict] = Field(default_factory=list)
    learnability: List[dict] = Field(default_factory=list)
    learning_progress: List[dict] = Field(default_factory=list)
    diversity: List[dict] = Field(default_factory=list)
    copeland_ranking: dict = Field(default_factory=dict)
    budget_plan: dict = Field(default_factory=dict)
    archive_refresh_plan: dict = Field(default_factory=dict)
    batch_plan: dict = Field(default_factory=dict)
    ued_nature_assertions: dict = Field(default_factory=dict)
    #: CC1 audit fix1 (§3): the final batch/launch decision hard gate.
    #: {batch_plan_ready, training_launch_authorized, launch_block_reasons,
    #:  shortfall} — both booleans are false unless EVERY structural
    #: condition holds at once.
    launch_gate: dict = Field(default_factory=dict)
    dry_run_certificate: dict = Field(default_factory=dict)


class BAGRUEdController:
    def __init__(self) -> None:
        assert_round_authorization()
        self.backend = DeterministicMockBackend()
        self.adapter = TrainingTrajectoryEvidenceAdapter(
            MockSymbolicAdapter(TEST_VOCABULARY))
        self.extractor = DeterministicEventExtractor()
        self.clip_selector = BehaviorClipSelector()
        self.board = ReviewBoard(self.backend)
        self.reconciler = ReviewBoardReconciler()
        self.cf_builder = CounterfactualEnvironmentBuilder()
        self.proposer = GlobalTaskParamsProposer()
        self.legality = LegalityGate()
        self.budget = BudgetAllocator()
        self.archive = ProposalArchive()
        self.supervision_guard = TrajectorySupervisionGuard()

    # ------------------------------------------------------------------
    def run_dry_run(self, raw_rollout: dict, *,
                    bundle_id: str = "synthetic_unsafe_rest_window",
                    total_updates: int = 8) -> DryRunResult:
        assert_round_authorization()
        assert_alpha_front_bounds()

        # 1-3. evidence intake + extraction + clips
        bundle = self.adapter.adapt(
            raw_rollout, bundle_id=bundle_id,
            source=EvidenceSource.SYNTHETIC_TEST_TRACE)
        anomalies = self.extractor.extract(bundle)
        manifest = self.extractor.detector_manifest()
        clips, dropped = self.clip_selector.select(bundle, anomalies)

        # 4-5. review board + reconciliation
        board_out = self.board.run(bundle, anomalies, clips, manifest)
        reconciliation = self.reconciler.reconcile(board_out)

        # 6-8. counterfactual environments -> descriptors -> legality
        interventions = self.board.parsed(board_out, C.ROLE_INTERVENTION_TUTOR)[
            "intervention_hypotheses"]
        plan: CounterfactualPlan = self.cf_builder.build(reconciliation,
                                                         interventions)
        descriptors = self.proposer.propose(plan)
        legal, rejected = self.legality.screen(descriptors)

        # proposal distribution (seeded, deterministic)
        conf_by_hyp = {h["hypothesis_id"]: h["confidence"]
                       for h in self.board.parsed(
                           board_out, C.ROLE_CAUSAL_FAILURE_ANALYST)[
                           "causal_hypotheses"]}
        critic_json = self.board.parsed(board_out, C.ROLE_CRITIC_SKEPTIC)
        dist = ProposalDistribution(
            legal, conf_by_hyp,
            critic_json["critic_penalty_by_intervention"])

        # 9. the THREE separate scoring families
        variant_ids = [d.descriptor_id for d in legal]
        regret = combined_regret_scores(build_mock_regret_evidences(variant_ids))

        student_load = behavior_failure_score(anomalies)
        gaps: List[BehavioralGapScore] = compute_behavioral_gaps(
            {vid: anomalies for vid in variant_ids},
            build_mock_reference_failure_scores(variant_ids))
        gap_by_id = {g.environment_id: g.behavioral_gap for g in gaps}

        success = build_mock_student_success_rates(variant_ids)
        learn = compute_learnability(success)
        lp = compute_learning_progress(build_mock_failure_history(variant_ids))
        retention = build_mock_global_retention(variant_ids)
        base_sigs = ANCHOR_SIGNATURES + self.archive.baseline_signatures()
        diversity = compute_diversity(legal, base_sigs)

        # 10. Soft Copeland (>=8 inputs; alpha split visible inside)
        pen_by_itv = critic_json["critic_penalty_by_intervention"]
        bundles = []
        for d in legal:
            pens = [pen_by_itv.get(i, 0.0)
                    for i in d.provenance.get("source_intervention_ids", [])]
            bundles.append(EnvironmentScoreBundle(
                environment_id=d.descriptor_id,
                front_regret=regret.per_environment.get(
                    d.descriptor_id, {}).get("front_regret", 0.0),
                global_regret=regret.per_environment.get(
                    d.descriptor_id, {}).get("global_regret", 0.0),
                behavioral_gap=gap_by_id.get(d.descriptor_id, student_load),
                learning_progress={s.environment_id: s.learning_progress
                                   for s in lp}[d.descriptor_id],
                learnability={s.environment_id: s.learnability
                              for s in learn}[d.descriptor_id],
                diversity={s.descriptor_id: s.diversity
                           for s in diversity}[d.descriptor_id],
                global_retention=retention[d.descriptor_id],
                critic_penalty=max(pens) if pens else 0.0,
                alpha_front=ALPHA_FRONT))
        ranking = soft_copeland_rank(bundles)

        # 11. budget: 12 UED + 4 anchors
        budget_plan: BudgetPlan = self.budget.allocate(ranking)

        # 12. archive refresh — DRY RUN ONLY
        score_by_id = {e.environment_id: e.copeland_score
                       for e in ranking.entries}
        refresh = self.archive.refresh(legal, score_by_id, dry_run=True)

        batch_plan: BatchPlan = BatchPlanner().plan(total_updates)

        # 13. CC1 audit fix1 (§3): the final batch/launch decision hard gate
        launch_gate = self._launch_gate(budget_plan, batch_plan, rejected,
                                        board_out)

        # UED-nature assertions (section: NOT an action-guidance system)
        ued_nature = self._ued_nature_assertions(board_out, legal)

        # final fail-closed sweep over the assembled result
        self.backend.assert_no_real_calls()
        result = DryRunResult(
            bundle=bundle,
            detector_manifest=manifest,
            anomalies=[a.model_dump() for a in anomalies],
            clips=[c.model_dump() for c in clips],
            clips_dropped=dropped,
            board=board_out.model_dump(),
            reconciliation=reconciliation.model_dump(),
            counterfactual_plan=plan.model_dump(),
            descriptors=[d.model_dump() for d in legal],
            rejected_descriptors=rejected,
            proposal_distribution_hash=dist.distribution_hash(),
            regret_scores=regret.model_dump(),
            behavioral_gaps=[g.model_dump() for g in gaps],
            learnability=[s.model_dump() for s in learn],
            learning_progress=[s.model_dump() for s in lp],
            diversity=[s.model_dump() for s in diversity],
            copeland_ranking=ranking.model_dump(),
            budget_plan=budget_plan.model_dump(),
            archive_refresh_plan=refresh,
            batch_plan=batch_plan.model_dump(),
            ued_nature_assertions=ued_nature,
            launch_gate=launch_gate,
        )
        sup = self.supervision_guard.assert_clean(
            result.model_dump(), label="full_dry_run_result")
        cert = self._certificate(board_out, budget_plan, sup, student_load,
                                 launch_gate)
        cert["certificate_hash"] = canonical_sha256(
            {k: v for k, v in cert.items() if k != "certificate_hash"})
        result.dry_run_certificate.update(cert)
        return result

    # ------------------------------------------------------------------
    def _launch_gate(self, budget_plan, batch_plan, rejected_descriptors,
                     board_out) -> dict:
        """CC1 audit fix1 (§3): the final batch/launch decision HARD GATE.

        BATCH_PLAN_READY and TRAINING_LAUNCH_AUTHORIZED both require ALL of
        the following at once:
          * budget_plan.status == OK (no unresolved shortfall — a plan that
            merely "looks" 12 entries via duplication does NOT pass: the
            BudgetPlan schema rejects duplicates and status governs),
          * exactly UED_ACTIVE_SLOTS (12) selected UED slots,
          * exactly the 4 fixed GLOBAL canonical anchors,
          * total_envs == NUM_ENVS (16), rollout_length == ROLLOUT_LENGTH
            (128), transitions_per_update == TRANSITIONS_PER_UPDATE (2048),
          * every selected descriptor legal (no rejected descriptor),
          * no unresolved guard violation on the board outputs.

        Any failure -> BOTH flags false + structured launch_block_reasons.
        Duplication to reach 12, slot/anchor/k/batch/transitions reduction
        and silent continuation are all FORBIDDEN — the diagnostic dry-run
        output is still produced, relabeled BLOCKED_DRY_RUN.
        """
        reasons: List[str] = []
        shortfall = max(0, C.UED_ACTIVE_SLOTS - len(budget_plan.ued_slots))
        if budget_plan.status != "OK":
            reasons.append(f"budget_plan_status={budget_plan.status}")
        if len(budget_plan.ued_slots) != C.UED_ACTIVE_SLOTS:
            reasons.append(
                f"selected_ued_slots={len(budget_plan.ued_slots)} "
                f"!= {C.UED_ACTIVE_SLOTS}")
        if len(set(budget_plan.ued_slots)) != len(budget_plan.ued_slots):
            reasons.append("duplicate_ued_slots_forbidden")
        if list(budget_plan.anchor_slots) != list(C.GLOBAL_CANONICAL_ANCHOR_IDS):
            reasons.append(
                f"canonical_anchor_slots={list(budget_plan.anchor_slots)} "
                f"!= the {C.GLOBAL_CANONICAL_ANCHORS} fixed global anchors")
        if batch_plan.num_envs != C.NUM_ENVS:
            reasons.append(f"total_envs={batch_plan.num_envs} "
                           f"!= {C.NUM_ENVS}")
        if batch_plan.rollout_length != C.ROLLOUT_LENGTH:
            reasons.append(f"rollout_length={batch_plan.rollout_length} "
                           f"!= {C.ROLLOUT_LENGTH}")
        if batch_plan.transitions_per_update != C.TRANSITIONS_PER_UPDATE:
            reasons.append(
                f"transitions_per_update="
                f"{batch_plan.transitions_per_update} "
                f"!= {C.TRANSITIONS_PER_UPDATE}")
        if rejected_descriptors:
            reasons.append(
                f"illegal_descriptors={len(rejected_descriptors)} "
                f"(all selected descriptors must be legal)")
        if budget_plan.shortfall_note:
            reasons.append(
                f"unresolved_shortfall: {budget_plan.shortfall_note}")
        if not (board_out.supervision_guard_status == "PASS"
                and board_out.leakage_guard_status == "PASS"):
            reasons.append(
                f"unresolved_guard_violation: supervision="
                f"{board_out.supervision_guard_status} leakage="
                f"{board_out.leakage_guard_status}")
        ready = not reasons
        return dict(
            batch_plan_ready=ready,
            training_launch_authorized=ready,
            launch_block_reasons=reasons,
            shortfall=shortfall,
            checks=dict(
                budget_plan_status=budget_plan.status,
                selected_ued_slots=len(budget_plan.ued_slots),
                canonical_anchor_slots=len(budget_plan.anchor_slots),
                total_envs=batch_plan.num_envs,
                rollout_length=batch_plan.rollout_length,
                transitions_per_update=batch_plan.transitions_per_update,
                rejected_descriptors=len(rejected_descriptors),
                board_supervision_guard_status=board_out.supervision_guard_status,
                board_leakage_guard_status=board_out.leakage_guard_status),
            note="readiness gate only; actual training additionally requires "
                 f"TRAINING_AUTHORIZED=true (currently "
                 f"{C.TRAINING_AUTHORIZED}); diagnostic dry-run output stays "
                 "available and is labeled BLOCKED_DRY_RUN when not ready")

    def _ued_nature_assertions(self, board_out, legal) -> dict:
        tutor = self.board.parsed(board_out, C.ROLE_INTERVENTION_TUTOR)
        analyst = self.board.parsed(board_out, C.ROLE_CAUSAL_FAILURE_ANALYST)
        supervision_clean = all(
            self.supervision_guard.scan(e.parsed_json)["passed"]
            for e in board_out.envelopes)
        return dict(
            method_is_environment_induction=True,
            no_action_guidance_to_student=supervision_clean,
            no_reward_shaping_emitted=supervision_clean,
            no_expert_demonstration_used=True,
            intervention_axes_all_legal=all(
                set(i["mutation_axes"]) <= set(C.MUTATION_AXES)
                for i in tutor["intervention_hypotheses"]),
            descriptor_fields_all_mock=all(
                set(d.model_dump()) <= set(C.MOCK_TASKPARAMS_FIELD_WHITELIST)
                for d in legal),
            analyst_categories_within_vocabulary=all(
                h["cause_category"] in C.CAUSE_CATEGORIES
                for h in analyst["causal_hypotheses"]),
            global_scope=C.TRAINING_SCOPE,
            tier3_only_training=C.TIER3_ONLY_TRAINING,
        )

    def _certificate(self, board_out, budget_plan, supervision_report,
                     student_load, launch_gate) -> dict:
        # CC1 audit fix1 (§3): a gate that is not fully ready relabels the
        # whole run as a BLOCKED diagnostic dry-run (training_authorized is
        # false regardless — the director flag is never set in this package).
        run_class = ("ENGINEERING_DRY_RUN" if launch_gate["batch_plan_ready"]
                     else "BLOCKED_DRY_RUN")
        return dict(
            record_version="bagr_ued.dry_run_certificate.v1",
            bagr_ued_version=C.BA_BAGR_UED_VERSION,
            run_class=run_class,
            batch_plan_ready=launch_gate["batch_plan_ready"],
            training_launch_authorized=launch_gate["training_launch_authorized"],
            launch_block_reasons=launch_gate["launch_block_reasons"],
            launch_gate_shortfall=launch_gate["shortfall"],
            training_authorized=C.TRAINING_AUTHORIZED,
            performance_claim_authorized=False,
            formal_evaluation_authorized=C.FORMAL_EVALUATION_AUTHORIZED,
            real_llm_calls=int(self.backend.real_calls),
            real_llm_calls_authorized=C.REAL_LLM_CALLS_AUTHORIZED,
            mock_llm_calls=int(self.backend.mock_calls),
            real_environment_rollouts=0,
            formal_front_bank_used=False,
            formal_back_bank_used=False,
            formal_full_profile_used=False,
            # the supervision guard assert_clean() already ran over the whole
            # result; reaching this line means nothing was emitted
            student_action_guidance_emitted=False,
            reward_shaping_emitted=False,
            expert_trajectory_used=False,
            real_taskparams_adapter=C.REAL_TASKPARAMS_ADAPTER,
            real_canonical_critic_reject_derivation_rule=
                C.REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE,
            real_canonical_critic_selection_policy=
                C.REAL_CANONICAL_CRITIC_SELECTION_POLICY,
            gpu_used=False,
            training_started=False,
            formal_evaluation_started=False,
            push_performed=False,
            training_scope=C.TRAINING_SCOPE,
            tier3_only_training=C.TIER3_ONLY_TRAINING,
            global_signal_required=C.GLOBAL_SIGNAL_REQUIRED,
            dry_run_env_count=C.NUM_ENVS,
            dry_run_ued_slots=C.UED_ACTIVE_SLOTS,
            dry_run_anchor_slots=C.GLOBAL_CANONICAL_ANCHORS,
            dry_run_transitions=C.TRANSITIONS_PER_UPDATE,
            review_interval_transitions=C.REVIEW_INTERVAL_TRANSITIONS,
            budget_plan_status=budget_plan.status,
            ued_slots_allocated=len(budget_plan.ued_slots),
            anchor_slots_allocated=len(budget_plan.anchor_slots),
            board_supervision_guard_status=board_out.supervision_guard_status,
            board_leakage_guard_status=board_out.leakage_guard_status,
            student_behavior_failure_score_mock=student_load,
            certificate_hash="",
        )
