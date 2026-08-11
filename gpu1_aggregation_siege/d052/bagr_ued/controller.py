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

import dataclasses
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
from d052.bagr_ued.launch_gate import (
    CONTEXT_VERSION,
    LaunchContext,
    LaunchGate,
    compute_clip_batch_hash,
    evaluate_launch_context,
    evaluate_launch_gate,
)
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
from d052.bagr_ued.symbolic_behavior_clip import (
    build_symbolic_clip_payload,
    validate_symbolic_clip_payload,
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
    #: CC3 fix2 (§12): bounded, de-identified, per-step SYMBOLIC behavior
    #: clip payloads (with provenance + payload hashes) the board received
    symbolic_behavior_clips: List[dict] = Field(default_factory=list)
    board: dict = Field(default_factory=dict)
    #: CC3 fix2 (§13): provisional out-of-taxonomy hypotheses surfaced by the
    #: auditor — audit trail ONLY; forbidden in selector/budget/archive
    provisional_anomaly_hypotheses: List[dict] = Field(default_factory=list)
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
    #: CC3 fix2 (§4): the strong-typed LaunchGate, serialized. Carries THREE
    #: unambiguous booleans — structural_batch_ready /
    #: director_training_authorized / final_training_launch_authorized (= the
    #: first two ANDed) — plus the four hash bindings archive.commit verifies.
    launch_gate: dict = Field(default_factory=dict)
    #: CC3 fix3 (§1): the strong-typed LaunchContext, serialized — the whole
    #: review-window decision state (seven conditions + the FULL six-way hash
    #: binding + the shared clip batch hash). commit/refresh(non-dry) require
    #: it alongside the gate.
    launch_context: dict = Field(default_factory=dict)
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

        # CC3 fix2 (§12): bounded, de-identified, per-step SYMBOLIC behavior
        # clip payloads — built + validated fail-closed (both guards +
        # raw-exposure scan + source admissibility + payload hash + limits)
        # BEFORE the board and the certificate may rely on them.
        #
        # CC3 fix3 (§10): the payload batch is built EXACTLY ONCE here and
        # shared verbatim with the board and the certificate — the board does
        # not rebuild its own copy. One batch, one hash, two consumers.
        symbolic_payloads = [build_symbolic_clip_payload(bundle, c)
                             for c in clips]
        symbolic_clips: List[dict] = []
        for payload in symbolic_payloads:
            report = validate_symbolic_clip_payload(payload)
            assert report["passed"], (
                f"SYMBOLIC_CLIP_VALIDATION_FAILED: {report['findings']}")
            symbolic_clips.append(payload.model_dump())
        # CC3 fix3 (§7): per-window clip cap re-asserted over the payload
        # batch (the selector already enforces it; defense in depth)
        if len(symbolic_payloads) > C.MAX_CLIPS_PER_REVIEW_WINDOW:
            raise AssertionError(
                f"CLIP_PER_WINDOW_LIMIT_EXCEEDED: {len(symbolic_payloads)} "
                f"clip payloads > MAX_CLIPS_PER_REVIEW_WINDOW="
                f"{C.MAX_CLIPS_PER_REVIEW_WINDOW}")
        per_episode_count: Dict[str, int] = {}
        for p in symbolic_payloads:
            per_episode_count[p.episode_id] = \
                per_episode_count.get(p.episode_id, 0) + 1
        for episode_id, count in sorted(per_episode_count.items()):
            assert count <= C.MAX_CLIPS_PER_EPISODE, (
                f"CLIP_PER_EPISODE_LIMIT_EXCEEDED: episode {episode_id} "
                f"carries {count} clips > MAX_CLIPS_PER_EPISODE="
                f"{C.MAX_CLIPS_PER_EPISODE}")
        clip_batch_hash = compute_clip_batch_hash(symbolic_payloads)

        # 4-5. review board + reconciliation (the board consumes the SHARED
        # payload batch and binds its hash; provisional out-of-taxonomy
        # hypotheses are surfaced by the auditor but never enter the selector
        # chain)
        board_out = self.board.run(bundle, anomalies, clips, manifest,
                                   symbolic_payloads=symbolic_payloads)
        assert board_out.symbolic_clip_batch_hash == clip_batch_hash, (
            "SYMBOLIC_CLIP_BATCH_DIVERGENCE: the board bound a different "
            "clip batch hash than the controller-built batch (CC3 fix3 §10)")
        reconciliation = self.reconciler.reconcile(board_out)
        provisional = list(self.board.parsed(
            board_out, C.ROLE_BEHAVIOR_AUDITOR).get(
            "provisional_anomaly_hypotheses", []))

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

        # 12. archive refresh — DRY RUN ONLY (no gate needed for dry_run=True;
        # the gate is required only by the active commit path, CC3 fix2 §7)
        score_by_id = {e.environment_id: e.copeland_score
                       for e in ranking.entries}
        refresh = self.archive.refresh(legal, score_by_id, dry_run=True)

        batch_plan: BatchPlan = BatchPlanner().plan(total_updates)

        # 13. CC3 fix2 (§4-§8): the strong-typed final batch/launch gate.
        # CC3 fix2 (§15-§16): legality semantics — UNSELECTED rejected
        # proposals are recorded but do NOT block a structurally-satisfied
        # LEGAL batch; the gate checks the SELECTED side only.
        legal_ids = [d.descriptor_id for d in legal]
        slot_set = set(budget_plan.ued_slots)
        selected_descriptors = [d for d in legal if d.descriptor_id in slot_set]
        launch_gate: LaunchGate = evaluate_launch_gate(
            budget_plan, batch_plan, selected_descriptors, rejected,
            board_out, legal_ids=legal_ids)

        # CC3 fix3 (§1): the strong-typed LaunchContext assembled from the
        # SAME gate (one binding, never two divergent records). In the dry
        # run the extra window conditions stay FALSE (fail-closed defaults) —
        # review certificate / provenance chain / simulator probe / selection
        # completion must be positively established by the real-simulator
        # path before final authorization can ever become true.
        launch_context: LaunchContext = evaluate_launch_context(
            launch_gate, board_out, symbolic_payloads=symbolic_payloads)

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
            symbolic_behavior_clips=symbolic_clips,
            provisional_anomaly_hypotheses=provisional,
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
            launch_gate=dataclasses.asdict(launch_gate),
            launch_context=dataclasses.asdict(launch_context),
        )
        sup = self.supervision_guard.assert_clean(
            result.model_dump(), label="full_dry_run_result")
        cert = self._certificate(board_out, budget_plan, sup, student_load,
                                 launch_gate, launch_context,
                                 clip_batch_hash, dropped,
                                 symbolic_clips, provisional)
        cert["certificate_hash"] = canonical_sha256(
            {k: v for k, v in cert.items() if k != "certificate_hash"})
        result.dry_run_certificate.update(cert)
        return result

    # ------------------------------------------------------------------
    # NOTE (CC3 fix2 §4-§7): the fix1 dict-returning ``_launch_gate`` method
    # was REMOVED. The launch decision is now the strong-typed LaunchGate
    # evaluated by d052.bagr_ued.launch_gate.evaluate_launch_gate (called in
    # run_dry_run above); archive.commit requires that gate object and
    # re-verifies its four hash bindings. There is no dict gate and no
    # ``training_launch_authorized`` double-meaning field anywhere.

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
                     student_load, launch_gate: LaunchGate,
                     launch_context: LaunchContext,
                     clip_batch_hash: str, clips_dropped: int,
                     symbolic_clips: List[dict],
                     provisional: List[dict]) -> dict:
        # CC3 fix2 (§4-§8): a gate that is not STRUCTURALLY ready relabels
        # the whole run as a BLOCKED diagnostic dry-run. The certificate
        # carries the THREE unambiguous gate booleans separately — no field
        # named training_launch_authorized exists anywhere (forbidden, §4).
        run_class = ("ENGINEERING_DRY_RUN"
                     if launch_gate.structural_batch_ready
                     else "BLOCKED_DRY_RUN")
        shortfall = max(0, C.UED_ACTIVE_SLOTS - len(budget_plan.ued_slots))

        # CC3 fix2 (§12): verify the certificate claims over the actual
        # symbolic clip payloads — the board reviewed REAL per-step symbolic
        # evidence, and no raw action integer / raw state / formal trajectory
        # is exposed in it.
        raw_action_int_findings = []
        raw_state_findings = []
        for dump in symbolic_clips:
            rep = validate_symbolic_clip_payload(dump)
            for f in rep["findings"]:
                if f["code"] == "RAW_ACTION_INTEGER_EXPOSED":
                    raw_action_int_findings.append(f)
                elif f["code"] == "RAW_STATE_EXPOSED":
                    raw_state_findings.append(f)

        # CC3 fix2 (§13): provisional hypotheses are surfaced but must NOT
        # reach the selector/budget/archive — prove it by id-disjointness
        # against everything the budget/selector chain selected.
        provisional_ids = {p.get("hypothesis_id", "") for p in provisional}
        selector_side_ids = set(budget_plan.ued_slots) | \
            set(budget_plan.anchor_slots)
        provisional_in_selector = len(provisional_ids & selector_side_ids)

        return dict(
            record_version="bagr_ued.dry_run_certificate.v2",
            bagr_ued_version=C.BA_BAGR_UED_VERSION,
            run_class=run_class,
            # CC3 fix2 (§4): the three UNAMBIGUOUS gate booleans
            structural_batch_ready=launch_gate.structural_batch_ready,
            director_training_authorized=
                launch_gate.director_training_authorized,
            final_training_launch_authorized=
                launch_gate.final_training_launch_authorized,
            launch_block_reasons=list(launch_gate.reasons),
            launch_gate_shortfall=shortfall,
            gate_version=launch_gate.gate_version,
            gate_batch_plan_hash=launch_gate.batch_plan_hash,
            gate_selected_descriptor_hash=launch_gate.selected_descriptor_hash,
            gate_guard_report_hash=launch_gate.guard_report_hash,
            gate_legality_report_hash=launch_gate.legality_report_hash,
            # CC3 fix3 (§2): the full six-way binding also carries critic +
            # director-authorization hashes
            gate_critic_report_hash=launch_gate.critic_report_hash,
            gate_director_authorization_hash=
                launch_gate.director_authorization_hash,
            # CC3 fix3 (§1): the LaunchContext state of this window
            context_version=launch_context.context_version,
            context_review_certificate_valid=
                launch_context.review_certificate_valid,
            context_provenance_valid=launch_context.provenance_valid,
            context_guards_passed=launch_context.guards_passed,
            context_simulator_probe_complete=
                launch_context.simulator_probe_complete,
            context_selection_complete=launch_context.selection_complete,
            context_final_training_launch_authorized=
                launch_context.final_training_launch_authorized,
            launch_context_reasons=list(launch_context.reasons),
            # CC3 fix3 (§10): board and certificate bind ONE shared batch
            clip_payload_batch_hash=clip_batch_hash,
            board_symbolic_clip_batch_hash=
                board_out.symbolic_clip_batch_hash,
            board_and_certificate_share_clip_batch=
                (board_out.symbolic_clip_batch_hash == clip_batch_hash),
            # CC3 fix3 (§7): clip caps enforced with an explicit drop count
            clips_dropped_by_caps=clips_dropped,
            max_clips_per_episode=C.MAX_CLIPS_PER_EPISODE,
            max_clips_per_review_window=C.MAX_CLIPS_PER_REVIEW_WINDOW,
            # CC3 fix2 (§12): symbolic behavior clip evidence claims
            behavior_review_has_symbolic_clips=bool(symbolic_clips),
            symbolic_behavior_clip_count=len(symbolic_clips),
            raw_action_integer_exposed=bool(raw_action_int_findings),
            raw_state_exposed=bool(raw_state_findings),
            formal_trajectory_exposed=False,
            symbolic_clip_schema_version="bagr_ued.symbolic_clip.v1",
            # CC3 fix2 (§13): provisional discovery contract
            provisional_anomaly_hypotheses_surfaced=len(provisional),
            provisional_anomaly_hypotheses_in_selector=provisional_in_selector,
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
