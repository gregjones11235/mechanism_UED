"""Role: Critic/Skeptic (task section 9).

Independent adversarial review across nine dimensions:
  evidence_sufficiency / causal_over_attribution / implementation_bug_coverage /
  legal_taskparams_convertibility / action_guidance_leakage /
  formal_info_usage / tier3_only_bias / counterfactual_controls /
  falsifiability.

The mock rule runs the REAL guards (TrajectorySupervisionGuard,
FormalEvaluationLeakageGuard) against the upstream role outputs — so a Tutor/
Analyst emitting "don't sleep / move away" is caught HERE as well as at the
board boundary (defense in depth).

Two evidence blocks are kept STRICTLY SEPARATE:
  reject_derivation_evidence     — why a hypothesis/intervention is rejected
  selection_recommendation_evidence — what the selector may use for ranking
and two canonical rules are echoed verbatim as PENDING — the schema validator
REFUSES any other value, so this package cannot secretly freeze
REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE or
REAL_CANONICAL_CRITIC_SELECTION_POLICY.
"""
from __future__ import annotations

import json
from typing import Dict, List

from pydantic import Field, field_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard)
from d052.bagr_ued.review_contracts import CONTEXT_CLOSE, CONTEXT_OPEN, RoleEnvelope
from d052.bagr_ued.trajectory_supervision_guard import TrajectorySupervisionGuard
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_CRITIC_SKEPTIC
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the Critic/Skeptic role of the BA-BAGR-UED review board.
Adversarially review the upstream outputs: evidence sufficiency, causal
over-attribution, missed implementation-bug explanation, legality of the
TaskParams conversion, action-guidance leakage, formal-evaluation info
usage, Tier3-only bias, missing counterfactual controls, unfalsifiability.
Keep reject-derivation evidence and selection-recommendation evidence
SEPARATE. The two canonical critic rules remain PENDING — you may not fix them.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the CriticSkepticOutput schema.
"""

DIMENSIONS = (
    "evidence_sufficiency",
    "causal_over_attribution",
    "implementation_bug_coverage",
    "legal_taskparams_convertibility",
    "action_guidance_leakage",
    "formal_info_usage",
    "tier3_only_bias",
    "counterfactual_controls",
    "falsifiability",
)


class CritiqueItem(CanonicalModel):
    dimension: str
    status: str = Field(pattern=r"^(pass|concern|fail)$")
    evidence: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)

    @field_validator("dimension")
    @classmethod
    def _dim(cls, v: str) -> str:
        if v not in DIMENSIONS:
            raise ValueError(f"UNKNOWN_CRITIQUE_DIMENSION: {v!r}")
        return v


class CriticSkepticOutput(CanonicalModel):
    critique_items: List[CritiqueItem] = Field(default_factory=list)
    #: SEPARATE from selection_recommendation_evidence (never merged)
    reject_derivation_evidence: Dict[str, object] = Field(default_factory=dict)
    #: SEPARATE from reject_derivation_evidence (never merged)
    selection_recommendation_evidence: Dict[str, object] = Field(
        default_factory=dict)
    critic_reject_hypothesis_ids: List[str] = Field(default_factory=list)
    critic_reject_intervention_ids: List[str] = Field(default_factory=list)
    critic_penalty_by_intervention: Dict[str, float] = Field(default_factory=dict)
    real_canonical_critic_reject_derivation_rule: str = Field(min_length=1)
    real_canonical_critic_selection_policy: str = Field(min_length=1)

    @field_validator("real_canonical_critic_reject_derivation_rule")
    @classmethod
    def _rule_pending(cls, v: str) -> str:
        if v != C.REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE:
            raise ValueError(
                "CRITIC_RULE_FROZEN_FORBIDDEN: REAL_CANONICAL_CRITIC_REJECT_"
                "DERIVATION_RULE must remain PENDING this round")
        return v

    @field_validator("real_canonical_critic_selection_policy")
    @classmethod
    def _policy_pending(cls, v: str) -> str:
        if v != C.REAL_CANONICAL_CRITIC_SELECTION_POLICY:
            raise ValueError(
                "CRITIC_POLICY_FROZEN_FORBIDDEN: REAL_CANONICAL_CRITIC_"
                "SELECTION_POLICY must remain PENDING this round")
        return v


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> CriticSkepticOutput:
    return CriticSkepticOutput.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """Deterministic adversarial review running the real guards."""
    supervision = TrajectorySupervisionGuard()
    leakage = FormalEvaluationLeakageGuard()

    hypotheses = context.get("causal_hypotheses", [])
    interventions = context.get("intervention_hypotheses", [])
    proposals = context.get("alternative_environment_proposals", [])
    findings = context.get("behavior_findings", [])

    reject_hyps: List[str] = []
    reject_itvs: List[str] = []
    penalty: Dict[str, float] = {}
    items: List[dict] = []

    # 1. evidence sufficiency (per finding confidence)
    weak = [f["finding_id"] for f in findings if f["confidence"] < 0.5]
    items.append(dict(
        dimension="evidence_sufficiency",
        status="concern" if weak else "pass",
        evidence=[f"low-confidence findings: {weak}" if weak
                  else f"all {len(findings)} findings have confidence >= 0.5"],
        references=[f["finding_id"] for f in findings][:8]))

    # 2. causal over-attribution: high confidence on thin support
    over = [h["hypothesis_id"] for h in hypotheses
            if h["confidence"] > 0.85 and len(h["supporting_evidence"]) <= 1]
    if over:
        reject_hyps.extend(over)
    items.append(dict(
        dimension="causal_over_attribution",
        status="fail" if over else "pass",
        evidence=([f"confidence>0.85 with <=1 supporting item: {over}"] if over
                  else ["no hypothesis asserts strong confidence on thin "
                        "support; statements are hedged with contradicting "
                        "evidence"]),
        references=over))

    # 3. implementation-bug coverage
    cats = {h["cause_category"] for h in hypotheses}
    covered = "implementation_or_adapter_bug" in cats
    items.append(dict(
        dimension="implementation_bug_coverage",
        status="pass" if covered else "concern",
        evidence=[("an implementation_or_adapter_bug hypothesis is present "
                   "(requires code inspection, not environment induction)")
                  if covered else
                  "no implementation/adapter-bug hypothesis was generated; "
                  "verify the evidence adapter before trusting causes"],
        references=sorted(cats)))

    # 4. legal TaskParams convertibility
    illegal = [i["intervention_id"] for i in interventions
               if any(a not in C.MUTATION_AXES for a in i["mutation_axes"])]
    if illegal:
        reject_itvs.extend(illegal)
    items.append(dict(
        dimension="legal_taskparams_convertibility",
        status="fail" if illegal else "pass",
        evidence=([f"axes outside vocabulary: {illegal}"] if illegal else
                  ["all intervention axes are within the legal mutation-axis "
                   "vocabulary; REAL_TASKPARAMS_ADAPTER="
                   + C.REAL_TASKPARAMS_ADAPTER + " (mock adapter only)"]),
        references=[i["intervention_id"] for i in interventions]))

    # 5. action-guidance leakage — run the REAL supervision guard upstream
    sup_report = supervision.scan(
        {"behavior_findings": findings, "causal_hypotheses": hypotheses,
         "intervention_hypotheses": interventions,
         "alternative_environment_proposals": proposals},
        label="upstream_role_outputs")
    leaked_hyps = sorted({h["hypothesis_id"] for h in hypotheses
                          if not supervision.scan(h)["passed"]})
    leaked_itvs = sorted({i["intervention_id"] for i in interventions
                          if not supervision.scan(i)["passed"]})
    if leaked_hyps:
        reject_hyps.extend(leaked_hyps)
    if leaked_itvs:
        reject_itvs.extend(leaked_itvs)
    items.append(dict(
        dimension="action_guidance_leakage",
        status="fail" if not sup_report["passed"] else "pass",
        evidence=([f"supervision guard findings: "
                   f"{json.dumps(sup_report['findings'], ensure_ascii=False)}"]
                  if not sup_report["passed"] else
                  ["TrajectorySupervisionGuard: no supervision keys and no "
                   "direct action advice in any upstream role output"]),
        references=leaked_hyps + leaked_itvs))

    # 6. formal info usage — run the REAL leakage guard over the context
    leak_report = leakage.scan(context, label="board_context")
    items.append(dict(
        dimension="formal_info_usage",
        status="fail" if not leak_report["passed"] else "pass",
        evidence=([f"leakage guard findings: "
                   f"{json.dumps(leak_report['findings'], ensure_ascii=False)}"]
                  if not leak_report["passed"] else
                  ["FormalEvaluationLeakageGuard: no FORMAL_FRONT/BACK/FULL, "
                   "FROZEN_BANK, or certificate-private-state provenance in "
                   "the context"]),
        references=[]))

    # 7. tier3-only bias
    families = sorted({p["environment_family"] for p in proposals})
    axes = sorted({a for i in interventions for a in i["mutation_axes"]})
    bias = len(families) < 3
    items.append(dict(
        dimension="tier3_only_bias",
        status="concern" if bias else "pass",
        evidence=[f"explorer families={families}; tutor axes={axes}; "
                  f"scope={C.TRAINING_SCOPE} "
                  f"(tier3_only_training={C.TIER3_ONLY_TRAINING})"],
        references=families))

    # 8. counterfactual controls
    no_control = [i["intervention_id"] for i in interventions
                  if i.get("counterfactual_groups")
                  and "control" not in i["counterfactual_groups"]]
    if no_control:
        reject_itvs.extend(no_control)
    items.append(dict(
        dimension="counterfactual_controls",
        status="fail" if no_control else "pass",
        evidence=([f"interventions missing a control group: {no_control}"]
                  if no_control else
                  ["every intervention carries a 'control' group plus "
                   "single-axis groups"]),
        references=no_control))

    # 9. falsifiability
    unfalsifiable = [h["hypothesis_id"] for h in hypotheses
                     if not h["testable_prediction"].strip()
                     or (h["cause_category"] not in
                         ("implementation_or_adapter_bug", "unknown")
                         and not h["required_counterfactual_variables"])]
    if unfalsifiable:
        reject_hyps.extend(unfalsifiable)
    items.append(dict(
        dimension="falsifiability",
        status="fail" if unfalsifiable else "pass",
        evidence=([f"unfalsifiable hypotheses: {unfalsifiable}"]
                  if unfalsifiable else
                  ["every non-implementation hypothesis carries a testable "
                   "prediction and counterfactual variables"]),
        references=unfalsifiable))

    # penalties (soft signal for the selector; separate from rejection)
    for i in interventions:
        iid = i["intervention_id"]
        p = 0.0
        p += 0.15 if (bias and iid) else 0.0
        p += 0.1 if any(f["confidence"] < 0.5 for f in findings
                        if f["finding_id"] in
                        {h["finding_id"] for h in hypotheses
                         if h["hypothesis_id"] in i["target_hypothesis_ids"]}) \
            else 0.0
        penalty[iid] = round(min(1.0, p), 4)

    reject_hyps = sorted(set(reject_hyps))
    reject_itvs = sorted(set(reject_itvs))

    return dict(
        critique_items=items,
        reject_derivation_evidence=dict(
            rejected_hypotheses=reject_hyps,
            rejected_interventions=reject_itvs,
            basis=["causal_over_attribution", "action_guidance_leakage",
                   "counterfactual_controls", "falsifiability",
                   "legal_taskparams_convertibility"],
            note="This block derives REJECTIONS only. It must not be merged "
                 "with selection_recommendation_evidence."),
        selection_recommendation_evidence=dict(
            critic_penalty_by_intervention=penalty,
            global_family_coverage=families,
            tutor_axis_coverage=axes,
            note="This block is a soft selection signal only. It must not be "
                 "merged with reject_derivation_evidence. Consumption of this "
                 "signal by a real selector awaits "
                 "REAL_CANONICAL_CRITIC_SELECTION_POLICY (PENDING)."),
        critic_reject_hypothesis_ids=reject_hyps,
        critic_reject_intervention_ids=reject_itvs,
        critic_penalty_by_intervention=penalty,
        real_canonical_critic_reject_derivation_rule=
            C.REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE,
        real_canonical_critic_selection_policy=
            C.REAL_CANONICAL_CRITIC_SELECTION_POLICY,
    )


def run(context: dict, backend, sequence: int) -> RoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return RoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, sequence=sequence, prompt=prompt,
        raw_response=raw, parsed_dump=parsed.model_dump())
