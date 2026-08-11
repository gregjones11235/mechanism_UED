"""ReviewBoardReconciler (task section 10).

Deterministic reconciliation of the role outputs. NOT a majority vote: each
item is decided by explicit, versioned RULES over the role envelopes + the
critic's structured verdicts, and EVERY decision item is bound to the exact
role output hash, evidence span hash, prompt version, backend/model identity,
and reconciliation rule version that produced it.

LLM (mock) outputs are CANDIDATE hypotheses only — this module never lets a
role output directly override the selector or curriculum; it forwards
supported/accepted items downstream with their provenance bindings.

Decisions:
  accepted_behavior_findings   confidence >= MIN_FINDING_CONFIDENCE and the
                               critic did not rule evidence insufficient
  supported_causal_hypotheses  not rejected by critic, falsifiable (prediction
                               + counterfactual variables, unless the category
                               is implementation/unknown), parent finding
                               accepted
  contested_hypotheses         critic concern but no rejection (e.g. thin
                               evidence) — kept, forwarded as unresolved
  rejected_hypotheses          critic reject (over-attribution, advice leak,
                               unfalsifiable) or missing testability
  accepted_intervention_hypotheses  >=1 supported target, critic convertibility
                               pass, control group present, not rejected
  required_counterfactual_tests     union of counterfactual variables of
                               supported hypotheses (deduped, sorted)
  unresolved_uncertainties     contested items + open critic dimensions
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from d052.bagr_ued import constants as C
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.review_contracts import ReviewBoardOutput
from d052.schemas.common import CanonicalModel

MIN_FINDING_CONFIDENCE = 0.5
#: categories whose testability is by code inspection, not environment axes
_NON_ENV_CATEGORIES = frozenset({"implementation_or_adapter_bug", "unknown"})


class ReconciledItem(CanonicalModel):
    """One reconciled decision with full provenance bindings (section 10)."""

    item_id: str = Field(min_length=1)
    item_kind: str = Field(min_length=1)
    decision: str = Field(pattern=r"^(accepted|supported|contested|rejected)$")
    reason: str = Field(min_length=1)
    bound_role_output_hashes: Dict[str, str] = Field(default_factory=dict)
    evidence_span_hashes: List[str] = Field(default_factory=list)
    prompt_versions: Dict[str, str] = Field(default_factory=dict)
    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    reconciliation_rule_version: str = C.RECONCILIATION_RULE_VERSION


class ReconciliationResult(CanonicalModel):
    bundle_id: str = Field(min_length=1)
    accepted_behavior_findings: List[ReconciledItem] = Field(default_factory=list)
    supported_causal_hypotheses: List[ReconciledItem] = Field(default_factory=list)
    contested_hypotheses: List[ReconciledItem] = Field(default_factory=list)
    rejected_hypotheses: List[ReconciledItem] = Field(default_factory=list)
    accepted_intervention_hypotheses: List[ReconciledItem] = Field(
        default_factory=list)
    required_counterfactual_tests: List[str] = Field(default_factory=list)
    unresolved_uncertainties: List[str] = Field(default_factory=list)
    reconciliation_rule_version: str = C.RECONCILIATION_RULE_VERSION
    reconciliation_hash: str = ""

    def finalize_hash(self) -> "ReconciliationResult":
        payload = self.model_dump()
        payload.pop("reconciliation_hash", None)
        object.__setattr__(self, "reconciliation_hash", canonical_sha256(payload))
        return self


class ReviewBoardReconciler:
    """Pure deterministic reconciliation over a ReviewBoardOutput."""

    def reconcile(self, board: ReviewBoardOutput) -> ReconciliationResult:
        env = {e.role: e for e in board.envelopes}
        auditor = env[C.ROLE_BEHAVIOR_AUDITOR]
        analyst = env[C.ROLE_CAUSAL_FAILURE_ANALYST]
        tutor = env[C.ROLE_INTERVENTION_TUTOR]
        critic = env[C.ROLE_CRITIC_SKEPTIC]

        findings = auditor.parsed_json["behavior_findings"]
        hypotheses = analyst.parsed_json["causal_hypotheses"]
        interventions = tutor.parsed_json["intervention_hypotheses"]
        critic_json = critic.parsed_json
        reject_hyps = set(critic_json["critic_reject_hypothesis_ids"])
        reject_itvs = set(critic_json["critic_reject_intervention_ids"])

        # which findings the critic considered evidence-insufficient
        insufficient = set()
        for item in critic_json["critique_items"]:
            if item["dimension"] == "evidence_sufficiency" and \
                    item["status"] != "pass":
                insufficient.update(item.get("references", []))
        over_attributed = set()
        for item in critic_json["critique_items"]:
            if item["dimension"] == "causal_over_attribution":
                over_attributed.update(item.get("references", []))

        def span_hashes_for(finding_id: str) -> List[str]:
            for a in findings:
                if a["finding_id"] == finding_id:
                    return [canonical_sha256(
                        {"finding_id": finding_id,
                         "span_ids": a.get("evidence_span_ids", [])})]
            return []

        bindings = {
            "backend_id": critic.backend_id,
            "model_id": critic.model_id,
        }

        # --- findings -------------------------------------------------------
        accepted_findings: List[ReconciledItem] = []
        accepted_finding_ids = set()
        for f in sorted(findings, key=lambda x: x["finding_id"]):
            fid = f["finding_id"]
            ok = f["confidence"] >= MIN_FINDING_CONFIDENCE and \
                fid not in insufficient
            accepted_findings.append(ReconciledItem(
                item_id=fid,
                item_kind="behavior_finding",
                decision="accepted" if ok else "contested",
                reason=(f"confidence={f['confidence']} >= "
                        f"{MIN_FINDING_CONFIDENCE} and critic evidence "
                        f"sufficiency pass" if ok else
                        f"confidence={f['confidence']} or critic evidence-"
                        f"sufficiency concern"),
                bound_role_output_hashes={C.ROLE_BEHAVIOR_AUDITOR:
                                          auditor.response_hash},
                evidence_span_hashes=span_hashes_for(fid),
                prompt_versions={C.ROLE_BEHAVIOR_AUDITOR: auditor.prompt_version},
                **bindings))
            if ok:
                accepted_finding_ids.add(fid)

        # --- hypotheses -----------------------------------------------------
        supported: List[ReconciledItem] = []
        contested: List[ReconciledItem] = []
        rejected: List[ReconciledItem] = []
        supported_hyp_ids = set()
        for h in sorted(hypotheses, key=lambda x: x["hypothesis_id"]):
            hid = h["hypothesis_id"]
            fid = h["finding_id"]
            env_testable = h["cause_category"] not in _NON_ENV_CATEGORIES
            falsifiable = bool(h["testable_prediction"].strip()) and \
                (not env_testable or bool(h["required_counterfactual_variables"]))
            if hid in reject_hyps or not falsifiable or \
                    fid not in accepted_finding_ids:
                reason = ("critic rejection" if hid in reject_hyps
                          else "unfalsifiable (no testable prediction / "
                               "counterfactual variables)" if not falsifiable
                          else "parent finding not accepted")
                rejected.append(self._hyp_item(hid, "rejected", reason, analyst,
                                               critic, fid, bindings))
            elif fid in insufficient or hid in over_attributed:
                contested.append(self._hyp_item(
                    hid, "contested", "critic concern (thin evidence / "
                    "over-attribution) without rejection", analyst, critic,
                    fid, bindings))
            else:
                supported.append(self._hyp_item(
                    hid, "supported", "falsifiable, critic-pass, parent "
                    "finding accepted", analyst, critic, fid, bindings))
                supported_hyp_ids.add(hid)

        # --- interventions ---------------------------------------------------
        accepted_itv: List[ReconciledItem] = []
        for itv in sorted(interventions, key=lambda x: x["intervention_id"]):
            iid = itv["intervention_id"]
            targets = set(itv["target_hypothesis_ids"])
            has_supported = bool(targets & supported_hyp_ids)
            has_control = "control" in itv.get("counterfactual_groups", [])
            ok = has_supported and has_control and iid not in reject_itvs
            reason = (
                f"supported targets={sorted(targets & supported_hyp_ids)}, "
                f"control group present, critic convertibility pass"
                if ok else
                ("critic rejection" if iid in reject_itvs else
                 "no supported target hypothesis" if not has_supported
                 else "missing control group"))
            accepted_itv.append(ReconciledItem(
                item_id=iid,
                item_kind="intervention_hypothesis",
                decision="accepted" if ok else "rejected",
                reason=reason,
                bound_role_output_hashes={C.ROLE_INTERVENTION_TUTOR:
                                          tutor.response_hash,
                                          C.ROLE_CRITIC_SKEPTIC:
                                          critic.response_hash},
                evidence_span_hashes=[],
                prompt_versions={C.ROLE_INTERVENTION_TUTOR: tutor.prompt_version,
                                 C.ROLE_CRITIC_SKEPTIC: critic.prompt_version},
                **bindings))

        # --- required counterfactual tests -----------------------------------
        tests = sorted({a for h in hypotheses
                        if h["hypothesis_id"] in supported_hyp_ids
                        for a in h["required_counterfactual_variables"]})

        # --- unresolved -------------------------------------------------------
        unresolved = [f"contested_hypothesis:{i.item_id}" for i in contested]
        unresolved += [f"{i.item_kind}:{i.item_id}" for i in accepted_itv
                       if i.decision != "accepted"]
        for item in critic_json["critique_items"]:
            if item["status"] == "concern":
                unresolved.append(f"critic_concern:{item['dimension']}")
        unresolved = sorted(set(unresolved))

        result = ReconciliationResult(
            bundle_id=board.bundle_id,
            accepted_behavior_findings=accepted_findings,
            supported_causal_hypotheses=supported,
            contested_hypotheses=contested,
            rejected_hypotheses=rejected,
            accepted_intervention_hypotheses=accepted_itv,
            required_counterfactual_tests=tests,
            unresolved_uncertainties=unresolved,
        )
        return result.finalize_hash()

    @staticmethod
    def _hyp_item(hid: str, decision: str, reason: str, analyst, critic,
                  fid: str, bindings: dict) -> ReconciledItem:
        return ReconciledItem(
            item_id=hid,
            item_kind="causal_hypothesis",
            decision=decision,
            reason=reason,
            bound_role_output_hashes={C.ROLE_CAUSAL_FAILURE_ANALYST:
                                      analyst.response_hash,
                                      C.ROLE_CRITIC_SKEPTIC:
                                      critic.response_hash},
            evidence_span_hashes=[canonical_sha256(
                {"finding_id": fid, "hypothesis_id": hid})],
            prompt_versions={C.ROLE_CAUSAL_FAILURE_ANALYST:
                             analyst.prompt_version,
                             C.ROLE_CRITIC_SKEPTIC: critic.prompt_version},
            **bindings)
