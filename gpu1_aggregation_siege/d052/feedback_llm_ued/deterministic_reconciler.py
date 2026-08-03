"""DeterministicReconciler — rule-based, fail-closed closure from the
AdaptiveEnvironmentDesigner's proposals to an executable curriculum plan.

Same doctrine as the BA-BAGR-UED board reconciler: the LLM proposes, rules
dispose. Everything here is pure, deterministic, and loud — every drop,
cap, relabel and top-up is logged, and legality violations raise instead of
being silently coerced.

Rules (in application order):

 1. proposals parse into ``FamilyAllocation`` (malformed = hard error);
 2. every cited feedback id must exist in ``known_feedback_ids``
    (fail-closed cross-check, no dangling citations);
 3. honesty: an allocation with NO cited feedback id is forced to
    ``is_exploration=True`` and may only use EXPLORATION decisions;
    an allocation citing feedback while flagged exploration is a masquerade
    and raises; a RETIRE without cited feedback raises as well (retirement
    is a verdict, not exploration);
 4. REQUEST_CONTROL escalates to a human: logged, zero budget;
 5. RETIRE removes a family from the dynamic budget (cites feedback);
    if a family is both retired and re-proposed active, retirement wins;
 6. duplicates per family: first by (priority, order) wins;
 7. allocation cap: MAX_INTERVENTIONS budget-bearing allocations;
 8. exploration cap: MAX_EXPLORATION_PROPOSALS allocations,
    EXPLORATION_SLOT_CAP slots total;
 9. slot fill of DYNAMIC_UED_SLOTS: exploration allocations reserve up to
    EXPLORATION_SLOT_CAP first (so uncited exploration is never starved by
    core demand), then core fills greedily in decision-priority order
    (RETAIN > EXPAND_BUDGET > MUTATE > REDUCE_BUDGET); leftover budget tops
    up the highest-priority allocation; an empty result raises
    INSUFFICIENT_DYNAMIC_ALLOCATION;
10. the 4 global canonical anchors are ALWAYS reserved (1 slot each),
    outside the dynamic budget.

The output carries everything the PlanRevisionRecord needs (per-family
modifications incl. old/new slots), so plan_k -> plan_{k+1} stays auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CurriculumPlan,
    FamilyAllocation,
)

#: deterministic budget priority (lower = served first)
DECISION_PRIORITY = {
    C.DECISION_RETAIN: 0,
    C.DECISION_EXPAND_BUDGET: 1,
    C.DECISION_MUTATE: 2,
    C.DECISION_REDUCE_BUDGET: 3,
}
_EXPLORATION_PRIORITY = len(DECISION_PRIORITY)
#: total slot cap for exploration allocations inside the dynamic budget
EXPLORATION_SLOT_CAP = 2

Proposal = Union[FamilyAllocation, Dict[str, object]]


@dataclass
class ReconciledPlan:
    plan: CurriculumPlan
    #: per-family modification dicts ready for PlanModification
    modifications: List[dict] = field(default_factory=list)
    log: List[dict] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)
    request_control: bool = False


class DeterministicReconciler:
    """Pure closure of designer proposals into a legal, hashed plan."""

    def reconcile(self, *, window: int, mode: str,
                  proposals: Sequence[Proposal],
                  known_feedback_ids,
                  previous_plan_id: str = "",
                  previous_slots: Optional[Mapping[str, int]] = None
                  ) -> ReconciledPlan:
        log: List[dict] = []
        allocs = [self._coerce(p, i) for i, p in enumerate(proposals)]

        # -- 2. dangling feedback citations fail closed ----------------------
        known = set(known_feedback_ids)
        cited = {fid for a in allocs for fid in a.based_on_feedback_ids}
        missing = sorted(cited - known)
        if missing:
            raise ValueError(
                f"UNKNOWN_FEEDBACK_ID: {missing} — designer cited feedback "
                f"that does not exist in the SimulatorFeedbackStore")

        # -- 3. honesty relabel ---------------------------------------------
        fixed: List[FamilyAllocation] = []
        for a in allocs:
            if not a.based_on_feedback_ids:
                if a.decision == C.DECISION_RETIRE:
                    raise ValueError(
                        "RETIRE_REQUIRES_FEEDBACK: retirement is a verdict, "
                        f"not exploration (family {a.environment_family!r})")
                if a.decision not in C.EXPLORATION_DECISIONS:
                    raise ValueError(
                        f"EXPLORATION_DECISION_ONLY: uncited allocation for "
                        f"family {a.environment_family!r} may only use "
                        f"{sorted(C.EXPLORATION_DECISIONS)}, got "
                        f"{a.decision!r}")
                if not a.is_exploration:
                    log.append(dict(rule="forced_exploration_label",
                                    family=a.environment_family))
                    a = a.model_copy(update={"is_exploration": True})
            elif a.is_exploration:
                raise ValueError(
                    f"MASQUERADE_FORBIDDEN: allocation for family "
                    f"{a.environment_family!r} cites feedback and may not be "
                    f"flagged exploration")
            fixed.append(a)

        # -- 4/5. escalations + retirements ---------------------------------
        request_control = False
        retired_allocs: List[Tuple[int, FamilyAllocation]] = []
        active: List[Tuple[int, FamilyAllocation]] = []
        for i, a in enumerate(fixed):
            if a.decision == C.DECISION_REQUEST_CONTROL:
                request_control = True
                log.append(dict(rule="request_control_escalation",
                                family=a.environment_family, order=i,
                                reason=a.reason))
                continue
            if a.decision == C.DECISION_RETIRE:
                retired_allocs.append((i, a))
                log.append(dict(rule="retired", family=a.environment_family,
                                order=i))
                continue
            active.append((i, a))
        retired_families = {a.environment_family for _, a in retired_allocs}
        kept: List[Tuple[int, FamilyAllocation]] = []
        for i, a in active:
            if a.environment_family in retired_families:
                log.append(dict(rule="retire_overrides_active",
                                family=a.environment_family, order=i))
                continue
            kept.append((i, a))

        # -- 6. dedup per family: (priority, order) wins ---------------------
        best: Dict[str, Tuple[Tuple[int, int], FamilyAllocation]] = {}
        for i, a in kept:
            prio = (DECISION_PRIORITY[a.decision]
                    if not a.is_exploration else _EXPLORATION_PRIORITY)
            key = (prio, i)
            cur = best.get(a.environment_family)
            if cur is None or key < cur[0]:
                if cur is not None:
                    log.append(dict(rule="duplicate_dropped",
                                    family=a.environment_family,
                                    kept_order=key[1],
                                    dropped_order=cur[0][1]))
                best[a.environment_family] = (key, a)
            else:
                log.append(dict(rule="duplicate_dropped",
                                family=a.environment_family,
                                kept_order=cur[0][1], dropped_order=i))
        deduped = [entry for entry in sorted(best.values(),
                                             key=lambda t: t[0])]

        # -- 7. intervention cap ---------------------------------------------
        capped = deduped
        if len(capped) > C.MAX_INTERVENTIONS:
            drops = capped[C.MAX_INTERVENTIONS:]
            capped = capped[:C.MAX_INTERVENTIONS]
            for (_, a) in drops:
                log.append(dict(rule="intervention_cap_dropped",
                                family=a.environment_family))

        # -- 8. exploration caps ----------------------------------------------
        explorations = [e for e in capped if e[1].is_exploration]
        core = [e for e in capped if not e[1].is_exploration]
        if len(explorations) > C.MAX_EXPLORATION_PROPOSALS:
            drops = explorations[C.MAX_EXPLORATION_PROPOSALS:]
            explorations = explorations[:C.MAX_EXPLORATION_PROPOSALS]
            for (_, a) in drops:
                log.append(dict(rule="exploration_cap_dropped",
                                family=a.environment_family))

        # -- 9. slot fill ---------------------------------------------------------
        # Exploration reserves its (capped) slice FIRST so uncited exploration
        # is never starved by core demand; core then fills greedily in
        # decision-priority order.
        remaining = C.DYNAMIC_UED_SLOTS
        granted: List[List[object]] = []          # [allocation, slots]
        expl_slots = 0
        for (_i, a) in explorations:
            want = min(max(0, int(a.slots)), EXPLORATION_SLOT_CAP - expl_slots)
            g = min(want, remaining)
            if g <= 0:
                log.append(dict(rule="zero_budget_dropped",
                                family=a.environment_family))
                continue
            expl_slots += g
            granted.append([a, g])
            remaining -= g
        ordered = sorted(core, key=lambda e: (DECISION_PRIORITY[e[1].decision],
                                              e[0]))
        for (_i, a) in ordered:
            want = max(0, int(a.slots))
            g = min(want, remaining)
            if g <= 0:
                log.append(dict(rule="zero_budget_dropped",
                                family=a.environment_family))
                continue
            granted.append([a, g])
            remaining -= g
        if remaining > 0:
            if not granted:
                raise ValueError(
                    "INSUFFICIENT_DYNAMIC_ALLOCATION: designer proposals "
                    "cannot fill the dynamic budget and no prior allocation "
                    "can absorb it")
            # leftover tops up the highest-priority CORE allocation (never an
            # exploration one, which is capped); fall back to any grant only
            # if no core allocation was funded at all
            target = next((g for g in granted if not g[0].is_exploration),
                          granted[0])
            target[1] += remaining
            log.append(dict(rule="leftover_top_up",
                            family=target[0].environment_family,
                            added=remaining))

        # -- 10. plan assembly ---------------------------------------------------
        final = sorted(granted, key=lambda g: g[0].environment_family)
        final_allocs = [a.model_copy(update={"slots": int(g)})
                        for a, g in final]
        retained = sorted(a.environment_family for a, _ in final
                          if not a.is_exploration
                          and a.decision == C.DECISION_RETAIN)
        mutated = sorted(a.environment_family for a, _ in final
                         if not a.is_exploration
                         and a.decision != C.DECISION_RETAIN)
        explored = sorted(a.environment_family for a, _ in final
                          if a.is_exploration)
        retired_list = sorted({a.environment_family for _, a in retired_allocs})
        plan_payload = dict(
            window=window, mode=mode, previous_plan_id=previous_plan_id,
            allocations=[a.model_dump() for a in final_allocs],
            retired_families=retired_list,
        )
        plan_id = f"plan-{window:04d}-" + canonical_sha256(plan_payload)[:16]
        plan = CurriculumPlan(
            plan_id=plan_id, window=window, previous_plan_id=previous_plan_id,
            mode=mode, allocations=final_allocs,
            retained_families=retained, mutated_families=mutated,
            retired_families=retired_list, explored_families=explored)

        prev = dict(previous_slots or {})
        modifications: List[dict] = []
        for a, g in final:
            modifications.append(dict(
                environment_family=a.environment_family,
                decision=a.decision,
                reason=a.reason,
                based_on_feedback_ids=list(a.based_on_feedback_ids),
                is_exploration=a.is_exploration,
                old_slots=prev.get(a.environment_family),
                new_slots=int(g)))
        for _i, a in sorted(retired_allocs, key=lambda t: t[1].environment_family):
            modifications.append(dict(
                environment_family=a.environment_family,
                decision=C.DECISION_RETIRE,
                reason=a.reason,
                based_on_feedback_ids=list(a.based_on_feedback_ids),
                is_exploration=False,
                old_slots=prev.get(a.environment_family),
                new_slots=0))

        stats = dict(
            n_proposals=len(proposals),
            n_budget_allocations=len(final),
            n_retired=len(retired_list),
            dynamic_slots=C.DYNAMIC_UED_SLOTS,
            exploration_slots=expl_slots,
            anchor_ids=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
            anchor_slots=C.GLOBAL_ANCHOR_SLOTS,
            final_batch=C.FINAL_BATCH,
            rule_version=C.RECONCILE_RULE_VERSION,
            request_control=request_control,
        )
        return ReconciledPlan(plan=plan, modifications=modifications, log=log,
                              stats=stats, request_control=request_control)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _coerce(p: Proposal, order: int) -> FamilyAllocation:
        if isinstance(p, FamilyAllocation):
            return p
        if isinstance(p, dict):
            return FamilyAllocation(**p)          # malformed => hard error
        raise ValueError(
            f"ILLEGAL_PROPOSAL_TYPE at order {order}: {type(p)!r}")
