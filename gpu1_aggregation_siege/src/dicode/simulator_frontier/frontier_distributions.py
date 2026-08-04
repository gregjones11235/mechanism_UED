"""The 12 dynamic frontier distributions + 4 shared standard-reset anchors (P0-6).

A ``FrontierDistribution`` names one dynamic frontier distribution with all
ten required elements: bucket, eligible states, start-state weights,
TaskParams range, seed distribution, stochasticity range, memory mode, goal
family, evidence hash and retention constraint.

The four anchors are NEVER self-invented: composition is hard-blocked until
the controller-signed shared anchor manifest binds via
``anchor_manifest.bind_anchor_manifest``.  Without it, ``compose_12_plus_4``
raises ``BLOCKED_SHARED_ANCHOR_MANIFEST``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .anchor_manifest import (
    ANCHOR_SLOT_COUNT,
    BLOCKED_SHARED_ANCHOR_MANIFEST,
    DYNAMIC_DISTRIBUTION_COUNT,
    AnchorManifest,
    RetentionContract,
    bind_anchor_manifest,
)
from .archive_schema import FrontierArchiveEntry  # noqa: F401  (type clarity)
from .discovery_provenance import DiscoveryProvenance
from .errors import InvalidEvidenceError, ProductionBlockedError
from .evidence_selector import verify_selection_evidence
from .llm_contracts import PlannerOutput, assert_planner_output_bound
from .memory_modes import MemoryRestoreMode
from .provenance import SearchActionLeakageGuard

DISTRIBUTION_SCHEMA = "simulator_frontier.frontier-distribution/v1"

_WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FrontierDistribution:
    """One dynamic frontier distribution (ten elements, fail-closed fields)."""

    distribution_id: str
    bucket: tuple[Any, ...]
    eligible_states: tuple[str, ...]
    start_state_weights: Mapping[str, float]
    taskparam_ranges: Mapping[str, Any]
    seed_distribution: Mapping[str, Any]
    stochasticity_range: Mapping[str, Any]
    memory_mode: str
    goal_family: str
    evidence_hash: str
    retention_constraint: str

    def __post_init__(self) -> None:
        if not str(self.distribution_id).strip():
            raise InvalidEvidenceError("FrontierDistribution.distribution_id is empty")
        if not self.eligible_states:
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: eligible_states is empty")
        if len(set(self.eligible_states)) != len(self.eligible_states):
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: duplicate eligible states")
        weights = dict(self.start_state_weights)
        if set(weights) != set(self.eligible_states):
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: start_state_weights keys must "
                "equal eligible_states exactly")
        total = 0.0
        for state_id, weight in weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                    or float(weight) < 0.0 or float(weight) > 1.0:
                raise InvalidEvidenceError(
                    f"distribution {self.distribution_id}: weight for {state_id!r} must "
                    f"be in [0, 1], got {weight!r}")
            total += float(weight)
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: start_state_weights must sum to 1, "
                f"got {total}")
        try:
            mode = MemoryRestoreMode(str(self.memory_mode))
        except ValueError as exc:
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: unknown memory mode "
                f"{self.memory_mode!r}") from exc
        if mode is MemoryRestoreMode.ZERO_MEMORY:
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: ZERO_MEMORY is ablation-only and "
                "can never back a production frontier distribution")
        if not str(self.goal_family).strip():
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: goal_family is empty")
        if not str(self.evidence_hash).strip():
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: evidence_hash is empty")
        if not str(self.retention_constraint).strip():
            raise InvalidEvidenceError(
                f"distribution {self.distribution_id}: retention_constraint is empty")


@dataclass(frozen=True)
class FrontierDistributionPlan:
    """12 validated dynamic distributions bound to the 4-anchor manifest."""

    distributions: tuple[FrontierDistribution, ...]
    anchor_binding: Mapping[str, Any]
    plan_hash: str


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_frontier_distribution_plan(plan: FrontierDistributionPlan, *,
                                        archive: Any,
                                        evidence_hashes: Sequence[str]) -> None:
    """Fail-closed validation of a full 12-distribution plan.

    Checks: exactly 12 distributions with unique ids; every eligible state
    exists in the archive and its entry is TRAINING_DISCOVERY-bound; each
    distribution's evidence hash is among the measured evidence hashes; and
    no distribution field carries an action-guidance key.
    """
    if not isinstance(plan, FrontierDistributionPlan):
        raise InvalidEvidenceError("expected a FrontierDistributionPlan")
    if len(plan.distributions) != DYNAMIC_DISTRIBUTION_COUNT:
        raise InvalidEvidenceError(
            f"frontier plan requires exactly {DYNAMIC_DISTRIBUTION_COUNT} dynamic "
            f"distributions, got {len(plan.distributions)}")
    ids = [d.distribution_id for d in plan.distributions]
    if len(set(ids)) != len(ids):
        raise InvalidEvidenceError(f"duplicate distribution ids: {ids}")
    if not plan.anchor_binding.get("bound"):
        raise InvalidEvidenceError(
            "frontier plan requires a bound anchor manifest (anchor_binding.bound)")
    allowed_hashes = {str(h) for h in evidence_hashes}
    for distribution in plan.distributions:
        if distribution.evidence_hash not in allowed_hashes:
            raise InvalidEvidenceError(
                f"distribution {distribution.distribution_id}: evidence_hash is not "
                "among the measured evidence hashes (never accept unmeasured claims)")
        SearchActionLeakageGuard.validate_aggregate(asdict(distribution))
        for state_id in distribution.eligible_states:
            try:
                entry, _encoded = archive.get(state_id)
            except KeyError as exc:
                raise InvalidEvidenceError(
                    f"distribution {distribution.distribution_id}: eligible state "
                    f"{state_id!r} is not in the archive") from exc
            if entry.discovery_provenance != DiscoveryProvenance.TRAINING_DISCOVERY.value:
                raise InvalidEvidenceError(
                    f"distribution {distribution.distribution_id}: state {state_id!r} is "
                    f"not TRAINING_DISCOVERY-bound (got {entry.discovery_provenance!r})")


def compose_12_plus_4(distributions: Sequence[FrontierDistribution], *,
                      manifest: AnchorManifest | None,
                      retention: RetentionContract,
                      archive: Any,
                      evidence_hashes: Sequence[str]) -> FrontierDistributionPlan:
    """Compose the 12 dynamic distributions with the 4 shared anchors.

    Hard-blocked until the controller-signed shared anchor manifest is bound:
    anchor science content is never self-invented.  After composition the
    whole plan is validated against the archive and the measured evidence.
    """
    if manifest is None:
        raise ProductionBlockedError(
            f"{BLOCKED_SHARED_ANCHOR_MANIFEST}: the shared frozen anchor manifest must "
            "be issued and signed by the controller; composition refuses to invent "
            "anchor science")
    anchor_binding = bind_anchor_manifest(manifest, retention)
    if len(anchor_binding.get("anchor_ids", ())) != ANCHOR_SLOT_COUNT:
        raise InvalidEvidenceError("anchor binding must expose exactly "
                                   f"{ANCHOR_SLOT_COUNT} anchors")
    plan_hash = _canonical_sha256({
        "schema": DISTRIBUTION_SCHEMA,
        "distributions": [asdict(d) for d in distributions],
        "anchor_binding": dict(anchor_binding),
    })
    plan = FrontierDistributionPlan(distributions=tuple(distributions),
                                    anchor_binding=anchor_binding,
                                    plan_hash=plan_hash)
    validate_frontier_distribution_plan(plan, archive=archive,
                                        evidence_hashes=evidence_hashes)
    return plan


# ---------------------------------------------------------------------------
# CC4 follow-up (P0-10): deterministic compiler from the TYPED planner output
# to exactly the 12 dynamic frontier distributions.
#
# Before this compiler existed, ``one_window_pipeline`` consumed a caller-
# supplied ``frontier_distributions`` tuple that had NO connection to the
# planner's typed output — the planner could say one thing while an arbitrary
# hand-built distribution list was executed.  The compiler closes that gap:
# every dynamic distribution is a pure, deterministic function of the
# validated ``PlannerOutput`` plus the minted selection evidence, and the
# result is re-validated against the archive.  Nothing is invented, nothing
# is caller-supplied, and any malformed / unbound / non-production input is
# rejected fail-closed.
# ---------------------------------------------------------------------------

PLANNER_COMPILER_VERSION = "planner-distribution-compiler/v1"

# Canonical ids of the 12 dynamic distribution slots.  The planner's
# ``start_distribution`` MUST be keyed by exactly these slot ids, each slot
# carrying a non-empty ``state_id -> weight`` mapping.
DISTRIBUTION_SLOT_IDS = tuple(f"D{i:02d}" for i in range(DYNAMIC_DISTRIBUTION_COUNT))


@dataclass(frozen=True)
class PlannerCompilation:
    """The deterministic result of compiling one typed planner output.

    ``compilation_hash`` binds the plan (id + hash), the plan's evidence hash,
    the minted selection-evidence hash, the compiler version and every
    compiled distribution, so a downstream consumer can re-derive the exact
    same 12 distributions from the same inputs and detect any drift.
    """

    plan_id: str
    plan_hash: str
    plan_evidence_hash: str
    selection_evidence_hash: str
    distributions: tuple[FrontierDistribution, ...]
    compilation_hash: str
    compiler_version: str = PLANNER_COMPILER_VERSION


def _compile_slot_weights(slot: str, slot_map: Any) -> tuple[tuple[str, ...], Mapping[str, float]]:
    """Normalize one slot's ``state_id -> weight`` mapping to sum exactly 1."""
    if not isinstance(slot_map, Mapping) or len(slot_map) == 0:
        raise InvalidEvidenceError(
            f"distribution slot {slot}: start_distribution slot must be a non-empty "
            f"state_id->weight mapping, got {type(slot_map).__name__}")
    raw: dict[str, float] = {}
    for state_id, weight in slot_map.items():
        sid = str(state_id).strip()
        if not sid:
            raise InvalidEvidenceError(f"distribution slot {slot}: empty state_id")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or not float(weight) > 0.0 or not math.isfinite(float(weight)):
            raise InvalidEvidenceError(
                f"distribution slot {slot}: weight for {sid!r} must be a finite "
                f"number > 0, got {weight!r}")
        if sid in raw:
            raise InvalidEvidenceError(
                f"distribution slot {slot}: duplicate state_id {sid!r}")
        raw[sid] = float(weight)
    total = sum(raw.values())
    states = tuple(sorted(raw))
    weights: dict[str, float] = {}
    running = 0.0
    for index, sid in enumerate(states):
        if index == len(states) - 1:
            last = 1.0 - running
            weights[sid] = 0.0 if last < 0.0 else last
        else:
            value = raw[sid] / total
            weights[sid] = value
            running += value
    return states, weights


def _compile_retention_constraint(plan: PlannerOutput) -> str:
    constraints = ";".join(str(c) for c in plan.retention_constraints)
    return (f"anchor_ratio>={float(plan.anchor_ratio):.6f}"
            + (f"|{constraints}" if constraints else ""))


def compile_planner_to_frontier_distributions(
        plan: Any, *,
        plan_evidence_hash: str,
        selection_evidence: Any,
        archive: Any) -> PlannerCompilation:
    """Deterministically compile the typed planner output into 12 distributions.

    Fail-closed contract:

    * ``plan`` must be a validated ``PlannerOutput`` (arbitrary mappings and
      foreign types are refused) and must RE-BIND to ``plan_evidence_hash``
      (its ``plan_hash`` is recomputed; a stale or tampered plan raises).
    * ``selection_evidence`` must be a minted, untampered ``SelectionEvidence``;
      its ``evidence_hash`` is stamped onto every compiled distribution so the
      distributions are provably backed by the measured branch evidence.
    * ``plan.memory_mode`` must be a production mode (never ZERO_MEMORY).
    * ``plan.taskparam_ranges`` / ``seed_distribution`` /
      ``stochasticity_distribution`` must be non-empty mappings — these fields
      actually back the mixed-start execution and may not be silently empty.
    * ``plan.start_distribution`` must be keyed by EXACTLY the 12 slot ids in
      ``DISTRIBUTION_SLOT_IDS``; each slot is a non-empty positive-weight
      ``state_id -> weight`` mapping, normalized to sum exactly 1.
    * Every eligible state must exist in the archive, be
      TRAINING_DISCOVERY-bound, and all states within one slot must share one
      archive bucket (the slot's bucket is the measured archive bucket, never
      invented).
    """
    if isinstance(plan, Mapping):
        raise InvalidEvidenceError(
            "compile_planner_to_frontier_distributions requires a typed "
            "PlannerOutput, not a mapping (hand-built distributions are never "
            "accepted on the production path)")
    if not isinstance(plan, PlannerOutput):
        raise InvalidEvidenceError(
            f"compile_planner_to_frontier_distributions requires a typed "
            f"PlannerOutput, got {type(plan).__name__}")

    # Re-bind the plan to the evidence hash it claims (defense in depth: the
    # two-LLM gate already verified this; a drifted plan must not compile).
    assert_planner_output_bound(plan, evidence_hash=plan_evidence_hash)
    # The evidence we stamp onto every distribution must be genuine.
    verify_selection_evidence(selection_evidence)

    try:
        memory_mode = MemoryRestoreMode(str(plan.memory_mode))
    except ValueError as exc:
        raise InvalidEvidenceError(
            f"planner memory_mode {plan.memory_mode!r} is not a valid memory mode") from exc
    if memory_mode is MemoryRestoreMode.ZERO_MEMORY:
        raise InvalidEvidenceError(
            "planner memory_mode ZERO_MEMORY can never back production "
            "frontier distributions (ablation-only)")

    for field_name in ("taskparam_ranges", "seed_distribution", "stochasticity_distribution"):
        value = getattr(plan, field_name)
        if not isinstance(value, Mapping) or len(value) == 0:
            raise InvalidEvidenceError(
                f"planner {field_name} must be a non-empty mapping: these fields "
                "back the mixed-start execution and may not be silently empty "
                "(fail closed rather than execute a dead distribution)")

    start_map = plan.start_distribution
    if not isinstance(start_map, Mapping):
        raise InvalidEvidenceError(
            f"planner start_distribution must be a mapping keyed by the 12 slot "
            f"ids, got {type(start_map).__name__}")
    missing = [s for s in DISTRIBUTION_SLOT_IDS if s not in start_map]
    extra = [k for k in start_map if k not in DISTRIBUTION_SLOT_IDS]
    if missing or extra:
        raise InvalidEvidenceError(
            f"planner start_distribution must be keyed by EXACTLY the 12 slot ids "
            f"{list(DISTRIBUTION_SLOT_IDS)}; missing={missing} extra={extra}")

    goal_family = f"FRONTIER:{selection_evidence.feasibility_class.value}"
    retention_constraint = _compile_retention_constraint(plan)
    stamped_evidence_hash = str(selection_evidence.evidence_hash)

    compiled: list[FrontierDistribution] = []
    for slot in DISTRIBUTION_SLOT_IDS:
        states, weights = _compile_slot_weights(slot, start_map[slot])
        bucket: tuple[Any, ...] | None = None
        for state_id in states:
            try:
                entry, _encoded = archive.get(state_id)
            except KeyError as exc:
                raise InvalidEvidenceError(
                    f"distribution slot {slot}: eligible state {state_id!r} is not "
                    "in the archive") from exc
            if entry.discovery_provenance != DiscoveryProvenance.TRAINING_DISCOVERY.value:
                raise InvalidEvidenceError(
                    f"distribution slot {slot}: state {state_id!r} is not "
                    f"TRAINING_DISCOVERY-bound (got {entry.discovery_provenance!r})")
            entry_bucket = tuple(entry.bucket())
            if bucket is None:
                bucket = entry_bucket
            elif entry_bucket != bucket:
                raise InvalidEvidenceError(
                    f"distribution slot {slot}: eligible states span multiple archive "
                    f"buckets ({bucket} vs {entry_bucket}); one slot must live in one "
                    "measured bucket")
        assert bucket is not None
        compiled.append(FrontierDistribution(
            distribution_id=f"{plan.plan_id}::{slot}",
            bucket=bucket,
            eligible_states=states,
            start_state_weights=weights,
            taskparam_ranges=dict(plan.taskparam_ranges),
            seed_distribution=dict(plan.seed_distribution),
            stochasticity_range=dict(plan.stochasticity_distribution),
            memory_mode=memory_mode.value,
            goal_family=goal_family,
            evidence_hash=stamped_evidence_hash,
            retention_constraint=retention_constraint,
        ))

    compilation_hash = _canonical_sha256({
        "compiler_version": PLANNER_COMPILER_VERSION,
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "plan_evidence_hash": plan_evidence_hash,
        "selection_evidence_hash": stamped_evidence_hash,
        "distributions": [asdict(d) for d in compiled],
    })
    return PlannerCompilation(
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        plan_evidence_hash=plan_evidence_hash,
        selection_evidence_hash=stamped_evidence_hash,
        distributions=tuple(compiled),
        compilation_hash=compilation_hash,
    )
