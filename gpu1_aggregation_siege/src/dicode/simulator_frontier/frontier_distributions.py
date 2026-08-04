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
