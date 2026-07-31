"""ProposalArchive with dry-run refresh (task sections 1 / 14).

Deterministic store keyed by descriptor_hash. This round the controller only
ever calls ``refresh(..., dry_run=True)`` — it returns the add/update plan
WITHOUT mutating the archive, so the 8192-transition review cadence can be
exercised end-to-end without pretending a real curriculum archive changed.
``commit`` exists for the future real path and refuses to run while
TRAINING_AUTHORIZED is false; it additionally refuses while the controller
launch gate is not fully ready (CC1 audit fix1, task §3/§4).
"""
from __future__ import annotations

from typing import Dict, List

from d052.bagr_ued import constants as C
from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.bagr_ued.hashing import canonical_sha256


class ProposalArchive:
    def __init__(self) -> None:
        #: descriptor_hash -> entry
        self.entries: Dict[str, dict] = {}

    def baseline_signatures(self) -> List[Dict[str, str]]:
        from d052.bagr_ued.diversity import signature
        return [signature(TaskParamsDescriptor.model_validate(e["descriptor"]))
                for e in self.entries.values()]

    def refresh(self, descriptors: List[TaskParamsDescriptor],
                score_by_descriptor_id: Dict[str, float],
                *, dry_run: bool) -> dict:
        would_add, would_update, unchanged = [], [], []
        for d in sorted(descriptors, key=lambda x: x.descriptor_id):
            score = score_by_descriptor_id.get(d.descriptor_id, 0.0)
            existing = self.entries.get(d.descriptor_hash)
            if existing is None:
                would_add.append(dict(descriptor_id=d.descriptor_id,
                                      descriptor_hash=d.descriptor_hash,
                                      score=score))
            elif existing["last_score"] != score:
                would_update.append(dict(descriptor_id=d.descriptor_id,
                                         descriptor_hash=d.descriptor_hash,
                                         old_score=existing["last_score"],
                                         new_score=score))
            else:
                unchanged.append(d.descriptor_id)
        plan = dict(dry_run=dry_run,
                    would_add=would_add,
                    would_update=would_update,
                    unchanged=sorted(unchanged),
                    plan_hash=canonical_sha256(
                        {"would_add": would_add, "would_update": would_update,
                         "unchanged": sorted(unchanged)}))
        if not dry_run:
            self.commit(descriptors, score_by_descriptor_id)
        return plan

    def commit(self, descriptors: List[TaskParamsDescriptor],
               score_by_descriptor_id: Dict[str, float],
               *, launch_gate: dict | None = None) -> None:
        # CC1 audit fix1 (§3/§4): an active-archive commit while the launch
        # gate is not fully ready FAILS CLOSED before anything else is
        # considered.
        if launch_gate is not None and not launch_gate.get("batch_plan_ready"):
            raise AssertionError(
                "ACTIVE_ARCHIVE_COMMIT_BLOCKED: launch gate not ready: "
                f"launch_block_reasons="
                f"{launch_gate.get('launch_block_reasons')}")
        if not C.TRAINING_AUTHORIZED:
            raise AssertionError(
                "ARCHIVE_COMMIT_UNAUTHORIZED: TRAINING_AUTHORIZED=false this "
                "round; only refresh(dry_run=True) is permitted")
        for d in descriptors:
            self.entries[d.descriptor_hash] = dict(
                descriptor=d.model_dump(),
                descriptor_id=d.descriptor_id,
                last_score=score_by_descriptor_id.get(d.descriptor_id, 0.0))
