"""ProposalArchive with dry-run refresh (task sections 1 / 14).

Deterministic store keyed by descriptor_hash. This round the controller only
ever calls ``refresh(..., dry_run=True)`` — it returns the add/update plan
WITHOUT mutating the archive, so the 8192-transition review cadence can be
exercised end-to-end without pretending a real curriculum archive changed.

CC3 fix2 (task §6-§7) — the commit path is UNBYPASSABLE:

  * ``commit`` REQUIRES a strong-typed ``LaunchGate`` (keyword-required, no
    default — omitting it is a TypeError; passing None or a non-LaunchGate
    fails closed). The fix1 bypass interface ``commit(..., launch_gate=None)``
    as a way to skip the gate NO LONGER EXISTS.
  * commit re-verifies ALL four gate hashes against the CURRENT values
    (batch plan, selected descriptors, guard report, legality report) — any
    mismatch -> ARCHIVE_COMMIT_REJECTED. A gate from another batch / another
    descriptor set / a tampered plan cannot commit.
  * ``refresh(..., dry_run=False)`` REQUIRES a gate (None -> immediate
    REFRESH_GATE_REQUIRED fail closed) and NEVER rebuilds a default-PASS gate
    internally; ``refresh(..., dry_run=True)`` may run gate-less but then
    reports training_authorized=false and does NOT touch the archive.
  * even a fully authorized gate still meets the package-level
    TRAINING_AUTHORIZED backstop (false this round -> ARCHIVE_COMMIT_UNAUTHORIZED).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from d052.bagr_ued import constants as C
from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.launch_gate import (
    GATE_VERSION,
    LaunchGate,
    compute_batch_plan_hash,
    compute_guard_report_hash,
    compute_legality_report_hash,
    compute_selected_descriptor_hash,
)


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
                *, dry_run: bool,
                launch_gate: LaunchGate | None = None) -> dict:
        """Plan (dry_run=True) or gate+commit (dry_run=False) an archive refresh.

        CC3 fix2 §7: dry_run=True permits launch_gate=None but then MUST NOT
        commit and reports training_authorized=false; dry_run=False requires
        the gate up-front (None -> fail closed; never a rebuilt default gate).
        """
        if not dry_run and launch_gate is None:
            raise AssertionError(
                "REFRESH_GATE_REQUIRED: refresh(dry_run=False) requires a "
                "strong-typed LaunchGate; the archive never rebuilds a "
                "default-PASS gate internally")
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
                    # CC3 fix2 §7: the dry-run path NEVER authorizes training
                    training_authorized=False,
                    plan_hash=canonical_sha256(
                        {"would_add": would_add, "would_update": would_update,
                         "unchanged": sorted(unchanged)}))
        if not dry_run:
            self.commit(descriptors, score_by_descriptor_id,
                        launch_gate=launch_gate)
        return plan

    def commit(self, descriptors: List[TaskParamsDescriptor],
               score_by_descriptor_id: Dict[str, float],
               *, launch_gate: LaunchGate,
               batch_plan=None, board_out=None,
               legal_ids: Sequence[str] = (),
               rejected_descriptors: Sequence[dict] = ()) -> None:
        """Active-archive commit — UNBYPASSABLE gate + hash binding (fix2 §6).

        ``launch_gate`` is keyword-REQUIRED with no default: calling commit
        without it is a TypeError (runtime fail closed). None / any
        non-LaunchGate value fails closed as ARCHIVE_COMMIT_REJECTED, as does
        a version mismatch, a gate whose final authorization is false, or a
        gate whose four hashes do not match the CURRENT values. When the
        current-state objects (batch_plan / board_out / legality inputs) are
        supplied, the hashes are recomputed and compared literally.
        """
        # 1. strong-type verification — no bypass interface exists
        if not isinstance(launch_gate, LaunchGate):
            raise AssertionError(
                "ARCHIVE_COMMIT_REJECTED: launch_gate must be a strong-typed "
                f"LaunchGate, got {type(launch_gate).__name__!r}; "
                f"commit(launch_gate=None) is not a bypass")
        if launch_gate.gate_version != GATE_VERSION:
            raise AssertionError(
                "ARCHIVE_COMMIT_REJECTED: gate_version mismatch: "
                f"{launch_gate.gate_version!r} != {GATE_VERSION!r}")

        # 2. authorization: structural AND director AND final, all true
        if not launch_gate.structural_batch_ready:
            raise AssertionError(
                "ARCHIVE_COMMIT_REJECTED: structural_batch_ready=false; "
                f"reasons={list(launch_gate.reasons)}")
        if not launch_gate.director_training_authorized:
            raise AssertionError(
                "ARCHIVE_COMMIT_REJECTED: director_training_authorized=false")
        if not launch_gate.final_training_launch_authorized:
            raise AssertionError(
                "ARCHIVE_COMMIT_REJECTED: final_training_launch_authorized="
                "false")

        # 3. hash binding: every gate hash must match the CURRENT state
        if batch_plan is not None:
            cur = compute_batch_plan_hash(batch_plan)
            if cur != launch_gate.batch_plan_hash:
                raise AssertionError(
                    "ARCHIVE_COMMIT_REJECTED: batch_plan_hash mismatch — "
                    f"gate={launch_gate.batch_plan_hash[:12]}… "
                    f"current={cur[:12]}… (gate from another batch plan)")
        if board_out is not None:
            cur = compute_guard_report_hash(board_out)
            if cur != launch_gate.guard_report_hash:
                raise AssertionError(
                    "ARCHIVE_COMMIT_REJECTED: guard_report_hash mismatch — "
                    "guard state changed after the gate was issued")
        if legal_ids or rejected_descriptors:
            cur = compute_legality_report_hash(legal_ids,
                                               rejected_descriptors)
            if cur != launch_gate.legality_report_hash:
                raise AssertionError(
                    "ARCHIVE_COMMIT_REJECTED: legality_report_hash mismatch — "
                    "legality evidence changed after the gate was issued")
        cur_sel = compute_selected_descriptor_hash(descriptors)
        if cur_sel != launch_gate.selected_descriptor_hash:
            raise AssertionError(
                "ARCHIVE_COMMIT_REJECTED: selected_descriptor_hash mismatch — "
                "the descriptors to commit are not the gated selection "
                "(duplicate / illegal / foreign candidate in the commit set)")

        # 4. package-level authorization backstop (false this round)
        if not C.TRAINING_AUTHORIZED:
            raise AssertionError(
                "ARCHIVE_COMMIT_UNAUTHORIZED: TRAINING_AUTHORIZED=false this "
                "round; only refresh(dry_run=True) is permitted")
        for d in descriptors:
            self.entries[d.descriptor_hash] = dict(
                descriptor=d.model_dump(),
                descriptor_id=d.descriptor_id,
                last_score=score_by_descriptor_id.get(d.descriptor_id, 0.0))
