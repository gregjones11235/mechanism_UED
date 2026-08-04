"""C15 / P1-7: controller snapshot, atomic save/load, cross-window restore.

A snapshot captures the COMPLETE frozen state of the double-window state
machine at a window boundary:

* the phase map, the comparison mode, and the completed WindowRecords (the
  next window index is their count);
* the ledger and the SimulatorFeedbackStore dumps (every record's content
  hash rides along — C14 recomputation re-verifies each one at load);
* revisions, plans (by id AND by window), the per-window feedback index,
  the retirement registry (``_retired_at``) and the human-reopen
  authorization;
* the anchor binding (ids + label), the runtime grants (P0-6: a restored
  REAL run keeps its authorization), the envelope/sequence counters, the
  probe-runner and backend usage counters, the training log, the human
  decision artifacts and the board hashes.

Everything serializes through the canonical JSON, and the payload carries a
recomputable ``snapshot_hash``. Load FAILS CLOSED (``HASH_CHAIN_BROKEN``) on:

* a snapshot hash that does not reproduce from the payload;
* any hypothesis / feedback / plan / revision record whose content hash does
  not reproduce (C14 recomputation, re-raised as chain corruption);
* a hypothesis revision chain with broken status linkage, non-monotone
  windows, or a final status the chain does not end in.

``save_controller`` writes atomically (tmp file + ``os.replace``), so a
crashed write can never leave a half-written snapshot posing as a real one.

Restore equivalence (the test proves it): freeze-point snapshot -> restore
-> continue reproduces the UNINTERRUPTED run's RunSummary byte-for-byte
(same process), and a fresh subprocess restoring the same file reproduces
the identical summary hash.
"""
from __future__ import annotations

import json
import os
from typing import Dict

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued.controller import (
    FeedbackUEDController,
    WindowRecord,
)
from d052.feedback_llm_ued.feedback_contracts import CurriculumPlan
from d052.feedback_llm_ued.human_decision import HumanDecisionArtifact
from d052.feedback_llm_ued.hypothesis_ledger import (
    HypothesisLedger,
    HypothesisRecord,
)
from d052.feedback_llm_ued.plan_revision import PlanRevisionRecord
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.student_binding import TrainingStepRecord
from d052.schemas.common import is_sha256_hex

SNAPSHOT_VERSION = "feedback_llm_ued.snapshot.v1"


class SnapshotCorrupted(RuntimeError):
    """The snapshot failed hash-chain verification — refuse to restore."""


def _hash_payload(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "snapshot_hash"}
    return canonical_sha256(body)


def snapshot_controller(ctl: FeedbackUEDController) -> dict:
    """Capture the complete frozen state of ``ctl`` at a window boundary.

    Safe ONLY between windows (after a freeze or a REQUEST_CONTROL stop):
    the loop only ever mutates state inside ``_run_window``, so the captured
    state is exactly a legal resume point.
    """
    payload: Dict[str, object] = dict(
        snapshot_version=SNAPSHOT_VERSION,
        mode=ctl.mode,
        #: P0-6: the runtime grants are persisted so a restored REAL run
        #: resumes with its authorization intact (an all-false grant set
        #: restores as the historical constants-only gate)
        runtime_authorization=dict(
            real_llm_backend=(
                ctl.runtime_authorization.real_llm_backend),
            real_envcoder=ctl.runtime_authorization.real_envcoder,
            real_probe=ctl.runtime_authorization.real_probe,
            real_training=ctl.runtime_authorization.real_training),
        human_reopen_families=sorted(ctl.human_reopen_families),
        anchor_ids=list(ctl.anchor_ids),
        anchor_binding=ctl.anchor_binding,
        seeded=ctl._seeded,
        ledger=ctl.ledger.dump(),
        store=ctl.store.dump(),
        revisions=[r.model_dump() for r in ctl.revisions],
        plans={pid: plan.model_dump() for pid, plan in ctl.plans.items()},
        plans_by_window={str(w): plan.plan_id
                         for w, plan in ctl._plans_by_window.items()},
        window_feedback={str(w): list(fids)
                         for w, fids in ctl._window_feedback.items()},
        phases={str(w): phase for w, phase in ctl._phases.items()},
        sequence=ctl._sequence,
        retired_at=dict(ctl._retired_at),
        probe_calls=ctl.runner.probe_calls,
        total_transitions=ctl.runner.total_transitions,
        backend_usage=dict(real_calls=ctl.backend.usage.real_calls,
                           replay_calls=ctl.backend.usage.replay_calls,
                           mock_calls=ctl.backend.usage.mock_calls,
                           failed_calls=ctl.backend.usage.failed_calls),
        training_log=[dict(status=t.status,
                           student_training_transitions=(
                               t.student_training_transitions),
                           reason=t.reason,
                           #: P0-11: the round-trip pass flag rides in the
                           #: snapshot; pre-P0-11 snapshots lack the key and
                           #: restore as False (never attested)
                           checkpoint_round_trip_pass=(
                               t.checkpoint_round_trip_pass))
                      for t in ctl.training_log],
        envelopes=[dict(role=e.role, window=e.window, sequence=e.sequence,
                        request_hash=e.request_hash,
                        response_hash=e.response_hash,
                        prompt_sha256=e.prompt_sha256)
                   for e in ctl.envelopes],
        boards={str(w): h for w, h in ctl.board_hashes.items()},
        completed_records=[r.to_dict() for r in ctl._completed_records],
        human_decision_artifacts=[a.model_dump()
                                  for a in ctl.human_decision_artifacts],
    )
    payload["snapshot_hash"] = _hash_payload(payload)
    return payload


def _verify_hypothesis_chain(dump: dict) -> HypothesisRecord:
    """Reconstruct one ledger record and verify its revision chain.

    The model validator recomputes the content hash (C14); on top of that
    the chain itself must be internally consistent: consecutive entries link
    new_status -> previous_status, windows never decrease, every
    previous_record_hash is a legal sha256, and the record's final status is
    exactly where the chain ends.
    """
    try:
        rec = HypothesisRecord(**dump)
    except Exception as exc:
        raise SnapshotCorrupted(
            f"HASH_CHAIN_BROKEN: hypothesis {dump.get('hypothesis_id')!r} "
            f"failed content-hash recomputation: {exc}") from exc
    previous_status: object = None
    previous_window = -1
    for i, entry in enumerate(rec.revision_history):
        if not is_sha256_hex(str(entry.get("previous_record_hash", ""))):
            raise SnapshotCorrupted(
                f"HASH_CHAIN_BROKEN: hypothesis {rec.hypothesis_id!r} "
                f"revision entry {i} carries an illegal "
                f"previous_record_hash")
        window = int(entry.get("window", -1))
        if window < previous_window:
            raise SnapshotCorrupted(
                f"HASH_CHAIN_BROKEN: hypothesis {rec.hypothesis_id!r} "
                f"revision windows are not monotone at entry {i}")
        previous_window = window
        if i > 0 and entry.get("previous_status") != previous_status:
            raise SnapshotCorrupted(
                f"HASH_CHAIN_BROKEN: hypothesis {rec.hypothesis_id!r} "
                f"revision entry {i} previous_status "
                f"{entry.get('previous_status')!r} does not link to the "
                f"previous entry's new_status {previous_status!r}")
        previous_status = entry.get("new_status")
    if rec.revision_history and rec.status != previous_status:
        raise SnapshotCorrupted(
            f"HASH_CHAIN_BROKEN: hypothesis {rec.hypothesis_id!r} final "
            f"status {rec.status!r} does not match the chain's last "
            f"new_status {previous_status!r}")
    return rec


def verify_snapshot_integrity(payload: dict) -> None:
    """Recompute the snapshot hash and re-verify every chained record.

    Any mismatch raises ``SnapshotCorrupted`` with HASH_CHAIN_BROKEN — the
    snapshot is evidence, and corrupted evidence must never restore.
    """
    carried = payload.get("snapshot_hash", "")
    if payload.get("snapshot_version") != SNAPSHOT_VERSION:
        raise SnapshotCorrupted(
            f"HASH_CHAIN_BROKEN: snapshot_version "
            f"{payload.get('snapshot_version')!r} != {SNAPSHOT_VERSION!r}")
    recomputed = _hash_payload(payload)
    if recomputed != carried:
        raise SnapshotCorrupted(
            f"HASH_CHAIN_BROKEN: snapshot carried hash {carried!r} but the "
            f"payload recomputes to {recomputed!r} — the snapshot was "
            "tampered with or serialized through a non-canonical encoding")
    for dump in payload.get("ledger", []):
        _verify_hypothesis_chain(dump)
    for dump in payload.get("store", []):
        try:
            SimulatorFeedbackRecord(**dump)      # C14 recomputation
        except Exception as exc:
            raise SnapshotCorrupted(
                f"HASH_CHAIN_BROKEN: feedback {dump.get('feedback_id')!r} "
                f"failed content-hash recomputation: {exc}") from exc
    for dump in payload.get("revisions", []):
        try:
            PlanRevisionRecord(**dump)
        except Exception as exc:
            raise SnapshotCorrupted(
                f"HASH_CHAIN_BROKEN: revision {dump.get('revision_id')!r} "
                f"failed content-hash recomputation: {exc}") from exc
    for plan_id, dump in payload.get("plans", {}).items():
        try:
            CurriculumPlan(**dump)
        except Exception as exc:
            raise SnapshotCorrupted(
                f"HASH_CHAIN_BROKEN: plan {plan_id!r} failed content-hash "
                f"recomputation: {exc}") from exc
    for dump in payload.get("human_decision_artifacts", []):
        try:
            HumanDecisionArtifact(**dump)
        except Exception as exc:
            raise SnapshotCorrupted(
                f"HASH_CHAIN_BROKEN: human decision artifact "
                f"{dump.get('artifact_id')!r} failed recomputation: {exc}"
            ) from exc
    for dump in payload.get("completed_records", []):
        WindowRecord(**dump)                     # structural re-check


def restore_controller(payload: dict, *, backend=None,
                       probe_runner=None) -> FeedbackUEDController:
    """Rebuild a controller whose ``run()`` CONTINUES the snapshotted run.

    The backend/probe-runner seams are injected exactly like at construction
    (default: deterministic mock backend + symbolic runner); their USAGE
    counters are restored from the snapshot so the resumed RunSummary
    accounting is byte-identical to the uninterrupted run.
    """
    verify_snapshot_integrity(payload)
    #: P0-6: rebuild the persisted runtime grants. Absent (pre-P0-6
    #: snapshots) or all-false restores the historical constants-only
    #: gate; any real grant restores EXECUTION_MODE_REAL so a REAL run
    #: resumes with the SAME authorization — never silently downgraded.
    #: The grant set itself is hash-covered by snapshot_hash, and
    #: RealRuntimeAuthorization re-checks grant consistency fail-closed.
    grants = payload.get("runtime_authorization")
    authorization = None
    if isinstance(grants, dict) and any(bool(v) for v in grants.values()):
        authorization = RealRuntimeAuthorization(**grants)
    ctl = FeedbackUEDController(
        payload["mode"], backend=backend, probe_runner=probe_runner,
        human_reopen_families=tuple(payload.get("human_reopen_families", ())),
        runtime_authorization=authorization)
    # anchor binding: the observable state rides in the snapshot (the
    # manifest object itself is never serialized)
    ctl.anchor_ids = tuple(payload["anchor_ids"])
    ctl.anchor_binding = payload["anchor_binding"]
    ctl._seeded = bool(payload.get("seeded", True))
    # ledger (order preserved) — records already chain-verified above
    ctl.ledger = HypothesisLedger()
    for dump in payload["ledger"]:
        ctl.ledger.register(HypothesisRecord(**dump))
    ctl.store = SimulatorFeedbackStore()
    for dump in payload["store"]:
        ctl.store.add(SimulatorFeedbackRecord(**dump))
    ctl.revisions = [PlanRevisionRecord(**d) for d in payload["revisions"]]
    ctl.plans = {pid: CurriculumPlan(**dump)
                 for pid, dump in payload["plans"].items()}
    ctl._plans_by_window = {int(w): ctl.plans[pid]
                            for w, pid in payload["plans_by_window"].items()}
    ctl._window_feedback = {int(w): list(fids)
                            for w, fids in payload["window_feedback"].items()}
    ctl._phases = {int(w): phase for w, phase in payload["phases"].items()}
    ctl._sequence = int(payload["sequence"])
    ctl._retired_at = dict(payload["retired_at"])
    ctl.runner.probe_calls = int(payload["probe_calls"])
    ctl.runner.total_transitions = int(payload["total_transitions"])
    usage = payload["backend_usage"]
    ctl.backend.usage.real_calls = int(usage["real_calls"])
    ctl.backend.usage.replay_calls = int(usage["replay_calls"])
    ctl.backend.usage.mock_calls = int(usage["mock_calls"])
    ctl.backend.usage.failed_calls = int(usage["failed_calls"])
    ctl.training_log = [TrainingStepRecord(**d)
                        for d in payload["training_log"]]
    #: envelopes/boards restore as AUDIT METADATA (hash-bound identity);
    #: live envelope/BoardOutput objects only exist within their own window
    ctl.envelopes = [dict(d) for d in payload["envelopes"]]
    ctl.board_hashes = {int(w): h for w, h in payload["boards"].items()}
    ctl._completed_records = [WindowRecord(**d)
                              for d in payload["completed_records"]]
    ctl.human_decision_artifacts = [HumanDecisionArtifact(**d)
                                    for d in
                                    payload["human_decision_artifacts"]]
    return ctl


def save_controller(ctl: FeedbackUEDController, path: str) -> dict:
    """Atomic snapshot write: tmp file + ``os.replace`` (crash-safe)."""
    payload = snapshot_controller(ctl)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return payload


def load_controller(path: str, *, backend=None,
                    probe_runner=None) -> FeedbackUEDController:
    """Load + verify + restore. HASH_CHAIN_BROKEN on any tamper."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return restore_controller(payload, backend=backend,
                              probe_runner=probe_runner)
