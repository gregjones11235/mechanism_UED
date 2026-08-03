"""Strong-typed, UNBYPASSABLE LaunchGate (CC3 fix2, task §4-§8).

fix1 carried the launch decision as a plain dict whose single boolean
``training_launch_authorized`` conflated two DIFFERENT concepts:

  * STRUCTURAL batch readiness — every structural condition of the 12+4 /
    16x128=2048 batch plan holds at once;
  * DIRECTOR training authorization — the out-of-package director decision
    (TRAINING_AUTHORIZED) that permits real training at all.

This module replaces that dict with an unambiguous frozen dataclass:

    final_training_launch_authorized =
        structural_batch_ready AND director_training_authorized

and THIS ROUND ``director_training_authorized`` is ALWAYS false (the director
flag is never set inside this package), so even a structurally perfect batch
keeps ``final_training_launch_authorized = false``. A field named
``training_launch_authorized`` expressing both concepts at once is FORBIDDEN.

Hash binding (task §6): the gate carries four content hashes — batch plan,
selected descriptors, guard report, legality report. ``ProposalArchive.commit``
requires a LaunchGate (keyword-required, non-None, isinstance-verified,
version-checked) and re-verifies ALL four hashes against the CURRENT values;
any mismatch -> ARCHIVE_COMMIT_REJECTED. There is no commit path without a
gate: ``archive.commit(..., launch_gate=None)`` as a bypass interface is
FORBIDDEN — the parameter has no default and None fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from d052.bagr_ued import constants as C
from d052.bagr_ued.hashing import canonical_sha256

GATE_VERSION = "bagr_ued.launch_gate.v1"

#: CC3 fix3 (§1): strong-typed, UNBYPASSABLE launch CONTEXT version
CONTEXT_VERSION = "bagr_ued.launch_context.v1"


# ---------------------------------------------------------------------------
# current-state hash helpers — the SINGLE source of truth shared by the gate
# evaluator and the archive commit verifier (so "gate hash == current" is a
# literal recomputation, not a copy).
# ---------------------------------------------------------------------------

def compute_batch_plan_hash(batch_plan) -> str:
    return canonical_sha256(batch_plan.model_dump())


def compute_selected_descriptor_hash(selected_descriptors) -> str:
    return canonical_sha256(
        [d.model_dump() for d in sorted(selected_descriptors,
                                        key=lambda d: d.descriptor_id)])


def compute_guard_report_hash(board_out) -> str:
    #: CC3 fix3 (§10): the guard report hash additionally binds the SHARED
    #: symbolic clip batch hash the board recorded ("" for boards without
    #: one — e.g. legacy test fixtures — keeping gate/commit agreement).
    return canonical_sha256(dict(
        supervision_guard_status=board_out.supervision_guard_status,
        leakage_guard_status=board_out.leakage_guard_status,
        symbolic_clip_batch_hash=getattr(board_out,
                                         "symbolic_clip_batch_hash", "")))


def compute_legality_report_hash(legal_ids: Sequence[str],
                                 rejected: Sequence[dict]) -> str:
    return canonical_sha256(dict(
        legal_ids=sorted(legal_ids),
        rejected=sorted(rejected,
                        key=lambda r: r.get("descriptor_id", ""))))


# CC3 fix3 (§2): the two hash bindings added to the four above, so the FULL
# six-way binding (batch / descriptor / legality / guard / critic / director
# authorization) is carried by the gate and the context alike.

def compute_critic_report_hash(board_out) -> str:
    """Hash over the Critic/Skeptic envelope content (hard rejections +
    penalties + required controls). A board that ran no critic hashes the
    explicit absence — the binding is never silently skipped."""
    for e in board_out.envelopes:
        if e.role == C.ROLE_CRITIC_SKEPTIC:
            return canonical_sha256(dict(
                role=e.role,
                prompt_version=getattr(e, "prompt_version", ""),
                parsed_json=e.parsed_json))
    return canonical_sha256(dict(role=C.ROLE_CRITIC_SKEPTIC,
                                 parsed_json={}, critic_envelope="ABSENT"))


def compute_director_authorization_hash(
        director_training_authorized: bool,
        authorization_record: dict | None = None) -> str:
    """Hash binding the director authorization DECISION itself (value +
    declared source). The default source is the package constant — a real
    authorization must supply an explicit record."""
    record = dict(authorization_record) if authorization_record else dict(
        source=C.DIRECTOR_AUTHORIZATION_SOURCE_DEFAULT)
    record["director_training_authorized"] = bool(
        director_training_authorized)
    return canonical_sha256(record)


def compute_clip_batch_hash(symbolic_payloads) -> str:
    """Hash over the SHARED symbolic clip payload batch (their content
    hashes, sorted). Empty batch -> hash of the explicit empty list."""
    hashes = []
    for p in symbolic_payloads:
        dump = p.model_dump() if hasattr(p, "model_dump") and \
            not isinstance(p, dict) else p
        hashes.append(str(dump.get("clip_payload_sha256", "")))
    return canonical_sha256(sorted(hashes))


@dataclass(frozen=True)
class LaunchGate:
    """The unambiguous final batch/launch decision (CC3 fix2 §4).

    Deliberately a frozen dataclass, not a dict: strong type, no default
    construction, version-verifiable. ``final_training_launch_authorized`` is
    the ONLY authorization to treat as "may train" — and it is false this
    round by construction.
    """

    structural_batch_ready: bool
    director_training_authorized: bool
    final_training_launch_authorized: bool
    batch_plan_hash: str
    selected_descriptor_hash: str
    guard_report_hash: str
    legality_report_hash: str
    #: CC3 fix3 (§2): the critic envelope content hash and the director
    #: authorization decision hash — completing the six-way binding. Defaults
    #: keep legacy hand-construction compiling; evaluate_launch_gate ALWAYS
    #: fills them, and archive.commit re-verifies them via the LaunchContext.
    critic_report_hash: str = ""
    director_authorization_hash: str = ""
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    gate_version: str = GATE_VERSION

    def __post_init__(self) -> None:
        # the defining contract, enforced at construction so no caller can
        # ever hand out a gate that violates it
        expected = self.structural_batch_ready and \
            self.director_training_authorized
        if self.final_training_launch_authorized != expected:
            raise ValueError(
                "LAUNCH_GATE_CONTRACT_VIOLATED: final_training_launch_"
                "authorized must equal structural_batch_ready AND "
                "director_training_authorized")
        if self.gate_version != GATE_VERSION:
            raise ValueError(
                f"LAUNCH_GATE_VERSION_MISMATCH: {self.gate_version!r} != "
                f"{GATE_VERSION!r}")


def _critic_hard_reject_ids(board_out) -> set:
    """Intervention ids the Critic/Skeptic hard-rejected (task §5: a SELECTED
    proposal carrying a hard-rejected source intervention blocks readiness)."""
    for e in board_out.envelopes:
        if e.role == C.ROLE_CRITIC_SKEPTIC:
            return set(e.parsed_json.get("critic_reject_intervention_ids", []))
    return set()


def evaluate_launch_gate(budget_plan, batch_plan, selected_descriptors,
                         rejected_descriptors, board_out, *,
                         director_training_authorized: bool =
                             C.TRAINING_AUTHORIZED,
                         legal_ids=None,
                         director_authorization_record: dict | None = None
                         ) -> LaunchGate:
    """Evaluate the structural batch gate + director authorization.

    ``selected_descriptors`` = the legal descriptors whose ids are the budget
    plan's UED slots; ``rejected_descriptors`` = LegalityGate rejection
    records. CC3 fix2 §15-§16: UNSELECTED illegal proposals are recorded but
    DO NOT block a structurally-satisfied LEGAL batch — only selected-side
    violations block (selected descriptor illegal, selector referencing a
    rejected candidate, fewer than 12 legal candidates, ...).

    CC3 fix3 (§2): the gate now carries the FULL six-way hash binding — the
    four fix2 hashes plus the critic envelope hash and the director
    authorization decision hash.
    """
    reasons: List[str] = []
    selected_ids = list(budget_plan.ued_slots)
    rejected_ids = sorted({r.get("descriptor_id", "")
                           for r in rejected_descriptors})
    legal_ids = sorted(legal_ids) if legal_ids is not None else \
        sorted(d.descriptor_id for d in selected_descriptors)

    # -- budget plan status / shortfall -------------------------------------
    if budget_plan.status != "OK":
        reasons.append(f"budget_plan_status={budget_plan.status}")
    if budget_plan.shortfall_note:
        reasons.append(f"unresolved_shortfall: {budget_plan.shortfall_note}")

    # -- slot / anchor structure --------------------------------------------
    if len(selected_ids) != C.UED_ACTIVE_SLOTS:
        reasons.append(f"selected_ued_slots={len(selected_ids)} "
                       f"!= {C.UED_ACTIVE_SLOTS}")
    if len(set(selected_ids)) != len(selected_ids):
        reasons.append("duplicate_selected_ued_slots_forbidden")
    if list(budget_plan.anchor_slots) != list(C.GLOBAL_CANONICAL_ANCHOR_IDS):
        reasons.append(
            f"canonical_anchor_slots={list(budget_plan.anchor_slots)} "
            f"!= the {C.GLOBAL_CANONICAL_ANCHORS} fixed global anchors")

    # -- batch arithmetic ----------------------------------------------------
    if batch_plan.num_envs != C.NUM_ENVS:
        reasons.append(f"total_envs={batch_plan.num_envs} != {C.NUM_ENVS}")
    if batch_plan.rollout_length != C.ROLLOUT_LENGTH:
        reasons.append(f"rollout_length={batch_plan.rollout_length} "
                       f"!= {C.ROLLOUT_LENGTH}")
    if batch_plan.transitions_per_update != C.TRANSITIONS_PER_UPDATE:
        reasons.append(
            f"transitions_per_update={batch_plan.transitions_per_update} "
            f"!= {C.TRANSITIONS_PER_UPDATE}")

    # -- legality (SELECTED side only — §15/§16 semantics) --------------------
    inter = sorted(set(selected_ids) & set(rejected_ids))
    if inter:
        reasons.append(
            f"selected_descriptor_illegal: selector references rejected "
            f"candidate(s) {inter}")
    missing_legal = sorted(set(selected_ids) - set(legal_ids))
    if missing_legal:
        reasons.append(
            f"selected_descriptor_without_legality_evidence: {missing_legal}")
    if len(legal_ids) < C.UED_ACTIVE_SLOTS and len(inter) == 0 \
            and not missing_legal:
        reasons.append(
            f"legal_candidates={len(legal_ids)} < {C.UED_ACTIVE_SLOTS}")

    # -- selected descriptors consistent with the recorded selection ---------
    sel_desc_ids = sorted(d.descriptor_id for d in selected_descriptors)
    if sel_desc_ids != sorted(set(selected_ids)):
        reasons.append(
            f"selected_descriptors_inconsistent_with_budget_slots: "
            f"descriptors={sel_desc_ids} slots={sorted(set(selected_ids))}")

    # -- critic hard reject on a SELECTED proposal ----------------------------
    critic_rejects = _critic_hard_reject_ids(board_out)
    for d in selected_descriptors:
        src = set(d.provenance.get("source_intervention_ids", []))
        hit = sorted(src & critic_rejects)
        if hit:
            reasons.append(
                f"selected_proposal_critic_hard_reject: "
                f"{d.descriptor_id} <- intervention(s) {hit}")

    # -- unresolved guard violations ------------------------------------------
    if not (board_out.supervision_guard_status == "PASS"
            and board_out.leakage_guard_status == "PASS"):
        reasons.append(
            f"unresolved_guard_violation: supervision="
            f"{board_out.supervision_guard_status} leakage="
            f"{board_out.leakage_guard_status}")

    # -- CC3 fix3 (§2): the critic envelope must EXIST to be hash-bound ------
    critic_report_hash = compute_critic_report_hash(board_out)
    if not any(e.role == C.ROLE_CRITIC_SKEPTIC for e in board_out.envelopes):
        reasons.append(
            "missing_critic_envelope: the critic_report_hash cannot bind a "
            "board that never ran the Critic/Skeptic")
    director_authorization_hash = compute_director_authorization_hash(
        director_training_authorized, director_authorization_record)

    # -- required provenance / hashes present ---------------------------------
    batch_plan_hash = compute_batch_plan_hash(batch_plan)
    selected_descriptor_hash = compute_selected_descriptor_hash(
        selected_descriptors)
    guard_report_hash = compute_guard_report_hash(board_out)
    legality_report_hash = compute_legality_report_hash(legal_ids,
                                                        rejected_descriptors)
    for name, h in (("batch_plan_hash", batch_plan_hash),
                    ("selected_descriptor_hash", selected_descriptor_hash),
                    ("guard_report_hash", guard_report_hash),
                    ("legality_report_hash", legality_report_hash),
                    ("critic_report_hash", critic_report_hash),
                    ("director_authorization_hash",
                     director_authorization_hash)):
        if not (isinstance(h, str) and len(h) == 64):
            reasons.append(f"missing_required_provenance_hash: {name}")

    structural = not reasons
    director = bool(director_training_authorized)
    return LaunchGate(
        structural_batch_ready=structural,
        director_training_authorized=director,
        final_training_launch_authorized=structural and director,
        batch_plan_hash=batch_plan_hash,
        selected_descriptor_hash=selected_descriptor_hash,
        guard_report_hash=guard_report_hash,
        legality_report_hash=legality_report_hash,
        critic_report_hash=critic_report_hash,
        director_authorization_hash=director_authorization_hash,
        reasons=tuple(reasons),
        gate_version=GATE_VERSION)


# ---------------------------------------------------------------------------
# CC3 fix3 (§1): the strong-typed, UNOMITTABLE LaunchContext
# ---------------------------------------------------------------------------

#: the six structural/authorization conditions the context binds. The final
#: authorization is their conjunction — computed, never supplied.
_CONTEXT_CONDITIONS = (
    "structural_batch_ready",
    "review_certificate_valid",
    "provenance_valid",
    "guards_passed",
    "simulator_probe_complete",
    "selection_complete",
    "director_training_authorized",
)


@dataclass(frozen=True)
class LaunchContext:
    """The full-window launch decision state (CC3 fix3 §1).

    Where the LaunchGate binds the BATCH structure, the LaunchContext binds
    the whole review-window state around it: the review certificate, the
    provenance chain, the guard verdicts, the simulator probe completion and
    the selection completion — plus the director authorization. It is a
    frozen dataclass with NO default construction of the final flag:

        final_training_launch_authorized = AND of ALL seven conditions

    is enforced in __post_init__, so no caller can ever hand out a context
    whose final flag disagrees with its conditions. ``archive.commit`` and
    ``archive.refresh(dry_run=False)`` REQUIRE it alongside the LaunchGate —
    there is no gate-only commit path (CC3 fix3 §3).
    """

    structural_batch_ready: bool
    review_certificate_valid: bool
    provenance_valid: bool
    guards_passed: bool
    simulator_probe_complete: bool
    selection_complete: bool
    director_training_authorized: bool
    final_training_launch_authorized: bool
    #: the FULL six-way hash binding (identical to the gate's)
    batch_plan_hash: str
    selected_descriptor_hash: str
    legality_report_hash: str
    guard_report_hash: str
    critic_report_hash: str
    director_authorization_hash: str
    #: hash of the SHARED symbolic clip payload batch the board and the
    #: certificate both consumed (CC3 fix3 §10)
    clip_batch_hash: str
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    context_version: str = CONTEXT_VERSION

    def __post_init__(self) -> None:
        expected = all(getattr(self, name) for name in _CONTEXT_CONDITIONS)
        if self.final_training_launch_authorized != expected:
            raise ValueError(
                "LAUNCH_CONTEXT_CONTRACT_VIOLATED: final_training_launch_"
                "authorized must equal the conjunction of "
                f"{list(_CONTEXT_CONDITIONS)}")
        if self.context_version != CONTEXT_VERSION:
            raise ValueError(
                f"LAUNCH_CONTEXT_VERSION_MISMATCH: {self.context_version!r} "
                f"!= {CONTEXT_VERSION!r}")
        for name in ("batch_plan_hash", "selected_descriptor_hash",
                     "legality_report_hash", "guard_report_hash",
                     "critic_report_hash", "director_authorization_hash",
                     "clip_batch_hash"):
            value = getattr(self, name)
            if not (isinstance(value, str) and len(value) == 64):
                raise ValueError(
                    f"LAUNCH_CONTEXT_HASH_INVALID: {name} must be a 64-char "
                    f"sha256 hex digest, got {value!r}")


def evaluate_launch_context(gate: LaunchGate, board_out, *,
                            review_certificate_valid: bool = False,
                            provenance_valid: bool = False,
                            simulator_probe_complete: bool = False,
                            selection_complete: bool = False,
                            symbolic_payloads=(),
                            extra_reasons: Sequence[str] = ()) -> LaunchContext:
    """Assemble the strong-typed LaunchContext from the gate + window state.

    Every extra condition DEFAULTS FALSE (fail-closed): a dry run / any path
    that has not positively established the review certificate, provenance,
    simulator probes and selection cannot reach final authorization. The
    six hashes are carried over from the gate — the gate and the context are
    ONE binding, never two divergent records.
    """
    if not isinstance(gate, LaunchGate):
        raise AssertionError(
            "LAUNCH_CONTEXT_REQUIRES_GATE: evaluate_launch_context needs a "
            f"strong-typed LaunchGate, got {type(gate).__name__!r}")
    guards_passed = (getattr(board_out, "supervision_guard_status", "") ==
                     "PASS" and
                     getattr(board_out, "leakage_guard_status", "") == "PASS")
    reasons: List[str] = list(gate.reasons)
    if not review_certificate_valid:
        reasons.append("review_certificate_not_established")
    if not provenance_valid:
        reasons.append("provenance_chain_not_established")
    if not guards_passed:
        reasons.append("guards_not_passed")
    if not simulator_probe_complete:
        reasons.append("simulator_probe_incomplete")
    if not selection_complete:
        reasons.append("selection_incomplete")
    reasons.extend(extra_reasons)
    return LaunchContext(
        structural_batch_ready=gate.structural_batch_ready,
        review_certificate_valid=bool(review_certificate_valid),
        provenance_valid=bool(provenance_valid),
        guards_passed=bool(guards_passed),
        simulator_probe_complete=bool(simulator_probe_complete),
        selection_complete=bool(selection_complete),
        director_training_authorized=gate.director_training_authorized,
        final_training_launch_authorized=False if not (
            gate.structural_batch_ready and review_certificate_valid and
            provenance_valid and guards_passed and simulator_probe_complete
            and selection_complete and gate.director_training_authorized
        ) else True,
        batch_plan_hash=gate.batch_plan_hash,
        selected_descriptor_hash=gate.selected_descriptor_hash,
        legality_report_hash=gate.legality_report_hash,
        guard_report_hash=gate.guard_report_hash,
        critic_report_hash=gate.critic_report_hash,
        director_authorization_hash=gate.director_authorization_hash,
        clip_batch_hash=compute_clip_batch_hash(symbolic_payloads),
        reasons=tuple(reasons),
        context_version=CONTEXT_VERSION)
