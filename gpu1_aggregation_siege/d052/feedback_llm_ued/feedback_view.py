"""FeedbackView — the ONLY surface through which the Review Board touches
probe feedback (structural basis of the comparison-mode isolation).

The board never receives a store. It receives a view:

* ``NormalFeedbackView``   — read-only frozen snapshot of explicitly injected
                             records (the controller selects EXACTLY window
                             k-1 — the CC3 C9 gate's exact lag);
* ``NullFeedbackView``     — STRUCTURAL blocking: the type holds no store, no
                             records and no ids — feedback is unreachable by
                             construction, not by prompt omission (the
                             legacy empty control; superseded for the static
                             mode by the shape-matched mask below);
* ``MaskedFeedbackView``   — P0-12 shape-matched no-feedback control
                             (static mode): presents EXACTLY the window k-1
                             record set (same item count, same prompt field
                             set) with every value replaced by a controlled
                             NULL/MASK value — the board runs the same
                             computation with no feedback content;
* ``PermutedFeedbackView`` — frozen recomputable permutation with anonymized
                             identities (shuffled mode): the permutation
                             derives ONLY from (mode, board window, window
                             scope, seed_schedule_hash) — never from runtime
                             randomness — and every record is shown under an
                             anonymized id with EVERY identity side channel
                             removed or consistently anonymized at the prompt
                             layer AND the evidence layer: candidate id /
                             mutation axes / axis values / held-constant axes
                             masked; the redundant family-grain predicted
                             signature dropped; the numeric fields — exact
                             probe rates are deterministic per-candidate-hash
                             fingerprints, so they are a pairing channel when
                             joined against the honest store — published ONLY
                             as per-family window aggregates (a public
                             function of the (window, family) partition,
                             identical in both layers, gaps and severity
                             rebuilt from the aggregates). The real
                             candidate<->feedback pairing is therefore
                             unrecoverable from the board context; the only
                             de-anonymization path is ``resolve_citation``,
                             held by the controller.

Every view exposes the same prompt payload shape, so the six roles are
identical across modes — only what they can see differs. Citations coming
back out of the board are de-anonymized through ``resolve_citation`` — the
ONLY authorized mapping path, held by the controller (the honest
bookkeeper), never by a board role.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Sequence, Tuple, runtime_checkable

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import (
    SEVERITY_NONE,
    BehaviorFailureEvidence,
    severity_for,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
)
from d052.schemas.common import is_sha256_hex

VIEW_LABEL_NORMAL = "normal"
VIEW_LABEL_NULL = "null"
VIEW_LABEL_MASKED = "masked"
VIEW_LABEL_PERMUTED = "permuted"

#: sentinel standing in for every masked identity-bearing field
MASKED_IDENTITY = "MASKED_IDENTITY"

#: P0-12: the controlled NULL/MASK values of the shape-matched no-feedback
#: control. Every record field is replaced by a fixed, content-free value:
#: NEUTRAL is the only match state that carries no direction and is not
#: counted as ungraded (an ungraded control board would escalate the
#: critic); 0.0 rates / empty collections carry no probe content.
MASKED_MATCH_STATE = C.MATCH_DIRECTION_NEUTRAL
MASKED_RATE = 0.0


@runtime_checkable
class FeedbackView(Protocol):
    label: str
    window_scope: int

    def records(self) -> List[SimulatorFeedbackRecord]:
        ...

    def to_prompt_payload(self) -> List[dict]:
        ...

    def behavior_evidence(self) -> List[BehaviorFailureEvidence]:
        """The BoardContext evidence layer, derived ONLY from the records the
        view presents (CC3 C9 gate: the board context is built from the view,
        never from the raw store). Null view: empty. Normal view: real-id
        evidence. Permuted view: anonymized evidence CONSISTENT with the
        prompt payload — the same anonymized feedback ids, candidate id
        masked, and the SAME family-level window aggregates for every
        numeric field (rates, gaps, severity): the evidence layer carries
        no identity side channel the prompt layer does not.
        """
        ...

    def resolve_citation(self, citation: str) -> str:
        """Map a board citation back to a SimulatorFeedbackStore id.

        The ONLY authorized de-anonymization path. Views fail closed on any
        citation they did not present.
        """
        ...


def _assert_exact_window(records: Sequence[SimulatorFeedbackRecord],
                         window_scope: int) -> None:
    """CC3 C9 gate (defense in depth): a view scoped to window w presents
    EXACTLY window-w records. Older, current and future records all fail
    closed as STALE_FEEDBACK_ID — the window lag is exactly one window."""
    for record in records:
        if record.window != window_scope:
            raise ValueError(
                f"STALE_FEEDBACK_ID: record {record.feedback_id!r} is from "
                f"window {record.window}; a view scoped to window "
                f"{window_scope} presents EXACTLY that window's frozen "
                f"feedback (older/current/future records fail closed)")


def record_payload(record: SimulatorFeedbackRecord) -> Dict[str, object]:
    """The coarse, episode-level slice of one record the board may see.

    Environment-level configuration (mutation axes / axis values / held axes)
    is included — it is TaskParams-level information, never an action-
    guidance carrier. Probe metrics are reduced to the two episode-level
    success rates.
    """
    metrics = record.stage2_metrics or record.stage1_metrics
    return dict(
        feedback_id=record.feedback_id,
        candidate_id=record.candidate_id,
        window=record.window,
        environment_family=record.environment_family,
        mutation_axes=list(record.mutation_axes),
        axis_values=dict(record.axis_values),
        held_constant_axes=dict(record.held_constant_axes),
        distinguishes_hypothesis_ids=list(record.distinguishes_hypothesis_ids),
        expected_observed_match=record.expected_observed_match,
        expected_signature=dict(record.expected_signature),
        student_success_rate=(metrics.student_success_rate
                              if metrics is not None else 0.0),
        reference_success_rate=(metrics.reference_success_rate
                                if metrics is not None else 0.0))


class NormalFeedbackView:
    """Read-only frozen snapshot of explicitly injected feedback records."""

    label = VIEW_LABEL_NORMAL

    def __init__(self, records: Sequence[SimulatorFeedbackRecord], *,
                 window_scope: int) -> None:
        if window_scope < 0:
            raise ValueError(f"ILLEGAL_VIEW_WINDOW_SCOPE: {window_scope}")
        _assert_exact_window(records, window_scope)
        self.window_scope = window_scope
        #: sorted + tuple: immutable snapshot, deterministic order
        self._records = tuple(sorted(records, key=lambda r: r.feedback_id))
        self._ids = frozenset(r.feedback_id for r in self._records)

    @classmethod
    def from_store(cls, store, *,
                   evidence_window: int) -> "NormalFeedbackView":
        """CC3 C9 gate: the view presents EXACTLY ``evidence_window``'s
        frozen records (window k's board reads window k-1 and nothing else).
        """
        records = [r for r in store.all() if r.window == evidence_window]
        return cls(records, window_scope=evidence_window)

    def records(self) -> List[SimulatorFeedbackRecord]:
        return list(self._records)

    def to_prompt_payload(self) -> List[dict]:
        return [record_payload(r) for r in self._records]

    def behavior_evidence(self) -> List[BehaviorFailureEvidence]:
        return [BehaviorFailureEvidence.from_record(r)
                for r in self._records]

    def resolve_citation(self, citation: str) -> str:
        if citation not in self._ids:
            raise ValueError(
                f"UNKNOWN_FEEDBACK_CITATION: {citation!r} is not a record "
                f"this view presents (window scope {self.window_scope})")
        return citation


class NullFeedbackView:
    """STRUCTURAL no-feedback view (static-no-feedback mode).

    Holds no store reference, no record list, no id index — there is nothing
    inside this object that could leak feedback, so isolation does not depend
    on prompt construction discipline. The constructor takes NO arguments:
    the type cannot even be handed a store.
    """

    label = VIEW_LABEL_NULL
    window_scope = -1

    def records(self) -> List[SimulatorFeedbackRecord]:
        return []

    def to_prompt_payload(self) -> List[dict]:
        return []

    def behavior_evidence(self) -> List[BehaviorFailureEvidence]:
        return []

    def resolve_citation(self, citation: str) -> str:
        raise ValueError(
            f"NULL_VIEW_HAS_NO_FEEDBACK: {citation!r} cannot be resolved — "
            f"the static mode's feedback view is structurally empty")


def _masked_payload(masked_id: str, record: SimulatorFeedbackRecord
                    ) -> Dict[str, object]:
    """P0-12: one record's board-visible slice under the shape-matched
    mask — the EXACT field set of :func:`record_payload`, every value
    replaced by its controlled NULL/MASK value. The board prompt of the
    no-feedback control is therefore byte-identical in SHAPE to the
    normal mode's prompt (same item count, same keys) and carries no
    feedback CONTENT."""
    return dict(
        feedback_id=masked_id,
        candidate_id=MASKED_IDENTITY,
        window=record.window,
        environment_family=MASKED_IDENTITY,
        mutation_axes=[],
        axis_values={},
        held_constant_axes={},
        distinguishes_hypothesis_ids=[],
        expected_observed_match=MASKED_MATCH_STATE,
        expected_signature={},
        student_success_rate=MASKED_RATE,
        reference_success_rate=MASKED_RATE)


def _masked_evidence(masked_id: str, record: SimulatorFeedbackRecord
                     ) -> BehaviorFailureEvidence:
    """P0-12: BoardContext evidence for one masked record — the SAME
    anonymized id the prompt payload shows, every numeric field the
    controlled null (0.0 / severity none), every identity masked. The
    evidence layer carries no channel the prompt layer does not."""
    return BehaviorFailureEvidence(
        feedback_id=masked_id,
        candidate_id=MASKED_IDENTITY,
        window=record.window,
        environment_family=MASKED_IDENTITY,
        student_success_rate=MASKED_RATE,
        reference_success_rate=MASKED_RATE,
        return_shortfall=MASKED_RATE,
        behavior_activation_gap=MASKED_RATE,
        front_progress_gap=MASKED_RATE,
        reference_gap=MASKED_RATE,
        severity=SEVERITY_NONE,
        expected_observed_match=MASKED_MATCH_STATE)


class MaskedFeedbackView:
    """P0-12: the SHAPE-MATCHED no-feedback control (static mode).

    Presents EXACTLY the window k-1 records the normal mode would see —
    the same item count, the same deterministic order, the same prompt
    field set — but every value is replaced by its controlled NULL/MASK
    value (``_masked_payload`` / ``_masked_evidence``): ids and family
    masked, axes / signatures / hypothesis bindings emptied, match state
    neutral, rates zero. The board therefore runs the SAME computation
    (six roles, same prompt shape, same item count, same lifecycle
    queries) with NO feedback content — the honest no-feedback control
    the comparison demands (an empty view changed the prompt shape and
    could not isolate feedback's contribution).

    * ``records()`` still presents the wrapped records for the pooled-
      episode COMPUTE aggregate only (episode counts are compute, matched
      across modes by construction); no content derived from them ever
      reaches a prompt;
    * ``resolve_citation`` fails closed for every citation — the control
      can never act on feedback (a verdict citing a masked id has no
      referent; the board's mock rules derive nothing from masked
      content, so no citation is produced).
    """

    label = VIEW_LABEL_MASKED

    def __init__(self, records: Sequence[SimulatorFeedbackRecord], *,
                 window_scope: int, board_window: int) -> None:
        if window_scope < 0:
            raise ValueError(f"ILLEGAL_VIEW_WINDOW_SCOPE: {window_scope}")
        if board_window < 0:
            raise ValueError(f"ILLEGAL_VIEW_BOARD_WINDOW: {board_window}")
        _assert_exact_window(records, window_scope)
        self.window_scope = window_scope
        self.board_window = board_window
        ordered = sorted(records, key=lambda r: r.feedback_id)
        self._records = tuple(ordered)
        self._payloads: List[Dict[str, object]] = []
        self._evidence: List[BehaviorFailureEvidence] = []
        for slot, record in enumerate(ordered):
            masked_id = f"masked-w{board_window:02d}-{slot:03d}"
            self._payloads.append(_masked_payload(masked_id, record))
            self._evidence.append(_masked_evidence(masked_id, record))

    @property
    def masked_ids(self) -> Tuple[str, ...]:
        return tuple(p["feedback_id"] for p in self._payloads)

    def records(self) -> List[SimulatorFeedbackRecord]:
        """The wrapped records — pooled-episode COMPUTE aggregate only;
        their content never reaches a prompt payload or evidence item."""
        return list(self._records)

    def to_prompt_payload(self) -> List[dict]:
        return [dict(p) for p in self._payloads]

    def behavior_evidence(self) -> List[BehaviorFailureEvidence]:
        return list(self._evidence)

    def resolve_citation(self, citation: str) -> str:
        raise ValueError(
            f"MASKED_VIEW_CITATION_NOT_RESOLVABLE: {citation!r} cannot be "
            "resolved — the shape-matched mask presents no feedback "
            "content; the no-feedback control can never act on feedback")


def _permutation_unit(seed: str, index: int) -> float:
    """Hash-derived pseudo-uniform value in [0, 1) — frozen, no RNG."""
    digest = canonical_sha256(dict(seed=seed, index=index))
    return int(digest[:16], 16) / (16 ** 16)


def family_level_metrics(records: Sequence[SimulatorFeedbackRecord]
                         ) -> Dict[str, Dict[str, object]]:
    """CC4 C9 gate: the shuffled view's numeric anonymization.

    Exact probe rates and the gaps derived from them are deterministic
    functions of the candidate hash (see ``simulator_probe``), so at full
    precision they are per-candidate fingerprints: anyone holding the honest
    store could join a shuffled payload item against it and recover the true
    candidate<->feedback pairing despite the anonymized ids. The permuted
    view therefore publishes ONLY per-family window aggregates — the mean
    Student/Reference success rates, the mean activation / front-progress
    gaps, the return shortfall and reference gap rebuilt from those
    aggregates, and the severity of that rebuilt gap.

    These aggregates are a deterministic function of the PUBLIC (window,
    family) partition — strictly less information than the family and window
    the payload shows anyway plus the honest store any joining adversary
    already holds — so they add NO identifying power: within one family of
    one window every record publishes byte-identical numbers. The same
    aggregates feed the prompt payload and the evidence layer (consistent
    anonymization at both layers).
    """
    by_family: Dict[str, List[SimulatorFeedbackRecord]] = {}
    for record in records:
        by_family.setdefault(record.environment_family, []).append(record)
    aggregates: Dict[str, Dict[str, object]] = {}
    for family in sorted(by_family):
        exact = [BehaviorFailureEvidence.from_record(r)
                 for r in by_family[family]]
        n = len(exact)
        student = round(sum(e.student_success_rate for e in exact) / n, 6)
        reference = round(sum(e.reference_success_rate for e in exact) / n, 6)
        activation = round(sum(e.behavior_activation_gap for e in exact) / n,
                           6)
        progress = round(sum(e.front_progress_gap for e in exact) / n, 6)
        shortfall = round(max(0.0, reference - student), 6)
        gap = round(max(shortfall, activation, progress), 6)
        aggregates[family] = dict(
            student_success_rate=student,
            reference_success_rate=reference,
            behavior_activation_gap=activation,
            front_progress_gap=progress,
            return_shortfall=shortfall,
            reference_gap=gap,
            severity=severity_for(gap))
    return aggregates


def _anonymized_payload(anon_id: str, record: SimulatorFeedbackRecord,
                        coarse: Dict[str, Dict[str, object]]
                        ) -> Dict[str, object]:
    """One record's board-visible slice under an anonymized identity.

    Every identity side channel is removed or consistently anonymized:
    candidate id / mutation axes / axis values / held-constant axes are
    masked; the family-grain predicted signature is dropped (redundant with
    the visible family-level hypotheses, identity-correlated at family
    granularity); the exact episode-level rates — per-candidate-hash
    fingerprints — are replaced by the family-level window aggregates
    (``family_level_metrics``), identical to the evidence layer's numbers.
    What moves together with the record is loop-essential coarse content
    only: window, family, distinguished hypotheses (family granularity),
    expected-vs-observed match state.
    """
    payload = record_payload(record)
    payload["feedback_id"] = anon_id
    payload["candidate_id"] = MASKED_IDENTITY
    payload["mutation_axes"] = []
    payload["axis_values"] = {}
    payload["held_constant_axes"] = {}
    payload["expected_signature"] = {}
    fam = coarse[record.environment_family]
    payload["student_success_rate"] = fam["student_success_rate"]
    payload["reference_success_rate"] = fam["reference_success_rate"]
    return payload


def _anonymized_evidence(anon_id: str, record: SimulatorFeedbackRecord,
                         coarse: Dict[str, Dict[str, object]]
                         ) -> BehaviorFailureEvidence:
    """BoardContext evidence for one permuted record under its anonymized id.

    CC3 C9 gate + CC4 hardening: the evidence layer carries NO identity side
    channel and NO identifying numeric channel — the feedback id is the SAME
    anonymized id the prompt payload shows (evidence<->payload consistency),
    the candidate id is masked, and EVERY numeric field (both success rates,
    all three gaps, reference gap, severity) is the SAME family-level window
    aggregate the prompt payload publishes (``family_level_metrics``). What
    moves together with the record is loop-essential coarse content only:
    window, family, match state.
    """
    evidence = BehaviorFailureEvidence.from_record(record)
    fam = coarse[record.environment_family]
    payload = evidence.model_dump()
    payload["feedback_id"] = anon_id
    payload["candidate_id"] = MASKED_IDENTITY
    payload["student_success_rate"] = fam["student_success_rate"]
    payload["reference_success_rate"] = fam["reference_success_rate"]
    payload["return_shortfall"] = fam["return_shortfall"]
    payload["behavior_activation_gap"] = fam["behavior_activation_gap"]
    payload["front_progress_gap"] = fam["front_progress_gap"]
    payload["reference_gap"] = fam["reference_gap"]
    payload["severity"] = fam["severity"]
    return BehaviorFailureEvidence(**payload)


class PermutedFeedbackView:
    """Frozen recomputable anonymized permutation (shuffled mode).

    Construction is a pure function of (records, mode, board_window,
    window_scope, seed_schedule_hash):

    * records are sorted by feedback_id (canonical order);
    * ``permutation_seed = canonical_sha256({mode, board_window,
      window_scope, seed_schedule_hash, purpose})`` — derived ONLY from the
      frozen seed schedule, never from runtime randomness;
    * slot ``j`` presents the record at permuted position ``order[j]`` (the
      indices sorted by hash-derived pseudo-uniform keys) under the
      anonymized id ``anon-w{board_window:02d}-{j:03d}``;
    * every identity side channel is removed or consistently anonymized at
      BOTH the prompt layer and the evidence layer (``_anonymized_payload``
      / ``_anonymized_evidence``): ids/axes masked, family-grain predicted
      signature dropped, exact per-candidate rates and gaps replaced by the
      shared per-family window aggregates (``family_level_metrics``).

    Two views constructed with identical inputs present bit-identical
    payloads, evidence and mappings (recomputable); the real
    candidate<->feedback pairing never appears in any payload, and no
    published number refines the public (window, family) partition
    (negative-tested: uniqueness/re-identification tests).
    """

    def __init__(self, records: Sequence[SimulatorFeedbackRecord], *,
                 window_scope: int, board_window: int, mode: str,
                 seed_schedule_hash: str) -> None:
        if window_scope < 0:
            raise ValueError(f"ILLEGAL_VIEW_WINDOW_SCOPE: {window_scope}")
        if board_window < 0:
            raise ValueError(f"ILLEGAL_VIEW_BOARD_WINDOW: {board_window}")
        if mode != C.MODE_SHUFFLED_FEEDBACK:
            raise ValueError(
                f"PERMUTED_VIEW_REQUIRES_SHUFFLED_MODE: {mode!r}")
        if not is_sha256_hex(seed_schedule_hash):
            raise ValueError(
                f"ILLEGAL_SEED_SCHEDULE_HASH: {seed_schedule_hash!r}")
        _assert_exact_window(records, window_scope)
        self.window_scope = window_scope
        self.board_window = board_window

        ordered = sorted(records, key=lambda r: r.feedback_id)
        self._permutation_seed = canonical_sha256(dict(
            mode=mode,
            board_window=board_window,
            window_scope=window_scope,
            seed_schedule_hash=seed_schedule_hash,
            purpose="permuted_feedback_view.v1"))
        order = sorted(
            range(len(ordered)),
            key=lambda i: (_permutation_unit(self._permutation_seed, i), i))
        self._records = tuple(ordered[i] for i in order)

        #: CC4 C9 gate: one set of family-level window aggregates, shared by
        #: the prompt payload and the evidence layer (consistent
        #: anonymization; a pure function of the presented record set).
        self._coarse = family_level_metrics(records)

        self._anon_to_real: Dict[str, str] = {}
        self._payloads: List[Dict[str, object]] = []
        self._evidence: List[BehaviorFailureEvidence] = []
        for slot, record in enumerate(self._records):
            anon_id = f"anon-w{board_window:02d}-{slot:03d}"
            self._anon_to_real[anon_id] = record.feedback_id
            self._payloads.append(
                _anonymized_payload(anon_id, record, self._coarse))
            self._evidence.append(
                _anonymized_evidence(anon_id, record, self._coarse))
        self.label = f"{VIEW_LABEL_PERMUTED}:{self._permutation_seed[:16]}"

    @property
    def permutation_seed(self) -> str:
        return self._permutation_seed

    def records(self) -> List[SimulatorFeedbackRecord]:
        return list(self._records)

    def to_prompt_payload(self) -> List[dict]:
        return [dict(p) for p in self._payloads]

    def behavior_evidence(self) -> List[BehaviorFailureEvidence]:
        """Anonymized evidence — the BoardContext built from this view never
        carries a real feedback id or candidate id (no identity side
        channel at the evidence layer)."""
        return list(self._evidence)

    def resolve_citation(self, citation: str) -> str:
        try:
            return self._anon_to_real[citation]
        except KeyError:
            raise ValueError(
                f"UNKNOWN_ANONYMIZED_CITATION: {citation!r} is not an "
                f"anonymized feedback id presented by this window's "
                f"permuted view") from None
