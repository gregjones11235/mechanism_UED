"""FeedbackView — the ONLY surface through which the Review Board touches
probe feedback (structural basis of the comparison-mode isolation).

The board never receives a store. It receives a view:

* ``NormalFeedbackView``   — read-only frozen snapshot of explicitly injected
                             records (the controller selects window <= k-1);
* ``NullFeedbackView``     — STRUCTURAL blocking: the type holds no store, no
                             records and no ids — feedback is unreachable by
                             construction, not by prompt omission (static
                             mode; the board context built from it carries a
                             zero feedback payload);
* ``PermutedFeedbackView`` — frozen recomputable permutation with anonymized
                             identities (shuffled mode): the permutation
                             derives ONLY from (mode, board window, window
                             scope, seed_schedule_hash) — never from runtime
                             randomness — and every record is shown under an
                             anonymized id with its identity side channels
                             (candidate id / mutation axes / axis values /
                             held-constant axes) masked, so the real
                             candidate<->feedback pairing is unrecoverable
                             from the board context.

Every view exposes the same prompt payload shape, so the six roles are
identical across modes — only what they can see differs. Citations coming
back out of the board are de-anonymized through ``resolve_citation`` — the
ONLY authorized mapping path, held by the controller (the honest
bookkeeper), never by a board role.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Sequence, runtime_checkable

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
)
from d052.schemas.common import is_sha256_hex

VIEW_LABEL_NORMAL = "normal"
VIEW_LABEL_NULL = "null"
VIEW_LABEL_PERMUTED = "permuted"

#: sentinel standing in for every masked identity-bearing field
MASKED_IDENTITY = "MASKED_IDENTITY"


@runtime_checkable
class FeedbackView(Protocol):
    label: str
    window_scope: int

    def records(self) -> List[SimulatorFeedbackRecord]:
        ...

    def to_prompt_payload(self) -> List[dict]:
        ...

    def resolve_citation(self, citation: str) -> str:
        """Map a board citation back to a SimulatorFeedbackStore id.

        The ONLY authorized de-anonymization path. Views fail closed on any
        citation they did not present.
        """
        ...


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
        self.window_scope = window_scope
        #: sorted + tuple: immutable snapshot, deterministic order
        self._records = tuple(sorted(records, key=lambda r: r.feedback_id))
        self._ids = frozenset(r.feedback_id for r in self._records)

    @classmethod
    def from_store(cls, store, *, max_window: int) -> "NormalFeedbackView":
        records = [r for r in store.all() if r.window <= max_window]
        return cls(records, window_scope=max_window)

    def records(self) -> List[SimulatorFeedbackRecord]:
        return list(self._records)

    def to_prompt_payload(self) -> List[dict]:
        return [record_payload(r) for r in self._records]

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

    def resolve_citation(self, citation: str) -> str:
        raise ValueError(
            f"NULL_VIEW_HAS_NO_FEEDBACK: {citation!r} cannot be resolved — "
            f"the static mode's feedback view is structurally empty")


def _permutation_unit(seed: str, index: int) -> float:
    """Hash-derived pseudo-uniform value in [0, 1) — frozen, no RNG."""
    digest = canonical_sha256(dict(seed=seed, index=index))
    return int(digest[:16], 16) / (16 ** 16)


def _anonymized_payload(anon_id: str,
                        record: SimulatorFeedbackRecord) -> Dict[str, object]:
    """One record's board-visible slice under an anonymized identity.

    The identity side channels (candidate id / mutation axes / axis values /
    held-constant axes) are masked; the evidence content (window, family,
    distinguished hypotheses, expected-vs-observed match, predicted
    signature, episode-level rates) moves together with the record, so a
    verdict graded on this payload binds to exactly the record shown.
    """
    payload = record_payload(record)
    payload["feedback_id"] = anon_id
    payload["candidate_id"] = MASKED_IDENTITY
    payload["mutation_axes"] = []
    payload["axis_values"] = {}
    payload["held_constant_axes"] = {}
    return payload


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
    * every identity side channel is masked (``_anonymized_payload``).

    Two views constructed with identical inputs present bit-identical
    payloads and mappings (recomputable); the real candidate<->feedback
    pairing never appears in any payload (negative-tested).
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

        self._anon_to_real: Dict[str, str] = {}
        self._payloads: List[Dict[str, object]] = []
        for slot, record in enumerate(self._records):
            anon_id = f"anon-w{board_window:02d}-{slot:03d}"
            self._anon_to_real[anon_id] = record.feedback_id
            self._payloads.append(_anonymized_payload(anon_id, record))
        self.label = f"{VIEW_LABEL_PERMUTED}:{self._permutation_seed[:16]}"

    @property
    def permutation_seed(self) -> str:
        return self._permutation_seed

    def records(self) -> List[SimulatorFeedbackRecord]:
        return list(self._records)

    def to_prompt_payload(self) -> List[dict]:
        return [dict(p) for p in self._payloads]

    def resolve_citation(self, citation: str) -> str:
        try:
            return self._anon_to_real[citation]
        except KeyError:
            raise ValueError(
                f"UNKNOWN_ANONYMIZED_CITATION: {citation!r} is not an "
                f"anonymized feedback id presented by this window's "
                f"permuted view") from None
