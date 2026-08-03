"""Stage 8: deterministic Soft Copeland selector (G4 gate).

Self-contained stdlib replica of the canonical Soft Copeland semantics
of this branch (``d052/selectors/copeland.py`` + ``base.py``), pinned
to the canonical protocol version and the supervisor-frozen source
SHA256s. The E1 runtime NEVER imports d052 selector code; equivalence
is enforced by the cross-implementation fixture gate
``tests/e1_formal/test_copeland_parity.py`` (per-candidate scores,
full pairwise matrix, final order and canonical result hash must ALL
be exactly equal; mismatch is a hard failure with no skip).

Replicated semantics (canonical_v2):

* aggregate strength = mean of a candidate's normalized role scores
  (0.0 if none), summed in the caller's insertion order exactly as the
  canonical implementation does;
* full pairwise Copeland over ``sorted(candidate_ids)``: the stronger
  candidate of each unordered pair scores +1, the weaker 0, a tie +0.5
  each — order-independent by construction;
* critic policy: hard_veto pre-filters rejected candidates;
  soft_penalty subtracts the normalized penalty from the base score;
  score_only records but ignores;
* ranking tie-break is ALWAYS (composite DESC, candidate_id ASC);
* top-k finalize with NO backfill / NO k-reduction / NO re-LLM;
  shortfall -> honest INSUFFICIENT status + note;
* selection_hash = canonical sha256 of
  {selector, critic_policy, k, seed, sorted(selected_ids)}.

E1-specific contract layered on top:

* RETENTION IS DISABLED with NO substitute metric (G3): no retention
  field, score or filter exists here. Retention enters only through
  the frozen-anchor pre/post evaluation gate in ``anchor_manifest``;
* every consumed signal must be selector-admissible provenance and
  must carry ``has_real_probe=True``; any missing real evidence fails
  closed with ``SELECTION_BLOCKED_NO_REAL_EVIDENCE`` — dynamic
  candidates are simply not promoted (batch degrades to anchors +
  REUSE upstream).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .canonical import canonical_sha256
from .schemas import E1Code, E1SchemaError, assert_selector_admissible

#: --- G4 pins (supervisor-frozen; see reports/d052_canonical_artifacts_SHA256SUMS)
COPELAND_PROTOCOL_VERSION = "canonical_v2"
#: sha256 of the canonical source file ``d052/selectors/copeland.py``
#: (committed blob bytes; the working copy may differ only by CRLF)
COPELAND_SOURCE_SHA256 = (
    "80a60829537c87bafcc17aef7715cd37f6fdad0027cc16f27832744f11f6d613"
)
#: sha256 of ``d052/legacy/canonical_constants.py``
COPELAND_CONSTANTS_SHA256 = (
    "32c7a1c9dd28fc0388d213591061cd7eb5e1a1944fc68ee1ab448c1eec822bf2"
)
#: sha256 of ``d052/selectors/base.py`` (shared ranking machinery)
COPELAND_BASE_SHA256 = (
    "c9d0858548176e50a5ce561258ac0863fb8908b9b789c9293116702ad2ede108"
)
COPELAND_SOURCE_PATH = "d052/selectors/copeland.py"
COPELAND_CONSTANTS_PATH = "d052/legacy/canonical_constants.py"
COPELAND_BASE_PATH = "d052/selectors/base.py"

#: canonical selector/policy/status vocabulary (values MUST equal d052)
SELECTOR_NAME = "SOFT_COPELAND"
CRITIC_HARD_VETO = "hard_veto"
CRITIC_SOFT_PENALTY = "soft_penalty"
CRITIC_SCORE_ONLY = "score_only"
_VALID_POLICIES = frozenset(
    {CRITIC_HARD_VETO, CRITIC_SOFT_PENALTY, CRITIC_SCORE_ONLY}
)
STATUS_OK = "OK"
STATUS_INSUFFICIENT = "INSUFFICIENT_ELIGIBLE_CANDIDATES"

# fail-closed codes
COPELAND_PARITY_MISMATCH = "COPELAND_PARITY_MISMATCH"
COPELAND_SOURCE_SHA_MISMATCH = "COPELAND_SOURCE_SHA_MISMATCH"
SELECTOR_BAD_TYPE = "SELECTOR_BAD_TYPE"
SELECTOR_MISSING_FIELD = "SELECTOR_MISSING_FIELD"
SELECTOR_UNKNOWN_FIELD = "SELECTOR_UNKNOWN_FIELD"
SELECTOR_OUT_OF_RANGE = "SELECTOR_OUT_OF_RANGE"
DUPLICATE_CANDIDATE_SIGNALS = "DUPLICATE_CANDIDATE_SIGNALS"


class SelectorError(E1SchemaError):
    """Fail-closed selector violation; ``code`` is greppable."""


@dataclass(frozen=True)
class E1CandidateSignals:
    """One candidate's selector-side signals (provenance-checked).

    ``role_scores`` preserves the caller's insertion order exactly
    (the canonical implementation sums ``role_scores.values()`` in
    insertion order; parity is bit-for-bit).
    """

    candidate_id: str
    role_scores: Tuple[Tuple[str, float], ...]
    critic_reject: bool
    critic_penalty: float
    provenance: str
    has_real_probe: bool

    def role_scores_dict(self) -> Dict[str, float]:
        return dict(self.role_scores)


_SIGNAL_FIELDS = frozenset(
    {
        "candidate_id",
        "role_scores",
        "critic_reject",
        "critic_penalty",
        "provenance",
        "has_real_probe",
    }
)


def _require_float_in_unit(name: str, value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectorError(
            SELECTOR_BAD_TYPE, f"{ctx}: {name} must be a number"
        )
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise SelectorError(
            SELECTOR_OUT_OF_RANGE,
            f"{ctx}: {name} outside [0, 1]: {value}",
        )
    return value


def consume_candidate_signal(mapping: Any, ctx: str) -> E1CandidateSignals:
    """Parse one candidate's signals fail-closed (no defaults except
    the canonical ones: critic_reject=False, critic_penalty=0.0 —
    mirrored from the canonical schema)."""
    if not isinstance(mapping, Mapping):
        raise SelectorError(
            SELECTOR_BAD_TYPE,
            f"{ctx}: signals must be a mapping, got {type(mapping).__name__}",
        )
    unknown = sorted(k for k in mapping if k not in _SIGNAL_FIELDS)
    if unknown:
        raise SelectorError(
            SELECTOR_UNKNOWN_FIELD,
            f"{ctx}: unknown signal field(s) {unknown}",
        )
    if "candidate_id" not in mapping:
        raise SelectorError(
            SELECTOR_MISSING_FIELD, f"{ctx}: missing 'candidate_id'"
        )
    candidate_id = mapping["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise SelectorError(
            SELECTOR_BAD_TYPE, f"{ctx}: candidate_id must be non-empty str"
        )
    role_scores: List[Tuple[str, float]] = []
    if "role_scores" in mapping:
        raw = mapping["role_scores"]
        if not isinstance(raw, Mapping):
            raise SelectorError(
                SELECTOR_BAD_TYPE, f"{ctx}: role_scores must be a mapping"
            )
        for role, value in raw.items():
            if not isinstance(role, str) or not role.strip():
                raise SelectorError(
                    SELECTOR_BAD_TYPE,
                    f"{ctx}: role name must be non-empty str, got {role!r}",
                )
            role_scores.append(
                (role, _require_float_in_unit(f"role_scores[{role}]", value, ctx))
            )
    critic_reject = False
    if "critic_reject" in mapping:
        if not isinstance(mapping["critic_reject"], bool):
            raise SelectorError(
                SELECTOR_BAD_TYPE, f"{ctx}: critic_reject must be bool"
            )
        critic_reject = mapping["critic_reject"]
    critic_penalty = 0.0
    if "critic_penalty" in mapping:
        critic_penalty = _require_float_in_unit(
            "critic_penalty", mapping["critic_penalty"], ctx
        )
    if "provenance" not in mapping:
        raise SelectorError(
            SELECTOR_MISSING_FIELD, f"{ctx}: missing 'provenance'"
        )
    provenance = assert_selector_admissible(mapping["provenance"], ctx)
    if "has_real_probe" not in mapping:
        raise SelectorError(
            SELECTOR_MISSING_FIELD, f"{ctx}: missing 'has_real_probe'"
        )
    if not isinstance(mapping["has_real_probe"], bool):
        raise SelectorError(
            SELECTOR_BAD_TYPE, f"{ctx}: has_real_probe must be bool"
        )
    return E1CandidateSignals(
        candidate_id=candidate_id.strip(),
        role_scores=tuple(role_scores),
        critic_reject=critic_reject,
        critic_penalty=critic_penalty,
        provenance=provenance,
        has_real_probe=mapping["has_real_probe"],
    )


def consume_candidate_signals(
    mappings: Any, ctx: str
) -> Tuple[E1CandidateSignals, ...]:
    """Parse all candidate signals; ids must be unique (fail-closed)."""
    if not isinstance(mappings, (list, tuple)):
        raise SelectorError(
            SELECTOR_BAD_TYPE,
            f"{ctx}: signals must be a sequence of mappings",
        )
    signals = tuple(
        consume_candidate_signal(raw, f"{ctx}[{i}]")
        for i, raw in enumerate(mappings)
    )
    ids = [sig.candidate_id for sig in signals]
    if len(set(ids)) != len(ids):
        raise SelectorError(
            DUPLICATE_CANDIDATE_SIGNALS,
            f"{ctx}: duplicate candidate_id in {ids}",
        )
    return signals


# ---------------------------------------------------------------------------
# Canonical semantics replica (pure stdlib)
# ---------------------------------------------------------------------------
def strength(role_scores: Mapping[str, float]) -> float:
    """Aggregate normalized role strength = mean over values (0 if none).

    Sums in the mapping's insertion order, exactly as the canonical
    implementation does (bit-for-bit parity requirement).
    """
    vals = list(role_scores.values())
    return sum(float(v) for v in vals) / len(vals) if vals else 0.0


def copeland_scores(
    signals: Sequence[E1CandidateSignals],
) -> Dict[str, float]:
    """Deterministic full-pairwise Copeland score per candidate_id."""
    by_id = {sig.candidate_id: sig for sig in signals}
    ids = sorted(by_id)  # deterministic iteration
    strengths = {cid: strength(by_id[cid].role_scores_dict()) for cid in ids}
    score = {cid: 0.0 for cid in ids}
    for i, a in enumerate(ids):
        sa = strengths[a]
        for b in ids[i + 1:]:
            sb = strengths[b]
            if sa > sb:
                score[a] += 1.0
            elif sa < sb:
                score[b] += 1.0
            else:
                score[a] += 0.5
                score[b] += 0.5
    return score


def pairwise_matrix(signals: Sequence[E1CandidateSignals]) -> Dict[Tuple[str, str], str]:
    """Full pairwise outcomes over sorted ids (audit/parity surface).

    Key ``(a, b)`` with ``a < b`` (lexicographic) maps to the WINNER's
    candidate_id, or ``"tie"``.
    """
    by_id = {sig.candidate_id: sig for sig in signals}
    ids = sorted(by_id)
    strengths = {cid: strength(by_id[cid].role_scores_dict()) for cid in ids}
    matrix = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if strengths[a] > strengths[b]:
                matrix[(a, b)] = a
            elif strengths[a] < strengths[b]:
                matrix[(a, b)] = b
            else:
                matrix[(a, b)] = "tie"
    return matrix


def eligible_ids(
    signals: Sequence[E1CandidateSignals], critic_policy: str
) -> List[str]:
    """Candidates eligible under the critic policy (order preserved)."""
    if critic_policy == CRITIC_HARD_VETO:
        return [sig.candidate_id for sig in signals if not sig.critic_reject]
    return [sig.candidate_id for sig in signals]


def composite_score(sig: E1CandidateSignals, critic_policy: str, base: float) -> float:
    """Apply critic policy to a base score, deterministically."""
    if critic_policy == CRITIC_SOFT_PENALTY:
        return base - sig.critic_penalty
    return base  # hard_veto (already filtered) / score_only


def rank_candidates(
    signals: Sequence[E1CandidateSignals], critic_policy: str
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """Rank eligible candidates by (composite DESC, candidate_id ASC).

    Returns (ordered [(candidate_id, composite)], sorted rejected ids).
    """
    if critic_policy not in _VALID_POLICIES:
        raise SelectorError(
            SELECTOR_OUT_OF_RANGE,
            f"critic_policy {critic_policy!r} not in {sorted(_VALID_POLICIES)}",
        )
    rejected = sorted(sig.candidate_id for sig in signals if sig.critic_reject)
    base_scores = copeland_scores(signals)
    by_id = {sig.candidate_id: sig for sig in signals}
    ranked = [
        (
            cid,
            composite_score(by_id[cid], critic_policy, base_scores[cid]),
        )
        for cid in eligible_ids(signals, critic_policy)
    ]
    ranked.sort(key=lambda t: (-float(t[1]), t[0]))
    return ranked, rejected


def compute_selection_hash(
    selector: str,
    critic_policy: str,
    k: int,
    seed: int,
    selected_ids: Sequence[str],
) -> str:
    """Canonical result hash — byte-identical to the canonical impl."""
    return canonical_sha256(
        {
            "selector": selector,
            "critic_policy": critic_policy,
            "k": k,
            "seed": seed,
            "selected_ids": sorted(selected_ids),
        }
    )


@dataclass(frozen=True)
class SelectionOutcome:
    """Audit-grade selection result (no backfill semantics)."""

    selector: str
    critic_policy: str
    k_requested: int
    seed: int
    candidate_count_in: int
    eligible_count: int
    selected_ids: Tuple[str, ...]
    rejected_by_critic: Tuple[str, ...]
    status: str
    selection_hash: str
    shortfall_note: str


def select_soft_copeland(
    signals: Sequence[E1CandidateSignals],
    *,
    k: int,
    seed: int,
    critic_policy: str = CRITIC_HARD_VETO,
) -> SelectionOutcome:
    """Deterministic Soft Copeland top-k (canonical_v2 semantics)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SelectorError(
            SELECTOR_OUT_OF_RANGE, f"k must be an int >= 1, got {k!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SelectorError(
            SELECTOR_BAD_TYPE, f"seed must be an int, got {seed!r}"
        )
    ranked, rejected = rank_candidates(signals, critic_policy)
    chosen = [cid for cid, _ in ranked]
    if len(chosen) >= k:
        selected = tuple(chosen[:k])
        status = STATUS_OK
        note = ""
    else:
        selected = tuple(chosen)
        status = STATUS_INSUFFICIENT
        note = (
            f"only {len(ranked)} eligible candidates for k={k} under "
            f"critic_policy={critic_policy}; selected all {len(chosen)}; "
            "NO backfill / NO k-reduction / NO re-LLM"
        )
    return SelectionOutcome(
        selector=SELECTOR_NAME,
        critic_policy=critic_policy,
        k_requested=k,
        seed=seed,
        candidate_count_in=len(signals),
        eligible_count=len(ranked),
        selected_ids=selected,
        rejected_by_critic=tuple(rejected),
        status=status,
        selection_hash=compute_selection_hash(
            SELECTOR_NAME, critic_policy, k, seed, selected
        ),
        shortfall_note=note,
    )


def select_dynamic_batch(
    signals: Sequence[E1CandidateSignals],
    *,
    k: int,
    seed: int,
    critic_policy: str = CRITIC_HARD_VETO,
) -> SelectionOutcome:
    """E1 promotion gate: select ONLY on real probe-backed signals.

    Any missing real evidence (no signals at all, or any signal lacking
    ``has_real_probe``) fails closed with
    ``SELECTION_BLOCKED_NO_REAL_EVIDENCE`` — dynamic candidates are not
    promoted and the upstream batch trains NOTHING while blocked (C13:
    zero updates, no anchors-only sneak; REUSE requires a fully
    verified previous window). There is NO retention substitute and NO
    archive-priority substitute on this path.
    """
    if len(signals) == 0:
        raise SelectorError(
            E1Code.SELECTION_BLOCKED_NO_REAL_EVIDENCE,
            "no candidate signals at all; selection is blocked (no real "
            "evidence, no substitute)",
        )
    missing = sorted(
        sig.candidate_id for sig in signals if not sig.has_real_probe
    )
    if missing:
        raise SelectorError(
            E1Code.SELECTION_BLOCKED_NO_REAL_EVIDENCE,
            f"candidates without a real dual probe: {missing}; selection "
            "is blocked — archive priors or heuristics never substitute "
            "for real probe evidence",
        )
    return select_soft_copeland(
        signals, k=k, seed=seed, critic_policy=critic_policy
    )
