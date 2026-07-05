"""Endorsement — the cross-rating bid term: how much OTHER Proposers endorse a proposal.

This is the "multi-FM market" signal a single FM structurally cannot produce (§2.3, §7.5).
Each Proposer rates every *other* Proposer's proposal; a proposal's Endorsement is the
aggregate of the votes it receives from others. Proposals endorsed by the majority score high
→ filters out a single FM's idiosyncratic proposals.

KEY THEORETICAL PROPERTY (matters for keeping the bid submodular): Endorsement is defined as a
**per-proposal independent score** (sum/mean of votes it receives). A function that assigns each
element an independent value and sums over the selected set is MODULAR. Modular = both submodular
and supermodular. Since (submodular Coverage) + (non-negative modular Endorsement) stays
submodular, adding Endorsement to the bid does NOT break the (1-1/e) greedy guarantee — *as long
as Endorsement is modular*, i.e. a proposal's endorsement does not depend on which OTHER proposals
were selected. We enforce that here by computing it once, independent of the selected set.

v1: votes are objective (computed by the auctioneer, e.g. embedding agreement / rubric).
v2: votes can come from Proposers' self-reports aggregated by a market scoring rule (§7.6 ④);
    the modular structure is preserved as long as the aggregate is per-proposal.
"""

from __future__ import annotations

from collections.abc import Mapping

from .proposal import Proposal

# A cross-rating matrix: votes[rater_id][proposal_id] = score in [0, 1].
CrossRatings = Mapping[str, Mapping[str, float]]


def endorsement_scores(
    proposals: list[Proposal],
    cross_ratings: CrossRatings,
    *,
    exclude_self: bool = True,
) -> dict[str, float]:
    """Aggregate endorsement per proposal = mean vote received from OTHER proposers.

    Args:
        proposals: the candidate set.
        cross_ratings: cross_ratings[rater_proposer_id][proposal_id] -> vote in [0,1].
        exclude_self: if True, a Proposer's votes on its own proposals are ignored (anti-self-dealing).

    Returns:
        {proposal_id: endorsement_score}, where score is the mean of valid votes received.
        A proposal with no valid votes gets 0.0.

    Note: this is computed independently per proposal (no dependence on a selected subset),
    which is exactly what keeps the resulting bid term MODULAR.
    """
    scores: dict[str, float] = {}
    for p in proposals:
        votes: list[float] = []
        for rater_id, rated in cross_ratings.items():
            if exclude_self and rater_id == p.proposer_id:
                continue
            if p.proposal_id in rated:
                v = rated[p.proposal_id]
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"vote for {p.proposal_id} from {rater_id} not in [0,1]: {v}")
                votes.append(v)
        scores[p.proposal_id] = float(sum(votes) / len(votes)) if votes else 0.0
    return scores
