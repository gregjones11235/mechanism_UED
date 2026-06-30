"""Proposal data contract for the auction (description) layer.

A ``Proposal`` is one candidate level description produced by one Proposer for one parent.
It is the unit the auction selects over. Field names are chosen to map 1:1 onto DiCode's
``generation_result`` dict (gen_manager.py ``_organize_data``) so the eventual
``dicode_adapter`` is pure field-shuffling, not a rewrite:

    DiCode generation_result      <->  Proposal
    -------------------------          --------
    generated_task_id             <->  proposal_id
    parent_task_id                <->  parent_task_id
    docstring                     <->  docstring
    reasoning                     <->  reasoning
    (no equivalent)               <->  proposer_id      (which Proposer produced it: A/C arms)
    (no equivalent)               <->  achievements     (objective coverage anchor, §7 option 3)
    (no equivalent)               <->  skill_tags       (semantic dependency-chain labels)
    (no equivalent)               <->  self_report      (v2 only: Proposer self-reported bid)

``achievements`` is the objective ground-truth anchor: a subset of the 67 Craftax achievements
(see craftax_achievements.ALL_ACHIEVEMENTS). It is what Coverage submodularity is proven over.
``self_report`` is None for v1 (objective bid computed by auctioneer) and populated for v2
(Proposer self-reports its valuation in the same LLM call). The v1/v2 ablation "drop self-report"
is therefore literally ``self_report = None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .craftax_achievements import ALL_ACHIEVEMENTS


@dataclass(frozen=True)
class SelfReport:
    """v2-only: a Proposer's self-reported valuation, emitted in the same call as the docstring.

    This is *private information* the Proposer reports about its own proposal. Because it can
    influence selection/price, it creates the manipulation incentive that makes a real auction
    (strategy-proofness) meaningful (v1_experiment.md §7.4). In v1 this is absent (None).
    """

    claimed_achievements: frozenset[str]  # which achievements the Proposer claims to teach
    confidence: float                     # self-reported confidence in [0, 1]
    claimed_learning_gain: float          # self-reported expected learning gain (>= 0)

    def __post_init__(self) -> None:
        bad = self.claimed_achievements - ALL_ACHIEVEMENTS
        if bad:
            raise ValueError(f"SelfReport.claimed_achievements has unknown names: {sorted(bad)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.claimed_learning_gain < 0.0:
            raise ValueError(f"claimed_learning_gain must be >= 0, got {self.claimed_learning_gain}")


@dataclass(frozen=True)
class Proposal:
    """One candidate level description, the unit the auction selects over."""

    proposal_id: str
    proposer_id: str               # identifies the Proposer (heterogeneous base / persona)
    parent_task_id: str            # the parent this evolved from (DiCode lineage)
    docstring: str                 # the level description (eq. 8 product)
    reasoning: str                 # the FM's reasoning text
    achievements: frozenset[str]   # objective anchor: subset of the 67 Craftax achievements
    skill_tags: tuple[str, ...] = ()           # semantic dependency-chain labels (free-form)
    self_report: SelfReport | None = None      # v2 only; None in v1

    def __post_init__(self) -> None:
        bad = self.achievements - ALL_ACHIEVEMENTS
        if bad:
            raise ValueError(
                f"Proposal {self.proposal_id!r} has unknown achievement names: {sorted(bad)}. "
                f"Legal values are the 67 craftax achievements (craftax_achievements.ALL_ACHIEVEMENTS)."
            )
        if not self.proposal_id:
            raise ValueError("proposal_id must be non-empty")
        if not self.proposer_id:
            raise ValueError("proposer_id must be non-empty")

    @property
    def is_v2(self) -> bool:
        """True iff this proposal carries a Proposer self-report (v2 mode)."""
        return self.self_report is not None
