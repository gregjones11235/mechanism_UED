"""Synthetic Proposal generators for offline testing (no LLM, no craftax, no GPU).

Lets tests control coverage overlap so submodularity / greedy behavior can be checked on
controlled inputs. Deterministic given a seed (random.Random), so tests are reproducible.
"""

from __future__ import annotations

import random

from .craftax_achievements import ALL_ACHIEVEMENTS
from .proposal import Proposal

_ALL = sorted(ALL_ACHIEVEMENTS)


def random_proposal(
    rng: random.Random,
    proposal_id: str,
    *,
    proposer_id: str = "mock",
    parent_task_id: str = "task_0",
    n_achievements: int | None = None,
) -> Proposal:
    """A proposal covering a random subset of achievements."""
    if n_achievements is None:
        n_achievements = rng.randint(1, 6)
    n_achievements = min(n_achievements, len(_ALL))
    achs = frozenset(rng.sample(_ALL, n_achievements))
    return Proposal(
        proposal_id=proposal_id,
        proposer_id=proposer_id,
        parent_task_id=parent_task_id,
        docstring=f"mock level {proposal_id}",
        reasoning="mock",
        achievements=achs,
    )


def random_pool(rng: random.Random, n: int, **kw) -> list[Proposal]:
    """A pool of n random proposals."""
    return [random_proposal(rng, f"p{i}", **kw) for i in range(n)]
