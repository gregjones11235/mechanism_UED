"""Learnability — the bid term rewarding proposals aimed at the student's LEARNABLE edge.

方法设计_v2.md §3 (fourth bid term). Balances the depth-biased terms (Coverage + AmbitionGain)
so "stay-safe / consolidation" proposals aren't systematically out-bid.

★ OFFICIAL DiCode semantics (dicode-learnability-p-is-training-byproduct): p is NOT measured by an
extra rollout. In DiCode, learnability = p*(1-p) where p = a task's training success rate
(scoring.py:236, num_successes/num_finished from the training trajectory). p only exists for tasks
that have ALREADY been trained. A freshly proposed candidate has no p of its own.

So we use the candidate's PARENT task's learnability as a proxy for "how learnable is evolving
along this lineage right now": the parent's learnability is already stored in the archive
(node["priority_score"] / node["learnability"] = p*(1-p)) from when it was trained. If the parent
sits in the learnable band (p≈0.5, learnability high ≈0.25), the student is at the learnable edge
of that chain and its children are likely well-targeted. A candidate whose parent has no
learnability (never trained, e.g. a seed) contributes 0 — it simply relies on the other three terms.

parent_learnability[parent_task_id] in [0, 0.25]: the stored p*(1-p) of that parent. (Max of
p*(1-p) is 0.25 at p=0.5.) Missing parent => 0.

MODULARITY: per-proposal independent (a function of the proposal's own parent_task_id and the fixed
parent-learnability map), so it is MODULAR — adding it keeps the assembled bid submodular (same
argument as ambition.py / endorsement.py). It does NOT depend on the selected set.
"""

from __future__ import annotations

from collections.abc import Mapping

from .proposal import Proposal


def learnability_gain(
    proposal: Proposal,
    parent_learnability: Mapping[str, float] | None,
) -> float:
    """Learnability(proposal) = parent_learnability[proposal.parent_task_id], else 0.

    Args:
        proposal: the candidate. Its ``parent_task_id`` keys into the map.
        parent_learnability: {parent_task_id: p*(1-p)} for parents that have been trained (read
            straight from the archive, no recomputation). None or missing key => 0.

    Returns:
        A non-negative scalar in [0, 0.25]. Per-proposal independent => modular.
    """
    if not parent_learnability:
        return 0.0
    value = parent_learnability.get(proposal.parent_task_id, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    # Clamp defensively to the theoretical p*(1-p) range [0, 0.25]; negatives/NaN -> 0.
    if not (value == value):  # NaN
        return 0.0
    return max(0.0, min(0.25, value))
