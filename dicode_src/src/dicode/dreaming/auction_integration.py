"""Adapter between DiCode's generation pipeline and the offline ``auction`` module.

Keeps all DiCode<->auction translation in one place so gen_manager stays minimally touched.
Two jobs:
  1. Parse the achievements a generated docstring teaches ("Relevant Achievements: A, B" line),
     mapping DiCode's UPPERCASE enum names to the auction module's lowercase achievement names.
  2. Turn DiCode ``parsed_response`` dicts (reasoning/description) into ``auction.Proposal``s and
     turn the auction's selected ``Proposal``s back into the index list DiCode expects.

The docstring format is fixed by DiCode's own prompt (paper p23/p85), e.g.:
    Relevant Achievements: ENTER_GNOMISH_MINES, DEFEAT_GNOME_WARRIOR
    Completed Achievements: MAKE_STONE_PICKAXE, MAKE_STONE_SWORD, ...
We take *Relevant* (what the level actively teaches) as the proposal's coverage; *Completed*
are prerequisites, not what the level teaches, so they are not coverage.
"""

from __future__ import annotations

import re

from auction.craftax_achievements import ALL_ACHIEVEMENTS
from auction.proposal import Proposal

# --- AmbitionGain data source: global_agent_profile (skill_*) -> target_gap ------------------


def profile_to_target_gap(global_agent_profile: dict | None) -> dict[str, float]:
    """Convert DiCode's global_agent_profile into the AmbitionGain target_gap map.

    global_agent_profile has keys like ``skill_collect_wood`` with values in 0..100 (the student's
    success rate on that achievement on the TARGET Craftax env, see gen_manager
    _format_global_agent_profile / online_evaluation.py). AmbitionGain wants
        target_gap[achievement] = 1 - SR_on_target   in [0,1]
    so an achievement the student aces (SR~100) -> gap 0, one it fails (SR~0) -> gap 1.

    Missing / unknown skills are omitted (AmbitionGain treats missing as gap 0). Values are clamped
    to [0,1]. Returns {} if the profile is absent (AmbitionGain then contributes 0 everywhere).
    """
    if not global_agent_profile:
        return {}
    gap: dict[str, float] = {}
    for key, value in global_agent_profile.items():
        if not key.startswith("skill_"):
            continue
        name = key[len("skill_"):].lower()
        if name not in ALL_ACHIEVEMENTS:
            continue
        try:
            sr = float(value) / 100.0  # skill_* values are 0..100
        except (TypeError, ValueError):
            continue
        sr = min(1.0, max(0.0, sr))
        gap[name] = 1.0 - sr
    return gap

# Matches the "Relevant Achievements:" line and captures everything until the next "...:" header
# or blank line. Achievements may wrap across lines (see paper p85 "Completed Achievements" wrap).
_RELEVANT_RE = re.compile(
    r"Relevant\s+Achievements\s*:\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]+:|\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_relevant_achievements(docstring: str) -> frozenset[str]:
    """Extract the lowercase achievement names from a docstring's 'Relevant Achievements:' line.

    Unknown / malformed tokens are dropped (robust to LLM typos); returns only names that are in
    the ground-truth 67-achievement set. Returns empty frozenset if the section is absent.
    """
    if not docstring:
        return frozenset()
    m = _RELEVANT_RE.search(docstring)
    if not m:
        return frozenset()
    raw = m.group(1)
    names = set()
    for tok in re.split(r"[,\n]", raw):
        name = tok.strip().lower()
        if name in ALL_ACHIEVEMENTS:
            names.add(name)
    return frozenset(names)


def parsed_response_to_proposal(
    parsed: dict,
    *,
    proposal_id: str,
    proposer_id: str,
    parent_task_id: str,
) -> Proposal:
    """Convert one DiCode parsed_response (has 'description', 'reasoning') into a Proposal.

    ``achievements`` is parsed from the description's 'Relevant Achievements:' line — the
    objective coverage anchor for the auction (§7 option 3).
    """
    docstring = parsed.get("description") or ""
    return Proposal(
        proposal_id=proposal_id,
        proposer_id=proposer_id,
        parent_task_id=parent_task_id,
        docstring=docstring,
        reasoning=parsed.get("reasoning") or "",
        achievements=parse_relevant_achievements(docstring),
    )
