"""Completed-admission gate — the v6 §3.4 code backstop that keeps mid-chain links TRAINED.

v6_design.md §3.4. DiCode's scaffold lets a proposer move a prerequisite into ``Completed
Achievements`` (satisfied by the initial world state, so the agent does NOT train it). The v5 disease:
proposers compressed away mid-chain links the student had NOT actually mastered, severing the tier3
chain so it never transferred. The fix here is a HARD code gate (not just a prompt ask):

  A prerequisite may stay in ``Completed`` ONLY IF the student has genuinely mastered it (held-out SR
  high). Any ``Completed`` entry that is an UNMASTERED link of the current siege focus is illegally
  compressed — this module pulls it OUT of ``Completed`` and INTO ``Relevant`` (so it is actually
  trained in the level), and rewrites the docstring accordingly.

This mirrors the SiegeNotebook's B-layer philosophy: the LLM proposes the Completed list; the code
enforces the invariant it cannot violate. ``unmastered`` is supplied by
``SiegeNotebook.unmastered_links(live_profile)`` — computed from LIVE SR, so the gate is always current.

Scope: the gate only ever MOVES an unmastered focus-link from Completed to Relevant. It never removes
a Relevant achievement, never touches Completed entries that are genuinely mastered, and is a no-op
when there is no active siege focus (unmastered set empty) — so it cannot disturb the baseline path.

Zero external deps (no craftax/jax/LLM) so it is unit-testable offline. Achievement-name validity is
checked against the ground-truth set in craftax_achievements.
"""

from __future__ import annotations

import re

from auction.craftax_achievements import ALL_ACHIEVEMENTS

# Section headers as the proposers emit them (see prompt OUTPUT FORMAT / auction_integration).
# We capture the body up to the next "Header:" line, a blank line, or end-of-string. Bodies may wrap
# across lines (paper p85), so DOTALL with a lookahead terminator.
_SECTION_RE_TMPL = r"({header})\s*:\s*(.*?)(?=\n\s*[A-Z][A-Za-z ]+:|\n\s*\n|\Z)"
_RELEVANT_RE = re.compile(_SECTION_RE_TMPL.format(header=r"Relevant\s+Achievements"),
                          re.IGNORECASE | re.DOTALL)
_COMPLETED_RE = re.compile(_SECTION_RE_TMPL.format(header=r"Completed\s+Achievements"),
                           re.IGNORECASE | re.DOTALL)


def _parse_names(body: str) -> list[str]:
    """Split a section body into lowercase known-achievement names, order-preserving, de-duplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in re.split(r"[,\n]", body or ""):
        name = tok.strip().lower()
        # tolerate a trailing period or stray punctuation the LLM sometimes appends
        name = name.strip(" .;")
        if name in ALL_ACHIEVEMENTS and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_completed_achievements(docstring: str) -> frozenset[str]:
    """Lowercase achievement names from a docstring's 'Completed Achievements:' line (symmetric to
    parse_relevant_achievements). Empty frozenset if absent/malformed."""
    if not docstring:
        return frozenset()
    m = _COMPLETED_RE.search(docstring)
    return frozenset(_parse_names(m.group(2))) if m else frozenset()


def _rewrite_section(docstring: str, regex: re.Pattern, header_label: str, names: list[str]) -> str:
    """Replace (or, if absent, do nothing to) a section's body with ``names`` (comma-joined UPPER).

    Names are emitted UPPERCASE to match the proposers' own style (e.g. MAKE_STONE_PICKAXE). Only the
    body after the header is replaced; surrounding text/indentation is preserved.
    """
    joined = ", ".join(n.upper() for n in names)

    def _sub(m: re.Match) -> str:
        return f"{m.group(1)}: {joined}"

    new, n = regex.subn(_sub, docstring, count=1)
    return new if n else docstring


def enforce_completed_gate(docstring: str, unmastered: set[str]) -> tuple[str, list[str]]:
    """Pull every UNMASTERED focus-link out of ``Completed`` and into ``Relevant``; rewrite docstring.

    Args:
        docstring: the proposer's task description (contains Relevant/Completed sections).
        unmastered: skills that are NOT yet mastered (from SiegeNotebook.unmastered_links). An entry
            in Completed that is in this set is an illegal compression.

    Returns:
        (new_docstring, moved) where ``moved`` is the list of achievement names pulled back into
        Relevant. If nothing needs moving (no focus / no violation), returns the docstring unchanged
        and an empty list — a strict no-op, so the baseline path is never disturbed.
    """
    if not docstring or not unmastered:
        return docstring, []

    completed = _parse_names(_COMPLETED_RE.search(docstring).group(2)) if _COMPLETED_RE.search(docstring) else []
    if not completed:
        return docstring, []

    unmastered_l = {u.lower() for u in unmastered}
    to_move = [c for c in completed if c in unmastered_l]
    if not to_move:
        return docstring, []

    new_completed = [c for c in completed if c not in unmastered_l]

    relevant = _parse_names(_RELEVANT_RE.search(docstring).group(2)) if _RELEVANT_RE.search(docstring) else []
    # append moved links to Relevant (preserve existing order; skip if somehow already present)
    for m in to_move:
        if m not in relevant:
            relevant.append(m)

    new = _rewrite_section(docstring, _COMPLETED_RE, "Completed Achievements", new_completed)
    new = _rewrite_section(new, _RELEVANT_RE, "Relevant Achievements", relevant)
    return new, to_move
