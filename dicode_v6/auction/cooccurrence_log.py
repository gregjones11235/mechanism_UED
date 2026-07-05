"""CooccurrenceLog — cross-session achievement co-occurrence counts from held-out eval (v6 §3.8, (c)).

v6_design.md §3.3/§3.8. The (c) co-occurrence signal answers "in the held-out episodes where the
student SUCCEEDS at a deep achievement, which OTHER achievements did it also reach?" — i.e. which
prerequisites this specific student actually strings together on the way to a hard skill, vs which it
never manages. That is evidence the modeler cannot get from aggregate SR alone (which only says
"place_torch = 9%", not "in the episodes where it beat the gnome warrior, it also had torches 95% of
the time"). It grounds the siege prereq chain in the student's REAL trajectories instead of the LLM's
guess (v6_design.md §3.3 (c)).

Why persistent + accumulated (v6_design.md §3.8): tier3 SR ~10% -> few successful episodes per eval
-> co-occurrence is SPARSE. A single session's counts are too noisy; we accumulate raw counts across
sessions (never averaging away) so the frequencies stabilise as evidence builds.

Design: the eval (craftax_evaluation) computes, per session, two fixed-shape integer arrays over the
67-achievement universe (order = craftax_achievements enum order):
  - ``count[i]``     = # finished held-out episodes where achievement i was reached
  - ``cooc[i][j]``   = # finished held-out episodes where BOTH i and j were reached
This log ADDS each session's arrays into running totals and persists them, then answers:
  ``prereq_frequencies(deep)`` = { j: cooc[deep][j] / count[deep] } for every j != deep
    = "of the episodes where the student reached `deep`, in what fraction did it also reach j?"
High-frequency co-reached skills are the prerequisites this student genuinely strings together.

Storage: a plain JSON file ``cooccurrence_counts.json`` in the run cwd (same place as
``student_profile_history.json`` / ``siege_notebook.json``), atomic-written, reloaded on resume.

This module is pure-python (no jax/craftax) so it is unit-testable offline; the eval side only has to
hand it two integer arrays per session.
"""

from __future__ import annotations

import json
import os

from auction.craftax_achievements import (
    ACHIEVEMENT_TO_VALUE,
    NUM_ACHIEVEMENTS,
    VALUE_TO_ACHIEVEMENT,
)

_DEFAULT_PATH = "cooccurrence_counts.json"

# Minimum EMPIRICAL SUCCESS RATE of the deep skill before its co-occurrence frequencies are trusted
# (user 2026-07-05). Held-out eval runs num_envs=1024 episodes/session, so support is NOT sparse in
# absolute terms (a tier3 skill at ~12% SR gives ~120 successful episodes PER session). The right
# guard is therefore RELATIVE, not an absolute count: only trust a deep skill's prereq co-occurrence
# once the student clears it often enough for the conditional frequencies to mean something —
# ``count[deep] / total_finished >= MIN_SR``. Below this the deep skill is essentially never solved and
# any co-occurrence is single-run noise; the caller then falls back to (b) mechanics only.
# 3% chosen against v5's measured tier3 ceiling (~12%): tier3 passes comfortably, while genuinely
# near-impossible skills (SR < 3%) are filtered out.
MIN_SR = 0.03

# Legacy alias kept so any external reference to the old absolute-count guard still imports; unused by
# the relative-SR path below.
MIN_SUPPORT = 5


class CooccurrenceLog:
    """Cross-session accumulated achievement co-occurrence counts over the 67-achievement universe."""

    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_PATH
        self._count, self._cooc, self._sessions, self._total = self._load()

    # ---- persistence (mirrors StudentProfileLog: atomic write, resume-safe) ---------------------

    def _load(self):
        n = NUM_ACHIEVEMENTS
        if not os.path.exists(self.path):
            return [0] * n, [[0] * n for _ in range(n)], [], 0
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            count = data.get("count") or [0] * n
            cooc = data.get("cooc") or [[0] * n for _ in range(n)]
            sessions = data.get("sessions") or []
            # ``total`` (accumulated finished-episode denominator) may be absent in a pre-v6-SR-guard
            # file; default 0 -> SR guard treats it as "no denominator yet" (empty hint) until the next
            # add_session supplies one. Resume-safe.
            total = int(data.get("total") or 0)
            # shape-guard: a corrupt / stale-shape file falls back to empty rather than crashing.
            if len(count) != n or len(cooc) != n or any(len(row) != n for row in cooc):
                return [0] * n, [[0] * n for _ in range(n)], [], 0
            return count, cooc, sessions, total
        except (json.JSONDecodeError, OSError, TypeError):
            return [0] * n, [[0] * n for _ in range(n)], [], 0

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "count": self._count,
                    "cooc": self._cooc,
                    "sessions": self._sessions,
                    "total": self._total,
                },
                f,
            )
        os.replace(tmp, self.path)

    # ---- accumulate one session's counts --------------------------------------------------------

    def add_session(self, session_idx: int, count_vec, cooc_mat, names=None, total=None) -> None:
        """Add one held-out eval's per-achievement count vector + co-occurrence matrix into the totals.

        ``count_vec``: ints per achievement (finished episodes reaching each achievement).
        ``cooc_mat``:  square ints (finished episodes reaching BOTH i and j).
        ``names``: optional list of achievement names labelling the columns/rows of count_vec/cooc_mat.
            If given, the arrays are REMAPPED by name into the canonical 67-achievement order (robust to
            the eval iterating achievements in a different order). If None, count_vec/cooc_mat are
            assumed already in canonical craftax enum order and length 67.
        ``total``: this session's number of FINISHED held-out episodes (the SR denominator, e.g. ~1024).
            Accumulated across sessions so the relative-SR guard (``MIN_SR``) can read
            ``count[deep] / total``. Optional (default None -> not incremented) so older callers/tests
            that don't pass it degrade gracefully to "no denominator" rather than crashing.
        Idempotent per session (re-adding the same session is skipped, so a resumed run doesn't
        double-count).
        """
        if session_idx in self._sessions:
            return  # already counted (resume idempotency)
        n = NUM_ACHIEVEMENTS
        if names is None:
            for i in range(n):
                self._count[i] += int(count_vec[i])
                crow, row = self._cooc[i], cooc_mat[i]
                for j in range(n):
                    crow[j] += int(row[j])
        else:
            # map each source column k -> canonical index via its name; skip unknown names.
            canon = [ACHIEVEMENT_TO_VALUE.get(str(nm).lower()) for nm in names]
            for k, ci in enumerate(canon):
                if ci is None:
                    continue
                self._count[ci] += int(count_vec[k])
                srow, drow = cooc_mat[k], self._cooc[ci]
                for l, cj in enumerate(canon):
                    if cj is None:
                        continue
                    drow[cj] += int(srow[l])
        if total is not None:
            self._total += int(total)
        self._sessions.append(session_idx)
        self._save()

    # ---- query ----------------------------------------------------------------------------------

    def support(self, achievement: str) -> int:
        """How many accumulated successful episodes reached ``achievement`` (its count)."""
        idx = ACHIEVEMENT_TO_VALUE.get(achievement.lower())
        return int(self._count[idx]) if idx is not None else 0

    def total_finished(self) -> int:
        """Accumulated number of FINISHED held-out episodes (the SR denominator)."""
        return int(self._total)

    def deep_sr(self, achievement: str) -> float:
        """Empirical success rate of ``achievement`` = count[ach] / accumulated total finished episodes.

        0.0 when no denominator has been recorded yet (older file / total never supplied).
        """
        if self._total <= 0:
            return 0.0
        return self.support(achievement) / self._total

    def prereq_frequencies(self, deep_achievement: str, min_freq: float = 0.0) -> dict[str, float]:
        """Co-occurrence frequencies: of episodes reaching ``deep_achievement``, fraction reaching each j.

        Returns {other_achievement: freq in [0,1]}, sorted high->low, excluding the deep skill itself
        and anything below ``min_freq``. Empty dict when the deep skill's EMPIRICAL SR
        (count[deep]/total_finished) is below ``MIN_SR`` (user 2026-07-05: a relative guard, not an
        absolute count — held-out eval has ~1024 episodes/session so absolute counts are never sparse;
        what matters is whether the student solves the deep skill often enough for the conditional
        frequencies to mean something). Below MIN_SR the caller falls back to (b) mechanics only.
        """
        idx = ACHIEVEMENT_TO_VALUE.get((deep_achievement or "").lower())
        if idx is None:
            return {}
        denom = self._count[idx]
        # Relative-SR guard: trust co-occurrence only once the deep skill is solved often enough.
        if self._total <= 0 or (denom / self._total) < MIN_SR:
            return {}
        out: dict[str, float] = {}
        row = self._cooc[idx]
        for j in range(NUM_ACHIEVEMENTS):
            if j == idx:
                continue
            freq = row[j] / denom
            if freq >= min_freq and row[j] > 0:
                out[VALUE_TO_ACHIEVEMENT[j]] = freq
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def render_prereq_hint(self, deep_achievement: str, top_k: int = 8, min_freq: float = 0.3) -> str:
        """Compact text for the modeler prompt: the skills most often co-reached with a deep skill.

        Empty string when the deep skill's empirical SR is below MIN_SR (solved too rarely to trust),
        so the prompt simply omits the (c) hint and the modeler leans on (b) mechanics — exactly the
        phased fallback in v6_design.md §3.8.
        """
        freqs = self.prereq_frequencies(deep_achievement, min_freq=min_freq)
        if not freqs:
            return ""
        supp = self.support(deep_achievement)
        sr_pct = int(round(self.deep_sr(deep_achievement) * 100))
        items = list(freqs.items())[:top_k]
        body = ", ".join(f"{name} {int(round(f * 100))}%" for name, f in items)
        return (
            f"When the student SUCCEEDED at {deep_achievement} ({supp} successful episodes, "
            f"~{sr_pct}% SR), it also reached: {body}. (High % = a prerequisite it genuinely strings "
            f"together; low/absent = a link it does NOT yet have when it wins — a candidate unmastered "
            f"link to train.)"
        )
