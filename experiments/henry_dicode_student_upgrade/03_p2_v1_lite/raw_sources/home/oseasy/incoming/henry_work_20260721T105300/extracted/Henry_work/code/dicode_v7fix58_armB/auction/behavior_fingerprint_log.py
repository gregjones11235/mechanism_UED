"""BehaviorFingerprintLog — cross-session ACTION fingerprint of the student's SUCCESSFUL episodes.

v6 problem-2 (user 2026-07-05). The (c) co-occurrence log (auction/cooccurrence_log.py) answers
"WHICH other achievements did the student also reach when it succeeded at a deep skill?" — a
correlation over the achievement bitmask. It cannot answer "HOW did the student behave in the episodes
where it won?" — the positioning / action mix / pacing that actually got the win. That "how" is exactly
what the siege ``style_note`` is meant to carry (a transferable "this is how I beat the wall"), yet the
modeler currently invents it from craftax priors because it never SEES what the student really did.

This log closes that gap with a CHEAP behavioural fingerprint (not full trajectories, see the design
note in problem-2): for every deep achievement, accumulate — over the FINISHED held-out episodes that
REACHED it — the per-episode ACTION-COUNT totals and the episode-length totals. From those it answers:

  ``fingerprint(deep)`` = {
      "n": episodes that reached deep,
      "avg_len": mean steps in a winning episode,
      "actions": { action_name: mean times used per winning episode }, sorted high->low,
  }

i.e. "when the student beats <deep>, a typical winning episode is ~84 steps and uses PLACE_STONE ~2.1x,
DO ~30x, MAKE_IRON_PICKAXE 1x, ..." — grounded evidence the modeler folds into the style_note instead
of guessing. Positioning is only partially captured (an action histogram is not a coordinate), which is
the accepted limitation of the cheap route; it still tells the modeler the real MOTION/CRAFT/COMBAT mix.

Why persistent + accumulated (mirrors CooccurrenceLog / StudentProfileLog): tier3 wins are relatively
rare, so a single session's fingerprint is noisy. We accumulate raw per-session totals (never averaging
away) and divide only at query time, so the means stabilise as evidence builds. Atomic-written +
reloaded on resume, lives in the run cwd next to siege_notebook.json / cooccurrence_counts.json.

Pure-python (no jax/craftax) so it is unit-testable offline; the eval side only has to hand it, per
session, three arrays keyed by achievement: action-count totals [num_ach, action_dim], step totals
[num_ach], and reached-episode counts [num_ach].
"""

from __future__ import annotations

import json
import os

from auction.craftax_achievements import (
    ACHIEVEMENT_TO_VALUE,
    NUM_ACHIEVEMENTS,
    VALUE_TO_ACHIEVEMENT,
)

_DEFAULT_PATH = "behavior_fingerprint.json"

# Canonical craftax action order (craftax.craftax.constants.Action, action_dim = 43). Kept as a module
# constant so this file stays pure-python (no craftax import) and offline-testable; the eval side emits
# columns in THIS order (or passes ``action_names`` for a name-remap, like CooccurrenceLog does).
ACTION_NAMES: list[str] = [
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE", "PLACE_TABLE",
    "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD", "REST", "DESCEND", "ASCEND",
    "MAKE_DIAMOND_PICKAXE", "MAKE_DIAMOND_SWORD", "MAKE_IRON_ARMOUR", "MAKE_DIAMOND_ARMOUR",
    "SHOOT_ARROW", "MAKE_ARROW", "CAST_FIREBALL", "CAST_ICEBALL", "PLACE_TORCH", "DRINK_POTION_RED",
    "DRINK_POTION_GREEN", "DRINK_POTION_BLUE", "DRINK_POTION_PINK", "DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW", "READ_BOOK", "ENCHANT_SWORD", "ENCHANT_ARMOUR", "MAKE_TORCH",
    "LEVEL_UP_DEXTERITY", "LEVEL_UP_STRENGTH", "LEVEL_UP_INTELLIGENCE", "ENCHANT_BOW",
]
NUM_ACTIONS = len(ACTION_NAMES)

# The four "pure movement / idle" actions — used to report a compact move-vs-act ratio so a style_note
# can say "this win was heavy exploration" vs "heavy crafting/combat" without listing 43 numbers.
_MOVE_ACTIONS = {"NOOP", "LEFT", "RIGHT", "UP", "DOWN", "SLEEP", "REST"}

# Minimum EMPIRICAL SUCCESS RATE of the deep skill before its fingerprint is trusted (mirrors
# CooccurrenceLog.MIN_SR): below this the skill is solved too rarely for the mean action mix to mean
# anything, so the hint is omitted and the modeler leans on (b) mechanics / (c) co-occurrence only.
MIN_SR = 0.03


class BehaviorFingerprintLog:
    """Cross-session accumulated action fingerprint of successful held-out episodes, per achievement."""

    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_PATH
        # _act[i]  = accumulated action-count totals over winning episodes of achievement i [action_dim]
        # _steps[i]= accumulated step totals over those episodes
        # _n[i]    = accumulated # of winning episodes for achievement i
        # _total   = accumulated # of FINISHED episodes (the SR denominator, like CooccurrenceLog._total)
        self._act, self._steps, self._n, self._sessions, self._total = self._load()

    # ---- persistence (mirrors CooccurrenceLog: atomic write, resume-safe) -----------------------

    def _load(self):
        n, a = NUM_ACHIEVEMENTS, NUM_ACTIONS
        empty = ([[0.0] * a for _ in range(n)], [0.0] * n, [0] * n, [], 0)
        if not os.path.exists(self.path):
            return empty
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            act = data.get("act") or [[0.0] * a for _ in range(n)]
            steps = data.get("steps") or [0.0] * n
            cnt = data.get("n") or [0] * n
            sessions = data.get("sessions") or []
            total = int(data.get("total") or 0)
            # shape-guard: a corrupt / stale-shape file falls back to empty rather than crashing.
            if len(act) != n or any(len(row) != a for row in act) or len(steps) != n or len(cnt) != n:
                return empty
            return act, steps, cnt, sessions, total
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return empty

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"act": self._act, "steps": self._steps, "n": self._n,
                 "sessions": self._sessions, "total": self._total},
                f,
            )
        os.replace(tmp, self.path)

    # ---- accumulate one session's fingerprint ---------------------------------------------------

    def add_session(
        self, session_idx: int, act_totals, step_totals, reached_counts,
        names=None, action_names=None, total=None,
    ) -> None:
        """Add one held-out eval's per-achievement action/step totals over winning episodes.

        ``act_totals``: [num_ach, action_dim] — summed action counts over the FINISHED episodes that
            reached each achievement (row i = the total action histogram of achievement i's wins).
        ``step_totals``: [num_ach] — summed episode lengths over those same winning episodes.
        ``reached_counts``: [num_ach] — # winning episodes per achievement (its cooc count[i]).
        ``names``: optional achievement labels for the ROWS; remapped into canonical order if given
            (robust to the eval iterating achievements in its own order). If None, rows are assumed in
            canonical craftax enum order and length NUM_ACHIEVEMENTS.
        ``action_names``: optional labels for the COLUMNS of act_totals; remapped into canonical action
            order if given. If None, columns are assumed already in canonical ACTION_NAMES order.
        ``total``: # FINISHED episodes this session (the SR denominator), accumulated for the MIN_SR
            guard. Optional -> degrades to "no denominator" if absent.

        Idempotent per session (resume-safe): re-adding the same session is skipped.
        """
        if session_idx in self._sessions:
            return
        n, a = NUM_ACHIEVEMENTS, NUM_ACTIONS
        # resolve row (achievement) -> canonical index
        row_idx = (
            [ACHIEVEMENT_TO_VALUE.get(str(nm).lower()) for nm in names]
            if names is not None else list(range(n))
        )
        # resolve column (action) -> canonical index
        col_idx = (
            [_action_to_index(str(nm)) for nm in action_names]
            if action_names is not None else list(range(a))
        )
        for src_r, ci in enumerate(row_idx):
            if ci is None or ci >= n:
                continue
            self._steps[ci] += float(step_totals[src_r])
            self._n[ci] += int(reached_counts[src_r])
            src_row, dst_row = act_totals[src_r], self._act[ci]
            for src_c, aj in enumerate(col_idx):
                if aj is None or aj >= a:
                    continue
                dst_row[aj] += float(src_row[src_c])
        if total is not None:
            self._total += int(total)
        self._sessions.append(session_idx)
        self._save()

    # ---- query ----------------------------------------------------------------------------------

    def total_finished(self) -> int:
        return int(self._total)

    def support(self, achievement: str) -> int:
        """# accumulated winning episodes for ``achievement``."""
        idx = ACHIEVEMENT_TO_VALUE.get(achievement.lower())
        return int(self._n[idx]) if idx is not None else 0

    def deep_sr(self, achievement: str) -> float:
        idx = ACHIEVEMENT_TO_VALUE.get(achievement.lower())
        if idx is None or self._total <= 0:
            return 0.0
        return self._n[idx] / self._total

    def fingerprint(self, achievement: str, top_k: int = 8, min_avg: float = 0.05) -> dict | None:
        """Mean per-winning-episode behaviour for ``achievement``, or None if solved too rarely.

        Returns {"n", "avg_len", "move_frac", "actions": {name: mean_uses_per_ep}} where ``actions`` is
        the top_k non-movement actions by mean usage (movement folded into ``move_frac`` so the modeler
        sees the interesting crafting/combat mix, not 5 walk actions dominating the list). None when the
        deep skill's empirical SR is below MIN_SR (single-run noise) — the caller then omits the hint.
        """
        idx = ACHIEVEMENT_TO_VALUE.get((achievement or "").lower())
        if idx is None:
            return None
        n_ep = self._n[idx]
        if n_ep <= 0 or self._total <= 0 or (n_ep / self._total) < MIN_SR:
            return None
        row = self._act[idx]
        total_actions = sum(row) or 1.0
        move_sum = sum(row[i] for i, nm in enumerate(ACTION_NAMES) if nm in _MOVE_ACTIONS)
        # mean uses-per-episode for every action; keep the informative (non-movement) ones.
        per_ep = {ACTION_NAMES[i]: row[i] / n_ep for i in range(NUM_ACTIONS)}
        informative = {
            name: v for name, v in per_ep.items()
            if name not in _MOVE_ACTIONS and v >= min_avg
        }
        informative = dict(sorted(informative.items(), key=lambda kv: -kv[1])[:top_k])
        return {
            "n": int(n_ep),
            "avg_len": round(self._steps[idx] / n_ep, 1),
            "move_frac": round(move_sum / total_actions, 2),
            "actions": {k: round(v, 2) for k, v in informative.items()},
        }

    def render_fingerprint_hint(self, achievement: str, top_k: int = 8) -> str:
        """Compact modeler-prompt text of how a typical WINNING episode of ``achievement`` behaves.

        Empty string when solved too rarely (fingerprint None) — phased fallback: the prompt omits it
        and the modeler leans on (b) mechanics + (c) co-occurrence, exactly like the (c) hint does.
        """
        fp = self.fingerprint(achievement, top_k=top_k)
        if fp is None:
            return ""
        sr_pct = int(round(self.deep_sr(achievement) * 100))
        acts = ", ".join(f"{name} {mean}x" for name, mean in fp["actions"].items()) or "(mostly movement)"
        # v7fix5.0 P1: label the counts as PRESSES — RL policies spam craft/interact buttons, so
        # "MAKE_DIAMOND_SWORD 41x" is ~41 attempts, NOT 41 crafts (the s207 misdiagnosis read
        # winners' press-spam as "winners build diamond gear before the fight" while the
        # achievement SR for that craft sat at 3-5%). Cross-check any crafting claim against the
        # SR profile before treating a press count as evidence the item was ever made.
        return (
            f"When the student WON {achievement} ({fp['n']} winning episodes, ~{sr_pct}% SR), a typical "
            f"winning episode ran ~{int(round(fp['avg_len']))} steps, was {int(round(fp['move_frac']*100))}% "
            f"movement/idle, and pressed: {acts} (ACTION PRESSES incl. failed attempts — a craft press "
            f"is NOT a successful craft; verify against the SR profile). (This is the real action mix / "
            f"pacing — ground the style_note in it, not an imagined tactic.)"
        )


def _action_to_index(name: str) -> int | None:
    """Canonical column index for an action name (case-insensitive), or None if unknown."""
    try:
        return ACTION_NAMES.index(name.upper())
    except ValueError:
        return None
