"""ChainOrderLog — DIRECTED skill-chain order + break-link mining from held-out eval (v6fix7 P2).

Why this exists (v6fix7 design §P2): the (c) co-occurrence matrix (cooccurrence_log) answers "WHICH
skills co-occur with a deep win" but throws away ORDER — the frontier survey (2026-07) flagged that
as the pipeline's biggest naive point. Two things only order can give:

  1. **Directed chains from successful episodes**: sort each finished episode's achievements by the
     step each was FIRST reached (``first_step``), then accumulate adjacent 2-gram / 3-gram counts.
     "collect_iron -> make_iron_sword -> defeat_troll" is evidence of the path the student actually
     walks, not just a bag of correlated skills.
  2. **Break-link mining from FAILED episodes**: for a siege wall with an ordered prereq chain, the
     episodes that do NOT reach the wall still tell us the DEEPEST chain link they died after. The
     distribution of that break link is the strongest focus-selection evidence there is — and its
     movement DEEPER across sessions ("chain-frontier advance") is progress even while the wall's SR
     sits at 0% (the tier4 patience signal, wired to SiegeNotebook.note_chain_progress / the P1a
     blacklist escape).

Same persistence contract as CooccurrenceLog (accumulate raw counts across sessions, never average
away; atomic JSON in the run cwd; per-session idempotency for resume). Pure python — no jax/craftax —
so it is unit-testable offline; the eval side hands it a per-env ``first_step`` matrix + finished mask.
"""

from __future__ import annotations

import json
import os

from auction.cooccurrence_log import MIN_SR
from auction.craftax_achievements import (
    ACHIEVEMENT_TO_VALUE,
    NUM_ACHIEVEMENTS,
    VALUE_TO_ACHIEVEMENT,
)

_DEFAULT_PATH = "chain_order_counts.json"

# Minimum number of FAILING episodes (w.r.t. one wall, one session) before that session's break-link
# distribution is trusted. Held-out eval runs ~1024 episodes; for an unsolved tier3/4 wall nearly all
# of them fail, so this only filters genuinely tiny samples (e.g. a wall that is almost mastered).
FRONTIER_MIN_FAILS = 64

# How much the mean break-link depth (in chain-link units) must rise ABOVE the best previous
# comparable session before we call it a frontier ADVANCE. With n_fail >= FRONTIER_MIN_FAILS the
# sample noise on the mean is well under this; 0.25 links means "a quarter of the cohort moved one
# link deeper" — a real distribution shift, not jitter.
FRONTIER_ADVANCE_LINKS = 0.25

# Bound on the persisted per-(session, wall) failure summaries. A full run is ~153 sessions and can
# track up to ~6 walls at once (active foci + retired), i.e. ~900 entries end-to-end; 1200 keeps the
# WHOLE run's history (the frontier ratchet compares against the best PREVIOUS session — truncating
# old entries would shorten that memory and let noise re-earn an "advance" it already spent).
_FAIL_HIST_MAX = 1200

# Greedy backward-walk guards for the dominant success path: stop when the best immediate
# predecessor explains under this fraction of the successor's episodes, and never walk deeper than
# this many hops (the prompt line stays one line).
_PATH_MIN_FRAC = 0.15
_PATH_MAX_HOPS = 3

# v6fix9 P0: a chain link whose GLOBAL empirical SR is at or above this is "universal" — virtually
# every episode reaches it regardless of the wall (place_table sits at ~99.5%). When such a link is
# the modal deepest-achieved link of the failing episodes, that is an ARTIFACT of (universal link ×
# deep position in the LLM-submitted chain order), not evidence the chain "snaps" there: the honest
# reading is "failures walk the whole proposed chain and still fail the final step". The old
# unconditional "most often break AFTER X, before <target> (missing in 100%)" sentence was tautologic
# for that shape and fed the make_iron_armour "cannot craft under pressure" misdiagnosis (22 sessions).
# Calibrated on job 3691755 cumulative data (77824 episodes, 2026-07-08): the empirical SR spectrum
# has a hole between collect_coal (0.881, a REAL armour-chain discriminator link, still drifting up)
# and wake_up (0.940); 0.92 sits in it with margin for coal's drift, so coal keeps its located-break
# wording while the torch/arrow/stone-tool near-universals fall on the artifact-guard side. The
# guard is wording-level only (the missing histogram always renders), so boundary drift degrades
# phrasing, never data.
_UNIVERSAL_SR = 0.92

# v6fix9 P1 — death-timing captions (PROVISIONAL until first-run calibration; a Craftax day is a few
# hundred steps and the observed mean episode is ~1200): a failing episode that survives LONGer than
# this past its deepest chain link was demonstrably NOT interrupted at the frontier; one that dies
# within SHORT of it plausibly was. In between, the render states the number without a verdict.
_SURVIVAL_LONG = 300
_SURVIVAL_SHORT = 50

# v6fix9 P1 — inventory forensics. The winners' median stockpile per resource is the yardstick (no
# hand-coded recipe table and no hand-picked resource list — the columns are the WHOLE inventory
# struct, enumerated programmatically at the eval side; user 2026-07-08 leak review). Only trust the
# yardstick once at least this many winning episodes back it.
_INV_MIN_SUCC = 8
# Readiness ratchet: the share of failures matching the winners' median stockpile must rise this far
# above the best previous comparable session to count as progress (n_fail >= FRONTIER_MIN_FAILS on
# both sides -> se ~1.4pp, so 5pp is ~3.5σ before the ratchet even applies).
_INV_ADVANCE_FRAC = 0.05

# v6fix9 #5 — frontier saturation: when mean break-link depth is within two ratchet-clicks
# (2 x FRONTIER_ADVANCE_LINKS) of the chain length, the depth channel is exhausted for this wall
# (enabler walls whose links are all mastered live here); progress evidence must come from the
# inventory channel instead, and the render stops implying "no frontier movement = no progress".
_SATURATION_HEADROOM = 2 * FRONTIER_ADVANCE_LINKS

# v6fix10 ⑦ — CHAIN-INCOMPLETE verdict. When at least this share of failing episodes reach the
# self-reported chain's TAIL link while the wall has ZERO wins ever (and no inventory gap explains
# it), the reported chain is complete-in-failures yet never sufficient — by pigeonhole it MISSES a
# prerequisite the LLM never named (the defeat_kobold case: chain [enter_dungeon, iron_sword,
# skeleton] all reached in 82% of failures, 0 wins in 22 sessions, because the true door — the
# sewers descent — was absent from the reported chain, so the missing histogram could never show it).
_CHAIN_COMPLETE_FRAC = 0.6
# v6fix10.1 hazard-2: "zero wins" must tolerate the 1-in-1024 fluke — with ~1000 episodes/session a
# wall whose true SR is epsilon (sewers-line skills sit at exactly 0.1%) accrues ~1 lucky win per
# session, and a strict support()>0 check would permanently disable the verdict for exactly the
# walls it exists for. A win RATE at/below this epsilon still counts as "never wins".
_CHAIN_INCOMPLETE_WIN_EPS = 0.01

# ---- v7fix5.0 P1: access frontier + conditional pass-through (access-wall root-cause fix) --------
# Calibrated on the s213 gnome forensics (2026-07-14): held-out enter_gnomish_mines reach = 18.6%
# (the 95.5% -> 18.6% cliff at the floor1->2 descent), cond past it = 153/190 = 80.5% ~= trained
# 78% — combat transferred, access binds. 0.35 sits above every observed capped link (mines
# 18-23%, sewers ~0) and below every passing one (dungeon 95%, orc fights 76-86%); 0.60 separates
# "execution transferred" (gnome 0.81) from zero-win walls (kobold cond 0.005). Sample guards keep
# a 2%-reach link (reached_n ~20 of 1024) from certifying on noise.
ACCESS_CAP_REACH = 0.35        # a link REACHED by fewer than this fraction of episodes is a frontier
ACCESS_COND_TRANSFERRED = 0.60  # cond above this = execution past the frontier has transferred
ACCESS_MIN_EPISODES = 50        # minimum episodes in the fail_hist entry to judge at all
ACCESS_MIN_REACHED = 50         # minimum episodes PAST the frontier for the cond certificate


def _median_int(values) -> int | None:
    """Median of an int list (lower middle for even n — conservative for stockpile readings)."""
    s = sorted(values)
    return s[(len(s) - 1) // 2] if s else None


class ChainOrderLog:
    """Cross-session directed 2-gram/3-gram chain counts + per-wall break-link history."""

    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_PATH
        (
            self._count,      # count[i] = finished episodes reaching achievement i
            self._gram2,      # gram2[i][j] = finished episodes where j FOLLOWS i adjacently in time
            self._gram3,      # sparse {"a>b>c": count} adjacent triples (paper/ablation material)
            self._fail_hist,  # per-(session, wall) break-link summaries, append-only bounded
            self._sessions,   # session idempotency
            self._total,      # accumulated finished episodes (SR denominator, same guard as cooc)
        ) = self._load()

    # ---- persistence (mirrors CooccurrenceLog: atomic write, resume-safe) -----------------------

    def _load(self):
        n = NUM_ACHIEVEMENTS
        empty = ([0] * n, [[0] * n for _ in range(n)], {}, [], [], 0)
        if not os.path.exists(self.path):
            return empty
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            count = data.get("count") or [0] * n
            gram2 = data.get("gram2") or [[0] * n for _ in range(n)]
            gram3 = data.get("gram3") or {}
            fail_hist = data.get("fail_hist") or []
            sessions = data.get("sessions") or []
            total = int(data.get("total") or 0)
            if len(count) != n or len(gram2) != n or any(len(row) != n for row in gram2):
                return empty  # stale-shape file -> start clean rather than crash
            if not isinstance(gram3, dict) or not isinstance(fail_hist, list):
                return empty
            return count, gram2, gram3, fail_hist, sessions, total
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return empty

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "count": self._count,
                    "gram2": self._gram2,
                    "gram3": self._gram3,
                    "fail_hist": self._fail_hist,
                    "sessions": self._sessions,
                    "total": self._total,
                },
                f,
            )
        os.replace(tmp, self.path)

    # ---- accumulate one session -----------------------------------------------------------------

    def add_session(
        self,
        session_idx: int,
        first_step_rows,
        finished,
        names=None,
        chain_targets: dict[str, list[str]] | None = None,
        end_steps=None,
        died=None,
        max_inv=None,
        inv_names=None,
    ) -> None:
        """Fold one held-out eval's per-env first-achievement-step matrix into the running totals.

        ``first_step_rows``: [num_envs][num_columns] ints — the step at which each achievement was
            FIRST reached in that env's episode, -1 if never reached.
        ``finished``: [num_envs] bools — whether that env's episode terminated inside the eval
            horizon. Only finished episodes are counted (same convention as CooccurrenceLog).
        ``names``: optional column labels; if given, columns are remapped by name into the canonical
            achievement order (unknown names dropped). If None, columns are assumed to already be in
            canonical craftax enum order.
        ``chain_targets``: {wall_skill: [ordered link names, shallow -> deep]} — the walls to mine
            break-link distributions for (active sieges + retired walls, from
            SiegeNotebook.chain_targets()). Episodes NOT reaching the wall contribute their deepest
            achieved link to that wall's per-session failure summary.
        v6fix9 P1 (all optional; absent -> the corresponding forensic fields are simply not stored,
        old callers keep working byte-identically):
        ``end_steps``: [num_envs] ints — the step each episode ENDED at (its length).
        ``died``: [num_envs] bools — whether the episode ended by death (vs the step cap). Craftax
            episodes end only by death or the cap, so the caller derives this as end < max_steps.
        ``max_inv``: [num_envs][num_inv_cols] ints — per-episode PEAK inventory count per column.
        ``inv_names``: labels for max_inv's columns (the whole inventory struct, enumerated
            programmatically at the eval side — no hand-picked resource list).
        Idempotent per session (resume-safe)."""
        if session_idx in self._sessions:
            return
        n = NUM_ACHIEVEMENTS
        if names is None:
            canon = list(range(n))
        else:
            canon = [ACHIEVEMENT_TO_VALUE.get(str(nm).lower()) for nm in names]

        # per-wall failure aggregation for THIS session. A target with an EMPTY chain is skipped
        # entirely: a freshly-opened focus has no prereq_tree yet, and mining it would only produce
        # a junk "(none) 100%" entry that renders as the misleading "the chain's first link is the
        # real wall" line — there is no chain to break yet.
        targets = {}
        for t, links in (chain_targets or {}).items():
            ll = [str(l).lower() for l in (links or [])]
            if ll:
                targets[str(t).lower()] = ll
        fail_agg = {
            t: {"n_fail": 0, "n_succ": 0, "depth_sum": 0, "last_link": {}, "missing": {},
                "died": 0, "after": [], "inv_fail_rows": [], "inv_succ_rows": []}
            for t in targets
        }
        # v6fix9 P1: forensic inputs are optional — absent means the fields are simply not stored.
        has_death = end_steps is not None and died is not None
        inv_cols = [str(n) for n in (inv_names or [])]
        has_inv = max_inv is not None and bool(inv_cols)

        for env_i, (row, fin) in enumerate(zip(first_step_rows, finished)):
            if not fin:
                continue  # unfinished (horizon-truncated) episodes carry no full-episode evidence
            self._total += 1
            seq = []  # [(step, canonical_idx)]
            for k, ci in enumerate(canon):
                if ci is None:
                    continue
                step = int(row[k])
                if step >= 0:
                    seq.append((step, ci))
            seq.sort()
            idxs = [ci for _, ci in seq]
            reached = set(idxs)
            for ci in idxs:
                self._count[ci] += 1
            for a, b in zip(idxs, idxs[1:]):
                self._gram2[a][b] += 1
            for a, b, c in zip(idxs, idxs[1:], idxs[2:]):
                key = f"{VALUE_TO_ACHIEVEMENT[a]}>{VALUE_TO_ACHIEVEMENT[b]}>{VALUE_TO_ACHIEVEMENT[c]}"
                self._gram3[key] = self._gram3.get(key, 0) + 1

            reached_names = {VALUE_TO_ACHIEVEMENT[ci] for ci in reached}
            step_by_name = {VALUE_TO_ACHIEVEMENT[ci]: st for st, ci in seq}
            for t, links in targets.items():
                agg = fail_agg[t]
                if t in reached_names:
                    agg["n_succ"] += 1
                    if has_inv:
                        agg["inv_succ_rows"].append(env_i)
                    continue
                agg["n_fail"] += 1
                if has_inv:
                    agg["inv_fail_rows"].append(env_i)
                if has_death and bool(died[env_i]):
                    agg["died"] += 1
                # Two depth readings per failing episode:
                #   - ``deepest`` (position of the deepest achieved link, 1-based) — only for the
                #     modal break-link LINE (which link the episode died after);
                #   - ``achieved`` (COUNT of chain links achieved) — the metric behind mean_depth /
                #     the frontier ratchet. Count is ORDER-INVARIANT, so the frontier comparison
                #     survives the LLM reordering the prereq_tree between sessions (it re-submits
                #     the tree every session; a position-based mean would silently lose all its
                #     comparable history on every reorder).
                deepest = 0
                achieved = 0
                for pos, link in enumerate(links, start=1):
                    if link in reached_names:
                        deepest = pos
                        achieved += 1
                last = links[deepest - 1] if deepest > 0 else "(none)"
                agg["last_link"][last] = agg["last_link"].get(last, 0) + 1
                agg["depth_sum"] += achieved
                for link in links:
                    if link not in reached_names:
                        agg["missing"][link] = agg["missing"].get(link, 0) + 1
                # v6fix9 P1: how long did this failure SURVIVE past its deepest achieved link?
                # Long survival refutes "interrupted at the frontier"; near-zero supports it.
                if has_death and achieved > 0:
                    deepest_step = max(
                        (step_by_name[l] for l in links if l in step_by_name), default=-1
                    )
                    if deepest_step >= 0:
                        agg["after"].append(int(end_steps[env_i]) - int(deepest_step))

        for t, links in targets.items():
            agg = fail_agg[t]
            if agg["n_fail"] + agg["n_succ"] == 0:
                continue
            mean_depth = (agg["depth_sum"] / agg["n_fail"]) if agg["n_fail"] else 0.0
            entry = {
                "session": int(session_idx),
                "target": t,
                "links": links,
                "n_fail": agg["n_fail"],
                "n_succ": agg["n_succ"],
                "mean_depth": round(mean_depth, 4),
                "last_link": agg["last_link"],
                "missing": agg["missing"],
            }
            # v6fix9 P1 forensic fields — only stored when the inputs were supplied this session
            # (old files / old callers keep the old shape; every reader treats them as optional).
            if has_death and agg["n_fail"]:
                entry["died_frac"] = round(agg["died"] / agg["n_fail"], 4)
                med_after = _median_int(agg["after"])
                if med_after is not None:
                    entry["after_deepest_med"] = int(med_after)
            if has_inv and len(agg["inv_succ_rows"]) >= _INV_MIN_SUCC and agg["inv_fail_rows"]:
                inv_summary = {}
                for c, col in enumerate(inv_cols):
                    try:
                        succ_vals = [int(max_inv[r][c]) for r in agg["inv_succ_rows"]]
                        fail_vals = [int(max_inv[r][c]) for r in agg["inv_fail_rows"]]
                    except (IndexError, TypeError, ValueError):
                        continue
                    succ_med = _median_int(succ_vals)
                    if not succ_med:
                        continue  # winners don't stockpile this column -> no yardstick, skip
                    fail_med = _median_int(fail_vals) or 0
                    ready = sum(1 for v in fail_vals if v >= succ_med) / len(fail_vals)
                    inv_summary[col] = {
                        "succ_med": int(succ_med),
                        "fail_med": int(fail_med),
                        "ready_frac": round(ready, 4),
                    }
                if inv_summary:
                    entry["inv"] = inv_summary
            self._fail_hist.append(entry)
        self._fail_hist = self._fail_hist[-_FAIL_HIST_MAX:]
        self._sessions.append(session_idx)
        self._save()

    # ---- queries ----------------------------------------------------------------------------------

    def support(self, achievement: str) -> int:
        """Accumulated finished episodes that reached ``achievement``."""
        idx = ACHIEVEMENT_TO_VALUE.get((achievement or "").lower())
        return int(self._count[idx]) if idx is not None else 0

    def total_finished(self) -> int:
        return int(self._total)

    def deep_sr(self, achievement: str) -> float:
        if self._total <= 0:
            return 0.0
        return self.support(achievement) / self._total

    def dominant_path(self, target: str, hops: int = _PATH_MAX_HOPS) -> list[tuple[str, float]]:
        """Greedy backward walk on the 2-gram graph: the most common immediate predecessor of the
        target, then of THAT skill, etc. Returns [(skill, frac), ...] shallow -> deep, where ``frac``
        is the fraction of episodes reaching the successor whose immediate predecessor was ``skill``.
        Empty when the target's empirical SR is below MIN_SR (too rarely solved to trust — same
        relative guard as the (c) co-occurrence hint)."""
        idx = ACHIEVEMENT_TO_VALUE.get((target or "").lower())
        if idx is None or self._total <= 0 or (self._count[idx] / self._total) < MIN_SR:
            return []
        path: list[tuple[str, float]] = []
        seen = {idx}
        cur = idx
        for _ in range(max(0, hops)):
            denom = self._count[cur]
            if denom <= 0:
                break
            best_i, best_c = -1, 0
            for i in range(NUM_ACHIEVEMENTS):
                c = self._gram2[i][cur]
                if c > best_c and i not in seen:
                    best_i, best_c = i, c
            frac = best_c / denom if denom else 0.0
            if best_i < 0 or frac < _PATH_MIN_FRAC:
                break
            path.append((VALUE_TO_ACHIEVEMENT[best_i], frac))
            seen.add(best_i)
            cur = best_i
        return list(reversed(path))

    def _fail_entries(self, target: str) -> list[dict]:
        t = (target or "").lower()
        return [e for e in self._fail_hist if e.get("target") == t]

    def latest_fail_summary(self, target: str) -> dict | None:
        entries = self._fail_entries(target)
        return entries[-1] if entries else None

    def _modal_break(self, entry: dict):
        """(mode_link, pct, is_tail, is_universal) for one fail_hist entry, or None without data.

        is_tail / is_universal are the two artifact conditions (v6fix9 P0): a modal break at the
        chain's final link, or at a near-universal link, does not locate a snapping link."""
        n_fail = int(entry.get("n_fail", 0))
        last = dict(entry.get("last_link") or {})
        if not last or n_fail <= 0:
            return None
        links = entry.get("links") or []
        mode_link, mode_cnt = max(last.items(), key=lambda kv: kv[1])
        pct = int(round(mode_cnt / n_fail * 100))
        is_tail = bool(links) and mode_link == links[-1]
        is_universal = mode_link != "(none)" and self.deep_sr(mode_link) >= _UNIVERSAL_SR
        return mode_link, pct, is_tail, is_universal

    def frontier_saturated(self, target: str) -> bool:
        """v6fix9 #5: True when the chain-depth channel is exhausted for this wall — mean break-link
        depth is within two ratchet-clicks of the chain length (all links near-universally reached),
        so 'no frontier movement' carries no information and progress must be read off the inventory
        channel (inventory_advanced) instead."""
        entry = self.latest_fail_summary(target)
        if not entry:
            return False
        links = entry.get("links") or []
        if not links:
            return False
        return float(entry.get("mean_depth", 0.0)) >= len(links) - _SATURATION_HEADROOM

    def inventory_advanced(self, target: str) -> bool:
        """v6fix9 #5: the inventory-readiness ratchet — failures stockpiling measurably closer to
        the winners' median. True when, in the latest comparable session (same link SET, n_fail >=
        FRONTIER_MIN_FAILS on both sides, inventory data present), some inventory column's
        ready_frac rose by at least _INV_ADVANCE_FRAC over the best previous comparable reading of
        that same column. This is the progress/unlock evidence for walls whose chain frontier is
        SATURATED (frontier_advanced is structurally dead there)."""
        entries = [
            e for e in self._fail_entries(target)
            if e.get("n_fail", 0) >= FRONTIER_MIN_FAILS and e.get("inv")
        ]
        if len(entries) < 2:
            return False
        latest = entries[-1]
        latest_set = set(latest.get("links") or [])
        prev = [e for e in entries[:-1] if set(e.get("links") or []) == latest_set]
        if not prev:
            return False
        for col, d in (latest.get("inv") or {}).items():
            hist = [
                float(e["inv"][col].get("ready_frac", 0.0))
                for e in prev if col in (e.get("inv") or {})
            ]
            if hist and float(d.get("ready_frac", 0.0)) >= max(hist) + _INV_ADVANCE_FRAC:
                return True
        return False

    def chain_incomplete(self, target: str) -> bool:
        """v6fix10 ⑦: True when the self-reported chain is COMPLETE in failures yet the wall has
        ZERO wins ever — the chain must be missing an unnamed prerequisite (the missing histogram
        only covers reported links, so it can never surface the omission itself). Conditions: a
        trustworthy failure sample; >= _CHAIN_COMPLETE_FRAC of failures REACH the chain's tail
        link; zero accumulated wins for the wall; and no inventory gap vs winners (a quantity gap
        is a different, P1-diagnosable disease — resource_shortfall, not a missing link)."""
        entry = self.latest_fail_summary(target)
        if not entry or int(entry.get("n_fail", 0)) < FRONTIER_MIN_FAILS:
            return False
        links = entry.get("links") or []
        if not links:
            return False
        # v6fix10.1 hazard-2: win RATE across this wall's tracked sessions, not a strict zero — a
        # fluke episode (SR 0.098% on a 1024-env eval) is not evidence the reported chain suffices.
        entries = self._fail_entries(target)
        wins = sum(int(e.get("n_succ", 0)) for e in entries)
        total = wins + sum(int(e.get("n_fail", 0)) for e in entries)
        if total > 0 and (wins / total) > _CHAIN_INCOMPLETE_WIN_EPS:
            return False  # the wall IS being won beyond fluke rate — the chain can be sufficient
        inv = entry.get("inv") or {}
        if any(int(d.get("succ_med", 0)) > int(d.get("fail_med", 0)) for d in inv.values()):
            return False  # a measured quantity gap explains the losses; not a missing link
        n_fail = int(entry["n_fail"])
        tail = links[-1]
        tail_reached = 1.0 - (int((entry.get("missing") or {}).get(tail, 0)) / n_fail)
        return tail_reached >= _CHAIN_COMPLETE_FRAC

    def access_frontier(self, target: str) -> dict | None:
        """v7fix5.0 P1: the wall's ACCESS FRONTIER — the shallowest chain link most episodes never
        reach — plus the conditional pass-through past it.

        Root cause this mechanizes (s213 forensics, defeat_gnome_warrior): 96% of held-out
        failures never entered the mines (reach 18.6%), while episodes that DID reach them won at
        cond = n_succ/(n_succ+n_fail-missing[link]) = 153/190 = 81% ~= the trained SR — combat had
        fully transferred and the binding constraint was ACCESS, yet the attribution blamed the
        diamond-gear chain (action-press counts misread as crafting successes). The frontier and
        cond are computed from fields fail_hist has always carried; no new telemetry.

          frontier    = FIRST link (prereq order, shallow->deep) with reach_frac < ACCESS_CAP_REACH
                        ("shallowest binding link": kobold's frontier is the mines, not the sewers)
          cond        = n_succ / reached_n over the SAME episodes (execution quality past the gate)
          certified   = cond >= ACCESS_COND_TRANSFERRED with reached_n >= ACCESS_MIN_REACHED —
                        "execution transferred, ONLY access binds" (gnome-shaped). Zero-win walls
                        (kobold/enchant) keep frontier for attribution but are NOT certified: the
                        chain still has unsolved work past the frontier, so parking them is not
                        justified by this evidence alone.
        None when there is no frontier or the episode sample is too small (< ACCESS_MIN_EPISODES)."""
        entry = self.latest_fail_summary(target)
        if not entry:
            return None
        n_fail = int(entry.get("n_fail", 0))
        n_succ = int(entry.get("n_succ", 0))
        total = n_fail + n_succ
        if total < ACCESS_MIN_EPISODES:
            return None
        links = [str(l).lower() for l in (entry.get("links") or [])]
        if not links:
            return None
        miss = {str(k).lower(): int(v) for k, v in (entry.get("missing") or {}).items()}
        for idx, link in enumerate(links):
            reach = 1.0 - miss.get(link, 0) / total
            if reach < ACCESS_CAP_REACH:
                reached_n = total - miss.get(link, 0)
                cond = (n_succ / reached_n) if reached_n > 0 else 0.0
                return {
                    "frontier": link,
                    "frontier_idx": idx,
                    "reach_frac": round(reach, 4),
                    "cond": round(cond, 4),
                    "certified": bool(
                        reached_n >= ACCESS_MIN_REACHED and cond >= ACCESS_COND_TRANSFERRED
                    ),
                    "n_episodes": total,
                    "reached_n": reached_n,
                }
        return None

    def forensics(self, target: str) -> dict | None:
        """v6fix9 P2: the machine-readable failure summary the attribution gate cross-checks the
        modeler's causal claims against (missing histogram / artifact-aware break shape / death
        timing / inventory gaps). None when there is no trustworthy failure sample — the gate then
        cannot verify and coerces claims to 'unknown' only when they CONTRADICT data, never when
        data is merely absent."""
        entry = self.latest_fail_summary(target)
        if not entry or entry.get("n_fail", 0) < FRONTIER_MIN_FAILS:
            return None
        n_fail = int(entry["n_fail"])
        t = (target or "").lower()
        miss = dict(entry.get("missing") or {})
        missing_top = sorted(
            ((k, v / n_fail) for k, v in miss.items() if k != t and v > 0),
            key=lambda kv: -kv[1],
        )[:3]
        mb = self._modal_break(entry)
        inv = entry.get("inv") or {}
        inv_gaps = [
            (c, int(d["succ_med"]), int(d["fail_med"]), float(d["ready_frac"]))
            for c, d in inv.items() if d.get("succ_med", 0) > d.get("fail_med", 0)
        ]
        # v6fix10 ⑥: the wall's own AMBIENT after-deepest survival (median over its history) — the
        # relative yardstick for the interrupted_by_combat check. fix9 first-run calibration: the
        # absolute 50-step prior can never fire (ambient observed 134-468 across every wall); an
        # interruption pattern is "well below this wall's own ambient", not "below a fixed count".
        _afters = [
            int(e["after_deepest_med"]) for e in self._fail_entries(target)
            if e.get("after_deepest_med") is not None
        ]
        return {
            "n_fail": n_fail,
            "links": list(entry.get("links") or []),
            "missing_top": [(k, round(f, 4)) for k, f in missing_top],
            "break_at_final": bool(mb and (mb[2] or mb[3])),
            "died_frac": entry.get("died_frac"),
            "after_deepest_med": entry.get("after_deepest_med"),
            "after_deepest_ambient_med": _median_int(_afters),
            "chain_incomplete": self.chain_incomplete(target),
            "inv_gaps": sorted(inv_gaps, key=lambda x: -(x[1] - x[2])),
            # v7fix5.0 P1: access frontier + conditional pass-through ride the forensics pack the
            # attribution gate already consumes — no new plumbing.
            "access": self.access_frontier(target),
        }

    def frontier_advanced(self, target: str) -> bool:
        """True when the LATEST session's mean break-link depth for ``target`` rose by at least
        FRONTIER_ADVANCE_LINKS above the best PREVIOUS comparable session (same link SET, enough
        failing episodes on both sides). This is the P1a patience/blacklist-escape signal: failure
        episodes dying measurably deeper along the chain = progress even at 0% wall SR.

        Comparable = same SET of chain links (order ignored): mean_depth counts achieved links, so
        it is order-invariant, and the LLM re-submits the prereq_tree every session — requiring the
        exact ordered list would silently discard the whole comparison history on every reorder.
        Membership changes (a link added/removed) DO reset comparability, as they must: the depth
        denominator changed."""
        entries = [e for e in self._fail_entries(target) if e.get("n_fail", 0) >= FRONTIER_MIN_FAILS]
        if len(entries) < 2:
            return False
        latest = entries[-1]
        latest_set = set(latest.get("links") or [])
        prev = [e for e in entries[:-1] if set(e.get("links") or []) == latest_set]
        if not prev:
            return False
        best_prev = max(float(e.get("mean_depth", 0.0)) for e in prev)
        return float(latest.get("mean_depth", 0.0)) >= best_prev + FRONTIER_ADVANCE_LINKS

    # ---- prompt rendering ---------------------------------------------------------------------------

    def render_chain_hint(self, target: str) -> str:
        """The CHAIN EVIDENCE block for one wall, <= 6 lines (design §P2.3 + v6fix9 P0/P1 forensics:
        success path / missing histogram / break link with artifact guard / death timing /
        inventory gap / frontier). Empty when there is
        neither a trustworthy success path (SR below MIN_SR) nor a trustworthy failure sample
        (fewer than FRONTIER_MIN_FAILS failing episodes) — the prompt then simply omits it and the
        modeler leans on (b) mechanics + (c) co-occurrence, the usual phased fallback."""
        t = (target or "").lower()
        lines: list[str] = []

        path = self.dominant_path(t)
        if path:
            hops = "->".join(f"{name}({int(round(frac * 100))}%)" for name, frac in path)
            lines.append(
                f"{t}: successful episodes most often arrive via {hops}->{t} "
                f"(% = share of the successor's wins with that immediate predecessor)."
            )

        entry = self.latest_fail_summary(t)
        if entry and entry.get("n_fail", 0) >= FRONTIER_MIN_FAILS:
            n_fail = int(entry["n_fail"])
            links = entry.get("links") or []
            last = dict(entry.get("last_link") or {})
            miss = dict(entry.get("missing") or {})
            # v6fix9 P0 forensic line: the FULL missing histogram (top 3). This is the field that
            # separates "a chain link is unreachable" from "every link is reached and the gap is
            # inside the final step" — it was collected from day one but never rendered, which is
            # how a narrative could survive 22 sessions of contradicting data.
            top_missing = sorted(
                ((k, v) for k, v in miss.items() if k != t and v > 0), key=lambda kv: -kv[1]
            )[:3]
            if top_missing:
                parts = ", ".join(
                    f"{k} missing in {int(round(v / n_fail * 100))}%" for k, v in top_missing
                )
                lines.append(
                    f"{t}: forensics over {n_fail} failing episodes — {parts}; links not listed "
                    f"were reached in nearly every failure. (Achievements are binary: 'reached "
                    f"once' can still be too little of a resource for the recipe.)"
                )
            # v7fix5.0 P1: the access-frontier verdict line — the number the diamond-gear
            # misdiagnosis never saw. Rendered for EVERY wall with a frontier; the cond
            # certificate ("execution transferred") only when earned.
            ax = self.access_frontier(t)
            if ax:
                aline = (
                    f"{t}: BINDING-ACCESS={ax['frontier']} — only "
                    f"{int(round(ax['reach_frac'] * 100))}% of episodes ever reach it"
                )
                if ax.get("certified"):
                    aline += (
                        f", and those that do complete the wall at "
                        f"{int(round(ax['cond'] * 100))}% (execution has TRANSFERRED; the wall's "
                        f"own step is NOT the problem). Tactics, gear or drills below this link "
                        f"cannot move held-out — the frontier link itself must be sieged first."
                    )
                else:
                    aline += (
                        f" (pass-through beyond it: {int(round(ax['cond'] * 100))}%). Everything "
                        f"downstream is unmeasurable until this link opens — attribute to the "
                        f"frontier, not to gear or tactics past it."
                    )
                lines.append(aline)
            if last:
                mode_link, mode_cnt = max(last.items(), key=lambda kv: kv[1])
                pct = int(round(mode_cnt / n_fail * 100))
                is_tail = bool(links) and mode_link == links[-1]
                is_universal = mode_link != "(none)" and self.deep_sr(mode_link) >= _UNIVERSAL_SR
                if mode_link == "(none)":
                    lines.append(
                        f"{t}: {pct}% of {n_fail} failing episodes die before reaching ANY chain "
                        f"link — the chain's first link is the real wall."
                    )
                elif is_tail or is_universal:
                    # Artifact guard (v6fix9 P0): a modal break at the chain's tail, or at a
                    # near-universal link, does NOT locate a snapping link — say what it actually
                    # means instead of implying "reaches the station, fails under pressure".
                    if self.chain_incomplete(t):
                        # v6fix10 ⑦: complete-in-failures + ZERO wins ever + no quantity gap —
                        # the old "gap is inside the final step" wording re-created the armour
                        # misleader for combat walls (defeat_kobold: every reported link reached,
                        # 0 wins, because the TRUE door was never in the reported chain). Say the
                        # only thing the data supports: the chain itself is missing a link.
                        lines.append(
                            f"{t}: {pct}% of failures reach the END of your reported chain "
                            f"({mode_link}) yet the wall has ZERO wins ever and no resource "
                            f"gap explains it — your reported chain is COMPLETE in failures but "
                            f"NOT SUFFICIENT: it is MISSING an unnamed prerequisite. Name what "
                            f"else must be true before this wall is even reachable (a place the "
                            f"student must get to, a state it must be in) and EXPAND the "
                            f"prereq_tree; do not re-attack on the current chain."
                        )
                    else:
                        why = []
                        if is_universal:
                            why.append("a near-universal skill reached in ~every episode")
                        if is_tail:
                            why.append("the final link of the proposed chain")
                        lines.append(
                            f"{t}: {pct}% of failures reach {mode_link} ({' and '.join(why)}) and "
                            f"STILL fail the wall — the chain does not snap at a missing link; the "
                            f"gap is inside the final step itself (resource QUANTITY, crafting "
                            f"context, or follow-through). Do not attribute it to pressure or "
                            f"interruption without direct evidence."
                        )
                else:
                    try:
                        nxt = links[links.index(mode_link) + 1]
                    except (ValueError, IndexError):
                        nxt = t
                    miss_pct = int(round(miss.get(nxt, n_fail) / n_fail * 100)) if nxt != t else 100
                    lines.append(
                        f"{t}: failing episodes (n={n_fail}) most often break AFTER {mode_link} "
                        f"({pct}%), before {nxt} (missing in {miss_pct}% of failures) — that link is "
                        f"where the chain snaps."
                    )
            # v6fix9 P1: death-timing line — the direct interruption test. Only rendered when the
            # eval side supplied death telemetry (old data renders exactly as before).
            if entry.get("died_frac") is not None:
                died_pct = int(round(float(entry["died_frac"]) * 100))
                dline = f"{t}: {died_pct}% of failures end by DEATH (the rest hit the step cap)"
                med_after = entry.get("after_deepest_med")
                if med_after is not None:
                    dline += (
                        f"; median survival AFTER the deepest reached link = {int(med_after)} steps"
                    )
                    if int(med_after) >= _SURVIVAL_LONG:
                        dline += (
                            " — failures live long past the chain frontier: NOT an "
                            "interruption problem"
                        )
                    elif int(med_after) <= _SURVIVAL_SHORT:
                        dline += (
                            " — failures die right at the frontier: interruption/lethality "
                            "is plausible"
                        )
                lines.append(dline + ".")
            # v6fix9 P1: inventory gap vs the winners' own stockpile (no recipe table — the
            # yardstick is the winners' median). Top-3 by gap; omitted when no gap or no data.
            inv = entry.get("inv") or {}
            gaps = sorted(
                ((c, d) for c, d in inv.items()
                 if int(d.get("succ_med", 0)) > int(d.get("fail_med", 0))),
                key=lambda cd: -(int(cd[1]["succ_med"]) - int(cd[1]["fail_med"])),
            )[:3]
            if gaps:
                parts = ", ".join(
                    f"{c}: winners' median {int(d['succ_med'])} vs failures' "
                    f"{int(d['fail_med'])} (only {int(round(float(d['ready_frac']) * 100))}% of "
                    f"failures reach the winners' level)"
                    for c, d in gaps
                )
                lines.append(f"{t}: resource gap vs winning episodes — {parts}.")
            # frontier movement (the patience signal, stated so the modeler reads 0% SR correctly);
            # same set-based comparability as frontier_advanced.
            _entry_set = set(entry.get("links") or [])
            comparable = [
                e for e in self._fail_entries(t)
                if e.get("n_fail", 0) >= FRONTIER_MIN_FAILS
                and set(e.get("links") or []) == _entry_set
            ]
            if len(comparable) >= 2:
                prev_best = max(float(e.get("mean_depth", 0.0)) for e in comparable[:-1])
                cur = float(entry.get("mean_depth", 0.0))
                if cur >= prev_best + FRONTIER_ADVANCE_LINKS:
                    lines.append(
                        f"{t}: break-link frontier ADVANCING (mean chain depth {prev_best:.1f} -> "
                        f"{cur:.1f} links) — failures are dying deeper; real progress even while the "
                        f"wall's SR is still ~0%."
                    )
        return "\n".join(lines)
