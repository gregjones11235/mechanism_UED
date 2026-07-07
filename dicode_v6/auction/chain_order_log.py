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
            t: {"n_fail": 0, "n_succ": 0, "depth_sum": 0, "last_link": {}, "missing": {}}
            for t in targets
        }

        for row, fin in zip(first_step_rows, finished):
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
            for t, links in targets.items():
                agg = fail_agg[t]
                if t in reached_names:
                    agg["n_succ"] += 1
                    continue
                agg["n_fail"] += 1
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

        for t, links in targets.items():
            agg = fail_agg[t]
            if agg["n_fail"] + agg["n_succ"] == 0:
                continue
            mean_depth = (agg["depth_sum"] / agg["n_fail"]) if agg["n_fail"] else 0.0
            self._fail_hist.append(
                {
                    "session": int(session_idx),
                    "target": t,
                    "links": links,
                    "n_fail": agg["n_fail"],
                    "n_succ": agg["n_succ"],
                    "mean_depth": round(mean_depth, 4),
                    "last_link": agg["last_link"],
                    "missing": agg["missing"],
                }
            )
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
        """The CHAIN EVIDENCE block for one wall, <= 3 lines (design §P2.3). Empty when there is
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
            if last:
                mode_link, mode_cnt = max(last.items(), key=lambda kv: kv[1])
                pct = int(round(mode_cnt / n_fail * 100))
                if mode_link == "(none)":
                    lines.append(
                        f"{t}: {pct}% of {n_fail} failing episodes die before reaching ANY chain "
                        f"link — the chain's first link is the real wall."
                    )
                else:
                    try:
                        nxt = links[links.index(mode_link) + 1]
                    except (ValueError, IndexError):
                        nxt = t
                    miss = dict(entry.get("missing") or {})
                    miss_pct = int(round(miss.get(nxt, n_fail) / n_fail * 100)) if nxt != t else 100
                    lines.append(
                        f"{t}: failing episodes (n={n_fail}) most often break AFTER {mode_link} "
                        f"({pct}%), before {nxt} (missing in {miss_pct}% of failures) — that link is "
                        f"where the chain snaps."
                    )
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
