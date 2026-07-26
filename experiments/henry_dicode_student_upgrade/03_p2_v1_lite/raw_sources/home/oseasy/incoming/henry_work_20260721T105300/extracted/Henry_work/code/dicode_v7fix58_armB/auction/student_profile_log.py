"""StudentProfileLog — per-session snapshots of the student's held-out skill profile.

v5-debate (2026-07-03, v5_design.md §3): the modeler needs the student's *current* held-out SR
over time — NOT the frozen A/B/C/D archive status, which goes stale (a level graded 'A' and never
re-trained keeps its 'A' label even after the student forgets it; see the tier2->tier3 dependency-
collapse diagnosis). This log records, once per generation session, the full 67-achievement held-out
success-rate vector, appended to a time series. From it the modeler can tell:
  - RISING vs FLAT-NEAR-ZERO (normal-early-weak vs genuinely stalled)
  - a skill whose SR was once high and has since dropped (FORGETTING -> system rehearsal rescue)
  - single-reading dips that don't persist across snapshots (NOISY)

Storage: a plain JSON file ``student_profile_history.json`` in the run's working directory (the same
cwd where ``task_graph.graphml`` lives), so it survives resume and is trivially inspectable offline.
The log is append-only and idempotent per session (re-recording the same session overwrites, so a
resumed run doesn't duplicate).
"""

from __future__ import annotations

import json
import os

_DEFAULT_PATH = "student_profile_history.json"


class StudentProfileLog:
    """Append-only time series of the student's held-out skill profile, keyed by session index."""

    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_PATH
        # history: list of {"session": int, "profile": {skill_name: sr_percent}}
        self._history: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self) -> None:
        # Atomic-ish write: temp file then replace, so an interrupted write can't corrupt the log.
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _normalize(global_agent_profile: dict | None) -> dict[str, float]:
        """Extract {achievement_name: sr_percent} from a global_agent_profile dict.

        global_agent_profile keys look like ``skill_collect_wood`` with values in 0..100. We strip the
        ``skill_`` prefix and keep the percent as-is (0..100). Non-skill keys are ignored.
        """
        out: dict[str, float] = {}
        if not global_agent_profile:
            return out
        for key, value in global_agent_profile.items():
            if not isinstance(key, str) or not key.startswith("skill_"):
                continue
            if value is None:
                continue
            name = key[len("skill_"):].lower()
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def record(self, session_idx: int, global_agent_profile: dict | None) -> None:
        """Record (or overwrite) this session's held-out profile snapshot, then persist."""
        profile = self._normalize(global_agent_profile)
        # Overwrite if this session already recorded (idempotent on resume), else append.
        for entry in self._history:
            if entry.get("session") == session_idx:
                entry["profile"] = profile
                self._save()
                return
        self._history.append({"session": session_idx, "profile": profile})
        self._history.sort(key=lambda e: e.get("session", 0))
        self._save()

    def series_for(self, achievement: str, last_k: int | None = None) -> list[tuple[int, float]]:
        """Return [(session, sr_percent), ...] for one achievement, oldest first.

        Missing readings (achievement absent in a snapshot) are skipped, not zero-filled — absence in
        the global eval means SR was 0 or the skill wasn't reported; caller decides how to treat it.
        """
        name = achievement.lower()
        series = [
            (e["session"], e["profile"][name])
            for e in self._history
            if name in e.get("profile", {})
        ]
        if last_k is not None:
            series = series[-last_k:]
        return series

    def latest(self) -> dict[str, float]:
        """Most recent full profile snapshot ({achievement: sr_percent}), or {} if empty."""
        return dict(self._history[-1]["profile"]) if self._history else {}

    def recent(self, last_k: int) -> list[dict]:
        """The last k session snapshots (each {"session", "profile"}), oldest first."""
        return list(self._history[-last_k:])

    def forgetting_candidates(
        self,
        drop_pp: float = 20.0,
        min_peak: float = 40.0,
        combat_drop_pp: float | None = 10.0,
        combat_min_peak: float | None = 15.0,
    ) -> list[dict]:
        """Achievements whose SR peaked historically but has since dropped — FORGETTING signal.

        For each achievement seen in the log, compute its running peak SR and its latest SR. Flag it if
        it once reached >= ``min_peak`` and has since fallen by >= ``drop_pp`` percentage points. This
        is a cheap statistical prefilter the modeler can lean on (it still makes the final call, since a
        single-snapshot dip may be NOISY rather than true forgetting).

        FAMILY-SPLIT (v6 §2⑥, user 2026-07-05): a COMBAT achievement's milestone peak can be as low as
        ~15% (a wall broken only occasionally), so the default 40% peak bar would NEVER flag a
        forgetting combat milestone — violating the rule "anything worth recording as success
        experience must be able to trigger rehearsal". So COMBAT-family skills use a lower peak / drop
        (``combat_min_peak`` / ``combat_drop_pp``), gear/gather/explore keep the higher default. When
        the combat overrides are None they fall back to the shared ``min_peak`` / ``drop_pp``.

        Returns a list of {"achievement", "peak", "peak_session", "latest", "drop"} sorted by drop desc.
        """
        from auction.craftax_achievements import family_of

        c_peak = combat_min_peak if combat_min_peak is not None else min_peak
        c_drop = combat_drop_pp if combat_drop_pp is not None else drop_pp

        # Collect every achievement name that appears anywhere in the history.
        names: set[str] = set()
        for e in self._history:
            names.update(e.get("profile", {}).keys())

        out: list[dict] = []
        for name in names:
            series = self.series_for(name)
            if len(series) < 2:
                continue
            peak = -1.0
            peak_session = None
            for sess, sr in series:
                if sr > peak:
                    peak = sr
                    peak_session = sess
            latest = series[-1][1]
            drop = peak - latest
            is_combat = family_of(name.lower()) == "COMBAT"
            peak_bar = c_peak if is_combat else min_peak
            drop_bar = c_drop if is_combat else drop_pp
            if peak >= peak_bar and drop >= drop_bar:
                out.append(
                    {
                        "achievement": name,
                        "peak": peak,
                        "peak_session": peak_session,
                        "latest": latest,
                        "drop": drop,
                    }
                )
        out.sort(key=lambda d: -d["drop"])
        return out
