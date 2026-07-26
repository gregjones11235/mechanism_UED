"""v7fix5.2 designcheck — seat routing (retire->park) + access-root auto-nomination.

Root cause pinned (fix51 run s228-243, 2026-07-15): fix5.0's diagnosis layer worked perfectly
(ACCESS_CAPPED override 8/8, access_caps even found the transitive root make_iron_armour) but its
therapy layer never started — watch stayed {}, both seats welded to uncertified-starved deep walls
the cond>=0.6 certificate could not evict, the one free seat (s228) went to learn_fireball against
a rendered BINDING-ACCESS instruction the healthy modeler ignored 8/8, and the s238 budget
retirement was refilled by the gateway channel the same boundary. fix5.2 adds ZERO new judgement
(no thresholds, no verdicts, no eviction triggers) and two transmissions: P0 routes an
already-decreed death of a frontier-starved wall to a PARK (WATCH, full state, no strikes), and
P1 deterministically nominates the chased access ROOT through every existing admission gate.

Wiring-level assertions (grep-style, no jax needed). Run: python v7fix52_designcheck.py.
"""

import re

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


SNB = _read("auction", "siege_notebook.py")
MOD = _read("auction", "modeler.py")
GM = _read("src", "dicode", "dreaming", "gen_manager.py")
TESTS = _read("auction", "tests", "test_siege_fix52_seat_routing.py")


def _method_body(src: str, name: str) -> str:
    """The text of one method: from its def to the next def at the same indent."""
    m = re.search(rf"\n    def {name}\(.*?(?=\n    def )", src, re.S)
    assert m, f"method {name} not found"
    return m.group(0)


ROUTER = _method_body(SNB, "_retire_or_park")
NOMINATE = _method_body(SNB, "_access_auto_nominate")
VETO = _method_body(SNB, "_access_auto_veto")
CHASE = _method_body(SNB, "_access_root_of")
WATCHPASS = _method_body(SNB, "_process_watch")
ZEROWIN = _method_body(SNB, "zero_win_walls")

CHECKS = [
    # --- S.1: starved is derived, not invented ---
    ("S.1 starved = existing access-cap verdict minus the certificate — the router consults "
     "_access_cap only, introduces NO new numeric threshold, and certified caps never park here",
     "cap = self._access_cap(sl)" in ROUTER
     and 'if not cap or cap.get("certified"):' in ROUTER
     # the docstring may cite fix5.0's calibrated numbers; the CODE must not compare
     # against any new float threshold (starved is a boolean read off the stored verdict).
     and not re.search(r"[<>]=?\s*0\.\d",
                       re.sub(r'""".*?"""', "", ROUTER, flags=re.S))),
    # --- S.2: the old path is byte-identical for non-starved deaths ---
    ("S.2 every retirement site routes through _retire_or_park and _archive_retirement is "
     "called from the router ALONE (reach>=35%/no-frontier deaths take the old path verbatim)",
     SNB.count("self._archive_retirement(") == 1
     and "self._archive_retirement(" in ROUTER
     and SNB.count("self._retire_or_park(") == 6),
    # --- S.3: a park is not a strike ---
    ("S.3 the park branch never touches the retired registry — no count, no cooldown, no "
     "blacklist, no failed-tactic archive; it records focus_parked_frontier_starved instead",
     '"focus_parked_frontier_starved"' in ROUTER
     and '"retired"' not in ROUTER
     and "failed_notes" not in ROUTER),
    # --- S.4: slot economics are structural freebies (park = leave foci) ---
    ("S.4 relay slot and DISCOUNT share free on park BY CONSTRUCTION: both are computed over "
     "active foci only, and the router moves the wall from foci to watch",
     'self._nb.setdefault("watch", {})[sl] = w' in ROUTER
     and 'for foc in self._nb.get("foci", []):' in ZEROWIN
     and SNB.count("for f in self._nb.get(\"foci\", []) if self._relay_active(f)") >= 2),
    # --- S.5: full state preserved, bookkeeping stripped on resume ---
    ("S.5 park copies the WHOLE focus dict (relay ladder, notes, attribution ride along); "
     "resume strips only the park bookkeeping keys",
     "w = dict(foc)" in ROUTER
     and 'w["park_event"] = str(event)' in ROUTER
     and 'f.pop("park_event", None)' in WATCHPASS
     and 'f.pop("park_frontier_sr", None)' in WATCHPASS),
    # --- S.6: the wake-up condition ---
    ("S.6 a starved-parked watcher is HELD until its park frontier moves +focus_improve_pp "
     "(same evidence bar as the blacklist escape hatch), riding the fix5.0 certified hold; "
     "a dissolved cap (reach>=35%) falls through to the ordinary stall-resume",
     'if stalled and _cap50 and w.get("park_event"):' in WATCHPASS
     and "float(_cur) >= float(_base) + self.th.focus_improve_pp" in WATCHPASS
     and WATCHPASS.index('_cap50.get("certified")')
     < WATCHPASS.index('w.get("park_event")')),
    # --- S.7: the chase terminates ---
    ("S.7 root chase: visited-set cycle guard + caps-table depth bound; a root has no frontier "
     "by definition so P0 can never park the root (no route loop)",
     "seen = {str(wall).lower()}" in CHASE
     and "f not in seen and len(seen) <= len(caps) + 1" in CHASE),
    # --- S.8: nomination walks the existing gates ---
    ("S.8 access_auto admission = the SAME gate primitives in reconcile order (scope / conquered "
     "/ maintenance / cooldown / blacklist+escape / expand-with-P2b-exemption); a veto stands "
     "down with an audit note, never bypasses",
     "self._is_valid_focus(sl, latest_profile)" in VETO
     and "self._is_conquered_and_held(sl, latest_profile)" in VETO
     and "self._maintenance_block_reason(sl, latest_profile)" in VETO
     and "self.th.cooldown_sessions - (session_idx - last_ret)" in VETO
     and "self._blacklist_count(reg) >= self.th.blacklist_retirements" in VETO
     and "self._has_new_evidence(sl, reg, latest_profile)" in VETO
     and "access_frontier=True" in VETO
     and "access_auto_vetoed" in NOMINATE),
    # --- S.9: settlement order + never evict + never steal the relay slot ---
    ("S.9 access_auto settles BEFORE the LLM reconcile (first pick of a naturally-free seat), "
     "never rewrites the foci list (no eviction), and an enter_* root opens as an ACCESS-LINK "
     "relay ONLY when the relay slot is free — busy slot -> ordinary DEPTH siege (N3)",
     SNB.index("self._access_auto_nominate(session_idx, latest_profile)")
     < SNB.index("self._reconcile_foci(session_idx, latest_profile, proposed_foci)")
     and 'self._nb["foci"] = [' not in NOMINATE
     and "if n_relays < self.th.relay_max:" in NOMINATE
     and 'opened_by="access_auto"' in NOMINATE),
    # --- S.10: the LLM is informed, not obligated; level-dose machinery untouched ---
    ("S.10 render/prompt additions only — PARKED walls render their wake-up condition, the "
     "modeler prompt teaches access_auto/park semantics (adopt, don't fight), both seat events "
     "print as [siege][PARK]/[siege][ACCESS-AUTO]; the fix ships no level-allocation change "
     "(the relay reading-dose guarantee is the untouched fix44 machinery, pinned by its tests)",
     "PARKED frontier-starved since" in SNB
     and "ACCESS-ROOT AUTO-NOMINATION (v7fix5.2)" in MOD
     and "[siege][PARK]" in GM and "[siege][ACCESS-AUTO]" in GM),
    # --- S.11: the tests exist and pin the behaviours ---
    ("S.11 offline tests pin: starved park on every retirement family / certified & no-cap "
     "deaths retire unchanged / hold-until-frontier-moves / relay state round-trip / chase "
     "(transitive root, cycle) / gate vetoes / fresh-run inertness",
     "def test_budget_death_of_starved_wall_parks" in TESTS
     and "def test_relay_stall_death_parks_with_ladder_state" in TESTS
     and "def test_no_cap_death_retires_exactly_as_before" in TESTS
     and "def test_parked_wall_held_until_frontier_moves" in TESTS
     and "def test_chase_reaches_transitive_root" in TESTS
     and "def test_chase_cycle_guard" in TESTS
     and "def test_access_auto_respects_blacklist_veto" in TESTS
     and "def test_fresh_run_no_caps_is_inert" in TESTS),
]


def main() -> int:
    n_pass = 0
    for i, (label, ok) in enumerate(CHECKS, 1):
        status = "PASS" if ok else "FAIL"
        print(f"{status} {i:>2} {label}")
        n_pass += bool(ok)
    print(f"\n{n_pass}/{len(CHECKS)} fix5.2 design points hold")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
