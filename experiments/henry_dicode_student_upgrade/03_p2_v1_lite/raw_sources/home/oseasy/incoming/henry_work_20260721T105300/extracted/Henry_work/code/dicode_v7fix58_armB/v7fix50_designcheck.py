"""v7fix5.0 designcheck — access-wall layer (frontier forensics + attribution override +
ACCESS_CAPPED park + expand-gate frontier exemption).

Root cause pinned (s213, 2026-07-14): held-out gnome capped by enter_gnomish_mines access
(reach 18.6%, cond past it 81% ~= trained 78%) while the attribution blamed the diamond-gear
chain; every siege training level bypassed the floor1->2 grind. Wiring-level assertions
(grep-style, no jax needed). Run: python v7fix50_designcheck.py -> prints N/N PASS.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


COL = _read("auction", "chain_order_log.py")
MOD = _read("auction", "modeler.py")
SNB = _read("auction", "siege_notebook.py")
GM = _read("src", "dicode", "dreaming", "gen_manager.py")
BFP = _read("auction", "behavior_fingerprint_log.py")
TESTS = _read("auction", "tests", "test_siege_fix50_access.py")

CHECKS = [
    # --- A.1-A.4: frontier forensics (chain_order_log) ---
    ("A.1 access_frontier: shallowest-frontier rule (first link under ACCESS_CAP_REACH, prereq order)",
     "def access_frontier" in COL
     and "for idx, link in enumerate(links):" in COL
     and "if reach < ACCESS_CAP_REACH:" in COL),
    ("A.2 calibrated constants shipped (0.35 reach / 0.60 cond / 50+50 sample guards)",
     "ACCESS_CAP_REACH = 0.35" in COL and "ACCESS_COND_TRANSFERRED = 0.60" in COL
     and "ACCESS_MIN_EPISODES = 50" in COL and "ACCESS_MIN_REACHED = 50" in COL),
    ("A.3 cond = n_succ/reached_n with BOTH sample guards on the certificate",
     "cond = (n_succ / reached_n) if reached_n > 0 else 0.0" in COL
     and "reached_n >= ACCESS_MIN_REACHED and cond >= ACCESS_COND_TRANSFERRED" in COL
     and "if total < ACCESS_MIN_EPISODES:" in COL),
    ("A.4 access pack rides forensics() and the chain hint renders BINDING-ACCESS",
     '"access": self.access_frontier(target)' in COL
     and "BINDING-ACCESS=" in COL and "execution has TRANSFERRED" in COL),
    # --- A.5-A.6: attribution override (modeler) ---
    ("A.5 deterministic override to (chain_unreached, frontier, verified) with llm_said audit,"
     " placed BEFORE the probs-rejection return",
     '"overridden": "ACCESS_CAPPED"' in MOD
     and '"llm_said_class": cls, "llm_said_key": key' in MOD
     and MOD.index('"overridden": "ACCESS_CAPPED"') < MOD.index('"rejected": cls}')),
    ("A.6 no override when the LLM already names the frontier (normal verified path stands)",
     'if not (cls == "chain_unreached" and key == _fr):' in MOD),
    ("A.7 prompt: BINDING-ACCESS instruction + enter_* legal target + NATURAL-spawn earn-the-descent"
     " guidance (floor-1 spawns are R6-illegal for non-relay levels — natural spawn is correct AND"
     " legal: the floor above the entrance is reached ~95% anyway)",
     "BINDING-ACCESS (v7fix5.0)" in MOD and "NATURAL spawn" in MOD
     and "NEVER pre-credit that floor's" in MOD and "native floor - 1" not in MOD),
    # --- A.8-A.11: notebook machinery ---
    ("A.8 access_caps in the notebook schema (survives _coerce/reload)",
     '"access_caps": {"session": None, "caps": {}}' in SNB
     and "def note_access_caps" in SNB and "def _access_cap" in SNB
     and "def access_frontiers" in SNB),
    ("A.9 gap gate parks ONLY certified caps, to WATCH, counters frozen, before the over-gap calc",
     'if _cap and _cap.get("certified"):' in SNB
     and "focus_parked_access_capped" in SNB
     and SNB.index('if _cap and _cap.get("certified"):') < SNB.index(
         "over = (\n                trained_pct is not None")
     and re.search(r'_cap and _cap\.get\("certified"\).{0,2000}?w\["gap_stall"\] = 0', SNB, re.S)),
    ("A.10 watch-resume HOLD while the cap lasts (no park->resume oscillation)",
     "_cap50 = self._access_cap(sl)" in SNB
     and 'if stalled and _cap50 and _cap50.get("certified"):' in SNB),
    ("A.11 expand gate: access_frontier waiver mirrors the relay waiver; capacity still binds;"
     " reconcile keys it off access_frontiers(); the relay fall-through keeps it",
     "access_frontier: bool = False" in SNB
     and "if access_frontier:\n            return True" in SNB
     and "_is_frontier = sl in self.access_frontiers()" in SNB
     and "expand_exempt_access_frontier" in SNB
     and SNB.count("access_frontier=_is_frontier") >= 2),
    # --- A.12-A.13: gen_manager feed + audit visibility ---
    ("A.12 gen_manager feeds note_access_caps each session from the forensics pack ([siege][access])",
     "note_access_caps" in GM and "[siege][access] wall=" in GM
     and GM.index("note_access_caps") > GM.index("chain_targets().keys()")),
    ("A.13 the [siege][attrib] line shows OVERRIDDEN + what the LLM originally said",
     "OVERRIDDEN={_a['overridden']}" in GM and "llm_said" in GM),
    # --- A.14: fingerprint press-count labelling ---
    ("A.14 fingerprint counts labelled ACTION PRESSES (a craft press is NOT a successful craft)",
     "ACTION PRESSES incl. failed attempts" in BFP and "NOT a successful craft" in BFP),
    # --- A.16: deterministic proposer-side contract (not routed through the rephrasable style_note) ---
    ("A.16 SIEGE_DIRECTIVE injects the ACCESS-LINK CONTRACT for enter_* foci (natural spawn,"
     " earn the descent, no set_monsters_killed on the approach floor)",
     "ACCESS-LINK CONTRACT for" in GM
     and '_fsk.startswith("enter_") and not _sysbuilt_relay' in GM
     and "Do NOT call " in GM),
    # --- B.1-B.4 (v7fix5.1): GLM empty-response guard (root-caused 2026-07-15: think:true at v7
    #     prompt sizes -> reasoning heavy tail burns the whole max_tokens budget -> finish_reason
    #     "length" with content "" on 20-50% of calls; probe: think OFF 5/5, think ON 1/10 dead) ---
    ("B.1 llm.py carries finish_reason in BOTH return paths (truncation visible in logs forever)",
     _read("src", "dicode", "dreaming", "llm.py").count('"finish_reason":') >= 2),
    ("B.2 diagnose_siege retries an EMPTY/unparseable parse on the shared attempt budget"
     " (not only attribution violations) and logs the finish_reason",
     "EMPTY/unparseable response (finish_reason=" in MOD
     and MOD.index("EMPTY/unparseable response") < MOD.index('viols = su.get("attrib_violations")')),
    ("B.3 modeler think:false in the live config (the root-cause switch, user-approved 2026-07-15;"
     " split on the LAST 'modeler:' so proposers' think flags can't satisfy it)",
     "think: false" in _read("conf", "gen_manager", "auction_c_v6siege.yaml").rsplit("\nmodeler:", 1)[1]
     and "think: true" not in _read("conf", "gen_manager", "auction_c_v6siege.yaml").rsplit("\nmodeler:", 1)[1]),
    ("B.4 empty-retry regression tests shipped (retry-then-good / budget-exhaust / one-call healthy path)",
     all(t in _read("auction", "tests", "test_modeler_empty_retry.py")
         for t in ("test_g1_empty_response_retried_then_good_wins",
                   "test_g2_all_empty_degrades_after_budget",
                   "test_g3_good_first_response_makes_one_call",
                   "test_g4_retry_log_carries_finish_reason"))),
    # --- A.15: regression tests shipped ---
    ("A.15 fix50 test file covers frontier math / override audit / park / hold / exemption / reload",
     "test_gnome_frontier_and_cond_certified" in TESTS
     and "test_kobold_frontier_is_mines_not_sewers_and_uncertified" in TESTS
     and "test_attribution_downstream_claim_overridden_with_audit" in TESTS
     and "test_certified_cap_parks_wall_to_watch" in TESTS
     and "test_uncertified_cap_leaves_style_gate_armed" in TESTS
     and "test_capped_watcher_held_then_released" in TESTS
     and "test_expand_gate_waived_for_named_frontier" in TESTS
     and "test_access_caps_survive_reload" in TESTS),
]

failed = [name for name, ok in CHECKS if not ok]
for name, ok in CHECKS:
    print(("PASS " if ok else "FAIL ") + name)
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed.")
if failed:
    raise SystemExit(1)
