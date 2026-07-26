# v7fix3 design-point checklist: every wiring the design doc promises, verified against source.
# Pure static checks (AST + string), no jax needed — runs identically local & Oscar.
import ast, io, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# Validate the tree this script ships in (like v7fix48/v7fix50 designchecks) — locally that is
# dicode_v7/, on Oscar the live training dir (~/dicode_v7fix47). The old hardcoded ~/dicode_v7fix3
# pointed at a stale fix3-era copy, so a modern signature (e.g. the fix5.0 multi-line
# _may_open_new_focus) read as a P3.1 failure though the live tree was correct.
R = os.path.dirname(os.path.abspath(__file__))

def src(p):
    return open(os.path.join(R, p), encoding="utf-8").read()

NB = src("auction/siege_notebook.py")
GM = src("src/dicode/dreaming/gen_manager.py")
MD = src("auction/modeler.py")
LV = src("auction/level_validator.py")
LM = src("auction/level_meta.py")
EV = src("src/dicode/evolution_efficient.py")
CA = src("auction/craftax_achievements.py")
ECO = src("src/dicode/dreaming/prompts/dicode/persona_ecology_coop.py")
CFG = src("conf/gen_manager/auction_c_v6siege.yaml")
DFL = src("conf/gen_manager/default.yaml")

checks = [
    # ---- P0 ----
    ("P0.1 modeler taught TIER LOCK rule (7)", "TIER LOCK" in MD and "tier_locked" in MD),
    ("P0.2 modeler taught upgrade rule (g)", "UPGRADED IN PLACE" in MD and "relay_attached" in MD),
    ("P0.3 journal zero-win>=3 upgrade hint", "★ZERO-WIN x" in NB and "UPGRADE the running focus" in NB),
    # ---- P1 ----
    ("P1.1 tier_of() exists + tier4 gate on LLM path", "def tier_of" in CA and "tier_locked(" in NB),
    ("P1.2 tier gate teaches relay_r0_floor", "re-propose THIS wall WITH relay_r0_floor" in NB),
    ("P1.3 auto-open _viable excludes tier4", "tier_of(sl) >= 4" in NB.split("def _viable")[1][:900]),
    ("P1.4 knob tier4_relay_only (th+cfg)", "tier4_relay_only" in NB and "siege_tier4_relay_only" in CFG),
    ("P1.5 tier_locked mirrored into the journal (LLM-visible)",
     "★TIER-LOCKED LAST SESSION" in NB and '"tier_locked_last": []' in NB),
    # ---- P2 ----
    ("P2.1 kept() branch routes relay_r0_floor to attach", "_attach_relay(" in NB.split("if sl in active:")[1][:700]),
    ("P2.2 attach guards: zero-win/capacity/range/already", all(s in NB for s in
        ("has held-out wins", "relay campaign(s)", "outside 1..", "relay already running @ floor"))),
    ("P2.3 attach resets old-form state machines", "foc[\"gap_forced\"] = False" in NB and "foc[\"ladder_level\"] = 0" in NB),
    ("P2.4 attach announces via RELAY-OPEN channel", "UPGRADED to SPAWN-ANNEAL RELAY" in NB),
    ("P2.5 knob relay_attach", "self.th.relay_attach" in NB and "siege_relay_attach" in CFG),
    # ---- P3 ----
    # v7fix5.0: the signature grew an access_frontier param (same waiver family) — the P3 point
    # pinned here is that the relay= kwarg exists and gates the waiver, not the exact def line.
    ("P3.1 _may_open_new_focus(relay=) waiver",
     "def _may_open_new_focus(" in NB
     and "relay: bool = False," in NB
     and "if relay and self.th.relay_expand_exempt:" in NB),
    ("P3.2 call site passes relay_requested", "relay=relay_requested" in NB),
    ("P3.3 fall-through re-checks ordinary gate", "relay=False" in NB.split("treating this as a normal focus proposal")[1][:800]),
    ("P3.4 knob relay_expand_exempt", "relay_expand_exempt" in NB and "siege_relay_expand_exempt" in CFG),
    # ---- P4 ----
    ("P4.1 ecology persona module + registered", "ECOLOGY curriculum designer" in ECO and "persona_ecology_coop" in DFL),
    ("P4.2 personas config = [ambitious, ecology]", "- ecology_coop" in CFG),
    ("P4.3 SIEGE_DIRECTIVE gated off for ecology", '"" if _is_eco else siege_directive_text' in GM),
    ("P4.4 ECOLOGY_DIRECTIVE only to ecology", "_eco_directive if _is_eco else \"\"" in GM),
    ("P4.5 ecology brief: 3 sections", all(s in GM for s in ("STARVED FAMILIES", "DECLINING SKILLS", "BREADTH SPAWN FRONTIER"))),
    ("P4.6 tag strip before note/select/archive", "siege tag stripped" in GM and GM.index("siege tag stripped") < GM.index("note_siege_level_type(_w")),
    ("P4.7 non-focus drill_target survives strip", "_dt in _focus_set" in GM),
    ("P4.8 _proposer_idx recorded for role quota", 'parsed["_proposer_idx"] = proposer_idx' in GM),
    ("P4.9 role quota + backfill in _coop_select", "coop_role_quota" in GM and "backfilled" in GM),
    ("P4.10 role quota config [10,8]", "coop_role_quota: [10, 8]" in CFG),
    ("P4.11 modeler told about the split", "ECOLOGY designer" in MD),
    ("P4.12 ecology_kept telemetry", "siege/ecology_kept" in GM),
    # ---- P5 ----
    ("P5.1 full-price cap in activation", "focus_force_cap" in EV and "full-price" in EV),
    ("P5.2 zero-win lane still 2", "zero_win_force_cap" in EV),
    ("P5.3 knob focus_force_cap=8", "FOCUS_FORCE_CAP = 8" in NB and "siege_focus_force_cap: 8" in CFG),
    # ---- P6 ----
    ("P6.1 notebook frontier state + persist schema", '"breadth_frontier": BREADTH_FRONTIER_START' in NB and "def breadth_frontier" in NB),
    ("P6.2 frontier advance API", "def note_breadth_frontier_reading" in NB and "floor {cur + 1} unlocked" in NB),
    ("P6.3 R6 always-on (validator)", "if not description:" in LV and "R6_SPAWN runs even with NO active foci" in LV),
    ("P6.4 BREADTH lane rules in R6", "is_breadth_lane" in LV and "two lanes" in LV),
    ("P6.5 reroll loop no longer exits on empty foci", "NO early-exit on empty foci" in GM),
    ("P6.6 frontier passed to validator", "breadth_frontier=breadth_frontier" in GM),
    ("P6.7 quota drop in keep loop", "[breadth][QUOTA]" in GM),
    ("P6.8 relay levels exempt from quota", "_is_relay_level" in GM),
    ("P6.9 frontier sweep from BREADTH nodes", "_note_breadth_frontier_readings" in GM and "[breadth][FRONTIER]" in GM),
    ("P6.10 sweep wired into siege session", "self._note_breadth_frontier_readings(session_idx=session_idx)" in GM),
    ("P6.11 spec teaches the two lanes", "BREADTH SPAWN FRONTIER" in LM),
    ("P6.12 check_compilation msg updated", "breadth" in GM.split("spawn-floor mismatch")[1][:700]),
    ("P6.13 knobs frontier_sr/quota", "siege_breadth_frontier_sr" in CFG and "siege_breadth_spawn_quota" in CFG),
    ("P6.14 frontier telemetry", "siege/breadth_frontier" in GM),
]

# AST cross-checks (the fix2 lesson: verify class ownership statically, not by memory)
gm_tree = ast.parse(GM)
gm_classes = {n.name: {m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
              for n in ast.walk(gm_tree) if isinstance(n, ast.ClassDef)}
checks += [
    ("AST: _render_ecology_directive ∈ TaskGenerator", "_render_ecology_directive" in gm_classes.get("TaskGenerator", set())),
    ("AST: _ecology_proposer_idxs ∈ TaskGenerator", "_ecology_proposer_idxs" in gm_classes.get("TaskGenerator", set())),
    ("AST: _note_breadth_frontier_readings ∈ TaskGenerator", "_note_breadth_frontier_readings" in gm_classes.get("TaskGenerator", set())),
    ("AST: check_compilation ∈ EnvGenerator", "check_compilation" in gm_classes.get("EnvGenerator", set())),
]
nb_tree = ast.parse(NB)
nb_classes = {n.name: {m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
              for n in ast.walk(nb_tree) if isinstance(n, ast.ClassDef)}
checks += [
    ("AST: _attach_relay ∈ SiegeNotebook", "_attach_relay" in nb_classes.get("SiegeNotebook", set())),
    ("AST: breadth_frontier ∈ SiegeNotebook", "breadth_frontier" in nb_classes.get("SiegeNotebook", set())),
]

# ---- v7fix3.1 audit fixes (2026-07-10 post-launch audit; job 3838221 killed & relaunched) ----
AMB = src("src/dicode/dreaming/prompts/dicode/persona_ambitious_coop.py")
checks += [
    ("A1 relay-capacity fall-through re-checks tier-4 (P1 bypass closed)",
     "open as a natural-spawn siege" in NB),
    ("A2 door gate skips tier-4 doors (P1 bypass closed)", "tier_of(door) >= 4" in NB),
    ("A3 cooldown waived for relay re-proposal after ORDINARY retirement",
     "cooldown_waived(" in NB and 'reg.get("last_event") != "focus_retired_relay_stalled"' in NB),
    ("A3b retirement registry records last_event", 'reg["last_event"] = str(event)' in NB),
    ("A3c RETIRED journal teaches the relay exemption", "exempt from the cooldown" in NB),
    ("A4 siege directive TEAM NOTE (split-team teaching)",
     "TEAM NOTE" in GM and "entirely YOUR" in GM),
    ("A4b ambitious persona knows the ecology split", "ECOLOGY designer" in AMB),
    ("A5 SEAT BUDGET no longer diverts the siege arm to coverage",
     "spend your remaining proposals on coverage" not in GM),
    ("A6 ecology brief reachability note", "EVERY member reads 0%" in GM),
    ("A7 dead code removed (stall ratchet + upgrade constant)",
     "ZERO_WIN_UPGRADE_MAX_SR = " not in NB and 'foc["stall_sessions"]' not in NB
     and "focus_min_stall_sessions=" not in NB and "siege_focus_min_stall_sessions:" not in CFG),
]

fails = 0
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
    fails += (not ok)
print(f"\n{len(checks) - fails}/{len(checks)} design points verified" + ("" if not fails else "  <<< FAILURES"))
sys.exit(1 if fails else 0)
