"""v7fix5.5 designcheck — batch 1: P0 scaffold-facts disclosure + L1 render diet.

Design doc: fable_research_reports/v7fix55假设探针课程环设计.md (P0 + P3/L1, hard rules §A).
Motivation: the fix54 radius probe proved the entry cliff is a spawn-context effect with
lighting the prime mechanical suspect, yet the world rule "radius stages torch-light the
spawn and the down ladder; entry stages pre-light nothing" was never disclosed to the
modeler — structurally undiagnosable. Meanwhile the 14k-char render diluted salience:
37% was per-wall repeated teaching prose in RETIRED WALLS and full-text style notes for
chains unrelated to the current attack. Hard rules pinned here: every cut ships a double
assertion (new format present + old redundancy gone); compressed content SURVIVES in the
notebook JSON (render-only, reversible); disclosure sentences are COMPUTED from the stage
table, never template constants. Run: python v7fix55_designcheck.py -> prints N/N PASS.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


SNB = _read("auction", "siege_notebook.py")
GM = _read("src", "dicode", "dreaming", "gen_manager.py")
T55 = _read("auction", "tests", "test_siege_fix55_render.py")

from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RUNG_LADDER_RADII,
    SiegeNotebook,
)


def _mk_nb(tmp):
    nb = SiegeNotebook(os.path.join(tmp, f"nb_{len(os.listdir(tmp))}.json"))
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof["defeat_kobold"] = 0.0
    nb.apply_llm_update(
        1, prof,
        {"foci": [{"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": 3}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    return nb


def _at_stage(nb, stage, floor=2):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    r["sub_stage"] = stage
    nb._save()
    return nb.render_for_prompt()


tmp = tempfile.mkdtemp()
nb = _mk_nb(tmp)
txt6 = _at_stage(nb, 6)          # radius leg, no lock
txt5 = _at_stage(nb, 5)          # radius + lock + slow clock
txt8 = _at_stage(nb, 8)          # easiest radius rung
txt0 = _at_stage(nb, 0)          # FULL

nb2 = _mk_nb(tmp)
nb2._nb["retired"] = {"defeat_gnome_warrior": {
    "count": 2, "last_session": 9, "sr_at_retirement": 12.0,
    "failed_notes": ["Old tactic one. Body A.", "Old tactic two. Body B.", "Latest tactic in full."],
    "last_event": "focus_retired_stalled",
    "failure_attribution_at_retirement": {"class": "resource_shortfall", "verified": True},
}}
nb2._save()
txt_ret = nb2.render_for_prompt()
nb2._nb["retired"]["defeat_gnome_warrior"]["last_event"] = "focus_retired_relay_stalled"
nb2._save()
txt_ret_relay = nb2.render_for_prompt()

nb3 = _mk_nb(tmp)
foc3 = nb3.foci()[0]
foc3["prereq_tree"] = [{"skill": "make_iron_sword", "state": "wall", "sr": 30.0}]
foc3["style_note"] = "live kobold tactic"
nb3._nb["verified_chains"] = [
    {"target": "make_iron_sword", "links": [], "category": "enabler", "status": "verified",
     "last_recorded_sr": 80, "last_recorded_session": 5, "style_note": "hot enabler prose"},
    {"target": "defeat_zombie", "links": [], "category": "combat_milestone",
     "status": "verified", "last_recorded_sr": 90, "last_recorded_session": 4,
     "style_note": "cold milestone prose"},
    {"target": "defeat_kobold", "links": [], "category": "combat_milestone",
     "status": "progress", "last_recorded_sr": 5, "last_recorded_session": 6,
     "style_note": "stale recorded kobold prose"},
]
nb3._save()
txt_ch = nb3.render_for_prompt()
nb3_reloaded = SiegeNotebook(nb3.path)

CHECKS = [
    ("B1.1 ONE stage table, two callers: relay_scaffold delegates to _stage_knobs and the "
     "render computes the NEXT stage through the same helper (no drift-prone second copy; "
     "batch-3 routes the insert's NEXT to its return stage through the SAME helper)",
     "return self._stage_knobs(r, stage)" in SNB
     and "self._stage_knobs(r, _next_stage55)" in SNB
     and 'int(_ins_r55.get("return_stage", 0) or 0)' in SNB),
    ("B1.2 SCAFFOLD FACTS are rendered at scaffolded stages and COMPUTED from the table "
     "(the radius value changes the sentence; a template constant cannot)",
     "SCAFFOLD FACTS" in txt6
     and f"within {RUNG_LADDER_RADII[2]} tiles of the down ladder" in txt6
     and f"within {RUNG_LADDER_RADII[0]} tiles of the down ladder" in txt8
     and "SCAFFOLD FACTS" not in txt0),
    ("B1.3 the LIGHTING world rule is disclosed and lives in exactly ONE source "
     "(_scaffold_fact_clauses) — the fact string appears once in the module, computed at "
     "render time, torch-lit at radius stages / NONE on the next-entry diff",
     SNB.count("torch-lit (9x9)") == 1
     and "torch-lit (9x9)" in txt6
     and "pre-light -> NONE" in txt5),
    ("B1.4 the NEXT-stage diff lists CHANGED knobs only (6->5: lock + clock arrive, spawn "
     "and light unchanged and unlisted)",
     "NEXT after graduating = stage 5" in txt6
     and "up-ladder -> REMOVED" in txt6
     and "spawn -> " not in txt6),
    ("B1.5 facts-not-levers wording + the fix53 DESCENT REGIME line untouched (its "
     "designcheck literal survives this batch)",
     "never yours to change" in txt6
     and "DESCENT REGIME at this stage (code-set, not yours to " in SNB),
    ("B1.6 R1 double assertion — new format IN: latest note full + older notes one-lined "
     "with the diagnosis tag on the wall line",
     "failed tactic (latest): Latest tactic in full." in txt_ret
     and "earlier failed tactic 1: Old tactic one…" in txt_ret
     and "Body A." not in txt_ret
     and "diagnosis: resource_shortfall (verified)" in txt_ret),
    ("B1.7 R1 double assertion — old redundancy OUT: the per-wall 3-line EXCEPTION prose is "
     "gone from the module; the exemption is taught ONCE in the header and ONLY when a "
     "rendered wall earned it (v7fix3.1 rule preserved)",
     "(EXCEPTION: if this wall is depth-blocked" not in SNB
     and txt_ret.count("exempt from the cooldown") == 1
     and "exempt from the cooldown" not in txt_ret_relay),
    ("B1.8 R2/R3: hot chains keep prose, cold chains are one-line records, an active-focus "
     "target defers to its style-so-far line (ONE tactic text, the newest)",
     "hot enabler prose" in txt_ch
     and "defeat_zombie" in txt_ch and "cold milestone prose" not in txt_ch
     and "live kobold tactic" in txt_ch and "stale recorded kobold prose" not in txt_ch),
    ("B1.9 compression is render-only and REVERSIBLE: every suppressed note survives in the "
     "notebook JSON across a reload",
     nb3_reloaded.verified_chains()[1]["style_note"] == "cold milestone prose"
     and SiegeNotebook(nb2.path).retired_registry()["defeat_gnome_warrior"]["failed_notes"][0]
     == "Old tactic one. Body A."),
    ("B1.10 gen_manager RELAY-BUILD stage string carries the lighting fact both ways, "
     "derived from the scaffold dict",
     "spawn & down ladder torch-lit (9x9 each)" in GM
     and "no scaffold pre-light (the floor's own light only)" in GM
     and 'scaffold.get("down_ladder_radius") is not None' in GM),
    ("B1.11 the fix5.5 batch-1 test suite pins disclosure/diff/diet/reversibility",
     all(t in T55 for t in (
         "test_scaffold_facts_rendered_at_scaffold_stage",
         "test_scaffold_facts_computed_from_stage_table",
         "test_scaffold_facts_absent_at_full_and_kit_strip",
         "test_next_stage_diff_lock_and_clock",
         "test_next_stage_diff_spawn_and_lighting",
         "test_next_stage_full_boundary",
         "test_retired_latest_note_full_older_one_lined",
         "test_retired_exemption_taught_once_and_only_when_earned",
         "test_chain_prose_hot_kept_cold_one_lined",
         "test_chain_of_active_focus_defers_to_style_so_far",
         "test_relay_build_stage_string_carries_lighting",
     ))),
]

# ==================================================================================================
# batch 2 — PROBE-AS-TOOL (design §batch-2 施工设计 + §B hard rules)
# ==================================================================================================
from auction.siege_notebook import (  # noqa: E402
    PROBE_AXES,
    PROBE_KINDS,
    PROBE_SENSORS,
)

WB = _read("src", "minicraftax", "world_builder.py")
RP = _read("src", "dicode", "evaluation", "rung_probe.py")
RD = _read("experiments", "training", "run_dicode.py")
MOD = _read("auction", "modeler.py")

pnb = _mk_nb(tmp)
_pr = pnb.foci()[0]["relay"]
_pr["spawn_floor"] = 2
_pr["sub_stage"] = 6
_pr["rung_trained"] = [12.0, 13.0, 12.5]
_pr["readings_since_transition"] = 6
_pr["gain_log"] = [0.0, 0.0]
pnb._save()
_avail_txt = pnb.render_for_prompt()          # BEFORE any request: the availability offer
pnb._admit_probe_request({"wall": "defeat_zombie", "kind": "diagnose",
                          "justification": "x"}, 5)
_rej_nonrelay = "not_an_active_relay_wall" in (pnb._nb.get("probe_receipt") or "")
pnb._admit_probe_request({"wall": "defeat_kobold", "kind": "diagnose",
                          "justification": "flat 12 -> 13 -> 12.5",
                          "filter": {"field": "mana", "op": "==", "value": 1}}, 5)
_pend = pnb._nb.get("probe_pending") or {}
_fallback_ok = _pend and _pend.get("filter") is None and "unknown_sensor" in str(
    _pend.get("filter_error"))
_pend_txt = pnb.render_for_prompt()           # AFTER the accept: pending + receipt mirror
pnb.deliver_probe_report({"wall": "defeat_kobold", "kind": "diagnose", "ckpt_step": 9000,
                          "n_envs": 256, "success_pct": 20.0, "died_pct": 70.0,
                          "timeout_pct": 10.0, "marginals": {"sleeping": {"rate_pct": 40.0}},
                          "snapshots": []}, 5)
_rep_txt = pnb.render_for_prompt()
pnb._admit_probe_request({"wall": "defeat_kobold", "kind": "diagnose",
                          "justification": "flat 12 -> 13 -> 12.5"}, 7)
_budget_rej = "budget_exhausted" in (pnb._nb.get("probe_receipt") or "")
_vk, _vd, _ve = pnb.probe_variant_knobs("defeat_kobold", "pre_light", "harder")

CHECKS += [
    ("B2.1 the LLM's whole freedom = 4 gated fields: kind/axis enums pinned, a non-relay wall "
     "is refused, the receipt mirrors into the journal",
     PROBE_KINDS == ("diagnose", "whatif") and len(PROBE_AXES) == 6
     and _rej_nonrelay and "PROBE RECEIPT" in _pend_txt and "PROBE PENDING" in _pend_txt),
    ("B2.2 filter compiler: whitelist fields/ops/ranges from PROBE_SENSORS; a compile failure "
     "FALLS BACK to uniform random (recorded, no reprompt — anti negotiation loop)",
     _fallback_ok and "uniform random" in _pend_txt),
    ("B2.3 budget = code counter per wall per rolling window; over-budget is refused with a "
     "receipt", _budget_rej),
    ("B2.4 trigger = rung stall OR verified citation (Tier-1 rides the fix47 handshake: "
     "numbers checked against rung_trained)",
     "_verify_probe_justification" in SNB and "_relay_progressing(foc)" in SNB),
    ("B2.5 probe output is NEVER load-bearing: delivery writes probe_reports only (no foci / "
     "rung keys), and the state keys live in the _empty_notebook whitelist",
     '"probe_ledger": {}' in SNB and '"probe_pending": None' in SNB
     and '"probe_reports": {}' in SNB
     and "PROBE REPORT (diagnose, measured s5" in _rep_txt),
    ("B2.6 ONE sensor catalog: the executor derives snap fields from PROBE_SENSORS and "
     "asserts equality at runtime; the journal's availability line renders from the same "
     "table",
     "from auction.siege_notebook import PROBE_SENSORS" in RP
     and "set(rec) == set(PROBE_SENSORS)" in RP
     and "PROBE TOOL available" in _avail_txt and "ladder_dist" in _avail_txt),
    ("B2.7 variants are template-rendered, never string surgery: gen_manager exposes "
     "_relay_level_build(wall, floor, scaffold, stripped); the executor calls it for base AND "
     "variant; pre_light decouples light from anchor via a set_starting_floor kwarg whose "
     "None default keeps the coupled pre-5.5 behaviour (v7fix5.7: the axis is a 3-level "
     "graded ladder — one notch per step, so harder-from-lit lands on 'ladder', not dark)",
     "def _relay_level_build" in GM and "_relay_level_build" in RP
     and "pre_light: bool | str | None = None" in WB
     and 'elif _pl == "ladder":' in WB
     and "_stamp, _ladder_only = self._down_ladder_spawn_radius is not None, False" in WB
     and _ve is None and _vk.get("pre_light") == "ladder"),
    ("B2.8 main-thread execution hook rides Step 4c in run_dicode, guarded so a probe can "
     "never break training, and a crashing probe drops its request WITH the budget spent",
     "Step 4c" in RD and "run_pending_probe" in RD and "clear_probe_pending" in RD),
    ("B2.9 modeler passthrough is SHAPE-only (semantics live in the notebook's single write "
     "path) and the system prompt teaches the tool",
     '"probe_request": probe' in MOD and "THE PROBE TOOL" in MOD),
    ("B2.10 the batch-2 test suite pins gates/budget/variants/lifecycle/single-source",
     all(t in _read("auction", "tests", "test_siege_fix55_probe.py") for t in (
         "test_probe_state_keys_survive_coerce_and_reload",
         "test_rejects_non_relay_wall_and_bad_enums",
         "test_trigger_needs_stall_or_verified_citation",
         "test_budget_window_and_single_pending",
         "test_filter_compiles_or_falls_back_random",
         "test_variant_steps_are_code_chosen",
         "test_pre_light_decouples_from_anchor",
         "test_lifecycle_render_and_stale",
         "test_whatif_report_renders_paired_delta",
         "test_probe_request_flows_through_apply_llm_update",
         "test_relay_level_build_renders_variant_from_knobs",
         "test_rung_probe_snap_catalog_single_source",
     ))),
]

# ==================================================================================================
# batch 3 — P2 hypothesis loop (design §batch-3 施工设计: free attribution + in-machine verify)
# ==================================================================================================
from auction.siege_notebook import RUNG_INSERT_STAGE  # noqa: E402


def _hyp3(axis="uplock", evidence="paired delta +12.3pp"):
    return {"hypothesis": "dark-death loop", "evidence": evidence,
            "intervention": {"axis": axis, "direction": "easier"},
            "prediction": "SR should rise >= 8pp"}


def _stall3(nb_, floor=2):
    r_ = nb_.foci()[0]["relay"]
    r_["spawn_floor"] = floor
    ms_, _, _, _ = nb_._ladder_shape(r_)
    r_["sub_stage"] = ms_ - 1
    r_["rung_trained"] = [12.0, 13.0, 12.5]
    r_["readings_since_transition"] = 6
    r_["gain_log"] = [0.0, 0.0]
    nb_._save()
    return r_


def _whatif3(nb_, r_, axis="uplock", delta=12.3):
    nb_._nb.setdefault("probe_reports", {})["defeat_kobold"] = {
        "wall": "defeat_kobold", "kind": "whatif", "delivered_session": 10, "n_envs": 256,
        "axis": axis, "direction": "easier", "delta_pp": delta,
        "spawn_floor": int(r_["spawn_floor"]), "sub_stage": int(r_["sub_stage"])}
    nb_._save()


# free verdict -> compile
h1 = _mk_nb(tmp)
r1 = _stall3(h1)
_k1 = int(r1["sub_stage"])
_ms1 = _k1 + 1
_whatif3(h1, r1, delta=12.3)
h1.admit_hypothesis("defeat_kobold", _hyp3(), 11)
_e1 = h1._nb["hypothesis_log"][-1]
_compiled_ok = (
    _e1["status"] == "verified_compiled" and r1["sub_stage"] == RUNG_INSERT_STAGE
    and r1["stage_insert"]["knobs"].get("uplock") is True
    and r1["rung_history"][-1]["event"] == "hypothesis_insert"
    and h1._nb.get("probe_pending") is None
    and h1.relay_scaffold("defeat_kobold")["sub_stage"] == RUNG_INSERT_STAGE
)
_ins_render = h1.render_for_prompt()
# insert graduates back to the return stage under the EXISTING machinery
# (v7fix5.7-P2': graduation judges the last-3 window mean — 3 above-bar readings needed)
_g1 = None
for _i57 in range(3):
    _g1 = h1.note_rung_reading("defeat_kobold", h1.th.rung_graduate_sr + 5.0,
                               session_idx=12 + _i57)
_grad_ok = ("RUNG_INSERT_GRADUATED" in (_g1 or "") and r1["sub_stage"] == _k1
            and "stage_insert" not in r1 and _e1["status"] == "insert_graduated")
# refuted free verdict
h2 = _mk_nb(tmp)
r2 = _stall3(h2)
_whatif3(h2, r2, delta=2.1)
h2.admit_hypothesis("defeat_kobold", _hyp3(evidence="paired delta +2.1pp"), 11)
_refut_ok = (h2._nb["hypothesis_log"][-1]["status"] == "refuted"
             and "stage_insert" not in r2
             and "★REFUTED" in h2.render_for_prompt())
# Tier-1 rejection + Tier-2 scheduling on a diagnose report
h3 = _mk_nb(tmp)
r3 = _stall3(h3)
h3._nb.setdefault("probe_reports", {})["defeat_kobold"] = {
    "wall": "defeat_kobold", "kind": "diagnose", "delivered_session": 10, "n_envs": 256,
    "success_pct": 12.5, "died_pct": 80.3, "timeout_pct": 7.2,
    "spawn_floor": 2, "sub_stage": int(r3["sub_stage"]),
    "marginals": {"light": {"med": 0.05}}}
h3._save()
h3.admit_hypothesis("defeat_kobold", _hyp3(evidence="most runs just die"), 11)
_t1_rej = h3._nb["hypothesis_log"][-1]["status"] == "rejected_tier1"
h3._nb["probe_reports"]["defeat_kobold"].pop("hypothesized", None)
h3.admit_hypothesis("defeat_kobold", _hyp3(evidence="died 80.3% at light med 0.05"), 11)
_p3 = h3._nb.get("probe_pending") or {}
_t2_ok = (
    h3._nb["hypothesis_log"][-1]["status"] == "verify_scheduled"
    and _p3.get("kind") == "whatif"
    and _p3.get("verify_hypothesis_id") == h3._nb["hypothesis_log"][-1]["id"]
    and [11, "verify"] in h3._nb["probe_ledger"]["defeat_kobold"]
    and h3._probe_budget_left("defeat_kobold", 11) == {"diagnose": 1, "whatif": 1}
)
# delivered verify verdict -> compile; stale context -> record only
h3.deliver_probe_report({"wall": "defeat_kobold", "kind": "whatif", "delta_pp": 15.0}, 12)
h3.hypothesis_housekeeping(13)
_hk_ok = (h3._nb["hypothesis_log"][-1]["status"] == "verified_compiled"
          and r3["sub_stage"] == RUNG_INSERT_STAGE)
h4 = _mk_nb(tmp)
r4 = _stall3(h4)
h4._nb.setdefault("probe_reports", {})["defeat_kobold"] = dict(
    h3._nb["probe_reports"]["defeat_kobold"], verdict_done=False)
h4._nb["probe_reports"]["defeat_kobold"]["verify_of"] = "h_missing"
h4._nb["hypothesis_log"] = [dict(_e1, id="h_missing", wall="defeat_kobold",
                                 status="verify_scheduled")]
h4._nb["probe_reports"]["defeat_kobold"]["verify_sub_stage"] = int(r4["sub_stage"]) - 1
h4._save()
h4.hypothesis_housekeeping(13)
_stale_ok = (h4._nb["hypothesis_log"][-1]["status"] == "stale_context"
             and "stage_insert" not in r4)
# single-insert invariant + fresh ratchet
h6 = _mk_nb(tmp)
r6 = _stall3(h6)
_key6 = "2:%d" % RUNG_INSERT_STAGE
r6.setdefault("best_by_rung", {})[_key6] = 77.0
h6._save()
_whatif3(h6, r6, delta=12.3)
h6.admit_hypothesis("defeat_kobold", _hyp3(), 11)
_fresh_ok = (r6["sub_stage"] == RUNG_INSERT_STAGE and r6.get("best_rung_trained") is None
             and _key6 not in (r6.get("best_by_rung") or {}))
_e6b = {"id": "h6b", "wall": "defeat_kobold", "session": 12, "axis": "needs_clock",
        "direction": "easier", "hypothesis": "x", "evidence": "y", "prediction": "z",
        "status": "recorded"}
h6._nb["hypothesis_log"].append(_e6b)
_no_stack = (h6._try_compile_hypothesis(h6.foci()[0], _e6b, 12, 15.0) is False
             and _e6b.get("note") == "insert_already_active"
             and h6._schedule_hypothesis_verify(h6.foci()[0], _e6b, 12) is False
             and h6._nb.get("probe_pending") is None)
# R0 pin: a credit-only insert at the target floor is refused
h5 = _mk_nb(tmp)
r5 = _stall3(h5, floor=3)
_whatif3(h5, r5, axis="monster_credit", delta=20.0)
h5.admit_hypothesis("defeat_kobold", _hyp3(axis="monster_credit",
                                           evidence="paired delta +20.0pp"), 11)
_r0_ok = (h5._nb["hypothesis_log"][-1]["status"] == "compile_refused"
          and "r0_pin" in h5._nb["hypothesis_log"][-1]["note"]
          and "stage_insert" not in r5)

CHECKS += [
    ("B3.1 double-call statically pinned: gen_manager builds a SECOND scientist LLM with "
     "think=True next to the think-off bookkeeping modeler; absent scientist = dormant loop "
     "(hypothesize_probe returns {} and nothing else runs)",
     "think=True," in GM and 'modeler_cfg.get("scientist", True)' in GM
     and "scientist_llm=scientist_llm" in GM
     and "if self.scientist_llm is None or not context_text:" in MOD),
    ("B3.2 the HYPOTHESIS block is gated: enums pinned, Tier-1 citation vs the report's OWN "
     "numbers (uncited narrative rejected, cited numbers admitted)",
     _t1_rej and h3._nb["hypothesis_log"][-1]["status"] != "rejected_tier1"),
    ("B3.3 free-verdict shortcut: a triggering whatif that measured the SAME axis+direction "
     "at the SAME stage settles the hypothesis with NO second probe",
     _compiled_ok and _refut_ok),
    ("B3.4 Tier-2 verify rides the batch-2 executor with its OWN ledger kind (modeler "
     "diagnose/whatif budget untouched) and the single pending slot",
     _t2_ok),
    ("B3.5 compile = INSERTED rung at a distinct stage id OUTSIDE every ladder range — the "
     "fix4.6 exact-stage reading filter isolates it with zero filter changes",
     RUNG_INSERT_STAGE > _ms1 + 10 and _compiled_ok
     and "INSERTED rung" in _ins_render and "★VERIFIED" in _ins_render),
    ("B3.6 the EXISTING state machine governs the insert: graduate -> return stage; stall -> "
     "insert removed + the normal regress path with the budget charged (self-healing, no LLM "
     "retraction)",
     _grad_ok and "RUNG_INSERT_STALLED" in SNB
     and '_regress_budget("stall at the hypothesis-inserted rung")' in SNB),
    ("B3.7 delivered verify verdicts compile through housekeeping (worker thread, BEFORE the "
     "big call) — delivery itself stays a pure store",
     _hk_ok and "hypothesis_housekeeping" in GM
     and GM.index("hypothesis_housekeeping") < GM.index("guidance = modeler.diagnose_siege")),
    ("B3.8 stale-context guard: a verdict landing after the ladder moved records but never "
     "compiles", _stale_ok),
    ("B3.9 the fix4.7 Q1 R0 pin is re-applied at compile: a credit-only insert at the target "
     "floor is refused", _r0_ok),
    ("B3.10 hypothesis_log lives in the _empty_notebook whitelist (fix4.2 lesson) and the "
     "scientist prompt pins the closed menu + honest-unknown rule",
     '"hypothesis_log": []' in SNB and "SCIENTIST_SYSTEM_PROMPT" in MOD
     and "menu is CLOSED" in MOD and "honest unknown" in MOD
     and "THE SCIENTIST PASS" in MOD),
    ("B3.11 the batch-3 test suite pins gates/verdicts/insert machinery/switches",
     all(t in _read("auction", "tests", "test_siege_fix55_hypothesis.py") for t in (
         "test_shape_gate_rejects_bad_axis_and_records",
         "test_tier1_rejects_uncited_evidence_and_passes_report_numbers",
         "test_one_scientist_shot_per_report",
         "test_verify_schedules_whatif_pending_with_own_budget_kind",
         "test_verify_budget_one_per_window_and_slot_respected",
         "test_free_verdict_from_triggering_whatif_compiles_insert",
         "test_free_verdict_below_bar_refutes_without_probe",
         "test_housekeeping_verdict_compiles_verified_verify_report",
         "test_housekeeping_stale_context_records_but_never_compiles",
         "test_r0_pin_refuses_credit_only_insert",
         "test_insert_graduates_back_to_return_stage",
         "test_insert_stall_removes_insert_and_regresses_normally",
         "test_single_insert_invariant_no_insert_on_insert",
         "test_insert_ratchet_is_fresh_per_insert",
         "test_pre_light_override_reaches_facts_and_scaffold",
         "test_master_switch_off_is_a_noop",
     ))),
    ("B3.12 single-insert invariant + per-insert fresh ratchet (2026-07-17 timeline audit): "
     "a verdict landing while an insert is ACTIVE refuses to compile (the overwrite would "
     "lose the return stage) and Tier-2 scheduling waits without burning budget; each "
     "compile drops the previous insert's floor:50 ratchet key (a different knob set is a "
     "different task — the fix53/54 surgery principle)",
     _fresh_ok and _no_stack),
]

n_ok = sum(bool(v) for _, v in CHECKS)
for label, v in CHECKS:
    print(("PASS " if v else "FAIL ") + " " + label)
print(f"{n_ok}/{len(CHECKS)} fix5.5 design points hold")
sys.exit(0 if n_ok == len(CHECKS) else 1)
