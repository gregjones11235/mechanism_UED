"""v7fix5.5 batch 3 — P2 hypothesis loop: free attribution + in-machine verification.

Design (fable_research_reports/v7fix55假设探针课程环设计.md, batch-3 施工设计):
a probe report triggers ONE scientist pass whose ROOT-CAUSE HYPOTHESIS block is folded through
code gates (shape enums -> Tier-1 evidence citation vs report/reading numbers); on a stalled
rung the intervention is verified by a paired whatif (Tier-2, own "verify" budget, single
pending slot) or — free-verdict shortcut — by the triggering whatif itself when it measured the
same axis+direction at the same stage. delta >= hypothesis_verify_delta_pp compiles an INSERTED
rung (sub_stage = RUNG_INSERT_STAGE, distinct int -> the fix4.6 reading filter isolates it with
zero changes) that the EXISTING state machine governs: graduate -> return stage; stall -> insert
removed + normal regress (budget charged) — a wrong hypothesis self-heals. The fix4.7 Q1 R0 pin
is re-applied at compile. Everything lands in the append-only hypothesis_log.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    PROBE_BUDGET_WINDOW,
    RUNG_INSERT_STAGE,
    RUNG_WIN,
    SiegeNotebook,
    SiegeThresholds,
)

WALL = "defeat_kobold"


def _mature_profile(extra=None):
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _open_relay(nb, wall=WALL, r0=3, session=1):
    prof = _mature_profile({wall: 0.0})
    nb.apply_llm_update(
        session, prof,
        {"foci": [{"skill": wall, "prereq_tree": [], "relay_r0_floor": r0}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    return prof


def _set_relay(nb, floor=2, stage=None, stalled=True, readings=(12.0, 13.0, 12.5)):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    if stage is None:
        stage, _, _, _ = nb._ladder_shape(r)
        stage -= 1  # a mid-ladder stage with room both ways
    r["sub_stage"] = int(stage)
    r["rung_trained"] = list(readings)
    r["readings_since_transition"] = 6 if stalled else 1
    r["gain_log"] = [0.0, 0.0] if stalled else [5.0, 5.0]
    nb._save()
    return r


def _diag_report(nb, session=10, stage=None):
    r = nb.foci()[0]["relay"]
    nb._nb.setdefault("probe_reports", {})[WALL] = {
        "wall": WALL, "kind": "diagnose", "delivered_session": session,
        "n_envs": 256, "success_pct": 12.5, "died_pct": 80.3, "timeout_pct": 7.2,
        "spawn_floor": int(r["spawn_floor"]),
        "sub_stage": int(stage if stage is not None else r["sub_stage"]),
        "marginals": {"light": {"p25": 0.0, "med": 0.05, "p75": 0.2}},
    }
    nb._save()
    return nb._nb["probe_reports"][WALL]


def _whatif_report(nb, session=10, axis="uplock", direction="easier", delta=12.3):
    r = nb.foci()[0]["relay"]
    nb._nb.setdefault("probe_reports", {})[WALL] = {
        "wall": WALL, "kind": "whatif", "delivered_session": session, "n_envs": 256,
        "axis": axis, "direction": direction, "delta_pp": delta,
        "base_success_pct": 10.0, "variant_success_pct": 10.0 + delta,
        "spawn_floor": int(r["spawn_floor"]), "sub_stage": int(r["sub_stage"]),
    }
    nb._save()
    return nb._nb["probe_reports"][WALL]


def _hyp(axis="uplock", direction="easier", evidence="died 80.3% with light med 0.05"):
    return {
        "hypothesis": "students die in the dark before reaching the ladder",
        "evidence": evidence,
        "intervention": {"axis": axis, "direction": direction},
        "prediction": "zero-shot SR should rise by at least 8pp",
    }


def _log(nb):
    return nb._nb.get("hypothesis_log") or []


# ---- gates -----------------------------------------------------------------------------------------

def test_shape_gate_rejects_bad_axis_and_records(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    _diag_report(nb)
    nb.admit_hypothesis(WALL, _hyp(axis="gravity"), 11)
    assert _log(nb)[-1]["status"] == "rejected_shape"
    assert "bad shape" in nb.last_hypothesis_decision


def test_tier1_rejects_uncited_evidence_and_passes_report_numbers(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    _diag_report(nb)
    nb.admit_hypothesis(WALL, _hyp(evidence="roughly 55% die to gnomes"), 11)
    assert _log(nb)[-1]["status"] == "rejected_tier1"
    _diag_report(nb)  # fresh report (the previous one was consumed)
    nb.admit_hypothesis(WALL, _hyp(evidence="died 80.3%, light median 0.05"), 11)
    assert _log(nb)[-1]["status"] == "verify_scheduled"


def test_one_scientist_shot_per_report(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    rep = _diag_report(nb, session=10)
    assert nb.hypothesis_scientist_due(11)["wall"] == WALL
    nb.admit_hypothesis(WALL, _hyp(axis="gravity"), 11)   # even a rejection consumes the shot
    assert rep["hypothesized"] is True
    assert nb.hypothesis_scientist_due(12) is None


def test_scientist_due_skips_stale_and_verify_reports(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    _diag_report(nb, session=10)
    assert nb.hypothesis_scientist_due(30) is None          # stale
    nb._nb["probe_reports"][WALL]["delivered_session"] = 29
    nb._nb["probe_reports"][WALL]["verify_of"] = "h1_x"
    assert nb.hypothesis_scientist_due(30) is None          # verify reports never re-theorize


# ---- Tier-2 scheduling -----------------------------------------------------------------------------

def test_verify_schedules_whatif_pending_with_own_budget_kind(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    _diag_report(nb)
    nb.admit_hypothesis(WALL, _hyp(), 11)
    pend = nb._nb["probe_pending"]
    assert pend["kind"] == "whatif" and pend["verify_hypothesis_id"] == _log(nb)[-1]["id"]
    assert pend["verify_floor"] == r["spawn_floor"]
    assert pend["verify_sub_stage"] == r["sub_stage"]
    assert [11, "verify"] in nb._nb["probe_ledger"][WALL]
    # the modeler's own diagnose/whatif budget is untouched by the verify ledger kind
    assert nb._probe_budget_left(WALL, 11) == {"diagnose": 1, "whatif": 1}


def test_verify_budget_one_per_window_and_slot_respected(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    _diag_report(nb)
    nb._nb.setdefault("probe_ledger", {})[WALL] = [[8, "verify"]]
    nb.admit_hypothesis(WALL, _hyp(), 11)
    assert _log(nb)[-1]["status"] == "recorded"             # budget spent -> waits
    assert nb._nb["probe_pending"] is None
    # window rolls past -> housekeeping schedules it (still within freshness window)
    nb._nb["probe_ledger"][WALL] = [[11 - PROBE_BUDGET_WINDOW, "verify"]]
    nb.hypothesis_housekeeping(12)
    assert _log(nb)[-1]["status"] == "verify_scheduled"


def test_recorded_expires_past_freshness_window(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    _diag_report(nb)
    nb._nb.setdefault("probe_ledger", {})[WALL] = [[10, "verify"]]  # blocks scheduling
    nb.admit_hypothesis(WALL, _hyp(), 11)
    assert _log(nb)[-1]["status"] == "recorded"
    nb.hypothesis_housekeeping(20)
    assert _log(nb)[-1]["status"] == "expired"


# ---- verdicts + compile ----------------------------------------------------------------------------

def test_free_verdict_from_triggering_whatif_compiles_insert(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    k_before = int(r["sub_stage"])
    _whatif_report(nb, axis="uplock", direction="easier", delta=12.3)
    nb.admit_hypothesis(WALL, _hyp(evidence="paired delta +12.3pp"), 11)
    e = _log(nb)[-1]
    assert e["status"] == "verified_compiled" and e["return_stage"] == k_before
    assert r["sub_stage"] == RUNG_INSERT_STAGE
    ins = r["stage_insert"]
    assert ins["knobs"]["uplock"] is True and ins["hypothesis_id"] == e["id"]
    assert r["rung_history"][-1]["event"] == "hypothesis_insert"
    assert nb._nb["probe_pending"] is None                  # no probe spent
    sc = nb.relay_scaffold(WALL)
    assert sc["sub_stage"] == RUNG_INSERT_STAGE and sc["uplock"] is True


def test_free_verdict_below_bar_refutes_without_probe(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    _whatif_report(nb, axis="uplock", direction="easier", delta=2.1)
    nb.admit_hypothesis(WALL, _hyp(evidence="paired delta +2.1pp"), 11)
    e = _log(nb)[-1]
    assert e["status"] == "refuted" and e["delta_pp"] == 2.1
    assert r["sub_stage"] != RUNG_INSERT_STAGE and "stage_insert" not in r
    assert nb._nb["probe_pending"] is None
    assert "★REFUTED" in nb.render_for_prompt()


def test_housekeeping_verdict_compiles_verified_verify_report(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    _diag_report(nb)
    nb.admit_hypothesis(WALL, _hyp(), 11)
    hid = _log(nb)[-1]["id"]
    nb.deliver_probe_report({"wall": WALL, "kind": "whatif", "delta_pp": 15.0,
                             "axis": "uplock", "direction": "easier"}, 12)
    assert nb._nb["probe_reports"][WALL]["verify_of"] == hid
    nb.hypothesis_housekeeping(13)
    assert _log(nb)[-1]["status"] == "verified_compiled"
    assert r["sub_stage"] == RUNG_INSERT_STAGE
    assert "VERIFIED" in nb.last_hypothesis_decision


def test_housekeeping_stale_context_records_but_never_compiles(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    _diag_report(nb)
    nb.admit_hypothesis(WALL, _hyp(), 11)
    nb.deliver_probe_report({"wall": WALL, "kind": "whatif", "delta_pp": 15.0}, 12)
    r["sub_stage"] = int(r["sub_stage"]) - 1                # the ladder moved meanwhile
    nb._save()
    nb.hypothesis_housekeeping(13)
    assert _log(nb)[-1]["status"] == "stale_context"
    assert r["sub_stage"] != RUNG_INSERT_STAGE and "stage_insert" not in r


def test_r0_pin_refuses_credit_only_insert(nb_path):
    nb = SiegeNotebook(nb_path, thresholds=SiegeThresholds(rung_r0_scaffold=True))
    _open_relay(nb, r0=3)
    r = _set_relay(nb, floor=3)                             # AT the target floor
    _whatif_report(nb, axis="monster_credit", direction="easier", delta=20.0)
    nb.admit_hypothesis(
        WALL, _hyp(axis="monster_credit", evidence="paired delta +20.0pp"), 11
    )
    e = _log(nb)[-1]
    assert e["status"] == "compile_refused" and "r0_pin" in e["note"]
    assert r["sub_stage"] != RUNG_INSERT_STAGE and "stage_insert" not in r


# ---- the inserted rung under the existing state machine --------------------------------------------

def _compile_insert(nb):
    r = nb.foci()[0]["relay"]
    _whatif_report(nb, axis="uplock", direction="easier", delta=12.3)
    nb.admit_hypothesis(WALL, _hyp(evidence="paired delta +12.3pp"), 11)
    assert r["sub_stage"] == RUNG_INSERT_STAGE
    return r, _log(nb)[-1]


def test_insert_graduates_back_to_return_stage(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    k = int(r["sub_stage"])
    r, e = _compile_insert(nb)
    # P2': graduation judges the last-RUNG_WIN window mean — the window fills first.
    for i in range(RUNG_WIN - 1 + nb.th.rung_substage_graduate_x):
        status = nb.note_rung_reading(WALL, nb.th.rung_graduate_sr + 5.0, session_idx=12 + i)
    assert "RUNG_INSERT_GRADUATED" in status
    assert r["sub_stage"] == k and "stage_insert" not in r
    assert e["status"] == "insert_graduated"
    assert r["rung_history"][-1]["event"] == "rung_insert_graduated"


def test_insert_stall_removes_insert_and_regresses_normally(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    k = int(r["sub_stage"])
    max_stage, _, _, _ = nb._ladder_shape(r)
    assert k < max_stage
    r, e = _compile_insert(nb)
    burn0 = int(r.get("regress_count", 0))
    status = None
    for i in range(RUNG_WIN - 1 + nb.th.rung_stall_readings):   # P2': window fills first
        status = nb.note_rung_reading(WALL, 1.0, session_idx=12 + i)
    assert "RUNG_INSERT_STALLED" in status
    assert r["sub_stage"] == k + 1 and "stage_insert" not in r
    assert e["status"] == "insert_stalled"
    assert int(r.get("regress_count", 0)) == burn0 + 1      # regress-family move, budget charged


def test_single_insert_invariant_no_insert_on_insert(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r, _ = _compile_insert(nb)
    ret = r["stage_insert"]["return_stage"]
    e2 = {"id": "h2x", "wall": WALL, "session": 12, "axis": "needs_clock",
          "direction": "easier", "hypothesis": "x", "evidence": "y", "prediction": "z",
          "status": "recorded"}
    nb._nb["hypothesis_log"].append(e2)
    assert nb._try_compile_hypothesis(nb.foci()[0], e2, 12, 15.0) is False
    assert e2["status"] == "compile_refused" and e2["note"] == "insert_already_active"
    assert r["stage_insert"]["return_stage"] == ret          # original resume point intact
    assert nb._schedule_hypothesis_verify(nb.foci()[0], e2, 12) is False
    assert nb._nb["probe_pending"] is None                   # verify budget not burnt either


def test_insert_ratchet_is_fresh_per_insert(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    key = f"{int(r['spawn_floor'])}:{RUNG_INSERT_STAGE}"
    r.setdefault("best_by_rung", {})[key] = 77.0             # a previous insert's stale best
    nb._save()
    r, _ = _compile_insert(nb)
    assert r["best_rung_trained"] is None                    # genuinely-new-rung semantics
    assert key not in (r.get("best_by_rung") or {})


def test_insert_neither_streak_leaves_it_active(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r, _ = _compile_insert(nb)
    nb.note_rung_reading(WALL, 25.0, session_idx=12)        # mid reading: no move either way
    assert r["sub_stage"] == RUNG_INSERT_STAGE and isinstance(r.get("stage_insert"), dict)


# ---- disclosure + renders + switches ---------------------------------------------------------------

def test_pre_light_override_reaches_facts_and_scaffold(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb, stage=2)                             # an entry stage (no radius)
    _whatif_report(nb, axis="pre_light", direction="easier", delta=25.0)
    nb.admit_hypothesis(WALL, _hyp(axis="pre_light", evidence="paired delta +25.0pp"), 11)
    assert _log(nb)[-1]["status"] == "verified_compiled"
    sc = nb.relay_scaffold(WALL)
    # v7fix5.7: from a dark entry stage, one easier notch = "ladder" (down-ladder stamp only)
    assert sc["pre_light"] == "ladder" and sc["down_ladder_radius"] is None
    facts = nb._scaffold_fact_clauses(sc)
    assert "torch-lit" in facts["pre-light"]                # override wins over the radius rule
    assert "entry" in facts["spawn"]
    txt = nb.render_for_prompt()
    assert "INSERTED rung" in txt and "★VERIFIED" in txt


def test_hypothesis_log_survives_coerce_and_reload(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    _diag_report(nb)
    nb.admit_hypothesis(WALL, _hyp(), 11)
    re = SiegeNotebook(nb_path)
    assert re._nb["hypothesis_log"][-1]["status"] == "verify_scheduled"
    assert re._nb["probe_pending"]["verify_hypothesis_id"] == re._nb["hypothesis_log"][-1]["id"]


def test_master_switch_off_is_a_noop(nb_path):
    nb = SiegeNotebook(nb_path, thresholds=SiegeThresholds(hypothesis_loop=False))
    _open_relay(nb)
    _set_relay(nb)
    rep = _diag_report(nb)
    assert nb.hypothesis_scientist_due(11) is None
    nb.admit_hypothesis(WALL, _hyp(), 11)
    nb.hypothesis_housekeeping(12)
    assert _log(nb) == [] and "hypothesized" not in rep
    assert nb._nb["probe_pending"] is None
