"""Regression tests for +skill_preflight.r3_floor_clause (surgical R3-floor ablation).

Provenance: matched-seed archive audit 2026-08-09 — the gate's LLM repair rewrote 24/27
trained mob levels from floor 2 to floor 0 (descriptions left stale), starving the floor-2
cluster. The surgical flag removes ONLY the floor channel from R3's scaffolded set.

Contract under test:
  * default (True): behavior byte-identical to the shipped gate — the floor-2 mob level
    is still an R3 violation (regression witness);
  * False: that level passes; the premark and inventory channels of R3 still fire;
    R1/R2 are untouched; the evidence string never indicts the floor.
"""
from dicode.skill_preflight.scaffold_gate import check_code

SR = {
    "defeat_gnome_warrior": 0.0,
    "enter_gnomish_mines": 0.20,
    "enter_dungeon": 0.90,
    "collect_iron": 0.30,
    "make_stone_pickaxe": 0.50,
    "collect_wood": 0.95,
}


def lvl(relevant, premark=(), floor=0, inv=""):
    body = [
        "class T:",
        "    def build(self, builder):",
        f"        self.relevant_achievements = [{', '.join('Achievement.' + r for r in relevant)}]",
    ]
    if premark:
        body.append(
            f"        self.completed_achievements = [{', '.join('Achievement.' + p for p in premark)}]"
        )
    if inv:
        body.append(f"        builder.set_player_inventory({inv})")
    if floor:
        body.append(f"        builder.set_starting_floor({floor})")
    return "\n".join(body) + "\n"


FLOOR_MOB = lvl(["DEFEAT_GNOME_WARRIOR"], floor=2)


def test_default_floor_mob_still_violates():
    v = check_code(FLOOR_MOB, SR)
    assert not v.ok and "R3_focus_prereq_scaffolded" in v.violations
    assert "set_starting_floor" in v.evidence


def test_surgical_floor_mob_passes():
    v = check_code(FLOOR_MOB, SR, r3_floor_clause=False)
    assert v.ok and v.violations == []


def test_surgical_premark_channel_alive():
    v = check_code(
        lvl(["DEFEAT_GNOME_WARRIOR"], premark=["ENTER_GNOMISH_MINES"]),
        SR, r3_floor_clause=False,
    )
    assert "R3_focus_prereq_scaffolded" in v.violations


def test_surgical_inventory_channel_alive():
    v = check_code(
        lvl(["COLLECT_IRON"], inv='{"pickaxe": 2}'),
        SR, r3_floor_clause=False,
    )
    assert "R3_focus_prereq_scaffolded" in v.violations


def test_surgical_r1_r2_untouched():
    v = check_code(
        lvl(["DEFEAT_GNOME_WARRIOR"], premark=["COLLECT_WOOD", "DEFEAT_GNOME_WARRIOR"]),
        SR, r3_floor_clause=False,
    )
    assert {"R1_premark_mastered", "R2_focus_premarked"} <= set(v.violations)


def test_surgical_evidence_never_indicts_floor():
    v = check_code(
        lvl(["DEFEAT_GNOME_WARRIOR"], premark=["ENTER_GNOMISH_MINES"], floor=2),
        SR, r3_floor_clause=False,
    )
    assert not v.ok and "set_starting_floor" not in v.evidence
