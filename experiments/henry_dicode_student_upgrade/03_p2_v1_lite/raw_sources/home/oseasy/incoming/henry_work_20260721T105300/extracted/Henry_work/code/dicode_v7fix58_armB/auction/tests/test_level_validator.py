"""v6fix7 P0.2 — SiegeLevelValidator rules + reroll/fallback wiring.

Pins the code-enforced level rules that used to live only in the prompt:
  R2_GIFT   the siege focus / drill target can never be gifted via Completed;
  R3_CHAIN  a drill may not compress its own execution chain into Completed;
  R1_SHAPE  a drill's Relevant carries nothing unrelated;
  R1_SUPERSET (warn-only) DEPTH/BREADTH keep the lineage superset rule;
  R4_WORLD  (warn-only) forced-back links should be world-supported.
Plus the gen_manager reroll loop: violators are re-queried with feedback; after the budget the
mechanical fallback moves gift/chain names back into Relevant. Siege-off / no-focus = strict no-op.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction.level_validator import (  # noqa: E402
    RULE_CHAIN,
    RULE_GIFT,
    RULE_SHAPE,
    RULE_SUPERSET,
    apply_fallback_fixes,
    render_violation_feedback,
    reroll_worthy,
    validate_level,
)

_GM_PATH = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
_spec = importlib.util.spec_from_file_location("dicode_v6_gen_manager_validator_test", _GM_PATH)
_gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gm)
TaskGenerator = _gm.TaskGenerator


FOCI = [
    {
        "skill": "make_iron_pickaxe",
        "prereq_tree": [
            {"skill": "collect_iron", "state": "CONSOLIDATED", "sr": 88.0},
            {"skill": "collect_coal", "state": "CONSOLIDATED", "sr": 94.0},
            {"skill": "place_furnace", "state": "CONSOLIDATED", "sr": 98.0},
            {"skill": "place_table", "state": "CONSOLIDATED", "sr": 100.0},
        ],
    }
]

DRILL_META = {
    "type": "CONSOLIDATE",
    "drill_target": "make_iron_pickaxe",
    "siege_wall": "make_iron_pickaxe",
}


def _doc(relevant: str, completed: str) -> str:
    return (
        "Objective: drill the iron pickaxe.\n"
        f"Relevant Achievements: {relevant}\n"
        f"Completed Achievements: {completed}\n"
        "World:\n- Player: floor 0, iron and coal veins adjacent, table placed.\n"
    )


# ---------- rules ----------


def test_clean_drill_passes():
    doc = _doc("MAKE_IRON_PICKAXE, COLLECT_IRON, COLLECT_COAL, PLACE_FURNACE, PLACE_TABLE", "COLLECT_WOOD")
    assert validate_level(doc, DRILL_META, FOCI) == []


def test_gift_of_focus_detected_any_type():
    doc = _doc("COLLECT_DIAMOND", "MAKE_IRON_PICKAXE, COLLECT_WOOD")
    violations = validate_level(doc, {"type": "DEPTH", "drill_target": None, "siege_wall": None}, FOCI)
    assert [v.rule for v in violations] == [RULE_GIFT]
    assert violations[0].fixable_moves == ["make_iron_pickaxe"]
    assert reroll_worthy(violations)


def test_drill_chain_compression_detected():
    doc = _doc("MAKE_IRON_PICKAXE", "COLLECT_IRON, COLLECT_COAL, COLLECT_WOOD")
    violations = validate_level(doc, DRILL_META, FOCI)
    rules = {v.rule for v in violations}
    assert RULE_CHAIN in rules
    chain_v = next(v for v in violations if v.rule == RULE_CHAIN)
    assert set(chain_v.fixable_moves) == {"collect_iron", "collect_coal"}


def test_drill_unrelated_relevant_detected():
    doc = _doc("MAKE_IRON_PICKAXE, DEFEAT_ZOMBIE, EAT_COW", "COLLECT_WOOD")
    violations = validate_level(doc, DRILL_META, FOCI)
    shape = [v for v in violations if v.rule == RULE_SHAPE]
    assert len(shape) == 1
    assert "DEFEAT_ZOMBIE" in shape[0].message and "EAT_COW" in shape[0].message
    assert reroll_worthy(violations)


def test_superset_rule_warn_only_for_depth():
    doc = _doc("COLLECT_DIAMOND", "COLLECT_WOOD")
    violations = validate_level(
        doc,
        {"type": "DEPTH", "drill_target": None, "siege_wall": None},
        FOCI,
        parent_relevant={"collect_diamond", "defeat_zombie"},
    )
    assert [v.rule for v in violations] == [RULE_SUPERSET]
    assert not reroll_worthy(violations)  # warn-only: never burns a reroll by itself


def test_no_foci_is_strict_noop():
    doc = _doc("DEFEAT_ZOMBIE", "MAKE_IRON_PICKAXE")
    assert validate_level(doc, DRILL_META, []) == []


def test_unmastered_links_are_allowed_in_drill_relevant():
    doc = _doc("MAKE_IRON_PICKAXE, COLLECT_IRON, DEFEAT_SKELETON", "COLLECT_WOOD")
    violations = validate_level(doc, DRILL_META, FOCI, unmastered={"defeat_skeleton"})
    assert violations == []  # forced unmastered link rides along legally


# ---------- fallback + feedback ----------


def test_fallback_moves_gift_and_chain_back():
    doc = _doc("COLLECT_DIAMOND", "MAKE_IRON_PICKAXE, COLLECT_IRON, COLLECT_WOOD")
    violations = validate_level(doc, DRILL_META, FOCI)
    fixed, moved = apply_fallback_fixes(doc, violations)
    assert set(moved) == {"make_iron_pickaxe", "collect_iron"}
    assert "MAKE_IRON_PICKAXE" in fixed.split("Relevant Achievements:")[1].split("\n")[0]
    # collect_wood (genuinely mastered, not a chain/gift name) must stay in Completed
    assert "COLLECT_WOOD" in fixed.split("Completed Achievements:")[1].split("\n")[0]


def test_feedback_mentions_every_rule():
    doc = _doc("MAKE_IRON_PICKAXE, DEFEAT_ZOMBIE", "COLLECT_IRON")
    violations = validate_level(doc, DRILL_META, FOCI)
    text = render_violation_feedback(violations)
    assert "REJECTED" in text
    for v in violations:
        assert v.rule in text


# ---------- gen_manager reroll wiring ----------


class _NotebookStub:
    """Mirrors the real SiegeNotebook read API used by _siege_validate_and_reroll."""

    def __init__(self, foci):
        self._foci = foci

    def foci(self):
        return list(self._foci)

    def focus_skills(self):
        return [f["skill"] for f in self._foci if isinstance(f.get("skill"), str)]

    def required_form(self, skill):
        return None


class _RerollLLM:
    """Always returns a clean drill on the reroll query."""

    def __init__(self):
        self.calls = []

    def query(self, system_prompt, user_prompts):
        self.calls.append(list(user_prompts))
        clean = _doc(
            "MAKE_IRON_PICKAXE, COLLECT_IRON, COLLECT_COAL, PLACE_FURNACE, PLACE_TABLE",
            "COLLECT_WOOD",
        )
        return [
            {
                "content": '<level_meta>{"type": "CONSOLIDATE", "drill_target": "make_iron_pickaxe", '
                '"siege_wall": "make_iron_pickaxe"}</level_meta>'
                f"<docstring>{clean}</docstring>"
            }
            for _ in user_prompts
        ]


class _ArchiveStub:
    def get_task_descriptions(self, ids):
        return {i: "Relevant Achievements: COLLECT_WOOD" for i in ids}


def _bare_generator(foci):
    gm = object.__new__(TaskGenerator)
    gm._siege_notebook = _NotebookStub(foci)
    gm.archive = _ArchiveStub()
    gm.llm = _RerollLLM()
    return gm


def test_reroll_replaces_violating_level():
    gm = _bare_generator(FOCI)
    bad = {
        "description": _doc("COLLECT_DIAMOND", "MAKE_IRON_PICKAXE"),
        "level_meta": {"type": "DEPTH", "drill_target": None, "siege_wall": None},
        "reasoning": "r",
    }
    out = gm._siege_validate_and_reroll(
        [bad], ["user prompt 0"], "sys", [["parent_0"]], proposer_idx=0, siege_unmastered=set()
    )
    assert len(out) == 1
    assert "MAKE_IRON_PICKAXE" not in (out[0]["description"].split("Completed Achievements:")[1])
    # the reroll prompt carried the violation feedback
    assert any("REJECTED" in p for call in gm.llm.calls for p in call)


def test_reroll_noop_when_no_foci():
    gm = _bare_generator([])
    bad = {"description": _doc("X", "MAKE_IRON_PICKAXE"), "level_meta": None}
    out = gm._siege_validate_and_reroll(
        [bad], ["u0"], "sys", [["p0"]], proposer_idx=0, siege_unmastered=set()
    )
    assert out[0] is bad
    assert gm.llm.calls == []  # zero LLM traffic on the no-focus path


def test_reroll_skips_none_placeholders():
    gm = _bare_generator(FOCI)
    clean = {
        "description": _doc(
            "MAKE_IRON_PICKAXE, COLLECT_IRON, COLLECT_COAL, PLACE_FURNACE, PLACE_TABLE",
            "COLLECT_WOOD",
        ),
        "level_meta": DRILL_META,
    }
    out = gm._siege_validate_and_reroll(
        [None, clean], ["u0", "u1"], "sys", [["p0"], ["p1"]], proposer_idx=1, siege_unmastered=set()
    )
    assert out[0] is None and out[1] is clean
    assert gm.llm.calls == []
