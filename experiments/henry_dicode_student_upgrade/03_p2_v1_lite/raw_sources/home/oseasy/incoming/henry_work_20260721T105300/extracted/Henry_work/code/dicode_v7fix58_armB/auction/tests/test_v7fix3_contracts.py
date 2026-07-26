"""v7fix3 §9 — three-party communication contracts, pinned as PAIRED anchors.

The fix-series recurring failure mode is one-sided drift: a gate learns a new refusal but no
prompt teaches the recovery (v7 first run: ⑦ swallowed relay proposals it never taught), or a
spec changes vocabulary but the consumer keeps the old words (v7fix1: Inventory TypeErrors).
Every test here asserts BOTH SIDES of one contract, so editing either side alone fails a test.

Pure python — reads module source / prompt strings only. No jax/LLM.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import auction.modeler as modeler_mod  # noqa: E402
import auction.siege_notebook as nb_mod  # noqa: E402
from auction.level_meta import LEVEL_META_SPEC_TEXT  # noqa: E402
from auction.level_validator import validate_level  # noqa: E402

_NB_SRC = open(nb_mod.__file__, encoding="utf-8").read()
_MODELER_SRC = open(modeler_mod.__file__, encoding="utf-8").read()

_ECO_PERSONA_PATH = os.path.join(
    _REPO, "src", "dicode", "dreaming", "prompts", "dicode", "persona_ecology_coop.py"
)
_AMB_PERSONA_PATH = os.path.join(
    _REPO, "src", "dicode", "dreaming", "prompts", "dicode", "persona_ambitious_coop.py"
)


def _load_prompts(path):
    ns: dict = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    return ns["system_prompt"], ns["user_prompt"]


# ---- contract 1: tier_locked refusal <-> modeler TIER LOCK teaching -------------------------------


def test_tier_locked_contract_pair():
    # gate side: the refusal names the exact recovery field.
    assert "tier_locked(" in _NB_SRC
    assert "re-propose THIS wall WITH relay_r0_floor" in _NB_SRC
    # prompt side: the modeler is taught the rule AND the decision word it will see.
    assert "TIER LOCK" in _MODELER_SRC
    assert "tier_locked" in _MODELER_SRC


# ---- contract 2: relay_attached upgrade <-> journal hint <-> modeler rule (g) ---------------------


def test_relay_attach_contract_pair():
    # gate side: the attach decision + the zero-win refusal wording exist.
    assert "relay_attached(" in _NB_SRC
    assert "relay_attach_refused(" in _NB_SRC
    # journal side: the zero-win hint teaches the SAME re-proposal format.
    assert "★ZERO-WIN" in _NB_SRC and "UPGRADE the running focus" in _NB_SRC
    # modeler side: rule (g) names the decision word and the field.
    assert "relay_attached" in _MODELER_SRC
    assert "UPGRADED IN PLACE" in _MODELER_SRC


# ---- contract 3: R6 breadth lane <-> level_meta spec <-> ecology persona --------------------------


def test_breadth_spawn_lane_contract_pair():
    # validator side: the two-lane teaching message (produce it live, not via source grep).
    desc = "Relevant Achievements: EAT_COW\nCompleted Achievements: NONE"
    meta = {"type": "DEPTH", "drill_target": None, "siege_wall": None,
            "spawn_floor": 2, "spawn_kit": None}
    v = validate_level(desc, meta, foci=[], breadth_frontier=1)
    assert v and "BREADTH" in v[0].message and "frontier" in v[0].message
    # spec side: the proposer-facing vocabulary names the same two lanes.
    assert "BREADTH SPAWN FRONTIER" in LEVEL_META_SPEC_TEXT
    assert "SPAWN-ANNEAL RELAY" in LEVEL_META_SPEC_TEXT
    # persona side: the ecology designer is told the lever and its bound.
    sys_p, user_p = _load_prompts(_ECO_PERSONA_PATH)
    assert "BREADTH SPAWN FRONTIER" in sys_p + user_p
    assert "spawn_floor" in sys_p + user_p


# ---- contract 4: role split (P4) — persona pair + modeler note ------------------------------------


def test_role_split_contract_pair():
    sys_e, user_e = _load_prompts(_ECO_PERSONA_PATH)
    # the ecology persona consumes ECOLOGY_DIRECTIVE (and never the siege directive).
    assert "{ECOLOGY_DIRECTIVE}" in user_e
    assert "{SIEGE_DIRECTIVE}" not in user_e
    # the siege persona is unchanged: still consumes SIEGE_DIRECTIVE.
    _, user_a = _load_prompts(_AMB_PERSONA_PATH)
    assert "{SIEGE_DIRECTIVE}" in user_a
    # the modeler knows the team is split (its TYPE guidance serves the siege designer).
    assert "ECOLOGY designer" in _MODELER_SRC


def test_ecology_persona_keeps_dicode_hardcore_blocks_verbatim():
    """The parser/coder contract: the ecology persona must keep every DiCode hardcore anchor the
    downstream parse + env-codegen relies on, byte-identical in the load-bearing lines."""
    sys_e, user_e = _load_prompts(_ECO_PERSONA_PATH)
    sys_a, user_a = _load_prompts(_AMB_PERSONA_PATH)
    anchors = [
        # knowledge base / toolkit skeleton (placeholders the system prompt is .format()ed with)
        "{CONSTANTS}", "{MOBS}", "{GAME_MECHANICS}", "{WORLD_GEN}", "{API_DOCS}",
        "<game_rules>", "</game_rules>", "<api_docs>", "</api_docs>",
        # output format contract
        "<reasoning>", "</reasoning>", "<docstring>", "</docstring>",
        "Relevant Achievements:", "Completed Achievements:",
        "**CRITICAL RULE: MANAGING ACHIEVEMENT LISTS**",
        "**SPECIFICITY REQUIREMENT (NON-NEGOTIABLE)**",
        "★HARD RULE — DO NOT COMPRESS AWAY A PREREQUISITE THE STUDENT HAS NOT MASTERED.",
        # docstring template fields the env coder keys on
        "Objective: [", "Description: [", "World:", "- Player: [", "- Map: [", "- Mechanics: [",
    ]
    for a in anchors:
        assert a in sys_e, f"ecology system prompt lost hardcore anchor: {a!r}"
        assert a in sys_a, f"anchor drifted out of the AMBITIOUS persona too: {a!r}"
    # user-prompt placeholders the prompt builder supplies for every persona
    for ph in ("{MASTERED_TASK}", "{TASK_PERFORMANCE_CONTEXT}", "{GLOBAL_AGENT_PROFILE}",
               "{PARENT_CHILD_HISTORY}", "{MY_TURN_ORDER}", "{MODELER_GUIDANCE}",
               "{PEER_ALREADY_MADE}", "{REFERENCE_LEVEL}", "{LEVEL_META_SPEC}"):
        assert ph in user_e, f"ecology user prompt lost placeholder: {ph!r}"


# ---- contract 5: P5 full-price cap wording <-> config anchor --------------------------------------


def test_force_cap_contract_pair():
    ev_path = os.path.join(_REPO, "src", "dicode", "evolution_efficient.py")
    ev_src = open(ev_path, encoding="utf-8").read()
    assert "focus_force_cap" in ev_src and "full-price" in ev_src
    assert nb_mod.FOCUS_FORCE_CAP == 8  # D3 calibration: above fix8's healthy 6-10/session band
    th = nb_mod.SiegeThresholds()
    assert th.focus_force_cap == 8 and th.zero_win_force_cap == 2
