"""v6fix7 P0.1 — <level_meta> block: spec rendering, tolerant parsing, retry gating, node attrs.

Pins the whole metadata chain: spec text only renders on siege sessions (baseline prompt
byte-unchanged), the parser tolerates absence/mild malformation, CONSOLIDATE without a
drill_target counts as incomplete (so the retry loop re-queries), and parsed metadata lands on
the archive node (skipped entirely when absent).
"""

import importlib.util
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))  # dicode_v6/
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction.level_meta import (  # noqa: E402
    LEVEL_META_SPEC_TEXT,
    level_meta_complete,
    parse_level_meta,
    render_level_meta_spec,
)

# Load gen_manager BY ABSOLUTE PATH (same rationale as test_siege_directive_style_note.py: the
# conda env's editable .pth may otherwise resolve `dicode` to a sibling checkout).
_GM_PATH = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
_spec = importlib.util.spec_from_file_location("dicode_v6_gen_manager_lvlmeta_test", _GM_PATH)
_gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gm)
TaskGenerator = _gm.TaskGenerator
TaskArchive = _gm.TaskArchive


# ---------- spec rendering ----------


def test_spec_empty_when_siege_off():
    assert render_level_meta_spec(False) == ""


def test_spec_present_when_siege_on():
    text = render_level_meta_spec(True)
    assert text == LEVEL_META_SPEC_TEXT
    assert "<level_meta>" in text and "drill_target" in text


# ---------- parsing ----------


def test_parse_valid_block():
    content = (
        "<reasoning>...</reasoning>\n"
        '<level_meta>{"type": "CONSOLIDATE", "drill_target": "make_iron_pickaxe", '
        '"siege_wall": "make_iron_pickaxe"}</level_meta>\n'
        "<docstring>Objective: drill.</docstring>"
    )
    meta = parse_level_meta(content)
    assert meta == {
        "type": "CONSOLIDATE",
        "drill_target": "make_iron_pickaxe",
        "siege_wall": "make_iron_pickaxe",
    }


def test_parse_absent_block_is_none():
    assert parse_level_meta("<docstring>Objective: x.</docstring>") is None
    assert parse_level_meta("") is None
    assert parse_level_meta(None) is None


def test_parse_repairs_pythonish_literals():
    content = "<level_meta>{'type': 'depth', 'drill_target': None, 'siege_wall': 'Enter_Dungeon'}</level_meta>"
    meta = parse_level_meta(content)
    assert meta["type"] == "DEPTH"
    assert meta["drill_target"] is None
    assert meta["siege_wall"] == "enter_dungeon"


def test_parse_null_variants_and_case():
    content = '<level_meta>{"type": "Breadth", "drill_target": "null", "siege_wall": ""}</level_meta>'
    meta = parse_level_meta(content)
    assert meta == {"type": "BREADTH", "drill_target": None, "siege_wall": None}


def test_parse_unknown_type_is_none():
    assert parse_level_meta('<level_meta>{"type": "SIEGE"}</level_meta>') is None


def test_parse_garbage_json_is_none():
    assert parse_level_meta("<level_meta>{type: DEPTH,}</level_meta>") is None


# ---------- completeness (retry gating) ----------


def test_complete_consolidate_requires_target():
    assert not level_meta_complete({"type": "CONSOLIDATE", "drill_target": None, "siege_wall": None})
    assert level_meta_complete(
        {"type": "CONSOLIDATE", "drill_target": "make_iron_pickaxe", "siege_wall": None}
    )


def test_complete_depth_needs_no_target():
    assert level_meta_complete({"type": "DEPTH", "drill_target": None, "siege_wall": None})
    assert not level_meta_complete(None)
    assert not level_meta_complete({"type": "??", "drill_target": None, "siege_wall": None})


# ---------- wiring: _parse_generation_response carries level_meta ----------


def _bare_generator():
    return object.__new__(TaskGenerator)


def test_parse_generation_response_extracts_meta():
    gm = _bare_generator()
    content = (
        "<reasoning>r</reasoning>"
        '<level_meta>{"type": "DEPTH", "drill_target": null, "siege_wall": null}</level_meta>'
        "<docstring>Objective: x.</docstring>"
    )
    parsed = gm._parse_generation_response({"content": content})
    assert parsed["description"] == "Objective: x."
    assert parsed["level_meta"]["type"] == "DEPTH"


def test_parse_generation_response_meta_absent_is_none():
    gm = _bare_generator()
    parsed = gm._parse_generation_response({"content": "<docstring>Objective: x.</docstring>"})
    assert parsed["description"] == "Objective: x."
    assert parsed["level_meta"] is None


# ---------- wiring: retry loop re-queries when meta required but missing ----------


class _FakeLLM:
    """First query returns a metadata-less response; the retry returns a complete one."""

    def __init__(self):
        self.calls = 0

    def query(self, system_prompt, user_prompts):
        self.calls += 1
        if self.calls == 1:
            return [{"content": "<docstring>Objective: no meta.</docstring>"} for _ in user_prompts]
        return [
            {
                "content": '<level_meta>{"type": "DEPTH", "drill_target": null, "siege_wall": null}'
                "</level_meta><docstring>Objective: with meta.</docstring>"
            }
            for _ in user_prompts
        ]


def test_retry_loop_requires_meta_when_flagged():
    gm = _bare_generator()
    gm.llm = _FakeLLM()
    out = gm._query_and_parse_responses("sys", ["u1"], max_retries=3, require_level_meta=True)
    assert len(out) == 1
    assert out[0]["description"] == "Objective: with meta."
    assert gm.llm.calls == 2  # retried exactly once for the missing block


def test_retry_loop_ignores_meta_when_not_required():
    gm = _bare_generator()
    gm.llm = _FakeLLM()
    out = gm._query_and_parse_responses("sys", ["u1"], max_retries=3, require_level_meta=False)
    assert len(out) == 1
    assert out[0]["description"] == "Objective: no meta."
    assert gm.llm.calls == 1  # baseline behaviour: no extra query


# ---------- wiring: archive node attrs ----------


def _bare_archive():
    import networkx as nx

    ar = object.__new__(TaskArchive)
    ar.graph = nx.DiGraph()
    ar._lock = threading.Lock()
    return ar


def test_set_level_meta_writes_attrs_and_skips_none():
    ar = _bare_archive()
    ar.graph.add_node("task_1")
    ar.set_level_meta(
        "task_1", {"type": "CONSOLIDATE", "drill_target": "make_iron_pickaxe", "siege_wall": None}
    )
    node = ar.graph.nodes["task_1"]
    assert node["level_type"] == "CONSOLIDATE"
    assert node["drill_target"] == "make_iron_pickaxe"
    assert "siege_wall" not in node  # None never written (graphml-safe)


def test_set_level_meta_missing_node_is_noop():
    ar = _bare_archive()
    ar.set_level_meta("ghost", {"type": "DEPTH"})  # must not raise
    assert "ghost" not in ar.graph.nodes
