"""v7fix5.1 — GLM empty-response guard on the modeler (root-caused 2026-07-15).

The disease: with think:true at v7 prompt sizes, GLM's reasoning length is heavy-tailed and
20-50% of calls burned the WHOLE max_tokens budget inside thinking — the API returns HTTP 200
with finish_reason="length" and content="" — and diagnose_siege silently degraded to empty
guidance with zero retries (its reroll loop only fired on attribution violations). These tests
pin the guard:

  G1  an empty first response is RETRIED (same attempt budget), and a good second response wins;
  G2  all-empty responses exhaust the budget (1 + _ATTRIB_REROLL_MAX calls), degrade to the
      empty-but-valid structure, and never crash;
  G3  a good first response makes exactly ONE call (no retry tax on the healthy path);
  G4  the retry log line carries the finish_reason so truncation stays diagnosable from .out.
"""

import json

import pytest

from auction.modeler import _ATTRIB_REROLL_MAX, Modeler
from auction.student_profile_log import StudentProfileLog


GOOD_JSON = json.dumps(
    {
        "student_states": {"collect_wood": "MASTERED"},
        "guidance_per_parent": {},
        "siege_update": {"foci": []},
    }
)


class _ScriptedLLM:
    """query() pops scripted responses; records how many calls were made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def query(self, system_prompt, user_prompts):
        self.calls += 1
        if self._responses:
            return [self._responses.pop(0)]
        return [{"content": "", "finish_reason": "length"}]


def _make_modeler(llm, tmp_path):
    profile_log = StudentProfileLog(path=str(tmp_path / "profile.json"))
    return Modeler(llm, archive=None, profile_log=profile_log)


def _diagnose(modeler):
    return modeler.diagnose_siege(
        session_idx=226,
        parent_ids=[],
        parent_context={},
        notebook_text="(empty siege notebook)",
        combat_targets=["defeat_kobold"],
    )


def test_g1_empty_response_retried_then_good_wins(tmp_path):
    llm = _ScriptedLLM(
        [
            {"content": "", "finish_reason": "length"},  # the s226 production failure shape
            {"content": GOOD_JSON, "finish_reason": "stop"},
        ]
    )
    out = _diagnose(_make_modeler(llm, tmp_path))
    assert llm.calls == 2, "empty parse must retry, not silently degrade"
    assert out["student_states"].get("collect_wood") == "MASTERED"


def test_g2_all_empty_degrades_after_budget(tmp_path):
    llm = _ScriptedLLM([])  # every call comes back empty/truncated
    out = _diagnose(_make_modeler(llm, tmp_path))
    assert llm.calls == 1 + _ATTRIB_REROLL_MAX, "must exhaust exactly the shared attempt budget"
    assert out["student_states"] == {}
    assert out["guidance_per_parent"] == {}
    assert out["siege_update"]["foci"] == []


def test_g3_good_first_response_makes_one_call(tmp_path):
    llm = _ScriptedLLM([{"content": GOOD_JSON, "finish_reason": "stop"}])
    out = _diagnose(_make_modeler(llm, tmp_path))
    assert llm.calls == 1, "healthy path must pay no retry tax"
    assert out["student_states"].get("collect_wood") == "MASTERED"


def test_g4_retry_log_carries_finish_reason(tmp_path, capsys):
    llm = _ScriptedLLM(
        [
            {"content": "", "finish_reason": "length"},
            {"content": GOOD_JSON, "finish_reason": "stop"},
        ]
    )
    _diagnose(_make_modeler(llm, tmp_path))
    logged = capsys.readouterr().out
    assert "EMPTY/unparseable response" in logged
    assert "finish_reason=length" in logged
