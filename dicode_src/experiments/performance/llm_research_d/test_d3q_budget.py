"""Strong counterexample tests for the D3Q shared POST budget state machine.

Run from the module directory:  python -m pytest test_d3q_budget.py
"""
import json
import os
import tempfile

import pytest

from d3q_budget import BudgetExceededError, D3QLedger, ProviderBudget, SlotBudget


def _event(**overrides):
    base = {
        "ts_utc": "2026-08-15T01:00:00Z",
        "slot_id": "slot_r1_small_p00",
        "model": "qwen2.5-coder:14b",
        "provider": "ollama",
        "kind": "initial",
        "attempt_index": 1,
    }
    base.update(overrides)
    return base


def test_slot_fourth_post_rejected_even_if_provider_has_budget():
    slot = SlotBudget("slot_r1_small_p00")
    provider = ProviderBudget("ollama")  # plenty of provider budget left
    assert slot.reserve("initial") == 1
    assert slot.reserve("transport_retry") == 2
    assert slot.reserve("semantic_repair") == 3
    assert provider.post_count == 0
    with pytest.raises(BudgetExceededError) as exc:
        slot.reserve("initial")
    assert exc.value.slot_id == "slot_r1_small_p00"
    assert "slot=" in str(exc.value)
    assert slot.post_count == 3


def test_provider_109th_post_rejected():
    provider = ProviderBudget("deepseek_official")
    for _ in range(108):
        provider.reserve("initial")
    assert provider.post_count == 108
    with pytest.raises(BudgetExceededError) as exc:
        provider.reserve("initial")
    assert exc.value.provider == "deepseek_official"
    assert exc.value.limit == 108
    assert provider.post_count == 108


def test_kind_mixing_consumes_same_slot_budget():
    slot = SlotBudget("slot_r2_large_p05")
    for kind in ("initial", "transport_retry", "semantic_repair"):
        slot.reserve(kind)
    assert slot.post_count == 3
    with pytest.raises(BudgetExceededError):
        slot.reserve("transport_retry")
    with pytest.raises(BudgetExceededError):
        slot.reserve("semantic_repair")


def test_invalid_kind_rejected_and_not_recorded():
    slot = SlotBudget("slot_r1_small_p00")
    with pytest.raises(ValueError):
        slot.reserve("bogus")
    assert slot.post_count == 0
    with pytest.raises(ValueError):
        ProviderBudget("ollama").check("bogus")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        ledger = D3QLedger(path)
        with pytest.raises(ValueError):
            ledger.reserve(**_event(kind="bogus"))
        # A rejected reserve must not create or write the ledger file.
        assert not os.path.exists(path)


def test_ledger_records_all_required_fields_with_indices():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        ledger = D3QLedger(path)
        event = ledger.reserve(**_event())
        assert event["post_index_in_slot"] == 1
        assert event["post_index_for_provider"] == 1
        with open(path, encoding="utf-8") as handle:
            line = json.loads(handle.read())
        assert line == event
        for field in (
            "ts_utc", "slot_id", "model", "provider", "kind", "attempt_index",
            "post_index_in_slot", "post_index_for_provider",
        ):
            assert field in line


def test_ledger_resume_rebuilds_budget_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        ledger = D3QLedger(path)
        for i, kind in enumerate(("initial", "transport_retry", "semantic_repair")):
            ledger.reserve(**_event(kind=kind, attempt_index=i + 1))
        ledger.reserve(
            **_event(slot_id="slot_r3_small_p11", provider="deepseek_official")
        )
        resumed = D3QLedger(path)
        assert resumed.slot_post_count("slot_r1_small_p00") == 3
        assert resumed.slot_post_count("slot_r3_small_p11") == 1
        assert resumed.provider_post_count("ollama") == 3
        assert resumed.provider_post_count("deepseek_official") == 1
        # The exhausted slot is still rejected after resume.
        with pytest.raises(BudgetExceededError) as exc:
            resumed.reserve(**_event())
        assert exc.value.slot_id == "slot_r1_small_p00"


def test_ledger_provider_109th_rejected_without_record():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        ledger = D3QLedger(path)
        for i in range(108):
            ledger.reserve(
                **_event(slot_id=f"slot_b{i:02d}", provider="deepseek_official", kind="initial")
            )
        assert ledger.provider_post_count("deepseek_official") == 108
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        with pytest.raises(BudgetExceededError) as exc:
            ledger.reserve(
                **_event(slot_id="slot_fresh", provider="deepseek_official", kind="initial")
            )
        assert exc.value.provider == "deepseek_official"
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == before


def test_rejected_slot_reserve_writes_no_ledger_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        ledger = D3QLedger(path)
        for i, kind in enumerate(("initial", "transport_retry", "semantic_repair")):
            ledger.reserve(**_event(kind=kind, attempt_index=i + 1))
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        with pytest.raises(BudgetExceededError):
            ledger.reserve(**_event())
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == before


def test_ledger_corrupt_line_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(_event()) + "\n")
            handle.write("not valid json\n")
        with pytest.raises(ValueError):
            D3QLedger(path)


def test_ledger_exceeding_slot_limit_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for i in range(4):
                event = _event(kind="initial", attempt_index=i + 1)
                event["post_index_in_slot"] = i + 1
                event["post_index_for_provider"] = i + 1
                handle.write(json.dumps(event) + "\n")
        with pytest.raises(BudgetExceededError):
            D3QLedger(path)