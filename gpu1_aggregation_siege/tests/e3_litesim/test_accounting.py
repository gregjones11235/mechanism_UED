import pytest

from dicode.e3_litesim.diagnostics.accounting import (CATEGORIES,
                                                      TransitionAccounting)


def test_totals_and_hash():
    acc = TransitionAccounting()
    acc.record("probe", 100)
    acc.record("training", 50)
    acc.record("original", 25)
    assert acc.total == 175
    final = acc.finalize(student_version="v")
    assert final["total_simulator_transitions"] == 175
    assert final["accounting_hash"]


def test_unknown_category_rejected():
    acc = TransitionAccounting()
    with pytest.raises(KeyError):
        acc.record("bogus", 1)


def test_state_bank_categories_present():
    assert "state_bank_build" in CATEGORIES
    assert "state_bank_validation" in CATEGORIES


def test_conservation_holds_for_all_categories():
    acc = TransitionAccounting()
    for cat in CATEGORIES:
        acc.record(cat, 2)
    final = acc.finalize(student_version="v")
    assert final["conservation_ok"]
    assert final["total_simulator_transitions"] == 2 * len(CATEGORIES)