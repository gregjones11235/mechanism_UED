import pytest

from dicode.e3_litesim.diagnostics.accounting import TransitionAccounting


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