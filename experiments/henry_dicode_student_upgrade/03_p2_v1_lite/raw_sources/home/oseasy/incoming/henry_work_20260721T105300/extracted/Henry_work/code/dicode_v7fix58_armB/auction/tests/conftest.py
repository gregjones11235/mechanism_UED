"""Shared test scaffolding for auction/tests.

v7fix4.2 introduced the deep-wall relay trigger: after K ignored mature decisions with a free
slot and no live relay, the notebook FORCE-OPENS a relay campaign (``relay_forced``). That is
correct in production — it exists precisely so relay start cannot depend on anyone's mood — but
it would inject a surprise kobold campaign into every long-running mature-notebook scenario
written before fix4.2 (ladder retirement, watch, budget, cooldown tests all walk 3+ sessions).
So the trigger tick is a no-op by default in tests; suites that exercise the trigger opt back
in with ``@pytest.mark.relay_trigger`` (see test_siege_fix42_relay_autoconvert.py). The
AUTOCONVERT branch itself stays live everywhere — it only fires on deep-wall proposals, which
only the tests that mean to make them make.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "relay_trigger: run with the v7fix4.2 relay-trigger tick LIVE (it is a no-op otherwise)",
    )


@pytest.fixture(autouse=True)
def _disarm_relay_trigger(request, monkeypatch):
    if request.node.get_closest_marker("relay_trigger"):
        yield
        return
    from auction import siege_notebook

    monkeypatch.setattr(
        siege_notebook.SiegeNotebook,
        "_relay_trigger_tick",
        lambda self, *a, **k: None,
    )
    yield
