"""Integration/wiring tests for the v6fix7 P2 chain-order mining (mirrors test_behav_wiring).

The rehearsal + (c) co-occurrence bugs were both CROSS-OBJECT wiring faults: a perfectly unit-tested
module that the eval side looked up on the wrong holder, silently no-op'ing for a whole run. These
tests pin the P2 wiring contract the same way: replicate the exact pop + holder-resolve +
add_session + frontier-dispatch dance from online_evaluation.run_session_evaluation against a fake
GenManager whose log/notebook hang off .task_generator (the real structure), and assert the chain
data actually accumulates, renders, and delivers the frontier signal into the notebook.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUCTION = os.path.dirname(_HERE)
if _AUCTION not in sys.path:
    sys.path.insert(0, _AUCTION)

from chain_order_log import ChainOrderLog  # noqa: E402
from craftax_achievements import ACHIEVEMENT_TO_VALUE, NUM_ACHIEVEMENTS  # noqa: E402
from siege_notebook import MATURITY_MIN_SNAPSHOTS, SiegeNotebook  # noqa: E402

from auction.tests.test_siege_notebook import _mature_profile  # noqa: E402


class _FakeTG:
    """Mirrors gen_manager.task_generator: the log + notebook live HERE, not on the GM."""
    def __init__(self, chain_log, notebook):
        self._chain_log = chain_log
        self._siege_notebook = notebook


class _FakeGM:
    def __init__(self, chain_log, notebook):
        self.task_generator = _FakeTG(chain_log, notebook)


def _emulate_online_eval_chain(gen_manager, session_idx, metrics):
    """The exact pop + holder-resolve + add_session + frontier-dispatch logic from
    online_evaluation.run_session_evaluation (kept in sync by hand — the point is the WIRING)."""
    import numpy as np
    chain_first_step = metrics.pop("_chain_first_step", None)
    chain_finished = metrics.pop("_chain_finished", None)
    chain_names = metrics.pop("_chain_names", None)

    holder = getattr(gen_manager, "task_generator", None) or gen_manager
    chain_log = getattr(holder, "_chain_log", None)
    if chain_log is not None and chain_first_step is not None and chain_finished is not None:
        notebook = getattr(holder, "_siege_notebook", None)
        targets = notebook.chain_targets() if notebook is not None else {}
        chain_log.add_session(
            session_idx,
            np.asarray(chain_first_step).astype(int).tolist(),
            np.asarray(chain_finished).astype(bool).tolist(),
            names=chain_names,
            chain_targets=targets,
        )
        advanced = [t for t in targets if chain_log.frontier_advanced(t)]
        if notebook is not None:
            for t in advanced:
                notebook.note_chain_progress(t)


def _metrics(episodes):
    """metrics dict shaped like craftax_evaluation returns: canonical-order first-step rows."""
    rows = []
    for ep in episodes:
        row = [-1] * NUM_ACHIEVEMENTS
        for name, step in ep.items():
            row[ACHIEVEMENT_TO_VALUE[name]] = step
        rows.append(row)
    return {
        "_chain_first_step": rows,
        "_chain_finished": [True] * len(rows),
        "_chain_names": None,
    }


def _gm_with_focus(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    nb.apply_llm_update(
        1, _mature_profile({"defeat_troll": 3.0}),
        {"foci": [{"skill": "defeat_troll", "prereq_tree": [
            {"skill": "collect_iron", "role": "gear"},
            {"skill": "make_iron_sword", "role": "gear"},
        ]}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == ["defeat_troll"]
    log = ChainOrderLog(str(tmp_path / "chain.json"))
    return _FakeGM(log, nb), log, nb


def test_chain_wiring_accumulates_against_focus(tmp_path):
    gm, log, nb = _gm_with_focus(tmp_path)
    assert not hasattr(gm, "_chain_log")  # guard the exact getattr-on-GM fault mode
    # 100 failures dying after collect_iron + a few wins
    eps = [{"collect_iron": 2}] * 100 + [
        {"collect_iron": 2, "make_iron_sword": 6, "defeat_troll": 9}
    ] * 10
    _emulate_online_eval_chain(gm, 11, _metrics(eps))
    entry = log.latest_fail_summary("defeat_troll")
    assert entry is not None and entry["n_fail"] == 100  # mined against the ACTIVE focus's chain
    assert log.support("defeat_troll") == 10
    assert log.render_chain_hint("defeat_troll") != ""


def test_chain_wiring_delivers_frontier_signal_to_notebook(tmp_path):
    gm, log, nb = _gm_with_focus(tmp_path)
    _emulate_online_eval_chain(gm, 11, _metrics([{"collect_iron": 2}] * 100))
    foc = nb.foci()[0]
    assert not foc.get("chain_frontier_advanced")  # first session: nothing to compare yet
    # next eval: failures die a full link deeper -> frontier advance -> flag lands on the focus
    _emulate_online_eval_chain(
        gm, 12, _metrics([{"collect_iron": 2, "make_iron_sword": 6}] * 100)
    )
    assert nb.foci()[0].get("chain_frontier_advanced") is True


def test_chain_wiring_noop_without_log(tmp_path):
    """siege off: _chain_log is None -> pops happen, nothing accumulates, no file appears."""
    class _BareTG:
        _chain_log = None
    class _BareGM:
        task_generator = _BareTG()
    metrics = _metrics([{"collect_wood": 0}] * 4)
    _emulate_online_eval_chain(_BareGM(), 11, metrics)
    assert "_chain_first_step" not in metrics  # popped even when inactive (wandb stays clean)
    assert not os.path.exists(os.path.join(str(tmp_path), "chain_order_counts.json"))
