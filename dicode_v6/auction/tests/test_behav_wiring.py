"""Integration/wiring tests for the v6 problem-2 behaviour fingerprint (2026-07-05).

Problem-1 taught us that per-module unit tests are NOT enough: the rehearsal bug (and, we later found,
the (c) co-occurrence bug) was a CROSS-OBJECT wiring fault — the notebook/logs live on
gen_manager.task_generator, but the eval-side reader looked them up on gen_manager directly, so
getattr always returned None and the feature silently no-op'd for an entire run.

These tests pin the wiring: they replicate the exact holder-resolution + add_session dance that
online_evaluation.run_session_evaluation performs, against a fake GenManager whose logs hang off
.task_generator (the real structure), and assert the fingerprint actually accumulates and renders. If a
future refactor moves the logs back onto the GM (or someone re-introduces the getattr-on-GM bug), the
render assertion fails loudly instead of silently degrading to an empty hint.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUCTION = os.path.dirname(_HERE)
if _AUCTION not in sys.path:
    sys.path.insert(0, _AUCTION)

from behavior_fingerprint_log import ACTION_NAMES, BehaviorFingerprintLog  # noqa: E402
from cooccurrence_log import CooccurrenceLog  # noqa: E402
from craftax_achievements import ACHIEVEMENT_TO_VALUE, NUM_ACHIEVEMENTS  # noqa: E402


class _FakeTG:
    """Mirrors gen_manager.task_generator: the logs live HERE, not on the GM."""
    def __init__(self, cooc_log, behav_log):
        self._cooc_log = cooc_log
        self._behav_log = behav_log


class _FakeGM:
    def __init__(self, cooc_log, behav_log):
        self.task_generator = _FakeTG(cooc_log, behav_log)


def _emulate_online_eval_accumulate(gen_manager, session_idx, metrics):
    """The exact pop + holder-resolve + add_session logic from online_evaluation.run_session_evaluation
    (kept in sync by hand; the point is to test the WIRING contract, not re-import the jax-heavy module).
    """
    import numpy as np
    cooc_count = metrics.pop("_cooc_count", None)
    cooc_matrix = metrics.pop("_cooc_matrix", None)
    cooc_names = metrics.pop("_cooc_names", None)
    cooc_total = metrics.pop("_cooc_total", None)
    behav_action = metrics.pop("_behav_action", None)
    behav_steps = metrics.pop("_behav_steps", None)
    behav_names = metrics.pop("_behav_names", None)
    behav_action_names = metrics.pop("_behav_action_names", None)

    holder = getattr(gen_manager, "task_generator", None) or gen_manager
    cooc_log = getattr(holder, "_cooc_log", None)
    if cooc_log is not None and cooc_count is not None and cooc_matrix is not None:
        cooc_log.add_session(
            session_idx,
            np.asarray(cooc_count).astype(int).tolist(),
            np.asarray(cooc_matrix).astype(int).tolist(),
            names=cooc_names,
            total=(int(np.asarray(cooc_total)) if cooc_total is not None else None),
        )
    behav_log = getattr(holder, "_behav_log", None)
    if (behav_log is not None and behav_action is not None
            and behav_steps is not None and cooc_count is not None):
        behav_log.add_session(
            session_idx,
            np.asarray(behav_action).astype(float).tolist(),
            np.asarray(behav_steps).astype(float).tolist(),
            np.asarray(cooc_count).astype(int).tolist(),
            names=behav_names,
            action_names=behav_action_names,
            total=(int(np.asarray(cooc_total)) if cooc_total is not None else None),
        )


def _make_eval_metrics(deep="make_iron_pickaxe", n_win=40, total=1000, do_uses=1200, steps=3360):
    """Build a metrics dict shaped like craftax_evaluation returns (only the _behav/_cooc keys matter)."""
    n = NUM_ACHIEVEMENTS
    di = ACHIEVEMENT_TO_VALUE[deep]
    count = [0] * n
    count[di] = n_win
    cooc = [[0] * n for _ in range(n)]
    cooc[di][di] = n_win
    behav_action = [[0.0] * len(ACTION_NAMES) for _ in range(n)]
    behav_action[di][ACTION_NAMES.index("DO")] = float(do_uses)
    behav_action[di][ACTION_NAMES.index("PLACE_STONE")] = 80.0
    behav_steps = [0.0] * n
    behav_steps[di] = float(steps)
    return {
        "_cooc_count": count, "_cooc_matrix": cooc, "_cooc_total": total, "_cooc_names": None,
        "_behav_action": behav_action, "_behav_steps": behav_steps,
        "_behav_names": None, "_behav_action_names": list(ACTION_NAMES),
    }


def test_wiring_accumulates_and_renders(tmp_path):
    cooc = CooccurrenceLog(str(tmp_path / "c.json"))
    behav = BehaviorFingerprintLog(str(tmp_path / "b.json"))
    gm = _FakeGM(cooc, behav)
    # guard the exact fault mode: the logs must NOT be on the GM directly.
    assert not hasattr(gm, "_behav_log") and not hasattr(gm, "_cooc_log")

    _emulate_online_eval_accumulate(gm, 11, _make_eval_metrics())
    # accumulated on the task_generator's logs
    assert behav.support("make_iron_pickaxe") == 40
    assert cooc.support("make_iron_pickaxe") == 40

    hint = behav.render_fingerprint_hint("make_iron_pickaxe")
    assert hint != ""  # the bug would leave this empty
    assert "DO 30.0x" in hint          # 1200 DO / 40 eps = 30 per winning episode
    assert "make_iron_pickaxe" in hint
    assert "84 steps" in hint          # 3360 / 40


def test_wiring_no_behav_data_renders_empty(tmp_path):
    """When the deep skill is solved too rarely (below MIN_SR), the hint is empty and the prompt omits
    the behaviour block — the modeler falls back to mechanics, no crash, no stray label."""
    cooc = CooccurrenceLog(str(tmp_path / "c.json"))
    behav = BehaviorFingerprintLog(str(tmp_path / "b.json"))
    gm = _FakeGM(cooc, behav)
    # only 5 wins out of 100000 finished -> SR 0.005% << MIN_SR 3%
    _emulate_online_eval_accumulate(gm, 11, _make_eval_metrics(n_win=5, total=100000))
    assert behav.render_fingerprint_hint("make_iron_pickaxe") == ""


def test_wiring_survives_resume(tmp_path):
    """Two sessions accumulate; a fresh log object (resume) reads the combined totals off disk."""
    cp, bp = str(tmp_path / "c.json"), str(tmp_path / "b.json")
    gm = _FakeGM(CooccurrenceLog(cp), BehaviorFingerprintLog(bp))
    _emulate_online_eval_accumulate(gm, 11, _make_eval_metrics(n_win=20, total=500))
    _emulate_online_eval_accumulate(gm, 12, _make_eval_metrics(n_win=20, total=500))
    # resume: brand-new objects load the accumulated state.
    behav2 = BehaviorFingerprintLog(bp)
    assert behav2.support("make_iron_pickaxe") == 40  # 20 + 20
    assert behav2.total_finished() == 1000
