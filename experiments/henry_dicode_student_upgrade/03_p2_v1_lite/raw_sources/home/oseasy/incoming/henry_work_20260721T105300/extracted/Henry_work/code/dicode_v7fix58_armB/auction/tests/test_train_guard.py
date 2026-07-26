"""v7fix4.8 — offline tests for the session training-health guard (train_guard.py).

Covers the calibration contract from the fast-arm s191 collapse postmortem
(fable_research_reports/v7fix4.8value护栏与必死关准入制方案.md §1.3):

  P0  the four historical SELF-RECOVERING value flare-ups (session v_loss mean 42-86,
      entropy min >= 0.28) must NOT trip — tripping would discard healthy progress;
  P1  the fatal runaway (first-session v_loss mean 13976, entropy min 0.16) must trip
      on its FIRST session, as must nan/inf and entropy collapse;
  P2  the held-out red line is DROP-based: a from-scratch run (legitimately low early
      collect_wood) never trips, a 100 -> 32 crash from a healthy baseline does;
  P3  consecutive-revert bookkeeping.

train_guard.py is pure python, loaded by file path (no jax needed) — spawn_kit precedent.
"""

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_GUARD = os.path.join(_REPO, "src", "dicode", "train_guard.py")

_spec = importlib.util.spec_from_file_location("train_guard", _GUARD)
train_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_guard)


@pytest.fixture(autouse=True)
def _fresh_state():
    train_guard.reset_session_stats()
    train_guard._prev_heldout["collect_wood"] = None
    train_guard._prev_heldout["mean_return"] = None
    train_guard._consecutive_reverts = 0
    yield


def _feed(v_losses, entropies):
    for v, e in zip(v_losses, entropies):
        train_guard.record_update(v, e)


# ---------------------------------------------------------------- P0 no false trips
def test_healthy_session_no_trip():
    _feed([0.4] * 100, [0.8] * 100)
    assert train_guard.session_verdict() is None


def test_healthy_session_start_spike_no_trip():
    # session-start spikes reach ~2.3 v_loss and entropy dips to ~0.28 historically
    _feed([2.3] + [0.4] * 99, [0.28] + [0.8] * 99)
    assert train_guard.session_verdict() is None


def test_self_recovering_flare_worst_case_no_trip():
    # banner s167 flare (fast arm): session v_loss MEAN 86, entropy min 0.28 — must NOT trip
    _feed([86.0] * 100, [0.8] * 99 + [0.28])
    assert train_guard.session_verdict() is None


def test_baseline_flare_no_trip():
    # BASELINE also flares (2/145 blocks, vmean up to 196, entropy 0.54) — flares are a
    # general property of the stack, not a deep-level artifact; must NOT trip.
    _feed([196.0] * 100, [0.54] * 100)
    assert train_guard.session_verdict() is None


def test_empty_session_no_trip():
    assert train_guard.session_verdict() is None


# ---------------------------------------------------------------- P1 real trips
def test_fatal_runaway_trips_on_first_session():
    # banner s190/191 first collapsed block: v_loss mean 13976
    _feed([13976.0] * 100, [0.5] * 100)
    reason = train_guard.session_verdict()
    assert reason is not None and "value_loss" in reason


def test_vloss_threshold_boundary():
    _feed([999.0] * 10, [0.5] * 10)
    assert train_guard.session_verdict() is None
    train_guard.reset_session_stats()
    _feed([1001.0] * 10, [0.5] * 10)
    assert train_guard.session_verdict() is not None


def test_entropy_collapse_trips():
    # fatal block hit entropy 0.16 on its way to 0.001
    _feed([0.5] * 100, [0.5] * 99 + [0.14])
    reason = train_guard.session_verdict()
    assert reason is not None and "entropy" in reason


def test_nan_trips():
    _feed([0.4, float("nan"), 0.4], [0.8, 0.8, 0.8])
    assert train_guard.session_verdict() is not None


def test_reset_clears_stats():
    _feed([13976.0] * 10, [0.5] * 10)
    train_guard.reset_session_stats()
    assert train_guard.session_verdict() is None


# ---------------------------------------------------------------- P2 held-out red line
def test_fresh_run_low_collect_wood_never_trips():
    # from-scratch run: no healthy baseline yet -> early low readings are legitimate
    assert train_guard.heldout_verdict({"skill_collect_wood": 12.0, "mean_return": 1.0}) is None


def test_heldout_crash_from_healthy_baseline_trips():
    train_guard.note_heldout({"skill_collect_wood": 100.0, "mean_return": 46.1})
    reason = train_guard.heldout_verdict({"skill_collect_wood": 32.3, "mean_return": 1.66})
    assert reason is not None and "collect_wood" in reason


def test_heldout_mean_return_crash_trips():
    train_guard.note_heldout({"skill_collect_wood": 55.0, "mean_return": 46.1})
    # collect_wood baseline (55) is below the 60 prev-floor, but mean_return crash still trips
    reason = train_guard.heldout_verdict({"skill_collect_wood": 52.0, "mean_return": -0.9})
    assert reason is not None and "mean_return" in reason


def test_heldout_normal_fluctuation_no_trip():
    train_guard.note_heldout({"skill_collect_wood": 100.0, "mean_return": 46.1})
    assert train_guard.heldout_verdict({"skill_collect_wood": 96.0, "mean_return": 44.0}) is None


def test_heldout_baseline_not_updated_by_reverted_session():
    # the caller only calls note_heldout on ACCEPTED sessions; a crash reading must not
    # become the new baseline (else the next crashed session would compare crash-to-crash)
    train_guard.note_heldout({"skill_collect_wood": 100.0, "mean_return": 46.1})
    assert train_guard.heldout_verdict({"skill_collect_wood": 0.0, "mean_return": -0.9}) is not None
    # baseline unchanged -> still trips next session
    assert train_guard.heldout_verdict({"skill_collect_wood": 0.0, "mean_return": -0.9}) is not None


# ---------------------------------------------------------------- P3 revert bookkeeping
def test_consecutive_reverts_count_and_reset():
    assert train_guard.register_verdict(True) == 1
    assert train_guard.register_verdict(True) == 2
    assert train_guard.register_verdict(False) == 0
    assert train_guard.register_verdict(True) == 1
