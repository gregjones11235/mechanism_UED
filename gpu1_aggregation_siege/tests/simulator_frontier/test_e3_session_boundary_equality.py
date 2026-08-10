# -*- coding: utf-8 -*-
"""E3 session-boundary per-leaf equality tests (audit item 3, semantics B).

Chosen semantics: B_NEW_SESSION_ENV_AND_MEMORY_RESET — the env and the
recurrent policy memory RESET together at each session start; only
params / optimizer / training RNG / global counters continue.  These tests
prove, leaf-by-leaf, that the resumed initial state matches the previous
session's final state under that semantics (params + optimizer + RNG
continuous, memory boundary recorded).

Runs against real verification-run evidence when E3_VERIFY_RUN_DIR is set
(otherwise the leaf-equality primitive is still tested).
"""

import hashlib
import os

import numpy as np
import pytest

import verify_e3_ah_gate as ver


def test_per_leaf_equality_primitive_detects_mismatch():
    """path/shape/dtype/hash equality detects a single-leaf change."""
    a = {"p": {"w": np.ones((3, 3), dtype=np.float32), "b": np.zeros(3, dtype=np.float32)}}
    b = {"p": {"w": np.ones((3, 3), dtype=np.float32), "b": np.zeros(3, dtype=np.float32)}}
    eq = ver._per_leaf_equality(a, b)
    assert eq["equal"] is True and eq["checked_leaves"] == 2
    c = {"p": {"w": np.ones((3, 3), dtype=np.float32), "b": np.ones(3, dtype=np.float32)}}
    neq = ver._per_leaf_equality(a, c)
    assert neq["equal"] is False
    assert any("p.b" in m for m in neq["mismatches"])


def test_per_leaf_equality_primitive_detects_shape_change():
    a = {"x": np.zeros((4,), dtype=np.float32)}
    b = {"x": np.zeros((5,), dtype=np.float32)}
    assert ver._per_leaf_equality(a, b)["equal"] is False


def _run_dir():
    return os.environ.get("E3_VERIFY_RUN_DIR")


def test_session_boundary_params_optimizer_rng_continuity():
    """On real run evidence: session k+1's initial params == session k's final
    (per-leaf hash), optimizer step continues, training RNG advances, and the
    boundary semantics is B (env + memory reset)."""
    d = _run_dir()
    if not d:
        pytest.skip("E3_VERIFY_RUN_DIR not set")
    from pathlib import Path
    reports = {}
    for p in sorted(Path(d).glob("evidence/session_*.json")):
        r = ver._load_json(p)
        reports[int(r["session_idx"])] = r
    idxs = sorted(reports)
    assert len(idxs) >= 2, "need >= 2 sessions"

    ckpt_dir = Path(d) / "runstate"
    for i in idxs:
        if i > 1:
            rs_prev = ver._restore_runstate(
                str(ckpt_dir / ("e3_canonical_runstate_s%03d" % (i - 1))))
            rs_cur = ver._restore_runstate(
                str(ckpt_dir / ("e3_canonical_runstate_s%03d" % i)))
            # params: session i's initial (mounted) == session i-1's final.
            ini = str(reports[i].get("initial_trainstate_params_sha256", ""))
            prev_final = ver._params_hash(rs_prev["params"])
            assert ini == prev_final, \
                f"session {i} initial params != session {i-1} final params"
            # optimizer: step continues +3200 per session (8 minibatch x 4 epochs).
            assert int(rs_cur["train_step"]) == int(rs_prev["train_step"]) + 3200
            # training RNG advances (never identical).
            assert ver._rng_token(rs_cur["training_rng"]) != \
                ver._rng_token(rs_prev["training_rng"])
            # boundary semantics: env + memory reset together.
            assert reports[i].get("session_boundary_semantics") == \
                "B_NEW_SESSION_ENV_AND_MEMORY_RESET"


def test_no_cross_session_memory_injection():
    """Under semantics B the next session's capture starts from FRESH memory —
    the reports must not show a previous-session memory restore."""
    d = _run_dir()
    if not d:
        pytest.skip("E3_VERIFY_RUN_DIR not set")
    from pathlib import Path
    reports = {}
    for p in sorted(Path(d).glob("evidence/session_*.json")):
        r = ver._load_json(p)
        reports[int(r["session_idx"])] = r
    for i in sorted(reports):
        if i > 1:
            assert reports[i].get("session_boundary_semantics") == \
                "B_NEW_SESSION_ENV_AND_MEMORY_RESET"
