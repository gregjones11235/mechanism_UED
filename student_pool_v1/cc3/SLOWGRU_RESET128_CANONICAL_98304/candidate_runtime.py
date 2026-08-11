#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Candidate runtime binding: SLOWGRU_RESET128_CANONICAL_98304.

Thin per-capsule binding over the shared THIN_GTRXL128_SLOWGRU_RUNTIME
(../slowgru_runtime/slowgru_runtime.py). Implements the unified CC3 ABI:

  load_candidate(checkpoint_contract=None) -> handle
  init_memory(batch_size) -> memory_state
  policy_step(observation, memory_state, done_mask, true_done=None)
      -> (action, memory_state_new, extras)
  reset_memory(memory_state, reset_mask) -> memory_state_new
  on_segment_boundary(memory_state) -> (memory_state_new, info)   # RESET128 contract
  candidate_metadata() -> dict

carry_mode=RESET128: on_segment_boundary resets the slow longstate to init at each
128-step segment boundary (fast window memories carry) — NOT unified with the
persistent candidate's boundary behavior.
Identity is fail-closed: load_candidate recomputes file SHA + params SHA against
checkpoint_contract.json and refuses to serve on any mismatch.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RT_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "slowgru_runtime"))
if _RT_DIR not in sys.path:
    sys.path.insert(0, _RT_DIR)

import slowgru_runtime as _sr  # noqa: E402

CANDIDATE_ID = "SLOWGRU_RESET128_CANONICAL_98304"
CARRY_MODE = "RESET128"

with open(os.path.join(_HERE, "checkpoint_contract.json"), encoding="utf-8") as _f:
    _DEFAULT_CONTRACT = json.load(_f)
assert _DEFAULT_CONTRACT["candidate_id"] == CANDIDATE_ID
assert _DEFAULT_CONTRACT["carry_mode"] == CARRY_MODE

_STATE = {"handle": None}


def load_candidate(checkpoint_contract=None):
    handle = _sr.load_candidate(checkpoint_contract or _DEFAULT_CONTRACT)
    if handle["carry_mode"] != CARRY_MODE:
        raise RuntimeError("CARRY_MODE_MISMATCH %s != %s" % (handle["carry_mode"], CARRY_MODE))
    _STATE["handle"] = handle
    return handle


def _h():
    if _STATE["handle"] is None:
        raise RuntimeError("load_candidate() must be called first")
    return _STATE["handle"]


def seed_policy_rng(seed):
    return _sr.seed_policy_rng(_h(), seed)


def init_memory(batch_size):
    return _sr.init_memory(_h(), batch_size)


def policy_step(observation, memory_state, done_mask, true_done=None):
    return _sr.policy_step(_h(), observation, memory_state, done_mask, true_done)


def reset_memory(memory_state, reset_mask):
    return _sr.reset_memory(_h(), memory_state, reset_mask)


def on_segment_boundary(memory_state):
    return _sr.on_segment_boundary(_h(), memory_state)


def candidate_metadata():
    return _sr.candidate_metadata(_h())
