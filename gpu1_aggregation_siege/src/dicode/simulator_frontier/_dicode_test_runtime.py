"""TEST_ONLY / SYNTHETIC DiCode training runtime.  NOT_REAL_EXECUTION.

This module exists ONLY so the CanonicalDiCodeOneUpdateRuntime contract can
be exercised by the dedicated tests.  It is never referenced by a production
entry point: the trusted-signer entrypoint allowlist rejects it for
production bundles.  Do not import it outside tests, and never use it to run
real training.
"""

from __future__ import annotations

from typing import Any, Mapping

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def synthetic_run_session_training(plan, adapter, params, run_state, budget):
    """TEST_ONLY: returns a receipt with a bit-different params tree."""
    import numpy as np

    def bump(leaf):
        if isinstance(leaf, np.ndarray) and leaf.size > 0:
            return (leaf + np.asarray(0.001, dtype=leaf.dtype)).astype(leaf.dtype)
        return leaf

    new_params = {"w": bump(params["w"]), "b": bump(params["b"])}
    return {"params": new_params, "receipt_kind": "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT"}


def synthetic_run_training_session(*_args, **_kwargs):
    """TEST_ONLY placeholder for DiCode's inner training-session chain."""
    raise NotImplementedError(
        "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT: the TEST_ONLY runtime is "
        "never invoked for real training")


def synthetic_env_factory(*_args, **_kwargs):
    """TEST_ONLY placeholder environment factory."""
    raise NotImplementedError("SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT")


def synthetic_taskparam_apply(params_env, taskparams):
    """TEST_ONLY taskparam application placeholder (identity)."""
    return params_env
