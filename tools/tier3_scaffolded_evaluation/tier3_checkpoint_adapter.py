#!/usr/bin/env python3
"""CC4 Tier3 — checkpoint adapter (READ-ONLY Student checkpoint identity).

CC2 trains Students / writes checkpoints; CC3 consumes CC4's StudentProfile. This
adapter is CC4's read-only identity layer over a Student checkpoint so the evaluator
can bind "which exact params + which exact observation/action interface" an evaluation
used — WITHOUT ever training or mutating the checkpoint.

Guards (negative tests):
  NEG21 checkpoint params SHA mismatch        -> fail closed
  NEG22 checkpoint observation shape mismatch -> fail closed
  NEG23 evaluation tries to update params     -> fail closed (params are immutable here)

The params SHA is SEED-FREE and deterministic (canonical bytes of the params pytree,
reusing the serializer discipline). On this host real JAX checkpoints are absent, so
the adapter is exercised with synthetic params dicts (TESTED_SYNTHETIC); the same code
path hashes a real params pytree on a JAX host via the V3 encoder.
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402
import tier3_state_serializer as ser      # noqa: E402

SCHEMA = "mechanism_UED.tier3_checkpoint_identity/v1"
ADAPTER_VERSION = "tier3_checkpoint_adapter/v1"


class FailClosed(Exception):
    """Hard stop on any checkpoint-identity / integrity violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Params hashing (seed-free, deterministic)
# ---------------------------------------------------------------------------
def _params_bytes(params):
    """Canonical bytes of a params object. Plain JSON-able dicts use canonical JSON;
    anything with arrays/pytree leaves falls back to the V3 canonical encoder."""
    try:
        return ser.canonical_json_bytes(params)
    except ser.FailClosed:
        manifest = []
        return ser.v3mat.encode_node(params, (), manifest)


def params_sha256(params) -> str:
    return ser.sha256_bytes(_params_bytes(params))


# ---------------------------------------------------------------------------
# Checkpoint record
# ---------------------------------------------------------------------------
def make_checkpoint_record(params, observation_shape, action_space_id,
                           checkpoint_ref="<synthetic>", trained_by="CC2_TRAINING_RUN"):
    """Build a READ-ONLY checkpoint identity record. `params` is never stored by
    reference for later mutation; only its hash + interface descriptors are kept."""
    return {
        "schema": SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "checkpoint_ref": checkpoint_ref,
        "trained_by": trained_by,                    # provenance: CC2 trained it
        "params_sha256": params_sha256(params),
        "observation_shape": list(observation_shape),
        "action_space_id": action_space_id,
        "observation_schema": "canonical_craftax_symbolic",
        "trainable": False,                          # evaluation NEVER trains
        "writable": False,                           # evaluation NEVER writes params
    }


# ---------------------------------------------------------------------------
# NEG21 / NEG22 / NEG23 guards
# ---------------------------------------------------------------------------
def assert_params_identity(record: dict, expected_params_sha256: str):
    """NEG21: the checkpoint's params SHA must equal the expected (declared) SHA."""
    require(isinstance(record, dict) and record.get("params_sha256"),
            "FAIL CLOSED (NEG21): checkpoint record missing params_sha256")
    require(record["params_sha256"] == expected_params_sha256,
            "FAIL CLOSED (NEG21): checkpoint params_sha256 %s != expected %s "
            "(wrong / stale / tampered checkpoint)"
            % (record["params_sha256"][:16], expected_params_sha256[:16]))
    return True


def assert_observation_shape(record: dict, expected_shape):
    """NEG22: the checkpoint's observation interface must match the evaluator's."""
    require(list(record.get("observation_shape", [])) == list(expected_shape),
            "FAIL CLOSED (NEG22): checkpoint observation_shape %s != expected %s "
            "(the Student was trained on a different observation interface)"
            % (record.get("observation_shape"), list(expected_shape)))
    return True


def assert_evaluation_does_not_update_params(record_before: dict, record_after: dict):
    """NEG23: the params SHA must be IDENTICAL before and after an evaluation.

    Evaluation is inference-only; any change to params_sha256 means the evaluation
    updated the params, which is forbidden.
    """
    require(record_before.get("params_sha256") == record_after.get("params_sha256"),
            "FAIL CLOSED (NEG23): params_sha256 changed during evaluation "
            "(before=%s after=%s); evaluation must be inference-only and never update "
            "Student params." % (str(record_before.get("params_sha256"))[:16],
                                 str(record_after.get("params_sha256"))[:16]))
    require(record_after.get("trainable") is False and record_after.get("writable") is False,
            "FAIL CLOSED (NEG23): checkpoint record must remain trainable=False/writable=False")
    return True


def load_checkpoint_readonly(path: str):
    """Read a checkpoint file's bytes (READ-ONLY) and return (sha256, nbytes).

    Opens strictly 'rb'; nothing is written. The read-only SHA identity is exercisable
    on any file; unflattening a real params pytree additionally needs JAX (below).
    """
    require(path and os.path.isfile(path),
            "FAIL CLOSED: checkpoint path missing (%r)" % path)
    h = ser.v3mat.sha256_file(path)          # reuse V3 streaming file hash
    nbytes = os.path.getsize(path)
    return h, nbytes


def load_full_params_readonly(path: str):
    """Load a CC2 ``full_state.pkl`` checkpoint READ-ONLY and unflatten its params.

    CC2 interface contract (eval_phase2_unified.py): the pickle carries
    ``d["params"] = (leaves, treedef)`` and ``d["manifest"]["params_sha256"]`` (the
    DECLARED identity). Params are rebuilt with
    ``jax.tree_util.tree_unflatten(treedef, [jnp.asarray(l) for l in leaves])``.

    Returns ``(params, recomputed_params_sha256, manifest, file_sha256)``. The caller
    binds identity with assert_params_identity against manifest["params_sha256"]
    (NEG21). Nothing is ever written, trained, or mutated here. Requires JAX
    (BLOCKED_ENVIRONMENT otherwise).
    """
    require(ser.have_jax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): loading a real params pytree requires JAX "
            "(available=%s)" % ser.have_jax())
    import pickle
    import jax
    import jax.numpy as jnp
    file_sha, _nbytes = load_checkpoint_readonly(path)     # read-only 'rb' + streaming SHA
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    require(isinstance(d, dict) and "params" in d,
            "FAIL CLOSED: %r is not a CC2 full_state.pkl (no 'params' entry)" % path)
    leaves, treedef = d["params"]
    params = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(l) for l in leaves])
    manifest = d.get("manifest") if isinstance(d.get("manifest"), dict) else {}
    return params, params_sha256(params), manifest, file_sha


# ---------------------------------------------------------------------------
# Self-test (synthetic; runs on this host).
# ---------------------------------------------------------------------------
def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    params = {"layer0": {"w": [[0.1, 0.2], [0.3, 0.4]], "b": [0.0, 0.0]}, "n": 4}
    rec = make_checkpoint_record(params, observation_shape=(67, 7, 7),
                                 action_space_id="canonical_craftax_action_set")
    check("record_readonly_flags", rec["trainable"] is False and rec["writable"] is False)
    check("params_sha_deterministic", params_sha256(params) == rec["params_sha256"])

    # NEG21: params SHA mismatch rejected; exact match accepted.
    check("NEG21_exact_match_ok",
          assert_params_identity(rec, rec["params_sha256"]) is True)
    try:
        assert_params_identity(rec, "0" * 64)
        check("NEG21_sha_mismatch_rejected", False)
    except FailClosed:
        check("NEG21_sha_mismatch_rejected", True)

    # NEG22: observation shape mismatch rejected; match accepted.
    check("NEG22_shape_match_ok", assert_observation_shape(rec, (67, 7, 7)) is True)
    try:
        assert_observation_shape(rec, (68, 7, 7))
        check("NEG22_shape_mismatch_rejected", False)
    except FailClosed:
        check("NEG22_shape_mismatch_rejected", True)

    # NEG23: identical params after eval accepted; changed params rejected.
    rec_after_same = dict(rec)
    check("NEG23_unchanged_ok",
          assert_evaluation_does_not_update_params(rec, rec_after_same) is True)
    mutated = dict(rec)
    mutated["params_sha256"] = params_sha256({"layer0": {"w": [[9.9]]}})
    try:
        assert_evaluation_does_not_update_params(rec, mutated)
        check("NEG23_params_update_rejected", False)
    except FailClosed:
        check("NEG23_params_update_rejected", True)
    # a record flagged trainable/writable is also rejected
    flagged = dict(rec)
    flagged["trainable"] = True
    try:
        assert_evaluation_does_not_update_params(rec, flagged)
        check("NEG23_trainable_flag_rejected", False)
    except FailClosed:
        check("NEG23_trainable_flag_rejected", True)

    # read-only file identity works on any file (uses a real source file as bytes).
    p = str(audit.resolve_source_path("world_builder"))
    sha, nbytes = load_checkpoint_readonly(p)
    check("readonly_file_sha", sha == audit.SOURCE_FILES["world_builder"]["sha256"] and nbytes > 0)

    if problems:
        print("TIER3_CHECKPOINT_ADAPTER_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_CHECKPOINT_ADAPTER_SELF_TEST_PASS (NEG21/22/23 guards live)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_checkpoint_adapter.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
