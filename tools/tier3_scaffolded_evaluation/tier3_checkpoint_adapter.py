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

Two SHA regimes (both seed-free and deterministic):
  * REAL CC2 checkpoints: ``cc2_params_sha256`` — CC2's EXACT ``_params_sha``
    (sha256 over ``np.ascontiguousarray(np.asarray(leaf)).tobytes()`` in
    jax.tree_util.tree_leaves order). full_state.pkl carries ``d["params"]`` as a
    DIRECT pytree (numpy leaves) plus ``manifest["params_sha256"]`` (declared).
    ``load_full_params_readonly`` recomputes and binds it (NEG21) and records the
    checkpoint FILE SHA.
  * Synthetic records (self-test only): ``params_sha256`` via the V3 canonical encoder.
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


def cc2_params_sha256(params) -> str:
    """EXACT CC2 params identity (train_rmt16_p2replay.py ``_params_sha``, lines
    119-123), byte-for-byte the same algorithm CC2 uses to stamp
    ``manifest["params_sha256"]`` into full_state.pkl::

        h = hashlib.sha256()
        for v in jax.tree_util.tree_leaves(params):
            h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
        return h.hexdigest()

    Leaf order = jax tree_leaves order; leaves may be numpy OR jax arrays (CC2 saves
    ``tree_map(np.asarray, params)``; np.asarray is the identity on numpy leaves, so
    the recomputed SHA equals CC2's declared SHA on the loaded pytree). Requires JAX
    (tree_leaves); FAILS CLOSED otherwise.
    """
    require(ser.have_jax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): cc2_params_sha256 requires JAX "
            "(jax.tree_util.tree_leaves); available=%s" % ser.have_jax())
    import hashlib
    import numpy as np
    import jax
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


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
    """Load a CC2 ``full_state.pkl`` checkpoint READ-ONLY (REAL CC2 format).

    CC2 writer contract (train_rmt16_p2replay.py ``save_ckpt``, lines 940-977)::

        pickle.dump({"params": _to_np(params),          # DIRECT pytree, numpy leaves
                     "manifest": {"params_sha256": p_sha, "step": step, "arm": ARM,
                                  "carry_mode": args.carry_mode,
                                  "replay_mode": REPLAY_MODE,
                                  "gpu_uuid": args.gpu_uuid, "seed": args.seed,
                                  "config": {k: v for k, v in vars(cfg).items()},
                                  "phase4a_v2": _phase4a_v2_manifest_fields(),
                                  "tag": tag}}, f, protocol=4)

    i.e. ``d["params"]`` is the params pytree ITSELF (inner/apply convention, numpy
    leaves via ``tree_map(np.asarray, ...)``) — NOT a ``(leaves, treedef)`` pair.
    ``params = d["params"]`` (converted leaf-wise to jnp for apply).

    NOTE (verified on all 26 real checkpoints): ``manifest["config"] == {}`` on every
    real pickle — Cfg is a class-ATTRIBUTES config class, so ``vars(Cfg())`` is empty
    BY DESIGN. The network hyperparameters are frozen in the driver SOURCE (see the
    policy adapter's ``load_cfg_from_driver_source``), NOT in the pickle.

    The recomputed params SHA uses CC2's EXACT algorithm (``cc2_params_sha256``) and
    MUST equal ``manifest["params_sha256"]`` (NEG21 — fail closed on mismatch).

    Returns ``(params, recomputed_params_sha256, manifest, file_sha256)``. Nothing is
    ever written, trained, or mutated here. Requires JAX (BLOCKED_ENVIRONMENT else).
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
    require(isinstance(d, dict) and "params" in d and "manifest" in d,
            "FAIL CLOSED: %r is not a CC2 full_state.pkl (needs 'params' AND 'manifest')"
            % path)
    manifest = d["manifest"]
    require(isinstance(manifest, dict) and manifest.get("params_sha256"),
            "FAIL CLOSED: %r manifest missing declared params_sha256" % path)
    params = d["params"]                                    # DIRECT pytree (CC2 format)
    require(jax.tree_util.tree_leaves(params),
            "FAIL CLOSED: %r 'params' has no leaves (empty params pytree)" % path)
    params = jax.tree_util.tree_map(jnp.asarray, params)    # numpy leaves -> jnp for apply
    recomputed = cc2_params_sha256(params)
    require(recomputed == manifest["params_sha256"],
            "FAIL CLOSED (NEG21): CC2 params_sha256 mismatch — recomputed %s != declared %s "
            "(wrong / stale / tampered checkpoint, or non-CC2 pickle format)"
            % (recomputed[:16], str(manifest["params_sha256"])[:16]))
    return params, recomputed, manifest, file_sha


def make_cc2_checkpoint_record(params, manifest, file_sha256, observation_shape,
                               action_space_id, checkpoint_ref="<cc2_full_state.pkl>",
                               driver_source_sha256=None):
    """READ-ONLY identity record for a REAL CC2 checkpoint. params_sha256 uses the
    CC2 algorithm (== manifest declaration, verified at load); NEVER trainable.

    Provenance keys mirror the REAL save_ckpt manifest: ``replay_mode`` (falls back
    to the legacy ``replay`` key), ``run_class`` from ``phase4a_v2`` (e.g.
    long_run_98304), and ``manifest_config_empty`` records the observed config={}
    (Cfg is class-attributes — hyperparameters come from the SHA-bound driver source,
    bound separately via ``driver_source_sha256``)."""
    p4 = manifest.get("phase4a_v2")
    run_class = p4.get("run_class") if isinstance(p4, dict) else None
    return {
        "schema": SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "checkpoint_ref": checkpoint_ref,
        "trained_by": manifest.get("arm", "CC2_TRAINING_RUN"),
        "params_sha256": cc2_params_sha256(params),
        "declared_params_sha256": manifest.get("params_sha256"),
        "checkpoint_file_sha256": file_sha256,
        "checkpoint_step": manifest.get("step"),
        "carry_mode": manifest.get("carry_mode"),
        "replay_mode": manifest.get("replay_mode", manifest.get("replay")),
        "run_class": run_class,
        "driver_source_sha256": driver_source_sha256,
        "manifest_config_empty": not bool(manifest.get("config")),
        "observation_shape": list(observation_shape),
        "action_space_id": action_space_id,
        "observation_schema": "canonical_craftax_symbolic",
        "trainable": False,
        "writable": False,
    }


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

    # REAL CC2 full_state.pkl format (DIRECT pytree + exact _params_sha); JAX host only.
    if ser.have_jax():
        import tempfile
        import pickle
        import numpy as np
        cc2_params = {"encoder": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
                                  "bias": np.zeros(3, np.float32)},
                      "head": np.zeros(4, np.float32)}
        sha_cc2 = cc2_params_sha256(cc2_params)
        with tempfile.TemporaryDirectory() as td:
            cp = os.path.join(td, "full_state.pkl")
            with open(cp, "wb") as fh:
                pickle.dump({"params": cc2_params,
                             "manifest": {"params_sha256": sha_cc2, "step": 4096,
                                          "arm": "RMT16-Selftest", "carry_mode": "persistent",
                                          "replay_mode": "original_vtrace", "seed": 42,
                                          "phase4a_v2": {"run_class": "selftest",
                                                         "segment_len": 128},
                                          "config": {}}}, fh, protocol=4)
            lp, lsha, lman, fsha = load_full_params_readonly(cp)
            check("cc2_format_roundtrip",
                  lsha == sha_cc2 and lman["step"] == 4096
                  and lman["carry_mode"] == "persistent" and len(fsha) == 64)
            rec = make_cc2_checkpoint_record(lp, lman, fsha, (8335,),
                                             "canonical_craftax_action_set",
                                             driver_source_sha256="1" * 64)
            check("cc2_record_binding",
                  rec["params_sha256"] == sha_cc2
                  and rec["declared_params_sha256"] == sha_cc2
                  and rec["checkpoint_file_sha256"] == fsha
                  and rec["checkpoint_step"] == 4096
                  and rec["carry_mode"] == "persistent"
                  and rec["replay_mode"] == "original_vtrace"
                  and rec["run_class"] == "selftest"
                  and rec["driver_source_sha256"] == "1" * 64
                  and rec["manifest_config_empty"] is True
                  and rec["trainable"] is False and rec["writable"] is False)
            # NEG21 on the REAL format: tampered declared SHA -> fail closed.
            with open(cp, "wb") as fh:
                pickle.dump({"params": cc2_params,
                             "manifest": {"params_sha256": "0" * 64}}, fh, protocol=4)
            try:
                load_full_params_readonly(cp)
                check("cc2_NEG21_tamper_rejected", False)
            except FailClosed:
                check("cc2_NEG21_tamper_rejected", True)
            # non-CC2 pickle (no manifest) -> fail closed.
            with open(cp, "wb") as fh:
                pickle.dump({"params": cc2_params}, fh, protocol=4)
            try:
                load_full_params_readonly(cp)
                check("cc2_missing_manifest_rejected", False)
            except FailClosed:
                check("cc2_missing_manifest_rejected", True)
    else:
        # Without JAX the loader must FAIL CLOSED (never a fake PASS).
        try:
            load_full_params_readonly(p)
            check("cc2_load_blocked_without_jax", False)
        except FailClosed:
            check("cc2_load_blocked_without_jax", True)

    if problems:
        print("TIER3_CHECKPOINT_ADAPTER_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_CHECKPOINT_ADAPTER_SELF_TEST_PASS (NEG21/22/23 guards live; "
          "CC2 full_state.pkl format bound; env=%s)" % ser.environment_status())
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_checkpoint_adapter.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
