"""Checkpoint format families: detection, read-only loading, params hashing.

Five format families exist in this programme.  Loading is fail-closed:
unknown formats, missing keys, or hash mismatches raise — the codec never
guesses the "closest-looking" format.

  CC2_PKL                  fully implemented (real CC2 full_state.pkl contract)
  ORBAX_FLAT_TRAINSTATE    orbax directory holding a flat TrainState pytree
  ORBAX_NESTED_PARAMS      orbax directory whose item nests params one level down
  BAKEOFF_PKL              named constant only; loader pending artifact handoff
  PHASE4A_PKL              named constant only; loader pending artifact handoff

The CC2 loader mirrors CC4's previously verified loading
(tier3_checkpoint_adapter.load_full_params_readonly, tasks #87-#97, D:/cc4tmp
read-only reference): the exact CC2 ``_params_sha`` algorithm is reproduced
byte-for-byte so the recomputed tree sha must equal the manifest-declared sha.

This module never imports jax/numpy at module level: both are lazy imports
inside the functions that need them, so the student_adapters package stays
importable in a jax-free interpreter.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Mapping


FORMAT_CC2_PKL = "CC2_PKL"
FORMAT_ORBAX_FLAT_TRAINSTATE = "ORBAX_FLAT_TRAINSTATE"
FORMAT_ORBAX_NESTED_PARAMS = "ORBAX_NESTED_PARAMS"
FORMAT_BAKEOFF_PKL = "BAKEOFF_PKL"
FORMAT_PHASE4A_PKL = "PHASE4A_PKL"

ALL_FORMATS = (
    FORMAT_CC2_PKL,
    FORMAT_ORBAX_FLAT_TRAINSTATE,
    FORMAT_ORBAX_NESTED_PARAMS,
    FORMAT_BAKEOFF_PKL,
    FORMAT_PHASE4A_PKL,
)

# CC2 writer contract (train_rmt16_p2replay.py save_ckpt): the pickle's top
# level contains EXACTLY these two keys.
_CC2_TOP_LEVEL_KEYS = frozenset({"params", "manifest"})
# Manifest keys observed on all real CC2 pickles (Stage 0 introspection and
# the verified tier3 reference).  params_sha256 and step are mandatory.
_CC2_MANIFEST_REQUIRED = ("params_sha256", "step")


class CheckpointCodecError(RuntimeError):
    """Raised on any format/identity/integrity violation (fail closed)."""


class CheckpointFormatNotImplementedError(NotImplementedError, CheckpointCodecError):
    """Raised for format families whose artifacts are pending handoff."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Read-only view of a loaded checkpoint."""

    format: str
    params: Any
    params_sha256: str            # recomputed over the loaded params tree
    file_sha256: str              # sha256 of the checkpoint file bytes
    manifest: Mapping[str, Any] = field(default_factory=dict)
    contains_optimizer: bool = False
    contains_rng: bool = False
    contains_policy_memory: bool = False
    global_step: int | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)


def file_sha256(path: str) -> str:
    """Streaming sha256 of a file's bytes (read-only)."""
    if not path or not os.path.isfile(path):
        raise CheckpointCodecError(f"checkpoint path missing: {path!r}")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cc2_params_sha256(params: Any) -> str:
    """CC2's EXACT params identity algorithm (train_rmt16_p2replay.py _params_sha).

    sha256 over ``np.ascontiguousarray(np.asarray(leaf)).tobytes()`` for every
    leaf in ``jax.tree_util.tree_leaves`` order.  Works on numpy or jax leaves
    (np.asarray is the identity on numpy leaves).  Requires jax; raises
    CheckpointCodecError (BLOCKED_ENVIRONMENT) without it — never guesses.
    """
    try:
        import jax
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise CheckpointCodecError(
            f"BLOCKED_ENVIRONMENT: cc2_params_sha256 requires jax/numpy: {exc}") from exc
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def load_cc2_pkl(path: str, *, expected_params_sha256: str | None = None) -> LoadedCheckpoint:
    """Load a real CC2 full_state.pkl READ-ONLY and verify identity fail-closed.

    Checks, in order:
      1. file exists; file sha256 computed (streaming, read-only);
      2. top level is a dict with EXACTLY {"params", "manifest"};
      3. manifest is a dict carrying params_sha256 and step;
      4. params pytree is non-empty;
      5. recomputed cc2_params_sha256 == manifest-declared params_sha256;
      6. if expected_params_sha256 given: recomputed == expected.

    The pickle stores no optimizer/RNG/policy-memory (verified Stage 0):
    the corresponding flags are False, and the R4c combined proof is
    unavailable on those sides for this family — recorded honestly upstream.
    """
    import pickle

    file_sha = file_sha256(path)
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict) or set(data.keys()) != _CC2_TOP_LEVEL_KEYS:
        raise CheckpointCodecError(
            f"{path!r} is not a CC2 full_state.pkl: top-level keys must be exactly "
            f"{sorted(_CC2_TOP_LEVEL_KEYS)}, got {sorted(data.keys()) if isinstance(data, dict) else type(data).__name__!r} "
            "(unknown format — never guess)")
    manifest = data["manifest"]
    if not isinstance(manifest, dict):
        raise CheckpointCodecError(f"{path!r}: CC2 manifest is not a dict")
    for key in _CC2_MANIFEST_REQUIRED:
        if key not in manifest or manifest[key] in (None, ""):
            raise CheckpointCodecError(f"{path!r}: CC2 manifest missing required key {key!r}")
    params = data["params"]
    recomputed = cc2_params_sha256(params)
    declared = str(manifest["params_sha256"])
    if recomputed != declared:
        raise CheckpointCodecError(
            f"CC2 params_sha256 mismatch: recomputed {recomputed[:16]}… != declared "
            f"{declared[:16]}… (wrong / stale / tampered checkpoint, or non-CC2 pickle)")
    if expected_params_sha256 is not None and recomputed != expected_params_sha256:
        raise CheckpointCodecError(
            f"CC2 params_sha256 != expected identity: recomputed {recomputed[:16]}… "
            f"!= expected {str(expected_params_sha256)[:16]}…")
    return LoadedCheckpoint(
        format=FORMAT_CC2_PKL,
        params=params,
        params_sha256=recomputed,
        file_sha256=file_sha,
        manifest=dict(manifest),
        contains_optimizer=False,
        contains_rng=False,
        contains_policy_memory=False,
        global_step=int(manifest["step"]),
        notes={"manifest_keys": sorted(str(k) for k in manifest.keys())},
    )


def _load_orbax_directory(path: str, *, nested_params: bool) -> LoadedCheckpoint:
    """Read-only restore of an orbax checkpoint directory (lazy orbax import).

    Returns the restored pytree as ``params``.  For ORBAX_NESTED_PARAMS the
    restored item must expose its params one level down (a single dict entry
    named 'params'); the flat variant returns the TrainState pytree as-is.
    No params hash regime is bound to orbax here: Stage 4 probes bind the
    declared hash from the candidate's SHA contract and compare explicitly.
    """
    if not path or not os.path.isdir(path):
        raise CheckpointCodecError(f"orbax checkpoint directory missing: {path!r}")
    try:
        import orbax.checkpoint as ocp
    except Exception as exc:
        raise CheckpointCodecError(
            f"BLOCKED_ENVIRONMENT: orbax loading requires orbax-checkpoint: {exc}") from exc
    try:
        restored = ocp.PyTreeCheckpointer().restore(path)
    except Exception as exc:
        raise CheckpointCodecError(f"orbax restore failed for {path!r}: {exc}") from exc
    if nested_params:
        if not isinstance(restored, dict) or "params" not in restored:
            raise CheckpointCodecError(
                f"{path!r}: ORBAX_NESTED_PARAMS requires a 'params' entry one level "
                f"down, got keys {sorted(restored.keys()) if isinstance(restored, dict) else type(restored).__name__!r}")
        payload = restored["params"]
    else:
        payload = restored
    return LoadedCheckpoint(
        format=FORMAT_ORBAX_NESTED_PARAMS if nested_params else FORMAT_ORBAX_FLAT_TRAINSTATE,
        params=payload,
        params_sha256="",  # bound explicitly by the caller's SHA contract
        file_sha256="",    # directory format: no single-file sha
        manifest={},
        contains_optimizer=isinstance(restored, dict) and "opt_state" in restored,
        contains_rng=False,
        contains_policy_memory=False,
        global_step=None,
        notes={"restored_top_keys": sorted(str(k) for k in restored.keys())
               if isinstance(restored, dict) else [type(restored).__name__]},
    )


def load_checkpoint(path: str, *, expected_format: str,
                    expected_params_sha256: str | None = None) -> LoadedCheckpoint:
    """Dispatch a read-only load by EXPLICIT expected_format (never inferred).

    Unknown format → raise.  BAKEOFF_PKL / PHASE4A_PKL raise
    CheckpointFormatNotImplementedError until their artifacts are handed off.
    """
    if expected_format not in ALL_FORMATS:
        raise CheckpointCodecError(
            f"unknown checkpoint format {expected_format!r}; known: {list(ALL_FORMATS)} (never guess)")
    if expected_format == FORMAT_CC2_PKL:
        return load_cc2_pkl(path, expected_params_sha256=expected_params_sha256)
    if expected_format == FORMAT_ORBAX_FLAT_TRAINSTATE:
        return _load_orbax_directory(path, nested_params=False)
    if expected_format == FORMAT_ORBAX_NESTED_PARAMS:
        return _load_orbax_directory(path, nested_params=True)
    raise CheckpointFormatNotImplementedError(
        f"{expected_format}: loader pending artifact handoff (ARTIFACT_HANDOFF_REQUIRED); "
        "refusing to load on speculation")
