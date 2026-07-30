#!/usr/bin/env python3
"""CC4 Tier3 — frozen bank artifact serialization (总控六条边界 §一).

The frozen FRONT/BACK bank hashes (21aeb7dc… / c632e30d…) reproduce EXACTLY on the
pinned CPU backend (python 3.11 / jax 0.4.30 / numpy 1.26.4 / flax 0.8.5 /
craftax 1.4.5) and DRIFT on GPU with the identical pinned stack. Therefore the
frozen banks are handled as immutable artifacts:

  SERIALIZE (once, pinned CPU only): mint the canonical bank ONCE from the frozen
    result-blind schedule (n=8, base=10_000, stride=1), serialize the COMPLETE
    EnvState arrays into an immutable artifact directory, and record:
      file_sha256 (states.npz + manifest.json via SHA256SUMS),
      canonical_content_sha256 (== the frozen bank hash, recomputed from the
      ordered per-state payload hashes), bank kind, number of states, state IDs
      and seeds, per-leaf dtype/shape, generator source SHAs, and the pinned
      environment identity. Serializing on a non-CPU backend FAILS CLOSED
      (GPU_REGENERATION_DISABLED).
  LOAD (any eval device, READ-ONLY): the GPU evaluator only loads the artifact.
    After loading, arrays are converted back to the host-NumPy canonical
    representation, canonical_content_sha256 is RECOMPUTED per state (round-trip
    through the frozen V3 seed-free serializer) and per bank; ANY mismatch FAILS
    CLOSED. The bank generator is NEVER called on this path — regenerating the
    frozen banks during a formal evaluation is forbidden (总控 §一.7).

Certificate binding (总控 §一.8): the loader returns bank_source=
FROZEN_SERIALIZED_ARTIFACT, bank_regenerated_on_eval_device=false,
artifact_file_sha256, loaded_content_sha256 and device provenance (mint/load)
for the certificate.

The frozen state VALUES, ORDER, seeds and metric definitions are NEVER modified
here: serialization stores exactly what builder.materialize_start mints under the
frozen schedule, and loading only re-hashes and compares against the frozen
identities in tier3_state_bank_materializer.

Integrity layering (each fails closed):
  1. SHA256SUMS covers manifest.json AND states.npz bytes (byte-tamper gate).
  2. manifest.json records states_npz_sha256 + treedef_sha256 (cross-binding).
  3. check_frozen_manifest_bindings over the embedded PROCESS_A manifest
     (frozen VALUE bindings: bank hash, field manifest, seed schedule, source
     SHAs, predicate code SHA, canonical task source SHA — pure, host-free).
  4. Per-state ROUND-TRIP: tree_unflatten(host numpy leaves) -> V3 payload hash
     must equal the recorded frozen per-state payload hash (a tampered treedef
     blob or leaf array can only produce a pytree whose canonical serialization
     mismatches — fail closed).
  5. Bank-level recomputation: ordered payload hashes must recompute the FROZEN
     bank hash (order-sensitive; NEG06 semantics preserved).

treedef storage: the PyTreeDef is stored as a pickle blob keyed under "treedef",
sha-bound in the manifest (treedef_sha256) and semantically gated by the
per-state round-trip canonical hash (layer 4): a malicious or corrupted treedef
can only ever yield states that FAIL the frozen payload-hash gate. The artifact
is generated and consumed only inside the CC4 evaluation pipeline; the frozen
values themselves never depend on the blob.

Leaf-kind fidelity: the frozen craftax EnvState pytree contains PYTHON scalar
leaves (flax struct dynamic fields — py_int/py_bool/py_float alongside jax
arrays). The V3 canonical encoder tags Python scalars as T_BOOL/T_INT/T_FLOAT
(and np.generic as T_NPSCALAR) but arrays as T_ARRAY — so naively storing every
leaf as a NumPy array would silently change the canonical tag bytes and drift
the round-trip hash. The artifact therefore records a "kind" per leaf in
leaf_schema and the loader restores the ORIGINAL Python type before re-hashing;
the frozen per-state payload hash remains the semantic value gate.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import sys

# Runnable-as-script AND importable-as-package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
import tier3_source_audit as audit        # noqa: E402
import tier3_state_serializer as ser      # noqa: E402
import tier3_scaffold_builder as builder  # noqa: E402
import tier3_state_bank_materializer as mat  # noqa: E402

SCHEMA = "mechanism_UED.tier3_frozen_bank_artifact/v1"
ARTIFACT_VERSION = "tier3_frozen_bank_artifact/v1"

FRONT = mat.FRONT
BACK = mat.BACK


class FailClosed(Exception):
    """Hard stop on any artifact integrity / regeneration-policy violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _lf_sha256_file(path: str) -> str:
    """LF-normalized source SHA (EOL-independent; same canonical form repo-wide)."""
    with open(path, "rb") as fh:
        data = fh.read().decode("utf-8")
    return hashlib.sha256(data.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _module_lf_sha(module) -> str:
    return _lf_sha256_file(os.path.abspath(module.__file__))


def generator_source_shas() -> dict:
    """LF-SHA of every source that participates in bank generation (总控 §一.3
    'generator source SHA') — drift in any of them changes what a regeneration
    would produce, so the artifact binds all of them."""
    import tier3_event_predicates as pred
    import tier3_boundary_schema as bnd
    src = mat.source_shas_for_bank()
    return {
        "world_builder_sha256": src["world_builder_sha256"],
        "canonical_task_sha256": src["canonical_task_sha256"],
        "scaffold_builder_sha256": _module_lf_sha(builder),
        "state_bank_materializer_sha256": _module_lf_sha(mat),
        "state_serializer_sha256": _module_lf_sha(ser),
        "v3_serializer_sha256": _module_lf_sha(ser.v3mat),
        "event_predicates_sha256": _module_lf_sha(pred),
        "boundary_schema_sha256": _module_lf_sha(bnd),
        "source_audit_sha256": _module_lf_sha(audit),
        "frozen_bank_artifacts_sha256": _lf_sha256_file(os.path.abspath(__file__)),
        "predicate_code_sha256": bnd.predicate_code_sha256(),
    }


def current_env_identity() -> dict:
    """The ACTUAL pinned environment identity of THIS process (总控 §一.3)."""
    import platform
    import importlib
    out = {"python_version": platform.python_version(),
           "platform_system": platform.system(),
           "platform_machine": platform.machine()}
    for key, modname in (("jax_version", "jax"), ("jaxlib_version", "jaxlib"),
                         ("numpy_version", "numpy"), ("flax_version", "flax"),
                         ("craftax_version", "craftax")):
        v = None
        try:
            v = getattr(importlib.import_module(modname), "__version__", None)
        except Exception:
            v = None
        if not v:
            try:
                import importlib.metadata as md
                v = md.version(modname)
            except Exception:
                v = None
        require(v, "FAIL CLOSED: cannot determine %s for the artifact environment "
                "identity" % key)
        out[key] = str(v)
    import jax
    out["jax_devices"] = repr(jax.devices())
    out["jax_default_backend"] = str(jax.default_backend())
    return out


def _git_head_or_unavailable() -> str:
    import subprocess
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(audit.repo_root()),
                              capture_output=True, text=True, timeout=60)
        sha = (proc.stdout or "").strip()
        if proc.returncode == 0 and len(sha) == 40 and all(
                c in "0123456789abcdef" for c in sha):
            return sha
    except Exception:
        pass
    return "UNAVAILABLE"


def _leaf_kind(leaf) -> str:
    """The faithful-storage class of a pytree leaf (bool BEFORE int — bool is an
    int subclass; jax Array / numpy ndarray -> 'array'). The V3 canonical encoder
    emits T_BOOL/T_INT/T_FLOAT for Python scalars and T_NPSCALAR for np.generic
    but T_ARRAY for arrays; storing a scalar as an array would drift the frozen
    payload hash, so the artifact records the kind and the loader restores it."""
    if isinstance(leaf, bool):
        return "py_bool"
    if isinstance(leaf, int):
        return "py_int"
    if isinstance(leaf, float):
        return "py_float"
    import numpy as np
    if isinstance(leaf, np.generic):
        return "np_scalar"
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype") \
            and not isinstance(leaf, (str, bytes)):
        return "array"
    raise FailClosed("FAIL CLOSED: unsupported frozen-bank leaf type %r (refusing "
                     "to approximate its canonical encoding)" % type(leaf))


def _leaf_to_array(leaf):
    """Store any supported leaf as a plain NumPy array (the npz carrier)."""
    import numpy as np
    arr = np.asarray(leaf)
    require(arr.dtype.kind in "biufc",
            "FAIL CLOSED: leaf %r of kind %s has non-storable dtype %s (refusing "
            "object/void storage)" % (leaf, _leaf_kind(leaf), arr.dtype))
    return arr


def _leaf_schema_entry(kind: str, arr) -> dict:
    """Per-leaf schema. 'array' / 'np_scalar' bind EXACT dtype + shape; Python
    scalar kinds bind shape=[] and only a dtype CLASS — the np.asarray dtype of a
    Python int is platform-dependent (int32 on Windows, int64 on Linux), so an
    exact dtype would make the artifact needlessly host-specific. The round-trip
    canonical hash is the actual value gate."""
    if kind in ("array", "np_scalar"):
        return {"kind": kind, "dtype": str(arr.dtype),
                "shape": [int(d) for d in arr.shape]}
    return {"kind": kind, "dtype": str(arr.dtype.kind), "shape": []}


def _restore_leaf(kind: str, arr, key: str):
    """Reconstruct the ORIGINAL leaf type from the stored array so the V3
    canonical encoding (and therefore the frozen payload hash) is reproduced
    EXACTLY on load (总控 §一.5: convert back to the host-NumPy canonical
    representation)."""
    if kind == "array":
        return arr
    require([int(d) for d in arr.shape] == [],
            "FAIL CLOSED: artifact leaf %s (kind=%s) must be 0-dim, got shape %s"
            % (key, kind, list(arr.shape)))
    if kind == "np_scalar":
        return arr[()]                       # np.generic with its bound dtype
    if kind == "py_bool":
        require(arr.dtype.kind == "b",
                "FAIL CLOSED: artifact leaf %s kind=py_bool has dtype %s (artifact "
                "tampered)" % (key, arr.dtype))
        return bool(arr[()])
    if kind == "py_int":
        require(arr.dtype.kind in "iu",
                "FAIL CLOSED: artifact leaf %s kind=py_int has dtype %s (artifact "
                "tampered)" % (key, arr.dtype))
        return int(arr[()])
    if kind == "py_float":
        require(arr.dtype.kind == "f",
                "FAIL CLOSED: artifact leaf %s kind=py_float has dtype %s (artifact "
                "tampered)" % (key, arr.dtype))
        return float(arr[()])
    raise FailClosed("FAIL CLOSED: artifact leaf %s has unknown kind %r (older / "
                     "invalid artifact; fail closed)" % (key, kind))


def _assert_dir_fresh(path: str):
    """Artifact output must be fresh (missing or empty) — never overwrite an
    existing artifact (immutability discipline; mirrors the evaluator freshness
    gate, no rm -rf / no auto-rename)."""
    if os.path.exists(path):
        require(os.path.isdir(path),
                "FAIL CLOSED: artifact path %r exists and is not a directory" % path)
        require(not os.listdir(path),
                "FAIL CLOSED: artifact directory %r is not empty — refusing to "
                "overwrite an existing artifact (remove it manually, then re-run)"
                % path)
    return True


# ---------------------------------------------------------------------------
# SERIALIZE — pinned CPU backend only (总控 §一.1–§一.3)
# ---------------------------------------------------------------------------
def serialize_bank(scenario: str, out_root: str) -> dict:
    """Mint ONE frozen bank on the pinned CPU backend and serialize it into an
    immutable artifact directory <out_root>/<scenario>/ (states.npz +
    manifest.json + SHA256SUMS). Fails closed unless the runtime backend is CPU
    (GPU_REGENERATION_DISABLED) and unless the minted bank reproduces the FROZEN
    identity (check_frozen_manifest_bindings + PROCESS_B independence gate)
    BEFORE anything is written."""
    require(scenario in mat.FROZEN_BANK_HASH,
            "FAIL CLOSED: no frozen bank identity for scenario %r (expected front_l2/back_l2)"
            % scenario)
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): artifact serialization requires "
            "JAX+craftax (jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))
    import jax
    import numpy as np
    backend = str(jax.devices()[0].platform) if jax.devices() else "<none>"
    require(backend == "cpu",
            "FAIL CLOSED (GPU_REGENERATION_DISABLED): the frozen banks may ONLY be "
            "regenerated on the pinned CPU backend (JAX_PLATFORMS=cpu); this process "
            "runs on %r — GPU re-mints of the frozen bank hash are KNOWN TO DRIFT"
            % backend)

    bank_dir = os.path.join(out_root, scenario)
    _assert_dir_fresh(bank_dir)

    # 1. Mint under the frozen result-blind schedule + frozen identity gates.
    manifest = mat.process_a_materialize(scenario, mat.FROZEN_BANK_N,
                                         base=mat.FROZEN_SEED_BASE,
                                         stride=mat.FROZEN_SEED_STRIDE)
    binding = mat.check_frozen_manifest_bindings(scenario, manifest)
    mat.process_b_verify(manifest)          # independent re-mint agreement (CPU-legal)

    # 2. Re-mint the ordered states (SAME schedule order) and bind each to its
    #    PROCESS_A payload hash; capture the pytree leaves + treedef.
    seeds = [int(s) for s in manifest["seeds"]]
    states = [builder.materialize_start(scenario, int(s)) for s in seeds]
    treedef0 = jax.tree_util.tree_structure(states[0])
    leaves_per_state = []
    ordered_hashes = []
    leaf_kinds = None
    for i, st in enumerate(states):
        ph, _payload, _v3m = ser.envstate_payload_hash(st)
        require(ph == manifest["entries"][i]["state_payload_hash"],
                "FAIL CLOSED: minted state %d payload hash %s != PROCESS_A manifest "
                "%s (internal inconsistency; nothing written)"
                % (i, ph[:16], str(manifest["entries"][i]["state_payload_hash"])[:16]))
        require(jax.tree_util.tree_structure(st) == treedef0,
                "FAIL CLOSED: pytree structure differs across bank states (index %d)" % i)
        raw = jax.tree_util.tree_leaves(st)
        kinds = [_leaf_kind(l) for l in raw]
        if leaf_kinds is None:
            leaf_kinds = kinds
        require(kinds == leaf_kinds,
                "FAIL CLOSED: leaf kinds differ across bank states (index %d)" % i)
        leaves_per_state.append([_leaf_to_array(l) for l in raw])
        ordered_hashes.append(ph)
    leaf_count = len(leaves_per_state[0])
    leaf_schema = [_leaf_schema_entry(kind, arr)
                   for kind, arr in zip(leaf_kinds, leaves_per_state[0])]
    require(all(len(lv) == leaf_count for lv in leaves_per_state),
            "FAIL CLOSED: leaf count differs across bank states")

    # 3. Serialize: treedef blob + per-state leaf arrays (fixed key order).
    treedef_bytes = pickle.dumps(treedef0)
    arrays = {"treedef": np.frombuffer(treedef_bytes, dtype=np.uint8)}
    for i, leaves in enumerate(leaves_per_state):
        for j, l in enumerate(leaves):
            arrays["s%03d_l%04d" % (i, j)] = l
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    npz_bytes = buf.getvalue()
    npz_sha = _sha256_bytes(npz_bytes)

    # 4. The immutable manifest (总控 §一.3 metadata set).
    import datetime as _dt
    manifest_doc = {
        "schema": SCHEMA,
        "artifact_version": ARTIFACT_VERSION,
        "bank_kind": "FROZEN_SCAFFOLD_BANK",
        "scenario": scenario,
        "hash_label": mat.HASH_LABELS[scenario],
        "state_count": len(states),
        "state_ids": ["%s-bank%d" % (scenario, i) for i in range(len(states))],
        "seeds": seeds,
        "canonical_content_sha256": manifest["state_bank_hash"],
        "frozen_bank_hash": mat.FROZEN_BANK_HASH[scenario],
        "per_state_payload_hashes": ordered_hashes,
        "states_npz_sha256": npz_sha,
        "treedef_sha256": _sha256_bytes(treedef_bytes),
        "leaf_count": leaf_count,
        "leaf_schema": leaf_schema,
        "bank_manifest": manifest,
        "frozen_bindings_verified_at_serialization": binding,
        "generator_source_shas": generator_source_shas(),
        "frozen_constants": {
            "field_manifest_sha256": mat.FROZEN_FIELD_MANIFEST_SHA256,
            "predicate_code_sha256": mat.FROZEN_PREDICATE_CODE_SHA256,
            "canonical_task_sha256": mat.FROZEN_CANONICAL_TASK_SHA256,
            "seed_base": mat.FROZEN_SEED_BASE,
            "seed_stride": mat.FROZEN_SEED_STRIDE,
            "n": mat.FROZEN_BANK_N,
        },
        "pinned_env_identity": current_env_identity(),
        "generation_git_commit": _git_head_or_unavailable(),
        "generation_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    require(manifest_doc["canonical_content_sha256"] == mat.FROZEN_BANK_HASH[scenario],
            "FAIL CLOSED: serialized %s content sha %s != FROZEN %s (refusing to write)"
            % (scenario, manifest_doc["canonical_content_sha256"][:16],
               mat.FROZEN_BANK_HASH[scenario][:16]))
    manifest_bytes = (json.dumps(manifest_doc, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n").encode("utf-8")

    # 5. Write states.npz + manifest.json + SHA256SUMS (immutably).
    os.makedirs(bank_dir, exist_ok=True)
    with open(os.path.join(bank_dir, "states.npz"), "wb") as fh:
        fh.write(npz_bytes)
    with open(os.path.join(bank_dir, "manifest.json"), "wb") as fh:
        fh.write(manifest_bytes)
    with open(os.path.join(bank_dir, "SHA256SUMS"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("%s  manifest.json\n" % _sha256_bytes(manifest_bytes))
        fh.write("%s  states.npz\n" % npz_sha)
    return {
        "scenario": scenario,
        "artifact_dir": bank_dir,
        "state_count": len(states),
        "canonical_content_sha256": manifest_doc["canonical_content_sha256"],
        "frozen_bank_hash": mat.FROZEN_BANK_HASH[scenario],
        "states_npz_sha256": npz_sha,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "leaf_count": leaf_count,
        "generation_backend": backend,
    }


# ---------------------------------------------------------------------------
# LOAD — read-only, any device (总控 §一.4–§一.6, §一.8)
# ---------------------------------------------------------------------------
def load_bank(scenario: str, artifacts_root: str) -> dict:
    """Read-only load of ONE frozen bank artifact. Converts arrays back to host
    NumPy, recomputes canonical_content_sha256 per state AND per bank, and fails
    closed on ANY mismatch. NEVER calls the bank generator. Returns the binding
    record consumed by the evaluator certificate: bank_source=
    FROZEN_SERIALIZED_ARTIFACT, bank_regenerated_on_eval_device=false,
    artifact_file_sha256, loaded_content_sha256, device provenance (mint/load)
    plus the host NumPy EnvState pytrees under "states" (index = frozen bank
    index)."""
    require(scenario in mat.FROZEN_BANK_HASH,
            "FAIL CLOSED: no frozen bank identity for scenario %r" % scenario)
    require(ser.have_jax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): artifact loading requires jax "
            "(tree_unflatten)")
    import jax
    import numpy as np

    bank_dir = os.path.join(artifacts_root, scenario)
    manifest_path = os.path.join(bank_dir, "manifest.json")
    sums_path = os.path.join(bank_dir, "SHA256SUMS")
    npz_path = os.path.join(bank_dir, "states.npz")
    for p in (manifest_path, sums_path, npz_path):
        require(os.path.isfile(p) and os.path.getsize(p) > 0,
                "FAIL CLOSED: frozen bank artifact file missing or empty: %s "
                "(formal evaluation requires a serialized frozen bank artifact)" % p)

    # Layer 1: SHA256SUMS over BOTH artifact files (byte-tamper gate).
    with open(manifest_path, "rb") as fh:
        manifest_bytes = fh.read()
    manifest_sha = _sha256_bytes(manifest_bytes)
    sums = {}
    with open(sums_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                sha, name = line.split(None, 1)
                sums[name.strip()] = sha.strip()
    require(sums.get("manifest.json") == manifest_sha,
            "FAIL CLOSED: artifact manifest.json bytes do not match SHA256SUMS "
            "(artifact tampered: recorded=%s actual=%s)"
            % (str(sums.get("manifest.json"))[:16], manifest_sha[:16]))
    require(bool(sums.get("states.npz")),
            "FAIL CLOSED: artifact SHA256SUMS has no states.npz entry")

    art = json.loads(manifest_bytes.decode("utf-8"))
    require(art.get("schema") == SCHEMA and art.get("artifact_version") == ARTIFACT_VERSION,
            "FAIL CLOSED: artifact manifest schema/version %r/%r != %s/%s"
            % (art.get("schema"), art.get("artifact_version"), SCHEMA, ARTIFACT_VERSION))
    require(art.get("scenario") == scenario,
            "FAIL CLOSED: artifact scenario %r != requested %r"
            % (art.get("scenario"), scenario))
    require(art.get("bank_kind") == "FROZEN_SCAFFOLD_BANK",
            "FAIL CLOSED: artifact bank_kind %r != FROZEN_SCAFFOLD_BANK"
            % art.get("bank_kind"))
    require(art.get("frozen_bank_hash") == mat.FROZEN_BANK_HASH[scenario],
            "FAIL CLOSED: artifact declares frozen_bank_hash %s != FROZEN %s"
            % (str(art.get("frozen_bank_hash"))[:16],
               mat.FROZEN_BANK_HASH[scenario][:16]))

    with open(npz_path, "rb") as fh:
        npz_bytes = fh.read()
    npz_sha = _sha256_bytes(npz_bytes)
    require(npz_sha == sums["states.npz"] == art.get("states_npz_sha256"),
            "FAIL CLOSED: states.npz file_sha256 %s != SHA256SUMS/manifest record "
            "(artifact tampered)" % npz_sha[:16])

    npz = np.load(io.BytesIO(npz_bytes), allow_pickle=False)

    # Layer 2: treedef blob sha cross-binding.
    require("treedef" in npz.files, "FAIL CLOSED: artifact states.npz has no treedef blob")
    treedef_bytes = np.asarray(npz["treedef"]).astype(np.uint8).tobytes()
    require(_sha256_bytes(treedef_bytes) == art.get("treedef_sha256"),
            "FAIL CLOSED: artifact treedef blob sha != manifest treedef_sha256 "
            "(artifact tampered)")
    treedef = pickle.loads(treedef_bytes)   # sha-gated; round-trip hash-gated below

    # Layer 3: pure frozen VALUE bindings over the embedded PROCESS_A manifest.
    bank_manifest = art.get("bank_manifest")
    require(isinstance(bank_manifest, dict),
            "FAIL CLOSED: artifact manifest embeds no bank_manifest")
    binding = mat.check_frozen_manifest_bindings(scenario, bank_manifest)
    require(art.get("canonical_content_sha256") == mat.FROZEN_BANK_HASH[scenario],
            "FAIL CLOSED: artifact canonical_content_sha256 %s != FROZEN %s"
            % (str(art.get("canonical_content_sha256"))[:16],
               mat.FROZEN_BANK_HASH[scenario][:16]))
    require(art.get("per_state_payload_hashes")
            == [e["state_payload_hash"] for e in bank_manifest["entries"]],
            "FAIL CLOSED: artifact per_state_payload_hashes != embedded bank manifest "
            "entries (order/tamper detected)")

    # Layer 4 + 5: per-state host-NumPy round-trip + bank-level recomputation.
    n = int(art.get("state_count") or 0)
    require(n == mat.FROZEN_BANK_N and len(art.get("seeds") or []) == n
            and art.get("seeds") == bank_manifest["seeds"],
            "FAIL CLOSED: artifact state_count/seeds %r != frozen n=%d schedule"
            % (art.get("state_count"), mat.FROZEN_BANK_N))
    leaf_schema = art.get("leaf_schema") or []
    leaf_count = int(art.get("leaf_count") or -1)
    require(leaf_count == len(leaf_schema) and leaf_count > 0,
            "FAIL CLOSED: artifact leaf_schema/leaf_count inconsistent")
    npz_files = set(npz.files)
    states = []
    ordered = []
    for i in range(n):
        leaves = []
        for j in range(leaf_count):
            key = "s%03d_l%04d" % (i, j)
            require(key in npz_files,
                    "FAIL CLOSED: artifact states.npz missing leaf array %s" % key)
            arr = np.asarray(npz[key])           # host NumPy canonical representation
            want = leaf_schema[j]
            kind = want.get("kind")
            if kind in ("array", "np_scalar"):
                require(str(arr.dtype) == want["dtype"]
                        and [int(d) for d in arr.shape] == list(want["shape"]),
                        "FAIL CLOSED: artifact leaf %s dtype/shape %s/%s != recorded "
                        "%s (artifact tampered)"
                        % (key, arr.dtype, list(arr.shape), want))
            # Python scalar kinds: dtype-class gate only (platform-independent;
            # see _leaf_schema_entry) — restored to their original Python type.
            leaves.append(_restore_leaf(kind, arr, key))
        st = jax.tree_util.tree_unflatten(treedef, leaves)
        ph, _payload, _v3m = ser.envstate_payload_hash(st)   # host round-trip recompute
        require(ph == bank_manifest["entries"][i]["state_payload_hash"],
                "FAIL CLOSED: loaded %s state %d canonical content hash %s != recorded "
                "%s (artifact content drifted; fail closed)"
                % (scenario, i, ph[:16],
                   str(bank_manifest["entries"][i]["state_payload_hash"])[:16]))
        ordered.append(ph)
        states.append(st)
    recomputed = mat.state_bank_hash(ordered, scenario, mat.source_shas_for_bank())
    require(recomputed == mat.FROZEN_BANK_HASH[scenario],
            "FAIL CLOSED: loaded %s bank content sha %s != FROZEN %s"
            % (scenario, recomputed[:16], mat.FROZEN_BANK_HASH[scenario][:16]))

    canonical = builder.verify_canonical_task_source(mat.FROZEN_CANONICAL_TASK_SHA256)
    return {
        "verified": True,
        "scenario": scenario,
        "hash_label": mat.HASH_LABELS[scenario],
        "state_bank_hash": recomputed,
        "state_count": n,
        "seeds": [int(s) for s in art["seeds"]],
        "ordered_payload_hashes": ordered,
        "field_manifest_sha256": bank_manifest["field_manifest_sha256"],
        "predicate_code_sha256": binding["predicate_code_sha256"],
        "canonical_task_sha256": canonical["sha256"],
        "source_shas": bank_manifest["source_shas"],
        "boundary_predicate_version": bank_manifest["boundary_predicate_version"],
        "states": states,                       # host-NumPy EnvState pytrees
        "bank_source": "FROZEN_SERIALIZED_ARTIFACT",
        "bank_regenerated_on_eval_device": False,
        "artifact_file_sha256": npz_sha,
        "loaded_content_sha256": recomputed,
        "artifact_manifest_sha256": manifest_sha,
        "artifact_dir": bank_dir,
        "device_provenance": {
            "mint": art.get("pinned_env_identity") or {},
            "load": current_env_identity(),
        },
    }


def verify_artifacts(artifacts_root: str, scenarios=(FRONT, BACK)) -> dict:
    """Load + full gate verification of every requested bank artifact (any device)."""
    out = {}
    for sc in scenarios:
        b = load_bank(sc, artifacts_root)
        out[sc] = {
            "verified": True,
            "state_bank_hash": b["state_bank_hash"],
            "frozen_bank_hash": mat.FROZEN_BANK_HASH[sc],
            "state_count": b["state_count"],
            "artifact_file_sha256": b["artifact_file_sha256"],
            "loaded_content_sha256": b["loaded_content_sha256"],
            "artifact_manifest_sha256": b["artifact_manifest_sha256"],
        }
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    def rejects(fn):
        try:
            fn()
            return False
        except Exception:
            return True

    # Constants are frozen and match the materializer bindings.
    check("schema_constants", bool(SCHEMA) and bool(ARTIFACT_VERSION))
    check("frozen_hash_constants",
          mat.FROZEN_BANK_HASH[FRONT].startswith("21aeb7dc")
          and mat.FROZEN_BANK_HASH[BACK].startswith("c632e30d"))

    # Loading from a missing artifact dir fails closed (any host).
    check("missing_artifact_rejected",
          rejects(lambda: load_bank(FRONT, "/nonexistent-artifact-root-cc4")))

    if ser.have_jax_craftax():
        import jax
        import tempfile
        backend = str(jax.devices()[0].platform) if jax.devices() else "<none>"
        if backend == "cpu":
            # Full round-trip on the pinned CPU backend: serialize -> load -> gate.
            with tempfile.TemporaryDirectory() as td:
                s_front = serialize_bank(FRONT, td)
                s_back = serialize_bank(BACK, td)
                check("serialize_front_frozen_hash",
                      s_front["canonical_content_sha256"]
                      == mat.FROZEN_BANK_HASH[FRONT]
                      == s_front["frozen_bank_hash"])
                check("serialize_back_frozen_hash",
                      s_back["canonical_content_sha256"]
                      == mat.FROZEN_BANK_HASH[BACK])
                lf = load_bank(FRONT, td)
                lb = load_bank(BACK, td)
                check("load_front_verified",
                      lf["verified"] is True and lf["state_count"] == mat.FROZEN_BANK_N
                      and lf["loaded_content_sha256"] == mat.FROZEN_BANK_HASH[FRONT]
                      and lf["bank_source"] == "FROZEN_SERIALIZED_ARTIFACT"
                      and lf["bank_regenerated_on_eval_device"] is False
                      and len(lf["states"]) == mat.FROZEN_BANK_N
                      and lf["device_provenance"]["mint"]
                      and lf["device_provenance"]["load"])
                check("load_back_verified",
                      lb["loaded_content_sha256"] == mat.FROZEN_BANK_HASH[BACK])
                # state IDs / seeds frozen schedule
                check("load_seeds_frozen",
                      lf["seeds"] == mat.fixed_seed_schedule(
                          FRONT, mat.FROZEN_BANK_N, mat.FROZEN_SEED_BASE,
                          mat.FROZEN_SEED_STRIDE))
                # round-trip: loaded state 0 re-canonicalizes to its recorded hash
                ph0, _p, _m = ser.envstate_payload_hash(lf["states"][0])
                check("roundtrip_state0_hash",
                      ph0 == lf["ordered_payload_hashes"][0])
                # Python scalar leaves (T_BOOL/T_INT/T_FLOAT tags) must survive
                # the round-trip with their ORIGINAL types — the loader restores
                # them from the recorded leaf kinds; the frozen bank genuinely
                # carries Python scalar leaves, so at least one must be present.
                with open(os.path.join(td, FRONT, "manifest.json"), "rb") as fh:
                    man = json.loads(fh.read().decode("utf-8"))
                loaded_kinds = [_leaf_kind(l)
                                for l in jax.tree_util.tree_leaves(lf["states"][0])]
                check("scalar_leaf_kinds_roundtrip",
                      loaded_kinds == [e["kind"] for e in man["leaf_schema"]]
                      and any(k != "array" for k in loaded_kinds)
                      and all(type(l) in (bool, int, float)
                              for k, l in zip(loaded_kinds,
                                              jax.tree_util.tree_leaves(
                                                  lf["states"][0]))
                              if k.startswith("py_")))
                # TAMPER 1: one byte of states.npz -> file-sha gate rejects
                with open(os.path.join(td, FRONT, "states.npz"), "r+b") as fh:
                    fh.seek(20)
                    b = fh.read(1)
                    fh.seek(20)
                    fh.write(bytes([b[0] ^ 0xFF]))
                check("tamper_npz_rejected", rejects(lambda: load_bank(FRONT, td)))
            # TAMPER 2: trailing byte on manifest.json -> SHA256SUMS gate rejects
            with tempfile.TemporaryDirectory() as td:
                serialize_bank(FRONT, td)
                with open(os.path.join(td, FRONT, "manifest.json"), "ab") as fh:
                    fh.write(b" ")
                check("tamper_manifest_rejected", rejects(lambda: load_bank(FRONT, td)))
            # cross-scenario request against a front-only root -> rejected
            with tempfile.TemporaryDirectory() as td:
                serialize_bank(FRONT, td)
                check("cross_scenario_rejected", rejects(lambda: load_bank(BACK, td)))
            # overwrite of an existing artifact dir -> rejected (immutability)
            with tempfile.TemporaryDirectory() as td:
                serialize_bank(FRONT, td)
                check("overwrite_rejected", rejects(lambda: serialize_bank(FRONT, td)))
        else:
            # GPU host: serialization itself must fail closed.
            with tempfile.TemporaryDirectory() as td:
                check("gpu_serialize_rejected",
                      rejects(lambda: serialize_bank(FRONT, td)))

    if problems:
        print("TIER3_FROZEN_BANK_ARTIFACTS_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    env = ser.environment_status()
    print("TIER3_FROZEN_BANK_ARTIFACTS_SELF_TEST_PASS (env=%s)" % env)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Real minting imports minicraftax under <repo>/dicode_src/src (audited relpaths).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
    if "--self-test" in argv:
        return self_test()

    def _opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    if "--serialize" in argv:
        out = _opt("--out")
        require(out, "FAIL CLOSED: --serialize requires --out <DIR>")
        scenario = _opt("--scenario", "both")
        scenarios = (FRONT, BACK) if scenario == "both" else (scenario,)
        summary = {}
        for sc in scenarios:
            s = serialize_bank(sc, out)
            summary[sc] = s
            print("ARTIFACT_SERIALIZED scenario=%s states=%d "
                  "canonical_content_sha256=%s frozen_bank_hash=%s "
                  "states_npz_sha256=%s manifest_sha256=%s dir=%s"
                  % (sc, s["state_count"], s["canonical_content_sha256"],
                     s["frozen_bank_hash"], s["states_npz_sha256"],
                     s["manifest_sha256"], s["artifact_dir"]), flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if "--verify" in argv:
        root = _opt("--artifacts")
        require(root, "FAIL CLOSED: --verify requires --artifacts <DIR>")
        scenario = _opt("--scenario", "both")
        scenarios = (FRONT, BACK) if scenario == "both" else (scenario,)
        summary = verify_artifacts(root, scenarios)
        for sc, s in summary.items():
            print("ARTIFACT_VERIFIED scenario=%s state_count=%d "
                  "loaded_content_sha256=%s frozen_bank_hash=%s artifact_file_sha256=%s"
                  % (sc, s["state_count"], s["loaded_content_sha256"],
                     s["frozen_bank_hash"], s["artifact_file_sha256"]), flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    print("usage: tier3_frozen_bank_artifacts.py --self-test\n"
          "       tier3_frozen_bank_artifacts.py --serialize "
          "[--scenario {front_l2,back_l2,both}] --out <DIR>   (pinned CPU only)\n"
          "       tier3_frozen_bank_artifacts.py --verify --artifacts <DIR> "
          "[--scenario {front_l2,back_l2,both}]")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
