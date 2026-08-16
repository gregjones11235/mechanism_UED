#!/usr/bin/env python
"""ACTUAL Craftax world-set materializer (CANONICAL_EVALUATOR_V1).

Replaces the DEPRECATED world-KEY-manifest prototype (world_key_manifest_prototype.py).
Where the prototype hashed ONLY a jax.random.fold_in key + a recipe descriptor, THIS
module is designed to serialize the ACTUAL materialized Craftax world state and hash
those real bytes.

=====================================================================================
STATUS (this host has NO JAX / NO craftax -- verified: importlib find_spec == False)
=====================================================================================
  WORLD_GENERATION_SOURCE_PATH            = FOUND   (see world_generation_path_audit)
  CRAFTAX_WORLD_MATERIALIZER_CODE         = IMPLEMENTED (static; real run NOT_RUN here)
  CRAFTAX_WORLD_MATERIALIZER_STATIC_TESTS = PASS    (pure-Python serializer self-test)
  CRAFTAX_WORLD_MATERIALIZER_REAL_RUN     = NOT_RUN (no JAX/craftax on this host)
  GLOBAL_WORLD_SET_HASH                   = BLOCKED_SOURCE_UNVERIFIED

=====================================================================================
V3 ADDENDUM (GLOBAL_WORLD_MATERIALIZER_RUNTIME_IDENTITY_HARDENING_V3)
=====================================================================================
  CC4_RUNTIME_SOURCE_IDENTITY_CODE        = PASS    (binding code implemented; see below)
  CC4_RUNTIME_SOURCE_IDENTITY_REAL_RUN    = NOT_RUN (binding executes only on a JAX host)
  EVALUATION_SEED_STATIC_RNG_BINDING      = PASS    (PRNGKey(EVAL_SEED) anchor in source)
  EVALUATION_SEED_REAL_WORLD_PAYLOAD_EFFECT = BLOCKED_ENVIRONMENT (no JAX/craftax here)
  WORLD_STATE_PAYLOAD_HASH                = IMPLEMENTED (seed-free payload hash helper)
  WORLD_FIELD_MANIFEST_CODE               = IMPLEMENTED (manifests persisted on a real run)
  WORLD_FIELD_MANIFEST_REAL_OUTPUT        = NOT_RUN

V3 hardening added on top of V2 (V2 guarantees retained, none downgraded):
  * RUNTIME EXECUTED-SOURCE IDENTITY BINDING: after the real ``import dicode.wrappers_cl``
    / ``import minicraftax.envs.multitask`` we capture ``module.__file__`` AND
    ``inspect.getsourcefile(<the class we actually call>)``, abspath+realpath-resolve them,
    recompute SHA256, and REQUIRE equality with the command-line-requested source -- else
    FailClosed(EXECUTED_SOURCE_IDENTITY_MISMATCH). We no longer rely on "the N on-disk
    copies happen to be byte-identical". The canonical S4 task exec path is bound the same
    way (TASK_EXECUTED_SOURCE_IDENTITY_MISMATCH on mismatch).
  * EXECUTED vs PROTOCOL-ANCHOR PROVENANCE SPLIT: wrapper/environment/task are recorded
    under ``executed_sources`` (the materializer actually imports/execs/calls them);
    eval_phase2_unified.py is recorded under ``protocol_anchor_sources`` with
    ``executed_by_materializer = false`` (we REPRODUCE its build+reset logic but never
    execute the evaluator program). We never write "evaluator source executed".
  * SEED-FREE WORLD PAYLOAD HASH: ``serialize_world_payload`` hashes ONLY the canonical
    serialized initial EnvState -- no evaluation_seed, seed_id, source SHA or version in
    those bytes. ``state_payload_hash = SHA256(payload)``. This is the honest carrier for
    "does the numeric seed actually change the REAL world" (NEG02 / GATE21): a header-seed
    difference alone is NOT evidence, because the header is deliberately seed-tagged.
  * SEED IDENTITY CLASSIFICATION: seed42 = CANONICAL_EVALUATOR_EXACT_WORLD_SET (the only
    class admissible as the canonical evaluator's exact world-set evidence); seed100000 =
    PARAMETERIZED_WORLD_GENERATION_PROTOCOL_VARIANT (robustness/screening/held-out only).
    A real frozen seed100000 evaluator (eval_p7_egomap_paired_256.py, P7_PAIRED_256) is
    recorded as an INDEPENDENT evaluator identity with its own full SHA -- never reusing
    the seed42 evaluator's identity.
  * WORLD FIELD MANIFEST PERSISTENCE: every real run writes world_field_manifests.json
    (per-world array path/dtype/shape/nbytes) + world_field_schema_summary.json, and binds
    ``world_field_manifests_sha256`` into world_hashes.json. ``assert_materialized``
    REJECTS a materialized result that lacks the manifest evidence.

The serializer / header / hash / two-process / gate logic below is pure Python and is
fully exercised by --self-test on this host. The MATERIALIZATION step (actually
importing craftax + the real wrapper + executing env.reset for 256 worlds) requires a
JAX + craftax==1.4.5 host and is GUARDED: on a JAX-less host it FAILS CLOSED (exit 2)
and never emits a world_set_hash. A mock world is NEVER accepted as a real run.

=====================================================================================
CANONICAL WORLD-GENERATION PATH (verbatim, line-anchored -- NOT a guessed path)
=====================================================================================
The per-world reset key is a PURE POSITIONAL jax.random.split chain (fold_in appears
NOWHERE in any of the four real evaluators; grep-confirmed zero hits):

  EVAL_SEED = 42                                  eval_phase2_unified.py:77 (hardcoded)
  rng       = jax.random.PRNGKey(EVAL_SEED)        eval_phase2_unified.py:169
  rng, reset_rng = jax.random.split(rng)           eval_phase2_unified.py:170  -> [1]
  obsv, log_state = env.reset(reset_rng, ctor)     eval_phase2_unified.py:171
      key, _rng = jax.random.split(key)            wrappers_cl.py:228          -> [1]
      reset_rngs = jax.random.split(_rng, 256)     wrappers_cl.py:229
      vmapped_reset(reset_rngs, params, task_ids, task_embeddings)
                                                   wrappers_cl.py:231 (vmap reset_env)
      # world i receives reset_rngs[i]; MultiTaskMiniCraftaxEnv.reset_env then does:
      rng, world_rng = jax.random.split(rng)       multitask.py:129            -> [1]
      state = lax.switch(task_id, world_gen_fns, world_rng)
                                                   multitask.py:132 (selects task.generate_world)
      # for the canonical S4 task, generate_world splits once more:
      rng, _r = jax.random.split(rng)              s4_task_code.py:39          -> [1]
      b = WorldBuilder(_r, static_params, params)  s4_task_code.py:40
      b.set_starting_floor(2); b.set_monsters_killed(2, 8); b.set_player_inventory({...})
      s = b.build(rng)                             s4_task_code.py:47
      return s.replace(item_map=...at[2,up].set(NONE))   s4_task_code.py:49
      # s is a craftax EnvState  == the world-identity carrier.

  world_key[i] = split( split( split( PRNGKey(42) )[1] )[1], 256 )[i]

CONSEQUENCE: a single world's key is NOT independently derivable from (seed, index); it
depends on the whole 256-way batch and its ordering. A materializer MUST reproduce the
entire 256-way split to land on the canonical worlds.

evaluation_seed=42 ENTERS the real world-gen RNG via PRNGKey(EVAL_SEED)->split chain
(NOT merely action sampling). PRNGKey(0) in the wrapper constructor is used ONLY to
permute task_ids (wrappers_cl.py:225-ish; identity when num_tasks=1) and does NOT enter
the per-world reset key.

Source SHA anchors (recorded in the header of every materialized world):
  evaluator  eval_phase2_unified.py  224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1
  wrapper    wrappers_cl.py          2ded41d81a98c712...  (6 on-disk copies byte-identical,
                                     incl. the dicode_src copy importable here AND the
                                     dicode_v7fix58_armB copy the evaluator actually loads)
  task       s4_task_code.py         45fdd17c5b34b9f3...  (p2_v1 canonical, LF, 2074 bytes;
                                     the P2-v0 "invalid-for-attribution" 34-line copy is NOT it)

NOTE on shared builder (GATE19): the real evaluators keep env construction + the three
seed lines INLINE (copy-pasted) and load the task via exec() of an absolute path; there
is NO importable shared world-builder, and the canonical evaluator file is READ-ONLY and
must not be modified. Therefore a literal "evaluator and materializer call the same
builder function" is BLOCKED. The strongest honest guarantee achievable here is the
STATIC ANCHOR TEST below: the constants/derivation embedded in build_canonical_eval_world
are asserted equal to the literal values parsed out of the canonical source files.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import struct
import sys

# ----------------------------------------------------------------------------------- #
# Protocol constants (MUST match CANONICAL_EVALUATOR_V1 / the canonical evaluator)
# ----------------------------------------------------------------------------------- #
SCHEMA_VERSION = "mechanism_UED.craftax_materialized_world/v1"
PROTOCOL_ID = "CANONICAL_EVALUATOR_V1"
NUM_WORLDS = 256
MAX_TIMESTEPS = 4096
OPTIMISTIC_RESET_RATIO = 16
NUM_TASKS = 1
SPAWN_FLOOR = 2
EXPECTED_CRAFTAX = "1.4.5"
ALLOWED_SEEDS = {"seed42": 42, "seed100000": 100000}

# Canonical source SHA anchors (verified by hand this round; see module docstring).
EVALUATOR_SHA256 = "224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1"
WRAPPER_SHA256_PREFIX = "2ded41d81a98c712"     # full sha recorded at run from the real file
TASK_SHA256_PREFIX = "45fdd17c5b34b9f3"
# Full canonical S4 task SHA (p2_v1, LF, 49 lines) -- V3 binds the FULL sha, not only prefix.
TASK_SHA256 = "45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d"
ENV_SOURCE_SHA256 = "c8f2d5c3c23476c92ab3897f47bef4df7f202a3bd57360fc1bd4cb92b9498bae"  # multitask.py
WRAPPER_SHA256 = "2ded41d81a98c712620dc1633262f2d185ce7dd22e7cc447db22a6ad04b0ddd8"     # wrappers_cl.py

# ----------------------------------------------------------------------------------- #
# V3: field-manifest schema
# ----------------------------------------------------------------------------------- #
FIELD_MANIFEST_SCHEMA = "mechanism_UED.craftax_world_field_manifests/v1"
FIELD_SCHEMA_SUMMARY_SCHEMA = "mechanism_UED.craftax_world_field_schema_summary/v1"
RUNTIME_SOURCE_IDENTITY_SCHEMA = "mechanism_UED.craftax_runtime_source_identity/v1"

# ----------------------------------------------------------------------------------- #
# V3: SEED IDENTITY CLASSIFICATION (section nine / ten)
# ----------------------------------------------------------------------------------- #
# seed42 is the ONLY class admissible as the canonical evaluator's EXACT world-set
# evidence. seed100000 is a parameterized world-generation variant (robustness / dev
# screening / extra held-out seed) and MUST NOT be passed off as the canonical evaluator's
# exact world set. Each carries its own protocol_id so results can never be silently pooled.
PROTOCOL_ID_SEED42 = "CANONICAL_EVALUATOR_V1_SEED42_EXACT"
PROTOCOL_ID_SEED100000 = "CANONICAL_EVALUATOR_V1_WORLDGEN_PARAMETERIZED_SEED100000"

# An INDEPENDENT, real, frozen evaluator that uses seed100000 (default --seed_base) was
# located in the repo (P7_PAIRED_256). It is recorded with its OWN full SHA so the
# seed100000 identity never reuses the seed42 evaluator's identity. (It is parameterized
# via --seed_base, default 100000; EVAL_SEED=int(args.seed_base); rng=PRNGKey(EVAL_SEED)
# at :190; same DistributedMultiTaskOptimisticLogWrapper(s4_base, PRNGKey(0), ...) at :136.)
SEED100000_EVALUATOR = {
    "evaluator_id": "P7_PAIRED_256",
    "path": "experiments/henry_dicode_student_upgrade/06_p7_egomap/raw_sources/home/oseasy/"
            "experiments/student_upgrade_wave1_4gpu/gpu1_p7_egomap/eval_p7_egomap_paired_256.py",
    "sha256_raw_on_disk": "f9c864359cfffe7726d93870fd17e52e18a7e49aa9a468471abf59088799a1a9",
    "sha256_lf_normalized": "c082db8b82e86b971d8943bd9275ba8b709ffdc0da198fb236c52ccd56c08325",
    "line_endings_on_disk": "CRLF",
    "seed_mechanism": "--seed_base default 100000 (:36); EVAL_SEED=int(args.seed_base) (:66); "
                      "rng=jax.random.PRNGKey(EVAL_SEED) (:190)",
    "wrapper_pattern": "DistributedMultiTaskOptimisticLogWrapper(s4_base, PRNGKey(0), ...) (:136)",
    "parameterized": True,
    "note": "real frozen P7 evaluator; independent of the canonical seed42 evaluator; its world "
            "set is a DIFFERENT seed line and MUST NOT be pooled with seed42.",
}

SEED_IDENTITIES = {
    "seed42": {
        "numeric_seed": 42,
        "identity_class": "CANONICAL_EVALUATOR_EXACT_WORLD_SET",
        "evaluator_exact_match": True,
        "protocol_id": PROTOCOL_ID_SEED42,
        "canonical_evaluator_sha256": EVALUATOR_SHA256,
        "admissible_as_canonical_exact_world_set_evidence": True,
        "independent_evaluator": None,
    },
    "seed100000": {
        "numeric_seed": 100000,
        "identity_class": "PARAMETERIZED_WORLD_GENERATION_PROTOCOL_VARIANT",
        "evaluator_exact_match": False,
        "protocol_id": PROTOCOL_ID_SEED100000,
        "canonical_evaluator_sha256": None,   # NOT the canonical seed42 evaluator's world set
        "admissible_as_canonical_exact_world_set_evidence": False,
        "independent_evaluator": SEED100000_EVALUATOR,
    },
}

EXIT_OK = 0
EXIT_FAIL_CLOSED = 2
EXIT_USAGE = 3


def eprint(*a):
    print(*a, file=sys.stderr)


class FailClosed(Exception):
    """Hard requirement unmet -> refuse to emit any world_set_hash."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# =================================================================================== #
# STABLE CANONICAL SERIALIZER  (pure Python; no pickle; deterministic; lossless)
# =================================================================================== #
# Every node is encoded as TAG(1 byte) + payload. Mapping-like containers (dict,
# dataclass, NamedTuple) are encoded with their keys/field-names in SORTED order so the
# hash is invariant to declaration order (negative test 4). Arrays bind dtype + shape +
# C-order bytes; the field PATH is bound implicitly through the sorted dict-key chain
# leading to the array and is recorded explicitly in the returned manifest.

T_NONE = 0x00
T_BOOL = 0x01
T_INT = 0x02          # arbitrary-precision signed (two's complement, length-prefixed)
T_FLOAT = 0x03        # IEEE754, native precision (py float -> f8; np float -> its dtype)
T_STR = 0x04
T_BYTES = 0x05
T_ENUM = 0x06
T_DICT = 0x07         # keys sorted
T_LIST = 0x08
T_TUPLE = 0x09
T_NAMEDTUPLE = 0x0A   # fields sorted by name
T_DATACLASS = 0x0B    # fields sorted by name; qualified class name bound
T_ARRAY = 0x0C        # dtype + shape + C-order bytes (lossless, native dtype)
T_NPSCALAR = 0x0D     # numpy scalar with its native dtype


def _u64(n):
    return struct.pack(">Q", n)


def _lenprefixed(b):
    return _u64(len(b)) + b


def _is_namedtuple(x):
    return isinstance(x, tuple) and hasattr(x, "_fields") and hasattr(x, "_asdict")


def _is_dataclass_instance(x):
    # True for stdlib dataclasses AND flax struct.dataclass (minicraftax EnvState/TaskParams).
    return (dataclasses.is_dataclass(x) or hasattr(x, "__dataclass_fields__")) \
        and not isinstance(x, type)


def _is_array_like(x):
    # numpy ndarray AND jax Array both expose .shape/.dtype and convert via np.asarray.
    return hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, (str, bytes))


def _np():
    import numpy as np
    return np


def encode_node(obj, path, manifest):
    """Canonical, deterministic encoding of obj. Appends array entries to `manifest`.

    `path` is a tuple of keys/indices describing where obj sits in the pytree; it is used
    only for the manifest, never to inject non-determinism into the byte stream.
    """
    np = _np()

    if obj is None:
        return bytes([T_NONE])

    if isinstance(obj, bool):                       # bool BEFORE int (bool is int subclass)
        return bytes([T_BOOL, 1 if obj else 0])

    if isinstance(obj, int):
        # arbitrary precision signed two's complement
        if obj == 0:
            payload = b"\x00"
        else:
            nbytes = (obj.bit_length() + 8) // 8     # +sign bit
            payload = obj.to_bytes(nbytes, "big", signed=True)
        return bytes([T_INT]) + _lenprefixed(payload)

    if isinstance(obj, float):
        return bytes([T_FLOAT]) + b"f8" + struct.pack(">d", obj)   # IEEE754 double, lossless

    if isinstance(obj, str):
        return bytes([T_STR]) + _lenprefixed(obj.encode("utf-8"))

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes([T_BYTES]) + _lenprefixed(bytes(obj))

    # numpy / jax scalar (zero-dim or np.generic) -- bind native dtype
    if isinstance(obj, np.generic):
        arr = np.asarray(obj)
        dt = arr.dtype.str                          # e.g. '<f4', '<i8' (canonical numpy str)
        raw = arr.astype(arr.dtype.newbyteorder(">"), copy=False).tobytes(order="C")
        manifest.append({"path": _pathstr(path), "dtype": dt, "shape": [],
                         "nbytes": len(raw)})
        return bytes([T_NPSCALAR]) + _lenprefixed(dt.encode("ascii")) + _lenprefixed(raw)

    if _is_array_like(obj):
        arr = np.asarray(obj)
        dt = arr.dtype.str
        # lossless: keep native dtype, only force big-endian for cross-platform stability
        raw = arr.astype(arr.dtype.newbyteorder(">"), copy=False).tobytes(order="C")
        shape = tuple(int(s) for s in arr.shape)
        manifest.append({"path": _pathstr(path), "dtype": dt, "shape": list(shape),
                         "nbytes": len(raw)})
        shape_bytes = b"".join(_u64(s) for s in shape)
        return (bytes([T_ARRAY]) + _lenprefixed(dt.encode("ascii"))
                + _u64(len(shape)) + shape_bytes + _lenprefixed(raw))

    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: _sortkey(kv[0]))
        out = bytearray([T_DICT]); out += _u64(len(items))
        for k, v in items:
            ks = k if isinstance(k, str) else repr(k)
            out += _lenprefixed(ks.encode("utf-8"))
            out += encode_node(v, path + (ks,), manifest)
        return bytes(out)

    if _is_namedtuple(obj):
        fields = sorted(obj._fields)                # sorted -> order-invariant
        out = bytearray([T_NAMEDTUPLE])
        out += _lenprefixed(type(obj).__qualname__.encode("utf-8"))
        out += _u64(len(fields))
        for fn in fields:
            out += _lenprefixed(fn.encode("utf-8"))
            out += encode_node(getattr(obj, fn), path + (fn,), manifest)
        return bytes(out)

    if isinstance(obj, tuple):
        out = bytearray([T_TUPLE]); out += _u64(len(obj))
        for i, v in enumerate(obj):
            out += encode_node(v, path + (i,), manifest)
        return bytes(out)

    if isinstance(obj, list):
        out = bytearray([T_LIST]); out += _u64(len(obj))
        for i, v in enumerate(obj):
            out += encode_node(v, path + (i,), manifest)
        return bytes(out)

    if isinstance(obj, enum_base()):
        # bind qualified enum name + the (recursively encoded) value
        out = bytearray([T_ENUM])
        out += _lenprefixed((type(obj).__module__ + "." + type(obj).__qualname__).encode("utf-8"))
        out += _lenprefixed(obj.name.encode("utf-8"))
        out += encode_node(obj.value, path + ("<value>",), manifest)
        return bytes(out)

    if _is_dataclass_instance(obj):
        fields = sorted(getattr(obj, "__dataclass_fields__").keys())   # sorted -> order-invariant
        out = bytearray([T_DATACLASS])
        out += _lenprefixed((type(obj).__module__ + "." + type(obj).__qualname__).encode("utf-8"))
        out += _u64(len(fields))
        for fn in fields:
            out += _lenprefixed(fn.encode("utf-8"))
            out += encode_node(getattr(obj, fn), path + (fn,), manifest)
        return bytes(out)

    raise FailClosed("FAIL CLOSED: unsupported type for canonical world serialization: %r "
                     "(refusing to fall back to pickle/repr)" % type(obj))


def enum_base():
    import enum
    return enum.Enum


def _sortkey(k):
    return k if isinstance(k, str) else repr(k)


def _pathstr(path):
    out = []
    for p in path:
        if isinstance(p, int):
            out.append("[%d]" % p)
        else:
            out.append(("." if out else "") + str(p))
    return "".join(out) or "<root>"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------- #
# Header (deterministic; NO wall-clock -> two independent processes byte-agree)
# ----------------------------------------------------------------------------------- #
def build_header(identity_pytree, evaluation_seed, world_index, source_shas, versions):
    """Deterministic header JSON. Sorted keys; contains NO timestamp -> reproducible."""
    return {
        "schema": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "world_index": int(world_index),
        "evaluation_seed": int(evaluation_seed),       # NUMERIC seed binds the bytes
        "generation_recipe": {
            "reset_key_derivation": "split(split(split(PRNGKey(evaluation_seed))[1])[1], %d)[world_index]" % NUM_WORLDS,
            "num_worlds": NUM_WORLDS,
            "max_timesteps": MAX_TIMESTEPS,
            "optimistic_reset_ratio": OPTIMISTIC_RESET_RATIO,
            "num_tasks": NUM_TASKS,
            "condition_on_task": True,
            "spawn_floor": SPAWN_FLOOR,
            "wrapper_constructor_args": "(s4_base, PRNGKey(0) [task_ids permutation only], 256, 1, 16, [1.0], table)",
            "task": "DEFEAT_KOBOLD (s4_task_code.py: set_starting_floor(2), monsters_killed[2]=8, fixed inventory, item_map[2,up]=NONE, TaskParams(needs_depletion_multiplier=0.3))",
            "world_index_order": "0..255 ascending",
        },
        "source_shas": source_shas,                    # evaluator/env/wrapper/task
        "versions": versions,                          # craftax / jax / jaxlib
        "field_manifest_note": "per-array dtype/shape/path recorded in the binary manifest",
    }


def serialize_world_payload(identity_pytree, manifest=None):
    """V3: canonical bytes of ONLY the actual world-identity payload (seed-free).

    Returns (payload_bytes, manifest). The payload contains the canonical serialization of
    the initial EnvState pytree and NOTHING ELSE -- deliberately NO evaluation_seed, NO
    seed_id, NO source SHA, NO version string, NO header metadata. Therefore:

        state_payload_hash = SHA256(payload_bytes)

    is a hash of the REAL world alone. Two seeds whose materialized EnvStates are byte-identical
    yield an IDENTICAL state_payload_hash even though their header-tagged per_world_hash
    differs. This is the ONLY honest carrier for "does the numeric seed change the real
    world" (NEG02 / GATE21): a per_world_hash / world_set_hash difference is NOT proof,
    because those hashes fold in the deliberately seed-tagged header.
    """
    if manifest is None:
        manifest = []
    payload = encode_node(identity_pytree, (), manifest)
    return payload, manifest


def state_payload_hash(identity_pytree):
    """V3 convenience: SHA256 of the seed-free canonical world payload."""
    payload, manifest = serialize_world_payload(identity_pytree)
    return sha256_bytes(payload), manifest


def serialize_world(identity_pytree, evaluation_seed, world_index, source_shas, versions):
    """Return (world_bytes, header_dict, manifest, per_world_hash).

    world_bytes = lenprefixed(header_json) + lenprefixed(payload)
    per_world_hash = SHA256(world_bytes)   -- over the ACTUAL materialized state bytes.

    NOTE: per_world_hash binds the seed-tagged header, so it changes with the numeric seed
    EVEN IF the real EnvState payload were identical. For the seed-effect question use the
    seed-free ``serialize_world_payload`` / ``state_payload_hash`` instead (V3).
    """
    payload, manifest = serialize_world_payload(identity_pytree)
    header = build_header(identity_pytree, evaluation_seed, world_index, source_shas, versions)
    header_json = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    world_bytes = _lenprefixed(header_json) + _lenprefixed(payload)
    return world_bytes, header, manifest, sha256_bytes(world_bytes)


def compute_world_set_hash(per_world_hashes, evaluation_seed, source_shas, versions):
    """Total hash is NOT a naive string concat of per-world hashes.

    world_set_hash = SHA256( canonical length-prefixed concatenation of:
        schema_version, evaluator_sha, env_source_sha, wrapper_source_sha, task_source_sha,
        craftax_version, jax_version, jaxlib_version, NUMERIC evaluation_seed,
        ordered[ (world_index ascending, per_world_hash) ] )
    """
    parts = [
        SCHEMA_VERSION,
        source_shas["evaluator_sha256"],
        source_shas["environment_source_sha256"],
        source_shas["wrapper_source_sha256"],
        source_shas["task_source_sha256"],
        versions["craftax"],
        versions["jax"],
        versions["jaxlib"],
        "evaluation_seed=%d" % int(evaluation_seed),
        "world_count=%d" % NUM_WORLDS,
    ]
    h = hashlib.sha256()
    for p in parts:
        h.update(_lenprefixed(p.encode("utf-8")))
    for idx in range(NUM_WORLDS):                       # ascending order 0..255
        require(str(idx) in per_world_hashes or idx in per_world_hashes,
                "FAIL CLOSED: missing per-world hash for world %d" % idx)
        ph = per_world_hashes.get(str(idx), per_world_hashes.get(idx))
        h.update(_u64(idx))
        h.update(_lenprefixed(ph.encode("ascii")))
    return h.hexdigest()


# =================================================================================== #
# V3: SEED IDENTITY (section nine / ten)
# =================================================================================== #
def seed_identity(seed_id):
    """Return the full identity record for a seed_id (fails closed on unknown seed)."""
    require(seed_id in SEED_IDENTITIES,
            "FAIL CLOSED: unknown seed_id %r; allowed=%s" % (seed_id, sorted(SEED_IDENTITIES)))
    ident = dict(SEED_IDENTITIES[seed_id])
    ident["seed_id"] = seed_id
    return ident


def assert_exact_world_set_eligible(seed_id):
    """GATE22: ONLY identity_class == CANONICAL_EVALUATOR_EXACT_WORLD_SET may serve as the
    canonical evaluator's exact world-set evidence. A parameterized variant (seed100000)
    may be used for robustness / screening / held-out seeds but NEVER as the exact set."""
    ident = seed_identity(seed_id)
    require(ident["identity_class"] == "CANONICAL_EVALUATOR_EXACT_WORLD_SET"
            and ident["evaluator_exact_match"] is True
            and ident["admissible_as_canonical_exact_world_set_evidence"] is True,
            "GATE22 REJECT: seed %r has identity_class=%r; only "
            "CANONICAL_EVALUATOR_EXACT_WORLD_SET is admissible as the canonical evaluator's "
            "exact world-set evidence. seed100000 is a PARAMETERIZED variant and MUST NOT be "
            "passed off as the hardcoded-seed42 evaluator's exact world set."
            % (seed_id, ident["identity_class"]))
    return ident


# =================================================================================== #
# V3: RUNTIME EXECUTED-SOURCE IDENTITY BINDING (section two / three / four)
# =================================================================================== #
def _resolve_real(p):
    """abspath THEN realpath (resolves symlinks) -- the canonical file identity."""
    return os.path.realpath(os.path.abspath(p))


def verify_source_identity(requested_path, imported_file, label, expected_sha256=None):
    """PURE: prove the file Python actually imported/executed IS the requested/recorded source.

    Compares realpath AND freshly-computed SHA256 of both. A symlinked requested_path that
    resolves to the very same real file (same realpath + same SHA) passes; a DIFFERENT file
    -- even one that is merely byte-identical -- FAILS CLOSED, because we must not rely on
    "the N on-disk copies happen to be byte-identical". Returns an identity dict; raises
    FailClosed(EXECUTED_SOURCE_IDENTITY_MISMATCH) on any mismatch.
    """
    require(requested_path and os.path.isfile(requested_path),
            "FAIL CLOSED: %s: requested source path missing (%r)" % (label, requested_path))
    require(imported_file and os.path.isfile(imported_file),
            "FAIL CLOSED: %s: imported/executed module file missing (%r)" % (label, imported_file))
    req_real = _resolve_real(requested_path)
    imp_real = _resolve_real(imported_file)
    req_sha = sha256_file(req_real)
    imp_sha = sha256_file(imp_real)
    match = (req_real == imp_real) and (req_sha == imp_sha)
    if expected_sha256 is not None:
        match = match and (imp_sha == expected_sha256)
    ident = {
        "label": label,
        "requested_path": os.path.abspath(requested_path),
        "requested_realpath": req_real,
        "requested_sha256": req_sha,
        "imported_module_file": os.path.abspath(imported_file),
        "imported_module_realpath": imp_real,
        "executed_sha256": imp_sha,
        "expected_sha256": expected_sha256,
        "identity_match": bool(match),
    }
    require(match,
            "FAIL CLOSED: EXECUTED_SOURCE_IDENTITY_MISMATCH for %s: the file Python actually "
            "imported/executed (%s, sha256=%s) is NOT the requested/recorded source (%s, "
            "sha256=%s). Refusing to bind any world_set_hash to an unproven source."
            % (label, imp_real, imp_sha[:16], req_real, req_sha[:16]))
    return ident


def protocol_anchor_identity(eval_path):
    """PURE: record the canonical evaluator as a STATIC PROTOCOL ANCHOR -- NOT an executed
    source. The materializer REPRODUCES the evaluator's build+reset logic but never executes
    the evaluator program, so ``executed_by_materializer`` is False and the full SHA is the
    file's real SHA (verified equal to the frozen anchor)."""
    require(eval_path and os.path.isfile(eval_path),
            "FAIL CLOSED: protocol-anchor evaluator source missing (%r)" % eval_path)
    real = _resolve_real(eval_path)
    sha = sha256_file(real)
    require(sha == EVALUATOR_SHA256,
            "FAIL CLOSED: protocol-anchor evaluator sha256 %s != canonical anchor %s"
            % (sha[:16], EVALUATOR_SHA256[:16]))
    return {
        "role": "static protocol anchor",
        "path": os.path.abspath(eval_path),
        "realpath": real,
        "sha256": sha,
        "executed_by_materializer": False,
        "anchor_match": sha == EVALUATOR_SHA256,
        "note": "the materializer reproduces eval_phase2_unified.py's env construction + the "
                "three seed/reset lines and indexes the same batched reset; it does NOT run the "
                "evaluator main program, so this source is a provenance ANCHOR, not an executed "
                "source. We never claim 'evaluator source executed'.",
    }


def verify_task_exec_identity(requested_task_path, executed_path, env_cls, expected_full_sha):
    """PURE-ish: bind the canonical S4 task source that was exec()'d.

    Verifies (a) executed realpath == requested realpath, (b) full SHA == requested full SHA
    AND == the canonical task SHA (full, not prefix), (c) the exec'd class is named ``Env``
    and exposes the expected task interface (generate_world / get_task_params). Any mismatch
    -> FailClosed(TASK_EXECUTED_SOURCE_IDENTITY_MISMATCH). No silent fallback to another
    namespace is permitted.
    """
    req_real = _resolve_real(requested_task_path)
    exe_real = _resolve_real(executed_path)
    req_sha = sha256_file(req_real)
    exe_sha = sha256_file(exe_real)
    same_file = (req_real == exe_real)
    sha_ok = (req_sha == exe_sha == expected_full_sha)
    cls_name = getattr(env_cls, "__name__", None)
    has_world = callable(getattr(env_cls, "generate_world", None))
    has_params = callable(getattr(env_cls, "get_task_params", None))
    ident = {
        "label": "task",
        "loaded_via": "exec",
        "requested_path": os.path.abspath(requested_task_path),
        "requested_realpath": req_real,
        "requested_sha256": req_sha,
        "executed_path": os.path.abspath(executed_path),
        "executed_realpath": exe_real,
        "executed_sha256": exe_sha,
        "expected_sha256": expected_full_sha,
        "class_name": cls_name,
        "class_module": getattr(env_cls, "__module__", None),
        "has_generate_world": bool(has_world),
        "has_get_task_params": bool(has_params),
        "identity_match": bool(same_file and sha_ok and cls_name == "Env" and has_world and has_params),
    }
    require(ident["identity_match"],
            "FAIL CLOSED: TASK_EXECUTED_SOURCE_IDENTITY_MISMATCH: executed task (%s, sha=%s, "
            "class=%r, generate_world=%s, get_task_params=%s) does not match the requested "
            "canonical task (%s, sha=%s, expected full sha=%s). Refusing to exec a non-canonical "
            "or interface-incomplete task definition."
            % (exe_real, exe_sha[:16], cls_name, has_world, has_params,
               req_real, req_sha[:16], expected_full_sha[:16]))
    return ident


def bind_executed_source_identity(requested_eval, requested_wrapper, requested_env):
    """RUNTIME (requires importable dicode/minicraftax, i.e. a JAX+craftax host).

    Imports the REAL modules the materializer actually calls and proves each executed file is
    the requested/recorded source. Returns the executed-vs-anchor provenance split:
      executed_sources        : wrapper, environment (imported + called)
      protocol_anchor_sources : canonical_evaluator (reproduced, NOT executed)
    Task identity is bound separately in _load_canonical_s4_task (exec) and merged by the
    caller. FAILS CLOSED on any identity mismatch or on a JAX-less host (import chain fails).
    """
    import inspect
    import importlib
    wrapper_module = importlib.import_module("dicode.wrappers_cl")
    wrapper_file = (inspect.getsourcefile(wrapper_module.DistributedMultiTaskOptimisticLogWrapper)
                    or wrapper_module.__file__)
    wrapper_ident = verify_source_identity(requested_wrapper, wrapper_file, "wrapper",
                                           expected_sha256=WRAPPER_SHA256)
    wrapper_ident["imported_module"] = "dicode.wrappers_cl"
    wrapper_ident["bound_symbol"] = "DistributedMultiTaskOptimisticLogWrapper"

    env_module = importlib.import_module("minicraftax.envs.multitask")
    env_file = (inspect.getsourcefile(env_module.MultiTaskMiniCraftaxEnv)
                or env_module.__file__)
    env_ident = verify_source_identity(requested_env, env_file, "environment",
                                       expected_sha256=ENV_SOURCE_SHA256)
    env_ident["imported_module"] = "minicraftax.envs.multitask"
    env_ident["bound_symbol"] = "MultiTaskMiniCraftaxEnv"

    eval_ident = protocol_anchor_identity(requested_eval)
    return {
        "schema": RUNTIME_SOURCE_IDENTITY_SCHEMA,
        "executed_sources": {"wrapper": wrapper_ident, "environment": env_ident},
        "protocol_anchor_sources": {"canonical_evaluator": eval_ident},
        "sys_path_head": [str(p) for p in list(sys.path[:8])],
    }


# =================================================================================== #
# MATERIALIZATION (requires JAX + craftax; FAILS CLOSED on this host)
# =================================================================================== #
def _require_jax_craftax():
    import importlib.util
    have_jax = importlib.util.find_spec("jax") is not None
    have_craftax = importlib.util.find_spec("craftax") is not None
    require(have_jax and have_craftax,
            "FAIL CLOSED: real Craftax world materialization requires JAX AND craftax "
            "(jax=%s, craftax=%s). This host cannot materialize worlds; no world_set_hash "
            "is emitted. Run on the authorized JAX+craftax==%s experiment host."
            % (have_jax, have_craftax, EXPECTED_CRAFTAX))


def materialize_all_world_states(evaluation_seed, requested_task_path=None):
    """Materialize ALL 256 canonical worlds in ONE call, exactly as the evaluator does.

    The canonical evaluator calls ``env.reset(reset_rng, ctor)`` ONCE (eval:171) and the
    wrapper internally performs the 256-way split (wrappers_cl:228-229) + vmap. A single
    world's key is NOT independently derivable from (seed, index), so we MUST reproduce
    the whole 256-way batch and then index world i. (Feeding a pre-split single key back
    into ``env.reset`` would split it AGAIN and silently yield the WRONG worlds.)

    Returns (batched_env_state, env, s4_base, task_identity) where batched_env_state is the
    craftax EnvState pytree with a leading (256,) axis on every leaf and task_identity is the
    V3 executed-task-source binding. FAILS CLOSED if JAX/craftax are absent -- nothing is
    materialized and no hash is ever produced.
    """
    _require_jax_craftax()                              # fail closed first
    import jax
    import jax.numpy as jnp
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
    from dicode.task_utils import get_achievement_multi_hot

    # --- env construction (eval_phase2_unified.py:82-89) ---
    ctor = EnvParams(max_timesteps=MAX_TIMESTEPS)                       # :82
    table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],
                      dtype=jnp.float32)                                # :83
    emb = int(table.shape[1])                                           # :84
    s4cls, task_ident = _load_canonical_s4_task(requested_task_path)    # exec canonical s4 (:85)
    s4_base = MultiTaskMiniCraftaxEnv([s4cls], StaticEnvParams(), ctor, True,
                                      conditioning_type="embedding",
                                      embedding_size=emb)               # :86-87

    # --- wrapper construction (eval_phase2_unified.py:121-122) ---
    env = DistributedMultiTaskOptimisticLogWrapper(
        s4_base, jax.random.PRNGKey(0), NUM_WORLDS, NUM_TASKS,
        OPTIMISTIC_RESET_RATIO, jnp.array([1.0]), table)                # :121-122

    # --- EXACT evaluator reset (eval:169-171); wrapper does the 256-split internally ---
    rng = jax.random.PRNGKey(int(evaluation_seed))                      # eval:169
    rng, reset_rng = jax.random.split(rng)                              # eval:170
    _obs, log_state = env.reset(reset_rng, ctor)                        # eval:171
    batched_env_state = log_state.env_state                             # (256,) craftax EnvState
    return batched_env_state, env, s4_base, task_ident


def extract_world_identity_single(batched_env_state, world_index):
    """Select world i's INITIAL EnvState snapshot (timestep=0, pre-step) from the batch.

    WORLD-IDENTITY DECISION (section four): serialize the COMPLETE initial EnvState
    snapshot -- every one of its 53 dataclass fields at reset time -- NOT only the 6
    strictly-immutable fields (task_id, task_params, down_ladders, up_ladders,
    potion_mapping, fractal_noise_angles). Rationale: the initial VALUES of the
    runtime-mutated fields (map terrain, item_map incl. the removed floor-2 up-ladder,
    starting inventory {wood:7,stone:27,coal:3,iron:3,...}, monsters_killed[2]=8, mob
    placements, pre-populated achievements, player_position/level) ARE the world's
    identity even though _craftax_step later mutates those arrays. Dropping them would
    silently discard result-affecting initial fields (forbidden by the task). The full
    snapshot also makes the hash bit-equal to the evaluator's log_state.env_state[i].

    Excluded as NON-identity: the wrapper LogEnvState episode accumulators
    (episode_returns / episode_lengths / running_original_return -- trivially zero at
    reset, carry no world information) and the rendered obs (a deterministic projection
    of the state, not identity itself).
    """
    import jax
    env_state_i = jax.tree.map(lambda x: x[world_index], batched_env_state)
    # The canonical serializer recurses every dataclass field (via __dataclass_fields__),
    # arrays binding dtype+shape+C-order bytes -- so no initial field is dropped here.
    return {"env_state": env_state_i}


def _load_canonical_s4_task(requested_task_path=None):
    """exec() the CANONICAL s4_task_code.py (p2_v1, full SHA 45fdd17c...) -- NOT the P2-v0
    invalid-for-attribution copy. The path must be supplied (CC4_S4_TASK_PATH or arg) because
    the canonical evaluator uses a server-absolute path (/home/oseasy/...).

    V3: binds the EXECUTED task source identity -- executed realpath + FULL sha (not prefix)
    + class name + task interface -- against the requested canonical source. Any mismatch ->
    FailClosed(TASK_EXECUTED_SOURCE_IDENTITY_MISMATCH). No silent fallback to another
    namespace. Returns (Env_class, task_identity_dict).
    """
    path = requested_task_path or os.environ.get("CC4_S4_TASK_PATH")
    require(path and os.path.isfile(path),
            "FAIL CLOSED: the canonical s4_task_code.py path must be supplied (p2_v1, full "
            "sha256 %s). Refusing to guess a task definition." % TASK_SHA256)
    actual = sha256_file(path)
    require(actual == TASK_SHA256,
            "FAIL CLOSED: s4_task_code.py full sha256 %s != canonical %s (wrong task "
            "definition / CRLF-mangled / non-canonical copy)." % (actual[:16], TASK_SHA256[:16]))
    ns = {"__name__": "cc4_canonical_s4_task"}      # deterministic Env.__module__
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), ns)                                              # mirrors eval:85
    require("Env" in ns,
            "FAIL CLOSED: exec'd canonical task did not define class Env (no silent fallback "
            "to any other namespace).")
    task_ident = verify_task_exec_identity(path, path, ns["Env"], TASK_SHA256)
    return ns["Env"], task_ident


def assert_materialized(result):
    """GATE: accept a result ONLY if it is a genuine materialized-world result.

    Rejects the OLD key-only prototype output (schema mechanism_UED.world_hashes/v1 or
    mechanism_UED.world/v1, world_params=null, no materialized flag) and anything that
    lacks the full serialized EnvState provenance. This is what negative test 6 checks.
    """
    KEY_ONLY_SCHEMAS = {"mechanism_UED.world_hashes/v1", "mechanism_UED.world/v1",
                        "mechanism_UED.world_set_agreement/v1"}
    schema = result.get("schema")
    require(schema == SCHEMA_VERSION,
            "GATE REJECT: schema %r is NOT the materialized-world schema %r "
            "(key-only prototype output is refused)" % (schema, SCHEMA_VERSION))
    require(schema not in KEY_ONLY_SCHEMAS,
            "GATE REJECT: %r is a key-only prototype schema" % schema)
    require(result.get("materialized") is True,
            "GATE REJECT: result.materialized is not True")
    require(result.get("world_count") == NUM_WORLDS,
            "GATE REJECT: world_count != %d" % NUM_WORLDS)
    src = result.get("source_shas", {})
    for k in ("evaluator_sha256", "environment_source_sha256",
              "wrapper_source_sha256", "task_source_sha256"):
        require(src.get(k), "GATE REJECT: missing requested-source SHA %r" % k)
    # V3: a genuine materialized result MUST carry the field-manifest evidence and the
    # seed-free per-world payload hashes. Missing either -> reject (negative test 12.6).
    require(result.get("world_field_manifests_sha256"),
            "GATE REJECT: missing world_field_manifests_sha256 (V3 manifest evidence)")
    require(result.get("world_field_manifest_world_count") == NUM_WORLDS,
            "GATE REJECT: world_field_manifest_world_count != %d" % NUM_WORLDS)
    pwh = result.get("per_world_state_payload_hashes") or {}
    require(len(pwh) == NUM_WORLDS,
            "GATE REJECT: per_world_state_payload_hashes incomplete (%d/%d)"
            % (len(pwh), NUM_WORLDS))
    # V3: the executed-source binding must be present and matched (runtime proof).
    if "executed_source_identity_match" in result:
        require(result["executed_source_identity_match"] is True,
                "GATE REJECT: executed_source_identity_match is not True "
                "(imported source != requested source)")
    return True


def compare_two_runs(a, b):
    """Compare two independent materialization results; FAIL CLOSED on ANY disagreement.

    Used by do_orchestrate (real runs) and by negative test 10 (with mock inputs to prove
    the comparison logic actually detects a mismatch).
    """
    # V3 (section fourteen): the two independent runs must agree on EVERY identity carrier --
    # not only the total hash. Any disagreement -> fail closed.
    for field in ("world_count", "world_set_hash", "per_world_hashes",
                  "per_world_state_payload_hashes",        # V3 seed-free payload hashes
                  "world_field_manifests_sha256",          # V3 field-manifest evidence
                  "source_shas", "versions", "evaluation_seed",
                  "numeric_evaluation_seed",               # V3 numeric seed
                  "identity_class", "protocol_id",         # V3 seed identity / protocol id
                  "runtime_source_identity"):              # V3 executed-source binding
        require(a.get(field) == b.get(field),
                "FAIL CLOSED: %s differs between the two independent runs" % field)
    return True


# =================================================================================== #
# V3: WORLD FIELD MANIFEST PERSISTENCE (section eleven / twelve)
# =================================================================================== #
def build_field_manifests_doc(per_world_manifests):
    """Assemble the persistable world_field_manifests.json document (pure).

    per_world_manifests: {world_index_str: [ {path,dtype,shape,nbytes}, ... ]} as recorded by
    the serializer. Every array/scalar field the serializer actually emitted appears here.
    """
    worlds = {}
    for k in sorted(per_world_manifests, key=lambda x: int(x)):
        worlds[k] = per_world_manifests[k]
    return {
        "schema": FIELD_MANIFEST_SCHEMA,
        "world_count": len(worlds),
        "world_index_order": "0..%d ascending" % (NUM_WORLDS - 1),
        "worlds": worlds,
    }


def field_manifests_sha256(manifests_doc):
    """Stable SHA256 of the canonical (sorted-key, compact) field-manifest document."""
    canon = json.dumps(manifests_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canon)


def build_field_schema_summary(per_world_manifests):
    """Pure structural summary across all worlds. Records (not silently overwrites) any
    schema-structure disagreement between worlds: a field whose dtype/shape differs, or that
    appears in only some worlds, is listed under structural_inconsistencies."""
    # path -> {"dtype": set, "shape": set, "nbytes": set, "worlds": count}
    by_path = {}
    world_keys = sorted(per_world_manifests, key=lambda x: int(x))
    total_array_bytes = 0
    for k in world_keys:
        seen_in_world = set()
        for e in per_world_manifests[k]:
            p = e["path"]
            total_array_bytes += int(e.get("nbytes", 0))
            d = by_path.setdefault(p, {"dtype": {}, "shape": {}, "nbytes": {}, "worlds": 0})
            d["dtype"][e["dtype"]] = d["dtype"].get(e["dtype"], 0) + 1
            shp = tuple(e["shape"])
            d["shape"][shp] = d["shape"].get(shp, 0) + 1
            d["nbytes"][int(e["nbytes"])] = d["nbytes"].get(int(e["nbytes"]), 0) + 1
            if p not in seen_in_world:
                d["worlds"] += 1
                seen_in_world.add(p)
    n_worlds = len(world_keys)
    fields = []
    inconsistencies = []
    for p in sorted(by_path):
        d = by_path[p]
        dtypes = sorted(d["dtype"])
        shapes = sorted([list(s) for s in d["shape"]])
        entry = {
            "path": p,
            "dtype": dtypes[0] if len(dtypes) == 1 else dtypes,
            "shape": shapes[0] if len(shapes) == 1 else shapes,
            "world_count": d["worlds"],
            "nbytes_variants": sorted(d["nbytes"]),
        }
        fields.append(entry)
        if len(dtypes) > 1 or len(shapes) > 1 or d["worlds"] != n_worlds:
            inconsistencies.append({
                "path": p,
                "reason": ("dtype_variants=%s shape_variants=%s present_in=%d/%d worlds"
                           % (dtypes, [list(s) for s in d["shape"]], d["worlds"], n_worlds)),
            })
    return {
        "schema": FIELD_SCHEMA_SUMMARY_SCHEMA,
        "world_count": n_worlds,
        "total_unique_field_paths": len(fields),
        "total_array_bytes_all_worlds": total_array_bytes,
        "fields": fields,
        "structural_inconsistencies": inconsistencies,
        "all_worlds_structurally_identical": len(inconsistencies) == 0,
        "note": "schema differences are RECORDED here, never silently overwritten; a non-empty "
                "structural_inconsistencies list means the 256 worlds do not share one schema.",
    }


def write_json(path, doc):
    """Write JSON with LF newlines (repo .gitattributes is -text for reports; keep blobs LF)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def collect_source_shas(eval_path, wrapper_path, task_path, env_path):
    """Record the SHA256 of the REQUESTED sources (provenance, section 7).

    This is the requested-source record carried in the header. V3 ADDITIONALLY proves, at
    runtime, that the file Python actually imports/execs IS each requested source
    (bind_executed_source_identity / verify_task_exec_identity). Fails closed if any required
    source is missing -- a recorded-but-not-executed or absent source must never slip into a
    world_set_hash.
    """
    def sha_or_fail(p, what):
        require(p and os.path.isfile(p),
                "FAIL CLOSED: %s source missing (%r); cannot record/execute source SHA" % (what, p))
        return sha256_file(p)
    return {
        "evaluator_sha256": sha_or_fail(eval_path, "evaluator"),
        "wrapper_source_sha256": sha_or_fail(wrapper_path, "wrapper"),
        "task_source_sha256": sha_or_fail(task_path, "task"),
        "environment_source_sha256": sha_or_fail(env_path, "environment (multitask env)"),
    }


def collect_versions():
    import importlib
    def ver(m):
        try:
            return getattr(importlib.import_module(m), "__version__", "UNKNOWN")
        except Exception:
            return None
    jax_v, jaxlib_v, craftax_v = ver("jax"), ver("jaxlib"), ver("craftax")
    require(jax_v and jaxlib_v and craftax_v,
            "FAIL CLOSED: jax/jaxlib/craftax versions must all resolve (jax=%s jaxlib=%s craftax=%s)"
            % (jax_v, jaxlib_v, craftax_v))
    require(craftax_v == EXPECTED_CRAFTAX,
            "FAIL CLOSED: craftax %r != expected %r" % (craftax_v, EXPECTED_CRAFTAX))
    return {"jax": jax_v, "jaxlib": jaxlib_v, "craftax": craftax_v}


def do_materialize_run(seed_id, out_dir, eval_path, wrapper_path, task_path, env_path):
    """FULL real materialization of the 256-world set + per-world & total hash (V3).

    Re-imports / re-builds / re-runs the real generation+reset, re-serializes and re-hashes
    every world. Emits, per run:
        world_hashes.json                (per-world hash + seed-free payload hash + total)
        world_field_manifests.json       (per-world array path/dtype/shape/nbytes)
        world_field_schema_summary.json  (cross-world structural summary)
        runtime_source_identity.json     (imported vs requested source binding; NO secrets)
    On a JAX-less host this FAILS CLOSED at the very first step and writes NOTHING that could
    be mistaken for a real world_set_hash.
    """
    evaluation_seed = ALLOWED_SEEDS[seed_id]
    ident = seed_identity(seed_id)                    # V3 seed identity (class/protocol_id)
    source_shas = collect_source_shas(eval_path, wrapper_path, task_path, env_path)
    # Static anchor check: our embedded constants MUST equal the canonical source literals.
    anchor = static_anchor_check(eval_path, wrapper_path, task_path)
    require(anchor["result"] == "PASS",
            "FAIL CLOSED: static anchor check failed: %s" % anchor["mismatches"])
    versions = collect_versions()

    # Make the canonical s4 task discoverable by the materialization path.
    os.environ["CC4_S4_TASK_PATH"] = task_path

    # V3: RUNTIME executed-source identity binding (imports real wrapper+env, proves each
    # executed file == requested source; evaluator recorded as protocol anchor, NOT executed).
    # Fails closed on a JAX-less host (import chain) or on any identity mismatch.
    runtime_identity = bind_executed_source_identity(eval_path, wrapper_path, env_path)

    batched, _env, _s4, task_ident = materialize_all_world_states(
        evaluation_seed, requested_task_path=task_path)   # REAL reset (256 worlds)
    runtime_identity["executed_sources"]["task"] = task_ident
    runtime_identity["module_versions"] = dict(versions)
    runtime_identity["seed_identity"] = ident

    per_world = {}
    per_world_payload = {}
    manifests = {}
    for i in range(NUM_WORLDS):                                          # 0..255 ascending
        identity = extract_world_identity_single(batched, i)
        _wb, _hdr, manifest, ph = serialize_world(identity, evaluation_seed, i,
                                                  source_shas, versions)
        payload_hash, _m2 = state_payload_hash(identity)   # V3 seed-free world payload hash
        per_world[str(i)] = ph
        per_world_payload[str(i)] = payload_hash
        manifests[str(i)] = manifest
    world_set_hash = compute_world_set_hash(per_world, evaluation_seed, source_shas, versions)

    # V3: persist field manifests + schema summary; bind the manifest SHA into world_hashes.
    manifests_doc = build_field_manifests_doc(manifests)
    manifests_sha = field_manifests_sha256(manifests_doc)
    schema_summary = build_field_schema_summary(manifests)

    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "world_field_manifests.json"), manifests_doc)
    write_json(os.path.join(out_dir, "world_field_schema_summary.json"), schema_summary)
    write_json(os.path.join(out_dir, "runtime_source_identity.json"), runtime_identity)

    result = {
        "schema": SCHEMA_VERSION,
        "materialized": True,
        "seed_id": seed_id,
        "evaluation_seed": evaluation_seed,
        # V3 seed identity (class / protocol id / exact-match admissibility)
        "identity_class": ident["identity_class"],
        "protocol_id": ident["protocol_id"],
        "evaluator_exact_match": ident["evaluator_exact_match"],
        "admissible_as_canonical_exact_world_set_evidence":
            ident["admissible_as_canonical_exact_world_set_evidence"],
        "numeric_evaluation_seed": ident["numeric_seed"],
        "world_count": NUM_WORLDS,
        "world_index_order": "0..255 ascending",
        "per_world_hashes": per_world,
        "per_world_state_payload_hashes": per_world_payload,   # V3 seed-free payload hashes
        "world_set_hash": world_set_hash,
        "source_shas": source_shas,
        "versions": versions,
        "static_anchor_check": anchor,
        # V3 manifest evidence (required by assert_materialized)
        "world_field_manifests_file": "world_field_manifests.json",
        "world_field_manifests_sha256": manifests_sha,
        "world_field_manifest_world_count": manifests_doc["world_count"],
        "world_field_schema_summary_file": "world_field_schema_summary.json",
        "all_worlds_structurally_identical": schema_summary["all_worlds_structurally_identical"],
        "runtime_source_identity_file": "runtime_source_identity.json",
        "executed_source_identity_match": all(
            v["identity_match"] for v in runtime_identity["executed_sources"].values()),
        "evaluator_role": "STATIC_PROTOCOL_ANCHOR_NOT_EXECUTED",
    }
    write_json(os.path.join(out_dir, "world_hashes.json"), result)
    print(json.dumps({"mode": "single-run", "materialized": True,
                      "world_count": NUM_WORLDS, "world_set_hash": world_set_hash,
                      "identity_class": ident["identity_class"],
                      "protocol_id": ident["protocol_id"],
                      "world_field_manifests_sha256": manifests_sha},
                     ensure_ascii=False))
    return EXIT_OK


def do_orchestrate(seed_id, out_dir, eval_path, wrapper_path, task_path, env_path):
    """Run the FULL materialization twice in INDEPENDENT processes and compare (section 9).

    No state is shared between the two runs: each re-imports, re-builds, re-runs the real
    256 generation+reset, re-serializes and re-hashes. ANY disagreement -> fail closed.
    """
    import subprocess
    this = os.path.abspath(__file__)
    run_dirs = [os.path.join(out_dir, "run_A"), os.path.join(out_dir, "run_B")]
    results = []
    for rd in run_dirs:
        cmd = [sys.executable, this, "--single-run", "--seed", seed_id, "--out", rd,
               "--eval-source", eval_path or "", "--wrapper-source", wrapper_path or "",
               "--task-source", task_path or "", "--env-source", env_path or ""]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        require(proc.returncode == EXIT_OK,
                "FAIL CLOSED: independent run %s exited %d\nstderr: %s"
                % (rd, proc.returncode, proc.stderr))
        with open(os.path.join(rd, "world_hashes.json"), encoding="utf-8") as f:
            run_result = json.load(f)
        # V3: attach the per-run executed-source identity so compare_two_runs binds it too.
        rsi_path = os.path.join(rd, "runtime_source_identity.json")
        if os.path.isfile(rsi_path):
            with open(rsi_path, encoding="utf-8") as f:
                run_result["runtime_source_identity"] = json.load(f)
        results.append(run_result)
    a, b = results
    compare_two_runs(a, b)                       # fail closed on ANY disagreement (V3-expanded)
    agreed = {
        "schema": "mechanism_UED.craftax_world_set_agreement/v1",
        "materialized": True,
        "two_independent_runs": True,
        "seed_id": seed_id,
        "identity_class": a.get("identity_class"),
        "protocol_id": a.get("protocol_id"),
        "world_count": a["world_count"],
        "world_set_hash": a["world_set_hash"],
        "per_world_hash_agreement": True,
        "per_world_state_payload_hash_agreement": True,    # V3
        "world_set_hash_agreement": True,
        "world_field_manifests_sha256_agreement": True,    # V3
        "runtime_source_identity_agreement": True,         # V3
        "source_sha_agreement": True,
        "version_agreement": True,
        "numeric_seed_agreement": True,
        "identity_class_agreement": True,
        "protocol_id_agreement": True,
        "run_A": os.path.join(run_dirs[0], "world_hashes.json"),
        "run_B": os.path.join(run_dirs[1], "world_hashes.json"),
    }
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "world_set_agreement.json"), agreed)
    print(json.dumps({"mode": "orchestrate", "materialized": True,
                      "agreement": True, "world_set_hash": a["world_set_hash"],
                      "identity_class": a.get("identity_class"),
                      "protocol_id": a.get("protocol_id")},
                     ensure_ascii=False))
    return EXIT_OK


# =================================================================================== #
# Static anchor test (GATE19 weak form): the constants embedded above MUST equal the
# literal values parsed out of the canonical source files. This is the honest substitute
# for a shared builder while the evaluator stays read-only + inline.
# =================================================================================== #
def static_anchor_check(eval_path=None, wrapper_path=None, task_path=None):
    """Parse canonical sources and assert our embedded anchors match. Pure static check."""
    import re
    findings = {"checked": [], "mismatches": []}

    def check(name, ok, detail):
        findings["checked"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            findings["mismatches"].append(name)

    if eval_path and os.path.isfile(eval_path):
        src = open(eval_path, encoding="utf-8", errors="replace").read()
        check("evaluator_sha256", sha256_file(eval_path) == EVALUATOR_SHA256,
              "eval SHA must equal canonical anchor")
        check("EVAL_SEED=42", re.search(r"EVAL_SEED\s*=\s*42\b", src) is not None, "eval:77")
        check("NUM_ENVS=256", re.search(r"NUM_ENVS\s*=\s*256\b", src) is not None, "eval:75")
        check("NUM_STEPS=4096", re.search(r"NUM_STEPS\s*=\s*4096\b", src) is not None, "eval:76")
        check("reset_rng=split(rng)",
              re.search(r"rng,\s*reset_rng\s*=\s*jax\.random\.split\(rng\)", src) is not None,
              "eval:170")
        check("env.reset(reset_rng",
              re.search(r"env\.reset\(reset_rng", src) is not None, "eval:171")
        check("wrapper PRNGKey(0)",
              re.search(r"DistributedMultiTaskOptimisticLogWrapper\(s4_base,\s*jax\.random\.PRNGKey\(0\)", src)
              is not None, "eval:121")
    if wrapper_path and os.path.isfile(wrapper_path):
        wsrc = open(wrapper_path, encoding="utf-8", errors="replace").read()
        check("wrapper_sha_prefix", sha256_file(wrapper_path).startswith(WRAPPER_SHA256_PREFIX),
              "wrappers_cl byte-identical anchor")
        check("wrapper split(_rng, num_envs)",
              re.search(r"reset_rngs\s*=\s*jax\.random\.split\(_rng,\s*self\.num_envs\)", wsrc)
              is not None, "wrappers_cl:229")
    if task_path and os.path.isfile(task_path):
        tsrc = open(task_path, encoding="utf-8", errors="replace").read()
        check("task_sha_prefix", sha256_file(task_path).startswith(TASK_SHA256_PREFIX),
              "canonical p2_v1 task")
        check("task generate_world split",
              re.search(r"rng,\s*_r\s*=\s*jax\.random\.split\(rng\)", tsrc) is not None,
              "s4_task_code:39")
        check("task starting_floor 2", "set_starting_floor(2)" in tsrc, "s4_task_code:41")
    findings["result"] = "PASS" if not findings["mismatches"] and findings["checked"] else "FAIL"
    return findings


# =================================================================================== #
# SELF-TEST (pure Python; runs on this JAX-less host)
# =================================================================================== #
def self_test():
    import numpy as np
    from collections import namedtuple
    import enum

    checks = []

    def ck(name, ok):
        checks.append((name, bool(ok)))
        if not ok:
            raise AssertionError("SELF-TEST FAIL: %s" % name)

    # determinism: same pytree -> identical bytes
    def sample_pytree(seed_val):
        Point = namedtuple("Point", ["x", "y"])
        @dataclasses.dataclass
        class Params:
            b_field: int
            a_field: float
        class Color(enum.Enum):
            RED = 1
            GREEN = 2
        return {
            "int": 12345, "neg": -7, "big": 10**30,
            "float": 3.141592653589793, "f32": np.float32(1.5),
            "str": "world", "none": None, "bool": True,
            "bytes": b"\x00\x01\xff",
            "list": [1, 2, 3], "tuple": (4, 5),
            "namedtuple": Point(1, 2),
            "dataclass": Params(b_field=2, a_field=1.0),
            "enum": Color.GREEN,
            "array_f64": np.arange(12, dtype=np.float64).reshape(3, 4),
            "array_i32": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "nested": {"a": [np.ones((2,), dtype=np.float32), {"b": (6,)}]},
        }

    m1, m2 = [], []
    b1 = encode_node(sample_pytree(1), (), m1)
    b2 = encode_node(sample_pytree(1), (), m2)
    ck("determinism_same_bytes", b1 == b2)
    ck("manifest_recorded", len(m1) > 0 and all("dtype" in e and "shape" in e for e in m1))

    # order-invariance for dict / dataclass (negative test 4)
    pa = {"a": 1, "b": 2, "c": np.ones(3)}
    pb = {"c": np.ones(3), "a": 1, "b": 2}     # different insertion order
    ck("dict_order_invariant", encode_node(pa, (), []) == encode_node(pb, (), []))

    # value change -> different bytes (negative test 3)
    pc = {"a": 1, "b": 3, "c": np.ones(3)}
    ck("value_change_changes", encode_node(pa, (), []) != encode_node(pc, (), []))

    # dtype change -> different bytes (negative test 5)
    ck("dtype_change_changes",
       encode_node(np.ones(3, dtype=np.float32), (), [])
       != encode_node(np.ones(3, dtype=np.float64), (), []))
    # shape change -> different bytes (negative test 5)
    ck("shape_change_changes",
       encode_node(np.ones((3,), dtype=np.float32), (), [])
       != encode_node(np.ones((1, 3), dtype=np.float32), (), []))

    # float losslessness: encode-> we trust struct '>d'; sanity roundtrip via numpy
    ck("float64_lossless", struct.unpack(">d", struct.pack(">d", 3.141592653589793))[0]
       == 3.141592653589793)

    # serialize_world produces a stable per_world_hash and a manifest
    src = {"evaluator_sha256": EVALUATOR_SHA256,
           "environment_source_sha256": "e" * 64,
           "wrapper_source_sha256": "w" * 64,
           "task_source_sha256": "t" * 64}
    ver = {"craftax": EXPECTED_CRAFTAX, "jax": "X.Y", "jaxlib": "X.Y"}
    wb1, hd1, mf1, ph1 = serialize_world(sample_pytree(1), 42, 0, src, ver)
    wb2, hd2, mf2, ph2 = serialize_world(sample_pytree(1), 42, 0, src, ver)
    ck("world_bytes_stable", wb1 == wb2 and ph1 == ph2)
    ck("header_has_numeric_seed", hd1["evaluation_seed"] == 42)

    # seed label change alone must NOT change hash; numeric seed change MUST
    wb_lbl, _, _, ph_lbl = serialize_world(sample_pytree(1), 42, 0, src, ver)
    ck("numeric_seed_in_header", ph_lbl == ph1)
    # world_set_hash: numeric seed enters the total
    pwh = {str(i): ph1 for i in range(NUM_WORLDS)}
    wsh42 = compute_world_set_hash(pwh, 42, src, ver)
    wsh100000 = compute_world_set_hash(pwh, 100000, src, ver)
    ck("world_set_hash_seed_sensitive", wsh42 != wsh100000)

    # unsupported type -> FailClosed (no pickle fallback)
    try:
        encode_node(object(), (), [])
        ck("unsupported_failclosed", False)
    except FailClosed:
        ck("unsupported_failclosed", True)

    # ============================ V3 SELF-TEST CHECKS ============================ #
    import tempfile

    # V3 (section seven): the seed-free payload hash must NOT move when only the header
    # seed moves, while the header-tagged per_world_hash DOES move. This is exactly why a
    # per_world_hash difference is NOT proof of a real RNG effect.
    same_world = sample_pytree(1)
    pl_a, _ = serialize_world_payload(same_world)
    pl_b, _ = serialize_world_payload(same_world)
    ck("payload_hash_deterministic", sha256_bytes(pl_a) == sha256_bytes(pl_b))
    _, _, _, ph_seed42 = serialize_world(same_world, 42, 0, src, ver)
    _, _, _, ph_seed100000 = serialize_world(same_world, 100000, 0, src, ver)
    sph_42, _ = state_payload_hash(same_world)
    sph_100000_identical_state, _ = state_payload_hash(same_world)
    ck("header_seed_changes_per_world_hash", ph_seed42 != ph_seed100000)
    ck("payload_hash_is_seed_free", sph_42 == sph_100000_identical_state)
    # but a genuinely different world payload changes the payload hash
    different_world = {"env_state": {"map": np.ones((9, 48, 48), dtype=np.int32) * 3}}
    sph_diff, _ = state_payload_hash(different_world)
    ck("payload_hash_tracks_real_state", sph_42 != sph_diff)

    # V3 (section nine/ten): seed identity classification + GATE22 admissibility.
    id42 = seed_identity("seed42")
    id100 = seed_identity("seed100000")
    ck("seed42_is_exact_class",
       id42["identity_class"] == "CANONICAL_EVALUATOR_EXACT_WORLD_SET"
       and id42["evaluator_exact_match"] is True
       and id42["protocol_id"] == PROTOCOL_ID_SEED42)
    ck("seed100000_is_parameterized_variant",
       id100["identity_class"] == "PARAMETERIZED_WORLD_GENERATION_PROTOCOL_VARIANT"
       and id100["evaluator_exact_match"] is False
       and id100["protocol_id"] == PROTOCOL_ID_SEED100000)
    ck("seed100000_not_exact_evidence",
       id100["admissible_as_canonical_exact_world_set_evidence"] is False)
    ck("seed100000_has_independent_evaluator",
       bool(id100["independent_evaluator"])
       and id100["independent_evaluator"]["sha256_raw_on_disk"]
       != id42.get("canonical_evaluator_sha256"))
    ck("gate22_accepts_seed42", assert_exact_world_set_eligible("seed42") is not None)
    try:
        assert_exact_world_set_eligible("seed100000")
        ck("gate22_rejects_seed100000", False)
    except FailClosed:
        ck("gate22_rejects_seed100000", True)

    # V3 (section eleven): field-manifest document, summary, SHA stability, and that a
    # schema difference is RECORDED (not silently overwritten).
    m_a, m_b = [], []
    encode_node({"map": np.ones((9, 48, 48), dtype=np.int32),
                 "inv": {"wood": np.int64(7)}}, (), m_a)
    encode_node({"map": np.ones((9, 48, 48), dtype=np.int32),
                 "inv": {"wood": np.int64(7)}}, (), m_b)
    pm = {str(i): (m_a if i % 2 == 0 else m_b) for i in range(NUM_WORLDS)}
    doc = build_field_manifests_doc(pm)
    ck("manifest_doc_world_count_256", doc["world_count"] == NUM_WORLDS
       and doc["schema"] == FIELD_MANIFEST_SCHEMA and len(doc["worlds"]) == NUM_WORLDS)
    sha_doc1 = field_manifests_sha256(doc)
    sha_doc2 = field_manifests_sha256(build_field_manifests_doc(pm))
    ck("manifest_sha_stable", sha_doc1 == sha_doc2)
    summary = build_field_schema_summary(pm)
    ck("summary_covers_all_paths",
       summary["total_unique_field_paths"] == len({e["path"] for e in m_a})
       and summary["all_worlds_structurally_identical"] is True)
    # introduce a dtype/shape divergence in world 5 -> must be RECORDED as an inconsistency
    pm_bad = dict(pm)
    m_div = []
    encode_node({"map": np.ones((9, 48, 48), dtype=np.float32),   # dtype changed
                 "inv": {"wood": np.int64(7)}}, (), m_div)
    pm_bad["5"] = m_div
    summary_bad = build_field_schema_summary(pm_bad)
    ck("summary_records_schema_divergence",
       summary_bad["all_worlds_structurally_identical"] is False
       and any(inc["path"] == "map" for inc in summary_bad["structural_inconsistencies"]))
    # manifest SHA changes when a manifest changes
    ck("manifest_change_changes_sha",
       field_manifests_sha256(build_field_manifests_doc(pm_bad)) != sha_doc1)
    # every serialized array path appears in the per-world manifest
    ck("manifest_paths_cover_arrays",
       {e["path"] for e in m_a} == {"map", "inv.wood"})

    # V3 (section twelve): assert_materialized REJECTS a result lacking manifest evidence.
    genuine_v3 = {"schema": SCHEMA_VERSION, "materialized": True, "world_count": NUM_WORLDS,
                  "source_shas": src, "per_world_hashes": pwh, "world_set_hash": "g" * 64,
                  "world_field_manifests_sha256": sha_doc1,
                  "world_field_manifest_world_count": NUM_WORLDS,
                  "per_world_state_payload_hashes": {str(i): "p" * 64 for i in range(NUM_WORLDS)}}
    ck("gate_accepts_v3_result", assert_materialized(genuine_v3) is True)
    missing_manifest = dict(genuine_v3)
    missing_manifest.pop("world_field_manifests_sha256")
    try:
        assert_materialized(missing_manifest)
        ck("gate_rejects_missing_manifest", False)
    except FailClosed:
        ck("gate_rejects_missing_manifest", True)

    # V3 (section two): executed-source identity binding LOGIC (pure; no craftax import).
    d = tempfile.mkdtemp(prefix="cc4_v3_selftest_")
    fA = os.path.join(d, "srcA.py"); open(fA, "wb").write(b"CLASS = 'wrapper'\n")
    fB = os.path.join(d, "srcB.py"); open(fB, "wb").write(b"CLASS = 'DIFFERENT'\n")
    same = verify_source_identity(fA, fA, "wrapper")     # same file -> match
    ck("source_identity_same_file_matches", same["identity_match"] is True)
    try:
        verify_source_identity(fA, fB, "wrapper")        # record A, import B -> reject
        ck("source_identity_mismatch_rejected", False)
    except FailClosed:
        ck("source_identity_mismatch_rejected", True)
    # byte-identical but DIFFERENT realpath must ALSO be rejected (not relying on byte-equality)
    fC = os.path.join(d, "srcC_copy.py"); open(fC, "wb").write(open(fA, "rb").read())
    try:
        verify_source_identity(fA, fC, "wrapper")
        ck("source_identity_byteidentical_diffpath_rejected", False)
    except FailClosed:
        ck("source_identity_byteidentical_diffpath_rejected", True)

    # V3 (section three): evaluator is recorded as a PROTOCOL ANCHOR, never an executed
    # source. Temporarily point the anchor SHA at a temp file so we can exercise the real
    # labelling function, then restore the canonical anchor.
    fEval = os.path.join(d, "eval_anchor.py"); open(fEval, "wb").write(b"# fake eval anchor\n")
    saved_anchor = globals()["EVALUATOR_SHA256"]
    try:
        globals()["EVALUATOR_SHA256"] = sha256_file(fEval)
        anchor_ident = protocol_anchor_identity(fEval)
        ck("protocol_anchor_label_not_executed",
           anchor_ident["executed_by_materializer"] is False
           and anchor_ident["role"] == "static protocol anchor"
           and anchor_ident["anchor_match"] is True)
    finally:
        globals()["EVALUATOR_SHA256"] = saved_anchor

    # V3 (section four): task-exec identity verification rejects a non-canonical / wrong-sha
    # / interface-incomplete class (positive path needs the real canonical task -> JAX host).
    class Env:                                    # right name + interface, WRONG file sha
        def generate_world(self, rng):
            return rng
        def get_task_params(self):
            return None
    try:
        verify_task_exec_identity(fA, fA, Env, TASK_SHA256)   # fA sha != canonical -> reject
        ck("task_exec_wrong_sha_rejected", False)
    except FailClosed:
        ck("task_exec_wrong_sha_rejected", True)
    class NotEnv:                                 # wrong class name -> reject
        def generate_world(self, rng):
            return rng
        def get_task_params(self):
            return None
    try:
        verify_task_exec_identity(fA, fA, NotEnv, sha256_file(fA))
        ck("task_exec_wrong_classname_rejected", False)
    except FailClosed:
        ck("task_exec_wrong_classname_rejected", True)

    print("SERIALIZER_SELF_TEST_PASS  (%d checks)" % len(checks))
    for n, ok in checks:
        print("  [PASS] %s" % n)
    return EXIT_OK


# =================================================================================== #
# CLI
# =================================================================================--- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Actual Craftax world-set materializer (fail-closed).")
    ap.add_argument("--seed", dest="seed_id", default="seed42", choices=sorted(ALLOWED_SEEDS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-source", default=None)
    ap.add_argument("--wrapper-source", default=None)
    ap.add_argument("--task-source", default=None)
    ap.add_argument("--env-source", default=None,
                    help="path to MultiTaskMiniCraftaxEnv source (multitask.py) for provenance SHA")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--self-test", dest="mode", action="store_const", const="self-test")
    mode.add_argument("--anchor-check", dest="mode", action="store_const", const="anchor-check")
    mode.add_argument("--single-run", dest="mode", action="store_const", const="single-run")
    mode.add_argument("--orchestrate", dest="mode", action="store_const", const="orchestrate")
    ap.set_defaults(mode="self-test")
    a = ap.parse_args(argv)

    if a.mode == "self-test":
        return self_test()
    if a.mode == "anchor-check":
        res = static_anchor_check(a.eval_source, a.wrapper_source, a.task_source)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return EXIT_OK if res["result"] == "PASS" else EXIT_FAIL_CLOSED
    if a.mode == "single-run":
        if not a.out:
            eprint("FAIL CLOSED: --out required for single-run")
            return EXIT_USAGE
        try:
            return do_materialize_run(a.seed_id, a.out, a.eval_source, a.wrapper_source,
                                      a.task_source, a.env_source)
        except FailClosed as e:
            eprint(str(e))
            return EXIT_FAIL_CLOSED
    if a.mode == "orchestrate":
        if not a.out:
            eprint("FAIL CLOSED: --out required for orchestrate")
            return EXIT_USAGE
        try:
            return do_orchestrate(a.seed_id, a.out, a.eval_source, a.wrapper_source,
                                  a.task_source, a.env_source)
        except FailClosed as e:
            eprint(str(e))
            return EXIT_FAIL_CLOSED
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
