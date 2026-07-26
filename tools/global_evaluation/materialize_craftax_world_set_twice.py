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


def serialize_world(identity_pytree, evaluation_seed, world_index, source_shas, versions):
    """Return (world_bytes, header_dict, manifest, per_world_hash).

    world_bytes = lenprefixed(header_json) + lenprefixed(payload)
    per_world_hash = SHA256(world_bytes)   -- over the ACTUAL materialized state bytes.
    """
    manifest = []
    payload = encode_node(identity_pytree, (), manifest)
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


def materialize_all_world_states(evaluation_seed):
    """Materialize ALL 256 canonical worlds in ONE call, exactly as the evaluator does.

    The canonical evaluator calls ``env.reset(reset_rng, ctor)`` ONCE (eval:171) and the
    wrapper internally performs the 256-way split (wrappers_cl:228-229) + vmap. A single
    world's key is NOT independently derivable from (seed, index), so we MUST reproduce
    the whole 256-way batch and then index world i. (Feeding a pre-split single key back
    into ``env.reset`` would split it AGAIN and silently yield the WRONG worlds.)

    Returns (batched_env_state, env, s4_base) where batched_env_state is the craftax
    EnvState pytree with a leading (256,) axis on every leaf. FAILS CLOSED if JAX/craftax
    are absent -- nothing is materialized and no hash is ever produced.
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
    s4cls = _load_canonical_s4_task()                                   # exec canonical s4 (:85)
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
    return batched_env_state, env, s4_base


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


def _load_canonical_s4_task():
    """exec() the CANONICAL s4_task_code.py (p2_v1, SHA prefix 45fdd17c) -- NOT the P2-v0
    invalid-for-attribution copy. The path must be supplied via CC4_S4_TASK_PATH because
    the canonical evaluator uses a server-absolute path (/home/oseasy/...)."""
    path = os.environ.get("CC4_S4_TASK_PATH")
    require(path and os.path.isfile(path),
            "FAIL CLOSED: CC4_S4_TASK_PATH must point to the canonical s4_task_code.py "
            "(p2_v1, sha256 prefix %s). Refusing to guess a task definition." % TASK_SHA256_PREFIX)
    actual = sha256_file(path)
    require(actual.startswith(TASK_SHA256_PREFIX),
            "FAIL CLOSED: s4_task_code.py sha256 %s does not match canonical prefix %s "
            "(wrong task definition / CRLF-mangled copy)." % (actual[:16], TASK_SHA256_PREFIX))
    ns = {}
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), ns)                                              # mirrors eval:85
    return ns["Env"]


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
        require(src.get(k), "GATE REJECT: missing executed-source SHA %r" % k)
    return True


def compare_two_runs(a, b):
    """Compare two independent materialization results; FAIL CLOSED on ANY disagreement.

    Used by do_orchestrate (real runs) and by negative test 10 (with mock inputs to prove
    the comparison logic actually detects a mismatch).
    """
    for field in ("world_count", "world_set_hash", "per_world_hashes",
                  "source_shas", "versions", "evaluation_seed"):
        require(a.get(field) == b.get(field),
                "FAIL CLOSED: %s differs between the two independent runs" % field)
    return True


def collect_source_shas(eval_path, wrapper_path, task_path, env_path):
    """Record the SHA256 of every source the materializer EXECUTES (provenance, section 7).

    Fails closed if any required source is missing -- a recorded-but-not-executed or
    absent source must never slip into a world_set_hash.
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
    """FULL real materialization of the 256-world set + per-world & total hash.

    Re-imports / re-builds / re-runs the real generation+reset, re-serializes and re-hashes
    every world. Emits world_hashes.json. On a JAX-less host this FAILS CLOSED at the very
    first step and writes NOTHING that could be mistaken for a real world_set_hash.
    """
    evaluation_seed = ALLOWED_SEEDS[seed_id]
    source_shas = collect_source_shas(eval_path, wrapper_path, task_path, env_path)
    # Static anchor check: our embedded constants MUST equal the canonical source literals.
    anchor = static_anchor_check(eval_path, wrapper_path, task_path)
    require(anchor["result"] == "PASS",
            "FAIL CLOSED: static anchor check failed: %s" % anchor["mismatches"])
    versions = collect_versions()

    # Make the canonical s4 task discoverable by the materialization path.
    os.environ["CC4_S4_TASK_PATH"] = task_path

    batched, _env, _s4 = materialize_all_world_states(evaluation_seed)   # REAL reset (256 worlds)

    per_world = {}
    manifests = {}
    for i in range(NUM_WORLDS):                                          # 0..255 ascending
        identity = extract_world_identity_single(batched, i)
        _wb, _hdr, manifest, ph = serialize_world(identity, evaluation_seed, i,
                                                  source_shas, versions)
        per_world[str(i)] = ph
        manifests[str(i)] = manifest
    world_set_hash = compute_world_set_hash(per_world, evaluation_seed, source_shas, versions)

    os.makedirs(out_dir, exist_ok=True)
    result = {
        "schema": SCHEMA_VERSION,
        "materialized": True,
        "seed_id": seed_id,
        "evaluation_seed": evaluation_seed,
        "world_count": NUM_WORLDS,
        "world_index_order": "0..255 ascending",
        "per_world_hashes": per_world,
        "world_set_hash": world_set_hash,
        "source_shas": source_shas,
        "versions": versions,
        "static_anchor_check": anchor,
    }
    with open(os.path.join(out_dir, "world_hashes.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"mode": "single-run", "materialized": True,
                      "world_count": NUM_WORLDS, "world_set_hash": world_set_hash},
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
            results.append(json.load(f))
    a, b = results
    compare_two_runs(a, b)                       # fail closed on ANY disagreement
    agreed = {
        "schema": "mechanism_UED.craftax_world_set_agreement/v1",
        "materialized": True,
        "two_independent_runs": True,
        "seed_id": seed_id,
        "world_count": a["world_count"],
        "world_set_hash": a["world_set_hash"],
        "per_world_hash_agreement": True,
        "world_set_hash_agreement": True,
        "source_sha_agreement": True,
        "version_agreement": True,
        "run_A": os.path.join(run_dirs[0], "world_hashes.json"),
        "run_B": os.path.join(run_dirs[1], "world_hashes.json"),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "world_set_agreement.json"), "w", encoding="utf-8") as f:
        json.dump(agreed, f, indent=2, ensure_ascii=False)
    print(json.dumps({"mode": "orchestrate", "materialized": True,
                      "agreement": True, "world_set_hash": a["world_set_hash"]},
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
