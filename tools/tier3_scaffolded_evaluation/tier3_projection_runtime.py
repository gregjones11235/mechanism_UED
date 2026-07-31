#!/usr/bin/env python3
"""CC4 Tier3 — NON-RMT projection runtime registry + owner-runtime loaders +
engine policy-protocol adapters (contract NON_RMT_RUNTIME_ABI_BINDING_CLOSURE §一–§五).

REGISTRATION AUTHORITY (audited 2026-07-31; see
reports/tier3_scaffolded_evaluation/non_rmt_abi_binding_closure_20260731/):
  NON_RMT_RUNTIME_REGISTRATION_AUTHORITY = CC4_CAN_REGISTER_PROJECTIONS (conditional)
Conditions C1–C5 are enforced HERE, in code:
  C1 frozen common/ + engine byte-untouched — registration = NEW CC4 files only;
     the binding driver re-verifies common sums 57/57 before any binding.
  C2 ZERO REIMPLEMENTATION — every projection calls the owner's OWN SHA-bound
     runtime module (owner load_candidate fail-closed gates / policy_step /
     owner SHA-protocol functions). CC4 writes only the protocol mapping (the
     engine policy-protocol shell), greedy_argmax readout, boundary scheduling,
     and evidence. No network code, no memory mechanics, no hash definitions.
  C3 projection addendum documented (PROJECTION_RUNTIME_ADDENDUM.md).
  C4 honest labels: run_class=INTERFACE_SMOKE; performance_claim_authorized=false;
     the formal frozen scale FRONT/BACK/FULL = 8/8/64 comes from
     evaluation_profile.json and is DISTINCT from executed smoke episodes.
  C5 owner artifacts read-only; params/checkpoint SHA recomputed per the
     OWNER-DEFINED protocol (full64) by calling the owner's own hash functions;
     any mismatch fails closed; CC4 defines NO competing hash.

This module imports ONLY the standard library at the top level (the pure
self-test must run anywhere). Owner runtimes / JAX are imported lazily inside
the loaders and policy adapters. The engine is never imported here.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types

SCHEMA = "mechanism_UED.tier3_projection_runtime/v1"

# ---------------------------------------------------------------------------
# Frozen evaluation contract constants (sources: evaluation_profile.json +
# common SHA256SUMS + closing evidence; all LIVE-reverified by the binding
# driver before any binding — the constants here are the fail-closed reference).
# ---------------------------------------------------------------------------
FROZEN_MAX_TIMESTEPS = 4096
FROZEN_ACTION_MODE = "greedy_argmax"
FROZEN_OBSERVATION_SHAPE = (8335,)
FROZEN_ACTION_DIM = 43
# Formal frozen episode scale — extracted by the driver FROM evaluation_profile.json
# (scenarios.front_l2.n / scenarios.back_l2.n / scenarios.full.world_seed_set.count).
FROZEN_FRONT_EPISODE_COUNT = 8
FROZEN_BACK_EPISODE_COUNT = 8
FROZEN_FULL_EPISODE_COUNT = 64

# common/ frozen artifacts (byte SHA over on-disk files)
FROZEN_COMMON_RUNNER_SHA256 = "135332d3b30c60cb7b29c620dc931da852e99b2ca256c7a77dbf365dfc94075b"
FROZEN_COMMON_EVALUATOR_SHA256 = "a47ff97f9dc745c4f0cf015966b777f90c6dd6c7fe934b9b552a542df188a344"
FROZEN_EVALUATION_PROFILE_SHA256 = "7147370115621bda0500d55d8fd506a119ef8d6467a08329aaf6e088fbf9ea73"
FROZEN_METRIC_SCHEMA_SHA256 = "3a1712c4074dcb8fe8043c5a67e3ad7c730f252c533ad148a7181ba28f953da0"
FROZEN_ENVIRONMENT_LOCK_SHA256 = "453f1680dafe0f168c25c262f51de59ddc59559676aecd05f8f17389015c2ad3"
FROZEN_FULL_PROFILE_SHA256 = "2eceb288785a589f3f7f8b6989be7876bbe8da299128363ee008397d79039c1f"
FROZEN_FRONT_BANK_CONTENT_SHA256 = "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687"
FROZEN_BACK_BANK_CONTENT_SHA256 = "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566"
FROZEN_SHA256SUMS_SHA256 = "14892443e8a063f2a4dffc8ff3eb31b54959535308411a7ecc6388f9afe0a72e"
FROZEN_COMMON_SUMS_ENTRY_COUNT = 57
FROZEN_ABI_DOC_SHA256 = "61e52af6ff64a3071f8b64916c80906275dcb201d37feaa0382ed988d03d7f6a"
FROZEN_ASSEMBLY_MANIFEST_SHA256 = "52d1fa0fca2ceaa0e992a118b1d5812166e3347fef23100cad0b4389cc9136d4"

# Engine modules — LF-SHA (EOL-independent source identity), as recorded in
# assembly_manifest.engine_module_sha256 and in common_runner's fail-closed gate.
FROZEN_ENGINE_LF_SHA256 = {
    "tier3_evaluator.py": "54ae18db24c6a826d91bfc7ea49dee39a777b800e24dcc9b8897398def8da715",
    "tier3_candidate_runtime.py": "6af09be4efdb3eef66ef68579177085ac7d410109b40e10359160fbec23f681f",
}
FROZEN_CC2_ADAPTER_LF_SHA256 = "bdd268ffc1bab6f5ce0a150a813d96cfe1788476396a334fb4b65f2796a85317"

# CC2 base_gtrxl frozen modules (on-disk; the owner's load_candidate gate
# re-checks these against the capsule contract keys independently).
CC2_FROZEN_MODULE_DIR = ("/home/oseasy/cc2_data/cc2_source_2d0cc74/"
                         "gpu2_rmt16_phase4a_snapshot/runtime/frozen_modules")
CC2_FROZEN_MODULE_FILES = {
    "network_rmt16.py": "b5c37d7aa2e9cac1b4b395111262b4d8a11e20fd75a2930670336a68d86b8632",
    "rmt_memory_anchor.py": "4ff54fb4e932b2a1027803334919f0a19d7484e7e3a827b7f5188fa1283452b4",
    "rmt16_memory.py": "17e1a614c404e4edf176de7e8f9bd3f241059257fb24962d0df148960c7f6500",
}

# CC1 shared reference runtime (on-disk) + the dicode network module CC1 binds
# (dicode/network.py is byte-identical in the repo's audited dicode_src and in
# CC1's V7fix58 tree — verified 2026-07-31 — so pinning `dicode` resolution to
# the repo tree keeps the engine canonical-env source AND the owner network
# source provably the same bytes).
CC1_GTRXL128_REFERENCE_RUNTIME_SHA256 = "d3d4e55259cbe7cd57112bb26bfc765cae96ae88b170397ea3e76fe2fd0196b1"
FROZEN_DICODE_NETWORK_SHA256 = "172e1cd427bee8a31946bcf2936ea960c68e7ce3f5a71c55c377cf92c7c4c3c9"
# train_state_utils.py (CC1 orbax load path): the module body has ZERO wandb
# references; only the dicode.utils.general package-chain imports wandb.
CC1_TRAIN_STATE_UTILS_SHA256 = "cbd091f90b2592b7a2ef51251e75dbe51f3b8eb5b9d36a3024123602582b2248"

# CC3 shared slowgru runtime + arm network (on-disk; both arms' slowgru_network.py
# are byte-identical — one shared SHA).
CC3_SLOWGRU_RUNTIME_SHA256 = "d3b74d2ed6aee1affd54bb3a39bdb50162afa81dac0cab5866e4ed6450fcb24b"
CC3_SLOWGRU_NETWORK_SHA256 = "b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b"

# CC4 V1-era RMT16 capsules (both formal RMT16 candidates): the capsule's thin
# shim binds to the FROZEN V1 common runner at COMMON_ROOT, which delegates to
# the FROZEN engine's registered rmt16_gtrxl_cc2 family loader (contract
# verification path identical to tier3_evaluator's). CHECKPOINT_CONTRACT content
# SHA = canonical SHA of repo configs/tier3_cc2_final98304_checkpoint_contract_v1.json
# (both capsule copies byte-identical; cited verbatim from both shims'
# CHECKPOINT_CONTRACT_SHA256 constants).
RMT16_V1_COMMON_ROOT = "/home/oseasy/student_pool_v1/common"
RMT16_FROZEN_CONTRACT_CONTENT_SHA256 = "7dda2bc7517342b189a1f1ba949d620eb4d1c978e252b74f4e2bdeb61363f2e5"
RMT16_ENGINE_RUNTIME_FAMILY = "rmt16_gtrxl_cc2"
RMT16_FROZEN_ACTION_MODE = "greedy_argmax"

# CC4 evaluation-device GPU allowlist (UUIDs). GPU0/GPU1 are BANNED for CC4.
CC4_GPU_ALLOWED_UUIDS = (
    "GPU-8df11537-ab79-722d-606f-411966196c4c",   # GPU2
    "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",   # GPU3
)
CC4_GPU_BANNED_UUID_PREFIXES = ("GPU-e8c08612", "GPU-3c7a2864")  # GPU0 / GPU1

# SlowGRU owner-documented segment boundary (on_segment_boundary cadence).
SLOWGRU_SEGMENT_BOUNDARY_STEPS = 128
# Owner smoke rng seed declared in the CC3 capsule checkpoint contracts.
SLOWGRU_SMOKE_POLICY_RNG_SEED = 777


class FailClosed(Exception):
    """Hard stop on any projection/registration-contract violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Hashing helpers (raw-bytes file SHA; LF-normalized source SHA)
# ---------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    """SHA256 over raw file bytes (the owner pkl/file convention)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def lf_sha256_file(path):
    """LF-normalized SHA256 of a source file (EOL-independent source identity;
    matches tier3_evaluator._sha256_lf_file / assembly_manifest engine SHAs)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


# ---------------------------------------------------------------------------
# Module import (SHA-gated, unique names — every owner names its file
# candidate_runtime.py, so each gets a distinct module identity).
# ---------------------------------------------------------------------------
def import_module_from_file(unique_name, path, expected_sha256=None):
    """Import a module from PATH under UNIQUE_NAME. If expected_sha256 is given,
    the raw-bytes SHA of the file must match BEFORE exec (fail closed)."""
    require(os.path.isfile(path), "FAIL CLOSED: module path missing: %r" % path)
    if expected_sha256 is not None:
        got = sha256_file(path)
        require(got == expected_sha256,
                "FAIL CLOSED (OWNER_MODULE_SHA_MISMATCH): %s live %s != declared %s"
                % (path, got, expected_sha256))
    spec = importlib.util.spec_from_file_location(unique_name, path)
    require(spec is not None and spec.loader is not None,
            "FAIL CLOSED: cannot build import spec for %r" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Capsule file verification (registry constant == capsule SHA256SUMS == live)
# ---------------------------------------------------------------------------
CAPSULE_FILES = ("candidate_runtime.py", "candidate_manifest.json",
                 "checkpoint_contract.json", "evaluate_candidate.py")


def parse_sha256sums(path):
    """Parse a sha256sums file -> {relpath: sha}. Tolerates 'sha  rel' and
    'sha *rel' lines (GNU text/binary markers)."""
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            require(len(parts) == 2, "FAIL CLOSED: malformed sums line in %s: %r"
                    % (path, line[:80]))
            sha, name = parts[0], parts[1]
            if name.startswith("*"):
                name = name[1:]
            out[name.strip()] = sha.strip()
    return out


def verify_capsule_files(spec):
    """Three-way gate for the four capsule files: registry-declared SHA ==
    capsule's own SHA256SUMS entry == live bytes. Returns the evidence dict."""
    capsule = spec["source_capsule_root"]
    sums_path = os.path.join(capsule, "SHA256SUMS")
    sums = parse_sha256sums(sums_path) if os.path.isfile(sums_path) else {}
    evidence = {}
    for fn in CAPSULE_FILES:
        declared = spec["capsule_file_sha256"][fn]
        path = os.path.join(capsule, fn)
        require(os.path.isfile(path),
                "FAIL CLOSED: capsule file missing: %s" % path)
        live = sha256_file(path)
        sums_sha = sums.get(fn)
        require(live == declared,
                "FAIL CLOSED (CAPSULE_FILE_SHA_MISMATCH): %s live %s != registry %s"
                % (path, live, declared))
        require(sums_sha is None or sums_sha == declared,
                "FAIL CLOSED (CAPSULE_SUMS_INCONSISTENT): %s sums %s != registry %s"
                % (fn, sums_sha, declared))
        evidence[fn] = {"declared": declared, "live": live,
                        "capsule_sums": sums_sha, "match": True}
    return evidence


# ---------------------------------------------------------------------------
# Minimal documented wandb stub (CC1 orbax load path only)
# ---------------------------------------------------------------------------
def install_wandb_stub_if_needed():
    """SCOPE-DISCLOSED no-op stub, installed ONLY if `import wandb` fails.

    Why: dicode.utils.general's package chain imports wandb at import time, but
    train_state_utils.py itself (LF/raw SHA cbd091f9…) contains ZERO wandb
    references (grep-verified 2026-07-31). The stub is a ModuleType with a
    PEP-562 module __getattr__ returning a no-op factory: any `wandb.X`
    resolves to a callable returning a no-op object. No network, no state, no
    file IO. Every binding record discloses whether the stub was installed.
    """
    try:
        importlib.import_module("wandb")
        return {"installed": False, "reason": "wandb importable without stub"}
    except ImportError:
        pass
    stub = types.ModuleType("wandb")

    class _NoOp(object):
        def __getattr__(self, name):
            return _noop

        def __call__(self, *a, **k):
            return _NoOp()

    def _noop(*a, **k):
        return _NoOp()

    stub.__path__ = []                     # submodule-less package marker
    stub.__getattr__ = lambda name: _noop      # PEP 562 module-level __getattr__
    stub.__doc__ = ("CC4 minimal no-op wandb stub (package-chain import only; "
                    "train_state_utils.py has zero wandb references).")
    sys.modules["wandb"] = stub
    return {"installed": True,
            "module": "wandb",
            "reason": "ModuleNotFoundError: wandb on bare import (package-chain only)",
            "stub_scope": "sys.modules['wandb'] = no-op ModuleType with PEP562 __getattr__",
            "train_state_utils_sha256": CC1_TRAIN_STATE_UTILS_SHA256,
            "train_state_utils_wandb_references": 0}


def install_openai_stub_if_needed():
    """SCOPE-DISCLOSED import-only stub, installed ONLY if `import openai` fails.

    Why: CC1's build_stage4_env -> dicode.task_utils -> dicode.dreaming.gen_manager
    -> dicode.dreaming.llm imports `from openai import AsyncOpenAI` at import
    time, but the LLM class is NEVER instantiated on our path — the owner's
    stage4 env builder uses only task_utils.get_achievement_multi_hot (pure
    numpy / craftax-constant math). The CC4 contract FORBIDS any new LLM call,
    so instead of installing the real client library (and its httpx/pydantic
    chain) into the locked eval venv, `openai` is stubbed: AsyncOpenAI exists
    for the import, but INSTANTIATION RAISES, and any other attribute access
    RAISES too — an accidental LLM attempt fails loudly, never makes a call.
    Every binding record discloses whether the stub was installed.
    """
    try:
        importlib.import_module("openai")
        return {"installed": False, "reason": "openai importable without stub"}
    except ImportError:
        pass
    stub = types.ModuleType("openai")

    class _ForbiddenLLMClient(object):
        def __init__(self, *a, **k):
            raise RuntimeError(
                "CC4 eval venv: 'openai' is an import-only stub; "
                "instantiation (any LLM call) is forbidden by CC4 contract.")

    def _forbidden(name):
        raise RuntimeError(
            "CC4 eval venv: 'openai.%s' does not exist (import-only stub; "
            "LLM calls forbidden by CC4 contract)." % name)

    stub.AsyncOpenAI = _ForbiddenLLMClient
    # __path__ = [] marks the stub as a (submodule-less) package: CPython's
    # from-import machinery probes module.__path__ at the C level and would
    # otherwise hit __getattr__ below. Real attribute -> machinery is happy;
    # `import openai.anything` still fails closed (no submodule will resolve).
    stub.__path__ = []
    stub.__getattr__ = _forbidden          # PEP 562 module-level __getattr__
    stub.__doc__ = ("CC4 import-only openai stub (package-chain import "
                    "satisfaction only; instantiation / LLM calls forbidden).")
    sys.modules["openai"] = stub
    return {"installed": True,
            "module": "openai",
            "reason": "ModuleNotFoundError: openai via task_utils->gen_manager->llm (import chain only)",
            "stub_scope": ("sys.modules['openai'] = ModuleType; AsyncOpenAI exists "
                           "for import but RAISES on instantiation; other attrs RAISE"),
            "llm_calls_executed": 0,
            "contract_basis": "CC4 contract forbids new LLM calls"}


# ---------------------------------------------------------------------------
# CC1 shared runtime resolution + dicode resolution pinning
# ---------------------------------------------------------------------------
CC1_DEFAULT_SHARED_RUNTIME = ("/home/oseasy/git_work/student_pool_reference_gtrxl/"
                              "student_pool/shared_runtime")


def resolve_cc1_shared_runtime(capsule_root):
    """Owner resolution order (from cc1 candidate_runtime._shared_runtime_path):
    env CC1_SHARED_RUNTIME -> capsule environment_lock.json shared_runtime_path
    -> owner default."""
    env = os.environ.get("CC1_SHARED_RUNTIME")
    if env:
        return env
    lock = os.path.join(capsule_root, "environment_lock.json")
    if os.path.isfile(lock):
        try:
            p = read_json(lock).get("shared_runtime_path")
            if p:
                return p
        except Exception:
            pass
    return CC1_DEFAULT_SHARED_RUNTIME


def verify_cc1_shared_runtime(capsule_root):
    root = resolve_cc1_shared_runtime(capsule_root)
    path = os.path.join(root, "gtrxl128_reference_runtime.py")
    require(os.path.isfile(path),
            "FAIL CLOSED: cc1 shared runtime missing: %s" % path)
    live = sha256_file(path)
    require(live == CC1_GTRXL128_REFERENCE_RUNTIME_SHA256,
            "FAIL CLOSED (CC1_SHARED_RUNTIME_SHA_MISMATCH): %s live %s != frozen %s"
            % (path, live, CC1_GTRXL128_REFERENCE_RUNTIME_SHA256))
    return {"shared_runtime_root": root, "gtrxl128_reference_runtime_path": path,
            "gtrxl128_reference_runtime_sha256": live}


def pin_dicode_resolution(repo_root):
    """Pin the `dicode` package to the repo's audited dicode_src BEFORE any CC1
    owner load (owner ensure_dicode_path only inserts V7 src if `dicode` is not
    yet importable). dicode/network.py is byte-identical in both trees
    (== CC1's declared policy_source_sha256 172e1cd4…, verified 2026-07-31);
    the single differing module (wrappers_cl.py) is used ONLY by the owner's
    eval_env wrapper — never by the engine canonical-env / rollout path."""
    src = os.path.join(repo_root, "dicode_src", "src")
    net = os.path.join(src, "dicode", "network.py")
    require(os.path.isfile(net), "FAIL CLOSED: repo dicode network missing: %s" % net)
    live = sha256_file(net)
    require(live == FROZEN_DICODE_NETWORK_SHA256,
            "FAIL CLOSED (DICODE_NETWORK_SHA_MISMATCH): %s live %s != frozen %s"
            % (net, live, FROZEN_DICODE_NETWORK_SHA256))
    if src not in sys.path:
        sys.path.insert(0, src)
    dicode = importlib.import_module("dicode")
    importlib.import_module("dicode.network")
    return {"dicode_src": src,
            "dicode_package_file": getattr(dicode, "__file__", None),
            "dicode_network_sha256": live,
            "dicode_network_matches_cc1_policy_source": True}


# ---------------------------------------------------------------------------
# Owner runtime loaders (each calls the owner's OWN fail-closed load gate;
# CC4 adds only independent on-disk SHA re-verification of the bound sources).
# ---------------------------------------------------------------------------
def load_cc2_base_gtrxl(spec):
    """BASE_GTRXL (CC2): owner candidate_runtime.load_candidate(contract) gates
    frozen-module source SHAs vs the contract keys and recomputes
    canonical_params_sha (raising CHECKPOINT_PARAMS_SHA_MISMATCH on drift)."""
    capsule = spec["source_capsule_root"]
    contract_path = os.path.join(capsule, "checkpoint_contract.json")
    fm_live = {}
    for fn, want in CC2_FROZEN_MODULE_FILES.items():
        p = os.path.join(CC2_FROZEN_MODULE_DIR, fn)
        require(os.path.isfile(p),
                "FAIL CLOSED: cc2 frozen module missing: %s" % p)
        got = sha256_file(p)
        require(got == want,
                "FAIL CLOSED (CC2_FROZEN_MODULE_SHA_MISMATCH): %s live %s != frozen %s"
                % (p, got, want))
        fm_live[fn] = got
    mod = import_module_from_file(
        "cc4_proj_cc2_base_gtrxl_candidate_runtime",
        os.path.join(capsule, "candidate_runtime.py"),
        expected_sha256=spec["capsule_file_sha256"]["candidate_runtime.py"])
    candidate = mod.load_candidate(contract_path, verify_source_sha=True)
    return {"kind": "cc2_base_gtrxl", "module": mod, "candidate": candidate,
            "params": candidate.params, "checkpoint_path": contract_checkpoint_path(contract_path),
            "frozen_modules_live_sha256": fm_live, "wandb_stub": None,
            "import_stubs": None}


def load_cc1_gtrxl128(spec):
    """CONTROL / TEACHER (CC1): owner thin binding load_candidate() reads its OWN
    capsule contract/manifest, builds the stage4 env (OBS_DIM/ACTION_DIM), and
    loads params via the owner loader (pickle for teacher, orbax via dicode
    load_weights_only for control). The shared runtime R is SHA-gated here."""
    capsule = spec["source_capsule_root"]
    sr = verify_cc1_shared_runtime(capsule)
    contract_path = os.path.join(capsule, "checkpoint_contract.json")
    require(sha256_file(contract_path) == spec["capsule_file_sha256"]["checkpoint_contract.json"],
            "FAIL CLOSED: cc1 contract sha drift for %s" % capsule)
    mod = import_module_from_file(
        "cc4_proj_cc1_%s_candidate_runtime" % spec["candidate_id"].lower(),
        os.path.join(capsule, "candidate_runtime.py"),
        expected_sha256=spec["capsule_file_sha256"]["candidate_runtime.py"])
    # Import-chain completion (disclosed per-binding): the owner's stage4 env
    # builder pulls package-chain modules absent from the locked eval venv.
    # wandb -> documented no-op stub; openai -> import-only stub whose
    # instantiation RAISES (LLM calls forbidden by CC4 contract). Anything
    # else missing fails closed — never guessed, never silently stubbed.
    stubs = []
    loaded = None
    last_exc = None
    for _attempt in range(3):
        try:
            loaded = mod.load_candidate()
            break
        except ModuleNotFoundError as exc:
            last_exc = exc
            missing = getattr(exc, "name", "") or ""
            # Match the EXACT missing top-level module only — a submodule miss
            # (e.g. openai.types) means something tried to reach THROUGH the
            # import-only stub, which fails closed rather than widening it.
            if missing == "wandb":
                s = install_wandb_stub_if_needed()
            elif missing == "openai":
                s = install_openai_stub_if_needed()
            else:
                raise FailClosed(
                    "FAIL CLOSED: unexpected cc1 load failure: %r (missing "
                    "module %r; only bare 'wandb'/'openai' may be stubbed)"
                    % (exc, missing))
            require(s["installed"],
                    "FAIL CLOSED: %s stub not installed but its import failed" % exc)
            stubs.append(s)
    require(loaded is not None,
            "FAIL CLOSED: cc1 load still failing after stub retries: %r" % last_exc)
    stub = next((s for s in stubs if s.get("module") == "wandb"), None)
    R = getattr(mod, "R", None)
    require(R is not None,
            "FAIL CLOSED: cc1 candidate_runtime did not expose the shared "
            "runtime as module attribute .R")
    return {"kind": "cc1_gtrxl128", "module": mod, "loaded": loaded, "R": R,
            "params": loaded["params"],
            "checkpoint_path": loaded["contract"]["checkpoint_path"],
            "obs_dim": loaded.get("obs_dim"), "action_dim": loaded.get("action_dim"),
            "shared_runtime": sr, "wandb_stub": stub,
            "import_stubs": stubs or None}


def install_numpy2_pickle_compat_if_needed():
    """CC3 checkpoints were PICKLED UNDER numpy>=2: the ndarray reducer in the
    pickle stream references ``numpy._core.numeric._frombuffer`` (protocol-5
    in-band buffer reconstruction). The LOCKED CC4 venv pins numpy 1.26.4
    (jax 0.4.30 pin; environment_lock — numpy may NOT be upgraded).

    numpy 1.26.4 in this venv SHIPS THE OFFICIAL numpy2-pickle interop shim
    package at site-packages/numpy/_core/ (its own docstring: "This private
    module only contains stubs for interoperability with NumPy 2.0 pickled
    arrays") covering _dtype / _internal / multiarray / _multiarray_umath /
    umath — but NOT `numeric`, the one leaf the CC3 pickle stream references.
    This function COMPLETES the official shim by exactly that missing leaf:
    ``sys.modules["numpy._core.numeric"] = numpy.core.numeric`` (numpy 1.x's
    identical protocol-5 ``_frombuffer`` reconstructor). Gate: if
    ``numpy._core.numeric`` imports natively, do nothing. (hasattr(np, "_core")
    is NOT a valid gate: the on-disk shim package sets the attribute without
    providing the numeric leaf.)

    Verified scope (read-only byte scan of both CC3 pkls, 2026-07-31): exactly
    ONE numpy2-path reference each — ``numpy._core.numeric`` — nothing else.
    Any OTHER missing module still fails closed (no widening). Reconstruction
    fidelity is witnessed by the owner-protocol params_sha gate (driver G3):
    if the alias altered any numeric, the declared params SHA would not match
    and the binding would fail closed. No owner code is modified; the pkl
    bytes are read untouched (file SHA gate)."""
    import importlib
    import numpy as np
    try:
        importlib.import_module("numpy._core.numeric")
        return {"installed": False,
                "reason": "numpy._core.numeric imports natively (official "
                          "shim already complete)",
                "numpy_version": np.__version__}
    except ImportError:
        pass
    import numpy.core.numeric as _nc_numeric
    require(hasattr(_nc_numeric, "_frombuffer"),
            "FAIL CLOSED: numpy.core.numeric._frombuffer missing in locked "
            "numpy %s — cannot complete the numpy2 pickle interop shim"
            % np.__version__)
    sys.modules["numpy._core.numeric"] = _nc_numeric
    return {"installed": True,
            "alias": "sys.modules['numpy._core.numeric'] = numpy.core.numeric",
            "basis": "completes numpy 1.26.4's OWN numpy2-pickle interop shim "
                     "package (site-packages/numpy/_core/, docstring: 'stubs "
                     "for interoperability with NumPy 2.0 pickled arrays') by "
                     "its one missing leaf `numeric` (protocol-5 _frombuffer); "
                     "identical reconstructor semantics",
            "numpy_version": np.__version__,
            "scope": "single sys.modules entry; verified pkl reference set = "
                     "{numpy._core.numeric} only (byte scan both CC3 pkls); "
                     "any other missing module still fails closed",
            "fidelity_witness": "owner params_sha_packed gate (G3) + pkl file "
                                "SHA gate (G2) — both fail closed on drift",
            "owner_code_modified": False}


def load_cc3_slowgru(spec):
    """SLOWGRU_RESET128 / SLOWGRU_PERSISTENT (CC3): owner thin binding over
    slowgru_runtime. The thin module asserts capsule contract identity
    (candidate_id + carry_mode) AT IMPORT; load_candidate runs the triple
    fail-closed gate (file SHA + arm network_src SHA + params SHA) and raises
    CARRY_MODE_MISMATCH on any drift. CC4 re-verifies the shared runtime + arm
    network on-disk SHAs and seeds the owner policy rng (contract smoke_seed)."""
    capsule = spec["source_capsule_root"]
    contract_path = os.path.join(capsule, "checkpoint_contract.json")
    require(sha256_file(contract_path) == spec["capsule_file_sha256"]["checkpoint_contract.json"],
            "FAIL CLOSED: cc3 contract sha drift for %s" % capsule)
    contract = read_json(contract_path)
    # Shared runtime (owner imports it from ../slowgru_runtime relative to capsule)
    rt_path = os.path.normpath(os.path.join(capsule, "..", "slowgru_runtime",
                                            "slowgru_runtime.py"))
    require(os.path.isfile(rt_path),
            "FAIL CLOSED: cc3 shared runtime missing: %s" % rt_path)
    rt_sha = sha256_file(rt_path)
    require(rt_sha == CC3_SLOWGRU_RUNTIME_SHA256,
            "FAIL CLOSED (CC3_SLOWGRU_RUNTIME_SHA_MISMATCH): %s live %s != frozen %s"
            % (rt_path, rt_sha, CC3_SLOWGRU_RUNTIME_SHA256))
    # Arm network source (contract arm_src) — both arms byte-identical
    arm_src = contract.get("arm_src")
    require(arm_src, "FAIL CLOSED: cc3 contract missing arm_src for %s" % capsule)
    net_path = os.path.join(arm_src, "slowgru_network.py")
    require(os.path.isfile(net_path),
            "FAIL CLOSED: cc3 arm network missing: %s" % net_path)
    net_sha = sha256_file(net_path)
    require(net_sha == CC3_SLOWGRU_NETWORK_SHA256,
            "FAIL CLOSED (CC3_SLOWGRU_NETWORK_SHA_MISMATCH): %s live %s != frozen %s"
            % (net_path, net_sha, CC3_SLOWGRU_NETWORK_SHA256))
    mod = import_module_from_file(
        "cc4_proj_cc3_%s_candidate_runtime" % spec["candidate_id"].lower(),
        os.path.join(capsule, "candidate_runtime.py"),
        expected_sha256=spec["capsule_file_sha256"]["candidate_runtime.py"])
    np_compat = install_numpy2_pickle_compat_if_needed()
    handle = mod.load_candidate()          # owner triple gate + carry mode gate
    require(handle.get("carry_mode") == spec["carry_mode"],
            "FAIL CLOSED (CARRY_MODE_MISMATCH_CC4): loaded %r != registry %r"
            % (handle.get("carry_mode"), spec["carry_mode"]))
    _sr = getattr(mod, "_sr", None)
    require(_sr is not None,
            "FAIL CLOSED: cc3 candidate_runtime did not expose slowgru_runtime as _sr")
    mod.seed_policy_rng(SLOWGRU_SMOKE_POLICY_RNG_SEED)
    return {"kind": "cc3_slowgru", "module": mod, "handle": handle, "_sr": _sr,
            "params": handle["params"],
            "checkpoint_path": contract.get("checkpoint_path"),
            "arm_src": arm_src, "slowgru_runtime_path": rt_path,
            "slowgru_runtime_sha256": rt_sha, "slowgru_network_sha256": net_sha,
            "wandb_stub": None, "import_stubs": None,
            "numpy_pickle_compat": np_compat}


def load_cc4_rmt16_capsule(spec):
    """PERSISTENT/RESET128 RMT16 (CC4 V1-era capsule over a CC2 original-vtrace
    arm). Owner plumbing chain, every hop SHA-gated and NONE of it CC4-defined
    semantics:

        capsule candidate_runtime.py (thin shim, registry-pinned SHA)
          -> FROZEN V1 common_runner.py (FROZEN_COMMON_RUNNER_SHA256)
             -> FROZEN engine tier3_candidate_runtime.py (LF-SHA 6af09be4…),
                registered rmt16_gtrxl_cc2 family loader: verifies the pkl
                (file SHA + recomputed params SHA + driver-source SHA + CC2
                policy-source SHA) against the frozen final98304 contract and
                requires carry_mode == arm — the SAME verification path
                tier3_evaluator uses — then builds CC2RMT16Policy and returns
                the CandidateRuntime ABI (init_memory/policy_step/reset_memory/
                candidate_metadata; opaque memory snapshot; batch enforced 1).

    CC4 defines no hash and no semantics here: every SHA is owner-declared
    (capsule manifest/contract/shim), and the engine + driver RECOMPUTE them
    independently (fail closed). The runtime is carried in ctx["runtime"];
    params stay inside the engine runtime (read-only, captured by reference)."""
    capsule = spec["source_capsule_root"]
    contract_path = os.path.join(capsule, "checkpoint_contract.json")
    require(sha256_file(contract_path) == spec["capsule_file_sha256"]["checkpoint_contract.json"],
            "FAIL CLOSED: rmt16 capsule contract sha drift for %s" % capsule)
    # Hop 1 gate: the FROZEN V1 common runner the shim binds to.
    common_runner_path = os.path.join(RMT16_V1_COMMON_ROOT, "common_runner.py")
    require(os.path.isfile(common_runner_path),
            "FAIL CLOSED: V1 common runner missing: %s" % common_runner_path)
    runner_sha = sha256_file(common_runner_path)
    require(runner_sha == FROZEN_COMMON_RUNNER_SHA256,
            "FAIL CLOSED (V1_COMMON_RUNNER_SHA_MISMATCH): %s live %s != frozen %s"
            % (common_runner_path, runner_sha, FROZEN_COMMON_RUNNER_SHA256))
    # Hop 2 gate: the FROZEN engine the runner delegates to (LF-SHA, the same
    # identity the runner's own engine pin enforces — independent cross-check
    # of the same bytes in this checkout).
    engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "tier3_candidate_runtime.py")
    require(os.path.isfile(engine_path),
            "FAIL CLOSED: frozen engine missing: %s" % engine_path)
    engine_sha = lf_sha256_file(engine_path)
    require(engine_sha == FROZEN_ENGINE_LF_SHA256["tier3_candidate_runtime.py"],
            "FAIL CLOSED (FROZEN_ENGINE_LF_SHA_MISMATCH): %s live %s != frozen %s"
            % (engine_path, engine_sha,
               FROZEN_ENGINE_LF_SHA256["tier3_candidate_runtime.py"]))
    # Shim import (SHA-gated BEFORE exec) + identity cross-check BEFORE load.
    mod = import_module_from_file(
        "cc4_proj_rmt16_%s_candidate_runtime" % spec["carry_mode"].lower(),
        os.path.join(capsule, "candidate_runtime.py"),
        expected_sha256=spec["capsule_file_sha256"]["candidate_runtime.py"])
    require(getattr(mod, "COMMON_RUNNER_SHA256", None) == FROZEN_COMMON_RUNNER_SHA256,
            "FAIL CLOSED (RMT16_SHIM_RUNNER_PIN_MISMATCH): %s shim pins runner %r "
            "!= frozen V1 %s"
            % (capsule, getattr(mod, "COMMON_RUNNER_SHA256", None),
               FROZEN_COMMON_RUNNER_SHA256))
    fi = mod.frozen_identities()
    require(fi.get("candidate_id") == spec["candidate_id"],
            "FAIL CLOSED (RMT16_SHIM_ID_MISMATCH): shim candidate_id %r != registry %r"
            % (fi.get("candidate_id"), spec["candidate_id"]))
    require(fi.get("runtime_family") == RMT16_ENGINE_RUNTIME_FAMILY
            == spec["engine_runtime_family"],
            "FAIL CLOSED (RMT16_SHIM_FAMILY_MISMATCH): shim family %r"
            % (fi.get("runtime_family"),))
    require(fi.get("arm") == spec["carry_mode"],
            "FAIL CLOSED (RMT16_SHIM_ARM_MISMATCH): shim arm %r != registry carry %r"
            % (fi.get("arm"), spec["carry_mode"]))
    require(fi.get("params_sha256") == spec["declared_params_sha256"]["value"],
            "FAIL CLOSED (RMT16_SHIM_PARAMS_MISMATCH): shim params %r != registry %r"
            % (fi.get("params_sha256"), spec["declared_params_sha256"]["value"]))
    require(fi.get("checkpoint_file_sha256")
            == spec["declared_checkpoint_file_sha256"]["value"],
            "FAIL CLOSED (RMT16_SHIM_FILE_MISMATCH): shim file sha %r != registry %r"
            % (fi.get("checkpoint_file_sha256"),
               spec["declared_checkpoint_file_sha256"]["value"]))
    require(fi.get("checkpoint_contract_sha256") == RMT16_FROZEN_CONTRACT_CONTENT_SHA256,
            "FAIL CLOSED (RMT16_SHIM_CONTRACT_MISMATCH): shim contract content sha "
            "%r != frozen %s"
            % (fi.get("checkpoint_contract_sha256"),
               RMT16_FROZEN_CONTRACT_CONTENT_SHA256))
    require(fi.get("scientific_predicates_defined_here") is False
            and fi.get("trainable") is False and fi.get("immutable") is True,
            "FAIL CLOSED (RMT16_SHIM_FLAGS): shim must declare no predicates, "
            "non-trainable, immutable")
    # Owner load through the shim -> V1 runner -> frozen engine family loader
    # (engine raises on ANY contract/SHA/carry drift — fail closed inside).
    runtime = mod.load(spec["checkpoint_path"])
    for meth in ("init_memory", "policy_step", "reset_memory", "candidate_metadata"):
        require(callable(getattr(runtime, meth, None)),
                "FAIL CLOSED (RMT16_ABI_SURFACE): engine runtime missing %r" % meth)
    meta = runtime.candidate_metadata()
    require(meta.get("runtime_family") == RMT16_ENGINE_RUNTIME_FAMILY,
            "FAIL CLOSED (RMT16_META_FAMILY): %r" % (meta.get("runtime_family"),))
    require(meta.get("arm") == spec["carry_mode"],
            "FAIL CLOSED (RMT16_META_ARM): %r != %r"
            % (meta.get("arm"), spec["carry_mode"]))
    require(meta.get("carry_mode") == spec["carry_mode"],
            "FAIL CLOSED (RMT16_META_CARRY): %r != %r"
            % (meta.get("carry_mode"), spec["carry_mode"]))
    require(meta.get("action_mode") == RMT16_FROZEN_ACTION_MODE,
            "FAIL CLOSED (RMT16_META_ACTION_MODE): %r != %s"
            % (meta.get("action_mode"), RMT16_FROZEN_ACTION_MODE))
    require(meta.get("params_sha256") == fi["params_sha256"],
            "FAIL CLOSED (RMT16_META_PARAMS): engine-recomputed %r != shim-declared %r"
            % (meta.get("params_sha256"), fi["params_sha256"]))
    require(meta.get("checkpoint_file_sha256") == fi["checkpoint_file_sha256"],
            "FAIL CLOSED (RMT16_META_FILE): engine-recomputed %r != shim-declared %r"
            % (meta.get("checkpoint_file_sha256"), fi["checkpoint_file_sha256"]))
    require(meta.get("checkpoint_contract_sha256") == RMT16_FROZEN_CONTRACT_CONTENT_SHA256,
            "FAIL CLOSED (RMT16_META_CONTRACT): %r"
            % (meta.get("checkpoint_contract_sha256"),))
    require(meta.get("base_checkpoint_params_sha256")
            == fi["base_checkpoint_params_sha256"],
            "FAIL CLOSED (RMT16_META_BASE_PARAMS): %r != %r"
            % (meta.get("base_checkpoint_params_sha256"),
               fi["base_checkpoint_params_sha256"]))
    require(meta.get("checkpoint_path") == spec["checkpoint_path"],
            "FAIL CLOSED (RMT16_META_PATH): %r != %r"
            % (meta.get("checkpoint_path"), spec["checkpoint_path"]))
    return {"kind": "cc4_rmt16_capsule", "module": mod, "runtime": runtime,
            "frozen_identities": fi, "engine_metadata": meta,
            "checkpoint_path": spec["checkpoint_path"],
            "common_root": RMT16_V1_COMMON_ROOT,
            "common_runner_path": common_runner_path,
            "common_runner_sha256": runner_sha,
            "engine_path": engine_path, "engine_lf_sha256": engine_sha,
            "wandb_stub": None, "import_stubs": None}


def contract_checkpoint_path(contract_path):
    return read_json(contract_path).get("checkpoint_path")


# ---------------------------------------------------------------------------
# Owner-protocol SHA recomputation (C5: call the owner's OWN hash functions —
# never a CC4-redefined hash).
# ---------------------------------------------------------------------------
def recompute_params_sha_owner(ctx):
    kind = ctx["kind"]
    if kind == "cc2_base_gtrxl":
        return ctx["module"].canonical_params_sha(ctx["params"])
    if kind == "cc1_gtrxl128":
        return ctx["R"].params_sha256(ctx["params"])
    if kind == "cc3_slowgru":
        return ctx["_sr"].params_sha(ctx["handle"])
    if kind == "cc4_rmt16_capsule":
        # The engine's OWN hash function (the frozen verification path): reload
        # the pkl read-only and recompute CC2's canonical params SHA from the
        # bytes — never a CC4-redefined hash, never the shim's declared value.
        import tier3_checkpoint_adapter as ckpt
        _params, params_sha, _manifest, _file_sha = \
            ckpt.load_full_params_readonly(ctx["checkpoint_path"])
        return params_sha
    raise FailClosed("FAIL CLOSED: unknown loader kind %r" % kind)


def recompute_checkpoint_file_sha_owner(spec, ctx):
    """orbax dir -> owner R.dir_sha256 (CC1 protocol); pkl file -> raw bytes."""
    path = ctx["checkpoint_path"]
    require(path, "FAIL CLOSED: no checkpoint_path in owner context")
    if spec["checkpoint_kind"] == "orbax_dir":
        require(os.path.isdir(path),
                "FAIL CLOSED: orbax checkpoint dir missing: %s" % path)
        return ctx["R"].dir_sha256(path)
    require(os.path.isfile(path),
            "FAIL CLOSED: checkpoint pkl missing: %s" % path)
    return sha256_file(path)


# ---------------------------------------------------------------------------
# Engine policy-protocol adapters (reset() + __call__(obs, env_state) -> int).
# The engine loop calls int(policy_fn(obs, state)); policy.reset() per episode.
# Memory mechanics are OWNED by the owner runtime — these shells only map the
# engine protocol onto the owner's policy_step and read greedy_argmax.
# ---------------------------------------------------------------------------
class BaseGtrxlProjectionPolicy(object):
    """CC2 base_gtrxl: owner Candidate.policy_step already returns argmax(logits)
    (greedy). RMT+GTrXL memory mechanics live inside the owner Candidate."""

    def __init__(self, candidate):
        self.candidate = candidate
        self.ms = None

    def reset(self):
        self.ms = self.candidate.init_memory(1)

    def __call__(self, obs, env_state):
        import jax.numpy as jnp
        import numpy as np
        if self.ms is None:
            self.reset()
        o = jnp.asarray(np.asarray(obs)[None, :])
        done_mask = jnp.zeros((1,), dtype=jnp.bool_)
        action, _logits, self.ms = self.candidate.policy_step(o, self.ms, done_mask)
        return int(np.asarray(action).reshape(-1)[0])


class GTrXL128ProjectionPolicy(object):
    """CC1 control/teacher: owner module-level policy_step(loaded, obs, ms,
    done_mask, greedy=True) -> dict(action, logits, value, memory_state);
    greedy=True selects pi.mode() (owner greedy).

    BATCH-1 WORKAROUND (disclosed in every binding as `batch1_workaround`):
    the owner's dicode transformerXL.forward_eval does `x = x.squeeze()` after
    EVERY transformer layer; at batch size 1 that also removes the batch
    dimension, so layer 2's `jnp.concatenate([memories[:, :, i], x[:, None]])`
    mismatches ((1,128,256) vs (256,1) — observed on server 2026-07-31). The
    owner's own harness NEVER runs at B=1 (build_stage4_env smoke_batch_size
    >= 2; eval_bakeoff NUM_ENVS=256 — vectorized). Rows are fully independent
    (per-row Dense encoder / per-row attention; no cross-batch op anywhere in
    forward_eval), so this adapter calls the owner's policy_step UNMODIFIED at
    batch 2 with the row DUPLICATED and reads action/memory from row 0 —
    numerically identical to the intended batch-1 semantics. This is a
    protocol-shell batching choice (the projection's defined job), NOT a
    network modification; both CC1 candidates use the identical adapter."""

    EFFECTIVE_BATCH = 2
    READOUT_ROW = 0

    def __init__(self, module, loaded, R):
        self.module = module
        self.loaded = loaded
        self.R = R
        self.ms = None
        self.batch1_workaround = {
            "applied": True,
            "effective_batch": self.EFFECTIVE_BATCH,
            "rows_duplicated": True,
            "readout_row": self.READOUT_ROW,
            "reason": ("owner dicode transformerXL.forward_eval x.squeeze() "
                       "removes the batch dim at B=1 (layer-2 concat TypeError "
                       "(1,128,256) vs (256,1)); owner harness is always "
                       "vectorized (smoke_batch_size>=2, NUM_ENVS=256) and "
                       "never hits B=1"),
            "fidelity_basis": ("forward_eval rows are fully independent "
                               "(per-row encoder/attention, no cross-batch "
                               "op) -> duplicated-row batch 2 with row-0 "
                               "readout is numerically identical to B=1"),
            "owner_code_modified": False}

    def reset(self):
        import jax
        import jax.numpy as jnp
        m = self.R.init_memory(1)
        # carry the owner memory state at batch 2 (two identical rows)
        self.ms = jax.tree_util.tree_map(
            lambda a: jnp.concatenate([a, a], axis=0), m)

    def __call__(self, obs, env_state):
        import jax.numpy as jnp
        import numpy as np
        if self.ms is None:
            self.reset()
        o1 = jnp.asarray(np.asarray(obs)[None, :])
        o = jnp.concatenate([o1, o1], axis=0)              # (2, obs_dim)
        done_mask = jnp.zeros((self.EFFECTIVE_BATCH,), dtype=jnp.bool_)
        out = self.module.policy_step(self.loaded, o, self.ms, done_mask,
                                      greedy=True, rng=None)
        self.ms = out["memory_state"]                      # both rows stay equal
        return int(np.asarray(out["action"]).reshape(-1)[self.READOUT_ROW])


class SlowGRUProjectionPolicy(object):
    """CC3 slowgru: owner policy_step ALWAYS samples an action, but the memory
    update (mem_out from the network forward) is ACTION-INDEPENDENT — so reading
    argmax(extras['logits'][row]) is the faithful greedy readout of the identical
    forward pass (no reimplementation, no behavior change). Segment boundary:
    on_segment_boundary every 128 env steps (owner-documented segment length);
    true_done/done_mask stay False — the engine STOPS an episode at env done and
    never steps past it.

    BATCH-1 WORKAROUND (disclosed in every binding as `batch1_workaround`):
    the owner's slowgru_network.forward_eval delegates to the SAME dicode
    transformerXL.forward_eval (byte-identical module, crash site
    transformerXL.py:194) as CC1's GTrXL128; that function does
    `x = x.squeeze()` after EVERY transformer layer, and at batch size 1
    that also removes the batch dimension, so layer 2's
    `jnp.concatenate([memories[:, :, i], x[:, None]])` mismatches
    ((1,128,256) vs (256,1) — observed on server 2026-07-31). The owner's
    trainer is ALWAYS vectorized (PPO `_env_step` over E envs; this runtime's
    docstring states it replicates that trainer mechanics VERBATIM) and never
    runs forward_eval at B=1. Every op is per-row: transformerXL.forward_eval
    is per-row encoder/attention (no cross-batch op); the owner's own
    `_slow_update` header comment reads "vectorised over env axis; no
    cross-env mixing" (per-row buffer write, per-row attention pooling,
    per-row GRUCell); the fast-memory roll is along axis=1; the mask
    mechanics are per-row; on_segment_boundary is per-row (RESET128:
    longstate -> init_longstate(B) for the whole batch; PERSISTENT:
    identity). So this adapter calls the owner's policy_step /
    on_segment_boundary UNMODIFIED at batch 2 with the rows DUPLICATED and
    reads the row-0 action — numerically identical to the intended batch-1
    semantics, INCLUDING §四 boundary semantics at the shell's effective
    batch (row 0 undergoes the same per-row op it would at B=1). This is a
    protocol-shell batching choice (the projection's defined job), NOT a
    network modification; owner code is untouched. Both CC3 candidates use
    the identical adapter; the two families still diverge solely through the
    owner's mode-dependent on_segment_boundary."""

    EFFECTIVE_BATCH = 2
    READOUT_ROW = 0

    def __init__(self, module, segment_boundary_steps=SLOWGRU_SEGMENT_BOUNDARY_STEPS):
        self.module = module
        self.segment_boundary_steps = int(segment_boundary_steps)
        self.ms = None
        self.steps = 0
        self.boundary_invocations = 0
        self.boundary_info_log = []
        self.batch1_workaround = {
            "applied": True,
            "effective_batch": self.EFFECTIVE_BATCH,
            "rows_duplicated": True,
            "readout_row": self.READOUT_ROW,
            "reason": ("owner slowgru_network.forward_eval -> dicode "
                       "transformerXL.forward_eval x.squeeze() removes the "
                       "batch dim at B=1 (layer-2 concat TypeError "
                       "(1,128,256) vs (256,1) at transformerXL.py:194 — "
                       "SAME crash site and byte-identical module as CC1 "
                       "GTrXL128); owner trainer is always vectorized (PPO "
                       "_env_step over E envs) and never hits B=1"),
            "fidelity_basis": ("all ops per-row: transformerXL.forward_eval "
                               "per-row encoder/attention; owner _slow_update "
                               "own comment 'vectorised over env axis; no "
                               "cross-env mixing' (per-row buffer/pooling/"
                               "GRUCell); fast-memory roll axis=1; mask "
                               "mechanics per-row; on_segment_boundary "
                               "per-row -> duplicated-row batch 2 with row-0 "
                               "readout numerically identical to B=1, "
                               "including section-4 boundary semantics"),
            "owner_code_modified": False}

    def reset(self):
        import jax.numpy as jnp
        m = self.module.init_memory(1)
        # carry the owner memory state at batch 2 (two identical rows).
        # step_idx is a python int (non-array leaf) — kept an int.
        self.ms = dict(m)
        self.ms["memories"] = jnp.concatenate([m["memories"], m["memories"]], axis=0)
        self.ms["memories_mask"] = jnp.concatenate(
            [m["memories_mask"], m["memories_mask"]], axis=0)
        self.ms["memories_mask_idx"] = jnp.concatenate(
            [m["memories_mask_idx"], m["memories_mask_idx"]], axis=0)
        self.ms["longstate"] = dict(
            (k, jnp.concatenate([v, v], axis=0)) for k, v in m["longstate"].items())
        self.ms["true_done"] = jnp.concatenate([m["true_done"], m["true_done"]], axis=0)
        self.ms["step_idx"] = m["step_idx"]
        self.steps = 0

    def __call__(self, obs, env_state):
        import jax.numpy as jnp
        import numpy as np
        if self.ms is None:
            self.reset()
        o1 = jnp.asarray(np.asarray(obs)[None, :])
        o = jnp.concatenate([o1, o1], axis=0)              # (2, obs_dim)
        done_mask = jnp.zeros((self.EFFECTIVE_BATCH,), dtype=jnp.bool_)
        true_done = jnp.zeros((self.EFFECTIVE_BATCH,), dtype=jnp.bool_)
        _sampled_action, self.ms, extras = self.module.policy_step(
            o, self.ms, done_mask, true_done=true_done)
        # rows are fully independent and start equal -> both rows stay equal;
        # the row-0 greedy readout is the faithful argmax of the B=1 forward
        action = int(np.argmax(
            np.asarray(extras["logits"])[self.READOUT_ROW].reshape(-1)))
        self.steps += 1
        if self.segment_boundary_steps > 0 and \
                self.steps % self.segment_boundary_steps == 0:
            self.ms, info = self.module.on_segment_boundary(self.ms)
            self.boundary_invocations += 1
            self.boundary_info_log.append(
                {"after_step": self.steps,
                 "info": {str(k): str(v) for k, v in dict(info).items()}})
        return action


class Rmt16CapsuleProjectionPolicy(object):
    """CC4 RMT16 capsule: the frozen engine's CandidateRuntime.policy_step
    returns the owner CC2 GREEDY action (int(self._policy(obs, None)); the
    stateful RMT16 policy — memories/mask/idx/rmt_st snapshot — lives inside
    the engine runtime). init_memory(1) resets the policy and returns the
    opaque memory snapshot; policy_step(obs[1, obs_dim], memory_state, None)
    returns {"action": int, "memory_state": snapshot}.

    NO batch-1 workaround (unlike the CC1/CC3 shells): the engine ENFORCES
    batch exactly 1 (CandidateRuntime.check_batch_size fails closed on any
    other size — a batched rollout would silently change CC2's per-step
    dynamics), and CC2's rmt_step_forward pads 1->2 internally. Calling at
    B=1 with done_mask=None (the ABI default) is the engine's OWN protocol."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.ms = None
        self.batch1_workaround = {
            "applied": False,
            "reason": ("engine CandidateRuntime enforces batch==1 "
                       "(check_batch_size fails closed) and the owner CC2 "
                       "rmt_step_forward pads 1->2 internally; done_mask=None "
                       "is the engine ABI default"),
            "owner_code_modified": False}

    def reset(self):
        self.ms = self.runtime.init_memory(1)

    def __call__(self, obs, env_state):
        import jax.numpy as jnp
        import numpy as np
        if self.ms is None:
            self.reset()
        o = jnp.asarray(np.asarray(obs)[None, :])          # (1, obs_dim)
        out = self.runtime.policy_step(o, self.ms, None)
        self.ms = out["memory_state"]
        return int(np.asarray(out["action"]).reshape(-1)[0])


def build_policy(spec, ctx):
    kind = ctx["kind"]
    if kind == "cc2_base_gtrxl":
        return BaseGtrxlProjectionPolicy(ctx["candidate"])
    if kind == "cc1_gtrxl128":
        return GTrXL128ProjectionPolicy(ctx["module"], ctx["loaded"], ctx["R"])
    if kind == "cc3_slowgru":
        return SlowGRUProjectionPolicy(ctx["module"],
                                       SLOWGRU_SEGMENT_BOUNDARY_STEPS)
    if kind == "cc4_rmt16_capsule":
        return Rmt16CapsuleProjectionPolicy(ctx["runtime"])
    raise FailClosed("FAIL CLOSED: unknown loader kind %r" % kind)


# ---------------------------------------------------------------------------
# SlowGRU segment-boundary unit check (contract §四): the two families MUST
# diverge. Direct owner-semantics check, independent of smoke length (a 32-step
# smoke never reaches the 128-step boundary).
# ---------------------------------------------------------------------------
def slowgru_boundary_unit_check(module, carry_mode):
    """Perturb longstate leaves (+1.0) -> on_segment_boundary -> assert owner
    semantics per carry mode (info strings + longstate leaf behavior)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    require(carry_mode in ("RESET128", "PERSISTENT"),
            "FAIL CLOSED: unknown carry_mode %r" % carry_mode)
    ms = module.init_memory(1)
    init_ls = ms["longstate"]
    init_leaves = [np.asarray(l).copy() for l in jax.tree_util.tree_leaves(init_ls)]
    perturbed = jax.tree_util.tree_map(lambda l: jnp.asarray(l) + 1.0, init_ls)
    ms_p = dict(ms)
    ms_p["longstate"] = perturbed
    fast_before = [np.asarray(l).copy()
                   for l in jax.tree_util.tree_leaves(ms_p["memories"])]
    ms_out, info = module.on_segment_boundary(ms_p)
    info = {str(k): str(v) for k, v in dict(info).items()}
    out_leaves = [np.asarray(l) for l in jax.tree_util.tree_leaves(ms_out["longstate"])]
    fast_after = [np.asarray(l) for l in jax.tree_util.tree_leaves(ms_out["memories"])]
    fast_carried = all(np.array_equal(a, b) for a, b in zip(fast_before, fast_after))
    if carry_mode == "RESET128":
        restored = (len(out_leaves) == len(init_leaves)
                    and all(np.array_equal(a, b)
                            for a, b in zip(out_leaves, init_leaves)))
        require(restored,
                "FAIL CLOSED (RESET128_BOUNDARY_SEMANTICS): longstate was NOT "
                "restored to init after on_segment_boundary")
        require(info.get("boundary_action") == "LONGSTATE_RESET_TO_INIT",
                "FAIL CLOSED (RESET128_BOUNDARY_SEMANTICS): boundary_action %r "
                "!= LONGSTATE_RESET_TO_INIT" % info.get("boundary_action"))
        require(fast_carried,
                "FAIL CLOSED (RESET128_BOUNDARY_SEMANTICS): fast memories not carried")
    else:
        perturbed_leaves = [np.asarray(l) for l in jax.tree_util.tree_leaves(perturbed)]
        kept = (len(out_leaves) == len(perturbed_leaves)
                and all(np.array_equal(a, b)
                        for a, b in zip(out_leaves, perturbed_leaves)))
        require(kept,
                "FAIL CLOSED (PERSISTENT_BOUNDARY_SEMANTICS): longstate was "
                "cleared after on_segment_boundary (must FULL_CARRY)")
        require(info.get("boundary_action") == "FULL_CARRY_NO_CLEAR",
                "FAIL CLOSED (PERSISTENT_BOUNDARY_SEMANTICS): boundary_action %r "
                "!= FULL_CARRY_NO_CLEAR" % info.get("boundary_action"))
    return {"carry_mode": carry_mode, "boundary_info": info,
            "longstate_restored_to_init": bool(carry_mode == "RESET128"),
            "longstate_full_carried": bool(carry_mode == "PERSISTENT"),
            "fast_memories_carried": bool(fast_carried),
            "segment_boundary_steps": SLOWGRU_SEGMENT_BOUNDARY_STEPS}


# ---------------------------------------------------------------------------
# PROJECTION REGISTRY — the seven CC4-authored projection families (6 students
# + 1 teacher reference). All SHA constants below were audited 2026-07-31 from
# owner capsule SHA256SUMS / contracts / READY / interface-smoke records
# (read-only) and re-verified on disk; declaration_source cites the owner
# artifact each declared full64 comes from. The two RMT16 entries bind CC4's
# own V1-era capsules (owner load path: shim -> FROZEN V1 common runner ->
# FROZEN engine rmt16_gtrxl_cc2 family loader); evaluation binding is V2.
# ---------------------------------------------------------------------------
PROJECTION_REGISTRY = {
    "BASE_GTRXL_ORIGINAL_VTRACE_98304": {
        "candidate_id": "BASE_GTRXL_ORIGINAL_VTRACE_98304",
        "runtime_family": "base_gtrxl_cc2_projection",
        "owner": "CC2",
        "candidate_class": "STUDENT",
        "formal_student_ranking_eligible": True,
        "strong_student_selection_eligible": True,
        "reference_only": False,
        "student_rank": None,
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "loader_kind": "cc2_base_gtrxl",
        "checkpoint_kind": "pkl_file",
        "network_family": "RMT16_GTrXL (CC2 network_rmt16 + rmt_memory_anchor)",
        "memory_mode": "rmt16_window128_gtrxl",
        "carry_mode": "base_gtrxl",
        "source_capsule_root": "/home/oseasy/student_pool_v1/cc2/BASE_GTRXL_ORIGINAL_VTRACE_98304",
        "capsule_file_sha256": {
            "candidate_runtime.py": "31e28eb60fd89eb482cdde0273f236948f920ad053740a07eafda3e8bc03e4fb",
            "candidate_manifest.json": "6105b62518134101bdae9c7096b3f50662de4f6fdf60395d7d1e47431620cb19",
            "checkpoint_contract.json": "e63a3481cfe00714232bdc3bac1584c3ceeab709db5e539fb5c98ff89d5aa401",
            "evaluate_candidate.py": "5e9a1f380c7279341e956013da410543098d695d3b346069c581f2319dfb7ac4",
        },
        "declared_params_sha256": {
            "value": "78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2",
            "declaration_source": "cc2 capsule checkpoint_contract.json params_sha256 "
                "(contract sha e63a3481…); owner load_candidate recomputes "
                "canonical_params_sha and raises CHECKPOINT_PARAMS_SHA_MISMATCH",
        },
        "declared_checkpoint_file_sha256": {
            "value": "d71e30aebb307c6fc5b404543a5ba3e32b30e25857905ae20300be46600713ea",
            "declaration_source": "cc2 capsule checkpoint_contract.json "
                "checkpoint_file_sha256 (sha256 of pkl bytes)",
        },
        "params_hash_protocol": "cc2 candidate_runtime.canonical_params_sha: sha256 over "
            "per-leaf np.ascontiguousarray(np.asarray(leaf)).tobytes() in tree_leaves order",
        "checkpoint_file_hash_protocol": "sha256 of pkl file bytes",
        "bound_owner_runtime_sha256": {
            "candidate_runtime.py": "31e28eb60fd89eb482cdde0273f236948f920ad053740a07eafda3e8bc03e4fb",
            "network_rmt16.py": CC2_FROZEN_MODULE_FILES["network_rmt16.py"],
            "rmt_memory_anchor.py": CC2_FROZEN_MODULE_FILES["rmt_memory_anchor.py"],
            "rmt16_memory.py": CC2_FROZEN_MODULE_FILES["rmt16_memory.py"],
        },
        "gpu_allowed_cc4": list(CC4_GPU_ALLOWED_UUIDS),
    },
    # The two FORMAL RMT16 candidates (standing CC4 ownership). Owner plumbing:
    # the CC4 V1-era capsule's thin candidate_runtime.py binds to the FROZEN V1
    # common runner (FROZEN_COMMON_RUNNER_SHA256), which delegates to the FROZEN
    # engine tier3_candidate_runtime.py (LF-SHA 6af09be4…) — its registered
    # rmt16_gtrxl_cc2 family loader verifies the checkpoint (file SHA + params
    # SHA + driver-source SHA + CC2 policy-source SHA) against the frozen
    # final98304 contract, exactly as tier3_evaluator does. CC4 defines NO
    # value here: every SHA is the capsule/contract-declared owner value, and
    # the loader + driver recompute them independently (fail closed).
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304": {
        "candidate_id": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
        "runtime_family": "rmt16_gtrxl_cc2_persistent_projection",
        "engine_runtime_family": "rmt16_gtrxl_cc2",
        "owner": "CC4 capsule over CC2 original-vtrace arm",
        "candidate_class": "STUDENT",
        "formal_student_ranking_eligible": True,
        "strong_student_selection_eligible": True,
        "reference_only": False,
        "student_rank": None,
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "loader_kind": "cc4_rmt16_capsule",
        "checkpoint_kind": "pkl_file",
        "network_family": "RMT16_GTrXL (CC2 network_rmt16 via the frozen "
                          "engine's registered rmt16_gtrxl_cc2 family loader)",
        "memory_mode": "rmt16_window128_persistent",
        "carry_mode": "persistent",
        "replay_mode": "original_vtrace",
        "segment_len": 128,
        "checkpoint_path": "/home/oseasy/cc2_data/cc2_runs_76b294b/runs/"
                           "RMT16-LONG98304-PERSISTENT/ckpt/98304/full_state.pkl",
        "source_capsule_root": "/home/oseasy/student_pool_v1/cc4/"
                               "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
        "capsule_file_sha256": {
            "candidate_runtime.py": "12f2308ac5a8ddba19f62f6d63900fd375ef845e8e1c8062c3b914bd258c374d",
            "candidate_manifest.json": "e50a04380bafada12862a87088e2e1280b726c4b502fc56856f89c69454834a4",
            "checkpoint_contract.json": "1517483077ec9905030d36d401d216f995b5596f1eea2993bc28fcacbadca4d7",
            "evaluate_candidate.py": "143dfc1ec61816de4b709ec4647d666896c54a07a0666b5eeda82eb93e651a88",
        },
        "declared_params_sha256": {
            "value": "aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d",
            "declaration_source": "cc4 capsule candidate_manifest.json params_sha256 "
                "+ checkpoint_contract.json arms.persistent.params_sha256 "
                "(contract content sha 7dda2bc7…); the frozen engine family "
                "loader recomputes it from the pkl bytes and gates it against "
                "the contract (verify_checkpoint_against_contract)",
        },
        "declared_checkpoint_file_sha256": {
            "value": "2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723",
            "declaration_source": "cc4 capsule candidate_manifest.json "
                "checkpoint_file_sha256 + checkpoint_contract.json "
                "arms.persistent.checkpoint_file_sha256 (sha256 of pkl bytes)",
        },
        "params_hash_protocol": "tier3_checkpoint_adapter.load_full_params_readonly "
            "(frozen engine verification path): CC2 canonical params sha256 — "
            "sha256 over per-leaf np.ascontiguousarray(np.asarray(leaf)).tobytes() "
            "in jax tree_leaves order, recomputed from the pkl bytes",
        "checkpoint_file_hash_protocol": "sha256 of pkl file bytes",
        "bound_owner_runtime_sha256": {
            "candidate_runtime.py": "12f2308ac5a8ddba19f62f6d63900fd375ef845e8e1c8062c3b914bd258c374d",
            "common_runner.py (frozen V1)": FROZEN_COMMON_RUNNER_SHA256,
            "tier3_candidate_runtime.py (frozen engine LF-SHA)":
                FROZEN_ENGINE_LF_SHA256["tier3_candidate_runtime.py"],
        },
        "gpu_allowed_cc4": list(CC4_GPU_ALLOWED_UUIDS),
    },
    "RESET128_RMT16_ORIGINAL_VTRACE_98304": {
        "candidate_id": "RESET128_RMT16_ORIGINAL_VTRACE_98304",
        "runtime_family": "rmt16_gtrxl_cc2_reset128_projection",
        "engine_runtime_family": "rmt16_gtrxl_cc2",
        "owner": "CC4 capsule over CC2 original-vtrace arm",
        "candidate_class": "STUDENT",
        "formal_student_ranking_eligible": True,
        "strong_student_selection_eligible": True,
        "reference_only": False,
        "student_rank": None,
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "loader_kind": "cc4_rmt16_capsule",
        "checkpoint_kind": "pkl_file",
        "network_family": "RMT16_GTrXL (CC2 network_rmt16 via the frozen "
                          "engine's registered rmt16_gtrxl_cc2 family loader)",
        "memory_mode": "rmt16_window128_reset128",
        "carry_mode": "reset128",
        "replay_mode": "original_vtrace",
        "segment_len": 128,
        "checkpoint_path": "/home/oseasy/cc2_data/cc2_runs_76b294b/runs/"
                           "RMT16-LONG98304-RESET128/ckpt/98304/full_state.pkl",
        "source_capsule_root": "/home/oseasy/student_pool_v1/cc4/"
                               "RESET128_RMT16_ORIGINAL_VTRACE_98304",
        "capsule_file_sha256": {
            "candidate_runtime.py": "6d69902213a6051d2b2d40e1ad4975deca635d47bf441b3cd5aee54252bcf72c",
            "candidate_manifest.json": "d79a04e817395c8db8fa935235492a1c30f4cdd12346c00eef3f7ba9c3d77fe4",
            "checkpoint_contract.json": "1517483077ec9905030d36d401d216f995b5596f1eea2993bc28fcacbadca4d7",
            "evaluate_candidate.py": "fb7a1d321ebed135a4d28fd16676f44385159fa9604624f0e557430774fe736e",
        },
        "declared_params_sha256": {
            "value": "78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2",
            "declaration_source": "cc4 capsule candidate_manifest.json params_sha256 "
                "+ checkpoint_contract.json arms.reset128.params_sha256 "
                "(contract content sha 7dda2bc7…); the frozen engine family "
                "loader recomputes it from the pkl bytes and gates it against "
                "the contract (verify_checkpoint_against_contract)",
        },
        "declared_checkpoint_file_sha256": {
            "value": "de3a159f58f904c4ed0bce17bcb87e4b39b21b4ffd0cea557ce61b860727b638",
            "declaration_source": "cc4 capsule candidate_manifest.json "
                "checkpoint_file_sha256 + checkpoint_contract.json "
                "arms.reset128.checkpoint_file_sha256 (sha256 of pkl bytes)",
        },
        "params_hash_protocol": "tier3_checkpoint_adapter.load_full_params_readonly "
            "(frozen engine verification path): CC2 canonical params sha256 — "
            "sha256 over per-leaf np.ascontiguousarray(np.asarray(leaf)).tobytes() "
            "in jax tree_leaves order, recomputed from the pkl bytes",
        "checkpoint_file_hash_protocol": "sha256 of pkl file bytes",
        "bound_owner_runtime_sha256": {
            "candidate_runtime.py": "6d69902213a6051d2b2d40e1ad4975deca635d47bf441b3cd5aee54252bcf72c",
            "common_runner.py (frozen V1)": FROZEN_COMMON_RUNNER_SHA256,
            "tier3_candidate_runtime.py (frozen engine LF-SHA)":
                FROZEN_ENGINE_LF_SHA256["tier3_candidate_runtime.py"],
        },
        "gpu_allowed_cc4": list(CC4_GPU_ALLOWED_UUIDS),
    },
    "CONTROL_CONTINUOUS_98304": {
        "candidate_id": "CONTROL_CONTINUOUS_98304",
        "runtime_family": "gtrxl128_cc1_control_projection",
        "owner": "CC1",
        "candidate_class": "STUDENT",
        "formal_student_ranking_eligible": True,
        "strong_student_selection_eligible": True,
        "reference_only": False,
        "student_rank": None,
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "loader_kind": "cc1_gtrxl128",
        "checkpoint_kind": "orbax_dir",
        "network_family": "GTrXL128 (dicode.network.ActorCriticTransformer)",
        "memory_mode": "gtrxl_window128",
        "carry_mode": "gtrxl_window128",
        "source_capsule_root": "/home/oseasy/student_pool_v1/cc1/CONTROL_CONTINUOUS_98304",
        "capsule_file_sha256": {
            "candidate_runtime.py": "ed1a5c3f85dc79093149a3a9aece92ad99524ffb881e1a10e95adb9d77b1305b",
            "candidate_manifest.json": "56cb71ce3ce356ca0667ee6e2881d13fecbb15bceb4f9046d16c986210180148",
            "checkpoint_contract.json": "4c29226dd8c4535c1847cc1760a5edda6653a859c6e86ffaa63c6749f8fd4d84",
            "evaluate_candidate.py": "55569c02c8fe758279310d9fc247d1408e3ae5f6d4f5f52ba3c21f3941f568e6",
        },
        "declared_params_sha256": {
            "value": "4c313c58d2d01f44c98ceb2e18580577e7735cd210e5c3d9b4ee3db3b286d505",
            "declaration_source": "cc1 owner capsule READY.json + interface_smoke_result.json "
                "(/params_sha256, /checks/C3_params_sha256, /checks/C8_params_sha_after); "
                "the capsule contract intentionally carries "
                "expected_params_sha256=PENDING_COMPUTED_IN_SMOKE (owner design) — "
                "CC4 cites the owner-computed smoke/READY values, defines nothing",
        },
        "declared_checkpoint_file_sha256": {
            "value": "34819d779ad91e59cd62836c803ba9226ffa3623368c12d74044e58cd6db9913",
            "declaration_source": "cc1 owner capsule READY.json + interface_smoke_result.json "
                "(/checkpoint_file_sha256, /checks/C2_file_sha256); computed with "
                "gtrxl128_reference_runtime.dir_sha256 (CC1 orbax directory protocol)",
        },
        "params_hash_protocol": "gtrxl128_reference_runtime.params_sha256: "
            "sha256(b''.join(np.asarray(leaf).tobytes() for leaf in tree_leaves)) "
            "(eval_bakeoff convention; NO ascontiguousarray)",
        "checkpoint_file_hash_protocol": "gtrxl128_reference_runtime.dir_sha256: sha256 over "
            "sorted (relpath, file_sha256); per pair update(rel.encode('utf-8')); "
            "update(b'\\0'); update(sha.encode('ascii')); update(b'\\n')",
        "bound_owner_runtime_sha256": {
            "candidate_runtime.py": "ed1a5c3f85dc79093149a3a9aece92ad99524ffb881e1a10e95adb9d77b1305b",
            "gtrxl128_reference_runtime.py": CC1_GTRXL128_REFERENCE_RUNTIME_SHA256,
            "dicode/network.py": FROZEN_DICODE_NETWORK_SHA256,
        },
        "gpu_allowed_cc4": list(CC4_GPU_ALLOWED_UUIDS),
    },
    "SLOWGRU_RESET128_CANONICAL_98304": {
        "candidate_id": "SLOWGRU_RESET128_CANONICAL_98304",
        "runtime_family": "slowgru_reset128_cc3_projection",
        "owner": "CC3",
        "candidate_class": "STUDENT",
        "formal_student_ranking_eligible": True,
        "strong_student_selection_eligible": True,
        "reference_only": False,
        "student_rank": None,
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "loader_kind": "cc3_slowgru",
        "checkpoint_kind": "pkl_file",
        "network_family": "GTrXL128 + SlowGRU longstate (slowgru_network)",
        "memory_mode": "gtrxl_window128_slowgru_longstate",
        "carry_mode": "RESET128",
        "segment_boundary_steps": SLOWGRU_SEGMENT_BOUNDARY_STEPS,
        "smoke_policy_rng_seed": SLOWGRU_SMOKE_POLICY_RNG_SEED,
        "boundary_semantics": "on_segment_boundary: longstate=init_longstate(B) "
            "(LONGSTATE_RESET_TO_INIT), fast memories CARRIED — segment boundary reset",
        "source_capsule_root": "/home/oseasy/student_pool_v1/cc3/SLOWGRU_RESET128_CANONICAL_98304",
        "capsule_file_sha256": {
            "candidate_runtime.py": "e3fcd9a6553af7fcb6c67a072e8c5c0ef6f00a23ac9110c6d1666371505fe9c1",
            "candidate_manifest.json": "ef7c1fe256adb0a3075deca87b1bfc29f75c375f7f0e4fcc694fea506c6358a5",
            "checkpoint_contract.json": "f2c8998741d6479d401ccb79b3beebe65d605ca0346aa5bc075dcfab56b19381",
            "evaluate_candidate.py": "6ba8f777c4314f68fa184dc8016365a184ddd6bf70aeb5236bf5c4ba532edbd4",
        },
        "declared_params_sha256": {
            "value": "9d92c5b9e2e2148b2375c59f7f595d53b95f924d62436ebdccf8bf9ea3d59247",
            "declaration_source": "cc3 capsule checkpoint_contract.json (contract sha "
                "f2c89987…); owner loader gates file SHA + arm network_src SHA + "
                "params SHA fail closed",
        },
        "declared_checkpoint_file_sha256": {
            "value": "2c065fa88bcc8cfcb193deda6ef599522238b99bf7151f5eeab0b70e4420f2de",
            "declaration_source": "cc3 capsule checkpoint_contract.json "
                "checkpoint_file_sha256 (sha256 of pkl bytes)",
        },
        "params_hash_protocol": "slowgru_runtime.params_sha_packed over the pkl-embedded "
            "(leaves, treedef): sha256 of per-leaf ascontiguousarray.tobytes()",
        "checkpoint_file_hash_protocol": "sha256 of pkl file bytes",
        "bound_owner_runtime_sha256": {
            "candidate_runtime.py": "e3fcd9a6553af7fcb6c67a072e8c5c0ef6f00a23ac9110c6d1666371505fe9c1",
            "slowgru_runtime.py": CC3_SLOWGRU_RUNTIME_SHA256,
            "slowgru_network.py (arm_src)": CC3_SLOWGRU_NETWORK_SHA256,
        },
        "gpu_allowed_cc4": list(CC4_GPU_ALLOWED_UUIDS),
    },
    "SLOWGRU_PERSISTENT_CANONICAL_98304": {
        "candidate_id": "SLOWGRU_PERSISTENT_CANONICAL_98304",
        "runtime_family": "slowgru_persistent_cc3_projection",
        "owner": "CC3",
        "candidate_class": "STUDENT",
        "formal_student_ranking_eligible": True,
        "strong_student_selection_eligible": True,
        "reference_only": False,
        "student_rank": None,
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "loader_kind": "cc3_slowgru",
        "checkpoint_kind": "pkl_file",
        "network_family": "GTrXL128 + SlowGRU longstate (slowgru_network)",
        "memory_mode": "gtrxl_window128_slowgru_longstate",
        "carry_mode": "PERSISTENT",
        "segment_boundary_steps": SLOWGRU_SEGMENT_BOUNDARY_STEPS,
        "smoke_policy_rng_seed": SLOWGRU_SMOKE_POLICY_RNG_SEED,
        "boundary_semantics": "on_segment_boundary: memory returned UNCHANGED "
            "(FULL_CARRY_NO_CLEAR) — cross-segment carry preserved",
        "source_capsule_root": "/home/oseasy/student_pool_v1/cc3/SLOWGRU_PERSISTENT_CANONICAL_98304",
        "capsule_file_sha256": {
            "candidate_runtime.py": "a450029c9da347d975a515e7baf9a2c678e6e4ce159d83c9e9883243cadd25e4",
            "candidate_manifest.json": "8f74a3182f392704441aa36c3d94d0748a13a3273cc409af17379d88d86b9412",
            "checkpoint_contract.json": "7f0e731b4f4605ba4e39f7a23168a4604ebde23a12658080338a7c55a02d1ae9",
            "evaluate_candidate.py": "001f93c19e311825a21e8f3b015e3f83e742f30f99c656d9d2a918d79a32ff6f",
        },
        "declared_params_sha256": {
            "value": "99d734b48acfd3499e5b836c7f632a52b1d17a732c3764a24c1935fd82a77ecc",
            "declaration_source": "cc3 capsule checkpoint_contract.json (contract sha "
                "7f0e731b…); owner loader triple fail-closed gate; load_candidate "
                "raises CARRY_MODE_MISMATCH on any drift",
        },
        "declared_checkpoint_file_sha256": {
            "value": "0bc92c9ee28684ba507d6d6d728110000f11d7115126fbaf9137b1f8390a9c47",
            "declaration_source": "cc3 capsule checkpoint_contract.json "
                "checkpoint_file_sha256 (sha256 of pkl bytes)",
        },
        "params_hash_protocol": "slowgru_runtime.params_sha_packed over the pkl-embedded "
            "(leaves, treedef): sha256 of per-leaf ascontiguousarray.tobytes()",
        "checkpoint_file_hash_protocol": "sha256 of pkl file bytes",
        "bound_owner_runtime_sha256": {
            "candidate_runtime.py": "a450029c9da347d975a515e7baf9a2c678e6e4ce159d83c9e9883243cadd25e4",
            "slowgru_runtime.py": CC3_SLOWGRU_RUNTIME_SHA256,
            "slowgru_network.py (arm_src)": CC3_SLOWGRU_NETWORK_SHA256,
        },
        "gpu_allowed_cc4": list(CC4_GPU_ALLOWED_UUIDS),
    },
    "BASELINE_TEACHER_CKPT17500": {
        "candidate_id": "BASELINE_TEACHER_CKPT17500",
        "runtime_family": "gtrxl128_cc1_teacher_reference_projection",
        "owner": "CC1",
        "candidate_class": "TEACHER_REFERENCE",
        # Contract §五: the teacher is a REFERENCE ONLY — never in the 6-student
        # ranking; its binding may PASS but must NOT count toward
        # STUDENT_COMMON_BINDING_PASS_COUNT.
        "formal_student_ranking_eligible": False,
        "strong_student_selection_eligible": False,
        "reference_only": True,
        "student_rank": None,
        "budget_class": "UNMATCHED_REFERENCE",
        "training_steps": 17500,
        "training_seed": 42,
        "counts_toward_student_binding_count": False,
        "loader_kind": "cc1_gtrxl128",
        "checkpoint_kind": "pkl_file",
        "network_family": "GTrXL128 (dicode.network.ActorCriticTransformer)",
        "memory_mode": "gtrxl_window128",
        "carry_mode": "gtrxl_window128",
        "source_capsule_root": "/home/oseasy/student_pool_v1/cc1/BASELINE_TEACHER_CKPT17500",
        "capsule_file_sha256": {
            "candidate_runtime.py": "ed1a5c3f85dc79093149a3a9aece92ad99524ffb881e1a10e95adb9d77b1305b",
            "candidate_manifest.json": "addee6e757edf52d12302844901ae7f29e22e0772656c6938820f9f926fc7041",
            "checkpoint_contract.json": "58f5623d54f1941b3bd38c37a0096479f885369459dcaa52cd5be48e372f58cd",
            "evaluate_candidate.py": "55569c02c8fe758279310d9fc247d1408e3ae5f6d4f5f52ba3c21f3941f568e6",
        },
        "declared_params_sha256": {
            "value": "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5",
            "declaration_source": "teacher capsule checkpoint_contract.json "
                "expected_params_sha256 (== canonical base params)",
        },
        "declared_checkpoint_file_sha256": {
            "value": "a87924a34d898fceed874c16e7332703fe960f02abaa2f8443efaecdb7482d01",
            "declaration_source": "teacher capsule checkpoint_contract.json "
                "(file_sha256_method='sha256 of file bytes'); CC4 pool-readiness "
                "round already recomputed this value MATCH",
        },
        "params_hash_protocol": "gtrxl128_reference_runtime.params_sha256 "
            "(eval_bakeoff convention; same as CONTROL)",
        "checkpoint_file_hash_protocol": "sha256 of pkl file bytes "
            "(contract file_sha256_method)",
        "bound_owner_runtime_sha256": {
            "candidate_runtime.py": "ed1a5c3f85dc79093149a3a9aece92ad99524ffb881e1a10e95adb9d77b1305b",
            "gtrxl128_reference_runtime.py": CC1_GTRXL128_REFERENCE_RUNTIME_SHA256,
            "dicode/network.py": FROZEN_DICODE_NETWORK_SHA256,
        },
        "gpu_allowed_cc4": list(CC4_GPU_ALLOWED_UUIDS),
    },
}


def get_spec(candidate_id):
    require(candidate_id in PROJECTION_REGISTRY,
            "FAIL CLOSED: unknown candidate_id %r (registered: %s)"
            % (candidate_id, sorted(PROJECTION_REGISTRY)))
    return PROJECTION_REGISTRY[candidate_id]


def load_owner_runtime(spec):
    kind = spec["loader_kind"]
    if kind == "cc2_base_gtrxl":
        return load_cc2_base_gtrxl(spec)
    if kind == "cc1_gtrxl128":
        return load_cc1_gtrxl128(spec)
    if kind == "cc3_slowgru":
        return load_cc3_slowgru(spec)
    if kind == "cc4_rmt16_capsule":
        return load_cc4_rmt16_capsule(spec)
    raise FailClosed("FAIL CLOSED: unknown loader_kind %r" % kind)


# ---------------------------------------------------------------------------
# Pure structural self-test (NO jax / numpy / owner imports — runs anywhere).
# ---------------------------------------------------------------------------
def self_test():
    n = 0

    def check(cond, msg):
        nonlocal n
        require(cond, "FAIL CLOSED (self-test): %s" % msg)
        n += 1

    check(len(PROJECTION_REGISTRY) == 7, "registry must hold exactly 7 projections")
    families = [s["runtime_family"] for s in PROJECTION_REGISTRY.values()]
    check(len(set(families)) == 7, "runtime families must be pairwise distinct: %s" % families)
    students = [cid for cid, s in PROJECTION_REGISTRY.items()
                if s["candidate_class"] == "STUDENT"]
    teachers = [cid for cid, s in PROJECTION_REGISTRY.items()
                if s["candidate_class"] == "TEACHER_REFERENCE"]
    check(len(students) == 6, "exactly 6 student projections")
    check(teachers == ["BASELINE_TEACHER_CKPT17500"], "exactly 1 teacher reference")

    for cid, s in PROJECTION_REGISTRY.items():
        check(cid == s["candidate_id"], "%s: id mismatch" % cid)
        for fn in CAPSULE_FILES:
            check(_is_hex64(s["capsule_file_sha256"][fn]),
                  "%s: capsule %s sha not full64" % (cid, fn))
        check(_is_hex64(s["declared_params_sha256"]["value"]),
              "%s: declared params sha not full64" % cid)
        check(_is_hex64(s["declared_checkpoint_file_sha256"]["value"]),
              "%s: declared checkpoint sha not full64" % cid)
        check(bool(s["declared_params_sha256"]["declaration_source"]),
              "%s: params declaration_source missing" % cid)
        check(bool(s["declared_checkpoint_file_sha256"]["declaration_source"]),
              "%s: checkpoint declaration_source missing" % cid)
        for v in s["bound_owner_runtime_sha256"].values():
            check(_is_hex64(v), "%s: bound runtime sha not full64" % cid)
        check(s["student_rank"] is None, "%s: student_rank must be null pre-ranking" % cid)
        check(tuple(s["gpu_allowed_cc4"]) == CC4_GPU_ALLOWED_UUIDS,
              "%s: gpu allowlist mismatch" % cid)

    # Student/teacher eligibility invariants (contract §五)
    for cid in students:
        s = PROJECTION_REGISTRY[cid]
        check(s["formal_student_ranking_eligible"] is True, "%s ranking eligibility" % cid)
        check(s["strong_student_selection_eligible"] is True, "%s selection eligibility" % cid)
        check(s["reference_only"] is False, "%s reference_only" % cid)
        check(s["budget_class"] == "MATCHED_98304", "%s budget" % cid)
        check(s["training_steps"] == 98304, "%s steps" % cid)
    t = PROJECTION_REGISTRY["BASELINE_TEACHER_CKPT17500"]
    check(t["formal_student_ranking_eligible"] is False, "teacher ranking eligibility")
    check(t["strong_student_selection_eligible"] is False, "teacher selection eligibility")
    check(t["reference_only"] is True, "teacher reference_only")
    check(t["budget_class"] == "UNMATCHED_REFERENCE", "teacher budget")
    check(t["training_steps"] == 17500, "teacher steps")
    check(t["counts_toward_student_binding_count"] is False,
          "teacher must not count toward STUDENT_COMMON_BINDING_PASS_COUNT")

    # The two slowgru families are DISTINCT (contract §四)
    r = PROJECTION_REGISTRY["SLOWGRU_RESET128_CANONICAL_98304"]
    p = PROJECTION_REGISTRY["SLOWGRU_PERSISTENT_CANONICAL_98304"]
    check(r["carry_mode"] == "RESET128" and p["carry_mode"] == "PERSISTENT",
          "slowgru carry modes must differ")
    check(r["runtime_family"] != p["runtime_family"], "slowgru families must differ")
    check(r["capsule_file_sha256"]["candidate_runtime.py"]
          != p["capsule_file_sha256"]["candidate_runtime.py"],
          "slowgru capsule runtimes must differ")
    check(r["segment_boundary_steps"] == p["segment_boundary_steps"] == 128,
          "segment boundary steps")
    check(r["bound_owner_runtime_sha256"]["slowgru_runtime.py"]
          == p["bound_owner_runtime_sha256"]["slowgru_runtime.py"]
          == CC3_SLOWGRU_RUNTIME_SHA256, "shared slowgru_runtime SHA")

    # The two formal RMT16 candidates are DISTINCT arms of ONE engine family
    # (contract §六): distinct projection families (pairwise-distinct
    # invariant), shared engine family / V1 runner / engine LF-SHA / frozen
    # contract copy; distinct capsules, params and checkpoint files.
    rp = PROJECTION_REGISTRY["PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"]
    rr = PROJECTION_REGISTRY["RESET128_RMT16_ORIGINAL_VTRACE_98304"]
    check(rp["loader_kind"] == rr["loader_kind"] == "cc4_rmt16_capsule",
          "rmt16 loader kind")
    check(rp["carry_mode"] == "persistent" and rr["carry_mode"] == "reset128",
          "rmt16 carry modes must differ")
    check(rp["runtime_family"] != rr["runtime_family"],
          "rmt16 projection families must differ")
    check(rp["engine_runtime_family"] == rr["engine_runtime_family"]
          == RMT16_ENGINE_RUNTIME_FAMILY,
          "rmt16 arms share the single registered engine family rmt16_gtrxl_cc2")
    check(rp["capsule_file_sha256"]["candidate_runtime.py"]
          != rr["capsule_file_sha256"]["candidate_runtime.py"],
          "rmt16 capsule runtimes must differ")
    check(rp["capsule_file_sha256"]["checkpoint_contract.json"]
          == rr["capsule_file_sha256"]["checkpoint_contract.json"],
          "rmt16 arms share the byte-identical frozen contract copy")
    check(rp["declared_params_sha256"]["value"]
          != rr["declared_params_sha256"]["value"],
          "rmt16 arm params must differ")
    check(rp["declared_checkpoint_file_sha256"]["value"]
          != rr["declared_checkpoint_file_sha256"]["value"],
          "rmt16 arm checkpoint files must differ")
    check(rp["bound_owner_runtime_sha256"]["common_runner.py (frozen V1)"]
          == rr["bound_owner_runtime_sha256"]["common_runner.py (frozen V1)"]
          == FROZEN_COMMON_RUNNER_SHA256, "rmt16 shared frozen V1 runner SHA")
    check(rp["bound_owner_runtime_sha256"]
          ["tier3_candidate_runtime.py (frozen engine LF-SHA)"]
          == rr["bound_owner_runtime_sha256"]
          ["tier3_candidate_runtime.py (frozen engine LF-SHA)"]
          == FROZEN_ENGINE_LF_SHA256["tier3_candidate_runtime.py"],
          "rmt16 shared frozen engine LF-SHA")

    # CC1 control + teacher share the identical thin-binding bytes (as audited)
    c = PROJECTION_REGISTRY["CONTROL_CONTINUOUS_98304"]
    check(c["capsule_file_sha256"]["candidate_runtime.py"]
          == t["capsule_file_sha256"]["candidate_runtime.py"]
          == "ed1a5c3f85dc79093149a3a9aece92ad99524ffb881e1a10e95adb9d77b1305b",
          "cc1 thin-binding identity")
    check(c["capsule_file_sha256"]["candidate_manifest.json"]
          != t["capsule_file_sha256"]["candidate_manifest.json"],
          "cc1 manifests must differ")

    # Frozen contract constants internal consistency
    check((FROZEN_FRONT_EPISODE_COUNT, FROZEN_BACK_EPISODE_COUNT,
           FROZEN_FULL_EPISODE_COUNT) == (8, 8, 64), "frozen episode scale 8/8/64")
    check(FROZEN_MAX_TIMESTEPS == 4096 and FROZEN_ACTION_DIM == 43
          and FROZEN_OBSERVATION_SHAPE == (8335,), "frozen contract invariants")
    for v in (FROZEN_COMMON_RUNNER_SHA256, FROZEN_COMMON_EVALUATOR_SHA256,
              FROZEN_EVALUATION_PROFILE_SHA256, FROZEN_METRIC_SCHEMA_SHA256,
              FROZEN_ENVIRONMENT_LOCK_SHA256, FROZEN_FULL_PROFILE_SHA256,
              FROZEN_FRONT_BANK_CONTENT_SHA256, FROZEN_BACK_BANK_CONTENT_SHA256,
              FROZEN_SHA256SUMS_SHA256, FROZEN_ABI_DOC_SHA256,
              FROZEN_ASSEMBLY_MANIFEST_SHA256, FROZEN_CC2_ADAPTER_LF_SHA256,
              CC1_GTRXL128_REFERENCE_RUNTIME_SHA256, FROZEN_DICODE_NETWORK_SHA256,
              CC1_TRAIN_STATE_UTILS_SHA256, CC3_SLOWGRU_RUNTIME_SHA256,
              CC3_SLOWGRU_NETWORK_SHA256,
              RMT16_FROZEN_CONTRACT_CONTENT_SHA256):
        check(_is_hex64(v), "frozen constant not full64: %r" % v)
    for v in FROZEN_ENGINE_LF_SHA256.values():
        check(_is_hex64(v), "engine LF sha not full64")
    for v in CC2_FROZEN_MODULE_FILES.values():
        check(_is_hex64(v), "cc2 frozen module sha not full64")
    check(FROZEN_COMMON_SUMS_ENTRY_COUNT == 57, "common sums entry count")

    # GPU discipline
    check(len(CC4_GPU_ALLOWED_UUIDS) == 2, "exactly two allowed GPUs")
    for pre in CC4_GPU_BANNED_UUID_PREFIXES:
        check(not any(u.startswith(pre) for u in CC4_GPU_ALLOWED_UUIDS),
              "banned GPU prefix %s must not appear in allowlist" % pre)

    # canonical_json_bytes determinism
    check(sha256_bytes(canonical_json_bytes({"b": 1, "a": [1, 2]}))
          == sha256_bytes(canonical_json_bytes({"a": [1, 2], "b": 1})),
          "canonical json must be key-order independent")

    print("PROJECTION_RUNTIME_SELF_TEST_PASS checks=%d families=%d students=%d "
          "teacher=1" % (n, len(families), len(students)))
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--self-test":
        sys.exit(self_test())
    if argv and argv[0] == "--list":
        for cid, s in sorted(PROJECTION_REGISTRY.items()):
            print("%s family=%s owner=%s class=%s"
                  % (cid, s["runtime_family"], s["owner"], s["candidate_class"]))
        sys.exit(0)
    print("usage: tier3_projection_runtime.py --self-test | --list")
    sys.exit(2)
