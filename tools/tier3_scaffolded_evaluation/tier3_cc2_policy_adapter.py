#!/usr/bin/env python3
"""CC4 Tier3 — REAL CC2 RMT16 policy adapter (READ-ONLY; NO reimplementation).

Imports CC2's ACTUAL modules from ``--cc2_snapshot_root`` and drives them verbatim:

    network_rmt16.ActorCriticTransformerRMT16
    rmt16_memory.RMT16Config / rmt16_init
    rmt_memory_anchor.make_apply_eval_rmt / make_update_fn / rmt_step_forward

CC4 NEVER reimplements RMT/GTrXL state transition: every env step calls CC2's own
``rmt_step_forward`` (the single shared per-step transition used by CC2 collection
and reconstruction). Action selection is ``greedy_argmax`` (frozen Tier3 contract;
seed-free). params are READ-ONLY (NEG23 — SHA identical before/after). No optimizer,
no replay learner, no sampling.

Carry semantics (CC2 rmt_memory_anchor.rmt_advance_tokens, the ONLY Persistent vs
Reset128 difference): at each 128-step segment boundary,
  * carry_mode="persistent" : mem_tokens <- residual cross-attention update, carried;
  * carry_mode="reset128"   : mem_tokens <- 0 (cleared at the boundary).
carry_mode is read from the checkpoint manifest — never chosen here.

Source identity: every required CC2 .py is hashed LF-normalized; the aggregate
``cc2_policy_source_sha256`` is bound into the evaluation certificate. A module that
resolves from anywhere but the declared snapshot root -> fail closed.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402
import tier3_state_serializer as ser      # noqa: E402

SCHEMA = "mechanism_UED.tier3_cc2_policy_adapter/v1"
ADAPTER_VERSION = "tier3_cc2_policy_adapter/v1"

# Files that MUST exist in the snapshot root (the rmt16_replay_phase4a/src layout).
REQUIRED_FILES = ("network_rmt16.py", "rmt16_memory.py", "rmt_memory_anchor.py",
                  "memory_anchor.py")
# Symbols imported FROM CC2's modules (never re-implemented here).
REQUIRED_SYMBOLS = {
    "network_rmt16": ("ActorCriticTransformerRMT16",),
    "rmt16_memory": ("RMT16Config", "rmt16_init"),
    "rmt_memory_anchor": ("make_apply_eval_rmt", "make_update_fn", "rmt_step_forward"),
}
# manifest["config"] fields required to reconstruct the network EXACTLY as CC2 built it.
REQUIRED_MANIFEST_CONFIG_FIELDS = (
    "activation", "embed_size", "hidden_layers", "num_heads", "qkv_features",
    "num_layers", "gating", "gating_bias", "rmt_num_tokens", "window_mem", "num_steps")
CARRY_MODES = ("persistent", "reset128")


class FailClosed(Exception):
    """Hard stop on any adapter / source-identity / config violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# CC2 source identity (LF-normalized, EOL-independent)
# ---------------------------------------------------------------------------
def _sha256_lf_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def cc2_policy_source_identity(snapshot_root: str) -> dict:
    """Hash every required CC2 source file; fail closed on ANY missing file.
    Aggregate = sha256 over sorted 'filename:sha' lines (order-independent)."""
    require(snapshot_root and os.path.isdir(snapshot_root),
            "FAIL CLOSED: --cc2_snapshot_root %r is not a directory" % snapshot_root)
    files = {}
    for name in REQUIRED_FILES:
        p = os.path.join(snapshot_root, name)
        require(os.path.isfile(p),
                "FAIL CLOSED: CC2 snapshot root %r missing required source file %s"
                % (snapshot_root, name))
        files[name] = _sha256_lf_file(p)
    aggregate = hashlib.sha256(
        "\n".join("%s:%s" % (k, files[k]) for k in sorted(files)).encode("utf-8")
    ).hexdigest()
    return {"schema": SCHEMA, "adapter_version": ADAPTER_VERSION,
            "snapshot_root": os.path.abspath(snapshot_root), "files": files,
            "cc2_policy_source_sha256": aggregate}


def load_cc2_policy_modules(snapshot_root: str):
    """Import CC2's REAL modules from the snapshot root and verify every required
    symbol resolves FROM that root (a module cached from another path -> fail closed).

    Returns (modules, source_identity). Requires JAX (CC2 modules import it)."""
    require(ser.have_jax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): loading CC2 policy modules requires JAX "
            "(available=%s)" % ser.have_jax())
    src_id = cc2_policy_source_identity(snapshot_root)
    root = os.path.abspath(snapshot_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    modules = {}
    for mod_name, symbols in REQUIRED_SYMBOLS.items():
        try:
            m = importlib.import_module(mod_name)
        except Exception as exc:
            raise FailClosed(
                "FAIL CLOSED: cannot import CC2 module %r from %r: %r"
                % (mod_name, root, exc))
        resolved = os.path.realpath(getattr(m, "__file__", "") or "")
        require(resolved == os.path.realpath(os.path.join(root, mod_name + ".py")),
                "FAIL CLOSED: CC2 module %r resolved from %r, not the declared snapshot "
                "root %r (stale import / wrong root)" % (mod_name, resolved, root))
        missing = [s for s in symbols if not hasattr(m, s)]
        require(not missing,
                "FAIL CLOSED: CC2 module %r missing required symbol(s) %s" % (mod_name, missing))
        modules[mod_name] = m
    return modules, src_id


# ---------------------------------------------------------------------------
# Network / RMT config reconstruction FROM the checkpoint manifest
# ---------------------------------------------------------------------------
def build_network_from_manifest(modules: dict, manifest: dict, action_dim: int):
    """Reconstruct ActorCriticTransformerRMT16 + RMT16Config EXACTLY as CC2 built them
    (train_rmt16_p2replay.py lines 155-159 + 85-86), from manifest['config'].
    Returns (network, rmt_cfg, carry_mode). Any missing field / bad carry_mode -> fail
    closed. Nothing is trained or initialized here (params come from the checkpoint)."""
    require(isinstance(manifest, dict) and isinstance(manifest.get("config"), dict),
            "FAIL CLOSED: checkpoint manifest missing 'config' dict (needed to rebuild "
            "the CC2 network exactly)")
    cfg = manifest["config"]
    missing = [f for f in REQUIRED_MANIFEST_CONFIG_FIELDS if f not in cfg]
    require(not missing,
            "FAIL CLOSED: checkpoint manifest config missing field(s) %s; cannot rebuild "
            "the CC2 network exactly" % missing)
    carry_mode = manifest.get("carry_mode")
    require(carry_mode in CARRY_MODES,
            "FAIL CLOSED: manifest carry_mode %r not in %s" % (carry_mode, CARRY_MODES))
    network = modules["network_rmt16"].ActorCriticTransformerRMT16(
        action_dim=int(action_dim), activation=cfg["activation"],
        encoder_size=cfg["embed_size"], hidden_layers=cfg["hidden_layers"],
        num_heads=cfg["num_heads"], qkv_features=cfg["qkv_features"],
        num_layers=cfg["num_layers"], gating=cfg["gating"],
        gating_bias=cfg["gating_bias"], rmt_num_tokens=cfg["rmt_num_tokens"])
    rmt_cfg = modules["rmt16_memory"].RMT16Config(
        num_tokens=cfg["rmt_num_tokens"], segment_len=cfg["num_steps"],
        encoder_size=cfg["embed_size"])
    return network, rmt_cfg, carry_mode


# ---------------------------------------------------------------------------
# Stateful greedy policy driving CC2's rmt_step_forward
# ---------------------------------------------------------------------------
class CC2RMT16Policy:
    """Greedy_argmax policy over CC2's REAL GTrXL+RMT16 dynamics (seed-free).

    reset() initializes the episode state EXACTLY as CC2's driver (train lines
    201-204): memories zeros (1, window_mem, num_layers, embed), mem_mask zeros
    (1, num_heads, 1, window_mem+1) bool, mem_idx=window_mem (P2 convention),
    rmt16_init(1, rmt_cfg). __call__(obs, env_state) runs ONE CC2 rmt_step_forward
    (batch 1; make_apply_eval_rmt pads 1->2 internally per CC2) and returns
    int(argmax(logits)).

    The `done` vector fed to rmt_step_forward is always False: a Tier3 episode
    STOPS at env done (no step follows a terminal), so CC2's terminal-reset branch
    is never entered mid-episode. params are captured by reference and NEVER
    mutated (READ-ONLY; NEG23 checks the SHA before/after).
    """

    def __init__(self, modules: dict, network, params, rmt_cfg, carry_mode: str,
                 window_mem: int, num_heads: int, num_layers: int, embed_size: int):
        require(carry_mode in CARRY_MODES,
                "FAIL CLOSED: carry_mode %r not in %s" % (carry_mode, CARRY_MODES))
        self._anchor = modules["rmt_memory_anchor"]
        self._rmtm = modules["rmt16_memory"]
        self.params = params
        self.carry_mode = carry_mode
        self.window_mem = int(window_mem)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.embed_size = int(embed_size)
        self.rmt_cfg = rmt_cfg
        self.apply_eval_rmt = self._anchor.make_apply_eval_rmt(network)
        self.update_fn = self._anchor.make_update_fn(network, params)
        self.reset()

    def reset(self):
        """Fresh REAL GTrXL + RMT16 episode state (CC2 driver convention, N=1)."""
        import jax.numpy as jnp
        self.memories = jnp.zeros((1, self.window_mem, self.num_layers, self.embed_size))
        self.mem_mask = jnp.zeros((1, self.num_heads, 1, self.window_mem + 1), jnp.bool_)
        self.mem_idx = jnp.full((1,), self.window_mem, jnp.int32)   # P2 convention
        self.rmt_st = self._rmtm.rmt16_init(1, self.rmt_cfg)
        self._done_in = jnp.zeros((1,), jnp.bool_)

    def __call__(self, obs, env_state):
        import numpy as np
        import jax.numpy as jnp
        obs_batch = jnp.asarray(obs)[None, :]                       # (1, obs_dim)
        (self.memories, self.mem_mask, self.mem_idx, self.rmt_st,
         logits, _value, _mem_pre, _entering) = self._anchor.rmt_step_forward(
            self.apply_eval_rmt, self.params, self.memories, self.mem_mask,
            self.mem_idx, self.rmt_st, obs_batch, self._done_in,
            self.window_mem, self.num_heads, self.rmt_cfg, self.carry_mode,
            self.update_fn)
        return int(np.argmax(np.asarray(logits)[0]))                # greedy_argmax


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _default_snapshot_root() -> str:
    """The audited CC2 rmt16_replay_phase4a source snapshot inside this repo."""
    return str(audit.repo_root() / "experiments" / "henry_dicode_student_upgrade"
               / "13_rmt16_phase4a" / "raw_sources" / "home" / "oseasy" / "experiments"
               / "rmt16_replay_phase4a" / "src")


def _synthetic_manifest():
    """CC2's frozen bakeoff+P2 config values (train Cfg, lines 68-75) as a manifest —
    TEST-ONLY, clearly synthetic; a real run reads this from the checkpoint."""
    return {"params_sha256": "0" * 64, "step": -1, "arm": "RMT16-SYNTHETIC-SELFTEST",
            "carry_mode": "persistent",
            "config": {"activation": "relu", "embed_size": 256, "hidden_layers": 256,
                       "num_heads": 8, "qkv_features": 256, "num_layers": 2,
                       "gating": True, "gating_bias": 2.0, "window_mem": 128,
                       "num_steps": 128, "rmt_num_tokens": 16}}


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    root = _default_snapshot_root()
    # Pure source-identity hashing works on ANY host (no JAX needed).
    src_id = cc2_policy_source_identity(root)
    check("source_identity_complete",
          set(src_id["files"].keys()) == set(REQUIRED_FILES)
          and len(src_id["cc2_policy_source_sha256"]) == 64
          and all(len(v) == 64 for v in src_id["files"].values()))
    check("source_identity_deterministic",
          cc2_policy_source_identity(root)["cc2_policy_source_sha256"]
          == src_id["cc2_policy_source_sha256"])
    # A missing file fails closed.
    try:
        cc2_policy_source_identity(os.path.dirname(os.path.abspath(__file__)))
        check("missing_cc2_files_rejected", False)
    except FailClosed:
        check("missing_cc2_files_rejected", True)

    if ser.have_jax():
        import numpy as np
        import jax
        import jax.numpy as jnp
        modules, _sid = load_cc2_policy_modules(root)
        check("modules_loaded_from_root",
              set(modules.keys()) == set(REQUIRED_SYMBOLS.keys()))
        manifest = _synthetic_manifest()
        network, rmt_cfg, carry = build_network_from_manifest(modules, manifest, 43)
        check("network_built_carry_persistent", carry == "persistent")
        # manifest config missing a field -> fail closed
        bad = _synthetic_manifest()
        del bad["config"]["window_mem"]
        try:
            build_network_from_manifest(modules, bad, 43)
            check("missing_config_field_rejected", False)
        except FailClosed:
            check("missing_config_field_rejected", True)
        # bad carry_mode -> fail closed
        badc = _synthetic_manifest()
        badc["carry_mode"] = "sideways"
        try:
            build_network_from_manifest(modules, badc, 43)
            check("bad_carry_mode_rejected", False)
        except FailClosed:
            check("bad_carry_mode_rejected", True)
        # RANDOM-INIT params via CC2's own init_all (NOT training; self-test only).
        cfg = manifest["config"]
        rng = jax.random.PRNGKey(0)
        full = network.init(
            rng,
            jnp.zeros((2, cfg["window_mem"], cfg["num_layers"], cfg["embed_size"])),
            jnp.zeros((2, 8335)),
            jnp.zeros((2, cfg["num_heads"], 1, cfg["window_mem"] + 1), jnp.bool_),
            mem_tokens=jnp.zeros((2, cfg["rmt_num_tokens"], cfg["embed_size"])),
            seg_buf=jnp.zeros((2, cfg["num_steps"], cfg["embed_size"])),
            method=network.init_all)
        params = jax.tree_util.tree_map(jnp.asarray, full["params"])
        policy = CC2RMT16Policy(modules, network, params, rmt_cfg, carry,
                                cfg["window_mem"], cfg["num_heads"],
                                cfg["num_layers"], cfg["embed_size"])
        # Deterministic NON-zero obs: an all-zero obs drives the bias-free GTrXL
        # stack to an all-zero h_t (seg_buf stays 0), which would mask the segment
        # boundary update. Real Craftax observations are never all-zero.
        obs = np.random.default_rng(20260729).normal(size=8335).astype(np.float32)
        a0 = policy(obs, None)
        check("greedy_action_in_range", 0 <= a0 < 43)
        # Deterministic + stateful: replay after reset reproduces the sequence.
        seq1 = [a0] + [policy(obs, None) for _ in range(3)]
        policy.reset()
        seq2 = [policy(obs, None) for _ in range(4)]
        check("deterministic_after_reset", seq1 == seq2)
        # ---- carry_mode semantics through CC2's REAL per-step transition ----
        # update_rmt_tokens carries NO gate (new = tokens + LN(attn(Q=tokens,
        # kv=seg_buf))); the zero-init rmt_gate only gates the per-step READ path.
        # So even untrained params diverge at the boundary given a non-zero obs.
        def _run128(mode, steps=128):
            pol = CC2RMT16Policy(modules, network, params, rmt_cfg, mode,
                                 cfg["window_mem"], cfg["num_heads"],
                                 cfg["num_layers"], cfg["embed_size"])
            for _ in range(steps):
                pol(obs, None)
            return np.asarray(pol.rmt_st["mem_tokens"])

        # Before the 128-step segment boundary NEITHER mode touches mem_tokens.
        check("no_token_update_before_segment_boundary",
              np.all(_run128("persistent", 64) == 0.0)
              and np.all(_run128("reset128", 64) == 0.0))

        # AT the boundary: persistent carries the cross-attention update; reset128
        # clears mem_tokens to 0 — the ONLY Persistent vs Reset128 difference in
        # CC2's shared rmt_advance_tokens.
        tok_p = _run128("persistent", 128)
        tok_r = _run128("reset128", 128)
        check("carry_mode_semantics_diverge_at_boundary",
              not np.array_equal(tok_p, tok_r) and np.any(tok_p != 0.0)
              and np.all(tok_r == 0.0))
    else:
        # Without JAX, module loading must FAIL CLOSED (never a fake PASS).
        try:
            load_cc2_policy_modules(root)
            check("module_load_blocked_without_jax", False)
        except FailClosed:
            check("module_load_blocked_without_jax", True)

    if problems:
        print("TIER3_CC2_POLICY_ADAPTER_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_CC2_POLICY_ADAPTER_SELF_TEST_PASS (CC2 modules bound; greedy_argmax; "
          "carry_mode persistent/reset128 verified; env=%s)" % ser.environment_status())
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # CC2 network_rmt16 imports dicode.transformer.transformerXL (audited source tree).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_cc2_policy_adapter.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
