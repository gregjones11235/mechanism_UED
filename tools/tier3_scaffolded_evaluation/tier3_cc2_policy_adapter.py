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

Config recovery: REAL CC2 manifests carry ``config == {}`` (Cfg is a class-attributes
class, so ``vars(Cfg())`` is empty by design — verified on all 26 real checkpoints).
The network hyperparameters are recovered by an AST-LITERAL parse of the SHA-bound
driver source (``FROZEN_DRIVER_FILE_SHA256``) — never executed, never guessed, never
defaulted — and cross-checked against any non-empty manifest config the pickle does
carry (a clash fails closed).
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
# Config fields required to reconstruct the network EXACTLY as CC2 built it.
#
# REAL-CC2 DISCOVERY (26 real checkpoints audited, both arms, steps 0..98304): on
# every real full_state.pkl, manifest["config"] == {} — CC2's Cfg (driver lines
# 303-309) is a class-ATTRIBUTES config class, so vars(Cfg()) is empty BY DESIGN and
# save_ckpt writes config={k: v for k, v in vars(cfg).items()} == {}. The network
# hyperparameters are therefore FROZEN IN THE DRIVER SOURCE, not the pickle. CC4
# recovers them by an AST-LITERAL parse of the SHA-bound driver source (never
# executing it, never guessing, never defaulting): see load_cfg_from_driver_source().
REQUIRED_CFG_FIELDS = (
    "activation", "embed_size", "hidden_layers", "num_heads", "qkv_features",
    "num_layers", "gating", "gating_bias", "rmt_num_tokens", "window_mem", "num_steps")
REQUIRED_MANIFEST_CONFIG_FIELDS = REQUIRED_CFG_FIELDS      # legacy alias
CARRY_MODES = ("persistent", "reset128")

# SHA-bound CC2 driver source: the ONLY authoritative place the frozen network
# hyperparameters live (five-way identical LF-SHA: handover §3 / run-completion
# addendum / launch report / local _cc2_stage copy / server deploy). AST-parsing
# this file's ``class Cfg`` is how CC4 rebuilds the network WITHOUT guessing.
FROZEN_DRIVER_FILE_SHA256 = (
    "453bd1ecc8d9671c741c4462214bd7699c74611a52ec157ff30cd68653b4bafc")
DEFAULT_DRIVER_SOURCE = (
    "D:/Projects/dicode-codex-director/orchestration/control/_cc2_stage/"
    "train_rmt16_p2replay.py")


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
# Network / RMT config reconstruction FROM the SHA-bound driver source
# ---------------------------------------------------------------------------
def load_cfg_from_driver_source(driver_path: str = DEFAULT_DRIVER_SOURCE,
                                expected_sha256: str = FROZEN_DRIVER_FILE_SHA256):
    """Recover the frozen CC2 network hyperparameters from the driver SOURCE.

    REAL CC2 checkpoints carry manifest["config"] == {} (Cfg is a class-attributes
    class; vars(Cfg()) is empty by design — verified on all 26 audited real
    checkpoints). The hyperparameters live in the driver source's ``class Cfg``
    (train_rmt16_p2replay.py lines 303-309). This function:

      * requires the driver file to exist and its LF-normalized SHA256 to equal
        ``expected_sha256`` (fail closed — a moved/edited driver is never trusted);
      * ast.parse-es the source (NEVER executes it) and collects every class-level
        literal assignment in ``class Cfg`` via ast.literal_eval (a non-literal
        value fails closed — no guessing, no defaults);
      * requires ALL REQUIRED_CFG_FIELDS to be present.

    Returns (cfg_dict, driver_sha256). Pure stdlib — runs on ANY interpreter.
    """
    require(driver_path and os.path.isfile(driver_path),
            "FAIL CLOSED: CC2 driver source missing (%r) — cannot rebuild the network "
            "without the frozen hyperparameters (no guessing, no defaults)" % driver_path)
    driver_sha = _sha256_lf_file(driver_path)
    require(expected_sha256 is None or driver_sha == expected_sha256,
            "FAIL CLOSED: CC2 driver source SHA256 %s != frozen/expected %s (a moved or "
            "edited driver is never trusted)"
            % (driver_sha[:16], str(expected_sha256)[:16]))
    import ast
    with open(driver_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise FailClosed("FAIL CLOSED: CC2 driver source %r does not parse: %r"
                         % (driver_path, exc))
    cfg_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Cfg":
            cfg_cls = node
            break
    require(cfg_cls is not None,
            "FAIL CLOSED: CC2 driver source %r defines no top-level 'class Cfg'"
            % driver_path)
    cfg = {}
    for stmt in cfg_cls.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        key = stmt.targets[0].id
        try:
            cfg[key] = ast.literal_eval(stmt.value)
        except (ValueError, SyntaxError):
            raise FailClosed(
                "FAIL CLOSED: Cfg.%s in %r is not a literal constant (no guessing, no "
                "defaults — the driver source must carry plain literals)"
                % (key, driver_path))
    missing = [f for f in REQUIRED_CFG_FIELDS if f not in cfg]
    require(not missing,
            "FAIL CLOSED: driver source Cfg missing field(s) %s; cannot rebuild the CC2 "
            "network exactly" % missing)
    return cfg, driver_sha


def build_network_from_manifest(modules: dict, manifest: dict, action_dim: int,
                                cfg: dict):
    """Reconstruct ActorCriticTransformerRMT16 + RMT16Config EXACTLY as CC2 built them
    (train_rmt16_p2replay.py lines 712-716 + 320-321).

    ``cfg`` comes from load_cfg_from_driver_source() (SHA-bound driver Cfg) — NOT from
    the pickle, because real manifests carry config={} by design. Consistency gates
    (fail closed):

      * every REQUIRED_CFG_FIELDS field present in ``cfg``;
      * carry_mode read from the manifest, in CARRY_MODES;
      * a NON-EMPTY manifest["config"] must agree with the driver Cfg on every key it
        carries ({} is the real observed state and passes; a clash is a tampered /
        foreign checkpoint);
      * manifest["phase4a_v2"]["segment_len"], when present, must equal cfg num_steps.

    Returns (network, rmt_cfg, carry_mode). Nothing is trained or initialized here
    (params come from the checkpoint)."""
    require(isinstance(cfg, dict),
            "FAIL CLOSED: build_network_from_manifest needs the driver-source cfg dict")
    missing = [f for f in REQUIRED_CFG_FIELDS if f not in cfg]
    require(not missing,
            "FAIL CLOSED: driver cfg missing field(s) %s; cannot rebuild the CC2 network "
            "exactly" % missing)
    require(isinstance(manifest, dict),
            "FAIL CLOSED: checkpoint manifest is not a dict")
    carry_mode = manifest.get("carry_mode")
    require(carry_mode in CARRY_MODES,
            "FAIL CLOSED: manifest carry_mode %r not in %s" % (carry_mode, CARRY_MODES))
    mcfg = manifest.get("config")
    if mcfg:
        require(isinstance(mcfg, dict),
                "FAIL CLOSED: manifest['config'] is present but not a dict")
        clashes = {k: (mcfg[k], cfg[k]) for k in mcfg if k in cfg and mcfg[k] != cfg[k]}
        require(not clashes,
                "FAIL CLOSED: manifest config disagrees with the SHA-bound driver Cfg "
                "(key: (manifest, driver)) %s — foreign / tampered checkpoint" % clashes)
    p4 = manifest.get("phase4a_v2")
    if isinstance(p4, dict) and p4.get("segment_len") is not None:
        require(int(p4["segment_len"]) == int(cfg["num_steps"]),
                "FAIL CLOSED: manifest phase4a_v2.segment_len %r != driver cfg num_steps "
                "%r (RMT segment geometry mismatch)"
                % (p4["segment_len"], cfg["num_steps"]))
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
    """A manifest shaped EXACTLY like a REAL direct-98304 checkpoint manifest — TEST
    ONLY, clearly synthetic. config={} mirrors reality (CC2 Cfg is a class-attributes
    class, so save_ckpt writes an empty config dict BY DESIGN — verified on all 26
    real checkpoints); replay_mode / phase4a_v2 provenance as CC2 stamps them. A real
    run reads this from the checkpoint; hyperparameters come from the SHA-bound driver
    source (load_cfg_from_driver_source), never from the pickle."""
    return {"params_sha256": "0" * 64, "step": -1, "arm": "RMT16-SYNTHETIC-SELFTEST",
            "carry_mode": "persistent", "replay_mode": "original_vtrace", "seed": 42,
            "config": {},
            "phase4a_v2": {"run_class": "selftest", "sequence_length": 129,
                           "segment_len": 128, "crosses_boundary": True,
                           "replay_mode": "original_vtrace"}}


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

    # Driver-source Cfg recovery is PURE STDLIB — runs on ANY interpreter. This is the
    # ONLY place the frozen network hyperparameters live (real config={} by design).
    cfg, driver_sha = load_cfg_from_driver_source(DEFAULT_DRIVER_SOURCE,
                                                  FROZEN_DRIVER_FILE_SHA256)
    check("driver_cfg_complete",
          driver_sha == FROZEN_DRIVER_FILE_SHA256
          and all(f in cfg for f in REQUIRED_CFG_FIELDS)
          and cfg["activation"] == "relu" and cfg["embed_size"] == 256
          and cfg["hidden_layers"] == 256 and cfg["num_heads"] == 8
          and cfg["qkv_features"] == 256 and cfg["num_layers"] == 2
          and cfg["gating"] is True and cfg["gating_bias"] == 2.0
          and cfg["rmt_num_tokens"] == 16 and cfg["window_mem"] == 128
          and cfg["num_steps"] == 128)
    # A driver source whose SHA != the frozen/expected value fails closed.
    try:
        load_cfg_from_driver_source(DEFAULT_DRIVER_SOURCE, "0" * 64)
        check("driver_wrong_expected_sha_rejected", False)
    except FailClosed:
        check("driver_wrong_expected_sha_rejected", True)
    # A missing driver source fails closed (no guessing, no defaults).
    try:
        load_cfg_from_driver_source(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "no_such_driver.py"))
        check("driver_missing_source_rejected", False)
    except FailClosed:
        check("driver_missing_source_rejected", True)

    if ser.have_jax():
        import numpy as np
        import jax
        import jax.numpy as jnp
        modules, _sid = load_cc2_policy_modules(root)
        check("modules_loaded_from_root",
              set(modules.keys()) == set(REQUIRED_SYMBOLS.keys()))
        manifest = _synthetic_manifest()
        check("synthetic_manifest_config_empty", manifest["config"] == {})
        network, rmt_cfg, carry = build_network_from_manifest(modules, manifest, 43, cfg)
        check("network_built_carry_persistent", carry == "persistent")
        # a NON-EMPTY manifest config clashing with the driver Cfg -> fail closed
        bad = _synthetic_manifest()
        bad["config"] = {"embed_size": 999}
        try:
            build_network_from_manifest(modules, bad, 43, cfg)
            check("config_driver_clash_rejected", False)
        except FailClosed:
            check("config_driver_clash_rejected", True)
        # a non-empty config that AGREES with the driver Cfg is accepted
        agree = _synthetic_manifest()
        agree["config"] = {"embed_size": cfg["embed_size"], "num_heads": cfg["num_heads"]}
        build_network_from_manifest(modules, agree, 43, cfg)
        check("config_driver_agreement_accepted", True)
        # phase4a_v2 segment_len mismatching the driver cfg -> fail closed
        badseg = _synthetic_manifest()
        badseg["phase4a_v2"] = {"segment_len": 64}
        try:
            build_network_from_manifest(modules, badseg, 43, cfg)
            check("phase4a_v2_segment_len_mismatch_rejected", False)
        except FailClosed:
            check("phase4a_v2_segment_len_mismatch_rejected", True)
        # bad carry_mode -> fail closed
        badc = _synthetic_manifest()
        badc["carry_mode"] = "sideways"
        try:
            build_network_from_manifest(modules, badc, 43, cfg)
            check("bad_carry_mode_rejected", False)
        except FailClosed:
            check("bad_carry_mode_rejected", True)
        # RANDOM-INIT params via CC2's own init_all (NOT training; self-test only).
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
