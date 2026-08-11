#!/usr/bin/env python3
"""Phase4A-v2.3 (CC2 §五.1) — PURE-Python FROZEN experiment specification.

STDLIB ONLY. This module MUST NOT import jax / numpy / optax / orbax / craftax / flax (nor any
module that does), because the driver imports it BEFORE `import jax` so the FULL scientific
binding can complete pre-JAX (§五.2). It holds the canonical frozen experiment values — the
single source of truth mirrored by `phase4a_v2_runtime_config._reference_runtime_kwargs` and
matched by the formal YAML `scientific_config` blocks.

§五.1 constraint: the VALUES below are FROZEN. This round MUST NOT change any of them (no
hyperparameter / network / task / evaluator / seed / budget change). The driver's REAL imported
objects (Cfg / FullP2Config / K_BATCH / ANCHOR_INTERVAL / MIN_SEQUENCE_LENGTH / RL.W_ORIGINAL_VTRACE)
are verified against this spec AFTER `import jax` (§五.3); any drift =>
IMPORTED_RUNTIME_CONSTANTS_MISMATCH.
"""
import argparse
import hashlib
import json

# ---------------------------------------------------------------------------
# The frozen experiment spec (carry_mode-independent; carry_mode is supplied per arm).
# Mirrors f2b7aead driver Cfg + FullP2Config + K_BATCH + ANCHOR_INTERVAL(128) +
# MIN_SEQUENCE_LENGTH(129) + RL.W_ORIGINAL_VTRACE(1.0) + CLI defaults (seed=42,
# total_updates=12, save_every=2, sequence_length=129).
# ---------------------------------------------------------------------------
FROZEN_SPEC = dict(
    replay_mode="original_vtrace",
    allow_full_p2_legacy=False,
    sequence_length=129,
    segment_len=128,
    hindsight=False,
    awr=False,
    w_original_vtrace=1.0,
    base_checkpoint="ckpt17500",
    seed=42,
    total_updates=12,
    save_every=2,
    num_envs=16,
    num_steps=128,
    task="DEFEAT_KOBOLD",
    optimistic_reset_ratio=16,
    condition_on_task=True,
    replay_batch_size=4,
    replay_buffer_capacity=64,
    anchor_interval=128,
    min_sequence_length=129,
    eligible_only_sampling=True,
    ppo_lr=2.0e-5,
    ppo_max_grad_norm=1.0,
    ppo_gamma=0.999,
    ppo_gae_lambda=0.8,
    ppo_clip_eps=0.2,
    ppo_vf_coef=0.5,
    ppo_ent_coef=0.002,
    ppo_update_epochs=1,
    ppo_num_minibatches=2,
    ppo_value_target_clip_min=-50.0,
    ppo_value_target_clip_max=300.0,
    vtrace_rho_bar=1.0,
    vtrace_c_bar=1.0,
    vtrace_vt_clip_min=-50.0,
    vtrace_vt_clip_max=300.0,
    kl_replay_max=0.05,
    kl_run_max=0.1,
    actor_step_scales=[1.0, 0.5, 0.25, 0.125],
    policy_lag_gate_active=False,
    policy_lag_gate_mode="not_applicable_original_vtrace",
    policy_lag_max_policy_lag=None,
    legacy_full_p2_active=False,
    legacy_full_p2_max_policy_lag=16,
    ema_tau=0.995,
    ent_floor=0.05,
    grad_clip=1.0,
    adam_eps=1.0e-5,
    net_activation="relu",
    net_embed_size=256,
    net_num_heads=8,
    net_qkv_features=256,
    net_num_layers=2,
    net_gating=True,
    net_gating_bias=2.0,
    net_window_mem=128,
    net_rmt_num_tokens=16,
    evaluator="frozen_rmt16_evaluator",
)

# VALID_CARRY_MODES: every carry_mode the frozen protocol may run. base_gtrxl is the third arm
# (CC2 §二 BASE_GTRXL_ORIGINAL_VTRACE_98304): the SAME network module + SAME ckpt17500, but the
# RMT16 persistent-token READ path is skipped (mem_tokens=None) so the policy reduces to the pure
# GTrXL backbone. APPENDING here is additive and does NOT touch FROZEN_SPEC / FROZEN_SPEC_SHA256
# (spec_sha256 hashes only the FROZEN_SPEC dict, not VALID_CARRY_MODES) -> P/R identity unchanged.
VALID_CARRY_MODES = ("persistent", "reset128", "base_gtrxl")

# The two canonical arms that have a FORMAL-VTRACE profile YAML (configs/rmt16_phase4a_v2_<arm>.yaml).
# base_gtrxl is an engineering smoke / long_run_98304 candidate ONLY — it has NO formal_vtrace
# profile — so the frozen-spec-vs-formal-YAML self-test cross-check below is scoped to these two.
FORMAL_VTRACE_ARMS = ("persistent", "reset128")


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_sha256(spec=None):
    """Canonical SHA256 over the frozen spec (carry_mode-independent values)."""
    return hashlib.sha256(_canonical_json(spec if spec is not None else FROZEN_SPEC)
                          .encode("utf-8")).hexdigest()


# Frozen at import time — the driver / gates can pin this to detect any edit to the spec block.
FROZEN_SPEC_SHA256 = spec_sha256()


def build_kwargs(carry_mode):
    """Return the kwargs for `phase4a_v2_runtime_config.build_runtime_scientific_config` for
    `carry_mode`, with every OTHER value taken verbatim from the frozen spec. This is what the
    driver uses to build the runtime scientific config PRE-JAX (§五.2 step 6)."""
    if carry_mode not in VALID_CARRY_MODES:
        raise ValueError(
            f"FROZEN_SPEC_INVALID_CARRY_MODE: {carry_mode!r} not in {VALID_CARRY_MODES}")
    kw = dict(FROZEN_SPEC)
    kw["carry_mode"] = carry_mode
    return kw


def build_frozen_scientific_config(carry_mode):
    """Build the frozen scientific_config dict for `carry_mode` (lazy-imports the builder so this
    module stays stdlib-only at import time)."""
    import phase4a_v2_runtime_config as RTC  # lazy: RTC imports yaml, not this at top level
    return RTC.build_runtime_scientific_config(**build_kwargs(carry_mode))


def self_test():
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)

    print("phase4a_v2_frozen_spec --self-test (Phase4A-v2.3 §五.1)", flush=True)

    # (1) spec SHA is deterministic + non-empty
    check("spec_sha256 deterministic", spec_sha256() == FROZEN_SPEC_SHA256
          and len(FROZEN_SPEC_SHA256) == 64)

    # (2) build_kwargs supplies carry_mode + all frozen values; differs only by carry_mode
    kw_p = build_kwargs("persistent")
    kw_r = build_kwargs("reset128")
    diff_keys = {k for k in set(kw_p) | set(kw_r) if kw_p.get(k) != kw_r.get(k)}
    check("build_kwargs: arms differ ONLY in carry_mode", diff_keys == {"carry_mode"},
          f"diff={sorted(diff_keys)}")

    # (3) invalid carry_mode -> fail closed
    try:
        build_kwargs("bogus")
        check("build_kwargs invalid carry_mode -> raised", False, "no raise")
    except ValueError as e:
        check("build_kwargs invalid carry_mode -> raised", "FROZEN_SPEC_INVALID_CARRY_MODE" in str(e))

    # (4) cross-check: the frozen spec builds a scientific_config that matches BOTH formal YAMLs'
    # scientific_config (value binding). Lazy-import RTC + yaml here (allowed in self-test only).
    import os
    import phase4a_v2_runtime_config as RTC
    try:
        import yaml
    except ImportError:
        yaml = None
    snap = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    yaml_ok = True
    if yaml is None:
        yaml_ok = False
    else:
        # Cross-check only the canonical formal-vtrace arms (base_gtrxl has no formal-vtrace YAML).
        for arm in FORMAL_VTRACE_ARMS:
            yp = os.path.join(snap, "configs", f"rmt16_phase4a_v2_{arm}.yaml")
            with open(yp, encoding="utf-8") as f:
                ycfg = yaml.safe_load(f)
            formal_sci = RTC.canonical_scientific_config(ycfg["scientific_config"])
            frozen_sci = RTC.canonical_scientific_config(build_frozen_scientific_config(arm))
            diffs = RTC.deep_diff(formal_sci, frozen_sci)
            if diffs:
                yaml_ok = False
                print(f"    [{arm}] frozen-vs-YAML diffs: {diffs[:3]}", flush=True)
    check("frozen spec scientific_config == BOTH formal YAML scientific_config", yaml_ok)

    # (5) editing a spec value changes the SHA (tamper-evidence of the spec block itself)
    mutated = dict(FROZEN_SPEC)
    mutated["seed"] = 43
    check("editing spec value changes spec_sha256", spec_sha256(mutated) != FROZEN_SPEC_SHA256)

    n = len(results); n_pass = sum(results)
    print(f"SELF_TEST_SUMMARY total={n} pass={n_pass} fail={n - n_pass}", flush=True)
    print(f"FROZEN_SPEC_SHA256={FROZEN_SPEC_SHA256}", flush=True)
    return 0 if n_pass == n else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true",
                    help="run frozen-spec self-tests (stdlib + lazy RTC/yaml for YAML cross-check)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ap.error("--self-test is the only supported mode here; the driver imports this module.")


if __name__ == "__main__":
    raise SystemExit(main())
