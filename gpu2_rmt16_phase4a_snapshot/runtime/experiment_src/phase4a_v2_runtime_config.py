#!/usr/bin/env python3
"""Phase4A-v2.2 (CC2 §六) — PRE-REGISTERED YAML <-> REAL RUNTIME binding, fail closed.

Binds the pre-registered formal config (configs/rmt16_phase4a_v2_{persistent,reset128}.yaml)
to the ACTUAL values the driver will run with, and refuses to proceed on any mismatch:

  (§六.2) replay_mode=original_vtrace REQUIRES --formal_config; missing => exit with
          FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE BEFORE JAX import / env build / ckpt load.
          replay_mode=off / probe may remain exempt (legacy dev compat). No bypass parameter.
  (§六.3) arm binding: schema=rmt16_phase4a_v2, arm==carry_mode, scientific_config.carry_mode
          ==carry_mode (and replay_mode, when declared) => else FORMAL_CONFIG_ARM_MISMATCH.
  (§六.4) build_runtime_scientific_config(...) rebuilds the COMPLETE scientific_config from the
          driver's REAL execution values (Cfg / fp_cfg / K_BATCH / ANCHOR_INTERVAL / CLI args),
          and the formal YAML scientific_config must equal it under canonical comparison.
  (§六.5) base checkpoint identity incl. a FROZEN EXPECTED params SHA256 taken from reviewed
          evidence (both arms' frozen raw-probe summaries record the same base_sha256);
          mismatch => BASE_CHECKPOINT_SHA_MISMATCH. No frozen expectation => NOT_FROZEN (never
          fabricated).
  (§六.6) runtime_assignment (gpu_uuid / out_dir) is validated SEPARATELY from the scientific
          SHA => RUNTIME_ASSIGNMENT_MISMATCH.
  (§六.7) write_runtime_config_certificate(...) emits runtime_config_certificate.json; only
          scientific_config_match=true AND runtime_assignment_match=true AND validation_errors=[]
          AND certificate_status=PASS permits env/training to proceed.

PURE Python: stdlib + PyYAML ONLY. NO JAX / numpy / optax — this module is imported by the
driver BEFORE `import jax` and must run on a machine without JAX (local gate box).
"""
import argparse
import hashlib
import json
import os

import yaml

SCHEMA = "rmt16_phase4a_v2"

# ---------------------------------------------------------------------------
# §六.5 — FROZEN expected base checkpoint identity (NOT fabricated).
# Source: gpu2_rmt16_phase4a_snapshot/evidence/raw_probe/persistent_probe_summary.json and
# reset128_probe_summary.json — both frozen reviewed summaries record
# base_sha256 = d4e85af5...60f5 (IDENTICAL on both arms) with step0_params_in =
# ckpt/0/full_state.pkl; the driver computes base_sha over the INNER params of the loaded
# ckpt17500 the same way, so the comparison is apples-to-apples.
EXPECTED_BASE_CHECKPOINT_LABEL = "ckpt17500"
EXPECTED_BASE_CHECKPOINT_SHA256 = (
    "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5")
EXPECTED_BASE_CHECKPOINT_EVIDENCE = (
    "gpu2_rmt16_phase4a_snapshot/evidence/raw_probe/persistent_probe_summary.json and "
    "reset128_probe_summary.json -> base_sha256 (identical both arms), "
    "step0_params_in=ckpt/0/full_state.pkl")


# ---------------------------------------------------------------------------
# loading + canonicalization
# ---------------------------------------------------------------------------

def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_formal_config(path):
    """Load a pre-registered YAML with provenance. Returns
    {path, realpath, file_sha256, config}. Raises ValueError on unreadable/invalid input."""
    if not path:
        raise ValueError(
            "FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE: no formal config path supplied.")
    if not os.path.exists(path):
        raise ValueError(f"FORMAL_CONFIG_LOAD_ERROR: formal config not found: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"FORMAL_CONFIG_INVALID: {path} is not a YAML mapping.")
    return dict(path=str(path), realpath=os.path.realpath(path),
                file_sha256=_sha256_bytes(raw), config=config)


def canonical_scientific_config(scientific_config):
    """Canonical form of a scientific_config block: JSON round-trip with sorted keys (the ONLY
    explicit normalization — scalar values keep EXACT equality, no float tolerance)."""
    if not isinstance(scientific_config, dict):
        raise ValueError(
            f"FORMAL_CONFIG_INVALID: scientific_config must be a mapping, got "
            f"{type(scientific_config).__name__}")
    return json.loads(json.dumps(scientific_config, sort_keys=True, ensure_ascii=False))


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def scientific_config_sha256(scientific_config):
    """SHA256 over the canonical JSON of the scientific_config block."""
    return _sha256_bytes(canonical_json(canonical_scientific_config(scientific_config))
                         .encode("utf-8"))


# ---------------------------------------------------------------------------
# §六.4 — build the REAL runtime scientific config from execution values
# ---------------------------------------------------------------------------

def build_runtime_scientific_config(
        *, carry_mode, replay_mode, allow_full_p2_legacy,
        sequence_length, segment_len, hindsight, awr, w_original_vtrace,
        base_checkpoint, seed, total_updates, save_every,
        num_envs, num_steps, task, optimistic_reset_ratio, condition_on_task,
        replay_batch_size, replay_buffer_capacity, anchor_interval, min_sequence_length,
        eligible_only_sampling,
        ppo_lr, ppo_max_grad_norm, ppo_gamma, ppo_gae_lambda, ppo_clip_eps, ppo_vf_coef,
        ppo_ent_coef, ppo_update_epochs, ppo_num_minibatches,
        ppo_value_target_clip_min, ppo_value_target_clip_max,
        vtrace_rho_bar, vtrace_c_bar, vtrace_vt_clip_min, vtrace_vt_clip_max,
        kl_replay_max, kl_run_max, actor_step_scales,
        policy_lag_gate_active, policy_lag_gate_mode, policy_lag_max_policy_lag,
        legacy_full_p2_active, legacy_full_p2_max_policy_lag,
        ema_tau, ent_floor, grad_clip, adam_eps,
        net_activation, net_embed_size, net_num_heads, net_qkv_features, net_num_layers,
        net_gating, net_gating_bias, net_window_mem, net_rmt_num_tokens,
        evaluator="frozen_rmt16_evaluator"):
    """Rebuild scientific_config EXACTLY as the pre-registered YAML schema, from the driver's
    ACTUAL execution values (not a subset of CLI args). Every value comes from a live runtime
    object (Cfg / fp_cfg / K_BATCH / ANCHOR_INTERVAL / args) in the driver call site."""
    return dict(
        carry_mode=carry_mode,
        replay_mode=replay_mode,
        allow_full_p2_legacy=bool(allow_full_p2_legacy),
        sequence_length=int(sequence_length),
        segment_len=int(segment_len),
        crosses_boundary=bool(int(sequence_length) > int(segment_len)),
        hindsight=bool(hindsight),
        awr=bool(awr),
        w_original_vtrace=float(w_original_vtrace),
        base_checkpoint=base_checkpoint,
        seed=int(seed),
        total_updates=int(total_updates),
        save_every=int(save_every),
        num_envs=int(num_envs),
        num_steps=int(num_steps),
        task=task,
        optimistic_reset_ratio=int(optimistic_reset_ratio),
        condition_on_task=bool(condition_on_task),
        replay_batch_size=int(replay_batch_size),
        replay_buffer_capacity=int(replay_buffer_capacity),
        anchor_interval=int(anchor_interval),
        min_sequence_length=int(min_sequence_length),
        eligible_only_sampling=bool(eligible_only_sampling),
        ppo=dict(
            lr=float(ppo_lr), max_grad_norm=float(ppo_max_grad_norm),
            gamma=float(ppo_gamma), gae_lambda=float(ppo_gae_lambda),
            clip_eps=float(ppo_clip_eps), vf_coef=float(ppo_vf_coef),
            ent_coef=float(ppo_ent_coef), update_epochs=int(ppo_update_epochs),
            num_minibatches=int(ppo_num_minibatches),
            value_target_clip_min=float(ppo_value_target_clip_min),
            value_target_clip_max=float(ppo_value_target_clip_max)),
        vtrace=dict(
            rho_bar=float(vtrace_rho_bar), c_bar=float(vtrace_c_bar),
            vt_clip_min=float(vtrace_vt_clip_min), vt_clip_max=float(vtrace_vt_clip_max)),
        kl_replay_max=float(kl_replay_max),
        kl_run_max=float(kl_run_max),
        actor_step_scales=[float(x) for x in actor_step_scales],
        policy_lag=dict(
            active=bool(policy_lag_gate_active),
            mode=policy_lag_gate_mode,
            max_policy_lag=policy_lag_max_policy_lag,
            correction=dict(
                method="vtrace_importance_sampling",
                rho_bar=float(vtrace_rho_bar), c_bar=float(vtrace_c_bar))),
        exposure_contract=dict(
            same_replay_protocol="READY",
            matched_replay_exposure="NOT_RUN",
            matched_replay_content="NOT_CLAIMED",
            endogenous_replay_screening="READY_AFTER_SMOKE",
            buffer_kind="endogenous"),
        legacy_full_p2_only=dict(
            active=bool(legacy_full_p2_active),
            max_policy_lag=int(legacy_full_p2_max_policy_lag)),
        ema_tau=float(ema_tau),
        ent_floor=float(ent_floor),
        grad_clip=float(grad_clip),
        adam_eps=float(adam_eps),
        network=dict(
            activation=net_activation, embed_size=int(net_embed_size),
            num_heads=int(net_num_heads), qkv_features=int(net_qkv_features),
            num_layers=int(net_num_layers), gating=bool(net_gating),
            gating_bias=float(net_gating_bias), window_mem=int(net_window_mem),
            rmt_num_tokens=int(net_rmt_num_tokens)),
        evaluator=evaluator)


# ---------------------------------------------------------------------------
# deep diff (key-order-insensitive, exact scalar equality)
# ---------------------------------------------------------------------------

def deep_diff(formal, runtime, path="scientific_config"):
    """Return a list of {path, formal, runtime, kind} mismatches. EXACT equality — there is no
    float tolerance and no ignored key (any extra/missing key on either side is reported)."""
    diffs = []
    if isinstance(formal, bool) or isinstance(runtime, bool):
        if formal is not runtime:
            diffs.append(dict(path=path, formal=formal, runtime=runtime, kind="value"))
        return diffs
    if isinstance(formal, (int, float)) and isinstance(runtime, (int, float)):
        if float(formal) != float(runtime):
            diffs.append(dict(path=path, formal=formal, runtime=runtime, kind="value"))
        return diffs
    if isinstance(formal, dict) and isinstance(runtime, dict):
        for k in sorted(set(formal) | set(runtime)):
            sub = f"{path}.{k}"
            if k not in formal:
                diffs.append(dict(path=sub, formal="<absent>", runtime=runtime[k],
                                  kind="extra_in_runtime"))
            elif k not in runtime:
                diffs.append(dict(path=sub, formal=formal[k], runtime="<absent>",
                                  kind="missing_in_runtime"))
            else:
                diffs.extend(deep_diff(formal[k], runtime[k], sub))
        return diffs
    if isinstance(formal, list) and isinstance(runtime, list):
        if len(formal) != len(runtime):
            diffs.append(dict(path=path, formal=formal, runtime=runtime, kind="list_length"))
            return diffs
        for i, (a, b) in enumerate(zip(formal, runtime)):
            diffs.extend(deep_diff(a, b, f"{path}[{i}]"))
        return diffs
    if type(formal) is not type(runtime):
        diffs.append(dict(path=path, formal=formal, runtime=runtime, kind="type"))
    elif formal != runtime:
        diffs.append(dict(path=path, formal=formal, runtime=runtime, kind="value"))
    return diffs


# ---------------------------------------------------------------------------
# §六.2 / §六.3 — pre-JAX preflight gates
# ---------------------------------------------------------------------------

def preflight_require_formal_config(replay_mode, formal_config_path):
    """§六.2 fail closed: replay_mode=original_vtrace REQUIRES a pre-registered --formal_config.
    Called by the driver BEFORE `import jax` / env build / ckpt load. replay_mode=off (and
    probe, which forces off) remains exempt for legacy dev compat. Raises ValueError."""
    if replay_mode == "original_vtrace" and not formal_config_path:
        raise ValueError(
            "FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE: replay_mode=original_vtrace requires "
            "--formal_config <pre-registered yaml> (configs/rmt16_phase4a_v2_<arm>.yaml). "
            "No bypass parameter exists; the driver exits BEFORE JAX import / env build / "
            "checkpoint load.")


def validate_arm_binding(formal_record, carry_mode, replay_mode=None):
    """§六.3 fail closed: the formal YAML must be THIS arm's pre-registration. Raises
    ValueError('FORMAL_CONFIG_ARM_MISMATCH...') on schema / arm / carry_mode / replay_mode
    mismatch. Returns a small PASS record."""
    if not isinstance(formal_record, dict) or not isinstance(formal_record.get("config"), dict):
        raise ValueError("FORMAL_CONFIG_ARM_MISMATCH: no formal config loaded.")
    cfg = formal_record["config"]
    errors = []
    if cfg.get("schema") != SCHEMA:
        errors.append(f"schema={cfg.get('schema')!r} != {SCHEMA!r}")
    if cfg.get("arm") != carry_mode:
        errors.append(f"arm={cfg.get('arm')!r} != carry_mode={carry_mode!r}")
    sci = cfg.get("scientific_config") or {}
    if not isinstance(sci, dict):
        raise ValueError("FORMAL_CONFIG_ARM_MISMATCH: formal YAML has no scientific_config.")
    if sci.get("carry_mode") != carry_mode:
        errors.append(
            f"scientific_config.carry_mode={sci.get('carry_mode')!r} != carry_mode={carry_mode!r}")
    if replay_mode is not None and sci.get("replay_mode") is not None \
            and sci.get("replay_mode") != replay_mode:
        errors.append(
            f"scientific_config.replay_mode={sci.get('replay_mode')!r} != "
            f"replay_mode={replay_mode!r}")
    if errors:
        raise ValueError("FORMAL_CONFIG_ARM_MISMATCH: " + "; ".join(errors))
    return dict(arm_binding="PASS", schema=SCHEMA, arm=carry_mode)


# ---------------------------------------------------------------------------
# §六.5 — base checkpoint identity
# ---------------------------------------------------------------------------

def build_checkpoint_identity(ckpt_path, *,
                              expected_label=EXPECTED_BASE_CHECKPOINT_LABEL,
                              expected_sha256=EXPECTED_BASE_CHECKPOINT_SHA256,
                              expected_source=EXPECTED_BASE_CHECKPOINT_EVIDENCE):
    """Record the CLI-provided base checkpoint identity. The params SHA is filled in AFTER the
    driver loads the checkpoint (verify_checkpoint_params_sha). expected_sha256=None means no
    frozen expectation exists -> labeled NOT_FROZEN (never fabricated)."""
    rec = dict(
        base_checkpoint_label=expected_label,
        base_checkpoint_path=(str(ckpt_path) if ckpt_path else None),
        base_checkpoint_realpath=(os.path.realpath(str(ckpt_path)) if ckpt_path else None),
        base_checkpoint_step=None,
        base_checkpoint_params_sha256=None,
        base_checkpoint_expected_sha256=expected_sha256,
        base_checkpoint_expected_sha256_status=(
            "FROZEN_FROM_REVIEWED_EVIDENCE" if expected_sha256 else "NOT_FROZEN"),
        base_checkpoint_expected_source=(expected_source if expected_sha256 else None),
        base_checkpoint_match=None)
    if ckpt_path:
        p = str(ckpt_path).replace("\\", "/").rstrip("/")
        rec["base_checkpoint_step"] = os.path.basename(p) or os.path.basename(os.path.dirname(p))
        if expected_label and ("17500" not in p):
            rec["base_checkpoint_label_note"] = (
                f"path {ckpt_path!r} does not textually reference {expected_label}")
    return rec


def verify_checkpoint_params_sha(checkpoint_identity, loaded_params_sha256):
    """Fail closed: compare the loaded base params SHA against the FROZEN expectation (if one
    exists). Returns the updated identity; raises BASE_CHECKPOINT_SHA_MISMATCH on inequality."""
    ci = dict(checkpoint_identity or {})
    ci["base_checkpoint_params_sha256"] = loaded_params_sha256
    expected = ci.get("base_checkpoint_expected_sha256")
    if not expected:
        ci["base_checkpoint_match"] = "NOT_FROZEN"
        return ci
    if loaded_params_sha256 == expected:
        ci["base_checkpoint_match"] = "PASS"
        return ci
    raise ValueError(
        f"BASE_CHECKPOINT_SHA_MISMATCH: loaded base params sha256={loaded_params_sha256} != "
        f"frozen expected {expected} (source: {ci.get('base_checkpoint_expected_source')}).")


# ---------------------------------------------------------------------------
# §六.6/§六.7 — full validation + certificate
# ---------------------------------------------------------------------------

def validate_runtime_against_formal_config(formal_record, runtime_scientific_config, *,
                                           gpu_uuid=None, out_dir=None,
                                           checkpoint_identity=None,
                                           cli_args=None, runtime_constants=None):
    """Full runtime<->formal binding validation. Returns the certificate dict.

    scientific_config_match : canonical YAML scientific_config == canonical REAL runtime
                              scientific_config (deep diff empty) AND equal canonical SHA256.
    runtime_assignment_match: formal runtime_assignment (gpu_uuid, out_dir) == CLI placement
                              (validated SEPARATELY; not part of the scientific SHA).
    certificate_status      : PASS only if scientific_config_match AND runtime_assignment_match
                              AND validation_errors == []."""
    if formal_record is None:
        raise ValueError(
            "FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE: no formal config loaded.")
    config = formal_record.get("config") or {}
    formal_sci_raw = config.get("scientific_config")
    if not isinstance(formal_sci_raw, dict):
        raise ValueError("FORMAL_CONFIG_INVALID: formal YAML has no scientific_config block.")

    formal_sci = canonical_scientific_config(formal_sci_raw)
    runtime_sci = canonical_scientific_config(runtime_scientific_config)
    formal_sha = scientific_config_sha256(formal_sci_raw)
    runtime_sha = scientific_config_sha256(runtime_scientific_config)

    validation_errors = []
    diffs = deep_diff(formal_sci, runtime_sci)
    for d in diffs:
        validation_errors.append(
            f"FORMAL_CONFIG_RUNTIME_MISMATCH @ {d['path']}: formal={d['formal']!r} "
            f"runtime={d['runtime']!r} ({d['kind']})")
    if formal_sha != runtime_sha:
        validation_errors.append(
            f"FORMAL_CONFIG_RUNTIME_MISMATCH: scientific_config_sha256 formal={formal_sha} "
            f"!= runtime={runtime_sha}")
    scientific_config_match = (not diffs) and (formal_sha == runtime_sha)

    # §六.6 runtime assignment — separate from the scientific SHA.
    ra = config.get("runtime_assignment") or {}
    ra_errors = []
    if gpu_uuid is not None and ra.get("gpu_uuid") is not None \
            and str(ra.get("gpu_uuid")) != str(gpu_uuid):
        ra_errors.append(
            f"RUNTIME_ASSIGNMENT_MISMATCH: gpu_uuid runtime={gpu_uuid!r} != "
            f"formal={ra.get('gpu_uuid')!r}")
    f_out = str(ra.get("out_dir") or "").replace("\\", "/").rstrip("/")
    r_out = str(out_dir or "").replace("\\", "/").rstrip("/")
    if f_out and gpu_uuid is not None and out_dir is not None:
        if not (f_out == r_out or r_out.endswith("/" + f_out) or r_out.endswith(f_out)):
            ra_errors.append(
                f"RUNTIME_ASSIGNMENT_MISMATCH: out_dir runtime={out_dir!r} != formal="
                f"{ra.get('out_dir')!r}")
    validation_errors.extend(ra_errors)
    runtime_assignment_match = not ra_errors

    # §六.5 checkpoint identity (structural part; the params SHA is verified post-load).
    ci = dict(checkpoint_identity or {})
    if ci.get("base_checkpoint_label_note"):
        validation_errors.append(
            f"BASE_CHECKPOINT_LABEL_MISMATCH: {ci['base_checkpoint_label_note']}")
    if ci.get("base_checkpoint_match") == "FAIL":
        validation_errors.append("BASE_CHECKPOINT_SHA_MISMATCH: see checkpoint_identity.")

    certificate_status = ("PASS" if (scientific_config_match and runtime_assignment_match
                                     and not validation_errors) else "FAIL")

    return dict(
        schema=SCHEMA,
        certificate_version="phase4a_v2.2",
        formal_config_path=formal_record.get("path"),
        formal_config_realpath=formal_record.get("realpath"),
        formal_config_file_sha256=formal_record.get("file_sha256"),
        scientific_config_sha256=formal_sha,
        runtime_scientific_config_sha256=runtime_sha,
        scientific_config_match=scientific_config_match,
        scientific_config_diffs=diffs,
        runtime_assignment_match=runtime_assignment_match,
        arm=runtime_sci.get("carry_mode"),
        carry_mode=runtime_sci.get("carry_mode"),
        replay_mode=runtime_sci.get("replay_mode"),
        actual_cli_args=dict(cli_args or {}),
        runtime_constants=dict(runtime_constants or {}),
        checkpoint_identity=ci,
        validation_errors=validation_errors,
        certificate_status=certificate_status)


def write_runtime_config_certificate(certificate, path):
    """Write runtime_config_certificate.json (§六.7). Returns the absolute path."""
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(certificate, f, indent=2, sort_keys=True)
    return os.path.abspath(path)


def certificate_shas_record(certificate):
    """The compact SHA record embedded in checkpoints / summaries / launch status (§六.8)."""
    return dict(
        formal_config_file_sha256=certificate.get("formal_config_file_sha256"),
        scientific_config_sha256=certificate.get("scientific_config_sha256"),
        runtime_scientific_config_sha256=certificate.get("runtime_scientific_config_sha256"),
        runtime_config_certificate_status=certificate.get("certificate_status"),
        base_checkpoint_expected_sha256=(
            certificate.get("checkpoint_identity") or {}).get("base_checkpoint_expected_sha256"),
        base_checkpoint_expected_sha256_status=(
            certificate.get("checkpoint_identity") or {}).get(
                "base_checkpoint_expected_sha256_status"))


# ---------------------------------------------------------------------------
# §六.9 — self-test (>= 19 fail-closed negative cases; pure Python, no JAX, no training)
# ---------------------------------------------------------------------------

def _reference_runtime_kwargs(carry_mode="persistent"):
    """The REAL frozen runtime values of the driver (mirrors Cfg / FullP2Config / K_BATCH /
    ANCHOR_INTERVAL / MIN_SEQUENCE_LENGTH / args defaults). Used to build both the reference
    runtime scientific_config and — when no YAML file is reachable — the reference formal block,
    so a match PASS is the identity comparison."""
    return dict(
        carry_mode=carry_mode,
        replay_mode="original_vtrace",
        allow_full_p2_legacy=False,
        sequence_length=129, segment_len=128,
        hindsight=False, awr=False, w_original_vtrace=1.0,
        base_checkpoint="ckpt17500", seed=42, total_updates=12, save_every=2,
        num_envs=16, num_steps=128, task="DEFEAT_KOBOLD",
        optimistic_reset_ratio=16, condition_on_task=True,
        replay_batch_size=4, replay_buffer_capacity=64, anchor_interval=128,
        min_sequence_length=129, eligible_only_sampling=True,
        ppo_lr=2.0e-5, ppo_max_grad_norm=1.0, ppo_gamma=0.999, ppo_gae_lambda=0.8,
        ppo_clip_eps=0.2, ppo_vf_coef=0.5, ppo_ent_coef=0.002, ppo_update_epochs=1,
        ppo_num_minibatches=2, ppo_value_target_clip_min=-50.0,
        ppo_value_target_clip_max=300.0,
        vtrace_rho_bar=1.0, vtrace_c_bar=1.0, vtrace_vt_clip_min=-50.0,
        vtrace_vt_clip_max=300.0,
        kl_replay_max=0.05, kl_run_max=0.1,
        actor_step_scales=[1.0, 0.5, 0.25, 0.125],
        policy_lag_gate_active=False,
        policy_lag_gate_mode="not_applicable_original_vtrace",
        policy_lag_max_policy_lag=None,
        legacy_full_p2_active=False, legacy_full_p2_max_policy_lag=16,
        ema_tau=0.995, ent_floor=0.05, grad_clip=1.0, adam_eps=1.0e-5,
        net_activation="relu", net_embed_size=256, net_num_heads=8, net_qkv_features=256,
        net_num_layers=2, net_gating=True, net_gating_bias=2.0, net_window_mem=128,
        net_rmt_num_tokens=16,
        evaluator="frozen_rmt16_evaluator")


def _reference_formal_record(carry_mode="persistent",
                             gpu_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
                             out_dir="runs/RMT16-PERSISTENT-ORIGVTRACE-129"):
    """In-memory formal record: scientific_config = the reference runtime scientific_config
    (so a self-match is the identity), wrapped as a loaded YAML record."""
    sci = build_runtime_scientific_config(**_reference_runtime_kwargs(carry_mode))
    raw = canonical_json(dict(schema=SCHEMA, arm=carry_mode, scientific_config=sci,
                              runtime_assignment=dict(gpu_uuid=gpu_uuid, out_dir=out_dir)))
    return dict(path="<synthetic>", realpath="<synthetic>", file_sha256=_sha256_bytes(
        raw.encode("utf-8")),
        config=dict(schema=SCHEMA, arm=carry_mode, scientific_config=sci,
                    runtime_assignment=dict(gpu_uuid=gpu_uuid, out_dir=out_dir)))


def self_test():
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)

    def expect_mismatch(name, mutate, expect_substring=None):
        """Build the reference binding, apply `mutate(runtime_kwargs, formal_record)`, and
        require certificate_status=FAIL (and optionally an error mentioning expect_substring)."""
        kw = _reference_runtime_kwargs()
        rec = _reference_formal_record()
        mutate(kw, rec)
        runtime_sci = build_runtime_scientific_config(**kw)
        cert = validate_runtime_against_formal_config(
            rec, runtime_sci, gpu_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
            out_dir="runs/RMT16-PERSISTENT-ORIGVTRACE-129")
        errs = " | ".join(cert["validation_errors"])
        check(name, cert["certificate_status"] == "FAIL" and (
            expect_substring is None or expect_substring in errs), errs[:160])

    print("phase4a_v2_runtime_config --self-test (Phase4A-v2.2 §六.9)", flush=True)

    # (0) identity -> PASS
    kw0 = _reference_runtime_kwargs()
    rec0 = _reference_formal_record()
    cert0 = validate_runtime_against_formal_config(
        rec0, build_runtime_scientific_config(**kw0),
        gpu_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
        out_dir="runs/RMT16-PERSISTENT-ORIGVTRACE-129")
    check("identity -> certificate_status=PASS + shas equal",
          cert0["certificate_status"] == "PASS" and cert0["scientific_config_match"]
          and cert0["runtime_assignment_match"]
          and cert0["scientific_config_sha256"]
          == cert0["runtime_scientific_config_sha256"]
          and cert0["validation_errors"] == [])

    # (1..14) scientific field mismatches -> FAIL with FORMAL_CONFIG_RUNTIME_MISMATCH
    expect_mismatch("wrong seed -> FAIL",
                    lambda kw, r: kw.update(seed=43), "seed")
    expect_mismatch("wrong total_updates -> FAIL",
                    lambda kw, r: kw.update(total_updates=11), "total_updates")
    expect_mismatch("wrong save_every -> FAIL",
                    lambda kw, r: kw.update(save_every=3), "save_every")
    expect_mismatch("wrong sequence_length -> FAIL",
                    lambda kw, r: kw.update(sequence_length=130), "sequence_length")
    expect_mismatch("wrong replay_mode -> FAIL",
                    lambda kw, r: kw.update(replay_mode="off"), "replay_mode")
    expect_mismatch("wrong carry_mode -> FAIL",
                    lambda kw, r: kw.update(carry_mode="reset128"), "carry_mode")
    expect_mismatch("wrong task -> FAIL",
                    lambda kw, r: kw.update(task="DEFEAT_ZOMBIE"), "task")
    expect_mismatch("wrong PPO lr -> FAIL",
                    lambda kw, r: kw.update(ppo_lr=3.0e-5), "ppo.lr")
    expect_mismatch("wrong vtrace rho_bar -> FAIL",
                    lambda kw, r: kw.update(vtrace_rho_bar=2.0), "vtrace.rho_bar")
    expect_mismatch("policy_lag block mismatch -> FAIL",
                    lambda kw, r: kw.update(policy_lag_max_policy_lag=16),
                    "policy_lag.max_policy_lag")
    expect_mismatch("wrong network embed_size -> FAIL",
                    lambda kw, r: kw.update(net_embed_size=512), "network.embed_size")
    expect_mismatch("missing formal scientific key -> FAIL",
                    lambda kw, r: r["config"]["scientific_config"].pop("ema_tau"),
                    "scientific_config.ema_tau")
    expect_mismatch("extra runtime scientific key -> FAIL",
                    lambda kw, r: r["config"]["scientific_config"].update(
                        {"unregistered_key": 1}),
                    "unregistered_key")
    expect_mismatch("YAML scientific SHA mismatch (mutated formal) -> FAIL",
                    lambda kw, r: r["config"]["scientific_config"].update({"seed": 99}),
                    "seed")

    # (15..16) runtime_assignment mismatches (separate from scientific SHA)
    kw = _reference_runtime_kwargs(); rec = _reference_formal_record()
    cert = validate_runtime_against_formal_config(
        rec, build_runtime_scientific_config(**kw),
        gpu_uuid="GPU-00000000-wrong", out_dir="runs/RMT16-PERSISTENT-ORIGVTRACE-129")
    check("wrong GPU UUID -> RUNTIME_ASSIGNMENT_MISMATCH (scientific still matches)",
          cert["certificate_status"] == "FAIL" and not cert["runtime_assignment_match"]
          and cert["scientific_config_match"]
          and any("RUNTIME_ASSIGNMENT_MISMATCH" in e for e in cert["validation_errors"]))
    kw = _reference_runtime_kwargs(); rec = _reference_formal_record()
    cert = validate_runtime_against_formal_config(
        rec, build_runtime_scientific_config(**kw),
        gpu_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
        out_dir="runs/SOMEWHERE-ELSE")
    check("wrong out_dir -> RUNTIME_ASSIGNMENT_MISMATCH",
          cert["certificate_status"] == "FAIL" and not cert["runtime_assignment_match"]
          and any("RUNTIME_ASSIGNMENT_MISMATCH" in e for e in cert["validation_errors"]))

    # (17) checkpoint label mismatch -> FAIL
    kw = _reference_runtime_kwargs(); rec = _reference_formal_record()
    ci = build_checkpoint_identity("/ckpt/99999/params")
    cert = validate_runtime_against_formal_config(
        rec, build_runtime_scientific_config(**kw),
        gpu_uuid="GPU-8df11537-ab79-722d-606f-411966196c4c",
        out_dir="runs/RMT16-PERSISTENT-ORIGVTRACE-129", checkpoint_identity=ci)
    check("checkpoint label mismatch -> FAIL",
          cert["certificate_status"] == "FAIL"
          and any("BASE_CHECKPOINT_LABEL_MISMATCH" in e for e in cert["validation_errors"]))

    # (18) missing formal config -> fail closed
    try:
        validate_runtime_against_formal_config(
            None, build_runtime_scientific_config(**_reference_runtime_kwargs()))
        check("missing formal config -> raised", False, "no raise")
    except ValueError as e:
        check("missing formal config -> raised",
              "FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE" in str(e))

    # (19) --formal_config requirement for original_vtrace (pre-JAX preflight)
    try:
        preflight_require_formal_config("original_vtrace", None)
        check("preflight: original_vtrace w/o --formal_config -> raised", False, "no raise")
    except ValueError as e:
        check("preflight: original_vtrace w/o --formal_config -> raised",
              "FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE" in str(e))
    ok_exempt = True
    try:
        preflight_require_formal_config("off", None)  # off remains exempt
    except ValueError:
        ok_exempt = False
    check("preflight: replay_mode=off exempt (legacy dev compat)", ok_exempt)

    # (20) arm binding mismatches
    for mut_name, mut in (
            ("schema", lambda r: r["config"].update(schema="wrong_schema")),
            ("arm", lambda r: r["config"].update(arm="reset128")),
            ("scientific carry_mode",
             lambda r: r["config"]["scientific_config"].update(carry_mode="reset128")),
            ("scientific replay_mode",
             lambda r: r["config"]["scientific_config"].update(replay_mode="off"))):
        rec = _reference_formal_record()
        mut(rec)
        try:
            validate_arm_binding(rec, "persistent", replay_mode="original_vtrace")
            check(f"arm binding: {mut_name} mismatch -> raised", False, "no raise")
        except ValueError as e:
            check(f"arm binding: {mut_name} mismatch -> raised",
                  "FORMAL_CONFIG_ARM_MISMATCH" in str(e))

    # (21) frozen base checkpoint SHA: PASS on match, fail closed on mismatch
    ci = build_checkpoint_identity("/ckpt/17500/full_state.pkl")
    ci_ok = verify_checkpoint_params_sha(ci, EXPECTED_BASE_CHECKPOINT_SHA256)
    check("base checkpoint SHA == frozen expectation -> PASS",
          ci_ok["base_checkpoint_match"] == "PASS")
    try:
        verify_checkpoint_params_sha(build_checkpoint_identity("/ckpt/17500/full_state.pkl"),
                                     "0" * 64)
        check("base checkpoint SHA mismatch -> raised", False, "no raise")
    except ValueError as e:
        check("base checkpoint SHA mismatch -> raised", "BASE_CHECKPOINT_SHA_MISMATCH" in str(e))
    ci_nf = verify_checkpoint_params_sha(
        build_checkpoint_identity("/ckpt/17500/full_state.pkl", expected_sha256=None), "abc")
    check("no frozen expectation -> NOT_FROZEN (never fabricated)",
          ci_nf["base_checkpoint_match"] == "NOT_FROZEN"
          and ci_nf["base_checkpoint_expected_sha256_status"] == "NOT_FROZEN")

    # (22) canonical key-order invariance of the scientific SHA
    sci_a = build_runtime_scientific_config(**_reference_runtime_kwargs())
    sci_b = json.loads(json.dumps(sci_a))  # same content
    sci_b = dict(reversed(list(sci_b.items())))
    check("scientific SHA key-order invariant",
          scientific_config_sha256(sci_a) == scientific_config_sha256(sci_b))

    n = len(results); n_pass = sum(results)
    print(f"SELF_TEST_SUMMARY total={n} pass={n_pass} fail={n - n_pass}", flush=True)
    print(f"FAIL_CLOSED_NEGATIVE_CASES={n - 1} (>= 19 required by §六.9)", flush=True)
    return 0 if n_pass == n else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic fail-closed binding self-tests (no JAX, no files "
                         "required, no training)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ap.error("--self-test is the only supported mode here; the driver imports this module.")


if __name__ == "__main__":
    raise SystemExit(main())
