#!/usr/bin/env python3
"""Phase4A-v2.2/v2.3 (CC2 §六 / §四 / §六 / §七) — PRE-REGISTERED YAML <-> REAL RUNTIME binding,
fail closed.

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

Phase4A-v2.3 additions:
  (v2.3 §四) runtime_assignment completeness + STRICT identity: REQUIRED_RUNTIME_ASSIGNMENT_FIELDS
          {arm, gpu_uuid, out_dir} must all be present + non-empty (RUNTIME_ASSIGNMENT_INCOMPLETE);
          arm bound four ways (RUNTIME_ASSIGNMENT_ARM_MISMATCH); gpu_uuid EXACT, no suffix
          (RUNTIME_ASSIGNMENT_GPU_MISMATCH); out_dir matched by REALPATH equality anchored at
          --run_root — relative only, no `..`, no absolute path, no suffix match
          (RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH).
  (v2.3 §六) certificate STATE MACHINE: build_precheck_certificate (pre-JAX) ->
          PENDING_CHECKPOINT_IDENTITY or FAIL; finalize_certificate (post checkpoint load) ->
          PASS/FAIL, certificate_finalized flag; a FAIL finalize overwrites any stale certificate
          so no old PASS can survive a checkpoint failure.
  (v2.3 §七) certificate artifact binding: payload SHA (§七.1), ATOMIC tempfile+fsync+replace
          write (§六.4), detached file-SHA sidecar (§七.2), extended certificate_shas_record
          (§七.3), verify_certificate_artifact tamper detection (§七.4 ->
          RUNTIME_CONFIG_CERTIFICATE_TAMPERED).

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
# Phase4A-v2.3 (§四) — runtime_assignment COMPLETENESS + STRICT identity, fail closed
# ---------------------------------------------------------------------------
# v2.2 validated gpu_uuid/out_dir LOOSELY: absent fields were silently skipped and out_dir used
# a suffix match (a runtime path merely ENDING WITH the formal out_dir passed). v2.3 makes the
# assignment fail CLOSED: every required field must be present and non-empty; arm is bound four
# ways; gpu_uuid must match exactly; out_dir must match by REALPATH equality anchored at the
# driver's --run_root (relative path only; no `..`; no absolute path; no suffix match).

REQUIRED_RUNTIME_ASSIGNMENT_FIELDS = ("arm", "gpu_uuid", "out_dir")


def _is_nonempty_str(value):
    return isinstance(value, str) and value.strip() != ""


def resolve_runtime_assignment(config):
    """§四.1 Resolve the effective runtime assignment, fail closed. `arm` comes from the formal
    config's canonical top-level `arm` field (the frozen YAML's runtime_assignment block carries
    gpu_uuid/out_dir only; arm is NOT duplicated there); gpu_uuid/out_dir come from
    runtime_assignment. Missing / null / empty / non-string on ANY required field raises
    RUNTIME_ASSIGNMENT_INCOMPLETE — there is no default and no bypass (no fail-open)."""
    if not isinstance(config, dict):
        raise ValueError("RUNTIME_ASSIGNMENT_INCOMPLETE: formal config is not a mapping.")
    ra = config.get("runtime_assignment")
    if not isinstance(ra, dict):
        raise ValueError(
            "RUNTIME_ASSIGNMENT_INCOMPLETE: formal YAML has no runtime_assignment mapping "
            "(gpu_uuid/out_dir must be pre-registered).")
    resolved = dict(arm=config.get("arm"),
                    gpu_uuid=ra.get("gpu_uuid"),
                    out_dir=ra.get("out_dir"))
    missing = [k for k in REQUIRED_RUNTIME_ASSIGNMENT_FIELDS
               if not _is_nonempty_str(resolved.get(k))]
    if missing:
        raise ValueError(
            f"RUNTIME_ASSIGNMENT_INCOMPLETE: required runtime_assignment field(s) "
            f"missing/null/empty/non-string: {missing}. Required (all non-empty strings): "
            f"{list(REQUIRED_RUNTIME_ASSIGNMENT_FIELDS)} — arm from the formal top-level `arm`, "
            "gpu_uuid/out_dir from runtime_assignment. No default, no bypass.")
    return resolved


def _validate_out_dir_strict(formal_out_dir, cli_out, run_root):
    """§四.3 STRICT out_dir identity. The formal out_dir must be a RELATIVE path (no absolute
    path, no drive letter, no `..` segment) that resolves INSIDE run_root, and the realpath of
    the actual CLI --out must EQUAL realpath(run_root/formal_out_dir). Exact equality — the v2.2
    suffix match is gone. Returns a list of errors (empty == PASS)."""
    errors = []
    f = str(formal_out_dir).replace("\\", "/")
    if os.path.isabs(f) or f.startswith("/") or (len(f) >= 2 and f[1] == ":"):
        errors.append(
            f"RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH: formal out_dir must be a RELATIVE path under "
            f"--run_root, got {formal_out_dir!r}")
        return errors
    if ".." in f.split("/"):
        errors.append(
            f"RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH: formal out_dir must not contain a '..' "
            f"segment, got {formal_out_dir!r}")
        return errors
    if not run_root:
        errors.append(
            "RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH: --run_root is required to pin the strict "
            "out_dir identity (realpath equality; no suffix match).")
        return errors
    run_root_real = os.path.realpath(str(run_root))
    expected = os.path.realpath(os.path.join(run_root_real, f))
    if not (expected == run_root_real or expected.startswith(run_root_real + os.sep)):
        errors.append(
            f"RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH: formal out_dir resolves OUTSIDE --run_root: "
            f"{expected!r} not under {run_root_real!r}")
        return errors
    actual = os.path.realpath(str(cli_out))
    if actual != expected:
        errors.append(
            f"RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH: actual out realpath={actual!r} != expected "
            f"realpath(run_root/out_dir)={expected!r} (strict equality; no suffix match).")
    return errors


def validate_runtime_assignment(config, *, cli_carry, cli_gpu, cli_out, run_root=None):
    """§四.2/§四.3/§四.4 fail-closed runtime_assignment validation (SEPARATE from the scientific
    SHA). Returns a record: runtime_assignment_match + runtime_assignment_errors + the resolved
    assignment. Error codes:
      RUNTIME_ASSIGNMENT_INCOMPLETE      — required field missing/null/empty (raised by resolve)
      RUNTIME_ASSIGNMENT_ARM_MISMATCH    — runtime_assignment.arm / formal top-level arm /
                                           scientific_config.carry_mode / CLI carry_mode disagree
      RUNTIME_ASSIGNMENT_GPU_MISMATCH    — formal gpu_uuid != CLI --gpu_uuid (exact; no suffix)
      RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH— strict realpath mismatch (see _validate_out_dir_strict)
    """
    resolved = resolve_runtime_assignment(config)   # may raise RUNTIME_ASSIGNMENT_INCOMPLETE
    errors = []
    sci = config.get("scientific_config") or {}
    arms = dict(runtime_assignment_arm=resolved["arm"],
                formal_top_arm=config.get("arm"),
                scientific_config_carry_mode=sci.get("carry_mode"),
                cli_carry_mode=cli_carry)
    if len(set(str(v) for v in arms.values())) != 1:
        errors.append(
            f"RUNTIME_ASSIGNMENT_ARM_MISMATCH: four-way arm binding disagrees: {arms}")
    if str(resolved["gpu_uuid"]) != str(cli_gpu):
        errors.append(
            f"RUNTIME_ASSIGNMENT_GPU_MISMATCH: formal gpu_uuid={resolved['gpu_uuid']!r} != "
            f"cli --gpu_uuid={cli_gpu!r} (exact equality; no suffix match).")
    errors.extend(_validate_out_dir_strict(resolved["out_dir"], cli_out, run_root))
    return dict(runtime_assignment_match=(not errors),
                runtime_assignment_errors=errors,
                runtime_assignment_resolved=resolved,
                cli_carry_mode=cli_carry, cli_gpu_uuid=cli_gpu, cli_out=cli_out,
                run_root=run_root,
                cli_out_realpath=(os.path.realpath(str(cli_out)) if cli_out else None))


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
        # Phase4A-v2.3 (§四.3): STRICT equality only. The v2.2 suffix match
        # (r_out.endswith("/"+f_out) / r_out.endswith(f_out)) is REMOVED: a runtime path that
        # merely ENDS WITH the formal out_dir is no longer accepted. The driver's primary
        # assignment validation is validate_runtime_assignment() (realpath equality anchored at
        # --run_root); this legacy in-certificate check is kept strict-consistent with it.
        if f_out != r_out:
            ra_errors.append(
                f"RUNTIME_ASSIGNMENT_MISMATCH: out_dir runtime={out_dir!r} != formal="
                f"{ra.get('out_dir')!r} (strict equality; no suffix match)")
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


def certificate_shas_record(certificate, *, certificate_file_sha256=None,
                            certificate_sidecar_path=None):
    """The compact SHA record embedded in checkpoints / summaries / launch status (§六.8).

    Phase4A-v2.3 (§七.3): the record now ALSO binds the certificate's own artifact identity —
    payload SHA, final FILE SHA, sidecar path, finalized flag, and the full base-checkpoint
    comparison (loaded params SHA + match) — so checkpoint/summary readers can re-verify the
    certificate chain without trust. The v2.2 keys are all retained (superset)."""
    cert = certificate or {}
    ci = cert.get("checkpoint_identity") or {}
    rec = dict(
        formal_config_file_sha256=cert.get("formal_config_file_sha256"),
        scientific_config_sha256=cert.get("scientific_config_sha256"),
        runtime_scientific_config_sha256=cert.get("runtime_scientific_config_sha256"),
        runtime_config_certificate_status=cert.get("certificate_status"),
        base_checkpoint_expected_sha256=ci.get("base_checkpoint_expected_sha256"),
        base_checkpoint_expected_sha256_status=ci.get(
            "base_checkpoint_expected_sha256_status"))
    rec.update(dict(
        runtime_config_certificate_version=cert.get("certificate_version"),
        runtime_config_certificate_finalized=cert.get("certificate_finalized"),
        runtime_config_certificate_payload_sha256=cert.get("certificate_payload_sha256"),
        runtime_config_certificate_file_sha256=certificate_file_sha256,
        runtime_config_certificate_sidecar_path=certificate_sidecar_path,
        base_checkpoint_params_sha256=ci.get("base_checkpoint_params_sha256"),
        base_checkpoint_match=ci.get("base_checkpoint_match")))
    return rec


# ---------------------------------------------------------------------------
# Phase4A-v2.3 (§六) — certificate STATE MACHINE: PENDING_CHECKPOINT_IDENTITY -> PASS/FAIL
# ---------------------------------------------------------------------------
# v2.2 wrote the certificate ONCE with status PASS before the checkpoint params were loaded,
# then REWROTE it after load — so a checkpoint-SHA failure could leave a stale PASS on disk, and
# the full scientific binding ran only AFTER `import jax`. v2.3 introduces an explicit state
# machine:
#   1. PRE-JAX  build_precheck_certificate -> PENDING_CHECKPOINT_IDENTITY (full scientific
#      binding from the frozen pure-Python spec + formal identity + strict assignment), written
#      ATOMICALLY as a provisional, NOT-finalized certificate.
#   2. POST-import: the driver diffs the REAL imported constants against the frozen spec
#      (IMPORTED_RUNTIME_CONSTANTS_MISMATCH on drift) and binds the executed protocol source.
#   3. POST-load finalize_certificate -> PASS (from PENDING + checkpoint PASS/NOT_FROZEN) or
#      FAIL (anything else, incl. checkpoint error). The FAIL finalize OVERWRITES the on-disk
#      certificate, so NO stale PASS/PENDING survives a failure, and the driver exits nonzero.

CERTIFICATE_VERSION = "phase4a_v2.3"
CERTIFICATE_STATUS_PENDING = "PENDING_CHECKPOINT_IDENTITY"
CERTIFICATE_STATUS_PASS = "PASS"
CERTIFICATE_STATUS_FAIL = "FAIL"

# Phase4A-v2.4 (§四.1): the checkpoint flow is ONE unified fail-closed try/except in the driver
# (manager init -> restore -> structure -> params extraction -> params hash -> SHA compare).
# Whichever stage raises is recorded in the finalized FAIL certificate as
# `checkpoint_failure_stage`, so a reviewer can tell a missing-checkpoint failure from a restore
# exception, a structural break, a hash failure or a frozen-SHA mismatch WITHOUT rerunning.
# NONE = the checkpoint flow completed (the certificate then finalizes PASS, or FAILs for a
# non-checkpoint reason such as a non-PENDING precheck).
CHECKPOINT_FAILURE_STAGES = (
    "CHECKPOINT_MANAGER_INIT",
    "CHECKPOINT_RESTORE",
    "CHECKPOINT_STRUCTURE",
    "CHECKPOINT_PARAMS_EXTRACTION",
    "CHECKPOINT_PARAMS_HASH",
    "CHECKPOINT_SHA_COMPARE",
    "NONE",
)


def build_precheck_certificate(formal_record, runtime_scientific_config, *,
                               formal_identity_record=None,
                               assignment_record=None,
                               checkpoint_identity=None,
                               frozen_spec_sha256=None,
                               cli_args=None, runtime_constants=None,
                               snapshot_root=None, run_root=None):
    """§五.2/§六.1 the PRE-JAX precheck certificate. Binds, all before `import jax`:
      * formal_config_identity (§三: canonical path + file SHA + scientific SHA),
      * the FULL scientific_config (formal YAML vs the runtime scientific config built from the
        frozen pure-Python spec — canonical deep diff + SHA),
      * runtime_assignment (§四: completeness + 4-way arm + gpu + strict out_dir).
    certificate_status = PENDING_CHECKPOINT_IDENTITY when all three pass (NOT PASS — the base
    checkpoint params SHA can only be compared after the post-JAX load), else FAIL.
    certificate_finalized stays False until finalize_certificate()."""
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

    formal_config_identity = bool(
        isinstance(formal_identity_record, dict)
        and formal_identity_record.get("formal_config_identity") == "PASS")
    runtime_assignment_match = bool(
        isinstance(assignment_record, dict)
        and assignment_record.get("runtime_assignment_match"))
    if isinstance(assignment_record, dict):
        validation_errors.extend(assignment_record.get("runtime_assignment_errors") or [])

    ci = dict(checkpoint_identity or {})
    # params SHA cannot be compared pre-JAX: build_checkpoint_identity sets match=None, which we
    # relabel PENDING (undecided). A truthy match here would be a logic error and is left as-is
    # for the caller to surface.
    if not ci.get("base_checkpoint_match"):
        ci["base_checkpoint_match"] = "PENDING"

    if (scientific_config_match and runtime_assignment_match and formal_config_identity
            and not validation_errors):
        status = CERTIFICATE_STATUS_PENDING
    else:
        status = CERTIFICATE_STATUS_FAIL
        if not formal_config_identity and not any(
                e.startswith(("FORMAL_CONFIG", "RUNTIME_ASSIGNMENT"))
                for e in validation_errors):
            validation_errors.append(
                "FORMAL_CONFIG_IDENTITY_MISMATCH: canonical formal-config identity did not PASS "
                "(§三 path + content identity is REQUIRED for the precheck certificate).")

    return dict(
        schema=SCHEMA,
        certificate_version=CERTIFICATE_VERSION,
        certificate_status=status,
        certificate_finalized=False,
        frozen_spec_sha256=frozen_spec_sha256,
        formal_config_identity=dict(formal_identity_record or {}),
        formal_config_path=formal_record.get("path"),
        formal_config_realpath=formal_record.get("realpath"),
        formal_config_file_sha256=formal_record.get("file_sha256"),
        scientific_config_sha256=formal_sha,
        runtime_scientific_config_sha256=runtime_sha,
        scientific_config_match=scientific_config_match,
        scientific_config_diffs=diffs,
        runtime_assignment_match=runtime_assignment_match,
        runtime_assignment=dict(assignment_record or {}),
        snapshot_root=snapshot_root,
        run_root=run_root,
        arm=runtime_sci.get("carry_mode"),
        carry_mode=runtime_sci.get("carry_mode"),
        replay_mode=runtime_sci.get("replay_mode"),
        actual_cli_args=dict(cli_args or {}),
        runtime_constants=dict(runtime_constants or {}),
        checkpoint_identity=ci,
        validation_errors=validation_errors)


def finalize_certificate(precheck_certificate, checkpoint_identity=None, *,
                         checkpoint_error=None, checkpoint_failure_stage="NONE"):
    """§六.2/§六.3 deterministic finalization. Returns a NEW certificate (input not mutated):
      * checkpoint_error is None AND checkpoint match is PASS/NOT_FROZEN AND the precheck status
        was PENDING_CHECKPOINT_IDENTITY  ->  status PASS, certificate_finalized=True, and the
        validation_errors list is cleared.
      * anything else (checkpoint error, non-PENDING precheck, match not PASS/NOT_FROZEN)
        ->  status FAIL, certificate_finalized=True, reasons appended.
    A FAIL finalization MUST be written over the on-disk certificate (the driver calls
    write_certificate_atomic right after), so a checkpoint-SHA failure can never leave a stale
    PASS behind.

    Phase4A-v2.4 (§四.1): the finalized certificate ALSO carries `checkpoint_failure_stage`
    (one of CHECKPOINT_FAILURE_STAGES). On a checkpoint failure it names the exact stage that
    raised (manager init / restore / structure / params extraction / params hash / SHA compare);
    on a clean checkpoint flow it is NONE. An unknown stage value is rejected fail-closed."""
    if checkpoint_failure_stage not in CHECKPOINT_FAILURE_STAGES:
        raise ValueError(
            f"CHECKPOINT_FAILURE_STAGE_INVALID: {checkpoint_failure_stage!r} not in "
            f"{CHECKPOINT_FAILURE_STAGES}")
    cert = dict(precheck_certificate or {})
    ci = dict(checkpoint_identity or cert.get("checkpoint_identity") or {})
    cert["checkpoint_identity"] = ci
    if checkpoint_error is not None:
        ci["base_checkpoint_match"] = "FAIL"
        errs = list(cert.get("validation_errors") or [])
        errs.append(f"BASE_CHECKPOINT_FAILURE: {checkpoint_error}")
        cert["validation_errors"] = errs
        cert["certificate_status"] = CERTIFICATE_STATUS_FAIL
        cert["certificate_finalized"] = True
        cert["checkpoint_failure_stage"] = (
            checkpoint_failure_stage if checkpoint_failure_stage != "NONE"
            else "CHECKPOINT_SHA_COMPARE")
        return cert
    match = ci.get("base_checkpoint_match")
    pre_status = cert.get("certificate_status")
    if pre_status == CERTIFICATE_STATUS_PENDING and match in ("PASS", "NOT_FROZEN"):
        cert["certificate_status"] = CERTIFICATE_STATUS_PASS
        cert["certificate_finalized"] = True
        cert["validation_errors"] = []
        cert["checkpoint_failure_stage"] = "NONE"
        return cert
    errs = list(cert.get("validation_errors") or [])
    if pre_status != CERTIFICATE_STATUS_PENDING:
        errs.append(
            f"CERTIFICATE_NOT_PENDING_AT_FINALIZE: precheck status={pre_status!r} (a FAIL "
            "precheck can never finalize to PASS).")
    if match not in ("PASS", "NOT_FROZEN"):
        errs.append(f"BASE_CHECKPOINT_MATCH_NOT_PASS: base_checkpoint_match={match!r}")
    cert["validation_errors"] = errs
    cert["certificate_status"] = CERTIFICATE_STATUS_FAIL
    cert["certificate_finalized"] = True
    cert["checkpoint_failure_stage"] = checkpoint_failure_stage
    return cert


# ---------------------------------------------------------------------------
# Phase4A-v2.3 (§七) — certificate payload/file SHA, ATOMIC write, sidecar, tamper detection
# ---------------------------------------------------------------------------
# v2.2's certificate carried no SHA of ITSELF and was written non-atomically (direct open/write),
# so a checkpoint/summary could not be bound to the exact certificate FILE, and a crash mid-write
# could leave a truncated certificate. v2.3 embeds a payload SHA (§七.1), writes via
# tempfile+fsync+os.replace (§六.4), records the FINAL file SHA in a detached sidecar (§七.2),
# binds that file SHA into checkpoint/summary records (§七.3), and provides a tamper detector
# (§七.4) that fails closed on ANY modification.

# Fields computed AT WRITE TIME; excluded from the signed payload so the payload stays stable.
_CERTIFICATE_SELF_FIELDS = ("certificate_payload_sha256", "certificate_file_sha256",
                            "certificate_sidecar_path", "certificate_written_via")


def compute_certificate_payload_sha(certificate):
    """§七.1 SHA256 over the canonical JSON of the certificate EXCLUDING the self-fields. Covers
    certificate_status, certificate_finalized, validation_errors, both scientific SHAs, the FULL
    checkpoint_identity (incl. the loaded base params SHA) and every other payload field — so a
    FAIL->PASS flip, an edited error list or an edited base SHA all change this SHA."""
    payload = {k: v for k, v in (certificate or {}).items()
               if k not in _CERTIFICATE_SELF_FIELDS}
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def atomic_write_json(path, obj):
    """§六.4 ATOMIC JSON write: same-directory temp file -> write -> flush -> fsync -> os.replace.
    Readers can never observe a partial or truncated certificate. Returns the absolute path."""
    path = os.path.abspath(str(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def certificate_sidecar_path(certificate_path):
    """The detached file-SHA sidecar name (§七.2): runtime_config_certificate.json ->
    runtime_config_certificate.sha256."""
    p = str(certificate_path)
    if p.lower().endswith(".json"):
        return p[:-len(".json")] + ".sha256"
    return p + ".sha256"


def write_certificate_atomic(certificate, path):
    """§七.1/§七.2 FINAL certificate write. Embeds the payload SHA (§七.1), writes atomically
    (§六.4), computes the FINAL file SHA over the exact written bytes, and writes the detached
    sidecar `<name>.sha256` containing EXACTLY `<sha256>  <basename>\n` (two tokens; v2.4 §十
    verifies the basename token too). The certificate file is NOT written again after this call,
    so the sidecar SHA stays valid for the artifact's lifetime.

    Phase4A-v2.4 (§五.1): returns the FIFTH element `written_certificate` — the EXACT dict that
    was serialized to disk (payload SHA + written_via included). v2.3 mutated a LOCAL copy, so
    the caller's in-memory certificate never carried `certificate_payload_sha256` and the
    manifest/summary payload SHA was always null. v2.4 callers MUST immediately adopt it:
        RUNTIME_CONFIG_CERTIFICATE = written_certificate
    so manifest / summary / launch-status bind the SAME object that is on disk.
    Returns (cert_path, sidecar_path, file_sha256, payload_sha256, written_certificate)."""
    cert = dict(certificate or {})
    for k in _CERTIFICATE_SELF_FIELDS:
        cert.pop(k, None)
    payload_sha = compute_certificate_payload_sha(cert)
    cert["certificate_payload_sha256"] = payload_sha
    cert["certificate_written_via"] = "atomic_tempfile_fsync_replace"
    cert_path = atomic_write_json(path, cert)
    with open(cert_path, "rb") as f:
        file_sha = _sha256_bytes(f.read())
    sidecar = certificate_sidecar_path(cert_path)
    base = os.path.basename(cert_path)
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(f"{file_sha}  {base}\n")
        f.flush()
        os.fsync(f.fileno())
    return cert_path, sidecar, file_sha, payload_sha, cert


def verify_certificate_artifact(certificate_path, sidecar_path=None, *,
                                expected_file_sha256=None,
                                expected_payload_sha256=None):
    """§七.4 tamper detection over a written certificate. Recomputes the file SHA (over the disk
    bytes) and the payload SHA (over the parsed certificate minus self-fields) and requires ALL:
      * recomputed payload SHA == the embedded certificate_payload_sha256 (catches FAIL->PASS
        flips, edited validation_errors, edited base-checkpoint SHA),
      * file SHA == the detached sidecar's SHA (catches ANY byte change, including an attacker
        who also recomputes the payload SHA),
      * expected file/payload SHAs (e.g. from the summary/checkpoint manifest) match, if given.
    Any violation raises RUNTIME_CONFIG_CERTIFICATE_TAMPERED. Returns a PASS record.

    Phase4A-v2.4 (§十): the sidecar is validated as a WHOLE LINE, not just its first token. It
    MUST be exactly `<sha256>  <certificate basename>\n`: exactly two whitespace-separated
    tokens, token[0] == the recomputed file SHA, token[1] == os.path.basename of the
    certificate, and a single trailing newline. A correct SHA with a WRONG basename (sidecar
    transplanted from another certificate file), extra tokens, an empty sidecar, or a
    truncated/missing newline all raise RUNTIME_CONFIG_CERTIFICATE_TAMPERED."""
    with open(certificate_path, "rb") as f:
        raw = f.read()
    file_sha = _sha256_bytes(raw)
    try:
        cert = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise ValueError(
            f"RUNTIME_CONFIG_CERTIFICATE_TAMPERED: certificate is not valid JSON ({e})")
    payload_sha = compute_certificate_payload_sha(cert)
    errors = []
    embedded_payload = cert.get("certificate_payload_sha256")
    if embedded_payload != payload_sha:
        errors.append(
            f"payload sha tampered: embedded={embedded_payload} recomputed={payload_sha}")
    if expected_payload_sha256 is not None and payload_sha != expected_payload_sha256:
        errors.append(
            f"payload sha != expected: recomputed={payload_sha} "
            f"expected={expected_payload_sha256}")
    if expected_file_sha256 is not None and file_sha != expected_file_sha256:
        errors.append(
            f"file sha != expected: actual={file_sha} expected={expected_file_sha256}")
    if sidecar_path is None:
        sidecar_path = certificate_sidecar_path(certificate_path)
    if os.path.exists(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar_raw = f.read()
        parts = sidecar_raw.split()
        sidecar_sha = parts[0] if parts else None
        sidecar_base = parts[1] if len(parts) >= 2 else None
        expected_base = os.path.basename(str(certificate_path))
        if sidecar_sha != file_sha:
            errors.append(
                f"sidecar file sha mismatch: sidecar={sidecar_sha} actual={file_sha}")
        if len(parts) != 2:
            errors.append(
                f"sidecar format invalid: expected exactly 2 tokens '<sha256>  <basename>', "
                f"got {len(parts)} token(s) in {sidecar_raw!r}")
        elif sidecar_base != expected_base:
            errors.append(
                f"sidecar basename mismatch: sidecar references {sidecar_base!r} but the "
                f"certificate file basename is {expected_base!r}")
        elif not sidecar_raw.endswith("\n"):
            errors.append(
                "sidecar format invalid: missing trailing newline (truncated sidecar)")
    else:
        errors.append(f"sidecar missing: {sidecar_path}")
    if errors:
        raise ValueError("RUNTIME_CONFIG_CERTIFICATE_TAMPERED: " + " | ".join(errors))
    return dict(certificate_tamper_check="PASS", certificate_path=str(certificate_path),
                sidecar_path=str(sidecar_path), file_sha256=file_sha,
                payload_sha256=payload_sha)


# ---------------------------------------------------------------------------
# §六.9 — self-test (>= 19 fail-closed negative cases; pure Python, no JAX, no training)
# ---------------------------------------------------------------------------

def _reference_runtime_kwargs(carry_mode="persistent"):
    """The REAL frozen runtime values of the driver (mirrors Cfg / FullP2Config / K_BATCH /
    ANCHOR_INTERVAL / MIN_SEQUENCE_LENGTH / args defaults). Used to build both the reference
    runtime scientific_config and — when no YAML file is reachable — the reference formal block,
    so a match PASS is the identity comparison.

    Phase4A-v2.3 (§五.1): the single source of truth for these values is the pure-Python frozen
    spec (phase4a_v2_frozen_spec.FROZEN_SPEC), which the driver imports BEFORE `import jax` to
    perform the full pre-JAX scientific binding. This function DELEGATES to it, so the reference
    / self-test binding can never silently diverge from the frozen spec the driver binds."""
    import phase4a_v2_frozen_spec as FSPEC   # lazy import: FSPEC imports RTC only inside funcs
    return FSPEC.build_kwargs(carry_mode)


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

    # =======================================================================
    # Phase4A-v2.3 additions (§四 assignment fail-closed / §六 state machine / §七 SHA+sidecar)
    # =======================================================================
    import tempfile
    import shutil

    _GPU = "GPU-8df11537-ab79-722d-606f-411966196c4c"
    _OUT = "runs/RMT16-PERSISTENT-ORIGVTRACE-129"

    def _cfg_with_ra(arm="persistent", gpu=_GPU, out=_OUT, top_arm=None):
        return dict(schema=SCHEMA, arm=(top_arm if top_arm is not None else arm),
                    scientific_config=build_runtime_scientific_config(
                        **_reference_runtime_kwargs(arm)),
                    runtime_assignment=dict(gpu_uuid=gpu, out_dir=out))

    # (23) §四.1 completeness -> RUNTIME_ASSIGNMENT_INCOMPLETE (fail closed, no default)
    def expect_incomplete(name, cfg):
        try:
            resolve_runtime_assignment(cfg)
            check(name, False, "no raise")
        except ValueError as e:
            check(name, "RUNTIME_ASSIGNMENT_INCOMPLETE" in str(e))

    c = _cfg_with_ra(); c["runtime_assignment"]["gpu_uuid"] = None
    expect_incomplete("(23a) null gpu_uuid -> INCOMPLETE", c)
    c = _cfg_with_ra(); c["runtime_assignment"]["out_dir"] = ""
    expect_incomplete("(23b) empty out_dir -> INCOMPLETE", c)
    c = _cfg_with_ra(); del c["runtime_assignment"]["gpu_uuid"]
    expect_incomplete("(23c) missing gpu_uuid key -> INCOMPLETE", c)
    c = _cfg_with_ra(); c["arm"] = None
    expect_incomplete("(23d) null top-level arm -> INCOMPLETE", c)
    c = _cfg_with_ra(); del c["runtime_assignment"]
    expect_incomplete("(23e) no runtime_assignment block -> INCOMPLETE", c)
    c = _cfg_with_ra(); c["runtime_assignment"]["gpu_uuid"] = 12345
    expect_incomplete("(23f) non-string gpu_uuid -> INCOMPLETE", c)

    # (24) §四.2-4 fully consistent assignment (real temp run_root) -> PASS, and each of the
    # three mismatch classes fails closed independently.
    tmp_root = tempfile.mkdtemp(prefix="p4av23_ra_")
    try:
        cli_out_abs = os.path.join(tmp_root, _OUT)
        os.makedirs(cli_out_abs, exist_ok=True)
        rec = validate_runtime_assignment(_cfg_with_ra(), cli_carry="persistent",
                                          cli_gpu=_GPU, cli_out=cli_out_abs,
                                          run_root=tmp_root)
        check("(24a) consistent assignment -> PASS",
              rec["runtime_assignment_match"] and rec["runtime_assignment_errors"] == [])

        rec = validate_runtime_assignment(_cfg_with_ra(), cli_carry="reset128",
                                          cli_gpu=_GPU, cli_out=cli_out_abs,
                                          run_root=tmp_root)
        check("(24b) CLI carry_mode disagrees -> RUNTIME_ASSIGNMENT_ARM_MISMATCH",
              not rec["runtime_assignment_match"] and any(
                  "RUNTIME_ASSIGNMENT_ARM_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))

        rec = validate_runtime_assignment(
            _cfg_with_ra(top_arm="reset128"), cli_carry="persistent",
            cli_gpu=_GPU, cli_out=cli_out_abs, run_root=tmp_root)
        check("(24c) formal top-level arm disagrees -> RUNTIME_ASSIGNMENT_ARM_MISMATCH",
              any("RUNTIME_ASSIGNMENT_ARM_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))

        rec = validate_runtime_assignment(_cfg_with_ra(), cli_carry="persistent",
                                          cli_gpu="GPU-ffffffff-other",
                                          cli_out=cli_out_abs, run_root=tmp_root)
        check("(24d) gpu_uuid mismatch -> RUNTIME_ASSIGNMENT_GPU_MISMATCH",
              any("RUNTIME_ASSIGNMENT_GPU_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))

        # (25) §四.3 strict out_dir: absolute / `..` / missing run_root / wrong realpath /
        # outside run_root / THE SUFFIX TRAP v2.2 wrongly accepted.
        rec = validate_runtime_assignment(_cfg_with_ra(out="/abs/runs/x"),
                                          cli_carry="persistent", cli_gpu=_GPU,
                                          cli_out=cli_out_abs, run_root=tmp_root)
        check("(25a) absolute formal out_dir -> OUT_DIR_MISMATCH",
              any("RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))

        rec = validate_runtime_assignment(_cfg_with_ra(out="../escape"),
                                          cli_carry="persistent", cli_gpu=_GPU,
                                          cli_out=cli_out_abs, run_root=tmp_root)
        check("(25b) '..' formal out_dir -> OUT_DIR_MISMATCH",
              any("RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))

        rec = validate_runtime_assignment(_cfg_with_ra(), cli_carry="persistent",
                                          cli_gpu=_GPU, cli_out=cli_out_abs, run_root=None)
        check("(25c) missing --run_root -> OUT_DIR_MISMATCH",
              any("RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))

        other_abs = os.path.join(tempfile.gettempdir(), "p4av23_other_out")
        os.makedirs(other_abs, exist_ok=True)
        rec = validate_runtime_assignment(_cfg_with_ra(), cli_carry="persistent",
                                          cli_gpu=_GPU, cli_out=other_abs,
                                          run_root=tmp_root)
        check("(25d) --out elsewhere (same relative name possible) -> OUT_DIR_MISMATCH",
              any("RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))
        shutil.rmtree(other_abs, ignore_errors=True)

        # the suffix trap: formal out_dir "run" would SUFFIX-match ".../runs/...-129" under
        # v2.2's endswith(); v2.3 must reject (realpath inequality).
        trap_abs = os.path.join(tmp_root, "prefix_" + _OUT)
        os.makedirs(trap_abs, exist_ok=True)
        rec = validate_runtime_assignment(_cfg_with_ra(), cli_carry="persistent",
                                          cli_gpu=_GPU, cli_out=trap_abs, run_root=tmp_root)
        check("(25e) suffix-trap --out (ends with formal out_dir) -> OUT_DIR_MISMATCH",
              any("RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH" in e
                  for e in rec["runtime_assignment_errors"]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # (26) §六 certificate state machine: PENDING -> PASS / FAIL, finalized flag, stale-PASS guard
    fid_pass = dict(formal_config_identity="PASS")
    ra_pass = dict(runtime_assignment_match=True, runtime_assignment_errors=[])
    kw = _reference_runtime_kwargs(); rec = _reference_formal_record()
    pre = build_precheck_certificate(
        rec, build_runtime_scientific_config(**kw), formal_identity_record=fid_pass,
        assignment_record=ra_pass,
        checkpoint_identity=build_checkpoint_identity("/ckpt/17500/full_state.pkl"))
    check("(26a) all-pass precheck -> PENDING_CHECKPOINT_IDENTITY (NOT PASS) + not finalized",
          pre["certificate_status"] == CERTIFICATE_STATUS_PENDING
          and pre["certificate_finalized"] is False
          and pre["checkpoint_identity"]["base_checkpoint_match"] == "PENDING"
          and pre["validation_errors"] == [])

    fin = finalize_certificate(
        pre, verify_checkpoint_params_sha(
            pre["checkpoint_identity"], EXPECTED_BASE_CHECKPOINT_SHA256))
    check("(26b) finalize with checkpoint PASS -> PASS + finalized + errors cleared",
          fin["certificate_status"] == CERTIFICATE_STATUS_PASS
          and fin["certificate_finalized"] is True and fin["validation_errors"] == []
          and fin["checkpoint_identity"]["base_checkpoint_match"] == "PASS")
    check("(26b') finalize does not mutate the precheck certificate",
          pre["certificate_status"] == CERTIFICATE_STATUS_PENDING
          and pre["certificate_finalized"] is False)

    fin_nf = finalize_certificate(
        pre, verify_checkpoint_params_sha(
            build_checkpoint_identity("/ckpt/17500/x", expected_sha256=None), "abc"))
    check("(26c) finalize with NOT_FROZEN checkpoint -> PASS (never fabricated)",
          fin_nf["certificate_status"] == CERTIFICATE_STATUS_PASS
          and fin_nf["checkpoint_identity"]["base_checkpoint_match"] == "NOT_FROZEN")

    fin_err = finalize_certificate(pre, checkpoint_error="BASE_CHECKPOINT_SHA_MISMATCH: x != y",
                                   checkpoint_failure_stage="CHECKPOINT_SHA_COMPARE")
    check("(26d) finalize with checkpoint error -> FAIL + finalized + error + stage recorded",
          fin_err["certificate_status"] == CERTIFICATE_STATUS_FAIL
          and fin_err["certificate_finalized"] is True
          and any("BASE_CHECKPOINT_FAILURE" in e for e in fin_err["validation_errors"])
          and fin_err["checkpoint_identity"]["base_checkpoint_match"] == "FAIL"
          and fin_err["checkpoint_failure_stage"] == "CHECKPOINT_SHA_COMPARE")
    check("(26d') a PASS finalize records checkpoint_failure_stage=NONE",
          fin["checkpoint_failure_stage"] == "NONE")
    for _stage in ("CHECKPOINT_MANAGER_INIT", "CHECKPOINT_RESTORE", "CHECKPOINT_STRUCTURE",
                   "CHECKPOINT_PARAMS_EXTRACTION", "CHECKPOINT_PARAMS_HASH"):
        _f = finalize_certificate(pre, checkpoint_error=f"{_stage}: simulated fault",
                                  checkpoint_failure_stage=_stage)
        check(f"(26d'') stage {_stage} -> FAIL cert carries that exact stage",
              _f["certificate_status"] == CERTIFICATE_STATUS_FAIL
              and _f["certificate_finalized"] is True
              and _f["checkpoint_failure_stage"] == _stage
              and any("BASE_CHECKPOINT_FAILURE" in e for e in _f["validation_errors"]))
    try:
        finalize_certificate(pre, checkpoint_error="x", checkpoint_failure_stage="BOGUS_STAGE")
        check("(26d''') invalid checkpoint_failure_stage -> raised", False, "no raise")
    except ValueError as e:
        check("(26d''') invalid checkpoint_failure_stage -> raised",
              "CHECKPOINT_FAILURE_STAGE_INVALID" in str(e))

    kw_bad = _reference_runtime_kwargs(); kw_bad["seed"] = 43
    pre_bad = build_precheck_certificate(
        _reference_formal_record(), build_runtime_scientific_config(**kw_bad),
        formal_identity_record=fid_pass, assignment_record=ra_pass)
    check("(26e) scientific-mismatch precheck -> FAIL (not PENDING)",
          pre_bad["certificate_status"] == CERTIFICATE_STATUS_FAIL
          and any("FORMAL_CONFIG_RUNTIME_MISMATCH" in e
                  for e in pre_bad["validation_errors"]))
    fin_bad = finalize_certificate(
        pre_bad, verify_checkpoint_params_sha(
            build_checkpoint_identity("/ckpt/17500/x"), EXPECTED_BASE_CHECKPOINT_SHA256))
    check("(26f) a FAIL precheck can NEVER finalize to PASS (stale-PASS guard)",
          fin_bad["certificate_status"] == CERTIFICATE_STATUS_FAIL
          and fin_bad["certificate_finalized"] is True
          and any("CERTIFICATE_NOT_PENDING_AT_FINALIZE" in e
                  for e in fin_bad["validation_errors"]))

    pre_nofid = build_precheck_certificate(
        _reference_formal_record(),
        build_runtime_scientific_config(**_reference_runtime_kwargs()),
        formal_identity_record=None, assignment_record=ra_pass)
    check("(26g) precheck without formal-config identity -> FAIL + identity error",
          pre_nofid["certificate_status"] == CERTIFICATE_STATUS_FAIL
          and any("FORMAL_CONFIG_IDENTITY_MISMATCH" in e
                  for e in pre_nofid["validation_errors"]))

    # (27) §七 atomic write + payload SHA + file SHA + sidecar + tamper detection
    tmp_cert = tempfile.mkdtemp(prefix="p4av23_cert_")
    try:
        cpath = os.path.join(tmp_cert, "runtime_config_certificate.json")
        cpath, spath, fsha, psha, written = write_certificate_atomic(fin, cpath)
        on_disk = json.load(open(cpath, encoding="utf-8"))
        sidecar_line = open(spath, encoding="utf-8").read().split()
        check("(27a) atomic write: payload SHA embedded, sidecar = '<sha>  <basename>'",
              on_disk["certificate_payload_sha256"] == psha
              and sidecar_line == [fsha, "runtime_config_certificate.json"]
              and spath == cpath[:-len(".json")] + ".sha256")
        check("(27a') §五.1: returned written_certificate is byte-exact the disk JSON and "
              "carries the payload SHA (caller MUST adopt it)",
              written == on_disk and written["certificate_payload_sha256"] == psha
              and written["certificate_written_via"] == "atomic_tempfile_fsync_replace")
        check("(27b) no stray temp file remains (atomic replace)",
              not [f for f in os.listdir(tmp_cert) if ".tmp." in f])
        vr = verify_certificate_artifact(cpath)
        check("(27c) untampered certificate -> tamper check PASS",
              vr["certificate_tamper_check"] == "PASS"
              and vr["file_sha256"] == fsha and vr["payload_sha256"] == psha)
        vr2 = verify_certificate_artifact(
            cpath, expected_file_sha256=fsha, expected_payload_sha256=psha)
        check("(27c') expected-SHA verification (summary/manifest binding) -> PASS",
              vr2["certificate_tamper_check"] == "PASS")

        # tamper 1: the REAL threat model — a FAIL certificate flipped to PASS on disk (payload
        # changes; the attacker does NOT recompute the payload SHA). NOTE: the certificate under
        # test must actually be FAIL; flipping an already-PASS cert is a byte no-op and correctly
        # verifies clean (that is not tampering).
        write_certificate_atomic(fin_err, cpath)   # fin_err is FAIL (see 26d)
        on_disk_fail = json.load(open(cpath, encoding="utf-8"))
        assert on_disk_fail["certificate_status"] == "FAIL", "tamper test needs a FAIL cert"
        t1 = dict(on_disk_fail); t1["certificate_status"] = "PASS"; t1["validation_errors"] = []
        atomic_write_json(cpath, t1)
        try:
            verify_certificate_artifact(cpath)
            check("(27d) FAIL->PASS flip without payload recompute -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27d) FAIL->PASS flip without payload recompute -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e)
                  and "payload sha tampered" in str(e))

        # tamper 2: attacker ALSO recomputes + re-embeds the payload SHA -> the file bytes still
        # differ, so the detached sidecar file-SHA no longer matches (second layer catches it).
        t2 = dict(on_disk_fail); t2["certificate_status"] = "PASS"; t2["validation_errors"] = []
        t2["certificate_payload_sha256"] = compute_certificate_payload_sha(t2)
        atomic_write_json(cpath, t2)
        try:
            verify_certificate_artifact(cpath)
            check("(27e) recomputed-payload attack -> sidecar file-SHA TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27e) recomputed-payload attack -> sidecar file-SHA TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e)
                  and "sidecar file sha mismatch" in str(e))

        # restore + tamper 3: delete the sidecar
        write_certificate_atomic(fin, cpath)
        os.remove(certificate_sidecar_path(cpath))
        try:
            verify_certificate_artifact(cpath)
            check("(27f) missing sidecar -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27f) missing sidecar -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e) and "sidecar missing" in str(e))

        # restore + tamper 4: expected file SHA from a stale summary no longer matches
        _, _, fsha_new, psha_new, _ = write_certificate_atomic(fin, cpath)
        try:
            verify_certificate_artifact(cpath, expected_file_sha256="0" * 64)
            check("(27g) expected-file-SHA mismatch -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27g) expected-file-SHA mismatch -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e)
                  and "file sha != expected" in str(e))

        # tamper 5: corrupt the file to invalid JSON
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("{corrupt")
        try:
            verify_certificate_artifact(cpath)
            check("(27h) invalid-JSON certificate -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27h) invalid-JSON certificate -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e))

        # (27i-l) Phase4A-v2.4 (§十): sidecar WHOLE-LINE validation. A first-token-only check
        # would accept a sidecar transplanted from ANOTHER certificate file (correct SHA for a
        # different basename), padded with extra tokens, emptied, or truncated.
        _, sp_sl, fsha_sl, psha_sl, _ = write_certificate_atomic(fin, cpath)
        base_sl = os.path.basename(cpath)
        with open(sp_sl, "w", encoding="utf-8") as f:
            f.write(f"{fsha_sl}  some_other_certificate.json\n")   # right SHA, WRONG basename
        try:
            verify_certificate_artifact(cpath)
            check("(27i) sidecar correct-SHA-but-wrong-basename -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27i) sidecar correct-SHA-but-wrong-basename -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e)
                  and "basename mismatch" in str(e))
        with open(sp_sl, "w", encoding="utf-8") as f:
            f.write(f"{fsha_sl}  {base_sl}  extra_token\n")        # extra tokens
        try:
            verify_certificate_artifact(cpath)
            check("(27j) sidecar extra tokens -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27j) sidecar extra tokens -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e)
                  and "exactly 2 tokens" in str(e))
        with open(sp_sl, "w", encoding="utf-8") as f:
            f.write("")                                            # empty sidecar
        try:
            verify_certificate_artifact(cpath)
            check("(27k) empty sidecar -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27k) empty sidecar -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e)
                  and "exactly 2 tokens" in str(e))
        with open(sp_sl, "w", encoding="utf-8") as f:
            f.write(f"{fsha_sl}  {base_sl}")                       # truncated: no newline
        try:
            verify_certificate_artifact(cpath)
            check("(27l) sidecar missing trailing newline -> TAMPERED", False, "no raise")
        except ValueError as e:
            check("(27l) sidecar missing trailing newline -> TAMPERED",
                  "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" in str(e)
                  and "trailing newline" in str(e))

        # (28) certificate_shas_record carries the §七.3 artifact-binding keys (superset of v2.2)
        _, sp2, fsh2, psh2, _ = write_certificate_atomic(fin, cpath)
        rec28 = certificate_shas_record(
            json.load(open(cpath, encoding="utf-8")),
            certificate_file_sha256=fsh2, certificate_sidecar_path=sp2)
        need = {"formal_config_file_sha256", "scientific_config_sha256",
                "runtime_scientific_config_sha256", "runtime_config_certificate_status",
                "base_checkpoint_expected_sha256", "base_checkpoint_expected_sha256_status",
                "runtime_config_certificate_finalized",
                "runtime_config_certificate_payload_sha256",
                "runtime_config_certificate_file_sha256",
                "runtime_config_certificate_sidecar_path",
                "base_checkpoint_params_sha256", "base_checkpoint_match"}
        check("(28) certificate_shas_record carries v2.2 + §七.3 keys",
              need.issubset(set(rec28))
              and rec28["runtime_config_certificate_file_sha256"] == fsh2
              and rec28["runtime_config_certificate_payload_sha256"] == psh2
              and rec28["base_checkpoint_match"] == "PASS")
    finally:
        shutil.rmtree(tmp_cert, ignore_errors=True)

    # (29) §四.3 legacy in-certificate check: the v2.2 SUFFIX match is gone (strict equality)
    kw = _reference_runtime_kwargs(); rec = _reference_formal_record()
    cert = validate_runtime_against_formal_config(
        rec, build_runtime_scientific_config(**kw), gpu_uuid=_GPU,
        out_dir="some/prefix/runs/RMT16-PERSISTENT-ORIGVTRACE-129")   # ends with formal out_dir
    check("(29) legacy validate: out_dir merely ENDING WITH formal -> FAIL (suffix removed)",
          cert["certificate_status"] == "FAIL" and not cert["runtime_assignment_match"]
          and any("RUNTIME_ASSIGNMENT_MISMATCH" in e for e in cert["validation_errors"]))

    # (30) §五.1 single source of truth: reference kwargs == frozen spec kwargs
    import phase4a_v2_frozen_spec as FSPEC
    check("(30) _reference_runtime_kwargs delegates to frozen spec (no divergence)",
          _reference_runtime_kwargs("persistent") == FSPEC.build_kwargs("persistent")
          and _reference_runtime_kwargs("reset128") == FSPEC.build_kwargs("reset128"))

    n = len(results); n_pass = sum(results)
    print(f"SELF_TEST_SUMMARY total={n} pass={n_pass} fail={n - n_pass}", flush=True)
    print(f"FAIL_CLOSED_NEGATIVE_CASES={n - 1} (>= 19 required by §六.9; v2.3 adds §四/§六/§七 "
          "fail-closed cases)", flush=True)
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
