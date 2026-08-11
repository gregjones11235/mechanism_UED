"""W512 × P2-Replay — canonical contract identity + certificate helpers (CC2 corrected §二).

PURE Python (yaml / json / hashlib / os / inspect only; NO jax / numpy import at module top so it
is safe to import BEFORE `import jax`). This is the W512 analog of the RMT16 Phase4A formal-config
binding — but it is a SEPARATE, self-contained module: it does NOT import or modify the frozen
RMT16 identity machinery (phase4a_v2_frozen_spec / phase4a_v2_formal_identity / phase4a_v2_contract
are arm-specific to the RMT16 persistent/reset128/base_gtrxl arms and are FROZEN; adding W512 to
them would change FROZEN_SPEC_SHA256 and the P/R identities, which is FORBIDDEN). The W512 canonical
candidate is an INDEPENDENT student with its own contract identity, pinned here.

The W512 contract is FIXED by directive W512_RESET128_P2REPLAY_CANONICAL_98304:
    task=DEFEAT_KOBOLD, seed=42, total_env_steps=98304, num_envs=16, num_steps=128,
    network_family=W512, carry_mode=reset128, replay_mode=original_vtrace,
    hindsight=false, awr=false, base params SHA=d4e85af5..., "match Base/RMT16 formal contract
    EXCEPT network capacity".
Every NON-budget scientific constant below equals the frozen RMT16 formal protocol value (PPO /
V-trace / replay / KL gate / EMA / network trunk); the ONLY capacity difference is the W512
raw-history read (long_size=384 + delay_size=128) replacing the RMT16 persistent token.
"""
from __future__ import annotations
import os, json, hashlib, inspect

# ----------------------- canonical frozen scientific constants -----------------------
# These are the SINGLE SOURCE OF TRUTH for the W512 canonical scientific contract. The driver
# materializes the identical values in its Cfg / FullP2Config and the config YAML must match them
# (fail-closed deep compare). They equal the frozen RMT16 formal protocol EXCEPT the network
# capacity block (w512 long_size/delay_size replace rmt_num_tokens).
EXPECTED_BASE_PARAMS_SHA256 = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"

CANONICAL_NETWORK_FAMILY = "W512"
CANONICAL_CARRY_MODE = "reset128"
CANONICAL_REPLAY_MODE = "original_vtrace"
CANONICAL_TASK = "DEFEAT_KOBOLD"
CANONICAL_SEED = 42
CANONICAL_NUM_ENVS = 16
CANONICAL_NUM_STEPS = 128
CANONICAL_SEGMENT_LEN = 128
CANONICAL_SEQUENCE_LENGTH = 129          # crosses one 128-step boundary (MIN_SEQUENCE_LENGTH)
CANONICAL_ANCHOR_INTERVAL = 128
CANONICAL_MIN_SEQUENCE_LENGTH = 129
CANONICAL_REPLAY_BATCH_SIZE = 4
CANONICAL_REPLAY_BUFFER_CAPACITY = 64

# A stable, sorted dict of every NON-budget scientific constant the config YAML must match.
# (Budget = total_updates / save_every / total_env_steps, which differ smoke vs long and are bound
#  separately via the run_management block + run_class.)
CANONICAL_SCIENTIFIC_CONSTANTS = {
    "activation": "relu",
    "adam_eps": 1.0e-5,
    "anchor_interval": 128,
    "awr": False,
    "base_checkpoint": "ckpt17500",
    "carry_mode": "reset128",
    "clip_eps": 0.2,
    "condition_on_task": True,
    "ema_tau": 0.995,
    "embed_size": 256,
    "ent_coef": 0.002,
    "ent_floor": 0.05,
    "gae_lambda": 0.8,
    "gating": True,
    "gating_bias": 2.0,
    "gamma": 0.999,
    "grad_clip": 1.0,
    "hindsight": False,
    "kl_replay_max": 0.05,
    "kl_run_max": 0.1,
    "lr": 2.0e-5,
    "max_grad_norm": 1.0,
    "min_sequence_length": 129,
    "network_family": "W512",
    "num_envs": 16,
    "num_heads": 8,
    "num_layers": 2,
    "num_minibatches": 2,
    "num_steps": 128,
    "optimistic_reset_ratio": 16,
    "qkv_features": 256,
    "replay_batch_size": 4,
    "replay_buffer_capacity": 64,
    "replay_mode": "original_vtrace",
    "rho_bar": 1.0,
    "c_bar": 1.0,
    "seed": 42,
    "segment_len": 128,
    "sequence_length": 129,
    "task": "DEFEAT_KOBOLD",
    "update_epochs": 1,
    "vf_coef": 0.5,
    "vt_clip_min": -50.0,
    "vt_clip_max": 300.0,
    "value_target_clip_min": -50.0,
    "value_target_clip_max": 300.0,
    "w_original_vtrace": 1.0,
    "window_mem": 128,
    "actor_step_scales": [1.0, 0.5, 0.25, 0.125],
    # W512 capacity block (the ONLY difference vs the RMT16 trunk's rmt_num_tokens=16)
    "w512_long_size": 384,
    "w512_delay_size": 128,
    "w512_encoder_size": 256,
}

# Per-run_class budget binding (NON-scientific management layer).
CANONICAL_RUN_BUDGET = {
    "engineering_smoke": {"total_updates": 2, "save_every": 2, "total_env_steps": 4096},
    "long_run_98304": {"total_updates": 48, "save_every": 4, "total_env_steps": 98304},
}
CANONICAL_STEPS_PER_UPDATE = CANONICAL_NUM_ENVS * CANONICAL_NUM_STEPS   # 2048


# ----------------------- hashing helpers -----------------------
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_sha256(fn) -> str:
    """Stable SHA of a function's source (inspect.getsource) — executed-protocol identity."""
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_json_default).encode("utf-8")


def scientific_config_sha256(scientific_config: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(scientific_config)).hexdigest()


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ----------------------- config load + flatten -----------------------
def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_scientific_config(cfg: dict) -> dict:
    """Flatten the YAML scientific_config (with nested ppo/vtrace/network/policy_lag blocks) into
    the flat canonical key space used by CANONICAL_SCIENTIFIC_CONSTANTS. Only the canonical keys
    are extracted; unknown keys are ignored for the compare (but recorded by the caller)."""
    sci = cfg.get("scientific_config", cfg)
    ppo = sci.get("ppo", {}) or {}
    vt = sci.get("vtrace", {}) or {}
    net = sci.get("network", {}) or {}
    flat = {}
    # top-level scalars
    for k in ["carry_mode", "replay_mode", "sequence_length", "segment_len", "hindsight", "awr",
              "w_original_vtrace", "base_checkpoint", "seed", "num_envs", "num_steps", "task",
              "optimistic_reset_ratio", "condition_on_task", "replay_batch_size",
              "replay_buffer_capacity", "anchor_interval", "min_sequence_length",
              "kl_replay_max", "kl_run_max", "actor_step_scales", "ema_tau", "ent_floor",
              "grad_clip", "adam_eps"]:
        if k in sci:
            flat[k] = sci[k]
    # ppo block
    for k in ["lr", "max_grad_norm", "gamma", "gae_lambda", "clip_eps", "vf_coef", "ent_coef",
              "update_epochs", "num_minibatches", "value_target_clip_min",
              "value_target_clip_max"]:
        if k in ppo:
            flat[k] = ppo[k]
    # vtrace block
    for k in ["rho_bar", "c_bar", "vt_clip_min", "vt_clip_max"]:
        if k in vt:
            flat[k] = vt[k]
    # network block
    for k in ["activation", "embed_size", "num_heads", "qkv_features", "num_layers", "gating",
              "gating_bias", "window_mem"]:
        if k in net:
            flat[k] = net[k]
    flat["network_family"] = net.get("network_family", sci.get("network_family", "W512"))
    # W512 capacity block
    if "w512_long_size" in net:
        flat["w512_long_size"] = net["w512_long_size"]
    if "w512_delay_size" in net:
        flat["w512_delay_size"] = net["w512_delay_size"]
    if "w512_encoder_size" in net:
        flat["w512_encoder_size"] = net["w512_encoder_size"]
    return flat


# ----------------------- fail-closed binding -----------------------
def _norm(v):
    """Normalize a scalar/list for comparison (floats compared with tolerance-free exactness after
    float() coercion; lists compared element-wise)."""
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    return v


def diff_scientific_constants(flat_config: dict,
                              expected: dict = None) -> list:
    """Return a list of drift records {path, expected, config} for any canonical constant where the
    YAML config differs from the frozen expectation. Empty list == exact match."""
    expected = expected if expected is not None else CANONICAL_SCIENTIFIC_CONSTANTS
    drift = []
    for k, exp in sorted(expected.items()):
        got = flat_config.get(k, "<MISSING>")
        if _norm(got) != _norm(exp):
            drift.append({"path": k, "expected": exp, "config": got})
    return drift


def validate_runtime_budget(cfg: dict, run_class: str, cli_total_updates: int,
                            cli_save_every: int) -> list:
    """Fail-closed budget binding: the config's run_management block + the CLI budget must match the
    canonical per-run_class budget. Returns drift records (empty == OK)."""
    if run_class not in CANONICAL_RUN_BUDGET:
        return [{"path": "run_class", "expected": sorted(CANONICAL_RUN_BUDGET),
                 "config": run_class}]
    exp = CANONICAL_RUN_BUDGET[run_class]
    rm = cfg.get("run_management", {}) or {}
    drift = []
    pairs = [
        ("run_management.run_class", rm.get("run_class"), run_class),
        ("run_management.total_updates", rm.get("total_updates"), exp["total_updates"]),
        ("run_management.save_every", rm.get("save_every"), exp["save_every"]),
        ("run_management.total_env_steps", rm.get("total_env_steps"), exp["total_env_steps"]),
        ("cli.total_updates", int(cli_total_updates), exp["total_updates"]),
        ("cli.save_every", int(cli_save_every), exp["save_every"]),
    ]
    for path, got, want in pairs:
        if _norm(got) != _norm(want):
            drift.append({"path": path, "expected": want, "config": got})
    return drift


def validate_runtime_assignment(cfg: dict, cli_gpu: str, cli_out: str,
                                run_root: str = None) -> list:
    """Fail-closed hardware/placement binding: the config runtime_assignment gpu_uuid + out_dir must
    match the CLI. If run_root is given, realpath(run_root/out_dir) must equal realpath(cli_out)."""
    ra = cfg.get("runtime_assignment", {}) or {}
    drift = []
    if ra.get("gpu_uuid") != cli_gpu:
        drift.append({"path": "runtime_assignment.gpu_uuid", "expected": ra.get("gpu_uuid"),
                      "config": cli_gpu})
    out_dir = ra.get("out_dir")
    if run_root is not None:
        expected_out = os.path.realpath(os.path.join(run_root, str(out_dir)))
        actual_out = os.path.realpath(cli_out)
        if expected_out != actual_out:
            drift.append({"path": "runtime_assignment.out_dir(realpath)",
                          "expected": expected_out, "config": actual_out})
    return drift


def build_config_identity(config_path: str, cfg: dict) -> dict:
    """Compute the W512 config identity record: canonical path, file SHA, scientific SHA, the flat
    scientific config, and the fail-closed drift vs the frozen canonical constants (empty == PASS)."""
    flat = flatten_scientific_config(cfg)
    drift = diff_scientific_constants(flat)
    return dict(
        config_path=os.path.realpath(config_path),
        config_file_sha256=file_sha256(config_path),
        scientific_config_sha256=scientific_config_sha256(flat),
        scientific_config_flat=flat,
        canonical_constants_sha256=scientific_config_sha256(CANONICAL_SCIENTIFIC_CONSTANTS),
        scientific_constants_drift=drift,
        scientific_constants_match=bool(not drift),
    )


# ----------------------- certificate -----------------------
CERTIFICATE_STATUS_PENDING = "PENDING_CHECKPOINT_IDENTITY"
CERTIFICATE_STATUS_PASS = "PASS"
CERTIFICATE_STATUS_FAIL = "FAIL"


def build_precheck_certificate(config_identity: dict, budget_drift: list,
                               assignment_drift: list, run_class: str,
                               ckpt17500: str, cli_args: dict) -> dict:
    """Build the provisional PENDING certificate (before the base checkpoint SHA is verified).
    Any config/budget/assignment drift flips it straight to FAIL (fail closed)."""
    errors = []
    if not config_identity["scientific_constants_match"]:
        errors.append("SCIENTIFIC_CONSTANTS_MISMATCH: "
                      + " | ".join(f"{d['path']}: expected={d['expected']!r} config={d['config']!r}"
                                   for d in config_identity["scientific_constants_drift"]))
    if budget_drift:
        errors.append("BUDGET_BINDING_MISMATCH: "
                      + " | ".join(f"{d['path']}: expected={d['expected']!r} config={d['config']!r}"
                                   for d in budget_drift))
    if assignment_drift:
        errors.append("RUNTIME_ASSIGNMENT_MISMATCH: "
                      + " | ".join(f"{d['path']}: expected={d['expected']!r} config={d['config']!r}"
                                   for d in assignment_drift))
    status = CERTIFICATE_STATUS_PENDING if not errors else CERTIFICATE_STATUS_FAIL
    return dict(
        certificate_status=status,
        certificate_finalized=False,
        candidate_id="W512_RESET128_P2REPLAY_CANONICAL_98304",
        network_family=CANONICAL_NETWORK_FAMILY,
        carry_mode=CANONICAL_CARRY_MODE,
        replay_mode=CANONICAL_REPLAY_MODE,
        hindsight=False,
        awr=False,
        run_class=run_class,
        interruption_policy="RESTART_FROM_STEP0",
        base_checkpoint_expected_sha256=EXPECTED_BASE_PARAMS_SHA256,
        base_checkpoint_path=ckpt17500,
        config_identity=config_identity,
        cli_args={k: v for k, v in cli_args.items()},
        checkpoint_identity=dict(
            base_checkpoint_expected_sha256=EXPECTED_BASE_PARAMS_SHA256,
            base_checkpoint_expected_sha256_status="FROZEN",
            base_checkpoint_loaded_sha256=None,
            base_checkpoint_match=None),
        validation_errors=errors,
    )


def finalize_certificate(cert: dict, loaded_base_sha: str | None,
                         checkpoint_error: str | None = None) -> dict:
    """Finalize: PENDING + loaded SHA == frozen expectation -> PASS; else FAIL. Records the loaded
    SHA + match flag. Idempotent-ish: overwrites status/finalized/errors."""
    cert = dict(cert)
    ci = dict(cert["checkpoint_identity"])
    ci["base_checkpoint_loaded_sha256"] = loaded_base_sha
    match = bool(loaded_base_sha == EXPECTED_BASE_PARAMS_SHA256) if loaded_base_sha else False
    ci["base_checkpoint_match"] = match
    cert["checkpoint_identity"] = ci
    cert["certificate_finalized"] = True
    errors = list(cert.get("validation_errors") or [])
    if checkpoint_error is not None:
        errors.append("CHECKPOINT_FAILURE: " + checkpoint_error)
    elif not match:
        errors.append(
            f"BASE_CHECKPOINT_SHA_MISMATCH: loaded={loaded_base_sha} "
            f"expected={EXPECTED_BASE_PARAMS_SHA256}")
    if cert["certificate_status"] == CERTIFICATE_STATUS_FAIL:
        pass   # pre-existing precheck failure stays FAIL
    elif errors:
        cert["certificate_status"] = CERTIFICATE_STATUS_FAIL
    else:
        cert["certificate_status"] = CERTIFICATE_STATUS_PASS
    cert["validation_errors"] = errors
    return cert


def write_certificate_atomic(cert: dict, path: str) -> tuple:
    """Atomic write (tmp + os.replace). Returns (path, payload_sha256, file_sha256)."""
    payload = canonical_json_bytes(cert)
    payload_sha = hashlib.sha256(payload).hexdigest()
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path, payload_sha, file_sha256(path)
