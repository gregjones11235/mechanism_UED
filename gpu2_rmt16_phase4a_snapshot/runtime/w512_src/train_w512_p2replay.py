#!/usr/bin/env python3
"""W512 × P2-Replay — CANONICAL training driver (CC2 corrected §二).

Candidate: W512_RESET128_P2REPLAY_CANONICAL_98304. This driver MIRRORS the frozen
`train_rmt16_p2replay.py` training contract — same DEFEAT_KOBOLD task, same base checkpoint
(ckpt17500, params SHA d4e85af5...), same frozen PPO + Original-goal V-trace Replay protocol, same
cadence / KL gate / EMA / conservation / firewall — EXCEPT the network capacity: the RMT16
persistent token is replaced by the W512 raw-history read (384 long ring + 128 delay line; see
network_w512.py / w512_memory.py). The replay STACK (V-trace loss + transactional KL gate + EMA +
eligible-only sampler + P2 anchor conservation) is the FROZEN RMT/P2 machinery reused verbatim by
the w512_src modules; only reconstruction + the loss-region scan are W512-specific.

FIXED contract (fail-closed; any CLI/config deviation aborts BEFORE `import jax`):
    network_family=W512  carry_mode=reset128  replay_mode=original_vtrace
    hindsight=false  awr=false  task=DEFEAT_KOBOLD  seed=42  num_envs=16  num_steps=128
    sequence_length=129  (match Base/RMT16 formal contract EXCEPT network capacity)

Per outer update:
  1. collect_rollout_w512   (16 env x 128 steps; emits complete episodes + sparse GTrXL/W512 anchors)
  2. PPO MAIN update        (frozen hyperparams; w512_ppo.ppo_update_w512)
  3. REPLAY update          (sample_eligible(129, rng, 4) -> original_vtrace_update_w512;
                             transactional KL gate <=0.05 with actor step-scale retry + rollback; EMA)
  4. checkpoints at step 0/8192/.../98304 (long: save_every=4, 13 ckpts) or 0/4096 (smoke).

NO Hindsight / NO AWR (firewall asserted == 0 for the whole run). GPU2 ONLY (UUID pinned by config).
Interruption -> RESTART_FROM_STEP0 (no resume; this driver never resumes).
"""
import os, sys, json, time, hashlib, pickle, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt17500", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu_uuid", required=True)
ap.add_argument("--w512_config", required=True,
                help="path to the canonical W512 YAML (configs/w512_phase4a_<run>_canonical.yaml)")
ap.add_argument("--snapshot_root", default=None,
                help="root of the frozen experiment snapshot; pins the canonical config PATH.")
ap.add_argument("--run_root", default=None,
                help="root under which runtime_assignment.out_dir (relative) must resolve.")
ap.add_argument("--run_class", required=True,
                choices=["engineering_smoke", "long_run_98304"])
ap.add_argument("--total_updates", type=int, required=True)   # smoke=2, long=48
ap.add_argument("--save_every", type=int, required=True)      # smoke=2, long=4
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sequence_length", type=int, default=129)
args = ap.parse_args()

# ----------------------- FIXED contract pins (directive W512_RESET128_P2REPLAY_CANONICAL_98304) ----
# These are NOT CLI choices: the canonical contract is fixed. The config YAML + CLI budget are
# bound fail-closed against them PRE-JAX (before `import jax` / CUDA env / env build / ckpt load).
import w512_formal_identity as WID    # noqa: E402  (pure: stdlib + yaml only)

NETWORK_FAMILY = WID.CANONICAL_NETWORK_FAMILY          # "W512"
CARRY_MODE = WID.CANONICAL_CARRY_MODE                  # "reset128"
REPLAY_MODE = WID.CANONICAL_REPLAY_MODE                # "original_vtrace"
ARM = "W512-Reset128-OrigVtrace"

# ----------------------- PRE-JAX identity + binding + provisional certificate -----------------------
RUNTIME_CONFIG_CERTIFICATE = None
RUNTIME_CONFIG_CERTIFICATE_PATH = None
RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256 = None
RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256 = None
CONFIG_IDENTITY = None
try:
    _cfg = WID.load_yaml_config(args.w512_config)
    # canonical config PATH identity (anti-copy): realpath must live under snapshot_root/configs.
    if args.snapshot_root is not None:
        _real = os.path.realpath(args.w512_config)
        _allowed_dir = os.path.realpath(os.path.join(args.snapshot_root, "configs"))
        if os.path.dirname(_real) != _allowed_dir:
            raise ValueError(
                f"FORMAL_CONFIG_PATH_MISMATCH: realpath({args.w512_config})={_real} not in "
                f"{_allowed_dir} (byte copies / symlink escapes rejected).")
    CONFIG_IDENTITY = WID.build_config_identity(args.w512_config, _cfg)
    _budget_drift = WID.validate_runtime_budget(
        _cfg, args.run_class, args.total_updates, args.save_every)
    _assign_drift = WID.validate_runtime_assignment(
        _cfg, args.gpu_uuid, args.out, run_root=args.run_root)
    RUNTIME_CONFIG_CERTIFICATE = WID.build_precheck_certificate(
        CONFIG_IDENTITY, _budget_drift, _assign_drift, args.run_class,
        args.ckpt17500, cli_args={k: v for k, v in vars(args).items()})
    if RUNTIME_CONFIG_CERTIFICATE["certificate_status"] != WID.CERTIFICATE_STATUS_PENDING:
        raise ValueError(
            "W512_FORMAL_CONFIG_RUNTIME_MISMATCH: precheck certificate_status="
            + RUNTIME_CONFIG_CERTIFICATE["certificate_status"] + "; no `import jax` / env build / "
            "ckpt load will proceed. errors: "
            + " | ".join(RUNTIME_CONFIG_CERTIFICATE["validation_errors"]))
    (RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
     RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256) = WID.write_certificate_atomic(
        RUNTIME_CONFIG_CERTIFICATE, os.path.join(args.out, "runtime_config_certificate.json"))
    print(f"[w512-formal] precheck certificate_status="
          f"{RUNTIME_CONFIG_CERTIFICATE['certificate_status']} "
          f"scientific_config_sha={CONFIG_IDENTITY['scientific_config_sha256'][:16]} "
          f"config_file_sha={CONFIG_IDENTITY['config_file_sha256'][:16]} "
          f"canonical_sha={CONFIG_IDENTITY['canonical_constants_sha256'][:16]} "
          f"path={RUNTIME_CONFIG_CERTIFICATE_PATH}", flush=True)
except ValueError as e:
    raise SystemExit(str(e))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.dirname(HERE)
FROZEN = os.path.join(RUNTIME, "frozen_modules")
EXPERIMENT = os.path.join(RUNTIME, "experiment_src")
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in [HERE, FROZEN, EXPERIMENT, V7 + "/src", V7]:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

import jax, jax.numpy as jnp, numpy as np, optax
import orbax.checkpoint as ocp
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

from network_w512 import ActorCriticTransformerW512
import w512_memory as w5m
import w512_memory_anchor as WA
import rng_utils as RU
import w512_collect as WC
import w512_ppo as WP
import w512_replay_learner as WRL
import w512_replay_buffer as WRB
from full_p2_learner import FullP2Config, build_optimizer
# frozen replay-protocol constants + firewall counters (network-agnostic; reused unchanged)
from replay_buffer import ANCHOR_INTERVAL, MIN_SEQUENCE_LENGTH
from rmt_replay_learner import W_ORIGINAL_VTRACE
from phase4a_v2_counters import Phase4ACounters

SEQUENCE_LENGTH = int(args.sequence_length)
SEGMENT_LEN = WID.CANONICAL_SEGMENT_LEN                      # 128
if SEQUENCE_LENGTH <= SEGMENT_LEN:
    raise SystemExit(f"FATAL: replay_mode=original_vtrace requires sequence_length > {SEGMENT_LEN} "
                     f"(got {SEQUENCE_LENGTH}); the canonical experiment crosses one boundary.")
K_BATCH = WID.CANONICAL_REPLAY_BATCH_SIZE                    # 4
STEPS_PER_UPDATE = WID.CANONICAL_NUM_ENVS * WID.CANONICAL_NUM_STEPS   # 2048

# ----------------------- config (bakeoff frozen + P2 frozen + W512 capacity) -----------------------
class Cfg:
    activation = "relu"; embed_size = 256; hidden_layers = 256; num_heads = 8; qkv_features = 256
    num_layers = 2; gating = True; gating_bias = 2.0; window_mem = 128; window_grad = 64
    lr = 2e-5; max_grad_norm = 1.0; gamma = 0.999; gae_lambda = 0.8; clip_eps = 0.2; vf_coef = 0.5
    ent_coef = 0.002; update_epochs = 1; num_minibatches = 2; num_envs = 16; num_steps = 128
    optimistic_reset_ratio = 16; condition_on_task = True
    value_target_clip_min = -50.0; value_target_clip_max = 300.0
    # W512 capacity (the ONLY difference vs the RMT16 trunk's rmt_num_tokens)
    w512_long_size = 384; w512_delay_size = 128; w512_encoder_size = 256
cfg = Cfg()
ppo_cfg = dict(window_mem=cfg.window_mem, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
               embed=cfg.embed_size, lr=cfg.lr, max_grad_norm=cfg.max_grad_norm,
               gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_eps=cfg.clip_eps,
               vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef, update_epochs=cfg.update_epochs,
               num_minibatches=cfg.num_minibatches)
fp_cfg = FullP2Config(window_mem=cfg.window_mem, num_heads=cfg.num_heads,
                      num_layers=cfg.num_layers, embed=cfg.embed_size,
                      gamma=cfg.gamma, vt_clip_min=cfg.value_target_clip_min,
                      vt_clip_max=cfg.value_target_clip_max)
w5_cfg = w5m.W512Config(long_size=cfg.w512_long_size, delay_size=cfg.w512_delay_size,
                        encoder_size=cfg.w512_encoder_size)

# ----------------------- POST-JAX imported-constants binding (fail closed) -----------------------
# Rebuild the flat scientific config from the REAL imported objects and diff vs the frozen canonical
# constants. Any drift => finalize certificate FAIL + exit. With W512_POSTJAX_BINDING_SELFTEST=1 the
# driver exits HERE (CPU regression hook; no env build / ckpt load / training). Launchers never set it.
def _imported_flat_scientific():
    flat = dict(
        activation=cfg.activation, embed_size=int(cfg.embed_size), num_heads=int(cfg.num_heads),
        qkv_features=int(cfg.qkv_features), num_layers=int(cfg.num_layers),
        gating=bool(cfg.gating), gating_bias=float(cfg.gating_bias),
        window_mem=int(cfg.window_mem), num_envs=int(cfg.num_envs), num_steps=int(cfg.num_steps),
        optimistic_reset_ratio=int(cfg.optimistic_reset_ratio),
        condition_on_task=bool(cfg.condition_on_task),
        lr=float(cfg.lr), max_grad_norm=float(cfg.max_grad_norm), gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda), clip_eps=float(cfg.clip_eps), vf_coef=float(cfg.vf_coef),
        ent_coef=float(cfg.ent_coef), update_epochs=int(cfg.update_epochs),
        num_minibatches=int(cfg.num_minibatches),
        value_target_clip_min=float(cfg.value_target_clip_min),
        value_target_clip_max=float(cfg.value_target_clip_max),
        rho_bar=float(fp_cfg.rho_bar), c_bar=float(fp_cfg.c_bar),
        vt_clip_min=float(fp_cfg.vt_clip_min), vt_clip_max=float(fp_cfg.vt_clip_max),
        kl_replay_max=float(fp_cfg.kl_replay_max), kl_run_max=float(fp_cfg.kl_run_max),
        actor_step_scales=list(fp_cfg.actor_step_scales), ema_tau=float(fp_cfg.ema_tau),
        ent_floor=float(fp_cfg.ent_floor), grad_clip=float(fp_cfg.grad_clip),
        adam_eps=float(fp_cfg.adam_eps),
        network_family=NETWORK_FAMILY, carry_mode=CARRY_MODE, replay_mode=REPLAY_MODE,
        sequence_length=SEQUENCE_LENGTH, segment_len=SEGMENT_LEN, hindsight=False, awr=False,
        w_original_vtrace=float(W_ORIGINAL_VTRACE), base_checkpoint="ckpt17500",
        seed=int(args.seed), task=WID.CANONICAL_TASK,
        replay_batch_size=int(K_BATCH), replay_buffer_capacity=int(WID.CANONICAL_REPLAY_BUFFER_CAPACITY),
        anchor_interval=int(ANCHOR_INTERVAL), min_sequence_length=int(MIN_SEQUENCE_LENGTH),
        w512_long_size=int(cfg.w512_long_size), w512_delay_size=int(cfg.w512_delay_size),
        w512_encoder_size=int(cfg.w512_encoder_size))
    return flat

_IMPORTED_DRIFT = WID.diff_scientific_constants(_imported_flat_scientific())
if os.environ.get("W512_POSTJAX_BINDING_SELFTEST") == "1":
    if _IMPORTED_DRIFT:
        print("W512_POSTJAX_BINDING_SELFTEST=FAIL run_class=" + str(args.run_class) + " drift: "
              + " | ".join(f"{d['path']}: expected={d['expected']!r} imported={d['config']!r}"
                           for d in _IMPORTED_DRIFT), flush=True)
        raise SystemExit(3)
    print("W512_POSTJAX_BINDING_SELFTEST=PASS imported_constants_match=True run_class="
          + str(args.run_class) + " total_updates=" + str(int(args.total_updates))
          + " save_every=" + str(int(args.save_every)), flush=True)
    raise SystemExit(0)
if _IMPORTED_DRIFT:
    _msg = " | ".join(f"{d['path']}: expected={d['expected']!r} imported={d['config']!r}"
                      for d in _IMPORTED_DRIFT)
    RUNTIME_CONFIG_CERTIFICATE = WID.finalize_certificate(
        RUNTIME_CONFIG_CERTIFICATE, loaded_base_sha=None,
        checkpoint_error="IMPORTED_RUNTIME_CONSTANTS_MISMATCH: " + _msg)
    (RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
     RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256) = WID.write_certificate_atomic(
        RUNTIME_CONFIG_CERTIFICATE, RUNTIME_CONFIG_CERTIFICATE_PATH)
    raise SystemExit("IMPORTED_RUNTIME_CONSTANTS_MISMATCH: imported runtime constants drifted from "
                     "the frozen W512 canonical spec; no env build / ckpt load / training. drift: "
                     + _msg)

# executed replay-protocol SOURCE identity (inspect; not string labels)
EXECUTED_PROTOCOL_IDENTITY = dict(
    learner_qualname=getattr(WRL.original_vtrace_update_w512, "__qualname__",
                             str(WRL.original_vtrace_update_w512)),
    learner_source_sha256=WID.source_sha256(WRL.original_vtrace_update_w512),
    loss_source_sha256=WID.source_sha256(WRL.compute_loss_original_vtrace_rmt),
    sampler_qualname=getattr(WRB.W512ReplayBuffer.sample_eligible, "__qualname__",
                             str(WRB.W512ReplayBuffer.sample_eligible)),
    sampler_source_sha256=WID.source_sha256(WRB.W512ReplayBuffer.sample_eligible),
    rng_class="numpy.random.RandomState")
RUNTIME_CONFIG_CERTIFICATE["executed_protocol_identity"] = EXECUTED_PROTOCOL_IDENTITY
RUNTIME_CONFIG_CERTIFICATE["imported_scientific_config_sha256"] = (
    WID.scientific_config_sha256(_imported_flat_scientific()))

# dedicated deterministic replay-sampling RNG (reused by training; independent of rollout/action RNG)
replay_sample_rng = np.random.RandomState(args.seed + 7)

# ----------------------- DEFEAT_KOBOLD task (identical to the frozen bakeoff) -----------------------
S4_TASK_CODE = '''
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams(needs_depletion_multiplier=0.3)
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        b.set_starting_floor(2); b.set_monsters_killed(2,8)
        b.set_player_inventory({"wood":7,"stone":27,"coal":3,"iron":3,"sapling":1,"pickaxe":3,"sword":3,"bow":1,"arrows":7,"torches":10})
        s=b.build(rng); up=b.ladders_up[2]
        return s.replace(item_map=s.item_map.at[2,up[0],up[1]].set(ItemType.NONE.value))
'''
ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
target_achievement_1d = np.asarray(table[0]).astype(np.float32)

# ----------------------- helpers -----------------------
def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()

def _to_np(pt):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), pt)

def _merge(base, full):
    if isinstance(base, dict) and isinstance(full, dict):
        out = dict(full)
        for k in base:
            if k in full:
                out[k] = _merge(base[k], full[k])
        return out
    return base

# ----------------------- base checkpoint load + staged fail-closed identity -----------------------
base_inner = None
base_sha = None
_CHECKPOINT_ERROR = None
try:
    t0 = time.time()
    ckpt_mgr = ocp.CheckpointManager(os.path.dirname(args.ckpt17500))
    raw = ckpt_mgr.restore(int(os.path.basename(args.ckpt17500)))
    base_inner = raw["params"]["params"]
    base_sha = _params_sha(base_inner)
    print(f"[load] ckpt17500 leaves={len(jax.tree_util.tree_leaves(base_inner))} "
          f"sha={base_sha[:16]} ({time.time()-t0:.1f}s)", flush=True)
except Exception as exc:
    _CHECKPOINT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[load] ckpt17500 FAILED error={_CHECKPOINT_ERROR}", flush=True)

RUNTIME_CONFIG_CERTIFICATE = WID.finalize_certificate(
    RUNTIME_CONFIG_CERTIFICATE, loaded_base_sha=base_sha, checkpoint_error=_CHECKPOINT_ERROR)
(RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
 RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256) = WID.write_certificate_atomic(
    RUNTIME_CONFIG_CERTIFICATE, RUNTIME_CONFIG_CERTIFICATE_PATH)
print(f"[w512-formal] FINAL certificate_status="
      f"{RUNTIME_CONFIG_CERTIFICATE['certificate_status']} "
      f"base_checkpoint_match="
      f"{RUNTIME_CONFIG_CERTIFICATE['checkpoint_identity']['base_checkpoint_match']} "
      f"expected={WID.EXPECTED_BASE_PARAMS_SHA256[:16]} "
      f"loaded={base_sha[:16] if base_sha else None} "
      f"certificate_payload_sha={RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256[:16]}", flush=True)
if RUNTIME_CONFIG_CERTIFICATE["certificate_status"] != WID.CERTIFICATE_STATUS_PASS:
    raise SystemExit(
        "W512_FORMAL_CONFIG_RUNTIME_MISMATCH: certificate_status=FAIL; no training step will "
        "proceed. errors: " + " | ".join(RUNTIME_CONFIG_CERTIFICATE["validation_errors"]))

# ----------------------- env + network + compat init -----------------------
print("=" * 78, flush=True)
print(f"{ARM}  driver  (W512 canonical)", flush=True)
print(f"  carry_mode={CARRY_MODE} replay_mode={REPLAY_MODE} sequence_length={SEQUENCE_LENGTH} "
      f"run_class={args.run_class} gpu={args.gpu_uuid}", flush=True)
print(f"  devices={[str(d) for d in jax.devices()]}", flush=True)
print("=" * 78, flush=True)

base_env = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=EMB)
env = DistributedMultiTaskOptimisticLogWrapper(
    base_env, jax.random.PRNGKey(0), cfg.num_envs, 1, cfg.optimistic_reset_ratio,
    jnp.array([1.0]), table, probe_term=False)
env_params = env.default_params
assert env_params is not None, "env_params must resolve before collect/env.step"
ACTION_DIM = int(env.action_space(env_params).n)
OBS_DIM = int(env.observation_space(env_params).shape[0])
OBS_SHAPE = tuple(int(x) for x in env.observation_space(env_params).shape)
fp_cfg.action_dim = ACTION_DIM; fp_cfg.obs_dim = OBS_DIM

network = ActorCriticTransformerW512(
    action_dim=ACTION_DIM, activation=cfg.activation, hidden_layers=cfg.hidden_layers,
    encoder_size=cfg.embed_size, num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
    long_size=cfg.w512_long_size)
apply_eval_w512 = WA.make_apply_eval_w512(network)

# compat init from ckpt17500 (base params loaded; W512 params fresh, additive -> base SHA preserved)
rng_init = jax.random.PRNGKey(args.seed); rng_init, _rng = jax.random.split(rng_init)
_full = network.init(
    _rng,
    jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed_size)),
    jnp.zeros((2, OBS_DIM)),
    jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), jnp.bool_),
    long_buf=jnp.zeros((2, cfg.w512_long_size, cfg.embed_size)),
    long_mask=jnp.zeros((2, cfg.w512_long_size), jnp.bool_))
full_inner = _full["params"]
params = _merge(base_inner, full_inner)            # INNER: ckpt17500 base weights + fresh W512
_base_keys = [k for k in full_inner.keys() if not str(k).startswith("w512_")]
_merged_base_sha = _params_sha({k: params[k] for k in _base_keys if k in params})
assert _merged_base_sha == base_sha == WID.EXPECTED_BASE_PARAMS_SHA256, (
    f"W512 merge perturbed base params: merged={_merged_base_sha[:16]} base={base_sha[:16]}")
print(f"[merge] inner_leaves={len(jax.tree_util.tree_leaves(params))} base_sha={base_sha[:16]} "
      f"merged_base_sha={_merged_base_sha[:16]} (W512 additive -> base bit-exact)", flush=True)

# ----------------------- optimizers / EMA / buffers / state -----------------------
ppo_opt = WP.build_ppo_optimizer(ppo_cfg)
replay_opt = build_optimizer(cfg.lr, fp_cfg)
params = jax.tree_util.tree_map(jnp.asarray, params)
ppo_opt_state = ppo_opt.init(params)
replay_opt_state = replay_opt.init(params)
target_params = params                                  # EMA target init = online (ckpt17500)

replay = WRB.W512ReplayBuffer(capacity=WID.CANONICAL_REPLAY_BUFFER_CAPACITY, seed=args.seed)
pending = WC.W512PendingEpisodeBuffers(cfg.num_envs, first_episode_id=0, first_policy_version=0)
counters = Phase4ACounters()

rng = jax.random.PRNGKey(args.seed + 1)
rng, _rng = jax.random.split(rng)
obsv, env_state = env.reset(_rng, env_params)
memories = jnp.zeros((cfg.num_envs, cfg.window_mem, cfg.num_layers, cfg.embed_size))
mem_mask = jnp.zeros((cfg.num_envs, cfg.num_heads, 1, cfg.window_mem + 1), jnp.bool_)
mem_idx = jnp.full((cfg.num_envs,), cfg.window_mem, jnp.int32)
w512_state = WA.w512_fresh_state(cfg.num_envs, w5_cfg)
done_enter = jnp.zeros((cfg.num_envs,), jnp.bool_)
action_rng = RU.make_action_rng(args.seed)

scan_fn = WRL.make_scan_w512_loss(network, fp_cfg, w5_cfg, SEGMENT_LEN)

CKPT_DIR = os.path.join(args.out, "ckpt"); LOG_DIR = os.path.join(args.out, "out")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"{ARM}_train.jsonl")
mon_path = os.path.join(LOG_DIR, "W512_MONITOR.jsonl")

# ----------------------- W512 correctness probes (READ-ONLY; no training impact) -----------------------
def _leaf_norm_w512(subtree):
    return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(v))))
        for v in jax.tree_util.tree_leaves(subtree))))

def _w512_read_monitor(p, mem, obs, mask_adv, long_buf, long_mask):
    """READ-ONLY W512 read-path CONNECTIVITY + WIRING probe (analog of the RMT16 forced-open
    read-branch connectivity gate). The W512 read is z_t = h_t + tanh(w512_gate) * long_ctx_t with
    w512_gate ZERO-INITIALISED (so step0 == base GTrXL bit-exactly, by design); at init / early
    training the carried long buffer therefore has ZERO forward effect and a plain forward diff is
    uninformative. So we (a) FORCE the gate OPEN on a READ-ONLY param copy (tanh(1) != 0) and inject
    a synthetic NON-ZERO 384 long buffer, comparing logits vs a zeroed buffer (forward connectivity),
    AND (b) take grad(logits.mean()+value.mean()) at the forced-open gate to prove the W512 read
    params (w512_cross_attn / w512_ln) and the gate are WIRED to the output. Does NOT modify the
    training params / optimizer / network."""
    inj = jnp.ones_like(long_buf)                       # synthetic NON-ZERO long buffer
    inj_mask = jnp.ones_like(long_mask)                 # all slots valid
    zeros = jnp.zeros_like(inj); zmask = jnp.zeros_like(inj_mask)
    p_open = {**p, "w512_gate": jnp.ones_like(jnp.asarray(p["w512_gate"]))}   # forced-open copy
    lg_on, _v1, _m1, _h1 = apply_eval_w512(p_open, mem, obs, mask_adv, inj, inj_mask)
    lg_off, _v2, _m2, _h2 = apply_eval_w512(p_open, mem, obs, mask_adv, zeros, zmask)
    lon = jnp.asarray(lg_on); loff = jnp.asarray(lg_off)
    diff = float(jnp.max(jnp.abs(lon - loff)))
    pon = jax.nn.softmax(lon, axis=-1); poff = jax.nn.softmax(loff, axis=-1)
    kl = float(jnp.mean(jnp.sum(pon * (jnp.log(pon + 1e-12) - jnp.log(poff + 1e-12)), axis=-1)))
    top_changed = float(jnp.mean(jnp.asarray(jnp.argmax(lon, axis=-1) != jnp.argmax(loff, axis=-1))))
    def _loss(pp):
        lg, vl, _mo, _ht = apply_eval_w512(pp, mem, obs, mask_adv, inj, inj_mask)
        return jnp.asarray(lg).mean() + jnp.asarray(vl).mean()
    g = jax.grad(_loss)(p_open)
    return dict(w512_read_logit_diff=diff, w512_read_KL=kl, w512_read_top_action_frac=top_changed,
                w512_cross_attn_grad=_leaf_norm_w512(g["w512_cross_attn"]),
                w512_ln_grad=_leaf_norm_w512(g["w512_ln"]),
                w512_gate_grad=float(np.abs(np.asarray(g["w512_gate"])).max()))

def _reset128_probe():
    """READ-ONLY unit probe on w512_reset128_clear: a nonzero long buffer at episode-local
    seg_step=128 (a boundary) is CLEARED; at seg_step=5 (mid-segment) it is PRESERVED. Positive
    evidence the reset128 boundary clear executes correctly."""
    try:
        st = WA.w512_fresh_state(2, w5_cfg)
        st = {**st,
              "long_buf": jnp.ones((2, cfg.w512_long_size, cfg.w512_encoder_size), jnp.float32),
              "long_mask": jnp.ones((2, cfg.w512_long_size), jnp.bool_),
              "seg_step": jnp.array([SEGMENT_LEN, 5], jnp.int32)}
        clr = WA.w512_reset128_clear(st, SEGMENT_LEN)
        lb = np.asarray(clr["long_buf"]); lm = np.asarray(clr["long_mask"])
        cleared = bool(np.all(lb[0] == 0.0) and not np.any(lm[0]))
        preserved = bool(np.all(lb[1] == 1.0) and np.all(lm[1]))
        return dict(w512_reset128_clear_at_boundary=cleared,
                    w512_reset128_preserve_mid_segment=preserved,
                    w512_reset128_probe_ok=bool(cleared and preserved))
    except Exception as exc:
        return dict(w512_reset128_probe_ok=False, w512_reset128_probe_error=repr(exc))

RESET128_PROBE = _reset128_probe()
print(f"[w512-probe] reset128 {RESET128_PROBE}", flush=True)

# ----------------------- training-loop instrumentation -----------------------
update_count = 0
accepted_policy_updates = 0; kl_rejected_updates = 0
online_ppo_update_count = 0
replay_not_ready_skip_count = 0; replay_sample_success_count = 0; replay_update_success_count = 0
replay_first_success_update = None; replay_sequences_consumed = 0
gtrxl_window_finite_all = True
# W512 read-path connectivity is established ONLY by the synthetic-injection monitor in the loop
# (NOT seeded): reset128 clearing correctness is gated separately by RESET128_CLEAR_OK.
w512_read_ever_nonzero = False
replay_attempt_mask = []; replay_update_outer_updates = []; replay_batch_sizes = []
replay_sequence_lengths = []; eligible_count_by_outer_update = []

def _replay_stats(buf):
    lengths = [int(t.length) for t in buf]
    return dict(replay_buffer_trajectory_count=len(lengths),
                replay_max_trajectory_length=(max(lengths) if lengths else 0),
                replay_eligible_count_129=sum(1 for L in lengths if L >= 129),
                replay_eligible_count_512=sum(1 for L in lengths if L >= 512))

# ----------------------- checkpoint -----------------------
def _manifest_fields():
    return dict(
        candidate_id="W512_RESET128_P2REPLAY_CANONICAL_98304",
        network_family=NETWORK_FAMILY, carry_mode=CARRY_MODE, replay_mode=REPLAY_MODE,
        hindsight=False, awr=False, w_original_vtrace=float(W_ORIGINAL_VTRACE),
        sequence_length=SEQUENCE_LENGTH, segment_len=SEGMENT_LEN,
        crosses_boundary=bool(SEQUENCE_LENGTH > SEGMENT_LEN),
        anchor_interval=int(ANCHOR_INTERVAL), min_sequence_length=int(MIN_SEQUENCE_LENGTH),
        replay_batch_size=int(K_BATCH), run_class=args.run_class,
        interruption_policy="RESTART_FROM_STEP0",
        observation_shape=list(OBS_SHAPE), action_dim=ACTION_DIM,
        executed_protocol_identity=EXECUTED_PROTOCOL_IDENTITY,
        runtime_config_certificate_payload_sha256=RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
        runtime_config_certificate_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
        base_checkpoint_params_sha256=base_sha)

def save_ckpt(step, params, ppo_opt_state, replay_opt_state, target_params, tag):
    d = os.path.join(CKPT_DIR, str(step)); os.makedirs(d, exist_ok=True)
    p_sha = _params_sha(params)
    with open(os.path.join(d, "full_state.pkl"), "wb") as f:
        pickle.dump({"params": _to_np(params),
                     "manifest": {"params_sha256": p_sha, "step": step, "arm": ARM,
                                  "carry_mode": CARRY_MODE, "replay_mode": REPLAY_MODE,
                                  "network_family": NETWORK_FAMILY,
                                  "gpu_uuid": args.gpu_uuid, "seed": args.seed,
                                  "config": {k: v for k, v in vars(cfg).items()},
                                  "w512": _manifest_fields(), "tag": tag}}, f, protocol=4)
    with open(os.path.join(d, "train_state.pkl"), "wb") as f:
        pickle.dump({"params": _to_np(params), "ppo_opt_state": _to_np(ppo_opt_state),
                     "replay_opt_state": _to_np(replay_opt_state),
                     "target_params": _to_np(target_params),
                     "replay_buffer": replay.state_dict(), "pending": pending.state_dict(),
                     "rng": np.asarray(rng), "action_rng": RU.action_rng_state(action_rng),
                     "update_count": update_count, "global_step": step,
                     "memories": np.asarray(memories), "mem_mask": np.asarray(mem_mask),
                     "mem_idx": np.asarray(mem_idx),
                     "w512_state": _to_np(w512_state), "done_enter": np.asarray(done_enter),
                     "obsv": np.asarray(obsv),
                     "counters": {"accepted_policy_updates": accepted_policy_updates,
                                  "kl_rejected_updates": kl_rejected_updates,
                                  "replay_sequences_consumed": replay_sequences_consumed,
                                  "replay_sample_rng_state": replay_sample_rng.get_state(),
                                  "phase4a_v2": counters.snapshot()},
                     "manifest": {"params_sha256": p_sha, "step": step, "arm": ARM,
                                  "carry_mode": CARRY_MODE, "replay_mode": REPLAY_MODE,
                                  "network_family": NETWORK_FAMILY,
                                  "w512": _manifest_fields()}}, f, protocol=4)
    print(f"[ckpt] step={step} params_sha={p_sha[:16]} tag={tag}", flush=True)
    return p_sha

save_ckpt(0, params, ppo_opt_state, replay_opt_state, target_params, "step0")

# ----------------------- training loop -----------------------
for u in range(args.total_updates):
    t_u = time.time()
    # 1. collect
    trajs, carry, rollout, stats = WC.collect_rollout_w512(
        env, env_state, network, params, obsv, memories, mem_mask, mem_idx, w512_state,
        done_enter, rng, action_rng, pending, target_achievement_1d, cfg.num_steps,
        cfg.window_mem, cfg.num_heads, w5_cfg, SEGMENT_LEN,
        collected_update_count=update_count, apply_eval_w512=apply_eval_w512,
        env_params=env_params, outer_update_index=u, policy_version=counters.policy_version)
    env_state = carry["env_state"]; obsv = carry["obsv"]
    memories = carry["memories"]; mem_mask = carry["mem_mask"]; mem_idx = carry["mem_idx"]
    w512_state = carry["w512_state"]; done_enter = carry["done_enter"]; rng = carry["rng"]

    for t in trajs:
        assert bool(np.asarray(t.dones)[-1]), "HARD STOP episode-boundary: non-terminal trajectory"
        replay.insert(t)                       # validates GTrXL + W512 anchor conservation
        replay.counters.trajectories_collected += 1
    assert replay.counters.trajectories_collected == replay.counters.trajectories_inserted, \
        "HARD STOP conservation: collected != inserted"

    # 2. last_value + GAE + PPO main update (SAME forward path as collection)
    st_clr = WA.w512_reset128_clear(w512_state, SEGMENT_LEN)
    mi_adv, mm_adv = WA.w512_advance_mask(mem_idx, mem_mask, done_enter,
                                          cfg.window_mem, cfg.num_heads)
    _lg, last_value, _mo, _ht = apply_eval_w512(
        params, memories, obsv, mm_adv, st_clr["long_buf"], st_clr["long_mask"])
    advantages, targets = WP.compute_gae(
        rollout["rewards"], rollout["values"], rollout["dones"], np.asarray(last_value),
        cfg.gamma, cfg.gae_lambda, cfg.value_target_clip_min, cfg.value_target_clip_max)
    params, ppo_opt_state, ppo_metrics = WP.ppo_update_w512(
        network, params, ppo_opt_state, ppo_opt, rollout, advantages, targets,
        ppo_cfg, w5_cfg, SEGMENT_LEN, rng)
    update_count += 1
    online_ppo_update_count += 1
    counters.on_outer_update(cfg.num_envs, cfg.num_steps)
    counters.on_ppo_accepted()
    assert ppo_metrics["ppo_finite"], "HARD STOP NaN/Inf in PPO update"

    # 3. replay update (ORIGINAL-GOAL V-TRACE ONLY; no relabel, no AWR; firewall asserted)
    rep = {}; did_replay_update = False
    _rstats = _replay_stats(replay._buffer)
    _replay_attempted = False; _eligible = int(_rstats["replay_eligible_count_129"])
    _batch_size = 0; _seq_lens = []
    _batch = replay.sample_eligible(SEQUENCE_LENGTH, replay_sample_rng, K_BATCH)
    _replay_attempted = True
    _eligible = int(_batch.eligible_count)
    if _batch.status == "NOT_READY":
        replay_not_ready_skip_count += 1
        print(f"REPLAY_NOT_READY requested_sequence_length={SEQUENCE_LENGTH} "
              f"max_trajectory_length={_rstats['replay_max_trajectory_length']} "
              f"eligible_count={_batch.eligible_count}", flush=True)
    else:
        so = _batch.samples
        _batch_size = int(len(so)); _seq_lens = [int(x) for x in _batch.sequence_lengths]
        counters.on_replay_attempt(len(so))
        replay_sample_success_count += len(so)
        params, target_params, replay_opt_state, m = WRL.original_vtrace_update_w512(
            network, params, target_params, replay_opt_state, replay_opt,
            apply_eval_w512, scan_fn, so, fp_cfg, w5_cfg, SEGMENT_LEN)
        update_count += 1
        counters.on_replay_update_executed()
        did_replay_update = True
        assert bool(m["finite"]), "HARD STOP NaN/Inf in original_vtrace replay loss"
        if bool(m.get("policy_committed")):
            assert float(m["policy_kl"]) <= fp_cfg.kl_replay_max + 1e-12, \
                f"HARD STOP accepted policy_kl={float(m['policy_kl']):.5f} > {fp_cfg.kl_replay_max}"
            accepted_policy_updates += 1
            counters.on_replay_policy_committed()
        if bool(m.get("kl_rejected_update")):
            assert not bool(m.get("policy_committed")), "HARD STOP KL-rejected committed policy"
            kl_rejected_updates += 1
            counters.on_replay_kl_rejected()
        assert float(m["entropy"]) >= fp_cfg.ent_floor, \
            f"HARD STOP entropy collapse {float(m['entropy']):.4f} < {fp_cfg.ent_floor}"
        rep = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in m.items() if not isinstance(v, (list, dict))}
        rep["batch"] = len(so)
        rep["replay_sample_ids"] = list(_batch.sample_ids)
        rep["replay_start_offsets"] = list(_batch.start_offsets)
        replay_update_success_count += 1
        replay_sequences_consumed += len(so)
        if replay_first_success_update is None:
            replay_first_success_update = update_count
        counters.assert_hindsight_awr_disabled()      # firewall: Hindsight/AWR MUST stay 0
    replay_attempt_mask.append(bool(_replay_attempted))
    eligible_count_by_outer_update.append(int(_eligible))
    if did_replay_update:
        replay_update_outer_updates.append(int(u))
        replay_batch_sizes.append(int(_batch_size))
        replay_sequence_lengths.append(list(_seq_lens))
    # EMA target on PPO-only iterations (a replay update does its own EMA internally)
    if not did_replay_update:
        target_params = WRL.FPL.ema_update(params, target_params, fp_cfg.ema_tau)

    # ---- W512 read-path monitor + gtrxl window finite (post-update carried state) ----
    mon = _w512_read_monitor(params, memories, obsv, mm_adv,
                             st_clr["long_buf"], st_clr["long_mask"])
    if mon["w512_read_logit_diff"] > 0.0 or mon["w512_read_KL"] > 0.0:
        w512_read_ever_nonzero = True
    _gtrxl_mem = np.asarray(memories); _w512_long = np.asarray(w512_state["long_buf"])
    if not (bool(np.all(np.isfinite(_gtrxl_mem))) and bool(np.all(np.isfinite(_w512_long)))):
        gtrxl_window_finite_all = False
    mon.update(dict(update=u, global_step=(u + 1) * STEPS_PER_UPDATE, arm=ARM,
                    gtrxl_window_mem_maxabs=float(np.max(np.abs(_gtrxl_mem))),
                    w512_long_buf_maxabs=float(np.max(np.abs(_w512_long))),
                    w512_read_ever_nonzero=w512_read_ever_nonzero,
                    gtrxl_window_finite_all=gtrxl_window_finite_all,
                    accepted_policy_updates=accepted_policy_updates,
                    online_ppo_update_count=online_ppo_update_count,
                    replay_update_success_count=replay_update_success_count,
                    replay_not_ready_skip_count=replay_not_ready_skip_count))
    mon.update(_rstats)
    with open(mon_path, "a") as f:
        f.write(json.dumps(mon, default=str) + "\n")

    global_step = (u + 1) * STEPS_PER_UPDATE
    entry = dict(update=u, global_step=global_step, arm=ARM,
                 completed_episodes=stats["completed_episodes"],
                 mean_ep_return=stats["mean_ep_return"], mean_ep_length=stats["mean_ep_length"],
                 replay_size=len(replay), update_count=update_count, **ppo_metrics, **rep,
                 t_s=round(time.time() - t_u, 1))
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    print(f"[u{u}] gs={global_step} ppo_actor={ppo_metrics['ppo_actor']:.4f} "
          f"ent={ppo_metrics['ppo_entropy']:.4f} eps={stats['completed_episodes']} "
          f"replay={len(replay)} kl={rep.get('policy_kl','-')} "
          f"read_diff={mon['w512_read_logit_diff']:.3e} ({time.time()-t_u:.1f}s)", flush=True)

    # 4. checkpoint at save points
    if (u + 1) % args.save_every == 0 or (u + 1) == args.total_updates:
        save_ckpt(global_step, params, ppo_opt_state, replay_opt_state, target_params, "save")

# ----------------------- terminal W512 gates -----------------------
# FINAL Hindsight/AWR firewall: for replay_mode=original_vtrace all four counters MUST be 0.
counters.assert_hindsight_awr_disabled()
RESET128_CLEAR_OK = bool(RESET128_PROBE.get("w512_reset128_probe_ok", False))
REPLAY_HORIZON_REACHED = bool(replay_update_success_count > 0)
# W512 long_run_98304 is the formal MATCHED_98304 candidate; the replay horizon is REPORTED but,
# consistent with the RMT16 long_run_98304 philosophy, is NOT a hard EXIT gate (correctness +
# finite params + firewall are). training_contract_match (constants) is the ranking criterion.
REPLAY_HORIZON_REQUIRED_FOR_PASS = False
_params_finite = all(bool(np.isfinite(np.asarray(v)).all())
                     for v in jax.tree_util.tree_leaves(params)
                     if np.issubdtype(np.asarray(v).dtype, np.floating))
CORRECTNESS_OK = bool(gtrxl_window_finite_all and RESET128_CLEAR_OK
                      and w512_read_ever_nonzero and _params_finite)
ARM_GATES_PASS = bool(CORRECTNESS_OK
                      and (REPLAY_HORIZON_REACHED or not REPLAY_HORIZON_REQUIRED_FOR_PASS))
if not RESET128_CLEAR_OK:
    ARM_STATUS = "W512_RESET128_CLEAR_FAILED"
elif not gtrxl_window_finite_all:
    ARM_STATUS = "W512_WINDOW_NOT_FINITE"
elif not w512_read_ever_nonzero:
    ARM_STATUS = "W512_READ_PATH_NOT_ACTIVE"
elif not _params_finite:
    ARM_STATUS = "W512_PARAMS_NOT_FINITE"
elif ARM_GATES_PASS and REPLAY_HORIZON_REACHED:
    ARM_STATUS = "PASS"
elif ARM_GATES_PASS:
    ARM_STATUS = "PASS_REPLAY_HORIZON_NOT_REACHED"
else:
    ARM_STATUS = "FAIL"
print(f"[gates] arm={ARM} STATUS={ARM_STATUS} replay_upd={replay_update_success_count} "
      f"reset128_ok={RESET128_CLEAR_OK} read_active={w512_read_ever_nonzero} "
      f"gtrxl_finite={gtrxl_window_finite_all} params_finite={_params_finite}", flush=True)

# ----------------------- summary -----------------------
summary = dict(
    arm=ARM, candidate_id="W512_RESET128_P2REPLAY_CANONICAL_98304",
    network_family=NETWORK_FAMILY, carry_mode=CARRY_MODE, replay_mode=REPLAY_MODE,
    memory_mode="w512_raw_history", hindsight=False, awr=False,
    observation_shape=list(OBS_SHAPE), action_dim=ACTION_DIM,
    run_class=args.run_class, interruption_policy="RESTART_FROM_STEP0",
    formal_student_ranking_eligible=True, budget_class="MATCHED_98304",
    replay_horizon_required_for_pass=REPLAY_HORIZON_REQUIRED_FOR_PASS,
    total_updates=args.total_updates, global_step=args.total_updates * STEPS_PER_UPDATE,
    total_env_steps=args.total_updates * STEPS_PER_UPDATE,
    final_params_sha256=_params_sha(params), base_sha256=base_sha,
    base_checkpoint_params_sha256=base_sha,
    merged_base_sha256=_merged_base_sha,
    accepted_policy_updates=accepted_policy_updates, kl_rejected_updates=kl_rejected_updates,
    config={k: v for k, v in vars(cfg).items()},
    w512_manifest=_manifest_fields(),
    phase4a_v2_counters=counters.snapshot(),
    replay_update_count=counters.replay_update_count,
    accepted_replay_policy_update_count=counters.accepted_replay_policy_update_count,
    replay_attempt_count=counters.replay_attempt_count,
    replay_sequences_consumed=replay_sequences_consumed,
    policy_version=counters.policy_version, outer_update_index=counters.outer_update_index,
    global_env_steps=counters.global_env_steps,
    exposure_certificate=dict(
        outer_update_count=int(counters.outer_update_index),
        replay_attempt_mask=[bool(x) for x in replay_attempt_mask],
        replay_update_outer_updates=list(replay_update_outer_updates),
        replay_update_count=int(counters.replay_update_count),
        replay_sequences_consumed=int(replay_sequences_consumed),
        replay_batch_sizes=list(replay_batch_sizes),
        replay_sequence_lengths=[list(x) for x in replay_sequence_lengths],
        eligible_count_by_outer_update=list(eligible_count_by_outer_update)),
    p2_frozen=dict(rho_bar=fp_cfg.rho_bar, c_bar=fp_cfg.c_bar, beta=fp_cfg.beta,
                   w_max=fp_cfg.w_max, w_vtrace=fp_cfg.w_vtrace, w_awr=fp_cfg.w_awr,
                   kl_replay_max=fp_cfg.kl_replay_max, ema_tau=fp_cfg.ema_tau,
                   policy_lag_gate_active=False, max_policy_lag=None),
    executed_protocol_identity=EXECUTED_PROTOCOL_IDENTITY,
    runtime_config_certificate_status=RUNTIME_CONFIG_CERTIFICATE["certificate_status"],
    runtime_config_certificate_path=RUNTIME_CONFIG_CERTIFICATE_PATH,
    runtime_config_certificate_payload_sha256=RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
    runtime_config_certificate_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
    scientific_config_sha256=CONFIG_IDENTITY["scientific_config_sha256"],
    config_file_sha256=CONFIG_IDENTITY["config_file_sha256"],
    training_contract_match=bool(CONFIG_IDENTITY["scientific_constants_match"]
                                 and not _IMPORTED_DRIFT),
    reset128_probe=RESET128_PROBE,
    status="COMPLETE", arm_status=ARM_STATUS, arm_gates_pass=bool(ARM_GATES_PASS),
    reset128_clear_ok=RESET128_CLEAR_OK, w512_read_ever_nonzero=w512_read_ever_nonzero,
    gtrxl_window_finite_all=gtrxl_window_finite_all, params_finite=_params_finite,
    replay_horizon_reached=REPLAY_HORIZON_REACHED,
    online_ppo_update_count=online_ppo_update_count,
    replay_sample_success_count=replay_sample_success_count,
    replay_update_success_count=replay_update_success_count,
    replay_not_ready_skip_count=replay_not_ready_skip_count,
    replay_first_success_update=replay_first_success_update,
    step0_params_in="ckpt/0/full_state.pkl",
    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with open(os.path.join(LOG_DIR, f"{ARM}_train_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\n" + "=" * 78, flush=True)
print(f"{ARM} COMPLETE  final_params_sha={summary['final_params_sha256'][:16]}", flush=True)
print(f"  accepted_policy_updates={accepted_policy_updates} kl_rejected={kl_rejected_updates} "
      f"replay_updates={replay_update_success_count}", flush=True)
print("=" * 78, flush=True)
print(f"W512_ARM_FINAL_STATUS={ARM_STATUS}", flush=True)
print(f"RUNTIME_CONFIG_CERTIFICATE_STATUS={RUNTIME_CONFIG_CERTIFICATE['certificate_status']} "
      f"certificate_finalized={RUNTIME_CONFIG_CERTIFICATE['certificate_finalized']} "
      f"base_checkpoint_match="
      f"{RUNTIME_CONFIG_CERTIFICATE['checkpoint_identity']['base_checkpoint_match']} "
      f"certificate_payload_sha256={RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256} "
      f"certificate_file_sha256={RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256} "
      f"base_checkpoint_params_sha256={base_sha}", flush=True)
if not ARM_GATES_PASS:
    sys.exit(1)
