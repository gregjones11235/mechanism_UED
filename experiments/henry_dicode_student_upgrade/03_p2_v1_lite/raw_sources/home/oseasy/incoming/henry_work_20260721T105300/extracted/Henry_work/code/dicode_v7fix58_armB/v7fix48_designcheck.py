"""v7fix4.8 designcheck — guardrail layer (value-target clip + session watchdog).

Wiring-level assertions (grep-style, no jax needed): every check names the design
requirement it pins. Run: python v7fix48_designcheck.py  -> prints N/N PASS.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


def _schedule_body(src):
    """Live linear_schedule body, comment-stripped and whitespace-normalized, with the
    config-getter spelling unified (ppo_tr uses _cfg_get, train_state_utils uses getattr —
    the only sanctioned difference). Used to pin the two copies token-identical."""
    m = re.search(r"def linear_schedule\(count\):(.*?)\n\s*return lr\n", src, re.S)
    body = m.group(1) if m else ""
    lines = [re.sub(r"\s+", "", ln).replace("_cfg_get", "getattr") for ln in body.splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


PPO = _read("src", "dicode", "ppo_tr.py")
TRAINING = _read("src", "dicode", "training.py")
GUARD = _read("src", "dicode", "train_guard.py")
YAML = _read("conf", "training", "default.yaml")
TESTS = _read("auction", "tests", "test_train_guard.py")
TSU = _read("src", "dicode", "utils", "general", "train_state_utils.py")

CHECKS = [
    # --- G.1-G.5: value-target clip (the collapse fix itself) ---
    ("G.1 GAE targets clipped at the training GAE return",
     "jnp.clip(\n\t\t\t\t\tadvantages + traj_batch.value, vt_clip_min, vt_clip_max" in PPO),
    ("G.2 clip bounds read from config with safe defaults (-50/300)",
     'value_target_clip_min", -50.0' in PPO and 'value_target_clip_max", 300.0' in PPO),
    ("G.3 config keys shipped in training yaml",
     "value_target_clip_min: -50.0" in YAML and "value_target_clip_max: 300.0" in YAML),
    ("G.4 scoring-side GAE (no-update path) left UNclipped — PVL scores must see raw advantages",
     PPO.count("jnp.clip(\n\t\t\t\t\tadvantages + traj_batch.value") == 1
     and "return advantages\n" in PPO),
    ("G.5 _cfg_get tolerates DictConfig and plain namespaces",
     "def _cfg_get(config, key, default)" in PPO and "except AttributeError" in PPO),
    # --- W.1-W.8: session watchdog wiring ---
    ("W.1 per-update stats fed from the wandb callback (v_loss + entropy)",
     "record_update(float(v_loss), float(ent))" in PPO),
    ("W.2 stats buffer reset at every session start (run_training_session)",
     "reset_session_stats()" in PPO and PPO.index("reset_session_stats()") < PPO.index("train_jit = jax.jit")),
    ("W.3 verdict judged BEFORE any archive persistence",
     TRAINING.index("_session_guard_verdict(config, evaluation_metrics)")
     < TRAINING.index("_update_archive_with_metrics(\n            gen_manager")),
    ("W.4 revert returns the PRE-session train_state and PRE-session counters",
     "pre_session_train_state = rl_train_state" in TRAINING
     and re.search(r"return \(\s*rng,\s*pre_session_train_state,\s*pre_session_update_step,"
                   r"\s*pre_session_env_steps,", TRAINING)),
    ("W.5 revert uses the 0-updates return shape the main loop already skips ckpt/graph on",
     re.search(r"pre_session_env_steps,\s*\{\},\s*0,\s*\{\},\s*\{\},\s*(?:pre_session_sil_state,)?\s*\)", TRAINING)),  # v7fix56: optional 9th element MUST be the pre-session SIL pool
    ("W.6 debug callbacks flushed before judging (effects_barrier, AttributeError-safe)",
     "jax.effects_barrier()" in TRAINING and "except AttributeError" in TRAINING),
    ("W.7 consecutive reverts stop the job (RuntimeError, config-capped)",
     "guard_max_consecutive_reverts" in TRAINING and "raise RuntimeError" in TRAINING),
    ("W.8 grep-able revert marker for monitoring",
     "[guard][SESSION-REVERT]" in TRAINING),
    # --- C.1-C.4: calibration contract (thresholds must not drift from the postmortem) ---
    ("C.1 v_loss threshold 1000 (5x worst benign flare 196 incl. BASELINE, 14x under runaway 13976)",
     "guard_session_vloss_max: 1000.0" in YAML and 'guard_session_vloss_max", 1000.0' in TRAINING),
    ("C.2 entropy hard line 0.10 (RECALIBRATED 2026-07-19: gamma=0.99 benign band dips to 0.1287,"
     " old 0.15 false-killed sil010 via two adjacent benign dips; collapse 0.16->0.001 still"
     " crosses 0.10 same-session; code fallback stays 0.15, yaml must carry the calibrated value)",
     "guard_session_entropy_min: 0.10" in YAML and 'guard_session_entropy_min", 0.15' in TRAINING),
    ("C.3 self-recovering-flare regression tests exist (fast 86 AND baseline 196 must not trip)",
     "test_self_recovering_flare_worst_case_no_trip" in TESTS and "test_baseline_flare_no_trip" in TESTS),
    ("C.4 held-out red line is drop-based (fresh runs immune)",
     "collect_wood_prev_min" in GUARD and "test_fresh_run_low_collect_wood_never_trips" in TESTS),
    # --- L.1-L.3: LR-schedule clamp (the ROOT-CAUSE fix — negative LR past the horizon) ---
    ("L.1 fresh-init schedule clamped in ppo_tr (frac floored at 0 before min_lr blend)",
     "frac = jnp.maximum(frac, 0.0)" in PPO
     and PPO.index("frac = jnp.maximum(frac, 0.0)") < PPO.index("lr = config.min_lr + (config.lr - config.min_lr) * frac")),
    ("L.2 RESUME-path schedule clamped in train_state_utils.load_weights_only (the tx a resumed run actually trains with)",
     "frac = jnp.maximum(frac, 0.0)" in TSU
     and TSU.index("frac = jnp.maximum(frac, 0.0)") < TSU.index("lr = config.min_lr + (config.lr - config.min_lr) * frac")),
    ("L.3 both schedules blend to min_lr (not a hard-zero floor) so post-horizon training still learns",
     PPO.count("lr = config.min_lr + (config.lr - config.min_lr) * frac") >= 1
     and TSU.count("lr = config.min_lr + (config.lr - config.min_lr) * frac") >= 1),
    # --- L.4-L.8: v7fix4.9 second anneal leg (post-horizon re-anneal; 2e-6 too small to absorb
    #     level-pool rotations — s195 relay influx = session-long value flare starving the actor) ---
    ("L.4 second leg wired in BOTH schedules and gated on u >= lr_restart_at",
     "jnp.where(u >= config.lr_restart_at" in PPO and "jnp.where(u >= config.lr_restart_at" in TSU),
    ("L.5 leg ships DISABLED by default (fresh runs keep the plain clamped schedule)",
     "lr_restart: 0.0" in YAML and "lr_restart_warmup: 50" in YAML),
    ("L.6 warmup ramps from min_lr (continuous with the pre-restart floor, no 15x LR step)",
     "leg2 = config.min_lr + (leg2 - config.min_lr) * warm" in PPO
     and "leg2 = config.min_lr + (leg2 - config.min_lr) * warm" in TSU),
    ("L.7 leg fraction clipped both sides (never above lr_restart, decays to min_lr at horizon, never below)",
     "frac2 = jnp.clip((config.lr_restart_horizon - u) / span, 0.0, 1.0)" in PPO
     and "frac2 = jnp.clip((config.lr_restart_horizon - u) / span, 0.0, 1.0)" in TSU),
    ("L.8 the two schedule bodies are token-identical (fresh-init and resume paths cannot drift)",
     _schedule_body(PPO) == _schedule_body(TSU) and len(_schedule_body(PPO)) >= 8),
    # --- P.1-P.3: purity/accounting ---
    ("P.1 train_guard is pure python (no jax import — locally testable)",
     "import jax" not in GUARD),
    ("P.2 baseline for held-out drop updated only on ACCEPTED sessions",
     TRAINING.index("register_verdict(False)") < TRAINING.index("note_heldout(evaluation_metrics)")
     and "test_heldout_baseline_not_updated_by_reverted_session" in TESTS),
    ("P.3 healthy sessions clear the consecutive-revert counter",
     "train_guard.register_verdict(False)" in TRAINING),
]

failed = [name for name, ok in CHECKS if not ok]
for name, ok in CHECKS:
    print(("PASS " if ok else "FAIL ") + name)
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed.")
if failed:
    raise SystemExit(1)
