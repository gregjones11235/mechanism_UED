#!/usr/bin/env python3
"""Phase4A-v2 — 15 test gates (CC2 directive §十).

Run directly for a gated summary (any FAIL => exit 1, per "任一门禁失败立即停止汇报"):
    python test_phase4a_v2_gates.py
or under pytest:
    pytest test_phase4a_v2_gates.py -v

Layering:
  * Pure-Python / numpy / AST / subprocess gates run ANYWHERE (no JAX needed):
    GATE 1(method), 2, 3, 7, 9, 10, 11(counters), 12(static), 13(structural), 14, 15.
  * JAX gates run on the SERVER CPU (dicode310): GATE 4(monkeypatch), 5/6(ast+jax), 8, 13(numeric*).
    They pytest.skip when JAX is absent so the local suite stays green for what is provable locally.

GATE 13's numeric bit-exact re-verification requires a parameter-updating run, which is NOT
authorized this round (NEW_TRAINING_RUNS=0); GATE 13 is therefore proven by construction
(additive changes + off-mode counter equivalence unit test + static replay-guard check) and the
numeric equiv-hash rerun is recorded as deferred in the test report.
"""
import ast
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SNAPSHOT = os.path.dirname(_HERE)
_EXP = os.path.join(_SNAPSHOT, "runtime", "experiment_src")
_FRZ = os.path.join(_SNAPSHOT, "runtime", "frozen_modules")
for _p in (_EXP, _FRZ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_LAUNCHER = os.path.join(_EXP, "train_rmt16_p2replay.py")
_LEARNER = os.path.join(_FRZ, "rmt_replay_learner.py")

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

import numpy as np  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _func_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _tokens(func_node):
    """Set of Name ids + Attribute attrs referenced inside a function body."""
    toks = set()
    for n in ast.walk(func_node):
        if isinstance(n, ast.Name):
            toks.add(n.id)
        elif isinstance(n, ast.Attribute):
            toks.add(n.attr)
    return toks


def _count_calls(func_node, callee):
    c = 0
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == callee) or \
               (isinstance(f, ast.Attribute) and f.attr == callee):
                c += 1
    return c


def make_rmt_traj(L, seed=0, obs_dim=6, wm=8, layers=2, heads=2,
                  num_tokens=4, seg_len=8, n_ach=3, D=16):
    """Build a VALID synthetic RMTTrajectory of length L (anchor-conservation legal)."""
    from rmt_replay_buffer import RMTTrajectory
    from replay_buffer import anchor_steps_for_length
    rng = np.random.RandomState(seed)
    asteps = np.array(anchor_steps_for_length(L), dtype=np.int64)
    N = len(asteps)
    embed = D
    return RMTTrajectory(
        observations=rng.randn(L, obs_dim).astype(np.float32),
        actions=rng.randint(0, 5, size=(L,)).astype(np.int32),
        rewards=rng.randn(L).astype(np.float32),
        dones=np.concatenate([np.zeros(L - 1, bool), np.ones(1, bool)]),
        values=rng.randn(L).astype(np.float32),
        log_probs=(-rng.rand(L)).astype(np.float32),
        initial_memory=rng.randn(wm, layers, embed).astype(np.float32),
        achievements=np.zeros((L, n_ach), np.float32),
        target_achievements=np.zeros(n_ach, np.float32),
        next_observations=rng.randn(L, obs_dim).astype(np.float32),
        memory_anchors=rng.randn(N, wm, layers, embed).astype(np.float32),
        anchor_steps=asteps,
        anchor_masks=np.zeros((N, heads, 1, wm + 1), bool),
        anchor_idxs=np.zeros(N, np.int64),
        collected_update_count=0,
        outer_update_index=0,
        policy_version_at_collection=0,
        rmt_initial_tokens=rng.randn(num_tokens, D).astype(np.float32),
        rmt_initial_segbuf=rng.randn(seg_len, D).astype(np.float32),
        rmt_initial_segcount=0,
        rmt_anchor_tokens=rng.randn(N, num_tokens, D).astype(np.float32),
        rmt_anchor_segbuf=rng.randn(N, seg_len, D).astype(np.float32),
        rmt_anchor_segcount=np.zeros(N, np.int64),
    )


# ----------------------------------------------------------------------------
# GATE 1 — L512 reachability recomputable from original episode records (method)
# ----------------------------------------------------------------------------

def gate01():
    import recompute_probe_step as R
    rep = R._selftest()  # synthetic method self-test (no server data needed)
    if rep["first_ge512"]["first_ge512_resolved_env_step"] != 4180:
        return FAIL, f"method self-test wrong resolved step: {rep}"
    return PASS, ("recompute method verified on synthetic records "
                  "(real both-arm recompute runs on server episode jsonl; conclusion expected BOTH)")


# ----------------------------------------------------------------------------
# GATE 2 — completion_resolved_env_step formula correct across env_id / rollout_step
# ----------------------------------------------------------------------------

def gate02():
    from phase4a_v2_counters import (completion_resolved_env_step as cr,
                                     completion_global_step_deprecated as dep)
    NE, RS = 16, 128
    checks = []
    checks.append(cr(0, NE, RS, 0, 0) == 1)                 # very first resolved step
    checks.append(cr(0, NE, RS, 0, 15) == 16)               # last env of rollout_step 0
    checks.append(cr(0, NE, RS, 1, 0) == 17)                # first env of rollout_step 1
    checks.append(cr(1, NE, RS, 0, 0) == NE * RS + 1)       # update 1 start == 2049
    checks.append(cr(2, NE, RS, 5, 3) == 2 * 2048 + 5 * 16 + 3 + 1)  # == 4180
    # contiguity: within update 0 the 2048 resolved steps are exactly 1..2048
    seen = sorted(cr(0, NE, RS, r, e) for r in range(RS) for e in range(NE))
    checks.append(seen == list(range(1, NE * RS + 1)))
    # the deprecated formula is DIFFERENT (demonstrates the corrected bug)
    checks.append(dep(2, NE, RS, 5) == 2 * 2048 + 5)        # 4101, drops *num_envs/env_id/+1
    checks.append(cr(2, NE, RS, 5, 3) != dep(2, NE, RS, 5))
    if all(checks):
        return PASS, "formula exact across env_id/rollout_step; contiguity 1..2048; deprecated differs"
    return FAIL, f"formula checks failed: {checks}"


# ----------------------------------------------------------------------------
# GATE 3 — outer / PPO / Replay / policy_version counters are NOT mixed
# ----------------------------------------------------------------------------

def gate03():
    from phase4a_v2_counters import Phase4ACounters
    c = Phase4ACounters()
    # 3 outer iterations, PPO each, then 1 committed replay + 1 KL-rejected replay
    for _ in range(3):
        c.on_outer_update(16, 128)
        c.on_ppo_accepted()
    assert c.outer_update_index == 3
    assert c.global_env_steps == 3 * 2048
    assert c.online_ppo_update_count == 3
    assert c.policy_version == 3                      # PPO always commits
    # committed replay
    c.on_replay_attempt(4)
    c.on_replay_update_executed()
    c.on_replay_policy_committed()
    assert c.accepted_replay_policy_update_count == 1
    assert c.replay_update_count == 1
    assert c.policy_version == 4                      # advanced
    assert c.online_ppo_update_count == 3            # PPO counter untouched by replay
    # KL-rejected replay: executed but NOT committed
    c.on_replay_update_executed()
    c.on_replay_kl_rejected()
    assert c.replay_update_count == 2                 # executed count advanced
    assert c.accepted_replay_policy_update_count == 1 # accepted NOT advanced
    assert c.policy_version == 4                      # policy_version NOT advanced on rollback
    assert c.replay_attempt_count == 4
    return PASS, ("outer/PPO/replay/accepted/policy_version independent; "
                  "KL-rollback advances executed but not accepted/policy_version")


# ----------------------------------------------------------------------------
# GATE 4 — original_vtrace path does NOT call Hindsight (AST structural)
# ----------------------------------------------------------------------------

def gate04():
    tree = ast.parse(_read(_LEARNER))
    forbidden = {"relabel_sample_rmt", "relabel_trajectory_rmt", "rmt_hindsight", "RH"}
    for fn in ("compute_loss_original_vtrace_rmt", "original_vtrace_update_rmt"):
        node = _func_node(tree, fn)
        if node is None:
            return FAIL, f"missing function {fn}"
        hit = forbidden & _tokens(node)
        if hit:
            return FAIL, f"{fn} references Hindsight symbols {hit}"
    return PASS, "neither original_vtrace function references relabel/rmt_hindsight/RH"


# ----------------------------------------------------------------------------
# GATE 5 — original_vtrace path does NOT compute AWR (AST structural)
# ----------------------------------------------------------------------------

def gate05():
    tree = ast.parse(_read(_LEARNER))
    forbidden = {"awr", "awr_losses", "AWRConfig", "w_awr", "A"}
    for fn in ("compute_loss_original_vtrace_rmt", "original_vtrace_update_rmt"):
        node = _func_node(tree, fn)
        if node is None:
            return FAIL, f"missing function {fn}"
        hit = forbidden & _tokens(node)
        if hit:
            return FAIL, f"{fn} references AWR symbols {hit}"
    return PASS, "neither original_vtrace function references awr/AWRConfig/w_awr/A"


# ----------------------------------------------------------------------------
# GATE 6 — original_vtrace loss = ONE original RMT scan + corresponding target scan
# ----------------------------------------------------------------------------

def gate06():
    tree = ast.parse(_read(_LEARNER))
    loss = _func_node(tree, "compute_loss_original_vtrace_rmt")
    upd = _func_node(tree, "original_vtrace_update_rmt")
    if loss is None or upd is None:
        return FAIL, "missing original_vtrace functions"
    n_scan_loss = _count_calls(loss, "scan_fn")
    if n_scan_loss != 1:
        return FAIL, f"loss scan_fn calls = {n_scan_loss} (expected exactly 1)"
    n_target = _count_calls(upd, "_target_scan_rmt")
    n_recon = _count_calls(upd, "reconstruct_rmt_batch")
    if n_target != 1:
        return FAIL, f"update _target_scan_rmt calls = {n_target} (expected 1)"
    if n_recon != 2:  # online + target, BOTH original sample
        return FAIL, f"update reconstruct_rmt_batch calls = {n_recon} (expected 2: online+target)"
    # no second relabeled reconstruction / target / samples
    for fn, node in (("loss", loss), ("update", upd)):
        toks = _tokens(node)
        bad = {"recon_r", "target_vals_r", "samples_rel"} & toks
        if bad:
            return FAIL, f"{fn} references relabeled-side symbols {bad}"
    return PASS, "loss has exactly 1 online scan; update has 1 target scan + 2 original recon; no relabeled side"


# ----------------------------------------------------------------------------
# GATE 7 — sequence_length=129 crosses the 128 boundary
# ----------------------------------------------------------------------------

def gate07():
    import yaml
    msgs = []
    for arm in ("persistent", "reset128"):
        cfg = yaml.safe_load(_read(os.path.join(_SNAPSHOT, "configs",
                                                f"rmt16_phase4a_v2_{arm}.yaml")))
        sci = cfg["scientific_config"]
        if int(sci["sequence_length"]) != 129:
            msgs.append(f"{arm}: sequence_length={sci['sequence_length']}")
        if int(sci["segment_len"]) != 128:
            msgs.append(f"{arm}: segment_len={sci['segment_len']}")
        if bool(sci["crosses_boundary"]) is not True:
            msgs.append(f"{arm}: crosses_boundary not true")
        if not (int(sci["sequence_length"]) > int(sci["segment_len"])):
            msgs.append(f"{arm}: sequence_length does not exceed segment_len")
    src = _read(_LAUNCHER)
    if "default=129" not in src:
        msgs.append("launcher --sequence_length default != 129")
    if "SEQUENCE_LENGTH <= SEGMENT_LEN" not in src:
        msgs.append("launcher missing original_vtrace crossing guard")
    if msgs:
        return FAIL, "; ".join(msgs)
    return PASS, "both configs sequence_length=129 > segment_len=128 (crosses one boundary); launcher guard present"


# ----------------------------------------------------------------------------
# GATE 8 — Persistent step-129 entering token NONZERO; Reset128 token ZERO (JAX)
# ----------------------------------------------------------------------------

def gate08():
    if not HAVE_JAX:
        return SKIP, "JAX absent (runs on server CPU)"
    import rmt16_memory as rmtm
    from rmt_memory_anchor import rmt_advance_tokens
    cfg = rmtm.RMT16Config(num_tokens=4, segment_len=128, encoder_size=8)
    N = 2

    def update_fn(tokens, seg_buf):     # stub: nonzero "updated" tokens
        return jnp.ones_like(tokens) * 5.0

    # seg_count=127 -> one more stored h_t reaches 128 -> segment boundary fires
    st = {"mem_tokens": jnp.zeros((N, 4, 8)),
          "seg_buf": jnp.ones((N, 128, 8)),
          "seg_count": jnp.full((N,), 127, jnp.int32)}
    h_t = jnp.ones((N, 8))
    done = jnp.zeros((N,), jnp.bool_)
    st_p = rmt_advance_tokens(st, h_t, done, update_fn, cfg, "persistent")
    st_r = rmt_advance_tokens(st, h_t, done, update_fn, cfg, "reset128")
    p_max = float(jnp.max(jnp.abs(st_p["mem_tokens"])))
    r_max = float(jnp.max(jnp.abs(st_r["mem_tokens"])))
    if p_max != 5.0:
        return FAIL, f"persistent boundary token maxabs={p_max} (expected 5.0, nonzero carry)"
    if r_max != 0.0:
        return FAIL, f"reset128 boundary token maxabs={r_max} (expected 0.0, cleared)"
    if int(st_p["seg_count"][0]) != 0 or int(st_r["seg_count"][0]) != 0:
        return FAIL, "seg_count did not reset at boundary"
    return PASS, "at 128 boundary: persistent entering token nonzero (5.0), reset128 token zero; seg_count reset both"


# ----------------------------------------------------------------------------
# GATE 9 — eligible-only sampler never draws a too-short trajectory
# ----------------------------------------------------------------------------

def gate09():
    from rmt_replay_buffer import RMTReplayBuffer
    buf = RMTReplayBuffer(capacity=64, seed=1)
    long_a = make_rmt_traj(200, seed=1)
    long_b = make_rmt_traj(260, seed=2)
    short_c = make_rmt_traj(150, seed=3)   # eligible for legacy can_sample (>=129) but < 200
    ida = buf.insert(long_a); idb = buf.insert(long_b); idc = buf.insert(short_c)
    rng = np.random.RandomState(99)
    batch = buf.sample_eligible(200, rng, 8)
    if batch.status != "OK":
        return FAIL, f"expected OK, got {batch.status}"
    if len(batch.samples) != 8:
        return FAIL, f"batch size {len(batch.samples)} != 8 (not fixed)"
    if batch.eligible_count != 2:
        return FAIL, f"eligible_count {batch.eligible_count} != 2"
    if any(sl != 200 for sl in batch.sequence_lengths):
        return FAIL, "a sample has sequence_length != 200"
    if idc in batch.sample_ids:
        return FAIL, f"short trajectory {idc} (len 150 < 200) was drawn"
    if not all(sid in (ida, idb) for sid in batch.sample_ids):
        return FAIL, f"unexpected sample_ids {batch.sample_ids}"
    # NOT_READY when no trajectory is long enough
    buf2 = RMTReplayBuffer(capacity=64, seed=2)
    buf2.insert(make_rmt_traj(150, seed=4))
    b2 = buf2.sample_eligible(200, np.random.RandomState(1), 4)
    if b2.status != "NOT_READY" or b2.samples:
        return FAIL, f"expected NOT_READY/empty, got {b2.status}/{len(b2.samples)}"
    return PASS, ("8/8 samples length==200 from eligible {200,260}; short(150) never drawn; "
                  "empty eligible -> explicit NOT_READY (no exception, no short substitute)")


# ----------------------------------------------------------------------------
# GATE 10 — same buffer state + RNG => same sample IDs & offsets (bit-reproducible)
# ----------------------------------------------------------------------------

def gate10():
    from rmt_replay_buffer import RMTReplayBuffer

    def build():
        b = RMTReplayBuffer(capacity=64, seed=1)
        b.insert(make_rmt_traj(200, seed=1))
        b.insert(make_rmt_traj(260, seed=2))
        b.insert(make_rmt_traj(150, seed=3))
        return b

    b1 = build()
    x1 = b1.sample_eligible(200, np.random.RandomState(123), 6)
    x2 = b1.sample_eligible(200, np.random.RandomState(123), 6)  # same buffer, reset RNG
    if x1.sample_ids != x2.sample_ids or x1.start_offsets != x2.start_offsets \
            or x1.sequence_lengths != x2.sequence_lengths:
        return FAIL, "same buffer+RNG produced different draws"
    # independence from the buffer's hidden self._rng (must NOT be consumed)
    b1._rng.randint(0, 10 ** 6)
    b1._rng.randint(0, 10 ** 6)
    x3 = b1.sample_eligible(200, np.random.RandomState(123), 6)
    if x3.sample_ids != x1.sample_ids or x3.start_offsets != x1.start_offsets:
        return FAIL, "draw depends on hidden buffer._rng (should only use the passed rng)"
    # a fresh identical buffer reproduces the same draws (state+RNG determinism)
    b2 = build()
    y = b2.sample_eligible(200, np.random.RandomState(123), 6)
    if y.sample_ids != x1.sample_ids or y.start_offsets != x1.start_offsets:
        return FAIL, "fresh identical buffer + RNG did not reproduce draws"
    return PASS, f"deterministic: ids={x1.sample_ids} offsets={x1.start_offsets} (independent of hidden _rng)"


# ----------------------------------------------------------------------------
# GATE 11 — critic/actor KL rollback => policy_version semantics correct
# ----------------------------------------------------------------------------

def gate11():
    from phase4a_v2_counters import Phase4ACounters
    c = Phase4ACounters()
    c.on_outer_update(16, 128); c.on_ppo_accepted()       # policy_version 1
    # a replay update that runs but is KL-rolled-back (policy side reverted)
    c.on_replay_attempt(4)
    c.on_replay_update_executed()
    c.on_replay_kl_rejected()                              # MUST NOT advance policy_version
    if c.policy_version != 1:
        return FAIL, f"KL-rejected replay advanced policy_version to {c.policy_version}"
    if c.accepted_replay_policy_update_count != 0:
        return FAIL, "KL-rejected replay counted as accepted"
    if c.replay_update_count != 1:
        return FAIL, "executed replay update not counted"
    # now a committed replay update
    c.on_replay_attempt(4)
    c.on_replay_update_executed()
    c.on_replay_policy_committed()                         # MUST advance policy_version
    if c.policy_version != 2:
        return FAIL, f"committed replay did not advance policy_version ({c.policy_version})"
    if c.accepted_replay_policy_update_count != 1:
        return FAIL, "committed replay not counted as accepted"
    return PASS, ("KL-rollback: executed+1 but policy_version/accepted unchanged; "
                  "committed: policy_version+1 & accepted+1")


# ----------------------------------------------------------------------------
# GATE 12 — checkpoint carries all required state (static source inspection)
# ----------------------------------------------------------------------------

def gate12():
    src = _read(_LAUNCHER)
    tree = ast.parse(src)
    fn = _func_node(tree, "save_ckpt")
    if fn is None:
        return FAIL, "save_ckpt not found"
    seg = ast.get_source_segment(src, fn)
    required = ['"params"', '"ppo_opt_state"', '"replay_opt_state"', '"target_params"',
                '"replay_buffer"', '"pending"', '"rng"', '"action_rng"', '"memories"',
                '"mem_mask"', '"mem_idx"', '"rmt_state"', '"obsv"', '"counters"',
                '"phase4a_v2"', '"update_count"', '"replay_sequences_consumed"',
                '"replay_sample_rng_state"']
    missing = [k for k in required if k not in seg]
    if missing:
        return FAIL, f"train_state.pkl missing keys: {missing}"
    return PASS, ("checkpoint carries params/PPO opt/Replay opt/EMA(target)/rng/action_rng/"
                  "buffer/pending/GTrXL(memories,mask,idx)/RMT state/all counters + replay RNG state")


# ----------------------------------------------------------------------------
# GATE 13 — off-path (replay_mode=off) stays bit-exact (structural + counter equivalence)
# ----------------------------------------------------------------------------

def gate13():
    from phase4a_v2_counters import Phase4ACounters
    # (a) off-mode counter equivalence with the legacy single update_count
    c = Phase4ACounters()
    legacy = 0
    for u in range(8):
        if c.policy_version != legacy:
            return FAIL, f"iter {u}: policy_version {c.policy_version} != legacy update_count {legacy}"
        if u != legacy:
            return FAIL, f"iter {u}: outer index {u} != legacy update_count {legacy}"
        legacy += 1                       # legacy: PPO increments update_count once per outer
        c.on_outer_update(16, 128)
        c.on_ppo_accepted()
    if c.policy_version != legacy != 8:
        return FAIL, "final off-mode policy_version != legacy update_count"
    if c.replay_update_count or c.accepted_replay_policy_update_count or c.replay_attempt_count:
        return FAIL, "off-mode touched replay counters"
    c.assert_hindsight_awr_disabled()
    # (b) static: replay learner / relabel calls are reachable ONLY under REPLAY_ON branches
    src = _read(_LAUNCHER)
    for call, guard in (("RL.original_vtrace_update_rmt(", 'REPLAY_ON and REPLAY_MODE == "original_vtrace"'),
                        ("RL.full_p2_update_rmt(", 'REPLAY_ON and REPLAY_MODE == "full_p2_legacy"')):
        ic = src.index(call); ig = src.index(guard)
        if not (ig < ic):
            return FAIL, f"{call} not guarded by {guard}"
    ir = src.index("RH.relabel_sample_rmt(")
    if not (src.index('REPLAY_MODE == "full_p2_legacy"') < ir):
        return FAIL, "relabel_sample_rmt reachable outside full_p2_legacy branch"
    return PASS, ("off-mode counters == legacy update_count at every step; replay/relabel calls "
                  "guarded behind REPLAY_ON; changes additive => numeric path unchanged "
                  "(numeric equiv-hash rerun deferred: not authorized this round)")


# ----------------------------------------------------------------------------
# GATE 14 — Persistent/Reset128 config diff differs ONLY in carry_mode
# ----------------------------------------------------------------------------

def gate14():
    import config_diff_validator as V
    ok, report = V.validate()
    if not ok:
        return FAIL, f"config diff violations: {report['violations']}"
    if set(report.get("differing_paths", [])) != {"carry_mode"}:
        return FAIL, f"differing paths != {{carry_mode}}: {report.get('differing_paths')}"
    return PASS, "scientific_config differs only on carry_mode; §六 invariants hold on both arms"


# ----------------------------------------------------------------------------
# GATE 15 — full_p2_legacy requires explicit authorization (behavioral, pre-JAX exit)
# ----------------------------------------------------------------------------

def gate15():
    py = sys.executable
    base = [py, _LAUNCHER, "--carry_mode", "persistent",
            "--ckpt17500", "x", "--out", "y", "--gpu_uuid", "z"]
    # (a) missing --replay_mode -> argparse error (exit 2), before any JAX import
    r1 = subprocess.run(base, capture_output=True, text=True, timeout=120)
    if r1.returncode != 2:
        return FAIL, f"missing --replay_mode exit={r1.returncode} (expected 2); stderr={r1.stderr[-300:]}"
    # (b) full_p2_legacy WITHOUT --allow-full-p2-legacy -> ap.error (exit 2), before JAX import
    r2 = subprocess.run(base + ["--replay_mode", "full_p2_legacy"],
                        capture_output=True, text=True, timeout=120)
    if r2.returncode != 2:
        return FAIL, f"full_p2_legacy w/o flag exit={r2.returncode} (expected 2); stderr={r2.stderr[-300:]}"
    if "allow-full-p2-legacy" not in r2.stderr:
        return FAIL, f"full_p2_legacy rejection message missing; stderr={r2.stderr[-300:]}"
    return PASS, "missing --replay_mode => exit 2; full_p2_legacy without --allow-full-p2-legacy => exit 2 (pre-JAX)"


# ----------------------------------------------------------------------------
# registry + runners
# ----------------------------------------------------------------------------

GATES = [
    ("GATE01_recompute_method", gate01),
    ("GATE02_resolved_step_formula", gate02),
    ("GATE03_counters_not_mixed", gate03),
    ("GATE04_no_hindsight", gate04),
    ("GATE05_no_awr", gate05),
    ("GATE06_single_scan_loss", gate06),
    ("GATE07_seqlen_crosses_boundary", gate07),
    ("GATE08_carry_boundary_tokens", gate08),
    ("GATE09_eligible_only_sampler", gate09),
    ("GATE10_sampler_determinism", gate10),
    ("GATE11_policy_version_kl", gate11),
    ("GATE12_checkpoint_contents", gate12),
    ("GATE13_offpath_bit_exact", gate13),
    ("GATE14_config_univariate", gate14),
    ("GATE15_legacy_requires_auth", gate15),
]


def main():
    print(f"HAVE_JAX={HAVE_JAX}")
    n_pass = n_fail = n_skip = 0
    for name, fn in GATES:
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001
            status, detail = FAIL, f"EXCEPTION: {type(e).__name__}: {e}"
        if status == PASS:
            n_pass += 1
        elif status == SKIP:
            n_skip += 1
        else:
            n_fail += 1
        print(f"[{status}] {name}: {detail}")
    print("=" * 78)
    print(f"SUMMARY pass={n_pass} fail={n_fail} skip={n_skip} (jax gates skip locally, run on server)")
    if n_fail:
        print("GATES_RESULT=FAIL")
        return 1
    print("GATES_RESULT=PASS" if n_skip == 0 else "GATES_RESULT=PASS_LOCAL (jax gates pending server)")
    return 0


# ---- pytest wrappers ----
def _wrap(fn):
    def _t():
        status, detail = fn()
        if status == SKIP:
            import pytest
            pytest.skip(detail)
        assert status == PASS, detail
    return _t


for _name, _fn in GATES:
    globals()[f"test_{_name.lower()}"] = _wrap(_fn)


if __name__ == "__main__":
    sys.exit(main())
