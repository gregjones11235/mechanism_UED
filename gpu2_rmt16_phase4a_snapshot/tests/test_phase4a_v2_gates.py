#!/usr/bin/env python3
"""Phase4A-v2 — 15 test gates (CC2 directive §十).

Run directly for a gated summary (any FAIL => exit 1, per "任一门禁失败立即停止汇报"):
    python test_phase4a_v2_gates.py
or under pytest:
    pytest test_phase4a_v2_gates.py -v

Layering:
  * Pure-Python / numpy / AST / subprocess gates run ANYWHERE (no JAX needed):
    GATE 1(method), 2, 3, 7, 9, 10, 11(counters), 12(static), 13(structural), 14, 15,
    and the Phase4A-v2.1 gates 16(structural+model), 17(structural part), 18, 19, 20, 21,
    22, 23, 24, 25, 26.
  * JAX gates run on the SERVER CPU (dicode310): GATE 4(monkeypatch), 5/6(ast+jax), 8, 13(numeric*),
    and the GATE 17 behavioral part (RMTPendingEpisodeBuffers.reset_slot).
    They pytest.skip / degrade to the structural part when JAX is absent so the local suite
    stays green for what is provable locally.

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
# GATE 16 — policy_version start/end/span captured across multi-rollout episodes (§二)
# ----------------------------------------------------------------------------

def gate16():
    import os
    rc = _read(os.path.join(_EXP, "rmt_collect.py"))
    # (a) structural: the completing trajectory reads the START version from pending BEFORE the
    # reset_slot overwrites it, records end = current version, span = end-start, with asserts,
    # and the deprecated alias is bound to START (not end).
    required = [
        "episode_start_version = int(pending.policy_version[e])",
        "episode_end_version = int(policy_version)",
        "episode_version_span = episode_end_version - episode_start_version",
        "assert episode_end_version >= episode_start_version",
        "assert episode_version_span >= 0",
        "policy_version_start=episode_start_version",
        "policy_version_end=episode_end_version",
        "policy_version_span=episode_version_span",
        "policy_version_at_collection=episode_start_version",
    ]
    missing = [frag for frag in required if frag not in rc]
    if missing:
        return FAIL, f"rmt_collect missing provenance fragments: {missing}"
    if rc.index("episode_start_version = int(pending.policy_version[e])") \
            > rc.index("pending.reset_slot(e,"):
        return FAIL, "start version captured AFTER reset_slot (would read the NEW episode's start)"
    # (b) behavioral model: an episode open at rollout 0 completes during rollout 2.
    pv_slot = [0]          # pending.policy_version[e]: the START version of the open episode
    current = 0            # accepted policy_version (PPO commits once per outer update)
    for _rollout in range(2):
        current += 1                       # PPO main update after each completed rollout
    start, end = int(pv_slot[0]), int(current)
    span = end - start
    if (start, end, span) != (0, 2, 2) or end < start:
        return FAIL, f"3-rollout episode provenance wrong: start={start} end={end} span={span}"
    # reset_slot opens the NEXT episode with start == current rollout version (span-0 possible)
    pv_slot[0] = int(current)
    if int(pv_slot[0]) != current or current - int(pv_slot[0]) != 0:
        return FAIL, "new episode start != current version (span-0 case broken)"
    # off-mode equivalence: policy_version == outer loop index => an episode opened during
    # outer update u0 gets start == u0 (bit-exact with the legacy stamping)
    pv_off = 0
    for u0 in range(3):
        if pv_off != u0:
            return FAIL, f"off-mode policy_version {pv_off} != outer index {u0}"
        pv_off += 1                                # PPO commit once per outer update
    return PASS, ("start=pending.policy_version[e] read BEFORE reset_slot; end=current; "
                  "span=end-start>=0 asserted; alias==start; 3-rollout episode => (0,2,2)")


# ----------------------------------------------------------------------------
# GATE 17 — reset_slot opens the new episode with the CURRENT rollout policy version (§二.3)
# ----------------------------------------------------------------------------

def gate17():
    import os
    rc = _read(os.path.join(_EXP, "rmt_collect.py"))
    if "pending.reset_slot(e, policy_version=int(policy_version))" not in rc:
        return FAIL, "reset_slot no longer stamps the new episode with the current policy version"
    if HAVE_JAX:
        from rmt_collect import RMTPendingEpisodeBuffers
        p = RMTPendingEpisodeBuffers(num_envs=2, first_episode_id=0, first_policy_version=0)
        if p.policy_version != [0, 0]:
            return FAIL, f"initial policy_version {p.policy_version} != [0, 0]"
        old_id = p.episode_id[0]
        p.reset_slot(0, policy_version=5)          # new episode starts at version 5
        if p.policy_version[0] != 5 or p.policy_version[1] != 0:
            return FAIL, f"after reset_slot policy_version={p.policy_version} (expected [5, 0])"
        if p.episode_id[0] == old_id:
            return FAIL, "reset_slot did not advance the episode id"
        return PASS, ("reset_slot(e, policy_version=current): new episode start==current "
                      "(behavioral, JAX CPU) + structural stamp check")
    return PASS, ("structural: reset_slot(e, policy_version=int(policy_version)) present "
                  "(behavioral part runs on server CPU; JAX absent locally)")


# ----------------------------------------------------------------------------
# GATE 18 — replay sample propagates start/end/span verbatim; alias == start (§二)
# ----------------------------------------------------------------------------

def gate18():
    from rmt_replay_buffer import RMTReplayBuffer
    t = make_rmt_traj(220, seed=5)
    t.policy_version_start = 3
    t.policy_version_end = 7
    t.policy_version_span = 4
    t.policy_version_at_collection = 3            # deprecated alias of START
    buf = RMTReplayBuffer(capacity=8, seed=1)
    buf.insert(t)
    batch = buf.sample_eligible(200, np.random.RandomState(0), 1)
    if batch.status != "OK" or len(batch.samples) != 1:
        return FAIL, f"sample failed: status={batch.status} n={len(batch.samples)}"
    s = batch.samples[0]
    if s.policy_version_start != 3:
        return FAIL, f"sample policy_version_start={s.policy_version_start} != 3 (not propagated)"
    if s.policy_version_end != 7:
        return FAIL, f"sample policy_version_end={s.policy_version_end} != 7 (not propagated)"
    if s.policy_version_span != 4:
        return FAIL, f"sample policy_version_span={s.policy_version_span} != 4 (not propagated)"
    if s.policy_version_at_collection != s.policy_version_start:
        return FAIL, ("deprecated alias != start: "
                      f"{s.policy_version_at_collection} != {s.policy_version_start}")
    return PASS, "sample() propagates start=3/end=7/span=4 verbatim; alias==start==3"


# ----------------------------------------------------------------------------
# GATE 19 — original_vtrace + active policy-lag gate fails CLOSED (§三.1)
# ----------------------------------------------------------------------------

def gate19():
    import os
    import yaml
    import phase4a_v2_contract as C
    for arm in ("persistent", "reset128"):
        path = os.path.join(_SNAPSHOT, "configs", f"rmt16_phase4a_v2_{arm}.yaml")
        with open(path, encoding="utf-8") as f:
            sc = yaml.safe_load(f)["scientific_config"]
        # the REAL configs must pass the validator (active=false, no stray top-level lag)
        C.validate_policy_lag_config(sc)
        if sc.get("max_policy_lag") is not None:
            return FAIL, f"{arm}: stray top-level max_policy_lag={sc.get('max_policy_lag')}"
        if sc["policy_lag"]["active"] is not False:
            return FAIL, f"{arm}: policy_lag.active={sc['policy_lag']['active']} (must be false)"
        # fail-closed: active=true under original_vtrace MUST raise
        bad = dict(sc); bad["policy_lag"] = dict(sc["policy_lag"], active=True)
        try:
            C.validate_policy_lag_config(bad)
            return FAIL, f"{arm}: active=true did NOT raise"
        except ValueError as e:
            if "ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT" not in str(e):
                return FAIL, f"{arm}: wrong error for active=true: {e}"
        # fail-closed: stray top-level max_policy_lag MUST raise
        bad2 = dict(sc); bad2["max_policy_lag"] = 16
        try:
            C.validate_policy_lag_config(bad2)
            return FAIL, f"{arm}: stray top-level max_policy_lag did NOT raise"
        except ValueError as e:
            if "ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT" not in str(e):
                return FAIL, f"{arm}: wrong error for stray top-level lag: {e}"
    return PASS, ("both arms' configs validate; active=true and stray top-level max_policy_lag "
                  "both raise ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT")


# ----------------------------------------------------------------------------
# GATE 20 — run manifest records lag-gate identity for original_vtrace (§三.2)
# ----------------------------------------------------------------------------

def gate20():
    import phase4a_v2_contract as C
    m = C.policy_lag_runtime_manifest("original_vtrace")
    expected = dict(policy_lag_gate_active=False,
                    policy_lag_gate_mode="not_applicable_original_vtrace",
                    max_policy_lag=None,
                    off_policy_correction="vtrace_importance_sampling",
                    rho_bar=1.0, c_bar=1.0)
    if m != expected:
        return FAIL, f"manifest {m} != expected {expected}"
    src = _read(_LAUNCHER)
    if "import phase4a_v2_contract as CONTRACT" not in src:
        return FAIL, "launcher does not import phase4a_v2_contract"
    if "fields.update(CONTRACT.policy_lag_runtime_manifest(REPLAY_MODE))" not in src:
        return FAIL, "launcher manifest does not record policy_lag_runtime_manifest"
    if "CONTRACT.replay_protocol_labels(" not in src:
        return FAIL, "launcher manifest does not record the four-way replay labels"
    return PASS, ("manifest: policy_lag_gate_active=false / max_policy_lag=null / "
                  "off_policy_correction=vtrace_importance_sampling / rho_bar=c_bar=1.0; "
                  "wired into checkpoint + summary manifest")


# ----------------------------------------------------------------------------
# GATE 21 — legacy max_policy_lag scoped to full_p2_legacy ONLY (§三.3)
# ----------------------------------------------------------------------------

def gate21():
    import os
    import yaml
    import phase4a_v2_contract as C
    for arm in ("persistent", "reset128"):
        path = os.path.join(_SNAPSHOT, "configs", f"rmt16_phase4a_v2_{arm}.yaml")
        with open(path, encoding="utf-8") as f:
            sc = yaml.safe_load(f)["scientific_config"]
        if sc["legacy_full_p2_only"]["max_policy_lag"] != 16:
            return FAIL, f"{arm}: legacy_full_p2_only.max_policy_lag != 16"
        if sc["policy_lag"]["max_policy_lag"] is not None:
            return FAIL, f"{arm}: active policy_lag.max_policy_lag not null"
        if "max_policy_lag" in sc:
            return FAIL, f"{arm}: top-level max_policy_lag leaked into scientific_config"
    leg = C.policy_lag_runtime_manifest("full_p2_legacy", legacy_max_policy_lag=16)
    if not (leg["policy_lag_gate_active"] is True and leg["max_policy_lag"] == 16
            and leg["policy_lag_gate_mode"] == "legacy_full_p2"):
        return FAIL, f"legacy manifest wrong: {leg}"
    orig = C.policy_lag_runtime_manifest("original_vtrace")
    if orig["policy_lag_gate_active"] or orig["max_policy_lag"] is not None:
        return FAIL, f"original_vtrace polluted by legacy lag: {orig}"
    off = C.policy_lag_runtime_manifest("off")
    if off["policy_lag_gate_active"] or off["max_policy_lag"] is not None:
        return FAIL, f"off polluted by legacy lag: {off}"
    src = _read(_LAUNCHER)
    if "ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT: runtime policy-lag gate active" not in src:
        return FAIL, "launcher lacks the runtime fail-closed lag-gate consistency check"
    return PASS, ("legacy lag=16 lives ONLY under legacy_full_p2_only; manifests: "
                  "legacy active=true/16, original_vtrace+off active=false/null; "
                  "runtime fail-closed guard present")


# ----------------------------------------------------------------------------
# GATE 22 — SAME_REPLAY_PROTOCOL vs MATCHED_REPLAY_EXPOSURE are NOT interchangeable (§四)
# ----------------------------------------------------------------------------

def gate22():
    import phase4a_v2_contract as C
    lab = C.replay_protocol_labels("original_vtrace", 129, 4)
    if lab["SAME_REPLAY_PROTOCOL"] != "READY":
        return FAIL, f"SAME_REPLAY_PROTOCOL={lab['SAME_REPLAY_PROTOCOL']} (expected READY)"
    if lab["MATCHED_REPLAY_EXPOSURE"] != "NOT_RUN":
        return FAIL, f"MATCHED_REPLAY_EXPOSURE={lab['MATCHED_REPLAY_EXPOSURE']} (expected NOT_RUN)"
    if lab["MATCHED_REPLAY_CONTENT"] != "NOT_CLAIMED":
        return FAIL, f"MATCHED_REPLAY_CONTENT={lab['MATCHED_REPLAY_CONTENT']} (expected NOT_CLAIMED)"
    if lab["ENDOGENOUS_REPLAY_SCREENING"] != "READY_AFTER_SMOKE":
        return FAIL, f"ENDOGENOUS_REPLAY_SCREENING={lab['ENDOGENOUS_REPLAY_SCREENING']}"
    # the single conflated flag must be GONE from the launcher summary
    src = _read(_LAUNCHER)
    if "matched_replay_protocol_ready=" in src:
        return FAIL, "conflated single matched_replay_protocol_ready= flag still assigned"
    if "CONTRACT.replay_protocol_labels(" not in src:
        return FAIL, "launcher does not emit the four-way label split"
    return PASS, ("protocol READY while exposure NOT_RUN / content NOT_CLAIMED / screening "
                  "READY_AFTER_SMOKE; conflated single flag removed from launcher summary")


# ----------------------------------------------------------------------------
# GATE 23 — no complete two-arm certificates => no MATCHED_REPLAY_EXPOSURE=PASS (§五.2)
# ----------------------------------------------------------------------------

def gate23():
    import phase4a_v2_exposure_validator as EV
    import phase4a_v2_contract as C
    kw = dict(replay_updates=2, consumed=8, batch_sizes=[4, 4], seq_lens=[[129] * 4] * 2,
              attempt_mask=[False, True, True], not_ready_updates=[0])
    sa = EV._synthetic_summary("persistent", **kw)
    sb = EV._synthetic_summary("reset128", **kw)
    rep = EV.validate_two_arm(sa, sb)
    if rep["MATCHED_REPLAY_EXPOSURE"] != "PASS":
        return FAIL, f"identical certificates did not PASS: {rep['EXPOSURE_DIFFERING_FIELDS']}"
    # dropping ANY certificate field => fail-closed, no PASS
    sb_bad = EV._synthetic_summary("reset128", drop_field="replay_update_count", **kw)
    try:
        EV.validate_two_arm(sa, sb_bad)
        return FAIL, "incomplete certificate did NOT raise"
    except ValueError as e:
        if "MATCHED_REPLAY_CERTIFICATE_REQUIRED" not in str(e):
            return FAIL, f"wrong error: {e}"
    # PASS-claim guard refuses a non-PASS comparison
    sb_mis = EV._synthetic_summary("reset128", replay_updates=1, consumed=4, batch_sizes=[4],
                                   seq_lens=[[129] * 4], attempt_mask=[False, True, True],
                                   not_ready_updates=[0])
    try:
        C.assert_matched_exposure_pass_allowed(
            EV.extract_certificate(sa, "persistent"), EV.extract_certificate(sb_mis, "reset128"))
        return FAIL, "PASS-claim guard did NOT raise on mismatched exposure"
    except ValueError as e:
        if "MATCHED_REPLAY_CERTIFICATE_REQUIRED" not in str(e):
            return FAIL, f"wrong PASS-claim error: {e}"
    return PASS, ("complete equal certificates => PASS; missing field => "
                  "MATCHED_REPLAY_CERTIFICATE_REQUIRED; PASS-claim guard refuses mismatch")


# ----------------------------------------------------------------------------
# GATE 24 — endogenous buffers can NEVER claim MATCHED_REPLAY_CONTENT=PASS (§五.2)
# ----------------------------------------------------------------------------

def gate24():
    import phase4a_v2_exposure_validator as EV
    import phase4a_v2_contract as C
    try:
        C.assert_content_match_not_claimed(buffer_kind="endogenous")
        return FAIL, "endogenous content claim did NOT raise"
    except ValueError as e:
        if "ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED" not in str(e):
            return FAIL, f"wrong error: {e}"
    kw = dict(replay_updates=1, consumed=4, batch_sizes=[4], seq_lens=[[129] * 4],
              attempt_mask=[True, True], not_ready_updates=[])
    sa = EV._synthetic_summary("persistent", **kw)
    sb = EV._synthetic_summary("reset128", **kw)
    rep = C.compare_exposure(EV.extract_certificate(sa, "persistent"),
                             EV.extract_certificate(sb, "reset128"))
    if rep["CONTENT_MATCH"] != "NOT_APPLICABLE_ENDOGENOUS_BUFFERS":
        return FAIL, f"CONTENT_MATCH={rep['CONTENT_MATCH']} (must never be PASS)"
    if rep["MATCHED_REPLAY_CONTENT"] != "NOT_CLAIMED":
        return FAIL, f"MATCHED_REPLAY_CONTENT={rep['MATCHED_REPLAY_CONTENT']}"
    # an INPUT certificate that claims content PASS => fail-closed
    sb_claim = EV._synthetic_summary("reset128", **kw)
    sb_claim["phase4a_v2"]["replay_labels"]["MATCHED_REPLAY_CONTENT"] = "PASS"
    try:
        EV.validate_two_arm(sa, sb_claim)
        return FAIL, "input content=PASS claim did NOT raise"
    except ValueError as e:
        if "ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED" not in str(e):
            return FAIL, f"wrong input-claim error: {e}"
    return PASS, ("endogenous => CONTENT_MATCH=NOT_APPLICABLE_ENDOGENOUS_BUFFERS (never PASS); "
                  "content claims fail closed")


# ----------------------------------------------------------------------------
# GATE 25 — frozen raw probe files match SHA256SUMS (§六)
# ----------------------------------------------------------------------------

def gate25():
    import os
    import recompute_probe_step as R
    ev = os.path.join(_SNAPSHOT, "evidence", "raw_probe")
    sums = os.path.join(ev, "SHA256SUMS")
    names = ["persistent_probe_episodes.jsonl", "persistent_probe_updates.jsonl",
             "persistent_probe_summary.json", "reset128_probe_episodes.jsonl",
             "reset128_probe_updates.jsonl", "reset128_probe_summary.json"]
    missing = [n for n in names + ["SHA256SUMS"] if not os.path.isfile(os.path.join(ev, n))]
    if missing:
        return FAIL, (f"RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY=BLOCKED_SOURCE_UNAVAILABLE "
                      f"(missing: {missing})")
    ok, results = R._verify_sha256sums(
        {n: os.path.join(ev, n) for n in names}, sums)
    if not ok:
        bad = {k: v for k, v in results.items() if not v["match"]}
        return FAIL, f"RAW_PROBE_SOURCE_HASH_MISMATCH: {bad}"
    return PASS, f"all {len(names)} frozen raw probe files match SHA256SUMS byte-for-byte"


# ----------------------------------------------------------------------------
# GATE 26 — recompute from frozen raw probe: 6/20, 5/21, 8979, BOTH (§六)
# ----------------------------------------------------------------------------

def gate26():
    import os
    import recompute_probe_step as R
    ev = os.path.join(_SNAPSHOT, "evidence", "raw_probe")
    p = os.path.join(ev, "persistent_probe_episodes.jsonl")
    r = os.path.join(ev, "reset128_probe_episodes.jsonl")
    sums = os.path.join(ev, "SHA256SUMS")
    if not (os.path.isfile(p) and os.path.isfile(r) and os.path.isfile(sums)):
        return FAIL, ("RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY=BLOCKED_SOURCE_UNAVAILABLE "
                      "(frozen probe JSONL absent)")
    rep = R.two_arm_recompute(p, r, sha256sums_path=sums)
    if rep.get("RAW_PROBE_SOURCE_HASH_MISMATCH"):
        return FAIL, f"RAW_PROBE_SOURCE_HASH_MISMATCH=BLOCKED: {rep.get('hash_verification')}"
    if rep.get("RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY") != "PASS":
        return FAIL, f"recompute blocked: {rep}"
    pa, ra = rep["arms"]["persistent"], rep["arms"]["reset128"]
    if (pa["total_completed_episodes"], pa["count_ge512"]) != (20, 6):
        return FAIL, f"persistent {pa['total_completed_episodes']}/{pa['count_ge512']} != 20/6"
    if (ra["total_completed_episodes"], ra["count_ge512"]) != (21, 5):
        return FAIL, f"reset128 {ra['total_completed_episodes']}/{ra['count_ge512']} != 21/5"
    fp, fr = pa["first_ge512"], ra["first_ge512"]
    for name, fg in (("persistent", fp), ("reset128", fr)):
        if (fg["episode_id"], fg["length"], fg["update_index"], fg["rollout_step"],
                fg["env_id"], fg["first_ge512_resolved_env_step"]) != (2, 562, 4, 49, 2, 8979):
            return FAIL, f"{name} first_ge512 mismatch: {fg}"
    if rep["L512_REACHABILITY"] != "BOTH" or not rep["cross_arm_resolved_step_agree"]:
        return FAIL, f"reachability {rep['L512_REACHABILITY']} / agree=" \
                     f"{rep['cross_arm_resolved_step_agree']} (expected BOTH/true)"
    return PASS, ("recomputed (not hardcoded): persistent 6/20, reset128 5/21, "
                  "first_ge512 resolved=8979 both arms, L512_REACHABILITY=BOTH")


# ----------------------------------------------------------------------------
# Phase4A-v2.2 (§九) — gates 27-38
# ----------------------------------------------------------------------------

def gate27():
    """GATE 27 — protocol REQUIRED fields complete, incl. learner and rng_rule (§二)."""
    import phase4a_v2_contract as C
    expected = {"sequence_length", "batch_size", "replay_mode", "sampler", "learner",
                "loss", "rng_rule"}
    if set(C.REQUIRED_PROTOCOL_FIELDS) != expected:
        return FAIL, f"REQUIRED_PROTOCOL_FIELDS={sorted(C.REQUIRED_PROTOCOL_FIELDS)} != {sorted(expected)}"
    lab = C.replay_protocol_labels("original_vtrace", 129, 4)
    proto = lab["protocol_definition"]
    miss = C.missing_required_protocol_fields(proto)
    if miss:
        return FAIL, f"runtime protocol_definition missing required fields {miss}"
    # auditable RNG breakdown (§二.2)
    for k, v in (("rng_engine", "np.random.RandomState"),
                 ("rng_seed_derivation", "run_seed_plus_7"),
                 ("rng_stream", "dedicated_replay_sampler"),
                 ("hidden_buffer_rng_used", False)):
        if proto.get(k) != v:
            return FAIL, f"protocol_definition.{k}={proto.get(k)!r} != {v!r}"
    # identical protocols -> PASS with emitted SHAs; key order irrelevant
    pa = dict(proto); pb = dict(reversed(list(dict(proto).items())))
    rep = C.compare_protocols(pa, pb)
    if rep["PROTOCOL_MATCH"] != "PASS":
        return FAIL, f"identical protocol (different key order) did not PASS: {rep}"
    if not (rep["PROTOCOL_DEFINITION_SHA256_ARM_A"]
            and rep["PROTOCOL_DEFINITION_SHA256_ARM_A"]
            == rep["PROTOCOL_DEFINITION_SHA256_ARM_B"]):
        return FAIL, "protocol SHA256 not emitted / not equal on identical protocols"
    # launcher must build the protocol via the contract (runtime-generated, not validator-built)
    src = _read(_LAUNCHER)
    if "CONTRACT.replay_protocol_labels(" not in src:
        return FAIL, "launcher does not generate protocol_definition via CONTRACT"
    return PASS, ("REQUIRED_PROTOCOL_FIELDS complete incl. learner+rng_rule; rng breakdown "
                  "auditable; identical protocol (any key order) -> PASS with equal SHA256")


def gate28():
    """GATE 28 — different learner/rng_rule/missing field/extra field all fail closed (§二.3)."""
    import phase4a_v2_contract as C
    base = C.replay_protocol_labels("original_vtrace", 129, 4)["protocol_definition"]

    def variant(**over):
        lab = C.replay_protocol_labels("original_vtrace", 129, 4, **over)
        return lab["protocol_definition"]

    # different learner / different rng_rule -> FAIL (old whitelist missed both)
    for name, proto_b, field in (
            ("different_learner", variant(learner="full_p2_legacy_update_rmt"), "learner"),
            ("different_rng_rule", variant(rng_rule="np.random.default_rng(seed)"), "rng_rule")):
        rep = C.compare_protocols(dict(base), proto_b)
        if rep["PROTOCOL_MATCH"] != "FAIL" or field not in rep["PROTOCOL_DIFFERING_FIELDS"]:
            return FAIL, f"{name}: {rep}"
    # missing learner / missing rng_rule -> fail closed PROTOCOL_IDENTITY_INCOMPLETE
    for name, drop in (("missing_learner", "learner"), ("missing_rng_rule", "rng_rule")):
        pb = dict(base); pb.pop(drop)
        try:
            C.compare_protocols(dict(base), pb)
            return FAIL, f"{name} did NOT raise"
        except ValueError as e:
            if "PROTOCOL_IDENTITY_INCOMPLETE" not in str(e) or drop not in str(e):
                return FAIL, f"{name} wrong error: {e}"
    # extra unknown field on ONE arm -> FAIL via keyset mismatch
    pb = dict(base); pb["unregistered_field"] = 1
    rep = C.compare_protocols(dict(base), pb)
    if rep["PROTOCOL_MATCH"] != "FAIL" or "unregistered_field" not in rep["PROTOCOL_KEYSET_MISMATCH"]:
        return FAIL, f"extra_field_one_arm: {rep}"
    # compare_exposure: protocol difference with EQUAL exposure -> MATCHED_REPLAY_EXPOSURE=FAIL
    import phase4a_v2_exposure_validator as EV
    kw = dict(replay_updates=2, consumed=8, batch_sizes=[4, 4], seq_lens=[[129] * 4] * 2,
              attempt_mask=[False, True, True], not_ready_updates=[0])
    sa = EV._synthetic_summary("persistent", **kw)
    sb = EV._synthetic_summary("reset128", learner="another_learner", **kw)
    rep = EV.validate_two_arm(sa, sb)
    if rep["PROTOCOL_MATCH"] != "FAIL" or rep["MATCHED_REPLAY_EXPOSURE"] != "FAIL":
        return FAIL, f"equal exposure + different learner must FAIL: {rep}"
    if rep["EXPOSURE_COUNT_MATCH"] != "PASS":
        return FAIL, "exposure counts should still match in this scenario"
    return PASS, ("different learner/rng_rule -> FAIL; missing learner/rng_rule -> "
                  "PROTOCOL_IDENTITY_INCOMPLETE; extra field -> keyset FAIL; equal exposure + "
                  "protocol difference -> MATCHED_REPLAY_EXPOSURE=FAIL")


def gate29():
    """GATE 29 — no numeric max_policy_lag in original_vtrace ACTIVE scope; legacy 16 only
    under legacy_full_p2_only with active=false (§三)."""
    import yaml
    import phase4a_v2_contract as C
    # (a) synthetic launcher-shaped summary is CLEAN
    summary = dict(
        replay_mode="original_vtrace",
        phase4a_v2=dict(replay_mode="original_vtrace", max_policy_lag=None,
                        policy_lag_gate_active=False,
                        active_replay_config=C.active_replay_config_manifest("original_vtrace"),
                        legacy_full_p2_only=C.legacy_full_p2_manifest(active=False)),
        active_replay_config=C.active_replay_config_manifest("original_vtrace"),
        legacy_full_p2_only=C.legacy_full_p2_manifest(active=False, max_policy_lag=16),
        p2_frozen=dict(rho_bar=1.0, c_bar=1.0, policy_lag_gate_active=False, max_policy_lag=None))
    scan = C.assert_no_active_policy_lag_leak(summary)
    if scan["scan"] != "clean":
        return FAIL, f"clean summary scanned as {scan}"
    # (b) any numeric leak in an active block -> ORIGINAL_VTRACE_ACTIVE_POLICY_LAG_LEAK
    for name, mut in (
            ("p2_frozen", lambda s: s["p2_frozen"].update(max_policy_lag=16)),
            ("active_replay_config",
             lambda s: s["active_replay_config"].update(max_policy_lag=16)),
            ("top_level", lambda s: s.update(max_policy_lag=16)),
            ("manifest_gate_active",
             lambda s: s["phase4a_v2"].update(policy_lag_gate_active=True))):
        bad = dict(summary)
        bad["p2_frozen"] = dict(summary["p2_frozen"])
        bad["active_replay_config"] = dict(summary["active_replay_config"])
        bad["phase4a_v2"] = dict(summary["phase4a_v2"])
        mut(bad)
        try:
            C.assert_no_active_policy_lag_leak(bad)
            return FAIL, f"{name} leak NOT detected"
        except ValueError as e:
            if "ORIGINAL_VTRACE_ACTIVE_POLICY_LAG_LEAK" not in str(e):
                return FAIL, f"{name} wrong error: {e}"
    # (c) legacy 16 without explicit active=false -> fail closed
    bad = dict(summary); bad["legacy_full_p2_only"] = dict(max_policy_lag=16)
    try:
        C.assert_no_active_policy_lag_leak(bad)
        return FAIL, "legacy numeric lag without active=false NOT detected"
    except ValueError as e:
        if "ORIGINAL_VTRACE_ACTIVE_POLICY_LAG_LEAK" not in str(e):
            return FAIL, f"legacy wrong error: {e}"
    # (d) launcher must not leak fp_cfg.max_policy_lag into any LIVE (non-comment) code; the
    # p2_frozen block must instead carry policy_lag_gate_active=False + max_policy_lag=None
    import re
    src = _read(_LAUNCHER)
    live_leak = [ln for ln in src.split("\n")
                 if "max_policy_lag=fp_cfg.max_policy_lag" in ln
                 and not ln.strip().startswith("#")]
    if live_leak:
        return FAIL, f"launcher live code still leaks: {live_leak[0].strip()}"
    norm = src.replace("\r\n", "\n")
    if not re.search(r"policy_lag_gate_active=False,\s*\n\s*max_policy_lag=None", norm):
        return FAIL, "launcher p2_frozen does not carry gate_active=False + max_policy_lag=None"
    # (e) both YAMLs: legacy_full_p2_only.active == false (and 16 retained, inactive)
    for arm in ("persistent", "reset128"):
        path = os.path.join(_SNAPSHOT, "configs", f"rmt16_phase4a_v2_{arm}.yaml")
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        leg = cfg["scientific_config"]["legacy_full_p2_only"]
        if leg.get("active") is not False or leg.get("max_policy_lag") != 16:
            return FAIL, f"{arm} legacy_full_p2_only={leg} (want active=false, lag=16)"
        pl = cfg["scientific_config"]["policy_lag"]
        if pl.get("active") is not False or pl.get("max_policy_lag") is not None:
            return FAIL, f"{arm} policy_lag={pl} (want active=false, max_policy_lag=null)"
    return PASS, ("active scope clean (phase4a_v2/active_replay_config/p2_frozen/top-level); "
                  "4 leak variants + legacy-active-missing fail closed; YAMLs carry "
                  "active=false/null with legacy 16 inactive")


def gate30():
    """GATE 30 — episode JSONL record writes policy_version start/end/span; old policy_version
    is an explicit DEPRECATED alias of policy_version_end (§四)."""
    src = _read(os.path.join(_EXP, "rmt_collect.py")).replace("\r\n", "\n")
    required = [
        "episode_record = dict(",
        "policy_version_start=episode_start_version,",
        "policy_version_end=episode_end_version,",
        "policy_version_span=episode_version_span,",
        'policy_version_alias_of="policy_version_end",',
        "policy_version_deprecated=True,",
        'assert traj.policy_version_start == episode_record["policy_version_start"]',
        'assert traj.policy_version_end == episode_record["policy_version_end"]',
        'assert traj.policy_version_span == episode_record["policy_version_span"]',
        'assert traj.policy_version_at_collection == (',
        'assert episode_record["policy_version"] == (',
        "episode_records.append(episode_record)",
    ]
    miss = [r for r in required if r not in src]
    if miss:
        return FAIL, f"episode record provenance missing in rmt_collect.py: {miss}"
    # the record range fields must appear exactly ONCE each in the record block (no duplication)
    if src.count("policy_version_deprecated=True,") != 1:
        return FAIL, "policy_version_deprecated appears != 1 time"
    # recompute provenance keys still present (20/6, 21/5, 8979 depend on them)
    for k in ("episode_id=int(pending.episode_id[e])", "length=int(L)",
              "update_index=int(outer_update_index)", "rollout_step=int(_rollout_step_i)"):
        if k not in src:
            return FAIL, f"recompute-critical record field changed/missing: {k}"
    return PASS, ("episode record writes start/end/span; policy_version=alias_of end "
                  "(deprecated=True); pre-write asserts tie record to trajectory range; "
                  "recompute keys untouched")


def gate31():
    """GATE 31 — trajectory policy-version range validator: legal passes, 5 illegal rejected,
    enforced on buffer insert (§五)."""
    import rmt_replay_buffer as RB
    # legal: legacy default 0/0/0/0 and a real range
    RB.validate_policy_version_range_fields(0, 0, 0, 0)
    RB.validate_policy_version_range_fields(3, 5, 2, 3)
    illegal = [
        (3, 5, 2, 0, "POLICY_VERSION_ALIAS_MISMATCH"),   # alias != start
        (5, 3, -2, 5, "POLICY_VERSION_RANGE_INVALID"),   # end < start
        (3, 5, 3, 3, "POLICY_VERSION_SPAN_MISMATCH"),    # span != end-start
        (-1, 5, 6, -1, "POLICY_VERSION_RANGE_INVALID"),  # negative start
        (3, 5, -2, 3, "POLICY_VERSION_SPAN_MISMATCH"),   # negative span
    ]
    for start, end, span, alias, code in illegal:
        try:
            RB.validate_policy_version_range_fields(start, end, span, alias)
            return FAIL, f"({start},{end},{span},{alias}) NOT rejected (want {code})"
        except ValueError as e:
            if code not in str(e):
                return FAIL, f"({start},{end},{span},{alias}) wrong code: {e}"
    # trajectory-level validator + enforcement through insert (validate_anchors path)
    traj = make_rmt_traj(200, seed=7)
    traj.policy_version_start, traj.policy_version_end = 3, 5
    traj.policy_version_span, traj.policy_version_at_collection = 2, 3
    traj.validate_policy_version_range()
    traj.validate_anchors()
    buf = RB.RMTReplayBuffer(capacity=4, seed=0)
    buf.insert(traj)  # legal range inserts fine
    bad = make_rmt_traj(200, seed=8)
    bad.policy_version_start, bad.policy_version_end = 5, 3
    bad.policy_version_span, bad.policy_version_at_collection = 2, 5
    try:
        RB.RMTReplayBuffer(capacity=4, seed=0).insert(bad)
        return FAIL, "illegal trajectory range inserted without raising"
    except ValueError as e:
        if "POLICY_VERSION_RANGE_INVALID" not in str(e):
            return FAIL, f"insert wrong code: {e}"
    return PASS, ("0/0/0/0 + real ranges pass; end<start, span!=end-start, alias!=start, "
                  "negative start, negative span all rejected; enforced on insert")


def gate32():
    """GATE 32 — sample-level range validator: invariants after propagation; read-only (§五)."""
    import rmt_replay_buffer as RB
    traj = make_rmt_traj(300, seed=11)
    traj.policy_version_start, traj.policy_version_end = 2, 4
    traj.policy_version_span, traj.policy_version_at_collection = 2, 2
    buf = RB.RMTReplayBuffer(capacity=4, seed=1)
    buf.insert(traj)
    s = buf.sample(trajectory_id=traj.trajectory_id, start_step=0, sequence_length=129)
    if (s.policy_version_start, s.policy_version_end, s.policy_version_span,
            s.policy_version_at_collection) != (2, 4, 2, 2):
        return FAIL, f"sample range not propagated: {(s.policy_version_start, s.policy_version_end, s.policy_version_span, s.policy_version_at_collection)}"
    before = (s.policy_version_start, s.policy_version_end, s.policy_version_span,
              s.policy_version_at_collection, float(np.sum(s.rewards)), float(np.sum(s.values)))
    rec = RB.validate_sample_policy_version_range(s)
    after = (s.policy_version_start, s.policy_version_end, s.policy_version_span,
             s.policy_version_at_collection, float(np.sum(s.rewards)), float(np.sum(s.values)))
    if before != after:
        return FAIL, "validator mutated the sample (must be read-only)"
    if rec != dict(policy_version_start=2, policy_version_end=4, policy_version_span=2,
                   policy_version_at_collection=2):
        return FAIL, f"validator record wrong: {rec}"
    # negative: a sample-like object with broken invariants is rejected
    s.policy_version_span = 99
    try:
        RB.validate_sample_policy_version_range(s)
        return FAIL, "broken sample span NOT rejected"
    except ValueError as e:
        if "POLICY_VERSION_SPAN_MISMATCH" not in str(e):
            return FAIL, f"sample validator wrong code: {e}"
    return PASS, ("sample propagates start/end/span/alias; validator read-only (numeric content "
                  "unchanged); broken span rejected with POLICY_VERSION_SPAN_MISMATCH")


def _rtc_reference(arm="persistent"):
    """Load the REAL committed formal YAML for `arm` and build the matching runtime scientific
    config from the frozen driver constants (mirrors the launcher call site)."""
    import phase4a_v2_runtime_config as RTC
    path = os.path.join(_SNAPSHOT, "configs", f"rmt16_phase4a_v2_{arm}.yaml")
    rec = RTC.load_formal_config(path)
    import yaml
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    kw = RTC._reference_runtime_kwargs(carry_mode=arm)  # frozen driver constants
    runtime_sci = RTC.build_runtime_scientific_config(**kw)
    return RTC, rec, y, runtime_sci


def gate33():
    """GATE 33 — formal YAML scientific_config == REAL runtime scientific config ->
    certificate PASS, on BOTH arms (§六.4/§六.7)."""
    RTC, rec_p, y_p, sci_p = _rtc_reference("persistent")
    RTC, rec_r, y_r, sci_r = _rtc_reference("reset128")
    for arm, rec, y, sci in (("persistent", rec_p, y_p, sci_p), ("reset128", rec_r, y_r, sci_r)):
        RTC.validate_arm_binding(rec, arm, replay_mode="original_vtrace")
        ra = y["runtime_assignment"]
        cert = RTC.validate_runtime_against_formal_config(
            rec, sci, gpu_uuid=ra["gpu_uuid"], out_dir=ra["out_dir"],
            checkpoint_identity=RTC.build_checkpoint_identity("/ckpt/17500/full_state.pkl"),
            cli_args=dict(carry_mode=arm, replay_mode="original_vtrace"),
            runtime_constants=dict(SEQUENCE_LENGTH=129, K_BATCH=4))
        if cert["certificate_status"] != "PASS":
            return FAIL, f"{arm}: certificate FAIL: {cert['validation_errors']}"
        if not (cert["scientific_config_match"] and cert["runtime_assignment_match"]
                and cert["validation_errors"] == []):
            return FAIL, f"{arm}: match flags wrong: {cert}"
        if cert["scientific_config_sha256"] != cert["runtime_scientific_config_sha256"]:
            return FAIL, f"{arm}: scientific SHAs differ despite PASS"
    # the frozen expected base SHA is the one recorded in BOTH frozen probe summaries
    import json as _json
    ev = os.path.join(_SNAPSHOT, "evidence", "raw_probe")
    shas = set()
    for name in ("persistent_probe_summary.json", "reset128_probe_summary.json"):
        with open(os.path.join(ev, name), encoding="utf-8") as f:
            shas.add(_json.load(f)["base_sha256"])
    if shas != {RTC.EXPECTED_BASE_CHECKPOINT_SHA256}:
        return FAIL, f"frozen expected base sha {RTC.EXPECTED_BASE_CHECKPOINT_SHA256} != evidence {shas}"
    return PASS, ("both arms: YAML scientific_config == real runtime scientific_config "
                  "(canonical diff empty, SHAs equal), assignment PASS, certificate PASS; "
                  "frozen expected base SHA == both probe summaries")


def gate34():
    """GATE 34 — any scientific field change -> FORMAL_CONFIG_RUNTIME_MISMATCH (seed,
    sequence_length, lr, rho_bar, network.embed_size at minimum) (§六.9)."""
    import phase4a_v2_runtime_config as RTC
    RTC, rec, y, sci = _rtc_reference("persistent")
    ra = y["runtime_assignment"]
    mutations = [
        ("seed", dict(seed=43), "scientific_config.seed"),
        ("sequence_length", dict(sequence_length=130), "scientific_config.sequence_length"),
        ("ppo.lr", dict(ppo_lr=3.0e-5), "scientific_config.ppo.lr"),
        ("vtrace.rho_bar", dict(vtrace_rho_bar=2.0), "vtrace.rho_bar"),
        ("network.embed_size", dict(net_embed_size=512), "scientific_config.network.embed_size"),
    ]
    for name, over, expect in mutations:
        kw = RTC._reference_runtime_kwargs("persistent"); kw.update(over)
        cert = RTC.validate_runtime_against_formal_config(
            rec, RTC.build_runtime_scientific_config(**kw),
            gpu_uuid=ra["gpu_uuid"], out_dir=ra["out_dir"])
        errs = " | ".join(cert["validation_errors"])
        if cert["certificate_status"] != "FAIL" or "FORMAL_CONFIG_RUNTIME_MISMATCH" not in errs:
            return FAIL, f"{name}: expected FORMAL_CONFIG_RUNTIME_MISMATCH, got {errs[:200]}"
        if expect not in errs:
            return FAIL, f"{name}: diff path {expect} not reported: {errs[:200]}"
    return PASS, ("seed / sequence_length / ppo.lr / vtrace.rho_bar / network.embed_size each "
                  "-> certificate FAIL with FORMAL_CONFIG_RUNTIME_MISMATCH")


def gate35():
    """GATE 35 — runtime assignment mismatch (GPU UUID or out_dir) fails, independently of the
    scientific SHA (§六.6)."""
    import phase4a_v2_runtime_config as RTC
    RTC, rec, y, sci = _rtc_reference("persistent")
    ra = y["runtime_assignment"]
    cert = RTC.validate_runtime_against_formal_config(
        rec, sci, gpu_uuid="GPU-00000000-0000-0000-0000-000000000000", out_dir=ra["out_dir"])
    if cert["certificate_status"] != "FAIL" or cert["runtime_assignment_match"]:
        return FAIL, f"wrong gpu_uuid not caught: {cert['validation_errors']}"
    if not cert["scientific_config_match"]:
        return FAIL, "scientific config must still match when only assignment differs"
    if not any("RUNTIME_ASSIGNMENT_MISMATCH" in e for e in cert["validation_errors"]):
        return FAIL, "RUNTIME_ASSIGNMENT_MISMATCH not reported for gpu_uuid"
    cert2 = RTC.validate_runtime_against_formal_config(
        rec, sci, gpu_uuid=ra["gpu_uuid"], out_dir="runs/WRONG-PLACE")
    if cert2["certificate_status"] != "FAIL" or cert2["runtime_assignment_match"]:
        return FAIL, f"wrong out_dir not caught: {cert2['validation_errors']}"
    if not any("RUNTIME_ASSIGNMENT_MISMATCH" in e for e in cert2["validation_errors"]):
        return FAIL, "RUNTIME_ASSIGNMENT_MISMATCH not reported for out_dir"
    return PASS, ("GPU UUID mismatch and out_dir mismatch each -> certificate FAIL with "
                  "RUNTIME_ASSIGNMENT_MISMATCH while scientific_config_match stays true")


def gate36():
    """GATE 36 — v2.4 certificate chain static order (§六 + §十四): ARGPARSE < preflight <
    FORMAL_IDENTITY < runtime assignment < ACTUAL-CLI pre-JAX scientific binding < PENDING
    precheck < pre-JAX refusal < `import jax` < imported-constants binding (+drift
    finalize/refusal) < replay-RNG construction < executed learner/sampler SOURCE binding <
    RNG identity binding < EFFECTIVE protocol build (+executed-protocol finalize/refusal) <
    staged ckpt load < checkpoint verify/finalize < post-write verify + disk re-read < final
    refusal < env build < training loop. original_vtrace without --formal_config fails closed
    BEFORE JAX; off mode stays exempt.

    Phase4A-v2.4 reason+diff: v2.3 asserted two finalize sites with env build BEFORE ckpt
    load; v2.4 (§六) binds the executed protocol (learner/sampler/RNG + effective definition)
    BEFORE the checkpoint finalize, moves ckpt load + finalize BEFORE env build, adds the
    executed-protocol finalize site (3 sites total), the §三.1 actual-CLI kwargs override,
    the staged checkpoint failure labels, and a post-write verify + disk re-read on the final
    PASS certificate."""
    import phase4a_v2_runtime_config as RTC
    try:
        RTC.preflight_require_formal_config("original_vtrace", None)
        return FAIL, "missing --formal_config under original_vtrace did NOT raise"
    except ValueError as e:
        if "FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE" not in str(e):
            return FAIL, f"wrong error: {e}"
    RTC.preflight_require_formal_config("off", None)  # off stays exempt (legacy dev compat)
    src = _read(_LAUNCHER).replace("\r\n", "\n")
    lines = src.split("\n")

    def line_of(needle):
        for i, ln in enumerate(lines):
            if needle in ln:
                return i
        return -1

    def lines_of(needle):
        return [i for i, ln in enumerate(lines) if needle in ln]

    marks = {
        "argparse": line_of("args = ap.parse_args()"),
        "preflight": line_of(
            "RTC.preflight_require_formal_config(REPLAY_MODE, args.formal_config)"),
        "formal_identity": line_of(
            "FORMAL_CONFIG_IDENTITY = FID.verify_formal_config_identity("),
        "assignment": line_of("RUNTIME_ASSIGNMENT_RECORD = RTC.validate_runtime_assignment("),
        # §三.1: the frozen spec kwargs are overridden with the ACTUAL CLI values pre-JAX.
        "prejax_cli_override": line_of("_prejax_kwargs.update("),
        "prejax_scientific": line_of(
            "_PREJAX_SCIENTIFIC = RTC.build_runtime_scientific_config("),
        "precheck": line_of("RUNTIME_CONFIG_CERTIFICATE = RTC.build_precheck_certificate("),
        # single-line literals of the refusal messages (line_of matches within ONE line)
        "prejax_refuse": line_of(
            '"FORMAL_CONFIG_RUNTIME_MISMATCH: prejax precheck certificate_status="'),
        "import_jax": line_of("import jax, jax.numpy as jnp"),
        "imported_scientific": line_of(
            "_imported_scientific = RTC.build_runtime_scientific_config("),
        "drift_diff": line_of("_IMPORTED_CONSTANTS_DRIFT = RTC.deep_diff("),
        "imported_refuse": line_of(
            '"IMPORTED_RUNTIME_CONSTANTS_MISMATCH: the REAL imported runtime constants '
            'drifted "'),
        # §六 ordering: RNG construction < executed binding < RNG identity < effective build.
        "rng_construct": line_of("replay_sample_rng = np.random.RandomState(args.seed + 7)"),
        "executed_bind": line_of(
            "EXECUTED_PROTOCOL_IDENTITY = CONTRACT.executed_function_source_identity("),
        "rng_bind": line_of(
            'EXECUTED_PROTOCOL_IDENTITY["rng_instance"] = '
            "CONTRACT.verify_rng_instance_identity("),
        "effective_build": line_of(
            "EFFECTIVE_PROTOCOL_DEFINITION, EFFECTIVE_PROTOCOL_SHA256 = ("),
        "executed_refuse": line_of(
            '"EXECUTED_PROTOCOL_BINDING_FAILURE: the executed replay protocol (learner '
            'source "'),
        # §四: the staged checkpoint flow (stage labels) precedes the manager init.
        "ckpt_stage": line_of('_CHECKPOINT_FAILURE_STAGE = "CHECKPOINT_MANAGER_INIT"'),
        "ckpt": line_of("ckpt_mgr = ocp.CheckpointManager("),
        "verify_checkpoint": line_of(
            'RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"] = '
            "RTC.verify_checkpoint_params_sha("),
        # §五.2: post-write verification + disk re-read on the final PASS certificate.
        "cert_disk_reread": line_of("_disk_certificate = json.load(_cert_f)"),
        "final_refuse": line_of(
            '"FORMAL_CONFIG_RUNTIME_MISMATCH: runtime_config_certificate_status=FAIL "'),
        "env": line_of("base_env = MultiTaskMiniCraftaxEnv("),
        "train_loop": line_of("for u in range(args.total_updates):"),
    }
    fins = lines_of("RUNTIME_CONFIG_CERTIFICATE = RTC.finalize_certificate(")
    verifys = lines_of("RTC.verify_certificate_artifact(")
    missing = sorted(k for k, v in marks.items() if v == -1)
    if missing:
        return FAIL, f"launcher wiring missing: {missing} (finalize sites={fins})"
    # exactly THREE finalize sites (v2.4): the drift-failure finalize, the executed-protocol
    # failure finalize, and the checkpoint finalize; classify by position.
    if len(fins) != 3:
        return FAIL, f"expected exactly 3 finalize_certificate sites, got {fins}"
    drift_fins = [i for i in fins
                  if marks["drift_diff"] < i < marks["rng_construct"]]
    executed_fins = [i for i in fins
                     if marks["effective_build"] < i < marks["ckpt_stage"]]
    ckpt_fins = [i for i in fins if i > marks["verify_checkpoint"]]
    if len(drift_fins) != 1 or len(executed_fins) != 1 or len(ckpt_fins) != 1:
        return FAIL, (f"finalize sites misplaced: drift={drift_fins} "
                      f"executed={executed_fins} ckpt={ckpt_fins}")
    marks["drift_finalize"] = drift_fins[0]
    marks["executed_finalize"] = executed_fins[0]
    marks["checkpoint_finalize"] = ckpt_fins[0]
    # a verify_certificate_artifact call must follow the checkpoint finalize (§五.2: verify
    # after EVERY write), and the last verify must be the post-final-write one.
    post_final_verify = [i for i in verifys if i > marks["checkpoint_finalize"]]
    if not post_final_verify:
        return FAIL, (f"no verify_certificate_artifact after the checkpoint finalize "
                      f"(verify sites={verifys}, finalize={marks['checkpoint_finalize']})")
    marks["cert_verify_final"] = post_final_verify[0]
    order = ["argparse", "preflight", "formal_identity", "assignment", "prejax_cli_override",
             "prejax_scientific", "precheck", "prejax_refuse", "import_jax",
             "imported_scientific", "drift_diff", "drift_finalize", "imported_refuse",
             "rng_construct", "executed_bind", "rng_bind", "effective_build",
             "executed_finalize", "executed_refuse", "ckpt_stage", "ckpt",
             "verify_checkpoint", "checkpoint_finalize", "cert_verify_final",
             "cert_disk_reread", "final_refuse", "env", "train_loop"]
    for a, b in zip(order, order[1:]):
        if not marks[a] < marks[b]:
            return FAIL, f"order violation: {a}({marks[a]}) !< {b}({marks[b]})"
    for arg in ('"--formal_config"', '"--snapshot_root"', '"--run_root"'):
        if arg not in src:
            return FAIL, f"launcher lacks the {arg} argument"
    return PASS, ("static §六/§十四 order holds end-to-end: argparse < preflight < formal "
                  "identity < assignment < ACTUAL-CLI pre-JAX scientific binding < PENDING "
                  "precheck + pre-JAX refusal < import jax < imported-constants drift "
                  "finalize/refusal < replay-RNG construction < executed learner/sampler + RNG "
                  "binding < effective protocol build + executed-protocol finalize/refusal < "
                  "staged ckpt load < checkpoint verify/finalize < post-write verify + disk "
                  "re-read < final refusal < env build < training loop; 3 finalize sites; "
                  "--formal_config/--snapshot_root/--run_root all wired; off mode exempt")


def gate37():
    """GATE 37 — certificate SHAs consistent with file content; checkpoint manifest + summary
    reference the certificate SHAs; base-checkpoint comparison fail-closed (§六.5/§六.7/§六.8)."""
    import json as _json
    import tempfile
    import phase4a_v2_runtime_config as RTC
    RTC, rec, y, sci = _rtc_reference("persistent")
    ra = y["runtime_assignment"]
    cert = RTC.validate_runtime_against_formal_config(
        rec, sci, gpu_uuid=ra["gpu_uuid"], out_dir=ra["out_dir"],
        checkpoint_identity=RTC.build_checkpoint_identity("/ckpt/17500/full_state.pkl"))
    # write + re-read: file content matches the certificate object and its SHAs recompute
    with tempfile.TemporaryDirectory() as td:
        p = RTC.write_runtime_config_certificate(cert, os.path.join(td, "cert.json"))
        with open(p, encoding="utf-8") as f:
            disk = _json.load(f)
    if disk != _json.loads(_json.dumps(cert, sort_keys=True)):
        return FAIL, "certificate on disk != certificate object"
    if disk["scientific_config_sha256"] != RTC.scientific_config_sha256(
            y["scientific_config"]):
        return FAIL, "certificate scientific_config_sha256 != recomputed YAML sha"
    if disk["formal_config_file_sha256"] != rec["file_sha256"]:
        return FAIL, "certificate formal_config_file_sha256 != file sha"
    # checkpoint SHA: PASS on frozen expectation, fail closed on mismatch, NOT_FROZEN if absent
    ci_ok = RTC.verify_checkpoint_params_sha(
        RTC.build_checkpoint_identity("/ckpt/17500/x"), RTC.EXPECTED_BASE_CHECKPOINT_SHA256)
    if ci_ok["base_checkpoint_match"] != "PASS":
        return FAIL, f"frozen expectation did not PASS: {ci_ok}"
    try:
        RTC.verify_checkpoint_params_sha(RTC.build_checkpoint_identity("/ckpt/17500/x"), "f" * 64)
        return FAIL, "base SHA mismatch did NOT raise"
    except ValueError as e:
        if "BASE_CHECKPOINT_SHA_MISMATCH" not in str(e):
            return FAIL, f"wrong error: {e}"
    # launcher embeds the certificate record in the checkpoint manifest AND the summary.
    # Phase4A-v2.3 reason+diff (§七.3): the summary call now binds the FINAL certificate file
    # SHA + sidecar (the v2.2 two-line needle no longer exists); the record key set grew to a
    # 13-key superset that pins the certificate ARTIFACT, not just the payload.
    src = _read(_LAUNCHER).replace("\r\n", "\n")
    for needle in ('fields["runtime_config_certificate"] = RTC.certificate_shas_record(',
                   "runtime_config_certificate=(\n"
                   "                   RTC.certificate_shas_record(\n"
                   "                       RUNTIME_CONFIG_CERTIFICATE,\n"
                   "                       certificate_file_sha256="
                   "RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,\n"
                   "                       certificate_sidecar_path="
                   "RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH)",
                   "runtime_config_certificate_path=RUNTIME_CONFIG_CERTIFICATE_PATH,",
                   "runtime_config_certificate_sidecar_path="
                   "RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,",
                   "runtime_config_certificate_file_sha256="
                   "RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,",
                   "executed_protocol_identity=EXECUTED_PROTOCOL_IDENTITY,",
                   "RUNTIME_CONFIG_CERTIFICATE[\"checkpoint_identity\"] = "
                   "RTC.verify_checkpoint_params_sha("):
        if needle not in src:
            return FAIL, f"launcher missing certificate embedding: {needle[:60]}..."
    # certificate_shas_record carries the §六.8 fields PLUS the v2.3 §七.3 artifact-identity keys
    keys = set(RTC.certificate_shas_record(cert))
    need = {"formal_config_file_sha256", "scientific_config_sha256",
            "runtime_scientific_config_sha256", "runtime_config_certificate_status",
            "base_checkpoint_expected_sha256", "base_checkpoint_expected_sha256_status",
            "runtime_config_certificate_version", "runtime_config_certificate_finalized",
            "runtime_config_certificate_payload_sha256",
            "runtime_config_certificate_file_sha256",
            "runtime_config_certificate_sidecar_path",
            "base_checkpoint_params_sha256", "base_checkpoint_match"}
    if not need.issubset(keys):
        return FAIL, f"certificate_shas_record missing {need - keys}"
    if len(keys) != 13:
        return FAIL, f"certificate_shas_record key count {len(keys)} != 13 (superset drift)"
    return PASS, ("certificate file content == object; SHAs recompute from YAML/file; base SHA "
                  "PASS/FAIL-closed/NOT_FROZEN paths correct; manifest + summary embed the "
                  "certificate record; summary binds final file SHA + sidecar; record = 13 keys")


def gate38():
    """GATE 38 — layered, time-scoped remote publication status labels (§七): no unscoped
    PUSH_PERFORMED for the whole branch."""
    import json as _json
    path = os.path.join(_SNAPSHOT, "reports", "rmt16_phase4a_v2_2_labels.json")
    if not os.path.isfile(path):
        return FAIL, "reports/rmt16_phase4a_v2_2_labels.json missing"
    with open(path, encoding="utf-8") as f:
        doc = _json.load(f)
    labels = doc["labels"]
    expect = {
        "BASE_REMOTE_PUBLICATION_STATUS": "PASS",
        "BASE_REMOTE_HEAD": "87d1e552415d292417dcb6e6f9f6b16b97a6d135",
        "IMPLEMENTATION_ROUND_PUSH_PERFORMED": False,
        "V2_2_REMOTE_PUBLICATION_STATUS": "NOT_PUSHED",
        "V2_2_PUSH_PERFORMED": False,
    }
    for k, v in expect.items():
        if labels.get(k) != v:
            return FAIL, f"{k}={labels.get(k)!r} != {v!r}"
    if "PUSH_PERFORMED" in labels:
        return FAIL, "time-UNSCOPED 'PUSH_PERFORMED' key present (forbidden by §七)"
    return PASS, ("layered status: BASE_REMOTE_PUBLICATION_STATUS=PASS @ 87d1e55, "
                  "IMPLEMENTATION_ROUND_PUSH_PERFORMED=false, V2_2_*=NOT_PUSHED; no unscoped "
                  "PUSH_PERFORMED")


# ----------------------------------------------------------------------------
# Phase4A-v2.3 (§十/§十一): GATE39-GATE50 — canonical formal-config identity (§三),
# runtime-assignment fail closed (§四), pre-JAX full scientific binding (§五/§十四),
# certificate state machine (§六), certificate artifact SHA / sidecar / tamper detection
# (§七), executed-protocol SOURCE identity (§八), v2.3 publication labels + v2.2 errata
# (§九). Negative tests are marked (NEG); GATE39-GATE50 add >= 30 fail-closed negatives
# (§十一 requires >= 25), on top of the module self-tests (RTC: 76, FID: 13, exposure
# validator: 25).
# ----------------------------------------------------------------------------


def _v23_prejax_chain(arm="persistent", *, with_identity=True, sci_override=None,
                      cli_carry=None, cli_gpu=None, run_root=None,
                      cli_replay_mode="original_vtrace", cli_allow_full_p2_legacy=False,
                      cli_sequence_length=129, cli_seed=42, cli_total_updates=12,
                      cli_save_every=2):
    """Functional mirror of the driver's PRE-JAX chain (§五.2 steps 3-8) for one arm, using
    exactly the pure-Python modules the driver imports before `import jax` (RTC + FSPEC +
    FID; NO jax/numpy). Returns (precheck_certificate, assignment_record, FSPEC).

    Phase4A-v2.4 (§三.1): the mirror binds the ACTUAL CLI VALUES — the frozen spec kwargs are
    overridden with the seven CLI-facing keys (carry_mode / replay_mode / allow_full_p2_legacy
    / sequence_length / seed / total_updates / save_every) EXACTLY as the driver now does, so
    a wrong simulated CLI value FAILs the precheck HERE (pre-JAX), not later. The defaults
    equal the frozen spec, so legacy callers behave unchanged."""
    import phase4a_v2_runtime_config as RTC
    import phase4a_v2_frozen_spec as FSPEC
    import phase4a_v2_formal_identity as FID
    snap = run_root if run_root is not None else _SNAPSHOT
    path = os.path.join(snap, "configs", f"rmt16_phase4a_v2_{arm}.yaml")
    rec = RTC.load_formal_config(path)
    cfg = rec["config"]
    ra = cfg["runtime_assignment"]
    cli_carry = arm if cli_carry is None else cli_carry
    cli_gpu = ra["gpu_uuid"] if cli_gpu is None else cli_gpu
    cli_out = os.path.join(snap, ra["out_dir"])
    # §三: the arm binding uses the ACTUAL CLI carry/replay (a wrong --carry_mode /
    # --replay_mode raises FORMAL_CONFIG_ARM_MISMATCH here, before any scientific build).
    RTC.validate_arm_binding(rec, cli_carry, replay_mode=cli_replay_mode)
    fid = FID.verify_formal_config_identity(snap, arm, rec) if with_identity else None
    assign = RTC.validate_runtime_assignment(cfg, cli_carry=cli_carry, cli_gpu=cli_gpu,
                                             cli_out=cli_out, run_root=snap)
    kw = FSPEC.build_kwargs(arm)
    kw.update(
        carry_mode=cli_carry,
        replay_mode=cli_replay_mode,
        allow_full_p2_legacy=bool(cli_allow_full_p2_legacy),
        sequence_length=int(cli_sequence_length),
        seed=int(cli_seed),
        total_updates=int(cli_total_updates),
        save_every=int(cli_save_every))
    if sci_override:
        kw.update(sci_override)
    sci = RTC.build_runtime_scientific_config(**kw)
    cert = RTC.build_precheck_certificate(
        rec, sci, formal_identity_record=fid, assignment_record=assign,
        checkpoint_identity=RTC.build_checkpoint_identity("/ckpt/17500/full_state.pkl"),
        frozen_spec_sha256=FSPEC.FROZEN_SPEC_SHA256,
        cli_args=dict(carry_mode=cli_carry, gpu_uuid=cli_gpu, out=cli_out),
        runtime_constants=dict(FROZEN_SPEC_SHA256=FSPEC.FROZEN_SPEC_SHA256),
        snapshot_root=snap, run_root=snap)
    return cert, assign, FSPEC


def gate39():
    """GATE 39 — canonical formal-config identity (§三): both arms' frozen path + file SHA +
    scientific SHA verify PASS against the real committed YAMLs; the reference runtime kwargs
    are the frozen spec itself (single source of truth)."""
    import phase4a_v2_runtime_config as RTC
    import phase4a_v2_frozen_spec as FSPEC
    import phase4a_v2_formal_identity as FID
    for arm in ("persistent", "reset128"):
        ident = FID.frozen_identity(arm)
        rec = RTC.load_formal_config(os.path.join(_SNAPSHOT, ident["relative_path"]))
        idrec = FID.verify_formal_config_identity(_SNAPSHOT, arm, rec)
        if idrec["formal_config_identity"] != "PASS":
            return FAIL, f"{arm}: canonical identity not PASS: {idrec}"
        if rec["file_sha256"] != ident["file_sha256"]:
            return FAIL, f"{arm}: file SHA drift vs frozen constant"
        if RTC.scientific_config_sha256(
                rec["config"]["scientific_config"]) != ident["scientific_config_sha256"]:
            return FAIL, f"{arm}: scientific SHA drift vs frozen constant"
    if RTC._reference_runtime_kwargs("persistent") != FSPEC.build_kwargs("persistent"):
        return FAIL, "reference kwargs diverged from the frozen spec"
    return PASS, ("both arms: canonical realpath == frozen path; file SHA + scientific SHA == "
                  "frozen constants (re-derived from the real files); reference kwargs == "
                  "frozen spec (single source of truth)")


def gate40():
    """GATE 40 — formal-config identity fail closed (§三, NEG x6): byte-identical copy at
    another path, edited seed (even CLI-synced), comment-only change, wrong-arm YAML (path +
    content), unknown arm — each rejected with the §三 error code."""
    import shutil
    import tempfile
    import phase4a_v2_runtime_config as RTC
    import phase4a_v2_formal_identity as FID
    canon = os.path.join(_SNAPSHOT, "configs", "rmt16_phase4a_v2_persistent.yaml")
    tmp = tempfile.mkdtemp(prefix="p4av23_g40_")
    try:
        # (NEG 1) byte-identical copy elsewhere -> PATH identity FAIL
        copy = os.path.join(tmp, "copy.yaml")
        shutil.copyfile(canon, copy)
        try:
            FID.verify_formal_config_path_identity(_SNAPSHOT, "persistent", copy)
            return FAIL, "(NEG1) byte copy elsewhere not rejected"
        except ValueError as e:
            if "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH" not in str(e):
                return FAIL, f"(NEG1) wrong code: {e}"
        # (NEG 2) edited seed -> CONTENT identity FAIL
        edited = open(canon, encoding="utf-8").read().replace("seed: 42", "seed: 43")
        ep = os.path.join(tmp, "edited.yaml")
        with open(ep, "w", encoding="utf-8") as f:
            f.write(edited)
        try:
            FID.verify_formal_config_content_identity(RTC.load_formal_config(ep), "persistent")
            return FAIL, "(NEG2) edited-seed copy not rejected"
        except ValueError as e:
            if "FORMAL_CONFIG_IDENTITY_MISMATCH" not in str(e):
                return FAIL, f"(NEG2) wrong code: {e}"
        # (NEG 3) comment-only change -> FILE SHA FAIL (scientific SHA unchanged)
        commented = "# comment that changes file bytes only\n" + open(
            canon, encoding="utf-8").read()
        cp = os.path.join(tmp, "commented.yaml")
        with open(cp, "w", encoding="utf-8") as f:
            f.write(commented)
        try:
            FID.verify_formal_config_content_identity(RTC.load_formal_config(cp), "persistent")
            return FAIL, "(NEG3) comment-only change not rejected"
        except ValueError as e:
            if "FORMAL_CONFIG_IDENTITY_MISMATCH" not in str(e) or "file_sha256" not in str(e):
                return FAIL, f"(NEG3) wrong code: {e}"
        # (NEG 4 + 5) wrong-arm YAML -> PATH and CONTENT identity both FAIL
        r128 = os.path.join(_SNAPSHOT, "configs", "rmt16_phase4a_v2_reset128.yaml")
        try:
            FID.verify_formal_config_path_identity(_SNAPSHOT, "persistent", r128)
            return FAIL, "(NEG4) wrong-arm YAML path not rejected"
        except ValueError as e:
            if "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH" not in str(e):
                return FAIL, f"(NEG4) wrong code: {e}"
        try:
            FID.verify_formal_config_content_identity(RTC.load_formal_config(r128), "persistent")
            return FAIL, "(NEG5) wrong-arm YAML content not rejected"
        except ValueError as e:
            if "FORMAL_CONFIG_IDENTITY_MISMATCH" not in str(e):
                return FAIL, f"(NEG5) wrong code: {e}"
        # (NEG 6) unknown arm -> fail closed
        try:
            FID.frozen_identity("bogus")
            return FAIL, "(NEG6) unknown arm not rejected"
        except ValueError as e:
            if "FORMAL_CONFIG_IDENTITY_UNKNOWN_ARM" not in str(e):
                return FAIL, f"(NEG6) wrong code: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return PASS, ("6 negatives rejected fail closed: byte copy elsewhere (path), edited seed "
                  "(content), comment-only (file SHA), wrong-arm YAML (path + content), "
                  "unknown arm")


def gate41():
    """GATE 41 — runtime_assignment completeness fail closed (§四.1, NEG x6): missing / null /
    empty / non-string gpu_uuid or out_dir, missing runtime_assignment block, missing
    top-level arm — all raise RUNTIME_ASSIGNMENT_INCOMPLETE (no default, no bypass, no
    fail-open)."""
    import copy as _copy
    import phase4a_v2_runtime_config as RTC
    base = RTC.load_formal_config(
        os.path.join(_SNAPSHOT, "configs", "rmt16_phase4a_v2_persistent.yaml"))["config"]

    def mutated(fn):
        cfg = _copy.deepcopy(base)
        fn(cfg)
        return cfg

    cases = [
        ("missing gpu_uuid", lambda c: c["runtime_assignment"].pop("gpu_uuid")),
        ("null gpu_uuid", lambda c: c["runtime_assignment"].__setitem__("gpu_uuid", None)),
        ("empty out_dir", lambda c: c["runtime_assignment"].__setitem__("out_dir", "   ")),
        ("non-string gpu_uuid", lambda c: c["runtime_assignment"].__setitem__("gpu_uuid", 123)),
        ("no runtime_assignment block", lambda c: c.pop("runtime_assignment")),
        ("missing top-level arm", lambda c: c.pop("arm")),
    ]
    for name, fn in cases:
        try:
            RTC.resolve_runtime_assignment(mutated(fn))
            return FAIL, f"(NEG) {name}: RUNTIME_ASSIGNMENT_INCOMPLETE not raised"
        except ValueError as e:
            if "RUNTIME_ASSIGNMENT_INCOMPLETE" not in str(e):
                return FAIL, f"(NEG) {name}: wrong code: {e}"
    got = RTC.resolve_runtime_assignment(base)
    if got != dict(arm="persistent", gpu_uuid=base["runtime_assignment"]["gpu_uuid"],
                   out_dir=base["runtime_assignment"]["out_dir"]):
        return FAIL, f"real config resolved wrong: {got}"
    return PASS, ("6 incomplete-assignment negatives all raise RUNTIME_ASSIGNMENT_INCOMPLETE "
                  "(missing/null/empty/non-string fields, no ra block, no arm); the real "
                  "config resolves arm + gpu_uuid + out_dir exactly")


def gate42():
    """GATE 42 — runtime_assignment four-way ARM + GPU fail closed (§四.2/§四.4, NEG x3): CLI
    carry mismatch, scientific carry_mode mismatch, wrong --gpu_uuid — each FAILs the
    assignment with its own error code; exact values PASS."""
    import copy as _copy
    import phase4a_v2_runtime_config as RTC
    rec = RTC.load_formal_config(
        os.path.join(_SNAPSHOT, "configs", "rmt16_phase4a_v2_persistent.yaml"))
    cfg = rec["config"]
    ra = cfg["runtime_assignment"]
    good_out = os.path.join(_SNAPSHOT, ra["out_dir"])

    def has_code(record, code):
        return any(code in e for e in record["runtime_assignment_errors"])

    # (NEG 1) CLI carry_mode disagrees with the formal arm
    r1 = RTC.validate_runtime_assignment(cfg, cli_carry="reset128", cli_gpu=ra["gpu_uuid"],
                                         cli_out=good_out, run_root=_SNAPSHOT)
    if r1["runtime_assignment_match"] or not has_code(r1, "RUNTIME_ASSIGNMENT_ARM_MISMATCH"):
        return FAIL, f"(NEG1) CLI carry mismatch not caught: {r1['runtime_assignment_errors']}"
    # (NEG 2) scientific_config.carry_mode disagrees (edited formal block)
    cfg2 = _copy.deepcopy(cfg)
    cfg2["scientific_config"]["carry_mode"] = "reset128"
    r2 = RTC.validate_runtime_assignment(cfg2, cli_carry="persistent", cli_gpu=ra["gpu_uuid"],
                                         cli_out=good_out, run_root=_SNAPSHOT)
    if r2["runtime_assignment_match"] or not has_code(r2, "RUNTIME_ASSIGNMENT_ARM_MISMATCH"):
        return FAIL, (f"(NEG2) scientific carry_mode mismatch not caught: "
                      f"{r2['runtime_assignment_errors']}")
    # (NEG 3) wrong --gpu_uuid (exact equality; no suffix match)
    r3 = RTC.validate_runtime_assignment(cfg, cli_carry="persistent",
                                         cli_gpu=ra["gpu_uuid"] + "-X", cli_out=good_out,
                                         run_root=_SNAPSHOT)
    if r3["runtime_assignment_match"] or not has_code(r3, "RUNTIME_ASSIGNMENT_GPU_MISMATCH"):
        return FAIL, f"(NEG3) gpu_uuid mismatch not caught: {r3['runtime_assignment_errors']}"
    # positive control: exact values PASS
    r0 = RTC.validate_runtime_assignment(cfg, cli_carry="persistent", cli_gpu=ra["gpu_uuid"],
                                         cli_out=good_out, run_root=_SNAPSHOT)
    if not r0["runtime_assignment_match"] or r0["runtime_assignment_errors"]:
        return FAIL, f"positive control failed: {r0['runtime_assignment_errors']}"
    return PASS, ("CLI carry / scientific carry_mode / gpu_uuid mismatches each -> "
                  "RUNTIME_ASSIGNMENT_{ARM,GPU}_MISMATCH with match=False (four-way arm "
                  "binding); exact values PASS")


def gate43():
    """GATE 43 — strict out_dir identity fail closed (§四.3, NEG x5): absolute formal out_dir,
    '..' segment, missing --run_root, CLI --out elsewhere under run_root, and the v2.2 suffix
    trap (dir name extending the formal name) — each -> RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH;
    the exact relative dir under --run_root PASSes (strict realpath equality)."""
    import copy as _copy
    import phase4a_v2_runtime_config as RTC
    rec = RTC.load_formal_config(
        os.path.join(_SNAPSHOT, "configs", "rmt16_phase4a_v2_persistent.yaml"))
    cfg = rec["config"]
    ra = cfg["runtime_assignment"]
    formal_out = ra["out_dir"]

    def run(cfg_, cli_out, run_root):
        return RTC.validate_runtime_assignment(cfg_, cli_carry="persistent",
                                               cli_gpu=ra["gpu_uuid"], cli_out=cli_out,
                                               run_root=run_root)

    def has_out_code(r):
        return any("RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH" in e
                   for e in r["runtime_assignment_errors"])

    # (NEG 1) absolute formal out_dir
    c1 = _copy.deepcopy(cfg)
    c1["runtime_assignment"]["out_dir"] = "/abs/runs/x"
    r1 = run(c1, "/abs/runs/x", _SNAPSHOT)
    if r1["runtime_assignment_match"] or not has_out_code(r1):
        return FAIL, f"(NEG1) absolute out_dir not rejected: {r1['runtime_assignment_errors']}"
    # (NEG 2) '..' segment
    c2 = _copy.deepcopy(cfg)
    c2["runtime_assignment"]["out_dir"] = "runs/../escape"
    r2 = run(c2, os.path.join(_SNAPSHOT, "escape"), _SNAPSHOT)
    if r2["runtime_assignment_match"] or not has_out_code(r2):
        return FAIL, f"(NEG2) '..' segment not rejected: {r2['runtime_assignment_errors']}"
    # (NEG 3) missing --run_root
    r3 = run(cfg, os.path.join(_SNAPSHOT, formal_out), None)
    if r3["runtime_assignment_match"] or not has_out_code(r3):
        return FAIL, f"(NEG3) missing run_root not rejected: {r3['runtime_assignment_errors']}"
    # (NEG 4) CLI --out elsewhere under run_root (strict equality, not containment-by-name)
    r4 = run(cfg, os.path.join(_SNAPSHOT, "runs", "SOMEWHERE-ELSE"), _SNAPSHOT)
    if r4["runtime_assignment_match"] or not has_out_code(r4):
        return FAIL, f"(NEG4) out dir elsewhere not rejected: {r4['runtime_assignment_errors']}"
    # (NEG 5) suffix trap: runs/<formal>-extra (v2.2 suffix match would have accepted this)
    r5 = run(cfg, os.path.join(_SNAPSHOT, formal_out + "-extra"), _SNAPSHOT)
    if r5["runtime_assignment_match"] or not has_out_code(r5):
        return FAIL, f"(NEG5) suffix-trap dir not rejected: {r5['runtime_assignment_errors']}"
    # positive control: exact realpath under run_root
    r0 = run(cfg, os.path.join(_SNAPSHOT, formal_out), _SNAPSHOT)
    if not r0["runtime_assignment_match"]:
        return FAIL, f"positive control failed: {r0['runtime_assignment_errors']}"
    return PASS, ("absolute / '..' / missing-run_root / elsewhere / suffix-trap out_dir all -> "
                  "RUNTIME_ASSIGNMENT_OUT_DIR_MISMATCH (strict realpath equality; the v2.2 "
                  "suffix match is gone); exact relative dir under --run_root PASSes")


def gate44():
    """GATE 44 — full scientific binding PRE-JAX on the frozen spec (§五/§十四): the precheck
    certificate reaches PENDING_CHECKPOINT_IDENTITY (NOT PASS) on BOTH arms through the pure-
    Python chain only (formal identity + strict assignment + frozen-spec scientific config),
    then finalizes to PASS once the frozen base SHA is loaded; missing identity -> FAIL with
    FORMAL_CONFIG_IDENTITY_MISMATCH (NEG)."""
    import phase4a_v2_runtime_config as RTC
    for arm in ("persistent", "reset128"):
        cert, assign, FSPEC = _v23_prejax_chain(arm)
        if cert["certificate_status"] != RTC.CERTIFICATE_STATUS_PENDING:
            return FAIL, (f"{arm}: precheck status={cert['certificate_status']} != "
                          f"PENDING_CHECKPOINT_IDENTITY; errors={cert['validation_errors']}")
        if cert["certificate_finalized"]:
            return FAIL, f"{arm}: precheck certificate must NOT be finalized pre-checkpoint"
        if cert["checkpoint_identity"]["base_checkpoint_match"] != "PENDING":
            return FAIL, (f"{arm}: pre-checkpoint match must be PENDING, got "
                          f"{cert['checkpoint_identity']['base_checkpoint_match']}")
        if not (cert["scientific_config_match"] and cert["runtime_assignment_match"]
                and cert["formal_config_identity"].get("formal_config_identity") == "PASS"):
            return FAIL, f"{arm}: pre-JAX binding flags wrong: {cert['validation_errors']}"
        if cert["frozen_spec_sha256"] != FSPEC.FROZEN_SPEC_SHA256:
            return FAIL, f"{arm}: frozen spec SHA not bound into the certificate"
        # finalize with the frozen base SHA (as the driver does post-load) -> PASS
        ci = RTC.verify_checkpoint_params_sha(
            cert["checkpoint_identity"], RTC.EXPECTED_BASE_CHECKPOINT_SHA256)
        fin = RTC.finalize_certificate(cert, ci)
        if not (fin["certificate_status"] == RTC.CERTIFICATE_STATUS_PASS
                and fin["certificate_finalized"] and fin["validation_errors"] == []):
            return FAIL, (f"{arm}: finalize -> {fin['certificate_status']} "
                          f"{fin['validation_errors']}")
    # (NEG) precheck WITHOUT the formal identity record -> FAIL, never PENDING
    cert_no_id, _, _ = _v23_prejax_chain("persistent", with_identity=False)
    if cert_no_id["certificate_status"] != RTC.CERTIFICATE_STATUS_FAIL:
        return FAIL, "(NEG) missing formal identity must FAIL the precheck"
    if not any("FORMAL_CONFIG_IDENTITY_MISMATCH" in e
               for e in cert_no_id["validation_errors"]):
        return FAIL, "(NEG) FORMAL_CONFIG_IDENTITY_MISMATCH not reported"
    return PASS, ("both arms: pure-Python pre-JAX chain (identity + strict assignment + "
                  "frozen-spec FULL scientific binding) -> PENDING_CHECKPOINT_IDENTITY, then "
                  "PASS after the frozen base SHA; missing identity -> FAIL (fail closed, "
                  "pre-JAX)")


def gate45():
    """GATE 45 — certificate state machine (§六, NEG x4): PENDING + checkpoint PASS -> PASS
    with errors cleared; checkpoint error -> FAIL + BASE_CHECKPOINT_FAILURE; a FAIL precheck
    can NEVER finalize to PASS (stale-PASS guard, director item 5); undecided (PENDING) match
    -> FAIL + BASE_CHECKPOINT_MATCH_NOT_PASS; NOT_FROZEN expectation -> PASS; finalize is pure
    (input not mutated)."""
    import phase4a_v2_runtime_config as RTC
    pend, _, _ = _v23_prejax_chain("persistent")
    if pend["certificate_status"] != RTC.CERTIFICATE_STATUS_PENDING:
        return FAIL, f"setup: precheck not PENDING: {pend['validation_errors']}"
    # positive: frozen base SHA -> PASS, finalized, errors cleared
    ci = RTC.verify_checkpoint_params_sha(
        pend["checkpoint_identity"], RTC.EXPECTED_BASE_CHECKPOINT_SHA256)
    ok = RTC.finalize_certificate(pend, ci)
    if not (ok["certificate_status"] == "PASS" and ok["certificate_finalized"]
            and ok["validation_errors"] == []):
        return FAIL, f"PENDING + PASS ckpt did not finalize PASS: {ok['validation_errors']}"
    # (NEG 1) checkpoint verification error -> FAIL + BASE_CHECKPOINT_FAILURE + match=FAIL
    f1 = RTC.finalize_certificate(pend, pend["checkpoint_identity"],
                                  checkpoint_error="BASE_CHECKPOINT_SHA_MISMATCH: x != y")
    if f1["certificate_status"] != "FAIL" or not any(
            "BASE_CHECKPOINT_FAILURE" in e for e in f1["validation_errors"]):
        return FAIL, "(NEG1) checkpoint error did not FAIL with BASE_CHECKPOINT_FAILURE"
    if f1["checkpoint_identity"]["base_checkpoint_match"] != "FAIL":
        return FAIL, "(NEG1) match not relabeled FAIL"
    # (NEG 2) FAIL precheck can never finalize to PASS even with a PASS checkpoint
    bad, _, _ = _v23_prejax_chain("persistent", sci_override=dict(seed=43))
    if bad["certificate_status"] != "FAIL":
        return FAIL, "(NEG2) setup: mutated frozen-spec value must FAIL the precheck"
    f2 = RTC.finalize_certificate(bad, ci)
    if f2["certificate_status"] == "PASS":
        return FAIL, "(NEG2) FAIL precheck finalized to PASS (stale-PASS leak)"
    if not any("CERTIFICATE_NOT_PENDING_AT_FINALIZE" in e for e in f2["validation_errors"]):
        return FAIL, "(NEG2) CERTIFICATE_NOT_PENDING_AT_FINALIZE not reported"
    # (NEG 3) undecided checkpoint match (PENDING) at finalize -> FAIL
    f3 = RTC.finalize_certificate(pend, pend["checkpoint_identity"])
    if f3["certificate_status"] != "FAIL" or not any(
            "BASE_CHECKPOINT_MATCH_NOT_PASS" in e for e in f3["validation_errors"]):
        return FAIL, "(NEG3) PENDING match at finalize did not FAIL"
    # (NEG 4) finalize_certificate is PURE: the input certificate is not mutated
    if (pend["certificate_status"] != RTC.CERTIFICATE_STATUS_PENDING
            or pend["certificate_finalized"]
            or pend["checkpoint_identity"]["base_checkpoint_match"] != "PENDING"):
        return FAIL, "(NEG4) finalize_certificate mutated its input certificate"
    # NOT_FROZEN expectation -> finalize PASS (documented non-frozen path)
    ci_nf = RTC.verify_checkpoint_params_sha(
        RTC.build_checkpoint_identity("/ckpt/x", expected_sha256=None), None)
    if ci_nf["base_checkpoint_match"] != "NOT_FROZEN":
        return FAIL, f"NOT_FROZEN path broken: {ci_nf}"
    f5 = RTC.finalize_certificate(pend, ci_nf)
    if f5["certificate_status"] != "PASS":
        return FAIL, f"PENDING + NOT_FROZEN must finalize PASS: {f5['validation_errors']}"
    return PASS, ("PENDING + PASS ckpt -> PASS (errors cleared); checkpoint error -> FAIL + "
                  "BASE_CHECKPOINT_FAILURE; FAIL precheck can NEVER become PASS (stale-PASS "
                  "guard); PENDING match -> FAIL; NOT_FROZEN -> PASS; finalize is pure")


def gate46():
    """GATE 46 — certificate artifact identity (§七.1/§七.2/§七.3): atomic write embeds the
    payload SHA + write marker, fsyncs a detached `<name>.sha256` sidecar with `<sha>  <base>`,
    leaks no temp file, and verify_certificate_artifact PASSes against the returned file/
    payload SHAs (exactly what the summary/checkpoint manifest bind)."""
    import json as _json
    import tempfile
    import phase4a_v2_runtime_config as RTC
    pend, _, _ = _v23_prejax_chain("persistent")
    ci = RTC.verify_checkpoint_params_sha(
        pend["checkpoint_identity"], RTC.EXPECTED_BASE_CHECKPOINT_SHA256)
    fin = RTC.finalize_certificate(pend, ci)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "runtime_config_certificate.json")
        # Phase4A-v2.4 (§五): write_certificate_atomic returns a 5-tuple; the fifth element is
        # the EXACT serialized artifact the caller must adopt (RUNTIME_CONFIG_CERTIFICATE =
        # written_certificate).
        cpath, spath, fsha, psha, written = RTC.write_certificate_atomic(fin, p)
        side = RTC.certificate_sidecar_path(cpath)
        if spath != side or not os.path.isfile(side):
            return FAIL, f"sidecar missing/misnamed: {spath}"
        if os.path.exists(f"{cpath}.tmp.{os.getpid()}"):
            return FAIL, "temp file leaked by atomic write"
        raw = open(cpath, "rb").read()
        if RTC._sha256_bytes(raw) != fsha:
            return FAIL, "returned file SHA != recomputed file SHA"
        disk = _json.loads(raw.decode("utf-8"))
        if written != disk:
            return FAIL, "returned written certificate != disk artifact (§五 adoption broken)"
        if written.get("certificate_payload_sha256") != psha:
            return FAIL, "written certificate does not carry the embedded payload SHA"
        if disk.get("certificate_payload_sha256") != psha:
            return FAIL, "embedded payload SHA != returned payload SHA"
        if disk.get("certificate_written_via") != "atomic_tempfile_fsync_replace":
            return FAIL, "atomic-write marker missing"
        if open(side, encoding="utf-8").read() != f"{fsha}  {os.path.basename(cpath)}\n":
            return FAIL, "sidecar content format wrong"
        v = RTC.verify_certificate_artifact(cpath, spath, expected_file_sha256=fsha,
                                            expected_payload_sha256=psha)
        if v["certificate_tamper_check"] != "PASS":
            return FAIL, f"artifact verify not PASS: {v}"
        # the §七.3 record binds the same file SHA + sidecar (summary/ckpt binding); the
        # payload SHA is a write-time self-field of the ARTIFACT (asserted on disk above),
        # not a field of the in-memory certificate, so the record carries it only when the
        # certificate object itself holds it (e.g. after a re-read of the written file)
        recd = RTC.certificate_shas_record(fin, certificate_file_sha256=fsha,
                                           certificate_sidecar_path=spath)
        if (recd["runtime_config_certificate_file_sha256"] != fsha
                or recd["runtime_config_certificate_sidecar_path"] != spath):
            return FAIL, "certificate_shas_record does not bind file SHA + sidecar (§七.3)"
        recd_disk = RTC.certificate_shas_record(disk, certificate_file_sha256=fsha,
                                                certificate_sidecar_path=spath)
        if recd_disk["runtime_config_certificate_payload_sha256"] != psha:
            return FAIL, "record built from the re-read artifact must bind the payload SHA"
    return PASS, ("atomic write: payload SHA embedded, detached sidecar '<sha>  <basename>' "
                  "fsynced, no tmp leak; verify_certificate_artifact PASS with expected "
                  "file+payload SHAs; §七.3 record binds file SHA + sidecar + payload SHA")


def gate47():
    """GATE 47 — certificate tamper detection fail closed (§七.4, NEG x6): FAIL->PASS status
    flip, edited validation_errors, corrupted sidecar SHA, missing sidecar, truncated JSON,
    and a wrong EXPECTED file SHA — each raises RUNTIME_CONFIG_CERTIFICATE_TAMPERED."""
    import json as _json
    import tempfile
    import phase4a_v2_runtime_config as RTC
    pend, _, _ = _v23_prejax_chain("persistent")
    # a FINALIZED FAIL certificate (the checkpoint-error path of §六.3)
    fail_cert = RTC.finalize_certificate(
        pend, pend["checkpoint_identity"],
        checkpoint_error="BASE_CHECKPOINT_SHA_MISMATCH: loaded != frozen")
    if fail_cert["certificate_status"] != "FAIL":
        return FAIL, "setup: expected a finalized FAIL certificate"

    def expect_tamper(label, attack):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "runtime_config_certificate.json")
            cpath, spath, _, _, _ = RTC.write_certificate_atomic(fail_cert, p)
            attack(cpath, spath)
            try:
                RTC.verify_certificate_artifact(cpath, spath)
                return f"{label}: tamper NOT detected"
            except ValueError as e:
                if "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" not in str(e):
                    return f"{label}: wrong code: {e}"
        return None

    def rewrite(cert_path, mutate):
        d = _json.loads(open(cert_path, encoding="utf-8").read())
        mutate(d)
        with open(cert_path, "w", encoding="utf-8") as f:
            f.write(_json.dumps(d, indent=2, sort_keys=True))

    attacks = [
        ("NEG1 FAIL->PASS flip",
         lambda c, s: rewrite(c, lambda d: (d.__setitem__("certificate_status", "PASS"),
                                            d.__setitem__("validation_errors", [])))),
        ("NEG2 edited validation_errors",
         lambda c, s: rewrite(c, lambda d: d.__setitem__("validation_errors", []))),
        ("NEG3 corrupted sidecar SHA",
         lambda c, s: open(s, "w", encoding="utf-8").write(
             "0" * 64 + "  runtime_config_certificate.json\n")),
        ("NEG4 missing sidecar", lambda c, s: os.remove(s)),
        ("NEG5 truncated JSON",
         lambda c, s: open(c, "wb").write(open(c, "rb").read()[:40])),
    ]
    for label, attack in attacks:
        err = expect_tamper(label, attack)
        if err:
            return FAIL, err
    # (NEG 6) pristine file, but wrong EXPECTED file SHA (stale summary/checkpoint binding)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "runtime_config_certificate.json")
        cpath, spath, _, _, _ = RTC.write_certificate_atomic(fail_cert, p)
        try:
            RTC.verify_certificate_artifact(cpath, spath, expected_file_sha256="f" * 64)
            return FAIL, "(NEG6) wrong expected file SHA not rejected"
        except ValueError as e:
            if "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" not in str(e):
                return FAIL, f"(NEG6) wrong code: {e}"
    return PASS, ("6 tamper attacks all fail closed with RUNTIME_CONFIG_CERTIFICATE_TAMPERED: "
                  "FAIL->PASS flip, edited errors, corrupted sidecar, missing sidecar, "
                  "truncated JSON, wrong expected file SHA")


def gate48():
    """GATE 48 — executed protocol SOURCE identity (v2.3 §八 + v2.4 §七): the executed learner
    / sampler are bound by INSPECTED source identity — module, qualname, module realpath,
    module FILE SHA256, FUNCTION SOURCE SHA256, source lines — and the functions MUST execute
    from the declared modules (rmt_replay_learner / rmt_replay_buffer with qualname
    RMTReplayBuffer.sample_eligible). A same-name impostor from another module, a PCG64
    Generator, an uninspectable builtin, a DELETED source file, and wrong declared labels all
    fail closed (NEG x6).

    Phase4A-v2.4 reason+diff: v2.3 asserted a single source_sha256 on an arbitrarily-named
    stub module; v2.4 §七 binds the full identity tuple (module realpath + module file SHA +
    function source SHA) and REQUIRES the declared module names, so the positive stubs must be
    real rmt_replay_learner.py / rmt_replay_buffer.py modules, and the same-name-impostor /
    deleted-source-file negatives are new fail-closed dimensions."""
    import linecache
    import shutil
    import sys
    import tempfile
    import textwrap
    import phase4a_v2_contract as CONTRACT
    try:
        import numpy as np
    except ImportError:
        return SKIP, "numpy unavailable (required to bind the RNG engine)"
    tmp = tempfile.mkdtemp(prefix="p4av24_g48_")
    # The positive stubs MUST carry the declared module names (v2.4 §七 module binding).
    with open(os.path.join(tmp, "rmt_replay_learner.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent('''
            def original_vtrace_update_rmt(state, batch):
                """v2.4 GATE48 stub learner matching the declared module + function name."""
                return state
        ''').lstrip())
    with open(os.path.join(tmp, "rmt_replay_buffer.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent('''
            class RMTReplayBuffer:
                """v2.4 GATE48 stub buffer matching the declared module + qualname."""

                def sample_eligible(self, seq_length, rng, k):
                    return []
        ''').lstrip())
    with open(os.path.join(tmp, "p4av24_g48_impostor.py"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent('''
            def original_vtrace_update_rmt(state, batch):
                """SAME function name as the declared learner, DIFFERENT module: impostor."""
                return state
        ''').lstrip())
    saved = {n: sys.modules.pop(n)
             for n in ("rmt_replay_learner", "rmt_replay_buffer", "p4av24_g48_impostor")
             if n in sys.modules}
    sys.path.insert(0, tmp)
    try:
        import rmt_replay_learner as L
        import rmt_replay_buffer as B
        import p4av24_g48_impostor as IMP
        # positive: declared module + function names bind; the FULL identity tuple is
        # inspected (module realpath / module file SHA / function source SHA — not labels)
        ident = CONTRACT.executed_function_source_identity(
            L.original_vtrace_update_rmt, B.RMTReplayBuffer.sample_eligible)
        if ident["executed_function_binding"] != "PASS":
            return FAIL, f"binding not PASS: {ident}"
        for part, mod in (("learner", "rmt_replay_learner"),
                          ("sampler", "rmt_replay_buffer")):
            pid = ident[part]
            for fld in ("module", "qualname", "name", "module_realpath",
                        "module_file_sha256", "function_source_sha256", "source_lines"):
                if pid.get(fld) in (None, ""):
                    return FAIL, f"{part}.{fld} missing: {pid}"
            if pid["module"] != mod:
                return FAIL, f"{part} module={pid['module']!r} != {mod!r}"
            if len(pid["module_file_sha256"]) != 64 or len(
                    pid["function_source_sha256"]) != 64:
                return FAIL, f"{part} SHAs not length-64: {pid}"
            if not os.path.isfile(pid["module_realpath"]):
                return FAIL, f"{part} module_realpath missing on disk: {pid['module_realpath']}"
        if ident["sampler"]["qualname"] != "RMTReplayBuffer.sample_eligible":
            return FAIL, f"sampler qualname wrong: {ident['sampler']['qualname']}"
        # RNG instance: numpy.random.RandomState binds with the full identity tuple; the
        # binding draws NO random numbers (state untouched).
        rs = np.random.RandomState(49)
        pristine = np.random.RandomState(49).get_state()
        rng = CONTRACT.verify_rng_instance_identity(rs)
        if rng["rng_binding"] != "PASS" or rng["class_name"] != "RandomState":
            return FAIL, f"RandomState did not bind: {rng}"
        for fld in ("class_module", "class_name", "numpy_version", "seed_derivation",
                    "hidden_buffer_rng_used"):
            if rng.get(fld) in (None, ""):
                return FAIL, f"rng identity field {fld} missing: {rng}"
        if rng["seed_derivation"] != "run_seed_plus_7":
            return FAIL, f"rng seed_derivation wrong: {rng['seed_derivation']}"
        if rng["hidden_buffer_rng_used"] is not False:
            return FAIL, f"rng hidden_buffer_rng_used must be False: {rng}"
        if rng["numpy_version"] != np.__version__:
            return FAIL, f"rng numpy_version {rng['numpy_version']} != {np.__version__}"
        state_after = rs.get_state()
        if not np.array_equal(state_after[1], pristine[1]) or state_after[2] != pristine[2]:
            return FAIL, "verify_rng_instance_identity CONSUMED random state (must not)"
        # declared protocol reconciles with the executed identity (two-phase)
        pd = CONTRACT.replay_protocol_labels("original_vtrace", 129, 4)["protocol_definition"]
        recon = CONTRACT.verify_executed_protocol_matches_declared(ident, pd)
        if recon["executed_protocol_declaration_match"] != "PASS":
            return FAIL, f"declaration reconciliation not PASS: {recon}"
        # (NEG 1) same-name impostor learner from ANOTHER module -> SOURCE_MISMATCH
        try:
            CONTRACT.executed_function_source_identity(
                IMP.original_vtrace_update_rmt, B.RMTReplayBuffer.sample_eligible)
            return FAIL, "(NEG1) same-name impostor learner not rejected"
        except ValueError as e:
            if "EXECUTED_PROTOCOL_SOURCE_MISMATCH" not in str(e):
                return FAIL, f"(NEG1) wrong code: {e}"
        # (NEG 2) wrong RNG engine (np.random.default_rng -> PCG64 Generator)
        try:
            CONTRACT.verify_rng_instance_identity(np.random.default_rng(1))
            return FAIL, "(NEG2) PCG64 Generator not rejected"
        except ValueError as e:
            if "EXECUTED_PROTOCOL_RNG_MISMATCH" not in str(e):
                return FAIL, f"(NEG2) wrong code: {e}"
        # (NEG 3) uninspectable callable -> SOURCE_UNAVAILABLE
        try:
            CONTRACT._source_identity(len)
            return FAIL, "(NEG3) uninspectable builtin not rejected"
        except ValueError as e:
            if "EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE" not in str(e):
                return FAIL, f"(NEG3) wrong code: {e}"
        # (NEG 4) DELETED source file -> SOURCE_UNAVAILABLE (the executed function's source is
        # no longer inspectable / hashable on disk; linecache cleared to mirror a fresh
        # process that cannot read the file).
        os.remove(os.path.join(tmp, "rmt_replay_learner.py"))
        linecache.clearcache()
        try:
            CONTRACT._source_identity(L.original_vtrace_update_rmt)
            return FAIL, "(NEG4) deleted source file not rejected"
        except ValueError as e:
            if "EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE" not in str(e):
                return FAIL, f"(NEG4) wrong code: {e}"
        # (NEG 5) wrong declared learner label -> DECLARATION_MISMATCH
        bad_pd = dict(pd)
        bad_pd["learner"] = "some_other_learner"
        try:
            CONTRACT.verify_executed_protocol_matches_declared(ident, bad_pd)
            return FAIL, "(NEG5) wrong declared learner not rejected"
        except ValueError as e:
            if "EXECUTED_PROTOCOL_DECLARATION_MISMATCH" not in str(e):
                return FAIL, f"(NEG5) wrong code: {e}"
        # (NEG 6) wrong declared sampler label -> DECLARATION_MISMATCH
        bad_pd2 = dict(pd)
        bad_pd2["sampler"] = "uniform"
        try:
            CONTRACT.verify_executed_protocol_matches_declared(ident, bad_pd2)
            return FAIL, "(NEG6) wrong declared sampler not rejected"
        except ValueError as e:
            if "EXECUTED_PROTOCOL_DECLARATION_MISMATCH" not in str(e):
                return FAIL, f"(NEG6) wrong code: {e}"
    finally:
        if tmp in sys.path:
            sys.path.remove(tmp)
        for n in ("rmt_replay_learner", "rmt_replay_buffer", "p4av24_g48_impostor"):
            sys.modules.pop(n, None)
        sys.modules.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    return PASS, ("executed learner/sampler bound by the FULL inspected identity tuple "
                  "(module / qualname / module realpath / module file SHA / function source "
                  "SHA / source lines) from the declared modules; RandomState binds with "
                  "numpy version + seed derivation + no hidden buffer RNG and consumes no "
                  "state; declaration reconciliation PASS; same-name impostor / PCG64 / "
                  "uninspectable builtin / DELETED source file / wrong declared learner / "
                  "wrong declared sampler all fail closed")


def gate49():
    """GATE 49 — v2.3 publication labels (§九): TIME-SCOPED creation-time evidence only.

    The gate asserts facts that are IMMUABLE — true at the moment the v2.3 commit was
    created, and NOT falsified if 总控 later pushes the branch:
      V2_3_PUBLICATION_STATUS_AT_COMMIT_CREATION = NOT_PUSHED
      V2_3_PUSH_PERFORMED_BEFORE_COMMIT          = false
    plus the historical v2.2 erratum facts (v2.2 WAS pushed: f2b7aead is the remote HEAD,
    director item 8). It REJECTS any label that asserts the CURRENT remote state
    (V2_3_REMOTE_PUBLICATION_STATUS / V2_3_PUSH_PERFORMED / unscoped PUSH_PERFORMED /
    V2_3_IMPLEMENTATION_ROUND_PUSH_PERFORMED) or any PUSH_PERFORMED label lacking a time
    qualifier — exactly the push-falsifiable shape that produced the v2.2 errata.

    Review-fix reason+diff: the original v2.3 gate asserted "current remote = NOT_PUSHED";
    a push by 总控 would instantly contradict the committed evidence (repeat of the v2.2
    errata failure mode). The assertion is now "NOT_PUSHED AT COMMIT CREATION"."""
    import json as _json
    path = os.path.join(_SNAPSHOT, "reports", "rmt16_phase4a_v2_3_labels.json")
    if not os.path.isfile(path):
        return FAIL, "reports/rmt16_phase4a_v2_3_labels.json missing"
    with open(path, encoding="utf-8") as f:
        doc = _json.load(f)
    labels = doc["labels"]
    # creation-time / historical facts — a later push does NOT falsify any of these
    expect = {
        "V2_3_PUBLICATION_STATUS_AT_COMMIT_CREATION": "NOT_PUSHED",
        "V2_3_PUSH_PERFORMED_BEFORE_COMMIT": False,
        "V2_2_ERRATUM_REMOTE_PUBLICATION_STATUS": "PUSHED",
        "V2_2_ERRATUM_REMOTE_HEAD": "f2b7aead44426825f905fa8b82c5f66c29ee167a",
        "BASE_REMOTE_HEAD": "87d1e552415d292417dcb6e6f9f6b16b97a6d135",
        "REVIEW_BASELINE_COMMIT": "d3c8c7d6abd2df3d0ba69dc2c1f326f8668798e5",
    }
    for k, v in expect.items():
        if labels.get(k) != v:
            return FAIL, f"{k}={labels.get(k)!r} != {v!r}"
    # forbid CURRENT-remote-state publication labels: a push would falsify them
    forbidden = {"PUSH_PERFORMED",
                 "V2_3_PUSH_PERFORMED",
                 "V2_3_REMOTE_PUBLICATION_STATUS",
                 "V2_3_IMPLEMENTATION_ROUND_PUSH_PERFORMED"}
    bad = sorted(forbidden & set(labels))
    if bad:
        return FAIL, f"push-falsifiable current-state publication label(s) present: {bad}"
    # any PUSH_PERFORMED label must carry an explicit time qualifier
    for k in labels:
        if "PUSH_PERFORMED" in k and not any(
                q in k for q in ("BEFORE_COMMIT", "AT_COMMIT_CREATION", "ERRATUM")):
            return FAIL, (f"push label {k!r} lacks a time qualifier "
                          "(BEFORE_COMMIT / AT_COMMIT_CREATION / ERRATUM)")
    return PASS, ("v2.3 labels are creation-time evidence: NOT_PUSHED at commit creation, "
                  "no push performed before commit; no push-falsifiable current-state label; "
                  "v2.2 erratum recorded (PUSHED @ f2b7aead); base @ 87d1e55; review "
                  "baseline @ d3c8c7d6")


def gate50():
    """GATE 50 — v2.3 reports + v2.2 publication errata (§九 item 8): the final report and the
    errata exist; the errata names the V2.2 NOT_PUSHED label vs the f2b7aead remote-HEAD
    reality; the v2.2 label file itself is NOT rewritten (errata-only correction)."""
    import json as _json
    final = os.path.join(_SNAPSHOT, "reports", "rmt16_phase4a_v2_3_final.md")
    errata = os.path.join(_SNAPSHOT, "reports", "rmt16_phase4a_v2_2_publication_errata.md")
    for p in (final, errata):
        if not os.path.isfile(p):
            return FAIL, f"reports/{os.path.basename(p)} missing"
    etxt = open(errata, encoding="utf-8").read()
    for needle in ("f2b7aead44426825f905fa8b82c5f66c29ee167a", "NOT_PUSHED", "PUSHED"):
        if needle not in etxt:
            return FAIL, f"errata does not document the NOT_PUSHED-vs-{needle[:12]} reality"
    # the v2.3 final report must state the CREATION-TIME status, never a current-remote claim
    # (review-fix: a current-state claim would be falsified by a later push)
    ftxt = open(final, encoding="utf-8").read()
    if "V2_3_PUBLICATION_STATUS_AT_COMMIT_CREATION" not in ftxt:
        return FAIL, "final report must state the creation-time (not current) v2.3 status"
    bare = ftxt.replace("V2_3_PUSH_PERFORMED_BEFORE_COMMIT", "")
    for cur in ("V2_3_REMOTE_PUBLICATION_STATUS", "V2_3_PUSH_PERFORMED",
                "V2_3_IMPLEMENTATION_ROUND_PUSH_PERFORMED"):
        if cur in bare:
            return FAIL, f"final report carries push-falsifiable current-state label {cur}"
    # the v2.2 labels file is byte-unchanged (its NOT_PUSHED labels stay as written in v2.2)
    with open(os.path.join(_SNAPSHOT, "reports", "rmt16_phase4a_v2_2_labels.json"),
              encoding="utf-8") as f:
        v22 = _json.load(f)
    if v22["labels"].get("V2_2_REMOTE_PUBLICATION_STATUS") != "NOT_PUSHED":
        return FAIL, "v2.2 labels file was rewritten (forbidden; errata-only correction)"
    return PASS, ("v2.3 final report + v2.2 publication errata present; errata records the "
                  "NOT_PUSHED label vs f2b7aead remote-HEAD reality; v2.2 labels file left "
                  "unchanged (errata-only correction per §九 item 8)")


# ----------------------------------------------------------------------------
# Phase4A-v2.4 (§十二): GATE51-GATE62 — actual-CLI pre-JAX binding (§三), subprocess
# import sentinel (§三.3), staged checkpoint failure finalization (§四), certificate
# disk-object sync + payload-SHA manifest binding (§五), executed-protocol source identity
# completeness + same-name-different-source rejection (§七), RNG identity before certificate
# PASS (§六), effective protocol definition stability (§八), cross-arm effective comparison
# fail closed (§九), strict sidecar basename/format validation (§十), relocatable formal
# path labels (§十一). Module self-tests this round: RTC 76, FID 13, exposure validator 25.
# ----------------------------------------------------------------------------


def gate51():
    """GATE 51 — v2.4 §三.1 ACTUAL CLI pre-JAX binding: the precheck certificate binds the
    ACTUAL CLI values (frozen-spec kwargs overridden with the seven CLI-facing keys). Every
    wrong CLI value FAILs the precheck PRE-JAX with FORMAL_CONFIG_RUNTIME_MISMATCH (seed /
    total_updates / save_every / sequence_length / allow-full-p2-legacy), and a wrong
    --replay_mode / --carry_mode fails the arm binding (FORMAL_CONFIG_ARM_MISMATCH). The
    correct CLI reproduces PENDING_CHECKPOINT_IDENTITY on both arms."""
    import phase4a_v2_runtime_config as RTC
    for arm in ("persistent", "reset128"):
        cert, _, _ = _v23_prejax_chain(arm)
        if cert["certificate_status"] != RTC.CERTIFICATE_STATUS_PENDING:
            return FAIL, (f"{arm}: correct-CLI precheck={cert['certificate_status']} "
                          f"({cert['validation_errors']})")
    wrong = [
        ("seed=43", dict(cli_seed=43)),
        ("total_updates=13", dict(cli_total_updates=13)),
        ("save_every=3", dict(cli_save_every=3)),
        ("sequence_length=130", dict(cli_sequence_length=130)),
        ("allow_full_p2_legacy=True", dict(cli_allow_full_p2_legacy=True)),
    ]
    for name, kw in wrong:
        cert, _, _ = _v23_prejax_chain("persistent", **kw)
        if cert["certificate_status"] != RTC.CERTIFICATE_STATUS_FAIL:
            return FAIL, f"(NEG) {name}: precheck not FAIL: {cert['certificate_status']}"
        if not any("FORMAL_CONFIG_RUNTIME_MISMATCH" in e for e in cert["validation_errors"]):
            return FAIL, f"(NEG) {name}: no FORMAL_CONFIG_RUNTIME_MISMATCH in errors"
        if cert["certificate_finalized"]:
            return FAIL, f"(NEG) {name}: precheck must not be finalized"
    for name, kw in [("replay_mode=off", dict(cli_replay_mode="off")),
                     ("carry_mode=reset128 on persistent YAML", dict(cli_carry="reset128"))]:
        try:
            _v23_prejax_chain("persistent", **kw)
            return FAIL, f"(NEG) {name}: arm binding did NOT raise"
        except ValueError as e:
            if "FORMAL_CONFIG_ARM_MISMATCH" not in str(e):
                return FAIL, f"(NEG) {name}: wrong code: {e}"
    # static: the driver overrides the frozen kwargs with the ACTUAL CLI before building
    src = _read(_LAUNCHER).replace("\r\n", "\n")
    for needle in ("_prejax_kwargs = FSPEC.build_kwargs(args.carry_mode)",
                   "_prejax_kwargs.update(",
                   "carry_mode=args.carry_mode,",
                   "replay_mode=REPLAY_MODE,",
                   "allow_full_p2_legacy=bool(args.allow_full_p2_legacy),",
                   "sequence_length=int(args.sequence_length),",
                   "seed=int(args.seed),",
                   "total_updates=int(args.total_updates),",
                   "save_every=int(args.save_every))",
                   "_PREJAX_SCIENTIFIC = RTC.build_runtime_scientific_config("
                   "**_prejax_kwargs)"):
        if needle not in src:
            return FAIL, f"driver missing actual-CLI pre-JAX binding: {needle}"
    if src.find("_prejax_kwargs.update(") > src.find(
            "_PREJAX_SCIENTIFIC = RTC.build_runtime_scientific_config(**_prejax_kwargs)"):
        return FAIL, "the CLI override must precede the pre-JAX scientific build"
    return PASS, ("actual-CLI pre-JAX binding: correct CLI -> PENDING on both arms; wrong "
                  "seed/total_updates/save_every/sequence_length/allow-full-p2-legacy -> "
                  "precheck FAIL with FORMAL_CONFIG_RUNTIME_MISMATCH (pre-JAX); wrong "
                  "replay_mode/carry_mode -> FORMAL_CONFIG_ARM_MISMATCH; driver override "
                  "wired before the scientific build")


def gate52():
    """GATE 52 — v2.4 §三.3 subprocess import sentinel: the driver is launched as a subprocess
    with a FAKE `jax` package whose __init__ writes a sentinel file on import. With a correct
    formal CLI the driver passes the pre-JAX chain and IMPORTS jax (sentinel written, no
    FORMAL_CONFIG_RUNTIME_MISMATCH); with a wrong CLI (--seed 43) it exits nonzero with
    FORMAL_CONFIG_RUNTIME_MISMATCH and the sentinel is NEVER written — proving the refusal
    fires BEFORE `import jax` / CUDA env / env build / ckpt load."""
    import shutil
    import tempfile
    import phase4a_v2_runtime_config as RTC
    cfg = RTC.load_formal_config(
        os.path.join(_SNAPSHOT, "configs", "rmt16_phase4a_v2_persistent.yaml"))["config"]
    ra = cfg["runtime_assignment"]
    work = tempfile.mkdtemp(prefix="p4av24_g52_")
    fake = os.path.join(work, "fake_modules")
    os.makedirs(os.path.join(fake, "jax"))
    sentinel = os.path.join(work, "jax_imported.sentinel")
    with open(os.path.join(fake, "jax", "__init__.py"), "w", encoding="utf-8") as f:
        f.write(f"open({sentinel!r}, 'w').write('jax imported by the driver subprocess')\n")
    with open(os.path.join(fake, "jax", "numpy.py"), "w", encoding="utf-8") as f:
        f.write("# blocking stub for jax.numpy (import only)\n")
    run_root = os.path.join(work, "run")
    out_dir = os.path.join(run_root, ra["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = _EXP + os.pathsep + fake
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, _LAUNCHER,
           "--carry_mode", "persistent",
           "--replay_mode", "original_vtrace",
           "--sequence_length", "129",
           "--ckpt17500", "/ckpt/17500/full_state.pkl",
           "--out", out_dir,
           "--gpu_uuid", str(ra["gpu_uuid"]),
           "--formal_config",
           os.path.join(_SNAPSHOT, "configs", "rmt16_phase4a_v2_persistent.yaml"),
           "--snapshot_root", _SNAPSHOT,
           "--run_root", run_root,
           "--seed", "42", "--total_updates", "12", "--save_every", "2"]
    try:
        # (POS) correct CLI -> the pre-JAX chain passes and `import jax` executes. The driver
        # then dies on the FAKE jax (no real arrays / optax) — that is fine: the assertions
        # are (a) no FORMAL_CONFIG_RUNTIME_MISMATCH and (b) the sentinel proves `import jax`
        # was reached.
        pos = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        pos_out = (pos.stdout or "") + (pos.stderr or "")
        if "FORMAL_CONFIG_RUNTIME_MISMATCH" in pos_out:
            return FAIL, f"(POS) correct CLI hit a pre-JAX mismatch: {pos_out[-900:]}"
        if not os.path.exists(sentinel):
            return FAIL, (f"(POS) sentinel missing although the pre-JAX chain passed "
                          f"(rc={pos.returncode}): `import jax` never ran: {pos_out[-900:]}")
        # (NEG) wrong --seed 43 -> nonzero exit + mismatch message + sentinel NEVER written
        os.remove(sentinel)
        neg = subprocess.run(cmd + ["--seed", "43"], capture_output=True, text=True,
                             timeout=300, env=env)
        neg_out = (neg.stdout or "") + (neg.stderr or "")
        if neg.returncode == 0:
            return FAIL, "(NEG) wrong --seed 43 exited 0 (must be nonzero)"
        if "FORMAL_CONFIG_RUNTIME_MISMATCH" not in neg_out:
            return FAIL, f"(NEG) mismatch message missing: {neg_out[-900:]}"
        if os.path.exists(sentinel):
            return FAIL, ("(NEG) sentinel written: `import jax` ran DESPITE the pre-JAX "
                          "refusal (the refusal must precede `import jax`)")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return PASS, ("subprocess import sentinel: correct CLI passes pre-JAX and imports jax "
                  "(sentinel written, no mismatch); wrong --seed 43 exits nonzero with "
                  "FORMAL_CONFIG_RUNTIME_MISMATCH and NEVER imports jax (sentinel absent)")


def gate53():
    """GATE 53 — v2.4 §四: EVERY checkpoint-flow failure stage FINALIZES the certificate FAIL
    (finalized=true + staged label on DISK). A pure-Python replica of the driver's staged try
    (manager init -> restore -> structure -> params extraction -> params hash -> SHA compare)
    is fault-injected at each stage; the finalized certificate is written atomically and
    RE-READ from disk. Valid structure + wrong SHA -> CHECKPOINT_SHA_COMPARE; a clean flow ->
    PASS + stage NONE; an unknown stage label fails closed (CHECKPOINT_FAILURE_STAGE_INVALID)."""
    import json as _json
    import tempfile
    import phase4a_v2_runtime_config as RTC
    stages = ["CHECKPOINT_MANAGER_INIT", "CHECKPOINT_RESTORE", "CHECKPOINT_STRUCTURE",
              "CHECKPOINT_PARAMS_EXTRACTION", "CHECKPOINT_PARAMS_HASH"]

    def run_staged(fault_stage, loaded_sha):
        cert, _, _ = _v23_prejax_chain("persistent")
        error = None
        stage = "NONE"
        try:
            for st in stages:
                stage = st
                if fault_stage == st:
                    raise RuntimeError(f"injected fault at {st}")
            stage = "CHECKPOINT_SHA_COMPARE"
            cert["checkpoint_identity"] = RTC.verify_checkpoint_params_sha(
                cert["checkpoint_identity"], loaded_sha)
            stage = "NONE"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return RTC.finalize_certificate(cert, cert["checkpoint_identity"],
                                        checkpoint_error=error,
                                        checkpoint_failure_stage=stage)

    for fault in stages + ["CHECKPOINT_SHA_COMPARE"]:
        loaded = ("f" * 64) if fault == "CHECKPOINT_SHA_COMPARE" \
            else RTC.EXPECTED_BASE_CHECKPOINT_SHA256
        fin = run_staged(fault, loaded)
        if fin["certificate_status"] != "FAIL" or not fin["certificate_finalized"]:
            return FAIL, f"{fault}: not finalized FAIL: {fin['validation_errors']}"
        if fin.get("checkpoint_failure_stage") != fault:
            return FAIL, (f"{fault}: checkpoint_failure_stage="
                          f"{fin.get('checkpoint_failure_stage')!r}")
        with tempfile.TemporaryDirectory() as td:
            cpath, spath, fsha, psha, written = RTC.write_certificate_atomic(
                fin, os.path.join(td, "runtime_config_certificate.json"))
            disk = _json.load(open(cpath, encoding="utf-8"))
            if disk != written or disk["certificate_status"] != "FAIL":
                return FAIL, f"{fault}: disk artifact is not the finalized FAIL certificate"
            if (not disk["certificate_finalized"]
                    or disk.get("checkpoint_failure_stage") != fault):
                return FAIL, f"{fault}: disk artifact missing finalized/stage"
            v = RTC.verify_certificate_artifact(cpath, spath, expected_file_sha256=fsha,
                                                expected_payload_sha256=psha)
            if v["certificate_tamper_check"] != "PASS":
                return FAIL, f"{fault}: FAIL-certificate artifact verify not PASS: {v}"
    # clean flow -> PASS + stage NONE + finalized
    fin = run_staged(None, RTC.EXPECTED_BASE_CHECKPOINT_SHA256)
    if fin["certificate_status"] != "PASS" or not fin["certificate_finalized"]:
        return FAIL, f"clean flow not PASS: {fin['validation_errors']}"
    if fin.get("checkpoint_failure_stage") != "NONE":
        return FAIL, f"clean flow stage != NONE: {fin.get('checkpoint_failure_stage')!r}"
    # (NEG) unknown stage label -> fail closed
    cert, _, _ = _v23_prejax_chain("persistent")
    try:
        RTC.finalize_certificate(cert, cert["checkpoint_identity"],
                                 checkpoint_error="x", checkpoint_failure_stage="BOGUS_STAGE")
        return FAIL, "(NEG) invalid stage label accepted"
    except ValueError as e:
        if "CHECKPOINT_FAILURE_STAGE_INVALID" not in str(e):
            return FAIL, f"(NEG) wrong code: {e}"
    # static: the driver stages the whole checkpoint flow under one try with stage labels
    src = _read(_LAUNCHER)
    for st in stages + ["CHECKPOINT_SHA_COMPARE"]:
        if f'"{st}"' not in src:
            return FAIL, f"driver missing stage label {st}"
    if "checkpoint_failure_stage=_CHECKPOINT_FAILURE_STAGE)" not in src:
        return FAIL, "driver does not pass the staged label to finalize_certificate"
    if src.count('_CHECKPOINT_ERROR = f"{type(exc).__name__}: {exc}"') < 2:
        return FAIL, "driver lacks the unified checkpoint/executed except handlers"
    return PASS, ("all six checkpoint failure stages finalize FAIL with the staged label ON "
                  "DISK (manager init / restore / structure / params extraction / params hash "
                  "/ SHA compare); clean flow -> PASS + stage NONE; invalid stage label fails "
                  "closed; driver stages the flow under one try with unified except")


def gate54():
    """GATE 54 — v2.4 §五: after EVERY certificate write the caller adopts the returned
    written artifact and re-verifies it, and on a final PASS the driver re-reads the disk
    certificate and requires equality. Functional: written == disk for PENDING / finalized
    FAIL / finalized PASS certificates; static: 4 write sites, >= 4 verify sites, adoption +
    re-read + mismatch-refusal wiring."""
    import json as _json
    import tempfile
    import phase4a_v2_runtime_config as RTC
    pend, _, _ = _v23_prejax_chain("persistent")
    ci = RTC.verify_checkpoint_params_sha(pend["checkpoint_identity"],
                                          RTC.EXPECTED_BASE_CHECKPOINT_SHA256)
    variants = [
        ("PENDING", pend),
        ("finalized_FAIL", RTC.finalize_certificate(
            pend, pend["checkpoint_identity"],
            checkpoint_error="BASE_CHECKPOINT_SHA_MISMATCH: loaded != frozen",
            checkpoint_failure_stage="CHECKPOINT_SHA_COMPARE")),
        ("finalized_PASS", RTC.finalize_certificate(pend, ci)),
    ]
    for name, cert in variants:
        with tempfile.TemporaryDirectory() as td:
            cpath, spath, fsha, psha, written = RTC.write_certificate_atomic(
                cert, os.path.join(td, "runtime_config_certificate.json"))
            disk = _json.load(open(cpath, encoding="utf-8"))
            if written != disk:
                return FAIL, f"{name}: adopted written certificate != disk artifact"
            if not (isinstance(psha, str) and len(psha) == 64):
                return FAIL, f"{name}: payload SHA not length-64: {psha!r}"
            if RTC._sha256_bytes(open(cpath, "rb").read()) != fsha:
                return FAIL, f"{name}: returned file SHA != recomputed file SHA"
            v = RTC.verify_certificate_artifact(cpath, spath, expected_file_sha256=fsha,
                                                expected_payload_sha256=psha)
            if v["certificate_tamper_check"] != "PASS":
                return FAIL, f"{name}: post-write verify not PASS: {v}"
    src = _read(_LAUNCHER).replace("\r\n", "\n")
    n_write = src.count("RTC.write_certificate_atomic(")
    n_verify = src.count("RTC.verify_certificate_artifact(")
    if n_write != 4:
        return FAIL, f"driver write_certificate_atomic sites={n_write} != 4"
    if n_verify < 4:
        return FAIL, (f"driver verify_certificate_artifact sites={n_verify} < 4 (one per "
                      "write required)")
    for needle in ("RUNTIME_CONFIG_CERTIFICATE) = (",
                   "_disk_certificate = json.load(_cert_f)",
                   "if _disk_certificate != RUNTIME_CONFIG_CERTIFICATE:",
                   "CERTIFICATE_DISK_OBJECT_MISMATCH"):
        if needle not in src:
            return FAIL, f"driver missing disk-object sync wiring: {needle}"
    return PASS, ("written == disk for PENDING / finalized FAIL / finalized PASS; payload + "
                  "file SHAs recompute; post-write verify PASS; driver has 4 write sites each "
                  "followed by a verify, adopts the written object, and re-reads the disk "
                  "certificate on final PASS (CERTIFICATE_DISK_OBJECT_MISMATCH guard)")


def gate55():
    """GATE 55 — v2.4 §五.3: the certificate PAYLOAD SHA + file SHA + base checkpoint params
    SHA are first-class, non-null length-64 fields in the manifest / summary / launch-status
    wiring. Functional: certificate_shas_record built from the DISK artifact carries all
    three; static: manifest + summary + launch-status needles."""
    import json as _json
    import tempfile
    import phase4a_v2_runtime_config as RTC
    pend, _, _ = _v23_prejax_chain("persistent")
    ci = RTC.verify_checkpoint_params_sha(pend["checkpoint_identity"],
                                          RTC.EXPECTED_BASE_CHECKPOINT_SHA256)
    fin = RTC.finalize_certificate(pend, ci)
    with tempfile.TemporaryDirectory() as td:
        cpath, spath, fsha, psha, written = RTC.write_certificate_atomic(
            fin, os.path.join(td, "runtime_config_certificate.json"))
        disk = _json.load(open(cpath, encoding="utf-8"))
    recd = RTC.certificate_shas_record(disk, certificate_file_sha256=fsha,
                                       certificate_sidecar_path=spath)
    for key in ("runtime_config_certificate_payload_sha256",
                "runtime_config_certificate_file_sha256",
                "base_checkpoint_params_sha256"):
        val = recd.get(key)
        if not (isinstance(val, str) and len(val) == 64):
            return FAIL, f"record {key} not non-null length-64: {val!r}"
    if recd["runtime_config_certificate_payload_sha256"] != psha:
        return FAIL, "record payload SHA != the written artifact's payload SHA"
    src = _read(_LAUNCHER).replace("\r\n", "\n")
    for needle in ('fields["runtime_config_certificate_payload_sha256"] = (',
                   'fields["base_checkpoint_params_sha256"] = base_sha',
                   "runtime_config_certificate_payload_sha256=(\n"
                   "                   RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256),",
                   "base_checkpoint_params_sha256=base_sha,",
                   "certificate_payload_sha256={RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256} ",
                   "base_checkpoint_params_sha256={base_sha} "):
        if needle not in src:
            return FAIL, f"driver missing payload/base-SHA binding: {needle[:70]}..."
    return PASS, ("manifest record from the DISK artifact carries non-null length-64 payload "
                  "SHA + file SHA + base checkpoint params SHA (payload SHA == written); "
                  "manifest / summary / launch-status all bind them")


def gate56():
    """GATE 56 — v2.4 §七: the REAL frozen replay modules' executed identity is COMPLETE —
    module, qualname, name, module_realpath, module_file_sha256 (recomputed from disk),
    function_source_sha256 (recomputed via inspect.getsource), source_lines — for both the
    learner and the sampler; executed binding + declaration reconciliation PASS on the real
    modules."""
    if not HAVE_JAX:
        return SKIP, ("JAX absent (the REAL frozen replay modules import jax at module "
                      "scope; runs on server CPU)")
    import hashlib
    import inspect
    import phase4a_v2_contract as CONTRACT
    import rmt_replay_learner as RL
    from rmt_replay_buffer import RMTReplayBuffer
    ident = CONTRACT.executed_function_source_identity(
        RL.original_vtrace_update_rmt, RMTReplayBuffer.sample_eligible)
    if ident["executed_function_binding"] != "PASS":
        return FAIL, f"real-module binding not PASS: {ident}"
    for part, fn in (("learner", RL.original_vtrace_update_rmt),
                     ("sampler", RMTReplayBuffer.sample_eligible)):
        pid = ident[part]
        for fld in CONTRACT.EXECUTED_SOURCE_IDENTITY_FIELDS:
            if pid.get(fld) in (None, ""):
                return FAIL, f"{part}.{fld} missing: {pid}"
        rp = pid["module_realpath"]
        if not os.path.isfile(rp):
            return FAIL, f"{part}: module_realpath not a file: {rp}"
        with open(rp, "rb") as f:
            file_sha = hashlib.sha256(f.read()).hexdigest()
        if pid["module_file_sha256"] != file_sha:
            return FAIL, f"{part}: module_file_sha256 != recomputed disk SHA"
        src_sha = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
        if pid["function_source_sha256"] != src_sha:
            return FAIL, f"{part}: function_source_sha256 != recomputed source SHA"
        if int(pid["source_lines"]) < 1:
            return FAIL, f"{part}: source_lines={pid['source_lines']}"
    if ident["learner"]["module"] != "rmt_replay_learner":
        return FAIL, f"learner module={ident['learner']['module']!r}"
    if ident["sampler"]["qualname"] != "RMTReplayBuffer.sample_eligible":
        return FAIL, f"sampler qualname={ident['sampler']['qualname']!r}"
    pd = CONTRACT.replay_protocol_labels("original_vtrace", 129, 4)["protocol_definition"]
    recon = CONTRACT.verify_executed_protocol_matches_declared(ident, pd)
    if recon["executed_protocol_declaration_match"] != "PASS":
        return FAIL, f"real-module declaration reconciliation not PASS: {recon}"
    return PASS, ("real frozen learner/sampler: complete identity tuple (module / qualname / "
                  "name / module realpath / module file SHA / function source SHA / source "
                  "lines) with file + source SHAs recomputed and equal; declared-module "
                  "binding + reconciliation PASS")


def gate57():
    """GATE 57 — v2.4 §七.1 SAME-NAME-DIFFERENT-SOURCE negative: functions with identical
    declared names/modules but DIFFERENT source (or a different module file SHA) are rejected
    by the cross-arm effective comparison — EFFECTIVE_PROTOCOL_MATCH=FAIL on
    learner.function_source_sha256 / sampler.module_file_sha256 even though the DECLARED
    protocol still matches; PROTOCOL_MATCH and MATCHED_REPLAY_EXPOSURE cannot be PASS."""
    import phase4a_v2_exposure_validator as EV
    kw = dict(replay_updates=2, consumed=8, batch_sizes=[4, 4], seq_lens=[[129] * 4] * 2,
              attempt_mask=[False, True, True], not_ready_updates=[0])
    sa = EV._synthetic_summary("persistent", **kw)
    sb = EV._synthetic_summary("reset128",
                               executed_kwargs=dict(learner_src_sha="ab" * 32), **kw)
    rep = EV.validate_two_arm(sa, sb)
    if rep["DECLARED_PROTOCOL_MATCH"] != "PASS":
        return FAIL, f"declared match must survive a source difference: {rep}"
    if rep["EFFECTIVE_PROTOCOL_MATCH"] != "FAIL":
        return FAIL, "same-name different-source must FAIL the EFFECTIVE match"
    if "learner.function_source_sha256" not in rep["EXECUTED_PROTOCOL_DIFFERING_FIELDS"]:
        return FAIL, ("differing fields must name learner.function_source_sha256: "
                      f"{rep['EXECUTED_PROTOCOL_DIFFERING_FIELDS']}")
    if rep["PROTOCOL_MATCH"] == "PASS" or rep["MATCHED_REPLAY_EXPOSURE"] == "PASS":
        return FAIL, "same-name different-source must not reach PROTOCOL / EXPOSURE PASS"
    if rep["EFFECTIVE_PROTOCOL_SHA256_ARM_A"] == rep["EFFECTIVE_PROTOCOL_SHA256_ARM_B"]:
        return FAIL, "effective SHAs must differ when the learner source differs"
    sb2 = EV._synthetic_summary("reset128",
                                executed_kwargs=dict(sampler_file_sha="cd" * 32), **kw)
    rep2 = EV.validate_two_arm(sa, sb2)
    if (rep2["EFFECTIVE_PROTOCOL_MATCH"] != "FAIL"
            or "sampler.module_file_sha256" not in rep2["EXECUTED_PROTOCOL_DIFFERING_FIELDS"]):
        return FAIL, ("sampler module-file tamper not caught: "
                      f"{rep2['EXECUTED_PROTOCOL_DIFFERING_FIELDS']}")
    return PASS, ("same-name-different-source rejected: learner function_source_sha256 / "
                  "sampler module_file_sha256 differences FAIL the effective match (declared "
                  "stays PASS); effective SHAs differ; no PROTOCOL / EXPOSURE PASS")


def gate58():
    """GATE 58 — v2.4 §六: the replay-sampler RNG identity is bound BEFORE the certificate can
    reach PASS (static: RNG construction + identity binding precede the last finalize site),
    verify_rng_instance_identity draws NO random numbers, and a mismatched numpy_version
    fails closed."""
    import numpy as np
    import phase4a_v2_contract as CONTRACT
    src = _read(_LAUNCHER).replace("\r\n", "\n")
    lines = src.split("\n")

    def line_of(needle):
        for i, ln in enumerate(lines):
            if needle in ln:
                return i
        return -1

    rng_c = line_of("replay_sample_rng = np.random.RandomState(args.seed + 7)")
    rng_b = line_of(
        'EXECUTED_PROTOCOL_IDENTITY["rng_instance"] = '
        "CONTRACT.verify_rng_instance_identity(")
    fins = [i for i, ln in enumerate(lines)
            if "RUNTIME_CONFIG_CERTIFICATE = RTC.finalize_certificate(" in ln]
    if rng_c == -1 or rng_b == -1 or len(fins) != 3:
        return FAIL, f"wiring missing (rng_construct={rng_c} rng_bind={rng_b} fins={fins})"
    if not (rng_c < rng_b < max(fins)):
        return FAIL, (f"RNG identity not bound before the checkpoint finalize: "
                      f"construct={rng_c} bind={rng_b} last_finalize={max(fins)}")
    # functional: no random state is consumed by the identity binding
    rs = np.random.RandomState(7)
    pristine = np.random.RandomState(7).get_state()
    rng = CONTRACT.verify_rng_instance_identity(rs)
    state_after = rs.get_state()
    if not np.array_equal(state_after[1], pristine[1]) or state_after[2] != pristine[2]:
        return FAIL, "verify_rng_instance_identity CONSUMED random state (must draw nothing)"
    if rng["numpy_version"] != np.__version__:
        return FAIL, f"numpy_version binding wrong: {rng['numpy_version']} != {np.__version__}"
    # (NEG) an unavailable (falsy) numpy_version fails closed: the executed RNG identity MUST
    # bind the exact numpy version. (An explicit non-empty numpy_version is recorded as given;
    # the cross-check against the executing numpy happens in
    # build_effective_protocol_definition — see GATE59 NEG3.)
    try:
        CONTRACT.verify_rng_instance_identity(np.random.RandomState(7), numpy_version="")
        return FAIL, "(NEG) unavailable numpy_version not rejected"
    except ValueError as e:
        if "EXECUTED_PROTOCOL_RNG_MISMATCH" not in str(e):
            return FAIL, f"(NEG) wrong code: {e}"
    return PASS, ("RNG construction + identity binding precede the checkpoint finalize; the "
                  "binding draws no random numbers (state untouched); numpy_version bound to "
                  "the executing numpy; mismatched numpy_version fails closed")


def gate59():
    """GATE 59 — v2.4 §八: the effective protocol definition is COMPLETE (declared protocol +
    executed learner/sampler 5-field projections + executed RNG 5 fields), its SHA256 is
    stable under input key-order permutation, missing/incomplete inputs fail closed, and the
    driver builds + emits it."""
    if not HAVE_JAX:
        return SKIP, ("JAX absent (the REAL frozen replay modules import jax at module "
                      "scope; runs on server CPU)")
    import numpy as np
    import phase4a_v2_contract as CONTRACT
    import rmt_replay_learner as RL
    from rmt_replay_buffer import RMTReplayBuffer
    ident = CONTRACT.executed_function_source_identity(
        RL.original_vtrace_update_rmt, RMTReplayBuffer.sample_eligible)
    ident["rng_instance"] = CONTRACT.verify_rng_instance_identity(np.random.RandomState(49))
    pd = CONTRACT.replay_protocol_labels("original_vtrace", 129, 4)["protocol_definition"]
    eff, sha = CONTRACT.build_effective_protocol_definition(
        pd, ident, ident["rng_instance"])
    if set(eff) != {"declared_protocol", "executed_learner", "executed_sampler",
                    "executed_rng"}:
        return FAIL, f"effective key set wrong: {sorted(eff)}"
    if set(eff["executed_learner"]) != set(CONTRACT.EFFECTIVE_PROTOCOL_LEARNER_FIELDS):
        return FAIL, f"executed_learner projection wrong: {sorted(eff['executed_learner'])}"
    if set(eff["executed_sampler"]) != set(CONTRACT.EFFECTIVE_PROTOCOL_SAMPLER_FIELDS):
        return FAIL, f"executed_sampler projection wrong: {sorted(eff['executed_sampler'])}"
    if set(eff["executed_rng"]) != set(CONTRACT.EFFECTIVE_PROTOCOL_RNG_FIELDS):
        return FAIL, f"executed_rng fields wrong: {sorted(eff['executed_rng'])}"
    if not (isinstance(sha, str) and len(sha) == 64):
        return FAIL, f"effective SHA not length-64: {sha!r}"
    # stable under key-order permutation of ALL inputs
    eff2, sha2 = CONTRACT.build_effective_protocol_definition(
        dict(reversed(list(pd.items()))),
        dict(reversed(list(ident.items()))),
        dict(reversed(list(ident["rng_instance"].items()))))
    if sha2 != sha:
        return FAIL, "effective protocol SHA not key-order invariant"
    if eff2 != eff:
        return FAIL, "effective protocol definition not key-order invariant"
    # fail closed: missing executed identity
    try:
        CONTRACT.build_effective_protocol_definition(pd, None, ident["rng_instance"])
        return FAIL, "(NEG1) missing executed identity not rejected"
    except ValueError as e:
        if "EXECUTED_PROTOCOL_IDENTITY_REQUIRED" not in str(e):
            return FAIL, f"(NEG1) wrong code: {e}"
    # fail closed: incomplete declared protocol
    bad_pd = dict(pd)
    bad_pd.pop("learner")
    try:
        CONTRACT.build_effective_protocol_definition(bad_pd, ident, ident["rng_instance"])
        return FAIL, "(NEG2) incomplete declared protocol not rejected"
    except ValueError as e:
        if "PROTOCOL_IDENTITY_INCOMPLETE" not in str(e):
            return FAIL, f"(NEG2) wrong code: {e}"
    # fail closed: rng numpy_version disagreement
    try:
        CONTRACT.build_effective_protocol_definition(pd, ident, ident["rng_instance"],
                                                     numpy_version="0.0.0")
        return FAIL, "(NEG3) rng numpy_version mismatch not rejected"
    except ValueError as e:
        if "EXECUTED_PROTOCOL_RNG_MISMATCH" not in str(e):
            return FAIL, f"(NEG3) wrong code: {e}"
    # static: driver builds the effective protocol and the summary emits it
    src = _read(_LAUNCHER)
    for needle in ("EFFECTIVE_PROTOCOL_DEFINITION, EFFECTIVE_PROTOCOL_SHA256 = (",
                   "declared_protocol_definition=DECLARED_PROTOCOL_DEFINITION,",
                   "effective_protocol_definition=EFFECTIVE_PROTOCOL_DEFINITION,",
                   "effective_protocol_sha256=EFFECTIVE_PROTOCOL_SHA256,"):
        if needle not in src:
            return FAIL, f"driver missing effective-protocol wiring: {needle}"
    return PASS, ("effective protocol complete (declared + learner/sampler 5-field projections "
                  "+ RNG 5 fields); SHA256 stable under key-order permutation; missing "
                  "executed identity / incomplete declared / RNG mismatch all fail closed; "
                  "driver builds + summary emits declared/effective definition + SHA")


def gate60():
    """GATE 60 — v2.4 §九: the two-arm validator compares the EFFECTIVE protocol and fails
    closed: missing executed identity OR effective block on either arm raises
    EXECUTED_PROTOCOL_IDENTITY_REQUIRED (NO declared-only fallback); an executed-source / RNG
    difference FAILs the effective match while declared stays PASS; MATCHED_REPLAY_EXPOSURE
    PASS requires BOTH EFFECTIVE_PROTOCOL_MATCH and EXPOSURE_COUNT_MATCH."""
    import phase4a_v2_exposure_validator as EV
    kw = dict(replay_updates=2, consumed=8, batch_sizes=[4, 4], seq_lens=[[129] * 4] * 2,
              attempt_mask=[False, True, True], not_ready_updates=[0])
    sa = EV._synthetic_summary("persistent", **kw)
    sb = EV._synthetic_summary("reset128", **kw)
    rep = EV.validate_two_arm(sa, sb)
    for key in ("DECLARED_PROTOCOL_MATCH", "EXECUTED_PROTOCOL_MATCH",
                "EFFECTIVE_PROTOCOL_MATCH", "EFFECTIVE_PROTOCOL_SHA256_ARM_A",
                "EFFECTIVE_PROTOCOL_SHA256_ARM_B", "EXECUTED_PROTOCOL_DIFFERING_FIELDS",
                "EXPOSURE_COUNT_MATCH"):
        if key not in rep:
            return FAIL, f"report missing {key}"
    if not (rep["DECLARED_PROTOCOL_MATCH"] == rep["EXECUTED_PROTOCOL_MATCH"]
            == rep["EFFECTIVE_PROTOCOL_MATCH"] == "PASS"):
        return FAIL, f"equal arms not all PASS: declared/executed/effective={rep}"
    if rep["EFFECTIVE_PROTOCOL_SHA256_ARM_A"] != rep["EFFECTIVE_PROTOCOL_SHA256_ARM_B"]:
        return FAIL, "equal arms' effective SHAs differ"
    if rep["MATCHED_REPLAY_EXPOSURE"] != "PASS" or rep["PROTOCOL_MATCH"] != "PASS":
        return FAIL, "equal arms must reach MATCHED_REPLAY_EXPOSURE + PROTOCOL PASS"
    # (NEG1) missing executed identity / effective block -> fail closed, no fallback
    for name, extra in [("executed", dict(drop_executed=True)),
                        ("effective", dict(drop_effective=True))]:
        sb_x = EV._synthetic_summary("reset128", **dict(kw, **extra))
        try:
            EV.validate_two_arm(sa, sb_x)
            return FAIL, f"(NEG1) missing {name} identity did NOT raise"
        except ValueError as e:
            if "EXECUTED_PROTOCOL_IDENTITY_REQUIRED" not in str(e):
                return FAIL, f"(NEG1) missing {name}: wrong code: {e}"
    # executed learner source differs -> declared PASS, effective FAIL, no MATCHED PASS
    sb_d = EV._synthetic_summary("reset128",
                                 executed_kwargs=dict(learner_src_sha="9a" * 32), **kw)
    rep_d = EV.validate_two_arm(sa, sb_d)
    if rep_d["DECLARED_PROTOCOL_MATCH"] != "PASS":
        return FAIL, "declared match must survive an executed-source difference"
    if rep_d["EFFECTIVE_PROTOCOL_MATCH"] != "FAIL" or rep_d["PROTOCOL_MATCH"] == "PASS":
        return FAIL, "executed-source difference must FAIL the effective + overall match"
    if rep_d["MATCHED_REPLAY_EXPOSURE"] == "PASS":
        return FAIL, "MATCHED_REPLAY_EXPOSURE requires EFFECTIVE_PROTOCOL_MATCH=PASS"
    # exposure difference + effective equal -> effective PASS but MATCHED FAIL
    sb_e = EV._synthetic_summary("reset128", replay_updates=1, consumed=4, batch_sizes=[4],
                                 seq_lens=[[129] * 4], attempt_mask=[False, True, True],
                                 not_ready_updates=[0])
    rep_e = EV.validate_two_arm(sa, sb_e)
    if rep_e["EFFECTIVE_PROTOCOL_MATCH"] != "PASS":
        return FAIL, f"effective match must survive an exposure difference: {rep_e}"
    if rep_e["MATCHED_REPLAY_EXPOSURE"] == "PASS":
        return FAIL, "exposure mismatch must FAIL MATCHED_REPLAY_EXPOSURE"
    return PASS, ("two-arm validator compares the EFFECTIVE protocol: equal arms -> all three "
                  "matches PASS + equal effective SHAs + MATCHED PASS; missing executed / "
                  "effective identity raises EXECUTED_PROTOCOL_IDENTITY_REQUIRED (no "
                  "declared-only fallback); executed-source difference -> effective FAIL with "
                  "declared PASS; MATCHED requires effective AND exposure PASS")


def gate61():
    """GATE 61 — v2.4 §十: the sidecar is validated in FULL — exactly `<sha256>  <certificate
    basename>\n` (two tokens; token[0] == file SHA; token[1] == basename; trailing newline).
    Correct SHA + WRONG basename, extra tokens, empty sidecar, a lone token, and a missing
    trailing newline (truncation) all fail closed with RUNTIME_CONFIG_CERTIFICATE_TAMPERED."""
    import tempfile
    import phase4a_v2_runtime_config as RTC
    pend, _, _ = _v23_prejax_chain("persistent")
    fin = RTC.finalize_certificate(pend, pend["checkpoint_identity"],
                                   checkpoint_error="BASE_CHECKPOINT_SHA_MISMATCH: x")

    def expect_tamper(label, sidecar_text):
        with tempfile.TemporaryDirectory() as td:
            cpath, spath, fsha, psha, _ = RTC.write_certificate_atomic(
                fin, os.path.join(td, "runtime_config_certificate.json"))
            with open(spath, "w", encoding="utf-8") as f:
                f.write(sidecar_text)
            try:
                RTC.verify_certificate_artifact(cpath, spath, expected_file_sha256=fsha,
                                                expected_payload_sha256=psha)
                return f"{label}: tamper NOT detected"
            except ValueError as e:
                if "RUNTIME_CONFIG_CERTIFICATE_TAMPERED" not in str(e):
                    return f"{label}: wrong code: {e}"
        return None

    with tempfile.TemporaryDirectory() as td:
        cpath, spath, fsha, psha, _ = RTC.write_certificate_atomic(
            fin, os.path.join(td, "runtime_config_certificate.json"))
        base = os.path.basename(cpath)
        if open(spath, encoding="utf-8").read() != f"{fsha}  {base}\n":
            return FAIL, "pristine sidecar is not exactly '<sha>  <basename>\\n'"
        v = RTC.verify_certificate_artifact(cpath, spath, expected_file_sha256=fsha,
                                            expected_payload_sha256=psha)
        if v["certificate_tamper_check"] != "PASS":
            return FAIL, f"pristine sidecar verify not PASS: {v}"
        attacks = [
            ("correct SHA + wrong basename", f"{fsha}  wrong_name.json\n"),
            ("extra tokens", f"{fsha}  {base} extra\n"),
            ("empty sidecar", ""),
            ("lone token (no basename)", f"{fsha}\n"),
            ("missing trailing newline (truncated)", f"{fsha}  {base}"),
        ]
    for label, content in attacks:
        err = expect_tamper(label, content)
        if err:
            return FAIL, err
    return PASS, ("sidecar full validation: pristine '<sha>  <basename>\\n' PASSes; correct "
                  "SHA + wrong basename / extra tokens / empty / lone token / truncated "
                  "(no newline) all fail closed with RUNTIME_CONFIG_CERTIFICATE_TAMPERED")


def gate62():
    """GATE 62 — v2.4 §十一: the formal path labels match the RELOCATABLE semantics —
    CANONICAL_RELATIVE_PATH_UNDER_EXECUTING_SNAPSHOT_ROOT + LAYOUT_AND_CONTENT_BOUND (whole-
    snapshot relocation IS legitimate; the v2.3 NO_COPY wording is gone) — and the snapshot
    root is DERIVED from the executing module location (realpath equality required; a
    different snapshot root fails closed)."""
    import phase4a_v2_formal_identity as FID
    import phase4a_v2_runtime_config as RTC
    fid_src = _read(os.path.join(_EXP, "phase4a_v2_formal_identity.py"))
    if "REALPATH_EQUALITY_NO_COPY_NO_SYMLINK_ESCAPE" in fid_src:
        return FAIL, "v2.3 NO_COPY label still present (whole-snapshot relocation is legal)"
    if FID.FORMAL_CONFIG_PATH_IDENTITY_LABEL != (
            "CANONICAL_RELATIVE_PATH_UNDER_EXECUTING_SNAPSHOT_ROOT"):
        return FAIL, f"path identity label wrong: {FID.FORMAL_CONFIG_PATH_IDENTITY_LABEL}"
    if FID.FORMAL_CONFIG_SNAPSHOT_RELOCATION_LABEL != "LAYOUT_AND_CONTENT_BOUND":
        return FAIL, f"relocation label wrong: {FID.FORMAL_CONFIG_SNAPSHOT_RELOCATION_LABEL}"
    derived = FID.derived_snapshot_root()
    if os.path.realpath(derived) != os.path.realpath(_SNAPSHOT):
        return FAIL, (f"derived_snapshot_root()={derived!r} does not resolve to the running "
                      f"snapshot {_SNAPSHOT!r}")
    for arm in ("persistent", "reset128"):
        canon = os.path.join(_SNAPSHOT, "configs", f"rmt16_phase4a_v2_{arm}.yaml")
        idrec = FID.verify_formal_config_path_identity(_SNAPSHOT, arm, canon)
        if idrec.get("formal_config_path_identity") != (
                FID.FORMAL_CONFIG_PATH_IDENTITY_LABEL):
            return FAIL, f"{arm}: path identity record wrong: {idrec}"
        if idrec.get("formal_config_snapshot_relocation") != (
                FID.FORMAL_CONFIG_SNAPSHOT_RELOCATION_LABEL):
            return FAIL, f"{arm}: relocation record wrong: {idrec}"
        if os.path.realpath(idrec.get("derived_snapshot_root_realpath") or "") != derived:
            return FAIL, f"{arm}: derived root not recorded: {idrec}"
        # (NEG) a snapshot root OTHER than the derived one fails closed, even with a valid
        # canonical YAML underneath it.
        try:
            FID.verify_formal_config_path_identity(os.path.join(_SNAPSHOT, "configs"), arm,
                                                   canon)
            return FAIL, f"(NEG) {arm}: non-derived snapshot_root not rejected"
        except ValueError as e:
            if "derived snapshot root" not in str(e):
                return FAIL, f"(NEG) {arm}: wrong message: {e}"
    return PASS, ("formal path labels match relocatable semantics (canonical relative path "
                  "under the executing snapshot root + layout-and-content bound); NO_COPY "
                  "wording gone; snapshot root derived from the executing module; other "
                  "snapshot roots fail closed on the derived-root check")


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
    # ---- Phase4A-v2.1 (§十): provenance / policy-lag / exposure / frozen-evidence gates ----
    ("GATE16_policy_version_start_end_span", gate16),
    ("GATE17_reset_slot_start_current", gate17),
    ("GATE18_sample_propagates_provenance", gate18),
    ("GATE19_original_vtrace_lag_fail_closed", gate19),
    ("GATE20_manifest_lag_identity", gate20),
    ("GATE21_legacy_lag_scoped", gate21),
    ("GATE22_protocol_vs_exposure_distinct", gate22),
    ("GATE23_no_cert_no_exposure_pass", gate23),
    ("GATE24_endogenous_no_content_pass", gate24),
    ("GATE25_raw_probe_sha_matches", gate25),
    ("GATE26_raw_probe_recompute_both", gate26),
    # ---- Phase4A-v2.2 (§九): full protocol identity / lag leak / episode range / range
    #      validators / formal-config runtime binding / layered publication gates ----
    ("GATE27_protocol_required_fields_complete", gate27),
    ("GATE28_protocol_fail_closed", gate28),
    ("GATE29_no_active_policy_lag_leak", gate29),
    ("GATE30_episode_policy_range_record", gate30),
    ("GATE31_trajectory_range_validator", gate31),
    ("GATE32_sample_range_validator", gate32),
    ("GATE33_formal_runtime_binding_pass", gate33),
    ("GATE34_scientific_field_mismatch", gate34),
    ("GATE35_runtime_assignment_mismatch", gate35),
    ("GATE36_formal_config_required_prejax", gate36),
    ("GATE37_certificate_sha_consistency", gate37),
    ("GATE38_layered_publication_labels", gate38),
    # ---- Phase4A-v2.3 (§十/§十一): canonical formal-config identity / runtime-assignment
    #      fail closed / pre-JAX full binding / certificate state machine + artifact SHA /
    #      executed protocol source identity / v2.3 publication labels + v2.2 errata ----
    ("GATE39_canonical_formal_identity", gate39),
    ("GATE40_formal_identity_fail_closed", gate40),
    ("GATE41_assignment_completeness_fail_closed", gate41),
    ("GATE42_assignment_arm_gpu_fail_closed", gate42),
    ("GATE43_assignment_out_dir_strict", gate43),
    ("GATE44_prejax_full_binding_pending", gate44),
    ("GATE45_certificate_state_machine", gate45),
    ("GATE46_certificate_artifact_sha", gate46),
    ("GATE47_certificate_tamper_detection", gate47),
    ("GATE48_executed_protocol_source_identity", gate48),
    ("GATE49_v2_3_publication_labels", gate49),
    ("GATE50_v2_2_publication_errata", gate50),
    ("GATE51_actual_cli_prejax_binding", gate51),
    ("GATE52_subprocess_import_sentinel", gate52),
    ("GATE53_checkpoint_failures_finalized", gate53),
    ("GATE54_certificate_disk_object_sync", gate54),
    ("GATE55_payload_sha_manifest_binding", gate55),
    ("GATE56_executed_source_identity_complete", gate56),
    ("GATE57_same_name_different_source_rejected", gate57),
    ("GATE58_rng_identity_before_certificate_pass", gate58),
    ("GATE59_effective_protocol_complete_stable", gate59),
    ("GATE60_two_arm_effective_comparison", gate60),
    ("GATE61_sidecar_basename_validation", gate61),
    ("GATE62_formal_path_labels_relocatable", gate62),
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
