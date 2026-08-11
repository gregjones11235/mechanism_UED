import sys

PATH = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src/dicode/wrappers_cl.py"

# wrappers_cl.py uses TAB indentation + CRLF line endings (uniform). newline='' preserves CRLF.
T = "\t"
N = "\r\n"


def block(lines):
    """Join pre-indented lines with CRLF; each element is a full line WITHOUT terminator."""
    return N.join(lines) + N


# ---------- W1: add probe_term constructor kwarg (default False) to Distributed wrapper ----------
W1_OLD = block([
    T + T + "task_embeddings=None,",
    T + "):",
    T + T + "super().__init__(env)",
    T + T + "self.num_envs = num_envs",
])
W1_NEW = block([
    T + T + "task_embeddings=None,",
    T + T + "probe_term: bool = False,",
    T + "):",
    T + T + "super().__init__(env)",
    T + T + "# RMT16 Phase4A (CC2 directive 3): when False, step() emits an info dict IDENTICAL",
    T + T + "# to the original wrapper (bit-exact). When True, additive _term_* logging keys only.",
    T + T + "self.probe_term = bool(probe_term)",
    T + T + "self.num_envs = num_envs",
])

# ---------- W2: gated additive _term_* info keys in Distributed.step (logging path only) ----------
W2_OLD = block([
    T + T + 'info["returned_episode_lengths"] = new_episode_length * real_done',
    "",
    T + T + "# --- NEW, CORRECT, AND EFFICIENT RESET LOGIC ---",
])
W2_NEW = block([
    T + T + 'info["returned_episode_lengths"] = new_episode_length * real_done',
    T + T + "# ---- RMT16 Phase4A READ-ONLY termination-reason logging (CC2 directive 3) ----",
    T + T + "# GATED by self.probe_term (default False -> info dict IDENTICAL to original / bit-exact).",
    T + T + "# Additive _term_* arrays from the TERMINAL (post-step, pre-reset) env_state. Fixed shape",
    T + T + "# [num_envs], fixed dtype, pure JAX arrays, NO strings in the JIT path. done_reason strings",
    T + T + "# are mapped host-side in the collector. This does NOT alter reset/step/done/reward/obs or",
    T + T + "# the optimistic-reset logic below (this wrapper only auto-resets ALREADY-done envs; it never",
    T + T + "# truncates a live episode, so optimistic_reset/wrapper_reset are never a done CAUSE here).",
    T + T + "if self.probe_term:",
    T + T + T + 'info["_term_player_health"] = env_state.player_health',
    T + T + T + 'info["_term_player_level"] = env_state.player_level',
    T + T + T + 'info["_term_timestep"] = env_state.timestep',
    T + T + T + 'info["_term_is_dead"] = env_state.player_health <= 0',
    T + T + T + 'info["_term_done_steps"] = env_state.timestep >= int(getattr(params, "max_timesteps", 4096))',
    "",
    T + T + "# --- NEW, CORRECT, AND EFFICIENT RESET LOGIC ---",
])

REPLS = [("W1", W1_OLD, W1_NEW, 1), ("W2", W2_OLD, W2_NEW, 1)]
with open(PATH, "r", newline="") as f:
    src = f.read()
for tag, old, new, n in REPLS:
    c = src.count(old)
    if c != n:
        print("ABORT %s expected %d got %d" % (tag, n, c))
        sys.exit(1)
    src = src.replace(old, new)
with open(PATH, "w", newline="") as f:
    f.write(src)
print("WRAPPER_PATCHED (%d replacements)" % len(REPLS))
