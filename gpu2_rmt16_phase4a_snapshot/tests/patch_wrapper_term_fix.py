import sys

PATH = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src/dicode/wrappers_cl.py"

# wrappers_cl.py uses TAB indentation + CRLF line endings (uniform). newline='' preserves CRLF.
T = "\t"
N = "\r\n"


def block(lines):
    """Join pre-indented lines with CRLF; each element is a full line WITHOUT terminator."""
    return N.join(lines) + N


# ---------- F1: validate the time-limit threshold source in __init__ (probe logging path only) ----------
# Anchor = the two lines currently emitted by W1 (unique in the file). Insert the validation block
# BETWEEN `self.probe_term = bool(probe_term)` (super().__init__ already ran -> self._env exists)
# and `self.num_envs = num_envs`.
F1_OLD = block([
    T + T + "self.probe_term = bool(probe_term)",
    T + T + "self.num_envs = num_envs",
])
F1_NEW = block([
    T + T + "self.probe_term = bool(probe_term)",
    T + T + "# RMT16 Phase4A (CC2 addendum): validate the time-limit threshold source for _term_done_steps.",
    T + T + "# The env's REAL termination uses each task's self.params.max_timesteps (== the ctor EnvParams,",
    T + T + "# == 4096 here), NOT env.default_params.max_timesteps (craftax default 100000). When probe_term",
    T + T + "# is True we read EVERY task's params.max_timesteps, require >=1 task and FULL agreement, and",
    T + T + "# otherwise RAISE (NO silent fallback, NO hardcoded 4096). Pure host-side Python at __init__, a",
    T + T + "# compile-time constant; probe_term=False skips this entirely -> info/training remain bit-exact.",
    T + T + "if self.probe_term:",
    T + T + T + "_tasks = getattr(self._env, \"tasks\", None)",
    T + T + T + "if not _tasks:",
    T + T + T + T + "raise RuntimeError(",
    T + T + T + T + "    \"probe_term: base env exposes no .tasks; cannot validate time-limit threshold\")",
    T + T + T + "_thresholds = []",
    T + T + T + "for _t in _tasks:",
    T + T + T + T + "_tp = getattr(_t, \"params\", None)",
    T + T + T + T + "if _tp is None or not hasattr(_tp, \"max_timesteps\"):",
    T + T + T + T + T + "raise RuntimeError(",
    T + T + T + T + T + "    \"probe_term: a task lacks params.max_timesteps; cannot validate time-limit threshold\")",
    T + T + T + T + "_thresholds.append(int(_tp.max_timesteps))",
    T + T + T + "if len(set(_thresholds)) != 1:",
    T + T + T + T + "raise RuntimeError(",
    T + T + T + T + "    \"probe_term: inconsistent task max_timesteps across tasks: %r\" % (_thresholds,))",
    T + T + T + "self.probe_max_timesteps = _thresholds[0]",
    T + T + "self.num_envs = num_envs",
])

# ---------- F2: _term_done_steps uses the VALIDATED self.probe_max_timesteps (requirement 6) ----------
F2_OLD = block([
    T + T + T + 'info["_term_done_steps"] = env_state.timestep >= int(getattr(params, "max_timesteps", 4096))',
])
F2_NEW = block([
    T + T + T + 'info["_term_done_steps"] = env_state.timestep >= self.probe_max_timesteps',
])

REPLS = [("F1", F1_OLD, F1_NEW, 1), ("F2", F2_OLD, F2_NEW, 1)]
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
print("WRAPPER_FIX_PATCHED (%d replacements)" % len(REPLS))
