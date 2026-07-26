import sys

PATH = "train_rmt16_p2replay.py"   # run from src dir
N = "\n"


def block(lines):
    return N.join(lines) + N


# ---- D10: wire probe_term=PROBE into the wrapper construction ----
# Without this the wrapper always runs probe_term=False and never emits _term_* keys, so the
# collector termination-reason path is dead in real runs and the A/B gate would pass trivially.
# PROBE is defined earlier (D2). When PROBE is False this is bit-exact the original (default False).
D10_OLD = block([
    "env = DistributedMultiTaskOptimisticLogWrapper(",
    "    base_env, jax.random.PRNGKey(0), cfg.num_envs, 1, cfg.optimistic_reset_ratio,",
    "    jnp.array([1.0]), table)",
])
D10_NEW = block([
    "env = DistributedMultiTaskOptimisticLogWrapper(",
    "    base_env, jax.random.PRNGKey(0), cfg.num_envs, 1, cfg.optimistic_reset_ratio,",
    "    # RMT16 Phase4A (CC2): wire the probe so probe_term=True emits the additive _term_* keys.",
    "    # PROBE=False (frozen runs / A arm) -> probe_term=False -> info dict bit-exact original.",
    "    jnp.array([1.0]), table, probe_term=PROBE)",
])

REPLS = [("D10", D10_OLD, D10_NEW, 1)]
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
print("DRIVER_WIRE_PATCHED (%d replacements)" % len(REPLS))
