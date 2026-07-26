import json, os, pickle, sys

ROOT = "/home/oseasy/experiments/rmt16_replay_phase4a"
FROZEN_STEP0_SHA = "2f8cd875993ae10385dbb5dae530a557a0eb1008541b98de416cc7ae7ba2d93b"
A_OUT = os.path.join(ROOT, "runs/RMT16-GATE6-A")
B_OUT = os.path.join(ROOT, "runs/RMT16-GATE6-B")
ARM = "RMT16-Persistent-PPO"          # replay off -> -PPO suffix

# Fields that MUST be bit-exact identical between A (probe off) and B (probe on).
HASH_FIELDS = ["actions_hash", "rewards_hash", "dones_hash", "ard_hash",
               "params_sha", "ppo_opt_sha", "rmt_state_sha",
               "memories_sha", "mem_mask_sha", "mem_idx_sha"]
METRIC_FIELDS = ["ppo_actor", "ppo_entropy", "ppo_value"]
INT_FIELDS = ["global_step", "online_ppo_update_count"]


def load_equiv(out_dir):
    p = os.path.join(out_dir, "out", f"{ARM}_equiv.jsonl")
    rows = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows[int(d["update"])] = d
    return p, rows


def load_step0_sha(out_dir):
    p = os.path.join(out_dir, "ckpt/0/full_state.pkl")
    with open(p, "rb") as f:
        d = pickle.load(f)
    return p, d["manifest"]["params_sha256"] if "manifest" in d else d["params_sha256"]


report = {}
ok = True

# ---------------- Gate-5: step0 params SHA == frozen, and A == B ----------------
a_pkl, a_sha = load_step0_sha(A_OUT)
b_pkl, b_sha = load_step0_sha(B_OUT)
gate5 = (a_sha == FROZEN_STEP0_SHA and b_sha == FROZEN_STEP0_SHA and a_sha == b_sha)
report["gate5_step0_sha"] = dict(
    frozen=FROZEN_STEP0_SHA, A=a_sha, B=b_sha,
    A_matches_frozen=bool(a_sha == FROZEN_STEP0_SHA),
    B_matches_frozen=bool(b_sha == FROZEN_STEP0_SHA),
    A_equals_B=bool(a_sha == b_sha), PASS=bool(gate5))
print("[GATE5] A step0 sha=%s" % a_sha)
print("[GATE5] B step0 sha=%s" % b_sha)
print("[GATE5] frozen    =%s" % FROZEN_STEP0_SHA)
print("GATE5_STEP0_SHA=%s" % ("PASS" if gate5 else "FAIL"))
ok = ok and gate5

# ---------------- Gate-6: per-update exact equivalence of A vs B ----------------
a_path, a_rows = load_equiv(A_OUT)
b_path, b_rows = load_equiv(B_OUT)
print("[GATE6] A equiv file=%s n_updates=%d" % (a_path, len(a_rows)))
print("[GATE6] B equiv file=%s n_updates=%d" % (b_path, len(b_rows)))

per_update = {}
all_updates = sorted(set(a_rows) | set(b_rows))
for u in all_updates:
    a = a_rows.get(u); b = b_rows.get(u)
    if a is None or b is None:
        per_update[u] = dict(present_in_both=False)
        ok = False
        continue
    mism = {}
    for fld in HASH_FIELDS + INT_FIELDS:
        if a.get(fld) != b.get(fld):
            mism[fld] = {"A": a.get(fld), "B": b.get(fld)}
    for fld in METRIC_FIELDS:
        # bit-exact float compare (json roundtrip preserves IEEE754 doubles)
        av = float(a.get(fld, 0.0)); bv = float(b.get(fld, 0.0))
        if av != bv:
            mism[fld] = {"A": av, "B": bv, "absdiff": abs(av - bv)}
    per_update[u] = dict(present_in_both=True, n_mismatches=len(mism), mismatches=mism,
                         actions_hash_A=a.get("actions_hash"),
                         params_sha_A=a.get("params_sha"))
    print("[GATE6 u%d] mismatches=%d  actions_hash=%s  params_sha=%s"
          % (u, len(mism), str(a.get("actions_hash"))[:16], str(a.get("params_sha"))[:16]))
    if mism:
        ok = False

n_updates_ok = (len(a_rows) == 2 and len(b_rows) == 2)
report["gate6_equiv"] = dict(
    A_n_updates=len(a_rows), B_n_updates=len(b_rows),
    expected_updates=2, updates_compared=all_updates,
    per_update=per_update,
    all_fields_bit_exact=bool(ok and gate5),
    PASS=bool(ok and gate5 and n_updates_ok))

print("PROBE_INSTRUMENTATION_TRAINING_EQUIVALENCE=%s"
      % ("PASS" if (ok and gate5 and n_updates_ok) else "FAIL"))

with open(os.path.join(ROOT, "reports/gate6_equiv_compare.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print("[saved] reports/gate6_equiv_compare.json")
sys.exit(0 if (ok and gate5 and n_updates_ok) else 1)
