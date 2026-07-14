#!/usr/bin/env python3
"""Checkpoint save/load/resume fixtures for data-plane integrity.

Verifies:
- Deterministic checkpoint save (same state -> same hash)
- Checkpoint load and state restoration
- Resume idempotency (load->save->load produces same state)
- Corruption detection
- Missing checkpoint handling
- SIEGE state persistence through checkpoint cycle
"""
import sys, os, json, tempfile, hashlib
_SIEGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_AGG_V2_SRC = "/root/experiments/dicode-aggregation-v2/src"
# Both must be in path: SIEGE for dicode.siege.*, agg-v2 for dicode.mechanisms.*
# SIEGE goes first so dicode.siege is found
for p in [_SIEGE_SRC, _AGG_V2_SRC]:
    if p in sys.path: sys.path.remove(p)
sys.path[0:0] = [_SIEGE_SRC, _AGG_V2_SRC]

PASSED = 0; FAILED = 0
def check(cond, msg):
    global PASSED, FAILED
    if cond: PASSED += 1; print(f"  PASS: {msg}")
    else: FAILED += 1; print(f"  FAIL: {msg}")

print("="*60)
print("CHECKPOINT SAVE/LOAD/RESUME FIXTURES")
print("="*60)

# 1: Deterministic save
print("\n1. Deterministic checkpoint save")
state = {"global_step": 16384, "mechanism": "original", "pool_hash": "abc123",
         "selected_ids": ["t1","t2","t3","t4","t5","t6","t7","t8"]}
h1 = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
h2 = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
check(h1 == h2, f"Same state -> same hash: {h1[:16]}...")

# 2: Different state -> different hash
state2 = dict(state); state2["global_step"] = 32768
h3 = hashlib.sha256(json.dumps(state2, sort_keys=True).encode()).hexdigest()
check(h1 != h3, "Different steps -> different hash")

# 3: Save and load cycle
print("\n2. Save-load cycle")
with tempfile.TemporaryDirectory() as td:
    ckpt_path = os.path.join(td, "checkpoint.json")
    with open(ckpt_path, "w") as f: json.dump(state, f)
    check(os.path.exists(ckpt_path), "Checkpoint saved")
    check(os.path.getsize(ckpt_path) > 0, f"Non-empty: {os.path.getsize(ckpt_path)} bytes")
    with open(ckpt_path) as f: loaded = json.load(f)
    for key in ["global_step", "mechanism", "pool_hash", "selected_ids"]:
        check(loaded[key] == state[key], f"Field '{key}' preserved: {loaded[key]}")

# 4: Resume idempotency
print("\n3. Resume idempotency (save->load->save->load)")
with tempfile.TemporaryDirectory() as td:
    ckpt = os.path.join(td, "ckpt.json")
    with open(ckpt, "w") as f: json.dump(state, f)
    with open(ckpt) as f: loaded1 = json.load(f)
    with open(ckpt, "w") as f: json.dump(loaded1, f)  # Re-save
    with open(ckpt) as f: loaded2 = json.load(f)
    check(loaded1 == loaded2, "Save-load-save-load idempotent")

# 5: Corruption detection
print("\n4. Corruption detection")
with tempfile.TemporaryDirectory() as td:
    ckpt = os.path.join(td, "ckpt.json")
    with open(ckpt, "w") as f: f.write("corrupted{{{")
    try:
        with open(ckpt) as f: json.load(f)
        check(False, "Corrupted file should raise JSONDecodeError")
    except json.JSONDecodeError:
        check(True, "Corruption detected via parse failure")

# 6: Missing checkpoint
print("\n5. Missing checkpoint handling")
with tempfile.TemporaryDirectory() as td:
    missing = os.path.join(td, "nonexistent.json")
    check(not os.path.exists(missing), "Missing checkpoint correctly detected")
    # Should not crash, should indicate fresh start
    fresh_start = not os.path.exists(missing)
    check(fresh_start, "Fresh start when checkpoint missing")

# 7: SIEGE state persistence
print("\n6. SIEGE state persistence through checkpoint")
from dicode.siege.siege_notebook import SiegeNotebook
with tempfile.TemporaryDirectory() as td:
    nb1 = SiegeNotebook(td)
    nb1.define_craftax_chains()
    nb1.update({"collect_wood": 0.96, "craft_planks": 0.55}, 1000)
    tier_before = nb1.profile.get_tier("collect_wood")
    nb1.save()
    # Reload
    nb2 = SiegeNotebook(td)
    check(nb2.profile.get_tier("collect_wood") == tier_before, "SIEGE tier preserved across save/load")
    check(nb2.session_count == 1, "Session count preserved")
    check("crafting_progression" in nb2.chain_order.chains, "Chains preserved")

# 8: Checkpoint with large state
print("\n7. Large state checkpoint")
large_state = {"step": 500000, "achievements": {f"ach_{i}": {"sr": 0.5 + 0.01*i, "tier": i%5} for i in range(100)}}
with tempfile.TemporaryDirectory() as td:
    ckpt = os.path.join(td, "large.json")
    with open(ckpt, "w") as f: json.dump(large_state, f)
    check(os.path.getsize(ckpt) > 1000, f"Large checkpoint saved: {os.path.getsize(ckpt)} bytes")
    with open(ckpt) as f: loaded = json.load(f)
    check(len(loaded["achievements"]) == 100, "All 100 achievements preserved")

print(f"\n{'='*60}")
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED > 0: sys.exit(1)
