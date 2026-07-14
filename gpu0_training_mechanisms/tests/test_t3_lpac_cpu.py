#!/usr/bin/env python3
"""T3 LPAC real module tests — adaptation logic and controller interface."""
import sys, os
# IMPORTANT: our worktree src must come LAST in insert(0) to take priority
# over aggregation-v2 which has dicode/training.py (file, not package).
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dicode.training.lpac import LPACWrapper

PASSED = 0; FAILED = 0
def check(cond, msg):
    global PASSED, FAILED
    if cond: PASSED += 1; print(f"  PASS: {msg}")
    else: FAILED += 1; print(f"  FAIL: {msg}")

print("=" * 60)
print("T3 LPAC REAL MODULE TESTS")
print("=" * 60)

# 1. Disabled identity
print("\n1. Disabled: fixed settings, no adaptation")
cfg_off = type('C',(),{'enable_lpac':False})()
lpac_off = LPACWrapper(cfg_off)
check(not lpac_off.enabled, "Disabled flag respected")
e, t = lpac_off.update(0.0, 0.0, 0.01)
check(e == 0.01, "Entropy unchanged when disabled")
check(t == 1.0, "Temperature unchanged when disabled")
check(not lpac_off.has_nonzero_adaptation(), "No adaptation when disabled")

# 2. Enabled: configurable
print("\n2. Enabled: configurable parameters")
cfg_on = type('C',(),{'enable_lpac':True,'lpac_entropy_base':0.01,'lpac_entropy_max':0.05,
    'lpac_temperature_base':1.0,'lpac_temperature_min':0.5,'lpac_stagnation_window':5,
    'lpac_forgetting_threshold':0.05,'lpac_uncertainty_weight':0.1})()
lpac_on = LPACWrapper(cfg_on)
check(lpac_on.enabled, "Enabled flag active")
check(lpac_on.entropy_base == 0.01, "Entropy base configurable")

# 3. Adaptation: stagnation increases entropy
print("\n3. Adaptation: stagnation → increased entropy")
e1, t1 = lpac_on.update(0.0, 0.0, 0.01)
check(abs(e1 - 0.01) < 1e-6, f"First step: entropy={e1:.6f} (base)")
# Simulate stagnation (5 steps of flat progress)
for _ in range(4):
    lpac_on.update(0.0, 0.0, 0.01)
e5, t5 = lpac_on.update(0.0, 0.0, 0.01)
check(e5 > e1, f"After stagnation: entropy {e5:.4f} > {e1:.4f}")
check(t5 < t1, f"After stagnation: temperature {t5:.4f} < {t1:.4f} (more greedy)")

# 4. Adaptation: forgetting increases entropy
print("\n4. Adaptation: forgetting → increased entropy")
lpac_on.reset_history()
e_base, _ = lpac_on.update(0.0, 0.0, 0.01)
e_forget, _ = lpac_on.update(0.0, 0.03, 0.01)  # 3% forgetting
check(e_forget > e_base, f"Forgetting boost: {e_forget:.4f} > {e_base:.4f}")

# 5. Nonzero adaptation detected
print("\n5. Nonzero adaptation detected")
lpac_on.reset_history()
lpac_on.reset_history()
e_test_a, _ = lpac_on.update(0.0, 0.1, 0.01)  # extreme forgetting
check(abs(e_test_a - lpac_on.entropy_base) > 1e-6, f"LPAC nonzero adaptation: {e_test_a:.4f} != {lpac_on.entropy_base:.4f}")

# 6. Bounded outputs
print("\n6. Outputs bounded within configured ranges")
lpac_on.reset_history()
for _ in range(20):
    e, t = lpac_on.update(0.0, 0.1, 0.01)  # extreme forgetting
check(e <= lpac_on.entropy_max, f"Entropy {e:.4f} <= max {lpac_on.entropy_max}")
check(t >= lpac_on.temp_min, f"Temperature {t:.4f} >= min {lpac_on.temp_min}")
check(e >= lpac_on.entropy_base, f"Entropy {e:.4f} >= base {lpac_on.entropy_base}")

# 7. Reset clears history
print("\n7. Reset clears adaptation history")
lpac_on.reset_history()
e_after_reset, t_after_reset = lpac_on.update(0.0, 0.0, 0.01)
check(abs(e_after_reset - lpac_on.entropy_base) < 1e-6, f"Reset restores base entropy: {e_after_reset:.6f}")
check(t_after_reset == lpac_on.temp_base, "Reset restores base temperature")

# 8. No Tier label input
print("\n8. No Tier labels used as controller inputs")
# LPAC consumes normalized signals (progress, forgetting, entropy)
# It does NOT consume tier labels like 'Tier 3' or 'Tier 4'
import inspect
src = inspect.getsource(lpac_on.update)
check("tier" not in src.lower(), "No tier label in update logic")
check("Tier" not in src, "No Tier constant referenced")
# Controller feedback worlds are separate from reporting worlds (design guarantee)

print(f"\n{'='*60}")
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED > 0: sys.exit(1)
