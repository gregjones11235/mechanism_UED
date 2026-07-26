#!/usr/bin/env python3
"""Dump the d052_eval + p2 sections of the prior unified_eval state (read-only)."""
import json

STATE = "/home/oseasy/experiments/d052_unified_eval_20260722/orchestration_state.json"
d = json.load(open(STATE))
for key in ("d052_eval", "p2"):
    print("=" * 70)
    print("SECTION:", key)
    print("=" * 70)
    print(json.dumps(d.get(key, {}), indent=2, ensure_ascii=False))
    print()
