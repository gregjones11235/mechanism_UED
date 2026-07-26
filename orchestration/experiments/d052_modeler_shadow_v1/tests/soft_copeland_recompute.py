"""Faithful re-implementation of the D052 enhanced-launcher Soft Copeland wrapper.

Reuses the ORIGINAL dicode.mechanisms.aggregation._aggregate_soft_copeland (not
rewritten). Mirrors launch_d052_pure_dynamic_enhanced.py::aggregate_soft_copeland
exactly: role-field mapping, signal dict, weights, argsort top-8.
"""
import sys, json, hashlib
import numpy as np

SRC = "/home/oseasy/experiments/mechanism_UED_continuation_20260715/workers/gpu0_original/gpu0_training_mechanisms/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from dicode.mechanisms.aggregation import _aggregate_soft_copeland  # ORIGINAL

WEIGHTS = {"w_progression": 0.34, "w_retention": 0.33, "w_novelty": 0.33,
           "w_critic": 0.01, "w_monopoly": 0.01}

def sha16(x):
    d = x.encode() if isinstance(x, str) else x
    return hashlib.sha256(d).hexdigest()[:16]

def build_signals(cands, jents):
    n = len(cands)
    prog = np.full(n, 0.5); nov = np.full(n, 0.5); crit = np.zeros(n)
    by_role = {r: {} for r in ("tutor", "critic", "explorer")}
    for je in jents:
        tid = je["task_id"]; r = je["role"]; sc = je["judgment"].get("scores", {})
        by_role[r][tid] = sc
    for i, spc in enumerate(cands):
        cid = spc["task_id"]
        prog[i] = float(by_role["tutor"].get(cid, {}).get("progression_score", 0.5))
        crit[i] = float(by_role["critic"].get(cid, {}).get("critic_penalty", 0.0))
        nov[i] = float(by_role["explorer"].get(cid, {}).get("novelty_score", 0.5))
    ret = 1.0 - crit
    sig = {"progression": prog, "retention": ret, "novelty": nov,
           "critic_penalty": crit, "monopoly_penalty": np.zeros(n),
           "source_ids": np.array(["d052"] * n), "skill_counts": np.ones(n)}
    return sig, {"progression": prog, "retention": ret, "novelty": nov,
                 "critic_penalty": crit}

def recompute(cands, jents, k=8):
    """Returns (selected_ids, scores_list, signals_dict)."""
    sig, raw = build_signals(cands, jents)
    scores = _aggregate_soft_copeland(sig, WEIGHTS, 1.0)
    sel_idx = np.argsort(-scores)[:k]
    sel = [cands[i]["task_id"] for i in sel_idx]
    return sel, scores.tolist(), raw

def selection_hash(selected_ids):
    return sha16(json.dumps(sorted(selected_ids)))

def pool_hash(cands):
    spec = sorted([{"id": c["task_id"], "tp": c.get("task_params", {}),
                    "achs": sorted(c.get("target_achievements", [])),
                    "prov": c.get("_prov", {})} for c in cands], key=lambda x: x["id"])
    return sha16(json.dumps(spec, sort_keys=True))
