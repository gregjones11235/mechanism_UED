"""Deterministic OFFLINE replay of the historical B/C counterfactual (spec §3).

Recomputes the legacy Soft-Copeland selected-8 for arms B and C from the REAL
bundle judgments, using the ORIGINAL selector source whose sha256 the bundle's
selector_config.json pins (27492e8a...). Faithful to the bundle's own wrapper
(tests/soft_copeland_recompute.py):

    candidate order : sorted(task_id)
    progression     : tutor.progression_score        (default 0.5)
    critic_penalty  : critic.critic_penalty          (default 0.0)
    novelty         : explorer.novelty_score         (default 0.5)
    retention       : 1 - critic_penalty
    monopoly        : 0 ; source_ids "d052" ; skill_counts 1
    scores          : _aggregate_soft_copeland(sig, WEIGHTS, 1.0)
    selected        : task_ids[np.argsort(-scores)[:8]]
    selection_hash  : sha256(json.dumps(sorted(selected)))[:16]

NO LLM, NO RNG, NO mutation of judgments/selector/tie-break/normalization/order.
run_replay() is pure (no file writes); it returns the checks + recomputed values.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Optional

from d052.reconciliation.real_bundle import (
    BUNDLE_REL, REPLAY_INPUTS_REL, REPO_ROOT, load_bundle_json, load_judgments,
)

WEIGHTS = {"w_progression": 0.34, "w_retention": 0.33, "w_novelty": 0.33,
           "w_critic": 0.01, "w_monopoly": 0.01}


def _load_aggregation(path: Path):
    """Load the original selector module standalone (no package side effects)."""
    spec = importlib.util.spec_from_file_location("d052_replay_aggregation_original",
                                                  str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _build_signals(records):
    import numpy as np
    ids = sorted({r["task_id"] for r in records})      # candidate order invariant
    n = len(ids)
    idx = {t: i for i, t in enumerate(ids)}
    by_role = {role: {} for role in ("tutor", "critic", "explorer")}
    for r in records:
        by_role[r["role"]][r["task_id"]] = r["raw_scores"]
    prog = np.full(n, 0.5)
    nov = np.full(n, 0.5)
    crit = np.zeros(n)
    for t, i in idx.items():
        prog[i] = float(by_role["tutor"].get(t, {}).get("progression_score", 0.5))
        crit[i] = float(by_role["critic"].get(t, {}).get("critic_penalty", 0.0))
        nov[i] = float(by_role["explorer"].get(t, {}).get("novelty_score", 0.5))
    sig = {"progression": prog, "retention": 1.0 - crit, "novelty": nov,
           "critic_penalty": crit, "monopoly_penalty": np.zeros(n),
           "source_ids": np.array(["d052"] * n), "skill_counts": np.ones(n)}
    return ids, sig


def _run_arm(agg, arm: str, bdir: Optional[Path]) -> dict:
    import numpy as np
    records = load_judgments(arm, bdir)
    ids, sig = _build_signals(records)
    scores = agg._aggregate_soft_copeland(sig, WEIGHTS, 1.0)
    sel_idx = np.argsort(-scores)[:8]
    sel = [ids[i] for i in sel_idx]
    return {"n_candidates": len(ids),
            "selected8_score_order": sel,
            "selected8_sorted": sorted(sel),
            "selection_hash": _sha16(json.dumps(sorted(sel))),
            "scores": [float(x) for x in scores]}


def _pool_hash(pool_path: Path) -> Optional[str]:
    if not pool_path.exists():
        return None
    pool = json.loads(pool_path.read_text(encoding="utf-8"))["candidates"]
    spec = sorted([{"id": c["task_id"], "tp": c.get("task_params", {}),
                    "achs": sorted(c.get("target_achievements", [])),
                    "prov": c.get("_prov", {})} for c in pool], key=lambda x: x["id"])
    return _sha16(json.dumps(spec, sort_keys=True))


def run_replay(bdir: Optional[os.PathLike] = None,
               replay_inputs: Optional[os.PathLike] = None) -> dict:
    """Run the full offline replay twice (determinism) and check every anchor.

    Returns a dict with recomputed values, per-anchor checks, and ALL_ANCHORS_PASS.
    """
    bdir = Path(bdir) if bdir else REPO_ROOT / BUNDLE_REL
    rin = Path(replay_inputs) if replay_inputs else REPO_ROOT / REPLAY_INPUTS_REL

    agg_path = rin / "aggregation_original.py"
    agg_sha = hashlib.sha256(agg_path.read_bytes()).hexdigest()
    agg = _load_aggregation(agg_path)

    eb = load_bundle_json("expected_behavior.json", bdir)
    sc = load_bundle_json("selector_config.json", bdir)

    B1, C1 = _run_arm(agg, "B", bdir), _run_arm(agg, "C", bdir)
    B2, C2 = _run_arm(agg, "B", bdir), _run_arm(agg, "C", bdir)
    pool_hash = _pool_hash(rin / "frozen_candidate_pool_round4.json")

    b_set, c_set = set(eb["B_selected8"]), set(eb["C_selected8"])
    overlap = b_set & c_set
    change = 8 - len(overlap)
    jaccard = len(overlap) / len(b_set | c_set)

    checks = {
        "agg_source_sha_matches_bundle": agg_sha == sc["selector_source_sha256"],
        "B_selection_hash_match": B1["selection_hash"] == eb["B_selection_hash"],
        "C_selection_hash_match": C1["selection_hash"] == eb["C_selection_hash"],
        "B_selected8_exact_set": B1["selected8_sorted"] == sorted(eb["B_selected8"]),
        "C_selected8_exact_set": C1["selected8_sorted"] == sorted(eb["C_selected8"]),
        "candidate_count_32": B1["n_candidates"] == 32 and C1["n_candidates"] == 32,
        "change_is_4_of_8": change == 4 and eb["selected_set_change"] == "4/8",
        "jaccard_match": abs(jaccard - eb["jaccard"]) < 1e-3,
        "overlap_4": len(overlap) == 4,
        "B_determinism_bitidentical": (B1 == B2),
        "C_determinism_bitidentical": (C1 == C2),
        "selector_rng_seed_null": sc["rng_seed"] is None,
        "pool_hash_match": pool_hash == eb["legacy_pool_hash"],
    }
    return {
        "replay_mode": "OFFLINE_DETERMINISTIC_NO_LLM",
        "aggregation_source_sha256": agg_sha,
        "weights": WEIGHTS, "temperature": 1.0, "k": 8,
        "expected_anchors": {
            "B_selection_hash": eb["B_selection_hash"],
            "C_selection_hash": eb["C_selection_hash"],
            "legacy_pool_hash": eb["legacy_pool_hash"]},
        "recomputed": {
            "B_selection_hash": B1["selection_hash"],
            "C_selection_hash": C1["selection_hash"],
            "B_selected8_score_order": B1["selected8_score_order"],
            "C_selected8_score_order": C1["selected8_score_order"],
            "pool_hash": pool_hash,
            "overlap": sorted(overlap),
            "entered_C_only": sorted(c_set - b_set),
            "exited_B_only": sorted(b_set - c_set),
            "change": f"{change}/8", "jaccard": round(jaccard, 4)},
        "checks": checks,
        "ALL_ANCHORS_PASS": all(checks.values()),
    }
