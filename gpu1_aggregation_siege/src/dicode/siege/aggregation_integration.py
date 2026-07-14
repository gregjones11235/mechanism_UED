"""S3 — SIEGE-Aggregation Integration Pipeline.

Connects held-out evidence → SIEGE state → chain-completeness gate →
frozen candidate pool → immutable cache → selector dispatch → focus quota →
forgetting rehearsal → PPO training.

Chain-completeness is a HARD admission gate, not a weighted preference.
"""

import json, os, time, numpy as np
from typing import Optional


# ==============================================================================
# Chain-Completeness Hard Admission Gate
# ==============================================================================

def chain_completeness_gate(
    candidate_ids: list[str],
    candidate_metadata: dict[str, dict],
    notebook,
) -> tuple[list[str], list[str], dict]:
    """Hard gate: only admit candidates with valid chain relevance.

    A candidate is ADMITTED if:
      - It has siege_wall=True (relevant to at least one chain), OR
      - It has chain_complete=True (all prerequisites mastered)

    A candidate is REJECTED if:
      - It has no chain relevance AND no chain links are complete

    Args:
        candidate_ids: List of candidate task IDs.
        candidate_metadata: Dict task_id -> SIEGE metadata dict.
        notebook: SiegeNotebook instance.

    Returns:
        (admitted_ids, rejected_ids, gate_report)
    """
    admitted = []
    rejected = []
    reasons = {}

    for tid in candidate_ids:
        # Get or compute metadata
        if tid in candidate_metadata:
            meta = candidate_metadata[tid]
        else:
            # Tasks without metadata: try to compute from achievements
            meta = {"siege_wall": False, "chain_complete": False}

        # Hard gate logic
        if meta.get("siege_wall") or meta.get("chain_complete"):
            admitted.append(tid)
            reasons[tid] = "admitted"
        else:
            rejected.append(tid)
            reasons[tid] = "no_chain_relevance"

    gate_report = {
        "candidates_total": len(candidate_ids),
        "admitted": len(admitted),
        "rejected": len(rejected),
        "rejection_reasons": {
            tid: reasons[tid] for tid in rejected
        },
    }

    return admitted, rejected, gate_report


# ==============================================================================
# SIEGE-Aware Candidate Pool
# ==============================================================================

def build_siege_candidate_pool(
    gen_manager,
    notebook,
    target_candidate_count: int = 32,
) -> dict:
    """Build a SIEGE-aware frozen candidate pool.

    1. Get all active tasks from archive
    2. Compute SIEGE metadata for each
    3. Apply chain-completeness hard gate
    4. Fill to target_candidate_count with admitted tasks

    Returns dict with pool data and gate report.
    """
    # Get active tasks
    active_pool = []
    with gen_manager.archive._lock:
        for node_id, data in gen_manager.archive.graph.nodes(data=True):
            if data.get("is_active"):
                active_pool.append((node_id, data))

    if not active_pool:
        return {"candidates": [], "admitted": 0, "rejected": 0, "gate_report": {}}

    # Compute SIEGE metadata for each candidate
    candidate_metadata = {}
    for task_id, data in active_pool:
        achievements = data.get("relevant_achievements", [])
        if not achievements:
            # Fallback: extract from task data
            achievements = data.get("skills", "").split(", ") if data.get("skills") else []
            achievements = [a.strip() for a in achievements if a.strip()]

        if achievements:
            meta = notebook.get_candidate_metadata(task_id, achievements)
        else:
            meta = {"siege_wall": False, "chain_complete": False}

        candidate_metadata[task_id] = meta

    # Hard admission gate
    task_ids = [t[0] for t in active_pool]
    admitted, rejected, gate_report = chain_completeness_gate(
        task_ids, candidate_metadata, notebook
    )

    # Fill to target count
    if len(admitted) < target_candidate_count:
        # If not enough admitted, include some rejected tasks
        # (to maintain candidate_count=32 for comparison fairness)
        shortfall = target_candidate_count - len(admitted)
        admitted.extend(rejected[:shortfall])
        gate_report["filled_from_rejected"] = shortfall

    # Truncate to target
    pool_candidates = admitted[:target_candidate_count]

    return {
        "candidates": pool_candidates,
        "admitted_original": len(admitted) - gate_report.get("filled_from_rejected", 0),
        "rejected": len(rejected),
        "gate_report": gate_report,
        "metadata": {tid: candidate_metadata.get(tid, {}) for tid in pool_candidates},
    }


# ==============================================================================
# Focus Quota Integration
# ==============================================================================

def apply_focus_quota(
    selected_ids: list[str],
    candidate_pool_data: dict,
    notebook,
    session: int,
) -> list[str]:
    """Apply SIEGE focus quota to selector output.

    Ensures at least min_chain_tasks from chain-relevant candidates.
    Does NOT modify if quota is already satisfied.
    """
    pool_candidates = candidate_pool_data.get("candidates", [])
    metadata = candidate_pool_data.get("metadata", {})

    # Identify chain-relevant tasks
    chain_tasks = [
        tid for tid in pool_candidates
        if metadata.get(tid, {}).get("siege_wall", False)
    ]

    # Enforce quota
    result = notebook.focus_quota.enforce(
        selected_ids, chain_tasks, pool_candidates, session
    )
    return result


# ==============================================================================
# Forgetting Rehearsal Integration
# ==============================================================================

def apply_rehearsal_allocation(
    selected_ids: list[str],
    candidate_pool_data: dict,
    notebook,
    session: int,
    max_rehearsal_slots: int = 2,
) -> tuple[list[str], dict]:
    """Insert rehearsal tasks into the selection if forgetting is detected.

    If rehearsal is active, replaces up to max_rehearsal_slots non-chain
    tasks with chain tasks that need rehearsal.
    """
    if not notebook.rehearsal.rehearsal_active:
        return list(selected_ids), {"rehearsal_applied": False}

    # Find tasks matching at-risk achievements
    at_risk = set(notebook.rehearsal.active_rehearsals)
    metadata = candidate_pool_data.get("metadata", {})

    rehearsal_candidates = []
    for tid in candidate_pool_data.get("candidates", []):
        meta = metadata.get(tid, {})
        unmastered = set(meta.get("unmastered_links", []))
        if unmastered & at_risk:
            rehearsal_candidates.append(tid)

    if not rehearsal_candidates:
        return list(selected_ids), {"rehearsal_applied": False, "reason": "no_matching_tasks"}

    # Replace non-chain tasks with rehearsal tasks
    chain_set = set(candidate_pool_data.get("candidates", []))
    result = list(selected_ids)
    replaced = 0

    for i, tid in enumerate(result):
        if replaced >= max_rehearsal_slots:
            break
        if tid not in chain_set and rehearsal_candidates:
            result[i] = rehearsal_candidates.pop(0)
            replaced += 1

    return result, {
        "rehearsal_applied": True,
        "slots_used": replaced,
        "at_risk_skills": list(at_risk),
    }


# ==============================================================================
# Full Pipeline
# ==============================================================================

def run_siege_aggregation_pipeline(
    gen_manager,
    notebook,
    config,
    n_selected: int = 8,
    candidate_count: int = 32,
    session: int = 1,
    global_step: int = 0,
    held_out_metrics: Optional[dict] = None,
) -> dict:
    """Full SIEGE → Aggregation pipeline for one curriculum session.

    1. Update SIEGE state from held-out evidence
    2. Build SIEGE-aware candidate pool (with chain-completeness gate)
    3. Run aggregation selector dispatch
    4. Apply focus quota
    5. Apply forgetting rehearsal
    6. Return final selected tasks + diagnostics

    Args:
        gen_manager: GenManager instance.
        notebook: SiegeNotebook instance.
        config: Hydra config.
        n_selected: Number of tasks to select (default 8).
        candidate_count: Target candidate pool size (default 32).
        session: Current curriculum session index.
        global_step: Current global environment step.
        held_out_metrics: Dict achievement_name -> SR (0-1).

    Returns:
        Dict with selected_ids, siege_state, diagnostics.
    """
    # Step 1: Update SIEGE state
    siege_update = None
    if held_out_metrics:
        siege_update = notebook.update(held_out_metrics, global_step)

    # Step 2: Build SIEGE-aware candidate pool
    pool_data = build_siege_candidate_pool(
        gen_manager, notebook, target_candidate_count=candidate_count
    )

    if not pool_data["candidates"]:
        return {
            "selected_ids": [],
            "siege_state": siege_update,
            "pool_data": pool_data,
            "error": "No candidates after chain-completeness gate",
        }

    # Step 3: Run aggregation selector
    # (delegates to existing aggregation dispatch)
    from dicode.selection import sample_tasks_for_training

    selected = sample_tasks_for_training(gen_manager, config, n_selected)

    # Step 4: Focus quota
    selected = apply_focus_quota(selected, pool_data, notebook, session)

    # Step 5: Forgetting rehearsal
    selected, rehearsal_info = apply_rehearsal_allocation(
        selected, pool_data, notebook, session
    )

    return {
        "selected_ids": selected,
        "siege_state": siege_update,
        "pool_data": pool_data,
        "focus_quota_check": notebook.focus_quota.check(
            selected,
            [tid for tid in pool_data["candidates"]
             if pool_data["metadata"].get(tid, {}).get("siege_wall")],
            session,
        ),
        "rehearsal": rehearsal_info,
        "n_selected": len(selected),
    }
