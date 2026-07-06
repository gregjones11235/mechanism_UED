"""Task selection utilities for DiCode.

This module provides functions for:
1. Sampling tasks for training from the active set (prioritized replay).
2. Selecting frontier tasks for evolution (parent selection).
"""

# --- Standard Library ---
import random

# --- Third-Party ---
import numpy as np
from omegaconf import DictConfig

# --- Local Modules ---
from dicode.dreaming.gen_manager import GenManager, TaskArchive


def sample_tasks_for_training(
    gen_manager: GenManager, config: DictConfig, n: int
) -> list[str]:
    """Samples n tasks from the active set for training using prioritized replay.

    Implements PLR-style sampling:
    1. Gets all tasks where `is_active=True`.
    2. Calculates score weight (w_s) based on rank or top-k prioritization.
    3. Calculates staleness weight (w_c) based on time since last trained.
    4. Combines: w = (1 - staleness_coeff) * w_s + staleness_coeff * w_c
    5. Samples `n` tasks using combined weights.

    Args:
        gen_manager: The GenManager instance.
        config: Hydra configuration.
        n: Number of tasks to sample.

    Returns:
        List of sampled task IDs.
    """
    print("  [Sampling] Starting prioritized sampling from active set...")

    # Get active tasks
    active_pool = []
    with gen_manager.archive._lock:
        for node_id, data in gen_manager.archive.graph.nodes(data=True):
            if data.get("is_active"):
                active_pool.append((node_id, data))

    if not active_pool:
        print("  [Sampling] Warning: No active tasks found.")
        return []

    current_session_idx = gen_manager.session_idx
    staleness_coeff = config.dicode_manager.staleness_coeff
    method = config.dicode_manager.prioritization_method
    temperature = config.dicode_manager.prioritization_temperature
    topk_k = config.dicode_manager.topk_k

    num_active = len(active_pool)
    print(f"  [Sampling] Method: {method} | Active tasks: {num_active}")

    # Extract data
    node_ids = [d[0] for d in active_pool]
    scores = np.array([d[1].get("priority_score", 0.0) for d in active_pool])

    # Calculate score weights
    w_s = _calculate_score_weights(scores, method, temperature, topk_k)

    # Calculate staleness weights
    timestamps = np.array([d[1].get("session_last_trained", 0) for d in active_pool])
    staleness = np.maximum(0.0, current_session_idx - timestamps)
    w_c = staleness / staleness.sum() if staleness.sum() > 0 else np.ones(num_active) / num_active

    # Combine weights
    final_weights = (1.0 - staleness_coeff) * w_s + staleness_coeff * w_c

    if final_weights.sum() <= 0:
        print("  [Sampling] Warning: Final weights are zero. Using uniform.")
        probabilities = np.ones(num_active) / num_active
    else:
        probabilities = final_weights / final_weights.sum()

    # Sample
    num_to_sample = min(n, num_active)
    sampled_task_ids = np.random.choice(
        node_ids, size=num_to_sample, replace=False, p=probabilities
    ).tolist()

    print(f"  [Sampling] Sampled {len(sampled_task_ids)} tasks.")
    return sampled_task_ids


def append_rehearsal_tasks(
    gen_manager, config, siege_batch: list[str]
) -> list[str]:
    """v6 §3.6 REHEARSAL — APPEND rehearsal levels to the training batch WITHOUT touching the siege
    allotment (user 2026-07-04: the siege batch must keep its full N levels; rehearsal is EXTRA).

    Rehearsal levels are those that TEACH a protected (conquered/consolidated) skill that is CURRENTLY
    FORGETTING (SR has dropped). Design decision (user 2026-07-04): rehearsal is triggered ONLY on a
    detected FORGETTING signal, NOT proactively every session — so when nothing protected is slipping,
    the full training budget stays on the siege focus. (This departs from §3.6's original "proactive
    lock" wording toward a forgetting-triggered rescue; noted in v6_design.md §3.6.) A single chain may
    consume MULTIPLE rehearsal slots (one per forgetting skill it contains, times the per-skill knob) —
    so a badly-slipping chain gets rescued in proportion to how much of it is slipping.

    This returns ``siege_batch + rehearsal_extra`` where:
      - the siege batch is returned UNCHANGED and in full (never shrunk to make room),
      - rehearsal count ADAPTS to the forgetting load:
            quota = rehearsal_per_forgetting_skill * (#protected skills currently FORGETTING)
        capped by a fraction of the siege allotment AND an absolute total-batch cap:
            quota <= floor(len(siege_batch) * rehearsal_max_frac)
            len(siege_batch) + quota <= rehearsal_total_cap        # e.g. 24, wall-clock guard
      - when NO protected skill is forgetting, quota == 0 -> siege_batch returned unchanged.

    A strict no-op (returns siege_batch unchanged) when siege is off, the protected set is empty, no
    protected skill is forgetting, or no active level teaches a forgetting skill — so the baseline /
    v5y path is byte-for-byte unchanged.
    """
    # BUGFIX (2026-07-05): the siege notebook + profile log live on the TaskGenerator
    # (gen_manager.task_generator), NOT on the GenManager itself — that is where _ensure_modeler
    # constructs and apply_llm_update mutates them. The original code read them off gen_manager
    # directly, so getattr always returned None and rehearsal silently no-op'd for the ENTIRE v6-OLD
    # run (0 rehearsal log lines, 0 rescues) despite a non-empty protected_set on disk. Resolve the
    # holder first; fall back to gen_manager for forward-compat if the attr is ever hoisted up.
    holder = getattr(gen_manager, "task_generator", None) or gen_manager
    notebook = getattr(holder, "_siege_notebook", None)
    if notebook is None:
        # siege off / v5y path — stay byte-for-byte silent (no log) so baseline output is unchanged.
        return siege_batch
    # log #2 (user 2026-07-05): from here on siege IS on, so make every silent early-return audible —
    # "no rehearsal this session" is otherwise indistinguishable between "nothing protected yet",
    # "nothing forgetting", "quota clamped to 0", and "no active level teaches a forgetting skill".
    protected = {p.lower() for p in notebook.protected_set()}
    if not protected:
        print(
            "  [Sampling][rehearsal] skipped: protected set empty "
            "(nothing conquered/consolidated yet to rehearse)."
        )
        return siege_batch

    dm = config.dicode_manager
    per_skill = int(getattr(dm, "rehearsal_per_forgetting_skill", 2))
    max_frac = float(getattr(dm, "rehearsal_max_frac", 0.5))
    total_cap = int(getattr(dm, "rehearsal_total_cap", 24))

    # Trigger: which protected skills are CURRENTLY FORGETTING (SR peaked then dropped). Rehearsal
    # fires ONLY if this is non-empty (user 2026-07-04).
    forgetting: set[str] = set()
    plog = getattr(holder, "_profile_log", None)
    if plog is not None:
        try:
            # §2⑥ family-split thresholds from the notebook's config-resolved SiegeThresholds, so a
            # forgetting COMBAT milestone (low peak) can actually be flagged for rehearsal.
            th = getattr(notebook, "th", None)
            kw = {} if th is None else dict(
                drop_pp=th.forgetting_drop_pp,
                min_peak=th.forgetting_min_peak,
                combat_drop_pp=th.forgetting_combat_drop_pp,
                combat_min_peak=th.forgetting_combat_min_peak,
            )
            forgetting = {f["achievement"].lower() for f in plog.forgetting_candidates(**kw)}
        except Exception as e:  # noqa: BLE001 - never let a prefilter hiccup break training sampling
            # LOUD, not silent: a swallowed error here makes rehearsal a no-op that is indistinguishable
            # in the log from "nothing is forgetting" — exactly how v6-OLD ran an ENTIRE run with
            # rehearsal dead (holder mis-resolution) without anyone noticing. Print the full traceback so
            # a real bug (bad th/plog type, changed forgetting_candidates signature) surfaces immediately
            # instead of hiding behind the benign "all holding" message below.
            import traceback
            print(
                f"  [Sampling][rehearsal] WARNING: forgetting-detection raised "
                f"({type(e).__name__}: {e}); treating as no forgetting THIS session. "
                f"If this repeats, rehearsal is silently disabled — investigate. Traceback:"
            )
            traceback.print_exc()
            forgetting = set()
    forgetting &= protected
    if not forgetting:
        print(
            f"  [Sampling][rehearsal] skipped: {len(protected)} protected skill(s) all holding "
            "(none FORGETTING) -> full budget stays on the siege focus."
        )
        return siege_batch  # nothing protected is slipping -> keep all fire-power on the siege focus

    quota = per_skill * len(forgetting)
    frac_ceiling = int(len(siege_batch) * max_frac)          # ≤ half the siege allotment
    quota = min(quota, frac_ceiling)
    # absolute total cap: siege + rehearsal ≤ total_cap (e.g. 24). Never shrink siege_batch to fit.
    room_to_cap = max(0, total_cap - len(siege_batch))
    quota = max(0, min(quota, room_to_cap))
    if quota == 0:
        print(
            f"  [Sampling][rehearsal] skipped: {len(forgetting)} skill(s) FORGETTING but quota "
            f"clamped to 0 (frac_ceil={frac_ceiling}, room_to_cap={room_to_cap}; "
            f"siege_batch={len(siege_batch)} already at/over cap {total_cap})."
        )
        return siege_batch

    # Levels that TEACH a forgetting protected skill (Relevant Achievements ∩ forgetting). Exclude
    # levels already in the siege batch (no point rehearsing what's already trained this session).
    from dicode.dreaming.auction_integration import parse_relevant_achievements

    already = set(siege_batch)
    rehearsal_extra: list[str] = []
    with gen_manager.archive._lock:
        for node_id, data in gen_manager.archive.graph.nodes(data=True):
            if not data.get("is_active") or node_id in already:
                continue
            if parse_relevant_achievements(data.get("description", "") or "") & forgetting:
                rehearsal_extra.append(node_id)
    rehearsal_extra = rehearsal_extra[:quota]
    if not rehearsal_extra:
        print(
            f"  [Sampling][rehearsal] skipped: {len(forgetting)} skill(s) FORGETTING, quota={quota}, "
            "but NO active level teaches a forgetting skill (all such levels already in the siege "
            "batch, or none exist in the archive)."
        )
        return siege_batch

    print(
        f"  [Sampling][rehearsal] siege={len(siege_batch)} kept in full; {len(forgetting)} protected "
        f"skill(s) FORGETTING; quota={quota} (frac_ceil={frac_ceiling}, room_to_cap={room_to_cap}); "
        f"appended {len(rehearsal_extra)} rehearsal level(s) -> total "
        f"{len(siege_batch) + len(rehearsal_extra)}."
    )
    return siege_batch + rehearsal_extra


def _calculate_score_weights(
    scores: np.ndarray,
    method: str,
    temperature: float,
    topk_k: int,
) -> np.ndarray:
    """Calculates score-based sampling weights.

    Args:
        scores: Array of priority scores for each task.
        method: Either "rank" or "topk".
        temperature: Temperature for rank-based softmax.
        topk_k: Number of top tasks for top-k method.

    Returns:
        Normalized weight array.
    """
    num_tasks = len(scores)

    if method == "rank":
        # Rank-based prioritization (from PLR)
        order = (-scores).argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(num_tasks) + 1
        w_s = (1.0 / ranks) ** (1.0 / temperature)

    elif method == "topk":
        k = min(num_tasks, topk_k)
        topk_indices = np.argpartition(-scores, k - 1)[:k]
        topk_mask = np.zeros(num_tasks, dtype=bool)
        topk_mask[topk_indices] = True

        # Softmax only on top-k scores
        scores_masked = np.where(topk_mask, scores, -np.inf)
        stable_scores = scores_masked - np.max(scores_masked)
        exp_scores = np.exp(stable_scores)
        exp_scores = np.where(topk_mask, exp_scores, 0.0)
        w_s = exp_scores

    else:
        raise ValueError(f"Unknown prioritization_method: {method}")

    return w_s / w_s.sum()


def select_tasks_for_evolution(
    config: DictConfig, archive: TaskArchive, idx: int, k: int
) -> list[str]:
    """Selects frontier tasks for evolution (parent selection).

    A task is on the frontier if:
    1. It has an allowed status (configurable).
    2. It has not exceeded the maximum branching factor.
    3. (Optional) It does not have a viable child that supersedes it.

    Args:
        config: Hydra configuration.
        archive: The task archive.
        idx: Current session index.
        k: Number of tasks to select.

    Returns:
        List of task IDs for evolution.
    """
    # Determine selection strategy
    if config.ablation:
        return _select_tasks_unrestricted(archive, k)

    parent_selection = config.dicode_manager.parent_selection
    if parent_selection == "strict":
        return _select_tasks_frontier(
            archive,
            k,
            parent_statuses={"A", "B"},
            success_statuses={"A", "B", "C"},
            max_children=5,
            sort_by_score=True,
        )
    elif parent_selection == "lenient":
        return _select_tasks_frontier(
            archive,
            k,
            parent_statuses={"A", "B", "C", "D"},
            success_statuses={"A", "B", "C"},
            max_children=5,
            sort_by_score=False,
        )
    else:
        return _select_tasks_unrestricted(archive, k)


def _select_tasks_frontier(
    archive: TaskArchive,
    k: int,
    parent_statuses: set[str],
    success_statuses: set[str],
    max_children: int,
    sort_by_score: bool,
) -> list[str]:
    """Selects frontier tasks with lineage-aware filtering.

    Args:
        archive: The task archive.
        k: Number of tasks to select.
        parent_statuses: Set of allowed parent statuses.
        success_statuses: Child statuses that "graduate" a parent.
        max_children: Maximum children per node.
        sort_by_score: Whether to sort by priority_score before selection.

    Returns:
        List of selected task IDs.
    """
    eligible_tasks = []

    for node_id, data in archive.graph.nodes(data=True):
        status = data.get("status")

        # Filter 1: Allowed status
        if status not in parent_statuses:
            continue

        # Filter 2: Branching cap
        children = list(archive.graph.successors(node_id))
        if len(children) >= max_children:
            continue

        # Filter 3: No viable child
        has_viable_child = any(
            archive.graph.has_node(child_id)
            and archive.graph.nodes[child_id].get("status") in success_statuses
            for child_id in children
        )

        if not has_viable_child:
            eligible_tasks.append(node_id)

    print(f"  Found {len(eligible_tasks)} frontier tasks.")

    # Fallback: if no frontier tasks, use any mastered task
    if not eligible_tasks:
        print("  Warning: No frontier tasks found. Fallback to mastered tasks.")
        eligible_tasks = [
            n for n, d in archive.graph.nodes(data=True) if d.get("status") == "A"
        ]

    # Selection
    if sort_by_score:
        eligible_tasks.sort(
            key=lambda tid: archive.graph.nodes[tid].get("priority_score", 0.0),
            reverse=True,
        )
        selected = eligible_tasks[:k]
    else:
        selected = random.sample(eligible_tasks, min(k, len(eligible_tasks)))

    return _replicate_to_fill(selected, k)


def _select_tasks_unrestricted(archive: TaskArchive, k: int) -> list[str]:
    """Selects tasks without lineage restrictions (for ablation).

    Args:
        archive: The task archive.
        k: Number of tasks to select.

    Returns:
        List of selected task IDs.
    """
    allowed_statuses = {"A", "B", "C", "D"}
    eligible_tasks = [
        node_id
        for node_id, data in archive.graph.nodes(data=True)
        if data.get("status") in allowed_statuses
    ]

    print(f"  Found {len(eligible_tasks)} tasks (unrestricted).")

    if len(eligible_tasks) >= k:
        selected = random.sample(eligible_tasks, k)
    else:
        selected = eligible_tasks[:]

    return _replicate_to_fill(selected, k)


def _replicate_to_fill(selected: list[str], k: int) -> list[str]:
    """Replicates selected tasks to fill batch size k.

    Args:
        selected: List of selected task IDs.
        k: Target batch size.

    Returns:
        List of task IDs with length k (if possible).
    """
    num_selected = len(selected)
    if num_selected == 0:
        return []

    if num_selected >= k:
        return selected

    factor = k // num_selected
    remainder = k % num_selected
    result = selected * factor + selected[:remainder]
    random.shuffle(result)
    return result
