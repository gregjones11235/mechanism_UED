"""Performance-aware aggregation mechanisms for DiCode curriculum selection.

This module implements multiple aggregation strategies that combine
heterogeneous curriculum signals to improve training efficiency, final
performance, curriculum stability, and generalization.

Strategies:
  A1_raw_weighted: Raw linear combination of signal scores.
  A2_robust_weighted: Linear combination with robust (median/IQR) normalization.
  A3_soft_copeland: Pairwise preference aggregation via Soft Copeland scores.
  A4_budgeted_soft_copeland: Soft Copeland with budget caps per source/signal.
  A5_budgeted_retention_trigger: Budgeted Copeland + optional retention trigger.
  A6_entropy_regularized: A5 + entropy-regularized softmax sampling.

The retention_trigger and forgetting_index are diagnostic fields — this is
a general-purpose curriculum selector, not only an anti-forgetting mechanism.

All functions are pure and operate on NumPy arrays for testability.
"""

import json
import os
import time
from typing import Any, Callable, Optional

import numpy as np


# ==============================================================================
# Core Normalization
# ==============================================================================


def robust_normalize(
    values: np.ndarray,
    clip: float = 3.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Apply median/IQR robust normalization with clipping.

    Transforms: (value - median) / IQR, then clips to [-clip, +clip].

    Args:
        values: 1-D array of raw scores.
        clip: Clipping threshold in IQR units.
        epsilon: Small constant to avoid division by zero.

    Returns:
        Normalized 1-D array, same shape as input.
        If IQR is zero, returns zeros.
    """
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    iqr = max(iqr, epsilon)

    normalized = (values - median) / iqr
    normalized = np.clip(normalized, -clip, clip)

    # Handle NaN/Inf
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=clip, neginf=-clip)
    return normalized


# ==============================================================================
# Signal Score Computation
# ==============================================================================


def compute_signal_scores(
    task_ids: list[str],
    archive_graph,
    original_task_id: str = "original_craftax",
    original_sr_history: Optional[list[float]] = None,
) -> dict[str, np.ndarray]:
    """Compute heterogeneous curriculum signals for each candidate task.

    Signals computed:
      - progression: priority_score (learnability / PVL / MaxMC)
      - retention: 1.0 / (1.0 + sessions_since_last_trained)
      - novelty: inverse of times trained (encourages sampling newer tasks)
      - critic_penalty: normalized staleness penalty
      - monopoly_penalty: penalty for over-sampled sources/skills

    Args:
        task_ids: List of candidate task IDs.
        archive_graph: The networkx DiGraph with task node data.
        original_task_id: The ID of the original Craftax evaluation task.
        original_sr_history: History of original task success rates for
            computing forgetting index.

    Returns:
        Dict mapping signal name -> 1-D NumPy array of shape (n_tasks,).
    """
    n = len(task_ids)
    if n == 0:
        return {
            "progression": np.array([]),
            "retention": np.array([]),
            "novelty": np.array([]),
            "critic_penalty": np.array([]),
            "monopoly_penalty": np.array([]),
            "source_ids": np.array([]),
            "skill_counts": np.array([]),
        }

    progression = np.zeros(n, dtype=np.float64)
    retention = np.zeros(n, dtype=np.float64)
    novelty = np.zeros(n, dtype=np.float64)
    critic_penalty = np.zeros(n, dtype=np.float64)
    session_last_trained = np.zeros(n, dtype=np.float64)
    source_ids_list = []

    for i, tid in enumerate(task_ids):
        try:
            data = archive_graph.nodes[tid]
        except (KeyError, AttributeError):
            data = {}
        progression[i] = float(data.get("priority_score", 0.0))
        slt = int(data.get("session_last_trained", -1))
        session_last_trained[i] = slt

        # Retention: higher for tasks not recently trained (inverse recency)
        # Tasks never trained get high retention score (replay value)
        if slt >= 0:
            retention[i] = 1.0 / (1.0 + slt)
        else:
            retention[i] = 1.0  # Never trained - could be important

        # Novelty: approximate via inverse of times in performance_history
        perf_history = data.get("performance_history", [])
        times_trained = max(len(perf_history), 1)
        novelty[i] = 1.0 / times_trained

        # Monopoly placeholder — filled later based on source distribution
        source = data.get("type", data.get("status", "unknown"))
        source_ids_list.append(source)

    # --- Compute monopoly penalty based on source frequency ---
    unique_sources, source_counts = np.unique(source_ids_list, return_counts=True)
    source_freq = dict(zip(unique_sources, source_counts / n))
    monopoly_penalty = np.zeros(n, dtype=np.float64)
    for i, src in enumerate(source_ids_list):
        monopoly_penalty[i] = source_freq.get(src, 0.0)

    # --- Compute skill count diversity penalty ---
    skill_counts = np.ones(n, dtype=np.float64)  # placeholder

    # --- Critic penalty: based on staleness pattern ---
    # High penalty when a task dominates recent sessions
    if session_last_trained.max() > 0:
        critic_penalty = session_last_trained / max(1.0, session_last_trained.max())
    else:
        critic_penalty = np.zeros(n, dtype=np.float64)

    return {
        "progression": progression,
        "retention": retention,
        "novelty": novelty,
        "critic_penalty": critic_penalty,
        "monopoly_penalty": monopoly_penalty,
        "source_ids": np.array(source_ids_list),
        "skill_counts": skill_counts,
    }


# ==============================================================================
# Forgetting Statistics
# ==============================================================================


def compute_forgetting_stats(
    current_old_success: float,
    best_old_success_so_far: float,
    original_sr_history: Optional[list[float]] = None,
) -> dict[str, float]:
    """Compute forgetting statistics from original task performance.

    Args:
        current_old_success: Current old task success rate.
        best_old_success_so_far: Best old task success rate observed.
        original_sr_history: Optional history of original task success rates.

    Returns:
        Dict with keys: forgetting_index, best_old_success, current_old_success,
        anti_forgetting_mode.
    """
    if original_sr_history is None:
        original_sr_history = []

    forgetting_index = max(0.0, best_old_success_so_far - current_old_success)
    anti_forgetting_mode = forgetting_index > 0.0

    return {
        "forgetting_index": float(forgetting_index),
        "best_old_success": float(best_old_success_so_far),
        "current_old_success": float(current_old_success),
        "anti_forgetting_mode": bool(anti_forgetting_mode),
    }


# ==============================================================================
# Aggregation Strategies
# ==============================================================================


def _aggregate_raw_weighted(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """A1: Raw linear combination of signal scores."""
    final = np.zeros_like(signals["progression"], dtype=np.float64)

    if weights.get("w_progression", 0.0) > 0:
        final += weights["w_progression"] * signals["progression"]
    if weights.get("w_retention", 0.0) > 0:
        final += weights["w_retention"] * signals["retention"]
    if weights.get("w_novelty", 0.0) > 0:
        final += weights["w_novelty"] * signals["novelty"]
    if weights.get("w_critic", 0.0) > 0:
        final -= weights["w_critic"] * signals["critic_penalty"]
    if weights.get("w_monopoly", 0.0) > 0:
        final -= weights["w_monopoly"] * signals["monopoly_penalty"]

    return final


def _aggregate_robust_weighted(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """A2: Robust-weighted with median/IQR normalization per signal."""
    final = np.zeros_like(signals["progression"], dtype=np.float64)

    if weights.get("w_progression", 0.0) > 0 and len(signals["progression"]) > 0:
        norm_prog = robust_normalize(signals["progression"])
        final += weights["w_progression"] * norm_prog

    if weights.get("w_retention", 0.0) > 0 and len(signals["retention"]) > 0:
        norm_ret = robust_normalize(signals["retention"])
        final += weights["w_retention"] * norm_ret

    if weights.get("w_novelty", 0.0) > 0 and len(signals["novelty"]) > 0:
        norm_nov = robust_normalize(signals["novelty"])
        final += weights["w_novelty"] * norm_nov

    if weights.get("w_critic", 0.0) > 0 and len(signals["critic_penalty"]) > 0:
        norm_critic = robust_normalize(signals["critic_penalty"])
        final -= weights["w_critic"] * norm_critic

    if weights.get("w_monopoly", 0.0) > 0 and len(signals["monopoly_penalty"]) > 0:
        norm_mono = robust_normalize(signals["monopoly_penalty"])
        final -= weights["w_monopoly"] * norm_mono

    return final


def _aggregate_soft_copeland(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
    temperature: float = 1.0,
) -> np.ndarray:
    """A3: Soft Copeland pairwise preference aggregation.

    For each signal, compute pairwise preference matrix.
    Task i beats task j on signal s if score_i[s] > score_j[s].
    Aggregate pairwise wins into a Copeland score, apply temperature.
    """
    n = len(signals["progression"])
    if n <= 1:
        return np.ones(n, dtype=np.float64) if n > 0 else np.array([])

    copeland_scores = np.zeros(n, dtype=np.float64)

    signal_keys = ["progression", "retention", "novelty"]
    signal_weights = {
        "progression": weights.get("w_progression", 0.34),
        "retention": weights.get("w_retention", 0.33),
        "novelty": weights.get("w_novelty", 0.33),
    }
    penalty_keys = ["critic_penalty", "monopoly_penalty"]
    penalty_weights = {
        "critic_penalty": weights.get("w_critic", 0.01),
        "monopoly_penalty": weights.get("w_monopoly", 0.01),
    }

    for key in signal_keys:
        w = signal_weights.get(key, 0.0)
        if w <= 0 or key not in signals or len(signals[key]) == 0:
            continue
        values = robust_normalize(signals[key])
        for i in range(n):
            for j in range(n):
                if i != j and values[i] > values[j]:
                    copeland_scores[i] += w
                elif i != j and values[i] == values[j]:
                    copeland_scores[i] += 0.5 * w

    for key in penalty_keys:
        w = penalty_weights.get(key, 0.0)
        if w <= 0 or key not in signals or len(signals[key]) == 0:
            continue
        values = robust_normalize(signals[key])
        for i in range(n):
            for j in range(n):
                if i != j and values[i] > values[j]:
                    copeland_scores[i] -= w
                elif i != j and values[i] == values[j]:
                    copeland_scores[i] -= 0.5 * w

    # Apply temperature
    if temperature > 0:
        copeland_scores = copeland_scores / temperature

    # Normalize to [0, 1] range
    c_min = copeland_scores.min()
    c_max = copeland_scores.max()
    if c_max - c_min > 1e-8:
        copeland_scores = (copeland_scores - c_min) / (c_max - c_min)
    else:
        copeland_scores = np.ones(n, dtype=np.float64) / n

    return copeland_scores


# ==============================================================================
# Budget Caps
# ==============================================================================


def apply_budget_caps(
    scores: np.ndarray,
    task_ids: list[str],
    source_ids: np.ndarray,
    max_source_share: float = 0.5,
    max_signal_share: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Apply budget caps to prevent any single source/signal from dominating.

    A4: Caps the number of selected tasks from any single source.
    If a source exceeds max_source_share, its excess tasks are penalized.

    Args:
        scores: Aggregated scores for each task.
        task_ids: List of task IDs.
        source_ids: Array of source type strings for each task.
        max_source_share: Maximum fraction of selections from one source.
        max_signal_share: Maximum fraction of selections from one signal.

    Returns:
        Tuple of (adjusted_scores, budget_info dict).
    """
    n = len(scores)
    if n == 0:
        return scores, {"source_caps_applied": 0, "signal_caps_applied": 0}

    adjusted = scores.copy()
    budget_info = {"source_caps_applied": 0, "signal_caps_applied": 0}

    # Source budget caps
    unique_sources = np.unique(source_ids)
    max_allowed_per_source = max(1, int(n * max_source_share))

    for src in unique_sources:
        mask = source_ids == src
        src_count = mask.sum()
        if src_count > max_allowed_per_source:
            # Penalize excess tasks from this source
            src_indices = np.where(mask)[0]
            # Sort by score descending - keep top ones, penalize rest
            sorted_idx = src_indices[np.argsort(-adjusted[src_indices])]
            penalize_idx = sorted_idx[max_allowed_per_source:]
            adjusted[penalize_idx] *= 0.5  # Halve the score of excess tasks
            budget_info["source_caps_applied"] += len(penalize_idx)

    return adjusted, budget_info


# ==============================================================================
# Entropy Computation
# ==============================================================================


def compute_curriculum_entropy(
    scores: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """Compute entropy of the curriculum distribution.

    Args:
        scores: Score array for candidate tasks.
        temperature: Temperature for softmax.

    Returns:
        Entropy value in nats.
    """
    if len(scores) == 0:
        return 0.0

    # Softmax
    stable = scores - scores.max()
    exp_scores = np.exp(stable / max(temperature, 1e-8))
    probs = exp_scores / (exp_scores.sum() + 1e-8)

    # Entropy
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    return float(entropy)


# ==============================================================================
# Curriculum Sampling
# ==============================================================================


def sample_curriculum(
    scores: np.ndarray,
    task_ids: list[str],
    k: int,
    temperature: float = 1.0,
    entropy_regularization: float = 0.0,
) -> tuple[list[str], np.ndarray]:
    """Sample k tasks from the score distribution using softmax.

    Args:
        scores: Final aggregated scores for each candidate task.
        task_ids: List of candidate task IDs.
        k: Number of tasks to sample.
        temperature: Softmax temperature (lower = more greedy).
        entropy_regularization: Bonus entropy coefficient (A6). Higher values
            encourage more uniform sampling.

    Returns:
        Tuple of (selected_task_ids, sampling_probabilities).
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return [], np.array([])

    k = min(k, n)

    # Softmax with optional entropy regularization
    stable = scores - scores.max()
    exp_scores = np.exp(stable / max(temperature, 1e-8))

    # A6: Entropy regularization adds a uniform component
    if entropy_regularization > 0:
        uniform = np.ones(n) / n
        exp_scores = (1.0 - entropy_regularization) * exp_scores + entropy_regularization * uniform

    probs = exp_scores / (exp_scores.sum() + 1e-8)
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / (probs.sum() + 1e-8)

    # Sample without replacement
    try:
        selected_indices = np.random.choice(
            n, size=k, replace=False, p=probs
        )
    except ValueError:
        # Fallback to uniform if probabilities are invalid
        selected_indices = np.random.choice(n, size=k, replace=False)

    selected_task_ids = [task_ids[i] for i in selected_indices]
    return selected_task_ids, probs


# ==============================================================================
# Main Aggregation Selector
# ==============================================================================


def select_tasks_with_aggregation(
    gen_manager,
    config,
    n: int,
    original_task_id: str = "original_craftax",
    best_original_success: float = 0.0,
    current_original_success: float = 0.0,
    global_step: int = 0,
    run_name: str = "default",
    seed: int = 0,
    log_dir: Optional[str] = None,
) -> list[str]:
    """Main aggregation-based task selection entry point.

    Replaces the PLR-style sampling in sample_tasks_for_training when
    aggregation.enabled=True.

    Args:
        gen_manager: The GenManager instance.
        config: Hydra configuration.
        n: Number of tasks to select.
        original_task_id: ID of the original evaluation task.
        best_original_success: Best observed original task success.
        current_original_success: Current original task success.
        global_step: Global training step.
        run_name: Name for logging.
        seed: Random seed.
        log_dir: Directory for diagnostics JSONL output.

    Returns:
        List of selected task IDs.
    """
    agg_config = config.aggregation
    mode = agg_config.mode

    # Get active tasks
    active_pool = []
    with gen_manager.archive._lock:
        for node_id, data in gen_manager.archive.graph.nodes(data=True):
            if data.get("is_active"):
                active_pool.append((node_id, data))

    if not active_pool:
        print("  [Aggregation] Warning: No active tasks found.")
        return []

    num_active = len(active_pool)
    print(f"  [Aggregation] Mode: {mode} | Active tasks: {num_active}")

    task_ids = [d[0] for d in active_pool]

    # --- Compute signals ---
    signals = compute_signal_scores(
        task_ids, gen_manager.archive.graph, original_task_id
    )

    # --- Compute forgetting stats ---
    forgetting_stats = compute_forgetting_stats(
        current_original_success, best_original_success
    )

    # --- LLM Cache Integration ---
    use_llm_cache = agg_config.get("use_llm_cache", False)
    llm_cache_hits = 0
    llm_cache_misses = 0
    llm_total_cost = 0.0

    if use_llm_cache:
        from dicode.mechanisms.llm_cache import load_cache, get_cached_judgments_by_task_id

        cache_path = agg_config.get("llm_cache_path",
            "/root/experiments/dreaming-in-code-coop/mechanism_logs/llm_judgments_cache.jsonl")
        llm_cache = load_cache(cache_path)

        llm_w_prog = float(agg_config.get("llm_weight_progression", 0.5))
        llm_w_nov = float(agg_config.get("llm_weight_novelty", 0.5))
        llm_w_crit = float(agg_config.get("llm_weight_critic", 0.5))
        llm_w_ret = float(agg_config.get("llm_weight_retention", 0.2))

        for i, tid in enumerate(task_ids):
            # Look up ALL cached judgments for this task_id
            cached_for_task = get_cached_judgments_by_task_id(llm_cache, tid)

            if not cached_for_task:
                llm_cache_misses += 1
                continue

            for role, judgment in cached_for_task.items():
                if not judgment:
                    continue
                llm_cache_hits += 1
                scores_dict = judgment.get("scores", {})

                if role == "tutor":
                    llm_prog = float(scores_dict.get("progression_score", 0.5))
                    signals["progression"][i] = (
                        (1.0 - llm_w_prog) * signals["progression"][i]
                        + llm_w_prog * llm_prog
                    )
                elif role == "critic":
                    llm_critic = float(scores_dict.get("critic_penalty", 0.0))
                    signals["critic_penalty"][i] = (
                        (1.0 - llm_w_crit) * signals["critic_penalty"][i]
                        + llm_w_crit * llm_critic
                    )
                elif role == "explorer":
                    llm_nov = float(scores_dict.get("novelty_score", 0.5))
                    signals["novelty"][i] = (
                        (1.0 - llm_w_nov) * signals["novelty"][i]
                        + llm_w_nov * llm_nov
                    )

                # Retention hint
                retention_hint = float(scores_dict.get("retention_hint", 0.0))
                if retention_hint > 0:
                    signals["retention"][i] = (
                        (1.0 - llm_w_ret) * signals["retention"][i]
                        + llm_w_ret * retention_hint
                    )

        print(f"  [Aggregation] LLM cache: {llm_cache_hits} hits, {llm_cache_misses} misses (enabled={use_llm_cache})")

    # --- Get weights (with optional retention trigger boost) ---
    weights = {
        "w_progression": float(agg_config.get("w_progression", 0.34)),
        "w_retention": float(agg_config.get("w_retention", 0.33)),
        "w_novelty": float(agg_config.get("w_novelty", 0.33)),
        "w_critic": float(agg_config.get("w_critic", 0.01)),
        "w_monopoly": float(agg_config.get("w_monopoly", 0.01)),
    }

    retention_trigger = float(agg_config.get("retention_trigger", 0.15))
    anti_forgetting = False

    # A5/A6: Anti-forgetting retention trigger
    if forgetting_stats["forgetting_index"] > retention_trigger:
        anti_forgetting = True
        boost_factor = 1.0 + min(2.0, forgetting_stats["forgetting_index"] / retention_trigger)
        weights["w_retention"] *= boost_factor
        weights["w_progression"] *= 0.7
        weights["w_novelty"] *= 0.7
        print(f"  [Aggregation] Anti-forgetting TRIGGERED: "
              f"forgetting_index={forgetting_stats['forgetting_index']:.3f} > "
              f"threshold={retention_trigger}")

    # --- Aggregate ---
    temperature = float(agg_config.get("temperature", 1.0))
    entropy_reg = float(agg_config.get("entropy_regularization", 0.0))

    if mode == "raw_weighted":
        scores = _aggregate_raw_weighted(signals, weights)
    elif mode == "robust_weighted":
        scores = _aggregate_robust_weighted(signals, weights)
    elif mode == "soft_copeland":
        scores = _aggregate_soft_copeland(signals, weights, temperature)
    elif mode == "budgeted_soft_copeland":
        scores = _aggregate_soft_copeland(signals, weights, temperature)
        max_source_share = float(agg_config.get("max_source_share", 0.5))
        max_signal_share = float(agg_config.get("max_signal_share", 0.5))
        scores, budget_info = apply_budget_caps(
            scores, task_ids, signals["source_ids"],
            max_source_share, max_signal_share,
        )
    elif mode == "budgeted_retention_trigger":
        scores = _aggregate_soft_copeland(signals, weights, temperature)
        max_source_share = float(agg_config.get("max_source_share", 0.5))
        max_signal_share = float(agg_config.get("max_signal_share", 0.5))
        scores, budget_info = apply_budget_caps(
            scores, task_ids, signals["source_ids"],
            max_source_share, max_signal_share,
        )
        if not anti_forgetting:
            # Apply budget caps even without trigger (A5 baseline)
            pass
    elif mode == "entropy_regularized":
        scores = _aggregate_soft_copeland(signals, weights, temperature)
        max_source_share = float(agg_config.get("max_source_share", 0.5))
        max_signal_share = float(agg_config.get("max_signal_share", 0.5))
        scores, budget_info = apply_budget_caps(
            scores, task_ids, signals["source_ids"],
            max_source_share, max_signal_share,
        )
        entropy_reg = float(agg_config.get("entropy_regularization", 0.1))
    else:
        print(f"  [Aggregation] Unknown mode '{mode}'. Falling back to robust_weighted.")
        scores = _aggregate_robust_weighted(signals, weights)

    # --- Sample ---
    selected_task_ids, probs = sample_curriculum(
        scores, task_ids, n, temperature, entropy_reg,
    )

    # --- Compute diagnostics ---
    curriculum_entropy = compute_curriculum_entropy(scores, temperature)

    # Source distribution of selected tasks
    selected_sources = []
    for tid in selected_task_ids:
        try:
            node_data = gen_manager.archive.graph.nodes[tid]
        except (KeyError, AttributeError):
            node_data = {}
        src = node_data.get("type", node_data.get("status", "unknown"))
        selected_sources.append(src)
    unique_selected_sources, selected_source_counts = np.unique(selected_sources, return_counts=True)
    source_share = dict(zip(unique_selected_sources, selected_source_counts / max(1, len(selected_task_ids))))

    # Signal share
    max_signal_idx = np.argmax(scores) if len(scores) > 0 else -1
    signal_share = {}
    if len(scores) > 0 and len(task_ids) > 0:
        top_scores = np.sort(scores)[-min(3, len(scores)):]
        signal_share = {
            "max_score": float(scores[max_signal_idx]) if max_signal_idx >= 0 else 0.0,
            "top3_mean": float(top_scores.mean()) if len(top_scores) > 0 else 0.0,
            "score_std": float(scores.std()),
        }

    # --- Log diagnostics ---
    diagnostics = {
        "timestamp": time.time(),
        "run_name": run_name,
        "seed": seed,
        "global_step": global_step,
        "aggregation_mode": mode,
        "retention_trigger": retention_trigger,
        "forgetting_index": forgetting_stats["forgetting_index"],
        "best_old_success": forgetting_stats["best_old_success"],
        "current_old_success": forgetting_stats["current_old_success"],
        "anti_forgetting_mode": anti_forgetting,
        "weights": weights,
        "num_candidates": num_active,
        "selected_task_ids": selected_task_ids,
        "selected_task_sources": selected_sources,
        "signal_share": signal_share,
        "source_share": source_share,
        "skill_share": {},
        "curriculum_entropy": curriculum_entropy,
        "temperature": temperature,
        "entropy_regularization": entropy_reg,
        "llm_enabled": use_llm_cache,
        "llm_cache_hit_rate": (llm_cache_hits / max(1, llm_cache_hits + llm_cache_misses)),
        "llm_cache_hits": llm_cache_hits,
        "llm_cache_misses": llm_cache_misses,
        "warnings": [],
    }

    # Write diagnostics
    _write_diagnostics(diagnostics, log_dir)

    print(f"  [Aggregation] Selected {len(selected_task_ids)} tasks. "
          f"Entropy={curriculum_entropy:.3f}, "
          f"AntiForgetting={anti_forgetting}")

    return selected_task_ids


def _write_diagnostics(diagnostics: dict, log_dir: Optional[str] = None) -> None:
    """Write a single diagnostics row to the JSONL log file.

    Writes to two locations:
    1. The cwd-relative mechanism_logs/ (Hydra output dir)
    2. An absolute path for easy post-hoc analysis.
    """
    if log_dir is None:
        log_dir = "mechanism_logs"

    # Write to cwd-relative path (Hydra output directory)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "aggregation_selector.jsonl")

    _append_jsonl(log_path, diagnostics)

    # Also write to absolute path for sweep summarization
    abs_log_dir = "/root/experiments/dicode_runs/aggregation/mechanism_logs"
    os.makedirs(abs_log_dir, exist_ok=True)
    abs_log_path = os.path.join(abs_log_dir, "aggregation_selector.jsonl")
    _append_jsonl(abs_log_path, diagnostics)


def _append_jsonl(log_path: str, diagnostics: dict) -> None:

    # Convert numpy types to native Python for JSON serialization
    serializable = {}
    for key, value in diagnostics.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        elif isinstance(value, np.integer):
            serializable[key] = int(value)
        elif isinstance(value, dict):
            serializable[key] = {
                str(k): (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in value.items()
            }
        else:
            serializable[key] = value

    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(serializable) + "\n")
    except Exception as e:
        print(f"  [Aggregation] Warning: Could not write diagnostics: {e}")
