"""Auction-based curriculum aggregation mechanism.

Implements a multi-round auction where curriculum roles (Tutor, Critic,
Explorer) bid for task slots using virtual budgets. The aggregation
mechanism makes the final decision — no single LLM directly chooses
the final task set.

Auction types:
  - auction_raw: Standard first-price auction, each role bids utility
  - auction_budgeted: Budget-constrained auction with per-role caps

All functions are pure and operate on NumPy arrays for testability.
"""

import numpy as np
from typing import Optional


# ==============================================================================
# Auction Mechanism
# ==============================================================================


def run_auction_selection(
    candidate_ids: list[str],
    role_utilities: dict[str, np.ndarray],
    n_selected: int = 8,
    auction_type: str = "raw",
    role_budgets: Optional[dict[str, float]] = None,
    reserve_price: float = 0.0,
    seed: int = 0,
) -> dict:
    """Run a multi-round auction to select curriculum tasks.

    Each role bids its utility for each candidate task. Tasks are
    allocated round-by-round to the highest bidder. Budget constraints
    are enforced in budgeted mode.

    The aggregation mechanism makes the final decision — no role's
    bid alone determines the outcome.

    Args:
        candidate_ids: List of candidate task IDs.
        role_utilities: Dict mapping role name -> 1-D array of utilities
            (one per candidate). Utilities should be in [0, 1].
        n_selected: Number of tasks to select (default 8).
        auction_type: 'raw' (no budget) or 'budgeted' (budget-constrained).
        role_budgets: Dict mapping role name -> budget cap. Required for
            'budgeted' type. Budget is consumed as tasks are allocated.
        reserve_price: Minimum bid to be considered (default 0.0).
        seed: Random seed for tie-breaking.

    Returns:
        Dict with:
          - selected_ids: list of selected task IDs
          - winning_bids: dict mapping selected task_id -> (winning_role, bid_amount)
          - per_role_allocation: dict mapping role -> list of won task IDs
          - budget_state: dict mapping role -> remaining budget (budgeted only)
          - auction_rounds: number of rounds run
          - budget_changed_selection: bool (budgeted only)
          - total_utility: sum of winning bids
    """
    n = len(candidate_ids)
    if n == 0 or n_selected <= 0:
        return {
            "selected_ids": [],
            "winning_bids": {},
            "per_role_allocation": {},
            "budget_state": {},
            "auction_rounds": 0,
            "budget_changed_selection": False,
            "total_utility": 0.0,
        }

    k = min(n_selected, n)
    roles = list(role_utilities.keys())

    if not roles:
        return {
            "selected_ids": [],
            "winning_bids": {},
            "per_role_allocation": {},
            "budget_state": {},
            "auction_rounds": 0,
            "budget_changed_selection": False,
            "total_utility": 0.0,
        }

    # Initialize state
    rng = np.random.default_rng(seed)
    available = set(range(n))  # Indices still available
    selected = []
    winning_bids = {}
    per_role_allocation = {role: [] for role in roles}

    # Budget tracking
    budgets = {}
    if auction_type == "budgeted" and role_budgets:
        budgets = dict(role_budgets)
    initial_budgets = dict(budgets)

    # Run k rounds of auction
    for round_idx in range(k):
        best_bid = -np.inf
        best_candidate = -1
        best_role = ""

        for role in roles:
            utilities = role_utilities[role]
            for idx in available:
                bid = float(utilities[idx])

                # Apply budget constraint
                if auction_type == "budgeted" and role in budgets:
                    if budgets[role] < bid:
                        continue  # Can't afford this bid

                if bid > best_bid and bid >= reserve_price:
                    best_bid = bid
                    best_candidate = idx
                    best_role = role
                elif bid == best_bid and bid >= reserve_price:
                    # Tie-breaking: random choice
                    if rng.random() > 0.5:
                        best_bid = bid
                        best_candidate = idx
                        best_role = role

        if best_candidate < 0:
            break  # No valid bids remaining

        # Allocate task
        task_id = candidate_ids[best_candidate]
        selected.append(task_id)
        winning_bids[task_id] = (best_role, best_bid)
        per_role_allocation[best_role].append(task_id)
        available.remove(best_candidate)

        # Deduct budget
        if auction_type == "budgeted" and best_role in budgets:
            budgets[best_role] -= best_bid

    # Compute budget effect
    budget_changed_selection = False
    if auction_type == "budgeted" and initial_budgets:
        # Rerun without budgets to compare
        raw_result = run_auction_selection(
            candidate_ids=candidate_ids,
            role_utilities=role_utilities,
            n_selected=n_selected,
            auction_type="raw",
            seed=seed,
        )
        budget_changed_selection = (
            set(selected) != set(raw_result["selected_ids"])
        )

    return {
        "selected_ids": selected,
        "winning_bids": winning_bids,
        "per_role_allocation": per_role_allocation,
        "budget_state": budgets if auction_type == "budgeted" else {},
        "auction_rounds": len(selected),
        "budget_changed_selection": budget_changed_selection,
        "total_utility": float(sum(b for _, b in winning_bids.values())),
    }


# ==============================================================================
# Utility construction from signals
# ==============================================================================


def build_role_utilities_from_signals(
    signals: dict[str, np.ndarray],
    role_weights: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, np.ndarray]:
    """Build per-role utility vectors from curriculum signals.

    Each role sees a different combination of signals:
      - Tutor: progression + learnability (positive)
      - Critic: penalty + risk (negative — converted to positive utility)
      - Explorer: novelty + diversity (positive)

    Args:
        signals: Dict mapping signal name -> 1-D array.
        role_weights: Optional per-role signal weights.

    Returns:
        Dict mapping role name -> 1-D utility array in [0, 1].
    """
    if role_weights is None:
        role_weights = {
            "tutor": {"progression": 0.7, "retention": 0.3},
            "critic": {"critic_penalty_inv": 0.6, "novelty": 0.4},
            "explorer": {"novelty": 0.8, "retention": 0.2},
        }

    n = len(signals.get("progression", np.array([])))
    if n == 0:
        return {}

    utilities = {}

    for role, weights in role_weights.items():
        util = np.zeros(n, dtype=np.float64)

        for signal_name, weight in weights.items():
            if signal_name == "critic_penalty_inv":
                # Invert critic penalty: high penalty = low utility
                penalty = signals.get("critic_penalty", np.zeros(n))
                util += weight * (1.0 - penalty)
            elif signal_name in signals:
                util += weight * signals[signal_name]

        # Normalize to [0, 1]
        u_min = util.min()
        u_max = util.max()
        if u_max - u_min > 1e-8:
            util = (util - u_min) / (u_max - u_min)
        else:
            util = np.ones(n) * 0.5

        utilities[role] = util

    return utilities


# ==============================================================================
# Budgeted Soft Copeland (proper implementation)
# ==============================================================================


def run_budgeted_copeland_selection(
    candidate_ids: list[str],
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
    n_selected: int = 8,
    source_ids: Optional[np.ndarray] = None,
    max_source_share: float = 0.5,
    temperature: float = 1.0,
) -> dict:
    """Run Soft Copeland with budget caps and explicit selection-effect logging.

    This is the PROPER implementation of budgeted Soft Copeland that
    explicitly logs whether budget activation changed the selected set.

    Args:
        candidate_ids: List of task IDs.
        signals: Dict of signal arrays.
        weights: Signal weights dict.
        n_selected: Number of tasks to select.
        source_ids: Array of source type strings for budget caps.
        max_source_share: Max fraction from one source.
        temperature: Softmax temperature.

    Returns:
        Dict with selection results and budget diagnostics.
    """
    from dicode.mechanisms.aggregation import (
        _aggregate_soft_copeland,
        apply_budget_caps,
    )

    n = len(candidate_ids)
    if source_ids is None:
        source_ids = signals.get("source_ids", np.array(["unknown"] * n))

    # Step 1: Soft Copeland without budget
    scores_no_budget = _aggregate_soft_copeland(signals, weights, temperature)
    top_k_no_budget = set(
        np.array(candidate_ids)[np.argsort(-scores_no_budget)[:n_selected]]
    )

    # Step 2: Apply budget caps
    scores_budgeted, budget_info = apply_budget_caps(
        scores_no_budget.copy(), candidate_ids, source_ids,
        max_source_share=max_source_share,
    )

    # Step 3: Soft Copeland with budgeted scores
    top_k_budgeted = set(
        np.array(candidate_ids)[np.argsort(-scores_budgeted)[:n_selected]]
    )

    # Step 4: Determine budget effect
    budget_changed_selection = top_k_no_budget != top_k_budgeted
    tasks_swapped = top_k_no_budget.symmetric_difference(top_k_budgeted)

    return {
        "selected_ids": sorted(top_k_budgeted),
        "selected_ids_no_budget": sorted(top_k_no_budget),
        "budget_changed_selection": budget_changed_selection,
        "tasks_swapped": sorted(tasks_swapped),
        "budget_info": budget_info,
        "scores_no_budget": scores_no_budget,
        "scores_budgeted": scores_budgeted,
    }
