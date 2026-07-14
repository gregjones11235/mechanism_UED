"""DiCode aggregation mechanisms for curriculum selection.

This package implements multiple aggregation strategies that combine
heterogeneous curriculum signals to select tasks for training:

- robust_normalize: Median/IQR normalization with outlier clipping
- Raw weighted scoring
- Robust weighted scoring
- Soft Copeland pairwise aggregation
- Budgeted allocation (anti-monopoly)
- Entropy-regularized softmax sampling
- Anti-forgetting retention trigger
"""

from dicode.mechanisms.aggregation import (
    apply_budget_caps,
    compute_curriculum_entropy,
    compute_forgetting_stats,
    compute_signal_scores,
    robust_normalize,
    sample_curriculum,
    select_tasks_with_aggregation,
)

__all__ = [
    "robust_normalize",
    "compute_curriculum_entropy",
    "compute_forgetting_stats",
    "compute_signal_scores",
    "apply_budget_caps",
    "sample_curriculum",
    "select_tasks_with_aggregation",
]
