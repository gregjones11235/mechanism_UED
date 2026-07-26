"""Authorization-gated evaluation adapter (canonical_v2).

Builds a deterministic held-out evaluation plan from an execution-mapping
certificate; runs no evaluation this phase (no training => no results). Enforces
NO_RAW_DATA_NO_STRONG_CLAIM and keeps RESULTS_REUSABILITY=ENGINEERING_ONLY until
raw data exists.
"""
from d052.evaluation.adapter import (
    EvaluationAdapterError,
    assert_no_strong_claim,
    attach_results,
    build_evaluation_plan,
)

__all__ = [
    "EvaluationAdapterError",
    "assert_no_strong_claim",
    "attach_results",
    "build_evaluation_plan",
]
