"""Authorization-gated evaluation adapter (canonical_v2).

Builds a DETERMINISTIC held-out evaluation PLAN from an execution-mapping
certificate (which achievements to measure, metric=success_rate). This phase runs
no training and therefore produces NO results; the adapter enforces
NO_RAW_DATA_NO_STRONG_CLAIM: a report with no raw data may not carry a strong
claim, and RESULTS_REUSABILITY stays ENGINEERING_ONLY until raw data exists.
"""
from __future__ import annotations

from typing import Optional

from d052.schemas.execution import ExecutionMappingCertificate


class EvaluationAdapterError(Exception):
    NO_RAW_DATA_NO_STRONG_CLAIM = "NO_RAW_DATA_NO_STRONG_CLAIM"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def build_evaluation_plan(cert: ExecutionMappingCertificate) -> dict:
    """Deterministic held-out evaluation plan from a certificate. No results."""
    return {
        "candidate_id": cert.candidate_id,
        "chash": cert.chash,
        "canonical_names": list(cert.canonical_names),
        "canonical_ids": list(cert.canonical_ids),
        "metric": "success_rate",
        "evidence_source": "held_out_evaluation",
        "num_achievements": len(cert.canonical_names),
        "conditioning_type": cert.conditioning_type,
        "student_obs_dim": cert.student_obs_dim,
        # NO results this phase:
        "results": None,
        "strong_claim_permitted": False,
        "RESULTS_REUSABILITY": "ENGINEERING_ONLY",
    }


def assert_no_strong_claim(report: dict) -> None:
    """Fail-closed: a report without raw results may not make a strong claim."""
    if report.get("results") is None and report.get("strong_claim_permitted"):
        raise EvaluationAdapterError(
            EvaluationAdapterError.NO_RAW_DATA_NO_STRONG_CLAIM,
            "report has no raw results but strong_claim_permitted=True "
            "(NO_RAW_DATA_NO_STRONG_CLAIM)")


def attach_results(report: dict, results: dict) -> dict:
    """Attach raw held-out results, permitting strong claims ONLY with raw data.

    Provided for a future authorized phase; this phase never calls it. It exists
    to make the no-raw-data gate explicit and testable.
    """
    if not results:
        raise EvaluationAdapterError(
            EvaluationAdapterError.NO_RAW_DATA_NO_STRONG_CLAIM,
            "cannot attach empty results")
    out = dict(report)
    out["results"] = results
    out["strong_claim_permitted"] = True
    out["RESULTS_REUSABILITY"] = "SCIENTIFIC_WITH_EVIDENCE"
    assert_no_strong_claim(out)
    return out
