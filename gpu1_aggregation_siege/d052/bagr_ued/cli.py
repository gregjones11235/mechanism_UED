"""BA-BAGR-UED CLI (dry run).

Usage:
    python -m d052.bagr_ued.cli --reports-dir <dir>

Runs the synthetic unsafe_rest window through the full controller and writes
the section-16 artifacts. This round: ENGINEERING_DRY_RUN only — no training,
no formal evaluation, no real LLM calls, no push.
"""
from __future__ import annotations

import argparse
import json
import sys

from d052.bagr_ued import constants as C
from d052.bagr_ued.controller import BAGRUEdController
from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard)
from d052.bagr_ued.report_writer import finalize_sha256sums, write_all
from d052.bagr_ued.synthetic_traces import build_unsafe_rest_raw_rollout
from d052.bagr_ued.trajectory_supervision_guard import TrajectorySupervisionGuard


def build_guard_reports(result) -> dict:
    """Guard evidence: PASS over the real run + fail-closed demonstrations
    (demonstrations record CODES and labels only, never the offending text)."""
    dumped = result.model_dump()
    sup = TrajectorySupervisionGuard()
    leak = FormalEvaluationLeakageGuard()

    sup_run = sup.scan(dumped, label="full_dry_run_result")

    # demonstration 1: a supervision key in an output fails closed
    demo_key = sup.scan({"recommended_actions": ["<redacted>"]},
                        label="demo_supervision_key")
    # demonstration 2: direct action advice fails closed
    demo_advice = sup.scan({"note": "don't sleep near monsters"},
                           label="demo_action_advice")
    # demonstration 3: formal FRONT provenance fails closed
    demo_front = leak.scan({"provenance": {"source": C.SOURCE_FORMAL_FRONT}},
                           label="demo_formal_front")
    # demonstration 4: generative-training source is ALLOWED
    demo_allowed = leak.scan(
        {"provenance": {"source": C.SOURCE_GENERATIVE_TRAINING_ENV}},
        label="demo_generative_training")

    def redact(report: dict) -> dict:
        out = dict(report)
        out["findings"] = [dict(code=f["code"], path=f["path"],
                                detail=f["detail"].split(" matched ")[0])
                           for f in report.get("findings", [])]
        return out

    return dict(
        supervision=dict(
            guard="TrajectorySupervisionGuard",
            full_dry_run_result=redact(sup_run),
            fail_closed_demonstrations=dict(
                supervision_key=redact(demo_key),
                direct_action_advice=redact(demo_advice)),
            forbidden_supervision_keys=sorted(C.FORBIDDEN_SUPERVISION_KEYS)),
        leakage=dict(
            guard="FormalEvaluationLeakageGuard",
            board_input_status=result.model_dump()["board"][
                "leakage_guard_status"],
            fail_closed_demonstrations=dict(formal_front=redact(demo_front)),
            allowed_demonstrations=dict(generative_training=redact(demo_allowed)),
            forbidden_sources=sorted(C.FORBIDDEN_EVIDENCE_SOURCES),
            allowed_sources=sorted(C.ALLOWED_EVIDENCE_SOURCES)),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d052.bagr_ued")
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--bundle-id", default="synthetic_unsafe_rest_window")
    args = ap.parse_args(argv)

    controller = BAGRUEdController()
    raw_rollout = build_unsafe_rest_raw_rollout()
    result = controller.run_dry_run(raw_rollout, bundle_id=args.bundle_id)
    guard_reports = build_guard_reports(result)
    sums = write_all(result, raw_rollout, args.reports_dir,
                     guard_reports=guard_reports)
    finalize_sha256sums(args.reports_dir)  # re-hash after any rewrite

    cert = result.dry_run_certificate
    print(json.dumps(dict(
        status="DRY_RUN_COMPLETE",
        reports_dir=args.reports_dir,
        artifacts=len(sums),
        run_class=cert["run_class"],
        real_llm_calls=cert["real_llm_calls"],
        ued_slots=cert["ued_slots_allocated"],
        anchor_slots=cert["anchor_slots_allocated"],
        supervision_guard=sup_ok(sup_status(result)),
    ), sort_keys=True))
    return 0


def sup_status(result) -> str:
    return result.model_dump()["board"]["supervision_guard_status"]


def sup_ok(status: str) -> bool:
    return status == "PASS"


if __name__ == "__main__":
    sys.exit(main())
