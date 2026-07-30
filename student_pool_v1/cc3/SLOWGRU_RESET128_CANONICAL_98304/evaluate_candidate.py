#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 candidate evaluation entry: SLOWGRU_RESET128_CANONICAL_98304.

BOUNDARY (task section 七, still in force): formal metrics — graph_distance_progress,
transition_count, defeat_count, and the official ordering rules — are OWNED by the
CC4 common evaluator + frozen FRONT/BACK banks. This script deliberately implements
NONE of them. It only validates the CLI contract and refuses to run until the CC4
common contract is handed off (formal_eval_binding=WAITING_CC4_COMMON_CONTRACT).

Accepted CLI (per contract):
  --candidate-manifest --common-evaluator --front-bank --back-bank --profile
  --output-dir --gpu-uuid
"""
import argparse
import json
import os
import sys

FORMAL_EVAL_BINDING = "WAITING_CC4_COMMON_CONTRACT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-manifest", required=True)
    ap.add_argument("--common-evaluator", required=True)
    ap.add_argument("--front-bank", required=True)
    ap.add_argument("--back-bank", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gpu-uuid", required=True)
    args = ap.parse_args()

    with open(args.candidate_manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    allowed = {"GPU-8df11537-ab79-722d-606f-411966196c4c",
               "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"}
    report = dict(
        candidate_id=manifest.get("candidate_id"),
        formal_eval_binding=FORMAL_EVAL_BINDING,
        status="REFUSED_WAITING_CC4_COMMON_CONTRACT",
        cli_contract_ok=True,
        gpu_uuid_allowed=args.gpu_uuid in allowed,
        reason="CC3 不实现自有正式评估指标/排序规则。formal ranking 必须由 CC4 公共 "
               "evaluator（commit SHA + runner + checkpoint contract + FRONT/BACK banks + "
               "frozen seed/state schedule + CLI template + self-test PASS）执行。"
               "见 blocker_ledger B1_CC4_EVALUATOR_HANDOFF / B3_FROZEN_STATE_BANK_HANDOFF。",
        implemented_metrics=[],
        short_smoke_reward_policy="REFERENCE_ONLY_NOT_PERFORMANCE_JUDGMENT",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "evaluate_candidate_status.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    sys.exit(3)  # explicit: NOT an evaluation failure — the binding is not available yet


if __name__ == "__main__":
    main()
