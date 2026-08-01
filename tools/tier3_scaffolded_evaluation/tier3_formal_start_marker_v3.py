#!/usr/bin/env python3
"""CC4 Tier3 — V3 repair-authorization start marker.

Sole producer of <pool>/cc4/V3_REPAIR_AUTHORIZATION.json (+ .sha256 sidecar), the
start-authorization record the V3 formal driver (tier3_formal_evaluation_v3.py)
verifies BEFORE any V3 run. It records, verbatim and auditably, the 总控 ruling
CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_FORMAL_EVALUATION_V3:
  * the ruling task name and verdict;
  * the authorization SCOPE (repair the frozen classifier's NEG20 handling of the
    legitimate composite event "floor transition AND defeat_kobold"; new
    representation primary_outcome + secondary_events[] + taxonomy_status);
  * the PROHIBITIONS (no retraining, no checkpoint change, no candidate-level
    exemption, no sub-metric ranking, no V2 evidence delete/overwrite/rewrite,
    no merge/rebase/amend/force-push);
  * the V2 archive evidence SHAs (V2_STATUS = CLOSED_INCONCLUSIVE_PARTICIPATION,
    V2_WINNER = null) — referenced for continuity, never modified;
  * the frozen common/bank pin snapshot (identical to V2DT), the V3 taxonomy
    module LF-SHA, and the git HEAD at recording time.

Refuses to overwrite an existing marker (the record is append-only in time; a
second attempt fails closed). Independent of the V2 SECONDARY_AUDIT marker and the
already-flipped V2 READY.

Usage (server or local, CWD = repo root):
  python tools/tier3_scaffolded_evaluation/tier3_formal_start_marker_v3.py \
      --pool-cc4-dir /home/oseasy/student_pool_v1/cc4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj                            # noqa: E402
import tier3_projection_binding_smoke_v2 as smokev2                # noqa: E402
import tier3_evaluation_certificate_v3 as certmod                  # noqa: E402
import tier3_taxonomy_v3 as taxonomy_v3                            # noqa: E402
import tier3_formal_evaluation_v3 as driver                        # noqa: E402

# Canonical marker constants live in the driver (single source of truth); this
# tool produces exactly the shape driver.verify_v3_repair_start accepts.
V3_REPAIR_MARKER_NAME = driver.V3_REPAIR_MARKER_NAME
V3_REPAIR_MARKER_SCHEMA = driver.V3_REPAIR_MARKER_SCHEMA
V3_REPAIR_RULING_TASK = driver.V3_REPAIR_RULING_TASK
V3_REPAIR_VERDICT = driver.V3_REPAIR_VERDICT
V2_ARCHIVE_SUMMARY_SHA256 = driver.V2_ARCHIVE_SUMMARY_SHA256
V2_ARCHIVE_GATE_SHA256 = driver.V2_ARCHIVE_GATE_SHA256

# 总控 ruling, recorded verbatim for audit.
RULING_VERBATIM = (
    "维持 Student、checkpoint、runtime、state bank、FULL seeds、episode 数、horizon、"
    "greedy policy 和排名规则不变。授权修复冻结失败分类器 NEG20 对合法复合事件 "
    "floor transition AND defeat_kobold 无法赋单一标签的问题。不得重训 Student，"
    "不得采用候选级豁免，不得降级为子指标排名。V2 归档：FORMAL_GLOBAL_PERFORMANCE_"
    "EVALUATION=CLOSED_INCONCLUSIVE_PARTICIPATION，V2_WINNER=null，V2_STUDENT_"
    "RANKING_VALID=false。不得删除/覆盖/改写 V2 证据。V3 = 新的可审计 evaluator "
    "semantic repair。复合事件语义：primary_outcome / secondary_events[] / "
    "taxonomy_status；FRONT 过渡成功为 primary，同时 defeat_kobold 为 secondary，"
    "taxonomy_status=VALID_COMPOSITE_EVENT，不得 FailClosed；BACK/FULL 以 "
    "DEFEAT_KOBOLD 为 primary。FailClosed 只保留：状态损坏/必需字段缺失/互相矛盾且"
    "无法由事件时序解释的数据/非法值/证据哈希不匹配/未注册事件类型。新协议 "
    "FORMAL_EVALUATOR_PROTOCOL=V3_COMPOSITE_EVENT，NEG20_PROTOCOL=NEG20_V3_PRIMARY_"
    "SECONDARY_EVENTS。排名仍按冻结顺序，四级全平 INCONCLUSIVE。禁止：重训/改 "
    "checkpoint/候选级豁免/删 CONTROL/只排 5 个/FULL-only 或 BACK-only 排名/改 bank/"
    "改 seeds/改 episode 数/改 horizon/性能重试/smoke 替代/覆盖 V2 证据/force push/"
    "rebase/amend/merge。"
)
RULING_PROHIBITIONS = [
    "NO_RETRAINING",
    "NO_CHECKPOINT_MODIFICATION",
    "NO_CANDIDATE_LEVEL_EXEMPTION",
    "NO_SUBMETRIC_RANKING_DOWNGRADE",
    "NO_V2_EVIDENCE_DELETE_OVERWRITE_REWRITE",
    "NO_MERGE_REBASE_AMEND_FORCE_PUSH",
    "NO_PERFORMANCE_RETRY",
    "NO_SMOKE_SUBSTITUTION_FOR_PERFORMANCE",
    "NO_FULL_ONLY_OR_BACK_ONLY_RANKING",
]


def _git_head(repo_root):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as exc:
        proj.require(False, "FAIL CLOSED (V3_MARKER): cannot resolve git HEAD at "
                            "%s: %r" % (repo_root, exc))


def build_marker(pool_cc4_dir, recorded_at_utc, git_head):
    return {
        "schema": V3_REPAIR_MARKER_SCHEMA,
        "ruling_task": V3_REPAIR_RULING_TASK,
        "verdict": V3_REPAIR_VERDICT,
        "pool_cc4_dir": pool_cc4_dir,
        "recorded_at_utc": recorded_at_utc,
        "ruling": {
            "task_name": V3_REPAIR_RULING_TASK,
            "verbatim": RULING_VERBATIM,
            "authorization_scope": "repair the frozen failure classifier's NEG20 "
                "handling of the legitimate composite event (floor transition AND "
                "defeat_kobold) and complete the formal evaluation as V3",
            "new_representation": "primary_outcome + secondary_events[] + "
                "taxonomy_status",
            "new_protocols": {
                "formal_evaluator_protocol":
                    taxonomy_v3.FORMAL_EVALUATOR_PROTOCOL,
                "neg20_protocol": taxonomy_v3.NEG20_PROTOCOL,
            },
            "prohibitions": list(RULING_PROHIBITIONS),
            "ranking_rule": "frozen order [full success_count, front_l2 "
                "transition_count, front_l2 mean graph_distance_progress, back_l2 "
                "defeat_count], tie tolerance 1e-12, full four-level tie => "
                "INCONCLUSIVE",
        },
        "evidence": {
            "v2_archive_summary_sha256": V2_ARCHIVE_SUMMARY_SHA256,
            "v2_archive_gate_sha256": V2_ARCHIVE_GATE_SHA256,
            "v2_status": certmod.V2_ARCHIVE_STATUS,
            "v2_winner": certmod.V2_ARCHIVE_WINNER,
            "v2_evidence_modified_by_v3": False,
            "pins_snapshot": certmod.pins_snapshot(),
            "taxonomy_v3_lf_sha256": taxonomy_v3.module_lf_sha256(),
            "git_commit_head": git_head,
        },
    }


def write_marker(pool_cc4_dir, recorded_at_utc=None, git_head=None):
    """Write the V3 repair-authorization marker + SHA sidecar. Refuses to
    overwrite an existing marker (fail-closed). Returns the marker sha."""
    os.makedirs(pool_cc4_dir, exist_ok=True)
    marker_path = os.path.join(pool_cc4_dir, V3_REPAIR_MARKER_NAME)
    proj.require(not os.path.exists(marker_path),
                 "FAIL CLOSED (V3_MARKER): %s already exists — refusing to "
                 "overwrite the start-authorization record" % marker_path)
    proj.require(not os.path.exists(marker_path + ".sha256"),
                 "FAIL CLOSED (V3_MARKER): sidecar already exists — refusing to "
                 "overwrite")
    if recorded_at_utc is None:
        recorded_at_utc = smokev2.utc_now_iso()
    if git_head is None:
        repo_root = os.path.dirname(os.path.dirname(HERE))
        git_head = _git_head(repo_root)
    marker = build_marker(pool_cc4_dir, recorded_at_utc, git_head)
    smokev2.write_json(marker_path, marker)
    sha = proj.sha256_file(marker_path)
    with open(marker_path + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("%s  %s\n" % (sha, V3_REPAIR_MARKER_NAME))
    return sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-cc4-dir", default="/home/oseasy/student_pool_v1/cc4")
    ap.add_argument("--recorded-at-utc", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    sha = write_marker(args.pool_cc4_dir, recorded_at_utc=args.recorded_at_utc)
    print("V3_REPAIR_AUTHORIZATION_MARKER_WRITTEN sha256=%s path=%s"
          % (sha, os.path.join(args.pool_cc4_dir, V3_REPAIR_MARKER_NAME)))
    return 0


def run_self_test():
    import tempfile
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        proj.require(cond, "V3_MARKER_SELF_TEST FAIL: %s" % msg)

    with tempfile.TemporaryDirectory() as td:
        common = os.path.join(td, "common_v2")
        cc4d = os.path.join(td, "cc4")
        os.makedirs(common)
        sha = write_marker(cc4d, recorded_at_utc="1970-01-01T00:00:00+00:00",
                           git_head="0" * 40)
        ok(len(sha) == 64, "marker sha hex64")
        ok(os.path.isfile(os.path.join(cc4d, V3_REPAIR_MARKER_NAME)), "marker file")
        ok(os.path.isfile(os.path.join(cc4d, V3_REPAIR_MARKER_NAME + ".sha256")),
           "sidecar file")
        # the driver's start gate must accept exactly what this tool writes
        ref = driver.verify_v3_repair_start(common, cc4d)
        ok(ref["sha256"] == sha, "driver verifies produced marker sha")
        ok(ref["ruling_task"] == V3_REPAIR_RULING_TASK, "ruling task verbatim")
        ok(ref["verdict"] == V3_REPAIR_VERDICT, "verdict verbatim")
        # marker content: ruling + prohibitions + V2 archive + pins recorded
        m = proj.read_json(os.path.join(cc4d, V3_REPAIR_MARKER_NAME))
        ok(m["ruling"]["verbatim"] == RULING_VERBATIM, "verbatim recorded")
        ok("NO_RETRAINING" in m["ruling"]["prohibitions"], "prohibitions recorded")
        ok(m["evidence"]["v2_status"] == "CLOSED_INCONCLUSIVE_PARTICIPATION",
           "v2 archive status recorded")
        ok(m["evidence"]["v2_evidence_modified_by_v3"] is False,
           "v2-untouched flag recorded")
        ok(m["evidence"]["pins_snapshot"] == certmod.pins_snapshot(),
           "pins snapshot recorded")
        ok(m["evidence"]["taxonomy_v3_lf_sha256"] == taxonomy_v3.module_lf_sha256(),
           "taxonomy LF-SHA recorded")
        # refuse-overwrite
        try:
            write_marker(cc4d, recorded_at_utc="1970-01-01T00:00:00+00:00",
                         git_head="0" * 40)
            ok(False, "overwrite accepted")
        except proj.FailClosed as exc:
            ok("refusing to overwrite" in str(exc), "refuse overwrite fires")

    print("V3_MARKER_SELF_TEST_PASS checks=%d" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
