#!/usr/bin/env python3
"""CC4 正式全局性能评估证据 — 独立离线复验器（JAX-free，2026-08-01）。

对本证据目录（自服务器 /home/oseasy/student_pool_v1 原样回传，tarball sha
07f5c018…）做独立复验：全部证据文件重哈希、marker/preflight/READY 链、
7 份 bundle 结构与诚实标签、BLOCKED 结构化中止证据、日志证据，并**从原始
evaluation result 独立重算排名**与发布的 FORMAL_RANKING_SUMMARY_V2DT.json
比对（不复用排名工具代码）。npz/pkl 永不出服务器，此目录不应包含任何
权重文件（本复验器主动断言）。

用法：python verify_formal_evaluation_evidence.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC4 = os.path.join(HERE, "cc4")
COMMON = os.path.join(HERE, "common_v2")
LOGS = os.path.join(CC4, "formal_eval_logs")

# --- frozen pins（来自冻结公共评测器 / 审计记录；与被验证据交叉核对）-------
PINS = {
    "POOL_BINDING_GATE_V2DT_SHA256":
        "cec167117a7aa8e67a3d5eb60839e711e72d950135553e4035a87e6c9859a352",
    "MARKER_SHA256":
        "b08c1a9bf7055ac6b4a200c6f561374a0c34c2dee9ce6dd799212de3eb5f8351",
    "SUMMARY_SHA256":
        "3e8186417aefeb25729324ce5fb4bc6b56a58087c8d1ee67bc088ad37d5c1ac3",
    "GATE_SHA256":
        "51d3d6fb8efbc978875823cdc4576443c4d61f308840462c1bfa12da52fddc5b",
    "PREFLIGHT_SHA256":
        "5c23a8fccb3a61ffb3fdfa7be83c7eca24e7b3eff2e6676dcad85cc5eda29f7c",
    "COMMON_EVALUATOR_SHA256":
        "2978a0f625bc94e18c99649959e8c090f964cd66e5dafd6b93245f144a317037",
    "MARKER_HEAD": "b0d7e9237a9e096c53665c523ce9e04df19bbff6",
    "EXECUTION_HEAD": "6f5e270559b67c63ea55d733f9606283c52dad2f",
    "CLOSING_HEAD": "8d46bd30770d05c5eee11c9c58592e928fdd152d",
    "MARKER_RECORDED_AT_UTC": "2026-07-31T10:09:36.565527+00:00",
    "AUDIT_VERDICT": "PASS_FOR_FORMAL_GLOBAL_PERFORMANCE_EVALUATION_START",
    "ABORT_VERDICT": "ENGINE_TAXONOMY_REJECTED_FORMAL_EVALUATION_V2",
}
STUDENTS = [
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "RESET128_RMT16_ORIGINAL_VTRACE_98304",
    "BASE_GTRXL_ORIGINAL_VTRACE_98304",
    "CONTROL_CONTINUOUS_98304",
    "SLOWGRU_RESET128_CANONICAL_98304",
    "SLOWGRU_PERSISTENT_CANONICAL_98304",
]
TEACHER = "BASELINE_TEACHER_CKPT17500"
ALL_CANDIDATES = STUDENTS + [TEACHER]
BLOCKED_IDS = sorted(set(ALL_CANDIDATES) - {"CONTROL_CONTINUOUS_98304"})
FROZEN_RULE_ORDER = [
    "full success_count",
    "front_l2 transition_count",
    "front_l2 mean graph_distance_progress",
    "back_l2 defeat_count",
]
EXPECTED_CONTROL_TUPLE = (0, 0, 0.4196479859579006, 7)

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise SystemExit("VERIFY FAIL (check %d): %s" % (CHECKS, msg))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def scenario_line_hash(raw_lines, scenario):
    lines = [ln for ln in raw_lines
             if json.loads(ln)["scenario"] == scenario]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def main():
    # 0. 权重隔离断言：证据目录不得含任何 npz/pkl/ckpt/orbax
    for root, _dirs, files in os.walk(HERE):
        for fn in files:
            ok(not fn.endswith((".npz", ".pkl", ".ckpt", ".orbax")),
               "weight file in evidence dir: %s" % os.path.join(root, fn))

    # 1. 发布文件与 sidecar
    for name, pin in (("FORMAL_RANKING_SUMMARY_V2DT.json",
                       PINS["SUMMARY_SHA256"]),
                      ("FORMAL_EVALUATION_GATE_V2DT.json",
                       PINS["GATE_SHA256"]),
                      ("SECONDARY_AUDIT_PASS.json", PINS["MARKER_SHA256"])):
        path = os.path.join(CC4, name)
        ok(sha256_file(path) == pin, "%s sha != pin" % name)
        side = open(path + ".sha256", encoding="utf-8").read().split()
        ok(side[0] == pin and side[1] == name, "%s sidecar mismatch" % name)

    # 2. marker 链
    marker = read_json(os.path.join(CC4, "SECONDARY_AUDIT_PASS.json"))
    ok(marker.get("verdict") == PINS["AUDIT_VERDICT"], "marker verdict")
    ok((marker.get("evidence") or {}).get("binding_gate_sha256")
       == PINS["POOL_BINDING_GATE_V2DT_SHA256"], "marker binding-gate sha")
    ok(marker.get("git_commit_head") == PINS["MARKER_HEAD"], "marker head")
    ok(marker.get("recorded_at_utc") == PINS["MARKER_RECORDED_AT_UTC"],
       "marker recorded_at")
    ok(sha256_file(os.path.join(CC4, "POOL_BINDING_GATE_V2DT.json"))
       == PINS["POOL_BINDING_GATE_V2DT_SHA256"], "binding gate file sha")

    # 3. 跨 GPU 确定性预检记录
    pre_path = os.path.join(CC4, "CROSS_GPU_DETERMINISM_PREFLIGHT.json")
    ok(sha256_file(pre_path) == PINS["PREFLIGHT_SHA256"], "preflight sha")
    pre = read_json(pre_path)
    ok("PASS" in str(pre.get("verdict")), "preflight verdict")
    ok(pre.get("checks") == 180, "preflight checks == 180")

    # 4. READY 收口翻转
    ready = read_json(os.path.join(COMMON, "COMMON_EVALUATOR_V2_READY.json"))
    ok(ready.get("COMMON_EVALUATOR_V2_READY") is True, "READY v2 true")
    ok(ready.get("FORMAL_RANKING_STARTED") is True
       and ready.get("FORMAL_RANKING_PUBLISHED") is True, "ranking flags")
    ok(ready.get("formal_ranking_summary_sha256") == PINS["SUMMARY_SHA256"],
       "READY summary sha")
    ok(ready.get("formal_evaluation_gate_sha256") == PINS["GATE_SHA256"],
       "READY gate sha")
    ok(ready.get("formal_evaluation_started_at_utc")
       == PINS["MARKER_RECORDED_AT_UTC"], "READY started_at == marker")
    ok(ready.get("pending_gates") == [], "pending_gates retired")
    ok((ready.get("secondary_audit_marker") or {}).get("sha256")
       == PINS["MARKER_SHA256"], "READY marker ref")

    # 5. 冻结规则逐字（来自回传的 metric_schema.json）
    schema = read_json(os.path.join(COMMON, "metric_schema.json"))
    spr = schema["selection_predicate_rule"]
    ok(spr["order"] == FROZEN_RULE_ORDER, "schema rule order verbatim")
    ok(spr["all_equal_result"] == "INCONCLUSIVE", "schema all_equal_result")

    # 6. 七份 bundle
    bundle_facts = {}
    common_pins_seen = set()
    for cid in ALL_CANDIDATES:
        d = os.path.join(CC4, cid, "formal_evaluation_v2dt")
        is_blocked = cid in BLOCKED_IDS
        tag = cid + ": "
        # 6a. SHA256SUMS_FORMAL_V2DT 重哈希（6 行）
        sums = open(os.path.join(d, "SHA256SUMS_FORMAL_V2DT"),
                    encoding="utf-8").read().splitlines()
        ok(len(sums) == 6, tag + "sums line count")
        for line in sums:
            h, fn = line.split("  ", 1)
            ok(sha256_file(os.path.join(d, fn)) == h, tag + "sums " + fn)
        # 6b. READY
        r = read_json(os.path.join(d, "READY_FORMAL_V2DT.json"))
        ok(r.get("candidate_id") == cid, tag + "READY candidate_id")
        ok(r.get("run_class") == "FORMAL_EVALUATION", tag + "run_class")
        ok(r.get("rehearsal") is False, tag + "not rehearsal")
        ok(r.get("student_rank") is None, tag + "READY rank null")
        ok(r.get("scientific_claim_authorized") is False, tag + "sci flag")
        ok(r.get("teacher_included_in_student_ranking") is False,
           tag + "teacher flag")
        gates = r.get("gates") or {}
        if is_blocked:
            false_gates = sorted(k for k, v in gates.items() if not v)
            ok(r.get("evaluation_status") == "BLOCKED", tag + "status BLOCKED")
            ok(false_gates == ["G4_FORMAL_SCHEDULE_COMPLETE"],
               tag + "only G4 false, got %s" % false_gates)
            ok(gates.get("G12_CERTIFICATE_VERIFIED") is True,
               tag + "G12 certificate verified")
            fa = r.get("formal_abort") or {}
            ok(fa.get("verdict") == PINS["ABORT_VERDICT"], tag + "abort verdict")
            ok(fa.get("aborted_phase") == "evaluate_classification",
               tag + "abort phase")
            ok(fa.get("scenario") == "front_l2", tag + "abort scenario")
            ok(fa.get("episodes_completed_before_abort") == 8,
               tag + "abort episodes completed")
            ok(fa.get("scenarios_completed_before_abort") == ["full",
                                                              "front_l2"],
               tag + "abort scenarios completed")
            ok("NEG20" in str(fa.get("engine_message")), tag + "NEG20 class")
        else:
            ok(r.get("evaluation_status") == "PASS", tag + "status PASS")
            ok(r.get("READY_FORMAL_V2DT") is True and all(gates.values()),
               tag + "all gates true")
            ok(r.get("formal_abort") is None, tag + "no abort")
        # 6c. certificate
        c = read_json(os.path.join(d, "evaluation_certificate_v2dt.json"))
        ok(c.get("run_class") == "FORMAL_EVALUATION", tag + "cert run_class")
        ok(c.get("student_rank") is None, tag + "cert rank null")
        for flag in ("scientific_claim_authorized",
                     "scaffolded_results_can_replace_full_task",
                     "interface_smoke_substituted_for_performance",
                     "teacher_included_in_student_ranking"):
            ok(c.get(flag) is False, tag + "cert " + flag)
        ok((c.get("provenance") or {}).get("git_commit_head")
           == PINS["EXECUTION_HEAD"], tag + "cert execution head")
        pins = c.get("common_pins") or {}
        ok(pins.get("common_evaluator_sha256")
           == PINS["COMMON_EVALUATOR_SHA256"], tag + "cert evaluator pin")
        common_pins_seen.add(json.dumps(pins, sort_keys=True))
        ok(((c.get("audit") or {}).get("secondary_audit_marker_sha256")
            == PINS["MARKER_SHA256"])
           or json.dumps(PINS["MARKER_SHA256"]) in json.dumps(c),
           tag + "cert marker reference")
        # 6d. per-scenario results + episode records
        raw = open(os.path.join(d, "episode_records.jsonl"),
                   encoding="utf-8").read().splitlines()
        results = {sc: read_json(os.path.join(
            d, "evaluation_result_v2dt.%s.json" % sc))
            for sc in ("full", "front_l2", "back_l2")}
        for sc in ("full", "front_l2", "back_l2"):
            res = results[sc]
            ok(res.get("candidate_id") == cid, tag + sc + " candidate_id")
            ok(res.get("run_class") == "FORMAL_EVALUATION",
               tag + sc + " run_class")
            ok(res.get("rehearsal") is False, tag + sc + " not rehearsal")
            n = len([ln for ln in raw if json.loads(ln)["scenario"] == sc])
            if sc == "full":
                ok(res.get("episodes_executed") == 64 and n == 64,
                   tag + "full 64 episodes")
                ok(res.get("aborted_in_scenario") is False
                   and res.get("evaluation") is not None,
                   tag + "full complete with evaluation")
            elif sc == "front_l2":
                if is_blocked:
                    ok(res.get("episodes_executed") == 8 and n == 8,
                       tag + "front 8 rollouts before abort")
                    ok(res.get("aborted_in_scenario") is True
                       and res.get("evaluation") is None,
                       tag + "front aborted, no evaluation")
                else:
                    ok(res.get("episodes_executed") == 8 and n == 8,
                       tag + "front 8 episodes")
                    ok(res.get("aborted_in_scenario") is False
                       and res.get("evaluation") is not None,
                       tag + "front complete")
                ok(res.get("episode_records_sha256")
                   == scenario_line_hash(raw, "front_l2"),
                   tag + "front episode_records_sha256 recompute")
            else:  # back_l2
                if is_blocked:
                    ok(res.get("episodes_executed") == 0 and n == 0,
                       tag + "back unreached (0) under frozen FULL→FRONT→BACK")
                    ok(res.get("evaluation") is None,
                       tag + "back no evaluation (abort evidence)")
                else:
                    ok(res.get("episodes_executed") == 8 and n == 8,
                       tag + "back 8 episodes")
                    ok(res.get("evaluation") is not None,
                       tag + "back complete")
            if sc != "back_l2" or not is_blocked:
                ok(res.get("episode_records_sha256")
                   == scenario_line_hash(raw, sc),
                   tag + sc + " episode_records_sha256 recompute")
        bundle_facts[cid] = {
            "blocked": is_blocked,
            "results": results,
        }
    ok(len(common_pins_seen) == 1, "common pins uniform across 7 bundles")

    # 7. 独立排名重算（原生实现，不导入排名工具）
    def extract_tuple(facts):
        ev = {sc: facts["results"][sc]["evaluation"]
              for sc in ("full", "front_l2", "back_l2")}
        m = {sc: ev[sc]["metrics"] for sc in ev}
        assert m["full"]["primary"]["valid_starts"] == 64
        assert m["front_l2"]["primary"]["valid_starts"] == 8
        assert m["back_l2"]["primary"]["valid_starts"] == 8
        back_defeat = int(m["back_l2"]["primary"]["successes"])
        assert back_defeat == int(
            m["back_l2"]["diagnostics"]["survival"]["defeat_count"])
        return (int(m["full"]["primary"]["successes"]),
                int(m["front_l2"]["primary"]["successes"]),
                float(m["front_l2"]["dense"]["value"]),
                back_defeat)

    def compare(a, b, tol=1e-12):
        for av, bv in zip(a, b):
            if av - bv > tol:
                return -1
            if bv - av > tol:
                return 1
        return 0

    ok(compare((11, 0, 0.0, 0), (10, 9, 1.0, 9)) == -1, "cmp level1")
    ok(compare((10, 5, 0.5 + 5e-13, 3), (10, 5, 0.5, 3)) == 0, "cmp tolerance")
    ok(compare((1, 2, 3.0, 4), (1, 2, 3.0, 5)) == 1, "cmp level4")

    eligible = {cid for cid in STUDENTS
                if not bundle_facts[cid]["blocked"]}
    ok(eligible == {"CONTROL_CONTINUOUS_98304"}, "eligible set == CONTROL")
    control_tuple = extract_tuple(bundle_facts["CONTROL_CONTINUOUS_98304"])
    ok(control_tuple == EXPECTED_CONTROL_TUPLE,
       "CONTROL tuple %r != expected %r" % (control_tuple,
                                            EXPECTED_CONTROL_TUPLE))
    recomputed_status = ("INCONCLUSIVE_PARTICIPATION"
                         if len(eligible) < 6 else "ORDERED")
    ok(recomputed_status == "INCONCLUSIVE_PARTICIPATION",
       "recomputed status inconclusive (<6 eligible)")

    # 8. 与发布的 summary 比对
    summary = read_json(os.path.join(CC4, "FORMAL_RANKING_SUMMARY_V2DT.json"))
    ok(summary["ranking_status"] == recomputed_status, "summary status match")
    ok(summary["student_count_eligible"] == "%d/6" % len(eligible),
       "summary eligible count")
    ok(summary["top_ranked_student_id"] is None, "no top ranked (<6)")
    ok(summary["inconclusive_groups"] == [], "no full-tie groups")
    ok(summary["teacher_included_in_student_ranking"] is False
       and summary["scientific_claim_authorized"] is False
       and summary["scaffolded_results_can_replace_full_task"] is False
       and summary["interface_smoke_substituted_for_performance"] is False,
       "summary honest flags")
    ok(summary["escalation"] and "INCONCLUSIVE_PARTICIPATION"
       in summary["escalation"], "summary escalation to 总控")
    ok(summary["selection_predicate_rule"]["order"] == FROZEN_RULE_ORDER
       and summary["selection_predicate_rule"]["tie_tolerance"] == 1e-12,
       "summary rule verbatim")
    gh = summary["git_head_policy"]
    ok(gh["marker_git_commit_head"] == PINS["MARKER_HEAD"]
       and gh["execution_git_commit_head"] == PINS["EXECUTION_HEAD"]
       and gh["closing_git_commit_head"] == PINS["CLOSING_HEAD"]
       and gh["execution_heads_uniform"] is True
       and gh["execution_heads_equal_or_descended_from_marker"] is True,
       "git_head_policy disclosure")
    ok(all(p["candidate_id"] + "=" + ("EQUAL" if False else "DESCENDANT")
           in gh["bundle_head_relations"] or p["candidate_id"] + "=EQUAL"
           in gh["bundle_head_relations"]
           for p in summary["participants"]), "per-bundle head relations")
    for p in summary["participants"]:
        cid = p["candidate_id"]
        if cid == "CONTROL_CONTINUOUS_98304":
            ok(p["participant_status"] == "ELIGIBLE_COMPLETE"
               and p["eligible"] is True and p["student_rank"] is None
               and tuple(p["rule_tuple"][k] for k in FROZEN_RULE_ORDER)
               == EXPECTED_CONTROL_TUPLE, "summary CONTROL participant")
        else:
            ok(p["participant_status"] == "BLOCKED_ENGINE_ABORT"
               and p["eligible"] is False and p["student_rank"] is None
               and p["rule_tuple"] is None, "summary BLOCKED " + cid)
        if cid == TEACHER:
            ok(p["excluded_from_student_ranking"] is True
               and p["reference_only"] is True, "teacher excluded/reference")

    # 9. gate 文件
    gate = read_json(os.path.join(CC4, "FORMAL_EVALUATION_GATE_V2DT.json"))
    ok(gate["FORMAL_EVALUATION_GATE_V2DT_PASS"] is False, "gate honestly false")
    g = gate["gates"]
    ok(g["G1_ALL_6_STUDENTS_ELIGIBLE_COMPLETE"] is False
       and g["G2_TEACHER_REFERENCE_COMPLETE"] is False
       and g["G3_NO_ENGINE_ABORT"] is False, "participation gates false")
    ok(all(g[k] is True for k in (
        "G4_NO_REHEARSAL_IN_FORMAL_POOL", "G5_CERTIFICATES_ALL_VERIFY",
        "G6_PINS_UNIFORM_FROZEN", "G7_GIT_HEAD_UNIFORM",
        "G7b_GIT_HEAD_EQUAL_OR_DESCENDED_FROM_MARKER", "G8_RULE_VERBATIM",
        "G9_REGISTRY_RANK_NULL", "G10_RANKING_COMPUTED_HONEST")),
       "integrity gates all true")
    ok(gate["flip_policy"]
       == "PUBLISH_HONEST_INCONCLUSIVE_UNDER_ENGINE_BLOCK", "flip policy")
    ok(gate["foreign_gate_failures"] == [], "no foreign failures")
    ok(gate["engine_blocked_candidate_ids"] == BLOCKED_IDS, "blocked ids")
    ok(gate["formal_ranking_summary_sha256"] == PINS["SUMMARY_SHA256"]
       and gate["secondary_audit_marker_sha256"] == PINS["MARKER_SHA256"],
       "gate sha references")
    ok(gate["ranking_status"] == "INCONCLUSIVE_PARTICIPATION"
       and gate["student_common_eligible_count"] == "1/6"
       and gate["teacher_reference_binding"] == "FAIL", "gate outcome fields")
    for k in ("CHECKPOINTS_MODIFIED", "CONTROL_RETRAINED",
              "CANDIDATE_EXCEPTION_USED", "FROZEN_BANKS_MODIFIED",
              "RETRAINING_PERFORMED"):
        ok(gate[k] is False, "gate " + k)
    ok(all(f.split(": ", 1)[0] in BLOCKED_IDS
           for f in gate["gate_failures"]), "all failures from blocked ids")

    # 10. 禁夸扫描（对发布的两份 JSON 原文）
    for name in ("FORMAL_RANKING_SUMMARY_V2DT.json",
                 "FORMAL_EVALUATION_GATE_V2DT.json"):
        text = open(os.path.join(CC4, name), encoding="utf-8").read()
        for bad in ('"scientific_claim_authorized": true',
                    '"scaffolded_results_can_replace_full_task": true',
                    '"interface_smoke_substituted_for_performance": true',
                    '"teacher_included_in_student_ranking": true',
                    '"FORMAL_EVALUATION_GATE_V2DT_PASS": true',
                    "SCIENTIFIC_CLAIM: AUTHORIZED"):
            ok(bad not in text, "overclaim %r in %s" % (bad, name))

    # 11. 日志证据
    for cid in ALL_CANDIDATES:
        log = open(os.path.join(LOGS, cid + ".log"),
                   encoding="utf-8", errors="replace").read()
        ok("[done]" in log, cid + " log has [done]")
        if cid in BLOCKED_IDS:
            ok("NOT relaxed, NOT a formal score" in log
               and PINS["ABORT_VERDICT"] in log,
               cid + " log BLOCKED discipline line")
        else:
            ok("evaluation_status=PASS" in log, cid + " log PASS")
    for cid in ("PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
                "RESET128_RMT16_ORIGINAL_VTRACE_98304",
                "SLOWGRU_RESET128_CANONICAL_98304",
                "BASELINE_TEACHER_CKPT17500"):
        raw_log = [fn for fn in os.listdir(LOGS)
                   if fn.startswith(cid + "_rawcrash")]
        ok(len(raw_log) == 1, cid + " raw-crash log preserved")
        text = open(os.path.join(LOGS, raw_log[0]),
                    encoding="utf-8", errors="replace").read()
        ok("tier3_failure_taxonomy.FailClosed" in text and "NEG20" in text,
           cid + " raw crash NEG20 taxonomy class")
    for fn in ("_queue_gpu0.nohup", "_queue_gpu1.nohup", "_queue_gpu2.nohup",
               "_queue_gpu3.nohup", "_rerun_wave1_gpu0.nohup",
               "_rerun_wave1_gpu1.nohup", "_rerun_wave2_gpu1.nohup",
               "_rerun_wave2_gpu3.nohup"):
        ok(os.path.isfile(os.path.join(LOGS, fn)), "queue log " + fn)

    print("FORMAL_EVALUATION_EVIDENCE_VERIFIED checks=%d" % CHECKS)
    print("ranking_status=INCONCLUSIVE_PARTICIPATION eligible=1/6 "
          "(CONTROL_CONTINUOUS_98304) blocked=6 (5 students + teacher)")
    print("flip_policy=PUBLISH_HONEST_INCONCLUSIVE_UNDER_ENGINE_BLOCK "
          "gate_pass=False (participation gates; integrity gates all true)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
