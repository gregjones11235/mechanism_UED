#!/usr/bin/env python3
"""CC4 V3 复合事件语义修复正式评估证据 — 独立离线复验器（2026-08-01）。

对本证据目录（自服务器 /home/oseasy/student_pool_v1 原样回传，tarball sha
ecf55214e0d48871d782aa230285862f201e7dea8f5f29954bcb101defa81d0d）做独立复验：

  * 权重隔离（主动断言目录内无任何 npz/pkl/npy/ckpt/orbax——npz/pkl 永不出服务器）;
  * 发布件（V3 ranking summary / gate / V3 repair marker）+ sidecar 重哈希;
  * V3 修复授权 marker 链（逐字总控裁定 / 9 条禁令 / V2 归档 SHA / taxonomy LF-SHA / git HEAD）;
  * 跨 GPU 确定性预检（GPU2/GPU3，未放宽，零差异）;
  * READY_V3 收口翻转（FORMAL_RANKING_STARTED/PUBLISHED=true，summary/gate sha 引用）;
  * 冻结排名规则逐字（metric_schema.json，sha 钉）;
  * 7 份 bundle：SHA256SUMS_FORMAL_V3 重哈希、READY、证书统一钉、episode SHA 重算、
    逐臂 reuse provenance（FULL 离线复用 / FRONT 离线重分类 / BACK 补跑或复用重签）;
  * 复合事件层：每条 transition∧defeat 记录 taxonomy_status=VALID_COMPOSITE_EVENT;
  * **从原始 evaluation result 独立重算排名**（原生实现，不导入排名工具）与发布的
    FORMAL_RANKING_SUMMARY_V3.json 比对——复现 INCONCLUSIVE_FULL_TIE（RESET128 与
    BASE_GTRXL 四级全平），winner=null;
  * §四F 奇偶：CONTROL 四元组 == (0,0,0.4196479859579006,7) 精确;
  * §四H 双重断言：对全部 9 条提交的 transition∧defeat FRONT 记录，V3 判
    VALID_COMPOSITE_EVENT **且** 冻结 V1 classify_episode 仍抛 NEG20（修复是加性的、
    冻结面未改）——此节导入仓库内提交工具（均 JAX-free）;
  * V2 归档零触碰（G14）：marker/gate/summary 记录的 V2 summary/gate SHA == 冻结钉;
  * 禁夸扫描 + 日志证据。

用法：python verify_formal_evaluation_evidence_v3.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC4 = os.path.join(HERE, "cc4")
COMMON = os.path.join(HERE, "common_v2")
LOGS = os.path.join(CC4, "formal_eval_logs")
# 仓库内提交工具（§四H 双重断言用；均 JAX-free）。HERE 上三级为仓库根。
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "..",
                                      "tools", "tier3_scaffolded_evaluation"))

# --- 冻结钉（与被验证据交叉核对；全部来自冻结公共评测器 / 审计记录）-----------
PINS = {
    "TAR_SHA256":
        "ecf55214e0d48871d782aa230285862f201e7dea8f5f29954bcb101defa81d0d",
    "SUMMARY_SHA256":
        "dab522cf7bcc43ed74f0bc1e9cab20c01c98d972d7edceb2717f9dc18445b659",
    "GATE_SHA256":
        "c529ebf3ddbf37085b85b0a79018d9cc06ce5a096dc744d18a97a3e0c8b72528",
    "MARKER_SHA256":
        "efa68c85f95fa7e7ced144ea81040fec5d849cd97e61fd7b8a39ed97dc835e6a",
    "PREFLIGHT_SHA256":
        "afaa6b3b58cce828709224bb4b8114f0bebab5a0dc51070f2be86ce51d17d794",
    "COMMON_EVALUATOR_SHA256":
        "2978a0f625bc94e18c99649959e8c090f964cd66e5dafd6b93245f144a317037",
    "METRIC_SCHEMA_SHA256":
        "8ec4adcdfa6844b276f5f253470e14ea8ad52f1e64c398e5e2658e8a066645c7",
    "FRONT_BANK_CONTENT_SHA256":
        "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687",
    "BACK_BANK_CONTENT_SHA256":
        "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566",
    "TAXONOMY_V3_LF_SHA256":
        "01f06d09190a70898b11165aed016d5f7f96a1e0ca9366acc81dbd4d9d6a3da2",
    # V2 归档（冻结引用，V3 绝不触碰）
    "V2_ARCHIVE_SUMMARY_SHA256":
        "3e8186417aefeb25729324ce5fb4bc6b56a58087c8d1ee67bc088ad37d5c1ac3",
    "V2_ARCHIVE_GATE_SHA256":
        "51d3d6fb8efbc978875823cdc4576443c4d61f308840462c1bfa12da52fddc5b",
    "HEAD": "5f035ed238171729a47633b5c54f0b14da059082",  # marker==execution==closing
    "MARKER_RECORDED_AT_UTC": "2026-08-01T03:22:55.900130+00:00",
    "RULING_TASK":
        "CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_FORMAL_EVALUATION_V3",
    "VERDICT": "AUTHORIZED_COMPOSITE_EVENT_SEMANTIC_REPAIR_V3",
    "FORMAL_EVALUATOR_PROTOCOL": "V3_COMPOSITE_EVENT",
    "NEG20_PROTOCOL": "NEG20_V3_PRIMARY_SECONDARY_EVENTS",
    "GPU2_UUID": "GPU-8df11537-ab79-722d-606f-411966196c4c",
    "GPU3_UUID": "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",
}
V3_GPU_ALLOWED = {PINS["GPU2_UUID"], PINS["GPU3_UUID"]}

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
# 6 个做 BACK 补跑（V2 为 0/8 骨架）；CONTROL 全臂完整 → 复用重签。
BACK_COMPLETION_IDS = sorted(set(ALL_CANDIDATES) - {"CONTROL_CONTINUOUS_98304"})

FROZEN_RULE_ORDER = [
    "full success_count",
    "front_l2 transition_count",
    "front_l2 mean graph_distance_progress",
    "back_l2 defeat_count",
]
# 发布四元组（独立重算必须逐位复现）
EXPECTED_TUPLES = {
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304": (9, 3, 0.5905970705064548, 7),
    "RESET128_RMT16_ORIGINAL_VTRACE_98304": (14, 2, 0.5650157181747473, 8),
    "BASE_GTRXL_ORIGINAL_VTRACE_98304": (14, 2, 0.5650157181747473, 8),
    "CONTROL_CONTINUOUS_98304": (0, 0, 0.4196479859579006, 7),
    "SLOWGRU_RESET128_CANONICAL_98304": (17, 2, 0.5236034412438056, 7),
    "SLOWGRU_PERSISTENT_CANONICAL_98304": (17, 2, 0.575285501489573, 6),
    "BASELINE_TEACHER_CKPT17500": (19, 2, 0.5805684102905279, 7),
}
EXPECTED_CONTROL_TUPLE = (0, 0, 0.4196479859579006, 7)
# FRONT 复合（major>=2）计数；其中 transition∧defeat 子集（== 原 NEG20 复现数）
EXPECTED_COMPOSITE_FRONT = {
    "CONTROL_CONTINUOUS_98304": 0,
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304": 3,
    "RESET128_RMT16_ORIGINAL_VTRACE_98304": 2,
    "BASE_GTRXL_ORIGINAL_VTRACE_98304": 2,
    "SLOWGRU_RESET128_CANONICAL_98304": 2,
    "SLOWGRU_PERSISTENT_CANONICAL_98304": 2,
    "BASELINE_TEACHER_CKPT17500": 2,
}
EXPECTED_TRANSITION_DEFEAT = {
    "CONTROL_CONTINUOUS_98304": 0,
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304": 1,
    "RESET128_RMT16_ORIGINAL_VTRACE_98304": 2,
    "BASE_GTRXL_ORIGINAL_VTRACE_98304": 2,
    "SLOWGRU_RESET128_CANONICAL_98304": 1,
    "SLOWGRU_PERSISTENT_CANONICAL_98304": 1,
    "BASELINE_TEACHER_CKPT17500": 2,
}
EXPECTED_TIE_GROUP = ["BASE_GTRXL_ORIGINAL_VTRACE_98304",
                      "RESET128_RMT16_ORIGINAL_VTRACE_98304"]

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


def read_records(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def scenario_line_hash(path, scenario):
    """Canonical episode-records digest = sha256 of the RAW stored jsonl lines
    (byte-identical, no re-serialization) for `scenario`, joined by "\\n" with a
    trailing "\\n". Re-serializing with json.dumps(sort_keys=...) would NOT match
    the published episode_records_sha256 — the line bytes themselves are the pin."""
    with open(path, encoding="utf-8") as fh:
        raw = [ln.rstrip("\n") for ln in fh if ln.strip()]
    lines = [ln for ln in raw if json.loads(ln)["scenario"] == scenario]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def bundle_dir(cid):
    return os.path.join(CC4, cid, "formal_evaluation_v3")


def main():
    # ---- 0. 权重隔离：证据目录不得含任何权重文件 ----------------------------
    for root, _dirs, files in os.walk(HERE):
        for fn in files:
            ok(not fn.endswith((".npz", ".pkl", ".npy", ".ckpt", ".orbax")),
               "weight file in evidence dir: %s" % os.path.join(root, fn))

    # ---- 1. 发布件 + sidecar 重哈希 -----------------------------------------
    for name, pin in (("FORMAL_RANKING_SUMMARY_V3.json", PINS["SUMMARY_SHA256"]),
                      ("FORMAL_EVALUATION_GATE_V3.json", PINS["GATE_SHA256"]),
                      ("V3_REPAIR_AUTHORIZATION.json", PINS["MARKER_SHA256"])):
        path = os.path.join(CC4, name)
        ok(sha256_file(path) == pin, "%s sha != pin" % name)
        side = open(path + ".sha256", encoding="utf-8").read().split()
        ok(side[0] == pin and side[1] == name, "%s sidecar mismatch" % name)

    # ---- 2. V3 修复授权 marker 链 -------------------------------------------
    marker = read_json(os.path.join(CC4, "V3_REPAIR_AUTHORIZATION.json"))
    ok(marker.get("schema") == "mechanism_UED.tier3_v3_repair_authorization/v1",
       "marker schema")
    ok(marker.get("ruling_task") == PINS["RULING_TASK"], "marker ruling_task")
    ok(marker.get("verdict") == PINS["VERDICT"], "marker verdict")
    ok(marker.get("recorded_at_utc") == PINS["MARKER_RECORDED_AT_UTC"],
       "marker recorded_at")
    ruling = marker.get("ruling") or {}
    ok(len(ruling.get("verbatim") or "") > 100, "marker verbatim recorded")
    proh = ruling.get("prohibitions") or []
    for must in ("NO_RETRAINING", "NO_CHECKPOINT_MODIFICATION",
                 "NO_CANDIDATE_LEVEL_EXEMPTION",
                 "NO_SUBMETRIC_RANKING_DOWNGRADE",
                 "NO_V2_EVIDENCE_DELETE_OVERWRITE_REWRITE",
                 "NO_MERGE_REBASE_AMEND_FORCE_PUSH", "NO_PERFORMANCE_RETRY",
                 "NO_SMOKE_SUBSTITUTION_FOR_PERFORMANCE",
                 "NO_FULL_ONLY_OR_BACK_ONLY_RANKING"):
        ok(must in proh, "marker prohibition %s recorded" % must)
    ok(ruling.get("new_protocols") == {
        "formal_evaluator_protocol": PINS["FORMAL_EVALUATOR_PROTOCOL"],
        "neg20_protocol": PINS["NEG20_PROTOCOL"]}, "marker new protocols")
    mev = marker.get("evidence") or {}
    ok(mev.get("v2_archive_summary_sha256") == PINS["V2_ARCHIVE_SUMMARY_SHA256"],
       "marker V2 archive summary sha")
    ok(mev.get("v2_archive_gate_sha256") == PINS["V2_ARCHIVE_GATE_SHA256"],
       "marker V2 archive gate sha")
    ok(mev.get("v2_status") == "CLOSED_INCONCLUSIVE_PARTICIPATION"
       and mev.get("v2_winner") is None
       and mev.get("v2_evidence_modified_by_v3") is False,
       "marker V2 archive status/winner/untouched")
    ok(mev.get("taxonomy_v3_lf_sha256") == PINS["TAXONOMY_V3_LF_SHA256"],
       "marker taxonomy_v3 LF-SHA")
    ok(mev.get("git_commit_head") == PINS["HEAD"], "marker git head")

    # ---- 3. 跨 GPU 确定性预检（GPU2/GPU3，未放宽，零差异）-------------------
    pre_path = os.path.join(CC4, "CROSS_GPU_DETERMINISM_PREFLIGHT_V3.json")
    ok(sha256_file(pre_path) == PINS["PREFLIGHT_SHA256"], "preflight sha")
    pre = read_json(pre_path)
    ok(pre.get("schema") == "mechanism_UED.cross_gpu_preflight/v1",
       "preflight schema frozen")
    ok(pre.get("CROSS_GPU_DETERMINISM_PREFLIGHT") == "PASS", "preflight PASS")
    ok(pre.get("comparison_loosened") is False, "preflight NOT loosened")
    ok(pre.get("all_mismatches") == [], "preflight zero mismatches")
    ok(pre.get("first_difference") is None, "preflight no first difference")
    ok(pre.get("checkpoint_file_sha256") and pre.get("params_sha256"),
       "preflight records checkpoint/params sha")
    # GPU 纪律：预检以整数索引 CUDA_VISIBLE_DEVICES 表达（G16 接受该形式）；
    # 索引 2/3 ↔ UUID GPU-8df1…(GPU2)/GPU-f56a…(GPU3)，GPU0/1 对 V3 禁用。
    gpu_idx = {str((pre.get("gpu_a") or {}).get("cuda_visible_devices")),
               str((pre.get("gpu_b") or {}).get("cuda_visible_devices"))}
    ok(gpu_idx == {"2", "3"}, "preflight GPUs == {idx2,idx3}: %s" % gpu_idx)
    ok(pre.get("front_bank_content_sha256") == PINS["FRONT_BANK_CONTENT_SHA256"]
       and pre.get("back_bank_content_sha256")
       == PINS["BACK_BANK_CONTENT_SHA256"], "preflight bank content shas")

    # ---- 4. READY_V3 收口翻转 ------------------------------------------------
    ready = read_json(os.path.join(CC4, "COMMON_EVALUATOR_V3_READY.json"))
    ok(ready.get("COMMON_EVALUATOR_V3_READY") is True, "READY_V3 true")
    ok(ready.get("FORMAL_RANKING_STARTED") is True
       and ready.get("FORMAL_RANKING_PUBLISHED") is True, "ranking flags")
    ok(ready.get("FORMAL_RANKING_AUTHORIZED_V3") is True, "AUTH_V3 true")
    ok(ready.get("ranking_status") == "INCONCLUSIVE_FULL_TIE"
       and ready.get("formal_winner") is None, "READY ranking outcome")
    ok(ready.get("formal_ranking_summary_sha256") == PINS["SUMMARY_SHA256"],
       "READY summary sha")
    ok(ready.get("formal_evaluation_gate_sha256") == PINS["GATE_SHA256"],
       "READY gate sha")
    ok(ready.get("pending_gates") == [], "READY pending_gates retired")
    ok(ready.get("scientific_claim_authorized") is False
       and ready.get("single_training_seed") is True
       and ready.get("teacher_included_in_student_ranking") is False,
       "READY honest flags")
    ok(ready.get("flip_policy") == "V3_GATE_GREEN", "READY flip policy")
    ok((ready.get("v3_repair_marker") or {}).get("ruling_task")
       == PINS["RULING_TASK"], "READY marker ref")

    # ---- 5. 冻结排名规则逐字（回传的 metric_schema.json，sha 钉）------------
    ok(sha256_file(os.path.join(COMMON, "metric_schema.json"))
       == PINS["METRIC_SCHEMA_SHA256"], "metric_schema sha frozen")
    schema = read_json(os.path.join(COMMON, "metric_schema.json"))
    spr = schema["selection_predicate_rule"]
    ok(spr["order"] == FROZEN_RULE_ORDER, "schema rule order verbatim")
    ok(spr["all_equal_result"] == "INCONCLUSIVE", "schema all_equal_result")

    # ---- 6. 七份 bundle ------------------------------------------------------
    bundle_facts = {}
    common_pins_seen = set()
    for cid in ALL_CANDIDATES:
        d = bundle_dir(cid)
        tag = cid + ": "
        is_completion = cid in BACK_COMPLETION_IDS
        # 6a. SHA256SUMS_FORMAL_V3 重哈希（6 行）
        sums = open(os.path.join(d, "SHA256SUMS_FORMAL_V3"),
                    encoding="utf-8").read().splitlines()
        ok(len(sums) == 6, tag + "sums line count == 6")
        for line in sums:
            h, fn = line.split("  ", 1)
            ok(sha256_file(os.path.join(d, fn)) == h, tag + "sums " + fn)
        # 6b. READY_FORMAL_V3
        r = read_json(os.path.join(d, "READY_FORMAL_V3.json"))
        ok(r.get("candidate_id") == cid, tag + "READY candidate_id")
        ok(r.get("run_class") == "FORMAL_EVALUATION", tag + "run_class")
        ok(r.get("rehearsal") is False, tag + "not rehearsal")
        ok(r.get("evaluation_status") == "PASS", tag + "status PASS")
        ok(r.get("READY_FORMAL_V3") is True, tag + "READY true")
        ok(r.get("student_rank") is None, tag + "READY rank null")
        ok(r.get("scientific_claim_authorized") is False
           and r.get("teacher_included_in_student_ranking") is False
           and r.get("scaffolded_results_can_replace_full_task") is False,
           tag + "READY honest flags")
        gates = r.get("gates") or {}
        ok(bool(gates) and all(gates.values()), tag + "all per-bundle gates true")
        ok(r.get("formal_abort") is None, tag + "no abort")
        # 6c. certificate
        c = read_json(os.path.join(d, "evaluation_certificate_v3.json"))
        ok(c.get("schema") == "mechanism_UED.tier3_evaluation_certificate/v3",
           tag + "cert schema v3")
        ok(c.get("run_class") == "FORMAL_EVALUATION", tag + "cert run_class")
        ok(c.get("student_rank") is None, tag + "cert rank null")
        ok(c.get("taxonomy_v3_lf_sha256") == PINS["TAXONOMY_V3_LF_SHA256"],
           tag + "cert taxonomy_v3 LF-SHA")
        ok(c.get("neg20_protocol") == PINS["NEG20_PROTOCOL"],
           tag + "cert neg20 protocol")
        ok(c.get("common_evaluator_protocol_version")
           == PINS["FORMAL_EVALUATOR_PROTOCOL"], tag + "cert evaluator protocol")
        for flag in ("scientific_claim_authorized",
                     "scaffolded_results_can_replace_full_task",
                     "interface_smoke_substituted_for_performance",
                     "teacher_included_in_student_ranking"):
            ok(c.get(flag) is False, tag + "cert " + flag)
        ok((c.get("provenance") or {}).get("git_commit_head") == PINS["HEAD"],
           tag + "cert execution head == frozen HEAD")
        pins = c.get("common_pins") or {}
        ok(pins.get("common_evaluator_sha256")
           == PINS["COMMON_EVALUATOR_SHA256"], tag + "cert evaluator pin")
        ok(pins.get("metric_schema_sha256") == PINS["METRIC_SCHEMA_SHA256"],
           tag + "cert metric_schema pin")
        ok(pins.get("front_bank_content_sha256")
           == PINS["FRONT_BANK_CONTENT_SHA256"]
           and pins.get("back_bank_content_sha256")
           == PINS["BACK_BANK_CONTENT_SHA256"], tag + "cert bank content pins")
        common_pins_seen.add(json.dumps(pins, sort_keys=True))
        # GPU 纪律（G16）：证书记录的可见 GPU ⊆ {GPU2,GPU3}
        cgpu = (c.get("gpu") or {}).get("visible_gpu_uuids") or []
        ok(bool(cgpu) and set(cgpu) <= V3_GPU_ALLOWED,
           tag + "cert GPU ⊆ {GPU2,GPU3}: %s" % cgpu)
        # 6d. 逐臂 result + episode_records SHA 重算
        records_path = os.path.join(d, "episode_records.jsonl")
        raw = read_records(records_path)
        results = {sc: read_json(os.path.join(
            d, "evaluation_result_v3.%s.json" % sc))
            for sc in ("full", "front_l2", "back_l2")}
        for sc in ("full", "front_l2", "back_l2"):
            res = results[sc]
            ok(res.get("candidate_id") == cid, tag + sc + " candidate_id")
            ok(res.get("run_class") == "FORMAL_EVALUATION", tag + sc + " run_class")
            ok(res.get("rehearsal") is False, tag + sc + " not rehearsal")
            ok(res.get("neg20_protocol") == PINS["NEG20_PROTOCOL"],
               tag + sc + " neg20 protocol")
            n = len([x for x in raw if x["scenario"] == sc])
            want = 64 if sc == "full" else 8
            ok(res.get("episodes_executed") == want and n == want,
               tag + sc + " episode count == %d" % want)
            ok(res.get("aborted_in_scenario") is False
               and res.get("evaluation") is not None,
               tag + sc + " complete with evaluation (no abort)")
            ok(res.get("episode_records_sha256")
               == scenario_line_hash(records_path, sc),
               tag + sc + " episode_records_sha256 recompute")
        # 6e. reuse provenance（FULL 离线复用 / FRONT 离线重分类 / BACK 补跑或复用）
        rp_full = results["full"].get("reuse_provenance") or {}
        ok(rp_full.get("reuse_status") == "REUSED_PASS"
           and rp_full.get("classification_only") is True
           and rp_full.get("environment_rerun") is False
           and rp_full.get("source") == "V2_COMMITTED_EVIDENCE"
           and rp_full.get("source_v2_episode_sha256"),
           tag + "FULL reused offline (REUSED_PASS, no rerun, V2 source sha)")
        rg = rp_full.get("reuse_gate") or {}
        ok(len(rg) == 9 and all(rg.get(g) is True for g in (
            "R1_EPISODES_COMPLETE", "R2_RECORD_SHA_RECOMPUTE",
            "R3_V2_SUMS_REHASH", "R4_CHECKPOINT_PARAMS_OWNER_MATCH",
            "R5_RUNTIME_CAPSULE_MATCH", "R6_SCHEDULE_FROZEN",
            "R7_NO_PERFORMANCE_EARLY_STOP", "R8_ENGINE_LF_SHA_FROZEN",
            "R9_V3_RECLASSIFY_REPRODUCIBLE")),
           tag + "FULL reuse_gate R1-R9 all true")
        ok(rp_full.get("v3_classifier_sha256")
           == PINS["TAXONOMY_V3_LF_SHA256"], tag + "FULL v3 classifier sha")
        rp_front = results["front_l2"].get("reuse_provenance") or {}
        ok(rp_front.get("reuse_status") == "REUSED_RECLASSIFIED"
           and rp_front.get("classification_only") is True
           and rp_front.get("environment_rerun") is False
           and rp_front.get("source") == "V2_COMMITTED_EVIDENCE"
           and rp_front.get("source_v2_episode_sha256"),
           tag + "FRONT reclassified offline (source V2 sha present)")
        ok(rp_front.get("composite_episode_count")
           == EXPECTED_COMPOSITE_FRONT[cid],
           tag + "FRONT reuse composite count matches")
        prs = rp_front.get("per_record_source_sha256") or {}
        ok(len(prs) == 8
           and all(prs.get("front_l2-bank%d" % i) for i in range(8)),
           tag + "FRONT per-record source sha ×8")
        ok(rp_front.get("v3_classifier_sha256")
           == PINS["TAXONOMY_V3_LF_SHA256"], tag + "FRONT v3 classifier sha")
        rp_back = results["back_l2"].get("reuse_provenance") or {}
        if is_completion:
            ok(rp_back.get("reuse_status") == "COMPLETED"
               and rp_back.get("source") == "V3_FRESH_COMPLETION_RUN"
               and rp_back.get("environment_rerun") is True
               and rp_back.get("classification_only") is False
               and rp_back.get("source_v2_episode_sha256") is None,
               tag + "BACK completion (fresh run, not a retry)")
        else:
            ok(rp_back.get("reuse_status") == "REUSED_RESIGNED"
               and rp_back.get("source") == "V2_COMMITTED_EVIDENCE"
               and rp_back.get("environment_rerun") is False
               and rp_back.get("classification_only") is True
               and rp_back.get("source_v2_episode_sha256"),
               tag + "BACK reused+resigned (CONTROL; source V2 sha present)")
        ok(rp_back.get("v3_classifier_sha256")
           == PINS["TAXONOMY_V3_LF_SHA256"], tag + "BACK v3 classifier sha")
        # 6f. 复合事件层：计数匹配证书披露；每条复合 taxonomy_status 合法
        disc = c.get("composite_event_disclosure") or {}
        by_sc = disc.get("composite_episode_count_by_scenario") or {}
        ok(by_sc.get("front_l2") == EXPECTED_COMPOSITE_FRONT[cid]
           and by_sc.get("back_l2") == 0 and by_sc.get("full") == 0,
           tag + "composite counts match disclosure %s" % by_sc)
        cel = results["front_l2"]["evaluation"].get("composite_event_layer") or {}
        ok(cel.get("composite_episode_count") == EXPECTED_COMPOSITE_FRONT[cid],
           tag + "front composite_event_layer count")
        for pe in cel.get("per_episode") or []:
            if pe.get("taxonomy_status") == "VALID_COMPOSITE_EVENT":
                ok(bool(pe.get("primary_outcome"))
                   and isinstance(pe.get("secondary_events"), list)
                   and len(pe["secondary_events"]) >= 1,
                   tag + "composite episode has primary+secondary")
        bundle_facts[cid] = {"results": results, "records": raw,
                             "cert": c}
    # §六：7 份证书引用完全相同的公共钉集
    ok(len(common_pins_seen) == 1, "common pins uniform across 7 bundles")

    # ---- 7. 独立排名重算（原生实现，不导入排名工具）-------------------------
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

    ok(compare((11, 0, 0.0, 0), (10, 9, 1.0, 9)) == -1, "cmp level1 dominates")
    ok(compare((10, 5, 0.5 + 5e-13, 3), (10, 5, 0.5, 3)) == 0, "cmp tolerance")
    ok(compare((1, 2, 3.0, 4), (1, 2, 3.0, 5)) == 1, "cmp level4")

    recomputed = {cid: extract_tuple(bundle_facts[cid]) for cid in STUDENTS}
    for cid in STUDENTS:
        ok(recomputed[cid] == EXPECTED_TUPLES[cid],
           "recomputed tuple %s %r != published %r"
           % (cid, recomputed[cid], EXPECTED_TUPLES[cid]))
    # §四F：CONTROL 四元组精确奇偶
    ok(recomputed["CONTROL_CONTINUOUS_98304"] == EXPECTED_CONTROL_TUPLE,
       "CONTROL tuple bit-exact %r" % (recomputed["CONTROL_CONTINUOUS_98304"],))
    # teacher 元组（参考，不入学生排名）
    ok(extract_tuple(bundle_facts[TEACHER]) == EXPECTED_TUPLES[TEACHER],
       "teacher tuple reproduced")
    # 原生分组排名 → 检出四级全平组 {BASE_GTRXL, RESET128}
    ordered = sorted(STUDENTS,
                     key=lambda c: tuple(-v for v in recomputed[c]) + (c,))
    groups = []
    for cid in ordered:
        if groups and compare(recomputed[groups[-1][0]], recomputed[cid]) == 0:
            groups[-1].append(cid)
        else:
            groups.append([cid])
    tie_groups = sorted(sorted(g) for g in groups if len(g) > 1)
    ok(tie_groups == [EXPECTED_TIE_GROUP],
       "native recompute finds exactly the tie group %s, got %s"
       % (EXPECTED_TIE_GROUP, tie_groups))
    recomputed_status = "ORDERED" if not tie_groups else "INCONCLUSIVE_FULL_TIE"
    ok(recomputed_status == "INCONCLUSIVE_FULL_TIE",
       "recomputed status INCONCLUSIVE_FULL_TIE")
    recomputed_winner = None  # 非 ORDERED → 无 winner

    # ---- 8. 与发布的 summary 比对 -------------------------------------------
    summary = read_json(os.path.join(CC4, "FORMAL_RANKING_SUMMARY_V3.json"))
    ok(summary["ranking_status"] == recomputed_status, "summary status match")
    ok(summary["student_count_eligible"] == "6/6", "summary eligible 6/6")
    ok(summary["formal_winner"] is None
       and summary["top_ranked_student_id"] is None, "summary winner null")
    ok(summary["inconclusive_groups"] == [EXPECTED_TIE_GROUP],
       "summary inconclusive group")
    ok(summary["FORMAL_RANKING_AUTHORIZED_V3"] is True, "summary AUTH_V3 true")
    ok(summary["scientific_claim_authorized"] is False
       and summary["scaffolded_results_can_replace_full_task"] is False
       and summary["interface_smoke_substituted_for_performance"] is False
       and summary["teacher_included_in_student_ranking"] is False,
       "summary honest flags")
    rule = summary["selection_predicate_rule"]
    ok(rule["order"] == FROZEN_RULE_ORDER and rule["tie_tolerance"] == 1e-12
       and rule["all_equal_result"] == "INCONCLUSIVE"
       and rule["source"]["schema_sha256"] == PINS["METRIC_SCHEMA_SHA256"]
       and rule["source"]["verified_verbatim"] is True,
       "summary rule verbatim + schema-sha-pinned")
    gh = summary["git_head_policy"]
    ok(summary["v2_archive"]["v2_status"]
       == "CLOSED_INCONCLUSIVE_PARTICIPATION"
       and summary["v2_archive"]["v2_winner"] is None
       and summary["v2_archive"]["v2_summary_sha256"]
       == PINS["V2_ARCHIVE_SUMMARY_SHA256"]
       and summary["v2_archive"]["v2_gate_sha256"]
       == PINS["V2_ARCHIVE_GATE_SHA256"]
       and summary["v2_archive"]["v2_evidence_modified_by_v3"] is False,
       "summary V2 archive reference untouched")
    # composite 汇总
    ok(summary["composite_event_summary"]["total_composite_episodes"] == 13,
       "summary total composite == 13")
    # 每参与者元组 + 状态
    for p in summary["participants"]:
        cid = p["candidate_id"]
        t = p["rule_tuple"]
        tt = tuple(t[k] for k in FROZEN_RULE_ORDER)
        ok(tt == EXPECTED_TUPLES[cid], "summary participant tuple " + cid)
        if cid == TEACHER:
            ok(p["participant_status"] == "TEACHER_REFERENCE_ONLY"
               and p["reference_only"] is True
               and p["student_rank"] is None, "teacher reference-only")
        else:
            ok(p["participant_status"] == "ELIGIBLE_COMPLETE"
               and p["eligible"] is True, "student eligible+complete " + cid)
    # 排名位：SLOWGRU_PERSISTENT=1，SLOWGRU_RESET128=2，平局组 rank=None，CONTROL=6
    rank_by = {p["candidate_id"]: p["student_rank"]
               for p in summary["participants"]}
    ok(rank_by["SLOWGRU_PERSISTENT_CANONICAL_98304"] == 1
       and rank_by["SLOWGRU_RESET128_CANONICAL_98304"] == 2
       and rank_by["BASE_GTRXL_ORIGINAL_VTRACE_98304"] is None
       and rank_by["RESET128_RMT16_ORIGINAL_VTRACE_98304"] is None
       and rank_by["CONTROL_CONTINUOUS_98304"] == 6
       and rank_by[TEACHER] is None, "summary ranks (tie pair null)")
    ok(gh.get("execution_heads_uniform") is True
       and gh.get("execution_heads_equal_or_descended_from_marker") is True,
       "git_head_policy uniform + descended")

    # ---- 9. gate 文件 --------------------------------------------------------
    gate = read_json(os.path.join(CC4, "FORMAL_EVALUATION_GATE_V3.json"))
    ok(gate["FORMAL_EVALUATION_GATE_V3_PASS"] is True, "gate PASS true")
    ok(gate["FORMAL_RANKING_AUTHORIZED_V3"] is True, "gate AUTH_V3 true")
    g = gate["gates"]
    for k in ("G1_ALL_6_STUDENTS_ELIGIBLE_COMPLETE",
              "G2_TEACHER_REFERENCE_COMPLETE", "G3_NO_ENGINE_ABORT",
              "G4_NO_REHEARSAL_IN_FORMAL_POOL", "G5_CERTIFICATES_ALL_VERIFY",
              "G6_PINS_UNIFORM_FROZEN", "G7_GIT_HEAD_UNIFORM",
              "G7b_GIT_HEAD_EQUAL_OR_DESCENDED_FROM_MARKER", "G8_RULE_VERBATIM",
              "G9_REGISTRY_RANK_NULL", "G10_RANKING_COMPUTED_HONEST",
              "G11_FULL_REUSED_PASS_X7",
              "G12_FRONT_RECLASSIFIED_PROVENANCE_X7",
              "G13_BACK_COMPLETE_OR_REUSED_X7", "G14_V2_ARCHIVE_UNTOUCHED",
              "G15_V3_REPAIR_MARKER_VERIFIED", "G16_GPU_V3_ONLY"):
        ok(g.get(k) is True, "gate %s true" % k)
    ok(gate["flip_policy"] == "V3_GATE_GREEN", "gate flip policy green")
    ok(gate["ranking_status"] == "INCONCLUSIVE_FULL_TIE"
       and gate["formal_winner"] is None, "gate outcome fields")
    ok(gate["blocked_candidate_ids"] == [] and gate["gate_failures"] == []
       and gate["foreign_gate_failures"] == [], "gate no blocks/failures")
    ok(gate["student_common_eligible_count"] == "6/6", "gate eligible 6/6")
    # §九 禁止标志位
    ok(gate["CHECKPOINTS_MODIFIED"] is False
       and gate["STUDENTS_RETRAINED"] == 0
       and gate["CANDIDATE_EXCEPTIONS_USED"] == 0
       and gate["CANDIDATE_EXCEPTION_USED"] is False
       and gate["FULL_ONLY_RANKING_USED"] is False
       and gate["BACK_ONLY_RANKING_USED"] is False
       and gate["RETRAINING_PERFORMED"] is False
       and gate["FROZEN_BANKS_MODIFIED"] is False
       and gate["CONTROL_RETRAINED"] is False, "gate §九 prohibition flags")
    ok(gate["server_git_head"] == PINS["HEAD"]
       and gate["git_heads_uniform"] == [PINS["HEAD"]], "gate uniform HEAD")
    ok(gate["v3_repair_marker_sha256"] == PINS["MARKER_SHA256"],
       "gate marker sha ref")
    ok(gate["v2_archive"]["v2_summary_sha256"]
       == PINS["V2_ARCHIVE_SUMMARY_SHA256"]
       and gate["v2_archive"]["v2_gate_sha256"]
       == PINS["V2_ARCHIVE_GATE_SHA256"]
       and gate["v2_archive"]["v2_evidence_modified_by_v3"] is False,
       "gate V2 archive untouched (G14)")

    # ---- 10. 禁夸扫描（对发布的两份 JSON 原文）------------------------------
    for name in ("FORMAL_RANKING_SUMMARY_V3.json",
                 "FORMAL_EVALUATION_GATE_V3.json"):
        text = open(os.path.join(CC4, name), encoding="utf-8").read()
        for bad in ('"scientific_claim_authorized": true',
                    '"scaffolded_results_can_replace_full_task": true',
                    '"interface_smoke_substituted_for_performance": true',
                    '"teacher_included_in_student_ranking": true',
                    '"FORMAL_RANKING_AUTHORIZED_V3": false',
                    '"FORMAL_EVALUATION_GATE_V3_PASS": false',
                    "SCIENTIFIC_CLAIM: AUTHORIZED"):
            ok(bad not in text, "overclaim %r in %s" % (bad, name))

    # ---- 11. §四H 双重断言（导入仓库内提交工具；均 JAX-free）----------------
    sys.path.insert(0, TOOLS)
    import tier3_failure_taxonomy as taxonomy_v1
    import tier3_taxonomy_v3 as taxonomy_v3
    ok(taxonomy_v3.module_lf_sha256() == PINS["TAXONOMY_V3_LF_SHA256"],
       "taxonomy_v3 module LF-SHA == committed pin")
    total_td = 0
    total_composite = 0
    for cid in ALL_CANDIDATES:
        raw = bundle_facts[cid]["records"]
        front = [r for r in raw if r["scenario"] == "front_l2"]
        td = 0
        comp = 0
        for rec in front:
            v3 = taxonomy_v3.classify_episode_v3("front_l2", rec)
            if v3["taxonomy_status"] == "VALID_COMPOSITE_EVENT":
                comp += 1
                # 任何复合都不得 FailClosed，且有 primary+secondary
                ok(bool(v3["primary_outcome"])
                   and len(v3["secondary_events"]) >= 1,
                   cid + " composite has primary+secondary")
            trans = rec.get("front_floor_transition_reached") is True
            defeat = rec.get("defeat_kobold") is True
            if trans and defeat:
                td += 1
                # V3：合法复合事件（primary 过渡成功，secondary 含 DEFEAT_KOBOLD）
                ok(v3["taxonomy_status"] == "VALID_COMPOSITE_EVENT"
                   and v3["primary_outcome"]
                   == taxonomy_v3.FRONT_TRANSITION_SUCCESS
                   and taxonomy_v3.EV_DEFEAT_KOBOLD in v3["secondary_events"],
                   cid + " transition∧defeat → V3 VALID_COMPOSITE_EVENT")
                ok(taxonomy_v3.verify_record_sha(rec)
                   == rec.get("episode_record_sha256"),
                   cid + " record sha recomputes")
                # 冻结 V1：仍抛 NEG20（修复是加性的，冻结面未改）
                try:
                    taxonomy_v1.classify_episode(dict(rec))
                    ok(False, cid + " frozen V1 did NOT raise on composite")
                except taxonomy_v1.FailClosed as exc:
                    ok("NEG20" in str(exc),
                       cid + " frozen V1 raises NEG20 on composite")
        ok(td == EXPECTED_TRANSITION_DEFEAT[cid],
           "%s transition∧defeat count %d != %d"
           % (cid, td, EXPECTED_TRANSITION_DEFEAT[cid]))
        ok(comp == EXPECTED_COMPOSITE_FRONT[cid],
           "%s composite(major>=2) count %d != %d"
           % (cid, comp, EXPECTED_COMPOSITE_FRONT[cid]))
        total_td += td
        total_composite += comp
    ok(total_td == 9, "ORIGINAL_NEG20_REPRODUCTIONS_FIXED == 9 (got %d)"
       % total_td)
    ok(total_composite == 13, "total composite episodes == 13 (got %d)"
       % total_composite)

    # ---- 12. 日志证据 --------------------------------------------------------
    log_files = os.listdir(LOGS)
    for cid in ALL_CANDIDATES:
        matches = [fn for fn in log_files if fn.endswith(cid + ".log")]
        ok(len(matches) == 1, cid + " has exactly one run log")
        text = open(os.path.join(LOGS, matches[0]),
                    encoding="utf-8", errors="replace").read()
        ok("[done]" in text and "evaluation_status=PASS" in text,
           cid + " log [done]+PASS")
        if cid in BACK_COMPLETION_IDS:
            ok("[stage7/BACK] COMPLETED (8 fresh episodes)" in text,
               cid + " log BACK completion line")
        else:
            ok("[stage7/BACK] REUSED_RESIGNED" in text,
               cid + " log BACK reused-resigned line")
        ok("[stage7/FULL] REUSED_PASS" in text
           and "[stage7/FRONT] REUSED_RECLASSIFIED" in text,
           cid + " log FULL reuse + FRONT reclassify lines")
        ok("READY_FORMAL_V3=True" in text, cid + " log READY_FORMAL_V3=True")
    for fn in ("summary_gpu2.txt", "summary_gpu3.txt"):
        text = open(os.path.join(LOGS, fn),
                    encoding="utf-8", errors="replace").read()
        ok("QUEUE_DONE" in text and PINS["HEAD"] in text,
           fn + " queue done at frozen HEAD")

    print("FORMAL_EVALUATION_EVIDENCE_V3_VERIFIED checks=%d" % CHECKS)
    print("ranking_status=INCONCLUSIVE_FULL_TIE eligible=6/6 "
          "tie_group={BASE_GTRXL, RESET128} winner=null AUTH_V3=true")
    print("flip_policy=V3_GATE_GREEN gate_pass=true (G1-G16 all green)")
    print("composite: 13 episodes (9 transition∧defeat = NEG20 reproductions "
          "fixed); frozen V1 still raises NEG20 (repair additive)")
    print("V2 archive untouched: summary=3e818641… gate=51d3d6fb… "
          "(CLOSED_INCONCLUSIVE_PARTICIPATION, winner=null)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
