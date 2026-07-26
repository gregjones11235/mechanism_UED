#!/usr/bin/env python
# CC4 remediation [4/6] part A: server raw-data sync manifest (BLOCKED) + missing-data update.
# Read-only; nothing synced (server unreachable). Records expected paths + minimum data needs.
import json, os, hashlib
BASE=os.getcwd()
OUT=open(os.path.join(BASE,"audit_outputs","_remediation_outdir.txt")).read().strip()
def J(p,o):
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,indent=2,ensure_ascii=False)

NA="NOT_AVAILABLE_LOCALLY"
manifest={
 "task":"GLOBAL_EVALUATION_REMEDIATION sync step (read-only)",
 "sync_status":"BLOCKED",
 "connectivity_probes":[
   {"target":"github.com:443 (raw TCP)","result":"REACHABLE","note":"raw TCP connect succeeds"},
   {"target":"github.com via git (proxy 127.0.0.1)","result":"UNREACHABLE","note":"proxy down: 'Failed to connect via 127.0.0.1'"},
   {"target":"github.com via git (direct, proxy bypassed)","result":"UNREACHABLE","note":"HTTPS handshake fails after 21s => git fetch BLOCKED"},
   {"target":"117.50.183.232:23 (known sync host = dreaming-in-code orchestration, NOT mechanism_UED)","result":"UNREACHABLE","note":"timeout"},
   {"target":"mechanism_UED experiment server (where W512/P7/P8/P9 raw data live)","result":"ADDRESS_UNKNOWN/UNREACHABLE","note":"no reachable address from this env; prior audit: ssh origin unreachable"}
 ],
 "consequence":"Cannot freeze origin Henry-branch HEAD (FU-8 BLOCKED) and cannot sync W512/P7/P8/P9 per-world data. Sync deferred to an environment with server access. No guessing of missing fields (NO_SILENT_ASSUMPTION).",
 "discipline":["read-only","server originals not moved/renamed/mtime-changed","no missing-field inference","record expected path + minimum data need when not found"],
 "expected_server_data":[
   {"experiment":"W512","expected_paths":["bakeoff_phase1/<arm>/eval/*.json (per-world)","bakeoff_phase1/<arm>/ckpt/manifest","bakeoff_phase1/shared/eval_a_side_unified.py + eval_w512_p2replay.py"],
    "minimum_data_needed":["success_per_world[256]","floor_per_world","death_per_world","episode_length_per_world","evaluator_sha256 (per file)","world seed/list/hash","checkpoint params_sha256","checkpoint manifest","training config","ACTUAL L_SEQ (resolve 129 vs 512)"],
    "arms":["W512_Persistent_PPO","W512_Reset128_PPO","W512_Persistent_P2Replay","W512_Reset128_P2Replay","BASELINE(101)","Control(93)"],
    "status":"NOT_SYNCED","blocks":["W512_REPRODUCIBILITY","CL-07","CL-08","CL-09","L_SEQ freeze"]},
   {"experiment":"P7","expected_paths":["gpu1_p7_egomap/eval/*.jsonl (per-world)","gpu1_p7_egomap/ckpt/params_*.pkl + carry_*.pkl"],
    "minimum_data_needed":["success_per_world[256]","action_mode actually used","evaluator_sha256","world seed/list (seed100000 line)","checkpoint manifest + params SHA (hashlib currently unused)"],
    "status":"NOT_SYNCED","blocks":["P7_REPRODUCIBILITY","CL-11"]},
   {"experiment":"P8","expected_paths":["gpu2_p8_longmemory/eval/{migration,final}/*.json (per-world)","gpu2_p8_longmemory/ckpt/full_state.pkl"],
    "minimum_data_needed":["migration(64w)+final(256w) per-world arrays","evaluator_sha256 (both files)","checkpoint SHA per chunk","distill-stage record"],
    "status":"NOT_SYNCED","blocks":["P8_REPRODUCIBILITY","CL-12"]},
   {"experiment":"P9","expected_paths":["gpu3_p9_authentic_reset/eval/*.json (per-world)","gpu3_p9_authentic_reset/ckpt/full_state.pkl","compare_resume artifact"],
    "minimum_data_needed":["success_per_world[256]","compare_resume A/B2 SHAs","checkpoint manifest","resume config","evaluator_sha256"],
    "status":"NOT_SYNCED","blocks":["P9_REPRODUCIBILITY","CL-13","continuation exact-resume verification"]},
   {"experiment":"GLOBAL","expected_paths":["origin Henry-branch HEAD","per-experiment running code SHA manifest","canonical L_SEQ frozen design file","all report-referenced checkpoint paths + SHA"],
    "minimum_data_needed":["origin HEAD commit SHA","code SHA manifest","canonical L_SEQ frozen file"],
    "status":"NOT_SYNCED","blocks":["HEAD freeze (FU-8)","lineage code-SHA","L_SEQ canonical (MISS-6)"]}
 ],
 "locally_available_for_recompute":["Phase2 unified per-world arrays (10 arms) in student_upgrade_wave1_4gpu/reports/phase2_unified_eval.json => recomputed in this step"]
}
J(os.path.join(OUT,"server_raw_data_manifest.json"),manifest)
# raw_data_manifest.json (audit deliverable alias)
J(os.path.join(OUT,"raw_data_manifest.json"),manifest)
# SHA256SUMS of synced data: none
open(os.path.join(OUT,"server_raw_data_SHA256SUMS"),"w",encoding="utf-8").write(
 "# NO files synced: experiment server + GitHub git access BLOCKED (see server_raw_data_manifest.json).\n"
 "# This file will be populated with '<sha256>  <relpath>' lines when read-only sync succeeds on a connected host.\n")
# missing data updated (carry forward audit MD-1..MD-10 with remediation status)
missing={
 "principle":"NO_RAW_DATA_NO_STRONG_CLAIM. Missing raw data caps claims; NOT a FAIL, NOT a performance failure.",
 "sync_attempted":"yes (single bounded probes; no polling)","sync_result":"BLOCKED",
 "items":[
   {"id":"MD-1","asset":"W512 6-arm per-world arrays + params SHA + manifest + actual L_SEQ","status":"NOT_SYNCED (server unreachable)","gates_blocked":["W512_REPRODUCIBILITY","L_SEQ freeze"],"remediation":"FU: read-only sync on connected host; then recompute via recompute_fixed.py"},
   {"id":"MD-2","asset":"P7 per-world + action_mode + evaluator SHA + manifest + params SHA","status":"NOT_SYNCED","gates_blocked":["P7_REPRODUCIBILITY"],"remediation":"sync + force params-SHA recording (hashlib unused)"},
   {"id":"MD-3","asset":"P8 migration+final per-world + evaluator SHAs + chunk SHA","status":"NOT_SYNCED","gates_blocked":["P8_REPRODUCIBILITY"],"remediation":"sync; hard-fail silent fallback (GATE4)"},
   {"id":"MD-4","asset":"P9 per-world + compare_resume artifact + manifest + resume config","status":"NOT_SYNCED","gates_blocked":["P9_REPRODUCIBILITY","continuation exact-resume"],"remediation":"sync + produce compare_resume via exact_resume_harness (authorized)"},
   {"id":"MD-5","asset":"RMT16 Phase4A/smoke/prefreeze/resume tests","status":"NOT_SYNCED + CC2 domain","gates_blocked":["RMT16 performance"],"remediation":"CC2 owns; CC4 read-only"},
   {"id":"MD-6","asset":"P2-Full-A formal per-world eval","status":"ABSENT","gates_blocked":["CL-10"],"remediation":"requires BASE_GTRXL+Replay control + training (not authorized)"},
   {"id":"MD-7","asset":"Phase3 attribution/longrun final per-world JSONs","status":"server-only","gates_blocked":["CL-15 numbers"],"remediation":"sync"},
   {"id":"MD-8","asset":"origin Henry-branch HEAD SHA + code SHA manifest","status":"BLOCKED (git access via proxy down + direct HTTPS fails)","gates_blocked":["HEAD freeze","lineage code-SHA"],"remediation":"FU-8 on connected host"},
   {"id":"MD-9","asset":"explicit world-set hashes in evaluator outputs","status":"tooling built (build_world_manifest.py); materialized hash needs JAX","gates_blocked":["GLOBAL_WORLD_SET_HASH_AVAILABLE","GATE2/3"],"remediation":"run builder --materialize on JAX host"},
   {"id":"MD-10","asset":"canonical L_SEQ (129 vs 512) frozen file","status":"NOT_FOUND locally; partial primary evidence: smoke/repro=129 (run_p2_full_smoke.py, posthoc), formal Level-B=512 (run_p2_full_smoke.py:66 comment 'formal run uses 512'; requirement matrix)","gates_blocked":["cross-line replay comparison"],"remediation":"freeze canonical on server"}
 ],
 "L_SEQ_primary_evidence_found":{
   "smoke_repro_L_SEQ_129":[".local/p2_full_audit/src/run_p2_full_smoke.py:66 L_SEQ=129 (comment: formal run uses 512)",".local/p2_full_audit/src_fullp2/run_p2_full_levelB.py:74 L_SEQ=129",".local/p2_full_audit/posthoc_attribution/posthoc_attribution.py:65 L_SEQ=129"],
   "formal_512_reference":[".local/p2_full_audit/src/run_p2_full_smoke.py:66 comment 'formal run uses 512'",".local/p2_full_audit/reports/p2_full_a_v1_henry_requirement_matrix.md (L_SEQ=129 loss-window documented; RMT16/P2-Full-A v2.1 doc specifies 512)"],
   "interpretation":"Both 129 and 512 exist as distinct run configs; canonical freeze must state which applies to each reported number. W512 repro used 129; RMT16 frozen config uses 512. Cross-line comparison requires this to be pinned."
 }
}
J(os.path.join(OUT,"global_missing_raw_data_updated.json"),missing)
J(os.path.join(OUT,"missing_data.json"),missing)
print("WROTE server_raw_data_manifest.json, raw_data_manifest.json, server_raw_data_SHA256SUMS, global_missing_raw_data_updated.json, missing_data.json")
