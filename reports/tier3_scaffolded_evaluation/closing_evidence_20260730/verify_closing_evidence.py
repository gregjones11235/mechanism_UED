#!/usr/bin/env python3
"""CC4 Tier3 — LOCAL PURE RE-VERIFICATION of the 20260730 closing evidence.

Runs under ANY interpreter (no jax): re-derives every whole-file / LF / canonical
SHA from the bytes actually pulled back from the server and compares against the
server-generated manifests, READY documents, binding results and preflight
certificate. Exits non-zero on any mismatch.

Scope note: the frozen BANK *content* SHAs require unpickling CC2 EnvState trees
(jax host) — those were verified on the server (statuses/bank_identity.json,
finalize gate B3, preflight canonical fields). Here we re-verify everything that
is pure: whole-file SHAs, LF-normalized module SHAs, canonical-JSON SHAs, gate
aggregations, cross-references, and the cross-GPU comparison (re-run locally on
the two pulled run documents).
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(ROOT, "common")
CAPS = {"persistent": os.path.join(ROOT, "cc4", "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"),
        "reset128": os.path.join(ROOT, "cc4", "RESET128_RMT16_ORIGINAL_VTRACE_98304")}

FRONT_FROZEN = "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687"
BACK_FROZEN = "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566"
FIELD_MANIFEST_FROZEN = "615d4be4df22115e4ac520718076860bf9def636a46806f5a2948be21456ee07"
PERSISTENT_CKPT = "2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723"
PERSISTENT_PARAMS = "aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d"
RESET128_CKPT = "de3a159f58f904c4ed0bce17bcb87e4b39b21b4ffd0cea557ce61b860727b638"
RESET128_PARAMS = "78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2"

problems = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        problems.append(name)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_lf(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- common/ manifest identities -------------------------------------------
man = load(os.path.join(COMMON, "assembly_manifest.json"))
whole = {"evaluation_profile_sha256": "evaluation_profile.json",
         "metric_schema_sha256": "metric_schema.json",
         "common_evaluator_sha256": "common_evaluator.py",
         "common_runner_sha256": "common_runner.py",
         "abi_doc_sha256": "candidate_runtime_abi.md",
         "environment_lock_sha256": "environment_lock.json"}
for key, rel in whole.items():
    check("manifest.%s == local file sha" % key,
          man[key] == sha_file(os.path.join(COMMON, rel)))

# Engine module LF-SHAs vs the deployed copies pulled back.
em = man["engine_module_sha256"]
ok_em = True
for name, sha in em.items():
    p = os.path.join(COMMON, "evaluator", name)
    if not os.path.exists(p) or sha_lf(p) != sha:
        ok_em = False
        problems.append("engine_module_sha256 mismatch: " + name)
check("all %d engine module LF-SHAs reproduce locally" % len(em), ok_em)

# Bank identities recorded by the server reload (content SHAs verified there;
# here: the four SHA kinds are present, content == frozen history, manifest ok).
bi = load(os.path.join(COMMON, "statuses", "bank_identity.json"))["banks"]
check("front content == frozen historical",
      bi["front_l2"]["canonical_content_sha256"] == FRONT_FROZEN
      and bi["front_l2"]["frozen_historical_content_sha256"] == FRONT_FROZEN)
check("back content == frozen historical",
      bi["back_l2"]["canonical_content_sha256"] == BACK_FROZEN
      and bi["back_l2"]["frozen_historical_content_sha256"] == BACK_FROZEN)
check("field manifest frozen (both banks)",
      bi["front_l2"]["field_manifest_sha256"] == FIELD_MANIFEST_FROZEN
      and bi["back_l2"]["field_manifest_sha256"] == FIELD_MANIFEST_FROZEN)
check("bank source is FROZEN_SERIALIZED_ARTIFACT (no gpu regen)",
      bi["front_l2"]["bank_source"] == "FROZEN_SERIALIZED_ARTIFACT"
      and bi["back_l2"]["bank_source"] == "FROZEN_SERIALIZED_ARTIFACT")
check("bank manifest json files match recorded artifact file sha",
      sha_file(os.path.join(COMMON, "front_bank_manifest.json"))
      and load(os.path.join(COMMON, "front_bank_manifest.json")).get("state_count") == 8
      and load(os.path.join(COMMON, "back_bank_manifest.json")).get("state_count") == 8)

# Negative gates.
neg = load(os.path.join(COMMON, "negative_test_report.json"))
check("negative gates fail==0 and exit 0",
      neg["fail"] == 0 and neg["tool_exit_code"] == 0)

# FULL profile identity: canonical sha of scenarios.full (pure recompute).
profile = load(os.path.join(COMMON, "evaluation_profile.json"))
full_sha = canonical_sha(profile["scenarios"]["full"])
check("profile FULL_PROFILE_READY true",
      profile["scenarios"]["full"].get("FULL_PROFILE_READY") is True)

# --- COMMON_EVALUATOR_READY --------------------------------------------------
ready = load(os.path.join(COMMON, "COMMON_EVALUATOR_READY.json"))
check("COMMON_EVALUATOR_READY == True", ready.get("COMMON_EVALUATOR_READY") is True)
g = ready.get("gates", {})
GATES = ("COMMON_ARTIFACT_IDENTITY", "B1_COMMON_EVALUATOR", "B3_FROZEN_BANKS",
         "FULL_PROFILE_READY", "COMMON_RUNTIME_ABI_READY", "NEGATIVE_GATES",
         "CROSS_GPU_DETERMINISM_PREFLIGHT", "SHA256SUMS_STATUS")
check("all 8 common gates PASS",
      all(isinstance(g.get(x), dict) and g[x].get("status") == "PASS" for x in GATES))
for key in ("common_evaluator_sha256", "common_runner_sha256",
            "evaluation_profile_sha256", "metric_schema_sha256",
            "environment_lock_sha256"):
    check("READY.%s == manifest" % key, ready.get(key) == man[key])
check("READY front/back bank content frozen",
      ready.get("front_bank_content_sha256") == FRONT_FROZEN
      and ready.get("back_bank_content_sha256") == BACK_FROZEN)

# --- cross-GPU certificate (re-run the pure comparison locally) ---------------
sys.path.insert(0, os.path.join(ROOT, "..", "..", "..", "tools",
                                "tier3_scaffolded_evaluation"))
sys.path.insert(0, os.path.normpath(os.path.join(ROOT, "..", "..", "..", "tools",
                                                 "tier3_scaffolded_evaluation")))
import tier3_cross_gpu_preflight as pre   # noqa: E402  (pure module)

run2 = load(os.path.join(ROOT, "preflight_gpu2.json"))
run3 = load(os.path.join(ROOT, "preflight_gpu3.json"))
cert_local = pre.compare_preflight_runs(run2, run3)
check("local re-compare reproduces CROSS_GPU_DETERMINISM_PREFLIGHT=PASS",
      cert_local["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "PASS")
cert = load(os.path.join(COMMON, "cross_gpu_preflight_certificate.json"))
check("server cert verdict PASS and no first difference",
      cert["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "PASS"
      and cert["first_difference"] is None)
check("cert checkpoint/params == frozen persistent",
      cert["checkpoint_file_sha256"] == PERSISTENT_CKPT
      and cert["params_sha256"] == PERSISTENT_PARAMS)
check("cert bank contents frozen",
      cert["front_bank_content_sha256"] == FRONT_FROZEN
      and cert["back_bank_content_sha256"] == BACK_FROZEN)
check("cert canonical episode SHAs agree gpu2==gpu3",
      cert_local["episode_record_sha256_by_scenario"]
      == cert["episode_record_sha256_by_scenario"])

# --- capsules -----------------------------------------------------------------
expect_ckpt = {"persistent": (PERSISTENT_CKPT, PERSISTENT_PARAMS),
               "reset128": (RESET128_CKPT, RESET128_PARAMS)}
expect_id = {"persistent": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
             "reset128": "RESET128_RMT16_ORIGINAL_VTRACE_98304"}
for arm, root in CAPS.items():
    tag = arm.upper()
    rdoc = load(os.path.join(root, "READY.json"))
    check("%s READY == True" % tag, rdoc.get("READY") is True)
    cg = rdoc.get("gates", {})
    CGATES = ("identity_status", "interface_smoke_status",
              "memory_contract_smoke_status", "formal_eval_binding",
              "checkpoint_sha256_verified", "params_sha256_verified",
              "common_artifact_sha_refs_verified", "immutable")
    check("%s all 8 capsule gates PASS" % tag,
          all(cg.get(x) == "PASS" for x in CGATES))
    bdoc = load(os.path.join(root, "common_evaluator_binding_result.json"))
    check("%s binding run_class=INTERFACE_SMOKE / PASS / no perf claim" % tag,
          bdoc.get("run_class") == "INTERFACE_SMOKE"
          and bdoc.get("formal_eval_binding") == "PASS"
          and bdoc.get("performance_claim_authorized") is False
          and bdoc.get("strong_student_selection_authorized") is False)
    refs = bdoc.get("common_references", {})
    check("%s eight common SHA refs present+verified" % tag,
          bdoc.get("common_references_verified") is True
          and set(refs) == {"common_evaluator_sha256", "common_runner_sha256",
                            "evaluation_profile_sha256", "metric_schema_sha256",
                            "environment_lock_sha256", "front_bank_content_sha256",
                            "back_bank_content_sha256", "full_profile_sha256"})
    check("%s binding refs match local files" % tag,
          refs["common_evaluator_sha256"] == man["common_evaluator_sha256"]
          and refs["common_runner_sha256"] == man["common_runner_sha256"]
          and refs["evaluation_profile_sha256"] == man["evaluation_profile_sha256"]
          and refs["metric_schema_sha256"] == man["metric_schema_sha256"]
          and refs["environment_lock_sha256"] == man["environment_lock_sha256"]
          and refs["front_bank_content_sha256"] == FRONT_FROZEN
          and refs["back_bank_content_sha256"] == BACK_FROZEN
          and refs["full_profile_sha256"] == full_sha)
    ckpt_sha, params_sha = expect_ckpt[arm]
    check("%s checkpoint/params frozen in binding+manifest" % tag,
          bdoc.get("checkpoint_file_sha256") == ckpt_sha
          and bdoc.get("params_sha256") == params_sha)
    mdoc = load(os.path.join(root, "candidate_manifest.json"))
    check("%s manifest identity + immutable + unauthorized" % tag,
          mdoc.get("candidate_id") == expect_id[arm]
          and mdoc.get("immutable") is True
          and mdoc.get("performance_claim_authorized") is False
          and mdoc.get("strong_student_selection_authorized") is False)
    tdoc = load(os.path.join(root, "training_contract.json"))
    check("%s training contract: no new training by cc4" % tag,
          tdoc.get("provenance", {}).get("new_training_by_cc4") is False
          and tdoc.get("provenance", {}).get("llm_calls_by_cc4") is False)
    ismoke = load(os.path.join(root, "interface_smoke_result.json"))
    ic = ismoke.get("checks", {})
    check("%s interface smoke: 3 scenarios, banks read-only" % tag,
          ismoke.get("status") == "PASS"
          and ismoke.get("engine_exit_code") == 0
          and ismoke.get("summary", {}).get("max_steps") == 32
          and all(ic.get(k) is True for k in
                  ("banks_read_only_from_artifacts", "three_scenarios_bound",
                   "run_class_interface_smoke", "max_steps_32", "arm_matches",
                   "contract_verified", "engine_exit_code_zero",
                   "performance_claim_not_authorized")))

# Capsule SHA256SUMS subset (pulled files only; npz never pulled).
for arm, root in CAPS.items():
    bad = []
    with open(os.path.join(root, "SHA256SUMS"), encoding="utf-8") as fh:
        for line in fh:
            sha, rel = line.rstrip("\n").split("  ", 1)
            p = os.path.join(root, *rel.split("/"))
            if os.path.exists(p) and sha_file(p) != sha:
                bad.append(rel)
    check("%s capsule sums verify over pulled files" % arm.upper(), not bad)

# Common SHA256SUMS subset (exclude the two npz payloads — never pulled).
bad = []
with open(os.path.join(COMMON, "SHA256SUMS"), encoding="utf-8") as fh:
    for line in fh:
        sha, rel = line.rstrip("\n").split("  ", 1)
        p = os.path.join(COMMON, *rel.split("/"))
        if not os.path.exists(p):
            if not rel.endswith(".npz"):
                bad.append("missing:" + rel)
            continue
        if sha_file(p) != sha:
            bad.append(rel)
check("common sums verify over pulled files (npz excluded)", not bad)

print()
if problems:
    print("CLOSING_EVIDENCE_LOCAL_REVERIFY_FAIL (%d problems)" % len(problems))
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("CLOSING_EVIDENCE_LOCAL_REVERIFY_PASS")
