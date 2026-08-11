"""Real prompt-registry + frozen StudentProfile contract verification (spec §5).

OFFLINE verification of the bundle's prompt_registry.json and student_profile.json:

  * all 6 prompt_hash_sha256 == sha256(full_text)
  * arms B and C differ ONLY by the appended frozen StudentProfile block
    (byte-identical prefix/suffix per role; profile block identical across roles)
  * raw_summary / candidate_block / profile_json registry hashes reproduced
  * student_profile profile_hash_sha256 == sha256(canon_json(llm_interpretation))
  * a canonical StudentProfile CAN be built from machine_facts completion rates
    (conservative: unmeasured of the 67 -> SR 0.0)

Also reconciles the bundle's REAL role model pins against the canonical
ROLE_REGISTRY pins (mismatches are recorded, never silently reconciled).
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from d052.profiling.student_profile import StudentProfile, build_student_profile
from d052.reconciliation.real_bundle import load_bundle_json, sha256_hex
from d052.roles.protocol import ROLE_REGISTRY, RoleName

PROFILE_INSERT_HEADER = ("Modeler StudentProfile (frozen interpretation of the "
                         "above evidence):\n")


def verify_prompt_registry(registry: Optional[dict] = None) -> dict:
    """Verify the real prompt registry contract. Pure/offline; returns checks."""
    pr = registry if registry is not None else load_bundle_json("prompt_registry.json")
    prompts = pr["prompts"]
    checks: Dict[str, object] = {}

    # 1) six prompt hashes over full_text
    checks["all_6_prompt_hashes_match"] = all(
        sha256_hex(p["full_text"]) == p["prompt_hash_sha256"] for p in prompts.values())

    # 2) B vs C: only the frozen profile block is inserted (per role)
    insertion_ok, blocks = True, {}
    for role in ("tutor", "critic", "explorer"):
        b = prompts[f"B_{role}"]["full_text"]
        c = prompts[f"C_{role}"]["full_text"]
        i = next(k for k in range(min(len(b), len(c))) if b[k] != c[k])
        j = len(c) - (len(b) - i)
        inserted = c[i:j]
        ok = (c[:i] == b[:i] and c[j:] == b[i:]
              and inserted.startswith(PROFILE_INSERT_HEADER))
        pj = inserted[len(PROFILE_INSERT_HEADER):]
        if pj.endswith("\n\n"):
            pj = pj[:-2]  # trailing separator belongs to the prompt glue
        blocks[role] = pj
        insertion_ok = insertion_ok and ok
    checks["B_C_only_difference_is_appended_frozen_profile"] = insertion_ok
    checks["profile_block_byte_identical_across_roles"] = len(set(blocks.values())) == 1

    # 3) registry sub-hashes reproduced from the rendered prompts
    bt = prompts["B_tutor"]["full_text"]
    raw_hdr = "Student information (deterministic, computed by code):\n"
    cand_hdr = "\n\nCandidates to evaluate"
    s1, s2 = bt.index(raw_hdr) + len(raw_hdr), bt.index(cand_hdr)
    checks["raw_summary_sha256_reproduced"] = (
        sha256_hex(bt[s1:s2]) == pr["raw_summary_sha256"])
    seg = bt[bt.index(cand_hdr) + 2:]
    cands_txt = seg[seg.index("\n") + 1: seg.index("\n\nEvaluate ALL 32")]
    checks["candidate_block_sha256_reproduced"] = (
        sha256_hex(cands_txt) == pr["candidate_block_sha256"])

    # 4) registry profile_json_sha256 is over the student_profile.json serialization
    #    of the SAME object embedded in arm C (deep-equal; recorded key order)
    sp = load_bundle_json("student_profile.json")
    checks["profile_json_sha256_reproduced"] = (
        sha256_hex(json.dumps(sp["llm_interpretation"], ensure_ascii=False))
        == pr["profile_json_sha256"])
    checks["embedded_profile_deep_equal_llm_interpretation"] = (
        json.loads(blocks["tutor"]) == sp["llm_interpretation"])

    checks["prompt_version"] = pr["prompt_version"]
    checks["ok"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def verify_student_profile(profile: Optional[dict] = None) -> dict:
    """Verify profile hashes + build the canonical StudentProfile from facts."""
    sp = profile if profile is not None else load_bundle_json("student_profile.json")
    ll = sp["llm_interpretation"]
    canon_ll = json.dumps(ll, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    checks = {
        "profile_hash_sha256_reproduced":
            sha256_hex(canon_ll) == sp["profile_hash_sha256"],
        "profile_hash_formula": "sha256(canon_json(llm_interpretation))",
    }
    # canonical profile from machine facts (completion_rate as SR proxy)
    sr = {p["achievement_id"].lower(): float(p["completion_rate"])
          for p in sp["machine_facts"]["per_achievement_completion"]}
    built = build_student_profile(sr)
    checks["canonical_profile_built"] = True
    checks["canonical_profile_summary"] = {
        "measured_count": built.measured_count,
        "mastered_count": built.mastered_count,
        "proficient_count": built.proficient_count,
        "overall_mastery": round(built.overall_mastery, 6)}
    checks["insufficient_evidence_kept_null"] = all(
        s.get("best_sr") is None and s.get("recent_delta") is None
        for s in ll["skills"])
    checks["ok"] = (checks["profile_hash_sha256_reproduced"]
                    and checks["canonical_profile_built"]
                    and checks["insufficient_evidence_kept_null"])
    return checks, built


def reconcile_role_pins(registry: Optional[dict] = None) -> dict:
    """Bundle REAL model pins vs canonical ROLE_REGISTRY pins (record divergence)."""
    pr = registry if registry is not None else load_bundle_json("prompt_registry.json")
    # the real pins are recorded per judgment; the registry fixes models in protocol
    # invariants, so read them from the bundle protocol.json controlled_invariants
    proto = load_bundle_json("protocol.json")
    models_used = proto["controlled_invariants"]["models"]
    rows = {}
    for role in ("tutor", "critic", "explorer"):
        canon = ROLE_REGISTRY[RoleName(role)]
        rows[role] = {"bundle_model": models_used[role],
                      "canonical_provider": canon.provider,
                      "canonical_exact_model_id": canon.exact_model_id,
                      "exact_model_id_match":
                          canon.exact_model_id in models_used[role]}
    return {"rows": rows,
            "all_exact_match": all(r["exact_model_id_match"] for r in rows.values()),
            "disposition": "real ids preserved losslessly via RoleJudgment optional "
                           "provenance; ROLE_REGISTRY divergence affects ONLY future "
                           "LLM calls (none this round); flagged for director"}
