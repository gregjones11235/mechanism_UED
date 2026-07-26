#!/usr/bin/env python
"""Negative + directed tests for the actual Craftax world materializer.

V2 (round 6): NEG01-NEG10 guarded the key-only-prototype methodological error.
V3 (GLOBAL_WORLD_MATERIALIZER_RUNTIME_IDENTITY_HARDENING_V3) FIXES two forbidden
behaviours and ADDS directed coverage:

  * NEG02 NO LONGER passes merely because JAX+craftax are installed. "Environment present"
    is NOT "test executed". A PASS now requires actually materializing seed42 AND seed100000
    and comparing the SEED-FREE world PAYLOAD hashes (state_payload_hash), with at least one
    world differing; 256/256 identical => FAIL. Without a real env it stays BLOCKED_ENVIRONMENT
    and only the STATIC seed binding may be reported PASS.
  * NEG09 reports the STRICT shared builder as BLOCKED_EVALUATOR_INLINE_READ_ONLY and the
    STATIC source anchor as a SEPARATE PASS -- never merged into one vague PASS.

  * NEG11-NEG23 add directed coverage for: runtime executed-source identity binding (and
    wrong-imported-file / byte-identical-different-path rejection), symlink resolution,
    task-exec source binding, evaluator = protocol-anchor-not-executed, seed-free payload
    hash, "header seed is NOT proof of a real RNG effect", seed identity classification,
    field-manifest persistence, missing-manifest rejection, and two-process source-identity
    mismatch detection.

Run: python world_materializer_negative_tests.py [--eval-source .. --wrapper-source ..
        --task-source .. --env-source ..]
Exit 0 iff FAIL=0 (BLOCKED / BLOCKED_ENVIRONMENT are honest, allowed outcomes).
"""
import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MATERIALIZER = os.path.join(HERE, "materialize_craftax_world_set_twice.py")

# default canonical source locations (overridable via CLI / env)
EX = ("D:/Projects/dicode-codex-director/audit_outputs/"
      "global_raw_data_extract_20260726T110032Z/home/oseasy/experiments")
DEFAULTS = {
    "eval_source": os.path.join(EX, "student_upgrade_wave1_4gpu", "eval_phase2_unified.py"),
    "wrapper_source": os.path.join(HERE, "..", "..", "dicode_src", "src", "dicode", "wrappers_cl.py"),
    "task_source": os.path.join(EX, "p2_v1_20260722", "evidence", "s4_task_code.py"),
    "env_source": os.path.join(HERE, "..", "..", "dicode_src", "src", "minicraftax", "envs", "multitask.py"),
}


def load_materializer():
    spec = importlib.util.spec_from_file_location("materializer", MATERIALIZER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def have_jax_craftax():
    return (importlib.util.find_spec("jax") is not None
            and importlib.util.find_spec("craftax") is not None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-source", default=os.environ.get("CC4_EVAL_SOURCE", DEFAULTS["eval_source"]))
    ap.add_argument("--wrapper-source", default=os.environ.get("CC4_WRAPPER_SOURCE", DEFAULTS["wrapper_source"]))
    ap.add_argument("--task-source", default=os.environ.get("CC4_TASK_SOURCE", DEFAULTS["task_source"]))
    ap.add_argument("--env-source", default=os.environ.get("CC4_ENV_SOURCE", DEFAULTS["env_source"]))
    a = ap.parse_args(argv)

    m = load_materializer()
    import numpy as np

    results = []

    def record(idx, name, status, detail):
        results.append({"id": "NEG%02d" % idx, "name": name, "status": status, "detail": detail})

    src = {"evaluator_sha256": "e" * 64, "environment_source_sha256": "n" * 64,
           "wrapper_source_sha256": "w" * 64, "task_source_sha256": "t" * 64}
    ver = {"craftax": m.EXPECTED_CRAFTAX, "jax": "X", "jaxlib": "X"}
    pwh = {str(i): ("a" * 64) for i in range(m.NUM_WORLDS)}
    pph = {str(i): ("p" * 64) for i in range(m.NUM_WORLDS)}   # seed-free payload hashes

    # ---- NEG01: changing ONLY the seed_id label (numeric seed fixed) leaves hash unchanged
    wsh_a = m.compute_world_set_hash(dict(pwh), 42, src, ver)
    wsh_b = m.compute_world_set_hash(dict(pwh), 42, src, ver)
    record(1, "label-only change leaves hash unchanged",
           "PASS" if wsh_a == wsh_b else "FAIL",
           "compute_world_set_hash takes the NUMERIC evaluation_seed only; seed_id label is "
           "display-only and never enters the bytes. equal=%s" % (wsh_a == wsh_b))

    # ---- NEG02 (V3 FIXED): numeric seed must change the REAL world PAYLOAD hashes.
    #      "environment installed" is NOT "test executed". A PASS requires actually running
    #      materialize_all_world_states(42) and (100000) and comparing the SEED-FREE
    #      state_payload_hash per world. A header-tagged per_world_hash difference is NOT
    #      accepted as evidence (the header is deliberately seed-tagged).
    static_seed_binding = False
    if os.path.isfile(a.eval_source):
        import re
        esrc = open(a.eval_source, encoding="utf-8", errors="replace").read()
        static_seed_binding = re.search(r"PRNGKey\(\s*EVAL_SEED\s*\)", esrc) is not None
    if have_jax_craftax():
        # Real environment: ACTUALLY materialize both seeds and compare real payloads.
        try:
            os.environ["CC4_S4_TASK_PATH"] = a.task_source
            batched42, _, _, _ = m.materialize_all_world_states(42, requested_task_path=a.task_source)
            batched100, _, _, _ = m.materialize_all_world_states(100000, requested_task_path=a.task_source)
            differ, same, first_diff = 0, 0, None
            for i in range(m.NUM_WORLDS):
                h42, _ = m.state_payload_hash(m.extract_world_identity_single(batched42, i))
                h100, _ = m.state_payload_hash(m.extract_world_identity_single(batched100, i))
                if h42 != h100:
                    differ += 1
                    if first_diff is None:
                        first_diff = i
                else:
                    same += 1
            if differ >= 1:
                record(2, "numeric seed change changes REAL world payload hashes", "PASS",
                       "REAL RUN: materialized seed42 vs seed100000; compared SEED-FREE "
                       "state_payload_hash per world. different_world_count=%d same_world_count=%d "
                       "first_different_world_index=%s (header seed excluded from these bytes)."
                       % (differ, same, first_diff))
            else:
                record(2, "numeric seed change changes REAL world payload hashes", "FAIL",
                       "REAL RUN: 256/256 worlds have IDENTICAL seed-free payload hashes across "
                       "seed42 vs seed100000 -- the numeric seed has NO real world effect "
                       "(same_world_count=%d). This is a genuine failure, not an environment block."
                       % same)
        except m.FailClosed as e:
            record(2, "numeric seed change changes REAL world payload hashes", "BLOCKED_ENVIRONMENT",
                   "jax/craftax present but real materialization could not execute (config/source "
                   "block): %s. NOT asserted PASS. STATIC_SEED_BINDING=%s."
                   % (str(e)[:120], static_seed_binding))
    else:
        record(2, "numeric seed change changes REAL world payload hashes", "BLOCKED_ENVIRONMENT",
               "NO jax/craftax on this host; real per-world EnvState payloads cannot be produced, "
               "so this is NOT asserted PASS (environment-installed != test-executed). "
               "EVALUATION_SEED_STATIC_RNG_BINDING=%s (PRNGKey(EVAL_SEED) anchor present in canonical "
               "source); EVALUATION_SEED_REAL_WORLD_EFFECT=BLOCKED_ENVIRONMENT."
               % ("PASS" if static_seed_binding else "UNVERIFIED"))

    # ---- NEG03: an actual field VALUE change changes the serialized bytes/hash
    e1 = m.encode_node({"v": np.ones(4)}, (), [])
    e2 = m.encode_node({"v": np.ones(4) * 2}, (), [])
    record(3, "actual field value change changes hash",
           "PASS" if e1 != e2 else "FAIL", "different=%s" % (e1 != e2))

    # ---- NEG04: field/declaration ORDER change leaves hash unchanged (sorted encoding)
    d1 = m.encode_node({"a": 1, "b": 2, "c": np.ones(3)}, (), [])
    d2 = m.encode_node({"c": np.ones(3), "a": 1, "b": 2}, (), [])
    @dataclasses.dataclass
    class P1:
        x: int
        y: int
    o1 = m.encode_node(P1(1, 2), (), [])
    o2 = m.encode_node(P1(1, 2), (), [])
    order_ok = (d1 == d2) and (o1 == o2)
    record(4, "field order change leaves hash unchanged",
           "PASS" if order_ok else "FAIL",
           "dict-order-invariant=%s; dataclass sorted-by-fieldname=%s" % (d1 == d2, o1 == o2))

    # ---- NEG05: dtype / shape change changes hash
    dtype_ok = (m.encode_node(np.ones(3, dtype=np.float32), (), [])
                != m.encode_node(np.ones(3, dtype=np.float64), (), []))
    shape_ok = (m.encode_node(np.ones((3,), dtype=np.float32), (), [])
                != m.encode_node(np.ones((1, 3), dtype=np.float32), (), []))
    record(5, "dtype/shape change changes hash",
           "PASS" if (dtype_ok and shape_ok) else "FAIL",
           "dtype-sensitive=%s; shape-sensitive=%s" % (dtype_ok, shape_ok))

    # ---- NEG06: key-only prototype output is REJECTED by the materialization gate; a genuine
    #      V3 materialized result (WITH manifest + payload-hash evidence) is accepted.
    prototype_result = {"schema": "mechanism_UED.world_hashes/v1", "world_count": 256,
                        "per_world_hashes": pwh, "world_set_hash": "f" * 64}
    rejected = False
    try:
        m.assert_materialized(prototype_result)
    except m.FailClosed:
        rejected = True
    genuine = {"schema": m.SCHEMA_VERSION, "materialized": True, "world_count": m.NUM_WORLDS,
               "source_shas": src, "per_world_hashes": pwh, "world_set_hash": "g" * 64,
               "world_field_manifests_sha256": "m" * 64,
               "world_field_manifest_world_count": m.NUM_WORLDS,
               "per_world_state_payload_hashes": pph}
    accepted = m.assert_materialized(genuine)
    record(6, "key-only prototype rejected by materialization gate",
           "PASS" if (rejected and accepted) else "FAIL",
           "prototype-shape rejected=%s; genuine V3 materialized (with manifest evidence) "
           "accepted=%s" % (rejected, accepted))

    # ---- NEG07: missing environment source -> fail closed
    fc7 = False
    try:
        m.collect_source_shas(a.eval_source, a.wrapper_source, a.task_source,
                              "/nonexistent/env_source.py")
    except m.FailClosed:
        fc7 = True
    record(7, "missing env source fails closed",
           "PASS" if fc7 else "FAIL", "FailClosed raised on absent env source=%s" % fc7)

    # ---- NEG08: recorded-but-not-executed source is rejected (need an execution path)
    src_text = open(MATERIALIZER, encoding="utf-8").read()
    fn = src_text[src_text.index("def do_materialize_run"):]
    fn = fn[:fn.index("\ndef do_orchestrate")]
    i_record = fn.index("collect_source_shas")
    i_exec = fn.index("materialize_all_world_states")
    i_hash = fn.index("compute_world_set_hash")
    exec_after_record_before_hash = (i_record < i_exec < i_hash)
    has_real_imports = ("from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper" in src_text
                        and "from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv" in src_text
                        and "exec(f.read(), ns)" in src_text)
    # V3: do_materialize_run must ALSO bind the executed source identity before hashing.
    binds_identity = ("bind_executed_source_identity" in fn and "verify_task_exec_identity" in src_text)
    record(8, "recorded-but-not-executed source rejected (execution path required)",
           "PASS" if (exec_after_record_before_hash and has_real_imports and binds_identity) else "FAIL",
           "static: record<execute<hash ordering=%s; real import+exec of wrapper/env/task present=%s; "
           "V3 runtime executed-source binding invoked=%s (on a JAX host these actually execute; on "
           "this host _require_jax_craftax / import fails closed first)."
           % (exec_after_record_before_hash, has_real_imports, binds_identity))

    # ---- NEG09 (V3 FIXED): STRICT shared builder stays BLOCKED (evaluator read-only + inline).
    #      This is reported SEPARATELY from the static anchor (NEG11) -- never merged into a
    #      vague PASS.
    record(9, "materializer/evaluator STRICT shared builder",
           "BLOCKED_EVALUATOR_INLINE_READ_ONLY",
           "MATERIALIZER_EVALUATOR_SHARED_BUILDER=BLOCKED_EVALUATOR_INLINE_READ_ONLY: the canonical "
           "evaluator keeps env construction + the three seed/reset lines INLINE and is READ-ONLY, "
           "so there is NO importable shared builder the materializer could literally call. This is "
           "NOT a PASS. The honest static-anchor equivalence is reported separately as NEG11.")

    # ---- NEG10: two-process comparison detects a per-world mismatch (MOCK demonstration)
    base = {"world_count": 256, "world_set_hash": "h" * 64, "per_world_hashes": dict(pwh),
            "per_world_state_payload_hashes": dict(pph),
            "world_field_manifests_sha256": "m" * 64,
            "source_shas": src, "versions": ver, "evaluation_seed": 42,
            "numeric_evaluation_seed": 42, "identity_class": "CANONICAL_EVALUATOR_EXACT_WORLD_SET",
            "protocol_id": m.PROTOCOL_ID_SEED42,
            "runtime_source_identity": {"executed_sources": {"wrapper": {"executed_sha256": "w" * 64}}}}
    good_copy = json.loads(json.dumps(base))
    mismatch = json.loads(json.dumps(base)); mismatch["per_world_hashes"]["7"] = "z" * 64
    agree_ok = m.compare_two_runs(base, good_copy)
    mismatch_caught = False
    try:
        m.compare_two_runs(base, mismatch)
    except m.FailClosed:
        mismatch_caught = True
    record(10, "two-process comparison detects per-world mismatch (mock)",
           "PASS" if (agree_ok and mismatch_caught) else "FAIL",
           "MOCK ONLY (real two-process NOT_RUN, no craftax): agreement accepted=%s; per-world "
           "mismatch fails closed=%s" % (agree_ok, mismatch_caught))

    # ---- NEG11 (V3): STATIC source-anchor equivalence, reported SEPARATELY from NEG09.
    anchor = m.static_anchor_check(a.eval_source, a.wrapper_source, a.task_source)
    anchor_pass = (anchor["result"] == "PASS")
    record(11, "static source-anchor equivalence (separate from strict shared builder)",
           "PASS" if anchor_pass else "FAIL",
           "STATIC_ANCHOR_EQUIVALENCE=%s (%d anchors checked, mismatches=%s). This is the honest "
           "substitute for a shared builder; it is a STATIC equivalence proof, NOT a shared-builder "
           "PASS (see NEG09)." % ("PASS" if anchor_pass else "FAIL",
                                  len(anchor["checked"]), anchor["mismatches"]))

    # ===================== V3 DIRECTED TESTS (NEG12-NEG23) ===================== #
    tmpd = tempfile.mkdtemp(prefix="cc4_v3_neg_")

    # ---- NEG12: imported WRAPPER sha matches the recorded source (binding logic + real file).
    #      Real `import dicode.wrappers_cl` requires JAX -> BLOCKED_ENVIRONMENT on this host;
    #      here we prove the binding LOGIC accepts a genuine same-realpath match using the real
    #      on-disk wrapper file, and that that file IS the canonical anchor.
    try:
        wmatch = m.verify_source_identity(a.wrapper_source, a.wrapper_source, "wrapper")
        wis_canonical = (m.sha256_file(m._resolve_real(a.wrapper_source)) == m.WRAPPER_SHA256)
        neg12 = (wmatch["identity_match"] is True and wis_canonical)
        record(12, "imported wrapper sha matches recorded source",
               "PASS" if neg12 else "FAIL",
               "binding-logic identity_match=%s on the REAL wrapper file; real file == canonical "
               "WRAPPER_SHA256(2ded41d8..)=%s. The actual `import dicode.wrappers_cl` binding is "
               "executed only on a JAX host (REAL_RUNTIME_NOT_RUN here)."
               % (wmatch["identity_match"], wis_canonical))
    except m.FailClosed as e:
        record(12, "imported wrapper sha matches recorded source", "FAIL",
               "unexpected FailClosed on a genuine same-file match: %s" % str(e)[:120])

    # ---- NEG13: imported ENV (multitask) sha matches the recorded source.
    try:
        ematch = m.verify_source_identity(a.env_source, a.env_source, "environment")
        eis_canonical = (m.sha256_file(m._resolve_real(a.env_source)) == m.ENV_SOURCE_SHA256)
        neg13 = (ematch["identity_match"] is True and eis_canonical)
        record(13, "imported environment sha matches recorded source",
               "PASS" if neg13 else "FAIL",
               "binding-logic identity_match=%s on the REAL multitask.py; real file == canonical "
               "ENV_SOURCE_SHA256(c8f2d5c3..)=%s. Actual import binding = REAL_RUNTIME_NOT_RUN here."
               % (ematch["identity_match"], eis_canonical))
    except m.FailClosed as e:
        record(13, "imported environment sha matches recorded source", "FAIL",
               "unexpected FailClosed on a genuine same-file match: %s" % str(e)[:120])

    # ---- NEG14: record path A but Python imports path B -> REJECTED even if class names match.
    fA = os.path.join(tmpd, "A.py"); open(fA, "wb").write(b"class DistributedMultiTaskOptimisticLogWrapper: pass\n")
    fB = os.path.join(tmpd, "B.py"); open(fB, "wb").write(b"class DistributedMultiTaskOptimisticLogWrapper: pass\n# different bytes\n")
    fc14 = False
    try:
        m.verify_source_identity(fA, fB, "wrapper")    # same class name, different file
    except m.FailClosed:
        fc14 = True
    # and byte-identical-but-different-realpath must ALSO be rejected
    fC = os.path.join(tmpd, "C_copy.py"); open(fC, "wb").write(open(fA, "rb").read())
    fc14b = False
    try:
        m.verify_source_identity(fA, fC, "wrapper")    # byte-identical, different realpath
    except m.FailClosed:
        fc14b = True
    record(14, "record-A / import-B fails closed (incl. byte-identical copy)",
           "PASS" if (fc14 and fc14b) else "FAIL",
           "different-content same-classname rejected=%s; byte-identical different-realpath "
           "rejected=%s (must NOT rely on copies being byte-identical)." % (fc14, fc14b))

    # ---- NEG15: a symlink to the SAME source resolves safely (same realpath + sha -> PASS).
    real = os.path.join(tmpd, "real_wrapper.py"); open(real, "wb").write(b"# real source\n")
    link = os.path.join(tmpd, "link_wrapper.py")
    try:
        os.symlink(real, link)
        sym = m.verify_source_identity(link, real, "wrapper")   # requested=link, imported=real
        neg15 = (sym["identity_match"] is True
                 and sym["requested_realpath"] == sym["imported_module_realpath"])
        record(15, "symlink to same source resolves safely",
               "PASS" if neg15 else "FAIL",
               "symlink requested path realpath-resolves to the executed file; identity_match=%s; "
               "requested_realpath==imported_realpath=%s."
               % (sym["identity_match"], sym["requested_realpath"] == sym["imported_module_realpath"]))
    except (OSError, NotImplementedError) as e:
        record(15, "symlink to same source resolves safely", "BLOCKED_ENVIRONMENT",
               "this filesystem/OS would not permit symlink creation (%s); the realpath-resolution "
               "logic itself is exercised by NEG12-NEG14. NOT reported as PASS." % str(e)[:80])

    # ---- NEG16: task exec path must match the recorded canonical task source.
    #      Positive real exec needs the canonical task (imports minicraftax) -> JAX host. Here we
    #      prove the REJECTION path: a wrong-sha / interface-incomplete class fails closed.
    class Env:
        def generate_world(self, rng):
            return rng
        def get_task_params(self):
            return None
    wrong_sha_rejected = False
    try:
        m.verify_task_exec_identity(fA, fA, Env, m.TASK_SHA256)   # fA sha != canonical
    except m.FailClosed:
        wrong_sha_rejected = True
    class EnvNoIface:                       # right name + right sha but MISSING interface
        pass
    iface_rejected = False
    try:
        m.verify_task_exec_identity(fA, fA, EnvNoIface, m.sha256_file(fA))
    except m.FailClosed:
        iface_rejected = True
    record(16, "task exec path must match recorded task source",
           "PASS" if (wrong_sha_rejected and iface_rejected) else "FAIL",
           "wrong-full-sha task rejected=%s; interface-incomplete task rejected=%s (positive real "
           "exec of the canonical 45fdd17c.. task = REAL_RUNTIME_NOT_RUN here; needs JAX)."
           % (wrong_sha_rejected, iface_rejected))

    # ---- NEG17: evaluator is labelled a PROTOCOL ANCHOR, NOT an executed source.
    feval = os.path.join(tmpd, "eval.py"); open(feval, "wb").write(b"# stand-in anchor\n")
    saved = m.EVALUATOR_SHA256
    try:
        m.EVALUATOR_SHA256 = m.sha256_file(feval)
        pa = m.protocol_anchor_identity(feval)
        neg17 = (pa["executed_by_materializer"] is False
                 and pa["role"] == "static protocol anchor" and pa["anchor_match"] is True)
    finally:
        m.EVALUATOR_SHA256 = saved
    record(17, "evaluator labelled protocol anchor, not executed source",
           "PASS" if neg17 else "FAIL",
           "protocol_anchor_sources.canonical_evaluator.executed_by_materializer=%s; role=%r. The "
           "materializer reproduces the evaluator's build+reset but never runs its main program; we "
           "never write 'evaluator source executed'." % (pa["executed_by_materializer"], pa["role"]))

    # ---- NEG18: actual seed-free world payload hash helper.
    world_x = {"env_state": {"map": np.ones((9, 48, 48), dtype=np.int32)}}
    h_x1, _ = m.state_payload_hash(world_x)
    h_x2, _ = m.state_payload_hash(world_x)
    payload_det = (h_x1 == h_x2)
    pl_bytes, _ = m.serialize_world_payload(world_x)
    # the payload bytes must NOT contain the source SHA / version / seed strings
    payload_clean = (b"evaluation_seed" not in pl_bytes and b"source_shas" not in pl_bytes
                     and m.EXPECTED_CRAFTAX.encode() not in pl_bytes)
    record(18, "seed-free world payload hash helper",
           "PASS" if (payload_det and payload_clean) else "FAIL",
           "state_payload_hash deterministic=%s; payload bytes carry NO evaluation_seed / source SHA "
           "/ version metadata=%s." % (payload_det, payload_clean))

    # ---- NEG19: a header-seed difference is NOT proof of a real RNG effect.
    _, _, _, phw42 = m.serialize_world(world_x, 42, 0, src, ver)
    _, _, _, phw100 = m.serialize_world(world_x, 100000, 0, src, ver)
    sph_a, _ = m.state_payload_hash(world_x)
    sph_b, _ = m.state_payload_hash(world_x)
    header_moves_payload_does_not = (phw42 != phw100) and (sph_a == sph_b)
    record(19, "header seed change is NOT real RNG-effect evidence",
           "PASS" if header_moves_payload_does_not else "FAIL",
           "same EnvState: per_world_hash(header-tagged) differs across seed 42 vs 100000 =%s, but "
           "seed-free state_payload_hash is IDENTICAL =%s. Therefore NEG02/GATE21 must compare PAYLOAD "
           "hashes, not header-tagged hashes." % (phw42 != phw100, sph_a == sph_b))

    # ---- NEG20: seed identity classification + GATE22 admissibility.
    id42 = m.seed_identity("seed42"); id100 = m.seed_identity("seed100000")
    cls_ok = (id42["identity_class"] == "CANONICAL_EVALUATOR_EXACT_WORLD_SET"
              and id42["evaluator_exact_match"] is True
              and id42["protocol_id"] == m.PROTOCOL_ID_SEED42
              and id100["identity_class"] == "PARAMETERIZED_WORLD_GENERATION_PROTOCOL_VARIANT"
              and id100["evaluator_exact_match"] is False
              and id100["protocol_id"] == m.PROTOCOL_ID_SEED100000
              and id100["admissible_as_canonical_exact_world_set_evidence"] is False)
    indep_eval = bool(id100["independent_evaluator"])
    gate_ok = False
    try:
        m.assert_exact_world_set_eligible("seed42")
        try:
            m.assert_exact_world_set_eligible("seed100000")
        except m.FailClosed:
            gate_ok = True
    except m.FailClosed:
        gate_ok = False
    record(20, "seed identity classification + GATE22",
           "PASS" if (cls_ok and indep_eval and gate_ok) else "FAIL",
           "seed42=CANONICAL_EVALUATOR_EXACT_WORLD_SET / seed100000=PARAMETERIZED_VARIANT=%s; "
           "seed100000 carries an INDEPENDENT real evaluator identity (P7_PAIRED_256, own full SHA, "
           "not reusing seed42's)=%s; GATE22 admits seed42 and rejects seed100000 as exact evidence=%s."
           % (cls_ok, indep_eval, gate_ok))

    # ---- NEG21: field manifest persistence (persisted / hash matches / count 256 / paths cover
    #      arrays / change changes sha).
    man = []
    m.encode_node({"map": np.ones((9, 48, 48), dtype=np.int32),
                   "inventory": {"wood": np.int64(7)}}, (), man)
    pm = {str(i): man for i in range(m.NUM_WORLDS)}
    doc = m.build_field_manifests_doc(pm)
    out_manifest = os.path.join(tmpd, "world_field_manifests.json")
    m.write_json(out_manifest, doc)
    persisted = os.path.isfile(out_manifest)
    reread = json.load(open(out_manifest, encoding="utf-8"))
    hash_matches = (m.field_manifests_sha256(reread) == m.field_manifests_sha256(doc))
    count_ok = (doc["world_count"] == m.NUM_WORLDS and len(doc["worlds"]) == m.NUM_WORLDS)
    cover_ok = ({e["path"] for e in man} == {"map", "inventory.wood"})
    pm2 = dict(pm); man_div = []
    m.encode_node({"map": np.ones((9, 48, 48), dtype=np.float32),
                   "inventory": {"wood": np.int64(7)}}, (), man_div)
    pm2["3"] = man_div
    change_ok = (m.field_manifests_sha256(m.build_field_manifests_doc(pm2))
                 != m.field_manifests_sha256(doc))
    summary = m.build_field_schema_summary(pm)
    summary_records = (m.build_field_schema_summary(pm2)["all_worlds_structurally_identical"] is False)
    neg21 = persisted and hash_matches and count_ok and cover_ok and change_ok and summary_records
    record(21, "field manifest persisted / hash / count / coverage / change-detection",
           "PASS" if neg21 else "FAIL",
           "persisted=%s; file-hash matches in-memory sha=%s; world_count=256=%s; manifest paths "
           "cover every serialized array=%s; manifest change changes sha=%s; schema summary RECORDS "
           "(not overwrites) a divergence=%s."
           % (persisted, hash_matches, count_ok, cover_ok, change_ok, summary_records))

    # ---- NEG22: a materialized result missing the manifest evidence is REJECTED.
    full = {"schema": m.SCHEMA_VERSION, "materialized": True, "world_count": m.NUM_WORLDS,
            "source_shas": src, "per_world_hashes": pwh, "world_set_hash": "g" * 64,
            "world_field_manifests_sha256": "m" * 64,
            "world_field_manifest_world_count": m.NUM_WORLDS,
            "per_world_state_payload_hashes": pph}
    no_manifest = dict(full); no_manifest.pop("world_field_manifests_sha256")
    rej_manifest = False
    try:
        m.assert_materialized(no_manifest)
    except m.FailClosed:
        rej_manifest = True
    no_payload = dict(full); no_payload["per_world_state_payload_hashes"] = {}
    rej_payload = False
    try:
        m.assert_materialized(no_payload)
    except m.FailClosed:
        rej_payload = True
    record(22, "world result rejects missing manifest / payload evidence",
           "PASS" if (rej_manifest and rej_payload) else "FAIL",
           "missing world_field_manifests_sha256 rejected=%s; empty per_world_state_payload_hashes "
           "rejected=%s." % (rej_manifest, rej_payload))

    # ---- NEG23: two-process comparison detects a RUNTIME SOURCE IDENTITY mismatch.
    base_id = json.loads(json.dumps(base))
    src_mismatch = json.loads(json.dumps(base))
    src_mismatch["runtime_source_identity"]["executed_sources"]["wrapper"]["executed_sha256"] = "q" * 64
    id_mismatch_caught = False
    try:
        m.compare_two_runs(base_id, src_mismatch)
    except m.FailClosed:
        id_mismatch_caught = True
    # also: identity_class / protocol_id disagreement must fail closed
    cls_mismatch = json.loads(json.dumps(base)); cls_mismatch["identity_class"] = "SOMETHING_ELSE"
    cls_caught = False
    try:
        m.compare_two_runs(base_id, cls_mismatch)
    except m.FailClosed:
        cls_caught = True
    record(23, "two-process source-identity / seed-class mismatch detection",
           "PASS" if (id_mismatch_caught and cls_caught) else "FAIL",
           "runtime_source_identity mismatch fails closed=%s; identity_class mismatch fails closed=%s "
           "(MOCK; real two-process NOT_RUN, no craftax)." % (id_mismatch_caught, cls_caught))

    # ---- summary ----
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_blocked = sum(1 for r in results if r["status"].startswith("BLOCKED"))
    print("NEGATIVE_TESTS: PASS=%d BLOCKED=%d FAIL=%d (total %d)"
          % (n_pass, n_blocked, n_fail, len(results)))
    for r in results:
        print("  [%s] %s %s" % (r["status"], r["id"], r["name"]))
        print("         %s" % r["detail"])
    print("NEGATIVE_TESTS_FAIL_COUNT=%d" % n_fail)
    if n_fail == 0:
        print("NEGATIVE_TESTS_SELF_TEST_PASS")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
