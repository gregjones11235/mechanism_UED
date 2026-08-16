import importlib.util
import json
from pathlib import Path

import numpy as np

PERF = Path(__file__).parents[4] / "experiments" / "performance"


def _a2():
    spec = importlib.util.spec_from_file_location("perf48_a2_minimal", PERF / "perf48_a2_minimal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _doc(params="p", **kw):
    d = {"params_sha256_before": "b", "params_sha256_after": params, "optimizer_sha256_before": "o",
         "optimizer_sha256_after": "oa", "checkpoint_reloaded_params_sha256": "crp",
         "checkpoint_reloaded_optimizer_sha256": "cro", "input_rng_sha256": "i", "train_rng_sha256": "t",
         "rng_sha256_before": "rb", "outer_rng_after_sha256": "oa2", "task_ids": ["t1"], "task_assignment_sha256": "ta",
         "task_code_hashes": ["c"], "embedding_hash": "e", "reset_selection_semantics": {"pass": True},
         "global_update_step": 700, "global_env_steps": 91750400, "updates": 100, "env_steps": 13107200,
         "scoring_fingerprint": "f", "checkpoint_loadable": True, "gpu_uuid": "GPU-1",
         "wrappers_cl_sha256": "w", "conditioning_type": "one_hot", "conditioning_shape": [16, 67],
         "conditioning_dtype": "float32", "score_function": "learnability", "compact_scoring_payload": False,
         "measured_session_wall_s": 1000.0, "train_wall_s": 950.0}
    d.update(kw)
    return d


def _profile_out(tmp_path, with_files=True, good_schema=True, with_phases=True, good_hash=True, excl_ok=True):
    a = _a2()
    out = tmp_path / "prof"
    out.mkdir(exist_ok=True)
    if with_files:
        events = []
        phases = ["train_build", "train_lower_compile", "train_execute", "scoring_transfer", "scoring_cpu"]
        if not with_phases:
            phases = ["train_build"]
        for ph in phases:
            e = {f: (1 if f in ("start_monotonic_ns", "end_monotonic_ns") else None) for f in a.EVENT_FIELDS}
            e["phase"] = ph; e["status"] = "ok"; e["cache_hit"] = False; e["duration_s"] = 0.0
            events.append(e)
        (out / "events.jsonl").write_text("\n".join(json.dumps(x) for x in events))
        (out / "events.csv").write_text("x")
        (out / "critical_path.json").write_text(json.dumps({"sessions": {"s0": {"session_wall": 10.0, "exclusive_phase_totals": {"a": 1.0, "b": 2.0}}}} if excl_ok else {"sessions": {"s0": {"session_wall": 1.0, "exclusive_phase_totals": {"a": 5.0, "b": 5.0}}}}))
    prof_doc = _doc(params="prof")
    prof_doc["profiling"] = {
        "enabled": True,
        "events_csv_sha256": a.sha256_file(out / "events.csv") if with_files and good_hash else ("bad" if with_files else None),
        "critical_path_sha256": a.sha256_file(out / "critical_path.json") if with_files and good_hash else ("bad" if with_files else None),
    }
    off_out = tmp_path / "off"
    off_out.mkdir(exist_ok=True)
    off_doc = _doc(params="off")
    return a, prof_doc, out, off_doc, off_out


def test_semantic_diffs_empty_and_detects():
    a = _a2()
    assert a.semantic_diffs(_doc(), _doc()) == []
    assert a.semantic_diffs(_doc(), _doc(params="different")) == ["params_sha256_after"]
    assert a.semantic_diffs(_doc(), _doc(score_function="pvl")) == ["score_function"]


def test_compute_overhead_formula():
    a = _a2()
    off_a = _doc(params="a", measured_session_wall_s=1000.0, train_wall_s=950.0)
    prof = _doc(params="p", measured_session_wall_s=1005.0, train_wall_s=960.0)
    off_b = _doc(params="b", measured_session_wall_s=1000.0, train_wall_s=950.0)
    oh = a.compute_overhead(off_a, prof, off_b)
    assert abs(oh["off_ref_measured_s"] - 1000.0) < 1e-9
    assert abs(oh["overhead_measured"] - 0.005) < 1e-9
    assert abs(oh["overhead_train"] - (960 - 950) / 950) < 1e-9


def test_compute_memory():
    a = _a2()
    off_a = _doc(params="a", gpu_peak_memory_mib=34000, gpu_min_free_mib=12000)
    prof = _doc(params="p", gpu_peak_memory_mib=34500, gpu_min_free_mib=11500)
    off_b = _doc(params="b", gpu_peak_memory_mib=33800, gpu_min_free_mib=12500)
    m = a.compute_memory(off_a, prof, off_b)
    assert m["off_peak_ref_mib"] == 34000
    assert m["peak_delta_mib"] == 500
    assert m["off_min_free_ref_mib"] == 12000


def test_conclusion_matrix():
    a = _a2()
    ok_oh = {"overhead_measured": 0.004}
    ok_mem = {"peak_delta_mib": 100, "profile_min_free_mib": 10000}
    assert a.conclusion(["x"], True, ok_oh, ok_mem) == "REJECTED_SEMANTIC_MISMATCH"
    assert a.conclusion([], False, ok_oh, ok_mem) == "REJECTED_PROFILING_CONTRACT"
    assert a.conclusion([], True, ok_oh, {"peak_delta_mib": 600, "profile_min_free_mib": 10000}) == "REJECTED_PROFILING_OVERHEAD"
    assert a.conclusion([], True, ok_oh, {"peak_delta_mib": 100, "profile_min_free_mib": 3000}) == "REJECTED_PROFILING_OVERHEAD"
    assert a.conclusion([], True, {"overhead_measured": 0.021}, ok_mem) == "REJECTED_PROFILING_OVERHEAD"
    assert a.conclusion([], True, {"overhead_measured": 0.004}, ok_mem) == "A2_MINIMAL_PASS"
    assert a.conclusion([], True, {"overhead_measured": 0.01}, ok_mem) == "A2_MINIMAL_PASS_WITH_OVERHEAD_CONCERN"


def test_verify_profile_contract_clean(tmp_path):
    a, prof_doc, out, off_doc, off_out = _profile_out(tmp_path)
    ok, msgs = a.verify_profile_contract(prof_doc, out, off_doc, off_out)
    assert ok, msgs


def test_verify_profile_contract_failures(tmp_path):
    a, prof_doc, out, off_doc, off_out = _profile_out(tmp_path, with_files=False)
    ok, msgs = a.verify_profile_contract(prof_doc, out, off_doc, off_out)
    assert not ok and any("missing" in m for m in msgs)

    a, prof_doc, out, off_doc, off_out = _profile_out(tmp_path, with_phases=False)
    ok, msgs = a.verify_profile_contract(prof_doc, out, off_doc, off_out)
    assert not ok and any("missing required" in m for m in msgs)

    a, prof_doc, out, off_doc, off_out = _profile_out(tmp_path, good_hash=False)
    ok, msgs = a.verify_profile_contract(prof_doc, out, off_doc, off_out)
    assert not ok and any("sha256 mismatch" in m for m in msgs)

    a, prof_doc, out, off_doc, off_out = _profile_out(tmp_path, excl_ok=False)
    ok, msgs = a.verify_profile_contract(prof_doc, out, off_doc, off_out)
    assert not ok and any("exclusive totals" in m for m in msgs)

    # P0_OFF residual artifact
    a, prof_doc, out, off_doc, off_out = _profile_out(tmp_path)
    (off_out / "events.jsonl").write_text("{}")
    ok, msgs = a.verify_profile_contract(prof_doc, out, off_doc, off_out)
    assert not ok and any("residual" in m for m in msgs)
