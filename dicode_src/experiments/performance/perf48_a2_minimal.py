#!/usr/bin/env python3
"""A2 minimal gate (5 arms, no E0, no full 13-arm gate).

A2.1 (deterministic semantic pair, --deterministic-xla):
    P0_OFF_DET -> P0_PROFILE_DET   (mid checkpoint, repeat 0)
    - compare all semantic fields; any diff -> REJECTED_SEMANTIC_MISMATCH
    - verify the P0_PROFILE reporting contract (JSONL/CSV/critical_path, schema,
      required phases, hash match, no double-counting) and P0_OFF zero residue

A2.2 (production sandwich, no flag):
    P0_OFF_A -> P0_PROFILE -> P0_OFF_B
    overhead = (P0_PROFILE.measured_session_wall_s - mean(OFF_A,OFF_B))/mean
    - <0.5% -> A2_MINIMAL_PASS ; 0.5-2% -> A2_MINIMAL_PASS_WITH_OVERHEAD_CONCERN ;
      >=2%  -> REJECTED_PROFILING_OVERHEAD ; memory delta>512MiB or min_free<4GiB -> REJECTED_PROFILING_OVERHEAD

Reuses perf48_pair_benchmark.run_arm / validate_result. Fail-closed.
Run on server: <dicode310-python> perf48_a2_minimal.py
"""
from __future__ import annotations
import datetime
import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

OLD = Path("/home/oseasy/e2_data_disk2/skill_preflight_runs/perf48_p0_gpu3_91a75e5_20260811T110147Z")
RUNS = Path("/home/oseasy/e2_data_disk2/skill_preflight_runs")
STAGE = Path("/tmp/perf48_a2_stage")
UUID = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"
PY = "/home/oseasy/miniconda3/envs/dicode310/bin/python"
SOURCE_COMMIT = "91a75e5+perf48-6ceffef"

SEM = [
    "params_sha256_before", "params_sha256_after", "optimizer_sha256_before", "optimizer_sha256_after",
    "checkpoint_reloaded_params_sha256", "checkpoint_reloaded_optimizer_sha256",
    "input_rng_sha256", "train_rng_sha256", "rng_sha256_before", "outer_rng_after_sha256",
    "task_ids", "task_assignment_sha256", "task_code_hashes", "embedding_hash",
    "reset_selection_semantics", "global_update_step", "global_env_steps", "updates", "env_steps",
    "scoring_fingerprint", "checkpoint_loadable", "gpu_uuid", "wrappers_cl_sha256",
    "conditioning_type", "conditioning_shape", "conditioning_dtype", "score_function",
    "compact_scoring_payload",
]

REQUIRED_PHASES = {"train_build", "train_lower_compile", "train_execute", "scoring_transfer", "scoring_cpu"}
EVENT_FIELDS = {"run_id", "session", "phase", "parent_phase", "start_monotonic_ns", "end_monotonic_ns",
                "duration_s", "status", "cache_hit", "task_signature", "request_id", "overlap_group"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def semantic_diffs(a: dict, b: dict) -> list[str]:
    return [f for f in SEM if a.get(f) != b.get(f)]


def verify_profile_contract(prof_doc: dict, prof_out: Path, off_doc: dict, off_out: Path) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    for fname in ("events.jsonl", "events.csv", "critical_path.json"):
        if not (prof_out / fname).exists():
            ok = False; msgs.append(f"P0_PROFILE missing {fname}")
    jsonl = prof_out / "events.jsonl"
    if jsonl.exists():
        events = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        if not events:
            ok = False; msgs.append("events.jsonl empty")
        elif any(set(e) != EVENT_FIELDS for e in events):
            ok = False; msgs.append("event schema mismatch")
        phases = {e.get("phase") for e in events}
        if REQUIRED_PHASES - phases:
            ok = False; msgs.append(f"missing required phases {sorted(REQUIRED_PHASES - phases)}")
        stored_csv = prof_doc.get("profiling", {}).get("events_csv_sha256")
        stored_cp = prof_doc.get("profiling", {}).get("critical_path_sha256")
        if stored_csv and (prof_out / "events.csv").exists() and sha256_file(prof_out / "events.csv") != stored_csv:
            ok = False; msgs.append("events.csv sha256 mismatch")
        if stored_cp and (prof_out / "critical_path.json").exists() and sha256_file(prof_out / "critical_path.json") != stored_cp:
            ok = False; msgs.append("critical_path.json sha256 mismatch")
        # exclusive totals must not exceed the session wall (no double-counting)
        try:
            cp = json.loads((prof_out / "critical_path.json").read_text())
            for sess, srep in (cp.get("sessions") or {}).items():
                wall = srep.get("session_wall", 0.0)
                excl = sum(srep.get("exclusive_phase_totals", {}).values())
                if excl > wall * 1.0001:
                    ok = False; msgs.append(f"exclusive totals {excl:.3f} > session_wall {wall:.3f} for {sess}")
        except Exception as exc:
            ok = False; msgs.append(f"critical_path parse error: {exc!r}")
    for fname in ("events.jsonl", "events.csv", "critical_path.json"):
        if (off_out / fname).exists():
            ok = False; msgs.append(f"P0_OFF residual artifact {fname}")
    return ok, msgs


def compute_overhead(off_a: dict, prof: dict, off_b: dict) -> dict:
    off_ref_meas = (off_a["measured_session_wall_s"] + off_b["measured_session_wall_s"]) / 2.0
    off_ref_train = (off_a["train_wall_s"] + off_b["train_wall_s"]) / 2.0
    return {
        "off_a_measured_s": off_a["measured_session_wall_s"],
        "off_b_measured_s": off_b["measured_session_wall_s"],
        "profile_measured_s": prof["measured_session_wall_s"],
        "off_ref_measured_s": off_ref_meas,
        "overhead_measured": (prof["measured_session_wall_s"] - off_ref_meas) / off_ref_meas,
        "overhead_train": (prof["train_wall_s"] - off_ref_train) / off_ref_train,
        "off_a_train_s": off_a["train_wall_s"],
        "profile_train_s": prof["train_wall_s"],
        "off_b_train_s": off_b["train_wall_s"],
    }


def compute_memory(off_a: dict, prof: dict, off_b: dict) -> dict:
    off_peak = max(off_a.get("gpu_peak_memory_mib", 0), off_b.get("gpu_peak_memory_mib", 0))
    off_min = min(off_a.get("gpu_min_free_mib", 1e12), off_b.get("gpu_min_free_mib", 1e12))
    return {
        "profile_peak_mib": prof.get("gpu_peak_memory_mib"),
        "off_peak_ref_mib": off_peak,
        "peak_delta_mib": round(prof.get("gpu_peak_memory_mib", 0) - off_peak, 1),
        "profile_min_free_mib": prof.get("gpu_min_free_mib"),
        "off_min_free_ref_mib": off_min,
    }


def conclusion(diffs: list[str], contract_ok: bool, overhead: dict, memory: dict) -> str:
    if diffs:
        return "REJECTED_SEMANTIC_MISMATCH"
    if not contract_ok:
        return "REJECTED_PROFILING_CONTRACT"
    if memory["peak_delta_mib"] > 512 or (memory["profile_min_free_mib"] or 0) < 4096:
        return "REJECTED_PROFILING_OVERHEAD"
    oh = overhead["overhead_measured"]
    if oh >= 0.02:
        return "REJECTED_PROFILING_OVERHEAD"
    if oh < 0.005:
        return "A2_MINIMAL_PASS"
    return "A2_MINIMAL_PASS_WITH_OVERHEAD_CONCERN"


def atomic_json(path: Path, value: dict) -> None:
    import tempfile, os
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ROOT = RUNS / f"perf48_a2_minimal_91a75e5_{TS}"
    if ROOT.exists():
        sys.exit(f"ROOT already exists: {ROOT}")
    ROOT.mkdir(parents=True)
    TOOLS = ROOT / "tools"; TOOLS.mkdir()
    for name in ("perf48_pair_benchmark.py", "perf48_training_kernel_harness.py",
                 "perf48_supervisor.py", "perf48_replay_manifest.py"):
        shutil.copy(STAGE / name, TOOLS / name)
    print(f"ROOT={ROOT}", flush=True)

    spec = importlib.util.spec_from_file_location("bench", TOOLS / "perf48_pair_benchmark.py")
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    manifest = json.loads((OLD / "manifest.json").read_text())

    def make_args(det: bool) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            root=str(ROOT), manifest=str(OLD / "manifest.json"), manifest_sha256=manifest["manifest_sha256"],
            harness=str(TOOLS / "perf48_training_kernel_harness.py"), python=PY, gpu_index=3, gpu_uuid=UUID,
            source_e0=str(OLD / "sources" / "e0" / "dicode_src"),
            source_perf48=str(OLD / "sources" / "perf48" / "dicode_src"),
            config_e0=str(OLD / "configs" / "e0.yaml"),
            config_perf48_off=str(OLD / "configs" / "perf48_off.yaml"),
            config_perf48_profile=str(OLD / "configs" / "perf48_profile.yaml"),
            source_commit=SOURCE_COMMIT, deterministic_xla=det,
        )

    result: dict = {"root": str(ROOT), "manifest_sha256": manifest["manifest_sha256"],
                    "source_commit": SOURCE_COMMIT, "gpu_uuid": UUID}
    try:
        # ---- A2.1 deterministic semantic pair ----
        print("=== A2.1 P0_OFF_DET -> P0_PROFILE_DET ===", flush=True)
        args_det = make_args(True)
        sd = ROOT / "semantic_pair"
        off_doc = bench.run_arm(args_det, "mid", 0, "P0_OFF", sd / "p0_off_det")
        print(f"  P0_OFF_DET params_after={off_doc['params_sha256_after']} wall={off_doc['measured_session_wall_s']}", flush=True)
        prof_doc = bench.run_arm(args_det, "mid", 0, "P0_PROFILE", sd / "p0_profile_det")
        print(f"  P0_PROFILE_DET params_after={prof_doc['params_sha256_after']} wall={prof_doc['measured_session_wall_s']}", flush=True)
        diffs = semantic_diffs(off_doc, prof_doc)
        contract_ok, contract_msgs = verify_profile_contract(prof_doc, sd / "p0_profile_det", off_doc, sd / "p0_off_det")
        pair = {"a": "P0_OFF_DET", "b": "P0_PROFILE_DET", "diff_count": len(diffs), "diffs": diffs,
                "contract_ok": contract_ok, "contract_msgs": contract_msgs,
                "a_params": off_doc["params_sha256_after"], "b_params": prof_doc["params_sha256_after"]}
        atomic_json(sd / "PAIR.json", pair)
        result["a21"] = pair
        print(f"  diffs={diffs} contract_ok={contract_ok} msgs={contract_msgs}", flush=True)
        if diffs or not contract_ok:
            result["conclusion"] = conclusion(diffs, contract_ok, {}, {"peak_delta_mib": 0, "profile_min_free_mib": 1e12})
            atomic_json(ROOT / "A2_MINIMAL_RESULT.json", result)
            print(f"=== A2_VERDICT {result['conclusion']} (A2.2 not run) ===", flush=True)
            return

        # ---- A2.2 production sandwich ----
        print("=== A2.2 P0_OFF_A -> P0_PROFILE -> P0_OFF_B ===", flush=True)
        args_nodet = make_args(False)
        ps = ROOT / "production_sandwich"
        off_a = bench.run_arm(args_nodet, "mid", 0, "P0_OFF", ps / "p0_off_a")
        prof = bench.run_arm(args_nodet, "mid", 0, "P0_PROFILE", ps / "p0_profile")
        off_b = bench.run_arm(args_nodet, "mid", 0, "P0_OFF", ps / "p0_off_b")
        oh = compute_overhead(off_a, prof, off_b)
        mem = compute_memory(off_a, prof, off_b)
        sand = {"overhead": oh, "memory": mem,
                "off_a_params": off_a["params_sha256_after"], "profile_params": prof["params_sha256_after"], "off_b_params": off_b["params_sha256_after"]}
        atomic_json(ps / "SANDWICH.json", sand)
        result["a22"] = sand
        print(f"  overhead_measured={oh['overhead_measured']:.4f} train={oh['overhead_train']:.4f} mem_delta={mem['peak_delta_mib']} min_free={mem['profile_min_free_mib']}", flush=True)
        result["conclusion"] = conclusion([], True, oh, mem)
        atomic_json(ROOT / "A2_MINIMAL_RESULT.json", result)
        print(f"=== A2_VERDICT {result['conclusion']} ===", flush=True)
    except Exception as exc:
        import traceback
        result["conclusion"] = "REJECTED_RUNTIME_FAILURE"
        result["error"] = repr(exc)
        atomic_json(ROOT / "A2_MINIMAL_RESULT.json", result)
        atomic_json(ROOT / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        print(f"=== A2_VERDICT REJECTED_RUNTIME_FAILURE: {exc!r} ===", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
