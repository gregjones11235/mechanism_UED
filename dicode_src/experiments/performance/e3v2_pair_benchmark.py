#!/usr/bin/env python3
"""Fail-closed runner and semantic/aggregate gates for the E3v2 replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping


# Deliberately match complete fatal diagnostics, rather than arbitrary paths.
FATAL_RE = re.compile(
    r"(?i)^(?:[a-z][\w-]*:\s*)?(?:traceback(?: \(most recent call last\))?:|(?:out[ -]?of[ -]?memory|oom|cuda\s*xid|xid\b[^\n]*\b\d+|segmentation fault|segfault|checkpoint[^\n]{0,30}corrupt)\b)"
)
CSV_HEADER = "timestamp,gpu_index,gpu_uuid,memory_used_mib,memory_free_mib,utilization_gpu_pct,app_pid,app_classification"


def gpu_snapshot(index: int, uuid: str | None = None) -> str:
    return subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def gpu_apps(index: int) -> str:
    return subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory,gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def assert_gpu_free(index: int, uuid: str) -> None:
    snap = gpu_snapshot(index, uuid)
    if uuid not in snap or gpu_apps(index).strip():
        raise RuntimeError("GPU unavailable or UUID mismatch")


def proc_chain(pid: int) -> tuple[list[int], bool]:
    chain: list[int] = []
    cur = int(pid)
    while cur > 1:
        chain.append(cur)
        try:
            text = Path(f"/proc/{cur}/status").read_text()
        except OSError:
            return chain, False
        line = next((x for x in text.splitlines() if x.startswith("PPid:")), "")
        try:
            cur = int(line.split()[1]) if line else 1
        except (IndexError, ValueError):
            cur = 1
    return chain, True


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def classify_pid(pid: int, roots: list[int] | set[int]) -> dict[str, Any]:
    chain, alive = proc_chain(pid)
    data: dict[str, Any] = {"pid": int(pid), "ancestry": chain, "cmdline": _cmdline(pid)}
    if not alive:
        data["classification"] = "stale_transient"
    elif set(chain) & set(int(x) for x in roots):
        data["classification"] = "owned_descendant"
    else:
        data["classification"] = "external"
    return data


def classify_gpu_apps(text: str, trainer_pid: int, uuid: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    bad: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        pid_text = line.split(",", 1)[0].strip()
        if not pid_text.isdigit() or uuid not in line:
            bad.append(line)
            continue
        row = classify_pid(int(pid_text), [trainer_pid, os.getpid()])
        row["gpu_line"] = line
        rows.append(row)
        # A process can disappear between nvidia-smi and /proc inspection;
        # stale_transient is evidence only, not a safety violation.
        if row["classification"] == "external":
            bad.append(line)
    return rows, bad


def descendants(pid: int) -> list[int]:
    """Return pid and descendants from a single process snapshot."""
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
    except Exception:
        return [int(pid)]
    tree: dict[int, list[int]] = {}
    for row in rows:
        bits = row.split()
        if len(bits) == 2:
            tree.setdefault(int(bits[1]), []).append(int(bits[0]))
    out: list[int] = []
    stack = [int(pid)]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(tree.get(current, []))
    return sorted(set(out))


def stop_owned(pid: int, term_timeout: float = 2.0) -> list[int]:
    """TERM then KILL the owned root and descendants, never its ancestors."""
    owned = descendants(pid)
    # ps gives us enough ancestry to ensure children are signalled first.
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
        parent = {int(b.split()[0]): int(b.split()[1]) for b in rows if len(b.split()) == 2}
        def depth(x: int) -> int:
            n = 0
            while x in parent and parent[x] in owned and parent[x] != x:
                n += 1; x = parent[x]
            return n
        owned = sorted(set(owned), key=depth, reverse=True)
    except Exception:
        owned = sorted(set(owned), reverse=True)
    for child in owned:
        try:
            os.kill(child, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + term_timeout
    while time.monotonic() < deadline:
        alive = []
        for child in owned:
            try:
                os.kill(child, 0); alive.append(child)
            except OSError:
                pass
        if not alive:
            break
        time.sleep(0.05)
    survivors = []
    for child in owned:
        try:
            os.kill(child, 0); survivors.append(child); os.kill(child, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass
    return owned


def _snapshot_values(snapshot: str) -> tuple[str, int, int, int, str]:
    line = next((x.strip() for x in snapshot.splitlines() if x.strip()), "")
    fields = [x.strip() for x in line.split(",")]
    if len(fields) < 5:
        raise ValueError("invalid nvidia-smi snapshot")
    return fields[1], int(re.search(r"\d+", fields[2]).group()), int(re.search(r"\d+", fields[3]).group()), int(re.search(r"\d+", fields[4]).group()), fields[0]


def monitor_popen(proc: subprocess.Popen[Any], index: int, uuid: str, out: str | Path):
    """Start sampler; return stop event, thread, and shared violation list."""
    stop = threading.Event(); violations: list[str] = []; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    csv = out / "gpu_memory.csv"; evidence = out / "gpu_evidence.log"
    if not csv.exists(): csv.write_text(CSV_HEADER + "\n")

    def loop() -> None:
        while not stop.is_set():
            timestamp = time.time()
            try:
                snap = gpu_snapshot(index, uuid); apps = gpu_apps(index)
                gpu_uuid, util, used, free, gpu_index = _snapshot_values(snap)
                rows, bad = classify_gpu_apps(apps, proc.pid, uuid)
                app_pid = rows[0]["pid"] if rows else ""
                app_class = rows[0]["classification"] if rows else "none"
                with csv.open("a") as f:
                    f.write(f"{timestamp:.6f},{gpu_index},{gpu_uuid},{used},{free},{util},{app_pid},{app_class}\n")
                if bad or gpu_uuid != uuid:
                    violations.extend(bad or ["uuid_change"])
                    evidence.open("a").write(f"{timestamp:.6f} violation=" + repr(bad or ["uuid_change"]) + "\n")
            except Exception as exc:
                violations.append(str(exc)); evidence.open("a").write(f"{timestamp:.6f} error={exc}\n")
            stop.wait(2.0)

    thread = threading.Thread(target=loop, name="e3v2-gpu-monitor", daemon=True); thread.start()
    return stop, thread, violations


def arm_gpu_metrics(csv_path: str | Path) -> tuple[int, int | None]:
    used: list[int] = []; free: list[int] = []
    path = Path(csv_path)
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 6:
                try:
                    used.append(int(fields[3])); free.append(int(fields[4]))
                except ValueError:
                    continue
    return max(used, default=0), min(free, default=None)


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fatal_in(paths: list[str | Path]) -> str | None:
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(errors="replace").splitlines()[-1000:]:
            if FATAL_RE.search(line.strip()):
                return line
    return None


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


SEMANTIC_FIELDS = (
    "params_sha256_before", "params_sha256_after", "optimizer_sha256_before", "optimizer_sha256_after",
    "checkpoint_reloaded_params_sha256", "checkpoint_reloaded_optimizer_sha256", "rng_sha256_before",
    "outer_rng_after_sha256", "task_ids", "task_assignment_sha256", "task_code_hashes", "embedding_hash",
    "reset_selection_semantics", "global_update_step",
    "global_env_steps", "updates", "env_steps", "scoring_fingerprint", "score_function", "wrappers_cl_sha256",
    "conditioning_type", "conditioning_shape", "conditioning_dtype",
)


def _same(a: Mapping[str, Any], b: Mapping[str, Any], *names: str) -> bool:
    """Compare a semantic value while accepting the manifest's historical aliases."""
    av = next((a[n] for n in names if n in a), None)
    bv = next((b[n] for n in names if n in b), None)
    return av == bv


def compare_pair(e0: Mapping[str, Any], e3: Mapping[str, Any], min_speedup: float | None = None) -> str:
    for field in SEMANTIC_FIELDS:
        if field not in e0 or field not in e3 or e0.get(field) != e3.get(field):
            return "REJECTED_SEMANTIC_MISMATCH"
    for aliases in (("input_rng_sha256", "rng_input_sha256"), ("train_rng_sha256", "rng_train_sha256"),
                    ("rng_sha256_after", "rng_after_sha256"), ("task_ids", "task_order"),
                    ("task_code_hashes", "code_hashes"), ("wrappers_cl_sha256", "wrapper_hash"),
                    ("reset_selection_semantics", "reset_semantics")):
        if not any(name in e0 for name in aliases) or not any(name in e3 for name in aliases) or not _same(e0, e3, *aliases):
            return "REJECTED_SEMANTIC_MISMATCH"
    if any(field not in e0 or field not in e3 for field in ("gpu_uuid", "classification", "llm_api_calls")):
        return "REJECTED_SEMANTIC_MISMATCH"
    if e0.get("gpu_uuid") != e3.get("gpu_uuid") or e0.get("classification") != e3.get("classification") or e0.get("llm_api_calls") != e3.get("llm_api_calls"):
        return "REJECTED_SEMANTIC_MISMATCH"
    if e0.get("compact_scoring_payload") is not False or e3.get("compact_scoring_payload") is not True:
        return "REJECTED_SEMANTIC_MISMATCH"
    if "checkpoint_loadable" not in e0 or "checkpoint_loadable" not in e3 or not e0.get("checkpoint_loadable") or not e3.get("checkpoint_loadable"):
        return "REJECTED_RUNTIME_FAILURE"
    return "SEMANTIC_PASS"


def aggregate(pairs: list[Mapping[str, Any]]) -> dict[str, Any]:
    import statistics
    if len(pairs) != 6:
        return {"conclusion": "REJECTED_RUNTIME_FAILURE", "pair_count": len(pairs)}
    d0 = [float(p["e0"].get("train_wall_s", p["e0"].get("duration_s", 0))) for p in pairs]
    d3 = [float(p["e3"].get("train_wall_s", p["e3"].get("duration_s", 0))) for p in pairs]
    th0 = [float(p["e0"].get("env_steps", 0)) / max(d, 1e-12) for p, d in zip(pairs, d0)]
    th3 = [float(p["e3"].get("env_steps", 0)) / max(d, 1e-12) for p, d in zip(pairs, d3)]
    md0, md3 = statistics.mean(d0), statistics.mean(d3); med0, med3 = statistics.median(d0), statistics.median(d3)
    mean_t0, mean_t3 = statistics.mean(th0), statistics.mean(th3)
    duration_imp = (md0 - md3) / md0 if md0 else 0.0; median_imp = (med0 - med3) / med0 if med0 else 0.0
    throughput_imp = (mean_t3 - mean_t0) / mean_t0 if mean_t0 else 0.0
    regressions = [i for i, (a, b) in enumerate(zip(d0, d3)) if b > a * 1.01]
    peak_deltas = [float(p["e3"].get("gpu_peak_memory_mib", 0)) - float(p["e0"].get("gpu_peak_memory_mib", 0)) for p in pairs]
    min_free = min(float(p[a].get("gpu_min_free_mib", 0)) for p in pairs for a in ("e0", "e3"))
    runtime_bad = max(peak_deltas, default=0) > 512 or min_free < 4096
    runtime_markers = ("runtime_failure", "fatal_error", "oom", "xid", "checkpoint_error", "gpu_violation")
    runtime_bad = runtime_bad or any(any(bool(p[a].get(k)) for k in runtime_markers) or p[a].get("checkpoint_loadable") is False for p in pairs for a in ("e0", "e3"))
    speed_pass = duration_imp >= .10 and median_imp >= .10 and throughput_imp >= .10
    conclusion = "REJECTED_RUNTIME_FAILURE" if runtime_bad else ("E3V2_SCORING_PAYLOAD_PASS" if speed_pass and not regressions else "REJECTED_NO_SPEEDUP")
    return {"conclusion": conclusion, "mean_duration_e0": md0, "mean_duration_e3": md3, "median_duration_e0": med0,
            "median_duration_e3": med3, "duration_improvement": duration_imp, "median_duration_improvement": median_imp,
            "throughput_improvement": throughput_imp, "per_pair_regressions": regressions,
            "max_peak_delta_mib": max(peak_deltas, default=0), "min_free_mib": min_free, "pair_count": 6}


def validate_result(doc: Mapping[str, Any], *, manifest_sha256: str, stage: str, repeat: int, arm: str, gpu_uuid: str | None = None, source_commit: str | None = None) -> dict[str, Any]:
    checks = {"manifest_sha256": manifest_sha256, "stage": stage, "repeat": repeat, "arm": arm, "classification": "TRAINING_KERNEL_BENCHMARK", "llm_api_calls": 0}
    if source_commit is not None:
        checks["source_commit"] = source_commit
    for key, expected in checks.items():
        if doc.get(key) != expected:
            raise RuntimeError(f"invalid result {key}")
    if gpu_uuid is not None and doc.get("gpu_uuid") != gpu_uuid:
        raise RuntimeError("invalid result gpu_uuid")
    evidence = doc.get("runtime_source_evidence")
    required_runtime = {"run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state"}
    if not isinstance(evidence, Mapping) or evidence.get("verified") is not True or not required_runtime.issubset(set(evidence.get("paths", {}))) or not required_runtime.issubset(set(evidence.get("hashes", {}))):
        raise RuntimeError("invalid runtime source evidence")
    checkpoint = doc.get("checkpoint_path")
    if not doc.get("checkpoint_exists") or not doc.get("checkpoint_loadable") or not checkpoint or not Path(checkpoint).exists():
        raise RuntimeError("invalid checkpoint result")
    return dict(doc)


def _failure(out: Path, message: str, proc: subprocess.Popen[Any] | None = None, **extra: Any) -> None:
    if proc is not None:
        stop_owned(proc.pid)
    atomic_json(out / "failure.json", {"error": message, **extra})


def run_arm(args: Any, stage: str, repeat: int, arm: str, out: str | Path) -> dict[str, Any]:
    out = Path(out); out.mkdir(parents=True, exist_ok=True); result = out / "RESULT.json"
    manifest_sha = args.manifest_sha256
    if result.exists():
        return validate_result(json.loads(result.read_text()), manifest_sha256=manifest_sha, stage=stage, repeat=repeat, arm=arm, gpu_uuid=args.gpu_uuid, source_commit=args.source_commit)
    source = args.source_e0 if arm == "E0" else args.source_e3v2; config = args.config_e0 if arm == "E0" else args.config_e3v2
    env = dict(os.environ); env.pop("JAX_PLATFORMS", None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, WANDB_MODE="offline", PYTHONPATH=str(Path(source) / "src"))
    for key, subdir in (("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"), ("XDG_CACHE_HOME", "cache"), ("WANDB_DIR", "wandb")):
        path = out / subdir; path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
    cmd = [args.python, args.harness, "--manifest", args.manifest, "--config", config, "--out", str(out), "--required-gpu-uuid", args.gpu_uuid,
           "--source-commit", args.source_commit, "--stage", stage, "--repeat", str(repeat), "--arm", arm, "--mode", "run"]
    assert_gpu_free(args.gpu_index, args.gpu_uuid)
    so = (out / "trainer.stdout").open("w"); se = (out / "trainer.stderr").open("w")
    proc: subprocess.Popen[Any] | None = None
    try:
        proc = subprocess.Popen(cmd, env=env, text=True, stdout=so, stderr=se, start_new_session=True)
        stop, thread, violations = monitor_popen(proc, args.gpu_index, args.gpu_uuid, out)
        try:
            while proc.poll() is None:
                fatal = fatal_in([out / "trainer.stdout", out / "trainer.stderr"])
                if violations or fatal:
                    _failure(out, "GPU safety violation" if violations else "fatal trainer output", proc, violations=violations, fatal=fatal)
                    raise RuntimeError("trainer stopped")
                time.sleep(.1)
            if proc.returncode:
                _failure(out, f"{arm} failed rc={proc.returncode}", proc)
                raise RuntimeError(f"{arm} failed rc={proc.returncode}")
        finally:
            stop.set(); thread.join(timeout=5)
    except Exception:
        if not (out / "failure.json").exists(): _failure(out, "trainer launch failure", proc)
        raise
    finally:
        so.close(); se.close()
    if not result.exists():
        _failure(out, "missing RESULT.json"); raise RuntimeError("missing RESULT.json")
    doc = validate_result(json.loads(result.read_text()), manifest_sha256=manifest_sha, stage=stage, repeat=repeat, arm=arm, gpu_uuid=args.gpu_uuid, source_commit=args.source_commit)
    peak, minimum = arm_gpu_metrics(out / "gpu_memory.csv"); doc.update(gpu_peak_memory_mib=peak, gpu_min_free_mib=minimum, gpu_uuid=args.gpu_uuid)
    atomic_json(result, doc)
    return doc


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("root", "manifest", "harness", "python", "gpu_index", "gpu_uuid", "source_e0", "source_e3v2", "config_e0", "config_e3v2", "source_commit"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    args = p.parse_args(); args.gpu_index = int(args.gpu_index)
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("e3v2_replay_manifest", Path(__file__).with_name("e3v2_replay_manifest.py")); module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module)
    manifest = module.load_manifest(args.manifest); args.manifest_sha256 = manifest["manifest_sha256"]; pairs: list[dict[str, Any]] = []
    failed_pair: str | None = None
    try:
        for stage in ("early", "mid", "late"):
            for repeat in (0, 1):
                order = next(x for x in manifest["stages"] if x["name"] == stage)["repeats"][repeat]["order"]
                pair_dir = root / "pairs" / f"{stage}_repeat_{repeat}"; failed_pair = str(pair_dir); rows: dict[str, Any] = {}
                for arm in order: rows[arm] = run_arm(args, stage, repeat, arm, pair_dir / arm.lower())
                status = compare_pair(rows["E0"], rows["E3V2"])
                if status != "SEMANTIC_PASS":
                    conclusion = "REJECTED_SEMANTIC_MISMATCH" if status == "REJECTED_SEMANTIC_MISMATCH" else "REJECTED_RUNTIME_FAILURE"
                    atomic_json(root / "E3V2_PAIR_RESULT.json", {"conclusion": conclusion, "completed_pair_count": len(pairs), "evidence": str(pair_dir), "failed_pair": str(pair_dir), "pairs": pairs, "manifest_sha256": args.manifest_sha256, "gpu_uuid": args.gpu_uuid})
                    raise RuntimeError(status)
                atomic_json(pair_dir / "PAIR.json", {"status": status, "e0": rows["E0"], "e3": rows["E3V2"]})
                pairs.append({"e0": rows["E0"], "e3": rows["E3V2"], "status": status})
    except Exception as exc:
        if not (root / "E3V2_PAIR_RESULT.json").exists() or json.loads((root / "E3V2_PAIR_RESULT.json").read_text()).get("conclusion") not in {"REJECTED_SEMANTIC_MISMATCH", "REJECTED_RUNTIME_FAILURE"}:
            atomic_json(root / "E3V2_PAIR_RESULT.json", {"conclusion": "REJECTED_RUNTIME_FAILURE", "completed_pair_count": len(pairs), "evidence": failed_pair, "failed_pair": failed_pair, "error": str(exc), "pairs": pairs, "manifest_sha256": args.manifest_sha256, "gpu_uuid": args.gpu_uuid})
        raise
    agg = aggregate(pairs)
    final = {**agg, "pairs": pairs, "pair_count": 6, "manifest_sha256": args.manifest_sha256, "gpu_uuid": args.gpu_uuid}
    atomic_json(root / "E3V2_PAIR_RESULT.json", final)
    if agg.get("conclusion") == "REJECTED_RUNTIME_FAILURE":
        raise RuntimeError("runtime gate failed")


if __name__ == "__main__":
    main()
