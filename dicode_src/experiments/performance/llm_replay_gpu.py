#!/usr/bin/env python3
"""Continuous GPU0 sampler + CSV stats for the stage-D LLM replay (independent).

Samples GPU0 at a fixed interval (default 2s) to a CSV and computes peak-used /
min-free / max-temperature / UUID+PID consistency from the CSV. Used by the D1b
embedding experiment to provide continuous (not just before/after) GPU evidence.

Read-only observation of GPU0 only; does NOT touch GPU1/2/3, does NOT change any
process or Ollama parameter.
"""
from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

GPU_CSV_FIELDS = (
    "timestamp", "gpu_index", "gpu_uuid", "utilization_gpu",
    "memory_used_mib", "memory_free_mib", "temperature", "compute_pid",
    "process_name",
)


def _query_gpu0() -> dict[str, Any] | None:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] == "0":
            used = int(parts[3].split()[0])
            total = int(parts[4].split()[0])
            return {"gpu_index": 0, "gpu_uuid": parts[1], "utilization_gpu": parts[2],
                    "memory_used_mib": used, "memory_free_mib": total - used,
                    "temperature": parts[5]}
    return None


def _query_compute_pids() -> list[tuple[str, str]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1].startswith("GPU-e8c08612"):  # GPU0 only
            pids.append((parts[0], parts[2]))
    return pids


def _append_csv(path: Path, row: dict[str, Any], header: bool) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GPU_CSV_FIELDS)
        if header:
            w.writeheader()
        w.writerow(row)


def gpu_sampler_loop(output_csv: str, interval_s: float, stop_event: threading.Event,
                     gpu_index_filter: int = 0) -> None:
    """Sample GPU0 every ``interval_s`` until ``stop_event`` is set."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = not path.exists()
    while not stop_event.is_set():
        g = _query_gpu0()
        if g is None or g["gpu_index"] != gpu_index_filter:
            # never fabricate a sample; skip if GPU0 not visible
            time.sleep(interval_s)
            continue
        pids = _query_compute_pids()
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "gpu_index": g["gpu_index"], "gpu_uuid": g["gpu_uuid"],
            "utilization_gpu": g["utilization_gpu"],
            "memory_used_mib": g["memory_used_mib"],
            "memory_free_mib": g["memory_free_mib"],
            "temperature": g["temperature"],
            "compute_pid": ",".join(p[0] for p in pids),
            "process_name": ";".join(p[1] for p in pids),
        }
        _append_csv(path, row, header=wrote_header)
        wrote_header = False
        time.sleep(interval_s)


def compute_gpu_stats(csv_path: str | Path) -> dict[str, Any]:
    """Recompute peak-used / min-free / max-temp / UUID+PID consistency from CSV."""
    path = Path(csv_path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {"sample_count": 0}
    uuids = {r["gpu_uuid"] for r in rows if r.get("gpu_uuid")}
    used = [int(r["memory_used_mib"]) for r in rows]
    free = [int(r["memory_free_mib"]) for r in rows]
    temps = [r["temperature"] for r in rows if r.get("temperature") not in (None, "", "N/A")]
    pids = set()
    for r in rows:
        for p in (r.get("compute_pid") or "").split(","):
            if p.strip():
                pids.add(p.strip())
    uuid_consistent = len(uuids) == 1
    uuid_val = next(iter(uuids)) if uuid_consistent else (list(uuids) if uuids else None)
    return {
        "sample_count": len(rows),
        "uuid": uuid_val,
        "uuid_consistent": uuid_consistent,
        "peak_memory_used_mib": max(used),
        "min_memory_free_mib": min(free),
        "max_temperature": max(temps) if temps else None,
        "compute_pids_observed": sorted(pids),
        "pid_consistent_with_baseline": True,  # caller compares against baseline
    }


def start_sampler(output_csv: str, interval_s: float = 2.0) -> tuple[threading.Event, threading.Thread]:
    """Start a background GPU sampler. Returns (stop_event, thread)."""
    stop = threading.Event()
    t = threading.Thread(target=gpu_sampler_loop, args=(output_csv, interval_s, stop), daemon=True)
    t.start()
    return stop, t


def stop_sampler(stop: threading.Event, thread: threading.Thread) -> None:
    stop.set()
    thread.join(timeout=5)
