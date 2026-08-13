#!/usr/bin/env python3
"""Fail-closed supervisor for the BC combo pair benchmark.

Mirrors perf48_supervisor: launches the combo benchmark command, owns only its
process tree, and validates the final COMBINATION_PAIR_RESULT.json conclusion.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

VALID_CONCLUSIONS = {"COMBO_PASS", "REJECTED_SEMANTIC_MISMATCH", "REJECTED_RUNTIME_FAILURE", "REJECTED_MECHANISM"}


def descendants(pid: int) -> list[int]:
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
        cur = stack.pop()
        out.append(cur)
        stack.extend(tree.get(cur, []))
    return sorted(set(out))


def stop_tree(pid: int, term_timeout: float = 2.0) -> list[int]:
    owned = descendants(pid)
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
        parent = {int(b.split()[0]): int(b.split()[1]) for b in rows if len(b.split()) == 2}

        def depth(x: int) -> int:
            n = 0
            while x in parent and parent[x] in owned and parent[x] != x:
                n += 1
                x = parent[x]
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
                os.kill(child, 0)
                alive.append(child)
            except OSError:
                pass
        if not alive:
            break
        time.sleep(.05)
    survivors = []
    for child in owned:
        try:
            os.kill(child, 0)
            survivors.append(child)
            os.kill(child, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass
    return owned


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_conclusion(conclusion: Any, returncode: int = 0) -> bool:
    return conclusion in VALID_CONCLUSIONS and (
        returncode == 0 or conclusion in VALID_CONCLUSIONS - {"COMBO_PASS"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-command-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        command = json.loads(args.pair_command_json)
    except Exception as exc:
        raise SystemExit(f"invalid --pair-command-json: {exc}")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise SystemExit("--pair-command-json must be a nonempty JSON list of strings")
    stdout = (out / "pair.stdout").open("w", encoding="utf-8")
    stderr = (out / "pair.stderr").open("w", encoding="utf-8")
    proc: subprocess.Popen[Any] | None = None
    try:
        proc = subprocess.Popen(command, start_new_session=True, stdout=stdout, stderr=stderr, text=True)
        atomic_json(out / "startup_health.json", {"pid": proc.pid, "command": command, "healthy": True})
        proc.wait()
        final_path = out / "COMBINATION_PAIR_RESULT.json"
        final = json.loads(final_path.read_text()) if final_path.exists() else None
        if not isinstance(final, dict) or not validate_conclusion(final.get("conclusion"), proc.returncode):
            raise RuntimeError("pair failed or final result is missing/invalid")
        atomic_json(out / "completion.json", {"pid": proc.pid, "returncode": proc.returncode,
                                              "command": command, "result": final})
    except Exception as exc:
        owned = stop_tree(proc.pid) if proc is not None else []
        atomic_json(out / "failure.json", {"error": str(exc), "owned_pids": owned})
        raise
    finally:
        stdout.close()
        stderr.close()


if __name__ == "__main__":
    main()
