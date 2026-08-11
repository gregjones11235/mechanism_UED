"""Build and verify the frozen P0_PROFILE training-kernel replay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np

BUDGET = {"timesteps": 2_000_000_000, "num_envs": 1024, "num_steps": 128, "updates": 100}
STAGES = ("early", "mid", "late")
RNG_ALGORITHM = "sha256-little-endian-u32:P0_PROFILE_REPLAY_RNG_V1:{stage}:{repeat}"
CONDITIONING_TYPE = "one_hot"
CONDITIONING_DIM = 67
ARMS = ("E0", "P0_OFF", "P0_PROFILE")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> Any:
    """Canonicalize values before JSON hashing (including non-finite floats)."""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, dict):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(x) for x in value]
    return value


def fingerprint(value: Any) -> str:
    return sha256_bytes(json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Hash sorted relative paths and bytes, preventing path/content collisions."""
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(path).as_posix().encode()
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8, "little")); h.update(rel)
        h.update(len(data).to_bytes(8, "little")); h.update(data)
    return h.hexdigest()


def _u32_rng(stage: str, repeat: int) -> list[int]:
    digest = hashlib.sha256(f"P0_PROFILE_REPLAY_RNG_V1:{stage}:{repeat}".encode()).digest()
    return [int.from_bytes(digest[0:4], "little"), int.from_bytes(digest[4:8], "little")]


def _conditioning_info(path: Path, task_count: int, conditioning_dim: int = CONDITIONING_DIM) -> dict[str, Any]:
    """Validate and fingerprint the frozen one-hot conditioning table."""
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"unable to load conditioning table: {path}") from exc
    if values.dtype != np.dtype("float32") or values.ndim != 2 or values.shape[0] != task_count + 1 or values.shape[1] != conditioning_dim:
        raise ValueError("conditioning table must be finite float32 [task_count+1, 67]")
    if not np.isfinite(values).all():
        raise ValueError("conditioning table contains non-finite values")
    values = np.ascontiguousarray(values)
    h = hashlib.sha256(); h.update(repr(tuple(values.shape)).encode()); h.update(str(values.dtype).encode()); h.update(values.tobytes())
    return {"path": str(path), "sha256": _file_sha256(path), "content_sha256": h.hexdigest(), "shape": list(values.shape), "dtype": str(values.dtype)}


def _resolve(path: str | os.PathLike[str], base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def _parse_bool(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _graph_tasks(graph_path: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    try:
        import networkx as nx
        graph = nx.read_graphml(graph_path)
    except Exception as exc:
        raise ValueError(f"unable to read GraphML: {graph_path}") from exc
    nodes = {str(n): attrs for n, attrs in graph.nodes(data=True)}
    by_attr: dict[str, dict[str, Any]] = {}
    for node, attrs in nodes.items():
        for key in ("id", "task_id", "taskId"):
            if key in attrs:
                by_attr[str(attrs[key])] = attrs
    result = []
    for task_id in task_ids:
        attrs = nodes.get(str(task_id)) or by_attr.get(str(task_id))
        if attrs is None:
            raise ValueError(f"task id {task_id!r} missing from graph")
        code = attrs.get("code", attrs.get("task_code", ""))
        if code is None or not str(code).strip():
            raise ValueError(f"task id {task_id!r} has empty code")
        description = attrs.get("description", "")
        result.append({
            "id": str(task_id),
            "code_sha256": sha256_bytes(str(code).encode()),
            "description_sha256": sha256_bytes(str(description).encode()),
            "session_created": attrs.get("session_created", attrs.get("sessionCreated")),
            "is_active": _parse_bool(attrs.get("is_active", attrs.get("isActive", True))),
        })
    return result


def _mapping(spec: Mapping[str, Any], names: tuple[str, ...], base: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    aliases = {"source": ("source", "source_files"), "config": ("config", "config_files")}
    for section in ("source", "config"):
        value = next((spec.get(alias) for alias in aliases[section] if spec.get(alias) is not None), None)
        if value is None:
            continue
        if isinstance(value, list):
            value = {str(Path(v).name): v for v in value}
        if not isinstance(value, Mapping):
            raise ValueError(f"{section} must be a file mapping")
        section_out = {}
        for label, raw in value.items():
            if isinstance(raw, Mapping):
                raw = raw.get("path")
            if not isinstance(raw, (str, os.PathLike)):
                raise ValueError(f"{section} mapping entry {label!r} requires a path")
            p = _resolve(raw, base)
            if not p.is_file():
                raise ValueError(f"missing {section} file: {p}")
            section_out[str(label)] = {"path": str(p), "sha256": _file_sha256(p)}
        out[section] = section_out
    if set(out) != {"source", "config"} or not out["source"] or not out["config"]:
        raise ValueError("stage spec requires both source and config file mappings")
    return out


def _stage_specs(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    stages = spec.get("stages")
    if isinstance(stages, Mapping):
        return {s: stages[s] for s in STAGES if s in stages}
    if isinstance(stages, list):
        return {str(x["name"]): x for x in stages if isinstance(x, Mapping) and "name" in x}
    return {s: spec[s] for s in STAGES if isinstance(spec.get(s), Mapping)}


def _build_strict(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("budget") != BUDGET:
        raise ValueError(f"budget must equal {BUDGET}")
    if spec.get("conditioning_type") != CONDITIONING_TYPE:
        raise ValueError("conditioning_type must be one_hot")
    base = Path(spec.get("base_dir", ".")).resolve()
    mappings = _mapping(spec, ("source", "config", "source_files", "config_files"), base)
    stages = _stage_specs(spec)
    if set(stages) != set(STAGES):
        raise ValueError("stage spec must contain exactly early, mid, and late stages")
    manifest_stages = []
    for name in STAGES:
        st = stages[name]
        graph_raw = st.get("graph", st.get("graphml", st.get("graph_path")))
        checkpoint_raw = st.get("checkpoint", st.get("checkpoint_dir", st.get("checkpoint_path")))
        if not graph_raw or not checkpoint_raw:
            raise ValueError(f"{name} requires graph and checkpoint paths")
        graph_path = _resolve(graph_raw, base)
        checkpoint = _resolve(checkpoint_raw, base)
        if not graph_path.is_file() or not checkpoint.is_dir():
            raise ValueError(f"invalid graph/checkpoint path for {name}")
        task_ids = [str(x) for x in st.get("task_ids", [])]
        if not task_ids or any(not x.strip() for x in task_ids) or len(set(task_ids)) != len(task_ids):
            raise ValueError(f"{name} task_ids must be unique non-empty strings")
        try:
            global_step = int(st["global_step"]); env_steps = int(st["initial_env_steps"])
        except Exception as exc:
            raise ValueError(f"{name} global_step/initial_env_steps required") from exc
        if env_steps != global_step * 1024 * 128:
            raise ValueError(f"{name} initial_env_steps mismatch")
        nums = re.findall(r"\d+", checkpoint.name)
        if not nums or int(nums[-1]) != global_step:
            raise ValueError(f"{name} checkpoint basename/global_step mismatch")
        files = [p for p in checkpoint.rglob("*") if p.is_file()]
        if not any(p.name == "_CHECKPOINT_METADATA" for p in files):
            raise ValueError(f"{name} checkpoint missing _CHECKPOINT_METADATA")
        if "archive_reconstruction_limit" not in st or not isinstance(st["archive_reconstruction_limit"], str) or not st["archive_reconstruction_limit"].strip():
            raise ValueError(f"{name} archive_reconstruction_limit non-empty text required")
        tasks = _graph_tasks(graph_path, task_ids)
        conditioning_raw = st.get("conditioning_path")
        if not conditioning_raw:
            raise ValueError(f"{name} requires frozen conditioning_path")
        conditioning = _conditioning_info(_resolve(conditioning_raw, base), len(task_ids))
        repeats = [{"repeat": i, "order": ["P0_OFF", "P0_PROFILE"] if i == 0 else ["P0_PROFILE", "P0_OFF"], "rng": _u32_rng(name, i)} for i in range(2)]
        manifest_stages.append({
            "name": name, "graph": {"path": str(graph_path), "sha256": _file_sha256(graph_path)},
            "checkpoint": {"path": str(checkpoint), "sha256": _tree_sha256(checkpoint), "basename": checkpoint.name},
            "conditioning": conditioning,
            "global_step": global_step, "initial_env_steps": env_steps, "task_ids": task_ids, "task_count": len(task_ids),
            "tasks": tasks, "archive_reconstruction_limit": st["archive_reconstruction_limit"],
            "embedding": {"hash": conditioning["content_sha256"], "shape": conditioning["shape"], "dtype": conditioning["dtype"]},
            "repeats": repeats, "orders": [r["order"] for r in repeats], "rng": [r["rng"] for r in repeats],
        })
    return {
        "classification": "TRAINING_KERNEL_BENCHMARK", "not_end_to_end_ued": True, "llm_api_calls": 0,
        "budget": dict(BUDGET), "conditioning_type": CONDITIONING_TYPE, "conditioning_dim": CONDITIONING_DIM, "stage_count": 3, "pair_count": 6,
        "source_config": mappings, "stages": manifest_stages,
        "source_gate": {"stage": "early", "repeat": 0, "order": ["E0", "P0_OFF"]},
        "rng_algorithm": RNG_ALGORITHM,
    }


def build_manifest(stage_spec: Mapping[str, Any]) -> dict[str, Any]:
    return _build_strict(stage_spec)


def _without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "manifest_sha256"}


def write_manifest(manifest: Mapping[str, Any], output: str | os.PathLike[str]) -> dict[str, Any]:
    data = dict(manifest); data.pop("manifest_sha256", None); data["manifest_sha256"] = fingerprint(_without_hash(data))
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(canonical(data), f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return data


def _verify_task_hashes(stage: Mapping[str, Any]) -> None:
    graph = Path(stage["graph"]["path"])
    tasks = _graph_tasks(graph, [str(x) for x in stage["task_ids"]])
    if tasks != stage["tasks"]:
        raise ValueError(f"task hashes changed for {stage['name']}")


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_sha256") != fingerprint(_without_hash(data)):
        raise ValueError("manifest_sha256 mismatch")
    if data.get("classification") != "TRAINING_KERNEL_BENCHMARK" or data.get("not_end_to_end_ued") is not True or data.get("llm_api_calls") != 0:
        raise ValueError("invalid manifest classification gates")
    if data.get("budget") != BUDGET or data.get("conditioning_type") != CONDITIONING_TYPE or data.get("conditioning_dim") != CONDITIONING_DIM or data.get("stage_count") != 3 or data.get("pair_count") != 6:
        raise ValueError("invalid manifest budget/stage counts")
    stages = data.get("stages", [])
    if len(stages) != 3 or [s.get("name") for s in stages] != list(STAGES):
        raise ValueError("invalid stage set/order")
    for stage in stages:
        ids = [str(x) for x in stage.get("task_ids", [])]
        if not ids or len(ids) != len(set(ids)) or any(not x.strip() for x in ids):
            raise ValueError("invalid task id order")
        if stage.get("orders") != [["P0_OFF", "P0_PROFILE"], ["P0_PROFILE", "P0_OFF"]]:
            raise ValueError("invalid pair order")
        if stage.get("task_count") != len(stage.get("task_ids", [])) or len(stage.get("repeats", [])) != 2:
            raise ValueError("invalid task/repeat count")
        if [r.get("order") for r in stage.get("repeats", [])] != stage.get("orders"):
            raise ValueError("repeat order mismatch")
        expected_rng = [_u32_rng(stage["name"], 0), _u32_rng(stage["name"], 1)]
        if stage.get("rng") != expected_rng or [r.get("rng") for r in stage.get("repeats", [])] != expected_rng:
            raise ValueError("invalid replay RNG")
        if int(stage["initial_env_steps"]) != int(stage["global_step"]) * 1024 * 128:
            raise ValueError("invalid initial_env_steps")
        if _file_sha256(Path(stage["graph"]["path"])) != stage["graph"]["sha256"]:
            raise ValueError("graph file hash changed")
        cp = Path(stage["checkpoint"]["path"])
        nums = re.findall(r"\d+", cp.name)
        if not cp.is_dir() or not nums or int(nums[-1]) != int(stage["global_step"]):
            raise ValueError("checkpoint basename/global_step mismatch")
        if _tree_sha256(cp) != stage["checkpoint"]["sha256"]:
            raise ValueError("checkpoint hash changed")
        if not any(p.name == "_CHECKPOINT_METADATA" for p in cp.rglob("*") if p.is_file()):
            raise ValueError("checkpoint metadata missing")
        _verify_task_hashes(stage)
        conditioning = stage.get("conditioning", {})
        cond = _conditioning_info(Path(conditioning.get("path", "")), len(ids))
        if cond != conditioning:
            raise ValueError("conditioning table changed")
        if stage.get("embedding", {}).get("hash") != cond["content_sha256"]:
            raise ValueError("conditioning compatibility hash changed")
    for section in data.get("source_config", {}).values():
        for entry in section.values():
            if _file_sha256(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("source/config file hash changed")
    if data.get("source_gate") != {"stage": "early", "repeat": 0, "order": ["E0", "P0_OFF"]}:
        raise ValueError("invalid source gate")
    return data


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--stage-spec", required=True); p.add_argument("--output", required=True)
    args = p.parse_args(); spec_path = Path(args.stage_spec) if not args.stage_spec.lstrip().startswith("{") else None
    if spec_path is not None and spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8")); spec.setdefault("base_dir", str(spec_path.parent))
    else:
        spec = json.loads(args.stage_spec); spec.setdefault("base_dir", str(Path.cwd()))
    write_manifest(build_manifest(spec), args.output)


if __name__ == "__main__":
    main()
