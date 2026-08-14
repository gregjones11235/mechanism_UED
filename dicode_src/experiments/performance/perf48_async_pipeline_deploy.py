#!/usr/bin/env python3
"""Derive a source-complete async-pipeline manifest from a dual manifest."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_manifest = _load_sibling("perf48_async_manifest", "perf48_combo_manifest.py")

CLASSIFICATION = "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE"
ASYNC_REQUIRED_SOURCE_FILES = (
    "experiments/training/run_dicode.py",
    "src/dicode/skill_preflight/async_preflight.py",
    "src/dicode/training.py",
    "src/dicode/ppo_tr.py",
    "src/dicode/evaluation/online_evaluation.py",
    "src/dicode/scoring.py",
    "src/dicode/wrappers_cl.py",
    "experiments/performance/perf48_async_pipeline_deploy.py",
    "experiments/performance/perf48_async_pipeline_harness.py",
    "experiments/performance/perf48_async_pipeline_benchmark.py",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_sha256(document: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(document))
    value.pop("manifest_sha256", None)
    return _manifest.fingerprint(value)


def _hashed(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("result_sha256", None)
    result["result_sha256"] = _manifest.fingerprint(result)
    return result


def atomic_json(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    result = _hashed(document)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return result


def load_deploy_evidence(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if _hashed(document)["result_sha256"] != document.get("result_sha256"):
        raise ValueError("async deploy evidence hash mismatch")
    return document


def build_deploy(
    *, dual_manifest: str | Path, source: str | Path, source_commit: str,
    out: str | Path, validation_replay_receipt: str | Path | None = None,
) -> dict[str, Any]:
    parent_path = Path(dual_manifest).resolve()
    source_root = Path(source).resolve()
    out_path = Path(out)
    if out_path.exists():
        raise FileExistsError(f"async deploy output exists: {out_path}")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full lowercase git SHA")
    parent = _manifest.load_manifest(parent_path)
    dual = parent.get("dual_pipeline")
    if not isinstance(dual, Mapping) or dual.get("classification") != CLASSIFICATION:
        raise ValueError("parent is not a strict dual-pipeline manifest")
    parent_entries = parent.get("source_config", {}).get("source", {})
    if set(dual.get("required_source_files", [])) != set(parent_entries):
        raise ValueError("dual parent source set mismatch")
    required = tuple(dict.fromkeys((*parent_entries, *ASYNC_REQUIRED_SOURCE_FILES)))
    entries: dict[str, dict[str, str]] = {}
    for relative in required:
        path = (source_root / relative).resolve()
        if not path.is_file() or source_root not in path.parents:
            raise ValueError(f"missing or escaped async source: {relative}")
        entries[relative] = {"path": str(path), "sha256": file_sha256(path)}
    replay = None
    if validation_replay_receipt is not None:
        replay_path = Path(validation_replay_receipt).resolve()
        if not replay_path.is_file():
            raise ValueError("validation replay receipt missing")
        replay = {"path": str(replay_path), "file_sha256": file_sha256(replay_path)}
    out_path.mkdir(parents=True)
    derived = copy.deepcopy(parent)
    derived.pop("manifest_sha256", None)
    derived["source_config"]["source"] = entries
    derived["async_pipeline"] = {
        "classification": CLASSIFICATION,
        "source_commit": source_commit,
        "source_root": str(source_root),
        "required_source_files": list(required),
        "parent_manifest": {
            "path": str(parent_path),
            "file_sha256": file_sha256(parent_path),
            "manifest_sha256": parent["manifest_sha256"],
            "canonical_sha256": canonical_manifest_sha256(parent),
        },
        "matrix": {"stages": ["early", "mid", "late"], "repeats": [0, 1], "count": 6},
        "validation_cache": {
            "exercised": False,
            "speedup_included": False,
            "scope": "external_6f0625d_validation_replay_reference_only",
            "receipt": replay,
        },
    }
    manifest_path = out_path / "manifest.json"
    _manifest.write_manifest(derived, manifest_path)
    loaded = _manifest.load_manifest(manifest_path)
    evidence = atomic_json(
        out_path / "async_pipeline_deploy_evidence.json",
        {
            "classification": CLASSIFICATION,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": loaded["manifest_sha256"],
            "manifest_file_sha256": file_sha256(manifest_path),
            "parent_manifest": loaded["async_pipeline"]["parent_manifest"],
            "source_commit": source_commit,
            "source_root": str(source_root),
            "source": copy.deepcopy(entries),
            "validation_cache_exercised": False,
            "llm_api_calls": 0,
        },
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validation-replay-receipt")
    args = parser.parse_args(argv)
    print(json.dumps(build_deploy(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
