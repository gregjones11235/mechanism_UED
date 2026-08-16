#!/usr/bin/env python3
"""Derive a source-complete dual-pipeline manifest from frozen fastpath data."""
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


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_manifest = _load_sibling("perf48_dual_deploy_manifest", "perf48_combo_manifest.py")
_fastpath_deploy = _load_sibling(
    "perf48_dual_deploy_fastpath", "perf48_fastpath_deploy.py"
)

CLASSIFICATION = "RESEARCH_SCHEDULE_CHANGE_NOT_SEMANTIC_MAINLINE"
BASE_SOURCE_FILES = tuple(_fastpath_deploy.SOURCE_FILES)
DUAL_SOURCE_FILES = (
    "src/dicode/training.py",
    "experiments/performance/perf48_dual_pipeline_harness.py",
    "experiments/performance/perf48_dual_pipeline_benchmark.py",
    "experiments/performance/perf48_dual_pipeline_deploy.py",
)
ALL_SOURCE_FILES = BASE_SOURCE_FILES + DUAL_SOURCE_FILES


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("result_sha256", None)
    result["result_sha256"] = _manifest.fingerprint(result)
    return result


def _atomic_json(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    result = _hashed_document(document)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return result


def load_deploy_evidence(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if _hashed_document(document).get("result_sha256") != document.get(
        "result_sha256"
    ):
        raise ValueError("dual deploy evidence hash mismatch")
    return document


def build_deploy(
    *, fastpath_manifest: str | Path, source: str | Path,
    source_commit: str, out: str | Path,
) -> dict[str, Any]:
    parent_path = Path(fastpath_manifest).resolve()
    source_root = Path(source).resolve()
    out_path = Path(out)
    if out_path.exists():
        raise FileExistsError(f"dual deploy output exists: {out_path}")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full lowercase git SHA")
    parent = _manifest.load_manifest(parent_path)
    parent_sources = parent.get("source_config", {}).get("source", {})
    if set(parent_sources) != set(BASE_SOURCE_FILES):
        raise ValueError("fastpath parent manifest source set mismatch")

    source_entries: dict[str, dict[str, str]] = {}
    for relative in ALL_SOURCE_FILES:
        path = (source_root / relative).resolve()
        if not path.is_file() or source_root not in path.parents:
            raise ValueError(f"missing or escaped dual source file: {relative}")
        source_entries[relative] = {
            "path": str(path),
            "sha256": _file_sha256(path),
        }

    out_path.mkdir(parents=True)
    derived = copy.deepcopy(parent)
    derived.pop("manifest_sha256", None)
    derived["source_config"]["source"] = source_entries
    derived["dual_pipeline"] = {
        "classification": CLASSIFICATION,
        "source_commit": source_commit,
        "source_root": str(source_root),
        "required_source_files": list(ALL_SOURCE_FILES),
        "parent_manifest": {
            "path": str(parent_path),
            "manifest_sha256": parent["manifest_sha256"],
            "file_sha256": _file_sha256(parent_path),
        },
    }
    manifest_path = out_path / "manifest.json"
    _manifest.write_manifest(derived, manifest_path)
    loaded = _manifest.load_manifest(manifest_path)
    evidence = _atomic_json(
        out_path / "dual_pipeline_deploy_evidence.json",
        {
            "classification": CLASSIFICATION,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": loaded["manifest_sha256"],
            "manifest_file_sha256": _file_sha256(manifest_path),
            "parent_manifest": dict(loaded["dual_pipeline"]["parent_manifest"]),
            "source_commit": source_commit,
            "source_root": str(source_root),
            "source": copy.deepcopy(source_entries),
            "llm_api_calls": 0,
        },
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastpath-manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_deploy(
                fastpath_manifest=args.fastpath_manifest,
                source=args.source,
                source_commit=args.source_commit,
                out=args.out,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
