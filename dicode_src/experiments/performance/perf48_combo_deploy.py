#!/usr/bin/env python3
"""Deploy helper for the BC combo experiment against the FROZEN P0 materials.

Given the P0 run directory (perf48_p0_gpu3_91a75e5_20260811T110147Z), this:
  1. reads the frozen manifest.json stage metadata (task_ids / global_step /
     initial_env_steps / archive_reconstruction_limit)
  2. emits a combo stage-spec that reuses the SAME frozen checkpoint,
     conditioning.npy, task_graph.graphml and RNG pairing (never re-selected)
  3. generates the BC_OFF / BC_ON config overlays from configs/perf48_off.yaml
  4. builds + verifies the combo manifest (full SHA-256 of every input)

Pure CPU logic: no JAX/GPU required. Outputs (all under --out)::

    stage_spec.json          input to perf48_combo_manifest.py
    configs/bc_off.yaml
    configs/bc_on.yaml
    manifest.json            built + verified combo manifest
    sha256_manifest.json     every frozen input file + tree SHA-256
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

_COMBO = importlib.util.spec_from_file_location(
    "perf48_combo_manifest", Path(__file__).with_name("perf48_combo_manifest.py"))
_combo = importlib.util.module_from_spec(_COMBO)
assert _COMBO.loader
_COMBO.loader.exec_module(_combo)

_COMBO_CONFIG = importlib.util.spec_from_file_location(
    "perf48_combo_config", Path(__file__).with_name("perf48_combo_config.py"))
_combo_config = importlib.util.module_from_spec(_COMBO_CONFIG)
assert _COMBO_CONFIG.loader
_COMBO_CONFIG.loader.exec_module(_combo_config)

STAGES = ("early", "mid", "late")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = p.relative_to(path).as_posix().encode()
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8, "little")); h.update(rel)
        h.update(len(data).to_bytes(8, "little")); h.update(data)
    return h.hexdigest()


def _frozen_manifest(frozen_run: Path) -> dict[str, Any]:
    path = frozen_run / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    stages = {s["name"]: s for s in data.get("stages", [])}
    if set(stages) != set(STAGES):
        raise ValueError(f"frozen manifest must contain exactly {STAGES}")
    return stages


def build_stage_spec(frozen_run: Path, base_config: Path, out: Path) -> dict[str, Any]:
    frozen_run = Path(frozen_run)
    stages_meta = _frozen_manifest(frozen_run)
    stage_spec = {
        "base_dir": str(out),
        "budget": {"timesteps": 2_000_000_000, "num_envs": 1024, "num_steps": 128, "updates": 100},
        "conditioning_type": "one_hot",
        "source": {},
        "config": {"base": str(base_config)},
        "stages": {},
    }
    for name in STAGES:
        meta = stages_meta[name]
        stage_dir = frozen_run / "stages" / name
        stage_spec["stages"][name] = {
            "graph": str(stage_dir / "task_graph.graphml"),
            "checkpoint": str(stage_dir / "checkpoint"),
            "conditioning_path": str(stage_dir / "conditioning.npy"),
            "task_ids": [str(x) for x in meta["task_ids"]],
            "global_step": int(meta["global_step"]),
            "initial_env_steps": int(meta["initial_env_steps"]),
            "archive_reconstruction_limit": meta.get("archive_reconstruction_limit", "all"),
            "candidate_codes_dir": str(out / "frozen" / name / "candidate_codes"),
        }
    return stage_spec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-run", required=True,
                   help="P0 run dir: perf48_p0_gpu3_91a75e5_20260811T110147Z")
    p.add_argument("--config-base", required=True,
                   help="frozen base config (configs/perf48_off.yaml)")
    p.add_argument("--out", required=True, help="deploy target directory")
    p.add_argument("--source", required=True, help="dicode_src directory (source mapping)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    # source mapping: the runtime-bound files (verified by the harness)
    source_rel = (
        "src/dicode/dreaming/gen_manager.py",
        "src/dicode/task_utils.py",
        "src/dicode/evaluation/online_evaluation.py",
        "src/dicode/ppo_tr.py",
        "src/dicode/scoring.py",
        "src/dicode/setup.py",
        "src/dicode/skill_preflight/preflight.py",
        "src/dicode/skill_preflight/preflight_route.py",
        "src/dicode/skill_preflight/reuse_loaded_tasks.py",
        "src/dicode/craftax_evaluation.py",
        "src/dicode/wrappers_cl.py",
    )
    stage_spec = build_stage_spec(args.frozen_run, args.config_base, out)
    for rel in source_rel:
        path = source / rel
        if not path.is_file():
            raise ValueError(f"missing source file {path}")
        stage_spec["source"][rel] = str(path)

    spec_path = out / "stage_spec.json"
    spec_path.write_text(json.dumps(stage_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # config overlays
    cfg_off = out / "configs" / "bc_off.yaml"
    cfg_on = out / "configs" / "bc_on.yaml"
    _combo_config.write_overlay_yaml(args.config_base, arm="BC_OFF", out_path=cfg_off)
    _combo_config.write_overlay_yaml(args.config_base, arm="BC_ON", out_path=cfg_on)
    off = _combo_config.load_overlay(cfg_off)
    on = _combo_config.load_overlay(cfg_on)
    verdict = _combo_config.verify_overlay_pair(off, on)
    if not verdict["valid"]:
        raise RuntimeError(f"overlay gate failed: {verdict}")

    # build + verify manifest
    manifest = _combo.build_combo_manifest(stage_spec)
    manifest_path = out / "manifest.json"
    _combo.write_manifest(manifest, manifest_path)
    loaded = _combo.load_manifest(manifest_path)

    # full SHA-256 record of every frozen input
    sha_record: dict[str, Any] = {"frozen_run": str(Path(args.frozen_run).resolve()),
                                  "config_base": {"path": str(Path(args.config_base).resolve()),
                                                  "sha256": _file_sha256(Path(args.config_base))}}
    for name in STAGES:
        meta = next(s for s in loaded["stages"] if s["name"] == name)
        sha_record[f"stages.{name}"] = {
            "graph": {"path": meta["graph"]["path"], "sha256": meta["graph"]["sha256"]},
            "checkpoint": {"path": meta["checkpoint"]["path"], "tree_sha256": meta["checkpoint"]["sha256"]},
            "conditioning": {"path": meta["conditioning"]["path"], "sha256": meta["conditioning"]["sha256"],
                             "content_sha256": meta["conditioning"]["content_sha256"]},
            "candidate_codes": {tid: e["sha256"] for tid, e in meta["candidate_codes"].items()},
        }
    sha_record["manifest_sha256"] = loaded["manifest_sha256"]
    sha_record["overlay_diff_paths"] = verdict["diff_paths"]
    (out / "sha256_manifest.json").write_text(json.dumps(sha_record, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    print(json.dumps({"overlay_diff_paths": verdict["diff_paths"],
                      "manifest_sha256": loaded["manifest_sha256"],
                      "outputs": [str(spec_path), str(cfg_off), str(cfg_on), str(manifest_path)]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
