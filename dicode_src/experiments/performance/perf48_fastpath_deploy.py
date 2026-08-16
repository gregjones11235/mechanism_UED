#!/usr/bin/env python3
"""Build fastpath overlays and a source-bound manifest over frozen combo data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_manifest = _load_sibling("perf48_combo_manifest_fastpath_deploy", "perf48_combo_manifest.py")
_old_deploy = _load_sibling("perf48_combo_deploy_fastpath", "perf48_combo_deploy.py")
_config = _load_sibling("perf48_fastpath_config_deploy", "perf48_fastpath_config.py")

SOURCE_FILES = (
    "src/dicode/dreaming/gen_manager.py",
    "src/dicode/task_utils.py",
    "src/dicode/evaluation/online_evaluation.py",
    "src/dicode/ppo_tr.py",
    "src/dicode/scoring.py",
    "src/dicode/setup.py",
    "src/dicode/skill_preflight/preflight.py",
    "src/dicode/skill_preflight/preflight_route.py",
    "src/dicode/skill_preflight/reuse_loaded_tasks.py",
    "src/dicode/skill_preflight/learnability_summary.py",
    "src/dicode/craftax_evaluation.py",
    "src/dicode/wrappers_cl.py",
    "experiments/training/run_dicode.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_deploy(
    *, frozen_run: str | Path, base_config: str | Path,
    source: str | Path, out: str | Path,
) -> dict:
    frozen_run = Path(frozen_run)
    base_config = Path(base_config)
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    stage_spec = _old_deploy.build_stage_spec(frozen_run, base_config, out)
    for relative in SOURCE_FILES:
        path = source / relative
        if not path.is_file():
            raise ValueError(f"missing source file {path}")
        stage_spec["source"][relative] = str(path)

    overlay_paths = {}
    overlay_gates = {}
    for comparison, arms in _config.COMPARISONS.items():
        configs = []
        for arm in arms:
            target = out / "configs" / f"{arm.lower()}.yaml"
            config = _config.write_overlay_yaml(
                base_config, comparison=comparison, arm=arm, out_path=target
            )
            configs.append(config)
            overlay_paths[arm] = str(target)
            stage_spec["config"][arm.lower()] = str(target)
        overlay_gates[comparison] = _config.verify_overlay_pair(
            configs[0], configs[1], comparison=comparison
        )

    stage_spec_path = out / "stage_spec.json"
    stage_spec_path.write_text(
        json.dumps(stage_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = out / "manifest.json"
    _manifest.write_manifest(_manifest.build_combo_manifest(stage_spec), manifest_path)
    loaded = _manifest.load_manifest(manifest_path)
    record = {
        "manifest_sha256": loaded["manifest_sha256"],
        "frozen_manifest_classification": loaded["classification"],
        "fastpath_result_classification": "PERF48_FASTPATH_BENCHMARK",
        "frozen_run": str(frozen_run.resolve()),
        "base_config": {"path": str(base_config.resolve()), "sha256": _sha256(base_config)},
        "source": {
            relative: {"path": str((source / relative).resolve()), "sha256": _sha256(source / relative)}
            for relative in SOURCE_FILES
        },
        "configs": overlay_paths,
        "overlay_gates": overlay_gates,
        "validation_cache_exercised": False,
        "validation_cache_speedup_claimed": False,
    }
    (out / "fastpath_deploy_evidence.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", required=True)
    parser.add_argument("--config-base", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_deploy(
        frozen_run=args.frozen_run,
        base_config=args.config_base,
        source=args.source,
        out=args.out,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
