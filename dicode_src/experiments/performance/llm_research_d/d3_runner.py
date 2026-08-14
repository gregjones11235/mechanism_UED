#!/usr/bin/env python3
"""D3 small-vs-large model runner contract.

The current entry point is metadata-only by default.  It validates the frozen
12-prompt replay and the shared request budget, then delegates the read-only
provider/GPU gate to :mod:`d3_metadata_gate`.  A formal generation run must be
implemented behind an explicit gate after the metadata result has been
reviewed; this file intentionally cannot send completions while the gate is
closed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3_metadata_gate as metadata_gate  # noqa: E402

SMALL_MODEL = "qwen2.5-coder:14b"
LARGE_MODEL = "deepseek-v4-flash"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}
EXPECTED_STAGES = {"early": 4, "mid": 4, "late": 4}
ARM_ORDER = (
    ("small", "r1"), ("large", "r1"),
    ("large", "r2"), ("small", "r2"),
    ("small", "r3"), ("large", "r3"),
)
POST_LIMIT = 108
MAX_REPAIRS_PER_SLOT = 2


class D3ConfigError(ValueError):
    pass


class ProviderPostBudget:
    """Shared hard cap for generation, transport retry, and repair POSTs."""

    def __init__(self, limit: int = POST_LIMIT):
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        self.limit = int(limit)
        self.used = {"ollama": 0, "deepseek_official": 0}

    def reserve(self, provider: str, *, kind: str = "generation") -> int:
        if provider not in self.used:
            self.used[provider] = 0
        if self.used[provider] >= self.limit:
            raise RuntimeError(f"D3 provider POST budget exhausted: {provider}")
        self.used[provider] += 1
        return self.used[provider]


def load_frozen_manifest(path: str | Path) -> dict[str, Any]:
    """Load the existing reconstructed 12-prompt manifest via local imports."""
    from llm_replay_manifest import load_manifest

    return load_manifest(path)


def validate_d3_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    prompts = manifest.get("user_prompts")
    if not isinstance(prompts, list) or len(prompts) != 12:
        raise D3ConfigError("D3 requires exactly 12 frozen prompts")
    stages = manifest.get("prompt_stages") or {}
    if {k: len(v) for k, v in stages.items()} != EXPECTED_STAGES:
        raise D3ConfigError(f"D3 prompt stages must be early/mid/late x4: {stages}")
    for key, expected in SAMPLING.items():
        if manifest.get(key) != expected:
            raise D3ConfigError(f"sampling field {key} must equal {expected!r}")
    if len(manifest.get("candidate_slots", [])) != 12:
        raise D3ConfigError("candidate slot count must remain 12")
    if len(manifest.get("request_order", [])) != 12:
        raise D3ConfigError("request order must remain 12")
    return manifest


def d3_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_d3_manifest(manifest)
    return {
        "classification": "D3_MODEL_COMPARISON_CONTRACT",
        "small_model": SMALL_MODEL,
        "large_model": LARGE_MODEL,
        "large_model_size": "UNKNOWN",
        "sampling": dict(SAMPLING),
        "prompt_count": 12,
        "prompt_stages": dict(EXPECTED_STAGES),
        "repeat_count_per_model": 3,
        "arm_order": [list(x) for x in ARM_ORDER],
        "max_repairs_per_slot": MAX_REPAIRS_PER_SLOT,
        "post_limit_per_provider": POST_LIMIT,
        "post_budget_scope": "generation + transport_retry + repair",
        "preflight": {"gpu_index": 2, "updates": 40, "num_envs": 1024, "rollout_steps": 128},
        "manifest_sha256": manifest.get("manifest_sha256"),
        "metadata_only_until_gate_review": True,
    }


def run_metadata_only(*, manifest_path: str | Path, output_dir: str | Path,
                      source_commit: str, source_branch: str,
                      ollama_base_url: str = metadata_gate.DEFAULT_OLLAMA_BASE,
                      deepseek_base_url: str = metadata_gate.DEFAULT_DEEPSEEK_BASE,
                      large_model: str = LARGE_MODEL) -> dict[str, Any]:
    manifest = validate_d3_manifest(load_frozen_manifest(manifest_path))
    contract = d3_contract(manifest)
    gate = metadata_gate.build_gate(
        source_commit=source_commit, source_branch=source_branch,
        ollama_base_url=ollama_base_url, deepseek_base_url=deepseek_base_url,
        large_model=large_model,
    )
    # Keep contract and gate in one independent result without any provider
    # response content.  The gate itself records completion/embedding calls = 0.
    result = {"classification": "D3_METADATA_ONLY_RESULT", "contract": contract, "gate": gate}
    metadata_gate.write_gate_artifacts(gate, output_dir)
    result_path = Path(output_dir) / "D3_CONTRACT.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(HERE / "FROZEN_MANIFEST.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--source-branch", default="unknown")
    parser.add_argument("--ollama-base-url", default=os.environ.get("D3_OLLAMA_BASE_URL", metadata_gate.DEFAULT_OLLAMA_BASE))
    parser.add_argument("--deepseek-base-url", default=os.environ.get("D3_DEEPSEEK_BASE_URL", metadata_gate.DEFAULT_DEEPSEEK_BASE))
    parser.add_argument("--large-model", default=os.environ.get("D3_LARGE_MODEL", LARGE_MODEL))
    args = parser.parse_args(argv)
    result = run_metadata_only(
        manifest_path=args.manifest, output_dir=args.output_dir,
        source_commit=args.source_commit, source_branch=args.source_branch,
        ollama_base_url=args.ollama_base_url, deepseek_base_url=args.deepseek_base_url,
        large_model=args.large_model,
    )
    gate = result["gate"]
    print(json.dumps({"gate_status": gate["gate_status"], "gpu2_smoke_allowed": gate["gpu2_smoke_allowed"],
                      "completion_requests_total": gate["completion_requests_total"],
                      "embedding_requests_total": gate["embedding_requests_total"]}, indent=2))
    return 0 if gate["gate_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
