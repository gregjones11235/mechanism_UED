"""Run manifest system for DiCode DeepSeek-V4-Pro substitution baseline.

Provides:
- Manifest schema definition and validation
- Unique output directory creation with collision detection
- Manifest serialization to JSON
"""

import hashlib
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from dicode.dspro.config import DSPRO_CONFIG, DsproConfig


# ==============================================================================
# Manifest schema
# ==============================================================================

MANIFEST_SCHEMA_VERSION = "1.0.0"

REQUIRED_MANIFEST_FIELDS = [
    "manifest_schema_version",
    "run_id",
    "timestamp_utc",
    "branch",
    "commit",
    "python_version",
    "experiment_label",
    "provider",
    "requested_model",
    "returned_model",
    "model_id",
    "seed",
    "output_path",
]

OPTIONAL_MANIFEST_FIELDS = [
    "jax_version",
    "physical_gpu",
    "logical_gpu",
    "environment_steps",
    "max_environment_steps",
    "environment_steps_completed",
    "temperature",
    "max_tokens",
    "top_p",
    "think",
    "reasoning_effort",
    "max_retries",
    "prompt_version",
    "schema_version",
    "cache_version",
    "accounting_namespace",
    "exit_code",
    "api_call_count",
    "api_input_tokens",
    "api_output_tokens",
    "api_estimated_cost",
    "cache_hits",
    "cache_misses",
    "generation_statistics",
    "compilation_statistics",
    "checkpoint_path",
    "log_path",
    "phase",
    "phase_status",
    "duration_seconds",
    "command",
    "notes",
]


def generate_run_id() -> str:
    """Generate a unique run ID with timestamp and random component."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_uuid = uuid.uuid4().hex[:8]
    return f"dspro-gpu0-{ts}-{short_uuid}"


def resolve_output_dir(run_id: str, config: Optional[DsproConfig] = None) -> str:
    """Resolve the output directory for a run.

    Aborts if the directory already exists (no overwrite policy).

    Args:
        run_id: Unique run identifier.
        config: DsproConfig instance (uses singleton if None).

    Returns:
        Absolute path to the new output directory.

    Raises:
        FileExistsError: If the output directory already exists.
    """
    if config is None:
        config = DSPRO_CONFIG

    output_dir = os.path.join(config.output_root, run_id)

    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            f"Run collision detected. Each run requires a unique directory.\n"
            f"Use a different run_id or remove the existing directory manually."
        )

    return output_dir


def create_output_dir(run_id: str, config: Optional[DsproConfig] = None) -> str:
    """Create a unique output directory atomically.

    Args:
        run_id: Unique run identifier.
        config: DsproConfig instance.

    Returns:
        Absolute path to the created output directory.

    Raises:
        FileExistsError: If the output directory already exists.
        OSError: If directory creation fails.
    """
    output_dir = resolve_output_dir(run_id, config)
    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def get_jax_version() -> str:
    """Get the installed JAX version string."""
    try:
        import jax
        return jax.__version__
    except ImportError:
        return "unknown"


def get_git_info() -> dict[str, str]:
    """Get current git branch and commit."""
    import subprocess
    branch = "unknown"
    commit = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            text=True,
        ).strip()
    except Exception:
        pass
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            text=True,
        ).strip()
    except Exception:
        pass
    return {"branch": branch, "commit": commit}


def create_manifest(
    run_id: Optional[str] = None,
    seed: int = 0,
    environment_steps: int = 0,
    phase: str = "G0",
    output_path: Optional[str] = None,
    config: Optional[DsproConfig] = None,
    **extra_fields,
) -> dict[str, Any]:
    """Create a run manifest dictionary.

    Args:
        run_id: Unique run ID (generated if None).
        seed: Random seed.
        environment_steps: Number of environment steps (0 for non-training phases).
        phase: Current phase label (G0-G5).
        output_path: Resolved output directory path.
        config: DsproConfig instance.
        **extra_fields: Additional fields to include.

    Returns:
        Manifest dictionary.
    """
    if config is None:
        config = DSPRO_CONFIG

    if run_id is None:
        run_id = generate_run_id()

    if output_path is None:
        output_path = resolve_output_dir(run_id, config)

    git_info = get_git_info()

    manifest = {
        # Required fields
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git_info["branch"],
        "commit": git_info["commit"],
        "python_version": sys.version.split()[0],
        "experiment_label": config.experiment_label,
        "provider": config.provider,
        "requested_model": config.model_id,
        "returned_model": None,  # Populated after first API response
        "model_id": config.model_id,
        "seed": seed,
        "output_path": output_path,

        # Optional fields
        "jax_version": get_jax_version(),
        "physical_gpu": os.getenv("CUDA_VISIBLE_DEVICES", "unset"),
        "logical_gpu": None,
        "environment_steps": environment_steps,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "think": config.think,
        "reasoning_effort": config.reasoning_effort,
        "max_retries": config.max_retries,
        "prompt_version": config.prompt_version,
        "schema_version": config.schema_version,
        "cache_version": config.cache_version,
        "accounting_namespace": config.accounting_namespace,
        "phase": phase,
        "phase_status": "PENDING",
        "api_call_count": 0,
        "api_input_tokens": 0,
        "api_output_tokens": 0,
        "api_estimated_cost": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    # Merge extra fields
    manifest.update(extra_fields)

    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate a manifest dictionary.

    Args:
        manifest: Manifest dict to validate.

    Returns:
        List of validation error messages (empty = valid).
    """
    errors = []

    # Check required fields.
    # returned_model may be null before the first API call.
    NULLABLE_REQUIRED = {"returned_model"}

    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            errors.append(f"Missing required manifest field: {field}")
        elif field not in NULLABLE_REQUIRED and manifest[field] is None:
            errors.append(f"Required manifest field is null: {field}")

    # Check schema version
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"Manifest schema version mismatch: "
            f"got {manifest.get('manifest_schema_version')}, "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )

    # Check model_id is not a legacy alias (silent mapping to wrong model)
    model_id = manifest.get("model_id", "")
    requested = manifest.get("requested_model", "")
    if model_id in DSPRO_CONFIG.legacy_aliases or requested in DSPRO_CONFIG.legacy_aliases:
        errors.append(
            f"Model ID '{model_id or requested}' is a legacy alias. "
            f"Legacy aliases (deepseek-chat, deepseek-reasoner) silently map to "
            f"deepseek-v4-flash, not deepseek-v4-pro. "
            f"Use the official model ID 'deepseek-v4-pro'."
        )

    # Check model_id is not a generic alias
    if model_id in ("latest", "auto", ""):
        errors.append(f"Manifest model_id '{model_id}' is a disallowed alias.")

    # === GATE: returned_model must match requested_model when both are set ===
    returned = manifest.get("returned_model")
    if requested and returned and returned != requested:
        msg = (
            f"MODEL MISMATCH GATE FAILED: "
            f"requested_model='{requested}' but API returned_model='{returned}'. "
        )
        if returned in ("deepseek-v4-flash", "deepseek-chat"):
            msg += (
                f"This indicates the request was silently mapped via a legacy alias. "
                f"The pinned model 'deepseek-v4-pro' must be used directly."
            )
        else:
            msg += "The request was mapped to an unexpected model."
        errors.append(msg)

    # Check provider
    if manifest.get("provider") != DSPRO_CONFIG.provider:
        errors.append(
            f"Manifest provider '{manifest.get('provider')}' "
            f"does not match config provider '{DSPRO_CONFIG.provider}'."
        )

    # Check model_id matches config
    if manifest.get("model_id") != DSPRO_CONFIG.model_id:
        errors.append(
            f"Manifest model_id '{manifest.get('model_id')}' "
            f"does not match config model_id '{DSPRO_CONFIG.model_id}'."
        )

    return errors


def write_manifest(manifest: dict[str, Any], output_dir: str) -> str:
    """Write manifest to JSON file in the output directory.

    Args:
        manifest: Manifest dictionary.
        output_dir: Directory to write manifest to.

    Returns:
        Path to the written manifest file.

    Raises:
        ValueError: If manifest validation fails.
    """
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(f"Manifest validation failed: {'; '.join(errors)}")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path


def read_manifest(manifest_path: str) -> dict[str, Any]:
    """Read and validate a manifest from disk.

    Args:
        manifest_path: Path to manifest.json.

    Returns:
        Manifest dictionary.

    Raises:
        FileNotFoundError: If manifest doesn't exist.
        ValueError: If manifest validation fails.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(f"Manifest validation failed: {'; '.join(errors)}")

    return manifest


def compute_artifact_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for artifact validation."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
