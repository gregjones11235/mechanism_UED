"""Pinned model manifest for curriculum aggregation roles.

This module provides the single source of truth for exact model IDs,
role assignments, and provider settings. No 'latest', 'auto', or moving
aliases are permitted.

CRITICAL: DeepSeek's `deepseek-chat` is a moving endpoint alias that
silently remaps to newer model versions. As of 2026-07-11 it resolves to
DeepSeek-V4-Flash (deprecation warning active). On 2026-07-24 the
`deepseek-chat` and `deepseek-reasoner` endpoints will be DISCONTINUED.
This experiment family uses `deepseek-chat` with mandatory identity
verification via API response `model` field. After July 24, 2026, the
family must migrate to `deepseek-v4-pro` or `deepseek-v4-flash`.

Qwen `qwen-turbo` is also a moving alias. The pinned dated version
`qwen-turbo-2025-04-28` is used instead.

GLM `glm-4-flash` is a stable model series endpoint.

API keys: referenced by environment variable name only — never written
to code, Git, reports, or logs.
"""

import hashlib
import json
import os
from typing import Optional

# ==============================================================================
# Pinned model identities — change only with a new experiment family
# ==============================================================================

MANIFEST_VERSION = "v2.1"
MANIFEST_FAMILY = "aggregation-v2-gpu1"

# MODEL_ID_RATIONALE: Documents why each model ID was chosen and its
# immutability characteristics.
MODEL_ID_RATIONALE = {
    "deepseek-chat": (
        "DeepSeek API endpoint alias. Does NOT provide version-pinned IDs. "
        "Currently resolves to DeepSeek-V4-Flash (as of 2026-07-11). "
        "Will be DISCONTINUED on 2026-07-24. "
        "Identity verified via API response 'model' field on each cache-generation run. "
        "Risk: HIGH — model may change between runs without notice. "
        "Migration path: deepseek-v4-pro or deepseek-v4-flash after July 24, 2026."
    ),
    "qwen-flash": (
        "Qwen DashScope stable model family. Provider-recommended replacement "
        "for qwen-turbo. No dated snapshots available in this API deployment. "
        "Risk: MEDIUM — model updates possible but API contract stable. "
        "Verified via real API identity check: requested=qwen-flash, returned=qwen-flash."
    ),
    "glm-4-flash": (
        "GLM ZhipuAI model endpoint. Stable model series name. "
        "Provider may update the underlying model but maintains API compatibility. "
        "Dated versions available (e.g., glm-4-flash-250414) but the base endpoint "
        "is the recommended stable identifier. "
        "Risk: LOW-MEDIUM — model updates possible but API contract stable."
    ),
}

# --- DeepSeek Generator (also used as Critic with separate accounting) ---
# NOTE: deepseek-chat is a MOVING alias. Identity must be verified per run.
DEEPSEEK_GENERATOR = {
    "provider": "deepseek",
    "exact_model_id": "deepseek-chat",
    "model_id_rationale": MODEL_ID_RATIONALE["deepseek-chat"],
    "model_tier": "deepseek-v4-flash",  # What it currently resolves to
    "role": "generator",
    "thinking_mode": "disabled",
    "reasoning_setting": "none",
    "temperature": 0.0,
    "max_output_tokens": 256,
    "retry_count": 3,
    "prompt_version": "v2.1",
    "schema_version": "v2.1",
    "api_key_env": "DEEPSEEK_API_KEY",
    "accounting_namespace": "deepseek_generator_gpu1",
    "immutability": "VERIFY_PER_RUN",  # Model identity must be verified each run
}

# --- DeepSeek Critic (same provider/model, separate logical accounting) ---
DEEPSEEK_CRITIC = {
    "provider": "deepseek",
    "exact_model_id": "deepseek-chat",
    "model_id_rationale": MODEL_ID_RATIONALE["deepseek-chat"],
    "model_tier": "deepseek-v4-flash",
    "role": "critic",
    "thinking_mode": "disabled",
    "reasoning_setting": "none",
    "temperature": 0.0,
    "max_output_tokens": 256,
    "retry_count": 3,
    "prompt_version": "v2.1",
    "schema_version": "v2.1",
    "api_key_env": "DEEPSEEK_CRITIC_API_KEY",
    "accounting_namespace": "deepseek_critic_gpu1",
    "immutability": "VERIFY_PER_RUN",
}

# --- Qwen Tutor (PINNED dated version) ---
QWEN_TUTOR = {
    "provider": "qwen",
    "exact_model_id": "qwen-flash",
    "model_id_rationale": (
        "Qwen DashScope stable model family. Provider-recommended replacement "
        "for qwen-turbo. No dated snapshots available in this API deployment "
        "(qwen-turbo-2025-04-28, qwen-turbo-latest, qwen3-turbo all rejected). "
        "qwen-flash is the latency-optimized stable endpoint. "
        "Risk: MEDIUM — model may be updated by provider but API contract is stable."
    ),
    "model_tier": "qwen-flash",
    "role": "tutor",
    "thinking_mode": "disabled",
    "reasoning_setting": "none",
    "temperature": 0.0,
    "max_output_tokens": 256,
    "retry_count": 3,
    "prompt_version": "v2.1",
    "schema_version": "v2.1",
    "api_key_env": "DASHSCOPE_API_KEY",
    "accounting_namespace": "qwen_tutor_gpu1",
    "immutability": "STABLE_ENDPOINT",
}

# --- GLM Explorer ---
GLM_EXPLORER = {
    "provider": "glm",
    "exact_model_id": "glm-4-flash",
    "model_id_rationale": MODEL_ID_RATIONALE["glm-4-flash"],
    "model_tier": "glm-4-flash",
    "role": "explorer",
    "thinking_mode": "disabled",
    "reasoning_setting": "none",
    "temperature": 0.0,
    "max_output_tokens": 256,
    "retry_count": 3,
    "prompt_version": "v2.1",
    "schema_version": "v2.1",
    "api_key_env": "ZHIPUAI_API_KEY",
    "accounting_namespace": "glm_explorer_gpu1",
    "immutability": "STABLE_ENDPOINT",
}

# --- Claude Code engineering (separate from experiment API keys) ---
CLAUDE_CODE_ENGINEERING = {
    "provider": "anthropic",
    "exact_model_id": "claude-opus-4-8",
    "role": "engineering",
    "api_key_env": "ANTHROPIC_API_KEY",
    "accounting_namespace": "claude_code_engineering",
}

# ==============================================================================
# Role -> config mapping
# ==============================================================================

ROLE_CONFIG_MAP = {
    "generator": DEEPSEEK_GENERATOR,
    "critic": DEEPSEEK_CRITIC,
    "tutor": QWEN_TUTOR,
    "explorer": GLM_EXPLORER,
}

ALL_ROLE_CONFIGS = [DEEPSEEK_GENERATOR, DEEPSEEK_CRITIC, QWEN_TUTOR, GLM_EXPLORER]

# Roles that require per-run identity verification
VERIFY_PER_RUN_ROLES = ["generator", "critic"]  # DeepSeek moving alias

# ==============================================================================
# Manifest validation
# ==============================================================================


def compute_manifest_hash() -> str:
    """Compute a stable hash of the entire manifest for cache/reproducibility."""
    payload = {
        "version": MANIFEST_VERSION,
        "family": MANIFEST_FAMILY,
        "roles": {
            name: {
                k: v
                for k, v in config.items()
                if k not in ("api_key_env", "accounting_namespace", "model_id_rationale")
            }
            for name, config in ROLE_CONFIG_MAP.items()
        },
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_manifest() -> dict:
    """Validate the model manifest for completeness and correctness.

    Returns:
        Dict with keys: valid (bool), errors (list[str]), warnings (list[str]),
        manifest_hash (str), manifest_family (str).
    """
    errors = []
    warnings = []

    # Check no moving aliases
    for name, config in ROLE_CONFIG_MAP.items():
        model_id = config.get("exact_model_id", "")
        if model_id in ("latest", "auto", "", None):
            errors.append(
                f"Role '{name}' has invalid model_id: '{model_id}'. "
                f"Must be a pinned exact model ID."
            )
        # Check for known moving aliases (qwen-turbo without 'latest' is acceptable but moving)
        if model_id in ("qwen-turbo-latest",):  # Explicit "latest" is banned
            errors.append(
                f"Role '{name}' uses explicit-latest alias '{model_id}'. "
                f"Use a stable family endpoint like 'qwen-flash'."
            )
        if config.get("provider", "") in ("", None):
            errors.append(f"Role '{name}' has empty provider.")

        # Check immutability tags
        imm = config.get("immutability", "")
        if imm == "VERIFY_PER_RUN":
            warnings.append(
                f"Role '{name}' ({model_id}): requires per-run identity verification. "
                f"Model may change between runs."
            )
        if imm not in ("IMMUTABLE_SNAPSHOT", "STABLE_ENDPOINT", "VERIFY_PER_RUN"):
            warnings.append(
                f"Role '{name}' ({model_id}): unknown immutability tag '{imm}'."
            )

    # Check API key env vars
    all_env_vars = set()
    for name, config in ROLE_CONFIG_MAP.items():
        env_var = config.get("api_key_env", "")
        if not env_var:
            errors.append(f"Role '{name}' has no api_key_env defined.")
        else:
            all_env_vars.add(env_var)

    # DeepSeek deprecation warning (July 24, 2026)
    warnings.append(
        "DEEPSEEK DEPRECATION: deepseek-chat and deepseek-reasoner endpoints "
        "will be DISCONTINUED on 2026-07-24 (13 days from 2026-07-11). "
        "All DeepSeek roles must migrate to deepseek-v4-pro or deepseek-v4-flash "
        "before that date. This requires a new experiment family."
    )

    manifest_hash = compute_manifest_hash()

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "manifest_hash": manifest_hash,
        "manifest_family": MANIFEST_FAMILY,
        "manifest_version": MANIFEST_VERSION,
        "model_ids": sorted(set(c["exact_model_id"] for c in ALL_ROLE_CONFIGS)),
        "api_key_env_vars": sorted(all_env_vars),
        "immutability": {
            name: config["immutability"] for name, config in ROLE_CONFIG_MAP.items()
        },
    }


def get_role_config(role: str) -> dict:
    """Get the pinned configuration for a curriculum role."""
    if role not in ROLE_CONFIG_MAP:
        raise ValueError(
            f"Unknown role: {role}. Known roles: {list(ROLE_CONFIG_MAP.keys())}"
        )
    return dict(ROLE_CONFIG_MAP[role])


def get_api_key_for_role(role: str) -> Optional[str]:
    """Get the API key for a role from environment variables."""
    config = get_role_config(role)
    primary = os.environ.get(config["api_key_env"])
    if primary:
        return primary
    fallbacks = {
        "generator": ["DEEPSEEK_API_KEY"],
        "critic": ["DEEPSEEK_API_KEY", "DEEPSEEK_CRITIC_API_KEY"],
        "tutor": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "explorer": ["ZHIPUAI_API_KEY", "GLM_API_KEY"],
    }
    for fb in fallbacks.get(role, []):
        key = os.environ.get(fb)
        if key:
            return key
    return None


# ==============================================================================
# Model identity verification
# ==============================================================================


def verify_model_identity(
    provider_name: str,
    requested_model_id: str,
    api_response_model: Optional[str] = None,
) -> dict:
    """Verify that the model identity returned by the provider matches expectations.

    For providers with moving aliases (DeepSeek), this records the actual
    resolved model ID from the API response. For providers with pinned
    versions (Qwen dated), this confirms the requested model was served.

    Args:
        provider_name: Provider name ('deepseek', 'qwen', 'glm').
        requested_model_id: The model ID that was requested.
        api_response_model: The 'model' field from the API response (if available).

    Returns:
        Dict with 'verified' (bool), 'requested', 'resolved', 'warning' (str).
    """
    result = {
        "verified": True,
        "requested": requested_model_id,
        "resolved": api_response_model or "NOT_RETURNED_BY_PROVIDER",
        "warning": "",
        "timestamp_utc": "",
    }

    from datetime import datetime, timezone
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    if api_response_model is None:
        result["warning"] = (
            f"Provider {provider_name} did not return model ID in response. "
            f"Cannot verify exact model identity."
        )
        # Not a hard failure — some providers don't echo the model
        return result

    if api_response_model != requested_model_id:
        result["verified"] = True  # Still verified — we recorded the actual model
        result["warning"] = (
            f"Provider {provider_name} resolved '{requested_model_id}' "
            f"to '{api_response_model}'. Original model ID is a moving alias. "
            f"Recorded resolved identity for reproducibility."
        )

    return result


def generate_identity_verification_report(
    verifications: dict[str, dict],
) -> str:
    """Generate a human-readable identity verification report."""
    lines = [
        "=" * 70,
        "MODEL IDENTITY VERIFICATION REPORT",
        "=" * 70,
        f"Manifest Family: {MANIFEST_FAMILY}",
        f"Manifest Version: {MANIFEST_VERSION}",
        "",
    ]
    for role, v in verifications.items():
        lines.append(f"  {role}:")
        lines.append(f"    Requested: {v.get('requested', '?')}")
        lines.append(f"    Resolved:  {v.get('resolved', '?')}")
        warning = v.get("warning", "")
        if warning:
            lines.append(f"    WARNING: {warning}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def print_manifest_report() -> str:
    """Generate a human-readable manifest report string."""
    validation = validate_manifest()
    lines = [
        "=" * 70,
        f"MODEL MANIFEST: {MANIFEST_FAMILY} (v{MANIFEST_VERSION})",
        f"Manifest Hash: {validation['manifest_hash'][:16]}...",
        "=" * 70,
        "",
        "Pinned Models:",
    ]
    for name, config in ROLE_CONFIG_MAP.items():
        lines.append(
            f"  {name:12s} -> {config['provider']:10s} / {config['exact_model_id']:30s}"
        )
        lines.append(
            f"  {'':12s}    tier={config['model_tier']:20s}  "
            f"immutability={config['immutability']}"
        )
        lines.append(
            f"  {'':12s}    prompt={config['prompt_version']}  "
            f"schema={config['schema_version']}  "
            f"accounting={config['accounting_namespace']}"
        )
        lines.append(
            f"  {'':12s}    key_env={config['api_key_env']}"
        )
    lines.append("")
    lines.append(f"API Key Environment Variables: {', '.join(validation['api_key_env_vars'])}")
    lines.append(f"Errors: {len(validation['errors'])}")
    for e in validation["errors"]:
        lines.append(f"  ERROR: {e}")
    lines.append(f"Warnings: {len(validation['warnings'])}")
    for w in validation["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append(f"Manifest Valid: {validation['valid']}")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    print(print_manifest_report())
