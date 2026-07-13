"""Pinned substitution configuration for DiCode DeepSeek-V4-Pro baseline.

Every experimental DiCode LLM call is routed through one exact pinned
DeepSeek V4 Pro model ID. This module provides a single source of truth
for the substitution configuration and validates it at import time.

This is a model-substitution baseline.
It is not an exact reproduction under the original model conditions.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
# Fixed pinned model configuration
# ==============================================================================

@dataclass(frozen=True)
class DsproConfig:
    """Immutable DeepSeek V4 Pro substitution configuration.

    All fields are frozen after construction. No aliases, fallbacks,
    or runtime model resolution are permitted.
    """

    # Experiment identity
    experiment_label: str = "DiCode DeepSeek-V4-Pro substitution baseline"
    experiment_family: str = "dspro"
    is_substitution_baseline: bool = True
    is_exact_reproduction: bool = False

    # Provider
    provider: str = "deepseek"
    provider_display_name: str = "DeepSeek"

    # Exact pinned model ID — no aliases, no latest, no auto.
    # deepseek-v4-pro is the official V4 Pro model ID.
    # deepseek-chat is a LEGACY ALIAS that silently maps to deepseek-v4-flash.
    # Legacy aliases are REJECTED by the gate checker.
    model_id: str = "deepseek-v4-pro"
    model_id_is_alias: bool = False

    # Legacy aliases that must be rejected at the gate
    legacy_aliases: tuple = ("deepseek-chat", "deepseek-reasoner")

    # API configuration
    api_key_env: str = "EXP_DEEPSEEK_API_KEY"
    base_url_env: str = "EXP_DEEPSEEK_BASE_URL"
    default_base_url: str = "https://api.deepseek.com/v1"

    # Generation parameters
    max_tokens: int = 32768
    dspro_role_judge_max_tokens: int = 2048  # Minimal stable budget for thinking model (empirically calibrated: 9/9 at 1536, +margin)
    temperature: float = 0.6
    top_p: float = 0.95
    think: bool = False
    reasoning_effort: Optional[str] = None  # DeepSeek V4 Pro does not use reasoning_effort

    # Retry configuration
    max_retries: int = 3
    initial_retry_delay: float = 2.0

    # Embedding note: DeepSeek does not support embeddings.
    # The embedding model remains a separate local_embed provider.
    embedding_provider: str = "local"
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"

    # Prompt versioning
    prompt_version: str = "migrated-v1"
    schema_version: str = "1"
    cache_version: str = "v1"

    # Output isolation
    output_root: str = "/root/experiments/dicode_runs/dspro"

    # Version pinning
    config_version: str = "1.0.0"

    # Logical role labels (preserved in logs even though all use same model)
    logical_roles: tuple = (
        "task_generator",
        "env_generator",
        "interestingness_critic",
    )

    # API accounting namespace
    accounting_namespace: str = "dspro-gpu0"

    def validate(self) -> list[str]:
        """Validate the configuration. Returns list of issues (empty = valid)."""
        issues = []

        # Check API key is configured
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            issues.append(
                f"API key environment variable {self.api_key_env} is not set. "
                f"API-dependent tests and training will be blocked."
            )

        # Check model ID is not a legacy alias
        if self.model_id in self.legacy_aliases:
            issues.append(
                f"Model ID '{self.model_id}' is a legacy alias. "
                f"Use the official model ID 'deepseek-v4-pro' instead. "
                f"Legacy aliases silently map to different models."
            )

        # Check model ID is not a generic alias
        if self.model_id in ("latest", "auto", ""):
            issues.append(f"Model ID '{self.model_id}' is a disallowed alias.")

        # Check output root exists or can be created
        if not os.path.exists(self.output_root):
            try:
                os.makedirs(self.output_root, exist_ok=True)
            except OSError as e:
                issues.append(f"Cannot create output root {self.output_root}: {e}")

        return issues

    def to_manifest_dict(self) -> dict:
        """Export configuration fields for the run manifest."""
        return {
            "experiment_label": self.experiment_label,
            "experiment_family": self.experiment_family,
            "is_substitution_baseline": self.is_substitution_baseline,
            "provider": self.provider,
            "requested_model": self.model_id,
            "returned_model": None,  # Populated after first API call
            "model_id": self.model_id,
            "model_id_is_alias": self.model_id_is_alias,
            "legacy_aliases_rejected": list(self.legacy_aliases),
            "max_tokens": self.max_tokens,
            "dspro_role_judge_max_tokens": self.dspro_role_judge_max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "think": self.think,
            "reasoning_effort": self.reasoning_effort,
            "max_retries": self.max_retries,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "cache_version": self.cache_version,
            "accounting_namespace": self.accounting_namespace,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
        }


# Singleton instance
DSPRO_CONFIG = DsproConfig()


def get_dspro_config() -> DsproConfig:
    """Return the frozen dspro substitution configuration."""
    return DSPRO_CONFIG
