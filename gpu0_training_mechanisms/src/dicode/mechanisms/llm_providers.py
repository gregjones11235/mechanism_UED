"""LLM provider configurations and API calling utilities.

Uses only Python standard library (urllib, json) to avoid adding dependencies.
Supports Qwen (DashScope), DeepSeek, and GLM (ZhipuAI) as role providers.

API keys are read from environment variables only, never hard-coded.
"""

import json
import os
import time
import urllib.request
import urllib.error
from typing import Optional


# ==============================================================================
# Provider configurations
# ==============================================================================

PROVIDER_CONFIGS = {
    "qwen": {
        "name": "Qwen (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-turbo",
        "cheap_model": "qwen-turbo",
        "pricing_per_1k_input": 0.0003,   # USD per 1K input tokens
        "pricing_per_1k_output": 0.0006,  # USD per 1K output tokens
        "max_output_tokens": 256,
        "timeout_seconds": 60,
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "cheap_model": "deepseek-chat",
        "pricing_per_1k_input": 0.00014,  # USD per 1K input tokens
        "pricing_per_1k_output": 0.00028, # USD per 1K output tokens
        "max_output_tokens": 256,
        "timeout_seconds": 60,
    },
    "glm": {
        "name": "GLM (ZhipuAI)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "api_key_env": "ZHIPUAI_API_KEY",
        "default_model": "glm-4-flash",
        "cheap_model": "glm-4-flash",
        "pricing_per_1k_input": 0.0001,   # USD per 1K input tokens
        "pricing_per_1k_output": 0.0001,  # USD per 1K output tokens
        "max_output_tokens": 256,
        "timeout_seconds": 60,
    },
}

ROLE_PROVIDER_MAP = {
    "tutor": "qwen",
    "critic": "deepseek",
    "explorer": "glm",
}


def get_provider_config(provider_name: str) -> dict:
    """Get the configuration for a named provider."""
    if provider_name not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider_name}. Known: {list(PROVIDER_CONFIGS.keys())}")
    return dict(PROVIDER_CONFIGS[provider_name])


def get_api_key(provider_name: str) -> Optional[str]:
    """Get the API key for a provider from environment variables."""
    config = get_provider_config(provider_name)
    key = os.environ.get(config["api_key_env"])
    if not key:
        # Try alternate env var names
        alt_keys = {
            "qwen": ["QWEN_API_KEY", "DASHSCOPE_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "glm": ["GLM_API_KEY", "ZHIPUAI_API_KEY"],
        }
        for alt in alt_keys.get(provider_name, []):
            key = os.environ.get(alt)
            if key:
                break
    return key


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: ~4 chars per token for English/Chinese mix."""
    return max(1, len(text) // 3)


# ==============================================================================
# API Call
# ==============================================================================


def call_llm_api(
    provider_name: str,
    messages: list[dict],
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = 60,
) -> dict:
    """Call an LLM API and return the parsed response.

    Args:
        provider_name: One of 'qwen', 'deepseek', 'glm'.
        messages: List of dicts with 'role' and 'content'.
        model: Model name override (uses provider default if None).
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: success, content, input_tokens_est, output_tokens_est,
        estimated_cost, provider, model, error (if failed).
    """
    config = get_provider_config(provider_name)
    api_key = get_api_key(provider_name)

    if not api_key:
        return {
            "success": False,
            "error": f"No API key found for {provider_name}",
            "provider": provider_name,
            "model": model or config["default_model"],
            "input_tokens_est": 0,
            "output_tokens_est": 0,
            "estimated_cost": 0.0,
        }

    model = model or config["default_model"]
    max_tokens = min(max_tokens, config["max_output_tokens"])

    # Build request
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    url = config["base_url"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Estimate input tokens
    input_text = " ".join(m.get("content", "") for m in messages)
    input_tokens_est = estimate_tokens(input_text)

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            result = json.loads(raw)
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        return {
            "success": False,
            "error": f"HTTP {e.code}: {error_body}",
            "provider": provider_name,
            "model": model,
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": 0,
            "estimated_cost": _estimate_cost(provider_name, input_tokens_est, 0),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:500],
            "provider": provider_name,
            "model": model,
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": 0,
            "estimated_cost": _estimate_cost(provider_name, input_tokens_est, 0),
        }

    # Extract content from OpenAI-compatible response
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {
            "success": False,
            "error": f"Unexpected response format: {str(result)[:300]}",
            "provider": provider_name,
            "model": model,
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": 0,
            "estimated_cost": _estimate_cost(provider_name, input_tokens_est, 0),
        }

    output_tokens_est = estimate_tokens(content)

    return {
        "success": True,
        "content": content,
        "provider": provider_name,
        "model": model,
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "estimated_cost": _estimate_cost(provider_name, input_tokens_est, output_tokens_est),
    }


def _estimate_cost(provider_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given token usage."""
    config = PROVIDER_CONFIGS.get(provider_name, {})
    input_cost = (input_tokens / 1000.0) * config.get("pricing_per_1k_input", 0.0)
    output_cost = (output_tokens / 1000.0) * config.get("pricing_per_1k_output", 0.0)
    return round(input_cost + output_cost, 8)
