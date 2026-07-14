"""JSONL-based caching for LLM role judgments.

Cache keys include:
  - task hash
  - provider
  - model
  - role
  - prompt_version

Avoids redundant API calls by storing judgments to disk.
"""

import hashlib
import json
import os
from typing import Optional


CACHE_VERSION = "v1"


def compute_task_hash(task_summary: dict) -> str:
    """Compute a deterministic hash for a task summary dict.

    Uses a stable JSON serialization of the relevant fields.
    """
    fields = {
        "task_id": task_summary.get("task_id", ""),
        "description": task_summary.get("description", task_summary.get("summary", "")),
        "source": task_summary.get("source", ""),
        "skills": str(task_summary.get("skills", task_summary.get("skill_tag", ""))),
    }
    canonical = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_cache_key(
    task_summary: dict,
    provider: str,
    model: str,
    role: str,
    prompt_version: str = CACHE_VERSION,
) -> str:
    """Compute a cache key for a specific LLM judgment call.

    The key format: {task_hash}_{provider}_{model}_{role}_{prompt_version}
    """
    task_hash = compute_task_hash(task_summary)
    return f"{task_hash}_{provider}_{model}_{role}_{prompt_version}"


def load_cache(cache_path: str) -> dict[str, dict]:
    """Load all cached judgments from a JSONL file into a lookup dict.

    Args:
        cache_path: Path to the JSONL cache file.

    Returns:
        Dict mapping cache_key -> judgment dict.
    """
    cache = {}
    if not os.path.exists(cache_path):
        return cache

    try:
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry.get("cache_key", "")
                    if key:
                        cache[key] = entry
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return cache


def get_cached_judgment(
    cache: dict[str, dict],
    task_summary: dict,
    provider: str,
    model: str,
    role: str,
) -> Optional[dict]:
    """Look up a cached judgment. Returns None if not found."""
    key = compute_cache_key(task_summary, provider, model, role)
    entry = cache.get(key)
    if entry and entry.get("judgment"):
        return entry["judgment"]
    return None


def get_cached_judgments_by_task_id(
    cache: dict[str, dict],
    task_id: str,
) -> dict[str, dict]:
    """Look up ALL cached judgments for a given task_id.

    Returns a dict mapping role -> judgment dict.
    This is more robust than hash-based lookup because task_id is stable
    across export and training time.
    """
    results = {}
    for key, entry in cache.items():
        if entry.get("task_id") == task_id:
            role = entry.get("role", "")
            judgment = entry.get("judgment")
            if role and judgment:
                # Keep the most recent (or any — cache entries should be unique per role)
                results[role] = judgment
    return results


def write_cache_entry(
    cache_path: str,
    task_summary: dict,
    provider: str,
    model: str,
    role: str,
    judgment: dict,
    api_response: dict,
) -> None:
    """Write a new cache entry to the JSONL file.

    Args:
        cache_path: Path to the JSONL cache file.
        task_summary: The task summary that was judged.
        provider: Provider name.
        model: Model name.
        role: Role name.
        judgment: Parsed judgment dict.
        api_response: Full API response (for cost/time metadata).
    """
    entry = {
        "cache_key": compute_cache_key(task_summary, provider, model, role),
        "task_id": task_summary.get("task_id", "unknown"),
        "task_hash": compute_task_hash(task_summary),
        "provider": provider,
        "model": model,
        "role": role,
        "prompt_version": CACHE_VERSION,
        "judgment": judgment,
        "input_tokens_est": api_response.get("input_tokens_est", 0),
        "output_tokens_est": api_response.get("output_tokens_est", 0),
        "estimated_cost": api_response.get("estimated_cost", 0.0),
    }

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    try:
        with open(cache_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  [LLM Cache] Warning: Could not write cache entry: {e}")


def get_cache_stats(cache_path: str) -> dict:
    """Get statistics about the cache file."""
    cache = load_cache(cache_path)
    if not cache:
        return {"total_entries": 0, "by_provider": {}, "by_role": {}, "total_cost": 0.0}

    by_provider = {}
    by_role = {}
    total_cost = 0.0

    for entry in cache.values():
        provider = entry.get("provider", "unknown")
        role = entry.get("role", "unknown")
        by_provider[provider] = by_provider.get(provider, 0) + 1
        by_role[role] = by_role.get(role, 0) + 1
        total_cost += entry.get("estimated_cost", 0.0)

    return {
        "total_entries": len(cache),
        "by_provider": by_provider,
        "by_role": by_role,
        "total_cost": round(total_cost, 8),
    }
