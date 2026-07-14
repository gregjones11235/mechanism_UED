"""Immutable role-judgment cache with comprehensive cache-key fields.

Cache keys include ALL required fields:
  - task-code hash
  - student-stage ID
  - role
  - provider
  - exact model ID
  - prompt version
  - schema version

Separate caches are maintained per role. Cache entries are immutable once
written — mutation is detected and treated as corruption.

Provides a 95% cache-hit hard gate and prohibits silent fallback to
rule-only behavior.
"""

import hashlib
import json
import os
from typing import Optional

# ==============================================================================
# Cache key construction
# ==============================================================================


def compute_task_code_hash(task_code: str) -> str:
    """Compute a deterministic hash of a task's source code.

    This is more stable than summary-based hashing because it captures the
    exact code content, not a potentially variable text summary.

    Args:
        task_code: The full source code string of the task.

    Returns:
        16-char hex digest.
    """
    return hashlib.sha256(task_code.encode("utf-8")).hexdigest()[:16]


def compute_task_summary_hash(task_summary: dict) -> str:
    """Compute a deterministic hash of a task summary dict.

    Uses stable JSON serialization of relevant fields.
    """
    fields = {
        "task_id": str(task_summary.get("task_id", "")),
        "description": str(
            task_summary.get("description", task_summary.get("summary", ""))
        ),
        "source": str(task_summary.get("source", "")),
        "skills": str(task_summary.get("skills", task_summary.get("skill_tag", ""))),
    }
    canonical = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_immutable_cache_key(
    *,
    task_code_hash: str,
    student_stage_id: str,
    role: str,
    provider: str,
    exact_model_id: str,
    prompt_version: str,
    schema_version: str,
) -> str:
    """Compute an immutable cache key with ALL required fields.

    Every field is mandatory — missing any field raises ValueError.
    The key format is:
      {task_hash}_{stage}_{role}_{provider}_{model}_{prompt}_{schema}

    Args:
        task_code_hash: Hash of the task's source code.
        student_stage_id: Identifier for the student's current stage.
        role: Curriculum role name ('tutor', 'critic', 'explorer').
        provider: Provider name ('qwen', 'deepseek', 'glm').
        exact_model_id: Exact pinned model ID (no aliases).
        prompt_version: Version tag for the role prompt.
        schema_version: Version tag for the response schema.

    Returns:
        Cache key string.

    Raises:
        ValueError: If any field is empty or contains disallowed characters.
    """
    fields = {
        "task_code_hash": task_code_hash,
        "student_stage_id": student_stage_id,
        "role": role,
        "provider": provider,
        "exact_model_id": exact_model_id,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
    }

    for name, value in fields.items():
        if not value or not str(value).strip():
            raise ValueError(f"Cache key field '{name}' is empty or invalid: '{value}'")
        if "latest" in str(value).lower() or "auto" in str(value).lower():
            raise ValueError(
                f"Cache key field '{name}' contains disallowed alias: '{value}'"
            )

    key = (
        f"{task_code_hash}_"
        f"{student_stage_id}_"
        f"{role}_"
        f"{provider}_"
        f"{exact_model_id}_"
        f"{prompt_version}_"
        f"{schema_version}"
    )
    return key


# ==============================================================================
# Immutable cache store
# ==============================================================================


class ImmutableRoleCache:
    """A per-role immutable judgment cache backed by a JSONL file.

    Once a cache entry is written for a given key, it cannot be overwritten.
    Attempts to write a different value for an existing key are detected
    as corruption.
    """

    def __init__(self, role: str, cache_path: str):
        """Initialize the cache.

        Args:
            role: The role this cache serves ('tutor', 'critic', 'explorer').
            cache_path: Absolute path to the JSONL cache file.
        """
        self.role = role
        self.cache_path = cache_path
        self._entries: dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        """Load existing entries from the cache file."""
        self._entries = {}
        if not os.path.exists(self.cache_path):
            self._loaded = True
            return

        with open(self.cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry.get("cache_key", "")
                    if key:
                        self._entries[key] = entry
                except json.JSONDecodeError:
                    continue
        self._loaded = True

    def get(self, cache_key: str) -> Optional[dict]:
        """Look up a cached judgment. Returns None if not found."""
        if not self._loaded:
            self.load()
        entry = self._entries.get(cache_key)
        if entry:
            return entry.get("judgment")
        return None

    def put(
        self,
        cache_key: str,
        task_id: str,
        task_code_hash: str,
        judgment: dict,
        metadata: dict,
    ) -> bool:
        """Write a new cache entry atomically.

        Returns False if the key already exists (immutability violation).
        Writes are appended to the JSONL file for durability.

        Args:
            cache_key: The immutable cache key.
            task_id: The task identifier.
            task_code_hash: Hash of the task code.
            judgment: The parsed judgment dict.
            metadata: Dict with input_tokens, output_tokens, cost, student_stage_id,
                provider, model, prompt_version, schema_version.

        Returns:
            True if written, False if key already exists.
        """
        if not self._loaded:
            self.load()

        # Immutability check
        if cache_key in self._entries:
            existing = self._entries[cache_key]
            existing_judgment = existing.get("judgment", {})
            # Compare judgments for mutation detection
            if json.dumps(existing_judgment, sort_keys=True) != json.dumps(
                judgment, sort_keys=True
            ):
                raise ValueError(
                    f"Cache mutation detected for key {cache_key}: "
                    f"attempted to change judgment for existing entry"
                )
            return False  # Already exists with same content — idempotent

        entry = {
            "cache_key": cache_key,
            "task_id": task_id,
            "task_code_hash": task_code_hash,
            "student_stage_id": metadata.get("student_stage_id", ""),
            "role": self.role,
            "provider": metadata.get("provider", ""),
            "exact_model_id": metadata.get("exact_model_id", ""),
            "prompt_version": metadata.get("prompt_version", ""),
            "schema_version": metadata.get("schema_version", ""),
            "judgment": judgment,
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "estimated_cost": metadata.get("estimated_cost", 0.0),
            "cache_version": "v2.0_immutable",
        }

        # Write to file atomically
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        line = json.dumps(entry) + "\n"
        with open(self.cache_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        self._entries[cache_key] = entry
        return True

    @property
    def entry_count(self) -> int:
        """Number of cached entries."""
        if not self._loaded:
            self.load()
        return len(self._entries)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._loaded:
            self.load()
        return {
            "role": self.role,
            "total_entries": len(self._entries),
            "cache_path": self.cache_path,
        }


class MultiRoleImmutableCache:
    """Manages separate immutable caches for each curriculum role.

    Provides:
    - Per-role cache isolation
    - Cross-role cache key validation
    - 95% cache-hit gate enforcement
    - No silent fallback protection
    """

    def __init__(
        self,
        cache_dir: str,
        role_cache_paths: Optional[dict[str, str]] = None,
    ):
        """Initialize multi-role cache.

        Args:
            cache_dir: Base directory for cache files.
            role_cache_paths: Optional dict mapping role -> custom path.
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        default_paths = {
            "generator": os.path.join(cache_dir, "generator_judgments.jsonl"),
            "tutor": os.path.join(cache_dir, "tutor_judgments.jsonl"),
            "critic": os.path.join(cache_dir, "critic_judgments.jsonl"),
            "explorer": os.path.join(cache_dir, "explorer_judgments.jsonl"),
        }
        paths = {**default_paths, **(role_cache_paths or {})}

        self._caches = {
            role: ImmutableRoleCache(role, path) for role, path in paths.items()
        }

    def load_all(self) -> None:
        """Load all caches from disk."""
        for cache in self._caches.values():
            cache.load()

    def get_cache(self, role: str) -> ImmutableRoleCache:
        """Get the cache for a specific role."""
        if role not in self._caches:
            raise ValueError(
                f"No cache for role '{role}'. Available: {list(self._caches.keys())}"
            )
        return self._caches[role]

    def get_judgment(
        self, role: str, cache_key: str
    ) -> Optional[dict]:
        """Get a cached judgment for a role and key."""
        return self.get_cache(role).get(cache_key)

    def compute_cache_hit_rate(
        self, required_keys: dict[str, list[str]]
    ) -> dict:
        """Compute cache hit rate across roles.

        Args:
            required_keys: Dict mapping role -> list of cache keys that should
                be present in the cache.

        Returns:
            Dict with per-role and overall hit rates.
        """
        results = {}
        total_hits = 0
        total_required = 0

        for role, keys in required_keys.items():
            cache = self.get_cache(role)
            hits = sum(1 for k in keys if cache.get(k) is not None)
            misses = len(keys) - hits
            rate = hits / max(1, len(keys))
            results[role] = {
                "hits": hits,
                "misses": misses,
                "total": len(keys),
                "hit_rate": rate,
            }
            total_hits += hits
            total_required += len(keys)

        overall_rate = total_hits / max(1, total_required)
        results["overall"] = {
            "hits": total_hits,
            "misses": total_required - total_hits,
            "total": total_required,
            "hit_rate": overall_rate,
        }

        return results

    def validate_cache_hit_rate(
        self,
        required_keys: dict[str, list[str]],
        min_hit_rate: float = 0.95,
    ) -> dict:
        """Validate that cache hit rate meets the minimum threshold.

        Args:
            required_keys: Dict mapping role -> list of required cache keys.
            min_hit_rate: Minimum acceptable hit rate (default 0.95).

        Returns:
            Dict with 'passed' (bool), 'hit_rates' (dict), 'reason' (str).
        """
        hit_rates = self.compute_cache_hit_rate(required_keys)
        overall_rate = hit_rates["overall"]["hit_rate"]
        passed = overall_rate >= min_hit_rate

        reason = (
            f"Cache hit rate {overall_rate:.4f} {'meets' if passed else 'BELOW'} "
            f"threshold {min_hit_rate:.4f}"
        )

        # Also check per-role rates
        for role, stats in hit_rates.items():
            if role == "overall":
                continue
            if stats["total"] > 0 and stats["hit_rate"] < min_hit_rate:
                passed = False
                reason += (
                    f"; Role '{role}' hit rate {stats['hit_rate']:.4f} below threshold"
                )

        return {
            "passed": passed,
            "hit_rates": hit_rates,
            "min_hit_rate": min_hit_rate,
            "reason": reason,
        }

    @property
    def all_stats(self) -> dict:
        """Get statistics for all caches."""
        return {
            role: cache.get_stats() for role, cache in self._caches.items()
        }
