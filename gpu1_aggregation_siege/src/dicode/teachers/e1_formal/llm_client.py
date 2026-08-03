"""Replay-only LLM client for the E1 teacher (offline round).

Duck-types the LLM ``query`` surface but answers ONLY from a pinned
replay store. Discipline mirrors ``siege/production_dispatcher.py``:

* replay-only mode: a cache MISS is a HARD FAIL (RuntimeError) — there
  is NO fallback, NO live call, NO implicit default;
* record mode is DISABLED this round (no paid API), which is what
  keeps ``REAL_ENVCODER_USED == false`` provable by inspection;
* keys reuse the shared 7-field ``compute_immutable_cache_key``
  (provider="replay", exact_model_id="e1-replay-mock-v1",
  student_stage_id = evidence-window hash, task_code_hash = prompt
  envelope hash); the shared helper already rejects empty values and
  latest/auto aliases.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ...mechanisms.immutable_cache import compute_immutable_cache_key
from .manifest import E1_REPLAY_MODEL_ID, E1_REPLAY_PROVIDER

#: record mode is disabled for the whole offline round.
E1_RECORD_MODE_DISABLED_THIS_ROUND = True

HARD_FAIL_PREFIX = "HARD FAIL: replay cache miss"


def make_replay_key(
    *,
    role: str,
    evidence_hash: str,
    prompt_envelope_hash: str,
    prompt_version: str,
    schema_version: str,
) -> str:
    """Compute the 7-field immutable replay key (no defaults, no aliases)."""
    return compute_immutable_cache_key(
        task_code_hash=prompt_envelope_hash,
        student_stage_id=evidence_hash,
        role=role,
        provider=E1_REPLAY_PROVIDER,
        exact_model_id=E1_REPLAY_MODEL_ID,
        prompt_version=prompt_version,
        schema_version=schema_version,
    )


class ReplayLLMClient:
    """Deterministic replay client; HARD FAIL on any miss."""

    def __init__(self, store: Mapping[str, str], context: str = "replay-client"):
        if not isinstance(store, Mapping):
            raise TypeError(f"{context}: replay store must be a mapping")
        for key, value in store.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    f"{context}: replay store must map str -> str"
                )
        # defensive copy: caller cannot mutate the pinned store afterwards
        self._store: Dict[str, str] = dict(store)
        self._context = context

    def query(
        self,
        system_prompt: str,
        user_prompts: Sequence[str],
        *,
        cache_key: str,
        role: str,
    ) -> List[Dict[str, Any]]:
        """Answer from the replay store only (HARD FAIL on miss).

        ``system_prompt``/``user_prompts`` are accepted for interface
        fidelity with the production LLM surface; the answer depends
        ONLY on ``cache_key`` (deterministic double-run equality).
        ``role`` labels the HARD FAIL message on a miss.
        """
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError(f"{self._context}: cache_key is required")
        if cache_key not in self._store:
            raise RuntimeError(
                f"{HARD_FAIL_PREFIX} role={role} key={cache_key}"
            )
        return [{"content": self._store[cache_key]}]

    def record(self, *args: Any, **kwargs: Any) -> None:
        """Disabled this round; calling it is a hard failure."""
        raise RuntimeError(
            "HARD FAIL: E1 record mode is disabled this round "
            "(E1_RECORD_MODE_DISABLED_THIS_ROUND); no paid API; "
            "REAL_ENVCODER_USED must remain false"
        )
