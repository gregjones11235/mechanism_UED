"""Deterministic execution bindings for the frontier distribution fields.

CC4 follow-up (P0-11): before this module existed, the compiled
``FrontierDistribution`` objects reached the mixed-start rollout with their
``taskparam_ranges`` / ``seed_distribution`` / ``stochasticity_range``
treated as inert metadata: STEP08 flattened every distribution into one big
state list and rolled out with unseeded, fully deterministic policy steps.
The distribution fields never influenced what actually executed.

``resolve_distribution_binding`` closes that gap.  It is a pure, fail-closed
function that consumes ONE validated ``FrontierDistribution`` plus an episode
index and a seed base, and MINTS an immutable
``DistributionRuntimeBinding`` whose every field is mechanically derived
from the distribution itself:

* ``episode_seed`` — canonical sha256 over the distribution's
  ``seed_distribution``, ``distribution_id``, the episode index and the seed
  base (mod 2**31): the per-episode environment continuation RNG seed.
* ``epsilon`` / ``temperature`` — resolved from ``stochasticity_range``
  (scalar or ``[lo, hi]`` -> lower bound, deterministic); the action-level
  stochasticity actually applied during the episode.
* ``taskparams`` — the distribution's non-empty TaskParams mapping, carried
  to the injected taskparam application surface.

Mappings and foreign types are refused; empty, unknown, malformed or
out-of-range content raises ``InvalidEvidenceError``.  ``binding_hash`` is
mint-only (computed in ``__post_init__``, never a constructor argument), and
``verify_distribution_binding`` recomputes it so any tampered or
self-reported binding is rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import InvalidEvidenceError
from .frontier_distributions import FrontierDistribution

DISTRIBUTION_RUNTIME_VERSION = "distribution-runtime/v1"

_EPSILON_KEY = "epsilon"
_TEMPERATURE_KEY = "temperature"
_STOCHASTICITY_KEYS = (_EPSILON_KEY, _TEMPERATURE_KEY)
_SEED_MODULUS = 2 ** 31


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DistributionRuntimeBinding:
    """One immutable, hash-bound per-episode execution binding (mint-only).

    ``binding_hash`` is NOT a constructor argument: it is computed in
    ``__post_init__`` from the derived fields only, so no caller can supply
    (self-report) a hash that was not mechanically produced.
    """

    distribution_id: str
    episode_index: int
    episode_seed: int
    epsilon: float
    temperature: float
    taskparams: Mapping[str, Any]
    binding_hash: str = field(init=False)
    runtime_version: str = DISTRIBUTION_RUNTIME_VERSION

    def __post_init__(self) -> None:
        if not str(self.distribution_id).strip():
            raise InvalidEvidenceError(
                "DistributionRuntimeBinding.distribution_id is empty")
        if isinstance(self.episode_index, bool) \
                or not isinstance(self.episode_index, int) or self.episode_index < 0:
            raise InvalidEvidenceError(
                f"DistributionRuntimeBinding.episode_index must be a non-negative "
                f"int, got {self.episode_index!r}")
        if isinstance(self.episode_seed, bool) \
                or not isinstance(self.episode_seed, int) \
                or self.episode_seed < 0 or self.episode_seed >= _SEED_MODULUS:
            raise InvalidEvidenceError(
                f"DistributionRuntimeBinding.episode_seed must be an int in "
                f"[0, {_SEED_MODULUS}), got {self.episode_seed!r}")
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)) \
                or not math.isfinite(float(self.epsilon)) \
                or float(self.epsilon) < 0.0 or float(self.epsilon) > 1.0:
            raise InvalidEvidenceError(
                f"DistributionRuntimeBinding.epsilon must be in [0, 1], "
                f"got {self.epsilon!r}")
        if isinstance(self.temperature, bool) \
                or not isinstance(self.temperature, (int, float)) \
                or not math.isfinite(float(self.temperature)) \
                or float(self.temperature) <= 0.0:
            raise InvalidEvidenceError(
                f"DistributionRuntimeBinding.temperature must be finite and > 0, "
                f"got {self.temperature!r}")
        if not isinstance(self.taskparams, Mapping) or len(self.taskparams) == 0:
            raise InvalidEvidenceError(
                "DistributionRuntimeBinding.taskparams must be a non-empty mapping "
                "(a binding without taskparams would silently skip the taskparam "
                "distribution; fail closed)")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "binding_hash"
        }
        payload["taskparams"] = dict(self.taskparams)
        object.__setattr__(self, "binding_hash", _canonical_sha256(payload))


def _resolve_stochasticity_value(distribution_id: str, key: str, value: Any) -> float:
    """Scalar or ``[lo, hi]`` -> lower bound (deterministic execution value)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            for bound in (low, high):
                if isinstance(bound, bool) or not isinstance(bound, (int, float)) \
                        or not math.isfinite(float(bound)):
                    raise InvalidEvidenceError(
                        f"distribution {distribution_id}: stochasticity_range[{key!r}] "
                        f"bounds must be finite numbers, got {value!r}")
            if float(low) > float(high):
                raise InvalidEvidenceError(
                    f"distribution {distribution_id}: stochasticity_range[{key!r}] has "
                    f"lo > hi: {value!r}")
            return float(low)
        raise InvalidEvidenceError(
            f"distribution {distribution_id}: stochasticity_range[{key!r}] must be a "
            f"finite scalar or a [lo, hi] pair, got {value!r}")
    if not math.isfinite(float(value)):
        raise InvalidEvidenceError(
            f"distribution {distribution_id}: stochasticity_range[{key!r}] must be "
            f"finite, got {value!r}")
    return float(value)


def resolve_distribution_binding(distribution: Any, *, episode_index: int,
                                 seed_base: int) -> DistributionRuntimeBinding:
    """Mint the immutable per-episode execution binding for one distribution.

    Fail-closed contract:

    * ``distribution`` must be a typed ``FrontierDistribution`` (arbitrary
      mappings and foreign types are refused).
    * ``episode_index`` / ``seed_base`` must be non-negative ints (bools are
      refused).
    * ``seed_distribution`` must be a non-empty mapping; the episode seed is
      the canonical sha256 over the seed distribution, the distribution id,
      the episode index and the seed base (mod 2**31) — never caller-supplied.
    * ``stochasticity_range`` must be a non-empty mapping whose keys are a
      subset of {"epsilon", "temperature"} and which carries at least one of
      them; each value is a finite scalar or a ``[lo, hi]`` pair resolved to
      its lower bound.  Defaults when a key is absent: epsilon=0.0,
      temperature=1.0.
    * ``taskparam_ranges`` must be a non-empty mapping (passed through).
    """
    if isinstance(distribution, Mapping):
        raise InvalidEvidenceError(
            "resolve_distribution_binding requires a typed FrontierDistribution, "
            "not a mapping (hand-built distribution surfaces are never accepted)")
    if not isinstance(distribution, FrontierDistribution):
        raise InvalidEvidenceError(
            f"resolve_distribution_binding requires a typed FrontierDistribution, "
            f"got {type(distribution).__name__}")
    for name, value in (("episode_index", episode_index), ("seed_base", seed_base)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidEvidenceError(
                f"resolve_distribution_binding: {name} must be a non-negative int, "
                f"got {value!r}")

    distribution_id = str(distribution.distribution_id)

    seed_distribution = distribution.seed_distribution
    if not isinstance(seed_distribution, Mapping) or len(seed_distribution) == 0:
        raise InvalidEvidenceError(
            f"distribution {distribution_id}: seed_distribution is empty — the seed "
            "field must execute, never be silently defaulted (fail closed)")
    episode_seed = int(_canonical_sha256({
        "runtime_version": DISTRIBUTION_RUNTIME_VERSION,
        "seed_distribution": dict(seed_distribution),
        "distribution_id": distribution_id,
        "episode_index": int(episode_index),
        "seed_base": int(seed_base),
    }), 16) % _SEED_MODULUS

    stochasticity = distribution.stochasticity_range
    if not isinstance(stochasticity, Mapping) or len(stochasticity) == 0:
        raise InvalidEvidenceError(
            f"distribution {distribution_id}: stochasticity_range is empty — the "
            "stochasticity field must execute, never be silently defaulted "
            "(fail closed)")
    unknown = sorted(str(k) for k in stochasticity if str(k) not in _STOCHASTICITY_KEYS)
    if unknown:
        raise InvalidEvidenceError(
            f"distribution {distribution_id}: stochasticity_range carries unknown "
            f"keys {unknown}; only {list(_STOCHASTICITY_KEYS)} execute (fail closed "
            "rather than ignore fields)")
    if _EPSILON_KEY not in stochasticity and _TEMPERATURE_KEY not in stochasticity:
        raise InvalidEvidenceError(
            f"distribution {distribution_id}: stochasticity_range must carry at "
            f"least one of {list(_STOCHASTICITY_KEYS)}")
    epsilon = 0.0
    if _EPSILON_KEY in stochasticity:
        epsilon = _resolve_stochasticity_value(
            distribution_id, _EPSILON_KEY, stochasticity[_EPSILON_KEY])
        if epsilon < 0.0 or epsilon > 1.0:
            raise InvalidEvidenceError(
                f"distribution {distribution_id}: epsilon must be in [0, 1], "
                f"got {epsilon}")
    temperature = 1.0
    if _TEMPERATURE_KEY in stochasticity:
        temperature = _resolve_stochasticity_value(
            distribution_id, _TEMPERATURE_KEY, stochasticity[_TEMPERATURE_KEY])
        if temperature <= 0.0:
            raise InvalidEvidenceError(
                f"distribution {distribution_id}: temperature must be > 0, "
                f"got {temperature}")

    taskparams = distribution.taskparam_ranges
    if not isinstance(taskparams, Mapping) or len(taskparams) == 0:
        raise InvalidEvidenceError(
            f"distribution {distribution_id}: taskparam_ranges is empty — the "
            "taskparam field must execute, never be silently defaulted (fail closed)")

    return DistributionRuntimeBinding(
        distribution_id=distribution_id,
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        epsilon=float(epsilon),
        temperature=float(temperature),
        taskparams={str(k): v for k, v in taskparams.items()},
    )


def verify_distribution_binding(binding: Any) -> None:
    """Recompute the binding hash; reject mappings, foreign types and tamper.

    A binding is only accepted when its stored ``binding_hash`` exactly
    equals the hash recomputed from its own fields — self-reported or
    mutated bindings fail closed.
    """
    if isinstance(binding, Mapping):
        raise InvalidEvidenceError(
            "verify_distribution_binding requires a minted "
            "DistributionRuntimeBinding, not a mapping")
    if not isinstance(binding, DistributionRuntimeBinding):
        raise InvalidEvidenceError(
            f"verify_distribution_binding requires a minted "
            f"DistributionRuntimeBinding, got {type(binding).__name__}")
    payload = {
        f.name: getattr(binding, f.name)
        for f in fields(binding)
        if f.name != "binding_hash"
    }
    payload["taskparams"] = dict(binding.taskparams)
    expected = _canonical_sha256(payload)
    if expected != binding.binding_hash:
        raise InvalidEvidenceError(
            "binding_hash mismatch: the DistributionRuntimeBinding was tampered "
            "with or self-reported (fail closed)")
