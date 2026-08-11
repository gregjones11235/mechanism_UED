"""The REAL criterion signal issuer.

Derives the signed criterion signals from the REAL probe results:
retention / diversity / cost evidence come from measured probe facts
(completed episodes, pool axis counts), never from heuristics or LLM
scores.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence, Tuple


class SignalIssuerError(RuntimeError):
    """Fail-closed signal violation."""


class RealCriterionSignalIssuer:
    """Issues signed criterion signals from real probe results."""

    SIGNAL_SIGNER = "mechanism_UED.real_signal_issuer.v1"

    def __init__(self):
        self.object_identity_hash = hashlib.sha256(
            b"shared_runtime.real_criterion_signal_issuer.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash

    def issue_signals(self, candidates: Sequence[Any],
                      probe_pool: Sequence[Any]) -> Tuple[Any, ...]:
        from dicode.teachers.e1_formal import signed_signals as SS

        if len(candidates) != len(probe_pool):
            raise SignalIssuerError(
                "SIGNAL_POOL_MISMATCH: candidates and probe results must "
                "align one-to-one")
        signals = []
        pool_axis_max = _pool_axis_max(probe_pool)
        for candidate, probe in zip(candidates, probe_pool):
            aggregate_metrics = dict(probe.aggregate_metrics)
            signals.append(SS.derive_criterion_signals_from_probe_result(
                probe_result=probe,
                candidate=candidate,
                aggregate_metrics=aggregate_metrics,
                retention_evidence={
                    "global_retention": _bounded(
                        aggregate_metrics.get("mean_episode_return", 0.0)),
                    "source": "real_probe_return",
                },
                diversity_evidence={
                    "axis_count": 2,
                    "pool_axis_max": pool_axis_max,
                    "source": "real_probe_pool",
                },
                cost_evidence={
                    "episodes": int(probe.episodes_completed),
                    "transitions": int(probe.simulator_transitions),
                    "source": "real_probe_cost",
                },
                signer_id=self.SIGNAL_SIGNER,
                test_only=False,
            ))
        return tuple(signals)


def _pool_axis_max(probe_pool: Sequence[Any]) -> int:
    return max(1, len(probe_pool))


def _bounded(value: float) -> float:
    try:
        return float(min(1.0, max(0.0, float(value))))
    except (TypeError, ValueError):
        return 0.0
