"""TrainingTrajectoryEvidenceAdapter (task sections 1 / 3).

Adapts a raw generative-training rollout of the CURRENT Student into an
admissible TrajectoryEvidenceBundle:

    raw rollout dict  ->  (leakage guard)  ->  symbolic adapter  ->  bundle

Hard rules enforced on intake:
  * FormalEvaluationLeakageGuard runs FIRST; any formal/bank provenance fails
    closed before a single step is read;
  * raw frozen-state / bank payload keys anywhere in the input fail closed
    (the adapter never reads a FRONT/BACK frozen state payload);
  * raw actions resolve through the injected symbolic vocabulary — no
    hardcoded Craftax action integers;
  * atomic env events (damage_taken / chased / died / no_effect) are derived
    DETERMINISTICALLY from symbolic state summaries when the trace does not
    already carry them (health-band decrease -> damage_taken, etc.), so
    downstream detectors never re-derive semantics from raw state.

The adapter is the ONLY sanctioned path from training to evidence; Reference
trajectories are NOT demonstrations — a Reference evidence adapter would use
the same boundary and is equally forbidden from emitting action sequences as
supervision (TrajectorySupervisionGuard applies to its outputs too).
"""
from __future__ import annotations

from typing import Any, Dict, List

from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard)
from d052.bagr_ued.trajectory_evidence import (
    EpisodeEvidence,
    EvidenceSource,
    MockSymbolicAdapter,
    StepRecord,
    TrajectoryEvidenceBundle,
    TrajectoryEvidenceError,
)

#: keys that may never appear in a raw training trace handed to this adapter
_FORBIDDEN_RAW_KEYS = frozenset({
    "front_bank_states", "back_bank_states", "frozen_state", "private_state",
    "evaluation_certificate", "expert_action_sequence", "expert_trajectory",
    "reference_action_sequence",
})

_HEALTH_BAND_ORDER = {"none": 0, "critical": 1, "low": 2, "mid": 3, "high": 4}


class TrainingTrajectoryEvidenceAdapter:
    """Raw generative-training rollout -> admissible evidence bundle."""

    def __init__(self, symbolic_adapter: MockSymbolicAdapter) -> None:
        self.symbolic_adapter = symbolic_adapter
        self.leakage_guard = FormalEvaluationLeakageGuard()

    def adapt(self, raw_rollout: Dict[str, Any], *,
              bundle_id: str,
              source: EvidenceSource = EvidenceSource.GENERATIVE_TRAINING_ENV
              ) -> TrajectoryEvidenceBundle:
        # 0. provenance + payload gates (fail closed before reading steps)
        self.leakage_guard.assert_admissible_source(source)
        self.leakage_guard.assert_clean(raw_rollout, label="raw_rollout")
        self._assert_no_forbidden_raw_keys(raw_rollout)

        episodes: List[EpisodeEvidence] = []
        for raw_ep in raw_rollout.get("episodes", []):
            episodes.append(self._adapt_episode(raw_ep, source))
        return TrajectoryEvidenceBundle(
            bundle_id=bundle_id,
            source=source,
            symbolic_adapter_version=self.symbolic_adapter.version,
            episodes=episodes,
            leakage_guard_status="PASS",
        )

    # -- internals ----------------------------------------------------------
    def _assert_no_forbidden_raw_keys(self, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in _FORBIDDEN_RAW_KEYS:
                    raise TrajectoryEvidenceError(
                        TrajectoryEvidenceError.FORBIDDEN_PAYLOAD_KEY,
                        f"raw rollout carries forbidden key {k!r} (frozen-"
                        f"state/bank/expert payload may not enter evidence)")
                self._assert_no_forbidden_raw_keys(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._assert_no_forbidden_raw_keys(v)

    def _adapt_episode(self, raw_ep: Dict[str, Any],
                       source: EvidenceSource) -> EpisodeEvidence:
        steps: List[StepRecord] = []
        prev_health_band = None
        for raw_step in raw_ep.get("steps", []):
            action = raw_step.get("action", raw_step.get("action_int"))
            name, classes = self.symbolic_adapter.resolve_action(action)
            state = self.symbolic_adapter.summarize_state(
                raw_step.get("state_summary", {}))
            events = list(raw_step.get("env_events", []))
            events.extend(self._derive_events(state, prev_health_band))
            prev_health_band = state.get("health_band")
            steps.append(StepRecord(
                step_index=int(raw_step["step_index"]),
                symbolic_action=name,
                action_semantic_classes=classes,
                state_summary=state,
                env_events=sorted(set(events)),
            ))
        return EpisodeEvidence(
            episode_id=str(raw_ep["episode_id"]),
            source=source,
            steps=steps,
            outcome=raw_ep.get("outcome"),
            meta={k: v for k, v in raw_ep.get("meta", {}).items()},
        )

    @staticmethod
    def _derive_events(state: Dict[str, Any], prev_health_band: Any) -> List[str]:
        """Deterministic event labels from symbolic summaries (no raw state)."""
        events: List[str] = []
        band = state.get("health_band")
        if (prev_health_band in _HEALTH_BAND_ORDER and band in _HEALTH_BAND_ORDER
                and _HEALTH_BAND_ORDER[band] < _HEALTH_BAND_ORDER[prev_health_band]):
            events.append("damage_taken")
        if state.get("is_dead") is True:
            events.append("died")
        if state.get("hostile_distance_band") in ("adjacent", "near") and \
                state.get("hostile_engaged") is False:
            events.append("threat_present")
        return events
