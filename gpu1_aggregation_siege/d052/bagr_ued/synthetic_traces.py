"""Synthetic test traces + mock scoring evidence (task sections 14 / 15).

The required unsafe_rest synthetic trace: the current Student repeatedly
performs REST-class actions near a hostile without confirmed safety, then
takes damage, is chased, and dies — a fully symbolic generative-training-env
trace (SYNTHETIC_TEST_TRACE source) that exercises the whole board.

Also provides deterministic MOCK scoring evidence (regret / reference
behavior-failure / success rates / window history) so the scoring + Soft
Copeland + budget chain runs end-to-end this round. Every number here is
labeled mock; none of it is a performance claim.
"""
from __future__ import annotations

from typing import Dict, List

from d052.bagr_ued.regret_scorer import RegretEvidence, ScenarioScope
from d052.bagr_ued.trajectory_evidence import ActionVocabulary

CF_ENV_ID = "CF_UNSAFE_REST_V1"

#: injected DATA (not hardcoded Craftax integers in package code)
TEST_VOCABULARY = ActionVocabulary(
    version="synthetic_test_vocab.v1",
    semantics={
        "REST": ["rest_class"],
        "SLEEP": ["rest_class"],
        "ATTACK": ["combat_class"],
        "MOVE_TOWARD_HOSTILE": ["move_class"],
        "MOVE_AWAY": ["move_class"],
        "GATHER_WOOD": ["resource_class", "equipment_class"],
        "EAT": ["resource_class"],
        "CRAFT_TOOL": ["equipment_class"],
        "DO_NOTHING": [],
        "SCOUT": ["move_class"],
    },
    raw_int_map={0: "DO_NOTHING", 1: "MOVE_TOWARD_HOSTILE", 2: "MOVE_AWAY",
                 3: "ATTACK", 4: "GATHER_WOOD", 5: "REST", 6: "SLEEP",
                 7: "EAT", 8: "CRAFT_TOOL", 9: "SCOUT"},
)


def _step(i: int, action: str, *, hostile: str = "none", safe: bool = False,
          health: str = "high", resource: str = "mid", progress: int = 0,
          is_dead: bool = False, engaged: bool = False,
          events: List[str] | None = None) -> dict:
    return dict(
        step_index=i,
        action=action,
        state_summary=dict(
            hostile_distance_band=hostile,
            env_confirmed_safe=safe,
            health_band=health,
            resource_band=resource,
            progress_ordinal=progress,
            is_dead=is_dead,
            hostile_engaged=engaged),
        env_events=list(events or []))


def build_unsafe_rest_raw_rollout() -> dict:
    """The required synthetic trace (2 episodes, generative-training shaped)."""
    ep1_steps = [
        _step(0, "GATHER_WOOD", hostile="none", resource="mid", progress=1),
        _step(1, "GATHER_WOOD", hostile="far", resource="mid", progress=1),
        _step(2, "CRAFT_TOOL", hostile="far", resource="mid", progress=2,
              events=["achievement_craft_tool"]),
        _step(3, "SCOUT", hostile="far", progress=2),
        _step(4, "REST", hostile="near", safe=False, progress=2),
        _step(5, "REST", hostile="near", safe=False, health="low", progress=2,
              events=["damage_taken"]),
        _step(6, "SLEEP", hostile="adjacent", safe=False, health="low",
              progress=2, events=["damage_taken"]),
        _step(7, "MOVE_AWAY", hostile="adjacent", health="low", progress=2,
              events=["chased"]),
        _step(8, "MOVE_AWAY", hostile="near", health="low", progress=2,
              events=["chased"]),
        _step(9, "REST", hostile="near", safe=False, health="critical",
              progress=2, events=["damage_taken"]),
        _step(10, "DO_NOTHING", hostile="adjacent", health="critical",
              progress=2),
        _step(11, "DO_NOTHING", hostile="adjacent", health="critical",
              progress=2, events=["damage_taken"]),
        _step(12, "DO_NOTHING", hostile="adjacent", health="critical",
              progress=2, is_dead=True, events=["died"]),
    ]
    ep2_steps = [
        _step(0, "SCOUT", hostile="none", progress=1),
        _step(1, "DO_NOTHING", hostile="none", progress=1, events=["no_effect"]),
        _step(2, "DO_NOTHING", hostile="none", progress=1, events=["no_effect"]),
        _step(3, "DO_NOTHING", hostile="none", progress=1, events=["no_effect"]),
        _step(4, "DO_NOTHING", hostile="none", progress=1, events=["no_effect"]),
        _step(5, "GATHER_WOOD", hostile="none", progress=1),
        _step(6, "SCOUT", hostile="none", progress=1,
              resource="critical"),
        _step(7, "SCOUT", hostile="none", progress=1, resource="critical"),
        _step(8, "SCOUT", hostile="none", progress=1, resource="critical"),
        _step(9, "SCOUT", hostile="none", progress=1, resource="critical"),
        _step(10, "SCOUT", hostile="none", progress=1, resource="critical"),
        _step(11, "SCOUT", hostile="none", progress=1, resource="critical"),
        _step(12, "DO_NOTHING", hostile="none", progress=1),
    ]
    return dict(
        rollout_id="synthetic_unsafe_rest_rollout_v1",
        source="GENERATIVE_TRAINING_ENV",
        episodes=[
            dict(episode_id="ep_unsafe_rest_01", steps=ep1_steps,
                 outcome="death",
                 meta=dict(env_id=CF_ENV_ID, policy="current_student_mock")),
            dict(episode_id="ep_no_effect_02", steps=ep2_steps,
                 outcome="timeout",
                 meta=dict(env_id=CF_ENV_ID, policy="current_student_mock")),
        ])


# ---------------------------------------------------------------------------
# deterministic MOCK scoring evidence (dry run; not performance)
# ---------------------------------------------------------------------------

def build_mock_regret_evidences(environment_ids: List[str]) -> List[RegretEvidence]:
    """Mock front+global regret evidence per environment (deterministic)."""
    out: List[RegretEvidence] = []
    for i, eid in enumerate(sorted(environment_ids)):
        student = round(0.25 + 0.03 * (i % 5), 4)     # mock student success
        reference = round(0.80 + 0.01 * (i % 3), 4)   # mock reference success
        out.append(RegretEvidence(
            environment_id=eid, scope=ScenarioScope.FRONT,
            student_success_rate=student, reference_success_rate=reference,
            severity_weight=1.0))
        out.append(RegretEvidence(
            environment_id=eid, scope=ScenarioScope.GLOBAL,
            student_success_rate=round(student + 0.05, 4),
            reference_success_rate=reference,
            severity_weight=0.9))
    return out


def build_mock_student_success_rates(environment_ids: List[str]) -> Dict[str, float]:
    return {eid: round(0.25 + 0.03 * (i % 5), 4)
            for i, eid in enumerate(sorted(environment_ids))}


def build_mock_reference_failure_scores(environment_ids: List[str]) -> Dict[str, float]:
    """Mock Reference behavior-failure baseline (low; Reference behaves well)."""
    return {eid: 0.10 for eid in environment_ids}


def build_mock_failure_history(environment_ids: List[str]) -> Dict[str, List[float]]:
    """Single window this round -> learning_progress honestly 0 (no history)."""
    return {eid: [0.55] for eid in environment_ids}


def build_mock_global_retention(environment_ids: List[str]) -> Dict[str, float]:
    return {eid: 0.8 for eid in environment_ids}
