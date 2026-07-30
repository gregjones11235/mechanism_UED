"""Behavior taxonomy for the BA-BAGR-UED review board (task sections 4-8, 13).

Single source of truth for:
  * behavior patterns the deterministic detectors can raise (GLOBAL, not a
    floor2->floor3-only taxonomy — includes resource waste, combat freeze,
    exploration loops, tool misuse, long-term planning failure);
  * the closed cause-category vocabulary the CausalFailureAnalyst may use;
  * the legal mutation-axis vocabulary (environment induction ONLY — no
    action/reward/policy knob appears anywhere in this document);
  * the global environment families the Explorer may propose.

``taxonomy_document()`` is the artifact written to behavior_taxonomy.json;
its hash is bound into every reconciled item for replay.
"""
from __future__ import annotations

from typing import Dict, List

from d052.bagr_ued import constants as C
from d052.bagr_ued.hashing import canonical_sha256

TAXONOMY_VERSION = "bagr_ued.behavior_taxonomy.v1"

#: behavior_pattern -> (detector_id, description, global_scope_note)
BEHAVIOR_PATTERNS: Dict[str, Dict[str, str]] = {
    "unsafe_rest_near_hostile": dict(
        detector_id="unsafe_rest_near_hostile",
        description="Student performs REST/SLEEP-class actions while a hostile "
                    "entity is nearby and the environment has not confirmed "
                    "safety, followed by damage/chase/death.",
        scope="global survival hygiene (dangerous resting)"),
    "repeated_no_effect": dict(
        detector_id="repeated_no_effect_action",
        description="Long runs of actions that produce no observable state "
                    "change (possible tool misuse or action-semantics gap).",
        scope="global efficiency (resource/action waste)"),
    "oscillation_loop": dict(
        detector_id="oscillation_loop",
        description="Persistent A/B/A/B action alternation without progress "
                    "(exploration loop).",
        scope="global exploration health"),
    "combat_freeze": dict(
        detector_id="combat_freeze",
        description="Hostile adjacent while the Student repeats a non-combat "
                    "action and keeps taking damage (combat freeze).",
        scope="global threat engagement"),
    "resource_neglect": dict(
        detector_id="resource_neglect",
        description="Critical resource need persists while no resource-class "
                    "action is taken (resource waste / planning failure).",
        scope="global resource planning"),
    "unprepared_threat_approach": dict(
        detector_id="threat_approach_without_preparation",
        description="Distance to a hostile closes from far/none to near/"
                    "adjacent with no preparation (equipment/resource) actions "
                    "in the approach window.",
        scope="global preparation/planning"),
    "progress_regression": dict(
        detector_id="progress_regression",
        description="Achieved progress ordinal decreases and stays down across "
                    "a window (long-term planning failure).",
        scope="global long-horizon planning"),
    "premature_terminal": dict(
        detector_id="premature_terminal_behavior",
        description="Episode ends in death with no recent progress gain and a "
                    "terminal high-risk behavior pattern.",
        scope="global terminal decision-making"),
}

#: cause category -> description (the Analyst's closed vocabulary, section 6)
CAUSE_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "perception_or_observability": "The relevant signal (threat distance, safe "
        "zone, need level) may not be observable or discriminable from the "
        "Student's observation.",
    "memory_or_context_retention": "The Student may fail to retain the needed "
        "context (recent threat encounter, need trajectory) across steps.",
    "value_or_risk_misestimation": "The Student may misestimate the risk/value "
        "tradeoff (e.g. rest value over survival risk).",
    "resource_planning_failure": "The Student may fail to plan resource "
        "acquisition/consumption ahead of need.",
    "exploration_noise": "The behavior may be incidental exploration noise "
        "rather than a systematic policy defect.",
    "action_semantics_confusion": "The Student may misassociate an action's "
        "effect (action-semantics gap).",
    "distribution_shift": "The situation may lie outside the Student's training "
        "distribution.",
    "environment_ambiguity": "The environment itself may be ambiguous (no "
        "discriminable safety signal), i.e. not a Student defect at all.",
    "implementation_or_adapter_bug": "The observed anomaly may stem from an "
        "observation/action adapter bug rather than the policy.",
    "unknown": "Evidence insufficient to narrow the cause class.",
}

#: mutation axis -> description (legal TaskParams mutation vocabulary ONLY)
MUTATION_AXIS_DESCRIPTIONS: Dict[str, str] = {
    "threat_distance_grading": "Grade the initial/respawn distance between the "
        "Student and hostile entities.",
    "safe_rest_area_availability": "Vary the availability of areas where the "
        "environment confirms safety for rest.",
    "rest_need_pressure": "Vary the pressure of the rest/sleep need accumulator.",
    "threat_count": "Vary the number of simultaneous hostiles.",
    "view_occlusion": "Vary occlusion of the local view (partial observability "
        "of threats).",
    "resource_pressure": "Vary scarcity/pressure of key resources.",
    "day_night_rest_need": "Vary the coupling of the day/night cycle to rest "
        "need.",
    "visibility": "Vary overall visibility range.",
    "multi_threat_interference": "Vary interference between multiple threats.",
    "long_term_memory_requirement": "Vary how much long-horizon memory the task "
        "requires.",
    "global_task_conflict": "Vary conflict between concurrent global task "
        "objectives.",
}

ENVIRONMENT_FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "threat_distance_family": "Environments parametrized by threat-distance "
        "structure.",
    "resource_pressure_family": "Environments parametrized by resource "
        "pressure/scarcity.",
    "day_night_rest_need_family": "Environments parametrized by day/night-rest "
        "coupling.",
    "visibility_family": "Environments parametrized by visibility/occlusion.",
    "multi_threat_interference_family": "Environments parametrized by "
        "multi-threat interference.",
    "long_term_memory_family": "Environments parametrized by long-term memory "
        "requirement.",
    "global_task_conflict_family": "Environments parametrized by global "
        "task-objective conflict.",
}


def assert_taxonomy_consistent() -> None:
    """Fail-closed self-check (import-time integrity of the vocabulary)."""
    assert set(CAUSE_CATEGORY_DESCRIPTIONS) == set(C.CAUSE_CATEGORIES), \
        "CAUSE_CATEGORY_VOCAB_MISMATCH"
    assert set(MUTATION_AXIS_DESCRIPTIONS) == set(C.MUTATION_AXES), \
        "MUTATION_AXIS_VOCAB_MISMATCH"
    assert set(ENVIRONMENT_FAMILY_DESCRIPTIONS) == set(C.ENVIRONMENT_FAMILIES), \
        "ENVIRONMENT_FAMILY_VOCAB_MISMATCH"
    detector_ids = {p["detector_id"] for p in BEHAVIOR_PATTERNS.values()}
    assert len(detector_ids) == len(BEHAVIOR_PATTERNS), "DETECTOR_ID_COLLISION"
    # no pattern description may contain an action imperative (taxonomy is
    # descriptive, never prescriptive)
    for name, p in BEHAVIOR_PATTERNS.items():
        low = (p["description"] + p["scope"]).lower()
        for bad in ("you should", "the student should", "must flee", "must attack"):
            assert bad not in low, f"PRESCRIPTIVE_TAXONOMY:{name}:{bad}"


def taxonomy_document() -> dict:
    """The behavior_taxonomy.json artifact (hash-stamped)."""
    doc = dict(
        taxonomy_version=TAXONOMY_VERSION,
        bagr_ued_version=C.BA_BAGR_UED_VERSION,
        training_scope=C.TRAINING_SCOPE,
        tier3_only_training=C.TIER3_ONLY_TRAINING,
        behavior_patterns=BEHAVIOR_PATTERNS,
        cause_categories=CAUSE_CATEGORY_DESCRIPTIONS,
        mutation_axes=MUTATION_AXIS_DESCRIPTIONS,
        environment_families=ENVIRONMENT_FAMILY_DESCRIPTIONS,
        action_advice_policy="FORBIDDEN — taxonomy is descriptive; no pattern "
                             "carries an action recommendation",
    )
    doc["taxonomy_sha256"] = canonical_sha256(
        {k: v for k, v in doc.items() if k != "taxonomy_sha256"})
    return doc


def patterns_for_detector(detector_id: str) -> List[str]:
    return [name for name, p in BEHAVIOR_PATTERNS.items()
            if p["detector_id"] == detector_id]


assert_taxonomy_consistent()
