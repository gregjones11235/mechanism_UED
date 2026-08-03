"""Provenance and data-leakage guards for frontier statistics."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from .errors import ProvenanceViolationError


class DataSource(str, Enum):
    TRAINING_STANDARD_ROLLOUT = "TRAINING_STANDARD_ROLLOUT"
    TRAINING_FRONTIER_CAPTURE = "TRAINING_FRONTIER_CAPTURE"
    TRAINING_BRANCH_SEARCH = "TRAINING_BRANCH_SEARCH"
    TRAINING_FRONTIER_START = "TRAINING_FRONTIER_START"
    FORMAL_FRONT = "FORMAL_FRONT"
    FORMAL_BACK = "FORMAL_BACK"
    FORMAL_FULL = "FORMAL_FULL"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"


class FormalDataLeakageGuard:
    FORBIDDEN = {DataSource.FORMAL_FRONT, DataSource.FORMAL_BACK, DataSource.FORMAL_FULL}

    @classmethod
    def assert_allowed(cls, source: DataSource | str, consumer: str) -> None:
        source = DataSource(source)
        if source in cls.FORBIDDEN and consumer in {"FrontierArchive", "BranchOutcome", "FeasibilityEstimate", "curriculum", "Student optimizer"}:
            raise ProvenanceViolationError(f"{source.value} cannot enter {consumer}")


class SearchActionLeakageGuard:
    FORBIDDEN_KEYS = {"action_sequence", "waypoint", "route", "expert_trajectory", "logits", "hidden_states"}

    @classmethod
    def validate_aggregate(cls, value: Any) -> None:
        def walk(node: Any, path: str = "") -> None:
            if isinstance(node, Mapping):
                for key, child in node.items():
                    name = str(key).lower()
                    if name in cls.FORBIDDEN_KEYS:
                        raise ProvenanceViolationError(f"action guidance field forbidden at {path}.{key}")
                    walk(child, f"{path}.{key}" if path else str(key))
            elif isinstance(node, (list, tuple)):
                for i, child in enumerate(node): walk(child, f"{path}[{i}]")
        walk(value)
