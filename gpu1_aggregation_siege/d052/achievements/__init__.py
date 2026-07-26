"""Official Craftax-67 achievement registry (canonical_v2 single source).

Public surface:
    REGISTRY            shared AchievementRegistry instance
    AchievementRegistry the registry class
    AchievementError    fail-closed violation (unknown / empty / drift)
    NUM_ACHIEVEMENTS    67
    ACHIEVEMENT_SCHEMA  "craftax_67_v1"
"""
from d052.achievements.registry import (
    ACHIEVEMENT_SCHEMA,
    NUM_ACHIEVEMENTS,
    REGISTRY,
    AchievementError,
    AchievementRegistry,
)

__all__ = [
    "ACHIEVEMENT_SCHEMA",
    "NUM_ACHIEVEMENTS",
    "REGISTRY",
    "AchievementError",
    "AchievementRegistry",
]
