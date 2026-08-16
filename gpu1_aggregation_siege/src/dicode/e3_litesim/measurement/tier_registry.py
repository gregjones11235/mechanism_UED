"""Tier registry: capability families probed with comprehensive metrics.

Config-driven on purpose: the Tier3-front dark-corridor world generator is
declared here and validated on the GPU server; locally the registry runs the
tiers whose worlds are constructible (survive / combat / original-proxy).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class TierSpec:
    tier_id: str
    skill_family: str
    order: int
    task_module: str
    horizon: int
    success_predicate_name: str
    server_only: bool = False

    def make_env(self) -> Any:
        from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
        from minicraftax.envs.base import MiniCraftaxTrain

        mod = importlib.import_module(
            f"minicraftax.tasks.seed_tasks.{self.task_module}")
        task = mod.Env(static_params=StaticEnvParams(), params=EnvParams())
        return MiniCraftaxTrain(task=task)


def _pred_survive(state: Any) -> np.ndarray:
    return np.asarray(state.player_health) > 0


def _pred_combat(state: Any) -> np.ndarray:
    killed = np.asarray(state.monsters_killed)
    if killed.ndim > 1:
        killed = killed.sum(axis=-1)
    return killed > 0


def _pred_original_front(state: Any) -> np.ndarray:
    return np.asarray(state.player_level) >= 2


SUCCESS_PREDICATES = {
    "survived_horizon": _pred_survive,
    "monster_killed": _pred_combat,
    "reached_floor2": _pred_original_front,
}


DEFAULT_TIERS = (
    TierSpec("tier1_survive", "BASIC_SURVIVAL", 1, "survive", 64,
             "survived_horizon"),
    TierSpec("tier2_combat", "THREAT_MANAGEMENT", 2, "combat", 96,
             "monster_killed"),
    # Tier3 FRONT (Floor2 DARK corridor -> Floor3): the real dark-corridor
    # world generator is validated on the GPU server; locally the original
    # task (dungeon transition) is the registered proxy.
    TierSpec("tier3_front", "DARK_NAVIGATION", 3, "original", 128,
             "reached_floor2"),
)


class TierRegistry:
    def __init__(self, tiers: tuple = DEFAULT_TIERS) -> None:
        self._tiers = {t.tier_id: t for t in tiers}

    def ids(self) -> list:
        return [t.tier_id for t in sorted(self._tiers.values(),
                                          key=lambda t: t.order)]

    def get(self, tier_id: str) -> TierSpec:
        return self._tiers[tier_id]

    def predicate(self, tier_id: str) -> Callable[[Any], np.ndarray]:
        return SUCCESS_PREDICATES[self._tiers[tier_id].success_predicate_name]