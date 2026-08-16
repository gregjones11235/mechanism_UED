import jax
import numpy as np
from craftax.craftax.craftax_state import EnvParams

from dicode.e3_litesim.measurement.tier_registry import TierRegistry, TierSpec
from dicode.e3_litesim.runtime.slice_student import SliceStudentBackend


def small_registry() -> TierRegistry:
    return TierRegistry(tiers=(
        TierSpec("tier1_survive", "BASIC_SURVIVAL", 1, "survive", 16,
                 "survived_horizon"),
        TierSpec("tier2_combat", "THREAT_MANAGEMENT", 2, "combat", 24,
                 "monster_killed"),
        TierSpec("tier3_front", "DARK_NAVIGATION", 3, "original", 32,
                 "reached_floor2"),
    ))


def make_setup(seed=0, tier="tier1_survive"):
    env_params = EnvParams(max_timesteps=4096)
    registry = small_registry()
    env = registry.get(tier).make_env()
    obs0, _ = env.reset(jax.random.PRNGKey(seed), env_params)
    backend = SliceStudentBackend(int(np.asarray(obs0).shape[-1]),
                                  int(env.action_space(env_params).n))
    params = backend.initial_params(jax.random.PRNGKey(seed))
    return {"env": env, "env_params": env_params, "registry": registry,
            "backend": backend, "params": params}