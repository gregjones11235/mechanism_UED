import jax
import jax.numpy as jnp
from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.world_builder import WorldBuilder
from minicraftax.tasks.base_task import BaseTask
from craftax.craftax.constants import Achievement, ACHIEVEMENT_REWARD_MAP
from craftax.craftax.util.game_logic_utils import has_beaten_boss


class Env(BaseTask):
    """Default MiniCraftax task."""

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = []
        self.completed_achievements = []
        self.label = ""

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def is_terminal(self, state):
        done_steps = state.timestep >= self.params.max_timesteps
        is_dead = state.player_health <= 0
        defeated_boss = has_beaten_boss(state, self.static_params)

        if len(self.relevant_achievements) == 0:
            return done_steps | is_dead | defeated_boss

        achievement_coeff = ACHIEVEMENT_REWARD_MAP
        relevant_achievements = jnp.array([b.value for b in self.relevant_achievements])
        current_achievements = state.achievements.astype(int)
        mask = jnp.zeros_like(current_achievements).at[relevant_achievements].set(1)
        total_possible_achievements = (mask * achievement_coeff).sum()
        total_achievements_so_far = (current_achievements * mask * achievement_coeff).sum()

        task_solved = total_achievements_so_far >= total_possible_achievements

        return done_steps | is_dead | defeated_boss | task_solved

    def get_reward(self, prev_state, next_state):
        if len(self.relevant_achievements) == 0:
            return jnp.array(0.0)

        achievement_coeff = ACHIEVEMENT_REWARD_MAP
        relevant_achievements = jnp.array([b.value for b in self.relevant_achievements])
        achievement_delta = next_state.achievements.astype(int) - prev_state.achievements.astype(int)
        mask = jnp.zeros_like(achievement_delta).at[relevant_achievements].set(1)
        achievement_reward = (achievement_delta * mask * achievement_coeff).sum()

        return achievement_reward

    def is_success(self, state):
        if len(self.relevant_achievements) == 0:
            return jnp.array(False)

        achievement_coeff = ACHIEVEMENT_REWARD_MAP
        relevant_achievements = jnp.array([b.value for b in self.relevant_achievements])
        current_achievements = state.achievements.astype(int)
        mask = jnp.zeros_like(current_achievements).at[relevant_achievements].set(1)
        total_possible_achievements = (mask * achievement_coeff).sum()
        total_achievements_so_far = (current_achievements * mask * achievement_coeff).sum()

        return total_achievements_so_far >= total_possible_achievements

    def generate_world(self, rng: jax.Array) -> EnvState:
        builder = WorldBuilder(rng, self.static_params, self.params)
        return builder.build(rng)