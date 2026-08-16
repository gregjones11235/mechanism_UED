import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import Achievement, BlockType


class Env(BaseTask):
    """
    Collect wood from trees on the overworld.

    Achievements:
    - COLLECT_WOOD

    Completed Achievements:
    - None

    World:
    Spawn in the overworld with several trees placed near the player.
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)

        self.relevant_achievements = [Achievement.COLLECT_WOOD]
        self.completed_achievements = []
        self.label = "Achievements: " + ", ".join(
            a.name for a in self.relevant_achievements
        )

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        builder = WorldBuilder(rng, self.static_params, self.params)

        # Ensure there are trees close to the spawn point.
        builder.place_randomly_near(
            rng,
            level=0,
            block_type=BlockType.TREE,
            target_pos=builder.player_position,
            min_dist=1,
            max_dist=8,
            n=16,
            on_blocks=[BlockType.GRASS],
        )

        state = builder.build(rng)

        # Set any achievements that should already be completed at the start.
        if self.completed_achievements:
            achievement_values = jnp.array(
                [a.value for a in self.completed_achievements], dtype=jnp.int32
            )
            achievements = state.achievements.at[achievement_values].set(True)
            state = state.replace(achievements=achievements)

        return state