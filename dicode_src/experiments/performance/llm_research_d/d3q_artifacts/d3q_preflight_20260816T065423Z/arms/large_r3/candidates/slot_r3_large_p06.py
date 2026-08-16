import jax
import jax.numpy as jnp

from minicraftax.craftax_state import TaskParams, EnvState
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import Achievement, BlockType


class Env(BaseTask):
    """
    Collect Wood

    World:
    - The agent starts in the overworld.
    - Several trees are placed close to the spawn point so wood is easy to collect.

    Achievements:
    - COLLECT_WOOD

    Completed Achievements:
    - None
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.COLLECT_WOOD]
        self.completed_achievements = []
        self.label = ", ".join([a.name for a in self.relevant_achievements])

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, builder_rng, placement_rng, build_rng = jax.random.split(rng, 4)

        builder = WorldBuilder(builder_rng, self.static_params, self.params)

        player_pos = (
            self.static_params.map_size[0] // 2,
            self.static_params.map_size[1] // 2,
        )

        builder.place_randomly_near(
            placement_rng,
            level=0,
            block_type=BlockType.TREE,
            target_pos=player_pos,
            min_dist=1,
            max_dist=3,
            n=10,
            on_blocks=[BlockType.GRASS],
        )

        state = builder.build(build_rng)

        if len(self.completed_achievements) > 0:
            completed_indices = jnp.array(
                [achievement.value for achievement in self.completed_achievements]
            )
            state = state.replace(
                achievements=state.achievements.at[completed_indices].set(True)
            )

        return state