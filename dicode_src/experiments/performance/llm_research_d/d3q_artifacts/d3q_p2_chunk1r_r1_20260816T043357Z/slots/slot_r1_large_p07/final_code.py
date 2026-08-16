import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder

from craftax.craftax.constants import BlockType, Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams


class Env(BaseTask):
    """
    Task: Collect Wood

    The player must collect a single piece of wood from a tree on the overworld.

    Achievements:
    - COLLECT_WOOD: Obtain wood from a tree.
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.COLLECT_WOOD]
        self.completed_achievements = []
        self.label = "Collect Wood: [COLLECT_WOOD]"

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, world_rng, build_rng = jax.random.split(rng, 3)

        builder = WorldBuilder(world_rng, self.static_params, self.params)

        # Ensure trees are plentiful near the spawn point so that the
        # objective can be completed quickly and reliably.
        player_position = builder.player_position
        builder.place_randomly_near(
            rng,
            level=0,
            block_type=BlockType.TREE,
            target_pos=player_position,
            min_dist=1,
            max_dist=4,
            n=10,
            on_blocks=[BlockType.GRASS],
        )

        return builder.build(build_rng)