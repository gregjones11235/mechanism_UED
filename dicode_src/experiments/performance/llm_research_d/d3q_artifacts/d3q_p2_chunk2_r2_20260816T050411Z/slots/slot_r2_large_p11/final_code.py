import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import Achievement, BlockType


class Env(BaseTask):
    """
    Collect a piece of wood from a tree.

    World: A standard overworld with trees. The player spawns in the middle
    of the overworld with a tree directly above them, so the very first
    action can be DO to mine it and gain wood.
    """
    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.COLLECT_WOOD]
        self.completed_achievements = []
        self.label = "COLLECT_WOOD"

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        builder = WorldBuilder(rng, self.static_params, self.params)
        builder.set_starting_floor(0)

        # Place a tree immediately above the player to make the task trivial.
        builder.place_block(
            0,
            BlockType.TREE,
            (builder.player_position[0] - 1, builder.player_position[1]),
        )

        return builder.build(rng)