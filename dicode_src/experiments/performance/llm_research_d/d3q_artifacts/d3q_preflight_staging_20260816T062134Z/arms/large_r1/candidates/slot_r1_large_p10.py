import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import Achievement


class Env(BaseTask):
    """
    Name: Collect Wood

    Description: The player must collect at least one unit of wood by mining a tree.
    The overworld is populated with trees that can be chopped by facing them and pressing the DO action.

    Achievements:
    - COLLECT_WOOD

    Completed Achievements:
    - None

    World:
    The player spawns at the centre of the standard overworld with default attributes, full needs
    and an empty inventory. No modifications are made to the default world generation; the natural
    tree distribution provides ample opportunity to complete the objective.
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)

        self.relevant_achievements = [
            Achievement.COLLECT_WOOD,
        ]

        self.completed_achievements = []

        self.label = "COLLECT_WOOD"

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        params = self.params
        static_params = self.static_params

        rng, world_rng, build_rng = jax.random.split(rng, 3)
        builder = WorldBuilder(world_rng, static_params, params)
        state = builder.build(build_rng)

        return state