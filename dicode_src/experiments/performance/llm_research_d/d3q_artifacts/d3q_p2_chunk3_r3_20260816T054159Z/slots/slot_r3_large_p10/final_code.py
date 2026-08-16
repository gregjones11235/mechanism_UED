import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import Achievement


class Env(BaseTask):
    """Collect wood from the overworld.

    The player must collect at least one unit of wood from a tree.
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.COLLECT_WOOD]
        self.completed_achievements = []
        self.label = "COLLECT_WOOD"

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, world_rng = jax.random.split(rng)
        builder = WorldBuilder(world_rng, self.static_params, self.params)
        return builder.build(rng)