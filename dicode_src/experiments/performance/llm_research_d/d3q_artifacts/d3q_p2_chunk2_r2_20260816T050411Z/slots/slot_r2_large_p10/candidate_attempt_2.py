import jax
import jax.numpy as jnp
from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.world_builder import WorldBuilder
from minicraftax.tasks.base_task import BaseTask
from craftax.craftax.constants import Achievement


class Env(BaseTask):
    """Default MiniCraftax task."""

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = []
        self.completed_achievements = []
        self.label = ""

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        builder = WorldBuilder(rng, self.static_params, self.params)
        return builder.build(rng)