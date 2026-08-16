import jax
import jax.numpy as jnp
from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import Achievement


class Env(BaseTask):
    """
    A simple task where the agent must collect wood from trees.

    Achievements:
        - Collect Wood: Acquire at least one unit of wood.
    Completed Achievements:
        - None
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.COLLECT_WOOD]
        self.completed_achievements = []
        self.label = "Collect Wood"

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        world_rng, build_rng = jax.random.split(rng)
        builder = WorldBuilder(world_rng, self.static_params, self.params)
        return builder.build(build_rng)