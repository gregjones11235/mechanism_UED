import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import BlockType, Achievement


class Env(BaseTask):
    """
    Collect Wood

    The player must collect at least one piece of wood to complete the task.
    Wood is obtained by pressing the DO action while facing a tree, which
    causes the player to mine the tree and gain one wood.

    ## Achievements
    - COLLECT_WOOD: Obtain at least 1 wood in the inventory.

    ## Completed Achievements
    None

    ## World
    The player spawns on the standard overworld. Extra trees are placed near
    the player's spawn so that wood is easy to collect.

    ## Task Parameters
    The default task parameters are used.
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.COLLECT_WOOD]
        self.completed_achievements = []
        self.label = ", ".join([a.name for a in self.relevant_achievements])

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, world_rng, tree_rng, build_rng = jax.random.split(rng, 4)
        builder = WorldBuilder(world_rng, self.static_params, self.params)
        builder.set_starting_floor(0)

        player_pos = (
            self.static_params.map_size[0] // 2,
            self.static_params.map_size[1] // 2,
        )

        builder.place_randomly_near(
            tree_rng,
            level=0,
            block_type=BlockType.TREE,
            target_pos=player_pos,
            min_dist=1,
            max_dist=6,
            n=25,
            on_blocks=[BlockType.GRASS],
        )

        state = builder.build(build_rng)
        return state