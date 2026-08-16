import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.world_builder import WorldBuilder

from craftax.craftax.constants import Achievement, BlockType


class Env(BaseTask):
    """Mine Coal: Start on the overworld with a wooden pickaxe and mine the coal placed next to you.

    World:
        - The player starts on the overworld at the default spawn position.
        - A coal block is placed directly in front of the player.
        - The player is given a wooden pickaxe so that the coal can be mined immediately.

    Task Params:
        - No additional task parameters are modified.

    Achievements:
        - COLLECT_COAL

    Completed Achievements:
        - (none)
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.COLLECT_COAL]
        self.completed_achievements = []
        self.label = "COLLECT_COAL"

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, build_rng = jax.random.split(rng)

        builder = WorldBuilder(rng, self.static_params, self.params)

        # Give the player a wooden pickaxe so they can mine coal.
        builder.set_player_inventory({"pickaxe": 1})

        # Place coal directly in front of the default spawn position.
        player_pos = (
            self.static_params.map_size[0] // 2,
            self.static_params.map_size[1] // 2,
        )
        builder.place_block(0, BlockType.COAL, (player_pos[0] - 1, player_pos[1]))

        return builder.build(build_rng)