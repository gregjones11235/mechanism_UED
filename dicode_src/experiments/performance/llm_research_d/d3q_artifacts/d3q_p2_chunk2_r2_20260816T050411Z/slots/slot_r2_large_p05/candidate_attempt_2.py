import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder

from craftax.craftax.constants import Achievement, BlockType


class Env(BaseTask):
    """
    Collect wood and craft a wooden pickaxe.

    Achievements:
    - COLLECT_WOOD
    - MAKE_WOOD_PICKAXE

    Completed Achievements:
    - (none)

    World:
    - The player starts in the overworld with several trees nearby.
    - A crafting table is placed adjacent to the player spawn.

    Task Params:
    - None
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)

        self.relevant_achievements = [
            Achievement.COLLECT_WOOD,
            Achievement.MAKE_WOOD_PICKAXE,
        ]
        self.completed_achievements = []
        self.label = ", ".join([a.name for a in self.relevant_achievements])

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        builder = WorldBuilder(rng, self.static_params, self.params)

        player_pos = (int(builder.player_position[0]), int(builder.player_position[1]))

        # Place a crafting table next to where the player spawns.
        builder.place_block(
            0,
            BlockType.CRAFTING_TABLE,
            (player_pos[0], player_pos[1] + 1),
        )

        # Guarantee a good supply of wood immediately around the spawn.
        rng, _rng = jax.random.split(rng)
        builder.place_randomly_near(
            _rng,
            level=0,
            block_type=BlockType.TREE,
            target_pos=player_pos,
            min_dist=2,
            max_dist=8,
            n=20,
            on_blocks=[BlockType.GRASS],
        )

        state = builder.build(rng)

        if self.completed_achievements:
            achievement_indices = jnp.array(
                [a.value for a in self.completed_achievements]
            )
            state = state.replace(
                achievements=state.achievements.at[achievement_indices].set(True)
            )

        return state