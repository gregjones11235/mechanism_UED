import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.world_builder import WorldBuilder
from minicraftax.tasks.base_task import BaseTask
from craftax.craftax.constants import Achievement, BlockType


class Env(BaseTask):
    """
    Collect Wood

    The agent must collect wood by mining a tree. The world is the standard
    overworld, with no hostile mobs, and the player begins with an empty
    inventory. Several trees are placed near the player spawn so the agent can
    easily gather wood.

    World:
    - Start on the overworld (floor 0).
    - Place 10 trees on grass near the player spawn.
    - Disable all mob spawns.

    Achievements:
    - COLLECT_WOOD

    Completed Achievements:
    - None

    Task Params:
    - passive_spawn_multiplier: 0.0
    - melee_spawn_multiplier: 0.0
    - ranged_spawn_multiplier: 0.0
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [
            Achievement.COLLECT_WOOD,
        ]
        self.completed_achievements = []
        self.label = ", ".join([a.name for a in self.relevant_achievements])

    def get_task_params(self) -> TaskParams:
        return TaskParams(
            passive_spawn_multiplier=0.0,
            melee_spawn_multiplier=0.0,
            ranged_spawn_multiplier=0.0,
        )

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, builder_rng, placement_rng, build_rng = jax.random.split(rng, 4)

        builder = WorldBuilder(builder_rng, self.static_params, self.params)
        builder.set_starting_floor(0)

        player_pos = builder.player_position
        builder.place_randomly_near(
            placement_rng,
            level=0,
            block_type=BlockType.TREE,
            target_pos=(player_pos[0], player_pos[1]),
            min_dist=1,
            max_dist=5,
            n=10,
            on_blocks=[BlockType.GRASS],
        )

        return builder.build(build_rng)