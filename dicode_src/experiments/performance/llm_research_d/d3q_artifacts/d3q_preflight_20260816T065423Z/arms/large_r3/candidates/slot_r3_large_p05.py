import jax
import jax.numpy as jnp

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.world_builder import WorldBuilder
from craftax.craftax.constants import Achievement, BlockType
from minicraftax.tasks.base_task import BaseTask


class Env(BaseTask):
    """Eat a Cow.

    The player must find and eat a cow. The player starts with a stone sword
    and a cow placed nearby. Eating a cow requires moving next to it and
    using the DO action to attack it.
    """

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.EAT_COW]
        self.completed_achievements = []
        self.label = "EAT_COW"

    def get_task_params(self) -> TaskParams:
        return TaskParams()

    def generate_world(self, rng: jax.Array) -> EnvState:
        builder = WorldBuilder(rng, self.static_params, self.params)

        # Give the player a stone sword to make killing the cow easier.
        builder.set_player_inventory({"sword": 1})

        # Place a cow near the player so the task is immediately actionable.
        rng, cow_rng = jax.random.split(rng)
        builder.add_mobs_randomly_near(
            cow_rng,
            level=0,
            mob_name="passive",
            type_id=0,
            n=1,
            target_pos=builder.player_position,
            min_dist=1,
            max_dist=3,
            on_blocks=[BlockType.GRASS],
        )

        state = builder.build(rng)

        # Mark any pre-completed achievements.
        if self.completed_achievements:
            achievement_indices = jnp.array(
                [a.value for a in self.completed_achievements]
            )
            state = state.replace(
                achievements=state.achievements.at[achievement_indices].set(True)
            )

        return state