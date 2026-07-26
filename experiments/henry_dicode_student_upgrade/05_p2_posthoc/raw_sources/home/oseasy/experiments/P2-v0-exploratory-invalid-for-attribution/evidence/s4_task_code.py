import jax
from craftax.craftax.constants import Achievement, ItemType

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
    """Objective: Spawn-anneal relay rung for DEFEAT_KOBOLD (rung floor 2 scaffold stage 4 (spawn at the floor entry, no scaffold pre-light (the floor's own light only), 8/8 kills pre-credited, up-ladder REMOVED (committed descent — no retreat), survival clocks at 0.3x), winner-median kit).
    Description: The REAL full 9-level Craftax world (fresh world every episode), spawn on floor 2 with a winner-median starting kit {'wood': 7, 'stone': 27, 'coal': 3, 'iron': 3, 'sapling': 1, 'pickaxe': 3, 'sword': 3, 'bow': 1, 'arrows': 7, 'torches': 10}; reach and complete DEFEAT_KOBOLD from there.
    Relevant Achievements: DEFEAT_KOBOLD
    Completed Achievements: NONE
    World: standard world generation, starting floor 2."""

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.DEFEAT_KOBOLD]
        self.completed_achievements = []
        self.label = "DEFEAT_KOBOLD"

    def get_task_params(self) -> TaskParams:
        return TaskParams(needs_depletion_multiplier=0.3)

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, _rng = jax.random.split(rng)
        builder = WorldBuilder(_rng, self.static_params, self.params)
        builder.set_starting_floor(2)
        builder.set_monsters_killed(2, 8)
        builder.set_player_inventory({'wood': 7, 'stone': 27, 'coal': 3, 'iron': 3, 'sapling': 1, 'pickaxe': 3, 'sword': 3, 'bow': 1, 'arrows': 7, 'torches': 10})
        state = builder.build(rng)
        up = builder.ladders_up[2]
        state = state.replace(item_map=state.item_map.at[2, up[0], up[1]].set(ItemType.NONE.value))
        return state
