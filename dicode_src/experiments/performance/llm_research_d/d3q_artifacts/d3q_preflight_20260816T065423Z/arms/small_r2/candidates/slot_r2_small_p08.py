import jax
from craftax.craftax.constants import Achievement, BlockType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
    """Objective: Train the agent to navigate to and enter the Gnomish Mines.
    Description: The player must achieve `ENTER_GNOMISH_MINES`. The player starts in the Overworld with a wooden sword, some basic resources, and essential blocks like stone and coal. Mobs are enabled to encourage focused learning on combat while navigating and surviving in the Gnomish Mines. This task is designed to build on the agent's existing skills by introducing the new dependency of entering the Gnomish Mines.

    Relevant Achievements: ENTER_GNOMISH_MINES
    Completed Achievements: COLLECT_WOOD, PLACE_TABLE, EAT_COW, COLLECT_SAPLING, COLLECT_DRINK, MAKE_WOOD_PICKAXE, MAKE_WOOD_SWORD, PLACE_PLANT, WAKE_UP, PLACE_FURNACE, COLLECT_COAL
    World:
    - Player: Starts in the Overworld with a wooden sword (`{"sword": 1}`) and some basic resources (`{"wood": 20, "stone": 10, "coal": 5}`). 
    - Map: Default procedural Overworld (Floor 0). No explicit block modifications are made to the default map.
    - Mechanics: "needs_depletion_multiplier = 0.7", "passive_spawn_multiplier = 0.5", "melee_spawn_multiplier = 0.3", "ranged_spawn_multiplier = 0.2"
    """

    def __init__(self, static_params: StaticEnvParams, params: EnvParams):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.ENTER_GNOMISH_MINES]
        self.completed_achievements = [
            Achievement.COLLECT_WOOD,
            Achievement.PLACE_TABLE,
            Achievement.EAT_COW,
            Achievement.COLLECT_SAPLING,
            Achievement.COLLECT_DRINK,
            Achievement.MAKE_WOOD_PICKAXE,
            Achievement.MAKE_WOOD_SWORD,
            Achievement.PLACE_PLANT,
            Achievement.WAKE_UP,
            Achievement.PLACE_FURNACE,
            Achievement.COLLECT_COAL,
        ]
        self.label = "ENTER_GNOMISH_MINES"

    def get_task_params(self) -> TaskParams:
        """Return custom parameters for this task."""
        return TaskParams(
            needs_depletion_multiplier=0.7,
            passive_spawn_multiplier=0.5,
            melee_spawn_multiplier=0.3,
            ranged_spawn_multiplier=0.2,
        )

    def generate_world(self, rng: jax.Array) -> EnvState:
        """Generates the world for the task."""
        rng, build_rng, placement_rng = jax.random.split(rng, 3)

        builder = WorldBuilder(build_rng, self.static_params, self.params)

        builder.set_starting_floor(0)

        # --- ADDED SCAFFOLDING ---
        # 1. Give prerequisite pickaxe and a sword for safety
        builder.set_player_inventory(
            {"wood": 20, "stone": 10, "coal": 5, "sword": 1}
        )

        # 2. Place cows as a food source
        builder.add_mobs_randomly_near(
            rng,
            level=0,
            mob_name="passive",
            type_id=0,  # type_id 0 is Cow
            n=3,
            target_pos=builder.player_position,
            min_dist=4,
            max_dist=8,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )
        # --- END SCAFFOLDING ---

        return builder.build(rng)