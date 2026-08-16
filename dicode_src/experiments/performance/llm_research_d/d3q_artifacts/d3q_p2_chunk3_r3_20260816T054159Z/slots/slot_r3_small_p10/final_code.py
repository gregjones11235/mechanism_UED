import jax
from craftax.craftax.constants import Achievement, BlockType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
    """Objective: Improve the agent's ability to enter the Gnomish Mines by focusing on collecting ruby and sapphire, which are essential for enchanting.
    Description: The player must achieve `ENTER_GNOMISH_MINES` and collect ruby (`COLLECT_RUBY`) and sapphire (`COLLECT_SAPPHIRE`). The player starts in the Overworld with a wooden sword, some basic resources, and essential blocks like stone and coal. This task is designed to build on the agent's existing skills by introducing the new dependency of entering the Gnomish Mines and focusing on collecting valuable gems.

    Relevant Achievements: ENTER_GNOMISH_MINES, COLLECT_RUBY, COLLECT_SAPPHIRE
    Completed Achievements: MAKE_WOOD_SWORD, COLLECT_WOOD, PLACE_TABLE, MAKE_STONE_PICKAXE, MAKE_STONE_SWORD, COLLECT_COAL, COLLECT_IRON, PLACE_FURNACE

    World:
    - Player: Starts in the Overworld with a wooden sword (`{"sword": 1}`) and some basic resources (`{"wood": 20, "stone": 10, "coal": 5}`). 
    - Map: Default procedural Gnomish Mines (Floor 2). 5 `ZOMBIE` mobs (melee mob type_id=0) are placed within a 4-tile radius from the player. Additionally, 3 `SKELETON` mobs (ranged mob type_id=0) and 1 `COW` mob (passive mob type_id=1) are placed randomly within a 6-tile radius.
    - Mechanics: "needs_depletion_multiplier = 0.7", "passive_spawn_multiplier = 0.5", "melee_spawn_multiplier = 0.3", "ranged_spawn_multiplier = 0.2"
    """

    def __init__(self, static_params: StaticEnvParams, params: EnvParams):
        super().__init__(static_params, params)
        self.relevant_achievements = [
            Achievement.ENTER_GNOMISH_MINES,
            Achievement.COLLECT_RUBY,
            Achievement.COLLECT_SAPPHIRE,
        ]
        self.completed_achievements = [
            Achievement.MAKE_WOOD_SWORD,
            Achievement.COLLECT_WOOD,
            Achievement.PLACE_TABLE,
            Achievement.MAKE_STONE_PICKAXE,
            Achievement.MAKE_STONE_SWORD,
            Achievement.COLLECT_COAL,
            Achievement.COLLECT_IRON,
            Achievement.PLACE_FURNACE,
        ]
        self.label = "ENTER_GNOMISH_MINES"

    def get_task_params(self) -> TaskParams:
        """Return custom parameters for this task."""
        return TaskParams(
            passive_spawn_multiplier=0.5,  # Enable random cow spawns
            melee_spawn_multiplier=0.3,  # Enable zombie spawns
            ranged_spawn_multiplier=0.2,  # Enable skeleton spawns
            needs_depletion_multiplier=0.7,  # Needs are on, but slower than usual
        )

    def generate_world(self, rng: jax.Array) -> EnvState:
        """Generates the world for the task."""
        rng, build_rng, placement_rng, cow_rng, zombie_rng, skeleton_rng = jax.random.split(
            rng, 6
        )

        builder = WorldBuilder(build_rng, self.static_params, self.params)

        builder.set_starting_floor(0)

        # --- ADDED SCAFFOLDING ---
        # 1. Give prerequisite pickaxe and a sword for safety
        builder.set_player_inventory({"wood": 20, "stone": 10, "coal": 5, "sword": 1})

        # 2. Place cows as a food source
        builder.add_mobs_randomly_near(
            cow_rng,
            level=0,
            mob_name="passive",
            type_id=1,  # type_id 1 is Cow
            n=3,
            target_pos=builder.player_position,
            min_dist=4,
            max_dist=8,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )
        # --- END SCAFFOLDING ---

        # Place mobs in the Gnomish Mines (Floor 2)
        builder.set_starting_floor(2)

        # Place zombies within a 4-tile radius from the player
        builder.add_mobs_randomly_near(
            zombie_rng,
            level=2,
            mob_name="melee",
            type_id=0,  # type_id 0 is Zombie
            n=5,
            target_pos=builder.player_position,
            min_dist=4,
            max_dist=4,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        # Place skeletons within a 6-tile radius from the player
        builder.add_mobs_randomly_near(
            skeleton_rng,
            level=2,
            mob_name="ranged",
            type_id=0,  # type_id 0 is Skeleton
            n=3,
            target_pos=builder.player_position,
            min_dist=6,
            max_dist=6,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        return builder.build(rng)