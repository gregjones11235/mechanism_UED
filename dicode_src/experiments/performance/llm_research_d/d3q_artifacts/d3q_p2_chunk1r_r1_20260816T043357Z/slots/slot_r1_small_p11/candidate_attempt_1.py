import jax
from craftax.craftax.constants import Achievement, BlockType, MobType, ProjectileType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
    """
    Objective: Improve the agent's ability to survive and progress in the Gnomish Mines by defending against a variety of mobs, collecting iron, and crafting an iron pickaxe.

    Description: The player must achieve `MAKE_IRON_ARMOUR`, `COLLECT_IRON`, and `MAKE_IRON_PICKAXE` while surviving in the Gnomish Mines. The agent starts with a wooden sword (`{"sword": 1}`) and some basic resources (`{"wood": 20, "stone": 10, "coal": 5}`). The Gnomish Mines environment is filled with hostile mobs such as skeletons and zombies to test the agent's combat skills. Resources like iron are sparse, requiring efficient management.

    Relevant Achievements: MAKE_IRON_ARMOUR, COLLECT_IRON, MAKE_IRON_PICKAXE, ENTER_GNOMISH_MINES
    Completed Achievements: MAKE_WOOD_SWORD, ENTER_DUNGEON

    World:
    - Player: Starts in the Gnomish Mines with a wooden sword (`{"sword": 1}`) and some basic resources (`{"wood": 20, "stone": 10, "coal": 5}`). 
    - Map: Default procedural dungeon (Floor 2). 3 `SKELETON` mobs (ranged mob type_id=0), 2 `ZOMBIE` mobs (melee mob type_id=0), and 1 `COW` mob (passive mob type_id=1) are placed randomly within a 6-tile radius from the player. Additionally, 3 `GNOME_WARRIOR` mobs (melee mob type_id=1) and 2 `SKELETON_ARCHER` mobs (ranged mob type_id=1) are placed randomly within an 8-tile radius.
    - Mechanics: "needs_depletion_multiplier = 0.7", "passive_spawn_multiplier = 0.3", "melee_spawn_multiplier = 0.5", "ranged_spawn_multiplier = 0.4"
    """

    def __init__(self, static_params: StaticEnvParams, params: EnvParams):
        super().__init__(static_params, params)
        self.relevant_achievements = [
            Achievement.MAKE_IRON_ARMOUR,
            Achievement.COLLECT_IRON,
            Achievement.MAKE_IRON_PICKAXE,
            Achievement.ENTER_GNOMISH_MINES,
        ]
        self.completed_achievements = [
            Achievement.MAKE_WOOD_SWORD,
            Achievement.ENTER_DUNGEON,
        ]
        self.label = "MAKE_IRON_ARMOUR, COLLECT_IRON, MAKE_IRON_PICKAXE, ENTER_GNOMISH_MINES"

    def get_task_params(self) -> TaskParams:
        """Return custom parameters for this task."""
        return TaskParams(
            passive_spawn_multiplier=0.3,
            melee_spawn_multiplier=0.5,
            ranged_spawn_multiplier=0.4,
            needs_depletion_multiplier=0.7,
        )

    def generate_world(self, rng: jax.Array) -> EnvState:
        """Generates the world for the task."""
        (
            rng,
            build_rng,
            placement_rng,
            cow_rng,
            skeleton_rng,
            zombie_rng,
            gnome_warrior_rng,
            skeleton_archer_rng,
        ) = jax.random.split(rng, 8)

        builder = WorldBuilder(build_rng, self.static_params, self.params)

        builder.set_starting_floor(2)

        # --- ADDED SCAFFOLDING ---
        # 1. Give prerequisite sword and basic resources for safety
        builder.set_player_inventory({"sword": 1, "wood": 20, "stone": 10, "coal": 5})

        # 2. Place cows as a food source
        builder.add_mobs_randomly_near(
            cow_rng,
            level=2,
            mob_name="passive",
            type_id=1,  # type_id 1 is Cow
            n=1,
            target_pos=builder.player_position,
            min_dist=6,
            max_dist=6,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        # 3. Place skeletons as a ranged threat
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

        # 4. Place zombies as a melee threat
        builder.add_mobs_randomly_near(
            zombie_rng,
            level=2,
            mob_name="melee",
            type_id=0,  # type_id 0 is Zombie
            n=2,
            target_pos=builder.player_position,
            min_dist=6,
            max_dist=6,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        # 5. Place gnome warriors as a melee threat
        builder.add_mobs_randomly_near(
            gnome_warrior_rng,
            level=2,
            mob_name="melee",
            type_id=1,  # type_id 1 is Gnome Warrior
            n=3,
            target_pos=builder.player_position,
            min_dist=8,
            max_dist=8,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        # 6. Place skeleton archers as a ranged threat
        builder.add_mobs_randomly_near(
            skeleton_archer_rng,
            level=2,
            mob_name="ranged",
            type_id=1,  # type_id 1 is Gnome Archer
            n=2,
            target_pos=builder.player_position,
            min_dist=8,
            max_dist=8,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )
        # --- END SCAFFOLDING ---

        return builder.build(rng)