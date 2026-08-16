import jax
from craftax.craftax.constants import Achievement, BlockType, MobType, ProjectileType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
    """Objective: Craft an iron pickaxe and enter the Gnomish Mines when you have a wooden sword and stone is nearby.
    Description: The player must achieve the `MAKE_IRON_PICKAXE` and `ENTER_GNOMISH_MINES` achievements. The player starts on Floor 0 (the overworld) with a wooden sword and nearby cows. Stone is placed within a small radius around the player to encourage collection. One zombie is placed 4-8 tiles from the player's start. Iron ore is placed near the starting position to facilitate crafting an iron pickaxe. All player needs are enabled, and passive and melee mobs are enabled in case the starting ones despawn. Ranged mobs are disabled to focus on combat with basic tools.
    Relevant Achievements: MAKE_IRON_PICKAXE, ENTER_GNOMISH_MINES
    Completed Achievements: MAKE_WOOD_SWORD, COLLECT_STONE, PLACE_STONE
    World:
    - Player: Starts on floor 0 with a wooden sword (`{"sword": 1}`)
    - Map: 
      - One `ZOMBIE` (melee mob type_id=0) is placed randomly within 4-8 (Manhattan distance) tiles of the player.
      - Multiple `STONE` blocks are placed randomly within a 5x5 tile radius around the player's starting position.
      - Three `COW` (passive mob type_id=0) are placed randomly within 4-8 (Manhattan distance) tiles of the player.
      - Multiple `IRON` blocks are placed randomly within a 10x10 tile radius around the player's starting position.
    - Mechanics: 
      - "needs_depletion_multiplier = 1.0"
      - "passive_spawn_multiplier = 1.0"
      - "melee_spawn_multiplier = 1.0"
      - "ranged_spawn_multiplier = 0.0"
      - "health_recover_multiplier = 5.0"
    """

    def __init__(self, static_params: StaticEnvParams, params: EnvParams):
        super().__init__(static_params, params)
        self.relevant_achievements = [
            Achievement.MAKE_IRON_PICKAXE,
            Achievement.ENTER_GNOMISH_MINES,
        ]
        self.completed_achievements = [
            Achievement.MAKE_WOOD_SWORD,
            Achievement.COLLECT_STONE,
            Achievement.PLACE_STONE,
        ]
        self.label = "MAKE_IRON_PICKAXE_ENTER_GNOMISH_MINES"

    def get_task_params(self) -> TaskParams:
        """Return custom parameters for this task."""
        return TaskParams(
            passive_spawn_multiplier=1.0,  # Enable random cow spawns
            melee_spawn_multiplier=1.0,  # Enable zombie spawns
            ranged_spawn_multiplier=0.0,  # Disable skeleton spawns
            needs_depletion_multiplier=1.0,  # Needs are on with normal speed
            health_recover_multiplier=5.0,  # Health recovery is faster
        )

    def generate_world(self, rng: jax.Array) -> EnvState:
        """Generates the world for the task."""
        (
            rng,
            build_rng,
            placement_rng,
            zombie_rng,
            cow_rng,
            stone_rng,
            iron_rng,
        ) = jax.random.split(rng, 7)

        builder = WorldBuilder(build_rng, self.static_params, self.params)

        builder.set_starting_floor(0)

        # --- ADDED SCAFFOLDING ---
        # 1. Give prerequisite sword for safety
        builder.set_player_inventory({"sword": 1})

        # 2. Place cows as a food source
        builder.add_mobs_randomly_near(
            cow_rng,
            level=0,
            mob_name="passive",
            type_id=0,  # type_id 0 is Cow
            n=3,
            target_pos=builder.player_position,
            min_dist=4,
            max_dist=8,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        # 3. Place a zombie for combat practice
        builder.add_mobs_randomly_near(
            zombie_rng,
            level=0,
            mob_name="melee",
            type_id=0,  # type_id 0 is Zombie
            n=1,
            target_pos=builder.player_position,
            min_dist=4,
            max_dist=8,
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        # 4. Place multiple stone blocks near the player for collection
        builder.place_randomly_near(
            placement_rng,
            level=0,
            block_type=BlockType.STONE,
            target_pos=builder.player_position,
            min_dist=0,
            max_dist=5,
            n=25,  # Adjust number as needed
            on_blocks=[BlockType.GRASS, BlockType.PATH],
        )

        # 5. Place multiple iron blocks near the player for crafting
        builder.place_randomly_near(
            iron_rng,
            level=0,
            block_type=BlockType.IRON,
            target_pos=builder.player_position,
            min_dist=0,
            max_dist=10,
            n=25,  # Adjust number as needed
            on_blocks=[BlockType.STONE],
        )
        # --- END SCAFFOLDING ---

        return builder.build(rng)