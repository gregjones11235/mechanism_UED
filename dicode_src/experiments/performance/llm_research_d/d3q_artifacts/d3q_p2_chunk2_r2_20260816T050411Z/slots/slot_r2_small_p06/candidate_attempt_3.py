import jax
from craftax.craftax.constants import Achievement, BlockType, ItemType, MobType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
    """Objective: Navigate, survive, and collect resources in the Gnomish Mines.
    Description: The player must achieve `DEFEAT_ZOMBIE`, `DEFEAT_SKELETON`, `COLLECT_IRON`, `COLLECT_COAL` while navigating pitch-black environments in the Gnomish Mines. The world is set up with a pre-filled inventory to facilitate the agent's immediate focus on higher-tier tasks.
    Relevant Achievements: DEFEAT_ZOMBIE, DEFEAT_SKELETON, COLLECT_IRON, COLLECT_COAL
    Completed Achievements: ENTER_GNOMISH_MINES
    World:
    - Player: Starts on floor 2 with an inventory containing 10 torches and 1 wood pickaxe.
    - Map: 
      - Gnomish Mines (floor 2) is procedurally generated with darkness, coal deposits, iron ore veins, and a mix of hostile mobs like zombies and skeletons.
      - Ladder Down from floor 1 to floor 2 is placed at the starting position.
    - Mechanics: "needs_depletion_multiplier = 1.0", "passive_spawn_multiplier = 1.0", "melee_spawn_multiplier = 0.5", "ranged_spawn_multiplier = 0.5"
    """

    def __init__(self, static_params: StaticEnvParams, params: EnvParams):
        super().__init__(static_params, params)
        self.relevant_achievements = [
            Achievement.DEFEAT_ZOMBIE,
            Achievement.DEFEAT_SKELETON,
            Achievement.COLLECT_IRON,
            Achievement.COLLECT_COAL,
        ]
        self.completed_achievements = [Achievement.ENTER_GNOMISH_MINES]
        self.label = "DEFEAT_ZOMBIE, DEFEAT_SKELETON, COLLECT_IRON, COLLECT_COAL"

    def get_task_params(self) -> TaskParams:
        """Return custom parameters for this task."""
        return TaskParams(
            passive_spawn_multiplier=1.0,
            melee_spawn_multiplier=0.5,
            ranged_spawn_multiplier=0.5,
            needs_depletion_multiplier=1.0,
        )

    def generate_world(self, rng: jax.Array) -> EnvState:
        """Generates the world for the task."""
        rng, build_rng, placement_rng = jax.random.split(rng, 3)

        builder = WorldBuilder(build_rng, self.static_params, self.params)

        builder.set_starting_floor(2)
        
        # Set initial inventory
        builder.set_player_inventory({
            ItemType.TORCH.name: 10,
            ItemType.WOOD_PICKAXE.name: 1
        })

        # Place coal and iron randomly in the Gnomish Mines (floor 2)
        builder.place_randomly(
            placement_rng,
            level=2,
            block_type=BlockType.COAL,
            n=15,
            on_blocks=[BlockType.STONE],
        )
        builder.place_randomly(
            placement_rng,
            level=2,
            block_type=BlockType.IRON,
            n=10,
            on_blocks=[BlockType.STONE],
        )

        # Add zombies and skeletons randomly in the Gnomish Mines (floor 2)
        builder.add_mobs_randomly_near(
            rng,
            level=2,
            mob_name="melee",
            type_id=MobType.ZOMBIE.value,  # type_id for Zombie
            n=5,
            target_pos=builder.player_position,
            min_dist=5,
            max_dist=15,
            on_blocks=[BlockType.STONE],
        )
        builder.add_mobs_randomly_near(
            rng,
            level=2,
            mob_name="ranged",
            type_id=MobType.SKELETON.value,  # type_id for Skeleton
            n=3,
            target_pos=builder.player_position,
            min_dist=5,
            max_dist=15,
            on_blocks=[BlockType.STONE],
        )

        return builder.build(rng)