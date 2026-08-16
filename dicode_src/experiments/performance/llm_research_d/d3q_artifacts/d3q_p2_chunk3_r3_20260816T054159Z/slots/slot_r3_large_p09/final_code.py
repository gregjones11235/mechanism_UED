import jax
import jax.numpy as jnp

from craftax.craftax.constants import Achievement, BlockType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
	"""Objective: Improve the agent's ability to enter the Gnomish Mines by navigating through basic combat and collecting essential resources.
	Description: The player must achieve `ENTER_GNOMISH_MINES`. The player starts in the Overworld with a wooden sword, some basic resources, and essential blocks like stone and coal. Mobs are enabled to encourage focused learning on combat while preparing to enter the Gnomish Mines. This task is designed to build on the agent's existing skills by introducing the new dependency of entering this critical floor.

	Relevant Achievements: ENTER_GNOMISH_MINES
	Completed Achievements: MAKE_WOOD_SWORD, COLLECT_WOOD, PLACE_TABLE, MAKE_STONE_PICKAXE, MAKE_STONE_SWORD, COLLECT_COAL, COLLECT_IRON, PLACE_FURNACE

	World:
	- Player: Starts in the Overworld with a wooden sword (`{"sword": 1}`) and some basic resources (`{"wood": 20, "stone": 10, "coal": 5}`). 
	- Map: Default procedural Overworld (Floor 0). 5 `ZOMBIE` mobs (melee mob type_id=0) are placed within a 4-tile radius from the player. Additionally, 3 `SKELETON` mobs (ranged mob type_id=0) and 1 `COW` mob (passive mob type_id=1) are placed randomly within a 6-tile radius.
	- Mechanics: "needs_depletion_multiplier = 0.7", "passive_spawn_multiplier = 0.5", "melee_spawn_multiplier = 0.3", "ranged_spawn_multiplier = 0.2"
	"""

	def __init__(self, static_params: StaticEnvParams, params: EnvParams):
		super().__init__(static_params, params)

		self.relevant_achievements = [
			Achievement.ENTER_GNOMISH_MINES,
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
		return TaskParams(
			needs_depletion_multiplier=0.7,
			passive_spawn_multiplier=0.5,
			melee_spawn_multiplier=0.3,
			ranged_spawn_multiplier=0.2,
		)

	def generate_world(self, rng: jax.Array) -> EnvState:
		builder = WorldBuilder(rng, self.static_params, self.params)

		# Starting inventory
		builder.set_player_inventory({
			"sword": 1,
			"wood": 20,
			"stone": 10,
			"coal": 5,
		})

		player_pos = jnp.array(
			[
				self.static_params.map_size[0] // 2,
				self.static_params.map_size[1] // 2,
			],
			dtype=jnp.int32,
		)

		# Place hostile and passive mobs near the start
		rng, _rng = jax.random.split(rng)
		builder.add_mobs_randomly_near(
			rng=_rng,
			level=0,
			mob_name="melee",
			type_id=0,
			n=5,
			target_pos=player_pos,
			min_dist=0,
			max_dist=4,
			on_blocks=[BlockType.GRASS, BlockType.PATH, BlockType.SAND],
		)

		rng, _rng = jax.random.split(rng)
		builder.add_mobs_randomly_near(
			rng=_rng,
			level=0,
			mob_name="ranged",
			type_id=0,
			n=3,
			target_pos=player_pos,
			min_dist=0,
			max_dist=6,
			on_blocks=[BlockType.GRASS, BlockType.PATH, BlockType.SAND],
		)

		rng, _rng = jax.random.split(rng)
		builder.add_mobs_randomly_near(
			rng=_rng,
			level=0,
			mob_name="passive",
			type_id=1,
			n=1,
			target_pos=player_pos,
			min_dist=0,
			max_dist=6,
			on_blocks=[BlockType.GRASS, BlockType.PATH, BlockType.SAND],
		)

		state = builder.build(rng)

		# Mark the achievements that are considered already completed
		completed_ids = jnp.array([a.value for a in self.completed_achievements], dtype=jnp.int32)
		state = state.replace(
			achievements=state.achievements.at[completed_ids].set(True)
		)

		return state