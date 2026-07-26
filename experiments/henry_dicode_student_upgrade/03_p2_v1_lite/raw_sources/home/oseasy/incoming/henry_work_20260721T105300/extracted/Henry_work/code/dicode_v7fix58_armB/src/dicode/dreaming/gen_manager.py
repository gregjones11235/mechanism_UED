"""Task evolution and curriculum generation using LLM-based dreaming.

This module implements the core curriculum learning loop for DiCode, including:
- Task: Loading and managing individual task environments.
- TaskArchive: A graph-based archive for storing and querying tasks.
- TaskSelector: Strategies for selecting parent tasks for evolution.
- TaskGenerator: LLM-based generation of new task descriptions.
- EnvGenerator: LLM-based code generation with compilation validation.
- GenManager: The main orchestrator class for the evolution pipeline.
"""

# --- Standard Library ---
import copy
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import traceback
from textwrap import dedent

# --- Third-Party ---
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

# --- Local Modules ---
from dicode.dreaming.envgen_guards import (
    diff_world_specs,
    scan_banned_randomness,
    shape_mismatch_message,
)
from dicode.dreaming.llm import LLM
from dicode.dreaming.prompts.dicode.constants import context as CONSTANTS
from dicode.dreaming.prompts.dicode.minicraftax_api import context as API_DOCS
from dicode.dreaming.prompts.dicode.mobs import context as MOBS
from dicode.dreaming.prompts.dicode.mobs_code import context as MOBS_CODE
from dicode.dreaming.prompts.dicode.step_fn_nl import context as GAME_MECHANICS
from dicode.dreaming.prompts.dicode.world_gen_nl import context as WORLD_GEN
from dicode.dreaming.utils import distances_from_embeddings, smart_absolute_path
from minicraftax.envs.base import MiniCraftaxTrain

# Instruction for the embedding model to generate task embeddings.
EMBEDDING_INSTRUCTION = (
    "Generate an embedding for this Craftax task description to evaluate its "
    "conceptual similarity to other tasks. The embedding should capture the core "
    "gameplay loop, the primary skills the agent must use (e.g., navigation, "
    "crafting, combat), the overall strategic objective, and how the world is built."
)


# v7fix4.1 world-shape contract (see envgen_guards.py): the canonical shape/dtype
# template every generated world must match. Built once per process from a BLANK
# WorldBuilder under the default StaticEnvParams — exactly the params every task
# receives in training (the multitask env instantiates task_cls(StaticEnvParams(), ...)),
# and build() assembles the complete EnvState without touching world content.
_CANONICAL_WORLD_SPECS: dict | None = None


def _flatten_world_specs(shape_struct) -> dict:
    """Flattens a jax.eval_shape EnvState result into ``{path: (shape, dtype)}`` strings."""
    leaves = jax.tree_util.tree_flatten_with_path(shape_struct)[0]
    return {
        jax.tree_util.keystr(path): (str(tuple(leaf.shape)), str(leaf.dtype))
        for path, leaf in leaves
    }


def _canonical_world_specs() -> dict:
    global _CANONICAL_WORLD_SPECS
    if _CANONICAL_WORLD_SPECS is None:
        from minicraftax.world_builder import WorldBuilder

        def _blank_world(rng):
            return WorldBuilder(rng, StaticEnvParams(), EnvParams()).build(rng)

        # eval_shape is trace-only: no execution, no XLA compile.
        struct = jax.eval_shape(_blank_world, jax.random.PRNGKey(0))
        _CANONICAL_WORLD_SPECS = _flatten_world_specs(struct)
    return _CANONICAL_WORLD_SPECS


class Task:
    """Loads and wraps a task environment from a Python file.

    Attributes:
        path: Absolute path to the task's Python file.
        file: The raw source code of the task file.
        env: The wrapped MiniCraftaxTrain environment.
        task: The raw Env class instance.
        desc: The task's docstring description.
    """

    def __init__(self, path: str):
        """Initializes a Task by loading its environment from a file.

        Args:
            path: Absolute path to the task's Python file.
        """
        self.path = path

        with open(self.path) as file:
            self.file = file.read()

        self.env, self.task = self.load_env()
        doc = self.task.__doc__
        self.desc = dedent(doc).strip() if doc else ""

    def load_env(self) -> tuple:
        """Loads the environment class from the task file.

        Uses a unique module name based on the filename to ensure thread safety
        when loading multiple tasks in parallel (sys.modules is shared).

        Returns:
            A tuple of (MiniCraftaxTrain env, raw Env task instance).
        """
        module_name = os.path.splitext(os.path.basename(self.path))[0]
        spec = importlib.util.spec_from_file_location(module_name, self.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task = getattr(module, "Env")(
            static_params=StaticEnvParams(), params=EnvParams()
        )
        env = MiniCraftaxTrain(task=task)
        return env, task


class TaskArchive:
	"""Manages the NetworkX graph of tasks, which serves as the single source of
	truth for the curriculum. Handles loading, saving, and querying tasks.
	"""

	def __init__(self, config):
		"""Initializes the TaskArchive by loading an existing graph or creating a new one.

		Args:
		    config: The Hydra configuration object, used to find seed task paths.

		"""
		self.config = config
		self.graph, self.active_task_count = self.load_graph()
		self._lock = threading.Lock()

	def load_graph(self) -> tuple[nx.DiGraph, int]:
		"""Loads the task graph from a file. If no file exists, creates a new
		graph and populates it with the initial seed tasks from the config.

		Returns:
		    A tuple containing the (nx.DiGraph, active_task_count)

		"""
		graph_path = self.config.graph_path  # The standard file to save our graph

		if os.path.exists(graph_path):
			print(f"Loading existing task graph from {graph_path}...")
			g = nx.read_graphml(graph_path)

			active_count = 0
			for node, data in g.nodes(data=True):
				if "performance_history" in data:
					try:
						data["performance_history"] = json.loads(data["performance_history"])
					except (json.JSONDecodeError, TypeError):
						# If it fails, default to an empty list for safety.
						data["performance_history"] = []
				if (
					"is_active" not in data
					or data["is_active"] == "false"
					or data["is_active"] == False
				):
					data["is_active"] = False
				else:
					data["is_active"] = True
					active_count += 1  # Count loaded active tasks

				if "priority_score" not in data:
					data["priority_score"] = float(data.get("learnability_score", 0.0))
				else:
					data["priority_score"] = float(data["priority_score"])

				if "session_last_trained" not in data:
					data["session_last_trained"] = -1
				else:
					data["session_last_trained"] = int(data["session_last_trained"])

			print(f"    - Found {active_count} active tasks in loaded graph.")
			return g, active_count
		else:
			print("No graph file found. Creating new task graph from seed tasks...")
			g = nx.DiGraph()

			# Get the initial task paths from the hydra config
			for i, task_path in enumerate(self.config.example_paths):
				# Add each seed task as a node with initial attributes
				try:
					task = Task(smart_absolute_path(task_path))
					code = task.file
					desc = task.desc
					g.add_node(
						f"task_{i + 1}",
						status="seed",  # "seed" is a special type of success
						type="seed",
						description=desc,
						code=code,
						performance_history=[],
						session_created=0,  # Initialize empty list for metrics,
						is_active=False,
						priority_score=0.0,
						session_last_trained=-1,
					)
				except Exception as e:
					print(f"Warning: Could not load seed task {task_path}. Error: {e}")

			print(f"Created a new graph with {g.number_of_nodes()} seed tasks.")
			return g, 0

	def save_graph(self):
		"""Saves the graph, converting lists to JSON strings for GraphML compatibility."""
		graph_path = "task_graph.graphml"
		print(f"Saving task graph with {self.graph.number_of_nodes()} nodes to {graph_path}...")

		# Create a copy to avoid modifying the graph object that's currently in use.
		with self._lock:
			graph_to_save = copy.deepcopy(self.graph)

		# Loop through all nodes to find and convert JAX arrays
		for node, data in graph_to_save.nodes(data=True):
			if "performance_history" in data and isinstance(data["performance_history"], list):
				# --- NEW: Convert JAX arrays inside the history list ---
				converted_history = []
				for record in data["performance_history"]:
					converted_record = {}
					for key, value in record.items():
						# If the value is a JAX array, convert it to a float
						if isinstance(value, jax.Array):
							converted_record[key] = float(value)
						else:
							converted_record[key] = value
					converted_history.append(converted_record)

				# Now, serialize the cleaned list to a JSON string
				data["performance_history"] = json.dumps(converted_history)

		nx.write_graphml(graph_to_save, graph_path)

	def record_new_task(
		self, child_task: str, parent_tasks: list, description: str, session_id: int
	):
		"""Adds a new task and its parent edges to the graph."""
		with self._lock:
			if not self.graph.has_node(child_task):
				self.graph.add_node(
					child_task,
					status="desc_generated",  # New, more descriptive initial status
					type="generated",
					description=description,
					code="",  # Store code as an attribute
					performance_history=[],
					session_created=session_id,
					is_active=False,
					priority_score=0.0,
					session_last_trained=-1,
				)

			for parent_task in parent_tasks:
				if self.graph.has_node(parent_task):
					self.graph.add_edge(parent_task, child_task)
				else:
					print(f"Warning: Parent task {parent_task} not found in graph.")

	def set_level_meta(self, task_path: str, meta: dict):
		"""v6fix7 P0.1: persist machine-readable level metadata as node attributes.

		Only called when a <level_meta> block was actually parsed (siege sessions), so the baseline
		path never writes these attrs and its graphml stays byte-identical. None values are skipped
		(graphml cannot serialise None).
		"""
		with self._lock:
			if not self.graph.has_node(task_path):
				return
			node = self.graph.nodes[task_path]
			if meta.get("type"):
				node["level_type"] = str(meta["type"])
			if meta.get("drill_target"):
				node["drill_target"] = str(meta["drill_target"])
			if meta.get("siege_wall"):
				node["siege_wall"] = str(meta["siege_wall"])
			# v7: the declared spawn floor — the rung-reading filter keys off it (only levels at
			# the CURRENT rung's floor count as that rung's trained evidence). 0/None skipped:
			# natural-spawn levels keep the exact pre-v7 attribute set (graphml byte-parity).
			try:
				if int(meta.get("spawn_floor") or 0) > 0:
					node["spawn_floor"] = int(meta["spawn_floor"])
			except (TypeError, ValueError):
				pass
			# v7fix4 P2: system-built relay levels are flagged — the rung-reading filter accepts
			# ONLY these as a relay wall's trained evidence (an FM level attacking the same wall
			# must not feed the ladder: its authored world is not reality-anchored, and unpinned
			# fidelity axes are exactly how the v7fix3 ladder fake-graduated).
			if meta.get("system_built"):
				node["system_built"] = True
			# v7fix4.6: the scaffold sub-stage joins the rung-reading filter key (same-floor
			# levels from an EASIER stage must not fake-graduate the current stage — the fix9 #2
			# high-water family one dial further in). 0/None skipped: FULL rungs and every
			# pre-4.6 level keep their exact attribute set (graphml byte-parity).
			try:
				if int(meta.get("spawn_sub_stage") or 0) > 0:
					node["spawn_sub_stage"] = int(meta["spawn_sub_stage"])
			except (TypeError, ValueError):
				pass
			# v7fix5.3: the descent-regime knobs are persisted for forensics only (the reading
			# filter still keys off floor+sub_stage — the stage int uniquely determines both).
			# False/None skipped: non-regime levels keep their exact attribute set.
			if meta.get("spawn_uplock"):
				node["spawn_uplock"] = True
			try:
				_nm53 = meta.get("spawn_needs_multiplier")
				if _nm53 is not None and float(_nm53) < 1.0:
					node["spawn_needs_multiplier"] = float(_nm53)
			except (TypeError, ValueError):
				pass

	def update_node_status(self, task_path: str, status: str):
		"""Updates the status of a task node (e.g., 'success', 'boring', 'failed_compile').

		Args:
		    task_path: The path of the node to update.
		    status: The new status string.

		"""
		with self._lock:
			if self.graph.has_node(task_path):
				self.graph.nodes[task_path]["status"] = status
			else:
				print(f"Warning: Tried to update status for non-existent node {task_path}")

	def update_node_performance(self, session_idx: int, performance_data: dict):
		"""Updates the performance_history attribute for multiple tasks in the graph.

		Args:
		    session_idx: The current session number.
		    performance_data: A dict mapping task_path to its metrics, e.g.,
		                      {"path/to/task": {"success_rate": 0.8}}.

		"""
		with self._lock:
			for task_path, metrics in performance_data.items():
				if self.graph.has_node(task_path):
					# Ensure the performance_history list exists before appending
					if "performance_history" not in self.graph.nodes[task_path]:
						self.graph.nodes[task_path]["performance_history"] = []

					# Append the new performance record to the node's history
					self.graph.nodes[task_path]["performance_history"].append(
						{"session": session_idx, **metrics}
					)
				else:
					print(f"Warning: Tried to update performance for non-existent node {task_path}")

	def update_node_code(self, task_path: str, code: str):
		"""Sets the env_path attribute for a given task node."""
		with self._lock:
			if self.graph.has_node(task_path):
				self.graph.nodes[task_path]["code"] = code
			else:
				print(f"Warning: Tried to update code for non-existent node {task_path}")

	def get_tasks_by_status(self, statuses: list[str]) -> list[str]:
		"""Queries the graph and returns a list of task paths matching the given statuses.

		Args:
		    statuses: A list of status strings to filter by (e.g., ["seed", "success"]).

		Returns:
		    A list of task path strings (the node IDs).

		"""
		with self._lock:
			return [n for n, d in self.graph.nodes(data=True) if d.get("status") in statuses]

	def get_node_attributes(self, task_path: str) -> dict:
		"""Returns the full attribute dictionary for a given task node.
		Returns an empty dict if the node doesn't exist.
		"""
		with self._lock:
			if self.graph.has_node(task_path):
				return self.graph.nodes[task_path].copy()
			return {}

	def is_active(self, node: str) -> bool:
		"""Checks if a task is currently in the active training set."""
		with self._lock:
			if not self.graph.has_node(node):
				return False
			return self.graph.nodes[node].get("is_active", False)

	def get_task_descriptions(self, task_paths: list[str]) -> dict[str, str]:
		"""Efficiently retrieves the descriptions for a list of tasks from the
		graph's node attributes, avoiding repeated file I/O.

		Args:
		    task_paths: A list of task paths to get descriptions for.

		Returns:
		    A dictionary mapping each task path to its description string.

		"""
		with self._lock:
			descriptions = {}
			for path in task_paths:
				if self.graph.has_node(path):
					# .get("description", "") provides a default empty string if the attr is missing
					descriptions[path] = self.graph.nodes[path].get("description", "")
			return descriptions

	def get_task_codes(self, task_paths: list[str]) -> dict[str, str]:
		"""Efficiently retrieves the codes for a list of tasks from the
		graph's node attributes, avoiding repeated file I/O.

		Args:
		    task_paths: A list of task paths to get codes for.

		Returns:
		    A dictionary mapping each task path to its code string.

		"""
		with self._lock:
			codes = {}
			for path in task_paths:
				if self.graph.has_node(path):
					# .get("description", "") provides a default empty string if the attr is missing
					codes[path] = self.graph.nodes[path].get("code", "")
			return codes

	def get_max_session_idx(self) -> int:
		"""Scans the entire graph to find the highest session index recorded.

		This is useful for resuming an experiment to ensure the session counter
		continues from where it left off.

		Returns:
		    The highest session index found, or 0 if no sessions are recorded.

		"""
		with self._lock:
			max_idx = 0

			# Iterate through every node and its attributes in the graph
			for _, data in self.graph.nodes(data=True):
				# Check the session in which the task was created
				created_idx = data.get("session_created", 0)
				max_idx = max(max_idx, created_idx)

				# Check all session records in the task's performance history
				history = data.get("performance_history", [])
				for record in history:
					perf_idx = record.get("session", 0)
					max_idx = max(max_idx, perf_idx)

			return max_idx

	def get_evolutionary_path(self, task_id: str) -> list[dict]:
		"""Traces the direct ancestral path of a task from a root to the task itself.

		Args:
		    task_id: The ID of the node to trace back from.

		Returns:
		    A list of dictionaries, where each dictionary contains the details
		    of a task in the evolutionary path, ordered from oldest to newest.

		"""
		with self._lock:
			if not self.graph.has_node(task_id):
				return []

			path = []
			current_node_id = task_id

			# Traverse backwards until a node with no parents (a seed) is found
			while True:
				node_data = self.graph.nodes[current_node_id]

				# Extract the most recent performance record
				performance_history = node_data.get("performance_history", [])
				latest_sr = "N/A"
				if performance_history:
					# The history is appended, so the last entry is the most recent
					last_record = performance_history[-1]
					if "sr" in last_record and last_record["sr"] >= 0:
						latest_sr = f"{last_record['sr']:.2%}"  # Format as percentage

				path.append(
					{
						"id": current_node_id,
						"description": node_data.get("description", "No description available."),
						"status": node_data.get("status", "unknown"),
						"success_rate": latest_sr,
					}
				)

				# Get the predecessors of the current node
				predecessors = list(self.graph.predecessors(current_node_id))
				if not predecessors:
					break  # Stop if we've reached a root/seed node

				# In a simple evolution, we assume one parent. If multiple, we follow the first one.
				current_node_id = predecessors[0]

			# The path was built backwards, so we reverse it to get chronological order
			return path[::-1]

	def mark_as_re_evaluated(self, task_id: str):
		"""Sets a flag on a node to indicate it has been re-introduced for training."""
		with self._lock:
			if self.graph.has_node(task_id):
				# The 're_evaluated' attribute defaults to False if not present.
				self.graph.nodes[task_id]["re_evaluated"] = True
			else:
				print(f"Warning: Tried to mark non-existent node {task_id} as re-evaluated.")

	def update_node_learnability(self, task_id: str, score: float):
		"""Updates the learnability score for a task."""
		with self._lock:
			if self.graph.has_node(task_id):
				# Ensure score is a standard float, handle NaN/inf if necessary
				safe_score = float(score) if np.isfinite(score) else 0.0
				self.graph.nodes[task_id]["learnability_score"] = safe_score
			else:
				print(f"Warning: Tried to update learnability for non-existent node {task_id}")

	def remove_node(self, task_id: str):
		"""Removes a task node and its associated edges from the graph."""
		with self._lock:
			if self.graph.has_node(task_id):
				try:
					self.graph.remove_node(task_id)
					print(f"    - Node {task_id} removed from archive.")
				except Exception as e:
					print(f"    - Warning: Failed to remove node {task_id}. Error: {e}")
			else:
				print(f"    - Warning: Tried to remove non-existent node {task_id}.")

	def set_task_active_status(self, task_id: str, is_active: bool):
		"""Atomically sets the 'is_active' status of a task and updates the global counter."""
		with self._lock:
			if not self.graph.has_node(task_id):
				print(f"Warning: Tried to set active status for non-existent node {task_id}")
				return

			current_status = self.graph.nodes[task_id].get("is_active", False)

			if is_active and not current_status:
				# Activate: Set to True and increment counter
				self.graph.nodes[task_id]["is_active"] = True
				self.active_task_count += 1
			elif not is_active and current_status:
				# Deactivate: Set to False and decrement counter
				self.graph.nodes[task_id]["is_active"] = False
				self.active_task_count -= 1
				# Ensure count never goes below zero
				self.active_task_count = max(0, self.active_task_count)
			# else: no change needed (e.g., setting True when already True)

	def update_node_priority_score(self, task_id: str, score: float):
		"""Updates the priority_score for a task."""
		with self._lock:
			if self.graph.has_node(task_id):
				self.graph.nodes[task_id]["priority_score"] = float(score)
			else:
				print(f"Warning: Tried to update priority_score for non-existent node {task_id}")

	def update_node_session_last_trained(self, task_id: str, session_idx: int):
		"""Updates the session_last_trained for a task."""
		with self._lock:
			if self.graph.has_node(task_id):
				self.graph.nodes[task_id]["session_last_trained"] = int(session_idx)
			else:
				print(
					f"Warning: Tried to update session_last_trained for non-existent node {task_id}"
				)

	def update_node_reasoning(self, task_id: str, reasoning: str):
		"""Sets the reasoning attribute for a given task node."""
		with self._lock:
			if self.graph.has_node(task_id):
				self.graph.nodes[task_id]["reasoning"] = reasoning
			else:
				print(f"Warning: Tried to update reasoning for non-existent node {task_id}")


class TaskSelector:
	"""Handles the strategy for selecting which tasks to use as examples for generation.
	Provides methods to select based on task descriptions or full environment code.
	"""

	def __init__(self, archive: TaskArchive, embedding_model: LLM, config):
		"""Initializes the TaskSelector.

		Args:
			archive: The TaskArchive instance for accessing task data.
			embedding_model: LLM instance for generating embeddings.
			config: The Hydra configuration object.
		"""
		self.archive = archive
		self.embedding_model = embedding_model
		self.config = config
		self.example_usage_counts = self._initialize_example_usage_counts()

	def _initialize_example_usage_counts(self) -> dict[str, int]:
		"""Initializes the usage counts based on historical success, measured by
		the number of children (outgoing edges) each task has.
		"""
		print("Initializing example counts from historical success (graph edges)...")
		counts = {}
		for node in self.archive.graph.nodes():
			# The count is the number of successful children
			counts[node] = self.archive.graph.out_degree(node)
		return counts

	def select_similar_desc_tasks(
		self, pivot_task: str, statuses: list[str], num_examples: int
	) -> list[str]:
		"""Selects tasks similar to a pivot based on description embeddings.

		Args:
			pivot_task: The task ID to find similar tasks for.
			statuses: List of valid task statuses to consider.
			num_examples: Number of similar tasks to return.

		Returns:
			A list of task IDs ordered by similarity to the pivot.
		"""
		candidate_tasks = self.archive.get_tasks_by_status(statuses=statuses)
		if not candidate_tasks:
			print("Warning: No successful tasks in archive to select description examples from.")
			return []
		similar_tasks = self._order_similar_tasks(pivot_task, candidate_tasks)
		similar_tasks = similar_tasks[:num_examples]

		self._update_usage_counts(similar_tasks)
		return similar_tasks

	def select_pivot_task(
		self, statuses: list[str], not_use_tasks: list[str], sampling_method: str
	) -> str:
		"""Selects a pivot task for evolution using the specified sampling method.

		Args:
			statuses: List of valid task statuses to consider.
			not_use_tasks: List of task IDs to exclude from selection.
			sampling_method: Either 'frequency' (inverse frequency sampling) or 'random'.

		Returns:
			A task ID string, or an empty list if no candidates available.
		"""
		candidate_tasks = self.archive.get_tasks_by_status(statuses=statuses)
		candidate_tasks = [task for task in candidate_tasks if task not in not_use_tasks]
		if not candidate_tasks:
			print("Warning: No successful tasks in archive to select description examples from.")
			return []
		if sampling_method == "frequency":
			counts = np.array([self.example_usage_counts.get(path, 0) for path in candidate_tasks])
			inv_counts = 1.0 / (counts + 1)
			probabilities = inv_counts / inv_counts.sum()
			return np.random.choice(candidate_tasks, p=probabilities)
		elif sampling_method == "random":
			return np.random.choice(candidate_tasks)
		else:
			raise ValueError(f"Invalid sampling method: {sampling_method}")

	def _update_usage_counts(self, selected_tasks: list[str]):
		"""Updates usage counts for a list of selected tasks."""
		for task in selected_tasks:
			self.example_usage_counts[task] = self.example_usage_counts.get(task, 0) + 1

	def _order_similar_tasks(self, pivot_task: str, other_tasks: list[str]) -> list[str]:
		"""Orders a list of tasks based on their task description similarity to a pivot task description."""
		# Get descriptions directly from the graph attributes
		contents = self.archive.get_task_descriptions(other_tasks)
		pivot_content = contents.get(pivot_task, "")

		# Filter to only include tasks for which we found content
		valid_tasks = list(contents.keys())
		valid_contents = list(contents.values())

		# Get embeddings
		results = self.embedding_model.get_embedding(
			[pivot_content] + valid_contents, EMBEDDING_INSTRUCTION
		)

		pivot_vector = results[0]["embedding"]
		other_vectors = [result["embedding"] for result in results[1:]]

		# Calculate similarity and find top N
		similarities = distances_from_embeddings(
			pivot_vector, other_vectors, distance_metric="cosine"
		)
		sorted_indices = np.array(similarities).argsort()

		return [valid_tasks[i] for i in sorted_indices]


class TaskGenerator:
	"""Handles the creative process of generating new task descriptions using an LLM,
	based on selected examples and agent performance feedback.
	"""

	def __init__(
		self,
		task_generator_llm: LLM,
		archive: TaskArchive,
		selector: TaskSelector,
		config,
		proposer_llms: list[LLM] | None = None,
		modeler_llm: LLM | None = None,
		scientist_llm: LLM | None = None,
	):
		"""Initializes the TaskGenerator.

		Args:
		    task_generator_llm: An instance of the LLM class for generation.
		    archive: An instance of the TaskArchive.
		    config: The Hydra configuration object.
		    proposer_llms: Optional list of N heterogeneous Proposer LLMs for the auction
		        (multi-FM) method. If None, behaviour is the unchanged single-FM DiCode baseline.
		    modeler_llm: Optional GLM LLM for the v5-debate MODELER (cooperative-fill method). If None,
		        the modeler path is off and behaviour is the unchanged v4 auction / baseline.
		    scientist_llm: v7fix5.5 P2 — the second, statically think-ON instance of the modeler
		        LLM for the scientist pass (probe report -> ROOT-CAUSE HYPOTHESIS). None = the
		        hypothesis loop stays dormant (probe tool alone, batch-2 behaviour).

		"""
		self.llm = task_generator_llm
		# For the auction method: N Proposers. Defaults to [self.llm] so N=1 == baseline.
		self.proposer_llms = proposer_llms if proposer_llms else [task_generator_llm]
		self.archive = archive
		self.selector = selector
		self.config = config
		# v5-debate: the modeler is built lazily on first use (needs the per-run StudentProfileLog).
		self.modeler_llm = modeler_llm
		self.scientist_llm = scientist_llm
		self._modeler = None  # auction.modeler.Modeler, lazily constructed in evolve_mastered_coop
		self._profile_log = None  # auction.student_profile_log.StudentProfileLog, lazily constructed
		self._siege_notebook = None  # auction.siege_notebook.SiegeNotebook (v6), lazy iff config.siege
		self._cooc_log = None  # auction.cooccurrence_log.CooccurrenceLog (v6 §3.8 c), lazy iff config.siege
		self._behav_log = None  # auction.behavior_fingerprint_log.BehaviorFingerprintLog (v6 problem-2), lazy iff config.siege
		self._chain_log = None  # auction.chain_order_log.ChainOrderLog (v6fix7 P2), lazy iff config.siege
		# Rotating turn order for the cooperative sequential-fill method (advances once per session).
		self._coop_turn_offset = 0
		if config.mode != "reward":
			self.evolve_mastered_prompt = importlib.import_module(
				self.config.prompts.evolve_mastered
			)
		else:
			self.evolve_mastered_prompt = importlib.import_module(
				self.config.prompts.evolve_mastered_r
			)

		self.ablation_prompt = importlib.import_module(
			self.config.prompts.ablation
		)

		# v2 auction: per-proposer persona prompts, paired to self.proposer_llms BY INDEX
		# (config.personas is a separate list decoupled from config.proposers). Each entry maps
		# persona name "foo" -> prompt module config.prompts["persona_foo"]. Absent / shorter than
		# proposer_llms -> those proposers fall back to the shared evolve_mastered_prompt (baseline
		# persona-less behaviour, so N=1 or no-personas == unchanged).
		self.persona_prompts = self._load_persona_prompts()

		self.task_num_counter = self._initialize_task_counter()

	def _load_persona_prompts(self) -> list | None:
		"""Load the persona prompt module for each proposer (by index), or None if unconfigured.

		Returns a list aligned with self.proposer_llms: entry i is the imported prompt module for
		proposer i's persona, or self.evolve_mastered_prompt if proposer i has no persona. Returns
		None entirely if config.personas is absent (pure baseline / non-persona auction).
		"""
		personas_cfg = self.config.get("personas", None)
		if not personas_cfg:
			return None
		prompts_map = self.config.prompts
		modules = []
		for i in range(len(self.proposer_llms)):
			if i < len(personas_cfg) and personas_cfg[i]:
				key = f"persona_{personas_cfg[i]}"
				module_path = prompts_map.get(key, None)
				if module_path is None:
					raise KeyError(
						f"persona '{personas_cfg[i]}' (proposer {i}) has no prompt module "
						f"'{key}' in config.prompts. Add it to conf/gen_manager/default.yaml."
					)
				modules.append(importlib.import_module(module_path))
			else:
				# No persona for this proposer -> shared baseline prompt.
				modules.append(self.evolve_mastered_prompt)
		print(f"[auction] Loaded personas: {list(personas_cfg)} (paired to proposers by index).")
		return modules

	def _initialize_task_counter(self) -> int:
		"""Finds the highest task number from the archive to avoid overwriting."""
		max_num = -1
		task_paths = [n for n, d in self.archive.graph.nodes(data=True)]
		for path in task_paths:
			match = re.search(r"task_(\d+)", path)
			if match:
				max_num = max(max_num, int(match.group(1)))
		return max_num + 1

	def evolve_ablation(
		self, session_idx: int, mastered_tasks: list[str], global_agent_profile: dict | None = None
	) -> list[dict]:
		"""Generates new task descriptions for ablation experiments.

		This is a simplified version of `evolve_mastered` that uses a fixed prompt
		without agent performance context. Used for ablation studies where we
		want to remove the influence of performance-guided evolution.

		Args:
			session_idx: Current curriculum session index.
			mastered_tasks: List of task IDs the agent has mastered.
			global_agent_profile: Unused in ablation; kept for API compatibility.

		Returns:
			A list of dictionaries containing the generated task data.
		"""
		print(f"Generating {len(mastered_tasks)} new task descriptions (ablation mode)...")

		user_prompts = []
		parent_sets = []
		example_sets = []

		for mastered_task in mastered_tasks:
			task_examples = self.selector.select_similar_desc_tasks(
				mastered_task,
				statuses=self._get_valid_parent_statuses(),
				num_examples=self.config.num_examples,
			)
			parent_sets.append([mastered_task])
			example_sets.append(task_examples)
			# NOTE: Ablation mode uses an empty format() call (no context variables)
			user_prompts.append(self.ablation_prompt.user_prompt.format())

		if not user_prompts:
			print("Could not generate any prompts. Skipping task generation.")
			return []

		system_prompt = self._build_system_prompt(self.ablation_prompt)
		parsed_responses = self._query_and_parse_responses(system_prompt, user_prompts)

		return self._organize_data(
			parsed_responses, parent_sets, example_sets, session_idx, "mastered"
		)

	def evolve_mastered(
		self, session_idx: int, mastered_tasks: list[str], global_agent_profile: dict | None = None
	) -> list[dict]:
		"""Generates new task descriptions by evolving from mastered tasks.

		This is the main evolution method that creates new curriculum tasks based on
		previously mastered tasks and the agent's current performance profile.

		Args:
			session_idx: Current curriculum session index.
			mastered_tasks: List of task IDs the agent has mastered.
			global_agent_profile: Dictionary of agent skill metrics from evaluation.

		Returns:
			A list of dictionaries containing the generated task data.
		"""
		print(f"Generating {len(mastered_tasks)} new task descriptions...")

		user_prompts = []
		parent_sets = []
		example_sets = []

		global_profile_str = self._format_global_agent_profile(global_agent_profile)

		for mastered_task in mastered_tasks:
			task_examples = self.selector.select_similar_desc_tasks(
				mastered_task,
				statuses=self._get_valid_parent_statuses(),
				num_examples=self.config.num_examples,
			)

			parent_sets.append([mastered_task])
			example_sets.append(task_examples)

			# Build the user prompt with performance context
			example_str = self._format_file_mastered_task([mastered_task])
			task_performance_str = self._get_task_performance_str(mastered_task)

			if self.config.mode != "reward":
				user_prompts.append(
					self.evolve_mastered_prompt.user_prompt.format(
						MASTERED_TASK=example_str,
						TASK_PERFORMANCE_CONTEXT=task_performance_str,
						GLOBAL_AGENT_PROFILE=global_profile_str,
					)
				)
			else:
				user_prompts.append(
					self.evolve_mastered_prompt.user_prompt.format(
						MASTERED_TASK=example_str,
						GLOBAL_AGENT_PROFILE=global_profile_str,
					)
				)

		if not user_prompts:
			print("Could not generate any prompts. Skipping task generation.")
			return []

		system_prompt = self._build_system_prompt(self.evolve_mastered_prompt)
		parsed_responses = self._query_and_parse_responses(system_prompt, user_prompts)

		return self._organize_data(
			parsed_responses, parent_sets, example_sets, session_idx, "mastered"
		)

	def evolve_mastered_auction(
		self,
		session_idx: int,
		mastered_tasks: list[str],
		global_agent_profile: dict | None = None,
		*,
		k: int | None = None,
	) -> list[dict]:
		"""Auction variant of evolve_mastered: N Proposers each dream a description per parent,
		then an auction selects the complementary top-k (v1_experiment.md §7, 方法设计_v1 §2).

		Reuses the exact prompt construction of evolve_mastered (so the per-parent context is
		identical to baseline). The only change is the ②description layer: single FM -> N FMs +
		top-k auction. With a single proposer and k=None this reduces to baseline behaviour.

		Args:
			session_idx / mastered_tasks / global_agent_profile: same as evolve_mastered.
			k: how many descriptions to keep after the auction. Defaults to len(mastered_tasks)
			   (i.e. produce as many winners as baseline produces descriptions, for fair budget).
		"""
		# Lazy import keeps the auction dependency out of the baseline path / module import.
		from auction.selectors import GreedyTopKSelector, SelectionContext
		from .auction_integration import parsed_response_to_proposal, profile_to_target_gap

		print(
			f"[auction] Generating descriptions for {len(mastered_tasks)} parents "
			f"x {len(self.proposer_llms)} proposers..."
		)

		# Archive family tally is computed once and fed to whichever proposer is the Breadth
		# persona (its template references {ARCHIVE_FAMILY_COVERAGE}; others ignore it).
		archive_family_coverage = self._compute_archive_family_coverage()

		# Stage A: each Proposer independently dreams a description for every parent, USING ITS OWN
		# persona prompt (system + user). self.persona_prompts[i] is proposer i's persona module,
		# or None (-> shared baseline prompt for all, i.e. unchanged behaviour).
		proposals = []
		# parallel bookkeeping so a winning proposal maps back to its parent/example set.
		parent_of: dict[str, list[str]] = {}
		example_of: dict[str, list[str]] = {}
		parsed_of: dict[str, dict] = {}
		parent_sets: list[list[str]] = []
		example_sets: list[list[str]] = []
		pid_counter = 0
		for proposer_idx, proposer in enumerate(self.proposer_llms):
			# Pick this proposer's persona prompt module (fall back to baseline if no personas).
			if self.persona_prompts is not None:
				module = self.persona_prompts[proposer_idx]
			else:
				module = self.evolve_mastered_prompt

			# Build per-persona prompts: system from the persona module, user from the same module
			# (Breadth's user template additionally consumes the archive family tally).
			system_prompt = self._build_system_prompt(module)
			user_prompts, p_sets, e_sets = self._build_mastered_prompts(
				mastered_tasks,
				global_agent_profile,
				prompt_module=module,
				archive_family_coverage=archive_family_coverage,
			)
			if not user_prompts:
				print("Could not generate any prompts. Skipping task generation.")
				return []
			# parent/example sets are identical across proposers (same parents); keep the first.
			if not parent_sets:
				parent_sets, example_sets = p_sets, e_sets

			# Temporarily route _query_and_parse_responses through this proposer.
			prev_llm = self.llm
			self.llm = proposer
			try:
				parsed_responses = self._query_and_parse_responses(system_prompt, user_prompts)
			finally:
				self.llm = prev_llm

			# parsed_responses align with the (surviving) user_prompts order.
			for local_i, parsed in enumerate(parsed_responses):
				pid = f"prop_s{session_idx}_{pid_counter}"
				pid_counter += 1
				parent_set = parent_sets[local_i] if local_i < len(parent_sets) else []
				proposal = parsed_response_to_proposal(
					parsed,
					proposal_id=pid,
					proposer_id=f"proposer_{proposer_idx}",
					parent_task_id=parent_set[0] if parent_set else "",
				)
				proposals.append(proposal)
				parent_of[pid] = parent_set
				example_of[pid] = example_sets[local_i] if local_i < len(example_sets) else []
				parsed_of[pid] = parsed

		if not proposals:
			print("[auction] No proposals produced. Skipping.")
			return []

		# Stage B: bid signal sources.
		#  - AmbitionGain: target_gap from the student's current skill profile (no extra LLM calls).
		#  - Endorsement: cross-ratings from one round of Proposers rating each other's proposals.
		#  - Learnability: each candidate proxies its PARENT task's stored learnability (official
		#    DiCode p*(1-p), read from the archive; no extra rollout — auction_integration note).
		target_gap = profile_to_target_gap(global_agent_profile)
		cross_ratings = None
		if len(self.proposer_llms) > 1 and getattr(self.config, "auction_endorsement", True):
			cross_ratings = self._run_cross_rating(proposals, global_agent_profile)
		parent_learnability = self._build_parent_learnability(proposals)

		# Ability-gate (2026-07-02, v1_experiment.md §10.9): compute the deepest tier the student may
		# be pushed toward NOW from its tier mastery. AmbitionGain then soft-discounts gap on tiers
		# beyond it, so ambitious can't out-bid feasible on tiers the student can't reach. Gated only
		# when there is a target_gap (a student profile) AND the gate is enabled (default on).
		from auction.craftax_achievements import reachable_ceiling as _reachable_ceiling

		ability_gate_on = bool(getattr(self.config, "auction_ability_gate", True))
		mastery_threshold = float(getattr(self.config, "auction_mastery_threshold", 0.60))
		overreach_decay = float(getattr(self.config, "auction_overreach_decay", 0.4))
		ceiling = (
			_reachable_ceiling(target_gap, threshold=mastery_threshold)
			if (ability_gate_on and target_gap)
			else None
		)

		context = SelectionContext(
			target_gap=target_gap or None,
			cross_ratings=cross_ratings,
			parent_learnability=parent_learnability or None,
			w_cov=float(getattr(self.config, "auction_w_cov", 1.0)),
			w_end=float(getattr(self.config, "auction_w_end", 1.0)),
			w_amb=float(getattr(self.config, "auction_w_amb", 1.0)),
			w_lrn=float(getattr(self.config, "auction_w_lrn", 1.0)),
			reachable_ceiling=ceiling,
			overreach_decay=overreach_decay,
		)
		if ceiling is not None:
			print(
				f"[auction][gate] session={session_idx} reachable_ceiling=tier{ceiling} "
				f"(mastery_threshold={mastery_threshold}, overreach_decay={overreach_decay}); "
				f"ambition gap on tiers >{ceiling} soft-discounted."
			)

		# Stage C: auction selects complementary top-k (greedy submodular: Coverage + endorsement
		# + ambition + learnability).
		if k is None:
			k = len(mastered_tasks)
		selector = GreedyTopKSelector()
		winners = selector.select(proposals, k, context)
		print(
			f"[auction] Selected {len(winners)}/{len(proposals)} proposals (k={k}). "
			f"bid weights cov/end/amb/lrn={context.w_cov}/{context.w_end}/{context.w_amb}/{context.w_lrn}, "
			f"endorsement={'on' if cross_ratings else 'off'}, "
			f"ambition_skills={len(target_gap)}, learnable_parents={len(parent_learnability)}."
		)

		# --- Detailed bid-voice logging (方法设计_v2.md §3.5: watch for a term drowned/dominating).
		# Pure reporting; does not affect selection. Reports WEIGHTED contribution + share per term,
		# per winner AND aggregated, plus which persona/proposer got selected.
		try:
			from auction.selectors import bid_breakdown

			bd = bid_breakdown(winners, context, all_proposals=proposals)
			av = bd["avg_share"]
			to = bd["totals"]
			print(
				f"[auction][voice] session={session_idx} avg per-winner share "
				f"cov={av['cov']:.1%} end={av['end']:.1%} amb={av['amb']:.1%} lrn={av['lrn']:.1%} "
				f"| totals cov={to['cov']:.3f} end={to['end']:.3f} amb={to['amb']:.3f} lrn={to['lrn']:.3f} "
				f"| by_proposer={bd['by_proposer']}"
			)
			# Flag pathologies: a term contributing <5% (drowned) or >70% (dominating) on average.
			for term in ("cov", "end", "amb", "lrn"):
				s = av[term]
				if s < 0.05:
					print(f"[auction][voice][WARN] '{term}' share {s:.1%} < 5% — near-drowned (check w_{term}).")
				elif s > 0.70:
					print(f"[auction][voice][WARN] '{term}' share {s:.1%} > 70% — dominating (check w_{term}).")
			# Per-winner one-liners (compact; proposer + parent + weighted terms + shares).
			for w in bd["per_winner"]:
				sh = w["shares"]
				print(
					f"[auction][voice]   {w['proposal_id']} <{w['proposer_id']}> parent={w['parent_task_id']} "
					f"cov={w['cov']:.3f}({sh['cov']:.0%}) end={w['end']:.3f}({sh['end']:.0%}) "
					f"amb={w['amb']:.3f}({sh['amb']:.0%}) lrn={w['lrn']:.3f}({sh['lrn']:.0%}) total={w['total']:.3f}"
				)
			# FULL candidate pool (winners + LOSERS), for offline auction replay / counterfactual
			# bid analysis. Each row = a candidate ENTRY bid (coverage vs empty set, same normalization
			# as selection), tagged sel/rej. Winner-only logging lacked this: with the whole 36-candidate
			# pool we can re-rank under a different ambition discount and see which picks change (2026-07-02).
			from auction.selectors import pool_breakdown

			pool = pool_breakdown(winners, proposals, context)
			print(f"[auction][pool] session={session_idx} n_candidates={len(pool)} n_selected={len(winners)}")
			for r in pool:
				flag = "sel" if r["selected"] else "rej"
				print(
					f"[auction][pool]   [{flag}] {r['proposal_id']} <{r['proposer_id']}> "
					f"parent={r['parent_task_id']} "
					f"cov={r['cov']:.3f} end={r['end']:.3f} amb={r['amb']:.3f} lrn={r['lrn']:.3f} "
					f"total={r['total']:.3f}"
				)
		except Exception as e:  # logging must never break the run
			# This block is PURE logging (the real _organize_data return below is outside the try, so a
			# failure here can't corrupt training) — swallowing is correct. But make it LOUD: a silently-
			# vanished voice log (e.g. bid_breakdown/pool_breakdown field renamed upstream) would otherwise
			# read as "auction ran fine, just quiet". Surface the type so a broken breakdown is diagnosable.
			import traceback
			print(
				f"[auction][voice] WARNING: breakdown logging failed (non-fatal, training unaffected) "
				f"({type(e).__name__}: {e}). Voice diagnostics missing this session. Traceback:"
			)
			traceback.print_exc()

		# Rebuild the (parsed, parent, example) triplets for the winners and organize as usual.
		win_parsed = [parsed_of[w.proposal_id] for w in winners]
		win_parents = [parent_of[w.proposal_id] for w in winners]
		win_examples = [example_of[w.proposal_id] for w in winners]
		return self._organize_data(
			win_parsed, win_parents, win_examples, session_idx, "mastered"
		)

	def _ensure_modeler(self):
		"""Lazily build the StudentProfileLog + Modeler (v5-debate). Returns the Modeler or None.

		v6 (§3.5): when the siege switch is on, also build the persistent SiegeNotebook (like the
		profile log, it lives in the run cwd and survives resume). Purely additive; with siege off the
		notebook is never created and the v5 coop path is byte-for-byte unchanged.
		"""
		if self.modeler_llm is None:
			return None
		if self._modeler is None:
			from auction.modeler import Modeler
			from auction.student_profile_log import StudentProfileLog

			recent_k = int(getattr(self.config, "modeler_recent_k", 6))
			self._profile_log = StudentProfileLog()
			self._modeler = Modeler(
				self.modeler_llm, self.archive, self._profile_log, recent_k=recent_k,
				scientist_llm=self.scientist_llm,
			)
			print("[modeler] Modeler + StudentProfileLog initialised (GLM diagnostic agent).")
			if bool(self.config.get("siege", False)) and self._siege_notebook is None:
				from auction.siege_notebook import SiegeNotebook, SiegeThresholds

				# B-layer thresholds are config-tunable (user 2026-07-05): read the siege_* keys off the
				# gen_manager config; any missing key keeps its documented default.
				thresholds = SiegeThresholds.from_config(self.config)
				self._siege_notebook = SiegeNotebook(thresholds=thresholds)
				th = self._siege_notebook.th
				print(
					f"[siege] SiegeNotebook initialised (persistent siege journal); "
					f"active foci = {self._siege_notebook.focus_skills()}. thresholds: "
					f"mastered={th.mastered_sr} unmastered={th.unmastered_sr} "
					f"saturated={th.saturated_sr} record_delta_pp={th.record_delta_pp} "
					f"maturity(min_snap={th.maturity_min_snapshots},min_mastered={th.maturity_min_mastered},"
					f"skill_sr={th.maturity_skill_sr}) "
					f"focus(improve_pp={th.focus_improve_pp},"
					f"max={th.max_focus},expand_sr={th.focus_expand_sr})."
				)
				# v6 §3.8 (c): the cross-session co-occurrence log, fed by the held-out eval
				# (online_evaluation) and read by the siege modeler to ground prereq chains in real
				# trajectories. Lazily built alongside the notebook; absent when siege is off.
				from auction.cooccurrence_log import CooccurrenceLog

				self._cooc_log = CooccurrenceLog()
				print("[siege] CooccurrenceLog initialised (cross-session (c) co-occurrence).")
				# v6 problem-2: the behaviour fingerprint log — HOW the student behaves in the episodes
				# it wins (action mix / pacing), fed by the same held-out eval and read by the modeler to
				# ground the style_note in real actions instead of imagined tactics. Lazy alongside (c).
				from auction.behavior_fingerprint_log import BehaviorFingerprintLog

				self._behav_log = BehaviorFingerprintLog()
				print("[siege] BehaviorFingerprintLog initialised (problem-2 winning-episode behaviour).")
				# v6fix7 P2: directed chain order + break-link mining — the temporal upgrade of (c).
				# Fed by the same held-out eval (run_session_evaluation), read by the siege modeler
				# (CHAIN EVIDENCE) and by the P1a patience/blacklist machinery (frontier advance).
				from auction.chain_order_log import ChainOrderLog

				self._chain_log = ChainOrderLog()
				print("[siege] ChainOrderLog initialised (P2 directed chains + break-link mining).")
		return self._modeler

	def evolve_mastered_coop(
		self,
		session_idx: int,
		mastered_tasks: list[str],
		global_agent_profile: dict | None = None,
	) -> list[dict]:
		"""v5-debate COOPERATIVE-FILL method (v5_design.md 方案A): modeler diagnosis + sequential fill.

		Flow per session:
		  1. Record the student's current held-out profile into the time-series log.
		  2. MODELER (GLM) diagnoses the student's current state + recommends a level TYPE per parent.
		  3. For EACH parent, the proposers write levels IN TURN (order rotates each session). The
		     second proposer sees what the first already made ({PEER_ALREADY_MADE}) and covers a
		     different valuable TYPE — cooperative, not competitive. BOTH levels are kept (no auction
		     culling, like baseline). With a single proposer this is just "modeler-guided ambitious".

		Returns the same _organize_data structure as evolve_mastered / _auction, so downstream code
		(code generation, injection) is unchanged.
		"""
		from .auction_integration import parsed_response_to_proposal  # noqa: F401 (parity import)

		modeler = self._ensure_modeler()
		# 1. record profile snapshot for the time series (idempotent per session).
		if self._profile_log is not None:
			self._profile_log.record(session_idx, global_agent_profile)

		# 2. modeler diagnosis (once per session). parent_context = short parent descriptions.
		parent_context = {}
		for pid in mastered_tasks:
			detail = self._modeler.view.level_detail(pid) if modeler else {}
			parent_context[pid] = detail.get("description", "") or ""

		# v6 SIEGE (§3.2/§3.5): if the siege notebook is active, the modeler reads its previous page,
		# proposes an update, and we fold that update through the notebook's B-layer rules. The
		# resulting focus + still-unmastered links drive the SIEGE_DIRECTIVE and the §3.4 gate below.
		siege_active = self._siege_notebook is not None and modeler is not None
		siege_directive_text = ""
		siege_unmastered: set[str] = set()
		if siege_active:
			latest_profile = self._profile_log.latest() if self._profile_log else {}
			num_snapshots = len(self._profile_log.recent(10_000)) if self._profile_log else 0
			combat_targets = self._combat_target_names()
			cooc_hint = self._render_cooccurrence_hint()
			# v6fix7 P2: directed CHAIN EVIDENCE (order + break link) — appended to the (c) hint so
			# the modeler sees it every siege session without a diagnose_siege schema change.
			chain_hint = self._render_chain_hint()
			if chain_hint:
				cooc_hint = (cooc_hint + "\n" + chain_hint) if cooc_hint else chain_hint
			behav_hint = self._render_behavior_hint()  # v6 problem-2: winning-episode action fingerprint
			# v6fix7 P1b: drill-transfer GAP signal (SCALAR's train-eval gap, repurposed as the
			# drill "graduation" trigger) — appended to behav_hint so the modeler sees it every
			# siege session without a schema change.
			gap_hint = self._render_siege_gap_hint(latest_profile, session_idx=session_idx)
			# v7fix3 P6: advance the breadth spawn frontier from the freshest deep-spawn BREADTH
			# training readings (before the ecology directive renders, so it shows the new bound).
			self._note_breadth_frontier_readings(session_idx=session_idx)
			if gap_hint:
				behav_hint = (behav_hint + "\n" + gap_hint) if behav_hint else gap_hint
			# v6fix9 P2: the machine-readable forensic summaries the attribution gate cross-checks
			# the modeler's causal claims against (same walls the CHAIN EVIDENCE block renders).
			forensics: dict[str, dict] = {}
			chain_incomplete: set[str] = set()
			chain_log_obj = getattr(self, "_chain_log", None)
			# v6fix10.1 hazard-3a: the notebook's admission-deferral gate ("no forensics -> chain-
			# track one session before opening") must run ONLY when a chain log exists — the apply
			# call below passes forensics=None otherwise, telling the gate to stand down (else every
			# proposal would defer forever waiting for a tracker that does not exist).
			forensics_available = chain_log_obj is not None and hasattr(chain_log_obj, "forensics")
			if forensics_available:
				for wall in self._siege_notebook.chain_targets().keys():
					fx = chain_log_obj.forensics(wall)
					if fx:
						forensics[wall] = fx
					# v6fix10 ⑦: complete-in-failures + zero wins = the reported chain misses a
					# prerequisite -> the wall is not attackable until the LLM expands the chain.
					if hasattr(chain_log_obj, "chain_incomplete") and \
							chain_log_obj.chain_incomplete(wall):
						chain_incomplete.add(wall)
				# v7fix5.0 P1/P2: feed this session's access-cap verdicts to the notebook — they
				# drive the gap-gate ACCESS_CAPPED park, the watch-resume hold, and the
				# expand-gate frontier exemption. Wholesale overwrite each session (a cap is a
				# reading, not a ratchet), greppable as [siege][access].
				if hasattr(self._siege_notebook, "note_access_caps"):
					_caps = {
						w: fx["access"] for w, fx in forensics.items()
						if isinstance(fx.get("access"), dict) and fx["access"].get("frontier")
					}
					self._siege_notebook.note_access_caps(_caps, session_idx=session_idx)
					for _w, _c in _caps.items():
						print(
							f"[siege][access] wall={_w}: frontier={_c.get('frontier')} "
							f"reach={float(_c.get('reach_frac', 0.0)) * 100:.1f}% "
							f"cond={float(_c.get('cond', 0.0)) * 100:.1f}% "
							f"certified={bool(_c.get('certified'))}"
						)
			# v7fix5.5 P2: the hypothesis loop runs BEFORE the big bookkeeping call so the journal
			# it reads already carries this session's verdicts/hypotheses. Order: housekeeping
			# first (verdicts for delivered Tier-2 verify reports — >= bar compiles an inserted
			# rung, else REFUTED feeds back; plus retry-scheduling/expiry), then ONE scientist
			# pass on a fresh report. Guarded: the loop must never crash a session, and with no
			# scientist LLM (or siege_hypothesis_loop=false) the whole block is a no-op.
			try:
				_nb55 = self._siege_notebook
				if hasattr(_nb55, "hypothesis_housekeeping"):
					_nb55.hypothesis_housekeeping(session_idx)
					if getattr(_nb55, "last_hypothesis_decision", None):
						print(f"[siege][HYPOTHESIS] {_nb55.last_hypothesis_decision}")
					_due55 = (
						_nb55.hypothesis_scientist_due(session_idx)
						if getattr(modeler, "scientist_llm", None) is not None else None
					)
					if _due55:
						_wall55 = str(_due55.get("wall"))
						print(f"[siege][HYPOTHESIS] session={session_idx}: scientist pass on "
						      f"{_wall55}'s fresh probe report.")
						_hyp55 = modeler.hypothesize_probe(
							_wall55, _nb55.scientist_context(_wall55)
						)
						_nb55.admit_hypothesis(_wall55, _hyp55, session_idx)
						if getattr(_nb55, "last_hypothesis_decision", None):
							print(f"[siege][HYPOTHESIS] {_nb55.last_hypothesis_decision}")
			except Exception as _e55:  # noqa: BLE001 — the hypothesis loop must never kill a session
				print(f"[siege][HYPOTHESIS] loop failed ({type(_e55).__name__}: {_e55}); "
				      f"skipped this session.")
			guidance = modeler.diagnose_siege(
				session_idx, mastered_tasks, parent_context,
				notebook_text=self._siege_notebook.render_for_prompt(),
				combat_targets=combat_targets,
				cooc_hint=cooc_hint,
				behav_hint=behav_hint,
				forensics=forensics,
			)
			# v6fix9 P2 monitoring: one line per focus claim + one per final rejection.
			_su = guidance.get("siege_update") or {}
			for _f in _su.get("foci") or []:
				_a = _f.get("failure_attribution") or {}
				if _a:
					_rej = f" (REJECTED claim: {_a['rejected']})" if _a.get("rejected") else ""
					# v7fix5.0 P1 audit trail: show the deterministic access-frontier override AND
					# what the LLM originally said (the s207 misdiagnosis must stay visible in logs).
					_ovr = (
						f" (OVERRIDDEN={_a['overridden']}: llm_said "
						f"{_a.get('llm_said_class')}/{_a.get('llm_said_key')})"
						if _a.get("overridden") else ""
					)
					print(
						f"[siege][attrib] {_f.get('skill')}: class={_a.get('class')} "
						f"key={_a.get('key_missing_link')} verified={_a.get('verified', False)}"
						f"{_rej}{_ovr}"
					)
			for _v in _su.get("attrib_violations") or []:
				print(f"[siege][attrib] VIOLATION: {_v}")
			# log #3 (user 2026-07-05): capture the LLM's RAW proposal BEFORE the B-layer folds it, so
			# the log shows what the model asked for vs. what the hard constraints actually allowed.
			raw_su = guidance.get("siege_update") or {}
			# §2.6: siege_update is now {"foci": [{skill, prereq_tree}, ...]}; extract the proposed
			# focus skills + the union of their prereq skills, just for the [llm-proposal] log line.
			raw_foci = raw_su.get("foci") if isinstance(raw_su, dict) else None
			# v7fix1 visibility: a relay ask must be distinguishable from a normal focus ask in the
			# log — the first run could not tell "the LLM never proposed a relay" from "it proposed
			# one and a gate swallowed it".
			raw_focus = [
				(f"{f.get('skill')}@relay_r0={int(f['relay_r0_floor'])}"
				 if isinstance(f.get("relay_r0_floor"), (int, float)) and int(f["relay_r0_floor"]) >= 1
				 else f.get("skill"))
				for f in raw_foci if isinstance(f, dict) and f.get("skill")
			] if isinstance(raw_foci, list) else None
			raw_tree_skills = [
				str(it.get("skill"))
				for f in (raw_foci or []) if isinstance(f, dict)
				for it in (f.get("prereq_tree") or []) if isinstance(it, dict) and it.get("skill")
			] if isinstance(raw_foci, list) else []
			# fold the LLM's proposed update through every B-layer hard constraint, then persist.
			self._siege_notebook.apply_llm_update(
				session_idx, latest_profile, guidance.get("siege_update"),
				num_snapshots=num_snapshots,
				forensics=(forensics if forensics_available else None),
				chain_incomplete=chain_incomplete,
			)
			# v6fix10 ③: reset the per-session force-activation counters the zero-win discount
			# reads in attempt_to_activate_task (evolution workers run async within the session).
			self._siege_force_counts = {}
			siege_unmastered = self._siege_notebook.unmastered_links(latest_profile)
			siege_directive_text = self._render_siege_directive(latest_profile)
			nb = self._siege_notebook
			print(
				f"[siege] session={session_idx}: foci={nb.focus_skills()}; "
				f"{len(siege_unmastered)} unmastered link(s) forbidden in Completed; "
				f"{len(nb.verified_chains())} verified chain(s)."
			)
			# log #3: LLM proposal vs. the B-layer-corrected result. A divergence here (e.g. the LLM
			# asked to open a focus but the expand/scope gate refused) is exactly what we want visible.
			print(
				f"[siege][llm-proposal] session={session_idx}: "
				f"raw foci={raw_focus!r}, raw prereq_tree={raw_tree_skills} "
				f"-> applied foci={nb.focus_skills()!r}, "
				f"prereq_tree={[l.get('skill') for l in nb.prereq_links()]}"
			)
			# log #1 (user 2026-07-05): WHY the focus is/ isn't what it is — otherwise "focus=None" is
			# ambiguous between immature / no-proposal / switch-too-soon / scope-rejected.
			if nb.last_focus_decision:
				print(f"[siege][focus-decision] session={session_idx}: {nb.last_focus_decision}")
			# log #4: a conquest is the method's core positive signal — announce it loudly.
			if nb.last_conquest:
				print(f"[siege][CONQUEST] session={session_idx}: conquered {nb.last_conquest}")
			# v6fix8 logs: graduation (①), enabler-budget retirement (⑤), ranked_walls auto-open (②).
			if getattr(nb, "last_graduation", None):
				print(f"[siege][GRADUATE] session={session_idx}: {nb.last_graduation}")
			if getattr(nb, "last_budget_retire", None):
				print(f"[siege][BUDGET-RETIRE] session={session_idx}: {nb.last_budget_retire}")
			if getattr(nb, "last_auto_open", None):
				print(f"[siege][AUTO-OPEN] session={session_idx}: {nb.last_auto_open}")
			# v6fix10 logs: door substitution (①), yield / resume (②).
			if getattr(nb, "last_door_sub", None):
				print(f"[siege][DOOR-SUB] session={session_idx}: {nb.last_door_sub}")
			if getattr(nb, "last_yield", None):
				print(f"[siege][YIELD] session={session_idx}: {nb.last_yield}")
			if getattr(nb, "last_resume", None):
				print(f"[siege][RESUME] session={session_idx}: {nb.last_resume}")
			# v6fix10.1 hazard-1: the ④ shortcut arming the P3 early-stop is a monitored event.
			if getattr(nb, "last_attrib_arm", None):
				print(f"[siege][ATTRIB-ARM] session={session_idx}: {nb.last_attrib_arm}")
			# v7: a spawn-anneal relay campaign opening is THE deep-wall event — announce it.
			if getattr(nb, "last_relay_open", None):
				print(f"[siege][RELAY-OPEN] session={session_idx}: {nb.last_relay_open}")
			# v7fix5.2: retire->park routing (P0) and access-root nomination (P1) are the seat
			# events this fix exists for — announce both.
			if getattr(nb, "last_park", None):
				print(f"[siege][PARK] session={session_idx}: {nb.last_park}")
			if getattr(nb, "last_access_auto", None):
				print(f"[siege][ACCESS-AUTO] session={session_idx}: {nb.last_access_auto}")
			# v7fix5.5: the probe gate's verdict (accept / reject receipt) — greppable, and the
			# journal mirrors it so the modeler sees the same line next session.
			if getattr(nb, "last_probe_decision", None):
				print(f"[siege][PROBE] session={session_idx}: {nb.last_probe_decision}")
		else:
			guidance = (
				modeler.diagnose(session_idx, mastered_tasks, parent_context)
				if modeler
				else {"student_states": {}, "guidance_per_parent": {}}
			)
		if modeler:
			states = guidance.get("student_states", {})
			print(
				f"[modeler] session={session_idx} diagnosed {len(states)} skills; "
				f"guidance for {len(guidance.get('guidance_per_parent', {}))} parents."
			)

		# 3. sequential fill with rotating turn order.
		n_prop = len(self.proposer_llms)
		order = [(self._coop_turn_offset + i) % n_prop for i in range(n_prop)]
		self._coop_turn_offset = (self._coop_turn_offset + 1) % max(n_prop, 1)

		# Per-parent record of what earlier proposers already produced this round (for PEER_ALREADY_MADE).
		peer_made: dict[str, list[str]] = {pid: [] for pid in mastered_tasks}

		# v6fix7 P0.1: on siege sessions (notebook exists <=> siege config on) every proposer is
		# asked for the machine-readable <level_meta> block and a response without one is re-queried.
		from auction.level_meta import render_level_meta_spec

		_siege_on = getattr(self, "_siege_notebook", None) is not None
		_lm_spec = render_level_meta_spec(_siege_on)

		all_parsed: list[dict] = []
		all_parents: list[list[str]] = []
		all_examples: list[list[str]] = []

		# v7fix3 P4: the ECOLOGY role is bound to the persona name (config `personas`), stable
		# across the rotating turn order. An ecology proposer never sees the siege directive —
		# it gets the system-computed ecology brief instead — and its siege tags are stripped
		# at parse time below (the division of labour is architecture, not prompt hope).
		_eco_idxs = self._ecology_proposer_idxs()
		_eco_directive = ""
		if siege_active and _eco_idxs:
			_eco_directive = self._render_ecology_directive(
				self._profile_log.latest() if self._profile_log else {}
			)

		for turn_i, proposer_idx in enumerate(order):
			proposer = self.proposer_llms[proposer_idx]
			module = (
				self.persona_prompts[proposer_idx]
				if self.persona_prompts is not None
				else self.evolve_mastered_prompt
			)
			_is_eco = proposer_idx in _eco_idxs
			# Build per-parent extra fields (modeler guidance + peer-so-far + turn order).
			extra: dict[str, dict] = {}
			for pid in mastered_tasks:
				peer_txt = (
					"\n---\n".join(peer_made[pid]) if peer_made[pid] else "(you are first this round)"
				)
				extra[pid] = {
					"MODELER_GUIDANCE": modeler.render_guidance_for_parent(guidance, pid)
					if modeler
					else "(no modeler)",
					"PEER_ALREADY_MADE": peer_txt,
					"REFERENCE_LEVEL": self._coop_reference_text(guidance, pid),
					"MY_TURN_ORDER": f"You are proposer {turn_i + 1} of {n_prop} this round.",
					# v6 §3.4: the siege focus + still-unmastered links this proposer must NOT compress.
					# v7fix3 P4: the ecology proposer NEVER receives it (empty string) — fix11
					# showed the directive + persona precedence converts every proposer into a
					# siege arm (24/24 tagged, BREADTH 5/276, INTERMEDIATE starved).
					"SIEGE_DIRECTIVE": "" if _is_eco else siege_directive_text,
					# v7fix3 P4: the ecology brief (starved families / declines / spawn frontier).
					"ECOLOGY_DIRECTIVE": _eco_directive if _is_eco else "",
					# v6fix7 P0.1: ask for the machine-readable <level_meta> block on siege sessions
					# only (notebook exists <=> siege config on). Siege off -> stays "" (baseline).
					"LEVEL_META_SPEC": _lm_spec,
				}

			system_prompt = self._build_system_prompt(module)
			user_prompts, p_sets, e_sets = self._build_mastered_prompts(
				mastered_tasks,
				global_agent_profile,
				prompt_module=module,
				extra_fields_per_parent=extra,
			)
			if not user_prompts:
				continue

			prev_llm = self.llm
			self.llm = proposer
			try:
				# v6fix7 P0.2: on siege sessions request an ALIGNED list (None placeholders) so the
				# validator reroll and the p_sets pairing below can index prompts/parents safely.
				parsed_responses = self._query_and_parse_responses(
					system_prompt,
					user_prompts,
					require_level_meta=_siege_on,
					return_aligned=_siege_on,
				)
				if _siege_on:
					parsed_responses = self._siege_validate_and_reroll(
						parsed_responses,
						user_prompts,
						system_prompt,
						p_sets,
						proposer_idx,
						siege_unmastered,
					)
			finally:
				self.llm = prev_llm

			for local_i, parsed in enumerate(parsed_responses):
				if parsed is None:
					continue  # aligned siege list: a permanently-failed parse keeps its slot
				# v6 §3.4 code backstop: pull any still-unmastered siege link the proposer illegally
				# compressed into `Completed` back into `Relevant` (so it is actually trained). No-op
				# when siege is off / no focus / no violation, so the v5 coop path is unchanged.
				if siege_unmastered and isinstance(parsed, dict) and parsed.get("description"):
					from auction.completed_gate import enforce_completed_gate

					fixed, moved = enforce_completed_gate(parsed["description"], siege_unmastered)
					if moved:
						parsed["description"] = fixed
						print(
							f"[siege][gate] proposer_{proposer_idx}: moved {moved} from Completed to "
							f"Relevant (unmastered siege links must be trained)."
						)
				# v7fix3 P4: the ecology proposer cannot arm siege privileges — a siege_wall tag,
				# or a drill_target naming an ACTIVE focus, is stripped HERE, before it can reach
				# note_siege_level_type, the coop selector's siege partition, or the archive node
				# attrs that drive force-activation. A drill_target naming a NON-focus skill is
				# the ecology role's legitimate CONSOLIDATE job and stays (it confers no
				# privileges — only tags matching an active focus do). The level survives either
				# way as a normal candidate.
				_meta = parsed.get("level_meta") if isinstance(parsed, dict) else None
				if _is_eco and _meta:
					_nb_eco = getattr(self, "_siege_notebook", None)
					_focus_set = (
						{s.lower() for s in _nb_eco.focus_skills()} if _nb_eco is not None else set()
					)
					_dt = str(_meta.get("drill_target") or "").lower()
					if _meta.get("siege_wall") or (_dt and _dt in _focus_set):
						print(
							f"[coop][role] proposer_{proposer_idx} (ecology) siege tag stripped: "
							f"siege_wall={_meta.get('siege_wall')!r}, "
							f"drill_target={_meta.get('drill_target')!r} — ecology levels carry "
							f"no siege privileges."
						)
						_meta["siege_wall"] = None
						if _dt and _dt in _focus_set:
							_meta["drill_target"] = None
				# v6fix7 P1a: record which FORM attacked each wall (feeds the L2 forced flip).
				if _meta and _siege_on:
					_nb = getattr(self, "_siege_notebook", None)
					if _nb is not None:
						_walls = {
							w for w in (_meta.get("siege_wall"), _meta.get("drill_target")) if w
						}
						for _w in _walls & {s.lower() for s in _nb.focus_skills()}:
							_nb.note_siege_level_type(_w, _meta.get("type"))
				# v7fix3 P4: remember which proposer made it (role quota in _coop_select).
				if isinstance(parsed, dict):
					parsed["_proposer_idx"] = proposer_idx
				all_parsed.append(parsed)
				all_parents.append(p_sets[local_i] if local_i < len(p_sets) else [])
				all_examples.append(e_sets[local_i] if local_i < len(e_sets) else [])
				# Record this proposal so the NEXT proposer sees it (peer awareness).
				pid = p_sets[local_i][0] if local_i < len(p_sets) and p_sets[local_i] else None
				if pid is not None:
					desc = (parsed.get("description") or "") if isinstance(parsed, dict) else ""
					peer_made[pid].append(f"[proposer_{proposer_idx}] {desc}")

		if not all_parsed:
			print("[coop] No proposals produced. Skipping.")
			return []

		# v7fix3 P6: per-session quota on deep-spawn BREADTH levels (relay-wall levels are exempt —
		# their spawn_floor is the rung CONTRACT, not the breadth lane). v7fix2's healthy dormant
		# phase used 1-7/session; the quota bounds the lane without killing it.
		_nb_q = getattr(self, "_siege_notebook", None)
		if _nb_q is not None:
			_quota = int(getattr(getattr(_nb_q, "th", None), "breadth_spawn_quota", 6) or 0)
			_relay_set = set(_nb_q.relay_walls()) if hasattr(_nb_q, "relay_walls") else set()
			_kp, _kpa, _ke, _deep, _dropped = [], [], [], 0, 0
			for _p, _pa, _e in zip(all_parsed, all_parents, all_examples):
				_m = _p.get("level_meta") if isinstance(_p, dict) else None
				try:
					_sf = int((_m or {}).get("spawn_floor") or 0)
				except (TypeError, ValueError):
					_sf = 0
				_tags_q = {
					str((_m or {}).get("siege_wall") or "").lower(),
					str((_m or {}).get("drill_target") or "").lower(),
				} - {""}
				_is_relay_level = bool(_tags_q & _relay_set)
				if _sf >= 1 and not _is_relay_level:
					if _quota > 0 and _deep >= _quota:
						_dropped += 1
						continue
					_deep += 1
				_kp.append(_p)
				_kpa.append(_pa)
				_ke.append(_e)
			if _dropped:
				print(
					f"[breadth][QUOTA] session={session_idx}: {_dropped} deep-spawn level(s) "
					f"beyond the per-session breadth quota ({_quota}) dropped ({_deep} kept)."
				)
			all_parsed, all_parents, all_examples = _kp, _kpa, _ke

		# Optional QUALITY SELECTION (v5_design.md §8). coop_select_k=null -> keep all (变种0, 不筛);
		# coop_select_k=K -> select K from the candidate pool by a NON-competitive bid (default 方案A:
		# AmbitionGain + Learnability; Coverage/Endorsement off, see §8.2). Aligns v5y's per-round
		# retained count to baseline's for a clean comparison. Nothing competitive (no cross-rating).
		select_k = getattr(self.config, "coop_select_k", None)
		if select_k is not None and len(all_parsed) > int(select_k):
			all_parsed, all_parents, all_examples = self._coop_select(
				all_parsed, all_parents, all_examples, int(select_k), session_idx, global_agent_profile
			)

		print(
			f"[coop] session={session_idx}: {len(all_parsed)} levels kept from {n_prop} proposer(s) "
			f"(turn order {order}); selection={'top-%d' % int(select_k) if select_k is not None else 'none (all kept)'}."
		)
		# v7fix3 P4 monitoring: how many kept levels came from the ecology role this round.
		if _eco_idxs:
			_eco_kept = sum(
				1 for p in all_parsed
				if isinstance(p, dict) and p.get("_proposer_idx") in _eco_idxs
			)
			print(
				f"[coop][role] session={session_idx}: ecology levels kept = "
				f"{_eco_kept}/{len(all_parsed)}."
			)
			try:
				import wandb as _wandb
				if getattr(_wandb, "run", None) is not None:
					_wandb.log({"siege/ecology_kept": int(_eco_kept)}, commit=False)
			except Exception:  # noqa: BLE001 — telemetry must never break a session
				pass
		# v7fix4 P2: system-built relay levels join AFTER selection — they own no proposal seat
		# (the 18+8 seat economics are untouched); their training volume is bounded downstream by
		# the activation lanes (zero-win discount cap / full-price cap), exactly like FM siege
		# levels. FM codegen is skipped for them (the code ships with the entry).
		for _sys in self._system_relay_levels(session_idx):
			all_parsed.append(_sys)
			all_parents.append([])
			all_examples.append([])
		return self._organize_data(
			all_parsed, all_parents, all_examples, session_idx, "mastered"
		)

	def _coop_select(
		self, all_parsed, all_parents, all_examples, k, session_idx, global_agent_profile,
		siege_partition=True,
	):
		"""Select k candidates from the coop pool by AmbitionGain+Learnability (v5_design.md §8.2 方案A).

		Reuses the auction GreedyTopKSelector with Coverage/Endorsement weights = 0 (so it reduces to a
		modular AmbitionGain + Learnability top-k). NON-competitive: no cross-rating. Returns the three
		parallel lists filtered to the winners, preserving order. Bid weights are config-overridable
		(coop_w_amb / coop_w_lrn), defaulting to 1/1.
		"""
		from auction.selectors import GreedyTopKSelector, SelectionContext

		try:
			from .auction_integration import parsed_response_to_proposal, profile_to_target_gap
		except ImportError:  # spec-from-file loads (unit tests) have no package parent
			from dicode.dreaming.auction_integration import (
				parsed_response_to_proposal,
				profile_to_target_gap,
			)

		# v6fix7 P1b — SIEGE RESCUE: candidates tagged (via <level_meta>) as attacking or drilling an
		# ACTIVE focus must not be killed by the learnability cull. Pure-Learnability top-k otherwise
		# culls by parent-lineage p*(1-p), which systematically kills drill lineages exactly when
		# their p saturates — "learned, therefore discarded" — while the wall's held-out SR has not
		# transferred.
		# v6fix8 ④ — EXTRA SEATS, NOT RESERVED SEATS (user 2026-07-07): fix7's uncapped in-k bypass
		# let proposers tag 22/24 candidates siege and monopolise the whole round ("top-0 over the
		# remaining 2"), starving coverage/learnability — the tax behind fix7's mean_return lagging
		# baseline. Now ALL k seats stay a pure learnability competition (siege candidates compete on
		# merit like everyone else); siege-tagged candidates that LOST the competition are rescued
		# onto up to ``siege_select_cap`` (default 8) EXTRA seats — total kept <= k + cap. Extra
		# seats are dealt round-robin across the active foci (no single wall hoards them); siege
		# losers beyond the cap are dropped. This mirrors the rehearsal philosophy (user 2026-07-04:
		# the main allotment keeps its full N; protected additions are EXTRA, never displacing).
		# No-op when siege off / no focus / untagged.
		nb = getattr(self, "_siege_notebook", None)
		foci_order = [s.lower() for s in nb.focus_skills()] if nb is not None else []
		if foci_order and siege_partition:
			def _siege_wall_of(parsed):
				meta = parsed.get("level_meta") if isinstance(parsed, dict) else None
				if not meta:
					return None
				tags = {
					str(meta.get("siege_wall") or "").lower(),
					str(meta.get("drill_target") or "").lower(),
				}
				for f in foci_order:
					if f in tags:
						return f
				return None

			siege_idx = [i for i, p in enumerate(all_parsed) if _siege_wall_of(all_parsed[i]) is not None]
			if siege_idx:
				cap = int(getattr(self.config, "siege_select_cap", 8) or 0)
				# 1) the FULL k-seat competition runs undistorted over ALL candidates.
				sel_p, sel_pa, sel_e = self._coop_select(
					all_parsed, all_parents, all_examples, k, session_idx, global_agent_profile,
					siege_partition=False,
				)
				won = {id(p) for p in sel_p}
				# 2) siege-tagged LOSERS get up to ``cap`` extra seats, round-robin across foci
				#    (proposal order within a wall).
				by_wall: dict[str, list[int]] = {f: [] for f in foci_order}
				for i in siege_idx:
					if id(all_parsed[i]) not in won:
						by_wall[_siege_wall_of(all_parsed[i])].append(i)
				losers = sum(len(v) for v in by_wall.values())
				# v6fix10 ③ zero-win discount: a focus with NO held-out win ever gets only a
				# fraction of the extra-seat cap (per-wall share; the freed seats go to walls
				# WITH evidence via the same round-robin). First win ratchets it to full price.
				zero_walls: set[str] = set()
				try:
					_prof = self._profile_log.latest() if self._profile_log else {}
					zero_walls = nb.zero_win_walls(
						{str(kk).lower(): vv for kk, vv in (_prof or {}).items()}
					)
				except Exception:  # noqa: BLE001 — discount must never break selection
					zero_walls = set()
				frac = float(getattr(getattr(nb, "th", None), "zero_win_seat_frac", 0.5) or 0.5)
				import math as _math
				wall_cap = {
					f: (max(1, int(_math.ceil(cap * frac))) if f in zero_walls else cap)
					for f in foci_order
				}
				taken: dict[str, int] = {f: 0 for f in foci_order}
				extra: list[int] = []
				while len(extra) < min(cap, losers):
					advanced = False
					for f in foci_order:
						if by_wall[f] and len(extra) < cap and taken[f] < wall_cap[f]:
							extra.append(by_wall[f].pop(0))
							taken[f] += 1
							advanced = True
					if not advanced:
						break
				extra = sorted(extra)
				n_won = sum(1 for i in siege_idx if id(all_parsed[i]) in won)
				_disc = sorted(zero_walls & set(foci_order))
				print(
					f"[coop][select][siege] session={session_idx}: {len(siege_idx)} siege-tagged "
					f"candidate(s): {n_won} won the open top-{k} on merit; {len(extra)} rescued onto "
					f"EXTRA seats (cap {cap}, round-robin over foci={foci_order}); "
					f"{losers - len(extra)} dropped. Total kept = {k + len(extra)}."
				)
				if _disc:
					print(
						f"[siege][DISCOUNT] session={session_idx}: zero-win wall(s) {_disc} at "
						f"{frac:.0%} extra-seat share (cap {[wall_cap[f] for f in _disc]}) until "
						f"their first held-out win."
					)
				return (
					sel_p + [all_parsed[i] for i in extra],
					sel_pa + [all_parents[i] for i in extra],
					sel_e + [all_examples[i] for i in extra],
				)

		# Build Proposal objects with a stable index id so we can map winners back to the lists.
		proposals = []
		for i, parsed in enumerate(all_parsed):
			parent = all_parents[i][0] if i < len(all_parents) and all_parents[i] else ""
			proposals.append(
				parsed_response_to_proposal(
					parsed, proposal_id=f"coop_s{session_idx}_{i}", proposer_id="coop", parent_task_id=parent
				)
			)
		target_gap = profile_to_target_gap(global_agent_profile)
		parent_learnability = self._build_parent_learnability(proposals)
		context = SelectionContext(
			target_gap=target_gap or None,
			cross_ratings=None,  # NON-competitive: no endorsement in coop
			parent_learnability=parent_learnability or None,
			w_cov=float(getattr(self.config, "coop_w_cov", 0.0)),   # 方案A: Coverage OFF (§8.2)
			w_end=0.0,                                              # Endorsement OFF (non-competitive)
			w_amb=float(getattr(self.config, "coop_w_amb", 1.0)),
			w_lrn=float(getattr(self.config, "coop_w_lrn", 1.0)),
			reachable_ceiling=None,  # v4 already retired the ability gate; do not revive it
		)
		# v7fix3 P4: per-role quotas inside the k main seats. fix11 evidence: with a single open
		# competition, mid-band siege drills are exactly the most learnable candidates and sweep
		# all k seats (s27: 24/24 tagged, 18/18 won "on merit") — the ecology needs a floor, not
		# a prayer. Same bid scoring, each role ranks its own queue; unfilled quota backfills
		# across roles on merit. Falls back to the open competition when unconfigured.
		selector = GreedyTopKSelector()
		win_ids: set[int] | None = None
		role_quota = getattr(self.config, "coop_role_quota", None)
		if role_quota is not None:
			try:
				quotas = [int(q) for q in role_quota]
			except (TypeError, ValueError):
				quotas = []
			has_roles = any(
				isinstance(p, dict) and isinstance(p.get("_proposer_idx"), int) for p in all_parsed
			)
			if quotas and sum(quotas) == int(k) and has_roles:
				buckets: dict[int, list[int]] = {}
				for i, p in enumerate(all_parsed):
					ridx = p.get("_proposer_idx") if isinstance(p, dict) else None
					b = min(int(ridx), len(quotas) - 1) if isinstance(ridx, int) and ridx >= 0 else 0
					buckets.setdefault(b, []).append(i)
				win_ids = set()
				for b, q in enumerate(quotas):
					subset = [proposals[i] for i in buckets.get(b, [])]
					if not subset or q <= 0:
						continue
					for w in selector.select(subset, min(q, len(subset)), context):
						win_ids.add(int(w.proposal_id.rsplit("_", 1)[1]))
				spare = int(k) - len(win_ids)
				if spare > 0:
					rest = [proposals[i] for i in range(len(all_parsed)) if i not in win_ids]
					for w in selector.select(rest, min(spare, len(rest)), context):
						win_ids.add(int(w.proposal_id.rsplit("_", 1)[1]))
				per_role = {}
				for i in win_ids:
					p = all_parsed[i]
					ridx = p.get("_proposer_idx") if isinstance(p, dict) else None
					b = min(int(ridx), len(quotas) - 1) if isinstance(ridx, int) and ridx >= 0 else 0
					per_role[b] = per_role.get(b, 0) + 1
				print(
					f"[coop][select][role] session={session_idx}: quotas={quotas} -> "
					f"won per role={per_role}"
					f"{f' (backfilled {spare})' if spare > 0 else ''}."
				)
			elif quotas and sum(quotas) != int(k):
				print(
					f"[coop][select][role] WARNING: coop_role_quota={quotas} does not sum to "
					f"k={k} — falling back to the open competition."
				)
		if win_ids is None:
			winners = selector.select(proposals, k, context)
			win_ids = {int(w.proposal_id.rsplit("_", 1)[1]) for w in winners}
		win_idx = sorted(win_ids)
		print(
			f"[coop][select] session={session_idx} kept {len(win_idx)}/{len(all_parsed)} "
			f"(w_amb={context.w_amb}, w_lrn={context.w_lrn}, w_cov={context.w_cov}); "
			f"ambition_skills={len(target_gap)}, learnable_parents={len(parent_learnability)}."
		)
		return (
			[all_parsed[i] for i in win_idx],
			[all_parents[i] for i in win_idx],
			[all_examples[i] for i in win_idx],
		)

	def _ecology_proposer_idxs(self) -> set[int]:
		"""v7fix3 P4: proposer indices whose persona is the ECOLOGY role (config `personas`).

		Role identity is the persona NAME at the proposer's index — stable across the rotating
		turn order (personas are paired to proposers BY INDEX, see _load_persona_prompts).
		Empty set when personas are unconfigured or no ecology persona is listed (pure fix2
		behaviour: every proposer receives the siege directive).

		Defensive config access (v7fix3.1): _render_siege_directive calls this too, and that
		renderer is exercised on bare TaskGenerator test doubles with NO config attribute —
		same convention as the SEAT BUDGET block."""
		cfg = getattr(self, "config", None)
		personas_cfg = cfg.get("personas", None) if hasattr(cfg, "get") else None
		if not personas_cfg:
			return set()
		return {
			i for i, name in enumerate(personas_cfg)
			if isinstance(name, str) and name.strip().lower() == "ecology_coop"
		}

	def _render_ecology_directive(self, latest_profile: dict | None) -> str:
		"""v7fix3 P4: the ecology proposer's brief — computed from held-out telemetry, zero LLM cost.

		Three sections: (1) STARVED FAMILIES — held-out families with the lowest live means and
		their weakest skills; (2) DECLINING SKILLS — well off their historical peak (read-only
		scan, does not touch the rehearsal machinery); (3) BREADTH SPAWN FRONTIER — the deep-spawn
		lever and its current bound. Empty string when there is no profile data yet."""
		from auction.craftax_achievements import ALL_ACHIEVEMENTS, family_of

		profile = {
			str(k).lower(): float(v)
			for k, v in (latest_profile or {}).items()
			if isinstance(v, (int, float))
		}
		if not profile:
			return ""
		lines: list[str] = []
		# (1) starved families: mean live SR per family, weakest first; list each family's
		# weakest non-saturated skills so the brief is actionable, not just a label.
		fam_skills: dict[str, list[tuple[str, float]]] = {}
		for name in sorted(ALL_ACHIEVEMENTS):
			sr = profile.get(name)
			if sr is None:
				continue
			fam_skills.setdefault(family_of(name), []).append((name, sr))
		if fam_skills:
			fam_mean = {
				f: (sum(sr for _, sr in members) / len(members))
				for f, members in fam_skills.items()
			}
			lines.append("STARVED FAMILIES (held-out mean, weakest first — serve these):")
			for f in sorted(fam_mean, key=fam_mean.get):
				weakest = sorted(fam_skills[f], key=lambda kv: kv[1])[:4]
				weakest_s = ", ".join(f"{n} {sr:.0f}%" for n, sr in weakest)
				lines.append(f"  - {f}: mean {fam_mean[f]:.0f}% | weakest: {weakest_s}")
			# v7fix3.1: reachability note — the weakest-first sort necessarily puts the deepest
			# all-zero families on top, and those are usually depth-blocked (siege/relay
			# territory), not starving-for-exposure. Keep the brief from steering the ecology
			# designer at walls it cannot reach with a natural-spawn level.
			lines.append(
				"  NOTE: a family whose EVERY member reads 0% usually lives on floors natural "
				"spawn cannot reach — that is siege territory, not yours. Prefer starved "
				"families with at least one non-zero member, or serve a deep family via the "
				"BREADTH SPAWN FRONTIER lane below (within the stated floor bound)."
			)
		# (2) declining skills: >= 15pp off a peak of >= 25% (real capital, really slipping).
		declines: list[tuple[float, str, float, float]] = []
		history = self._profile_log.recent(10_000) if self._profile_log else []
		peaks: dict[str, float] = {}
		for snap in history:
			for name, sr in (snap.get("profile") or {}).items():
				if isinstance(sr, (int, float)):
					nl = str(name).lower()
					peaks[nl] = max(peaks.get(nl, 0.0), float(sr))
		for name, peak in peaks.items():
			cur = profile.get(name)
			if cur is None or peak < 25.0:
				continue
			drop = peak - float(cur)
			if drop >= 15.0:
				declines.append((drop, name, peak, float(cur)))
		if declines:
			declines.sort(reverse=True)
			lines.append("DECLINING SKILLS (well off their peak — refresh before they rot):")
			for drop, name, peak, cur in declines[:6]:
				lines.append(f"  - {name}: {cur:.0f}% now vs {peak:.0f}% peak (-{drop:.0f}pp)")
		# (3) the deep-spawn lever (P6) — frontier + quota so the proposer plans within bounds.
		nb = getattr(self, "_siege_notebook", None)
		if nb is not None and hasattr(nb, "breadth_frontier"):
			frontier = int(nb.breadth_frontier())
			quota = int(getattr(getattr(nb, "th", None), "breadth_spawn_quota", 6) or 0)
			lines.append(
				f"BREADTH SPAWN FRONTIER: floor {frontier}. A BREADTH level (no siege tags) may "
				f'declare "spawn_floor" up to {frontier} to train that floor\'s family directly; '
				f"at most {quota} deep-spawn levels are kept per round, and the frontier advances "
				f"only when the current floor's breadth levels are actually being won in training."
			)
		return "\n".join(lines)

	def _note_breadth_frontier_readings(self, session_idx: int | None = None) -> None:
		"""v7fix3 P6: feed the frontier the freshest trained SR of deep-spawn BREADTH levels AT the
		frontier floor; print + wandb-log an advance. Same recency window as the gap sweep."""
		import json as _json

		nb = getattr(self, "_siege_notebook", None)
		if nb is None or not hasattr(nb, "note_breadth_frontier_reading"):
			return
		frontier = int(nb.breadth_frontier())
		recency = int(getattr(getattr(self, "config", None), "siege_gap_trained_recency", 2) or 2)
		best: float | None = None
		with self.archive._lock:
			nodes = list(self.archive.graph.nodes(data=True))
		for _nid, data in nodes:
			if str(data.get("level_type", "")).upper() != "BREADTH":
				continue
			if data.get("siege_wall") or data.get("drill_target"):
				continue  # the breadth lane is untagged by definition (P6/R6)
			try:
				node_floor = int(data.get("spawn_floor") or 0)
			except (TypeError, ValueError):
				node_floor = 0
			if node_floor != frontier:
				continue
			ph = data.get("performance_history")
			try:
				ph = _json.loads(ph) if isinstance(ph, str) else (ph or [])
			except (TypeError, ValueError):
				ph = []
			recs = [r for r in ph if isinstance(r, dict) and r.get("sr") is not None]
			if session_idx is not None:
				recs = [
					r for r in recs
					if isinstance(r.get("session"), (int, float))
					and r["session"] >= session_idx - recency
				]
			if not recs:
				continue
			trained_pct = float(recs[-1]["sr"]) * 100.0
			if best is None or trained_pct > best:
				best = trained_pct
		if best is None:
			return
		msg = nb.note_breadth_frontier_reading(frontier, best, session_idx=session_idx)
		if msg:
			print(f"[breadth][FRONTIER] session={session_idx}: {msg}")
			try:
				import wandb as _wandb
				if getattr(_wandb, "run", None) is not None:
					_wandb.log(
						{"siege/breadth_frontier": int(nb.breadth_frontier())}, commit=False
					)
			except Exception:  # noqa: BLE001 — telemetry must never break a session
				pass

	def _coop_reference_text(self, guidance: dict, parent_id: str) -> str:
		"""Render the modeler-recommended reference level body for a parent, or '' if none."""
		if self._modeler is None:
			return ""
		g = (guidance.get("guidance_per_parent") or {}).get(parent_id) or {}
		ref = g.get("reference_level_id") or ""
		if not ref:
			return ""
		detail = self._modeler.view.level_detail(ref)
		if not detail:
			return ""
		return f"[reference level {ref}]\n{detail.get('description', '')}"

	def _render_siege_gap_hint(self, latest_profile: dict, session_idx: int | None = None) -> str:
		"""v6fix7 P1b — the drill "graduation" trigger (SCALAR's train-eval gap, p18 Fig.9, repurposed).

		For each ACTIVE focus, compare the best TRAINED SR among its siege-tagged levels (what the
		student achieves inside the scaffolded/drilled level) against the focus's HELD-OUT SR (the
		real game). A drill that is won in-level (trained >= 90%) while the wall's held-out SR lags
		far behind (gap >= 30pp) is OVERFIT TO THE CALM SANDBOX — the CONSOLIDATE definition's "add
		the pressure back as SR rises" clause must fire now, quantified instead of vibes. Rendered
		into the modeler's siege prompt; empty when siege off / no focus / no trained siege level.
		"""
		import json as _json

		nb = getattr(self, "_siege_notebook", None)
		if nb is None:
			return ""
		foci = {s.lower() for s in nb.focus_skills()}
		if not foci:
			return ""
		latest_profile = {str(k).lower(): v for k, v in (latest_profile or {}).items()}

		# v6fix9 P0.5-#2: only levels actually TRAINED within the last ``recency`` sessions count.
		# The old max(srs[-1]) over EVERY node this wall ever tagged was a never-falling high-water
		# mark: one drill hitting 90% once kept the wall reading "sandbox won" forever, so fix8 ③'s
		# "drill regressed / no reading -> counter resets" leg was structurally dead — and the P3
		# early-stop would inherit the same artifact. recency=2 tolerates exactly one sampling gap
		# (the siege focus quota keeps live drills in rotation ~every session; two absent sessions
		# means the drills really left the rotation -> trained=None -> the gate resets, by design).
		recency = int(getattr(getattr(self, "config", None), "siege_gap_trained_recency", 2) or 2)
		# v7: relay walls read ONLY levels declared at the CURRENT rung's spawn floor — a stale
		# deeper-rung drill still scoring 95% must not fake-graduate the new, harder rung.
		_rsf = getattr(nb, "required_spawn_floor", None)
		_relay_sys_only = str(
			getattr(getattr(self, "config", None), "siege_relay_worldgen", "base")
		) == "base"
		relay_floor: dict[str, int] = {}
		relay_stage: dict[str, int] = {}
		_rss = getattr(nb, "relay_sub_stage", None)
		if callable(_rsf):
			for wall in foci:
				f = _rsf(wall)
				if f is not None:
					relay_floor[wall] = int(f)
					# v7fix4.6: the scaffold sub-stage joins the reading filter key.
					relay_stage[wall] = int(_rss(wall)) if callable(_rss) else 0
		best_trained: dict[str, float] = {}
		with self.archive._lock:
			nodes = list(self.archive.graph.nodes(data=True))
		for _nid, data in nodes:
			tags = {
				str(data.get("siege_wall", "")).lower(),
				str(data.get("drill_target", "")).lower(),
			} & foci
			if not tags:
				continue
			try:
				node_floor = int(data.get("spawn_floor") or 0)
			except (TypeError, ValueError):
				node_floor = 0
			ph = data.get("performance_history")
			try:
				ph = _json.loads(ph) if isinstance(ph, str) else (ph or [])
			except (TypeError, ValueError):
				ph = []
			recs = [r for r in ph if isinstance(r, dict) and r.get("sr") is not None]
			if session_idx is not None:
				recs = [
					r for r in recs
					if isinstance(r.get("session"), (int, float))
					and r["session"] >= session_idx - recency
				]
			if not recs:
				continue
			trained_pct = float(recs[-1]["sr"]) * 100.0
			try:
				node_stage = int(data.get("spawn_sub_stage") or 0)
			except (TypeError, ValueError):
				node_stage = 0
			for wall in tags:
				if wall in relay_floor and node_floor != relay_floor[wall]:
					continue  # v7: wrong-rung level — not this rung's evidence
				# v7fix4.6: same floor but a different scaffold stage is ALSO the wrong rung —
				# a stale easier-stage level scoring 90% must not fake-graduate a harder stage.
				if wall in relay_floor and node_stage != relay_stage.get(wall, 0):
					continue
				# v7fix4 P2: a relay rung's evidence must come from SYSTEM-BUILT levels only — an
				# FM-authored level attacking the same wall is not reality-anchored, and unpinned
				# fidelity axes are exactly how the v7fix3 ladder fake-graduated (trained SR rose
				# as the spawn annealed up). Inactive on the "fm" ablation arm.
				if wall in relay_floor and _relay_sys_only and not data.get("system_built"):
					continue
				best_trained[wall] = max(best_trained.get(wall, 0.0), trained_pct)

		# v6fix8 ③: feed the HARD GATE (SiegeNotebook.note_transfer_gap counts consecutive over-gap
		# decisions and forces DEPTH at the cap) and log every reading — fix7 computed this hint but
		# left it prompt-only with zero logging, so the modeler read "52pp gap", kept refining calm
		# drills anyway, and the monitoring never saw it. Every ACTIVE focus is fed (a wall with no
		# trained siege level reads None -> the counter resets: no drills, no drill-transfer gap).
		for wall in sorted(foci):
			trained_v = best_trained.get(wall)
			held_v = latest_profile.get(wall)
			# v7: a relay wall's reading drives the RUNG state machine, not the gap gate (which
			# the notebook suspends for it anyway — mid-rung levels cannot move held-out, so the
			# trained-vs-held gap is 100pp by design, not by overfitting).
			# v7fix5.7-P2' T1 (fix56设计 §3.1): the state machine consumption MOVED to run_dicode
			# Step 4d (SiegeNotebook.consume_rung_eval, once per session at eval delivery). This
			# decision-cadence site used to be the ONLY consumer and ran every other session —
			# half the honest readings were never judged. It keeps ONLY the wandb telemetry
			# (this site owns the archive-side trained-max number).
			if wall in relay_floor:
				# read-only peek for telemetry; rung_eval_for mutates nothing.
				_ev56 = getattr(nb, "rung_eval_for", lambda _w, _s: None)(wall, session_idx)
				eval_v = float(_ev56["sr"]) if _ev56 else None
				# v7 telemetry (guarded: wandb runs in the training process; never fatal).
				try:
					import wandb as _wandb
					if getattr(_wandb, "run", None) is not None:
						_fc = getattr(nb, "_relay_foc", lambda _w: None)(wall)
						_r = (_fc or {}).get("relay") or {}
						_wandb.log({
							f"siege/rung_spawn_floor_{wall}": int(_r.get("spawn_floor", relay_floor[wall])),
							f"siege/rung_trained_sr_{wall}": float(trained_v) if trained_v is not None else -1.0,
							# v7fix5.6: the honest number the state machine actually consumed.
							f"siege/rung_zeroshot_sr_{wall}": float(eval_v) if eval_v is not None else -1.0,
							# v7fix4.6: scaffold stage (0 = FULL) + regress budget burn.
							f"siege/rung_sub_stage_{wall}": int(_r.get("sub_stage", 0)),
							f"siege/rung_regress_count_{wall}": int(_r.get("regress_count", 0)),
							# v7fix4 P3 monitoring: floor 0 alone cannot distinguish "just
							# annealed to natural spawn (kit still on)" from the KIT_STRIP
							# exam — and the exam is the run's SEWN=certificate claim.
							f"siege/relay_kit_stage_{wall}": int(bool(_r.get("kit_strip"))),
						}, commit=False)
				except Exception:  # noqa: BLE001 — telemetry must never break a session
					pass
				continue
			status = nb.note_transfer_gap(wall, trained_v, held_v, session_idx=session_idx)
			if status is not None and (trained_v is not None or status != "ok"):
				t_s = f"{trained_v:.0f}%" if trained_v is not None else "-"
				h_s = f"{held_v:.0f}%" if held_v is not None else "-"
				g_s = (
					f"{trained_v - float(held_v):.0f}pp"
					if trained_v is not None and held_v is not None else "-"
				)
				print(f"[siege][gap] wall={wall}: trained={t_s} held-out={h_s} gap={g_s} -> {status}")

		lines = []
		for wall, trained in sorted(best_trained.items()):
			held = latest_profile.get(wall)
			if held is None:
				continue
			if wall in relay_floor:
				continue  # v7: the journal's RELAY section reports rung state; no gap line
			gap = trained - float(held)
			if trained >= 90.0 and gap >= 30.0:
				lines.append(
					f"  - {wall}: best siege-level TRAINED SR {trained:.0f}% vs HELD-OUT {held:.0f}% "
					f"(gap {gap:.0f}pp) — the drill is won in its calm sandbox but NOT transferring. "
					"Per the CONSOLIDATE definition, ADD THE PRESSURE BACK now (reintroduce mobs/"
					"night/hunger stepwise) and converge toward the full game."
				)
			elif trained > 0:
				lines.append(
					f"  - {wall}: best siege-level trained SR {trained:.0f}% vs held-out {held:.0f}%."
				)
		if not lines:
			return ""
		return "DRILL-TRANSFER GAP (trained-in-level vs held-out real game):\n" + "\n".join(lines)

	def _render_cooccurrence_hint(self) -> str:
		"""v6 §3.8 (c): the (c) co-occurrence evidence text for the siege modeler.

		For the current focus (if any) and every deep COMBAT target with enough successful-episode
		support, list which skills the student ACTUALLY co-reaches when it succeeds — grounding the
		prereq chain in real trajectories, not the LLM's guess. Empty (so the prompt omits it and the
		modeler leans on (b) mechanics) when the log is absent or support is too sparse — the phased
		fallback in v6_design.md §3.8. Bounded to a few lines to keep the prompt tight."""
		log = getattr(self, "_cooc_log", None)
		if log is None:
			return ""
		targets: list[str] = []
		# v6fix9 P0.5-#4: ALL active foci, not just the primary. fix8's auto-open made multi-foci
		# the norm; a secondary non-COMBAT focus (collect_diamond, make_iron_armour, ...) got NO
		# co-occurrence evidence at all — the modeler wrote its style_note/attribution blind.
		if self._siege_notebook is not None:
			targets.extend(s.lower() for s in self._siege_notebook.focus_skills())
		# a few deep combat targets with real support, so the modeler has co-occurrence to reason with
		# (v6fix7 P1d: was `>= 0`, which is always true — the intended prefilter never filtered;
		# render_prereq_hint's MIN_SR guard was silently doing all the work).
		for name in self._combat_target_names():
			if name not in targets and log.support(name) >= 1:
				targets.append(name)
		lines = []
		omitted: list[str] = []
		for name in targets:
			hint = log.render_prereq_hint(name)
			if not hint:
				continue
			# v6fix9 P0.5-#3: over-budget targets WITH evidence are named, never silently dropped.
			if len(lines) >= 6:
				omitted.append(name)
				continue
			lines.append(f"  - {hint}")
		if omitted:
			lines.append(
				f"  - (+{len(omitted)} more with co-occurrence evidence omitted: {', '.join(omitted)})"
			)
		if not lines:
			return ""
		return (
			"REAL-TRAJECTORY CO-OCCURRENCE (from the student's own held-out successes — use this to "
			"pick the prereq chain from what it ACTUALLY strings together, not what you imagine):\n"
			+ "\n".join(lines)
		)

	def _render_chain_hint(self) -> str:
		"""v6fix7 P2: the CHAIN EVIDENCE block for the siege modeler — DIRECTED order upgrade of (c).

		For every active focus (and the tracked retired walls, capped): the dominant success path
		(time-ordered, from real winning episodes), where failing episodes most often break, and
		whether the break-link frontier is advancing (the "0% SR but dying deeper = progress" patience
		signal). Empty when the log is absent or no wall has trustworthy data — phased fallback to
		(b)+(c), same as every other hint."""
		log = getattr(self, "_chain_log", None)
		if log is None:
			return ""
		targets: list[str] = []
		foci: set[str] = set()
		if self._siege_notebook is not None:
			targets = list(self._siege_notebook.chain_targets().keys())
			foci = {s.lower() for s in self._siege_notebook.focus_skills()}
		# v6fix9 P0.5-#3: per-wall quotas instead of a global first-come cap. The old
		# `len(lines) >= 8: break` let the first walls eat the whole budget and silently dropped
		# the rest — and chain_targets orders retired walls LAST, so the evidence that vanished was
		# exactly the "do not reopen this wall / it earned an unlock" record. Active foci keep the
		# full block; retired walls keep their LAST lines (break/frontier — the reopen-relevant
		# ones); anything still over budget is NAMED instead of disappearing (no-silent-caps).
		lines: list[str] = []
		omitted: list[str] = []
		for name in targets:
			hint = log.render_chain_hint(name)
			if not hint:
				continue
			quota = 6 if name in foci else 2
			wall_lines = hint.splitlines()
			wall_lines = wall_lines[:quota] if name in foci else wall_lines[-quota:]
			# budget = MAX_FOCUS(3) x the full per-focus block (6) — every active focus always fits.
			if len(lines) + len(wall_lines) > 18:
				omitted.append(name)
				continue
			lines.extend(f"  - {ln}" for ln in wall_lines)
		if omitted:
			lines.append(
				f"  - (chain evidence for {len(omitted)} more tracked wall(s) omitted: "
				f"{', '.join(omitted)})"
			)
		if not lines:
			return ""
		return (
			"CHAIN EVIDENCE (directed, from real episode ORDER — stronger than co-occurrence: use it "
			"to pick WHICH link to train and to read progress on a wall whose SR is still 0%):\n"
			+ "\n".join(lines)
		)

	def _render_behavior_hint(self) -> str:
		"""v6 problem-2: the behaviour-fingerprint text for the siege modeler — HOW the student behaves
		in the episodes it WINS (action mix / pacing), so the style_note reflects what actually worked
		rather than an imagined tactic. For the current focus + a few deep combat targets with enough
		winning-episode support. Empty (prompt omits it) when the log is absent or every target is solved
		too rarely to trust — the same phased fallback as the (c) hint. Bounded to a few lines."""
		log = getattr(self, "_behav_log", None)
		if log is None:
			return ""
		targets: list[str] = []
		# v6fix9 P0.5-#4: ALL active foci first (see _render_cooccurrence_hint) — a secondary
		# non-COMBAT focus previously got no behaviour evidence for its style_note at all.
		if self._siege_notebook is not None:
			targets.extend(s.lower() for s in self._siege_notebook.focus_skills())
		for name in self._combat_target_names():
			if name not in targets:
				targets.append(name)
		lines = []
		omitted: list[str] = []
		for name in targets:
			hint = log.render_fingerprint_hint(name)
			if not hint:
				continue
			if len(lines) >= 5:  # keep the prompt bounded; name the overflow (no-silent-caps)
				omitted.append(name)
				continue
			lines.append(f"  - {hint}")
		if omitted:
			lines.append(
				f"  - (+{len(omitted)} more with behaviour evidence omitted: {', '.join(omitted)})"
			)
		if not lines:
			return ""
		return (
			"REAL-SUCCESS BEHAVIOUR (how the student ACTUALLY acted in the episodes it won — write the "
			"style_note to match this real action mix / pacing, not a tactic you merely imagine):\n"
			+ "\n".join(lines)
		)

	@staticmethod
	def _combat_target_names() -> list[str]:
		"""COMBAT-family achievement names fed to the siege modeler so it can prefer a stuck combat wall
		as the focus. A category label, not a course-chain prior — see craftax_achievements.family_of."""
		from auction.craftax_achievements import ALL_ACHIEVEMENTS, family_of

		return sorted(a for a in ALL_ACHIEVEMENTS if family_of(a) == "COMBAT")

	def _render_relay_kit_hint(self, wall: str) -> str:
		"""v7 spawn_kit evidence: the winner-median max-inventory stockpiles from the students'
		OWN successful episodes (fix9 P1 telemetry) — the leakage-clean yardstick the proposer
		derives a relay level's starting kit from. Prefers the relay wall's own winners; while the
		wall has none (the usual case — that is why it is a relay), falls back to the tracked
		target with the most winner evidence. Empty string when no inventory telemetry exists yet
		(the proposer then equips from mechanics knowledge, which the prompt already allows)."""
		log = getattr(self, "_chain_log", None)
		if log is None or not hasattr(log, "latest_fail_summary"):
			return ""
		nb = getattr(self, "_siege_notebook", None)

		def _inv_of(target: str):
			try:
				entry = log.latest_fail_summary(target)
			except Exception:  # noqa: BLE001 — hint must never break the directive
				return None, 0
			if not isinstance(entry, dict):
				return None, 0
			inv = entry.get("inv")
			return (inv or None), int(entry.get("n_succ", 0) or 0)

		candidates: list[str] = [wall]
		if nb is not None and hasattr(nb, "chain_targets"):
			try:
				candidates.extend(t for t in nb.chain_targets().keys() if t != wall)
			except Exception:  # noqa: BLE001
				pass
		best_target, best_inv, best_succ = None, None, -1
		for i, t in enumerate(candidates):
			inv, n_succ = _inv_of(t)
			if not inv:
				continue
			if i == 0:  # the wall's own winners always win the tie
				best_target, best_inv, best_succ = t, inv, n_succ
				break
			if n_succ > best_succ:
				best_target, best_inv, best_succ = t, inv, n_succ
		if not best_inv:
			return ""
		agg = self._winner_median_kit(best_inv)
		if not agg:
			return ""
		cols = sorted(agg.items(), key=lambda kv: -kv[1])[:6]
		items = ", ".join(f"{f} {v}" for f, v in cols)
		src = "its own winners" if best_target == wall else f"winners of {best_target}"
		return (
			f"WINNER-MEDIAN STOCKPILES (spawn_kit evidence, from {src}'s real episodes): {items} "
			f"— these are exact spawn_kit field names; use the medians as the kit yardstick and "
			f"do not hand out gear the students' own winning runs never carried."
		)

	@staticmethod
	def _winner_median_kit(inv: dict) -> dict[str, int]:
		"""v7fix2: collapse telemetry column labels onto the LEGAL spawn_kit vocabulary —
		armour_2 / potions_5 are flattened-array labels, not kit fields. armour keeps the best
		slot (a tier), potions sum across colours (a stock count). Shared by the proposer-facing
		hint and the v7fix4 system-built relay levels (single kit source, no drift)."""
		from minicraftax.spawn_kit import canonicalise_telemetry_label

		agg: dict[str, int] = {}
		for c, d in (inv or {}).items():
			f = canonicalise_telemetry_label(c)
			v = int(d.get("succ_med", 0) or 0)
			if f is None or v <= 0:
				continue
			agg[f] = agg.get(f, 0) + v if f == "potions" else max(agg.get(f, 0), v)
		return agg

	def _relay_kit_dict(self, wall: str) -> dict[str, int]:
		"""v7fix4 P2: the winner-median spawn kit for ``wall`` as a dict — same evidence source
		as _render_relay_kit_hint (the students' own winning episodes; the wall's own winners
		preferred, else the best-evidenced tracked target), consumed directly by the system-built
		relay level instead of round-tripping through the FM."""
		log = getattr(self, "_chain_log", None)
		if log is None or not hasattr(log, "latest_fail_summary"):
			return {}
		nb = getattr(self, "_siege_notebook", None)
		candidates: list[str] = [wall]
		if nb is not None and hasattr(nb, "chain_targets"):
			try:
				candidates.extend(t for t in nb.chain_targets().keys() if t != wall)
			except Exception:  # noqa: BLE001
				pass
		best_inv, best_succ = None, -1
		for i, t in enumerate(candidates):
			try:
				entry = log.latest_fail_summary(t)
			except Exception:  # noqa: BLE001
				entry = None
			inv = entry.get("inv") if isinstance(entry, dict) else None
			n_succ = int(entry.get("n_succ", 0) or 0) if isinstance(entry, dict) else 0
			if not inv:
				continue
			if i == 0:
				best_inv = inv
				break
			if n_succ > best_succ:
				best_inv, best_succ = inv, n_succ
		return self._winner_median_kit(best_inv or {})

	# v7fix4 P2 (fable_research_reports/v7fix4真实世界接力与栖息地保真方案.md): relay levels are
	# SYSTEM-BUILT on the real 9-level world generator — the exact distribution held-out evaluates
	# (a FRESH world every episode reset: generate_world receives the reset rng). FM authorship is
	# bypassed for relay levels entirely. Post-mortem: the FM's implicit KPI is trained SR, so any
	# fidelity axis not code-pinned gets 'helpfully' annealed away — v7fix3's levels moved the
	# lizard shallower along with the annealing spawn (trained SR ROSE 76->94->97 as the spawn
	# moved UP; a real ladder must lengthen the chain and dip), and SEWN certified a sandbox that
	# taught nothing transferable (held-out 0, gap 97pp). Instead of pinning axes one by one
	# (target floor, gauntlets, ladders, mob density, ...), the sandbox IS the real world with
	# exactly two sanctioned overrides: the spawn floor (the rung contract) and the starting kit
	# (winner-median; EMPTY at the KIT_STRIP rung, where trained == held-out by construction).
	# v7fix5.3: two more code-pinned scaffold knobs ride the same template — {task_params_args}
	# (needs_depletion_multiplier, the survival-clock anneal) and {build_tail} (either the plain
	# build return, or the UPLOCK block that removes the spawn floor's up-ladder ITEM post-build;
	# game_mechanics' ASCEND requires standing on LADDER_UP, so no item = no escape leg).
	# {constants_import} carries ItemType only when the lock is emitted, keeping every non-locked
	# level's code byte-identical to its pre-5.3 render.
	_RELAY_LEVEL_CODE = '''\
import jax
from craftax.craftax.constants import Achievement{constants_import}

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
    """{docstring}"""

    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.{wall_enum}]
        self.completed_achievements = []
        self.label = "{wall_enum}"

    def get_task_params(self) -> TaskParams:
        return TaskParams({task_params_args})

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, _rng = jax.random.split(rng)
        builder = WorldBuilder(_rng, self.static_params, self.params)
        builder.set_starting_floor({floor}{radius_arg})
{credit_line}{kit_line}{build_tail}
'''

	_RELAY_BUILD_PLAIN = "        return builder.build(rng)"
	_RELAY_BUILD_UPLOCK = (
		"        state = builder.build(rng)\n"
		"        up = builder.ladders_up[{floor}]\n"
		"        state = state.replace(item_map=state.item_map.at[{floor}, up[0], up[1]]"
		".set(ItemType.NONE.value))\n"
		"        return state"
	)

	@staticmethod
	def _quantile_radii(dists, quantiles) -> list[int]:
		"""v7fix5.4: distance samples -> sorted, deduped integer radii (smallest/easiest first).
		Pure math, unit-tested directly; collisions between adjacent quantiles merge (the ladder
		simply gets fewer rungs — _ladder_shape adapts to the resolved length)."""
		import numpy as _np
		d = _np.asarray(dists, dtype=_np.float64)
		return sorted({max(1, int(round(_np.quantile(d, float(q))))) for q in quantiles})

	def _calibrate_rung_radii(self, nb, floor: int, session_idx: int) -> None:
		"""v7fix5.4: measure ``floor``'s entry->down-ladder Manhattan distance distribution on
		freshly generated worlds and persist its quantile radii to the notebook (one-time per
		floor per run). CODE-owned end to end — the modeler never sets a radius (the v7fix3
		law: an FM freedom not pinned by code gets levelled), and the quantiles adapt to each
		floor's ACTUAL distance distribution instead of a hand-calibrated constant table."""
		import numpy as _np
		from craftax.craftax.craftax_state import EnvParams as _EP, StaticEnvParams as _SEP
		from minicraftax.world_builder import WorldBuilder as _WB

		m = int(getattr(nb.th, "rung_calib_samples", 64) or 64)
		qs = tuple(getattr(nb.th, "rung_ladder_quantiles", (0.05, 0.25, 0.50, 0.75, 0.90)))
		_sp, _ep = _SEP(), _EP()

		def _d2d(key):
			b = _WB(key, _sp, _ep)
			return jnp.abs(b.ladders_up[int(floor)] - b.ladders_down[int(floor)]).sum()

		# fixed key: deterministic radii for a given floor (resume-stable, test-stable)
		keys = jax.random.split(jax.random.PRNGKey(54_000 + int(floor)), m)
		try:
			dists = jax.vmap(_d2d)(keys)
		except Exception:  # noqa: BLE001 — vmap can trip on shape-dependent python paths
			dists = jnp.stack([_d2d(k) for k in keys])
		d = _np.asarray(dists, dtype=_np.float64)
		radii = self._quantile_radii(d, qs)
		nb.set_floor_radii(int(floor), radii)
		print(
			f"[siege][RUNG-CALIB] session={session_idx}: floor {int(floor)} entry->down-ladder "
			f"distances over {m} measured worlds P50={_np.median(d):.0f} "
			f"P90={_np.quantile(d, 0.9):.0f} -> quantile radii {radii} "
			f"(quantiles {list(qs)}; code-calibrated, never the modeler's to set)."
		)

	def _relay_level_build(self, wall: str, floor: int, scaffold, stripped: bool):
		"""(docstring, code, meta, stage) for ONE relay rung level from an EXPLICIT knob dict.

		v7fix5.5: extracted from _system_relay_levels so the probe executor renders the SAME
		template for the current stage and for a what-if variant (one knob moved by the
		notebook's step table) — string surgery on level code is how probe variants drift.
		``scaffold`` None = the exact pre-4.6 FULL level; an optional scaffold["pre_light"]
		overrides the light-stamp coupling (probe axis)."""
		kit = {} if stripped else self._relay_kit_dict(wall)
		# v7fix5.8 supply axis: an explicit kit_torches knob overrides the winner-median kit's
		# torch count. torch57 probe (2026-07-19, ckpt18300 paired cells): the dark cliff is a
		# SUPPLY constraint, not a placement-skill gap — kit 10 reads 31.1%, kit 26 reads 52.5%,
		# kit 0 collapses to 3.1% with the policy still pressing PLACE_TORCH on an empty pocket.
		# So the anneal rung that replaces the (doubly broken) fix5.7 light leg moves torch
		# supply, never light stamps. KIT_STRIP exams stay verbatim (held-out semantics).
		_kt58 = (scaffold or {}).get("kit_torches") if not stripped else None
		if _kt58 is not None and kit:
			kit = dict(kit)
			kit["torches"] = int(_kt58)
		kit_line = (
			f"        builder.set_player_inventory({kit!r})\n" if kit else ""
		)
		# v7fix4.6 P1: scaffold sub-stage knobs (None at FULL/kit-strip -> exact pre-4.6
		# level). radius rides set_starting_floor; the clear-gate pre-credit is emitted for
		# floors >= 1 ONLY (monsters_killed[0] inits to 10 — writing a smaller value would
		# LOCK the open overworld ladder). v7fix5.3: uplock removes the spawn floor's
		# up-ladder post-build (floors >= 1 only — floor 0 has no up-ladder) and
		# needs_multiplier < 1 slows the survival clocks via TaskParams; both come from the
		# notebook's stage table, never from the FM.
		radius_arg = ""
		credit_line = ""
		task_params_args = ""
		build_tail = self._RELAY_BUILD_PLAIN
		constants_import = ""
		_uplock = False
		_needs = 1.0
		if scaffold:
			_rad = scaffold.get("down_ladder_radius")
			if _rad is not None:
				radius_arg = f", down_ladder_radius={int(_rad)}"
			# v7fix5.5 probe axis: an explicit pre_light knob decouples the light stamp
			# from the spawn anchor (set_starting_floor kwarg; None = coupled default).
			_pl55 = scaffold.get("pre_light")
			if _pl55 is not None:
				# v7fix5.8: render "ladder" faithfully — bool() coerced it to True, so the
				# fix5.7 49-leg would have rebuilt the identical fully-lit world (torch57
				# also proved that leg semantically empty at the entry anchor: S50==S49
				# byte-for-byte). Bool knobs keep their exact pre-5.8 rendering.
				_pl_repr = "'ladder'" if _pl55 == "ladder" else repr(bool(_pl55))
				radius_arg += f", pre_light={_pl_repr}"
			_cred = int(scaffold.get("monster_credit") or 0)
			if _cred > 0 and int(floor) >= 1:
				credit_line = (
					f"        builder.set_monsters_killed({int(floor)}, {_cred})\n"
				)
			_needs = float(scaffold.get("needs_multiplier", 1.0) or 1.0)
			if _needs < 1.0:
				task_params_args = f"needs_depletion_multiplier={_needs}"
			_uplock = bool(scaffold.get("uplock")) and int(floor) >= 1
			if _uplock:
				build_tail = self._RELAY_BUILD_UPLOCK.format(floor=int(floor))
				constants_import = ", ItemType"
		# v7fix5.5 P0: the lighting fact joins the stage string (docstring + reasoning) —
		# derived from the scaffold dict, same rule as set_starting_floor: a radius spawn
		# torch-lights the spawn and down-ladder 9x9 neighbourhoods, an entry spawn stamps
		# no light. World-rule fact, never a template constant.
		# v7fix5.8 (arm B'): the disclosure must follow the ACTUAL build — an explicit
		# pre_light override decouples the stamp from the anchor (siege_notebook's _lit_clause
		# already obeyed this; THIS string ignored the override, so a radius+dark rung would
		# have LIED "torch-lit" to the journal/docstring — fix9 attribution law).
		_pl55d = scaffold.get("pre_light") if scaffold else None
		_LIT_BOTH_55 = "spawn & down ladder torch-lit (9x9 each)"
		_LIT_NONE_55 = "no scaffold pre-light (the floor's own light only)"
		if _pl55d is None:
			_light55 = (
				_LIT_BOTH_55
				if scaffold and scaffold.get("down_ladder_radius") is not None
				else _LIT_NONE_55
			)
		elif _pl55d == "ladder":
			_light55 = "down ladder torch-lit (9x9), spawn NOT (dark start, lit destination)"
		else:
			_light55 = _LIT_BOTH_55 if _pl55d else _LIT_NONE_55
		stage = (
			"KIT_STRIP exam: natural spawn, EMPTY kit — this world + this start IS the "
			"held-out distribution" if stripped else (
				f"rung floor {int(floor)} scaffold stage {int(scaffold['sub_stage'])} "
				f"(spawn {'within ' + str(scaffold['down_ladder_radius']) + ' of the down ladder' if scaffold.get('down_ladder_radius') is not None else 'at the floor entry'}, "
				f"{_light55}, "
				f"{int(scaffold.get('monster_credit') or 0)}/8 kills pre-credited"
				+ (", up-ladder REMOVED (committed descent — no retreat)" if _uplock else "")
				+ (f", survival clocks at {_needs:.1f}x" if _needs < 1.0 else "")
				+ (f", torch supply {int(_kt58)}" if _kt58 is not None else "")
				+ "), winner-median kit" if scaffold
				else f"rung floor {int(floor)}, winner-median kit"
			)
		)
		docstring = (
			f"Objective: Spawn-anneal relay rung for {wall.upper()} ({stage}).\n"
			f"    Description: The REAL full 9-level Craftax world (fresh world every episode), "
			f"spawn on floor {int(floor)}"
			+ ("" if not kit else f" with a winner-median starting kit {kit!r}")
			+ f"; reach and complete {wall.upper()} from there.\n"
			f"    Relevant Achievements: {wall.upper()}\n"
			f"    Completed Achievements: NONE\n"
			f"    World: standard world generation, starting floor {int(floor)}."
		)
		code = self._RELAY_LEVEL_CODE.format(
			docstring=docstring, wall_enum=wall.upper(), floor=int(floor),
			radius_arg=radius_arg, credit_line=credit_line, kit_line=kit_line,
			task_params_args=task_params_args, build_tail=build_tail,
			constants_import=constants_import,
		)
		meta = {
			"type": "DEPTH", "drill_target": None, "siege_wall": wall,
			"spawn_floor": int(floor), "spawn_kit": (kit or None),
			"system_built": True,
			# v7fix4.6: the rung-reading filter keys off sub_stage too (a stale easier-stage
			# level scoring high must not fake-graduate the harder stage — fix9 #2 family).
			"spawn_sub_stage": int(scaffold["sub_stage"]) if scaffold else 0,
			"spawn_ladder_radius": (scaffold or {}).get("down_ladder_radius"),
			"spawn_monster_credit": (scaffold or {}).get("monster_credit"),
			# v7fix5.3: forensic labels for the two new knobs (graphml attrs skip defaults —
			# set_level_meta guards keep non-regime levels' attribute set unchanged).
			"spawn_uplock": _uplock,
			"spawn_needs_multiplier": (_needs if _needs < 1.0 else None),
			# v7fix5.8: forensic label for the supply axis (None = winner-median, attr skipped).
			"spawn_kit_torches": (int(_kt58) if _kt58 is not None else None),
		}
		return docstring, code, meta, stage

	def _system_relay_levels(self, session_idx: int) -> list[dict]:
		"""v7fix4 P2: build this session's system relay levels (one parsed-proposal dict each).

		Empty list when siege is off / no active relay / the worldgen knob is set back to "fm"
		(the fix3-ablation arm). Level count per wall = ``siege_relay_levels_per_session``
		(default 2 — with per-reset world generation each level is already a DISTRIBUTION, not a
		single instance, so the count only needs to cover the activation lanes)."""
		nb = getattr(self, "_siege_notebook", None)
		if nb is None or not hasattr(nb, "relay_walls"):
			return []
		if str(getattr(getattr(self, "config", None), "siege_relay_worldgen", "base")) != "base":
			return []
		n_per = int(getattr(getattr(self, "config", None), "siege_relay_levels_per_session", 2) or 2)
		# v7fix5.4: quantile ladder — resolve an approach floor's radii from MEASURED worlds
		# BEFORE its first quantile build (no-op when the ladder is off / already calibrated).
		for _wall54 in nb.relay_walls():
			_cf54 = getattr(nb, "rung_calibration_needed", lambda _w: None)(_wall54)
			if _cf54 is not None:
				self._calibrate_rung_radii(nb, int(_cf54), session_idx)
		out: list[dict] = []
		for wall in nb.relay_walls():
			floor = nb.required_spawn_floor(wall)
			if floor is None:
				continue
			stripped = bool(getattr(nb, "relay_kit_stripped", lambda _w: False)(wall))
			scaffold = getattr(nb, "relay_scaffold", lambda _w: None)(wall)
			docstring, code, meta, stage = self._relay_level_build(
				wall, int(floor), scaffold, stripped
			)
			for _i in range(n_per):
				out.append({
					"description": docstring,
					"reasoning": (
						f"system-built relay rung level for {wall} (real-world generator; "
						f"{stage})"
					),
					"level_meta": dict(meta),
					"_system_code": code,
					"_proposer_idx": None,
				})
		if out:
			print(
				f"[siege][RELAY-BUILD] session={session_idx}: {len(out)} system relay level(s) "
				f"built on the REAL world generator for {sorted({p['level_meta']['siege_wall'] for p in out})} "
				f"(spawn floors {sorted({p['level_meta']['spawn_floor'] for p in out})})."
			)
		return out

	def _render_siege_directive(self, latest_profile: dict) -> str:
		"""Text of the current siege focus + still-unmastered links, injected as {SIEGE_DIRECTIVE}.

		Empty string when there is no active focus (siege off / student still early), so proposers see
		nothing and behave as plain v5 coop. When a focus is set, names the wall and the links that
		MUST stay in Relevant (the §3.4 prompt-side ask; the code gate enforces it regardless)."""
		nb = self._siege_notebook
		foci = nb.foci() if nb is not None else []
		if nb is None or not foci:
			return ""
		unmastered = sorted(nb.unmastered_links(latest_profile))
		# §2.6: up to max_focus parallel walls — render each with its own chain.
		if len(foci) == 1:
			lines = [f"ACTIVE SIEGE FOCUS (the hard wall to build toward): {foci[0].get('skill')}"]
		else:
			lines = [f"ACTIVE SIEGE FOCI ({len(foci)} parallel walls to build toward):"]
		# v7fix3.1: when the proposer team is split (P4), tell the SIEGE side explicitly — its
		# persona's generic "cover a TYPE your peer didn't" cooperation rule predates the split
		# and would otherwise divert siege proposals to coverage the ecology role already owns.
		if self._ecology_proposer_idxs():
			lines.append(
				"TEAM NOTE: your peer designer serves the ECOLOGY (the non-siege capability "
				"surface) and cannot tag siege levels — the siege is entirely YOUR "
				"responsibility. Do not divert proposals to general coverage; judge your peer's "
				"levels only to avoid duplicating a specific level, not to avoid a TYPE."
			)
		for foc in foci:
			if len(foci) > 1:
				lines.append(f"WALL: {foc.get('skill')}")
			# v7fix4 post-audit P2 wording: under system-built relay worldgen this wall's own
			# levels are off-limits to the FM (R6_SYSTEM_RELAY), so the chain header and tactic
			# line must not read as "build levels toward this wall" — the proposer's lane is the
			# LINK levels (untagged / other-foci) plus keeping the journal know-how current.
			_sysbuilt_relay = (
				isinstance(foc.get("relay"), dict) and not foc.get("relay_sewn")
				and str(getattr(getattr(self, "config", None),
						"siege_relay_worldgen", "base")) == "base"
			)
			tree = foc.get("prereq_tree", [])
			if tree:
				lines.append(
					"Prerequisite chain (train the still-unmastered LINKS as their own levels; "
					"do NOT tag levels for the wall itself — its rung levels are system-built):"
					if _sysbuilt_relay else
					"Prerequisite chain (train the whole chain up to the wall, don't just gift the last jump):"
				)
				for link in tree:
					# v6fix7 P0.3: the old blanket "mastered — may be scaffolded/compressed" tag
					# directly contradicted the isolation-drill rule (a drill must PERFORM its own
					# chain, mastered or not). Scope the permission to non-drill levels explicitly.
					tag = "STILL-UNMASTERED — MUST be trained (Relevant, NOT Completed)" \
						if link.get("state") != "CONSOLIDATED" \
						else "mastered — may be scaffolded/compressed in DEPTH/BREADTH levels; in a CONSOLIDATE drill of this wall it stays in Relevant and is performed"
					lines.append(f"  - {link.get('skill')} [{tag}] role={link.get('role') or '-'}")
			# §3.1 self-style: hand the proposer the modeler's accumulated ATTACK TACTIC for this wall —
			# the know-how distilled from the student's real winning episodes (action mix / pacing / what
			# was drowning out the target skill). Without this the proposer only knows WHICH wall to build
			# toward, not HOW to shape the level so the wall is actually practised: e.g. that a gear/craft
			# wall needs a zero-mob clean drill with stations adjacent and combat/survival stripped out,
			# so the craft signal isn't drowned by fighting. This is the one place a human player's "go
			# find a safe spot to grind this move" instinct enters the level design.
			style = str(foc.get("style_note", "")).strip()
			if style:
				_how = (
					"use it to shape the LINK levels; the wall's own rung levels are system-built"
					if _sysbuilt_relay else "shape the level to enact it"
				)
				lines.append(
					f"ATTACK TACTIC for {foc.get('skill')} (modeler's know-how from real winning episodes "
					f"— {_how}): {style}"
				)
			# v7fix5.0: ACCESS-LINK CONTRACT — deterministic (not routed through the style_note,
			# which gets rephrased): an enter_* focus trains the DESCENT ITSELF, so its levels must
			# leave the gate to be earned. The s213 root cause: every training level pre-credited
			# the clear-gate / spawned below it, so the floor1->2 grind (where 96% of held-out
			# failures die) was never in anyone's training distribution.
			_fsk = str(foc.get("skill", "")).lower()
			if _fsk.startswith("enter_") and not _sysbuilt_relay:
				lines.append(
					f"ACCESS-LINK CONTRACT for {_fsk}: its levels run from NATURAL spawn "
					f"(spawn_floor 0 — a non-zero spawn_floor is illegal for non-relay levels) and "
					f"the student must EARN the descent: clear the kill quota on the floor above "
					f"the entrance, find the down ladder, descend. Do NOT call "
					f"set_monsters_killed for that floor — pre-crediting the clear-gate trains "
					f"nothing this wall needs (the grind IS the skill). Kit for the approach "
					f"(torches, weapons) is fine."
				)
			# v7 SPAWN-ANNEAL RELAY: the rung contract for this wall's levels — spawn floor is
			# code-checked (R6_SPAWN), the kit evidence comes from the students' own winners.
			if isinstance(foc.get("relay"), dict) and not foc.get("relay_sewn"):
				_r = foc["relay"]
				_sf = int(_r.get("spawn_floor", 0))
				if _sysbuilt_relay:
					# v7fix4 P2: relay levels are SYSTEM-BUILT (real world generator + rung spawn +
					# winner-median kit) — the proposer must NOT author levels for this wall; its
					# fire belongs to the other foci/links. R6 still rejects any stray attempt.
					# v7fix4.6: the scaffold sub-stage is rendered for transparency (the ladder is
					# code-driven; the FM decides nothing here, it just should not be surprised
					# that trained SR jumps when the stage steps).
					_stage46 = int(_r.get("sub_stage", 0) or 0)
					# v7fix5.3: transparency for the descent-regime knobs too (still code-driven —
					# the proposer decides nothing, it just should not misread the rung's world).
					_sc53p = (
						getattr(self._siege_notebook, "relay_scaffold", lambda _w: None)(
							str(foc.get("skill", "")).lower()
						) or {}
					)
					_knob53p = ""
					if _sc53p.get("uplock") or float(_sc53p.get("needs_multiplier", 1.0)) < 1.0:
						_knob53p = (
							f" [descent regime: up-ladder removed, survival clocks at "
							f"{float(_sc53p.get('needs_multiplier', 1.0)):.1f}x]"
						)
					_stage_txt = (
						f", scaffold sub-stage {_stage46} of the descent ladder{_knob53p}"
						if _stage46 else ""
					)
					lines.append(
						f"★SPAWN-ANNEAL RELAY for {foc.get('skill')} (SYSTEM-BUILT): the system "
						f"itself builds this wall's rung levels on the REAL full-world generator "
						f"(current rung: spawn floor {_sf}{_stage_txt}) — do NOT author levels targeting this "
						f"wall; any you propose will be rejected by the spawn contract. Spend your "
						f"siege fire on the OTHER foci and still-unmastered links. Held-out SR for "
						f"this wall is EXPECTED to read 0 until the ladder anneals back to natural "
						f"spawn and passes the kitless exam."
					)
				else:
					lines.append(
						f"★SPAWN-ANNEAL RELAY for {foc.get('skill')} (code-enforced): every level "
						f"targeting this wall MUST spawn the student on FLOOR {_sf} (declare "
						f'"spawn_floor": {_sf} in <level_meta>; any other value is rejected) and '
						f"must contain the COMPLETE chain from that spawn to the target — descend, "
						f"clear, fight; no skipped floors, no teleports past content. Give a starting "
						f"kit (\"spawn_kit\") based on the winner-median stockpiles below. Held-out "
						f"SR for this wall is EXPECTED to read 0 until the relay anneals back to "
						f"natural spawn; the rung's TRAINED SR is what graduates it."
					)
					kit_hint = self._render_relay_kit_hint(str(foc.get("skill", "")).lower())
					if kit_hint:
						lines.append(kit_hint)
			# v6fix7 P1a: the escalation ladder binds the PROPOSER too. At L2+ the attack form for
			# this wall is FORCED (the other form froze); the validator rejects a siege level for
			# this wall built in the frozen form.
			lvl = int(foc.get("ladder_level", 0) or 0)
			req = nb.required_form(str(foc.get("skill", "")))
			if req and foc.get("gap_forced"):
				# v6fix8 ③: the gap gate fired — different reason than a ladder freeze, same channel.
				lines.append(
					f"★GAP GATE: REQUIRED FORM for {foc.get('skill')} = {req}. Its drills are won "
					"in their calm sandbox but held-out is NOT following — isolation drills are "
					f"suspended; if your level targets this wall you MUST build a {req} level under "
					"FULL pressure (mobs/night/hunger present, no stripped sandbox). This overrides "
					"MODELER_GUIDANCE and is code-enforced."
				)
			elif req:
				lines.append(
					f"★LADDER (frozen {foc.get('frozen_sessions', 0)} sessions): REQUIRED FORM for "
					f"{foc.get('skill')} = {req}. The previous form froze; if your level targets this "
					f"wall you MUST build a {req} level (this overrides MODELER_GUIDANCE and is "
					f"code-enforced)."
				)
			elif lvl >= 1:
				lines.append(
					f"LADDER NOTE: {foc.get('skill')} has been frozen {foc.get('frozen_sessions', 0)} "
					"session(s) (whole attack tree, no progress). Prefer a materially different level "
					"design for it this round — more of the same has not moved it."
				)
		if unmastered:
			lines.append(
				"FORBIDDEN in Completed this level (unmastered links — put them in Relevant): "
				+ ", ".join(u.upper() for u in unmastered)
			)
		# v6fix8 ④: announce the extra-seat budget — fix7's proposers tagged 22/24 candidates siege
		# and monopolised selection; with extra seats the main selection stays a pure merit
		# competition and over-tagging just wastes proposals (siege losers beyond the cap are dropped).
		# Defensive config access: this renderer is also exercised on bare TaskGenerator test doubles.
		cap = int(getattr(getattr(self, "config", None), "siege_select_cap", 8) or 0)
		if cap > 0:
			lines.append(
				f"SEAT BUDGET: every level competes for the main selection seats on merit; "
				f"siege-tagged levels that lose are rescued onto at most {cap} EXTRA seats "
				"(dealt round-robin across the foci) and any beyond that are DROPPED — a "
				"siege-tagged loser beyond the cap is a wasted proposal, so do not tag more "
				"levels than the seats can carry."
			)
		return "\n".join(lines)

	def _run_cross_rating(
		self, proposals: list, global_agent_profile: dict | None = None
	) -> dict[str, dict[str, float]]:
		"""One round of Endorsement: each Proposer rates the OTHER Proposers' proposals in [0,1].

		Returns cross_ratings[rater_proposer_id][proposal_id] = score. A Proposer never rates its own
		proposals (enforced downstream by endorsement_scores exclude_self, and here we simply skip
		them in the prompt). This is the multi-FM "market" signal a single FM cannot produce; a
		proposal endorsed by the majority of the OTHER models scores high, filtering idiosyncratic
		proposals. Cost = N LLM calls per round (one batched rating call per proposer).

		v2 (prompt设计稿_v2.md §4): NEUTRAL 3-criteria rubric (solvable-now / well-targeted /
		useful-toward-mastery) with the student profile injected, so ratings are stabler and account
		for the student's current level. Deliberately NO persona voice for raters (avoids a
		"hard-proposer systematically down-votes safe levels" bias; heterogeneity comes from the
		different base models).

		Robust to parse failures: any proposal a rater doesn't score just gets no vote from that rater
		(endorsement_scores averages over the votes that exist).
		"""
		import json

		profile_str = self._format_global_agent_profile(global_agent_profile)
		rater_prompt_sys = (
			"You are an expert reviewer on a curriculum-design team for a Craftax RL agent. Other "
			"designers proposed the candidate training levels below (NOT your own). Score each on how "
			"good a NEXT training level it is for THIS agent right now, using THREE explicit criteria, "
			"then give one overall score.\n\n"
			"Criteria (judge each, then combine into one overall score):\n"
			"  1. SOLVABLE-NOW: can the current agent plausibly complete it? (unsolvable/way-too-hard -> low)\n"
			"  2. WELL-TARGETED: is it aimed at a skill the agent is ready to learn (not already "
			"mastered, not hopelessly far)? (mistargeted -> low)\n"
			"  3. USEFUL-TOWARD-MASTERY: does clearing it move the agent toward mastering full Craftax "
			"(a real stepping stone, not a distractor)? (distracting -> low)\n\n"
			"Here is the agent's current performance profile (use it for criteria 1 & 2):\n"
			f"<agent_profile>\n{profile_str}\n</agent_profile>\n\n"
			"Return ONLY a JSON object mapping each proposal id to its OVERALL score in [0,1], e.g. "
			"{\"prop_s1_3\": 0.7, \"prop_s1_5\": 0.2}. No other text."
		)

		cross_ratings: dict[str, dict[str, float]] = {}
		for rater_idx, rater in enumerate(self.proposer_llms):
			rater_id = f"proposer_{rater_idx}"
			# Show only the OTHER proposers' proposals to this rater.
			others = [p for p in proposals if p.proposer_id != rater_id]
			if not others:
				continue
			listing = "\n\n".join(
				f"[id: {p.proposal_id}]\n{p.docstring}" for p in others
			)
			user = (
				f"Rate the following {len(others)} candidate levels on the three criteria and return "
				f"ONLY the JSON object of id -> overall score in [0,1].\n\n{listing}"
			)
			prev_llm = self.llm
			self.llm = rater
			try:
				resp = self.llm.query(rater_prompt_sys, [user])
			finally:
				self.llm = prev_llm

			content = (resp[0].get("content") or "") if resp else ""
			scores: dict[str, float] = {}
			# Extract the first JSON object in the response.
			match = re.search(r"\{.*\}", content, re.DOTALL)
			if match:
				try:
					raw = json.loads(match.group(0))
					valid_ids = {p.proposal_id for p in others}
					for pid, sc in raw.items():
						if pid in valid_ids:
							try:
								v = float(sc)
							except (TypeError, ValueError):
								continue
							scores[pid] = min(1.0, max(0.0, v))
				except json.JSONDecodeError:
					pass
			if scores:
				cross_ratings[rater_id] = scores
			print(f"[auction] cross-rating: {rater_id} scored {len(scores)}/{len(others)} others.")

		return cross_ratings if cross_ratings else None

	def _build_mastered_prompts(
		self,
		mastered_tasks: list[str],
		global_agent_profile: dict | None,
		prompt_module=None,
		archive_family_coverage: str | None = None,
		extra_fields_per_parent: dict[str, dict] | None = None,
	) -> tuple[list[str], list[list[str]], list[list[str]]]:
		"""Builds the per-parent user prompts.

		Args:
			prompt_module: which prompt module supplies user_prompt. Defaults to the shared
				evolve_mastered_prompt (baseline). For v2 personas, pass that proposer's persona
				module so its persona-specific user_prompt template is used.
			archive_family_coverage: only the Breadth persona's user_prompt has an
				{ARCHIVE_FAMILY_COVERAGE} placeholder; pass the pre-computed tally string for it.
				Other personas' templates don't reference it, so it's harmlessly ignored.
			extra_fields_per_parent: v5-debate coop path — {parent_id: {FIELD: value}} extra template
				fields (MODELER_GUIDANCE / PEER_ALREADY_MADE / REFERENCE_LEVEL / MY_TURN_ORDER) merged
				into the fields dict for that parent. _safe_format ignores any a template doesn't use,
				so non-coop personas are unaffected. Defaults to None (unchanged v4 behaviour).
		"""
		module = prompt_module or self.evolve_mastered_prompt
		user_prompts: list[str] = []
		parent_sets: list[list[str]] = []
		example_sets: list[list[str]] = []
		global_profile_str = self._format_global_agent_profile(global_agent_profile)
		# NOTE (v3, 2026-07-02): the prompt-side ability gate is GONE. Persona templates no longer
		# carry an {ABILITY_GATE} placeholder — over-reach control is bid-side only (ambition.py's
		# reachable_ceiling discount), matching baseline's plain evolve prompt (no tier constraint).

		for mastered_task in mastered_tasks:
			task_examples = self.selector.select_similar_desc_tasks(
				mastered_task,
				statuses=self._get_valid_parent_statuses(),
				num_examples=self.config.num_examples,
			)
			parent_sets.append([mastered_task])
			example_sets.append(task_examples)
			example_str = self._format_file_mastered_task([mastered_task])
			task_performance_str = self._get_task_performance_str(mastered_task)
			# Provide every field any persona template might reference; _safe_format only
			# substitutes the ones actually present in this module's template.
			fields = {
				"MASTERED_TASK": example_str,
				"TASK_PERFORMANCE_CONTEXT": task_performance_str,
				"GLOBAL_AGENT_PROFILE": global_profile_str,
				"ARCHIVE_FAMILY_COVERAGE": archive_family_coverage or "",
				# Only ambitious's template references this; _safe_format drops it for the others.
				"PARENT_CHILD_HISTORY": self._format_parent_child_history(mastered_task),
				# v5 coop fields default to empty; the coop path overrides per parent below.
				"MODELER_GUIDANCE": "",
				"PEER_ALREADY_MADE": "",
				"REFERENCE_LEVEL": "",
				"MY_TURN_ORDER": "",
				# v6 siege field (§3.4): current focus + links that must NOT be compressed. Empty
				# unless the siege path overrides it per parent; _safe_format drops it for non-siege
				# personas. Empty == baseline scaffold behaviour (no siege directive).
				"SIEGE_DIRECTIVE": "",
				# v7fix3 P4: ecology brief — empty unless the coop siege path fills it per parent.
				"ECOLOGY_DIRECTIVE": "",
				# v6fix7 P0.1: <level_meta> output spec. Empty unless the siege path overrides it
				# (siege off -> placeholder renders empty -> prompt byte-unchanged).
				"LEVEL_META_SPEC": "",
			}
			if extra_fields_per_parent and mastered_task in extra_fields_per_parent:
				fields.update(extra_fields_per_parent[mastered_task])
			user_prompts.append(self._safe_format(module.user_prompt, fields))
		return user_prompts, parent_sets, example_sets

	def _compute_archive_family_coverage(self) -> str:
		"""Tally how many archive levels teach each skill family (for the Breadth persona).

		Scans every archive node's docstring, parses its 'Relevant Achievements', maps each to a
		family (COMBAT/GATHER/CRAFT/EXPLORE), and counts a family once per level that touches it.
		Returns a compact string like 'COMBAT: 12 | GATHER: 3 | CRAFT: 8 | EXPLORE: 1'. Pure code,
		zero LLM. Empty archive -> all zeros.
		"""
		from auction.craftax_achievements import FAMILIES, family_of
		from .auction_integration import parse_relevant_achievements

		counts = {fam: 0 for fam in FAMILIES}
		for _node, data in self.archive.graph.nodes(data=True):
			desc = data.get("description", "") or ""
			achs = parse_relevant_achievements(desc)
			if not achs:
				continue
			fams_here = {family_of(a) for a in achs}
			for fam in fams_here:
				counts[fam] += 1
		return " | ".join(f"{fam}: {counts[fam]}" for fam in FAMILIES)

	def _build_parent_learnability(self, proposals: list) -> dict[str, float]:
		"""Map each candidate's parent_task_id -> that parent's stored learnability p*(1-p).

		Official DiCode learnability is a TRAINING by-product: a task's p (success rate) is measured
		while it trains, and learnability = p*(1-p) is stored on its archive node
		(update_node_learnability -> 'learnability_score'; falls back to 'priority_score'). A fresh
		candidate has no p of its own, so its Learnability bid proxies its PARENT's stored value.
		Parents never trained (e.g. seeds) are omitted -> Learnability contributes 0 for those.
		Pure archive read, no GPU / no rollout.
		"""
		result: dict[str, float] = {}
		parent_ids = {p.parent_task_id for p in proposals if p.parent_task_id}
		for pid in parent_ids:
			try:
				if not self.archive.graph.has_node(pid):
					continue
				node = self.archive.graph.nodes[pid]
			except Exception:
				continue
			val = node.get("learnability_score", None)
			if val is None:
				val = node.get("priority_score", None)
			if val is None:
				continue
			try:
				result[pid] = float(val)
			except (TypeError, ValueError):
				continue
		return result

	@staticmethod
	def _safe_format(template: str, fields: dict) -> str:
		"""str.format that only substitutes placeholders actually present in the template.

		A persona user_prompt may reference a subset of fields (e.g. Breadth uses
		ARCHIVE_FAMILY_COVERAGE, others don't; reward-mode omits TASK_PERFORMANCE_CONTEXT). Passing
		a superset of fields to str.format is fine (extra kwargs ignored), but a template field with
		no matching kwarg raises KeyError — here every known field is always supplied, so this is a
		thin wrapper that also tolerates literal braces defensively.
		"""
		try:
			return template.format(**fields)
		except KeyError as e:
			# A template referenced an unknown field: fill it blank and retry (robustness).
			missing = str(e).strip("'")
			fields = {**fields, missing: ""}
			return template.format(**fields)

	def _get_valid_parent_statuses(self) -> list[str]:
		"""Returns the list of task statuses that are valid for selecting examples."""
		return [
			"seed",
			"interesting",
			"desc_generated",
			"compile_success",
			"mastered",
			"learnable",
			"unlearnable",
			"A",
			"B",
			"C",
			"D",
			"example",
		]

	def _get_task_performance_str(self, task_id: str) -> str:
		"""Retrieves and formats the performance history for a specific task."""
		try:
			task_data = self.archive.graph.nodes[task_id]
			performance_history = task_data.get("performance_history", [])
			if performance_history:
				task_specific_profile = performance_history[-1]
				return self._format_task_performance_context(task_specific_profile)
			return "No specific performance data found for this task."
		except Exception as e:
			print(f"Warning: Could not retrieve performance history for {task_id}: {e}")
			return f"Error retrieving performance data: {e}"

	def _build_system_prompt(self, prompt_module) -> str:
		"""Builds the system prompt for task generation LLM calls."""
		return prompt_module.system_prompt.format(
			CONSTANTS=CONSTANTS,
			MOBS=MOBS,
			GAME_MECHANICS=GAME_MECHANICS,
			WORLD_GEN=WORLD_GEN,
			API_DOCS=API_DOCS,
		)

	def _siege_validate_and_reroll(
		self,
		parsed_responses: list,
		user_prompts: list[str],
		system_prompt: str,
		p_sets: list[list[str]],
		proposer_idx: int,
		siege_unmastered: set,
		max_rerolls: int = 2,
	) -> list:
		"""v6fix7 P0.2 — run the SiegeLevelValidator on each parsed proposal; reroll violators.

		Called INSIDE the coop loop's try block (self.llm is still routed to the current proposer).
		``parsed_responses`` must be ALIGNED with ``user_prompts`` (None placeholders allowed).
		Violating proposals are re-queried with explicit violation feedback appended to their own
		user prompt (at most ``max_rerolls`` rounds); leftovers get the mechanical fallback
		(Completed->Relevant moves) + a WARN. Strict no-op when siege is off or no focus is active,
		so the baseline / v5y path is unchanged.
		"""
		notebook = getattr(self, "_siege_notebook", None)
		if notebook is None or not parsed_responses:
			return parsed_responses
		_foci_attr = getattr(notebook, "foci", None)
		foci = _foci_attr() if callable(_foci_attr) else (_foci_attr or [])
		# v7fix3 P6: NO early-exit on empty foci — R6's spawn contract is always-on. v7fix2's
		# dormant phase (foci=[]) skipped validation entirely, which is how 33 unchecked deep-spawn
		# levels slipped through; the lane is legal now but BOUNDED (BREADTH + frontier + quota),
		# and the bound only exists if the validator actually runs. With foci=[] every other rule
		# is a structural no-op inside validate_level, so non-spawn behaviour is unchanged.

		from auction.level_validator import (
			RULE_SYS_RELAY,
			apply_fallback_fixes,
			render_violation_feedback,
			reroll_worthy,
			validate_level,
		)

		try:
			from .auction_integration import parse_relevant_achievements
		except ImportError:  # spec-from-file loads (unit tests) have no package parent
			from dicode.dreaming.auction_integration import parse_relevant_achievements

		def _parent_relevant(i: int) -> set | None:
			pid = p_sets[i][0] if i < len(p_sets) and p_sets[i] else None
			if not pid:
				return None
			try:
				desc = self.archive.get_task_descriptions([pid]).get(pid) or ""
			except Exception:
				return None
			achs = parse_relevant_achievements(desc)
			return set(achs) if achs else None

		# v6fix7 P1a L2: ladder-forced attack forms for frozen walls (empty dict when no wall is at L2+).
		required_forms = {}
		for _foc in foci:
			_skill = str(_foc.get("skill", ""))
			_req = notebook.required_form(_skill)
			if _req:
				required_forms[_skill.lower()] = _req
		# v7: relay rung spawn-floor contracts (empty dict when no relay is live).
		required_spawn_floors = {}
		_rsf = getattr(notebook, "required_spawn_floor", None)
		if callable(_rsf):
			for _foc in foci:
				_skill = str(_foc.get("skill", "")).lower()
				_floor = _rsf(_skill)
				if _floor is not None:
					required_spawn_floors[_skill] = int(_floor)
		# v7fix3 P6: the deepest floor an untagged BREADTH level may spawn on (floor 1 minimum).
		_bf_attr = getattr(notebook, "breadth_frontier", None)
		breadth_frontier = int(_bf_attr()) if callable(_bf_attr) else 1
		# v7fix4 post-audit hardening: under system-built relay worldgen the relay walls take no
		# FM levels AT ALL (R6_SYSTEM_RELAY) — plain R6 only rejects a mismatched floor, and the
		# directive prints the current rung floor, so a disobedient proposer could otherwise author
		# a passing level and eat the wall's 2 discounted force-activation slots (quarantined from
		# rung evidence but still crowding out the system-built levels). Empty on the "fm" arm.
		system_relay_walls: set[str] = (
			set(required_spawn_floors)
			if str(getattr(getattr(self, "config", None), "siege_relay_worldgen", "base")) == "base"
			else set()
		)

		current = list(parsed_responses)
		for round_i in range(max_rerolls + 1):
			pending: list[tuple[int, list]] = []
			for i, parsed in enumerate(current):
				if not isinstance(parsed, dict) or not parsed.get("description"):
					continue
				violations = validate_level(
					parsed["description"],
					parsed.get("level_meta"),
					foci,
					unmastered=siege_unmastered,
					parent_relevant=_parent_relevant(i),
					required_forms=required_forms,
					required_spawn_floors=required_spawn_floors,
					breadth_frontier=breadth_frontier,
					system_relay_walls=system_relay_walls,
				)
				if violations and reroll_worthy(violations):
					pending.append((i, violations))
				elif violations:
					for v in violations:
						print(
							f"[siege][validator] proposer_{proposer_idx} warn ({v.rule}): {v.message}"
						)
			if not pending:
				return current
			if round_i == max_rerolls:
				for i, violations in pending:
					fixed, moved = apply_fallback_fixes(current[i]["description"], violations)
					if moved:
						current[i]["description"] = fixed
					# v7fix4: the mechanical fix for a persistent R6_SYSTEM_RELAY violation is a
					# TAG STRIP — the level survives as an ordinary (untagged) candidate that
					# competes on merit, but it can no longer draw the relay wall's rescue seats
					# or its discounted force-activation slots. The description edit above cannot
					# fix a tag; without this the fallback-accept would keep the crowd-out open.
					if any(v.rule == RULE_SYS_RELAY for v in violations):
						_meta = current[i].get("level_meta")
						if isinstance(_meta, dict):
							_stripped = {
								k: _meta.pop(k, None) for k in ("siege_wall", "drill_target")
								if str(_meta.get(k) or "").lower() in system_relay_walls
							}
							if _stripped:
								print(
									f"[siege][validator] proposer_{proposer_idx} DEMOTED level "
									f"{i}: siege tags {sorted(_stripped)} stripped — its wall's "
									f"rung levels are system-built; the level stays as an "
									f"ordinary candidate."
								)
					rules = ",".join(sorted({v.rule for v in violations}))
					print(
						f"[siege][validator] proposer_{proposer_idx} FALLBACK after {max_rerolls} "
						f"rerolls (rules={rules}; moved={moved}) — accepted with mechanical fix only."
					)
				return current
			idxs = [i for i, _ in pending]
			reroll_prompts = [
				user_prompts[i] + render_violation_feedback(violations) for i, violations in pending
			]
			print(
				f"[siege][validator] proposer_{proposer_idx} reroll {round_i + 1}/{max_rerolls} for "
				f"{len(idxs)} level(s); rules="
				f"{sorted({v.rule for _, violations in pending for v in violations})}"
			)
			replacements = self._query_and_parse_responses(
				system_prompt,
				reroll_prompts,
				max_retries=3,
				require_level_meta=True,
				return_aligned=True,
			)
			for j, rep in enumerate(replacements):
				if isinstance(rep, dict) and rep.get("description"):
					current[idxs[j]] = rep
				# else: keep the old (violating) version — the fallback round will fix what it can.
		return current

	def _query_and_parse_responses(
		self,
		system_prompt: str,
		user_prompts: list[str],
		max_retries: int = 10,
		require_level_meta: bool = False,
		return_aligned: bool = False,
	) -> list[dict]:
		"""Queries the LLM and parses responses with a retry loop for failed parses.

		Args:
			system_prompt: The system prompt for the LLM.
			user_prompts: List of user prompts to send to the LLM.
			max_retries: Maximum number of retry attempts for failed parses.
			require_level_meta: v6fix7 P0.1 — when True (siege sessions, where the prompt contains
				the {LEVEL_META_SPEC} block), a response without a usable <level_meta> block counts
				as a parse failure and is re-queried, so TYPE-dependent enforcement always has data.
				False (default) keeps every non-siege call site byte-for-byte unchanged.
			return_aligned: v6fix7 P0.2 — when True, return a list the SAME length/order as
				``user_prompts`` with None placeholders for permanently-failed parses, so callers
				(validator reroll, p_sets pairing) can index prompts/parents safely. Default False
				keeps the historical filtered return for all existing call sites.

		Returns:
			A list of successfully parsed response dictionaries (filtered), or the aligned list
			with None placeholders when ``return_aligned`` is True.
		"""
		from auction.level_meta import level_meta_complete

		responses = self.llm.query(system_prompt, user_prompts)

		final_parsed_responses = [None] * len(responses)
		indices_to_retry = list(range(len(responses)))

		for attempt in range(max_retries):
			for i, original_index in enumerate(indices_to_retry):
				response = responses[i]
				parsed_data = self._parse_generation_response(response)
				if parsed_data.get("description") is not None and (
					not require_level_meta or level_meta_complete(parsed_data.get("level_meta"))
				):
					final_parsed_responses[original_index] = parsed_data

			indices_to_retry = [
				i for i, result in enumerate(final_parsed_responses) if result is None
			]

			if not indices_to_retry:
				print("Successfully parsed all LLM responses.")
				break

			if attempt < max_retries - 1:
				print(
					f"Failed to parse {len(indices_to_retry)} responses. "
					f"Retrying (Attempt {attempt + 2}/{max_retries})..."
				)
				prompts_to_retry = [user_prompts[i] for i in indices_to_retry]
				responses = self.llm.query(system_prompt, prompts_to_retry)
			else:
				print(
					f"Warning: Failed to parse {len(indices_to_retry)} responses "
					f"after {max_retries} attempts."
				)

		if return_aligned:
			return final_parsed_responses
		return [res for res in final_parsed_responses if res is not None]

	def _format_file_mastered_task(self, example_paths: list[str]) -> str:
		"""Formats a parent task description into a string for the LLM prompt."""
		descriptions = self.archive.get_task_descriptions(example_paths)
		return "\n".join([f"\n{desc}\n" for desc in descriptions.values()])

	def _format_global_agent_profile(self, evaluation_feedback: dict | None) -> str:
		"""Formats the global evaluation metrics into a string for the LLM prompt.

		v4 (2026-07-03): REVERTED to the plain DiCode baseline formatter — a flat list of the skills
		the agent has *demonstrated* (SR > 0), NO tier grouping, NO explicit 0% listing, NO tier
		semantic labels. Rationale: the v2/v3 per-tier formatter (which listed every achievement and
		labelled which depth tier each belongs to, e.g. "TIER 3: diamond gear / gnome-orc combat /
		magic") was a KNOWLEDGE LEAK — it handed the LLM Craftax's depth structure that baseline never
		gets. To keep the C-arm comparable to baseline AND to make curriculum shape emerge from the
		mechanism + the student's own training feedback (not from injected priors), we give the same
		neutral profile baseline gives. The ambitious persona instead learns "what failed from here"
		from the prior-children training outcomes ({PARENT_CHILD_HISTORY}), a real feedback signal, not
		an injected depth map.

		Args:
			evaluation_feedback: Dictionary of skill metrics from the agent's evaluation on the full
				Craftax game (keys like ``skill_collect_wood`` in 0..100).

		Returns:
			A formatted string describing the agent's demonstrated skill profile.
		"""
		if not evaluation_feedback:
			return "No overall performance evaluation is available for the agent yet."

		context_str = "This is the agent's *general* skill profile from the full Craftax game:\n"
		skill_lines = [
			f"- {key[len('skill_'):]}: {value:.2f}%"
			for key, value in evaluation_feedback.items()
			if key.startswith("skill_") and value > 0
		]
		if skill_lines:
			context_str += (
				"Its success rates on the skills it has shown are:\n" + "\n".join(skill_lines) + "\n"
			)
		else:
			context_str += "No skills have been demonstrated in the global evaluation yet.\n"
		return context_str

	def _format_parent_child_history(self, mastered_task: str) -> str:
		"""Levels already evolved FROM this parent + their trained SR (ambitious persona only).

		v4 (2026-07-03): the real feedback signal that replaces the injected tier map. For the parent
		task being evolved, list its children in the archive and each child's last trained success
		rate. A child whose SR stayed near zero marks a direction that FAILED from here — the student
		could not reach the situation that level required (the tier2->tier3 "dependency collapse":
		experiment_design.md §11.7 E). The ambitious persona uses this to re-aim shallower / scaffold
		rather than re-issue an unlearnable reach. Only ambitious's user_prompt has the
		{PARENT_CHILD_HISTORY} placeholder; the other personas' _safe_format silently drops this field.
		"""
		import json as _json

		g = getattr(self.archive, "graph", None)
		if g is None or not g.has_node(mastered_task):
			return "No levels have been evolved from this parent yet."
		from .auction_integration import parse_relevant_achievements

		K = 10  # show the last K trained-SR readings per child (v3c ph-length: p90=10, K=10 fully
		# shows 90% of children; the rest are truncated to their most recent 10 sessions — enough to
		# read the current trend without dumping ancient early history).
		lines: list[str] = []
		for child in g.successors(mastered_task):
			data = g.nodes[child]
			ph = data.get("performance_history")
			try:
				ph = _json.loads(ph) if isinstance(ph, str) else (ph or [])
			except (TypeError, ValueError):
				ph = []
			# full SR sequence over sessions (skip records with no sr)
			srs = [r.get("sr") for r in ph if isinstance(r, dict) and r.get("sr") is not None]
			if not srs:
				continue
			achs = parse_relevant_achievements(data.get("description", "") or "")
			goal = ", ".join(sorted(achs)) if achs else "unspecified goal"
			shown = srs[-K:]
			seq = ", ".join(f"{s * 100:.0f}%" for s in shown)
			prefix = "..., " if len(srs) > K else ""
			# Report the raw trained-SR TIME SERIES (oldest->newest), NO verdict. The proposer reads
			# the trend itself: a direction still climbing from zero is being learned (not failed);
			# only one flat near zero across sessions is genuinely unlearnable from here. Handing it a
			# hard "FAILED"/"learned" label (as v2/v3 did on the last reading alone) mis-flags a
			# late-blooming direction whose early readings are still 0 — see experiment_design.md §11.7.
			lines.append(
				f"  - child targeting [{goal}]: trained SR per session (oldest->newest) = [{prefix}{seq}]"
			)
		if not lines:
			return "No levels have been evolved from this parent yet."
		return (
			"Levels already evolved from this parent, and how the agent trained on each across "
			"sessions. Each is a TIME SERIES of trained success rate (oldest first). Read the trend "
			"yourself — do not judge a direction on a single low reading:\n" + "\n".join(lines)
		)


	# v7fix4.7 Q4 (2026-07-14 forensics lesson): craftax's calculate_inventory_achievements runs
	# at RESET, so a spawn kit flips every inventory-derived achievement to ~100% before the first
	# step, and reset also sets enter_* from the spawn floor. On a kitted / deep-spawn task these
	# ~100% rows are ARTIFACTS, not behaviour — they poisoned the 2026-07-14 diagnosis twice
	# (both "no lighting habit" and "flees to the surface" were read off pre-credited rows).
	_PRECREDITABLE_ACHIEVEMENTS = frozenset({
		# inventory-derived (kit items flip these at reset):
		"collect_wood", "collect_stone", "collect_coal", "collect_iron", "collect_diamond",
		"collect_ruby", "collect_sapphire", "collect_sapling",
		"make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe", "make_diamond_pickaxe",
		"make_wood_sword", "make_stone_sword", "make_iron_sword", "make_diamond_sword",
		"make_torch", "make_arrow", "find_bow",
		# floor-entry flags (reset sets enter_* for every floor at/above the spawn floor):
		"enter_dungeon", "enter_gnomish_mines", "enter_sewers", "enter_vault",
		"enter_troll_mines", "enter_fire_realm", "enter_ice_realm", "enter_graveyard",
	})

	def _format_task_performance_context(self, task_profile: dict) -> str:
		"""Formats the task-specific metrics into a string for the LLM prompt."""
		if not task_profile:
			return "No task-specific performance data available."

		context_str = "While training on this task, the agent achieved:\n"

		# Add the main SR for the task's goal (composite success)
		if "sr" in task_profile:
			# 'sr' is a 0-1 ratio, so MULTIPLY by 100
			sr_percent = task_profile["sr"] * 100.0
			context_str += f"- Main Goal Success Rate (SR): {sr_percent:.2f}%\n"

		# Add the SR for ALL individual achievements (goal-related and accidental)
		if "achievement_srs" in task_profile and task_profile["achievement_srs"]:
			context_str += "Detailed Skill SRs (including goal components and accidental skills):\n"
			dropped = []
			for key, value in task_profile["achievement_srs"].items():
				# v7fix4.7 Q4: a pre-creditable achievement pinned at ~100% carries no
				# behavioural signal — drop the row and say so, or the LLM reads kit
				# artifacts as skills (the 2026-07-14 mis-diagnosis pair).
				try:
					_v = float(value)
				except (TypeError, ValueError):
					_v = None
				if _v is not None and _v >= 99.5 and str(key) in self._PRECREDITABLE_ACHIEVEMENTS:
					dropped.append(str(key))
					continue
				# 'value' from achievement_srs is ALREADY 0-100, so DO NOT multiply
				context_str += f"  - {key}: {value:.2f}%\n"
			if dropped:
				context_str += (
					"  (omitted as reset-time pre-credit artifacts, ~100% by construction "
					"on this task's kit/spawn — NOT behaviour: " + ", ".join(sorted(dropped)) + ")\n"
				)
		else:
			context_str += "No detailed skill SRs were recorded.\n"

		return context_str

	def _parse_generation_response(self, response: dict) -> dict:
		"""Parses the LLM's response to extract both the reasoning and the docstring.
		Returns a dictionary with both fields.
		"""
		response_content = response.get("content", "")

		# Default values in case parsing fails
		parsed_data = {"reasoning": None, "description": None}

		# Extract reasoning
		if response_content is None:
			return parsed_data

		reasoning_match = re.search(
			r"<reasoning>\s*(.*?)\s*</reasoning>", response_content, re.DOTALL
		)
		if reasoning_match:
			parsed_data["reasoning"] = reasoning_match.group(1).strip()

		# Extract task description (docstring)
		desc_match = re.search(r"<docstring>\s*(.*?)\s*</docstring>", response_content, re.DOTALL)
		if desc_match:
			parsed_data["description"] = desc_match.group(1).strip()

		# v6fix7 P0.1: machine-readable level metadata. Tolerant — siege-off responses were never
		# asked for the block, so absence is normal and parses to None (baseline path unchanged).
		from auction.level_meta import parse_level_meta

		parsed_data["level_meta"] = parse_level_meta(response_content)

		return parsed_data

	def _organize_data(
		self,
		parsed_responses: list[dict],
		parent_sets: list[list[str]],
		example_sets: list[list[str]],
		session_idx: int,
		evolution_type: str,
	) -> list[dict]:
		"""Organizes parsed LLM responses into generation result dictionaries.

		Records new tasks in the archive and builds the result structure
		for downstream processing.

		Args:
			parsed_responses: List of parsed LLM response dictionaries.
			parent_sets: List of parent task ID lists for each response.
			example_sets: List of example task ID lists for each response.
			session_idx: Current curriculum session index.
			evolution_type: Type of evolution (e.g., 'mastered').

		Returns:
			A list of generation result dictionaries.
		"""
		generation_results = []
		for i, parsed_data in enumerate(parsed_responses):
			description = parsed_data.get("description")
			if description is None:
				continue  # Skip if the LLM failed to generate a valid docstring

			# Create the new task ID and add it to the archive
			new_task_id = f"task_{self.task_num_counter}"
			parent_tasks = parent_sets[i]
			self.archive.record_new_task(
				child_task=new_task_id,
				parent_tasks=parent_tasks,
				description=description,
				session_id=session_idx,
			)

			# v6fix7 P0.1: persist the machine-readable level metadata on the archive node so the
			# validator / siege quota / selection can act on TYPE. Absent (siege off) -> no attrs
			# written -> graphml byte-unchanged on the baseline path.
			level_meta = parsed_data.get("level_meta")
			if level_meta:
				self.archive.set_level_meta(new_task_id, level_meta)

			# Get the single parent ID
			parent_id = parent_tasks[0] if parent_tasks else None

			# Get the description dictionary and safely access the value
			parent_descriptions = self.archive.get_task_descriptions(parent_tasks)
			parent_description_str = parent_descriptions.get(parent_id, "Parent task not found")

			# Append all relevant data for this new task to our results list
			generation_results.append(
				{
					"generated_task_id": new_task_id,
					"parent_task_id": parent_id,
					"parent_task": parent_description_str,  # Assuming one parent for evolution
					"evolution_type": evolution_type,
					"reasoning": parsed_data.get("reasoning"),
					"docstring": description,
					"examples": example_sets[i],  # Keep track of examples used
					# v6fix7 P0.1: carried through so the coop selector / validator see it too.
					"level_meta": level_meta,
					# v7fix4 P2: a system-built relay level ships its own code — the worker skips
					# FM codegen for it (None for every FM-authored level, incl. the whole baseline).
					"_system_code": parsed_data.get("_system_code"),
				}
			)

			self.task_num_counter += 1

		return generation_results


class EnvGenerator:
	"""Handles the technical process of generating runnable environment code from a
	task description, including a reflection loop to fix compilation errors.
	"""

	def __init__(self, env_generator_llm: LLM, archive: TaskArchive, config):
		"""Initializes the EnvGenerator.

		Args:
		    env_generator_llm: An instance of the LLM class for code generation.
		    archive: An instance of the TaskArchive.
		    config: The Hydra configuration object.

		"""
		self.llm = env_generator_llm
		self.archive = archive
		self.config = config
		self.gen_env_prompt = importlib.import_module(self.config.prompts.env_generation)
		self.craftax_mechanics = importlib.import_module(self.config.prompts.craftax_code).context

		if config.mode != "reward":
			self.wrapper_mechanics = importlib.import_module(
				self.config.prompts.wrapper_mechanics
			).context
		else:
			self.wrapper_mechanics = importlib.import_module(
				self.config.prompts.wrapper_mechanics_r
			).context

	def generate(self, tasks_to_generate: list[dict]) -> dict:
		"""Generates and validates environment files, ensuring compilation for all tasks.
		If a task fails compilation after all reflection attempts, it is re-queued
		for generation from scratch until it succeeds.
		"""
		print(
			f"Generating environment code for {len(tasks_to_generate)} tasks with persistent retries..."
		)

		# Initialize state tracker for each task
		task_states = [
			{
				"task_info": task_info,
				"status": "needs_initial_generation",
				"final_code": None,
				"last_response": None,
				"error_msg": None,
				"reflection_count": 0,
			}
			for task_info in tasks_to_generate
		]

		# Main loop: continues until all tasks compile successfully
		round_num = 0
		while True:
			round_num += 1
			print(f"\n--- Generation/Reflection Round {round_num} ---")

			# Check for completion condition: if all tasks are successful, break the loop.
			if all(s["status"] == "success" for s in task_states):
				print("All tasks have been successfully compiled. Exiting loop.")
				break

			# Separate tasks by their current state
			tasks_for_initial_generation = [
				s for s in task_states if s["status"] == "needs_initial_generation"
			]
			tasks_for_reflection = [
				s for s in task_states if s["status"] == "pending_reflection"
			]

			initial_gen_prompts = []
			if tasks_for_initial_generation:
				print(
					f"Preparing {len(tasks_for_initial_generation)} tasks for initial generation..."
				)
				for state in tasks_for_initial_generation:
					# Reset reflection count for this new attempt
					state["reflection_count"] = 0
					example_str = self._format_code_examples(state["task_info"]["examples"])
					initial_gen_prompts.append(
						self.gen_env_prompt.user_prompt.format(
							CODE_EXAMPLES=example_str,
							TASK_DESCRIPTION=state["task_info"]["description"],
						)
					)

			reflection_prompts = []
			if tasks_for_reflection:
				print(f"Preparing {len(tasks_for_reflection)} tasks for reflection...")
				for state in tasks_for_reflection:
					reflection_prompts.append(
						self._build_reflection_prompt(
							state["last_response"],
							state["error_msg"],
							self._format_code_examples(state["task_info"]["examples"]),
							state["task_info"]["description"],
						)
					)

			system_prompt = self.gen_env_prompt.system_prompt.format(
				CRAFTAX_CODE=self.craftax_mechanics, MINICRAFTAX_CODE=self.wrapper_mechanics
			)

			# Query LLM in parallel for both batches
			new_initial_responses = []
			if initial_gen_prompts:
				new_initial_responses = self.llm.query(system_prompt, initial_gen_prompts)

			new_reflection_responses = []
			if reflection_prompts:
				new_reflection_responses = self.llm.query(system_prompt, reflection_prompts)

			# Update task states with LLM responses
			for i, state in enumerate(tasks_for_initial_generation):
				state["last_response"] = new_initial_responses[i]["content"]
				state["status"] = "pending_validation"

			for i, state in enumerate(tasks_for_reflection):
				state["last_response"] = new_reflection_responses[i]["content"]
				state["reflection_count"] += 1
				state["status"] = "pending_validation"

			# Validate all tasks that have received a new response
			tasks_to_validate = [s for s in task_states if s["status"] == "pending_validation"]
			print(f"Validating {len(tasks_to_validate)} new code attempts...")
			for state in tasks_to_validate:
				code_attempt = self._extract_file(state["last_response"])
				if not code_attempt:
					state["error_msg"] = "Could not extract Python code from the LLM response."
					print(f"Task {state['task_info']['task']}... FAILED (code extraction)")
				else:
					# v7fix2: pass the declared spawn floor through so check_compilation can
					# cross-check the declaration against the code's actual reset behaviour.
					from auction.level_meta import parse_level_meta as _plm

					_meta = _plm(state["last_response"])
					is_correct, error_msg = self.check_compilation(
						code_attempt,
						declared_spawn_floor=(
							None if _meta is None else int(_meta.get("spawn_floor") or 0)
						),
					)
					if is_correct:
						state["status"] = "success"
						state["final_code"] = code_attempt
						print(f"Task {state['task_info']['task']}... SUCCESS")
					else:
						state["error_msg"] = error_msg
						print(
							f"Task {state['task_info']['task']}... FAILED (compilation error): {error_msg}"
						)

				# Decide the next step for failed tasks
				if state["status"] != "success":
					if state["reflection_count"] >= self.config.num_reflections_max:
						# Ran out of reflection trials, reset for a completely new attempt
						print(
							f"Task {state['task_info']['task']} has failed max reflections. Re-queueing for generation from scratch."
						)
						state["status"] = "needs_initial_generation"
					else:
						# Still have reflection trials left
						state["status"] = "pending_reflection"

		# Final processing and archive update
		generation_results = {}
		for state in task_states:
			task_id = state["task_info"]["task"]
			self.archive.update_node_status(task_id, "compile_success")
			self.archive.update_node_code(task_id, state["final_code"])

			generation_results[task_id] = {
				"compiled": True,
				"code": state["final_code"],
				"error": None,
			}

		print("\nBatch environment generation complete.")
		return generation_results

	def generate_code_only(self, tasks_to_generate: list[dict]) -> dict[str, str | None]:
		"""Runs the LLM query to generate code for a batch of tasks, but does NOT compile.
		This method is safe to run in a background thread.

		Args:
		    tasks_to_generate: A list of dicts, e.g.,
		        [{'task': 'task_123', 'description': '...', 'examples': [...]}]

		Returns:
		    A dictionary mapping task_id to the generated code string (or None if extraction failed).

		"""
		print(f"    WORKER (Thread): Generating code for {len(tasks_to_generate)} tasks...")

		# 1. Prepare a batch of prompts
		user_prompts = []
		code_example_strs = []
		for task_info in tasks_to_generate:
			example_str = self._format_code_examples(task_info["examples"])
			code_example_strs.append(example_str)
			user_prompts.append(
				self.gen_env_prompt.user_prompt.format(
					CODE_EXAMPLES=example_str, TASK_DESCRIPTION=task_info["description"]
				)
			)

		system_prompt = self.gen_env_prompt.system_prompt.format(
			CRAFTAX_CODE=self.craftax_mechanics, MINICRAFTAX_CODE=self.wrapper_mechanics, MOBS=MOBS_CODE,
		)

		# Query the LLM for all prompts in parallel
		responses = self.llm.query(system_prompt, user_prompts)

		# Extract code and map to task_id
		results = {}
		for i, task_info in enumerate(tasks_to_generate):
			task_id = task_info["task"]
			response_content = responses[i].get("content")
			extracted_code = self._extract_file(response_content)
			results[task_id] = extracted_code  # Will be None if extraction failed

		print("    WORKER (Thread): Code generation complete.")
		return results

	def _build_reflection_prompt(
		self, failed_response_content: str, error_msg: str, code_examples_str: str, task_desc: str
	) -> str:
		"""Builds a reflection prompt to help the LLM fix its previous error.

		Args:
			failed_response_content: The LLM's previous response that failed.
			error_msg: The error message from the failed attempt.
			code_examples_str: Formatted code examples for context.
			task_desc: The task description being generated.

		Returns:
			A formatted reflection prompt string.
		"""
		prompt_template = self.gen_env_prompt.user_prompt_reflection_not_compilation_error
		return prompt_template.format(
			PREVIOUS_RESPONSE=failed_response_content, ERROR=error_msg, TASK_DESC=task_desc
		)

	def _format_code_examples(self, example_paths: list[str]) -> str:
		"""Formats the selected code examples into a string for the LLM prompt."""
		codes = self.archive.get_task_codes(example_paths)
		return "\n".join([f"<example>\n{code}\n</example>\n" for code in codes.values()])

	def check_compilation(self, code: str, declared_spawn_floor: int | None = None) -> tuple[bool, str]:
		"""Validates code by loading and running a full environment step on CPU.

		This ensures generated code is syntactically correct and produces valid
		JAX-compatible state. Runs strictly on CPU to avoid GPU memory conflicts
		with training.

		v7fix2: when ``declared_spawn_floor`` is given (the <level_meta> declaration), the
		floor the code ACTUALLY spawns on after env.reset() must match it. R6 only validates
		the declaration, and the rung readings key off the declared floor — a level declaring
		floor 3 while spawning at the overworld would poison the relay's readings as "valid".

		v7fix4.1: two more guards ride the same reflection loop (see envgen_guards.py) —
		a string-level ban on numpy.random / stdlib random (frozen into constants under
		JIT: the world silently stops varying per reset), and the world-shape contract
		(eval_shape of generate_world vs the canonical blank-builder world; EVERY leaf's
		shape/dtype must match, because training compiles all tasks into one lax.switch
		whose branches must have identical output types — the v7fix4 double-crash).

		Args:
			code: The Python source code to validate.
			declared_spawn_floor: the <level_meta> spawn_floor, or None to skip the check
				(no meta block — e.g. siege off — keeps the old behaviour byte-identical).

		Returns:
			A tuple of (success: bool, error_message: str).
		"""
		# v7fix4.1: cheap string-level guard first — banned randomness sources (np.random /
		# stdlib random) never reach exec. They freeze into constants under JIT, which no
		# runtime check below can see (the world just silently stops varying per reset).
		randomness_error = scan_banned_randomness(code)
		if randomness_error:
			return False, randomness_error

		temp_file = None
		module_name = None

		try:
			# Write code to temporary file
			with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
				f.write(code)
				temp_file = f.name

			# Get CPU device for isolated execution
			try:
				cpu_device = jax.devices("cpu")[0]
			except IndexError:
				cpu_device = jax.local_devices(backend="cpu")[0]

			# Load and validate environment on CPU
			with jax.default_device(cpu_device):
				temp_task = Task(temp_file)
				env = temp_task.env
				module_name = temp_task.task.__module__

				params = env.default_params
				key = jax.random.PRNGKey(0)

				# Define validation kernel that runs reset + step
				def _validate_on_cpu_impl(rng):
					rng, reset_key = jax.random.split(rng)
					obs, state = env.reset(reset_key, params)
					spawn_level = state.player_level  # v7fix2: the floor the code ACTUALLY starts on
					action = env.action_space(params).sample(rng)
					obs, state, reward, done, info = env.step(rng, state, action, params)

					# v7fix4.3: is_success is a lax.switch branch in MULTITASK training but is
					# never called by this solo env's step — a hallucinated attribute inside it
					# passes validation, enters the archive, and kills the whole training jit
					# (job 3929744 @s82: 'EnvParams' object has no attribute
					# 'achievement_reward_map'). Execute it here so the AttributeError surfaces
					# in the reflection loop instead.
					_ = temp_task.task.is_success(state)

					# Validate inventory field types to prevent JAX compilation errors
					for field_name, value in state.inventory.__dict__.items():
						if hasattr(value, "dtype") and value.dtype != jnp.int32:
							raise ValueError(
								f"Inventory field '{field_name}' has type {value.dtype}, expected int32."
							)
					return reward, spawn_level

				_validate_on_cpu = jax.jit(_validate_on_cpu_impl, backend="cpu")
				_, _spawn_level = _validate_on_cpu(key)

				# v7fix4.3: get_task_params is called at MULTITASK env init (clamp + stack) but
				# never on this solo path — same failure family as is_success above, one gate
				# earlier (it would crash training at init, before the first step). Plain python
				# call, no jit needed.
				_ = temp_task.task.get_task_params()

				# v7fix4.4: the conditioning table constructs EVERY archived task class as
				# cls(None, None) each session (training.py _generate_embeddings_for_session)
				# — an __init__ that dereferences its args passes every guard above because
				# this solo path always constructs with real params (baseline 3919416 @s96:
				# None.replace inside a task __init__ killed the whole run; 4th member of the
				# "training executes what solo validation never calls" family). Same probe
				# here, plus the two attributes that table actually reads; the wrapper turns
				# the raw AttributeError into a TEACHING message for the reflection loop.
				try:
					_probe = type(temp_task.task)(None, None)
					_ = _probe.relevant_achievements
					_ = _probe.label
				except Exception as _pe:
					raise ValueError(
						"Env(None, None) must be constructible: the training-side "
						"conditioning table instantiates every task class with None "
						"args each session and reads .relevant_achievements/.label. "
						"Do not dereference __init__ args before defaulting them "
						"(pattern: `static_params = static_params or "
						f"StaticEnvParams()`). It failed with: {_pe}"
					) from _pe

				# v7fix4.1 world-shape contract: the rollout above validates the task SOLO,
				# so a non-standard world (e.g. a custom StaticEnvParams with more mob slots)
				# is internally consistent and passes it — then explodes only when training
				# compiles ALL tasks into one lax.switch, whose branches must have identical
				# output shapes. eval_shape is trace-only (no execution, no XLA compile).
				world_struct = jax.eval_shape(
					temp_task.task.generate_world, jax.random.PRNGKey(0)
				)
				shape_mismatches = diff_world_specs(
					_canonical_world_specs(), _flatten_world_specs(world_struct)
				)

			if shape_mismatches:
				return False, shape_mismatch_message(shape_mismatches)

			# v7fix2: declared-vs-actual spawn-floor cross-check (see docstring). The message
			# teaches — it rides the reflection loop back to the generator.
			if declared_spawn_floor is not None and int(_spawn_level) != int(declared_spawn_floor):
				return False, (
					f"spawn-floor mismatch: <level_meta> declares spawn_floor="
					f"{int(declared_spawn_floor)} but env.reset() actually spawns the player on "
					f"floor {int(_spawn_level)}. Make generate_world call "
					f"builder.set_starting_floor({int(declared_spawn_floor)}); a non-zero "
					f"spawn_floor is legal only for relay levels (at exactly the rung floor the "
					f"SIEGE DIRECTIVE states) or for BREADTH ecology levels within the breadth "
					f"spawn frontier."
				)

			return True, ""

		except Exception as e:
			traceback.print_exc()
			return False, f"Compilation error: {str(e)}"

		finally:
			if temp_file and os.path.exists(temp_file):
				os.unlink(temp_file)
			if module_name and module_name in sys.modules:
				del sys.modules[module_name]

	def _extract_file(self, content: str) -> str | None:
		"""Extracts Python code from an LLM response wrapped in <code> tags.

		Args:
			content: The raw LLM response string.

		Returns:
			The extracted code string, or the original content if no tags found.
		"""
		if not content:
			return None
		code_match = re.search(r"<code>\s*(.*?)\s*</code>", content, re.DOTALL)
		if code_match:
			return code_match.group(1).strip()
		return content


class GenManager:
	"""Main orchestrator for the DiCode evolution pipeline.

	Coordinates task generation, code synthesis, and archive management
	for curriculum learning through LLM-based task evolution.
	"""

	def __init__(self, config):
		"""Initializes the GenManager pipeline.

		Args:
			config: The Hydra configuration object containing all settings.
		"""
		self.config_ = config
		self.config = config.gen_manager

		task_designer = LLM(
			provider=self.config.task_generator.provider,
			base_url=self.config.task_generator.base_url,
			model=self.config.task_generator.model,
			llm_type=self.config.task_generator.llm_type,
			max_tokens=self.config.task_generator.max_tokens,
			temperature=self.config.task_generator.temperature,
			top_p=self.config.task_generator.top_p,
			think=self.config.task_generator.think,
		)
		env_coder = LLM(
			provider=self.config.env_generator.provider,
			base_url=self.config.env_generator.base_url,
			model=self.config.env_generator.model,
			llm_type=self.config.env_generator.llm_type,
			max_tokens=self.config.env_generator.max_tokens,
			temperature=self.config.env_generator.temperature,
			top_p=self.config.env_generator.top_p,
			think=self.config.env_generator.think,
		)

		embedding_model = LLM(
			provider=self.config.embedding_model.provider,
			base_url=self.config.embedding_model.base_url,
			model=self.config.embedding_model.model,
			llm_type=self.config.embedding_model.llm_type,
			embedding_size=self.config.embedding_model.embedding_size,
		)
		# Optional N heterogeneous Proposer LLMs for the auction method (config.gen_manager.proposers).
		# Absent -> proposer_llms stays None -> TaskGenerator falls back to the single-FM baseline.
		proposer_llms = None
		proposers_cfg = self.config.get("proposers", None)
		if proposers_cfg:
			proposer_llms = [
				LLM(
					provider=pc.provider,
					base_url=pc.base_url,
					model=pc.model,
					llm_type=pc.llm_type,
					max_tokens=pc.max_tokens,
					temperature=pc.temperature,
					top_p=pc.top_p,
					think=pc.think,
				)
				for pc in proposers_cfg
			]
			print(f"[auction] Built {len(proposer_llms)} Proposer LLMs.")

		# Optional MODELER LLM for v5-debate (config.gen_manager.modeler). Absent -> modeler path off.
		modeler_llm = None
		scientist_llm = None
		modeler_cfg = self.config.get("modeler", None)
		if modeler_cfg:
			modeler_llm = LLM(
				provider=modeler_cfg.provider,
				base_url=modeler_cfg.base_url,
				model=modeler_cfg.model,
				llm_type=modeler_cfg.llm_type,
				max_tokens=modeler_cfg.max_tokens,
				temperature=modeler_cfg.temperature,
				top_p=modeler_cfg.top_p,
				think=modeler_cfg.get("think", False),
			)
			print(f"[modeler] Built MODELER LLM ({modeler_cfg.model}).")
			# v7fix5.5 P2: the scientist pass is the SAME model with think statically ON — the
			# 2026-07-17 A/B pinned think per call type (big bookkeeping call: off, or reasoning
			# burns the budget at 30k+ chars; small hypothesis call: on, mechanism depth). No
			# runtime switching; config gate modeler.scientist=false disables the loop.
			if modeler_cfg.get("scientist", True):
				scientist_llm = LLM(
					provider=modeler_cfg.provider,
					base_url=modeler_cfg.base_url,
					model=modeler_cfg.model,
					llm_type=modeler_cfg.llm_type,
					max_tokens=modeler_cfg.max_tokens,
					temperature=modeler_cfg.temperature,
					top_p=modeler_cfg.top_p,
					think=True,
				)
				print(f"[modeler] Built SCIENTIST LLM ({modeler_cfg.model}, think=on).")

		self.archive = TaskArchive(self.config)
		self.selector = TaskSelector(self.archive, embedding_model, self.config)
		self.task_generator = TaskGenerator(
			task_designer,
			self.archive,
			self.selector,
			self.config,
			proposer_llms=proposer_llms,
			modeler_llm=modeler_llm,
			scientist_llm=scientist_llm,
		)
		self.env_generator = EnvGenerator(env_coder, self.archive, self.config)

		self.session_idx = self.archive.get_max_session_idx() + 1

	def evolve_tasks(
		self, dict_of_tasks: dict[str, list[str]], global_agent_profile: dict
	) -> list[dict]:
		"""Orchestrates the full I/O-bound evolution pipeline for one session.

		This method is thread-safe and performs:
		1. Task description generation from mastered tasks (LLM Call 1)
		2. Code generation for new task descriptions (LLM Call 2)
		3. Result merging and preparation for the main thread

		Args:
			dict_of_tasks: Dictionary mapping task categories to task ID lists.
			global_agent_profile: Agent skill metrics from evaluation.

		Returns:
			A list of dictionaries containing generation results.
		"""
		print("    WORKER (Thread): Starting task design...")
		all_generation_results = []

		# --- 1. Evolve from Mastered Tasks (LLM Call 1a) ---

		if not self.config_.ablation:
			if dict_of_tasks.get("mastered"):
				# v5-debate COOP (modeler + sequential fill) when enabled; else v4 auction; else
				# unchanged single-FM baseline.
				use_modeler = bool(self.config.get("auction_modeler", False))
				use_auction = bool(self.config.get("auction", False))
				if use_modeler:
					mastered_results = self.task_generator.evolve_mastered_coop(
						self.session_idx,
						dict_of_tasks["mastered"],
						global_agent_profile,
					)
				elif use_auction:
					mastered_results = self.task_generator.evolve_mastered_auction(
						self.session_idx,
						dict_of_tasks["mastered"],
						global_agent_profile,
						k=self.config.get("auction_k", None),
					)
				else:
					mastered_results = self.task_generator.evolve_mastered(
						self.session_idx, dict_of_tasks["mastered"], global_agent_profile
					)
				all_generation_results.extend(mastered_results)

		else:
			if dict_of_tasks.get("mastered"):
				mastered_results = self.task_generator.evolve_ablation(
					self.session_idx, dict_of_tasks["mastered"], global_agent_profile
				)
				all_generation_results.extend(mastered_results)

		if not all_generation_results:
			print("    WORKER (Thread): No new tasks were designed in this session.")
			return []

		print(
			f"    WORKER (Thread): Task design finished. {len(all_generation_results)} designs created."
		)

		# Prepare for Code Generation
		# v7fix4 P2: system-built relay levels ship their own code — no FM codegen call for them
		# (that call is the run's biggest out-token sink AND the fidelity hole the fix closes).
		tasks_to_generate_code_for = [
			{
				"task": result["generated_task_id"],
				"description": result["docstring"],
				"examples": result["examples"],
			}
			for result in all_generation_results
			if not result.get("_system_code")
		]

		# Generate Code (LLM Call 2)
		compilation_results = self.env_generator.generate_code_only(tasks_to_generate_code_for)

		# Merge Generation and Compilation Results
		final_results_for_worker = []
		for gen_result in all_generation_results:
			task_id = gen_result["generated_task_id"]
			# v7fix4 P2: shipped code wins (system relay levels); FM levels read the codegen result.
			code_string = gen_result.get("_system_code") or compilation_results.get(task_id)

			gen_result["code_string"] = code_string

			if code_string:
				gen_result["compiled"] = None  # To be filled by main thread
				gen_result["code"] = code_string
				gen_result["error"] = None
			else:
				gen_result["compiled"] = False
				gen_result["code"] = ""
				gen_result["error"] = "Failed to extract code from LLM response."

			final_results_for_worker.append(gen_result)

		return final_results_for_worker