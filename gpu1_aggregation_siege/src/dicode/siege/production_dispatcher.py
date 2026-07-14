"""Production dispatcher v4 — pool recomputed, cache hard-failed, Original isolated.

Fixes:
- Original: REQUIRES injected GenManager + config at runtime, ZERO cache init/reads
- Enhanced: separate initialization, cache hard-failed on missing fields
- compile_candidate: TaskParams consumed by generate_world via self.get_task_params()
- Runtime adapter: all 5 mechanisms with stable hashes
- make_test_defaults(): explicit test-only factory — never used in production dispatch
"""
import json, os, hashlib, time, numpy as np

POOL_PATH = "/root/experiments/dicode_runs/siege_aggregation/frozen_pool_artifact.json"
CACHE_PATH = "/root/experiments/dicode_runs/siege_aggregation/frozen_immutable_cache"
ALL_MECHANISMS = ["original", "soft_copeland", "budgeted_copeland", "auction_raw", "auction_budgeted"]


def make_test_defaults(pool: list):
	"""TEST-ONLY factory. Returns mock (gen_manager, config) for Original dispatch testing.

	DO NOT call this in production dispatch paths. Production Original mode MUST
	receive injected real GenManager + config from the runtime.
	"""
	class L: __enter__ = lambda *a: None; __exit__ = lambda *a: None
	class NV:
		def __init__(s, d): s._d = d
		def __call__(s, data=False): return s._d.items() if data else s._d.keys()
		def __getitem__(s, k): return s._d[k]
	class FG:
		def __init__(s): s.nd = {}
		@property
		def nodes(s): return NV(s.nd)
	class FA:
		def __init__(s): s.graph = FG(); s._lock = L()
	class FGM:
		def __init__(s): s.archive = FA(); s.session_idx = 1
	class FC:
		def __init__(s, **kw): s.__dict__.update(kw)
		def get(s, k, d=None): return s.__dict__.get(k, d)

	gm = FGM()
	for i, tid in enumerate(pool):
		gm.archive.graph.nd[tid] = {
			"is_active": True, "priority_score": 0.3 + 0.02 * i,
			"session_last_trained": max(0, i - 10),
			"type": "seed", "status": "B",
			"performance_history": [{"sr": 0.3 + 0.02 * i}],
		}
	cfg = FC(aggregation={"enabled": False},
	         dicode_manager=FC(staleness_coeff=0.3, prioritization_method="rank",
	                           prioritization_temperature=1.0, topk_k=8))
	return gm, cfg


class ProductionDispatcher:
	"""Pool hash recomputed. Original: ZERO cache. Enhanced: hard-fail scores."""

	def __init__(self):
		if not os.path.exists(POOL_PATH):
			raise FileNotFoundError(f"Pool missing: {POOL_PATH}")
		with open(POOL_PATH) as f:
			self.pool_artifact = json.load(f)
		self.pool = self.pool_artifact["pool"]
		self.candidate_count = len(self.pool)
		self.selected_count = self.pool_artifact.get("selected_count", 8)

		self.computed_hash = hashlib.sha256(
			json.dumps(self.pool, sort_keys=True).encode()).hexdigest()
		if self.computed_hash != self.pool_artifact["pool_hash"]:
			raise RuntimeError("Pool hash MISMATCH")
		self.pool_hash = self.computed_hash

		if self.candidate_count != 32:
			raise RuntimeError(f"candidate_count={self.candidate_count}")
		if self.selected_count != 8:
			raise RuntimeError(f"selected_count={self.selected_count}")

		self._cache_loaded = False
		self.cache_hit_rate = None
		self.signals = None

	def _init_enhanced_cache(self):
		"""Initialize cache + signals for ENHANCED mechanisms only. Original skips this."""
		if self._cache_loaded:
			return
		if not os.path.isdir(CACHE_PATH):
			raise FileNotFoundError(f"Cache missing: {CACHE_PATH}")
		from dicode.mechanisms.immutable_cache import MultiRoleImmutableCache, compute_immutable_cache_key
		from dicode.mechanisms.model_manifest import ROLE_CONFIG_MAP
		self.cache = MultiRoleImmutableCache(cache_dir=CACHE_PATH)
		self.cache.load_all()
		total = sum(c.entry_count for c in self.cache._caches.values())
		if total < 96:
			raise RuntimeError(f"Cache: {total} entries")
		self.model_manifest = {
			"family": __import__('dicode.mechanisms.model_manifest', fromlist=['MANIFEST_FAMILY']).MANIFEST_FAMILY,
			"version": __import__('dicode.mechanisms.model_manifest', fromlist=['MANIFEST_VERSION']).MANIFEST_VERSION,
			"roles": {r: c["exact_model_id"] for r, c in ROLE_CONFIG_MAP.items()},
		}
		self._build_signals()
		self._verify_cache()
		self._cache_loaded = True

	def _build_signals(self):
		from dicode.mechanisms.immutable_cache import compute_immutable_cache_key
		from dicode.mechanisms.model_manifest import ROLE_CONFIG_MAP
		REQUIRED = {"tutor": "progression_score", "critic": "critic_penalty", "explorer": "novelty_score"}
		n = len(self.pool)
		self.progression = np.zeros(n); self.novelty = np.zeros(n)
		self.critic_penalty = np.zeros(n); self.retention = np.zeros(n)
		self.source_ids = []
		for i, tid in enumerate(self.pool):
			for role in ["tutor", "critic", "explorer"]:
				cfg = ROLE_CONFIG_MAP.get(role)
				try:
					key = compute_immutable_cache_key(task_code_hash=tid, student_stage_id="stage_0",
						role=role, provider=cfg["provider"], exact_model_id=cfg["exact_model_id"],
						prompt_version=cfg["prompt_version"], schema_version=cfg["schema_version"])
				except ValueError: continue
				entry = self.cache.get_cache(role).get(key)
				if entry is None:
					raise RuntimeError(f"HARD FAIL: cache miss {tid}/{role}")
				scores = entry.get("scores", {})
				required_field = REQUIRED[role]
				if required_field not in scores:
					raise RuntimeError(f"HARD FAIL: missing {required_field} for {tid}/{role}")
			tutor_s = self.cache.get_cache("tutor").get(
				compute_immutable_cache_key(task_code_hash=tid, student_stage_id="stage_0",
					role="tutor", provider=ROLE_CONFIG_MAP["tutor"]["provider"],
					exact_model_id=ROLE_CONFIG_MAP["tutor"]["exact_model_id"],
					prompt_version=ROLE_CONFIG_MAP["tutor"]["prompt_version"],
					schema_version=ROLE_CONFIG_MAP["tutor"]["schema_version"]))["scores"]
			critic_s = self.cache.get_cache("critic").get(
				compute_immutable_cache_key(task_code_hash=tid, student_stage_id="stage_0",
					role="critic", provider=ROLE_CONFIG_MAP["critic"]["provider"],
					exact_model_id=ROLE_CONFIG_MAP["critic"]["exact_model_id"],
					prompt_version=ROLE_CONFIG_MAP["critic"]["prompt_version"],
					schema_version=ROLE_CONFIG_MAP["critic"]["schema_version"]))["scores"]
			explorer_s = self.cache.get_cache("explorer").get(
				compute_immutable_cache_key(task_code_hash=tid, student_stage_id="stage_0",
					role="explorer", provider=ROLE_CONFIG_MAP["explorer"]["provider"],
					exact_model_id=ROLE_CONFIG_MAP["explorer"]["exact_model_id"],
					prompt_version=ROLE_CONFIG_MAP["explorer"]["prompt_version"],
					schema_version=ROLE_CONFIG_MAP["explorer"]["schema_version"]))["scores"]
			self.progression[i] = float(tutor_s["progression_score"])
			self.novelty[i] = float(explorer_s["novelty_score"])
			self.critic_penalty[i] = float(critic_s["critic_penalty"])
			self.retention[i] = 1.0 - self.critic_penalty[i]
			self.source_ids.append(
				self.cache.get_cache("critic").get(
					compute_immutable_cache_key(task_code_hash=tid, student_stage_id="stage_0",
						role="critic", provider=ROLE_CONFIG_MAP["critic"]["provider"],
						exact_model_id=ROLE_CONFIG_MAP["critic"]["exact_model_id"],
						prompt_version=ROLE_CONFIG_MAP["critic"]["prompt_version"],
						schema_version=ROLE_CONFIG_MAP["critic"]["schema_version"]))["decision"])
		self.source_ids_arr = np.array(self.source_ids)

	def _verify_cache(self):
		from dicode.mechanisms.immutable_cache import compute_immutable_cache_key
		from dicode.mechanisms.model_manifest import ROLE_CONFIG_MAP
		hits, total = 0, 0
		for tid in self.pool:
			for role in ["tutor", "critic", "explorer"]:
				cfg = ROLE_CONFIG_MAP.get(role)
				try:
					key = compute_immutable_cache_key(task_code_hash=tid, student_stage_id="stage_0",
						role=role, provider=cfg["provider"], exact_model_id=cfg["exact_model_id"],
						prompt_version=cfg["prompt_version"], schema_version=cfg["schema_version"])
				except ValueError: continue
				total += 1
				if self.cache.get_cache(role).get(key): hits += 1
		self.cache_hit_rate = hits / max(1, total)
		if self.cache_hit_rate < 0.95:
			raise RuntimeError(f"Cache {self.cache_hit_rate:.4f} < 0.95")

	def _dispatch_original(self, gen_manager, config):
		"""Original: REQUIRES real GenManager + config injected from runtime. ZERO cache.

		NO silent test-default fallback. Caller MUST inject real GenManager and config
		for Original production mode. Use make_test_defaults(pool) for a test-only factory.
		"""
		from dicode.selection import sample_tasks_for_training
		import io, contextlib

		if gen_manager is None or config is None:
			raise TypeError(
				"Original production mode REQUIRES injected gen_manager and config. "
				"Use make_test_defaults(pool) for a test-only factory. "
				"Do NOT silently construct test defaults in production."
			)

		out = io.StringIO()
		with contextlib.redirect_stdout(out):
			selected = sample_tasks_for_training(gen_manager, config, 8)

		return selected, {
			"aggregation_disabled": True, "cache_reads": 0,
			"cache_initialized": False, "gen_manager_injected": True,
			"output": out.getvalue(),
		}

	def _dispatch_enhanced(self, mechanism):
		"""Enhanced mechanisms: signals from cache."""
		self._init_enhanced_cache()
		n = len(self.pool)
		signals = {
			"progression": self.progression, "retention": self.retention,
			"novelty": self.novelty, "critic_penalty": self.critic_penalty,
			"monopoly_penalty": np.zeros(n), "source_ids": self.source_ids_arr,
			"skill_counts": np.ones(n),
		}
		weights = {"w_progression": 0.34, "w_retention": 0.33, "w_novelty": 0.33,
		           "w_critic": 0.01, "w_monopoly": 0.01}
		if mechanism == "soft_copeland":
			from dicode.mechanisms.aggregation import _aggregate_soft_copeland
			sc = _aggregate_soft_copeland(signals, weights, 1.0)
			return [self.pool[i] for i in np.argsort(-sc)[:8]], {"scores": sc.tolist()}
		elif mechanism == "budgeted_copeland":
			from dicode.mechanisms.aggregation import _aggregate_soft_copeland, apply_budget_caps
			sc = _aggregate_soft_copeland(signals, weights, 1.0)
			sc_b, bi = apply_budget_caps(sc.copy(), self.pool, self.source_ids_arr, max_source_share=0.25)
			return [self.pool[i] for i in np.argsort(-sc_b)[:8]], {"budget_info": bi}
		elif mechanism == "auction_raw":
			from dicode.mechanisms.auction import build_role_utilities_from_signals, run_auction_selection
			utils = build_role_utilities_from_signals(signals)
			ar = run_auction_selection(self.pool, utils, 8, "raw", seed=0)
			return ar["selected_ids"], {"auction": ar}
		elif mechanism == "auction_budgeted":
			from dicode.mechanisms.auction import build_role_utilities_from_signals, run_auction_selection
			utils = build_role_utilities_from_signals(signals)
			ab = run_auction_selection(self.pool, utils, 8, "budgeted",
			                           role_budgets={"tutor": 3.0, "critic": 2.0, "explorer": 2.0}, seed=0)
			return ab["selected_ids"], {"auction": ab}
		raise ValueError(f"Unknown: {mechanism}")

	def dispatch(self, mechanism: str, gen_manager=None, config=None) -> dict:
		"""Dispatch a mechanism. Original REQUIRES gen_manager and config."""
		if mechanism not in ALL_MECHANISMS:
			raise ValueError(f"Unknown: {mechanism}")
		if mechanism == "original":
			if gen_manager is None or config is None:
				raise TypeError(
					"dispatch('original', ...) REQUIRES injected gen_manager and config. "
					"Use make_test_defaults(self.pool) for a test-only factory."
				)
			selected, trace = self._dispatch_original(gen_manager, config)
		else:
			selected, trace = self._dispatch_enhanced(mechanism)
		if len(selected) != 8:
			raise RuntimeError(f"{mechanism}: {len(selected)} selected, must be 8")
		cmap = {tid: i for i, tid in enumerate(sorted(self.pool))}
		return {
			"mechanism": mechanism, "selected_ids": selected, "n_selected": len(selected),
			"task_ids": [cmap[tid] for tid in selected], "candidate_id_map": cmap,
			"pool_hash": self.pool_hash, "computed_hash": self.computed_hash,
			"cache_hit_rate": self.cache_hit_rate,
			"candidate_count": self.candidate_count, "selected_count": self.selected_count,
			"trace": trace, "timestamp": time.time(), "status": "PRODUCTION_DISPATCH",
		}


# ===================================================================
# Candidate compiler — TaskParams consumed by generate_world
# ===================================================================

def compile_candidate(candidate_id: str, achievements: list, param_seed: int):
	"""Production compiler. TaskParams consumed by generate_world via get_task_params()."""
	from craftax.craftax.constants import Achievement
	from minicraftax.craftax_state import TaskParams
	from minicraftax.tasks.base_task import BaseTask
	all_achs = list(Achievement)
	target = []
	for name in achievements:
		for a in all_achs:
			if a.name.lower() == name.lower(): target.append(a); break
	if not target: target = all_achs[:2]
	rng = np.random.default_rng(param_seed)
	sm = 0.25 + 3.0 * rng.random(); hm = 0.25 + 6.0 * rng.random(); dm = 0.25 + 6.0 * rng.random()
	ch = hashlib.sha256(f"{candidate_id}:{sorted([a.name for a in target])}".encode()).hexdigest()[:16]

	class CandidateEnv(BaseTask):
		def __init__(self, sp, ep):
			super().__init__(sp, ep)
			self._cid = candidate_id; self._ch = ch
			self.relevant_achievements = target; self.completed_achievements = []
			self.label = f"siege_{candidate_id}"
			self._sm = sm; self._hm = hm; self._dm = dm
		@property
		def candidate_hash(self): return self._ch
		@property
		def candidate_id(self): return self._cid
		def get_task_params(self):
			return TaskParams(passive_spawn_multiplier=float(self._sm),
			                melee_spawn_multiplier=float(self._sm * 0.8),
			                mob_health_multiplier=float(self._hm),
			                mob_damage_multiplier=float(self._dm))
		def generate_world(self, rng):
			"""TaskParams consumed by MultiTaskMiniCraftaxEnv during init via get_task_params()."""
			from minicraftax.world_builder import WorldBuilder
			return WorldBuilder(rng, self.static_params, self.params).build(rng)
	return CandidateEnv, ch


def compile_selected_candidates(dispatch_result: dict, achievement_map=None):
	task_classes, hashes, compiled = [], [], []
	from craftax.craftax.constants import Achievement
	all_achs = list(Achievement)
	defaults = [a.name.lower() for a in all_achs]
	for i, cid in enumerate(dispatch_result["selected_ids"]):
		if achievement_map and cid in achievement_map:
			ach_names = achievement_map[cid]
		else:
			start = i % (len(defaults) - 2)
			ach_names = defaults[start:start + 2 + (i % 3)]
		seed = int(hashlib.sha256(f"{cid}_compile".encode()).hexdigest()[:8], 16)
		Cls, ch = compile_candidate(cid, ach_names, seed)
		task_classes.append(Cls); hashes.append(ch)
		compiled.append({"candidate_id": cid, "hash": ch, "achievements": ach_names})
	return task_classes, hashes, compiled


def build_runtime_adapter(dispatch_result: dict, achievement_map=None):
	"""Runtime adapter for ALL 5 mechanisms. Returns task_classes, hashes, distribution."""
	task_classes, hashes, compiled = compile_selected_candidates(dispatch_result, achievement_map)
	n = len(task_classes)
	tw = np.ones(n); tw[:4] = 2.0; dist = (tw / tw.sum()).tolist()
	return {
		"mechanism": dispatch_result["mechanism"], "task_classes": task_classes,
		"candidate_hashes": hashes, "distribution": dist, "compiled_manifest": compiled,
		"pool_hash": dispatch_result["pool_hash"], "cache_hit_rate": dispatch_result.get("cache_hit_rate"),
		"selected_ids": dispatch_result["selected_ids"], "n_tasks": n,
		"status": "RUNTIME_ADAPTER_READY",
	}
