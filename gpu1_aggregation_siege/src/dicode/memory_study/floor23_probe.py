"""Floor2->Floor3 probe over the frozen tier3 FRONT_L2 contract.

For every (state, candidate, ho_mode) triple the probe:

  1. selects a capture from the shared result-blind bank (assignment depends
     ONLY on state identity - never on measured results);
  2. burns the capture segment into the candidate's memory via ho_burnin
     (mechanical G2 isolation receipt, fail-closed);
  3. rolls out from the frozen FRONT_L2 start state with the burned memory;
  4. scores the episode with the frozen tier3 predicates/metrics used AS A
     LIBRARY (zero bytes of tier3 code modified);
  5. writes a fully-provenanced result JSON per triple + an aggregate summary.

SYNTHETIC mode runs end-to-end without jax/craftax (mock runtime, synthetic
bank/states). REAL mode is a server execution (RUNBOOK in
docs/memory_study/HO_FLOOR23_DESIGN.md).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .ho_contract import (
    FailClosed,
    HOMode,
    HistoryCapture,
    IsolationContext,
    canonical_json_bytes,
    hash_pytree,
    sha256_hex,
)
from .ho_burnin import RNG_STREAM_ID_BURNIN, burnin_history
from .ho_capture_bank import GENERATOR_SYNTHETIC, assign_capture

SCENARIO = "front_l2"
RESULT_SCHEMA_ID = "mechanism_UED.memory_study_floor23_result/v1"
SUMMARY_SCHEMA_ID = "mechanism_UED.memory_study_floor23_summary/v1"
START_FLOOR = 2
EXIT_FLOOR = 3
DEFAULT_HO_MODES: Tuple[HOMode, ...] = (HOMode.BASE, HOMode.HO_ZERO,
                                        HOMode.HO_REAL)

REQUIRED_EPISODE_FIELDS = ("scenario", "valid_start", "episode_id",
                           "front_floor_transition_reached")


@dataclasses.dataclass(frozen=True)
class CandidateRuntime:
    """A measured candidate bound for probe execution.

    rollout_fn(start_state, memory) -> episode dict carrying the frozen tier3
    FRONT_L2 episode-record fields (see tier3_metrics._ep): scenario,
    valid_start, front_floor_transition_reached, graph_distance_progress,
    episode_id (+ optional timesteps/timed_out/player_died/classified_label).
    It may additionally carry from_level/to_level for the primary-event cross
    check against tier3_event_predicates."""

    candidate_id: str
    params: Any
    initial_memory: Any
    step_fn: Callable[[Any, Any, Any], Any]
    rollout_fn: Callable[[dict, Any], dict]
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)


def load_tier3_library(repo_root=None):
    """Import the frozen tier3 modules AS A LIBRARY (sys.path insertion, the
    same pattern tier3 itself uses). Returns (tier3_metrics,
    tier3_event_predicates). Fails closed when the tooling is absent."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    tdir = Path(repo_root) / "tools" / "tier3_scaffolded_evaluation"
    if not tdir.is_dir():
        raise FailClosed("TIER3_TOOLING_MISSING: %s" % tdir)
    tdir_s = str(tdir)
    if tdir_s not in sys.path:
        sys.path.insert(0, tdir_s)
    try:
        metrics = importlib.import_module("tier3_metrics")
        predicates = importlib.import_module("tier3_event_predicates")
    except Exception as exc:
        raise FailClosed("TIER3_TOOLING_IMPORT_FAILED: %s" % exc) from exc
    return metrics, predicates


def _state_identity(state: dict) -> Tuple[str, str]:
    state_id = state.get("state_id")
    if not state_id:
        raise FailClosed("STATE_MISSING_ID")
    payload_hash = state.get("payload_sha256")
    if not payload_hash:
        payload_hash = sha256_hex(canonical_json_bytes(
            {k: v for k, v in state.items()
             if k not in ("payload_sha256",)}))
    return str(state_id), str(payload_hash)


def _validate_episode(episode: dict, state_id: str, candidate_id: str,
                      mode: HOMode) -> None:
    for field in REQUIRED_EPISODE_FIELDS:
        if field not in episode:
            raise FailClosed(
                "EPISODE_FIELD_MISSING: %s (state=%s cand=%s mode=%s)"
                % (field, state_id, candidate_id, mode.value))
    if episode.get("scenario") != SCENARIO:
        raise FailClosed("EPISODE_SCENARIO_MISMATCH: %r"
                         % episode.get("scenario"))
    if episode.get("valid_start") is not True:
        raise FailClosed("EPISODE_INVALID_START: FRONT_L2 bank states must "
                         "produce valid_start=True episodes (state=%s cand=%s "
                         "mode=%s)" % (state_id, candidate_id, mode.value))
    if not isinstance(episode.get("front_floor_transition_reached"), bool):
        raise FailClosed("EPISODE_PRIMARY_EVENT_NOT_BOOL")
    progress = episode.get("graph_distance_progress")
    if progress is not None:
        if (not isinstance(progress, (int, float)) or isinstance(progress, bool)
                or not (0.0 <= float(progress) <= 1.0)):
            raise FailClosed("EPISODE_PROGRESS_OUT_OF_RANGE: %r" % (progress,))


def run_floor23_probe(states: Sequence[dict],
                      runtimes: Sequence[CandidateRuntime],
                      captures: Sequence[HistoryCapture],
                      out_root,
                      ho_modes: Sequence[HOMode] = DEFAULT_HO_MODES,
                      max_states: Optional[int] = None,
                      run_mode: str = "synthetic",
                      probe_seed: int = 0,
                      repo_root=None,
                      ) -> dict:
    """Execute the probe matrix and write results under out_root.

    Returns the aggregate summary dict (also written to out_root/summary.json).
    """
    if run_mode not in ("synthetic", "real"):
        raise FailClosed("UNKNOWN_RUN_MODE: %r" % (run_mode,))
    if not states:
        raise FailClosed("PROBE_NO_STATES")
    if not runtimes:
        raise FailClosed("PROBE_NO_CANDIDATES")
    if not captures:
        raise FailClosed("PROBE_NO_CAPTURES")
    if not ho_modes:
        raise FailClosed("PROBE_NO_HO_MODES")
    for cap in captures:
        cap.validate()
    metrics, predicates = load_tier3_library(repo_root)

    out = Path(out_root)
    res_dir = out / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    if max_states is not None:
        states = list(states)[:int(max_states)]

    seen_ids = set()
    for rt in runtimes:
        if rt.candidate_id in seen_ids:
            raise FailClosed("DUPLICATE_CANDIDATE_ID: %s" % rt.candidate_id)
        seen_ids.add(rt.candidate_id)

    episodes_by_arm: Dict[Tuple[str, str], List[dict]] = {}
    results_written = 0

    for state in states:
        state_id, state_payload_hash = _state_identity(state)
        capture = assign_capture(list(captures), state_id)
        for rt in runtimes:
            for mode in ho_modes:
                if not isinstance(mode, HOMode):
                    raise FailClosed("UNKNOWN_HO_MODE: %r" % (mode,))
                ctx = IsolationContext(
                    params_sha_before=hash_pytree(rt.params),
                    env_state_payload_hash=None,
                    rng_stream_id=RNG_STREAM_ID_BURNIN,
                    task_embedding_hash=str(
                        state.get("task_embedding_hash") or "NONE_DECLARED"),
                    timestep=int(state.get("timestep", 0)),
                    inventory_hash=str(
                        state.get("inventory_hash") or "NONE_DECLARED"),
                    position_hash=str(
                        state.get("position_hash") or "NONE_DECLARED"),
                    entities_hash=str(
                        state.get("entities_hash") or "NONE_DECLARED"))
                memory_after, receipt = burnin_history(
                    rt.step_fn, rt.params, rt.initial_memory, capture, mode,
                    ctx)
                episode = dict(rt.rollout_fn(state, memory_after))
                episode.setdefault("scenario", SCENARIO)
                episode.setdefault("episode_id", state_id)
                _validate_episode(episode, state_id, rt.candidate_id, mode)
                if ("from_level" in episode) and ("to_level" in episode):
                    derived = bool(predicates.front_floor_transition_reached(
                        episode["from_level"], episode["to_level"]))
                    if derived != bool(
                            episode["front_floor_transition_reached"]):
                        raise FailClosed(
                            "PRIMARY_EVENT_INCONSISTENT: tier3 predicate "
                            "disagrees with episode flag (state=%s cand=%s "
                            "mode=%s)" % (state_id, rt.candidate_id,
                                          mode.value))
                summary = metrics.summarize(SCENARIO, [episode])
                result = {
                    "schema": RESULT_SCHEMA_ID,
                    "run_mode": run_mode,
                    "scenario": SCENARIO,
                    "state_id": state_id,
                    "state_payload_hash": state_payload_hash,
                    "candidate_id": rt.candidate_id,
                    "candidate_provenance": rt.provenance,
                    "ho_mode": mode.value,
                    "capture_id": capture.capture_id,
                    "capture_bank_hash": capture.bank_hash,
                    "capture_policy_id": capture.capture_policy_id,
                    "receipt": dataclasses.asdict(receipt),
                    "episode": episode,
                    "metrics_summary": summary,
                    "probe_seed": int(probe_seed),
                }
                fname = "%s__%s__%s.json" % (state_id, rt.candidate_id,
                                             mode.value)
                (res_dir / fname).write_bytes(
                    canonical_json_bytes(result))
                results_written += 1
                episodes_by_arm.setdefault(
                    (rt.candidate_id, mode.value), []).append(episode)

    arms = []
    for (cid, mode_val), episodes in sorted(episodes_by_arm.items()):
        primary = metrics.compute_primary_metric(SCENARIO, episodes)
        dense = metrics.compute_dense_progress(SCENARIO, episodes)
        arms.append({
            "candidate_id": cid,
            "ho_mode": mode_val,
            "episodes": len(episodes),
            "primary_metric": primary,
            "dense_progress": dense,
        })
    bank_hashes = sorted({c.bank_hash for c in captures})
    summary_doc = {
        "schema": SUMMARY_SCHEMA_ID,
        "run_mode": run_mode,
        "scenario": SCENARIO,
        "probe_seed": int(probe_seed),
        "num_states": len(states),
        "num_candidates": len(runtimes),
        "ho_modes": [m.value for m in ho_modes],
        "capture_bank_hashes": bank_hashes,
        "capture_generator": (GENERATOR_SYNTHETIC
                              if run_mode == "synthetic" else "REAL"),
        "results_written": results_written,
        "arms": arms,
    }
    (out / "summary.json").write_bytes(canonical_json_bytes(summary_doc))
    return summary_doc


# ---------------------------------------------------------------------------
# SYNTHETIC helpers (plumbing exercises only - never a performance claim)
# ---------------------------------------------------------------------------

def synthetic_states(num_states: int, seed: int = 0,
                     obs_dim: int = 8) -> List[dict]:
    """Normalized FRONT_L2-shaped SYNTHETIC start states (dark corridor,
    player_level=2). Hashes are content-derived so isolation contexts are
    non-trivial. Not materialized from craftax: REAL banks are server-side."""
    import random as _random
    states = []
    for i in range(num_states):
        rng = _random.Random(seed * 1000 + i)
        body = {
            "state_id": "SYNSTATE%04d" % i,
            "schema": "mechanism_UED.memory_study_syn_state/v1",
            "generator": GENERATOR_SYNTHETIC,
            "player_level": START_FLOOR,
            "light_level": 0,
            "down_ladders": [],
            "up_ladders": [[int(rng.randint(0, 7)), int(rng.randint(0, 7))]],
            "monsters_killed_to_clear_level": 0,
            "timestep": 0,
            "task_embedding": [round(rng.random(), 6) for _ in range(4)],
            "inventory": [0] * 4,
            "position": [int(rng.randint(0, 7)), int(rng.randint(0, 7))],
            "entities": [{"mask": True, "level": START_FLOOR,
                          "health": 1.0, "category": "mob"}],
        }
        body["task_embedding_hash"] = sha256_hex(
            canonical_json_bytes(body["task_embedding"]))
        body["inventory_hash"] = sha256_hex(
            canonical_json_bytes(body["inventory"]))
        body["position_hash"] = sha256_hex(
            canonical_json_bytes(body["position"]))
        body["entities_hash"] = sha256_hex(
            canonical_json_bytes(body["entities"]))
        body["payload_sha256"] = sha256_hex(canonical_json_bytes(
            {k: v for k, v in body.items() if k != "payload_sha256"}))
        states.append(body)
    return states


def make_synthetic_candidate(candidate_id: str, success_bias: float = 0.5,
                             obs_dim: int = 8,
                             provenance: Optional[dict] = None
                             ) -> CandidateRuntime:
    """Deterministic mock candidate for SYNTHETIC end-to-end runs.

    step_fn folds each observation row into a memory trace (tuple of row
    hashes). rollout_fn decides the episode outcome from a hash of
    (candidate_id, state_id, memory-trace signature) thresholded by
    success_bias - so the three HO modes can produce different, reproducible
    outcomes without any jax. This is plumbing, not a performance model."""
    if not (0.0 <= float(success_bias) <= 1.0):
        raise FailClosed("SUCCESS_BIAS_OUT_OF_RANGE")
    params = {"w": [[0.1, 0.2], [0.3, 0.4]], "candidate_id": candidate_id}
    initial_memory: Tuple[str, ...] = ()

    def step_fn(_params: Any, memory: Any, obs_row: Any) -> Any:
        row_hash = hashlib.sha256(
            canonical_json_bytes([float(x) for x in obs_row])).hexdigest()[:16]
        return tuple(memory) + (row_hash,)

    def rollout_fn(state: dict, memory: Any) -> dict:
        state_id = str(state.get("state_id"))
        trace_sig = hashlib.sha256(
            canonical_json_bytes(list(memory))).hexdigest()
        decision = int(hashlib.sha256(
            ("%s|%s|%s" % (candidate_id, state_id, trace_sig))
            .encode("utf-8")).hexdigest(), 16) % 10000
        success = decision < int(round(float(success_bias) * 10000))
        progress = (1.0 if success else
                    round((decision % 100) / 100.0 * 0.9, 4))
        to_level = EXIT_FLOOR if success else START_FLOOR
        return {
            "scenario": SCENARIO,
            "valid_start": True,
            "episode_id": state_id,
            "front_floor_transition_reached": success,
            "corridor_exit_reached": success,
            "defeat_kobold": False,
            "graph_distance_progress": progress,
            "timesteps": 512 if success else 4096,
            "timed_out": not success,
            "player_died": False,
            "classified_label": ("SYN_SUCCESS" if success
                                 else "SYN_NO_TRANSITION"),
            "from_level": START_FLOOR,
            "to_level": to_level,
        }

    return CandidateRuntime(
        candidate_id=candidate_id,
        params=params,
        initial_memory=initial_memory,
        step_fn=step_fn,
        rollout_fn=rollout_fn,
        provenance=provenance or {"generator": GENERATOR_SYNTHETIC,
                                  "note": "SYNTHETIC_TEST_ONLY mock runtime"})