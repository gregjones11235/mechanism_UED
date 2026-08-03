"""Simulator probe abstraction + staged funnel (task §4).

The probe is the "simulator as verifier/feedback source" half of the loop:
candidates are standard-reset, environment-level TaskParams; the probe runs
Student and Reference episodes and returns ONLY coarse episode-level
statistics (ReferenceOutputGuard enforced at the runner boundary).

Runners
-------
* ``DeterministicSymbolicProbeRunner`` — the ONLY runner executable in this
  environment: no JAX/Craftax exists locally, so probes are deterministic
  symbolic rollouts derived from the candidate hash (reproducible, seedless).
  It is honest about itself: ``real_simulator=False`` and
  ``status=BLOCKED_NO_LOCAL_CRAFTAX`` are carried on every metric set through
  the candidate's ``real_adapter_status``.
* ``CraftaxPreflightProbeRunner`` — the real-Craftax SEAM, selectively reusing
  the JAX-free core of skill-preflight-ued_Mason (``skill_preflight_core``:
  route / PreflightResult / prereq readiness). Construction FAILS CLOSED while
  ``REAL_SIMULATOR_PROBE_AUTHORIZED`` is false or jax/craftax cannot import;
  nothing in this package may pretend otherwise.

Staged funnel (task §4): 64 raw candidates -> L1 legality/static -> L2 fast
probe (Student 2ep, Reference 1ep; legality/reset/step, behavior activation,
coarse progress, too-hard/too-easy via ``route``) keeping ~24 -> L3 full probe
(Student 4-8ep, Reference 2-4ep; regret, learnability, front progress, global
retention, simulator cost + family-diversity penalty) keeping 12 dynamic UED
slots, plus 4 frozen global canonical anchors = final batch of 16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    ProbeMetrics,
)
from d052.feedback_llm_ued.formal_isolation import ReferenceOutputGuard
from d052.feedback_llm_ued.skill_preflight_core import (
    Decision,
    PreflightResult,
    route,
)

_HEX_UNIT = float(0xFFFFFFFF)


class ProbeRunnerBlocked(RuntimeError):
    """Fail-closed signal that a real-simulator probe is not available."""


@runtime_checkable
class SimulatorProbeRunner(Protocol):
    runner_id: str
    real_simulator: bool
    status: str

    def probe(self, candidate: CandidateEnvironment, *, stage: str,
              student_episodes: int, reference_episodes: int) -> ProbeMetrics:
        ...


def _unit(candidate_hash: str, index: int) -> float:
    """Deterministic pseudo-uniform in [0,1] from one 8-hex slice of the hash."""
    seg = candidate_hash[index * 8:(index + 1) * 8]
    return int(seg, 16) / _HEX_UNIT


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def assert_episode_budget(stage: str, student_episodes: int,
                          reference_episodes: int) -> None:
    """Shared per-stage episode-budget gate (symbolic + real probe seams)."""
    if stage == "fast":
        if student_episodes > C.STAGE1_STUDENT_EPISODES or \
                reference_episodes > C.STAGE1_REFERENCE_EPISODES:
            raise ValueError(
                "FAST_PROBE_EPISODE_BUDGET_EXCEEDED: "
                f"student={student_episodes} reference={reference_episodes}")
    elif stage == "full":
        if not (C.STAGE2_STUDENT_EPISODES_MIN <= student_episodes
                <= C.STAGE2_STUDENT_EPISODES_MAX):
            raise ValueError(
                f"FULL_PROBE_STUDENT_EPISODES_OUT_OF_RANGE: "
                f"{student_episodes}")
        if not (C.STAGE2_REFERENCE_EPISODES_MIN <= reference_episodes
                <= C.STAGE2_REFERENCE_EPISODES_MAX):
            raise ValueError(
                f"FULL_PROBE_REFERENCE_EPISODES_OUT_OF_RANGE: "
                f"{reference_episodes}")
    else:
        raise ValueError(f"ILLEGAL_PROBE_STAGE: {stage!r}")


def static_legality_check(candidate: CandidateEnvironment) -> Tuple[bool, str]:
    """L1 static gate: mock-namespaced legality + adapter honesty."""
    if not candidate.legality_hint.startswith("MOCK_ONLY"):
        return False, "static_fail:legality_hint_missing"
    if candidate.real_adapter_status != C.REAL_SIMULATOR_PROBE_STATUS:
        return False, "static_fail:unexpected_real_adapter_status"
    if not candidate.candidate_hash:
        return False, "static_fail:missing_candidate_hash"
    return True, ""


class DeterministicSymbolicProbeRunner:
    """Deterministic symbolic probe (the only runner executable locally).

    Every metric is a pure function of the candidate hash + episode counts, so
    two runs over the same candidate batch are bit-identical. The Reference
    output path is guarded at the boundary: only the allowed coarse fields may
    ever leave ``probe``.
    """

    runner_id = "mock.feedback_llm_ued.symbolic_probe.v1"
    real_simulator = False
    status = C.REAL_SIMULATOR_PROBE_STATUS

    def __init__(self) -> None:
        self._guard = ReferenceOutputGuard()
        self.probe_calls = 0
        self.total_transitions = 0

    def probe(self, candidate: CandidateEnvironment, *, stage: str,
              student_episodes: int, reference_episodes: int) -> ProbeMetrics:
        if stage not in ("fast", "full"):
            raise ValueError(f"ILLEGAL_PROBE_STAGE: {stage!r}")
        if student_episodes <= 0 or reference_episodes <= 0:
            raise ValueError("ILLEGAL_PROBE_EPISODES: must be positive")
        self._check_episode_budget(stage, student_episodes, reference_episodes)

        h = candidate.candidate_hash
        fam_idx = C.ENVIRONMENT_FAMILIES.index(candidate.environment_family) \
            if candidate.environment_family in C.ENVIRONMENT_FAMILIES else 0
        fam_bias = ((fam_idx % 3) - 1) * 0.05

        difficulty = _unit(h, 0)
        student_sr = _clamp01(0.95 - difficulty + 0.1 * (_unit(h, 1) - 0.5)
                              + fam_bias)
        behavior_activation = _clamp01(0.2 + 0.7 * _unit(h, 2))
        front_progress = _clamp01(0.6 * student_sr + 0.4 * _unit(h, 3))
        made_progress = behavior_activation > 0.35 or front_progress > 0.25
        reference_sr = _clamp01(student_sr + 0.15 + 0.25 * _unit(h, 4))
        reference_mean_progress = _clamp01(0.7 * reference_sr
                                           + 0.3 * _unit(h, 5))
        reference_behavior_activation = _clamp01(behavior_activation + 0.10
                                                 + 0.15 * _unit(h, 5))
        global_retention = _clamp01(0.70 + 0.28 * _unit(h, 6))
        regret = max(0.0, reference_sr - student_sr)
        learnability = round(
            max(0.0, 1.0 - abs(student_sr - 0.4) / 0.45), 4)
        transitions = ((student_episodes + reference_episodes)
                       * C.ROLLOUT_LENGTH)

        metrics = ProbeMetrics(
            stage=stage,
            student_success_rate=round(student_sr, 6),
            student_behavior_activation=round(behavior_activation, 6),
            student_front_progress=round(front_progress, 6),
            reference_success_rate=round(reference_sr, 6),
            reference_mean_progress=round(reference_mean_progress, 6),
            reference_behavior_activation=round(
                reference_behavior_activation, 6),
            global_retention=round(global_retention, 6),
            regret=round(regret, 6),
            learnability=learnability,
            simulator_transitions=transitions,
            too_hard=(student_sr < C.PREFLIGHT_LEARNABLE_LOW
                      and not made_progress),
            too_easy=student_sr >= C.PREFLIGHT_TOO_EASY,
            probe_source=C.SOURCE_CANDIDATE_PROBE)
        # Reference/Student payloads leaving the runner MUST be clean of
        # action-guidance carriers (fail-closed).
        self._guard.assert_clean(
            metrics.model_dump(),
            label=f"probe:{candidate.candidate_id}:{stage}")
        self.probe_calls += 1
        self.total_transitions += transitions
        return metrics

    @staticmethod
    def _check_episode_budget(stage: str, student_episodes: int,
                              reference_episodes: int) -> None:
        assert_episode_budget(stage, student_episodes, reference_episodes)


class CraftaxPreflightProbeRunner:
    """Real-Craftax seam (selective reuse of Mason preflight; BLOCKED locally).

    NOT a merge of the Mason branch: only the JAX-free routing core
    (``skill_preflight_core``) is reused. This class is the single authorized
    place to touch the real simulator, and it refuses to construct while the
    authorization flag is false or jax/craftax cannot be imported.
    """

    runner_id = "feedback_llm_ued.craftax_preflight_probe.v1"
    real_simulator = True

    def __init__(self) -> None:
        if not C.REAL_SIMULATOR_PROBE_AUTHORIZED:
            raise ProbeRunnerBlocked(
                f"[{C.REAL_SIMULATOR_PROBE_STATUS}] "
                "REAL_SIMULATOR_PROBE_AUTHORIZED=false this round; the "
                "deterministic symbolic runner is the only permitted probe")
        try:                                   # pragma: no cover - blocked path
            import jax  # noqa: F401
            import craftax  # noqa: F401
        except Exception as exc:               # pragma: no cover
            raise ProbeRunnerBlocked(
                f"[{C.REAL_SIMULATOR_PROBE_STATUS}] jax/craftax import "
                f"failed: {exc}") from exc
        self.status = "READY"                  # pragma: no cover

    def probe(self, candidate: CandidateEnvironment, *, stage: str,
              student_episodes: int,
              reference_episodes: int) -> ProbeMetrics:  # pragma: no cover
        raise ProbeRunnerBlocked(
            f"[{C.REAL_SIMULATOR_PROBE_STATUS}] real Craftax probe is not "
            "reachable in this environment")


@dataclass
class ProbeBatch:
    """Full funnel outcome for one window."""

    window: int
    raw_candidate_ids: List[str] = field(default_factory=list)
    static_rejects: List[dict] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    stage1_results: List[dict] = field(default_factory=list)
    stage1_survivors: List[str] = field(default_factory=list)
    stage2_results: List[dict] = field(default_factory=list)
    dynamic_selected: List[str] = field(default_factory=list)
    anchor_ids: Tuple[str, ...] = C.GLOBAL_CANONICAL_ANCHOR_IDS
    total_simulator_transitions: int = 0

    @property
    def final_batch(self) -> List[str]:
        return list(self.dynamic_selected) + list(self.anchor_ids)

    @property
    def funnel_stats(self) -> Dict[str, int]:
        return dict(
            raw=len(self.raw_candidate_ids),
            static_rejects=len(self.static_rejects),
            duplicates=len(self.duplicates),
            stage1_probed=len(self.stage1_results),
            stage1_survivors=len(self.stage1_survivors),
            stage2_probed=len(self.stage2_results),
            stage2_selected=sum(1 for r in self.stage2_results
                                if r.get("selected")),
            dynamic_selected=len(self.dynamic_selected),
            anchors=len(self.anchor_ids),
            final_batch=len(self.final_batch),
            total_simulator_transitions=self.total_simulator_transitions,
        )


def _fast_preflight(candidate: CandidateEnvironment,
                    runner: SimulatorProbeRunner) -> PreflightResult:
    m = runner.probe(candidate, stage="fast",
                     student_episodes=C.STAGE1_STUDENT_EPISODES,
                     reference_episodes=C.STAGE1_REFERENCE_EPISODES)
    made_progress = (m.student_behavior_activation > 0.35
                     or m.student_front_progress > 0.25)
    decision: Decision = route(m.student_success_rate, made_progress,
                               learnable_low=C.PREFLIGHT_LEARNABLE_LOW,
                               too_easy=C.PREFLIGHT_TOO_EASY)
    return PreflightResult(
        action=decision.action, reason=decision.reason,
        sr=m.student_success_rate, any_partial_progress=made_progress,
        n_episodes=C.STAGE1_STUDENT_EPISODES,
        extra=dict(metrics=m.model_dump()))


def _full_score(m: ProbeMetrics, max_transitions: int) -> float:
    """Deterministic Stage-2 composite: learnability + progress + retention,
    penalized by regret and simulator cost."""
    cost = (m.simulator_transitions / max_transitions) if max_transitions else 0.0
    return (0.30 * m.learnability
            + 0.20 * m.student_front_progress
            + 0.20 * (1.0 - min(1.0, m.regret))
            + 0.15 * m.global_retention
            + 0.10 * m.student_behavior_activation
            - 0.05 * cost)


def run_staged_funnel(candidates: List[CandidateEnvironment],
                      runner: SimulatorProbeRunner, *, window: int,
                      raw_cap: int = C.RAW_CANDIDATES,
                      stage1_keep: int = C.STAGE1_KEEP,
                      stage2_keep: int = C.STAGE2_KEEP) -> ProbeBatch:
    """64 raw -> L1 static -> L2 fast (~24) -> L3 full (12 dynamic) + 4 anchors.

    Deterministic at every cut: static rejects, hash dedup, route-based
    accept/reject, score-sorted trimming with a family-diversity penalty on
    the final greedy pick.
    """
    batch = ProbeBatch(window=window)
    # probe cost is accounted PER BATCH (the runner's counter is cumulative
    # across windows; a window record must show its own window's cost)
    transitions_before = runner.total_transitions
    if len(candidates) > raw_cap:
        raise ValueError(
            f"RAW_CANDIDATE_CAP_EXCEEDED: {len(candidates)} > {raw_cap}")

    # -- L1: static legality + dedup ----------------------------------------
    seen_hashes: set = set()
    stage1_pool: List[CandidateEnvironment] = []
    for cand in candidates:
        batch.raw_candidate_ids.append(cand.candidate_id)
        ok, err = static_legality_check(cand)
        if not ok:
            batch.static_rejects.append(dict(candidate_id=cand.candidate_id,
                                             reason=err))
            continue
        if cand.candidate_hash in seen_hashes:
            batch.duplicates.append(cand.candidate_id)
            continue
        seen_hashes.add(cand.candidate_hash)
        stage1_pool.append(cand)

    # -- L2: fast probe + route ----------------------------------------------
    accepted: List[Tuple[float, CandidateEnvironment, PreflightResult]] = []
    for cand in stage1_pool:
        result = _fast_preflight(cand, runner)
        batch.stage1_results.append(dict(
            candidate_id=cand.candidate_id, action=result.action,
            reason=result.reason, sr=result.sr,
            any_partial_progress=result.any_partial_progress,
            metrics=result.extra["metrics"]))
        if result.action == "accept":
            sort_key = result.extra["metrics"]["student_front_progress"] \
                + result.extra["metrics"]["student_behavior_activation"]
            accepted.append((sort_key, cand, result))
    accepted.sort(key=lambda t: (-t[0], t[1].candidate_id))
    survivors = accepted[:stage1_keep]
    batch.stage1_survivors = [c.candidate_id for _, c, _ in survivors]

    # -- L3: full probe + composite score -------------------------------------
    if not survivors:
        batch.total_simulator_transitions = (runner.total_transitions
                                             - transitions_before)
        return batch
    student_ep = C.STAGE2_STUDENT_EPISODES_MAX
    reference_ep = C.STAGE2_REFERENCE_EPISODES_MAX
    max_transitions = (student_ep + reference_ep) * C.ROLLOUT_LENGTH
    scored: List[Tuple[float, CandidateEnvironment, ProbeMetrics]] = []
    full_probed: Dict[str, Tuple[float, ProbeMetrics]] = {}
    for _key, cand, _res in survivors:
        m_full = runner.probe(cand, stage="full",
                              student_episodes=student_ep,
                              reference_episodes=reference_ep)
        score = _full_score(m_full, max_transitions)
        full_probed[cand.candidate_id] = (score, m_full)
        scored.append((score, cand, m_full))
    scored.sort(key=lambda t: (-t[0], t[1].candidate_id))

    # greedy pick with a family-diversity penalty (deterministic)
    family_counts: Dict[str, int] = {}
    picked: List[Tuple[float, CandidateEnvironment, ProbeMetrics]] = []
    remaining = list(scored)
    while len(picked) < stage2_keep and remaining:
        best_i = None
        best_eff = None
        for i, (score, cand, m) in enumerate(remaining):
            eff = score - 0.10 * family_counts.get(cand.environment_family, 0)
            if best_eff is None or eff > best_eff or \
                    (eff == best_eff
                     and cand.candidate_id
                     < remaining[best_i][1].candidate_id):
                best_eff = eff
                best_i = i
        score, cand, m = remaining.pop(best_i)
        family_counts[cand.environment_family] = \
            family_counts.get(cand.environment_family, 0) + 1
        picked.append((score, cand, m))
    selected_ids = {cand.candidate_id for _s, cand, _m in picked}
    for cand_id in sorted(full_probed):
        score, m = full_probed[cand_id]
        batch.stage2_results.append(dict(
            candidate_id=cand_id, score=round(score, 6),
            selected=cand_id in selected_ids, metrics=m.model_dump()))
    # preserve deterministic selection order (score-desc greedy) in the output
    batch.dynamic_selected = [cand.candidate_id for _s, cand, _m in picked]
    batch.total_simulator_transitions = (runner.total_transitions
                                         - transitions_before)
    return batch
