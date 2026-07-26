"""Offline Phase-2.5 B/C counterfactual pipeline (canonical_v2).

Runs the matched modeler-ablation OFFLINE and deterministically:

    B = S1_THREE_ROLE          (modeler OFF)
    C = S2_FOUR_ROLE_MODELER   (modeler ON, bonus from the StudentProfile channel)

over a NEW legal shared frozen candidate pool, producing canonical B/C selected-8,
selection hashes, execution-mapping certificates, and a matched-counterfactual
manifest -- and (optionally) registering two DRAFT cells. It performs ZERO training
timesteps (gate 8).

HONESTY (NO_RAW_DATA_NO_STRONG_CLAIM / NO_SILENT_FALLBACK): the Modeler CC's real
Phase-2.5 migration package (live prompts + judgment cache) is NOT on disk yet, so
the role judgments + modeler judgment used here are CLEARLY-LABELED deterministic
offline FIXTURES (SYNTHETIC_FIXTURE / SYNTHETIC OFFLINE FIXTURE), not real Modeler
CC outputs. They exist to exercise and prove the canonical protocol end-to-end; the
real package must be reconciled when it arrives. Nothing here fabricates a strong
empirical claim or silently substitutes real data.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

from d052.achievements import REGISTRY
from d052.cells.registry import CellRegistry, validate_cell_spec
from d052.cells.spec import CellSpec
from d052.counterfactual.ablation import ABLATION_SCORING_ROLES, modeler_ablation_arms
from d052.counterfactual.judgment_cache import JudgmentCache
from d052.counterfactual.manifest import MatchedCounterfactualManifest, build_manifest
from d052.counterfactual.prompts import build_prompt_set, role_judgment_prompt_hash
from d052.counterfactual.protocol import (
    CounterfactualArm,
    MatchedVerification,
    verify_matched_bc,
)
from d052.counterfactual.student_modeler_channel import (
    MODELER_BONUS_WEIGHT,
    ModelerContext,
    build_modeler_context,
    modeler_bonus_for,
    student_profile_hash,
)
from d052.execution.mapper import build_execution_certificate, canonical_compiled_spec
from d052.generation import build_pool
from d052.profiling.modeler import (
    EvidenceCheck,
    MachineFacts,
    ModelerJudgment,
    Recommendation,
    StudentState,
)
from d052.profiling.student_profile import StudentProfile, build_student_profile
from d052.roles.protocol import ROLE_REGISTRY, RoleName
from d052.schemas.candidate import CandidatePool
from d052.schemas.execution import ExecutionMappingCertificate
from d052.schemas.roles import RoleJudgment, ScoringRole
from d052.schemas.selector import SelectionResult
from d052.selectors import select

# --- frozen Phase-2.5 offline parameters (deterministic) -------------------
PHASE25_POOL_ID = "phase25_canonical_shared_frozen_v1"
PHASE25_SEED = 20260726
PHASE25_K = 8
PHASE25_N_CANDIDATES = 16
SYNTHETIC_LABEL = "SYNTHETIC_FIXTURE_deterministic_offline_v1"
WEAKEST_INDEX = 8          # candidate whose target is the student's weakest skill
_WEAK_SR = 0.02
_OTHER_SR = 0.9
_ROLE_NAMES = [RoleName.TUTOR, RoleName.CRITIC, RoleName.EXPLORER]


@dataclass
class Phase25Result:
    pool: CandidatePool
    profile: StudentProfile
    modeler_judgment: ModelerJudgment
    modeler_context: ModelerContext
    student_profile_hash: str
    judgment_prompt_hash: str
    judgment_cache_hash: str
    prompt_set_b_hash: str
    prompt_set_c_hash: str
    arm_b: CounterfactualArm
    arm_c: CounterfactualArm
    verification: MatchedVerification
    selection_b: SelectionResult
    selection_c: SelectionResult
    certificates_b: List[ExecutionMappingCertificate]
    certificates_c: List[ExecutionMappingCertificate]
    manifest: MatchedCounterfactualManifest
    modeler_bonus_by_id: Dict[str, float] = field(default_factory=dict)

    @property
    def b_selected8(self) -> List[str]:
        return list(self.selection_b.selected_ids)

    @property
    def c_selected8(self) -> List[str]:
        return list(self.selection_c.selected_ids)

    @property
    def modeler_selection_change(self) -> int:
        return self.manifest.modeler_canonical_selection_change


# --- deterministic fixtures -------------------------------------------------
def phase25_target_names(n: int = PHASE25_N_CANDIDATES) -> List[str]:
    """The first ``n`` canonical achievement names (sorted) -> one per candidate."""
    names = sorted(REGISTRY.names)
    if n > len(names):
        raise ValueError(f"n={n} exceeds canonical achievement count {len(names)}")
    return names[:n]


def build_phase25_pool(pool_id: str = PHASE25_POOL_ID,
                       n: int = PHASE25_N_CANDIDATES) -> CandidatePool:
    """A NEW legal shared frozen pool: n candidates cand_00.., one canonical target."""
    names = phase25_target_names(n)
    tp = {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 1.0,
          "mob_health_multiplier": 1.0, "mob_damage_multiplier": 1.0}
    raw = [{"task_id": f"cand_{i:02d}", "task_params": dict(tp),
            "target_achievements": [names[i]]} for i in range(n)]
    return build_pool(pool_id, raw)


def build_phase25_student_profile(n: int = PHASE25_N_CANDIDATES) -> StudentProfile:
    """Deterministic held-out SR fixture: all targets strong except the weakest."""
    names = phase25_target_names(n)
    sr = {name: _OTHER_SR for name in names}
    sr[names[WEAKEST_INDEX]] = _WEAK_SR
    return build_student_profile(sr)


def build_phase25_modeler_judgment(profile: StudentProfile,
                                   n: int = PHASE25_N_CANDIDATES) -> ModelerJudgment:
    """A clearly-labeled SYNTHETIC modeler judgment focusing the weakest skill."""
    names = phase25_target_names(n)
    focus = names[WEAKEST_INDEX]
    d = ROLE_REGISTRY[RoleName.MODELER]
    facts = MachineFacts(latest_sr=dict(profile.per_achievement_sr),
                         recent_series=[], forgetting_prefilter=[], num_snapshots=1)
    return ModelerJudgment(
        machine_facts=facts, student_state=StudentState.RISING,
        recommendation=Recommendation.DEPTH,
        guidance=(f"{SYNTHETIC_LABEL}: prioritize the student's weakest modeler-"
                  f"flagged skill ({focus})"),
        siege_foci=[focus], evidence_check=EvidenceCheck.SUPPORTED,
        provider=d.provider, exact_model_id=d.exact_model_id,
        prompt_version=d.prompt_version)


def build_phase25_judgments(pool: CandidatePool) -> List[RoleJudgment]:
    """Deterministic SYNTHETIC per-candidate role judgments (tutor/critic/explorer).

    tutor/explorer headline scores decrease with candidate index (cand_00 best);
    critic penalty is constant with no veto, so the B<->C delta is attributable
    ONLY to the modeler bonus. Labeled synthetic; not real Modeler CC outputs.
    """
    judgments: List[RoleJudgment] = []
    for c in sorted(pool.candidates, key=lambda c: c.task_id):
        i = int(c.task_id.split("_")[1])
        base = round(1.0 - 0.01 * i, 6)

        def _pin(role: RoleName):
            d = ROLE_REGISTRY[role]
            return dict(provider=d.provider, exact_model_id=d.exact_model_id,
                        prompt_version=d.prompt_version)

        judgments.append(RoleJudgment(
            role=ScoringRole.TUTOR, candidate_id=c.task_id,
            scores={"progression_score": base},
            rationale=f"{SYNTHETIC_LABEL}: i={i}", **_pin(RoleName.TUTOR)))
        judgments.append(RoleJudgment(
            role=ScoringRole.EXPLORER, candidate_id=c.task_id,
            scores={"novelty_score": base},
            rationale=f"{SYNTHETIC_LABEL}: i={i}", **_pin(RoleName.EXPLORER)))
        judgments.append(RoleJudgment(
            role=ScoringRole.CRITIC, candidate_id=c.task_id,
            scores={"critic_penalty": 0.1}, critic_reject=False,
            rationale=f"{SYNTHETIC_LABEL}: i={i}", **_pin(RoleName.CRITIC)))
    return judgments


# --- the offline run --------------------------------------------------------
def compute_phase25(pool_id: str = PHASE25_POOL_ID, seed: int = PHASE25_SEED,
                    k: int = PHASE25_K,
                    n: int = PHASE25_N_CANDIDATES) -> Phase25Result:
    """Run the full matched B/C counterfactual offline (in memory, deterministic)."""
    pool = build_phase25_pool(pool_id, n)
    profile = build_phase25_student_profile(n)
    judgment = build_phase25_modeler_judgment(profile, n)
    context = build_modeler_context(profile, judgment)
    sp_hash = student_profile_hash(profile)

    jph = role_judgment_prompt_hash(_ROLE_NAMES)
    prompt_b = build_prompt_set("B", _ROLE_NAMES, modeler_enabled=False)
    prompt_c = build_prompt_set("C", _ROLE_NAMES, modeler_enabled=True,
                                student_profile_channel_id=context.context_hash)

    cache = JudgmentCache(pool.pool_hash, jph)
    cache.put_many(build_phase25_judgments(pool))
    cache_hash = cache.cache_hash()

    arm_b, arm_c = modeler_ablation_arms(
        pool_hash=pool.pool_hash, judgment_cache_hash=cache_hash,
        prompt_set_b=prompt_b, prompt_set_c=prompt_c, k=k, seed=seed,
        roles=list(ABLATION_SCORING_ROLES), modeler_bonus_weight=MODELER_BONUS_WEIGHT,
        student_profile_hash=sp_hash, modeler_context_hash=context.context_hash)

    verification = verify_matched_bc(arm_b, arm_c)          # GATE 1

    signals_b = cache.build_signals(pool, arm_b.selector)   # modeler OFF -> bonus 0
    selection_b = select(arm_b.selector, pool, signals_b)

    bonus_c = {c.task_id: modeler_bonus_for(c, context, modeler_enabled=True)
               for c in pool.candidates}
    signals_c = cache.build_signals(pool, arm_c.selector, modeler_bonus_by_id=bonus_c)
    selection_c = select(arm_c.selector, pool, signals_c)   # GATE 2 (replayable)

    arm_b = arm_b.model_copy(update={"selection_result": selection_b})
    arm_c = arm_c.model_copy(update={"selection_result": selection_c})

    by_id = {c.task_id: c for c in pool.candidates}
    certs_b = [build_execution_certificate(                  # GATE 4
                   by_id[cid], canonical_compiled_spec(by_id[cid], f"phase25_B/{cid}"))
               for cid in selection_b.selected_ids]
    certs_c = [build_execution_certificate(
                   by_id[cid], canonical_compiled_spec(by_id[cid], f"phase25_C/{cid}"))
               for cid in selection_c.selected_ids]
    all_ok = all(cert.executed_as_intended for cert in certs_b + certs_c)

    manifest = build_manifest(                               # binds gates 1-4 + 8
        pool_id=pool.pool_id, pool_hash=pool.pool_hash, arm_b=arm_b, arm_c=arm_c,
        verification=verification, selection_b=selection_b, selection_c=selection_c,
        certificates_b_count=len(certs_b), certificates_c_count=len(certs_c),
        all_certificates_executed_as_intended=all_ok,
        canonical_target_firewall="PASS",
        no_training_attestation={
            "D052_LONG_TRAINING_RUNS": 0, "scope": "single_cell_no_training",
            "timesteps_run": 0, "runner": "no_op_runner",
            "attestation": "offline selection+certification only; no cell launched"})

    return Phase25Result(
        pool=pool, profile=profile, modeler_judgment=judgment,
        modeler_context=context, student_profile_hash=sp_hash,
        judgment_prompt_hash=jph, judgment_cache_hash=cache_hash,
        prompt_set_b_hash=prompt_b.prompt_set_hash,
        prompt_set_c_hash=prompt_c.prompt_set_hash,
        arm_b=arm_b, arm_c=arm_c, verification=verification,
        selection_b=selection_b, selection_c=selection_c,
        certificates_b=certs_b, certificates_c=certs_c, manifest=manifest,
        modeler_bonus_by_id=bonus_c)


# --- artifact emission (no-overwrite) --------------------------------------
def _excl_write_json(path: str, obj) -> None:
    """Write JSON exclusively (fail if exists) -> NO_LEGACY_ARTIFACT_OVERWRITE."""
    text = (obj if isinstance(obj, str)
            else json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)


def emit_phase25_artifacts(result: Phase25Result, output_root: str) -> List[str]:
    """Write all Phase-2.5 artifacts under output_root (no-overwrite). Returns paths."""
    os.makedirs(output_root, exist_ok=True)
    written: List[str] = []

    def w(name: str, obj) -> None:
        p = os.path.join(output_root, name)
        _excl_write_json(p, obj)
        written.append(p)

    w("pool.json", json.loads(result.pool.model_dump_json()))
    w("judgment_cache.json", {
        "pool_hash": result.pool.pool_hash,
        "judgment_prompt_hash": result.judgment_prompt_hash,
        "cache_hash": result.judgment_cache_hash,
        "synthetic_label": SYNTHETIC_LABEL,
    })
    w("modeler_context.json", json.loads(result.modeler_context.model_dump_json()))
    w("arm_b.json", json.loads(result.arm_b.model_dump_json()))
    w("arm_c.json", json.loads(result.arm_c.model_dump_json()))
    w("selection_b.json", json.loads(result.selection_b.model_dump_json()))
    w("selection_c.json", json.loads(result.selection_c.model_dump_json()))
    w("certificates_b.json",
      [json.loads(c.model_dump_json()) for c in result.certificates_b])
    w("certificates_c.json",
      [json.loads(c.model_dump_json()) for c in result.certificates_c])
    w("matched_counterfactual_manifest.json",
      json.loads(result.manifest.model_dump_json()))
    w("summary.json", {
        "pool_id": result.pool.pool_id,
        "pool_hash": result.pool.pool_hash,
        "canonical_b_selected8": result.b_selected8,
        "canonical_c_selected8": result.c_selected8,
        "modeler_canonical_selection_change":
            f"{result.modeler_selection_change}/{result.manifest.selection_change_over}",
        "selection_hash_b": result.selection_b.selection_hash,
        "selection_hash_c": result.selection_c.selection_hash,
        "manifest_hash": result.manifest.manifest_hash,
        "all_certificates_executed_as_intended":
            result.manifest.all_certificates_executed_as_intended,
        "training_timesteps": result.manifest.training_timesteps,
        "synthetic_label": SYNTHETIC_LABEL,
    })
    return written


# --- DRAFT cell registration (no training) ---------------------------------
def build_phase25_cellspecs(result: Phase25Result,
                            created_by: str = "d052_phase25_canonical_offline_harness"
                            ) -> List[CellSpec]:
    """Two DRAFT cell specs (B and C). intended_total_timesteps=0 (no training)."""
    specs = []
    for label, arm, sel in (("B", result.arm_b, result.selection_b),
                            ("C", result.arm_c, result.selection_c)):
        specs.append(CellSpec(
            cell_id=f"phase25_{label}_{arm.selector.selector.value.lower()}",
            protocol_version="canonical_v2",
            title=f"Phase 2.5 canonical arm {label} ({arm.selector.selector.value})",
            hypothesis=(f"arm {label}: modeler "
                        + ("ON" if arm.modeler_enabled else "OFF")
                        + f" selection over {result.pool.pool_id}"),
            pool_id=result.pool.pool_id, pool_hash=result.pool.pool_hash,
            selector=arm.selector, candidate_ids=list(sel.selected_ids),
            selection_hash=sel.selection_hash,
            intended_total_timesteps=0,
            output_dir=f"gpu1_aggregation_siege/phase25_canonical_cells/cell_{label}",
            created_by=created_by))
    return specs


def register_phase25_cells(result: Phase25Result, cells_root: str,
                           actor: str = "d052_phase25_canonical_offline_harness"
                           ) -> List[dict]:
    """Register the two cells as DRAFT (validated structurally, NEVER launched)."""
    registry = CellRegistry(cells_root)
    out = []
    for spec in build_phase25_cellspecs(result):
        problems = validate_cell_spec(spec)         # must be a LEGAL draft
        if problems:
            raise ValueError(f"ILLEGAL_DRAFT_CELL {spec.cell_id}: {problems}")
        rec = registry.register(spec, actor=actor)  # DRAFT only; no training
        out.append({"cell_id": spec.cell_id, "state": rec.state.value,
                    "cell_identity_hash": spec.identity_hash(),
                    "selection_hash": spec.selection_hash,
                    "intended_total_timesteps": spec.intended_total_timesteps})
    return out
