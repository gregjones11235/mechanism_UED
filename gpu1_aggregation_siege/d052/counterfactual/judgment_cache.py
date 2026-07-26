"""Content-addressed judgment cache + deterministic signal reconstruction.

Task §Judgment cache 读取与确定性重放 (gate 2). A judgment cache stores one
RoleJudgment per (pool_hash, candidate_id, role, judgment_prompt_hash) and is itself
bound by a content ``cache_hash``. From a fixed cache + selector config the package
reconstructs SelectorSignals DETERMINISTICALLY (per-role rank_percentile_v1 over the
cached raw headline scores), so the same (pool, judgments, selector config, seed)
reproduces a bit-identical selection_hash -- independent of insertion order.

This is the seam where the real Modeler CC judgment cache will plug in: today the
offline harness fills it with clearly-labeled deterministic FIXTURE judgments; the
read/replay contract here is what guarantees replayability once real judgments
arrive. No silent recomputation, no provider calls.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from d052.counterfactual._hash import sha256_hex
from d052.normalization.rank_percentile import normalized_map, rank_percentile_v1
from d052.schemas.candidate import CandidatePool
from d052.schemas.roles import RoleJudgment, ScoringRole
from d052.schemas.selector import SelectorConfig
from d052.selectors.base import CandidateSignals, SelectorSignals


class JudgmentCacheError(Exception):
    MISSING_JUDGMENT = "MISSING_JUDGMENT"
    DUPLICATE_KEY_CONFLICT = "DUPLICATE_KEY_CONFLICT"
    CACHE_HASH_MISMATCH = "CACHE_HASH_MISMATCH"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def cache_key_hash(pool_hash: str, candidate_id: str, role: str,
                   judgment_prompt_hash: str) -> str:
    """Content key binding a judgment to its pool + candidate + role + prompt."""
    return sha256_hex({
        "pool_hash": pool_hash,
        "candidate_id": candidate_id,
        "role": role,
        "judgment_prompt_hash": judgment_prompt_hash,
    })


class JudgmentCache:
    """An ordered, content-addressed store of role judgments (deterministic)."""

    def __init__(self, pool_hash: str, judgment_prompt_hash: str) -> None:
        self.pool_hash = pool_hash
        self.judgment_prompt_hash = judgment_prompt_hash
        self._by_key: Dict[str, RoleJudgment] = {}

    def put(self, judgment: RoleJudgment) -> str:
        key = cache_key_hash(self.pool_hash, judgment.candidate_id,
                             judgment.role.value, self.judgment_prompt_hash)
        existing = self._by_key.get(key)
        if existing is not None and existing.model_dump() != judgment.model_dump():
            raise JudgmentCacheError(
                JudgmentCacheError.DUPLICATE_KEY_CONFLICT,
                f"conflicting judgment for key {key} (candidate="
                f"{judgment.candidate_id}, role={judgment.role.value}); a judgment "
                f"cache entry is immutable once written (NO silent overwrite)")
        self._by_key[key] = judgment
        return key

    def put_many(self, judgments: List[RoleJudgment]) -> None:
        for j in judgments:
            self.put(j)

    def get(self, candidate_id: str, role: ScoringRole) -> Optional[RoleJudgment]:
        key = cache_key_hash(self.pool_hash, candidate_id, role.value,
                             self.judgment_prompt_hash)
        return self._by_key.get(key)

    def require(self, candidate_id: str, role: ScoringRole) -> RoleJudgment:
        j = self.get(candidate_id, role)
        if j is None:
            raise JudgmentCacheError(
                JudgmentCacheError.MISSING_JUDGMENT,
                f"no cached judgment for candidate={candidate_id} role={role.value} "
                f"(a selection may not silently drop a candidate's judgment)")
        return j

    def entries(self) -> List[dict]:
        """Sorted, canonical entries (for hashing / serialization)."""
        out = []
        for key in sorted(self._by_key):
            j = self._by_key[key]
            out.append({"key_hash": key, "judgment": j.model_dump(mode="json")})
        return out

    def cache_hash(self) -> str:
        """Deterministic content hash over all cached judgments (order-insensitive)."""
        return sha256_hex({
            "pool_hash": self.pool_hash,
            "judgment_prompt_hash": self.judgment_prompt_hash,
            "entries": self.entries(),
        })

    def build_signals(self, pool: CandidatePool, config: SelectorConfig,
                      modeler_bonus_by_id: Optional[Mapping[str, float]] = None
                      ) -> SelectorSignals:
        """Reconstruct SelectorSignals deterministically from the cache.

        For each configured scoring role, the raw headline scores across the whole
        pool are normalized with rank_percentile_v1 (per-role independent), then
        assembled into CandidateSignals. The critic's hard-veto bit + normalized
        penalty come from the cached critic judgment. modeler_bonus is injected per
        candidate (0.0 for modeler-OFF arm B). Identical cache + config -> identical
        signals regardless of candidate/insertion order (gate 2).
        """
        bonus = dict(modeler_bonus_by_id or {})
        roles: List[ScoringRole] = list(config.roles)
        ids = [c.task_id for c in pool.candidates]

        # per-role raw headline column across the pool, then normalize independently
        norm_by_role: Dict[ScoringRole, Dict[str, float]] = {}
        for role in roles:
            raw = [(cid, self.require(cid, role).headline_score) for cid in ids]
            norm_by_role[role] = normalized_map(rank_percentile_v1(role, raw))

        critic_in_roles = ScoringRole.CRITIC in roles
        signals: List[CandidateSignals] = []
        for cid in ids:
            role_scores = {role.value: norm_by_role[role][cid] for role in roles}
            critic_reject = False
            critic_penalty = 0.0
            if critic_in_roles:
                cj = self.require(cid, ScoringRole.CRITIC)
                critic_reject = bool(cj.critic_reject)
                critic_penalty = norm_by_role[ScoringRole.CRITIC][cid]
            signals.append(CandidateSignals(
                candidate_id=cid,
                role_scores=role_scores,
                critic_reject=critic_reject,
                critic_penalty=critic_penalty,
                cost=1.0,
                modeler_bonus=float(bonus.get(cid, 0.0)),
            ))
        return SelectorSignals(pool_hash=pool.pool_hash, candidates=signals)
