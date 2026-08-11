"""BehaviorClipSelector (task section 1 / 3).

Selects LIMITED windows around anomalies for the review board — the board
never receives whole episodes, only clips justified by anomaly ids. Overlapping
windows merge (no duplicated evidence), clips are capped (bounded context), and
clip ids are content hashes (replay-stable).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import Field

from d052.bagr_ued.event_extractor import AnomalyCandidate
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.trajectory_evidence import (
    EvidenceSpan,
    TrajectoryEvidenceBundle,
)
from d052.schemas.common import CanonicalModel


class BehaviorClip(CanonicalModel):
    clip_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    span: EvidenceSpan
    reason_anomaly_ids: List[str] = Field(min_length=1)

    @property
    def span_hash(self) -> str:
        return canonical_sha256(self.span.span_hash_payload)


class BehaviorClipSelector:
    def __init__(self, context_steps: int = 4, max_clips: int = 16) -> None:
        self.context_steps = context_steps
        self.max_clips = max_clips

    def select(self, bundle: TrajectoryEvidenceBundle,
               anomalies: List[AnomalyCandidate]
               ) -> Tuple[List[BehaviorClip], int]:
        """Return (clips, dropped_count). NO silent cap: if more windows exist
        than max_clips, the caller receives the drop count and must log it."""
        by_ep: Dict[str, List[AnomalyCandidate]] = {}
        for a in anomalies:
            by_ep.setdefault(a.episode_id, []).append(a)

        clips: List[BehaviorClip] = []
        for episode_id in sorted(by_ep):
            ep = bundle.episode(episode_id)
            max_step = max((s.step_index for s in ep.steps), default=0)
            windows: List[dict] = []
            for a in sorted(by_ep[episode_id],
                            key=lambda x: x.evidence_span.start_step):
                lo = max(0, a.evidence_span.start_step - self.context_steps)
                hi = min(max_step, a.evidence_span.end_step + self.context_steps)
                windows.append(dict(lo=lo, hi=hi, anomaly_id=a.anomaly_id))
            # merge overlapping windows; keep their anomaly provenance
            merged: List[dict] = []
            for w in windows:
                if merged and w["lo"] <= merged[-1]["hi"] + 1:
                    merged[-1]["hi"] = max(merged[-1]["hi"], w["hi"])
                    merged[-1]["ids"].append(w["anomaly_id"])
                else:
                    merged.append(dict(lo=w["lo"], hi=w["hi"],
                                       ids=[w["anomaly_id"]]))
            for m in merged:
                span = EvidenceSpan(episode_id=episode_id,
                                    start_step=m["lo"], end_step=m["hi"])
                payload = dict(episode_id=episode_id,
                               span=span.span_hash_payload,
                               reason_anomaly_ids=sorted(set(m["ids"])))
                clips.append(BehaviorClip(
                    clip_id=f"clip:{canonical_sha256(payload)[:12]}",
                    episode_id=episode_id,
                    span=span,
                    reason_anomaly_ids=sorted(set(m["ids"]))))
        clips.sort(key=lambda c: (c.episode_id, c.span.start_step, c.clip_id))
        dropped = max(0, len(clips) - self.max_clips)
        return clips[:self.max_clips], dropped
