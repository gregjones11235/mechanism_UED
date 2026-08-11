"""BehaviorClipSelector (task section 1 / 3).

Selects LIMITED windows around anomalies for the review board — the board
never receives whole episodes, only clips justified by anomaly ids. Overlapping
windows merge (no duplicated evidence), clips are capped (bounded context), and
clip ids are content hashes (replay-stable).

CC3 fix3 (§7): the bounds are TWO independent hard caps, both enforced here
with an explicit drop count (NO silent truncation):

  * MAX_CLIPS_PER_EPISODE        — per-episode cap (earliest windows win,
                                   deterministic order);
  * MAX_CLIPS_PER_REVIEW_WINDOW  — per-review-window cap over all episodes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import Field

from d052.bagr_ued import constants as C
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
    def __init__(self, context_steps: int = 4,
                 max_clips_per_episode: int = C.MAX_CLIPS_PER_EPISODE,
                 max_clips_per_window: int = C.MAX_CLIPS_PER_REVIEW_WINDOW
                 ) -> None:
        self.context_steps = context_steps
        self.max_clips_per_episode = max_clips_per_episode
        self.max_clips_per_window = max_clips_per_window

    def select(self, bundle: TrajectoryEvidenceBundle,
               anomalies: List[AnomalyCandidate]
               ) -> Tuple[List[BehaviorClip], int]:
        """Return (clips, dropped_count). NO silent cap: clips dropped by the
        per-episode OR per-window cap are counted and surfaced to the caller
        (and recorded in the controller certificate)."""
        by_ep: Dict[str, List[AnomalyCandidate]] = {}
        for a in anomalies:
            by_ep.setdefault(a.episode_id, []).append(a)

        clips: List[BehaviorClip] = []
        dropped = 0
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
            ep_clips: List[BehaviorClip] = []
            for m in merged:
                span = EvidenceSpan(episode_id=episode_id,
                                    start_step=m["lo"], end_step=m["hi"])
                payload = dict(episode_id=episode_id,
                               span=span.span_hash_payload,
                               reason_anomaly_ids=sorted(set(m["ids"])))
                ep_clips.append(BehaviorClip(
                    clip_id=f"clip:{canonical_sha256(payload)[:12]}",
                    episode_id=episode_id,
                    span=span,
                    reason_anomaly_ids=sorted(set(m["ids"]))))
            # CC3 fix3 §7: per-episode cap (deterministic earliest-first)
            dropped += max(0, len(ep_clips) - self.max_clips_per_episode)
            clips.extend(ep_clips[:self.max_clips_per_episode])
        clips.sort(key=lambda c: (c.episode_id, c.span.start_step, c.clip_id))
        # CC3 fix3 §7: per-review-window cap over the whole batch
        dropped += max(0, len(clips) - self.max_clips_per_window)
        return clips[:self.max_clips_per_window], dropped
