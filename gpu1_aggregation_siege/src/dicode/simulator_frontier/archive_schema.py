"""Serializable frontier archive metadata."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FrontierArchiveEntry:
    state_id: str
    source_checkpoint_id: str
    source_episode_id: str
    source_seed: int
    source_timestep: int
    capture_reason: str
    floor: int
    gate_progress: float
    health_band: str
    threat_band: str
    resource_band: str
    inventory_stage: str
    achievement_snapshot: Mapping[str, Any]
    terminal: bool
    memory_mode: str
    encoded_state_ref: str
    state_hash: str
    provenance_hash: str
    created_at: str

    def bucket(self) -> tuple[Any, ...]:
        return (self.floor, self.health_band, self.threat_band, self.resource_band,
                self.inventory_stage, bool(self.terminal))
