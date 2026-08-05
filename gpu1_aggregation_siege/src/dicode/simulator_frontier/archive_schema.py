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
    # Additive R3/R9 binding fields (condition 2 + review condition 5 era).
    # Empty defaults mean "unbound": bound-ness is enforced by
    # student_binding.assert_entry_bound, never implied here.
    source_student_identity_hash: str = ""
    source_parameter_hash: str = ""
    source_memory_spec_hash: str = ""
    capture_student_id: str = ""
    discovery_provenance: str = ""
    # Director handoff (E3-DS section 3): a frontier entry is bound to the
    # selected Student's memory mode and the runtime bundle that captured it —
    # a state captured by one arm can never be handed to the other's training.
    source_memory_mode: str = ""
    runtime_bundle_hash: str = ""

    def bucket(self) -> tuple[Any, ...]:
        return (self.floor, self.health_band, self.threat_band, self.resource_band,
                self.inventory_stage, bool(self.terminal))
