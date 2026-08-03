"""Errors used by the simulator-frontier foundation layer."""


class SimulatorFrontierError(Exception):
    """Base class for explicit, fail-closed foundation errors."""


class InvalidEvidenceError(SimulatorFrontierError, ValueError):
    """Evidence cannot be safely interpreted."""


class SchemaMismatchError(SimulatorFrontierError, ValueError):
    """Encoded data does not match the expected schema."""


class ProvenanceViolationError(SimulatorFrontierError, ValueError):
    """A data object crosses a forbidden provenance boundary."""
