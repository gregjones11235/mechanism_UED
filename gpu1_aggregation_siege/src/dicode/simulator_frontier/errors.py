"""Errors used by the simulator-frontier foundation layer."""


class SimulatorFrontierError(Exception):
    """Base class for explicit, fail-closed foundation errors."""


class InvalidEvidenceError(SimulatorFrontierError, ValueError):
    """Evidence cannot be safely interpreted."""


class SchemaMismatchError(SimulatorFrontierError, ValueError):
    """Encoded data does not match the expected schema."""


class ProvenanceViolationError(SimulatorFrontierError, ValueError):
    """A data object crosses a forbidden provenance boundary."""


class ArchiveWriteGuardError(SimulatorFrontierError, ValueError):
    """A production archive write failed the internal guard chain (fail closed)."""


class BranchSearchBlockedError(SimulatorFrontierError):
    """A real actual-N branch search cannot honestly proceed (missing artifact/executor)."""


class ProductionBlockedError(SimulatorFrontierError):
    """A production path is blocked waiting on a controller-supplied input (never faked)."""
