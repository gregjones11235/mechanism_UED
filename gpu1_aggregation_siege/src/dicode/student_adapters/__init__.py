"""Shared StudentAdapter contract layer — the ONLY adapter/loader/registry.

Ownership: CC1/CC4 own this package; CC2/CC3 are consumers of these
contracts and must not fork a parallel framework.

Import discipline: this package's __init__ never triggers a jax import.
jax/numpy heavy lifting lives inside functions of checkpoint_codec (lazy) and
inside concrete architecture adapters (Stage 3+, also lazy).
"""

from .checkpoint_codec import (
    ALL_FORMATS,
    FORMAT_BAKEOFF_PKL,
    FORMAT_CC2_PKL,
    FORMAT_ORBAX_FLAT_TRAINSTATE,
    FORMAT_ORBAX_NESTED_PARAMS,
    FORMAT_PHASE4A_PKL,
    CheckpointCodecError,
    CheckpointFormatNotImplementedError,
    LoadedCheckpoint,
    cc2_params_sha256,
    file_sha256,
    load_cc2_pkl,
    load_checkpoint,
)
from .identity import (
    StudentIdentity,
    StudentIdentityError,
    identity_field_names,
    identity_to_mapping,
    validate_identity,
)
from .protocol import (
    ActionSpec,
    CheckpointSpec,
    MemoryFieldSpec,
    MemorySpec,
    ObsSpec,
    StudentAdapter,
)
from .registry import (
    RUNTIME_OVERRIDE_KEYS,
    StudentAdapterRegistry,
    StudentProfile,
    StudentRegistryError,
    default_profile_dir,
    load_student_profile,
    resolve_runtime_overrides,
)
from .fake import FakeStudentAdapter, fake_params_sha256

__all__ = [
    "ALL_FORMATS", "FORMAT_BAKEOFF_PKL", "FORMAT_CC2_PKL",
    "FORMAT_ORBAX_FLAT_TRAINSTATE", "FORMAT_ORBAX_NESTED_PARAMS",
    "FORMAT_PHASE4A_PKL", "CheckpointCodecError",
    "CheckpointFormatNotImplementedError", "LoadedCheckpoint",
    "cc2_params_sha256", "file_sha256", "load_cc2_pkl", "load_checkpoint",
    "StudentIdentity", "StudentIdentityError", "identity_field_names",
    "identity_to_mapping", "validate_identity",
    "ActionSpec", "CheckpointSpec", "MemoryFieldSpec", "MemorySpec",
    "ObsSpec", "StudentAdapter",
    "RUNTIME_OVERRIDE_KEYS", "StudentAdapterRegistry", "StudentProfile",
    "StudentRegistryError", "default_profile_dir", "load_student_profile",
    "resolve_runtime_overrides",
    "FakeStudentAdapter", "fake_params_sha256",
]
