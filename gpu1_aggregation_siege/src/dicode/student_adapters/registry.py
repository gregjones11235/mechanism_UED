"""Shared StudentAdapter registry, profile loading, and runtime overrides.

This is the ONE registry: CC2/CC3 consume it, they must not fork a parallel
framework.  Everything is fail-closed — unknown candidates, missing adapters,
placeholder identities, or profiles carrying absolute/server paths all raise.

Profiles are pure YAML (logical identity + interface parameters only, no
checkpoint paths, no checkpoint payloads).  Runtime overrides are parsed from
plain ``key=value`` argv entries (no hydra dependency).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .checkpoint_codec import ALL_FORMATS
from .identity import StudentIdentity, validate_identity
from .protocol import MemoryFieldSpec, MemorySpec


class StudentRegistryError(RuntimeError):
    """Raised on any registry/profile/override violation (fail closed)."""


# --- profile schema ---------------------------------------------------------

_REQUIRED_PROFILE_KEYS = (
    "candidate_id", "architecture_family", "checkpoint_format",
    "global_step", "total_env_steps", "params_sha256", "source_commit",
    "observation_shape", "action_count", "memory",
)
_OPTIONAL_PROFILE_KEYS = ("notes",)

# Strings that indicate an absolute / server path, which committed profiles
# must never contain.
_ABS_PATH_RE = re.compile(r"^([A-Za-z]:[/\\]|[/\\~])")
_IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_SERVER_MARKERS = ("//", "\\\\", "/home/", "oseasy@", "ssh://", "scp://")


def _scan_for_paths(value: Any, where: str) -> None:
    """Raise if any string under ``value`` looks like an absolute/server path."""
    if isinstance(value, str):
        if _ABS_PATH_RE.match(value) or "://" in value or _IPV4_RE.search(value):
            raise StudentRegistryError(f"profile field {where} carries a path-like value {value!r}; profiles are logical identity only")
        for marker in _SERVER_MARKERS:
            if marker in value:
                raise StudentRegistryError(f"profile field {where} carries server marker {marker!r} in {value!r}")
    elif isinstance(value, Mapping):
        for key, sub in value.items():
            _scan_for_paths(key, f"{where}.key")
            _scan_for_paths(sub, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, sub in enumerate(value):
            _scan_for_paths(sub, f"{where}[{i}]")


@dataclass(frozen=True)
class StudentProfile:
    """Logical identity + interface parameters of one Student candidate."""

    profile_name: str
    candidate_id: str
    architecture_family: str
    checkpoint_format: str
    global_step: int
    total_env_steps: int
    params_sha256: str
    source_commit: str
    observation_shape: tuple[int, ...]
    action_count: int
    memory_mode: str
    memory_fields: Mapping[str, MemoryFieldSpec]
    notes: Mapping[str, Any] = field(default_factory=dict)

    def memory_spec(self) -> MemorySpec:
        return MemorySpec(fields=dict(self.memory_fields), mode=self.memory_mode)

    def expected_identity(self) -> StudentIdentity:
        """Build the validated identity this profile expects at mount time."""
        return validate_identity(StudentIdentity(
            candidate_id=self.candidate_id,
            architecture_family=self.architecture_family,
            checkpoint_format=self.checkpoint_format,
            global_step=self.global_step,
            total_env_steps=self.total_env_steps,
            params_sha256=self.params_sha256,
            source_commit=self.source_commit,
            observation_shape=self.observation_shape,
            action_count=self.action_count,
            memory_spec_hash=self.memory_spec().spec_hash(),
        ))


def default_profile_dir() -> Path:
    """Repo-relative profile directory (never a server path)."""
    return Path(__file__).resolve().parents[3] / "conf" / "student_profiles"


def load_student_profile(path: str | Path) -> StudentProfile:
    """Load and validate one YAML profile; fail closed on any violation."""
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment dependent
        raise StudentRegistryError(f"BLOCKED_ENVIRONMENT: pyyaml required for profiles: {exc}") from exc

    profile_path = Path(path)
    if not profile_path.is_file():
        raise StudentRegistryError(f"student profile missing: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise StudentRegistryError(f"profile {profile_path} is not a mapping")

    keys = set(data.keys())
    missing = [k for k in _REQUIRED_PROFILE_KEYS if k not in keys]
    if missing:
        raise StudentRegistryError(f"profile {profile_path} missing required keys {missing}")
    unknown = keys - set(_REQUIRED_PROFILE_KEYS) - set(_OPTIONAL_PROFILE_KEYS)
    if unknown:
        raise StudentRegistryError(f"profile {profile_path} carries unknown keys {sorted(unknown)} (fail closed)")
    if "checkpoint_path" in keys or "checkpoint_dir" in keys:
        raise StudentRegistryError("profiles must never contain checkpoint paths (runtime override only)")

    _scan_for_paths(data, where="profile")

    if data["checkpoint_format"] not in ALL_FORMATS:
        raise StudentRegistryError(
            f"profile {profile_path}: unknown checkpoint_format {data['checkpoint_format']!r}; known: {list(ALL_FORMATS)}")

    shape = data["observation_shape"]
    if not isinstance(shape, (list, tuple)) or not shape or any(int(x) <= 0 for x in shape):
        raise StudentRegistryError(f"profile {profile_path}: observation_shape must be a non-empty positive list")

    memory = data["memory"]
    if not isinstance(memory, dict) or "mode" not in memory or "fields" not in memory:
        raise StudentRegistryError(f"profile {profile_path}: memory needs 'mode' and 'fields'")
    fields_raw = memory["fields"]
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise StudentRegistryError(f"profile {profile_path}: memory.fields must be a non-empty mapping")
    memory_fields: dict[str, MemoryFieldSpec] = {}
    for name, spec in fields_raw.items():
        if not isinstance(spec, dict) or "shape" not in spec:
            raise StudentRegistryError(f"profile {profile_path}: memory field {name!r} needs 'shape'")
        fshape = tuple(None if d is None else int(d) for d in spec["shape"])
        memory_fields[str(name)] = MemoryFieldSpec(shape=fshape, dtype=str(spec.get("dtype", "float32")))

    notes = data.get("notes", {}) or {}
    if not isinstance(notes, dict):
        raise StudentRegistryError(f"profile {profile_path}: notes must be a mapping")

    profile = StudentProfile(
        profile_name=profile_path.stem,
        candidate_id=str(data["candidate_id"]),
        architecture_family=str(data["architecture_family"]),
        checkpoint_format=str(data["checkpoint_format"]),
        global_step=int(data["global_step"]),
        total_env_steps=int(data["total_env_steps"]),
        params_sha256=str(data["params_sha256"]),
        source_commit=str(data["source_commit"]),
        observation_shape=tuple(int(x) for x in shape),
        action_count=int(data["action_count"]),
        memory_mode=str(memory["mode"]),
        memory_fields=memory_fields,
        notes=notes,
    )
    profile.expected_identity()  # validates all identity fields fail-closed
    return profile


# --- registry ---------------------------------------------------------------

AdapterFactory = Callable[[StudentProfile], Any]


class StudentAdapterRegistry:
    """Fail-closed registry: candidate_id → adapter factory.

    ``resolve`` raises when the candidate is unknown, no factory was
    registered, or the produced adapter does not structurally satisfy the
    StudentAdapter protocol.  Nothing is ever guessed or defaulted.
    """

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, candidate_id: str, factory: AdapterFactory) -> None:
        if not candidate_id or not str(candidate_id).strip():
            raise StudentRegistryError("candidate_id is empty")
        if not callable(factory):
            raise StudentRegistryError(f"factory for {candidate_id!r} is not callable")
        if candidate_id in self._factories:
            raise StudentRegistryError(f"candidate_id {candidate_id!r} already registered (double registration)")
        self._factories[str(candidate_id)] = factory

    def candidates(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def resolve(self, profile: StudentProfile) -> Any:
        from .protocol import StudentAdapter  # runtime_checkable

        factory = self._factories.get(profile.candidate_id)
        if factory is None:
            raise StudentRegistryError(
                f"no adapter registered for candidate {profile.candidate_id!r} "
                f"(known: {list(self.candidates())}); never guess")
        adapter = factory(profile)
        if not isinstance(adapter, StudentAdapter):
            raise StudentRegistryError(
                f"adapter for {profile.candidate_id!r} does not satisfy the StudentAdapter protocol")
        return adapter


# --- runtime overrides ------------------------------------------------------

RUNTIME_OVERRIDE_KEYS = (
    "student.profile",
    "student.checkpoint_path",
    "student.expected_params_sha256",
    "student.expected_source_commit",
)


def resolve_runtime_overrides(argv: Sequence[str]) -> dict[str, str]:
    """Parse ``key=value`` argv entries (pure python, no hydra).

    Only ``student.*`` keys are considered; unknown ``student.*`` keys raise
    (fail closed), duplicates raise, empty values raise.  Non-``student.*``
    entries are ignored so this composes with other argv consumers.
    """
    out: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        key = key.strip()
        if not key.startswith("student."):
            continue
        if key not in RUNTIME_OVERRIDE_KEYS:
            raise StudentRegistryError(
                f"unknown runtime override {key!r}; allowed: {list(RUNTIME_OVERRIDE_KEYS)}")
        if key in out:
            raise StudentRegistryError(f"duplicate runtime override {key!r}")
        if not value.strip():
            raise StudentRegistryError(f"runtime override {key!r} has an empty value")
        out[key] = value.strip()
    return out
