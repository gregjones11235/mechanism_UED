"""Fail-closed, atomic persistence for process-isolated training sessions.

The training checkpoint contains the Flax ``TrainState``.  This module stores
the surrounding session state (RNGs, counters, curriculum references and
provenance) beside that checkpoint so a fresh process can resume a complete
UED boundary instead of restoring weights only.

The payload is intentionally opaque to this module.  Callers are responsible
for constructing a trusted in-memory mapping; the store provides atomic
publication and integrity checks, but never silently repairs a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


class BoundaryIntegrityError(RuntimeError):
    """Raised when a boundary is missing, malformed or fails an integrity check."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | os.PathLike[str]) -> str:
    """Hash a file or a directory tree with stable relative-path ordering."""
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for child in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(root)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@dataclass(frozen=True)
class BoundaryManifest:
    """Authoritative metadata for one completed session boundary."""

    schema_version: int
    session_idx: int
    global_update_step: int
    global_env_steps: int
    payload_sha256: str
    references: dict[str, str]
    provenance: dict[str, str]

    @classmethod
    def from_payload(
        cls,
        *,
        session_idx: int,
        global_update_step: int,
        global_env_steps: int,
        payload: bytes,
        references: Mapping[str, str],
        provenance: Mapping[str, str],
    ) -> "BoundaryManifest":
        if session_idx < 0 or global_update_step < 0 or global_env_steps < 0:
            raise ValueError("boundary counters must be non-negative")
        return cls(
            schema_version=SCHEMA_VERSION,
            session_idx=int(session_idx),
            global_update_step=int(global_update_step),
            global_env_steps=int(global_env_steps),
            payload_sha256=sha256_bytes(payload),
            references={str(k): str(v) for k, v in references.items()},
            provenance={str(k): str(v) for k, v in provenance.items()},
        )

    def to_bytes(self) -> bytes:
        return _canonical_json(asdict(self))


class BoundaryStore:
    """Publish and restore one complete process boundary atomically."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)

    def _paths(self, session_idx: int) -> tuple[Path, Path]:
        if session_idx < 0:
            raise ValueError("session_idx must be non-negative")
        directory = self.root / f"session_{session_idx:06d}"
        return directory / "payload.pkl", directory / "manifest.json"

    def write(
        self,
        *,
        session_idx: int,
        global_update_step: int,
        global_env_steps: int,
        state: Mapping[str, Any],
        references: Mapping[str, str],
        provenance: Mapping[str, str],
    ) -> BoundaryManifest:
        """Write payload and manifest, publishing the manifest last.

        A manifest is considered committed only when both files exist and the
        recorded payload digest verifies. Existing committed boundaries are
        immutable; callers must use a new session index for a retry.
        """
        payload, manifest_path = self._paths(session_idx)
        if payload.exists() or manifest_path.exists():
            raise BoundaryIntegrityError(f"boundary already exists: session_{session_idx:06d}")

        payload_bytes = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
        manifest = BoundaryManifest.from_payload(
            session_idx=session_idx,
            global_update_step=global_update_step,
            global_env_steps=global_env_steps,
            payload=payload_bytes,
            references=references,
            provenance=provenance,
        )
        _atomic_write(payload, payload_bytes)
        _atomic_write(manifest_path, manifest.to_bytes())
        return manifest

    def read(self, session_idx: int) -> tuple[BoundaryManifest, dict[str, Any]]:
        payload_path, manifest_path = self._paths(session_idx)
        if not payload_path.is_file() or not manifest_path.is_file():
            raise BoundaryIntegrityError(f"incomplete boundary: session_{session_idx:06d}")
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = BoundaryManifest(**raw_manifest)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BoundaryIntegrityError(f"invalid manifest: {manifest_path}") from exc
        if manifest.schema_version != SCHEMA_VERSION:
            raise BoundaryIntegrityError(
                f"unsupported boundary schema: {manifest.schema_version}"
            )
        payload_bytes = payload_path.read_bytes()
        if sha256_bytes(payload_bytes) != manifest.payload_sha256:
            raise BoundaryIntegrityError(f"payload digest mismatch: {payload_path}")
        try:
            state = pickle.loads(payload_bytes)
        except Exception as exc:  # pragma: no cover - depends on corrupt pickle
            raise BoundaryIntegrityError(f"unreadable payload: {payload_path}") from exc
        if not isinstance(state, dict):
            raise BoundaryIntegrityError("boundary payload must deserialize to a dict")
        return manifest, state

    def latest(self) -> tuple[BoundaryManifest, dict[str, Any]] | None:
        if not self.root.exists():
            return None
        candidates = []
        for directory in self.root.glob("session_*"):
            try:
                candidates.append(int(directory.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        if not candidates:
            return None
        return self.read(max(candidates))


def verify_reference_digests(
    manifest: BoundaryManifest, actual: Mapping[str, str]
) -> None:
    """Require exact reference-key and digest equality before resuming."""
    expected = dict(manifest.references)
    observed = {str(k): str(v) for k, v in actual.items()}
    if expected != observed:
        raise BoundaryIntegrityError(
            f"boundary references differ: expected={sorted(expected)} observed={sorted(observed)}"
        )
