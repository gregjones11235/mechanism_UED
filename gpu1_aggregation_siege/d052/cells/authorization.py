"""CellAuthorization — explicit, per-cell, content-bound launch authorization.

NO_UNAUTHORIZED_TRAINING: a cell may only be launched if it carries a valid,
non-revoked authorization whose ``cell_identity_hash`` equals the cell spec's
current identity hash. If the spec changes after authorization, the hash no longer
matches and the authorization is void (must re-authorize). The ``authorization_hash``
is verified against recomputation so a token cannot be forged or reused across cells.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import Field, field_validator, model_validator

from d052.schemas.common import CanonicalModel, validate_sha256_hex

#: Scope labels. This phase only ever issues the no-training scope.
SCOPE_NO_TRAINING = "single_cell_no_training"
SCOPE_TRAINING = "single_cell_training"
LEGAL_SCOPES = frozenset({SCOPE_NO_TRAINING, SCOPE_TRAINING})


def compute_authorization_hash(cell_identity_hash: str, authorized_by: str,
                               scope: str, granted_total_timesteps: int) -> str:
    payload = {
        "cell_identity_hash": cell_identity_hash,
        "authorized_by": authorized_by,
        "scope": scope,
        "granted_total_timesteps": granted_total_timesteps,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class CellAuthorization(CanonicalModel):
    cell_id: str = Field(min_length=1)
    cell_identity_hash: str
    authorized_by: str = Field(min_length=1)
    scope: str
    granted_total_timesteps: int = Field(ge=0)
    authorization_hash: str
    revoked: bool = False

    @field_validator("cell_identity_hash", "authorization_hash")
    @classmethod
    def _hashes(cls, v: str) -> str:
        return validate_sha256_hex(v, "hash")

    @model_validator(mode="after")
    def _check(self) -> "CellAuthorization":
        if self.scope not in LEGAL_SCOPES:
            raise ValueError(
                f"UNKNOWN_SCOPE: {self.scope!r} not in {sorted(LEGAL_SCOPES)}")
        expected = compute_authorization_hash(
            self.cell_identity_hash, self.authorized_by, self.scope,
            self.granted_total_timesteps)
        if self.authorization_hash != expected:
            raise ValueError(
                f"AUTHORIZATION_HASH_MISMATCH: expected {expected}, "
                f"got {self.authorization_hash}")
        return self


def make_authorization(cell_id: str, cell_identity_hash: str, authorized_by: str,
                       scope: str,
                       granted_total_timesteps: int) -> CellAuthorization:
    """Convenience builder that computes the authorization_hash deterministically."""
    return CellAuthorization(
        cell_id=cell_id,
        cell_identity_hash=cell_identity_hash,
        authorized_by=authorized_by,
        scope=scope,
        granted_total_timesteps=granted_total_timesteps,
        authorization_hash=compute_authorization_hash(
            cell_identity_hash, authorized_by, scope, granted_total_timesteps),
        revoked=False,
    )
