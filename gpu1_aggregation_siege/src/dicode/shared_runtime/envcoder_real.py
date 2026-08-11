"""The REAL EnvCoder backend registry asset.

The real staged ladder (SYNTAX..TERMINAL_AUTORESET on the craftax
runtime) is implemented in ``dicode.teachers.e1_formal.envcoder_backends.
RealBackendAdapter``; this module wraps it as a registry asset with a
stable identity.
"""
from __future__ import annotations

import hashlib

from dicode.teachers.e1_formal.envcoder_backends import RealBackendAdapter


class RealEnvCoderBackend(RealBackendAdapter):
    """The real backend as a registered asset (stable identity)."""

    def __init__(self):
        self.object_identity_hash = hashlib.sha256(
            b"shared_runtime.real_envcoder_backend.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash
