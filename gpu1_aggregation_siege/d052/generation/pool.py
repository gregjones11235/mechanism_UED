"""Shared frozen candidate pool: build, persist (no-overwrite), load, verify.

Discipline enforced here:
  * candidate_pool_mode = shared_frozen -> ONE pool, consumed identically by every
    selector, identified by its content ``pool_hash`` (selectors hard-fail on a
    mismatch, as production_dispatcher already does).
  * NO_LEGACY_ARTIFACT_OVERWRITE -> the store creates pool files with O_EXCL and
    REFUSES to write over any existing file/dir; it never writes into an old
    experiment directory.
  * On load, the stored pool is re-validated and its recomputed pool_hash must
    equal the recorded one (tamper-evidence).
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, List, Mapping

from d052.generation.validator import canonicalize_candidate
from d052.schemas.candidate import Candidate, CandidatePool


class PoolError(Exception):
    """Fail-closed pool error with a stable ``code``."""

    EXISTS_NO_OVERWRITE = "EXISTS_NO_OVERWRITE"
    POOL_HASH_MISMATCH = "POOL_HASH_MISMATCH"
    NOT_FOUND = "NOT_FOUND"
    EMPTY_POOL = "EMPTY_POOL"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def build_pool(pool_id: str, raw_candidates: Iterable[Mapping[str, Any]]) -> CandidatePool:
    """Validate raw candidates and assemble a frozen CandidatePool.

    Each raw candidate is canonicalized (content-hashed). The pool_hash is the
    sha256 over the ordered candidate chashes. Empty input -> error.
    """
    candidates: List[Candidate] = [canonicalize_candidate(r) for r in raw_candidates]
    if not candidates:
        raise PoolError(PoolError.EMPTY_POOL, "cannot build a pool from 0 candidates")
    return CandidatePool(
        pool_id=pool_id,
        pool_hash=CandidatePool.hash_candidates(candidates),
        candidate_count=len(candidates),
        candidates=candidates,
        frozen=True,
    )


def _atomic_excl_write_text(path: str, text: str) -> None:
    """Create ``path`` exclusively (fail if it exists) and write text."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)


class SharedFrozenPoolStore:
    """Persist/load shared frozen pools under a root dir, one subdir per pool_id.

    Layout: ``<root>/<pool_id>/pool.json`` + ``<root>/<pool_id>/pool_manifest.json``.
    Writing is no-overwrite (O_EXCL); loading re-validates and re-hashes.
    """

    POOL_FILE = "pool.json"
    MANIFEST_FILE = "pool_manifest.json"

    def __init__(self, root: str) -> None:
        self.root = root

    def _pool_dir(self, pool_id: str) -> str:
        return os.path.join(self.root, pool_id)

    def exists(self, pool_id: str) -> bool:
        return os.path.exists(self._pool_dir(pool_id))

    def write(self, pool: CandidatePool) -> str:
        """Persist a frozen pool. REFUSES to overwrite anything existing."""
        d = self._pool_dir(pool.pool_id)
        if os.path.exists(d):
            raise PoolError(
                PoolError.EXISTS_NO_OVERWRITE,
                f"pool dir already exists: {d} (NO_LEGACY_ARTIFACT_OVERWRITE)")
        os.makedirs(d)  # created above the exists-check; safe
        pool_path = os.path.join(d, self.POOL_FILE)
        manifest_path = os.path.join(d, self.MANIFEST_FILE)
        try:
            _atomic_excl_write_text(pool_path, pool.model_dump_json(indent=2))
            manifest = {
                "pool_id": pool.pool_id,
                "pool_hash": pool.pool_hash,
                "candidate_count": pool.candidate_count,
                "frozen": pool.frozen,
                "protocol_version": pool.protocol_version,
            }
            _atomic_excl_write_text(
                manifest_path,
                json.dumps(manifest, indent=2, ensure_ascii=False))
        except FileExistsError as e:
            raise PoolError(
                PoolError.EXISTS_NO_OVERWRITE,
                f"refusing to overwrite existing pool file under {d}") from e
        return pool_path

    def load(self, pool_id: str) -> CandidatePool:
        """Load + re-validate a stored pool; verify recomputed pool_hash."""
        d = self._pool_dir(pool_id)
        pool_path = os.path.join(d, self.POOL_FILE)
        if not os.path.exists(pool_path):
            raise PoolError(PoolError.NOT_FOUND, f"no stored pool at {pool_path}")
        with open(pool_path, encoding="utf-8") as f:
            pool = CandidatePool.model_validate_json(f.read())
        # CandidatePool validation already recomputes + checks pool_hash; assert
        # the manifest agrees too.
        manifest_path = os.path.join(d, self.MANIFEST_FILE)
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("pool_hash") != pool.pool_hash:
                raise PoolError(
                    PoolError.POOL_HASH_MISMATCH,
                    f"manifest pool_hash != recomputed pool_hash for {pool_id}")
        return pool

    @staticmethod
    def assert_pool_matches(pool: CandidatePool, expected_hash: str) -> None:
        """Selectors call this to hard-fail if handed a different pool."""
        if pool.pool_hash != expected_hash:
            raise PoolError(
                PoolError.POOL_HASH_MISMATCH,
                f"selector received pool_hash {pool.pool_hash} but expected "
                f"{expected_hash}; all selectors must share ONE frozen pool")
