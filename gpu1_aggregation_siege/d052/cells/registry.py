"""CellRegistry — register / validate / prepare / authorize / launch / status.

The registry owns the mutable lifecycle record of each cell (state + history +
authorization + launch manifest), separate from the immutable CellSpec identity.

Hard rules enforced here:
  * NO_LEGACY_ARTIFACT_OVERWRITE: register() creates the cell dir exclusively and
    REFUSES an existing cell_id; new runs never write into legacy/frozen dirs
    (output_dir is checked against DENY_LEGACY_OUTPUT_PREFIXES).
  * prepare / validate / status NEVER launch training.
  * NO_UNAUTHORIZED_TRAINING: launch() requires state==AUTHORIZED and a valid,
    non-revoked authorization whose cell_identity_hash matches the current spec;
    a no-training authorization is structurally incapable of running timesteps
    (the runner is forced to the no-op and any non-zero timesteps FAIL the cell).
  * Every transition is checked against the state machine; illegal moves fail-closed.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from d052.cells.authorization import (
    SCOPE_NO_TRAINING,
    CellAuthorization,
)
from d052.cells.spec import ENVIRONMENT_VERSION, CellSpec
from d052.cells.states import CellState, assert_transition

INDEX_FILE = "registry.json"
CELLS_DIR = "cells"
RECORD_FILE = "record.json"

#: output_dir prefixes that would write a new run into legacy/frozen territory.
DENY_LEGACY_OUTPUT_PREFIXES = (
    "/root/",
    "audit_outputs/",
    "reports/d052_readonly",
    "experiments/",
    "checkpoints_legacy/",
)


class CellError(Exception):
    EXISTS_NO_OVERWRITE = "EXISTS_NO_OVERWRITE"
    NOT_FOUND = "NOT_FOUND"
    INVALID_CELL_ID = "INVALID_CELL_ID"
    ILLEGAL_TRANSITION = "ILLEGAL_TRANSITION"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    AUTHORIZATION_MISMATCH = "AUTHORIZATION_MISMATCH"
    REVOKED_AUTHORIZATION = "REVOKED_AUTHORIZATION"
    UNAUTHORIZED_LAUNCH = "UNAUTHORIZED_LAUNCH"
    NOT_READY = "NOT_READY"
    NO_TRAINING_VIOLATION = "NO_TRAINING_VIOLATION"
    LAUNCH_FAILED = "LAUNCH_FAILED"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def no_op_runner(record: "CellRecord") -> dict:
    """The ONLY runner permitted under a no-training authorization.

    Records intent, runs ZERO timesteps. D052_LONG_TRAINING_RUNS stays 0.
    """
    return {
        "trained": False,
        "timesteps_run": 0,
        "reason": "no-training phase (D052_LONG_TRAINING_RUNS=0); intent recorded",
        "cell_id": record.spec.cell_id,
        "intended_total_timesteps": record.spec.intended_total_timesteps,
    }


class HistoryEntry(BaseModel):
    seq: int
    from_state: CellState
    to_state: CellState
    actor: str
    reason: str = ""


class CellRecord(BaseModel):
    spec: CellSpec
    state: CellState = CellState.DRAFT
    seq: int = 0
    history: List[HistoryEntry] = Field(default_factory=list)
    authorization: Optional[CellAuthorization] = None
    prepared_bundle: Optional[dict] = None
    launch_manifest: Optional[dict] = None
    block_reason: str = ""


def validate_cell_spec(spec: CellSpec) -> List[str]:
    """Structural pre-flight checks. Returns a list of problems (empty == OK).

    These are deterministic and launch nothing.
    """
    problems: List[str] = []
    if not spec.candidate_ids:
        problems.append("no candidate_ids bound to the cell")
    if spec.environment_version != ENVIRONMENT_VERSION:
        problems.append(
            f"environment_version {spec.environment_version!r} != "
            f"{ENVIRONMENT_VERSION!r}")
    od = spec.output_dir.replace("\\", "/")
    if ".." in od.split("/"):
        problems.append(f"output_dir {spec.output_dir!r} contains path traversal")
    for pfx in DENY_LEGACY_OUTPUT_PREFIXES:
        if od.startswith(pfx) or f"/{pfx}" in od:
            problems.append(
                f"output_dir {spec.output_dir!r} would write into a "
                f"legacy/frozen area ({pfx}); NO_LEGACY_ARTIFACT_OVERWRITE")
    return problems


class CellRegistry:
    """File-backed cell lifecycle registry under ``root``.

    Layout: ``<root>/registry.json`` index + ``<root>/cells/<cell_id>/record.json``.
    """

    def __init__(self, root: str) -> None:
        self.root = root

    # --- paths + persistence ------------------------------------------------
    def _index_path(self) -> str:
        return os.path.join(self.root, INDEX_FILE)

    def _cell_dir(self, cell_id: str) -> str:
        return os.path.join(self.root, CELLS_DIR, cell_id)

    def _record_path(self, cell_id: str) -> str:
        return os.path.join(self._cell_dir(cell_id), RECORD_FILE)

    @staticmethod
    def _check_cell_id(cell_id: str) -> None:
        if (not cell_id) or ("/" in cell_id) or ("\\" in cell_id) \
                or (".." in cell_id):
            raise CellError(
                CellError.INVALID_CELL_ID,
                f"cell_id {cell_id!r} must be a plain directory-safe name")

    def _load_index(self) -> dict:
        p = self._index_path()
        if not os.path.exists(p):
            return {"cells": {}}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:
        os.makedirs(self.root, exist_ok=True)
        tmp = self._index_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._index_path())

    def _load(self, cell_id: str) -> CellRecord:
        self._check_cell_id(cell_id)
        p = self._record_path(cell_id)
        if not os.path.exists(p):
            raise CellError(CellError.NOT_FOUND, f"no cell {cell_id!r} at {p}")
        with open(p, encoding="utf-8") as f:
            return CellRecord.model_validate_json(f.read())

    def _save(self, rec: CellRecord) -> None:
        os.makedirs(self._cell_dir(rec.spec.cell_id), exist_ok=True)
        tmp = self._record_path(rec.spec.cell_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(rec.model_dump_json(indent=2))
        os.replace(tmp, self._record_path(rec.spec.cell_id))
        index = self._load_index()
        index["cells"][rec.spec.cell_id] = {
            "state": rec.state.value,
            "cell_identity_hash": rec.spec.identity_hash(),
            "seq": rec.seq,
        }
        self._save_index(index)

    def _set_state(self, rec: CellRecord, dst: CellState, actor: str,
                   reason: str = "") -> None:
        try:
            assert_transition(rec.state, dst)
        except ValueError as e:
            raise CellError(CellError.ILLEGAL_TRANSITION, str(e)) from e
        rec.seq += 1
        rec.history.append(HistoryEntry(
            seq=rec.seq, from_state=rec.state, to_state=dst, actor=actor,
            reason=reason))
        rec.state = dst
        self._save(rec)

    # --- lifecycle operations ----------------------------------------------
    def list_cells(self) -> List[str]:
        """All registered cell_ids (sorted). Read-only."""
        return sorted(self._load_index().get("cells", {}))

    def register(self, spec: CellSpec, *, actor: str) -> CellRecord:
        """Register a new cell in DRAFT. REFUSES an existing cell_id."""
        self._check_cell_id(spec.cell_id)
        d = self._cell_dir(spec.cell_id)
        if os.path.exists(d):
            raise CellError(
                CellError.EXISTS_NO_OVERWRITE,
                f"cell {spec.cell_id!r} already exists at {d} "
                f"(NO_LEGACY_ARTIFACT_OVERWRITE; choose a new cell_id)")
        os.makedirs(d)  # exclusive: existed-check above; concurrent-safe enough
        rec = CellRecord(spec=spec, state=CellState.DRAFT)
        self._save(rec)
        return rec

    def validate_cell(self, cell_id: str, *, actor: str) -> CellRecord:
        """DRAFT -> VALIDATED (or BLOCKED). Validates only; NEVER launches."""
        rec = self._load(cell_id)
        if rec.state is not CellState.DRAFT:
            raise CellError(
                CellError.NOT_READY,
                f"validate requires DRAFT, cell is {rec.state.value}")
        problems = validate_cell_spec(rec.spec)
        if problems:
            rec.block_reason = "; ".join(problems)
            self._save(rec)
            self._set_state(rec, CellState.BLOCKED, actor,
                            f"validation failed: {rec.block_reason}")
            return rec
        rec.block_reason = ""
        self._save(rec)
        self._set_state(rec, CellState.VALIDATED, actor, "spec validated")
        return rec

    def prepare(self, cell_id: str, *, actor: str) -> CellRecord:
        """VALIDATED -> READY: assemble the launch bundle. NEVER launches."""
        rec = self._load(cell_id)
        if rec.state is not CellState.VALIDATED:
            raise CellError(
                CellError.NOT_READY,
                f"prepare requires VALIDATED, cell is {rec.state.value}")
        rec.prepared_bundle = {
            "cell_id": rec.spec.cell_id,
            "cell_identity_hash": rec.spec.identity_hash(),
            "pool_hash": rec.spec.pool_hash,
            "selection_hash": rec.spec.selection_hash,
            "candidate_ids": sorted(rec.spec.candidate_ids),
            "output_dir": rec.spec.output_dir,
            "intended_total_timesteps": rec.spec.intended_total_timesteps,
            "launched": False,   # explicit: prepare does not launch
        }
        self._save(rec)
        self._set_state(rec, CellState.READY, actor,
                        "launch bundle prepared (not launched)")
        return rec

    def authorize(self, cell_id: str, authorization: CellAuthorization, *,
                  actor: str) -> CellRecord:
        """READY -> AUTHORIZED: bind a valid authorization to the cell."""
        rec = self._load(cell_id)
        if rec.state is not CellState.READY:
            raise CellError(
                CellError.NOT_READY,
                f"authorize requires READY, cell is {rec.state.value}")
        ih = rec.spec.identity_hash()
        if authorization.cell_id != cell_id:
            raise CellError(
                CellError.AUTHORIZATION_MISMATCH,
                f"authorization cell_id {authorization.cell_id!r} != {cell_id!r}")
        if authorization.revoked:
            raise CellError(
                CellError.REVOKED_AUTHORIZATION,
                "authorization is revoked")
        if authorization.cell_identity_hash != ih:
            raise CellError(
                CellError.AUTHORIZATION_MISMATCH,
                f"authorization bound to identity {authorization.cell_identity_hash} "
                f"but current spec identity is {ih}; re-authorize the current spec")
        if authorization.granted_total_timesteps != rec.spec.intended_total_timesteps:
            raise CellError(
                CellError.AUTHORIZATION_MISMATCH,
                f"granted_total_timesteps {authorization.granted_total_timesteps} "
                f"!= intended_total_timesteps {rec.spec.intended_total_timesteps}")
        rec.authorization = authorization
        self._save(rec)
        self._set_state(rec, CellState.AUTHORIZED, actor,
                        f"authorized by {authorization.authorized_by} "
                        f"(scope={authorization.scope})")
        return rec

    def launch(self, cell_id: str, *, actor: str,
               runner: Optional[Callable[[CellRecord], dict]] = None) -> CellRecord:
        """AUTHORIZED -> RUNNING -> COMPLETE (or FAILED). Authorization-gated.

        Under a no-training authorization the runner is FORCED to the no-op and
        any non-zero timesteps FAIL the cell -- so a no-training auth can never
        train, regardless of what runner is supplied.
        """
        rec = self._load(cell_id)
        if rec.state is not CellState.AUTHORIZED:
            raise CellError(
                CellError.UNAUTHORIZED_LAUNCH,
                f"launch requires AUTHORIZED, cell is {rec.state.value} "
                f"(NO_UNAUTHORIZED_TRAINING)")
        auth = rec.authorization
        if auth is None:
            raise CellError(
                CellError.UNAUTHORIZED_LAUNCH, "no authorization on record")
        if auth.revoked:
            raise CellError(
                CellError.REVOKED_AUTHORIZATION, "authorization is revoked")

        self._set_state(rec, CellState.RUNNING, actor, "launch started")
        effective_runner = no_op_runner if auth.scope == SCOPE_NO_TRAINING \
            else (runner or no_op_runner)
        try:
            artifact = effective_runner(rec)
        except Exception as e:  # runner failure -> FAILED (preserved), re-raise
            self._set_state(rec, CellState.FAILED, actor, f"runner error: {e}")
            raise CellError(CellError.LAUNCH_FAILED, str(e)) from e

        if auth.scope == SCOPE_NO_TRAINING \
                and int(artifact.get("timesteps_run", 0)) != 0:
            self._set_state(rec, CellState.FAILED, actor,
                            "no-training authorization produced timesteps")
            raise CellError(
                CellError.NO_TRAINING_VIOLATION,
                "a no-training authorization must run 0 timesteps")

        rec.launch_manifest = {
            "authorization_hash": auth.authorization_hash,
            "scope": auth.scope,
            "executed_by": actor,
            "runner_artifact": artifact,
            "timesteps_run": int(artifact.get("timesteps_run", 0)),
        }
        self._save(rec)
        self._set_state(rec, CellState.COMPLETE, actor, "launch complete")
        return rec

    def status(self, cell_id: str) -> dict:
        """Read-only status snapshot. NEVER launches."""
        rec = self._load(cell_id)
        return {
            "cell_id": rec.spec.cell_id,
            "state": rec.state.value,
            "cell_identity_hash": rec.spec.identity_hash(),
            "seq": rec.seq,
            "authorized": rec.authorization is not None
            and not rec.authorization.revoked,
            "scope": rec.authorization.scope if rec.authorization else None,
            "prepared": rec.prepared_bundle is not None,
            "launched": rec.launch_manifest is not None,
            "timesteps_run": (rec.launch_manifest or {}).get("timesteps_run", 0),
            "history": [(h.from_state.value, h.to_state.value, h.actor)
                        for h in rec.history],
        }
