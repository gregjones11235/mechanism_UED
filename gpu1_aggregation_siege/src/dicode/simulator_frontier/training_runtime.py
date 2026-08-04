"""Binding contract for the ORIGINAL training runtime (CC4 follow-up, P0-12).

Before this contract existed, the E3 window accepted the original loss and
the optimizer update as two plain, opaque callables: nothing bound them to
the actual training runtime, nothing could tell an injected original
loss/update from a reimplementation, and a callable could be substituted
after injection without detection.

``OriginalTrainingRuntime`` closes that gap.  It is MINTED (never
constructed piecemeal, never supplied as a mapping) from the two injected
callables plus controller-visible descriptors, and it binds them
mechanically:

* the sha256 of each callable's SOURCE TEXT (EOL-normalized) is captured at
  mint time — a substituted or reimplemented callable has different source
  and fails verification;
* ``runtime_hash`` is recomputed in ``__post_init__`` from the descriptors
  and the two source hashes, so a self-reported or tampered runtime is
  structurally impossible;
* ``verify_original_training_runtime`` recomputes BOTH source hashes from
  the stored callables and the runtime hash itself, rejecting mappings,
  foreign types and any drift.

The loss and update definitions are NEVER redefined here: this module only
binds and verifies callables that come from the shared, controller-authorized
training runtime (no second loss, no second optimizer).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Mapping

from .errors import InvalidEvidenceError

TRAINING_RUNTIME_VERSION = "original-training-runtime/v1"

BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME = (
    "BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _callable_source_sha256(name: str, fn: Any) -> str:
    """sha256 of source file + source text (EOL-normalized), fail-closed.

    Binding the file path together with the text means an identical body
    redefined in another file does NOT pass as the original definition.
    """
    if isinstance(fn, Mapping):
        raise InvalidEvidenceError(
            f"training runtime {name}: a mapping is never an acceptable "
            "loss/optimizer-update surface (fail closed)")
    if not callable(fn):
        raise InvalidEvidenceError(
            f"training runtime {name}: expected a callable, got "
            f"{type(fn).__name__}")
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise InvalidEvidenceError(
            f"training runtime {name}: cannot bind the callable — its source "
            f"text is unavailable ({exc!r}); a runtime whose definitions cannot "
            "be pinned by source hash is never accepted (fail closed)") from exc
    try:
        source_file = str(inspect.getsourcefile(fn) or "<unknown>")
    except TypeError:
        source_file = "<unknown>"
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(
        f"{source_file}\n::\n{normalized}".encode("utf-8")).hexdigest()


def _descriptor_payload(runtime_id: str, loss_name: str, optimizer_name: str,
                        contract_ref: str, loss_source_sha256: str,
                        optimizer_source_sha256: str) -> dict[str, Any]:
    return {
        "runtime_version": TRAINING_RUNTIME_VERSION,
        "runtime_id": runtime_id,
        "loss_name": loss_name,
        "optimizer_name": optimizer_name,
        "contract_ref": contract_ref,
        "loss_source_sha256": loss_source_sha256,
        "optimizer_source_sha256": optimizer_source_sha256,
    }


@dataclass(frozen=True)
class OriginalTrainingRuntime:
    """One bound original training runtime (mint-only, fail closed).

    ``runtime_hash`` is NOT a constructor argument: it is computed in
    ``__post_init__`` from the descriptors and the two callables' source
    hashes, so no caller can supply (self-report) the hash.
    """

    runtime_id: str
    loss_name: str
    optimizer_name: str
    contract_ref: str
    loss_fn: Callable[..., Any]
    optimizer_update_fn: Callable[..., Any]
    loss_source_sha256: str
    optimizer_source_sha256: str
    runtime_hash: str = field(init=False)
    runtime_version: str = TRAINING_RUNTIME_VERSION

    def __post_init__(self) -> None:
        for label, value in (("runtime_id", self.runtime_id),
                             ("loss_name", self.loss_name),
                             ("optimizer_name", self.optimizer_name),
                             ("contract_ref", self.contract_ref)):
            if not str(value).strip():
                raise InvalidEvidenceError(
                    f"OriginalTrainingRuntime.{label} is empty — the original "
                    "training runtime is never bound anonymously")
        for label, fn in (("loss_fn", self.loss_fn),
                          ("optimizer_update_fn", self.optimizer_update_fn)):
            if isinstance(fn, Mapping) or not callable(fn):
                raise InvalidEvidenceError(
                    f"OriginalTrainingRuntime.{label} must be a callable "
                    "(mappings are never accepted)")
        for label, digest in (("loss_source_sha256", self.loss_source_sha256),
                              ("optimizer_source_sha256",
                               self.optimizer_source_sha256)):
            text = str(digest)
            if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
                raise InvalidEvidenceError(
                    f"OriginalTrainingRuntime.{label} is not a lowercase sha256 "
                    f"hex digest: {text[:24]!r}…")
        expected = _canonical_sha256(_descriptor_payload(
            str(self.runtime_id), str(self.loss_name), str(self.optimizer_name),
            str(self.contract_ref), str(self.loss_source_sha256),
            str(self.optimizer_source_sha256)))
        object.__setattr__(self, "runtime_hash", expected)


def mint_original_training_runtime(*, loss_fn: Any, optimizer_update_fn: Any,
                                   runtime_id: str, loss_name: str,
                                   optimizer_name: str,
                                   contract_ref: str) -> OriginalTrainingRuntime:
    """Mint the immutable binding for the injected ORIGINAL loss + update.

    The callables must carry retrievable source text (builtins, C
    extensions and source-less partials fail closed: a runtime that cannot
    be pinned by source hash is never bound).  Descriptors must be non-empty
    strings identifying the controller-shared runtime contract.
    """
    loss_source_sha256 = _callable_source_sha256("loss_fn", loss_fn)
    optimizer_source_sha256 = _callable_source_sha256(
        "optimizer_update_fn", optimizer_update_fn)
    return OriginalTrainingRuntime(
        runtime_id=str(runtime_id),
        loss_name=str(loss_name),
        optimizer_name=str(optimizer_name),
        contract_ref=str(contract_ref),
        loss_fn=loss_fn,
        optimizer_update_fn=optimizer_update_fn,
        loss_source_sha256=loss_source_sha256,
        optimizer_source_sha256=optimizer_source_sha256,
    )


def verify_original_training_runtime(runtime: Any) -> None:
    """Recompute source hashes + runtime hash; reject any drift or fakes.

    A runtime is only accepted when:

    * it is a minted ``OriginalTrainingRuntime`` (mappings and foreign types
      are refused);
    * the CURRENT source hashes of the stored ``loss_fn`` /
      ``optimizer_update_fn`` equal the stored source hashes (a substituted
      callable fails closed);
    * the stored ``runtime_hash`` equals the hash recomputed from the
      descriptors and source hashes (tamper/self-report fails closed).
    """
    if isinstance(runtime, Mapping):
        raise InvalidEvidenceError(
            "verify_original_training_runtime requires a minted "
            "OriginalTrainingRuntime, not a mapping")
    if not isinstance(runtime, OriginalTrainingRuntime):
        raise InvalidEvidenceError(
            f"verify_original_training_runtime requires a minted "
            f"OriginalTrainingRuntime, got {type(runtime).__name__}")
    current_loss = _callable_source_sha256("loss_fn", runtime.loss_fn)
    if current_loss != runtime.loss_source_sha256:
        raise InvalidEvidenceError(
            "loss_fn source hash drift: the bound loss callable was substituted "
            "after minting (fail closed)")
    current_update = _callable_source_sha256(
        "optimizer_update_fn", runtime.optimizer_update_fn)
    if current_update != runtime.optimizer_source_sha256:
        raise InvalidEvidenceError(
            "optimizer_update_fn source hash drift: the bound update callable "
            "was substituted after minting (fail closed)")
    expected = _canonical_sha256(_descriptor_payload(
        str(runtime.runtime_id), str(runtime.loss_name),
        str(runtime.optimizer_name), str(runtime.contract_ref),
        str(runtime.loss_source_sha256), str(runtime.optimizer_source_sha256)))
    if expected != runtime.runtime_hash:
        raise InvalidEvidenceError(
            "runtime_hash mismatch: the OriginalTrainingRuntime was tampered "
            "with or self-reported (fail closed)")


def _runtime_descriptor_fields(runtime: OriginalTrainingRuntime) -> dict[str, Any]:
    """JSON-safe descriptor fields (callables excluded) for step records."""
    return {
        f.name: getattr(runtime, f.name)
        for f in fields(runtime)
        if f.name not in ("loss_fn", "optimizer_update_fn")
    }


def runtime_binding_summary(runtime: Any) -> dict[str, Any]:
    """Audit summary for pipeline step records (no callables, no leakage)."""
    if not isinstance(runtime, OriginalTrainingRuntime):
        raise InvalidEvidenceError(
            f"runtime_binding_summary requires a minted OriginalTrainingRuntime, "
            f"got {type(runtime).__name__}")
    summary = _runtime_descriptor_fields(runtime)
    summary["bound"] = True
    return summary
