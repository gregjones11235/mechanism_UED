"""Provenance + frozen-config recovery for the vendored read-only RMT16 subset.

Pure stdlib module (NO jax import): it records where every vendored file came
from (source paths + SHA bindings + the single recorded import fix) and it
recovers the frozen CC2 network hyperparameters from the SHA-bound driver
source by AST-literal parsing — never executed, never guessed, never defaulted
(mirrors the CC4 tier3 ``tier3_cc2_policy_adapter`` contract).

Driver-source SHA binding (forensically resolved 2026-08-03):
  * ``FROZEN_DRIVER_SOURCE_SHA256`` is the SHA256 of the staging/deploy copy
    ``orchestration/control/_cc2_stage/train_rmt16_p2replay.py`` (raw == LF,
    verified on this host).  It is the value bound by the frozen tier3
    checkpoint contract and by the RMT16 student profiles' ``source_commit``.
  * The archived snapshot under ``raw_sources/.../src/train_rmt16_p2replay.py``
    in this repo is a DIFFERENT snapshot (raw ``3610cc11…`` / LF ``02f5ef2c…``).
    Its ``class Cfg`` literals were AST-verified IDENTICAL to the frozen
    driver's ``Cfg`` field-for-field (2026-08-03), so either copy recovers the
    same frozen hyperparameters; the mount driver still SHA-gates whichever
    copy it is handed.
"""

from __future__ import annotations

import ast
import hashlib
import os

# --- vendored file provenance ------------------------------------------------

# Repo-relative root of the audited CC2 rmt16_replay_phase4a source snapshot
# (the same snapshot root CC4 tier3 bound into its evaluation certificates).
ARCHIVE_ROOT_RELATIVE = (
    "experiments/henry_dicode_student_upgrade/13_rmt16_phase4a/raw_sources/"
    "home/oseasy/experiments/rmt16_replay_phase4a/src"
)

# vendored module name -> provenance record.
# ``source_raw_sha256`` / ``source_lf_sha256`` bind the ARCHIVE copy;
# ``vendored_sha256`` binds the file as committed under architectures/.
VENDORED_FILES = {
    "rmt16_network.py": {
        "source_file": "network_rmt16.py",
        "source_raw_sha256": "73340d52ae7a661b4994d642d36af0374be63aeee9ed7deed320ded72fece439",
        "source_lf_sha256": "b5c37d7aa2e9cac1b4b395111262b4d8a11e20fd75a2930670336a68d86b8632",
        "vendored_sha256": "73340d52ae7a661b4994d642d36af0374be63aeee9ed7deed320ded72fece439",
        "import_fix": None,  # byte-identical vendor
    },
    "rmt16_memory.py": {
        "source_file": "rmt16_memory.py",
        "source_raw_sha256": "7cb575862592bbb4602ab0d8499b356ec2760eb7861d90742e38d80edf492675",
        "source_lf_sha256": "17e1a614c404e4edf176de7e8f9bd3f241059257fb24962d0df148960c7f6500",
        "vendored_sha256": "7cb575862592bbb4602ab0d8499b356ec2760eb7861d90742e38d80edf492675",
        "import_fix": None,  # byte-identical vendor
    },
    "rmt16_anchor.py": {
        "source_file": "rmt_memory_anchor.py",
        "source_raw_sha256": "ef06808f46e43d72ec4fae51ea18c3d21a34a9ee61e849a066c0a8140ffe9a05",
        "source_lf_sha256": "92c56b6375878e789fae2fddee0bf5a4fef25ad4eec83e67ab8c91ec65ea68e8",
        "vendored_sha256": "0d2d69b78d3bb07ec035fea9eceec6e78bd689206886517ec0e1d93977864822",
        # THE COMPLETE import-fix diff (two line edits, nothing else changed):
        #  1) drop the module-level line
        #       import memory_anchor as MA  # frozen P2 GTrXL anchor helpers (reused)
        #     because ``MA`` is referenced NOWHERE in rmt_memory_anchor.py
        #     (verified: the eval-forward subset this round mounts —
        #     make_apply_eval_rmt / make_update_fn / rmt_advance_tokens /
        #     rmt_step_forward — is self-contained) and ``memory_anchor.py``
        #     is a P2-replay reconstruction helper that is not vendored;
        #  2) rewrite the sibling-directory import
        #       import rmt16_memory as rmtm
        #     as the package-relative
        #       from . import rmt16_memory as rmtm
        #     (CC2 ran these files from one flat directory; inside the
        #     student_adapters.architectures package the sibling module is
        #     only reachable relatively).
        # Every other byte is unchanged; ``rmt_step_forward`` remains the ONE
        # per-step transition (never reimplemented by CC4).
        "import_fix": ("removed unused top-level line 'import memory_anchor as MA'; "
                       "rewrote 'import rmt16_memory as rmtm' as "
                       "'from . import rmt16_memory as rmtm'"),
    },
}

# The archived driver snapshot (repo-relative), with its own SHA bindings.
ARCHIVE_DRIVER_FILE = "train_rmt16_p2replay.py"
ARCHIVE_DRIVER_RAW_SHA256 = "3610cc11175e03122ccdd9da76219d809930d988a22eff490bd2b05ba7f4ec68"
ARCHIVE_DRIVER_LF_SHA256 = "02f5ef2ccb9dd2f640068291c46baf05f6074163623156974de5ff595fc78ffa"

# SHA-bound CC2 driver source (frozen tier3 checkpoint contract value): the
# staging/deploy copy orchestration/control/_cc2_stage/train_rmt16_p2replay.py.
# Raw == LF for that file (verified on this host 2026-08-03).
FROZEN_DRIVER_SOURCE_SHA256 = "453bd1ecc8d9671c741c4462214bd7699c74611a52ec157ff30cd68653b4bafc"
FROZEN_DRIVER_SOURCE_SEMANTICS = (
    "staging/deploy driver copy bound by the frozen tier3 checkpoint contract; "
    "the repo raw_sources archive driver is a distinct snapshot whose class Cfg "
    "is AST-verified field-for-field identical"
)

# --- frozen network hyperparameters ------------------------------------------

# Config fields required to reconstruct the network EXACTLY as CC2 built it
# (same contract as CC4 tier3 tier3_cc2_policy_adapter.REQUIRED_CFG_FIELDS).
REQUIRED_CFG_FIELDS = (
    "activation", "embed_size", "hidden_layers", "num_heads", "qkv_features",
    "num_layers", "gating", "gating_bias", "rmt_num_tokens", "window_mem",
    "num_steps",
)

# Frozen expectation, AST-extracted from the SHA-bound driver ``class Cfg``
# (and independently re-extracted from the archived driver copy, identical).
# Used ONLY as a fail-closed cross-check of any runtime AST parse — never as a
# silent default.
FROZEN_RMT16_CFG = {
    "activation": "relu",
    "embed_size": 256,
    "hidden_layers": 256,
    "num_heads": 8,
    "qkv_features": 256,
    "num_layers": 2,
    "gating": True,
    "gating_bias": 2.0,
    "rmt_num_tokens": 16,
    "window_mem": 128,
    "num_steps": 128,
}

CARRY_MODES = ("persistent", "reset128")


class DriverSourceError(RuntimeError):
    """Raised on any driver-source identity / config violation (fail closed)."""


def sha256_lf_file(path: str) -> str:
    """LF-normalized SHA256 of a file (EOL-independent binding)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def load_rmt16_cfg_from_driver_source(driver_path: str, expected_sha256: str):
    """Recover the frozen CC2 network hyperparameters from the driver SOURCE.

    Real CC2 checkpoints carry ``manifest["config"] == {}`` (Cfg is a
    class-attributes config class; ``vars(Cfg())`` is empty by design —
    verified by CC4 tier3 on all 26 audited real checkpoints).  The
    hyperparameters live in the driver source's ``class Cfg``.  This function:

      * requires the driver file to exist and its LF-normalized SHA256 to
        equal ``expected_sha256`` (fail closed — a moved/edited driver is
        never trusted);
      * ``ast.parse``-es the source (NEVER executes it) and collects every
        class-level literal assignment in ``class Cfg`` via
        ``ast.literal_eval`` (a non-literal value fails closed — no guessing,
        no defaults);
      * requires ALL ``REQUIRED_CFG_FIELDS`` to be present.

    Returns ``(cfg_dict, driver_sha256)``.  Pure stdlib.
    """
    if not driver_path or not os.path.isfile(driver_path):
        raise DriverSourceError(
            f"FAIL CLOSED: CC2 driver source missing ({driver_path!r}) — cannot rebuild "
            "the RMT16 network without the frozen hyperparameters (no guessing, no defaults)")
    driver_sha = sha256_lf_file(driver_path)
    if str(expected_sha256).lower() != driver_sha:
        raise DriverSourceError(
            f"FAIL CLOSED: CC2 driver source SHA256 {driver_sha[:16]}… != expected "
            f"{str(expected_sha256)[:16]}… (a moved or edited driver is never trusted)")
    with open(driver_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise DriverSourceError(
            f"FAIL CLOSED: CC2 driver source {driver_path!r} does not parse: {exc!r}") from exc
    cfg_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Cfg":
            cfg_cls = node
            break
    if cfg_cls is None:
        raise DriverSourceError(
            f"FAIL CLOSED: CC2 driver source {driver_path!r} defines no top-level 'class Cfg'")
    cfg: dict = {}
    for stmt in cfg_cls.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        key = stmt.targets[0].id
        try:
            cfg[key] = ast.literal_eval(stmt.value)
        except (ValueError, SyntaxError) as exc:
            raise DriverSourceError(
                f"FAIL CLOSED: Cfg.{key} in {driver_path!r} is not a literal constant "
                "(no guessing, no defaults)") from exc
    missing = [f for f in REQUIRED_CFG_FIELDS if f not in cfg]
    if missing:
        raise DriverSourceError(
            f"FAIL CLOSED: driver source Cfg missing field(s) {missing}; cannot rebuild "
            "the CC2 RMT16 network exactly")
    return cfg, driver_sha


def verify_frozen_cfg(cfg: dict) -> None:
    """Fail closed unless ``cfg`` agrees with the frozen expectation on all
    REQUIRED_CFG_FIELDS.  This is a cross-check of the runtime AST parse
    against the SHA-bound frozen values — never a silent default."""
    if not isinstance(cfg, dict):
        raise DriverSourceError("FAIL CLOSED: cfg must be a dict")
    clashes = {k: (cfg.get(k), FROZEN_RMT16_CFG[k])
               for k in REQUIRED_CFG_FIELDS if cfg.get(k) != FROZEN_RMT16_CFG[k]}
    if clashes:
        raise DriverSourceError(
            f"FAIL CLOSED: driver Cfg disagrees with the frozen RMT16 expectation "
            f"(key: (got, frozen)) {clashes}")
