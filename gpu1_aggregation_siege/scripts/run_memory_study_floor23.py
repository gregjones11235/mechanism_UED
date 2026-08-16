#!/usr/bin/env python
"""Memory Study Floor2->Floor3 probe CLI (TASK B1).

SYNTHETIC (local, jax-free) end-to-end:

    python gpu1_aggregation_siege/scripts/run_memory_study_floor23.py \
        --mode synthetic --out-root OUT [--max-states N] [--seed S]

REAL (server only; see docs/memory_study/HO_FLOOR23_DESIGN.md RUNBOOK):

    python gpu1_aggregation_siege/scripts/run_memory_study_floor23.py \
        --mode real --candidate-id cc3_slowgru_persistent \
        --bank-manifest PATH --capture-bank PATH --out-root PATH

Fail-closed discipline: missing assets / hash drift / local REAL attempts all
exit non-zero with a structured BLOCKED JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_GAS_ROOT = _HERE.parents[1]                 # gpu1_aggregation_siege
_SRC = _GAS_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dicode.memory_study import (  # noqa: E402
    FailClosed,
    HOMode,
    generate_synthetic_capture_bank,
    make_synthetic_candidate,
    run_floor23_probe,
    synthetic_states,
)

SCHEMA_BLOCKED = "mechanism_UED.memory_study_blocked/v1"


def _blocked(reason: str, **extra) -> int:
    doc = {"schema": SCHEMA_BLOCKED, "status": "BLOCKED",
           "reason": reason}
    doc.update(extra)
    print(json.dumps(doc, sort_keys=True, indent=2))
    return 2


def _parse_modes(text: str):
    modes = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            modes.append(HOMode(tok))
        except ValueError:
            raise FailClosed("UNKNOWN_HO_MODE: %r" % tok)
    if not modes:
        raise FailClosed("NO_HO_MODES_PARSED")
    return modes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["synthetic", "real"], required=True)
    ap.add_argument("--candidate-id", default=None,
                    help="REAL mode: candidate identity (binding per RUNBOOK)")
    ap.add_argument("--ho-modes", default="base,ho_zero,ho_real")
    ap.add_argument("--bank-manifest", default=None,
                    help="REAL mode: FRONT_L2 state bank manifest path")
    ap.add_argument("--capture-bank", default=None,
                    help="REAL mode: capture bank manifest path")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-states", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-states", type=int, default=4,
                    help="SYNTHETIC mode only")
    ap.add_argument("--num-captures", type=int, default=3,
                    help="SYNTHETIC mode only")
    ap.add_argument("--segment-len", type=int, default=4,
                    help="SYNTHETIC mode only")
    ap.add_argument("--obs-dim", type=int, default=8,
                    help="SYNTHETIC mode only (REAL uses canonical 8335)")
    args = ap.parse_args(argv)

    try:
        ho_modes = _parse_modes(args.ho_modes)
        out_root = Path(args.out_root)

        if args.mode == "real":
            missing = [name for name, val in
                       (("--candidate-id", args.candidate_id),
                        ("--bank-manifest", args.bank_manifest),
                        ("--capture-bank", args.capture_bank)) if not val]
            if missing:
                return _blocked("REAL_MODE_ASSETS_MISSING",
                                missing=missing,
                                runbook="docs/memory_study/"
                                        "HO_FLOOR23_DESIGN.md")
            try:
                import jax  # noqa: F401
            except Exception:
                return _blocked(
                    "REAL_MODE_BLOCKED_LOCAL_NO_JAX",
                    note="REAL probe runs only on the server (CC4 venv, "
                         "GPU2/3) per RUNBOOK; local environment has no "
                         "jax/craftax")
            if not Path(args.bank_manifest).is_file():
                return _blocked("STATE_BANK_MANIFEST_MISSING",
                                path=args.bank_manifest)
            if not Path(args.capture_bank).is_file():
                return _blocked("CAPTURE_BANK_MANIFEST_MISSING",
                                path=args.capture_bank)
            return _blocked("REAL_MODE_RUNTIME_BINDING_NOT_INJECTED",
                            note="server-side candidate binding is injected "
                                 "per RUNBOOK; this local CLI does not guess "
                                 "checkpoint paths")

        # SYNTHETIC end-to-end (jax-free)
        states = synthetic_states(args.num_states, seed=args.seed,
                                  obs_dim=args.obs_dim)
        manifest, captures = generate_synthetic_capture_bank(
            args.num_captures, args.segment_len, args.obs_dim,
            seed=args.seed + 777)
        runtimes = [
            make_synthetic_candidate("SYN_CAND_A", success_bias=0.60),
            make_synthetic_candidate("SYN_CAND_B", success_bias=0.35),
        ]
        summary = run_floor23_probe(
            states=states,
            runtimes=runtimes,
            captures=captures,
            out_root=out_root,
            ho_modes=ho_modes,
            max_states=args.max_states,
            run_mode="synthetic",
            probe_seed=args.seed)
        print(json.dumps({"status": "SYNTHETIC_PROBE_COMPLETE",
                          "summary_path": str(out_root / "summary.json"),
                          "results_written": summary["results_written"],
                          "arms": len(summary["arms"])},
                         sort_keys=True, indent=2))
        return 0
    except FailClosed as exc:
        return _blocked("FAIL_CLOSED", detail=str(exc))


if __name__ == "__main__":
    sys.exit(main())