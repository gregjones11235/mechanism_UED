"""Cell lifecycle CLI: register / validate / prepare / authorize / launch / status.

Discipline baked into the command surface:
  * ``validate``, ``prepare`` and ``status`` ONLY inspect/advance state -- they
    NEVER launch training.
  * ``launch`` is authorization-gated by the registry: it requires state AUTHORIZED
    and a valid, non-revoked authorization, and a no-training authorization can only
    ever run the no-op runner (0 timesteps). Launching anything real requires an
    explicit training-scope authorization, which this phase does not issue.

Exit codes: 0 success; 2 a CellError / validation failure (fail-closed).

Examples:
  python -m d052.cells.cli --root <root> register --spec spec.json --actor CC3
  python -m d052.cells.cli --root <root> validate --cell-id c1 --actor CC3
  python -m d052.cells.cli --root <root> prepare  --cell-id c1 --actor CC3
  python -m d052.cells.cli --root <root> authorize --cell-id c1 \\
      --authorization auth.json --actor CC3
  python -m d052.cells.cli --root <root> launch --cell-id c1 --actor CC3
  python -m d052.cells.cli --root <root> status --cell-id c1
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from d052.cells.authorization import CellAuthorization
from d052.cells.registry import CellError, CellRegistry
from d052.cells.spec import CellSpec


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="d052.cells.cli",
                                description="D052 canonical cell lifecycle CLI")
    p.add_argument("--root", required=True, help="cell registry root directory")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("register", help="register a new cell (DRAFT)")
    r.add_argument("--spec", required=True, help="path to CellSpec JSON")
    r.add_argument("--actor", required=True)

    for name, helpt in (("validate", "validate a DRAFT cell (never launches)"),
                        ("prepare", "prepare a VALIDATED cell (never launches)"),
                        ("status", "read-only status (never launches)")):
        s = sub.add_parser(name, help=helpt)
        s.add_argument("--cell-id", required=True)
        if name != "status":
            s.add_argument("--actor", required=True)

    a = sub.add_parser("authorize", help="authorize a READY cell")
    a.add_argument("--cell-id", required=True)
    a.add_argument("--authorization", required=True,
                   help="path to CellAuthorization JSON")
    a.add_argument("--actor", required=True)

    l = sub.add_parser("launch", help="launch an AUTHORIZED cell (gated)")
    l.add_argument("--cell-id", required=True)
    l.add_argument("--actor", required=True)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reg = CellRegistry(args.root)
    try:
        if args.command == "register":
            spec = CellSpec.model_validate(_read_json(args.spec))
            rec = reg.register(spec, actor=args.actor)
            print(json.dumps({"registered": rec.spec.cell_id,
                              "state": rec.state.value,
                              "cell_identity_hash": rec.spec.identity_hash()},
                             indent=2))
        elif args.command == "validate":
            rec = reg.validate_cell(args.cell_id, actor=args.actor)
            print(json.dumps({"cell_id": rec.spec.cell_id,
                              "state": rec.state.value,
                              "block_reason": rec.block_reason}, indent=2))
        elif args.command == "prepare":
            rec = reg.prepare(args.cell_id, actor=args.actor)
            print(json.dumps({"cell_id": rec.spec.cell_id,
                              "state": rec.state.value,
                              "launched": rec.prepared_bundle["launched"]},
                             indent=2))
        elif args.command == "authorize":
            auth = CellAuthorization.model_validate(
                _read_json(args.authorization))
            rec = reg.authorize(args.cell_id, auth, actor=args.actor)
            print(json.dumps({"cell_id": rec.spec.cell_id,
                              "state": rec.state.value,
                              "scope": rec.authorization.scope}, indent=2))
        elif args.command == "launch":
            rec = reg.launch(args.cell_id, actor=args.actor)  # no-op runner phase
            print(json.dumps({"cell_id": rec.spec.cell_id,
                              "state": rec.state.value,
                              "timesteps_run":
                                  rec.launch_manifest["timesteps_run"]},
                             indent=2))
        elif args.command == "status":
            print(json.dumps(reg.status(args.cell_id), indent=2))
        return 0
    except CellError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
