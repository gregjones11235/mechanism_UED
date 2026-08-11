"""Location + integrity of the REAL Phase 2.5 Canonical Migration Bundle.

The bundle lives at orchestration/experiments/d052_modeler_shadow_v1/artifacts/
d052_phase25_canonical_migration/ (same repo-relative path as the server source).
Everything here is READ-ONLY: loaders never write, and integrity checks verify
bytes against the bundle's OWN SHA256SUMS manifest (13 payload files).

Tamper-evidence formula (verified 192/192 on 2026-07-26):
    judgment_hash_sha256 == sha256(canon_json(source_row["judgment"]))
where canon_json = json.dumps(obj, sort_keys=True, ensure_ascii=False,
separators=(",", ":")) and source_row is the original per-role record in the
source_file under outputs/ (keys: anon_id, arm, judgment, model_rq, model_rt,
provider, role, task_id).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

#: repo root = worktree containing gpu1_aggregation_siege/ and orchestration/
REPO_ROOT = Path(__file__).resolve().parents[3]

BUNDLE_REL = ("orchestration/experiments/d052_modeler_shadow_v1/artifacts/"
              "d052_phase25_canonical_migration")
OUTPUTS_REL = "orchestration/experiments/d052_modeler_shadow_v1/outputs"
REPLAY_INPUTS_REL = "orchestration/experiments/d052_modeler_shadow_v1/replay_inputs"

#: the 14 frozen bundle files (13 payloads + manifest)
BUNDLE_FILES = (
    "expected_behavior.json", "field_mapping.json", "judgments_B.jsonl",
    "judgments_C.jsonl", "prompt_registry.json", "protocol.json",
    "ranking_B.json", "ranking_C.json", "regression_test_spec.md",
    "role_ablation.json", "salted_hash_audit.json", "selector_config.json",
    "student_profile.json", "SHA256SUMS",
)


def bundle_dir() -> Path:
    return REPO_ROOT / BUNDLE_REL


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(obj) -> str:
    """sha256 over canonical JSON (objects) or utf-8 (str/bytes)."""
    if isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
    elif isinstance(obj, str):
        data = obj.encode("utf-8")
    else:
        data = _canon(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_bundle_integrity(bdir: Optional[os.PathLike] = None) -> Dict[str, object]:
    """Verify every SHA256SUMS entry against the actual bundle bytes (read-only).

    Returns {ok: bool, verified: n, failed: [names], missing: [names]}.
    """
    d = Path(bdir) if bdir else bundle_dir()
    manifest = d / "SHA256SUMS"
    if not manifest.exists():
        return {"ok": False, "verified": 0, "failed": [], "missing": ["SHA256SUMS"]}
    verified, failed, missing = 0, [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, name = line.split(None, 1)
        name = name.lstrip("*").strip()
        p = d / name
        if not p.exists():
            missing.append(name)
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual == expected:
            verified += 1
        else:
            failed.append(name)
    return {"ok": not failed and not missing and verified == 13,
            "verified": verified, "failed": failed, "missing": missing}


def load_judgments(arm: str, bdir: Optional[os.PathLike] = None) -> List[dict]:
    """The flattened bundle judgment records for one arm (96 expected)."""
    if arm not in ("B", "C"):
        raise ValueError(f"arm must be B or C, got {arm!r}")
    d = Path(bdir) if bdir else bundle_dir()
    p = d / f"judgments_{arm}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_bundle_json(name: str, bdir: Optional[os.PathLike] = None) -> dict:
    d = Path(bdir) if bdir else bundle_dir()
    return json.loads((d / name).read_text(encoding="utf-8"))


def load_source_rows(source_file: str,
                     outputs_root: Optional[os.PathLike] = None) -> List[dict]:
    """Original per-role judgment rows from outputs/ (source_file is 'outputs/x.jsonl')."""
    root = Path(outputs_root) if outputs_root else REPO_ROOT / OUTPUTS_REL
    p = root / os.path.basename(source_file)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def original_judgment_for(record: dict,
                          outputs_root: Optional[os.PathLike] = None) -> Optional[dict]:
    """Locate the ORIGINAL judgment object a bundle record was flattened from."""
    for row in load_source_rows(record["source_file"], outputs_root):
        if row.get("task_id") == record["task_id"] and row.get("role") == record["role"]:
            return row.get("judgment")
    return None


def judgment_hash_formula(obj: dict) -> str:
    """The verified tamper-evidence hash: sha256 over canonical JSON."""
    return sha256_hex(obj)


def verify_judgment_hashes(arms=("B", "C"),
                           outputs_root: Optional[os.PathLike] = None) -> Dict[str, object]:
    """Re-verify every record's judgment_hash_sha256 against its original object."""
    checked, ok, mismatches, missing_original = 0, 0, [], []
    src_cache: Dict[str, List[dict]] = {}
    for arm in arms:
        for rec in load_judgments(arm):
            checked += 1
            sf = rec["source_file"]
            rows = src_cache.setdefault(sf, load_source_rows(sf, outputs_root))
            orig = next((r["judgment"] for r in rows
                         if r.get("task_id") == rec["task_id"]
                         and r.get("role") == rec["role"]), None)
            if orig is None:
                missing_original.append((arm, rec["task_id"], rec["role"]))
                continue
            if judgment_hash_formula(orig) == rec["judgment_hash_sha256"]:
                ok += 1
            else:
                mismatches.append((arm, rec["task_id"], rec["role"]))
            # flattened-fidelity cross-check (scores must not have drifted)
            if orig.get("scores") != rec["raw_scores"]:
                mismatches.append(("SCORES_DRIFT", arm, rec["task_id"]))
    return {"checked": checked, "ok": ok, "mismatches": mismatches,
            "missing_original": missing_original,
            "formula": "sha256(canon_json(source_row['judgment']))",
            "all_ok": ok == checked == 192 and not mismatches and not missing_original}
