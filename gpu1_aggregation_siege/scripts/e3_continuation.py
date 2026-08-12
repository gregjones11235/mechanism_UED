"""Dependency-light immutable T15 continuation prefix manifest helpers."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

SCHEMA = "simulator_frontier.e3-continuation/v1"

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_manifest(legacy_root: str, *, legacy_source_commit: str,
                   legacy_auth_hash: str, legacy_run_metadata_sha256: str,
                   legacy_journal_sha256: str, session30_diag: dict,
                   quarantine_planner: dict, journal_entries=None) -> dict:
    root = Path(legacy_root); evidence = root / "evidence"; reports = []
    refs_seen = []
    for i in range(1, 30):
        p = evidence / f"session_{i:03d}.json"
        if not p.exists(): raise ValueError("continuation evidence gap")
        d = json.loads(p.read_text(encoding="utf-8"))
        refs = d.get("durable_role_journal_refs", [])
        if d.get("session_idx") != i or len(refs) != 2: raise ValueError("continuation report invalid")
        expected_start = (i - 1) * 100
        expected_env = expected_start * 131072
        if d.get("start_global_update", expected_start) != expected_start or d.get("start_global_env_steps", expected_env) != expected_env:
            raise ValueError("continuation report start counter mismatch")
        if d.get("global_update_step") != i * 100 or d.get("global_env_steps") != i * 13107200:
            raise ValueError("continuation report counter mismatch")
        if d.get("source_commit", legacy_source_commit) != legacy_source_commit or d.get("authorization_manifest_hash", legacy_auth_hash) != legacy_auth_hash:
            raise ValueError("continuation report identity mismatch")
        if i > 1:
            previous = d.get("previous_checkpoint")
            expected_name = f"e3_canonical_runstate_s{i-1:03d}"
            if not previous or Path(str(previous)).name != expected_name:
                raise ValueError("continuation checkpoint chain mismatch")
            previous_path = Path(str(previous))
            runstate_root = (root / "runstate").resolve()
            try:
                if runstate_root not in previous_path.resolve().parents:
                    raise ValueError("continuation checkpoint path escapes legacy runstate")
            except FileNotFoundError:
                raise ValueError("continuation checkpoint path invalid") from None
        refs_seen.extend(str(r.get("key", "")) for r in refs if isinstance(r, dict))
        stem = root / "runstate" / f"e3_canonical_runstate_s{i:03d}"
        state, meta = Path(str(stem) + ".state.pkl"), Path(str(stem) + ".meta.json")
        if state.exists() and meta.exists():
            if hashlib.sha256(state.read_bytes()).hexdigest() != d.get("checkpoint_state_sha256"):
                raise ValueError("continuation state hash mismatch")
            md = json.loads(meta.read_text(encoding="utf-8"))
            if md.get("state_file_sha256") != d.get("checkpoint_state_sha256") or md.get("current_session_idx") != i:
                raise ValueError("continuation checkpoint metadata mismatch")
        elif (root / "runstate").exists():
            raise ValueError("continuation checkpoint missing")
        reports.append({"session": i, "sha256": _sha(p)})
    if len(refs_seen) != 58 or len(set(refs_seen)) != 58 or any(not k for k in refs_seen):
        raise ValueError("continuation journal refs invalid")
    if journal_entries is not None:
        by_key = {str(e.get("key")): e for e in journal_entries if isinstance(e, dict)}
        if set(refs_seen) != set(by_key) or len(by_key) != 58:
            raise ValueError("continuation journal prefix mismatch")
        for e in by_key.values():
            if e.get("source_commit") != legacy_source_commit or e.get("role") not in {"frontier_evidence_diagnostician", "curriculum_search_planner"}:
                raise ValueError("continuation journal identity mismatch")
            if e.get("session") not in range(1, 30):
                raise ValueError("continuation journal session mismatch")
        candidate_by_key = {}
        for i in range(1, 30):
            report = json.loads((evidence / f"session_{i:03d}.json").read_text(encoding="utf-8"))
            for ref in report.get("durable_role_journal_refs", []):
                if ref.get("key") in candidate_by_key:
                    raise ValueError("continuation duplicate report ref")
                candidate_by_key[str(ref.get("key"))] = (i, ref)
        for key, entry in by_key.items():
            i, ref = candidate_by_key.get(key, (None, None))
            if i is None or entry.get("session") != i or entry.get("role") != ref.get("role"):
                raise ValueError("continuation journal/report binding mismatch")
            if ref.get("evidence_hash") and entry.get("evidence_hash") != ref.get("evidence_hash"):
                raise ValueError("continuation evidence identity mismatch")
            if not entry.get("candidate") or (ref.get("candidate_id") and entry.get("candidate") != ref.get("candidate_id")):
                raise ValueError("continuation candidate identity missing or mismatched")
    if json.loads((evidence / "session_029.json").read_text())["global_update_step"] != 2900:
        raise ValueError("continuation final update mismatch")
    s029 = root / "runstate" / "e3_canonical_runstate_s029"
    state029, meta029 = Path(str(s029)+".state.pkl"), Path(str(s029)+".meta.json")
    if not state029.exists() or not meta029.exists():
        raise ValueError("continuation s029 checkpoint missing")
    if not all(quarantine_planner.get(k) for k in ("key", "content_hash", "validated_output_hash")) or quarantine_planner.get("error_code") != "FORBIDDEN_ACTION_GUIDANCE_FIELD":
        raise ValueError("continuation toxic planner quarantine incomplete")
    if not session30_diag.get("key"):
        raise ValueError("continuation diagnostician missing")
    return {"schema": SCHEMA, "legacy_root": str(root), "prefix_sessions": 29,
            "legacy_source_commit": legacy_source_commit, "legacy_auth_hash": legacy_auth_hash,
            "legacy_run_metadata_sha256": legacy_run_metadata_sha256,
            "legacy_journal_sha256": legacy_journal_sha256,
            "prefix_evidence_sha256": {f"{r['session']:03d}": r["sha256"] for r in reports},
            "prefix_state_sha256": _sha(state029), "prefix_meta_sha256": _sha(meta029),
            "prefix_final_global_update": 2900, "prefix_final_global_env_steps": 380108800,
            "session30_diag": dict(session30_diag), "quarantine_planner": dict(quarantine_planner)}

def canonical_hash(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
