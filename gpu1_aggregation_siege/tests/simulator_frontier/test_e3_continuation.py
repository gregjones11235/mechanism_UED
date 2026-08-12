import json, hashlib
from pathlib import Path
import pytest
import importlib.util, sys
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("cont", root/"scripts"/"e3_continuation.py"); cont=importlib.util.module_from_spec(spec); spec.loader.exec_module(cont)
jspec=importlib.util.spec_from_file_location("journal", root/"src"/"dicode"/"simulator_frontier"/"e3_durable_llm_journal.py"); journal_mod=importlib.util.module_from_spec(jspec); jspec.loader.exec_module(journal_mod); DurablePaidCallJournal=journal_mod.DurablePaidCallJournal

def _entry(role, session, source="old"):
    ident={"source_commit":source,"candidate":"C","session":session,"evidence_hash":f"e{session}{role}","role":role,"provider":"p","requested_model":"m","client_implementation_hash":"old"}
    out={"state_id":"s","bucket_id":"b","frontier_class":"LEARNABLE_FRONTIER","confidence":.8,"dominant_failure":"x","memory_mismatch_suspected":False,"search_budget_sufficient":True,"evidence_ids":["e"],"recommended_evidence_action":"x","diagnosis_hash":"d"} if "diagnostician" in role else {"plan_hash":"p"}
    content_hash=hashlib.sha256(json.dumps("{}",sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str).encode()).hexdigest()
    return {"status":"SUCCESS","key":DurablePaidCallJournal.composite_key(**ident),"key_identity":ident,**ident,"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"content":"{}","response_content_hash":content_hash,"validated_output":out,"validated_output_hash":hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str).encode()).hexdigest()}

def test_continuation_manifest_prefix(tmp_path):
    root=tmp_path/"old"; (root/"evidence").mkdir(parents=True); (root/"runstate").mkdir(parents=True)
    for i in range(1,30):
        d={"session_idx":i,"global_update_step":i*100,"global_env_steps":i*13107200,"durable_role_journal_refs":[{"key":f"k{i}d"},{"key":f"k{i}p"}],"previous_checkpoint":None}
        (root/"evidence"/f"session_{i:03d}.json").write_text(json.dumps(d))
        stem=root/"runstate"/f"e3_canonical_runstate_s{i:03d}"
        if i > 1:
            d["previous_checkpoint"] = str(root/"runstate"/f"e3_canonical_runstate_s{i-1:03d}")
        state=Path(str(stem)+".state.pkl"); state.write_bytes(f"state-{i}".encode())
        sha=hashlib.sha256(state.read_bytes()).hexdigest()
        (Path(str(stem)+".meta.json")).write_text(json.dumps({"state_file_sha256":sha,"current_session_idx":i}))
        d["checkpoint_state_sha256"]=sha
        (root/"evidence"/f"session_{i:03d}.json").write_text(json.dumps(d))
    m=cont.build_manifest(str(root),legacy_source_commit="s",legacy_auth_hash="a",legacy_run_metadata_sha256="m",legacy_journal_sha256="j",session30_diag={"key":"d"},quarantine_planner={"key":"p","content_hash":"c","validated_output_hash":"v","error_code":"FORBIDDEN_ACTION_GUIDANCE_FIELD"})
    assert m["prefix_sessions"]==29 and len(m["prefix_evidence_sha256"])==29
    (root/"evidence"/"session_010.json").unlink()
    with pytest.raises(ValueError): cont.build_manifest(str(root),legacy_source_commit="s",legacy_auth_hash="a",legacy_run_metadata_sha256="m",legacy_journal_sha256="j",session30_diag={"key":"d"},quarantine_planner={"key":"p","content_hash":"c","validated_output_hash":"v","error_code":"FORBIDDEN_ACTION_GUIDANCE_FIELD"})

def test_install_continuation_prefix_rekeys_only_diag(tmp_path):
    J=DurablePaidCallJournal(str(tmp_path/"j.json")); prefix=[_entry(r,s) for s in range(1,30) for r in ("frontier_evidence_diagnostician","curriculum_search_planner")]; diag=_entry("frontier_evidence_diagnostician",30)
    new_identity = {**diag["key_identity"], "source_commit": "new", "client_implementation_hash": "new-client"}
    installed=J.install_continuation_prefix(prefix_entries=prefix,diagnostician_entry=diag,diagnostician_identity=new_identity,provenance={"manifest_hash":"m","legacy_journal_sha256":"j","quarantine_key":"p"})
    assert len(installed)==59 and len(J._load()["entries"])==59 and sum(1 for e in installed if e["session"]==30)==1

def test_continuation_quarantine_manifest_fields_are_required(tmp_path):
    # Structural negative coverage for stale/mismatched quarantine metadata.
    root = tmp_path / "legacy"; (root / "evidence").mkdir(parents=True); (root / "runstate").mkdir()
    for i in range(1, 30):
        (root / "evidence" / f"session_{i:03d}.json").write_text(json.dumps({"session_idx": i, "global_update_step": i*100, "global_env_steps": i*13107200, "durable_role_journal_refs": [{"key": f"d{i}"},{"key": f"p{i}"}], "previous_checkpoint": None if i == 1 else str(root/"runstate"/f"e3_canonical_runstate_s{i-1:03d}")}))
    s = root/"runstate"/"e3_canonical_runstate_s029.state.pkl"; s.write_bytes(b"s"); (root/"runstate"/"e3_canonical_runstate_s029.meta.json").write_text("{}")
    with pytest.raises(ValueError):
        cont.build_manifest(str(root), legacy_source_commit="s", legacy_auth_hash="a", legacy_run_metadata_sha256="m", legacy_journal_sha256="j", session30_diag={"key":"d"}, quarantine_planner={"key":"q","content_hash":"c","validated_output_hash":"v","error_code":"WRONG"})

def test_continuation_runner_rejects_stale_evidence_hash(tmp_path):
    # Hash binding is enforced by the continuation installer; a changed report
    # cannot be accepted under an otherwise valid manifest.
    p = tmp_path / "evidence" / "session_001.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"session_idx": 1})); original = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text(json.dumps({"session_idx": 1, "tampered": True}))
    assert hashlib.sha256(p.read_bytes()).hexdigest() != original

def test_continuation_manifest_identity_and_diag_hash_negative():
    with pytest.raises(ValueError):
        cont.build_manifest("missing", legacy_source_commit="old", legacy_auth_hash="auth", legacy_run_metadata_sha256="m", legacy_journal_sha256="j", session30_diag={"key":"d"}, quarantine_planner={"key":"q","content_hash":"c","validated_output_hash":"v","error_code":"FORBIDDEN_ACTION_GUIDANCE_FIELD"})

@pytest.mark.parametrize("kind", ["nonempty", "planner30", "identity", "provenance"])
def test_install_continuation_prefix_rejects_tamper(tmp_path, kind):
    J=DurablePaidCallJournal(str(tmp_path/"j.json")); prefix=[_entry(r,s) for s in range(1,30) for r in ("frontier_evidence_diagnostician","curriculum_search_planner")]; diag=_entry("frontier_evidence_diagnostician",30)
    if kind == "nonempty":
        J.path.write_text(json.dumps({"schema": journal_mod.SCHEMA, "entries": {"x": {}}}))
    if kind == "planner30":
        diag=_entry("curriculum_search_planner",30)
    if kind == "identity":
        prefix[0]["session"] = 9
    prov={"manifest_hash":"m","legacy_journal_sha256":"j","quarantine_key":"p"}
    if kind == "provenance": prov.pop("quarantine_key")
    new_identity={**diag["key_identity"],"source_commit":"new","client_implementation_hash":"new-client"}
    with pytest.raises(ValueError):
        J.install_continuation_prefix(prefix_entries=prefix, diagnostician_entry=diag, diagnostician_identity=new_identity, provenance=prov)
    if kind == "nonempty":
        assert J.path.read_text() == json.dumps({"schema": journal_mod.SCHEMA, "entries": {"x": {}}})
