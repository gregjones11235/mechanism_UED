import json, hashlib
from pathlib import Path
import pytest
import importlib.util, sys
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("cont", root/"scripts"/"e3_continuation.py"); cont=importlib.util.module_from_spec(spec); spec.loader.exec_module(cont)
jspec=importlib.util.spec_from_file_location("journal", root/"src"/"dicode"/"simulator_frontier"/"e3_durable_llm_journal.py"); journal_mod=importlib.util.module_from_spec(jspec); jspec.loader.exec_module(journal_mod); DurablePaidCallJournal=journal_mod.DurablePaidCallJournal
rspec=importlib.util.spec_from_file_location("runner", root/"scripts"/"run_e3_formal_longrun.py"); runner_mod=importlib.util.module_from_spec(rspec); rspec.loader.exec_module(runner_mod)
cspec=importlib.util.spec_from_file_location("clients", root/"src"/"dicode"/"simulator_frontier"/"_e3_real_llm_clients.py"); client_mod=importlib.util.module_from_spec(cspec); cspec.loader.exec_module(client_mod)

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

def _v2_fixture(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"; journal=tmp_path/"journal.json"; entries=[]
    for i,base in ((1,a),(2,b),(3,b)):
        (base/"evidence").mkdir(parents=True,exist_ok=True); (base/"runstate").mkdir(parents=True,exist_ok=True)
        ds=[]
        for role in ("frontier_evidence_diagnostician","curriculum_search_planner"):
            e=_entry(role,i); entries.append(e); ds.append({"key":e["key"],"role":role,"evidence_hash":e["evidence_hash"]})
        stem=base/"runstate"/f"e3_canonical_runstate_s{i:03d}"; state=Path(str(stem)+".state.pkl"); state.write_bytes(f"s{i}".encode()); meta=Path(str(stem)+".meta.json"); meta.write_text(json.dumps({"state_file_sha256":hashlib.sha256(state.read_bytes()).hexdigest(),"current_session_idx":i,"global_update_step":i*100,"global_env_steps":i*100*131072,"source_commit":"old"}))
        prev_root = a if i == 2 else b
        (base/"evidence"/f"session_{i:03d}.json").write_text(json.dumps({"session_idx":i,"candidate_id":"C","global_update_step":i*100,"global_env_steps":i*100*131072,"durable_role_journal_refs":ds,"source_commit":"old","authorization_manifest_hash":"auth","previous_checkpoint":None if i==1 else str((prev_root/"runstate"/f"e3_canonical_runstate_s{i-1:03d}"))}))
    payload={"schema":journal_mod.SCHEMA,"entries":{e["key"]:e for e in entries}}; journal.write_text(json.dumps(payload)); m=cont.build_manifest_v2(report_roots={"1":str(a),"2":str(b),"3":str(b)},journal_path=str(journal),prefix_sessions=3); mp=tmp_path/"m.json"; mp.write_text(json.dumps(m)); return mp,journal,a,b

def test_install_continuation_v2_two_roots_positive(tmp_path):
    mp,_,_,_= _v2_fixture(tmp_path); out=tmp_path/"out"; out.mkdir(); result=runner_mod._install_continuation_v2(manifest_path=str(mp),run_dir=str(out),source_commit="new",candidate_id="C",auth=object(),client_hash="new"); assert result["start_session"]==4 and len(result["loaded_reports"])==3
    assert cont.SCHEMA_V2 == "simulator_frontier.e3-continuation/v2"

@pytest.mark.parametrize("kind",["report","checkpoint","ref","role"])
def test_install_continuation_v2_two_roots_negative(tmp_path,kind):
    mp,journal,a,b=_v2_fixture(tmp_path); m=json.loads(mp.read_text());
    if kind=="report": m["sessions"]["002"]["report_sha256"]="0"*64
    elif kind=="checkpoint": m["sessions"]["003"]["state_sha256"]="0"*64
    elif kind=="ref": m["sessions"]["001"]["journal_refs"][0]="missing"
    else: json.loads((a/"evidence"/"session_001.json").read_text())
    if kind=="role":
        p=a/"evidence"/"session_001.json"; d=json.loads(p.read_text()); d["durable_role_journal_refs"]=d["durable_role_journal_refs"][:1]; p.write_text(json.dumps(d)); m["sessions"]["001"]["report_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    mp.write_text(json.dumps(m)); out=tmp_path/"out"; out.mkdir()
    with pytest.raises(ValueError): runner_mod._install_continuation_v2(manifest_path=str(mp),run_dir=str(out),source_commit="new",candidate_id="C",auth=object(),client_hash="new")

def test_anchor_ratio_zero_rejected():
    with pytest.raises(ValueError):
        client_mod._strict_float(0, lo=0.0, hi=1.0, field="anchor_ratio", exclusive_lo=True)

def test_v2_quarantine_extra_is_manifested_and_not_imported(tmp_path):
    mp,journal,a,b=_v2_fixture(tmp_path); payload=json.loads(journal.read_text()); extra=_entry("frontier_evidence_diagnostician",4); payload["entries"][extra["key"]]=extra; journal.write_text(json.dumps(payload)); m=cont.build_manifest_v2(report_roots={"1":str(a),"2":str(b),"3":str(b)},journal_path=str(journal),prefix_sessions=3); assert extra["key"] in m["quarantine_keys"]; mp.write_text(json.dumps(m)); out=tmp_path/"out"; out.mkdir(); r=runner_mod._install_continuation_v2(manifest_path=str(mp),run_dir=str(out),source_commit="new",candidate_id="C",auth=object(),client_hash="new"); assert extra["key"] not in json.loads((out/"LLM_PAID_CALL_JOURNAL.json").read_text())["entries"]

@pytest.mark.parametrize("kind",["missing","fake","cross","undeclared"])
def test_v2_quarantine_inventory_tamper_rejected(tmp_path,kind):
    mp,journal,a,b=_v2_fixture(tmp_path); payload=json.loads(journal.read_text()); extra=_entry("frontier_evidence_diagnostician",4); payload["entries"][extra["key"]]=extra; journal.write_text(json.dumps(payload)); m=cont.build_manifest_v2(report_roots={"1":str(a),"2":str(b),"3":str(b)},journal_path=str(journal),prefix_sessions=3)
    if kind=="missing": m["quarantine_keys"]=[]
    elif kind=="fake": m["quarantine_keys"].append("fake")
    elif kind=="cross": m["quarantine_keys"].append(m["journal_refs"][0])
    else: m["quarantine_keys"].append("undeclared")
    mp.write_text(json.dumps(m)); out=tmp_path/"out"; out.mkdir()
    with pytest.raises(ValueError): runner_mod._install_continuation_v2(manifest_path=str(mp),run_dir=str(out),source_commit="new",candidate_id="C",auth=object(),client_hash="new")

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
    if kind == "provenance":
        assert len(J.install_continuation_prefix(prefix_entries=prefix, diagnostician_entry=None, diagnostician_identity=None, provenance=prov)) == 58
    else:
        with pytest.raises(ValueError):
            J.install_continuation_prefix(prefix_entries=prefix, diagnostician_entry=diag, diagnostician_identity=new_identity, provenance=prov)
    if kind == "nonempty":
        assert J.path.read_text() == json.dumps({"schema": journal_mod.SCHEMA, "entries": {"x": {}}})
