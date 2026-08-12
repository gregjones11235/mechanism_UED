import importlib.util
import json
import hashlib
import os
import pickle
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

VALID_TASKPARAM_RANGES = {
    "passive_spawn_multiplier": [1.0, 2.0],
    "melee_spawn_multiplier": [1.0, 2.0],
    "ranged_spawn_multiplier": [1.0, 2.0],
    "mob_health_multiplier": [1.0, 2.0],
    "mob_damage_multiplier": [1.0, 2.0],
    "melee_trigger_distance": [2, 4],
    "monsters_killed_to_clear_level": [2, 4],
    "needs_depletion_multiplier": [1.0, 2.0],
    "health_recover_multiplier": [1.0, 2.0],
    "health_loss_multiplier": [1.0, 2.0],
    "mana_recover_multiplier": [1.0, 2.0],
    "growing_plants_age": [2, 4],
}
VALID_START_DISTRIBUTION = {f"D{i:02d}": {"s": 1.0} for i in range(12)}


def test_validate_formal_journal_151x2_and_tamper_matrix(monkeypatch):
    import run_e3_formal_longrun as runner
    entries = {}
    results = []
    for session in range(1, 152):
        refs = []
        for role in ("frontier_evidence_diagnostician", "curriculum_search_planner"):
            key = f"k{session:03d}{role[0]}"
            event = {"key": key, "role": role, "paid_new": True,
                     "reused": False, "requested_model": "m",
                     "returned_model": "m", "usage": {"total_tokens": 1},
                     "response_content_hash": "c", "validated_output_hash": "v"}
            ident = {"session": session, "role": role,
                     "candidate": "C", "source_commit": "S"}
            entries[key] = {"key_identity": ident, "requested_model": "m",
                            "returned_model": "m", "usage": event["usage"],
                            "response_content_hash": "c",
                            "validated_output_hash": "v"}
            refs.append(event)
        results.append({"session_idx": session, "candidate_id": "C",
                        "source_commit": "S", "durable_role_journal_refs": refs})
    journal = {"entries": entries, "installed_target_keys": []}
    assert runner._validate_formal_journal(results, journal)
    bad = json.loads(json.dumps(results)); bad[0]["durable_role_journal_refs"][1]["key"] = "missing"
    assert not runner._validate_formal_journal(bad, journal)
    bad = json.loads(json.dumps(results)); bad[0]["durable_role_journal_refs"][0]["reused"] = True
    assert not runner._validate_formal_journal(bad, journal)
    bad = json.loads(json.dumps(results)); bad[0]["durable_role_journal_refs"] = bad[0]["durable_role_journal_refs"][:1]
    assert not runner._validate_formal_journal(bad, journal)
    journal2 = {"entries": dict(entries), "installed_target_keys": ["old"]}
    assert not runner._validate_formal_journal(results, journal2)


def test_journal_installed_preseed_ceiling(tmp_path):
    cls = _journal_cls()
    j = cls(str(tmp_path / "j.json"), max_success_keys=302)
    # Seed metadata-only installed keys to exercise the adjusted current ceiling.
    payload = {"schema": j._load()["schema"], "entries": {
        f"i{i}": {"status": "SUCCESS", "key": f"i{i}"} for i in range(3)},
        "installed_target_keys": [f"i{i}" for i in range(3)]}
    (tmp_path / "j.json").write_text(json.dumps(payload))
    # Tamper is fail-closed on load; this test documents that synthetic entries
    # must still be fully validated by install_preseed_entries in real use.
    with pytest.raises(ValueError):
        j._load()


def test_journal_three_installed_plus_302_current_ceiling(tmp_path, monkeypatch):
    cls = _journal_cls()
    path = tmp_path / "ceiling.json"
    j = cls(str(path), max_success_keys=302)
    installed = {f"i{i}": {"status": "SUCCESS", "key": f"i{i}"} for i in range(3)}
    path.write_text(json.dumps({"schema": j._load()["schema"], "entries": installed,
                                "installed_target_keys": list(installed)}))
    monkeypatch.setattr(j, "_valid_entry", lambda key, entry: True)
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    for session in range(1, 303):
        identity = {"source_commit": "S", "candidate": "C", "session": session,
                    "evidence_hash": f"e{session}", "role": "frontier_evidence_diagnostician",
                    "provider": "p", "requested_model": "m",
                    "client_implementation_hash": "h"}
        key = j.composite_key(**identity)
        j.record_success(key=key, identity=identity, returned_model="m",
                         usage=usage, validated_output={"ok": True},
                         response_content="x")
    with pytest.raises(ValueError):
        identity = {"source_commit": "S", "candidate": "C", "session": 303,
                    "evidence_hash": "e303", "role": "frontier_evidence_diagnostician",
                    "provider": "p", "requested_model": "m",
                    "client_implementation_hash": "h"}
        j.record_success(key=j.composite_key(**identity), identity=identity,
                         returned_model="m", usage=usage,
                         validated_output={"ok": True}, response_content="x")
    assert len(j._load()["entries"]) == 305


def _journal_cls():
    p = Path(__file__).resolve().parents[2] / "src" / "dicode" / "simulator_frontier" / "e3_durable_llm_journal.py"
    spec = importlib.util.spec_from_file_location("e3dj", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DurablePaidCallJournal


def test_budget_positive_and_negative():
    import e3_authorization as a
    a.resolve_e3_budget(candidate=a.FORMAL_BUDGET_CANDIDATE, sessions=151, scope="formal")
    for n in (150, 152):
        with pytest.raises(ValueError):
            a.resolve_e3_budget(candidate=a.FORMAL_BUDGET_CANDIDATE, sessions=n, scope="formal")


def test_journal_tamper_and_ceiling(tmp_path):
    J = _journal_cls(); j = J(str(tmp_path / "j.json"), max_success_keys=1)
    ident = dict(source_commit="a", candidate="b", session=1, evidence_hash="c",
                 role="frontier_evidence_diagnostician", provider="p", requested_model="m", client_implementation_hash="i")
    key = j.composite_key(**ident)
    j.record_success(key=key, identity=ident, returned_model="m",
                     usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                     validated_output={"ok": True}, response_content="{}",
                     )
    assert j.lookup(key, identity=ident)["key"] == key
    payload = json.loads((tmp_path / "j.json").read_text())
    payload["entries"][key]["content"] = "tampered"
    (tmp_path / "j.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        j.lookup(key, identity=ident)


def test_journal_concurrent_unique_and_usage_rejection(tmp_path):
    J = _journal_cls(); path = tmp_path / "concurrent.json"
    def one(i):
        j = J(str(path)); ident = dict(source_commit="a", candidate="b", session=i,
            evidence_hash=str(i), role="frontier_evidence_diagnostician", provider="p", requested_model="m",
            client_implementation_hash="i")
        key = j.composite_key(**ident)
        return j.record_success(key=key, identity=ident, returned_model="m",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            validated_output={"i": i}, response_content="{}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, range(16)))
    j = J(str(path)); assert len(j._load()["entries"]) == 16
    capped = J(str(tmp_path / "cap.json"), max_success_keys=2)
    for i in range(2): one_ident = dict(source_commit="c", candidate="d", session=i,
        evidence_hash=str(i), role="frontier_evidence_diagnostician", provider="p", requested_model="m", client_implementation_hash="i"); capped.record_success(key=capped.composite_key(**one_ident), identity=one_ident, returned_model="m", usage={"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}, validated_output={"i":i}, response_content="{}")
    third = dict(source_commit="c", candidate="d", session=3, evidence_hash="3",
                 role="frontier_evidence_diagnostician", provider="p", requested_model="m", client_implementation_hash="i")
    with pytest.raises(ValueError):
        capped.record_success(key=capped.composite_key(**third), identity=third,
            returned_model="m", usage={"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},
            validated_output={"i":3}, response_content="{}")
    assert len(capped._load()["entries"]) == 2


def test_journal_role_caps_and_preseed_rekey(tmp_path):
    J = _journal_cls()
    source_path = tmp_path / "source.json"
    source = J(str(source_path))
    old = dict(source_commit="old", candidate="b", session=1,
               evidence_hash="prompt-hash", role="frontier_evidence_diagnostician",
               provider="p", requested_model="m", client_implementation_hash="old-client")
    old_key = source.composite_key(**old)
    source.record_success(key=old_key, identity=old, returned_model="m",
        usage={"prompt_tokens": 1, "completion_tokens": 1024, "total_tokens": 1025},
        validated_output={"ok": True}, response_content="diag")
    payload = json.loads(source_path.read_text())
    payload["preseed_provenance"] = {
        "source_run": "old-run", "source_key": old_key,
        "source_journal_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_commit": "old", "source_client_implementation_hash": "old-client",
    }
    source_path.write_text(json.dumps(payload))
    target = J(str(tmp_path / "target.json"))
    new = dict(source_commit="new", candidate="b", session=1,
               evidence_hash="prompt-hash", role="frontier_evidence_diagnostician",
               provider="p", requested_model="m", client_implementation_hash="new-client")
    entry = target.install_preseed(source_entry=payload["entries"][old_key],
        identity=new, provenance=payload["preseed_provenance"])
    assert target.lookup(entry["key"], identity=new)["validated_output"] == {"ok": True}
    planner = dict(new, role="curriculum_search_planner", evidence_hash="plan")
    pkey = target.composite_key(**planner)
    target2 = J(str(tmp_path / "planner.json"))
    target2.record_success(key=pkey, identity=planner, returned_model="m",
        usage={"prompt_tokens": 1, "completion_tokens": 4096, "total_tokens": 4097},
        validated_output={"plan": True}, response_content="plan")
    with pytest.raises(ValueError):
        target2.record_success(key=target2.composite_key(**dict(planner, role="unknown")),
            identity=dict(planner, role="unknown"), returned_model="m",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            validated_output={}, response_content="x")
    with pytest.raises(ValueError):
        target2.record_success(key=pkey, identity=planner, returned_model="m",
            usage={"prompt_tokens": 1, "completion_tokens": 4097, "total_tokens": 4098},
            validated_output={}, response_content="x")


def test_journal_contiguous_dual_prefix_rekeys(tmp_path):
    J = _journal_cls(); source = J(str(tmp_path / "prefix.json"))
    entries = []
    for session, role in ((1, "frontier_evidence_diagnostician"),
                          (1, "curriculum_search_planner"),
                          (2, "frontier_evidence_diagnostician")):
        ident = dict(source_commit="old", candidate="b", session=session,
                     evidence_hash=f"h-{session}-{role}", role=role,
                     provider="p", requested_model="m", client_implementation_hash="old-client")
        key = source.composite_key(**ident)
        cap = 1024 if role == "frontier_evidence_diagnostician" else 4096
        source.record_success(key=key, identity=ident, returned_model="m",
            usage={"prompt_tokens": 1, "completion_tokens": cap, "total_tokens": cap + 1},
            validated_output={"session": session, "role": role}, response_content="{}")
        entries.append((key, source._load()["entries"][key], ident))
    target = J(str(tmp_path / "target-prefix.json"))
    identities = [dict(i, source_commit="new", client_implementation_hash="new-client")
                  for _, _, i in entries]
    installed = target.install_preseed_entries(
        entries=[e for _, e, _ in entries], identities=identities,
        provenance={"source_run": "old", "source_keys": [k for k, _, _ in entries]})
    assert [e["session"] for e in installed] == [1, 1, 2]
    assert {e["role"] for e in installed if e["session"] == 1} == {
        "frontier_evidence_diagnostician", "curriculum_search_planner"}
    assert installed[-1]["role"] == "frontier_evidence_diagnostician"


def test_preseed_role_maps_partial_prefix(monkeypatch):
    import run_e3_formal_longrun as runner
    mandatory, opportunistic = runner._preseed_role_maps([
        {"session": 1, "role": "frontier_evidence_diagnostician", "key": "d1"},
        {"session": 1, "role": "curriculum_search_planner", "key": "p1"},
        {"session": 2, "role": "frontier_evidence_diagnostician", "key": "d2"},
    ])
    assert mandatory == {1: "d1"} and opportunistic == {2: "d2"}
    assert runner._preseed_role_maps([
        {"session": 1, "role": "frontier_evidence_diagnostician", "key": "d1"},
    ]) == ({}, {1: "d1"})


def test_runner_preseed_install_rekeys_and_tamper_blocks(tmp_path, monkeypatch):
    import run_e3_formal_longrun as runner
    J = _journal_cls(); source_path = tmp_path / "preseed.json"
    source = J(str(source_path))
    ident = dict(source_commit="old", candidate="SLOWGRU_PERSISTENT_CANONICAL_98304",
                 session=1, evidence_hash="prompt", role="frontier_evidence_diagnostician",
                 provider="dashscope", requested_model="qwen-plus", client_implementation_hash="old-client")
    key = source.composite_key(**ident)
    source.record_success(key=key, identity=ident, returned_model="qwen-plus",
        usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        validated_output={"state_id":"s"}, response_content="diag")
    payload = json.loads(source_path.read_text())
    payload["preseed_provenance"] = {"source_run":"old-run", "source_key":key,
        "source_journal_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_commit":"old", "source_client_implementation_hash":"old-client"}
    source_path.write_text(json.dumps(payload))
    auth = SimpleNamespace(preseed_journal_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    run_dir = tmp_path / "run"; run_dir.mkdir()
    monkeypatch.delenv("E3_PRESEEDED_DIAGNOSTIC_KEY", raising=False)
    info = runner._install_preseed_journal(preseed_path=str(source_path), run_dir=str(run_dir), auth=auth,
        source_commit="new", candidate_id="SLOWGRU_PERSISTENT_CANONICAL_98304", client_hash="new-client")
    assert info["source_key"] == key and info["installed_key"] != key
    assert len(J(str(run_dir / "LLM_PAID_CALL_JOURNAL.json"))._load()["entries"]) == 1
    auth.preseed_journal_sha256 = "0" * 64
    with pytest.raises(ValueError):
        runner._install_preseed_journal(preseed_path=str(source_path), run_dir=str(tmp_path / "other"), auth=auth,
            source_commit="new", candidate_id="SLOWGRU_PERSISTENT_CANONICAL_98304", client_hash="new-client")


def test_dual_preseed_entries_atomically_rekey(tmp_path):
    J = _journal_cls(); source = J(str(tmp_path / "source.json"))
    entries = []; identities = []
    for role, evidence_hash in (("frontier_evidence_diagnostician", "diag"),
                                ("curriculum_search_planner", "plan")):
        ident = dict(source_commit="old", candidate="SLOWGRU_PERSISTENT_CANONICAL_98304",
                     session=1, evidence_hash=evidence_hash, role=role,
                     provider="dashscope", requested_model="qwen-plus", client_implementation_hash="old")
        key = source.composite_key(**ident)
        source.record_success(key=key, identity=ident, returned_model="qwen-plus",
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            validated_output={"role": role}, response_content=role)
        entries.append(source._load()["entries"][key])
        identities.append({**ident, "source_commit": "new", "client_implementation_hash": "new"})
    target = J(str(tmp_path / "target.json"))
    installed = target.install_preseed_entries(entries=entries, identities=identities,
        provenance={"source_run":"old", "source_key":"diag", "source_journal_sha256":"x",
                    "source_commit":"old", "source_client_implementation_hash":"old"})
    assert {entry["role"] for entry in installed} == {
        "frontier_evidence_diagnostician", "curriculum_search_planner"}
    assert len(target._load()["entries"]) == 2


def test_dual_preseed_client_path_reuses_both_without_transport(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    J = _journal_cls(); path = tmp_path / "dual.json"; journal = J(str(path))
    evidence = {"feasibility": {"state_id": "s"}, "archive_summary": {"bucket_id": "b", "evidence_ids": ["e"]}}
    diag = {"state_id":"s","bucket_id":"b","frontier_class":"LEARNABLE_FRONTIER","confidence":.8,
            "dominant_failure":"x","memory_mismatch_suspected":False,"search_budget_sufficient":True,
            "evidence_ids":["e"],"recommended_evidence_action":"x"}
    dh = clients._canonical_sha256({"evidence":{"feasibility":{"state_id":"s"},"data_source":""}})
    full_eh = clients._evidence_hash_of(evidence)
    from dicode.simulator_frontier.llm_contracts import compute_diagnostician_hash, compute_planner_hash
    diag["diagnosis_hash"] = compute_diagnostician_hash(diag, evidence_hash=full_eh)
    planner_input = {**evidence, "diagnostician_summary": diag}
    ph = clients._canonical_sha256(planner_input)
    valid_planner = {"plan_id":"plan-old", "based_on_diagnosis_hash":diag["diagnosis_hash"],
        "bucket_modifications":{}, "start_distribution":VALID_START_DISTRIBUTION,
        "taskparam_ranges":VALID_TASKPARAM_RANGES, "seed_distribution":{"s":[0,1]},
        "stochasticity_distribution":{"x":[0,1]}, "search_source":"STUDENT_DETERMINISTIC",
        "actual_n":4, "horizon":16, "memory_mode":"SAVED_POLICY_MEMORY",
        "anchor_ratio":0.2, "retention_constraints":["x"], "reason":"x"}
    valid_planner["plan_hash"] = compute_planner_hash(valid_planner, evidence_hash=full_eh)
    identities=[]; entries=[]
    for role, eh, out in (("frontier_evidence_diagnostician", dh, diag), ("curriculum_search_planner", ph, valid_planner)):
        old = dict(source_commit="old", candidate="b", session=1, evidence_hash=eh, role=role,
                   provider="dashscope", requested_model="m", client_implementation_hash="old")
        key = journal.composite_key(**old)
        journal.record_success(key=key, identity=old, returned_model="m",
            usage={"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},
            validated_output=out, response_content="{}")
        entries.append(journal._load()["entries"][key])
        identities.append({**old, "source_commit":"new", "client_implementation_hash":"new"})
    target = J(str(tmp_path / "dual_target.json")); target.install_preseed_entries(entries=entries, identities=identities,
        provenance={"source_run":"old","source_key":"k","source_journal_sha256":"x","source_commit":"old","source_client_implementation_hash":"old"})
    target_path = tmp_path / "dual_target.json"
    monkeypatch.setenv("QWEN_MODEL","m"); monkeypatch.setenv("E3_LLM_JOURNAL_PATH",str(target_path)); monkeypatch.setenv("E3_SOURCE_COMMIT","new"); monkeypatch.setenv("E3_CANDIDATE_ID","b"); monkeypatch.setenv("E3_SESSION_IDX","1"); monkeypatch.setenv("E3_CLIENT_FACTORY_IMPLEMENTATION_HASH","new")
    calls=[]; monkeypatch.setattr(clients,"_call_qwen",lambda *a,**k: calls.append(1))
    clients.clear_audit_events(); d=clients._DiagnosticianClient("s","b").complete(evidence); p=clients._PlannerClient("s",4,16).complete(planner_input)
    events=clients.drain_audit_events(); assert not calls and len(events)==2 and all(e["reused"] for e in events)


def test_preseed_reuse_diag_only_planner_transport(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("E3_PRESEEDED_DIAGNOSTIC_KEY", raising=False)
    import run_e3_formal_longrun as runner
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    from dicode.simulator_frontier.llm_contracts import compute_diagnostician_hash
    J = _journal_cls()
    evidence = {"feasibility":{"state_id":"s"}, "archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}
    prompt_hash = clients._canonical_sha256({"evidence":{"feasibility":{"state_id":"s"},"data_source":""}})
    diag = {"state_id":"s","bucket_id":"b","frontier_class":"LEARNABLE_FRONTIER",
            "confidence":0.8,"dominant_failure":"x","memory_mismatch_suspected":False,
            "search_budget_sufficient":True,"evidence_ids":["e"],"recommended_evidence_action":"x"}
    diag["diagnosis_hash"] = compute_diagnostician_hash(diag, evidence_hash=clients._evidence_hash_of(evidence))
    source_path = tmp_path / "preseed.json"; source = J(str(source_path))
    old = dict(source_commit="old", candidate="SLOWGRU_PERSISTENT_CANONICAL_98304", session=1,
               evidence_hash=prompt_hash, role="frontier_evidence_diagnostician", provider="dashscope",
               requested_model="m", client_implementation_hash="old-client")
    old_key = source.composite_key(**old)
    source.record_success(key=old_key, identity=old, returned_model="m",
        usage={"prompt_tokens":2,"completion_tokens":5,"total_tokens":7},
        validated_output=diag, response_content="diag")
    payload=json.loads(source_path.read_text()); payload["preseed_provenance"]={
        "source_run":"/old/run","source_key":old_key,
        "source_journal_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_commit":"old","source_client_implementation_hash":"old-client"}
    source_path.write_text(json.dumps(payload))
    target_dir=tmp_path / "target"; target_dir.mkdir()
    auth=SimpleNamespace(preseed_journal_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), provider="dashscope", requested_model="m")
    runner._install_preseed_journal(preseed_path=str(source_path), run_dir=str(target_dir), auth=auth,
        source_commit="new", candidate_id="SLOWGRU_PERSISTENT_CANONICAL_98304", client_hash="new-client")
    monkeypatch.setenv("QWEN_MODEL","m"); monkeypatch.setenv("E3_LLM_JOURNAL_PATH",str(target_dir / "LLM_PAID_CALL_JOURNAL.json"))
    monkeypatch.setenv("E3_SOURCE_COMMIT","new"); monkeypatch.setenv("E3_CANDIDATE_ID","SLOWGRU_PERSISTENT_CANONICAL_98304"); monkeypatch.setenv("E3_SESSION_IDX","1")
    monkeypatch.setenv("E3_CLIENT_FACTORY_IMPLEMENTATION_HASH", "new-client")
    calls=[]
    def fake(system,user,**kwargs):
        calls.append((system,kwargs))
        body={"bucket_modifications":{},"taskparam_ranges":VALID_TASKPARAM_RANGES,"seed_distribution":{"s":[0,1]},"stochasticity_distribution":{"x":[0,1]},"anchor_ratio":.2,"retention_constraints":["x"],"reason":"x","start_distribution":VALID_START_DISTRIBUTION}
        return {"content":json.dumps(body),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    monkeypatch.setattr(clients,"_call_qwen",fake); clients.clear_audit_events()
    d=clients._DiagnosticianClient("s","b").complete(evidence)
    clients._PlannerClient("s",4,16).complete({**evidence,"diagnostician_summary":d})
    events=clients.drain_audit_events()
    assert len(calls)==1 and calls[0][1]["max_tokens"]==4096
    assert [e["role"] for e in events] == ["frontier_evidence_diagnostician","curriculum_search_planner"]
    assert events[0]["reused"] and events[0]["paid_new"] is False
    assert events[1]["paid_new"] and events[1]["reused"] is False


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 1, "completion_tokens": 1},
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 0},
    {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 2},
    {"prompt_tokens": 1, "completion_tokens": 1025, "total_tokens": 1026},
    {"prompt_tokens": 1, "completion_tokens": 20001, "total_tokens": 20002},
])
def test_invalid_usage_rejected(tmp_path, usage):
    J = _journal_cls(); j = J(str(tmp_path / "u.json"))
    ident = dict(source_commit="a", candidate="b", session=1, evidence_hash="u",
                 role="frontier_evidence_diagnostician", provider="p", requested_model="m", client_implementation_hash="i")
    with pytest.raises(ValueError):
        j.record_success(key=j.composite_key(**ident), identity=ident,
                         returned_model="m", usage=usage, validated_output={}, response_content="")
    assert j._load()["entries"] == {}


@pytest.mark.parametrize("field", ["content", "validated_output", "requested_model",
                                    "returned_model", "key_identity", "usage_total",
                                    "usage_completion", "usage_sum"])
def test_existing_entry_tamper_hard_fails(tmp_path, field):
    J = _journal_cls(); path = tmp_path / "tamper.json"; j = J(str(path))
    ident = dict(source_commit="a", candidate="b", session=1, evidence_hash="t",
                 role="frontier_evidence_diagnostician", provider="p", requested_model="m", client_implementation_hash="i")
    key = j.composite_key(**ident)
    j.record_success(key=key, identity=ident, returned_model="m",
        usage={"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},
        validated_output={"ok":True}, response_content="{}")
    payload = json.loads(path.read_text()); entry = payload["entries"][key]
    if field == "content": entry["content"] = "x"
    elif field == "validated_output": entry["validated_output"] = {"bad": True}
    elif field == "requested_model": entry["requested_model"] = "other"
    elif field == "returned_model": entry["returned_model"] = "other"
    elif field == "key_identity": entry["key_identity"]["session"] = 99
    elif field == "usage_total": entry["usage"]["total_tokens"] = 0
    elif field == "usage_completion": entry["usage"]["completion_tokens"] = 1025
    else: entry["usage"]["total_tokens"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): j.lookup(key, identity=ident)


def test_run_metadata_helpers(tmp_path):
    import run_e3_formal_longrun as r
    payload={"source_commit":"a","candidate_id":"b","sessions":2}
    r.write_run_metadata_once(str(tmp_path), payload)
    raw=(tmp_path/"RUN_METADATA.json").read_bytes(); assert r.verify_run_metadata(str(tmp_path), payload)==payload
    assert (tmp_path/"RUN_METADATA.json").read_bytes()==raw
    with pytest.raises(ValueError): r.write_run_metadata_once(str(tmp_path), payload)
    r.append_resume_event(str(tmp_path), {"pid":1})
    assert (tmp_path/"RUN_METADATA.json").read_bytes()==raw
    assert json.loads((tmp_path/"RESUME_EVENTS.jsonl").read_text().strip())["pid"]==1
    (tmp_path/"RUN_METADATA.sha256").write_text("0"*64)
    with pytest.raises(ValueError): r.verify_run_metadata(str(tmp_path), payload)


def test_run_lease_releases_on_raise(tmp_path):
    import run_e3_formal_longrun as r
    with pytest.raises(RuntimeError):
        with r.run_lease(str(tmp_path/"x"), metadata={}):
            raise RuntimeError("boom")
    with r.run_lease(str(tmp_path/"x"), metadata={}):
        pass


def test_run_lease_cross_process(tmp_path):
    import run_e3_formal_longrun as r
    run_dir = tmp_path / "proc"
    run_dir.mkdir()
    child = tmp_path / "child_lease.py"
    child.write_text("import os,sys; sys.path.insert(0,sys.argv[1]); import run_e3_formal_longrun as r; from pathlib import Path\nwith r.run_lease(sys.argv[2], metadata={}):\n Path(sys.argv[3]).write_text(str(os.getpid())); sys.stdin.readline()\n")
    proc = subprocess.Popen([sys.executable, str(child), str(Path(__file__).resolve().parents[2]/"scripts"), str(run_dir), str(run_dir/"ready")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = run_dir / "ready"
        deadline = time.time() + 10
        while time.time() < deadline and not ready.exists(): time.sleep(.05)
        assert ready.exists()
        with pytest.raises(RuntimeError):
            with r.run_lease(str(run_dir), metadata={}): pass
        proc.stdin.write("\n"); proc.stdin.flush(); assert proc.wait(timeout=10) == 0
        with r.run_lease(str(run_dir), metadata={}) as rec:
            assert any(str(old.get("pid")) == ready.read_text() for old in rec.get("takeover_history", []))
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()


def test_resume_helper_counter_formula():
    import run_e3_formal_longrun as r
    assert 100 * r.ENV_STEPS_PER_UPDATE == 13_107_200


@pytest.mark.parametrize("metrics,ok", [
    ({"task": {"lp": float("nan"), "sr": 0.5}}, True),
    ({"task": {"lp": -1.0, "sr": 0.5}}, True),
    ({"task": {"lp": float("inf")}}, False),
    ({"task": {"sr": float("nan")}}, False),
    ({"task": {"nested": {"mean_return": float("inf")}}}, False),
    ({"task": 3.0}, False),
    ({"task": {"lp": [float("nan")]}}, False),
    ({"task": {"lp": complex(1, 2)}}, False),
])
def test_training_metrics_lp_nan_sentinel(metrics, ok):
    import run_e3_formal_longrun as r
    assert r._training_metrics_finite(metrics) is ok


def test_preseed_session_key_stale_fails_before_transport(monkeypatch):
    import run_e3_real_smoke as smoke
    monkeypatch.setenv("E3_SESSION_IDX", "2")
    monkeypatch.setenv("E3_PRESEEDED_DIAGNOSTIC_KEY", "stale")
    with pytest.raises(RuntimeError, match="PRESEEDED_SESSION_KEY_STALE"):
        smoke.run_two_real_llm_roles(object(), {"feasibility": {"state_id": "s"}})


def test_wrapper_session1_preseed_reused_clears_key(monkeypatch):
    smoke_path = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_real_smoke.py"
    spec = importlib.util.spec_from_file_location("e3_smoke_wrapper_test", smoke_path)
    smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
    clients = types.ModuleType("_e3_real_llm_clients")
    llm_contracts = types.ModuleType("llm_contracts")
    invocation_gate = types.ModuleType("invocation_gate")
    invocation_gate.InvocationReason = types.SimpleNamespace(REVISION_REQUIRED="REVISION_REQUIRED")
    invocation_gate.decide_invocation = lambda *a, **k: object()
    invocation_gate.evidence_hash_of = lambda value: "evidence"
    pkg = types.ModuleType("dicode.simulator_frontier")
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier", pkg)
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier._e3_real_llm_clients", clients)
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier.llm_contracts", llm_contracts)
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier.invocation_gate", invocation_gate)
    monkeypatch.setenv("E3_SESSION_IDX", "1")
    monkeypatch.setenv("E3_PRESEEDED_DIAGNOSTIC_KEY", "diag-key")
    monkeypatch.setenv("E3_PRESEEDED_DIAGNOSTIC_SESSION", "1")
    clients.clear_audit_events = lambda: None
    clients.drain_audit_events = lambda: [
        {"role": "frontier_evidence_diagnostician", "reused": True, "paid_new": False},
        {"role": "curriculum_search_planner", "reused": True, "paid_new": False},
    ]
    llm_contracts.run_two_llm_production = lambda *a, **k: {"role_order": [
        "frontier_evidence_diagnostician", "curriculum_search_planner"]}
    result = smoke.run_two_real_llm_roles(object(), {"feasibility": {"state_id": "s"}})
    assert len(result["audit_events"]) == 2
    assert "E3_PRESEEDED_DIAGNOSTIC_KEY" not in os.environ


def test_wrapper_session1_nonreused_preserves_key(monkeypatch):
    smoke_path = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_real_smoke.py"
    spec = importlib.util.spec_from_file_location("e3_smoke_wrapper_test_bad", smoke_path)
    smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
    clients = types.ModuleType("_e3_real_llm_clients")
    llm_contracts = types.ModuleType("llm_contracts")
    invocation_gate = types.ModuleType("invocation_gate")
    invocation_gate.InvocationReason = types.SimpleNamespace(REVISION_REQUIRED="REVISION_REQUIRED")
    invocation_gate.decide_invocation = lambda *a, **k: object()
    invocation_gate.evidence_hash_of = lambda value: "evidence"
    pkg = types.ModuleType("dicode.simulator_frontier")
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier", pkg)
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier._e3_real_llm_clients", clients)
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier.llm_contracts", llm_contracts)
    monkeypatch.setitem(sys.modules, "dicode.simulator_frontier.invocation_gate", invocation_gate)
    monkeypatch.setenv("E3_SESSION_IDX", "1")
    monkeypatch.setenv("E3_PRESEEDED_DIAGNOSTIC_KEY", "diag-key")
    monkeypatch.setenv("E3_PRESEEDED_DIAGNOSTIC_SESSION", "1")
    clients.clear_audit_events = lambda: None
    clients.drain_audit_events = lambda: [
        {"role": "frontier_evidence_diagnostician", "reused": False, "paid_new": True},
        {"role": "curriculum_search_planner", "reused": True, "paid_new": False},
    ]
    llm_contracts.run_two_llm_production = lambda *a, **k: {"role_order": [
        "frontier_evidence_diagnostician", "curriculum_search_planner"]}
    with pytest.raises(RuntimeError, match="PRESEEDED_DIAGNOSTIC_NOT_REUSED"):
        smoke.run_two_real_llm_roles(object(), {"feasibility": {"state_id": "s"}})
    assert os.environ.get("E3_PRESEEDED_DIAGNOSTIC_KEY") == "diag-key"


def test_two_role_client_journal_reuse(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("E3_PRESEEDED_DIAGNOSTIC_KEY", raising=False)
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(tmp_path / "roles.json"))
    monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "1")
    calls = []
    def fake(system, user, **kwargs):
        calls.append(system)
        if "Diagnostician" in system:
            body = {"frontier_class":"LEARNABLE_FRONTIER","confidence":0.8,"dominant_failure":"x","memory_mismatch_suspected":False,"search_budget_sufficient":True,"recommended_evidence_action":"x"}
        else:
            body = {"bucket_modifications":{},"taskparam_ranges":VALID_TASKPARAM_RANGES,"seed_distribution":{"seed":[0,1]},"stochasticity_distribution":{"x":[0,1]},"anchor_ratio":0.2,"retention_constraints":["x"],"reason":"x","start_distribution":VALID_START_DISTRIBUTION}
        return {"content": json.dumps(body), "requested_model":"m", "returned_model":"m", "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    monkeypatch.setattr(clients, "_call_qwen", fake)
    evidence={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}
    dc=clients._DiagnosticianClient("s","b"); pc=clients._PlannerClient("s",4,16)
    d=dc.complete(evidence); p=pc.complete({**evidence,"diagnostician_summary":d})
    d2=dc.complete(evidence); p2=pc.complete({**evidence,"diagnostician_summary":d2})
    assert len(calls)==2
    assert d2 == d and p2 == p
    J = _journal_cls(); entries = J(str(tmp_path / "roles.json"))._load()["entries"]
    assert len(entries) == 2
    assert {e.get("role") for e in entries.values()} == {
        "frontier_evidence_diagnostician", "curriculum_search_planner"}


def test_role_specific_transport_caps_and_preseed_miss_blocks(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("E3_PRESEEDED_DIAGNOSTIC_KEY", raising=False)
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(tmp_path / "caps.json"))
    monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "1")
    calls = []
    def fake(system, user, **kwargs):
        calls.append((system, user, kwargs))
        body = ({"frontier_class":"LEARNABLE_FRONTIER","confidence":.8,"dominant_failure":"x","memory_mismatch_suspected":False,"search_budget_sufficient":True,"recommended_evidence_action":"x"}
                if "Diagnostician" in system else {"bucket_modifications":{},"taskparam_ranges":VALID_TASKPARAM_RANGES,"seed_distribution":{"s":[0,1]},"stochasticity_distribution":{"x":[0,1]},"anchor_ratio":.2,"retention_constraints":["x"],"reason":"x","start_distribution":VALID_START_DISTRIBUTION})
        return {"content":json.dumps(body),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    monkeypatch.setattr(clients, "_call_qwen", fake)
    evidence={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}
    d=clients._DiagnosticianClient("s","b").complete(evidence)
    clients._PlannerClient("s",4,16).complete({**evidence,"diagnostician_summary":d})
    assert [kwargs["max_tokens"] for _, _, kwargs in calls] == [1024, 4096]
    planner_system, planner_user, _ = calls[1]
    assert "required_current_state_id" in planner_user
    assert all(slot in planner_user for slot in ("D00", "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10", "D11"))
    assert '"s"' in planner_user
    assert '"taskparam_lower_bounds"' in planner_user and '"growing_plants_age": 2' in planner_user
    assert "exact required_current_state_id" in planner_system
    monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(tmp_path / "missing-preseed.json"))
    monkeypatch.setenv("E3_PRESEEDED_DIAGNOSTIC_KEY", "missing-key")
    with pytest.raises(RuntimeError, match="PRESEED_DIAGNOSTIC_EVIDENCE_MISMATCH"):
        clients._DiagnosticianClient("s","b").complete(evidence)


def test_diagnosis_change_rekeys_planner(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("E3_PRESEEDED_DIAGNOSTIC_KEY", raising=False)
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(tmp_path / "d.json")); monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "1")
    calls=[]
    def fake(system,user,**kw):
        calls.append(system)
        body={"frontier_class":"LEARNABLE_FRONTIER","confidence":.8,"dominant_failure":"x","memory_mismatch_suspected":False,"search_budget_sufficient":True,"recommended_evidence_action":"x"} if "Diagnostician" in system else {"bucket_modifications":{},"taskparam_ranges":VALID_TASKPARAM_RANGES,"seed_distribution":{"s":[0,1]},"stochasticity_distribution":{"x":[0,1]},"anchor_ratio":.2,"retention_constraints":["x"],"reason":"x","start_distribution":VALID_START_DISTRIBUTION}
        return {"content":json.dumps(body),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    monkeypatch.setattr(clients,"_call_qwen",fake); evidence={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}
    dc=clients._DiagnosticianClient("s","b"); pc=clients._PlannerClient("s",4,16); d=dc.complete(evidence); pc.complete({**evidence,"diagnostician_summary":d})
    changed=dict(d); changed["confidence"]=.9
    from dicode.simulator_frontier.llm_contracts import compute_diagnostician_hash
    changed["diagnosis_hash"]=compute_diagnostician_hash(changed,evidence_hash=clients._evidence_hash_of(evidence)); clients.clear_audit_events(); dc.complete(evidence); pc.complete({**evidence,"diagnostician_summary":changed}); events=clients.drain_audit_events()
    assert len(calls)==3 and len(events)==2 and events[0]["reused"] and events[1]["paid_new"]
    assert len(_journal_cls()(str(tmp_path / "d.json"))._load()["entries"]) == 3


def test_role2_failure_resume_reuses_role1(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("E3_PRESEEDED_DIAGNOSTIC_KEY", raising=False)
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(tmp_path / "r.json")); monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "1")
    calls=[]; fail=[True]
    def fake(system,user,**kw):
        calls.append(system)
        if "Diagnostician" in system:
            body={"frontier_class":"LEARNABLE_FRONTIER","confidence":.8,"dominant_failure":"x","memory_mismatch_suspected":False,"search_budget_sufficient":True,"recommended_evidence_action":"x"}
        else:
            if fail:
                fail.clear()
                raise RuntimeError("planner failure")
            body={"bucket_modifications":{},"taskparam_ranges":VALID_TASKPARAM_RANGES,"seed_distribution":{"s":[0,1]},"stochasticity_distribution":{"x":[0,1]},"anchor_ratio":.2,"retention_constraints":["x"],"reason":"x","start_distribution":VALID_START_DISTRIBUTION}
        return {"content":json.dumps(body),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    monkeypatch.setattr(clients,"_call_qwen",fake); evidence={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}
    dc=clients._DiagnosticianClient("s","b"); pc=clients._PlannerClient("s",4,16); clients.clear_audit_events(); d=dc.complete(evidence)
    with pytest.raises(RuntimeError): pc.complete({**evidence,"diagnostician_summary":d})
    first_events = clients.drain_audit_events()
    assert len(first_events) == 1 and first_events[0]["role"] == "frontier_evidence_diagnostician"
    assert first_events[0]["paid_new"] and not first_events[0]["reused"]
    assert len(_journal_cls()(str(tmp_path / "r.json"))._load()["entries"]) == 1
    d2=dc.complete(evidence); p2=pc.complete({**evidence,"diagnostician_summary":d2}); events=clients.drain_audit_events()
    assert len(calls)==3 and d2==d and p2 and [e["role"] for e in events] == ["frontier_evidence_diagnostician","curriculum_search_planner"]
    assert events[0]["reused"] and not events[0]["paid_new"]
    assert events[1]["paid_new"] and not events[1]["reused"]


def _planner_fake_body(system):
    if "Diagnostician" in system:
        return {"frontier_class":"LEARNABLE_FRONTIER","confidence":.8,"dominant_failure":"x","memory_mismatch_suspected":False,"search_budget_sufficient":True,"recommended_evidence_action":"x"}
    return {"bucket_modifications":{},"taskparam_ranges":VALID_TASKPARAM_RANGES,"seed_distribution":{"s":[0,1]},"stochasticity_distribution":{"x":[0,1]},"anchor_ratio":.2,"retention_constraints":["x"],"reason":"x","start_distribution":VALID_START_DISTRIBUTION}


def test_diagnostician_cached_second_call_reuses(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "1"); monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(tmp_path / "d.json"))
    calls=[]; monkeypatch.setattr(clients, "_call_qwen", lambda s,u,**k: (calls.append(s) or {"content":json.dumps(_planner_fake_body(s)),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}))
    ev={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}; dc=clients._DiagnosticianClient("s","b"); clients.clear_audit_events(); d1=dc.complete(ev); d2=dc.complete(ev); events=clients.drain_audit_events()
    assert d1==d2 and len(calls)==1 and events[-1]["reused"]


def test_valid_planner_new_is_validated_and_saved(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "1"); path=tmp_path/"v.json"; monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(path))
    monkeypatch.setattr(clients, "_call_qwen", lambda s,u,**k: {"content":json.dumps(_planner_fake_body(s)),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
    ev={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}; d=clients._DiagnosticianClient("s","b").complete(ev); clients._PlannerClient("s",4,16).complete({**ev,"diagnostician_summary":d}); assert len(_journal_cls()(str(path))._load()["entries"])==2


def test_nested_action_new_planner_not_saved(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "2"); path=tmp_path/"n.json"; monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(path))
    def fake(s,u,**k):
        b=_planner_fake_body(s)
        if "Diagnostician" not in s:
            b["seed_distribution"]={"action": 1}
        return {"content":json.dumps(b),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    monkeypatch.setattr(clients,"_call_qwen",fake); ev={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}; d=clients._DiagnosticianClient("s","b").complete(ev)
    with pytest.raises(Exception): clients._PlannerClient("s",4,16).complete({**ev,"diagnostician_summary":d})
    assert len(_journal_cls()(str(path))._load()["entries"])==1


def test_toxic_cached_planner_rejected_without_reuse(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from dicode.simulator_frontier import _e3_real_llm_clients as clients
    monkeypatch.setenv("QWEN_MODEL", "m"); monkeypatch.setenv("E3_SOURCE_COMMIT", "a"); monkeypatch.setenv("E3_CANDIDATE_ID", "b"); monkeypatch.setenv("E3_SESSION_IDX", "3"); path=tmp_path/"t.json"; monkeypatch.setenv("E3_LLM_JOURNAL_PATH", str(path))
    calls=[]
    def fake(s,u,**k):
        calls.append((s,u)); return {"content":json.dumps(_planner_fake_body(s)),"requested_model":"m","returned_model":"m","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    monkeypatch.setattr(clients,"_call_qwen",fake)
    ev={"feasibility":{"state_id":"s"},"archive_summary":{"bucket_id":"b","evidence_ids":["e"]}}; d=clients._DiagnosticianClient("s","b").complete(ev); pc=clients._PlannerClient("s",4,16); pc.complete({**ev,"diagnostician_summary":d})
    payload=json.loads(path.read_text());
    for e in payload["entries"].values():
        if e.get("role")=="curriculum_search_planner":
            e["validated_output"]["bucket_modifications"]={"D00":{"action":"forbidden"}}
    import hashlib
    toxic = next(e for e in payload["entries"].values() if e.get("role")=="curriculum_search_planner")
    toxic["validated_output_hash"] = hashlib.sha256(json.dumps(toxic["validated_output"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()
    path.write_text(json.dumps(payload)); before=len(calls); clients.clear_audit_events()
    with pytest.raises(Exception, match="FORBIDDEN_ACTION_GUIDANCE_FIELD"):
        pc.complete({**ev,"diagnostician_summary":d})
    assert not any(e.get("role")=="curriculum_search_planner" and e.get("reused") for e in clients.drain_audit_events())
    assert len(calls) == before
def test_formal_disk_gate_new_child_and_escape(tmp_path, monkeypatch):
    import run_e3_formal_longrun as runner
    root = tmp_path / "data-root"
    root.mkdir()
    monkeypatch.setattr(runner, "FORMAL_RUN_ROOT", str(root))
    monkeypatch.setattr(runner.shutil, "disk_usage",
                        lambda path: type("U", (), {"free": 100 * (1024 ** 3)})())
    free, required = runner._assert_formal_disk_capacity(
        str(root / "new-run"))
    assert free >= required >= 70 * (1024 ** 3)
    with pytest.raises(RuntimeError, match="RUN_DIR_MUST_USE_DATA_DISK"):
        runner._assert_formal_disk_capacity(str(tmp_path / "escape"))


def test_formal_gpu_gate_rejects_wrong_cvd_without_query(monkeypatch):
    import run_e3_formal_longrun as runner
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        runner._assert_formal_gpu_binding()


def test_formal_gpu_gate_rejects_uuid_mismatch(monkeypatch):
    import run_e3_formal_longrun as runner
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(runner, "_gpu_uuid", lambda: "GPU-wrong")
    with pytest.raises(RuntimeError, match="UUID_MISMATCH"):
        runner._assert_formal_gpu_binding()


def test_formal_gpu_gate_rejects_cpu_fallback(monkeypatch):
    pytest.importorskip("jax")
    import run_e3_formal_longrun as runner
    import jax
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(runner, "_gpu_uuid", lambda: runner.EXPECTED_PHYSICAL_GPU_UUID)
    monkeypatch.setattr(jax, "devices", lambda: [type("CpuDevice", (), {"platform": "cpu"})()])
    with pytest.raises(RuntimeError, match="NO_GPU_JAX_DEVICE"):
        runner._assert_formal_gpu_binding()


def test_formal_gpu_gate_accepts_bound_gpu1(monkeypatch):
    pytest.importorskip("jax")
    import run_e3_formal_longrun as runner
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(runner, "_gpu_uuid", lambda: runner.EXPECTED_PHYSICAL_GPU_UUID)
    assert runner._assert_formal_gpu_binding() == runner.EXPECTED_PHYSICAL_GPU_UUID


def test_finite_gate_rejects_params_opt_and_metrics(tmp_path):
    jax = pytest.importorskip("jax")
    import run_e3_formal_longrun as runner
    good = SimpleNamespace(params=jax.numpy.ones((2,)),
                           opt_state=jax.numpy.ones((2,)))
    runner._assert_finite_training_artifacts(
        train_state=good, receipt={"training_metrics": {"task": {"loss": 1.0}},
                                   "evaluation_metrics": {"return": 0.0}})
    for state, receipt, message in (
        (SimpleNamespace(params=jax.numpy.array([jax.numpy.nan]),
                         opt_state=jax.numpy.ones((1,))), {}, "PARAMS"),
        (SimpleNamespace(params=jax.numpy.ones((1,)),
                         opt_state=jax.numpy.array([jax.numpy.inf])), {}, "OPT_STATE"),
        (good, {"training_metrics": {"task": {"loss": jax.numpy.nan}}}, "TRAINING_METRICS"),
    ):
        with pytest.raises(RuntimeError, match=message):
            runner._assert_finite_training_artifacts(train_state=state, receipt=receipt)


def _write_resume_fixture(root: Path, *, mutate=None, sessions=2):
    import run_e3_formal_longrun as runner
    (root / "evidence").mkdir(parents=True)
    (root / "runstate").mkdir(parents=True)
    previous = None
    for i in range(1, sessions + 1):
        stem = root / "runstate" / f"e3_canonical_runstate_s{i:03d}"
        state = Path(str(stem) + ".state.pkl")
        state.write_bytes(pickle.dumps({"params": {"x": float(i)},
                                       "opt_state": {"x": float(i)}}))
        sha = hashlib.sha256(state.read_bytes()).hexdigest()
        meta = {
            "schema": "simulator_frontier.canonical_runstate_checkpoint/v1",
            "codec_version": "simulator_frontier.runstate_codec/v1",
            "state_file_sha256": sha, "fields": [],
            "global_update_step": i * 100,
            "global_env_steps": i * 100 * runner.ENV_STEPS_PER_UPDATE,
            "current_session_idx": i, "plan_hash": "p",
            "runtime_bundle_hash": "r", "config_hash": "c",
            "source_commit": "source", "idempotency_token": f"e3-longrun-s{i}",
        }
        meta["checkpoint_hash"] = hashlib.sha256(
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        Path(str(stem) + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        report = {
            "schema": "simulator_frontier.e3_formal_longrun_session/v1",
            "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
            "session_idx": i, "current_session_idx": i,
            "candidate_id": "candidate", "source_commit": "source",
            "authorization_manifest_hash": "auth", "previous_checkpoint": previous,
            "num_updates_in_session": 100,
            "env_steps_per_update": runner.ENV_STEPS_PER_UPDATE,
            "start_global_update": (i - 1) * 100,
            "start_global_env_steps": (i - 1) * 100 * runner.ENV_STEPS_PER_UPDATE,
            "global_update_step": i * 100,
            "global_env_steps": i * 100 * runner.ENV_STEPS_PER_UPDATE,
            "fresh_process_restore_equivalent": True,
            "checkpoint_path": str(stem), "checkpoint_state_sha256": sha,
            "checkpoint_hash": meta["checkpoint_hash"],
            "checkpoint_content_hash": f"content-{i}",
        }
        (root / "evidence" / f"session_{i:03d}.json").write_text(
            json.dumps(report), encoding="utf-8")
        previous = str(stem)
    if mutate:
        mutate(root)


def test_resume_fixture_positive_and_chain_negative(tmp_path):
    import run_e3_formal_longrun as runner
    _write_resume_fixture(tmp_path)
    reports, start, previous = runner._load_and_validate_completed_sessions(
        str(tmp_path), candidate="candidate", source_commit="source",
        sessions=151, authorization_manifest_hash="auth")
    assert len(reports) == 2 and start == 3 and previous.endswith("s002")
    mutations = {
        "gap": lambda root: (root / "evidence" / "session_001.json").rename(root / "evidence" / "session_003.json"),
        "extra": lambda root: (root / "evidence" / "session_002.json").replace(root / "evidence" / "session_152.json"),
        "previous": lambda root: json.dump({**json.loads((root / "evidence" / "session_002.json").read_text()), "previous_checkpoint": "wrong"}, (root / "evidence" / "session_002.json").open("w")),
        "auth": lambda root: json.dump({**json.loads((root / "evidence" / "session_001.json").read_text()), "authorization_manifest_hash": "wrong"}, (root / "evidence" / "session_001.json").open("w")),
        "source": lambda root: json.dump({**json.loads((root / "evidence" / "session_001.json").read_text()), "source_commit": "wrong"}, (root / "evidence" / "session_001.json").open("w")),
        "counter": lambda root: json.dump({**json.loads((root / "evidence" / "session_001.json").read_text()), "global_update_step": 9}, (root / "evidence" / "session_001.json").open("w")),
        "state_sha": lambda root: (root / "runstate" / "e3_canonical_runstate_s001.state.pkl").write_bytes(b"tamper"),
        "meta_hash": lambda root: json.dump({**json.loads((root / "runstate" / "e3_canonical_runstate_s001.meta.json").read_text()), "checkpoint_hash": "0" * 64}, (root / "runstate" / "e3_canonical_runstate_s001.meta.json").open("w")),
        "orphan": lambda root: (root / "runstate" / "orphan.state.pkl").write_bytes(b"orphan"),
    }
    for _name, mutation in mutations.items():
        case = tmp_path / _name
        case.mkdir()
        _write_resume_fixture(case, mutate=mutation)
        with pytest.raises(ValueError):
            runner._load_and_validate_completed_sessions(
                str(case), candidate="candidate", source_commit="source",
                sessions=151, authorization_manifest_hash="auth")


def test_archive_attribute_failure_rolls_back_new_nodes(monkeypatch):
    pytest.importorskip("jax")
    import dicode.simulator_frontier.canonical_dicode_runtime as runtime
    from dicode.simulator_frontier import production_task_materializer as materializer
    Plan = runtime.CanonicalDiCodeTrainingBatchPlan
    dynamic_slots = tuple(f"plan-001::D{i:02d}" for i in range(12))
    slots = dynamic_slots + ("collecting", "combat", "crafting")
    valid = dict(VALID_TASKPARAM_RANGES)
    slot_distributions = {
        slot: ({"distribution_id": slot, "taskparam_ranges": valid,
                "evidence_hash": "e" * 64} if "::" in slot else
               {"distribution_id": slot, "evidence_hash": "e" * 64})
        for slot in slots}
    plan = Plan(
        plan_id="plan-001", curriculum_slots=slots,
        curriculum_weights={slot: 0.8 / 15 for slot in slots},
        original_task_included=True, original_task_proportion=0.2,
        curriculum_proportion_total=0.8, slot_distributions=slot_distributions,
        memory_bindings={slot: {} for slot in slots}, env_adapter_id="adapter")

    class Attrs(dict):
        def __init__(self, slot, fail_slot):
            super().__init__(); self.slot = slot; self.fail_slot = fail_slot
        def update(self, *args, **kwargs):
            if self.slot == self.fail_slot:
                raise RuntimeError("injected attribute failure")
            return super().update(*args, **kwargs)

    class Graph:
        def __init__(self): self.nodes = {}
        def has_node(self, node): return node in self.nodes
        def remove_node(self, node): self.nodes.pop(node, None)

    class Archive:
        def __init__(self, fail_record=None): self.graph = Graph(); self.fail_record = fail_record
        def record_new_task(self, child_task, parent_tasks, description, session_id):
            self.graph.nodes.setdefault(child_task, Attrs(child_task, dynamic_slots[7]))
            if child_task == self.fail_record:
                raise RuntimeError("injected record failure")

    archive = Archive()
    monkeypatch.setattr(runtime, "verify_frontier_distribution_environment_adapter", lambda adapter: None)
    monkeypatch.setattr(materializer, "require_anchor_bindings", lambda _m: [
        {"anchor_id": n, "taskparams": {}, "base_env_entrypoint": "x:y",
         "base_env_hash": "a" * 64, "world_set_ref": "w",
         "seed_policy_ref": "s", "reset_protocol": "STANDARD_RESET"}
        for n in ("collecting", "combat", "crafting")])
    monkeypatch.setattr(materializer, "resolve_taskparams", lambda *args, **kwargs: {k: 1 for k in valid})
    monkeypatch.setattr(materializer, "render_slot_env_module", lambda *args, **kwargs: ("class Env: pass\n", "c" * 64))
    monkeypatch.setattr(materializer, "canonical_sha256", lambda value: "d" * 64)
    monkeypatch.setattr(runtime, "_env_module_source", lambda *args: "class Env: pass\n")
    monkeypatch.setattr(runtime, "_import_entrypoint", lambda *args: object)
    monkeypatch.setattr(runtime, "class_source_sha256", lambda *args: "a" * 64)
    with pytest.raises(Exception):
        runtime.materialize_and_register(SimpleNamespace(env_implementation_hash="a" * 64), plan, archive,
                                          session_idx=1, anchor_manifest=object())
    assert archive.graph.nodes == {}
    existing = Archive()
    existing.graph.nodes[dynamic_slots[0]] = {"keep": "unchanged"}
    with pytest.raises(Exception):
        runtime.materialize_and_register(SimpleNamespace(env_implementation_hash="a" * 64), plan, existing,
                                          session_idx=1, anchor_manifest=object())
    assert existing.graph.nodes == {dynamic_slots[0]: {"keep": "unchanged"}}
    failed_record = Archive(fail_record=dynamic_slots[7])
    with pytest.raises(Exception):
        runtime.materialize_and_register(SimpleNamespace(env_implementation_hash="a" * 64), plan, failed_record,
                                          session_idx=1, anchor_manifest=object())
    assert failed_record.graph.nodes == {}
