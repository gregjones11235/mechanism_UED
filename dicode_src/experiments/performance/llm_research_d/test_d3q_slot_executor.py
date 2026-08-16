"""Local pytest for the D3Q Phase 1 slot executor (launcher + remote runner).

Coverage required by the phase contract:

* payload exactness for both arms (per-field equality + forbidden-field
  absence, including the ``thinking`` / ``extra_body`` / ``chat_template_kwargs``
  / ``enable_thinking`` / ``reasoning_effort`` checks);
* shared budget integration: slot 4th POST and provider 109th POST are
  rejected on a real ledger file; resume rebuilds identical state;
* frozen repair-template assembly: template bytes appear verbatim and the
  template sha256 stays ``beff6ea4...``;
* no-secret filter intercepts ``sk-...`` / ``Authorization: Bearer`` text;
* remote-runner offline unit logic: extract_code and static_lint
  classification over local samples (injected enum map, no craftax needed).

Run from the module directory:  python -m pytest test_d3q_slot_executor.py
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import d3q_budget
import d3q_slot_runner as runner
import d3q_slot_launcher as launcher

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "FROZEN_MANIFEST.json"
PHASE0 = HERE / "d3q_phase0_reconciliation_20260815T011126Z"
REPAIR_TEMPLATE = PHASE0 / "D3Q_FROZEN_REPAIR_TEMPLATE.json"


# ---------------------------------------------------------------------------
# Payload exactness.
# ---------------------------------------------------------------------------


def _manifest_prompts(prompt_index: int = 0):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data["system_prompt"], data["user_prompts"][prompt_index]


def test_large_payload_exact_shape():
    system, user = _manifest_prompts()
    payload = runner.build_payload("large", system, user)
    assert set(payload) == {
        "model", "messages", "temperature", "top_p", "max_tokens", "thinking"
    }
    assert payload == {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    for forbidden in ("extra_body", "chat_template_kwargs", "enable_thinking", "reasoning_effort"):
        assert forbidden not in payload


def test_small_payload_exact_shape():
    system, user = _manifest_prompts()
    payload = runner.build_payload("small", system, user)
    assert set(payload) == {"model", "messages", "temperature", "top_p", "max_tokens"}
    assert payload == {
        "model": "qwen2.5-coder:14b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
    }
    for forbidden in ("extra_body", "chat_template_kwargs", "enable_thinking", "reasoning_effort", "thinking"):
        assert forbidden not in payload


def test_payload_messages_are_manifest_bytes():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    system, user = _manifest_prompts(0)
    assert data["request_order"][0] == {"index": 0, "kind": "code", "slot": "task_1"}
    for arm in ("small", "large"):
        payload = runner.build_payload(arm, system, user)
        assert payload["messages"][0]["content"] == data["system_prompt"]
        assert payload["messages"][1]["content"] == data["user_prompts"][0]
    # byte-exact round trip
    payload = runner.build_payload("large", system, user)
    body = runner.payload_to_bytes(payload)
    assert json.loads(body.decode("utf-8")) == payload


def test_payload_serialization_deterministic():
    system, user = _manifest_prompts()
    a = runner.payload_to_bytes(runner.build_payload("small", system, user))
    b = runner.payload_to_bytes(runner.build_payload("small", system, user))
    assert a == b


def test_frozen_manifest_hash_binding():
    data = runner.load_frozen_manifest(MANIFEST)
    assert data["manifest_sha256"] == runner.FROZEN_MANIFEST_SHA256
    assert len(data["user_prompts"]) == 12
    assert len(data["request_order"]) == 12


def test_slot_id_and_arm_parsing():
    assert runner.parse_slot_id("slot_r1_small_p00") == ("1", "small", 0)
    assert runner.parse_slot_id("slot_r2_large_p11") == ("2", "large", 11)
    with pytest.raises(ValueError):
        runner.parse_slot_id("slot_r4_small_p00")
    with pytest.raises(ValueError):
        runner.parse_slot_id("bogus")
    assert runner.arm_to_provider_model("small") == ("ollama", "qwen2.5-coder:14b",
                                                     "http://127.0.0.1:11434/v1")
    assert runner.arm_to_provider_model("large") == ("deepseek_official", "deepseek-v4-flash",
                                                     "https://api.deepseek.com")


# ---------------------------------------------------------------------------
# Shared budget integration on a real ledger file.
# ---------------------------------------------------------------------------


def _event(**overrides):
    base = {
        "ts_utc": "2026-08-15T01:00:00Z",
        "slot_id": "slot_r1_small_p00",
        "model": "qwen2.5-coder:14b",
        "provider": "ollama",
        "kind": "initial",
        "attempt_index": 1,
    }
    base.update(overrides)
    return base


def test_slot_fourth_post_rejected_on_ledger(tmp_path):
    ledger = d3q_budget.D3QLedger(tmp_path / "ledger.jsonl")
    for i, kind in enumerate(("initial", "transport_retry", "semantic_repair")):
        ledger.reserve(**_event(kind=kind, attempt_index=i + 1))
    assert ledger.slot_post_count("slot_r1_small_p00") == 3
    with pytest.raises(d3q_budget.BudgetExceededError):
        ledger.reserve(**_event(kind="initial", attempt_index=4))
    with open(tmp_path / "ledger.jsonl", encoding="utf-8") as handle:
        assert len(handle.read().splitlines()) == 3


def test_provider_109th_post_rejected_on_ledger(tmp_path):
    ledger = d3q_budget.D3QLedger(tmp_path / "ledger.jsonl")
    for i in range(108):
        ledger.reserve(
            **_event(slot_id=f"slot_b{i:02d}", provider="deepseek_official",
                    model="deepseek-v4-flash", kind="initial", attempt_index=1)
        )
    assert ledger.provider_post_count("deepseek_official") == 108
    before = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    with pytest.raises(d3q_budget.BudgetExceededError):
        ledger.reserve(
            **_event(slot_id="slot_fresh", provider="deepseek_official",
                    model="deepseek-v4-flash", kind="initial", attempt_index=1)
        )
    assert (tmp_path / "ledger.jsonl").read_text(encoding="utf-8") == before


def test_ledger_resume_consistent(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = d3q_budget.D3QLedger(path)
    ledger.reserve(**_event(kind="initial", attempt_index=1))
    ledger.reserve(**_event(kind="transport_retry", attempt_index=2))
    ledger.reserve(**_event(kind="semantic_repair", attempt_index=3))
    resumed = d3q_budget.D3QLedger(path)
    assert resumed.slot_post_count("slot_r1_small_p00") == 3
    assert resumed.provider_post_count("ollama") == 3
    with pytest.raises(d3q_budget.BudgetExceededError):
        resumed.reserve(**_event(kind="initial", attempt_index=4))
    assert resumed.slot_post_count("slot_r1_small_p00") == 3


def test_fs_ledger_summary_no_double_load(tmp_path):
    # D3QLedger.__init__ auto-loads an existing ledger; a second explicit
    # load() would double-apply events.  fs_ledger_summary must report the
    # rebuilt counts exactly once.
    path = tmp_path / "ledger.jsonl"
    ledger = d3q_budget.D3QLedger(path)
    ledger.reserve(**_event(kind="initial", attempt_index=1))
    summary = runner.fs_ledger_summary(str(path))
    assert summary["exists"] is True
    assert summary["slot_counts"] == {"slot_r1_small_p00": 1}
    assert summary["provider_counts"] == {"ollama": 1}
    resumed = d3q_budget.D3QLedger(path)
    event = resumed.reserve(**_event(kind="transport_retry", attempt_index=2))
    assert event["post_index_in_slot"] == 2
    assert event["post_index_for_provider"] == 2
    summary = runner.fs_ledger_summary(str(path))
    assert summary["slot_counts"] == {"slot_r1_small_p00": 2}
    assert summary["provider_counts"] == {"ollama": 2}


def test_over_budget_ledger_fails_closed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(4):
            event = _event(kind="initial", attempt_index=i + 1)
            event["post_index_in_slot"] = i + 1
            event["post_index_for_provider"] = i + 1
            handle.write(json.dumps(event) + "\n")
    with pytest.raises(d3q_budget.BudgetExceededError):
        d3q_budget.D3QLedger(path)


# ---------------------------------------------------------------------------
# Repair template assembly.
# ---------------------------------------------------------------------------


def test_repair_template_sha256_frozen():
    data = json.loads(REPAIR_TEMPLATE.read_text(encoding="utf-8"))
    assert data["classification"] == "D3Q_FROZEN_REPAIR_TEMPLATE"
    assert data["template_sha256"] == runner.FROZEN_REPAIR_TEMPLATE_SHA256
    assert runner.sha256_text(data["template_text"]) == runner.FROZEN_REPAIR_TEMPLATE_SHA256


def test_repair_assembly_uses_template_bytes_verbatim():
    data = json.loads(REPAIR_TEMPLATE.read_text(encoding="utf-8"))
    template_text = data["template_text"]
    prompt = runner.assemble_repair_user_prompt(template_text, "code here", "boom")
    assert prompt.startswith(template_text)
    assert "code here" in prompt
    assert "boom" in prompt


def test_repair_assembly_sanitizes_secrets():
    data = json.loads(REPAIR_TEMPLATE.read_text(encoding="utf-8"))
    template_text = data["template_text"]
    dirty_error = "line 3: Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz01234567 failed"
    prompt = runner.assemble_repair_user_prompt(template_text, "code", dirty_error)
    assert not runner.contains_secret(prompt)
    assert "sk-abcdefghijklmnopqrstuvwxyz01234567" not in prompt
    assert "[REDACTED]" in prompt


# ---------------------------------------------------------------------------
# No-secret filter.
# ---------------------------------------------------------------------------


def test_no_secret_filter_intercepts_secret_shapes():
    dirty_samples = [
        "api_key=sk-abcdefghijklmnopqrstuvwxyz0123456789abcdef",
        "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz0123456789abcdef",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
        '{"header": {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}}',
    ]
    for sample in dirty_samples:
        assert runner.contains_secret(sample), f"missed: {sample[:60]}"


def test_no_secret_filter_passes_clean_text():
    clean_samples = [
        "temperature=0.6 top_p=0.95 max_tokens=8192",
        "Authorization header uses Bearer ollama placeholder",
        "sk- is not a token here",
        "private key material is never committed",
    ]
    for sample in clean_samples:
        assert not runner.contains_secret(sample), f"false positive: {sample[:60]}"

# ---------------------------------------------------------------------------
# extract_code / static_lint classification (offline, injected enum map).
# ---------------------------------------------------------------------------

FAKE_ENUM_MEMBERS = {
    "BlockType": {"STONE", "COAL", "WOOD"},
    "Achievement": {"COLLECT_COAL", "COLLECT_STONE"},
}
FAKE_INVENTORY_FIELDS = {"wood", "coal", "stone", "iron", "diamonds"}


def test_extract_code_classification():
    assert runner.extract_code("<code>def f():\n    return 1\n</code>") == "def f():\n    return 1"
    assert runner.extract_code("```python\nx = 1\n```") == "x = 1"
    assert runner.extract_code("plain code without tags") == "plain code without tags"
    assert runner.extract_code("") is None
    assert runner.extract_code(None) is None
    assert runner.extract_code("  ") == ""


def test_static_lint_valid_code_passes():
    code = (
        "from craftax.craftax.constants import BlockType\n"
        "from craftax.craftax.craftax_state import Inventory\n"
        "import jax.numpy as jnp\n"
        "x = BlockType.COAL\n"
        "inv = Inventory(wood=1)\n"
    )
    error_class, message = runner.static_lint(
        code, enum_members=FAKE_ENUM_MEMBERS, inventory_fields=FAKE_INVENTORY_FIELDS
    )
    assert error_class == "" and message == ""


def test_static_lint_syntax_error_classified():
    error_class, message = runner.static_lint("def foo(:", enum_members=FAKE_ENUM_MEMBERS,
                                              inventory_fields=FAKE_INVENTORY_FIELDS)
    assert error_class == "syntax_error"
    assert message.startswith("Compilation error:")


def test_static_lint_enum_error_classified():
    code = (
        "from craftax.craftax.constants import BlockType\n"
        "x = BlockType.NONEXISTENT\n"
    )
    error_class, message = runner.static_lint(code, enum_members=FAKE_ENUM_MEMBERS,
                                              inventory_fields=FAKE_INVENTORY_FIELDS)
    assert error_class == "api_enum_error"
    assert "invalid BlockType member NONEXISTENT" in message


def test_static_lint_inventory_error_classified():
    code = (
        "from craftax.craftax.craftax_state import Inventory\n"
        "inv = Inventory(bogus_kwarg=1)\n"
    )
    error_class, message = runner.static_lint(code, enum_members=FAKE_ENUM_MEMBERS,
                                              inventory_fields=FAKE_INVENTORY_FIELDS)
    assert error_class == "inventory_error"
    assert "invalid Inventory kwarg bogus_kwarg" in message


def test_static_lint_forbidden_import_classified():
    code = "import os\nx = os.getcwd()\n"
    error_class, message = runner.static_lint(code, enum_members=FAKE_ENUM_MEMBERS,
                                              inventory_fields=FAKE_INVENTORY_FIELDS)
    assert error_class == "dangerous_import"
    assert "forbidden import: os" in message
    code2 = "from subprocess import run\n"
    error_class2, _ = runner.static_lint(code2, enum_members=FAKE_ENUM_MEMBERS,
                                         inventory_fields=FAKE_INVENTORY_FIELDS)
    assert error_class2 == "dangerous_import"


def test_static_lint_dangerous_builtin_classified():
    code = "data = open('secret.txt').read()\n"
    error_class, message = runner.static_lint(code, enum_members=FAKE_ENUM_MEMBERS,
                                              inventory_fields=FAKE_INVENTORY_FIELDS)
    assert error_class == "dangerous_capability"
    assert "dangerous builtin call: open" in message


def test_static_lint_uses_frozen_maps_without_craftax_import(monkeypatch):
    # The remote runner must not import craftax/jax in-process; static lint
    # uses frozen maps instead, so blocking craftax imports must not matter.
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "craftax" or name.startswith("craftax."):
            raise ImportError("craftax unavailable for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    code = (
        "from craftax.craftax.constants import BlockType\n"
        "from craftax.craftax.craftax_state import Inventory\n"
        "import jax.numpy as jnp\n"
        "x = BlockType.COAL\n"
        "inv = Inventory(wood=1)\n"
    )
    error_class, message = runner.static_lint(code)
    assert error_class == "" and message == ""
    bad = (
        "from craftax.craftax.constants import BlockType\n"
        "x = BlockType.NONEXISTENT\n"
    )
    error_class, message = runner.static_lint(bad)
    assert error_class == "api_enum_error"


def test_frozen_craftax_maps_match_extraction():
    # Maps extracted from craftax 1.4.5 (remote venv skill_preflight_e0e1);
    # the recorded source-file sha256 pins the extraction artifact.
    assert runner.CRAFTAX_MAP_SOURCE_SHA256 == (
        "b37c1b654483b4139b2643f609cec21afc57f8570baa164bccf31aaccdabf994"
    )
    assert len(runner.CRAFTAX_BLOCK_TYPES) == 37
    assert len(runner.CRAFTAX_ACHIEVEMENTS) == 67
    assert len(runner.CRAFTAX_INVENTORY_FIELDS) == 16
    assert "STONE" in runner.CRAFTAX_BLOCK_TYPES
    assert "WAKE_UP" in runner.CRAFTAX_ACHIEVEMENTS
    assert "wood" in runner.CRAFTAX_INVENTORY_FIELDS


# ---------------------------------------------------------------------------
# classify_exception.
# ---------------------------------------------------------------------------


def test_classify_exception():
    import socket
    assert runner.classify_exception(TimeoutError()) == "timeout"
    assert runner.classify_exception(socket.timeout()) == "timeout"
    assert runner.classify_exception(ConnectionRefusedError()) == "connection_error"
    assert runner.classify_exception(RuntimeError()) == "unknown_error"


# ---------------------------------------------------------------------------
# post_chat_completion against a real local HTTP server.
# ---------------------------------------------------------------------------

SERVER_STATE = {"mode": "ok"}


class _FakeChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        mode = SERVER_STATE["mode"]
        self.send_response(SERVER_STATE["status"])
        if mode in ("ok", "invalid_json"):
            self.send_header("x-request-id", "req-test-123")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if mode == "ok":
            body = {
                "id": "chatcmpl-test",
                "model": "deepseek-v4-flash",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                          "prompt_cache_hit_tokens": 7},
            }
            self.wfile.write(json.dumps(body).encode("utf-8"))
        elif mode == "empty":
            pass
        elif mode == "invalid_json":
            self.wfile.write(b"not-json")
        elif mode in ("4xx", "5xx"):
            self.wfile.write(json.dumps({"error": {"message": "boom"}}).encode("utf-8"))

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_post_chat_completion_ok(fake_server):
    url = f"http://127.0.0.1:{fake_server.server_port}/v1/chat/completions"
    SERVER_STATE.update({"mode": "ok", "status": 200})
    result = runner.post_chat_completion(url, b"{}", {"Content-Type": "application/json"})
    assert result["error_class"] is None
    assert result["http_status"] == 200
    assert result["request_id"] == "req-test-123"
    assert result["decoded"]["model"] == "deepseek-v4-flash"
    assert result["finish_reason"] == "stop"
    usage = runner.extract_usage(result["decoded"])
    assert usage == {"prompt_tokens": 10, "completion_tokens": 20, "cached_tokens": 7}


def test_post_chat_completion_empty_response(fake_server):
    url = f"http://127.0.0.1:{fake_server.server_port}/v1/chat/completions"
    SERVER_STATE.update({"mode": "empty", "status": 200})
    result = runner.post_chat_completion(url, b"{}", {})
    assert result["error_class"] == "empty_response"
    assert result["http_status"] == 200


def test_post_chat_completion_invalid_json(fake_server):
    url = f"http://127.0.0.1:{fake_server.server_port}/v1/chat/completions"
    SERVER_STATE.update({"mode": "invalid_json", "status": 200})
    result = runner.post_chat_completion(url, b"{}", {})
    assert result["error_class"] == "invalid_json"


def test_post_chat_completion_http_error_classes(fake_server):
    url = f"http://127.0.0.1:{fake_server.server_port}/v1/chat/completions"
    SERVER_STATE.update({"mode": "4xx", "status": 404})
    result = runner.post_chat_completion(url, b"{}", {})
    assert result["error_class"] == "http_4xx"
    assert result["http_status"] == 404
    SERVER_STATE.update({"mode": "5xx", "status": 503})
    result = runner.post_chat_completion(url, b"{}", {})
    assert result["error_class"] == "http_5xx"
    assert result["http_status"] == 503


def test_post_chat_completion_connection_refused():
    result = runner.post_chat_completion("http://127.0.0.1:1/v1/chat/completions", b"{}", {})
    assert result["error_class"] == "connection_error"
    assert result["http_status"] == 0


# ---------------------------------------------------------------------------
# Dotenv parsing semantics (mirror d3_deepseek_provider).
# ---------------------------------------------------------------------------


def test_parse_env_file(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        "# comment\n"
        "EXP_DEEPSEEK_API_KEY=sk-abc\n"
        "EMPTY=\n"
        "QUOTED=\"hello world\"\n",
        encoding="utf-8",
    )
    values = runner.parse_env_file(path)
    assert values["EXP_DEEPSEEK_API_KEY"] == "sk-abc"
    assert values["EMPTY"] == ""
    assert values["QUOTED"] == "hello world"


def test_parse_env_file_rejects_export_and_dup(tmp_path):
    path = tmp_path / "env"
    path.write_text("export A=1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.parse_env_file(path)
    path.write_text("A=1\nA=2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.parse_env_file(path)


# ---------------------------------------------------------------------------
# Slot result schema validation (launcher).
# ---------------------------------------------------------------------------


def _slot_result_fixture():
    return {
        "classification": "D3Q_SLOT_RESULT",
        "schema_version": 1,
        "run_id": "d3q_p1_smoke_test",
        "slot_id": "slot_r1_small_p00",
        "arm": "small",
        "repeat": "1",
        "prompt_index": 0,
        "prompt_slot": "task_1",
        "provider": "ollama",
        "model": "qwen2.5-coder:14b",
        "initial_valid": True,
        "final_valid": True,
        "attempts": 1,
        "repair_requests": 0,
        "repair_success": 0,
        "empty_response": 0,
        "timeout": 0,
        "connection_error": 0,
        "http_4xx": 0,
        "http_5xx": 0,
        "invalid_json": 0,
        "extract_error": 0,
        "syntax_error": 0,
        "api_enum_error": 0,
        "cpu_jax_error": 0,
        "duplicate_code": 0,
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "cached_tokens": 0,
        "generation_wall_s": 1.5,
        "repair_wall_s": 0.0,
        "cpu_validation_wall_s": 30.0,
        "final_code_sha256": "a" * 64,
        "fatal_api_blocked": False,
        "attempts_detail": [],
        "ledger_counts": {"slot": 1, "provider": 1},
    }


def test_launcher_slot_result_schema_pass():
    launcher.verify_slot_result_schema(_slot_result_fixture())


def test_launcher_slot_result_schema_rejects_missing_field():
    result = _slot_result_fixture()
    del result["cpu_validation_wall_s"]
    with pytest.raises(launcher.LauncherError):
        launcher.verify_slot_result_schema(result)


def test_launcher_slot_result_schema_rejects_bad_counts():
    result = _slot_result_fixture()
    result["attempts"] = 4
    with pytest.raises(launcher.LauncherError):
        launcher.verify_slot_result_schema(result)
    result = _slot_result_fixture()
    result["repair_requests"] = 3
    with pytest.raises(launcher.LauncherError):
        launcher.verify_slot_result_schema(result)


# ---------------------------------------------------------------------------
# Budget enforcement helpers used by the launcher.
# ---------------------------------------------------------------------------


def test_launcher_budget_enforcement_before_slot():
    summary = {"slot_counts": {"slot_r1_small_p00": 3}, "provider_counts": {"ollama": 5}}
    with pytest.raises(launcher.LauncherError):
        launcher._enforce_budget_before_slot(summary, "slot_r1_small_p00", "ollama")
    summary = {"slot_counts": {}, "provider_counts": {"ollama": 108}}
    with pytest.raises(launcher.LauncherError):
        launcher._enforce_budget_before_slot(summary, "slot_r1_small_p00", "ollama")
    summary = {"slot_counts": {}, "provider_counts": {"ollama": 107}}
    launcher._enforce_budget_before_slot(summary, "slot_r1_small_p00", "ollama")


def test_launcher_no_secret_scan_directory(tmp_path):
    (tmp_path / "clean.json").write_text('{"temperature": 0.6}', encoding="utf-8")
    (tmp_path / "dirty.txt").write_text(
        "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz0123456789abcdef",
        encoding="utf-8",
    )
    scan = launcher._no_secret_scan_directory(tmp_path)
    assert not scan["passed"]
    assert any("dirty.txt" in item for item in scan["violations"])
    (tmp_path / "dirty.txt").unlink()
    scan = launcher._no_secret_scan_directory(tmp_path)
    assert scan["passed"]
    assert scan["files_scanned"] == 1


# ---------------------------------------------------------------------------
# Incident D3Q_PHASE2_INCIDENT_01 regressions:
# a slot legally exhausting its 3-POST budget must PASS the run, and a
# mid-run failure must preserve already-collected slot evidence.
# ---------------------------------------------------------------------------


def test_launcher_budget_after_slot_exact_three_is_legal():
    summary = {"slot_counts": {"slot_r1_small_p02": 3}, "provider_counts": {"ollama": 7}}
    launcher._enforce_budget_after_slot(
        summary, "slot_r1_small_p02", "ollama", expected_slot_posts=3
    )
    summary = {"slot_counts": {"slot_r1_small_p02": 4}, "provider_counts": {"ollama": 8}}
    with pytest.raises(launcher.LauncherError):
        launcher._enforce_budget_after_slot(
            summary, "slot_r1_small_p02", "ollama", expected_slot_posts=4
        )


def _fake_gpu_snapshot(command_runner, target, ssh_key):
    return {
        "gpu2": {"uuid": launcher.EXPECTED_GPU2_UUID, "external_compute_pids": []},
        "gpus": [],
    }


def _fake_ollama_snapshot(command_runner, target, ssh_key, python_path):
    return {
        "models": {runner.SMALL_MODEL: launcher.EXPECTED_OLLAMA_QWEN_DIGEST_PREFIX + "ffff"},
        "pids": ["1"],
    }


def _patch_launcher_remotes(monkeypatch, summary_fn, dispatch_fn, collect_log, result_fn):
    monkeypatch.setattr(launcher, "_gpu_snapshot", _fake_gpu_snapshot)
    monkeypatch.setattr(launcher, "_ollama_snapshot", _fake_ollama_snapshot)
    monkeypatch.setattr(launcher, "_remote_exec_root", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_deploy", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_verify_remote_hashes", lambda *a, **k: {"binding": "x"})
    monkeypatch.setattr(launcher, "_cleanup_exec_root", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_remote_ledger_summary", summary_fn)
    monkeypatch.setattr(launcher, "_dispatch_slot", dispatch_fn)

    def fake_collect(command_runner, target, ssh_key, exec_root, slot_id, local_slots_dir):
        slot_dir = Path(local_slots_dir) / slot_id
        slot_dir.mkdir(parents=True, exist_ok=True)
        collect_log.append(slot_id)

    monkeypatch.setattr(launcher, "_collect_slot_dir", fake_collect)
    monkeypatch.setattr(launcher, "_load_slot_result", result_fn)


def _run_launcher_kwargs(tmp_path, slots):
    return dict(
        slots=slots,
        ssh_target="oseasy@172.25.14.221",
        ssh_key=str(tmp_path / "fake_key"),
        remote_python="/home/x/bin/python",
        remote_env_file="/home/x/env.env",
        mason_worktree="/home/x/wt",
        mason_src="/home/x/wt/src",
        artifacts_dir=tmp_path / "artifacts",
        runner=lambda invocation: launcher.CommandResult(0),
    )


def test_launcher_exact_budget_slot_passes_end_to_end(tmp_path, monkeypatch):
    counts = {"slot_r1_small_p01": 0, "slot_r1_small_p02": 0}
    provider = {"ollama": 0}

    def summary_fn(command_runner, target, ssh_key, python_path, exec_root):
        return {"slot_counts": dict(counts), "provider_counts": dict(provider)}

    def dispatch_fn(command_runner, target, ssh_key, **kwargs):
        slot_id = kwargs["slot_id"]
        posts = 3 if slot_id == "slot_r1_small_p02" else 1
        counts[slot_id] = posts
        provider["ollama"] += posts
        return {"slot_id": slot_id}

    collect_log = []

    def result_fn(slot_dir, slot_id):
        return {
            "slot_id": slot_id,
            "initial_valid": True,
            "final_valid": True,
            "attempts": 1,
            "repair_requests": 0,
        }

    _patch_launcher_remotes(monkeypatch, summary_fn, dispatch_fn, collect_log, result_fn)
    result = launcher.run_launcher(
        run_id="d3q_fake_exact_budget",
        **_run_launcher_kwargs(tmp_path, ["slot_r1_small_p01", "slot_r1_small_p02"]),
    )
    assert result["status"] == "PASS", result.get("reason")
    assert collect_log == ["slot_r1_small_p01", "slot_r1_small_p02"]
    assert result["ledger_post"]["slot_counts"]["slot_r1_small_p02"] == 3


def test_launcher_midrun_failure_preserves_collected_evidence(tmp_path, monkeypatch):
    counts = {"slot_r1_small_p01": 0, "slot_r1_small_p02": 0}
    provider = {"ollama": 0}

    def summary_fn(command_runner, target, ssh_key, python_path, exec_root):
        return {"slot_counts": dict(counts), "provider_counts": dict(provider)}

    def dispatch_fn(command_runner, target, ssh_key, **kwargs):
        slot_id = kwargs["slot_id"]
        if slot_id == "slot_r1_small_p02":
            raise launcher.LauncherError("remote_command_failed")
        counts[slot_id] = 1
        provider["ollama"] += 1
        return {"slot_id": slot_id}

    collect_log = []

    def result_fn(slot_dir, slot_id):
        return {
            "slot_id": slot_id,
            "initial_valid": True,
            "final_valid": True,
            "attempts": 1,
            "repair_requests": 0,
        }

    _patch_launcher_remotes(monkeypatch, summary_fn, dispatch_fn, collect_log, result_fn)
    result = launcher.run_launcher(
        run_id="d3q_fake_midrun_failure",
        **_run_launcher_kwargs(tmp_path, ["slot_r1_small_p01", "slot_r1_small_p02"]),
    )
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "remote_command_failed"
    assert collect_log == ["slot_r1_small_p01"]
    published = tmp_path / "artifacts" / "d3q_fake_midrun_failure"
    assert (published / "slots" / "slot_r1_small_p01").is_dir()
    assert not (published / "slots" / "slot_r1_small_p02").exists()
    published_result = json.loads(
        (published / "D3Q_SLOT_LAUNCHER_RESULT.json").read_text(encoding="utf-8")
    )
    assert "slot_r1_small_p01" in published_result["slot_results"]
