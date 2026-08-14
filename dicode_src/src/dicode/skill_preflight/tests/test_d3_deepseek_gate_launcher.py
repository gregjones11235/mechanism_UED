"""Offline command-transcript tests for the external DeepSeek gate launcher."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "performance"
    / "llm_research_d"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import d3_deepseek_gate_launcher as launcher  # noqa: E402


FIXED_NOW = datetime(2026, 8, 14, 6, 7, 8, tzinfo=timezone.utc)
FIXED_TOKEN = "abcdef123456"
SSH_TARGET = "research@example.test"
SSH_KEY = Path("C:/declared/keys/d3_ed25519")
REMOTE_PYTHON = "/usr/bin/python3"
REMOTE_ENV = "/home/research/.config/dicode/experiment_llm.env"


class _Response:
    status = 200

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _secret() -> str:
    return "credential-" + uuid.uuid4().hex + uuid.uuid4().hex + uuid.uuid4().hex


def _passing_artifact(tmp_path: Path, secret: str) -> dict:
    env = tmp_path / "fake_remote.env"
    env.write_text(
        "EXP_DEEPSEEK_PROVIDER=deepseek\n"
        "EXP_DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
        "EXP_DEEPSEEK_MODEL=deepseek-v4-flash\n"
        f"EXP_DEEPSEEK_API_KEY={secret}\n",
        encoding="utf-8",
    )
    return launcher.gate.run_metadata_gate(
        env,
        urlopen=lambda *args, **kwargs: _Response(
            {"data": [{"id": "deepseek-v4-flash"}]}
        ),
        now=FIXED_NOW,
    )


def _artifact_bytes(artifact: dict) -> bytes:
    return (
        json.dumps(
            launcher.gate.canonical(artifact),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _manifest() -> dict[str, str]:
    return {
        "tool": hashlib.sha256(Path(launcher.gate.__file__).read_bytes()).hexdigest(),
        "provider": hashlib.sha256(
            Path(launcher.gate.provider.__file__).read_bytes()
        ).hexdigest(),
    }


class _FakeRemote:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.artifact_bytes = _artifact_bytes(artifact)
        self.download_bytes = self.artifact_bytes
        self.remote_artifact_sha = hashlib.sha256(self.download_bytes).hexdigest()
        self.manifest = _manifest()
        self.pre_hashes = dict(self.manifest)
        self.post_hashes = dict(self.manifest)
        self.gpu_uuid = launcher.EXPECTED_GPU2_UUID
        self.gpu_free_mib = 8192
        self.apps = ""
        self.gpu_returncode = 0
        self.root_exists = False
        self.gate_stdout = json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
        self.gate_stderr = ""
        self.calls: list[list[str]] = []
        self._hash_calls = 0

    def __call__(self, argv):
        call = list(argv)
        self.calls.append(call)
        if call[0] == "scp":
            source, destination = call[-2:]
            if source.startswith(SSH_TARGET + ":") and source.endswith(
                launcher.REMOTE_ARTIFACT_NAME
            ):
                Path(destination).write_bytes(self.download_bytes)
            return launcher.CommandResult(0)

        remote = call[-1]
        if "EXISTS" in remote and "ABSENT" in remote:
            state = "EXISTS" if self.root_exists else "ABSENT"
            return launcher.CommandResult(0, state + "\n")
        if "mkdir(mode=0o700)" in remote:
            self.root_exists = True
            return launcher.CommandResult(0, "CREATED\n")
        if "shutil.rmtree" in remote:
            self.root_exists = False
            return launcher.CommandResult(0, "REMOVED\n")
        if "--query-gpu=index,uuid,memory.free" in remote:
            stdout = f"2, {self.gpu_uuid}, {self.gpu_free_mib}\n"
            return launcher.CommandResult(self.gpu_returncode, stdout)
        if "--query-compute-apps=gpu_uuid,pid" in remote:
            return launcher.CommandResult(self.gpu_returncode, self.apps)
        if "--env-file" in remote:
            return launcher.CommandResult(0, self.gate_stdout, self.gate_stderr)
        if "hashlib.sha256" in remote and launcher.REMOTE_PROVIDER_NAME in remote:
            value = self.pre_hashes if self._hash_calls == 0 else self.post_hashes
            self._hash_calls += 1
            return launcher.CommandResult(
                0, json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            )
        if "hashlib.sha256" in remote:
            return launcher.CommandResult(0, self.remote_artifact_sha + "\n")
        raise AssertionError(f"unexpected fake argv: {call!r}")


def _run(tmp_path: Path, fake: _FakeRemote) -> dict:
    return launcher.run_launcher(
        ssh_target=SSH_TARGET,
        ssh_key=SSH_KEY,
        remote_python=REMOTE_PYTHON,
        remote_env_file=REMOTE_ENV,
        local_output_dir=tmp_path / "output",
        runner=fake,
        now=FIXED_NOW,
        token=FIXED_TOKEN,
    )


def _gate_calls(fake: _FakeRemote) -> list[list[str]]:
    return [call for call in fake.calls if call[0] == "ssh" and "--env-file" in call[-1]]


def test_success_exact_argv_sequence_hashes_cleanup_and_redaction(tmp_path):
    secret = _secret()
    fake = _FakeRemote(_passing_artifact(tmp_path, secret))
    result = _run(tmp_path, fake)

    assert result["status"] == "PASS"
    assert result["reason"] is None
    assert result["model"] == "deepseek-v4-flash"
    assert result["base_url"] == "https://api.deepseek.com"
    assert result["credential_variable"] == "EXP_DEEPSEEK_API_KEY"
    assert result["manifest_sha256"] == fake.manifest
    assert result["pre_execution_sha256"] == fake.manifest
    assert result["post_execution_sha256"] == fake.manifest
    assert result["remote_artifact_sha256"] == fake.remote_artifact_sha
    assert result["local_artifact_sha256"] == fake.remote_artifact_sha
    assert result["artifact_request_count"] == 1
    assert result["completion_requests"] == 0
    assert result["embedding_requests"] == 0
    assert result["cleanup_verified"] is True
    assert fake.root_exists is False
    assert len(fake.calls) == 15
    assert [call[0] for call in fake.calls[:5]] == ["ssh", "ssh", "scp", "scp", "ssh"]
    assert [call[0] for call in fake.calls[-4:]] == ["ssh", "scp", "ssh", "ssh"]
    assert "EXISTS" in fake.calls[0][-1] and "ABSENT" in fake.calls[0][-1]
    assert "mkdir(mode=0o700)" in fake.calls[1][-1]
    assert fake.calls[2][-2].endswith(launcher.REMOTE_PROVIDER_NAME)
    assert fake.calls[3][-2].endswith(launcher.REMOTE_GATE_NAME)
    assert "hashlib.sha256" in fake.calls[4][-1]
    assert "--query-gpu=index,uuid,memory.free" in fake.calls[5][-1]
    assert "--query-compute-apps=gpu_uuid,pid" in fake.calls[6][-1]
    assert "--env-file" in fake.calls[7][-1]
    assert "--query-gpu=index,uuid,memory.free" in fake.calls[8][-1]
    assert "--query-compute-apps=gpu_uuid,pid" in fake.calls[9][-1]
    assert "hashlib.sha256" in fake.calls[10][-1]
    assert "hashlib.sha256" in fake.calls[11][-1]
    assert fake.calls[12][-2].endswith(launcher.REMOTE_ARTIFACT_NAME)
    assert "shutil.rmtree" in fake.calls[13][-1]
    assert "EXISTS" in fake.calls[14][-1] and "ABSENT" in fake.calls[14][-1]
    assert all(isinstance(call, list) for call in fake.calls)
    assert all("shell=True" not in item for call in fake.calls for item in call)
    assert len(_gate_calls(fake)) == 1
    gate_argv = _gate_calls(fake)[0]
    assert "--model" not in gate_argv[-1]
    assert "--base-url" not in gate_argv[-1]
    runner_source = inspect.getsource(launcher._default_runner)
    assert "shell=False" in runner_source
    assert "shell=True" not in runner_source
    flattened = json.dumps(fake.calls) + json.dumps(result, sort_keys=True)
    assert secret not in flattened
    assert "Bearer " not in flattened
    assert launcher.REMOTE_ROOT_PREFIX in result["remote_root"]
    assert launcher.verify_launcher_result(result) == result
    saved = json.loads(
        (tmp_path / "output" / launcher.LOCAL_RESULT_NAME).read_text(encoding="utf-8")
    )
    assert saved == result


def test_pre_execution_hash_mismatch_fails_before_gate_and_cleans(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.pre_hashes["tool"] = "0" * 64
    result = _run(tmp_path, fake)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "hash_mismatch"
    assert _gate_calls(fake) == []
    assert result["cleanup_verified"] is True


def test_post_execution_hash_mutation_fails_after_one_gate_and_cleans(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.post_hashes["provider"] = "0" * 64
    result = _run(tmp_path, fake)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "hash_mismatch"
    assert len(_gate_calls(fake)) == 1
    assert result["artifact_request_count"] == 1
    assert result["cleanup_verified"] is True


def test_unexpected_gate_output_is_rejected_without_retry(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.gate_stdout += "unexpected\n"
    result = _run(tmp_path, fake)
    assert result["reason"] == "unexpected_remote_output"
    assert len(_gate_calls(fake)) == 1
    assert result["cleanup_verified"] is True


@pytest.mark.parametrize(
    ("uuid_value", "apps", "expected"),
    [
        ("GPU-wrong", "", "gpu2_uuid_mismatch"),
        (
            launcher.EXPECTED_GPU2_UUID,
            f"{launcher.EXPECTED_GPU2_UUID}, 4242\n",
            "gpu2_external_app",
        ),
    ],
)
def test_gpu2_identity_and_external_apps_fail_before_api(
    tmp_path, uuid_value, apps, expected
):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.gpu_uuid = uuid_value
    fake.apps = apps
    result = _run(tmp_path, fake)
    assert result["reason"] == expected
    assert _gate_calls(fake) == []
    assert result["cleanup_verified"] is True
    assert result["gpu_pre"]["uuid"] == uuid_value
    if apps:
        assert result["gpu_pre"]["external_compute_pids"] == [4242]


def test_nvidia_smi_unavailable_fails_before_api(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.gpu_returncode = 127
    result = _run(tmp_path, fake)
    assert result["reason"] == "gpu_unavailable"
    assert _gate_calls(fake) == []


def test_cleanup_prefix_safety_rejects_any_other_path_without_command():
    with pytest.raises(launcher.LauncherError, match="cleanup_failed"):
        launcher._validate_remote_root("/tmp/not-the-d3-root")
    with pytest.raises(launcher.LauncherError, match="cleanup_failed"):
        launcher._validate_remote_root("/home/research/persistent")


def test_downloaded_artifact_tamper_is_blocked(tmp_path):
    artifact = _passing_artifact(tmp_path, _secret())
    fake = _FakeRemote(artifact)
    tampered = json.loads(json.dumps(artifact))
    tampered["observed_utc"] = "2026-08-14T06:07:09Z"
    fake.download_bytes = _artifact_bytes(tampered)
    fake.remote_artifact_sha = hashlib.sha256(fake.download_bytes).hexdigest()
    result = _run(tmp_path, fake)
    assert result["reason"] == "artifact_tamper"
    assert result["cleanup_verified"] is True


def test_secret_like_remote_output_is_not_retained(tmp_path):
    secret = _secret()
    fake = _FakeRemote(_passing_artifact(tmp_path, secret))
    fake.gate_stderr = "Authorization: Bearer " + secret
    result = _run(tmp_path, fake)
    assert result["reason"] == "secret_like_output"
    saved = (tmp_path / "output" / launcher.LOCAL_RESULT_NAME).read_text(encoding="utf-8")
    assert secret not in saved
    assert "Bearer " not in saved
    assert len(_gate_calls(fake)) == 1


def test_existing_remote_root_is_never_created_or_cleaned(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.root_exists = True
    result = _run(tmp_path, fake)
    assert result["reason"] == "remote_root_exists"
    assert len(fake.calls) == 1
    assert fake.root_exists is True
