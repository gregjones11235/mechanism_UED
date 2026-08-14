"""Offline command-transcript tests for the external DeepSeek gate launcher."""
from __future__ import annotations

import hashlib
import json
import shlex
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
        self.cleanup_stdout = "REMOVED\n"
        self.gate_stdout = json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
        self.gate_stderr = ""
        self.calls: list[launcher.CommandInvocation] = []
        self._hash_calls = 0

    def __call__(self, invocation):
        assert isinstance(invocation, launcher.CommandInvocation)
        call = list(invocation.argv)
        self.calls.append(invocation)
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
            return launcher.CommandResult(0, self.cleanup_stdout)
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


def _run(tmp_path: Path, fake: _FakeRemote, *, local_unlinker=None) -> dict:
    return launcher.run_launcher(
        ssh_target=SSH_TARGET,
        ssh_key=SSH_KEY,
        remote_python=REMOTE_PYTHON,
        remote_env_file=REMOTE_ENV,
        local_output_dir=tmp_path / "output",
        runner=fake,
        local_unlinker=local_unlinker,
        now=FIXED_NOW,
        token=FIXED_TOKEN,
    )


def _gate_calls(fake: _FakeRemote) -> list[launcher.CommandInvocation]:
    return [
        call
        for call in fake.calls
        if call.argv[0] == "ssh" and "--env-file" in call.argv[-1]
    ]


def _expected_success_invocations(tmp_path: Path) -> list[launcher.CommandInvocation]:
    root = launcher._remote_root(FIXED_NOW, FIXED_TOKEN)
    remote_gate = root + "/" + launcher.REMOTE_GATE_NAME
    remote_provider = root + "/" + launcher.REMOTE_PROVIDER_NAME
    remote_artifact = root + "/" + launcher.REMOTE_ARTIFACT_NAME
    source_dir = Path(launcher.__file__).resolve().parent
    staging = (
        (tmp_path / "output").resolve()
        / f".{launcher.LOCAL_ARTIFACT_NAME}.{FIXED_TOKEN}.staging"
    )
    ssh_base = [
        "ssh",
        "-i",
        str(SSH_KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "--",
        SSH_TARGET,
    ]
    scp_base = [
        "scp",
        "-q",
        "-i",
        str(SSH_KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "--",
    ]

    def ssh(remote_command: list[str]) -> list[str]:
        return ssh_base + [shlex.join(remote_command)]

    def remote_python(script: str, *args: str) -> list[str]:
        return ssh([REMOTE_PYTHON, "-c", script, *args])

    argv = [
        remote_python(launcher._EXISTENCE_SCRIPT, root),
        remote_python(launcher._CREATE_SCRIPT, root),
        scp_base
        + [
            str(source_dir / launcher.REMOTE_PROVIDER_NAME),
            f"{SSH_TARGET}:{remote_provider}",
        ],
        scp_base
        + [
            str(source_dir / launcher.REMOTE_GATE_NAME),
            f"{SSH_TARGET}:{remote_gate}",
        ],
        remote_python(
            launcher._HASH_FILES_SCRIPT,
            remote_gate,
            remote_provider,
        ),
        ssh(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
        ),
        ssh(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
        ),
        ssh(
            [
                REMOTE_PYTHON,
                remote_gate,
                "--env-file",
                REMOTE_ENV,
                "--output",
                remote_artifact,
            ],
        ),
        ssh(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
        ),
        ssh(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
        ),
        remote_python(
            launcher._HASH_FILES_SCRIPT,
            remote_gate,
            remote_provider,
        ),
        remote_python(
            launcher._HASH_ONE_SCRIPT,
            remote_artifact,
        ),
        scp_base + [f"{SSH_TARGET}:{remote_artifact}", str(staging)],
        remote_python(launcher._CLEANUP_SCRIPT, root),
        remote_python(launcher._EXISTENCE_SCRIPT, root),
    ]
    return [launcher.CommandInvocation(tuple(item), shell=False) for item in argv]


def _assert_no_artifact_or_staging(tmp_path: Path) -> None:
    output = tmp_path / "output"
    assert not (output / launcher.LOCAL_ARTIFACT_NAME).exists()
    assert list(output.glob(f".{launcher.LOCAL_ARTIFACT_NAME}.*.staging")) == []


class _SelectiveUnlinker:
    def __init__(self, *, fail=(), already_absent=()):
        self.fail = frozenset(fail)
        self.already_absent = frozenset(already_absent)
        self.calls: list[str] = []

    @staticmethod
    def _label(path: Path) -> str:
        return "staging" if path.name.endswith(".staging") else "artifact"

    def __call__(self, path: Path) -> None:
        label = self._label(path)
        self.calls.append(label)
        if label in self.already_absent:
            path.unlink(missing_ok=True)
            raise FileNotFoundError
        if label in self.fail:
            raise OSError("synthetic unlink failure")
        path.unlink(missing_ok=True)


def _force_final_readback_failure(monkeypatch) -> None:
    original_load = launcher.gate.load_artifact

    def fail_final_readback(path):
        if Path(path).name == launcher.LOCAL_ARTIFACT_NAME:
            return {"invalid": "readback"}
        return original_load(path)

    monkeypatch.setattr(launcher.gate, "load_artifact", fail_final_readback)


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
    assert result["http_status"] == 200
    assert result["exact_model_advertised"] is True
    assert result["completion_requests"] == 0
    assert result["embedding_requests"] == 0
    assert result["cleanup_verified"] is True
    assert result["staging_cleanup_attempted"] is True
    assert result["staging_exists_after_cleanup"] is False
    assert result["published_artifact_cleanup_attempted"] is False
    assert result["published_artifact_exists_after_cleanup"] is True
    assert result["local_cleanup_failure_count"] == 0
    assert fake.root_exists is False
    assert fake.calls == _expected_success_invocations(tmp_path)
    assert len(fake.calls) == 15
    assert all(call.shell is False for call in fake.calls)
    assert len(_gate_calls(fake)) == 1
    gate_argv = _gate_calls(fake)[0].argv
    assert "--model" not in gate_argv[-1]
    assert "--base-url" not in gate_argv[-1]
    flattened = json.dumps([call.argv for call in fake.calls]) + json.dumps(
        result, sort_keys=True
    )
    assert secret not in flattened
    assert "Bearer " not in flattened
    assert launcher.REMOTE_ROOT_PREFIX in result["remote_root"]
    assert launcher.verify_launcher_result(result) == result
    saved = json.loads(
        (tmp_path / "output" / launcher.LOCAL_RESULT_NAME).read_text(encoding="utf-8")
    )
    assert saved == result
    artifact_path = tmp_path / "output" / launcher.LOCAL_ARTIFACT_NAME
    assert artifact_path.exists()
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == fake.remote_artifact_sha
    assert list((tmp_path / "output").glob("*.staging")) == []


def test_pre_execution_hash_mismatch_fails_before_gate_and_cleans(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.pre_hashes["tool"] = "0" * 64
    result = _run(tmp_path, fake)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "hash_mismatch"
    assert _gate_calls(fake) == []
    assert result["cleanup_verified"] is True
    _assert_no_artifact_or_staging(tmp_path)


def test_post_execution_hash_mutation_fails_after_one_gate_and_cleans(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.post_hashes["provider"] = "0" * 64
    result = _run(tmp_path, fake)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "hash_mismatch"
    assert len(_gate_calls(fake)) == 1
    assert result["artifact_request_count"] == 1
    assert result["cleanup_verified"] is True
    _assert_no_artifact_or_staging(tmp_path)


def test_unexpected_gate_output_is_rejected_without_retry(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.gate_stdout += "unexpected\n"
    result = _run(tmp_path, fake)
    assert result["reason"] == "unexpected_remote_output"
    assert len(_gate_calls(fake)) == 1
    assert result["cleanup_verified"] is True
    _assert_no_artifact_or_staging(tmp_path)


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
    _assert_no_artifact_or_staging(tmp_path)


def test_nvidia_smi_unavailable_fails_before_api(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.gpu_returncode = 127
    result = _run(tmp_path, fake)
    assert result["reason"] == "gpu_unavailable"
    assert _gate_calls(fake) == []
    _assert_no_artifact_or_staging(tmp_path)


@pytest.mark.parametrize("tool", ["ssh", "scp"])
@pytest.mark.parametrize("mutation", ["missing", "duplicated", "wrong"])
def test_transport_base_rejects_invalid_identity_selection(tool, mutation):
    base = (
        launcher._ssh_base(SSH_TARGET, SSH_KEY)
        if tool == "ssh"
        else launcher._scp_base(SSH_KEY)
    )
    option_index = base.index("IdentitiesOnly=yes")
    if mutation == "missing":
        invalid = base[: option_index - 1] + base[option_index + 1 :]
    elif mutation == "duplicated":
        invalid = (
            base[:option_index]
            + ["IdentitiesOnly=yes", "-o"]
            + base[option_index:]
        )
    else:
        invalid = list(base)
        invalid[option_index] = "IdentitiesOnly=no"
    with pytest.raises(launcher.LauncherError, match="remote_command_failed"):
        launcher._require_exact_identity_selection(invalid)


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
    _assert_no_artifact_or_staging(tmp_path)


def test_published_artifact_readback_failure_removes_final_and_staging(
    tmp_path, monkeypatch
):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    _force_final_readback_failure(monkeypatch)
    result = _run(tmp_path, fake)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "artifact_tamper"
    _assert_no_artifact_or_staging(tmp_path)


def test_staging_unlink_failure_does_not_skip_final_removal(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    unlinker = _SelectiveUnlinker(fail={"staging"})
    result = _run(tmp_path, fake, local_unlinker=unlinker)

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "cleanup_failed"
    assert result["staging_cleanup_attempted"] is True
    assert result["staging_exists_after_cleanup"] is True
    assert result["published_artifact_cleanup_attempted"] is True
    assert result["published_artifact_exists_after_cleanup"] is False
    assert result["local_cleanup_failure_count"] == 1
    assert unlinker.calls == ["staging", "artifact"]
    assert not (tmp_path / "output" / launcher.LOCAL_ARTIFACT_NAME).exists()


def test_final_unlink_failure_does_not_skip_staging_removal(tmp_path, monkeypatch):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    _force_final_readback_failure(monkeypatch)
    unlinker = _SelectiveUnlinker(fail={"artifact"})
    result = _run(tmp_path, fake, local_unlinker=unlinker)

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "cleanup_failed"
    assert result["staging_cleanup_attempted"] is True
    assert result["staging_exists_after_cleanup"] is False
    assert result["published_artifact_cleanup_attempted"] is True
    assert result["published_artifact_exists_after_cleanup"] is True
    assert result["local_cleanup_failure_count"] == 1
    assert unlinker.calls == ["staging", "artifact"]
    assert list((tmp_path / "output").glob("*.staging")) == []


def test_both_unlink_failures_are_attempted_and_recorded(tmp_path, monkeypatch):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    _force_final_readback_failure(monkeypatch)
    unlinker = _SelectiveUnlinker(fail={"staging", "artifact"})
    result = _run(tmp_path, fake, local_unlinker=unlinker)

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "cleanup_failed"
    assert result["staging_exists_after_cleanup"] is True
    assert result["published_artifact_exists_after_cleanup"] is True
    assert result["local_cleanup_failure_count"] == 2
    assert unlinker.calls == ["staging", "artifact"]


def test_already_absent_cleanup_target_is_not_a_cleanup_failure(
    tmp_path, monkeypatch
):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    _force_final_readback_failure(monkeypatch)
    unlinker = _SelectiveUnlinker(already_absent={"artifact"})
    result = _run(tmp_path, fake, local_unlinker=unlinker)

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "artifact_tamper"
    assert result["staging_exists_after_cleanup"] is False
    assert result["published_artifact_exists_after_cleanup"] is False
    assert result["local_cleanup_failure_count"] == 0
    assert unlinker.calls == ["staging", "artifact"]
    _assert_no_artifact_or_staging(tmp_path)


def test_cleanup_failure_prevents_artifact_publication(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.cleanup_stdout = "UNEXPECTED\n"
    result = _run(tmp_path, fake)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "cleanup_failed"
    assert result["external_artifact_hash_verified"] is False
    _assert_no_artifact_or_staging(tmp_path)


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
    _assert_no_artifact_or_staging(tmp_path)


def test_existing_remote_root_is_never_created_or_cleaned(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    fake.root_exists = True
    result = _run(tmp_path, fake)
    assert result["reason"] == "remote_root_exists"
    assert len(fake.calls) == 1
    assert fake.root_exists is True
    _assert_no_artifact_or_staging(tmp_path)


def _rehashed_result(result: dict, path: str, replacement) -> dict:
    changed = json.loads(json.dumps(result))
    target = changed
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement
    payload = {
        key: value for key, value in changed.items() if key != "artifact_sha256"
    }
    changed["artifact_sha256"] = launcher._canonical_sha256(payload)
    return changed


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("manifest_sha256.tool", "0" * 64),
        ("pre_execution_sha256", None),
        ("pre_execution_sha256.tool", "0" * 64),
        ("post_execution_sha256", None),
        ("post_execution_sha256.provider", "0" * 64),
        ("remote_artifact_sha256", None),
        ("local_artifact_sha256", None),
        ("local_artifact_sha256", "0" * 64),
        ("artifact_internal_sha256", None),
        ("artifact_path", None),
        ("artifact_status", "BLOCKED"),
        ("artifact_request_count", None),
        ("artifact_request_count", 0),
        ("http_status", None),
        ("http_status", 401),
        ("exact_model_advertised", None),
        ("exact_model_advertised", False),
        ("gpu_pre", None),
        ("gpu_pre.gpu_index", 1),
        ("gpu_pre.uuid", "GPU-wrong"),
        ("gpu_pre.memory_free_mib", launcher.MINIMUM_GPU2_FREE_MIB - 1),
        ("gpu_pre.external_compute_pids", [4242]),
        ("gpu_post", None),
        ("gpu_post.gpu_index", 1),
        ("gpu_post.uuid", "GPU-wrong"),
        ("gpu_post.memory_free_mib", launcher.MINIMUM_GPU2_FREE_MIB - 1),
        ("gpu_post.external_compute_pids", [4242]),
        ("external_execution_hashes_verified", False),
        ("external_artifact_hash_verified", False),
        ("cleanup_verified", False),
        ("staging_cleanup_attempted", False),
        ("staging_exists_after_cleanup", True),
        ("published_artifact_cleanup_attempted", True),
        ("published_artifact_exists_after_cleanup", False),
        ("local_cleanup_failure_count", 1),
    ],
)
def test_rehashed_pass_counterexamples_are_rejected(
    tmp_path, field, replacement
):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    result = _run(tmp_path, fake)
    counterexample = _rehashed_result(result, field, replacement)
    with pytest.raises(launcher.LauncherError, match="artifact_tamper"):
        launcher.verify_launcher_result(counterexample)


def test_rehashed_blocked_result_cannot_masquerade_as_complete_pass(tmp_path):
    fake = _FakeRemote(_passing_artifact(tmp_path, _secret()))
    result = _run(tmp_path, fake)
    masquerade = _rehashed_result(result, "status", "BLOCKED")
    masquerade = _rehashed_result(masquerade, "reason", "hash_mismatch")
    with pytest.raises(launcher.LauncherError, match="artifact_tamper"):
        launcher.verify_launcher_result(masquerade)
