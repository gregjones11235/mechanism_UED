# D3 server metadata and credential-safety gate

- Server Ollama `qwen2.5-coder:14b`: **PASS** (`/api/tags` HTTP 200; exact model present)
- Dedicated GPU2: **PASS** (UUID recorded; 1 MiB used, no compute PID)
- DeepSeek `/v1/models`: **BLOCKED** (HTTP 401 in the initial metadata gate)
- Safety event: a malformed metadata-only shell retry exposed a credential fragment in operator-visible output; no value is retained here
- Completion requests: `0`; embedding requests: `0`; post-stop external calls: `0`
- Final disposition: **BLOCKED_EXTERNAL_PROVIDER_CREDENTIAL_EXPOSURE**

The server gate is independent of the earlier local-Windows blocked gate. No remote process,
file, GPU, completion, or embedding was started or modified. Further external-provider work is
stopped pending credential rotation and explicit re-approval.
