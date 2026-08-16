# D3Q Qwen metadata gate

- Final status: BLOCKED_QWEN_MODEL_NOT_ADVERTISED
- Ollama /api/tags: exactly 1 GET, HTTP 200, exact qwen2.5-coder:14b present.
- Qwen /models: exactly 1 GET, HTTP 200, credential_echo=false; launcher-declared qwen-flash-2025-07-28 absent.
- Credential: EXP_QWEN_API_KEY present; value never serialized or logged.
- DeepSeek endpoint requests: 0; completion requests: 0; embedding requests: 0.
- GPU2 before/after: UUID GPU-8df11537-ab79-722d-606f-411966196c4c, free 45619 MiB, no external compute PID.
- GPU3/GPU1 were not touched; no remote process or file was started/modified.
- Cross-vendor IDs returned by the provider catalog were filtered and not persisted.
- The earlier model-uncertain gate is preserved in D3Q_FIRST_GATE_EVIDENCE.json with zero metadata requests.
