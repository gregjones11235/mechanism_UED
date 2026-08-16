# D3 DeepSeek Flash metadata gate evidence

- Classification: `D3_DEEPSEEK_GATE_LAUNCHER`
- Outcome: `BLOCKED`
- Sanitized reason: `artifact_request_count_invalid`
- Audited launcher commit: `d82bf904664153085e0d75dfda80edd1f4617d0b`
- Observed UTC: `2026-08-14T05:29:22Z`
- Fixed model: `deepseek-v4-flash`
- Fixed base URL: `https://api.deepseek.com`
- Credential declaration: `EXP_DEEPSEEK_API_KEY`
- HTTP status: `null`
- Exact model advertised: `false`
- Metadata request count: `0`
- Completion request count: `0`
- Embedding request count: `0`
- GPU index: `2`
- GPU UUID: `GPU-8df11537-ab79-722d-606f-411966196c4c`
- GPU free memory before gate: `45619 MiB`
- GPU compute applications before gate: `0`
- GPU post snapshot: `null` because the gate stopped before that phase
- Remote temporary-root cleanup verified: `true`
- Local staging retained: `false`
- Local public metadata artifact retained: `false`
- External execution hashes verified: `false`; post-execution hashes were not reached
- External artifact hash verified: `false`; no metadata artifact was published
- Provider observed SHA256: `94f5de686890d5ab36862c1846f4835314f01d9518a9813e7f297d6a0c185328`
- Gate tool observed SHA256: `fe07053a6d6d706237a0b0bb84d69b47cf571c4222e415b46e99d8267fa822dd`
- Gate internal artifact SHA256: `2b0966d5bfc049dc0835a4b6e7b1435a31c9113b6fe5974949c3c2516bc20742`
- Launcher result canonical SHA256: `c6e2501917cd215a316c24ad7aa62ed423f3320748e5d1d8a31ede0529d06c8b`

The authorized launcher was invoked exactly once and was not retried. It failed
closed because the audited gate reported zero metadata requests instead of the
required one. No completion, embedding, or training operation was attempted.
No response body, authorization header, credential material, cross-provider
identifier, or private-key material is retained in this evidence directory.
