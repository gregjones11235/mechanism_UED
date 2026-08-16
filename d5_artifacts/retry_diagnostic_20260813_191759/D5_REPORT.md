# D5 retry diagnosis (offline)

- Status: **INSUFFICIENT_REQUEST_TRACE**
- Conclusion: **NOT_REPRODUCED**
- Archived request events replayed: `0`
- Network/provider calls: `0`

Attempt 06 reports `863` transport-level retry events
(`289` chat and `574` embedding),
with `2201` eventual HTTP 200 responses and `0` non-200 responses.
Parse, empty-response, compilation, requeue, reflection, and API-repair counters are all zero in the
summary. The evidence does not retain request IDs, connection/timeout subtype, event-loop or client
identity, session boundaries, or per-request timestamps. Consequently the trigger cannot be separated
into connection reset, pool contention, timeout, or server close, and a retry fix would be speculation.

No external endpoint, GPU, credential, or source evidence was touched.
