# D4 embedding request-shape audit

- Status: **NO_REPLAY_PAYLOAD**
- Mainline eligible: **no**
- Network/completion/embedding calls: `0/0/0`
- Planned matrix: `max_in_flight={1,2,4}`, 3 repeats each
- Executed matrix: none

The frozen manifest preserves model, embedding dimension (`768`),
16 task order, batch-size sequence, lifecycle labels, and a task-text pool. Mason attempt 06 does
not preserve the exact serialized request message bytes, request/response bytes, returned embedding
values, or request-level timings. Therefore a max-in-flight replay would not be a faithful experiment.
Existing D1C summary rows are retained as historical context only and are not counted as D4 observations. The
manifest reports dimension `768`, while the archived provider-config summary
reports `1024`; this unresolved shape conflict is another replay gate.

No external provider, Ollama endpoint, GPU, or original evidence file was modified.
