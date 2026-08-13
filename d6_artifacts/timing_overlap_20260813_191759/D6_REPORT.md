# D6 timing/overlap audit (offline)

- Status: **INSUFFICIENT_SESSION_BOUNDARY_TRACE**
- Queue decision: **NO_QUEUE_JUSTIFIED_INSUFFICIENT_TRACE**
- Archived wall clock: `76400.00s`
- Evolution + compilation + training component sum: `96166.28s`
- Component-sum minus wall **upper bound**: `19766.28s` (`20.55%` of component sum; not measured interval overlap)

Evolution duration median/P95: `1516.07s` / `1713.86s`;
compilation median/P95: `168.95s` / `194.83s`.
These are phase-duration distributions, not synchronization waits. The archived run has no
critical-path/events file and no aligned producer-consumer/session wait spans, so synchronized
wait median/P95 and GPU-idle intervals are **not observable**. The requested fixed sequence
queue experiment is therefore not justified by evidence and was not launched; this is an
insufficiency-of-trace decision, not a claim that the threshold was below its gate.

No GPU, provider, or original timing artifact was touched.
