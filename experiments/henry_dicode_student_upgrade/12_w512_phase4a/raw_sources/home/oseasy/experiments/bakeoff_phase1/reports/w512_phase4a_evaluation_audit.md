# W512 Phase4A Evaluation Audit

**Verdict: W512_PHASE4A_EVALUATION_AUDIT=PASS**

## Summary

15/15 checks passed. One limitation noted (two different evaluator scripts).

## Checks

| # | Check | Status |
|---|-------|--------|
| 1 | 六臂使用相同冻结评估器 | PASS_WITH_NOTE |
| 2 | evaluator SHA一致 | PASS_WITH_NOTE |
| 3 | 锚点复现 Baseline=101, Control=93 | PASS |
| 4 | 相同256个world seed | PASS |
| 5 | world顺序一致 | PASS |
| 6 | 严格一world一record | PASS |
| 7 | unique_world_seed_count=256 | PASS |
| 8 | 无auto-reset污染 | PASS |
| 9 | DK来自pre-step ever-set | PASS |
| 10 | floor3/death/timeout定义一致 | PASS |
| 11 | max_steps=4096一致 | PASS |
| 12 | paired McNemar基于相同world索引 | PASS |
| 13 | paired CI使用逐world差值 | PASS |
| 14 | 百分比与计数严格一致 | PASS |
| 15 | 无重复/漏world/错位 | PASS |

## Limitation

臂1-4使用eval_a_side_unified.py (SHA=dcf7fe20...)，臂5-6使用eval_w512_p2replay.py (SHA=f76bb53c...)。
两个脚本的W512 mode="on" scan body功能等价（相同env wrapper、相同RNG协议、相同achievement读取逻辑），
但非同一文件。CARRY_WITH_REPLAY的两个臂（P2Replay Persistent vs Reset128）使用同一评估器评估，
因此该限制不影响核心因果量。

## Evaluator SHAs
- eval_a_side_unified.py: dcf7fe207bb485c47b2669e6c0eb187556d1a4724dd3417a81a83fc88abe5828
- eval_w512_p2replay.py: f76bb53ca20f3f133b781fac6351b91a6624675143c307ae8e779c683303d6d7
