# D 阶段最终收口报告

## 结论

**`D_PHASE_CLOSED_NO_PRODUCTION_OPTIMIZATION`**

- 阶段状态：`COMPLETE`
- 审核状态：`PASS_WITH_CONCERNS`
- 可纳入主线组合的 D 阶段优化：`[]`
- 本收口没有修改生产代码，也没有执行新的 LLM、API、GPU 或 provider 实验。

这意味着 D 阶段研究工作已结束，但当前证据没有放行任何生产优化。它不等于“所有方向均无收益”：只表示已有收益没有达到可安全外推至生产主线的证据门槛。

## Git 边界

- branch：`perf/llm-production-shape-d1c`
- base：`453dc356d29dce783dfb7c6e915f5195dc272fe1`
- D2 审计修复 HEAD：`4fa39478ef02d68ff528155bcfcef429562f7de4`
- 本任务不 push、不 merge。

## 分项判定

1. **D1 Chat 并发**：观察到约 `3.32%` 的有限改善，但每臂只有 2 次重复，不能证明大幅稳定收益，不进入组合。
2. **D1b 非批处理 embedding**：观察到约 `76.38%` 改善；这是合成 workload，不能外推生产。
3. **D1b 批处理 embedding**：压力 replay 中 mif=4 / mif=25 分别观察到约 `30.18%` / `27.30%` 改善；压力 replay 不等于 Mason 生产路径。
4. **D1c Mason retry**：96 个受控 batched 请求的 SDK retry 为 0，历史 retry 触发条件未复现，故不修改生产调度。
5. **D2 235B 对照**：仅完成阻塞证据审计；benchmark 执行臂为 0，不存在 235B 与 14B 的速度或质量结论。

## 证据绑定

- `CHAT_UNBOUNDED_RESULTS_AUDITED.json` raw SHA256: `08b6a15301cc58fc8a19abbcdd9c45c99c5ccf07e4c76125ee7fff04118331c4`
- `D1B_ALL_RESULTS.json` raw SHA256: `726012108ce8fcb67656727521ab39794bf55eaf9b9991de4ae97eb6ec8d1cae`
- `D1B_BATCH_RESULTS.json` raw SHA256: `08e4a0adc8e3af7327e75cebabbc1ff6b39d53b9a4431f63c1cce3b4faacbae0`
- `D1C_ALL_RESULTS.json` raw SHA256: `06574f6a264bb2ffdd0da243d1ca5a0fb699fcbb2b5f2c17dce995fb880551ed`
- `D1_ALL_RESULTS.json` raw SHA256: `0b009e1b02d161af78a39a54dcf23925ff418b44bb943fcf5db9f5ca67e60310`
- `D2_EVIDENCE_FINAL.json` raw SHA256: `1356d9cfb4ad8ecb2783b7019f3745583c2e5031a3c7d57e831363dfb0797dd2`
- `D2_RESULT.json` raw SHA256: `45e1b0b35d2cbc1cde984685f00981bda055d52f2c368c9cef048d51324a1f0c`

- `D_PHASE_FINAL_RESULT.json` internal canonical SHA256: `5c1545061fc99d7abf1ded2d30413631a667f3a509248763639b67dc903a3d5f`
- `D_PHASE_FINAL_RESULT.json` raw file SHA256: `76518f23c479948ddfcb463733227dfcb3683f46ced2e4a6b9a3f184b8286736`
- `d_phase_finalize.py` raw file SHA256: `de681ae6c2d81624f24b9828b4024abbb14cdff5473f3c0db3f6509554e6ddb1`

JSON 内部哈希使用 `canonical_json_sha256`，作用域为 `D_PHASE_FINAL_RESULT_FIELDS_EXCLUDING_ARTIFACT_SHA256`。输出报告自身 raw SHA 不写入自身，以避免自引用循环。

## 遗留关注项

- Chat 每臂仅 2 次重复，缺少预设置信区间。
- embedding 收益只来自合成/压力 replay，未在生产 Mason session 重现。
- Mason 574/575 transport retry 根因仍未定位。
- D2 因 provider 可用性与授权证据阻塞，没有运行双模型 benchmark。

## 后续主线含义

D 阶段不向当前 B/C 组合优化追加变量。未来若重新研究 D，必须作为新的独立研究线，以生产形状 replay、预设重复次数和 provider 授权门禁重新开始，不能把本阶段的合成收益直接当作生产收益。
