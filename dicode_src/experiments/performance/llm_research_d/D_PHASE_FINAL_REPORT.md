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

- `CHAT_UNBOUNDED_RESULTS_AUDITED.json` normalized UTF-8/LF SHA256: `c473fcdbfc75ac7864bc2d0197501f71ac8c5faae5a2c08f25ae9e4dc1bb6cf0`
- `D1B_ALL_RESULTS.json` normalized UTF-8/LF SHA256: `726012108ce8fcb67656727521ab39794bf55eaf9b9991de4ae97eb6ec8d1cae`
- `D1B_BATCH_RESULTS.json` normalized UTF-8/LF SHA256: `08e4a0adc8e3af7327e75cebabbc1ff6b39d53b9a4431f63c1cce3b4faacbae0`
- `D1C_ALL_RESULTS.json` normalized UTF-8/LF SHA256: `06574f6a264bb2ffdd0da243d1ca5a0fb699fcbb2b5f2c17dce995fb880551ed`
- `D1_ALL_RESULTS.json` normalized UTF-8/LF SHA256: `0b009e1b02d161af78a39a54dcf23925ff418b44bb943fcf5db9f5ca67e60310`
- `D2_EVIDENCE_FINAL.json` normalized UTF-8/LF SHA256: `1356d9cfb4ad8ecb2783b7019f3745583c2e5031a3c7d57e831363dfb0797dd2`
- `D2_RESULT.json` normalized UTF-8/LF SHA256: `45e1b0b35d2cbc1cde984685f00981bda055d52f2c368c9cef048d51324a1f0c`

- `D2_RESULT.json` internal canonical SHA256: `54b1e01d6afa01a98f8fa0396ad8e9ccfec6ca79d87f8e78c55e1ecf557acaa6`
- `D2_EVIDENCE_FINAL.json` internal canonical SHA256: `234b766a8c494d6fa0f3afd875270ae8486a96c4a72b747f81deb92f99c5e037`
- `D_PHASE_FINAL_RESULT.json` internal canonical SHA256: `381ebc489bb4068822b28fa67b67688e82b59934ce079e154a32b9ff4b36cd29`
- `D_PHASE_FINAL_RESULT.json` observed raw file SHA256（非输入门禁，本机审计值）: `d91615073b780ac0c0dc33d0b9f229560ef9da0c6f7b864960857a9ab4f93be5`
- `d_phase_finalize.py` observed raw file SHA256（非输入门禁，本机审计值）: `9c248f8117c685ae5c5adc1582b71598cb0c2cf9bc2edcebf9c6891839f3dcfb`

输入门禁哈希先以 UTF-8 解码，再把 CRLF/CR 规范化为 LF 后计算 SHA256；因此 fresh Linux 与 Windows checkout 可复现，同时 JSON 内容变化仍会被拒绝。JSON 内部哈希使用 `canonical_json_sha256`，作用域为 `D_PHASE_FINAL_RESULT_FIELDS_EXCLUDING_ARTIFACT_SHA256`。输出报告自身 raw SHA 不写入自身，以避免自引用循环。

每个输出文件自身使用 no-clobber 原子创建；JSON 与报告是两个独立 artifact，不宣称二者构成跨文件事务。

## 遗留关注项

- Chat 每臂仅 2 次重复，缺少预设置信区间。
- embedding 收益只来自合成/压力 replay，未在生产 Mason session 重现。
- Mason 574/575 transport retry 根因仍未定位。
- D2 因 provider 可用性与授权证据阻塞，没有运行双模型 benchmark。

## 后续主线含义

D 阶段不向当前 B/C 组合优化追加变量。未来若重新研究 D，必须作为新的独立研究线，以生产形状 replay、预设重复次数和 provider 授权门禁重新开始，不能把本阶段的合成收益直接当作生产收益。
