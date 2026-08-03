# Simulator-Centric Frontier-UED · 队友包审计

审计性质：只读包审计，不代表训练、PPO、课程或性能结果。

## 输入与完整性

- 输入包：`Henry_work2.zip`（本地用户提供的原始压缩包）
- SHA256：`226363969f50fe42b35bd3cdb03d6a0e7cba16be44c2d1463ffc375b0e907e62`
- 大小：112,352,362 bytes
- 本地隔离审计目录：`.tmp_simulator_team_audit/Henry_work2/`
- 解包统计：320 regular files，57 directories，约 164,667,281 bytes
- 主要内容：`code/dicode_v7fix60`、`code/dicode_v8`、v7/v8 designcheck、pv8 probe 结果、checkpoint 参考文件与 README/设计材料。

## 可复用与拒绝范围

### 可作为基础参考

- v8 `frontier_archive.py` 的事件捕获、分层去重、状态结构思路；
- v8 designcheck 中已明确的 fail-closed、实际分支数和未实现项声明；
- pv8_0 的 archive/restore smoke 作为初始状态保存证据；
- v7/v8 的字段和 Craftax wrapper 行为作为后续 CC1 对照输入。

### 本轮不迁移

- checkpoint、Orbax 目录、NPZ/缓存、运行日志和正式评估轨迹；
- LLM/provider 配置、API key、`.env`；
- `router_diagnosis_v8` 主流程、完整 PPO 接入、12+4 curriculum、长跑；
- 任何 designcheck、smoke 或 archive 结果作为训练改进/性能结论。

## 结论

队友包包含可审计的静态代码和有限的 archive/restore smoke，但没有本轮所禁止的完整 PPO/真实长程训练闭环。当前迁移仅实现独立的 `dicode.simulator_frontier` foundation API，并把所有未完成集成明确留给 CC1。
