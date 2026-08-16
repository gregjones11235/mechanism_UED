# Phase 0：本地小模型代码生成能力验证（Go/No-Go）

> 作者：Mason　｜　日期：2026-07-06　｜　分支：`skill-preflight-ued_Mason`
> 结论速览：**GO** — 本地 14B coder 生成 Craftax 环境代码，**首次编译通过率 12/12 = 100%**。

---

## 1. 目的

验证方案的**地基假设**：本地小模型（区别于 DiCode 默认的 Qwen3-235B）能否生成**可编译、可用的 Craftax 环境代码**。

这是整个"用本地小模型 + 多 LLM 达到/逼近 DiCode 成功率"方向的 go/no-go——如果小模型连可编译的环境代码都写不出，方案需要重新设计。因此在写任何方法代码（skill scheduler / preflight）之前，先做这个最小验证。

---

## 2. 环境

| 项 | 配置 |
|---|---|
| 平台 | RunPod，2× RTX A6000（48GB each） |
| DiCode 环境 | `uv sync --all-extras` + `uv pip install "jax[cuda12]"`（依赖由 `uv.lock` 锁定） |
| 本地模型服务 | **Ollama**（非 vLLM，原因见下）serving `qwen2.5-coder:14b`（量化版，~9GB） |
| GPU 分配 | GPU 0 → Ollama（14B，~15GB 占用）；GPU 1 → 留给 DiCode（编译检查在 CPU） |
| 模型接口 | Ollama 的 OpenAI 兼容接口 `http://localhost:11434/v1`，DiCode `local` provider 直连 |

**关键配置：**
- `.env`：`GENERATION_SERVER_URL=http://localhost:11434/v1`，`OPENAI_API_KEY=ollama`（占位，Ollama 不校验）
- provider 配置：复制 `conf/gen_manager/llm/local_gen.yaml` → `local_qwen14b.yaml`，`model` 改为 `qwen2.5-coder:14b`
- Ollama 起服务时设 `OLLAMA_CONTEXT_LENGTH=32768`（DiCode 生成 prompt 很大，见下）

---

## 3. 踩过的坑与解法（给组里复现的人参考）

这一节记录搭建过程中的实际障碍，避免重复踩坑。

### 3.1 vLLM 起不来 → 改用 Ollama
- 症状：vLLM 反复 `Engine core initialization failed` / `Failed to infer device type` / NVML 符号缺失（`nvmlDeviceGetHandleByIndex_v2`）。多次尝试（`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`--enforce-eager`、`LD_LIBRARY_PATH` 指向真库）均无效。
- 根因：RunPod 的 PyTorch 镜像与 vLLM 的 CUDA/驱动栈底层不匹配，vLLM 无法识别 GPU 设备。
- **解法：改用 Ollama。** Ollama 一次就认到 A6000（`library=CUDA name="NVIDIA RTX A6000"`），对环境挑剔度远低于 vLLM。Phase 0 只需生成十几个环境，不需要 vLLM 的高吞吐，Ollama 完全够用，且满足"本地部署"设定。

### 3.2 JAX 报 cuSPARSE / cuSolver 缺失 → 可忽略
- 症状：`Unable to load cuSPARSE`，JAX fallback 到 CPU。
- 影响：**无害。** `check_compilation` 本就在 CPU 上运行（`jax.default_device(cpu)`），JAX 走 CPU 不影响编译检查。（若之后跑训练需要 GPU JAX，再修 CUDA 库版本。）

### 3.3 Craftax 贴图缓存损坏 → 删除重建
- 症状：`import craftax` 崩在 `OSError: Invalid data stream`（`Loading Craftax textures from cache` 后）。
- 根因：贴图缓存文件 `craftax/craftax/assets/texture_cache.pbz2` 下载/写入不完整（损坏文件仅 276KB，正常约 1.75MB）。
- **解法：删除损坏文件让 craftax 重新渲染。** craftax 找不到缓存时会走 else 分支重新生成并重存（`constants.py` L1115）。删除后重新 import，打印 `Textures loaded and saved to cache.` 即修复，新缓存 1.75MB。

### 3.4 Ollama 上下文
- DiCode 生成 prompt 很大（系统 prompt 嵌入完整 MiniCraftax + Craftax 库代码 + mob 信息），本次实测 **system prompt ≈ 71,714 chars ≈ 17,928 tokens**，加 user prompt ≈ 2,300 tokens，共约 20K tokens 输入。
- 起 Ollama 时设 `OLLAMA_CONTEXT_LENGTH=32768` 覆盖，确保不截断。实测生成质量正常（100% 编译通过），未见上下文截断迹象。

---

## 4. 实验方法

脚本：`dicode_src/phase0.py`（standalone，不进主训练循环）。

**做法：**
1. 复用 DiCode 真实的环境生成 prompt（`prompts.cl_.gen_env` 的 `system_prompt` + `user_prompt`），填入真实的 Craftax/MiniCraftax/Mob 知识库上下文。
2. 用 DiCode 自带的 4 个 seed 任务（`collecting` / `combat` / `crafting` / `survive`）做 **leave-one-out**：拿其中一个的任务描述当 target，另 3 个的代码当 few-shot 示例。
3. 每个 target 让 14B 生成 **3 次**，共 **12 次生成**。
4. 每次生成：提取 `<code>` 内容 → **剥除 markdown 围栏** → 复刻 DiCode 的 `check_compilation`（写临时文件 → `Task` 加载 env → 在 CPU 上 `reset` + `step` → 校验 inventory dtype 为 int32）。
5. 统计首次编译通过率（**无反思**，纯裸首次），生成代码存 `phase0_out/`。

**说明：** 本实验测的是"首次、无反思"的裸通过率——DiCode 真实运行时对失败的会反思重试，因此进入训练的可用率 ≥ 此数。

---

## 5. 结果

```
Model: qwen2.5-coder:14b
First-try compile rate: 12/12 = 100%
  collecting : 3/3
  combat     : 3/3
  crafting   : 3/3
  survive    : 3/3
VERDICT: GO
```

**过程中的一个发现（已修）：** 首版脚本测得 10/12（83%），2 个 FAIL 均为 `SyntaxError: line 1`。排查发现是模型在 `<code>` 标签内又套了 markdown 围栏（` ```python ... ``` `），被代码提取逻辑连同围栏一起提取，导致首行语法错——**是提取格式问题，不是模型能力问题**。在 `extract_code` 中增加剥除 markdown 围栏的处理后，重跑得 **12/12 = 100%**，印证了这一判断。

生成的环境代码平均约 2,400–3,400 字符，包含正确的 import、`BaseTask` 继承、`__init__` / `get_task_params` / `generate_world` 实现，且能通过完整的 reset+step 校验。

---

## 6. 结论与下一步

**结论：GO。** 本地 14B coder 完全能生成可编译的 Craftax 环境代码（首次通过率 100%，格式修正后），方案的地基假设成立。之前最大的风险点（"小模型写不出有效环境代码则方案作废"）**已排除**。

**下一步（方法实现，见实现文件清单）：**
1. `skill_scheduler.py`：复用 `auction/craftax_achievements.py` 的 `reachable_ceiling()` / `tier_mastery()`，定位学习前沿。
2. `preflight.py`：复用 `craftax_evaluation.py:make_evaluate` 改成候选关卡冷 rollout + partial-progress 判据 + 分级漏斗。
3. 两处 hook（`run_dicode.py` / `evolution_efficient.py`）串入主循环。
4. 短 go/no-go run（砍小 timesteps）验证整条管线。

**待补充的深入检查（可选）：** 本实验只验证"可编译"，未验证"关卡是否有意义/可解"——建议人工抽查 `phase0_out/` 中几个生成环境的关卡设计质量，作为更深一层的确认。

---

## 附：复现步骤（简要）

```bash
# 1. DiCode 环境
cd dicode_src && uv sync --all-extras && uv pip install "jax[cuda12]"

# 2. Ollama + 本地 14B（GPU 0）
curl -fsSL https://ollama.com/install.sh | sh
tmux new-session -d -s ollama 'export OLLAMA_MODELS=/workspace/ollama_models; export CUDA_VISIBLE_DEVICES=0; export OLLAMA_HOST=0.0.0.0:11434; export OLLAMA_CONTEXT_LENGTH=32768; ollama serve'
ollama pull qwen2.5-coder:14b

# 3. 配置：.env 设 GENERATION_SERVER_URL=http://localhost:11434/v1；
#    conf/gen_manager/llm/local_qwen14b.yaml 的 model 改为 qwen2.5-coder:14b

# 4. 跑
uv run python phase0.py
# 若 craftax 报 texture 缓存损坏：删除 <site-packages>/craftax/craftax/assets/texture_cache.pbz2 后重跑
```
