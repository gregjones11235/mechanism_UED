# 方法组开跑清单（Phase 4 → Phase 5）

> 起草 2026-07-07。前置 = 干净 baseline 已跑完（run `32v02vi9` / DiCode-run-1783460542，task_graph 4→52+ 节点一路涨、零故障）。
> 目标 = 在**与 baseline 完全相同的环境**下跑方法组 +A / +A+B，构成三组消融。
> **核心原则：三组唯一的差别只能是 flag。环境（jax 0.6.2 / Ollama / 显存配置 / seed）全程不变。**

---

## 0. 环境不变性检查（每次开跑前必做）

方法组必须和 baseline 同环境，否则对照无效。**别做的事**：别换 pod、别改 jax 版本、别 `pkill ollama`（14B keep-alive 常驻着要复用）、别改 `XLA_PYTHON_CLIENT_MEM_FRACTION`。

每次开跑前，固定跑这段确认（venv + 4 个环境变量 + Ollama + 14B 全 GPU）：

```bash
source /workspace/venv/bin/activate
cd /workspace/mechanism_UED/dicode_src

# 4 个环境变量（和 baseline 完全一致）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export OLLAMA_KEEP_ALIVE=-1
export GENERATION_SERVER_URL=http://localhost:11434/v1
export EMBEDDING_SERVER_URL=http://localhost:11434/v1

# 确认 4 件事
python -c "import jax; print('jax', jax.__version__, jax.devices())"    # 期望 0.6.2 + [CudaDevice(id=0)]
curl -s http://localhost:11434/api/tags | python -c "import sys,json; print([m['name'] for m in json.load(sys.stdin)['models']])"  # 两个模型都在
grep -a "offloaded" /workspace/ollama_server.log | tail -1              # 期望 49/49（14B 全 GPU）
nvidia-smi | grep "MiB /"                                                # JAX 释放了、只剩 Ollama ~15G
```

> 如果 `offloaded` 不是 49/49（罕见，14B 被挤下过 GPU）→ 重跑一次预热：
> `curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5-coder:14b","prompt":"hi","stream":false}' > /dev/null`
> 再查一次 offload。

---

## 1. Phase 4a：+A（只开 skill scheduler）

先单独验证 ★A hook，别一次开两个（出问题不好定位）。**只加一个 flag：`+skill_preflight.use_scheduler=true`**（`+` 前缀是因为 config 里没有 `skill_preflight` 节点，要新增）。

```bash
# 承接第 0 节的环境变量
tmux kill-session -t train 2>/dev/null
tmux new-session -d -s train "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 \
GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && \
python experiments/training/run_dicode.py \
  seed=1 \
  use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED \
  training.total_timesteps=500000000 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  gen_manager.task_generator.max_tokens=8192 \
  gen_manager.env_generator.max_tokens=8192 \
  +skill_preflight.use_scheduler=true \
  > /workspace/run_A.log 2>&1"
sleep 3 && tmux ls
```

**盯什么（+A 特有信号）：**
```bash
tail -f /workspace/run_A.log | grep -aiE "SkillGraph|designs created|Saving task graph|Session finished|500|No new tasks|Traceback" --line-buffered
```
- ✅ **`[SkillGraph] frontier tier N, targets: [...]`** 出现 —— 说明 scheduler 在工作、定位到学习前沿（这是 +A 生效的直接证据，baseline 里没有这行）；
- ✅ `12 designs created` + `task graph N nodes` 涨 —— 生成受 target 影响但仍能编译；
- ❌ 若生成大量编译失败 / `No new tasks` 变多 —— 可能是 A-2 注入的 target 让 14B 产出跑偏，贴 log。

跑几个 session（到 ~8e7 或 session 8-10，和 baseline 对齐步数），确认健康 + `[SkillGraph]` 正常，就 `pkill -9 -f run_dicode` 停。**记下 run ID = arm_A。**

---

## 2. ★B 前置确认：default_params —— ✅ 已确认（2026-07-07）

> **结论：`has default_params: True`**（在 `survive.py` seed task 上实测）。
> preflight 的 B-1 hook 可直接使用 `_raw.default_params`，**无需改代码**，`use_preflight` 可放心开启。
> 下面的命令保留备用：**换 pod / 改动 minicraftax 环境类后建议重跑一次确认**。

+A+B 的 preflight hook 用 `Task(...).env.default_params`。`MiniCraftaxTrain` 继承自 `CraftaxSymbolicEnvNoAutoReset`，`default_params` 由 gymnax 风格父类提供。确认命令（避免 preflight 静默走 except 分支"kept all"、白开）：

```bash
source /workspace/venv/bin/activate
cd /workspace/mechanism_UED/dicode_src
# 找一个已存在的 task 路径（archive 里的），确认 default_params 可取
python -c "
from dicode.dreaming.gen_manager import Task
from dicode.dreaming.utils import smart_absolute_path
import glob
# 找一个 seed task 或 archive 里的 task 文件
cands = glob.glob('src/minicraftax/tasks/seed_tasks/*.py')
print('用', cands[0])
t = Task(smart_absolute_path(cands[0]))
print('has default_params:', hasattr(t.env, 'default_params'))
if hasattr(t.env, 'default_params'):
    print('default_params OK:', type(t.env.default_params).__name__)
"
```
- **`has default_params: True`** → ✅ 直接跑 +A+B（第 3 节）。
- **`False`** → preflight 的 env_params 取法要换。贴我输出，我照 `run_session_training` 给 sampled task 造 env_params 的方式改 B-1（只改那一处）。

---

## 3. Phase 4b：+A+B（scheduler + preflight，完整最小核心）

default_params 确认 True 后，**两个 flag 都开**：

```bash
tmux kill-session -t train 2>/dev/null
tmux new-session -d -s train "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 \
GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && \
python experiments/training/run_dicode.py \
  seed=1 \
  use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED \
  training.total_timesteps=500000000 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  gen_manager.task_generator.max_tokens=8192 \
  gen_manager.env_generator.max_tokens=8192 \
  +skill_preflight.use_scheduler=true \
  +skill_preflight.use_preflight=true \
  > /workspace/run_AB.log 2>&1"
sleep 3 && tmux ls
```

**盯什么（+A+B 特有信号）：**
```bash
tail -f /workspace/run_AB.log | grep -aiE "SkillGraph|Preflight|designs created|Saving task graph|Session finished|500|Traceback" --line-buffered
```
- ✅ `[SkillGraph] frontier tier ...`（scheduler 在工作）；
- ✅ **`[Preflight] kept X/Y new tasks`** 出现 —— preflight 在过滤（保留 X 个、拒绝 Y−X 个）。这是 +B 生效的直接证据；
- ✅ 被拒任务在 archive 里标 `preflight_*` 状态；
- ⚠️ 若每次都 `kept 0/Y`（全拒）或 `kept Y/Y`（全留）→ preflight 阈值可能需调，或走了 except 分支（说明 default_params 还是没取到），贴 log；
- ⚠️ 若明显变慢 —— preflight 每候选跑一次 eval-scale rollout（12 候选 × 完整 eval），若太慢考虑给 preflight 配小 num_envs/num_steps（Phase 3b 优化，先跑通）。

同样跑到和 baseline/+A 对齐的步数，`pkill -9 -f run_dicode` 停。**记下 run ID = arm_AB。**

---

## 4. Phase 5：三组消融对比

三个 run（同 seed=1、同环境、只差 flag）：

**★ 对齐目标（三组必须一致）：跑满 session 10 = `global_env_steps 157,286,400` = `update_step 1200`。**

| arm | flag | wandb run | 步数 | 含义 |
|---|---|---|---|---|
| baseline（纯 DiCode） | 无 | **32v02vi9** (`singleLLM_baseline`) | 157,286,400 ✅ | 对照 |
| +A（+skill graph） | `use_scheduler=true` | （arm_A，记下） | 对齐到 157,286,400 | 有图 vs 无图 |
| +A+B（完整最小核心） | 两个都开 | （arm_AB，记下） | 对齐到 157,286,400 | 主方法 |

**主判据（你 idea 的核心主张）**——不是看 make_iron_pickaxe 脱没脱离 29%（那只是烟雾测试），而是：
- **tier2-4 的 held-out 裸 SR**（铁装 / 钻石 / 附魔类）：+A+B 能否把 baseline 崩到 0% 的深层 tier 拉起来（哪怕到 ~10%）；
- **mean_performance / mean_return 的整体曲线**：三组同步数下谁更高；
- **curriculum 演化质量**：三组的 task_graph 节点增长、per-tier 任务分布（+A 的 scheduler 应让生成更集中在学习前沿的 tier）。

**在 wandb 里**：把三个 run 加进同一个 workspace 对比，看 `curriculum/num_tasks_compiled_cumulative`、各 tier 成就 SR、mean_performance。

> 统计严谨性（可选，有算力再做）：目前三组各 seed=1，是"受控单点对照"。若要说服力更强，每组补 seed=2,3 各跑一遍，报均值±方差。第一轮 seed=1 先把趋势跑出来。

---

## 5. 运维要点（贯穿始终）

1. **停训练用 `pkill -9 -f run_dicode`**，别用 Ctrl-C（杀不干净，会残留占显存）。停后 `nvidia-smi | grep "MiB /"` 确认 JAX 显存释放。
2. **Ollama 全程别关**——三组共用同一个 14B（keep-alive 常驻）。只有换 pod / 重启机器才需要重新 `ollama serve` + 预热。
3. **每个 arm 的 run ID + output dir 记进 `baseline_实验记录.md`**，别丢（对比时要按 ID 找）。
4. **环境别动**：jax 0.6.2、显存 0.75、seed=1、max_tokens 8192 —— 三组必须一致。任何一处变了，那一组就不能和另两组比。
5. **换 pod 后的环境恢复**（万一）：`source /workspace/venv/bin/activate`（venv 在卷上，不用重装）+ 起 Ollama（带 `OLLAMA_MODELS=/workspace/ollama_models`）+ 预热 14B + 确认 jax 0.6.2。

---

## 附：一眼看懂的开跑顺序

```
[baseline 32v02vi9 已完成]
   ↓ pkill -9 停掉，环境不动
第0节：环境不变性检查（jax/Ollama/49-49/显存）
   ↓
第1节：+A（只开 scheduler）→ 盯 [SkillGraph] → 跑到对齐步数 → pkill → 记 run_A
   ↓
第2节：确认 default_params True
   ↓
第3节：+A+B（两个都开）→ 盯 [Preflight] kept X/Y → 跑到对齐步数 → pkill → 记 run_AB
   ↓
第4节：三组进同一 wandb workspace 对比 tier2-4 held-out SR + mean_performance
```
