# Skill-Preflight UED 仓库使用说明

分支:`skill-preflight-ued_Mason`(`gregjones11235/mechanism_UED`)
维护:Mason · 最后更新 2026-07-25 · 英文版见 `REPO_GUIDE.md`

本分支新增的所有东西都是 **flag 门控、默认关闭**,不传任何 `+skill_preflight.*` /
`+training.sil_*` 就是原版 DiCode 行为。

**新增组件一览**

| 组件 | 位置 | 作用 |
|---|---|---|
| SkillGraph 前沿调度器 | `src/dicode/skill_preflight/skill_scheduler.py` | 决定每个 session 教哪些成就 |
| Preflight 任务过滤 | `src/dicode/skill_preflight/` | 丢掉编译不过、或到不了目标的生成任务 |
| Scaffold gate | 同上 | 拦截"双通道教学"(任务文本直接泄题) |
| 死因仪表 necropsy | `src/dicode/necropsy.py` | 在官方 eval 上挂逐死诊断 |
| SIL 采集 + BC | `experiments/analysis/sil_collect.py`、`src/dicode/sil_bc.py` | 采集成功片段,克隆回策略 |
| 判读器 | `experiments/analysis/necro_verdict.py`、`iron_triage.py` | 读尸体、打判决表 |

---

## 1. 环境搭建

### 1.1 机器

A100 SXM(80 GB)。**CUDA-13 的 pod** 要把加载路径指到自带的 CUDA-12 库,否则 JAX 看不到 GPU:

```bash
export LD_LIBRARY_PATH=$(python -c "import site,glob,os;print(':'.join(glob.glob(os.path.join(site.getsitepackages()[0],'nvidia','*','lib'))))"):$LD_LIBRARY_PATH
```

### 1.2 Python 环境

```bash
python -m venv /workspace/venv && source /workspace/venv/bin/activate
pip install -e /workspace/mechanism_UED/dicode_src
```

维护者机器上的实际版本(用
`pip freeze | grep -Ei "^(jax|orbax|flax|optax|chex|craftax|numpy|ml_dtypes)"` 复查):

```
python 3.11.15
jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt  0.6.2
orbax-checkpoint 0.11.18   flax 0.10.7   optax 0.2.5   chex 0.1.89
craftax 1.4.5   numpy 2.5.1   ml_dtypes 0.5.4
```

> **读别人的 checkpoint 时的兼容问题。** 报
> `sharding passed to deserialization should be specified... Got None`
> **不是文件坏了**,是 orbax 版本行为差异。真正要对齐的只有
> `orbax-checkpoint==0.11.18` 和 `jax==0.6.2`;numpy / ml_dtypes 的小版本不一致没关系。

杂项:`.env` 里如果有 `WANDB_API_KEY=unused` 删掉;Craftax 贴图缓存损坏时删掉
`texture_cache.pbz2` 让它重建。

### 1.3 Ollama(teacher 模型 + embedding)

```bash
export OLLAMA_MODELS=/workspace/ollama_models
ollama serve > /workspace/ollama_server.log 2>&1 &
ollama pull qwen2.5-coder:14b     # hydra 里叫 local_qwen14b
ollama pull nomic-embed-text
```

训练环境里要设 `OLLAMA_KEEP_ALIVE=-1`,否则模型在 session 之间被卸载,每个 session
都要重新加载一次。

### 1.4 开跑前的烟测

```bash
cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate
python -c "import jax; print(jax.devices())"                 # 应列出 GPU
curl -s http://localhost:11434/v1/models | head -c 200        # ollama 活着
```

---

## 2. 启动训练

```bash
FORK=/workspace/mechanism_UED/dicode_src/outputs/<run_name>
mkdir -p $FORK/rl_checkpoints

tmux new-session -d -s <run_name> "cd /workspace/mechanism_UED/dicode_src && \
source /workspace/venv/bin/activate && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 OLLAMA_KEEP_ALIVE=-1 \
  GENERATION_SERVER_URL=http://localhost:11434/v1 \
  EMBEDDING_SERVER_URL=http://localhost:11434/v1 && \
timeout 27000 python experiments/training/run_dicode.py \
  hydra.run.dir=$FORK seed=1 \
  use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED \
  training.total_timesteps=2000000000 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  gen_manager.task_generator.max_tokens=8192 \
  gen_manager.env_generator.max_tokens=8192 \
  +skill_preflight.use_scheduler=true \
  +skill_preflight.use_preflight=true \
  +skill_preflight.mastery_threshold=0.2 \
  +skill_preflight.frontier_mode=prereq \
  +skill_preflight.prereq_threshold=0.3 \
  +skill_preflight.use_scaffold_gate=true \
  > /workspace/run_<run_name>.log 2>&1"
```

> 上面模板里传的是 `mastery_threshold=0.2`,这是我们历史上一直用的值,**不是模块默认值
> (`0.6`)**。只有在复现已有实验臂时才照抄 0.2;新实验请自己决定用哪个,并在记录里写明。

点火后 15 分钟看两枚指纹,确认血统正常再去睡:

```bash
grep -a "Restored Optimizer Step Count" /workspace/run_<run_name>.log   # 血统
grep -a "SIL-BC. phase start"          /workspace/run_<run_name>.log   # 仅 SIL 臂
```

### 关键 flag

| Flag | 默认 | 作用 |
|---|---|---|
| `+skill_preflight.use_scheduler` | 关 | 打开 SkillGraph 调度器 |
| `+skill_preflight.use_preflight` | 关 | 训练前过滤生成的任务 |
| `+skill_preflight.frontier_mode` | `prereq` | `prereq` = 只有当直接前置都过 `prereq_threshold` 时,该技能才可作为目标 |
| `+skill_preflight.mastery_threshold` | **0.6** | 成功率一旦超过它,该技能不再作为教学目标 |
| `+skill_preflight.prereq_threshold` | 0.3 | 每个直接前置要跨过的门槛 |
| `+skill_preflight.use_scaffold_gate` | 关 | 拦掉泄题的任务 |
| `+training.sil_coef` | 0(空操作)| SIL 的 BC 损失权重 |
| `+training.sil_buffer` | — | 黄金片段库路径 |
| `+training.sil_burn` | 48 | BC 损失前的 burn-in 步数(0 = 消融)|

注意 `skill_scheduler.py` 里的 `max_target_achievements = 6`:合格技能按**最难优先**排序后
截断到 6 个。**真正决定谁被教的往往是这个名额上限,而不是阈值**——在下"某个阈值改动没效果"
的结论之前,先去日志里看前沿清单:

```bash
grep -a "SkillGraph" /workspace/run_<run_name>.log | tail -5
```

---

## 3. 评测

官方协议:1024 个冻结的 held-out Craftax 世界,seed 0。

```bash
python experiments/training/eval_checkpoints.py \
  hydra.run.dir=/tmp/<tag> use_wandb=false seed=0 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  +eval.ckpt_root=<run_dir>/rl_checkpoints \
  "+eval.steps=[1700]" +eval.tag=<TAG> +eval.details=true \
  2>&1 | grep -aE "RESULT|EVAL_DONE"

python experiments/analysis/necro_verdict.py /tmp/<tag>/eval_<TAG>_seed0_details.json
python experiments/analysis/iron_triage.py   /tmp/<tag>/eval_<TAG>_seed0_details.json
```

三个坑:eval 的参数前面必须带 `+`;必须重复训练时同样的 `gen_manager` 覆盖;
这里 `use_wandb=false` 没问题,但**训练时不行**。

`necro_verdict` 打印死亡楼层分布、分层步数/伤害/交战比账本、死亡上下文
(近敌/远敌距离、食物、水)、击杀矩阵。
`iron_triage` 打印逐成就达成率,以及达成者与未达成者的回报差(溢价)。
**溢价是被选择偏置污染的上界**——达成者本来就是更强的 episode,不能读成
"教会这个技能就能涨这么多分"。

---

## 4. SIL 管道(可选)

先从 donor checkpoint 采集黄金片段,再克隆回去:

```bash
python experiments/analysis/sil_collect.py \
  hydra.run.dir=/tmp/collect use_wandb=false seed=0 \
  +sil.ckpt_root=<donor_run>/rl_checkpoints '+sil.step=400' \
  +sil.mode=descend \
  +sil.tag=TAG +sil.out=/workspace/golden_buffer +sil.rollouts=8
```

四种触发模式:

| mode | 触发条件 | 教什么 |
|---|---|---|
| `descend` | 首次进入 2 层 | 下楼 |
| `stay` | 在 2 层连续驻留 64 步 | 在楼下活下来 |
| `resource` | 渴态补水 | 资源管理 |
| `skill` | 目标成就首次翻转(`+sil.skill=MAKE_IRON_ARMOUR`)| 备料 → 合成的完整序列 |

然后训练时加 `+training.sil_coef=1.0 +training.sil_buffer=/workspace/golden_buffer`。
库上限 512 条,按 episode 回报择优淘汰。

**条件向量很关键。** 采集器必须喂官方 eval 同款的多热成就向量;全零向量是分布外输入,
会悄悄压掉你想采的行为。这个坑曾经产出过一个错误判决——**如果采集率看起来不合理,
先查条件向量。**

---

## 5. 运维军规(踩出来的)

**续跑**
- **严禁原地续跑。** 一律:新 `hydra.run.dir` + 把最新 checkpoint 拷成
  `rl_checkpoints/0`,再从被续的那个 run 拷 `task_graph.graphml` 和 `runtime_analysis/`。
  原地续跑会**静默丢 checkpoint**(段内计数器与 orbax 的 `latest_step` 守卫竞争),不报错。
- **所有 fork / 续跑都必须保持 `training.total_timesteps=2000000000`。**
  它定义 LR 退火 horizon;改了它就改了整条曲线,而且超过 horizon 之后没有 clamp 的退火
  会让学习率穿过零、把 Adam 变成**梯度上升**(这曾导致七连崩;clamp 修复已在本分支)。

**存档与关机**
- orbax 是异步写盘,滞后大约 20 分钟。到收线点时,**等最后一个 `Checkpointing` 目录
  真正落盘再 kill**。
- `tmux kill-session` 杀不干净。用 `pgrep -af run_dicode` 取 PID 直接 kill,然后再 pgrep 验尸。

**预算**
- 约 25 分钟一个 session。17 个 session 的臂给 `timeout 27000`;SIL 臂更慢,给 27000–30000。

**日志与分析**
- `global_env_steps` 在续跑时会重置——**横轴一律用 `session`**。
- 训练必须 `use_wandb=true`,`false` 的路径在训练侧是坏的。
- wandb 的 `scan_history(keys=[...])` 会命中一个坏 API 路径。**全量扫,客户端再过滤。**
- 不要传 `+validation=default`。

**改文件**
- 仓库里的 py 文件是 **CRLF**。脚本化改动必须 `open(..., newline='')`,否则会把所有行尾重写一遍。

---

## 6. 判据规程(最值得照抄的一节)

这一节改过我们自己好几个结论。

**这套系统实测的噪声地板(官方 eval):**

| 来源 | 量级 |
|---|---|
| checkpoint 抖动(某臂末 10 个 session)| SD 0.37 – 0.54 |
| 同臂换 seed、位置对齐 | 1.13 分(取多 session 均值后 0.68)|
| 达成率针 — armour | 1.7 pp |
| 达成率针 — pickaxe / sword | 5 – 6 pp |
| 2 层驻留步数 | 效应 ~17 vs 噪声 ~4 |

由此四条纪律:

1. **单具尸体只是观察量,不是判决。** 分数一律报"末 N 个 session 的均值"或
   相邻三具尸体的平均。这一步是免费的,而且能把分辨 1 分所需的 seed 数从 8 降到 4。
2. **优先用行为针。** 2 层驻留在单 seed 下信噪比约 4;分数只有 0.9,分辨不了我们做的任何事。
3. **点火前预注册判据**——指标、阈值、以及"反向结果意味着什么"。
   我们有两个很漂亮的发现是这样死掉的,而且死得对。
4. **null 只有在干预确实到达被测路径时才算数。** 下"X 没用"的结论之前,
   先验证 X 真的改变了它该改变的东西(例如某个课程 flag 是否真的改变了前沿清单)。

---

## 7. 东西都在哪

- 实验跟踪:wandb 项目 `mechanism_UED/Skill_Preflight_UED`
- 训练输出:`dicode_src/outputs/<run_name>/rl_checkpoints/`
- 黄金库:`/workspace/golden_*`
- 日志:`/workspace/run_<run_name>.log`
- 设计卡、周报与交接文档和本文件放在一起,推导细节在 SIL 设计卡里。

### 参照实验臂

| 臂 | wandb id | 末 session |
|---|---|---|
| fork 源 / resumeAB | `2oyy46uv` | 154 |
| 从头跑 2e9,阈值 0.2 | `hdodsb5l` | 66 |
| 14B-only 基线(3e8,无 skill_preflight)| `mc75k0nx` | 23 |
| 阈值 0.4(3e8)| `506cdgz3` | 37 |
| 阈值 0.6(晚窗 fork)| `w8mlwsi8` | 172 |
| placebo seed-1 | `5qnqbjjb` | 186 |
| placebo seed-2 | `r369pox1` | 174 |
| phi tail | `gejgawhc` | 175 |
| SIL,双 donor 库 | `2awp5kbt` | 170 |
| SIL,纯基线库 | `vnpcrp0y` | 172 |
| LOCK seed-1 | `5dfg8rr1` | 174 |
| SKILL(armour 库)| `pgh95yfl` | 167 |

> 这些在 wandb 上**全都显示 `state: crashed`,这是正常的**——被 `timeout` 收掉的训练进程
> 就是这样退出的,checkpoint 完好无损。唯一一次真正的失败是 `5qnqbjjb`:它跑过了退火
> horizon,回报从 session 172 的 42.19 崩到 session 173 的 −0.90。那就是第 5 节说的
> "学习率穿过零"的故障,也正是本分支的 clamp 修复要防的东西。

有问题找 Mason。第 5 节没覆盖到的坑,踩到了告诉我,我加进去。
