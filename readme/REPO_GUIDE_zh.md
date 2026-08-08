# Skill-Preflight UED 仓库使用说明

分支:`skill-preflight-ued_Mason`(`gregjones11235/mechanism_UED`)
维护:Mason · 最后更新 2026-08-08 · 英文版见 `REPO_GUIDE.md`

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

# pyproject.toml 没有任何版本约束 → 新 venv 会解析到"当天最新",而裸 jax/jaxlib 是
# CPU wheel。必须手工钉:
pip install "jax[cuda12]==0.6.2" jaxlib==0.6.2 orbax-checkpoint==0.11.18 \
            flax==0.10.7 optax==0.2.5 chex==0.1.89 craftax==1.4.5

python -c "import jax; assert jax.default_backend()=='gpu'; print(jax.devices())"
```

> `uv.lock` 的版本对得上,但**不含 cuda plugin**,单跑 `uv sync` 仍是 CPU-only。

维护者机器上的实际版本(用
`pip freeze | grep -Ei "^(jax|orbax|flax|optax|chex|craftax|numpy|ml_dtypes)"` 复查):

```
python 3.12.3
jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt  0.6.2
orbax-checkpoint 0.11.18   flax 0.10.7   optax 0.2.5   chex 0.1.89
craftax 1.4.5   numpy 2.5.1   ml_dtypes 0.5.4
```

> **读别人的 checkpoint 时的兼容问题。** 报
> `sharding passed to deserialization should be specified... Got None`
> **不是文件坏了**,是 orbax 版本行为差异。真正要对齐的只有
> `orbax-checkpoint==0.11.18` 和 `jax==0.6.2`;numpy / ml_dtypes 的小版本不一致没关系。
> 另一个同样报法的原因是**设备/拓扑不匹配**(在 GPU 写的 ckpt 上强制 `JAX_PLATFORMS=cpu`
> 可逐字复现该报错)——**ckpt 一律在 GPU 上 restore**。
>
> **`craftax==1.4.5` 是科学关键的那一个 pin。** orbax/jax 决定你能不能*读*别人的 ckpt;
> craftax 版本不同就是环境不同,分数**不可比**(不是"读不出",是"不能比")。

杂项:`.env` 里如果有 `WANDB_API_KEY=unused` 删掉;Craftax 贴图缓存损坏时删掉
`texture_cache.pbz2` 让它重建。

### 1.3 Ollama(teacher 模型 + embedding)

`OLLAMA_KEEP_ALIVE` / `OLLAMA_CONTEXT_LENGTH` / `OLLAMA_MODELS` 都是**服务端**变量,
写在训练环境里**无效**。每张卡起一个 server:

```bash
pkill ollama; sleep 2; pgrep -af ollama          # 必须为空 —— 见下面的端口陷阱

nohup env OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES=0 \
     OLLAMA_KEEP_ALIVE=-1 OLLAMA_CONTEXT_LENGTH=32768 \
     OLLAMA_MODELS=/root/ollama_models ollama serve > /root/ollama_0.log 2>&1 &
sleep 6 && grep -E "OLLAMA_KEEP_ALIVE|OLLAMA_CONTEXT_LENGTH|OLLAMA_MODELS" /root/ollama_0.log

OLLAMA_HOST=127.0.0.1:11434 ollama pull qwen2.5-coder:14b   # hydra 名 local_qwen14b
OLLAMA_HOST=127.0.0.1:11434 ollama pull nomic-embed-text
grep "offloaded 49/49" /root/ollama_0.log        # 满层上卡;不是 49/49 就别开跑
```

第二张卡改用 `OLLAMA_HOST=127.0.0.1:11435 CUDA_VISIBLE_DEVICES=1`,并把该臂的
`GENERATION_SERVER_URL` / `EMBEDDING_SERVER_URL` 指到 `:11435`。

> **端口陷阱(吃过 9GB 下错盘的亏,而且零报错)。** 已有 server 占着 `:11434` 时,第二个
> `serve` 会以 `bind: address already in use` 退出,而随后的 `pull` 会**静默连到旧 server**、
> 继承它的 `OLLAMA_MODELS`。所以永远:pkill → 确认 `pgrep` 为空 → 起 → grep 出那三个变量
> → 才 pull。
>
> `ollama pull` 是**经由 server** 下载的,重启 `serve` 会中断进行中的 pull;训练期
> `/embeddings`、`/chat` 的 Retry 日志是良性的。
>
> 单卡显存账:trainer ~61.3 GB + ollama ~15.9 GB ≈ 77 GB / 80 GB —— 更大的 teacher
> 本地放不下。

### 1.4 开跑前的烟测

```bash
cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate
python -c "import jax; print(jax.devices())"                 # 应列出 GPU
python -c "import craftax; print(craftax.__version__)"       # 必须 1.4.5
curl -s http://localhost:11434/v1/models | head -c 200        # ollama 活着
python -c "from dicode.skill_preflight.prereq_graph import DIRECT_PREREQS as G; import hashlib,json; \
print(hashlib.md5(json.dumps({k:sorted(v) for k,v in sorted(G.items())}).encode()).hexdigest()[:12])"
# 期望 cfbb1c9a4558(未打乱的原图)—— 见 §6b
```

---

## 2. 启动训练

```bash
FORK=/workspace/mechanism_UED/dicode_src/outputs/<run_name>
mkdir -p $FORK/rl_checkpoints

tmux new-session -d -s <run_name> "cd /workspace/mechanism_UED/dicode_src && \
source /workspace/venv/bin/activate && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
  CUDA_VISIBLE_DEVICES=0 \
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
  gen_manager.task_generator.base_url=http://localhost:11434/v1 \
  gen_manager.env_generator.base_url=http://localhost:11434/v1 \
  gen_manager.embedding_model.base_url=http://localhost:11434/v1 \
  dicode_manager.additional_num_parents=15 \
  +skill_preflight.use_scheduler=true \
  +skill_preflight.use_preflight=true \
  +skill_preflight.mastery_threshold=0.6 \
  +skill_preflight.frontier_mode=prereq \
  +skill_preflight.prereq_threshold=0.3 \
  +skill_preflight.use_scaffold_gate=true \
  +skill_preflight.r3_mastered_exemption=true \
  > /root/run_<run_name>.log 2>&1"
```

> **双卡。** `CUDA_VISIBLE_DEVICES` 是**每臂**的,旧模板漏了;双卡请分别设 `0` / `1`,
> 并把每臂指向自己那个 ollama 端口。
>
> **base_url 也要作为 hydra override 传一份。** 环境变量**不会**写进
> `.hydra/overrides.yaml`;传进 hydra 才能让"这一臂用的哪个 teacher 端点"在几个月后
> 还查得出来。
>
> **阈值。** 上面这组是 2026 夏季的基座栈:`mastery_threshold=0.6` + `r3_mastered_exemption`。
> 早期臂用的是 `0.2`;阈值线已结案 —— 0.6 vs 0.2 从头跑,分数落在噪声带内,唯一的真实
> 差别是 0.6 使中层制造波动约翻倍。用哪个都行,但要写明。
>
> **上游原版基线** = 去掉全部 `+skill_preflight.*`,并把
> `dicode_manager.additional_num_parents` 改回 **2**(模块默认)。实测墙钟:全栈 2e9
> 每臂 60.3–66.0 h;原版约 52 h。
>
> **ckpt 要落在容器盘。** `/workspace` 是 MooseFS:缓冲写快,但 `fsync` 比本地盘慢约
> 1000×。规矩是 venv 与 repo 放 `/workspace`,`hydra.run.dir` 与 `OLLAMA_MODELS` 放
> `/root`,每 30 分钟 rsync 回卷(容器盘每次 pod 重启清零)。卡死自检:
> `timeout 10 ls /workspace >/dev/null 2>&1; echo $?`(124 = 挂住)。

点火后 15 分钟看两枚指纹,确认血统正常再去睡:

```bash
grep -a "Restored Optimizer Step Count" /root/run_<run_name>.log   # 血统
grep -a "SIL-BC. phase start"          /root/run_<run_name>.log   # 仅 SIL 臂
grep -a "offloaded 49/49"              /root/ollama_0.log         # teacher 满层上卡
grep -a "\[Preflight\] ERROR (kept all, gate inactive!)" /root/run_<run_name>.log   # 必须为空
```

最后一条是"最贵的沉默":配置里缺 `validation` 时 preflight 会抛异常,而异常被吞成这
一行 —— 门**静默失效**、所有生成关卡照单全收,这一臂看着健康但在科学上已经作废。

### 关键 flag

| Flag | 默认 | 作用 |
|---|---|---|
| `+skill_preflight.use_scheduler` | 关 | 打开 SkillGraph 调度器 |
| `+skill_preflight.use_preflight` | 关 | 训练前过滤生成的任务 |
| `+skill_preflight.frontier_mode` | `prereq` | `prereq` = 只有当直接前置都过 `prereq_threshold` 时,该技能才可作为目标 |
| `+skill_preflight.mastery_threshold` | **0.6** | 成功率一旦超过它,该技能不再作为教学目标 |
| `+skill_preflight.prereq_threshold` | 0.3 | 每个直接前置要跨过的门槛 |
| `+skill_preflight.use_scaffold_gate` | 关 | 拦掉泄题的任务 |
| `+skill_preflight.r3_mastered_exemption` | 关 | 允许规则 3 发放**已掌握**的直接前置 |
| `dicode_manager.additional_num_parents` | **2** | 每周期额外从 archive 抽的父任务数;基座栈用 **15**(≈25 父任务/周期,实测每个 2e9 臂 1899–1900 个任务)|
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
- **所有 fork / 续跑都必须保持 `training.total_timesteps=2000000000`,并把学习率设成
  该位置应有的值**(或还原退火计数)。optimizer reset 会把退火计数归零,于是 fork 全程
  跑满初始 LR,而该位置只需要约 1/10;该臂随后会在**总步数的固定位置**崩溃,死状固定:
  `value_loss → 1.5e10`、`grad_norm → 2.5e10`、`entropy → 0`、`return ≡ −0.90`。
  <br>**对旧版说法的更正:** 此前写作"未 clamp 的退火使 LR 穿过零 → Adam 变成梯度上升"。
  专门的崩溃测定台已将其证伪:值损失归一、critic/骨干梯度防火墙、退火 clamp 本身,
  三个候选全部仍死在同位;而 1/10 学习率的对照臂**穿越死刑位存活**。clamp 作为无害护栏
  保留,但 commit `ff6b956` 的 "ROOT CAUSE" 措辞需要勘误。

**存档与关机**
- orbax 是异步写盘,滞后大约 20 分钟。到收线点时,**等最后一个 `Checkpointing` 目录
  真正落盘再 kill**。
- `tmux kill-session` 杀不干净,会留下**孤儿进程继续占卡**(曾因此白占一张卡两小时)。
  用 `kill $(pgrep -f "outputs/<臂名>")` 按 PID 杀,再确认 `pgrep` 为空、显存掉下来,
  才能启动下一臂。

**预算**
- 约 25 分钟一个 session。17 个 session 的臂给 `timeout 27000`;SIL 臂更慢,给 27000–30000。

**日志与分析**
- `global_env_steps` 在续跑时会重置——**横轴一律用 `session`**。
- 训练必须 `use_wandb=true`,`false` 的路径在训练侧是坏的。
- wandb 的 `scan_history(keys=[...])` 会命中一个坏 API 路径。**全量扫,客户端再过滤。**
- 不要传 `+validation=default`。
- **循环内的 `evaluation/*` 曾读训练 env 的槽位**,导致任何被 reward wrapper 包裹的臂
  return 虚高而成就位干净(实测 +3.4,与 `2.0 × (0.86 + 0.73)` 分毫吻合)。现已修:每
  session 真跑独立 held-out 评测上报 `evaluation/*`,训练侧指标改名 `evaluation_shaped/*`,
  并有结构测试钉住。看到两者混用请当 bug。
- **`ep_len` 单独不作证据**:同配置两臂可差 55–82%,而 return 只差 0.30–1.13。

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
| 同配置重跑、早窗单个离线 ckpt | ±2.31 |
| **带宽随训练阶段变化** | **技能起飞窗内 ~7;约 s15 之后 0.3 – 1.1** |
| 达成率针 — armour | 1.7 pp |
| 达成率针 — pickaxe / sword | 5 – 6 pp |
| 2 层驻留步数 | 效应 ~17 vs 噪声 ~4 |

带宽**不是常数**:六个预算点上测出来是一个**峰** —— 起飞前近零,一臂铁链点着而另一臂
没点着时冲到 ~7,之后收敛到 ~1。**每个差值都要放在它所处阶段的带宽上读**:早窗的差距
无论看起来多大,基本都不可裁决。交叉验证:同一臂三个 seed 两两差均值 1.05,对照论文的
SE × √n 也落在 1.1–1.4。

由此五条纪律:

1. **单具尸体只是观察量,不是判决。** 分数一律报"末 N 个 session 的均值"或
   相邻三具尸体的平均。这一步是免费的,而且能把分辨 1 分所需的 seed 数从 8 降到 4。
2. **优先用行为针。** 2 层驻留在单 seed 下信噪比约 4;分数只有 0.9,分辨不了我们做的任何事。
3. **点火前预注册判据**——指标、阈值、以及"反向结果意味着什么"。
   我们有两个很漂亮的发现是这样死掉的,而且死得对。
4. **null 只有在干预确实到达被测路径时才算数。** 下"X 没用"的结论之前,
   先验证 X 真的改变了它该改变的东西(例如某个课程 flag 是否真的改变了前沿清单)。
5. **只有离线、多 ckpt 的评测有裁决权。** 训练期曲线在一季里误导过我们四次:三次是
   噪声(方向或量级与离线复测相反),一次是结构性的(上面那个泄漏)。训练期曲线用来看
   "它还活着",不用来判断"什么是真的"。

---

## 6b. 先验图与打乱图消融

`src/dicode/skill_preflight/prereq_graph.py` 里的 `DIRECT_PREREQS` 是 67 个 Craftax 成就
的一跳前置图(94 条边),**同时**供调度器(`frontier_mode=prereq`)与 scaffold gate 使用
—— 改动它会一次移动两个机制。

文件内含一段**环境变量门控的打乱图补丁**:不设 `SP_SHUFFLE_PREREQ` 时**逐字无操作**;
设了(如 `SP_SHUFFLE_PREREQ=20260804`)则用双边交换做随机重连,保持节点集合、边数、
每个节点的入度与出度、以及无环性 —— 只打乱"谁门着谁"。import 时打印指纹行:

```
[PREREQ-SHUFFLE] seed=20260804 nodes=67 edges=94 swaps=1880/2566 changed=79/94 degseq=OK
```

图指纹:原图 `cfbb1c9a4558`,seed 20260804 的打乱图 `1354f4e59b14`。自检:

```bash
python -c "from dicode.skill_preflight.prereq_graph import DIRECT_PREREQS as G; import hashlib,json; \
print(len(G), sum(len(v) for v in G.values()), \
hashlib.md5(json.dumps({k:sorted(v) for k,v in sorted(G.items())}).encode()).hexdigest()[:12])"
# 未设变量时 -> 67 94 cfbb1c9a4558
```

**不变量:边表必须 `sorted()`。** 首炉曾因遍历 `frozenset` 受 `PYTHONHASHSEED` 影响,
**每个进程各得一张不同的图**;指纹行当场抓获,该臂作废重跑。请保留这个排序,并要求任何
打乱臂的日志里 grep 得到那行指纹 —— 没有指纹,这一臂不算数。

动阈值之前值得知道的一个结构事实:在真图 + `prereq_threshold=0.3` 下,**有 27 个成就
整季"永不合法"**,因为它们某个门在我们跑过的所有臂里的历史最高掌握率都够不到门槛
(例如整个二层战斗簇挂在 `enter_gnomish_mines` 之后,而后者峰值只有 0.12)。
**正确而保守的知识 + 对可达性盲目的门控 = 一把锁**,锁住的恰恰是最需要课程的那批技能。
请把"图是对的"和"图对调度器是可用的"当成两件事。

---

## 7. 东西都在哪

- 实验跟踪:wandb 项目 `mechanism_UED/Skill_Preflight_UED`
- 训练输出:`dicode_src/outputs/<run_name>/rl_checkpoints/`
- 黄金库:`/workspace/golden_*`
- 日志:`/workspace/run_<run_name>.log`
- preflight 成本工作(基线、判据、陷阱):[PREFLIGHT_COST.md](PREFLIGHT_COST.md)
- 设计卡、周报与交接文档和本文件放在一起,推导细节在 SIL 设计卡里。
- **数字口径争议一律以数据总账(2026-08-07 收官版,Mason 处)为准**;本文件与总账冲突时,
  以总账为准。

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
