# Craftax 人类试玩 & "参考答案"核实

> 核实日期 2026-07-02。来源:官方仓库 [github.com/MichaelTMatthews/Craftax](https://github.com/MichaelTMatthews/Craftax)（README / tutorial.md / play_craftax.py 源码）、论文 [arXiv:2402.16801](https://arxiv.org/abs/2402.16801)。

---

## 一、结论：Craftax 有没有"参考答案"？

| 问题 | 答案 |
|---|---|
| **能人肉试玩吗** | ✅ 有，官方内置 `play_craftax`（完整版）/ `play_craftax_classic`（classic 版） |
| **有人类高分/通关路径吗** | ✅ 有一份"混合技能人类轨迹数据集"（官方 Google Drive）。其中 **run1 是唯一一条通关（拿全成就）的轨迹**。查看需用 craftax **v1.1.0 或更早** |
| **有脚本 / 硬编码最优 solver 吗** | ❌ 没有。仓库里没有任何 hardcoded bot / scripted 最优策略 |
| **人类到底多难通关** | 前身 Crafter：一位熟知机制的作者，在**可无限暂停思考**的 GUI 里玩了**约 5 小时**才第一次打出全成就 perfect run。人类上限存在但极难 |

### 分数口径（易踩坑）
- **完整 Craftax** 官方 scoreboard 把成绩报成"占 max return **226** 的百分比"。
- **DiCode 报的 mean return 48.33 是绝对值口径**，不是这个百分比，别混。
- **Craftax-Classic** 的 max return = **22**，又是另一套口径，别混。
- 成就总数 = **67**（full 版）。

### 对我们研究的用途 & 红线
- ✅ **可用**：人肉试玩建立"深层成就（gnome warrior / diamond gear / 后几层 / 魔法）到底多难连起来"的直觉；run1 通关轨迹可作**人类可达性证据**（证明深层成就人类能串起来，环境非不可解）；可做定性截图。
- ❌ **红线**：这条人类轨迹**不能当 RL 的"参考答案 / 最优策略"注入或对照**。我们对标 DiCode 是**纯 RL from scratch，不做模仿学习**；一旦引入人类先验（BC / 示范）就**改了 problem setting**，与 48.33 基线不可比，reviewer 会直接质疑。只适合"难度直觉 / 可达性论证 / 定性图"，不进 pipeline。

---

## 二、从零开始试玩操作步骤

> 前提：`play_craftax` 需要 **pygame 弹一个 GUI 窗口**（源码 `import pygame` 必需）。因此**最省事在本机 Windows 跑**（有原生显示）。WSL / Oscar 需要 X server，见文末"其它环境"。
> ⚠️ craftax 官方**只支持 Python 3.9–3.12**。本机默认是 3.13，**不能直接装**，必须先建一个 3.12 环境。

### 方式 A：本机 Windows（推荐，最省事）

**1. 建一个 Python 3.12 独立环境**（二选一）

conda（若已装 miniforge/anaconda）：
```powershell
conda create -n craftax python=3.12 -y
conda activate craftax
```
或 venv（需先有一个 3.12 解释器；没有就用 conda 那条）：
```powershell
py -3.12 -m venv C:\craftax_env
C:\craftax_env\Scripts\Activate.ps1
```

**2. 安装 craftax（CPU 就够玩，人玩不需要 GPU）**
```powershell
pip install craftax
```
> pygame 会作为依赖一起装上。若没自动装，手动 `pip install pygame`。

**3. 启动完整版试玩**
```powershell
play_craftax
```
classic 版：`play_craftax_classic`

**4. 等编译**（不是卡死）
> JAX 首次编译：约 **30s 出第一帧**，再约 **20s 才能走第一步**。控制键会在开玩时打印在终端。

---

## 三、键位表（源自 play_craftax.py 源码，完整版）

### 移动 & 核心交互
| 键 | 动作 |
|---|---|
| `W` / `A` / `S` / `D` | 上 / 左 / 下 / 右 移动 |
| `SPACE` | DO（交互：采集 / 攻击 / 喝水 / 吃等，朝向前方） |
| `Q` | NOOP（原地不动，跳过一帧） |
| `TAB` | 睡觉 Sleep |
| `E` | 休息 Rest |
| `,`（逗号） | 上楼 Ascend（回上一层 / overworld） |
| `.`（句号） | 下楼 Descend（下到地牢深层） |

### 放置
| 键 | 动作 |
|---|---|
| `T` | 放工作台 Table |
| `R` | 放石头 Stone |
| `F` | 放熔炉 Furnace |
| `P` | 放植物 Plant |
| `J` | 放火把 Torch |

### 制作 — 镐 / 剑（需邻近工作台等）
| 键 | 动作 | 键 | 动作 |
|---|---|---|---|
| `1` | 木镐 | `5` | 木剑 |
| `2` | 石镐 | `6` | 石剑 |
| `3` | 铁镐 | `7` | 铁剑 |
| `4` | 钻石镐 | `8` | 钻石剑 |

### 制作 — 装备 / 弹药 / 火把
| 键 | 动作 |
|---|---|
| `Y` | 铁甲 Iron Armour |
| `U` | 钻石甲 Diamond Armour |
| `O` | 造箭 Make Arrow |
| `I` | 射箭 Shoot Arrow |
| `[` | 造火把 Make Torch |

### 魔法 / 附魔 / 书（深层内容）
| 键 | 动作 |
|---|---|
| `G` | 火球 Cast Fireball |
| `H` | 冰球 Cast Iceball |
| `M` | 读书 Read Book |
| `K` | 附魔剑 Enchant Sword |
| `L` | 附魔甲 Enchant Armour |
| `;` | 附魔弓 Enchant Bow |

### 喝药水（6 色）
| 键 | 药水 | 键 | 药水 |
|---|---|---|---|
| `Z` | 红 Red | `V` | 粉 Pink |
| `X` | 绿 Green | `B` | 青 Cyan |
| `C` | 蓝 Blue | `N` | 黄 Yellow |

### 属性升级（RPG 深层机制）
| 键 | 动作 |
|---|---|
| `]` | 升敏捷 Dexterity |
| `-` | 升力量 Strength |
| `=` | 升智力 Intelligence |

> 大致进程直觉：木→石（1、2、5、6 造基础工具）→ 挖矿下楼（`.`）→ 铁/煤/熔炉（`F`、3、7）→ 钻石（4、8）→ 深层地牢的战斗 / 魔法 / 附魔（`G`/`H`/`K`/`L`/属性）。越往后 = DiCode / 我们 auction 要造垫脚石的那些坑。`tutorial.md` 有官方通关教程。

---

## 三点五、工具制作配方（源码 game_logic.py `do_crafting` 实测，2026-07-02）

> ★ **最容易卡的坑：制作判定是"相邻(near)工作台"，不是"站在工作台上"。** 把工作台放下后，人物常站在离它一格远或斜角的位置，那就不算 near，按键没反应。
> **正确做法：走到工作台的正上/下/左/右紧邻一格（不是斜对角、不是隔一格），再按制作键。**

| 物品 | 键 | 材料需求 | 站位要求 | 额外限制 |
|---|---|---|---|---|
| 木镐 / 木剑 | `1` / `5` | 木 ×1 | 挨工作台 | 身上还没镐/剑（`pickaxe<1` / `sword<1`） |
| 石镐 / 石剑 | `2` / `6` | 木 ×1 + 石 ×1 | 挨工作台 | — |
| 铁镐 / 铁剑 | `3` / `7` | 木 ×1 + 石 ×1 + 铁 ×1 + 煤 ×1 | **同时**挨工作台 **和** 熔炉 | — |
| 钻石镐 | `4` | 木 ×1 + 钻石 ×3 | 挨工作台 | — |
| 钻石剑 | `8` | 木 ×1 + 钻石 ×2 | 挨工作台 | — |

**要点：**
- **木镐/木剑不需要石头**，只要 ≥1 木头 + 挨工作台即可，一下就成。按 1/5 没反应 = 十有八九没紧贴工作台（或已经有镐/剑了）。
- **铁装那步最容易卡**：`is_at_crafting_table AND is_at_furnace` 两个相邻条件要**同时**满足 → **把工作台和熔炉放在一起**，让你站的那格同时相邻两者，再按 3/7。这也正是我们研究里"深层前置链"开始变难的起点。
- 采集木头：面朝树按 `SPACE`（DO），每次 +1 木。

---

## 三点六、楼层结构 + 钻石/魔法完整获取链（源码 game_logic.py / constants.py 实测，2026-07-02）

### 楼层结构（`.`=下潜 DESCEND，`,`=上升 ASCEND）
| Floor | 名称 | 备注 |
|---|---|---|
| 0 | overworld 地表 | 起点。**没有钻石/宝石/书** |
| 1 | gnomish mines 侏儒矿 | 开始有矿 |
| 2 | dungeon 地牢 | |
| 3 | sewers 下水道 | ★ **宝箱掉书**（学法术的书只在 floor 3/4 掉） |
| 4 | vaults 宝库 | ★ **宝箱掉书** |
| 5 | troll mines 巨魔矿 | |
| 6 | fire 火之领域 | 火附魔台 |
| 7 | ice 冰之领域 | 冰附魔台 |
| 8 | boss | necromancer 首领 |

> 每下一层新楼层 **player_xp += 1**（属性升级的唯一来源）。

### Tier 3：钻石装备
- **挖钻石硬条件：`pickaxe >= 3`（必须有铁镐）**。钻石矿只在地下楼层，地表没有。
- 配方（挨工作台）：钻石镐 `4` = 钻石×3 + 木×1；钻石剑 `8` = 钻石×2 + 木×1；钻石甲 `U` = 钻石×3。

### Tier 4：魔法（三条支线，都不是"造"出来的）
**A. 学 & 放法术（fireball / iceball）**
1. 下到 **Floor 3(sewers) 或 4(vaults)**，宝箱才掉**书**（源码 `is_looting_book` 条件 = `player_level==3 or ==4`）。
2. 有书后按 **`M`(READ_BOOK)** → **随机学到 fireball 或 iceball 之一**（每本书学一个未学的；两个都要=读两本）。
3. 学会后才能 **`G`(火球)/`H`(冰球)** 施放（`is_casting AND learned_spells[i]`），消耗 mana。没学过按了无效。

**B. 附魔（enchant sword/armour/bow）**
1. 需**宝石** sapphire(蓝)/ruby(红)，**挖宝石要 `pickaxe >= 4`（钻石镐）**。
2. 站**附魔台**（火/冰版，在 floor 6/7）旁，身上有对应武器 + 宝石×1 + mana≥9。
3. 按 **`K`(附魔剑)/`L`(附魔甲)/`;`(附魔弓)**。

**C. 属性升级（力量/敏捷/智力）**
- 条件 **`player_xp >= 1`**；XP 靠**每下一层新楼层 +1**。
- 攒到后按 **`-`(力量)/`]`(敏捷)/`=`(智力)**，每次花 1 XP。

### 一句话链条
铁镐 → 下矿挖钻石 → 钻石镐 → 挖 sapphire/ruby；同时下 Floor 3–4 开宝箱拿书 → `M` 读书学法术 → `G/H` 放 → 到附魔台 `K/L` 附魔。**这整段 tier3→tier4 正是 baseline 崩到 0%、DiCode / 我们 auction 要造垫脚石的核心区间**（tier 划分见 `auction/craftax_achievements.py` DEPTH_TIERS：铁装=tier2、钻石装=tier3、法术/附魔/属性=tier4）。

---

## 四、其它环境（不推荐，仅备查）

- **WSL（Ubuntu）**：无原生显示，需装 X server（Windows 侧跑 VcXsrv/X410 并 `export DISPLAY=...`），pygame 才能弹窗。折腾，不如直接本机 Windows。
- **Oscar（登录节点）**：禁跑计算 + 无显示，**不要在 Oscar 玩**。要玩就本机。
- 若只想**看** run1 通关轨迹（而非自己玩）：装 craftax **v1.1.0 或更早**，从官方 Google Drive 下轨迹数据集回放。

---

## 相关记忆
- 记忆 `craftax-human-play-and-reference-trajectories`（本文档的精简索引版）
- 成就集 67 个：记忆 `craftax-achievement-set-67`
- 分数口径 / 官方评测：记忆 `dicode-step-semantics-and-official-eval`
