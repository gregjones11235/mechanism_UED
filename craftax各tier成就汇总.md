# Craftax (full) 各 tier 成就汇总 —— 官方权威划分

> 记录 2026-07-06。**权威来源（三方交叉核实，完全一致）**：
> 1. `craftax==1.4.5` 本地源码 `constants.py` 的 `achievement_mapping()` + 两个列表；
> 2. **craftax 官方 GitHub `main` 分支** `MichaelTMatthews/Craftax/.../constants.py`（联网核实）；
> 3. **DiCode.pdf 论文附录 page 24-25 逐行成就表 + page 70 直接复制的源码**。
> 这正是 **DiCode baseline 实际算 reward 的定义**，即 SOTA 48.33 / wandb `skill_*` 背后的口径。
> ⚠️ 与我们自己代码 `dicode_v6/auction/craftax_achievements.py` 的 `DEPTH_TIERS`（手工划的 1-4 层）
> **不一致**——那是 design choice、且**已确认划错**（魔法、diamond 装备都归错档，见 §5）。
> **横比 baseline / 论文口径时，一律以本文档的官方四档为准。**
>
> 📌 **两个高频困惑点，官方权威答案**：
> - **魔法 fireball/iceball = Advanced（第3档，+5）**，比钻石装备深一档。
> - **diamond 跨两档**：`collect_diamond`(采) = Basic 第1档；`make_diamond_*`(钻石装备剑/镐/甲) =
>   **Intermediate 第2档**。**钻石装备不在第3档**（此前误以为 diamond 是 tier3，官方是第2档）。

---

## 0. 官方分档机制（reward 权重决定档位）

`achievement_mapping(value)`（constants.py:519）：
```
value <= 24                       -> reward 1  = BASIC
value in INTERMEDIATE_ACHIEVEMENTS -> reward 3  = INTERMEDIATE
value in VERY_ADVANCED_ACHIEVEMENTS-> reward 8  = VERY_ADVANCED
其余（else）                        -> reward 5  = ADVANCED
```
- **官方只有四档，名字是 BASIC / INTERMEDIATE / ADVANCED / VERY_ADVANCED**（无 "tier1-4" 叫法）。
- 若按深浅顺序编号 1→4：BASIC=第1档、INTERMEDIATE=第2档、**ADVANCED=第3档**、VERY_ADVANCED=第4档。
- reward 权重 1/3/5/8 就是深度权重（越深单次成就给分越高，是 mean_return 的构成）。
- ★ **魔法 learn/cast fireball/iceball 在 ADVANCED（第3档，reward 5）**，不在第4档，也不与钻石装备同档
  （钻石装备在 INTERMEDIATE 第2档）。深度上：钻石装备(第2) < 魔法(第3) < realm/necromancer(第4)。

---

## 1. BASIC（reward 1，第1档）— 25 个
早期：木石基础、易怪、生存、**整条铁器链到 iron_sword**、钻石采集、火把。
> 注意：`collect_diamond`(19)、`make_iron_pickaxe`(20)、`make_iron_sword`(21) 都在 BASIC ——
> **make_iron_pickaxe 是 BASIC，不是 tier3**（此前我误判为 tier3，纠正）。

```
0 collect_wood      1 place_table       2 eat_cow          3 collect_sapling
4 collect_drink     5 make_wood_pickaxe 6 make_wood_sword  7 place_plant
8 defeat_zombie     9 collect_stone    10 place_stone     11 eat_plant
12 defeat_skeleton 13 make_stone_pickaxe 14 make_stone_sword 15 wake_up
16 place_furnace   17 collect_coal    18 collect_iron    19 collect_diamond
20 make_iron_pickaxe 21 make_iron_sword 22 make_arrow     23 make_torch
24 place_torch
```

## 2. INTERMEDIATE（reward 3，第2档）— 18 个
中游：钻石/宝石装备、铁甲、首次深层下降（gnomish_mines/dungeon）、gnome/orc 战斗、弓、宝箱、药水。

```
25 make_diamond_sword  26 make_iron_armour    27 make_diamond_armour
28 enter_gnomish_mines 29 enter_dungeon
36 defeat_gnome_warrior 37 defeat_gnome_archer 38 defeat_orc_solider  39 defeat_orc_mage
50 eat_bat             51 eat_snail
52 find_bow            53 fire_bow
54 collect_sapphire    59 collect_ruby        60 make_diamond_pickaxe
61 open_chest          62 drink_potion
```

## 3. ADVANCED（reward 5，第3档）— 15 个 ★"tier3"就是这一档
后期：更深楼层（sewers/vault/troll_mines）、中层战斗（lizard/kobold/troll/deep_thing/knight/archer）、
**魔法（fireball/iceball 学习+施放）**、附魔。

```
30 enter_sewers      31 enter_vault       32 enter_troll_mines
40 defeat_lizard     41 defeat_kobold     42 defeat_troll        43 defeat_deep_thing
55 learn_fireball    56 cast_fireball     57 learn_iceball       58 cast_iceball
63 enchant_sword     64 enchant_armour
65 defeat_knight     66 defeat_archer
```

## 4. VERY_ADVANCED（reward 8，第4档）— 9 个
最深：火/冰领域、墓地、元素/necromancer/pigman 等终局怪。

```
33 enter_fire_realm  34 enter_ice_realm   35 enter_graveyard
44 defeat_pigman     45 defeat_fire_elemental 46 defeat_frost_troll
47 defeat_ice_elemental 48 damage_necromancer 49 defeat_necromancer
```

---

## 5. 对当前工作的含义
- **横比 baseline 的"tier3"= 官方 ADVANCED 档（第3档，15 个，含魔法）**，不是我们代码 DEPTH_TIERS[3]。
- v6 攻坚焦点 `make_iron_pickaxe` 官方口径是 **BASIC（第1档）**，是最基础的铁器链，不是深层成就——
  它 SR 落后 baseline 属于"基础链退化"，比落后深层更值得警惕。
- 我方代码 `DEPTH_TIERS` 与官方不一致（魔法归错档、iron_pickaxe 归 tier2 而非 BASIC）——
  若 AmbitionGain/ability-gate 依赖它，可能与 baseline reward 结构错配，**待独立核查是否需对齐官方**。
- 相关：[[craftax-achievement-set-67]]（67 全集 + wandb join 键）、本仓库 craftax_achievements.py。
