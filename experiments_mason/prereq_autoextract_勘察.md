# 前置图自动抽取可行性勘察(prereq_autoextract_spike)

> 2026-07-11。动机 = UED 纯度质疑的防御:把 `prereq_graph.py` 从 "hand-curated" 升级为
> "compiled from the environment source",证明先验量级是 O(环境规格) 而非 O(解法)。
> 方法 = 直接写抽取原型(`dicode_src/prereq_autoextract_spike.py`,纯 AST/字面量解析,
> 零 craftax import 零 jax)跑在 craftax 1.4.5 wheel 源码上,与手工图逐边 diff。
> **结论先行:78% 的边今天就全自动复现;修一天工程可到 ~90%;真正不可自动化的只有
> 两小类判断。且 spike 反向抓出手工图 3 条漏边 + 1 条不精确边(已修)——自动化不止
> 可辩护,比手工更可靠。**

---

## 1. 数字(67 节点 / 92 条边)

| 档位 | 节点 | 说明 |
|---|---|---|
| **A. 今日全自动**(spike 逐边复现) | **48/67(边覆盖 72/92 = 78%)** | 配方 11 条 make_* 线(can_craft_* 库存比较 + is_crafting_* 台站门)/ 挖矿镐级(can_mine_*)/ 楼层阶梯(LEVEL_ACHIEVEMENT_MAP)/ 怪物边(FLOOR_MOB_MAPPING × 贴图索引序 × 刷新概率,含 0 层怪零前置)/ place_plant·place_stone·place_furnace / 宝箱线(drink_potion、learn_*、cast_*) |
| **B. 半天工程可自动** | ≈7 | place_table/place_torch(把 F 组件的 200 字符文本窗换成正经 AST 作用域——现有 2 条噪声边即出自此)、open_chest←enter_dungeon(world_gen 宝箱楼层)、fire_bow(射击逻辑 bow>0+arrows>0)、enchant_*(附魔函数 + ENCHANTMENT_TABLE 楼层配置,多跳)、find_bow(掉落经中间变量,一跳数据流) |
| **C. 可平凡默认** | 4 | collect_wood/sapling/drink、wake_up——"未发现任何门控机制 → 空前置"策略即可(collect_wood 甚至可正抽:can_mine_tree=True);该默认策略本身是一条(温和的)判断 |
| **D. 真人工** | ≈4 节点 + 1 类横切判断 | 见 §3 |

A+B+C ≈ 59/67 节点(~88%),边覆盖估 ~90%。

## 2. ★ 意外收获:spike 抓出已交付图的 4 处错误(已修 + 回归钉死)

diff 的 "auto-extra" 逐条核对地面真相时发现,**错的是手工图不是抽取器**:

1. `make_iron_armour` 漏 `place_table`(game_logic l.713:铁甲要 table **且** furnace);
2. `make_diamond_armour` 漏 `place_table`(l.745);
3. `make_torch` 漏 `place_table`(l.785);
4. `find_bow` 的精确边是 `open_chest` 而非 `enter_dungeon`(弓是宝箱掉落物;FIND_BOW = `inventory.bow > 0` 库存判定)。

顺带确认一个语义地基:**COLLECT_*/FIND_BOW/MAKE_* 全族是库存状态成就**(game_logic ~2871-2972,`inventory.x > 0` / 装备等级即触发)——所以宝箱掉落合法授予它们,`ITEM_*_GRANTS` 的映射从"近似"升格为"字面游戏真值"(闸的 R3 库存授予检查因此站得更稳)。

论文一句话素材:*the extractor not only matches the hand graph, it corrected it —— 环境规格级先验用编译获得比用人手获得更可靠。*

## 3. 真正自动化不了的边界(诚实清单)

1. **AND 投影(横切判断,最重要的一类)**:抽取器能精确拿到**析取图**(sapphire = 挖矿∨宝箱;coal/iron/diamond 同;enchant = 火台∨冰台)——今日 diff 里 5 条 "auto-extra" 全是真实或路径。但连词语义的调度图必须每处选一条正则最廉分支,这个选择是人做的。可以再自动化一层(最浅路径成本模型),但判断只是移进了成本模型的定义里,不会消失。**论文表述:OR-graph is compiled; the AND-projection is the sole curriculum-design input.**
2. **necromancer 双成就**:boss 机制独立于通用怪物表,楼层边需专门读 boss 逻辑(可做,不通用)。
3. **eat_plant ← place_plant**:植物生长链散在多处,追踪性价比低。
4. C 档的"未见门控 → 空前置"默认策略。

## 4. 建议(押后队列排期)

- **半天**:补 B 档五个抽取件 + 把 F 组件窗口启发式换成 AST 作用域 → 加一个 CI 式测试:`extract_all() 的 AND 投影(给定正则路径表)必须逐边等于 DIRECT_PREREQS`——图与环境源码从此锁死,craftax 换版本即自动报警;
- 正则路径表(D.1)单列成一个 ~10 行的显式常量,论文里整表贴出——"全部人类课程先验在此"一目了然;
- 与 §6 消融配合:prereq-only 臂的增益 ÷ 这张 10 行表 = "先验性价比"的量化表述。

## 5. 复现

```bash
cd /workspace/mechanism_UED/dicode_src
pip download craftax==1.4.5 --no-deps -d /tmp && unzip -o /tmp/craftax-1.4.5*.whl -d /tmp/craftax_pkg
PYTHONPATH=src:. python prereq_autoextract_spike.py --craftax-src /tmp/craftax_pkg/craftax/craftax --diff
```
期望:auto-covered 55/67,exact 48,extra 7(5 真或路径 + 2 窗口噪声),missing 1(place_torch 的 make_torch,同窗口噪声)。
