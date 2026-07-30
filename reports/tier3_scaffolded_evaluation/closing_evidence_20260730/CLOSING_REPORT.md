# CC4 收口报告 — 公共 evaluator 交付 + 双 RMT16 capsule(2026-07-30)

合同:CC4_MATERIALIZE_COMMON_EVALUATOR_AND_RMT16_CAPSULES。
本轮只补齐公共 evaluator 交付与两个 RMT16 capsule;未启动任何正式性能评估、
训练、D052、消融。服务器 oseasy@172.25.14.221,GPU 仅用 2/3。

## 分支与提交

- 分支:`henry/tier3-scaffolded-evaluation`
- 本地/服务器 HEAD(一致,ff-only 同步):`7fe25a72fbe0e0e01dd74e25376a313698cee67b`
  (tree `9cec45eee346af395104a010d08922680e44e2b2`)
- 本轮新增 commit7 `7fe25a72`:装配器/胶囊在进程内 unpickle 前插入审计
  dicode 源树(minicraftax),修复服务器第三次装配的 fail closed。
- push 仍 BLOCKED_NETWORK(127.0.0.1:443 代理不可达;按总控为网络问题,
  非代码失败;服务器同步全部经 git bundle + ff-only)。

## 固定字段(§11)

| 字段 | 值 |
|---|---|
| CC4_CURRENT_TASK_CONTINUED | true(收口合同 §0–§11 全链) |
| CC4_EXISTING_PROGRESS_PRESERVED | true(commit4–6 工作保留;仅删除本 CC 三次失败装配的部分输出) |
| CC4_OTHER_JOBS_INTERRUPTED | false(cc1/cc2/cc3 目录与 director 孤儿进程 106885 均未触碰) |
| COMMON_ROOT | /home/oseasy/student_pool_v1/common |
| COMMON_ROOT_CREATED | true(56 文件;SHA256SUMS 57/57 核验通过) |
| COMMON_EVALUATOR_SHA256 | a47ff97f9dc745c4f0cf015966b777f90c6dd6c7fe934b9b552a542df188a344 |
| COMMON_RUNNER_SHA256 | 135332d3b30c60cb7b29c620dc931da852e99b2ca256c7a77dbf365dfc94075b |
| EVALUATION_PROFILE_SHA256 | 7147370115621bda0500d55d8fd506a119ef8d6467a08329aaf6e088fbf9ea73 |
| METRIC_SCHEMA_SHA256 | 3a1712c4074dcb8fe8043c5a67e3ad7c730f252c533ad148a7181ba28f953da0 |
| FRONT_BANK_FILE_SHA256 | 2c20e886c07c108d036622356c1c8136112f8e31bc1fbf61997044986a842d39 |
| FRONT_BANK_CONTENT_SHA256 | 21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687(== 冻结历史恒等) |
| BACK_BANK_FILE_SHA256 | 9e104772d11abcee9fec2fd22cf57742d19dd37e4aeb122fa313f4a555c2a5ae |
| BACK_BANK_CONTENT_SHA256 | c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566(== 冻结历史恒等) |
| FULL_PROFILE_SHA256 | 2eceb288785a589f3f7f8b6989be7876bbe8da299128363ee008397d79039c1f(profile["scenarios"]["full"] 规范 JSON) |
| ENVIRONMENT_LOCK_SHA256 | 453f1680dafe0f168c25c262f51de59ddc59559676aecd05f8f17389015c2ad3 |
| B1_COMMON_EVALUATOR | PASS |
| B3_FROZEN_BANKS | PASS(mint=CPU / load=GPU provenance;bank_source=FROZEN_SERIALIZED_ARTIFACT;field_manifest 615d4be4… 冻结) |
| FULL_PROFILE_READY | true |
| COMMON_RUNTIME_ABI_READY | true |
| NEGATIVE_GATES | PASS(49/49,fail=0,pending_commit3=0) |
| CROSS_GPU_DETERMINISM_PREFLIGHT | PASS(GPU2 vs GPU3,三场景 record SHA 逐位一致;first_difference=None) |
| COMMON_SHA256SUMS_STATUS | PASS(57/57) |
| COMMON_EVALUATOR_READY | **true**(8/8 门) |
| PERSISTENT_RMT16_CAPSULE_ROOT | /home/oseasy/student_pool_v1/cc4/PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 |
| PERSISTENT_RMT16_BINDING | PASS(run_class=INTERFACE_SMOKE;8 个 §7 SHA 引用全部核验;FRONT/BACK/FULL 各 1ep×32 步真实引擎运行) |
| PERSISTENT_RMT16_READY | **true**(8/8 门;11 文件;SHA256SUMS 12/12) |
| RESET128_RMT16_CAPSULE_ROOT | /home/oseasy/student_pool_v1/cc4/RESET128_RMT16_ORIGINAL_VTRACE_98304 |
| RESET128_RMT16_BINDING | PASS(同上) |
| RESET128_RMT16_READY | **true**(8/8 门;11 文件;SHA256SUMS 12/12) |
| FORMAL_POOL_EVALUATION_STARTED | false |
| NEW_TRAINING_STARTED | false |
| D052_RUNS_STARTED | 0 |
| ABLATION_RUNS_STARTED | 0 |

## 证据

- 服务器侧:三处 SHA256SUMS `sha256sum -c` 全过(common 57/57、双 capsule 各 12/12)。
- 本目录:服务器回传的小型证据 JSON(排除 npz/pkl)+ `verify_closing_evidence.py`
  纯门本地复验(任意解释器):**CLOSING_EVIDENCE_LOCAL_REVERIFY_PASS**。
- 接口 smoke 真实行为:persistent 臂 BACK bank0 于 22 步 SUCCESS_DEFEAT_KOBOLD;
  FRONT TIMEOUT_NO_TRANSITION;FULL TIMEOUT_NO_KOBOLD(greedy、32 步封顶)。
- 冻结恒等式全部复现:checkpoint 文件/params 五值、bank 内容两值、
  field_manifest、checkpoint 合同自哈希 7dda2bc7…。

## 未做(合同 §10 禁止项,严格遵守)

未启动 6-student 正式排名评估、未训练、未 D052、未消融、未新增候选 adapter、
未改 checkpoint、未变 FRONT/BACK 语义、未发明 FULL seeds、未 GPU 重生成银行、
未抢占进程、未 force push/rebase/amend/merge。
