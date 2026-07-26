# Fresh-Checkout 级证据完整性核验(CC4 合并前加固 · 二)

- UTC:`2026-07-26T13:15:53Z`
- 任务:`GLOBAL_EVALUATION_PREMERGE_EVIDENCE_HARDENING`
- **CC4_FRESH_CHECKOUT_INTEGRITY = PASS**

## 方法
- 另建**独立 detached worktree** `D:/c4f`,从对象库干净检出 CC4 HEAD `2a89f393113d26a4e022646ba9d26c4d8c2b0dad`(tree `3c8bc175fc9fdeec96707570f76ce11e3bda8c73`,status clean)。
- `sha256sum -c` 从 **worktree 根**(BASE 等价)执行,因为 SHA256SUMS 内路径为**仓库根相对**(`audit_outputs/global_remediation_…/X`)。
- 说明:用户字面命令 `cd <remediation_dir> && sha256sum -c SHA256SUMS` 因路径为根相对而得 0 OK / 54 FAILED——**已记录,非缺陷**;按实际路径格式从根执行得 54/54。

## 1. SHA256 校验(fresh checkout)
- **sha256sum -c:54 OK / 0 FAILED** ✓

## 2. .gitattributes 生效 · 无 CRLF/LF 字节转换
- `.gitattributes` blob:`e19ceacf56871f7cd3b6ceebddb747be84fe50d9`,内容:
  - `audit_outputs/global_remediation_20260726T095819Z/** -text`
  - `reports/global_remediation/** -text`
- 抽样(工作树文件字节 vs 提交 blob vs sumfile 三方一致):

| 文件 | workfile | blob | ==blob | ==sumfile |
|---|---|---|---|---|
| `reports/global_remediation/world_manifest_report.md` | c445eada88a7… | c445eada88a7… | YES | YES |
| `audit_outputs/global_remediation_20260726T095819Z/official_achievement_tiers.json` | 16f121800b88… | 16f121800b88… | YES | YES |
| `audit_outputs/global_remediation_20260726T095819Z/world_manifests/canonical_worlds_256_seed42.json` | e63c626aac15… | e63c626aac15… | YES | YES |
| `audit_outputs/global_remediation_20260726T095819Z/raw_data_manifest.json` | c1266cc4aaef… | c1266cc4aaef… | YES | YES |

- 结论:checkout **未发生** CRLF/LF 转换(此前 autocrlf 失配已由 Codex 字节修复 + `-text` 解决)。

## 3. artifact_inventory 一致性
- 条目数:**54**;size 匹配:**54/54**;sha256 匹配:**54/54**;失配:**0** ✓

## 4. 路径存在性
- SHA256SUMS 条目:**54**;路径全部存在:**54/54** ✓

## 5. 无冒充文件(未登记文件检查)
- 受跟踪:remediation 45 + reports 11 = **56**(=56)。
- 磁盘文件:remediation 45 + reports 11 = 56。
- **未登记文件:0**(NONE)✓;sumfile/inventory 路径全部受跟踪。

## 纪律
- 未修改 54 个冻结文件;未重写 SHA256SUMS;fresh checkout 全程只读;未 push/merge/rebase/force。
