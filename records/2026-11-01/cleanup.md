# C28 upload cleanup (2026-11-01T19:44:25.064386+08:00)

## 变更范围
- 文档：README、`docs/code_map.md`、`docs/data_flow_and_sources.md`、`docs/reports_catalog.md`
- `.gitignore`：锚定 `/runs/`、`/reports/*` 白名单，公开快照在 `records/2026-11-01/`
- 记录：`records/2026-11-01/*` 清单与 status（**无** 生产 `.py` 行为改动）

## 验证（C26 基线）
- 离线测试：625 unittest（C26，~214s OK；本任务未重跑）
- 冻结批次：14，与备份 `C26_20261101T114423Z` 一致：True
- C26 备份 130 个 baseline `.py` SHA 未变：见 `verify_src_py_baseline`
- **未** 删除生产代码；仅 Git 噪声抑制与 curated 记录上传

## 上传范围
- **已上传：** `src/`、`configs/`、`docs/`、`README.md`、`.gitignore`、`records/2026-11-01/`、`reports/foundation_status.md` 及既有 tracked reports
- **留服务器：** 原始 `runs/**`、live `reports/*` JSON/JSONL 指针与生成流（见 status.json `server_only_live_pointers`）

## 备份引用
-  opaque 相对名：`C26_20261101T114423Z`；meta head：`717a083b375f…`
