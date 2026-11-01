# Reports catalog

Updated: 2026-11-01T19:44:25.064386+08:00 (China time).

## 公开仓库内（已提交 / 可链接）

| 角色 | 相对路径 |
| --- | --- |
| **权威状态快照** | `records/2026-11-01/status.json` |
| 源码与配置清单 | `records/2026-11-01/source_manifest.json` |
| `reports/` 文件索引（不含正文） | `records/2026-11-01/report_catalog.json` |
| 冻结批次哈希 | `records/2026-11-01/campaign_inventory.json` |
| 工程 smoke 摘要 | `reports/foundation_status.md` |

**CURRENT 正式计数（V14 批次，只读折叠）：** qualified edit **0**；CURRENT formal SFT **0**；MC **0**；历史 qualified keep **8**；累计已用 8/0 regular/retries；控制器 **未运行**。

## 仅服务器本地（不提交 GitHub）

以下路径仍在部署机供 frozen 引用与运维，**不在** 本公开上传范围：

- `reports/data_collection_current.json`
- `reports/data_collection_progress_current.json`
- `reports/grounded_edit_admission_progress_data_edit_trial_20261101_v14.json`
- 其余 `reports/**/*.json` / `*.jsonl` 生成物与 `runs/**` ledger、teacher 语料

机器可读全量 `reports/` 清单（含上述 server-only 路径的哈希分类）见 `records/2026-11-01/report_catalog.json`。

## 策略

1. 禁止 hard reset、递归删除、或将 pending 重标为 PASS。  
2. 历史路径保留以服务 frozen replay。  
3. 运行时 scratch 由 `.gitignore` 抑制；**不** ignore `src/`、`tests/`、`docs/`、`configs/`。  
4. 公开 GitHub：**public** — 禁止密钥与完整 raw prompt/reply。
