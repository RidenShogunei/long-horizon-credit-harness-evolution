# 长程信用 Harness 稳定进化 — 仓库入口（当前）

**研究一（research-one）** 覆盖：在 ModularRSI 终端任务上收集 Plan-only 结构化监督（edit/keep 与准入链）、**初始化 SFT 管线**，以及目标 harness 上 **真实 K=2、M=2、L_after=2 长程 MC（G/D/U）** 实验设计。上述路线 **均未完成**；本回合仅做仓库整理与公开上传，**采集已停止**，不启动新预算或 paid 调用。

数据子命令入口：`python -m research1 data --help`。常用动作为：`prepare` 冻结批次目录与 `protocol.json`；`run` / `resume` 驱动采集或试验控制器（按协议路由 initialization、grounded edit trial、formal scale 或通用 engine）；`status` 只读折叠战役 `status.json` 与 ledger；`export` 经 CLI 内 `_export` 调用 `sft.export_dataset` 导出 CURRENT 数据集；`revoke` 维护 lineage。Foundation 诊断仍用 `python -m research1 preflight|calibrate|run-modularrsi|…`；历史见 `docs/experiment_protocol.md`（**非** 研究一当前权威协议）。

```bash
cd <repo>
export PYTHONPATH=src
python -m research1 data --help
```

## 最新研究一批次（已停止）

| 项 | 值 |
| --- | --- |
| 批次目录 | `runs/data_edit_trial_20261101_v14/` |
| 状态快照 | `GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT`（phase `finished`） |
| 更新时间 | 2026-11-01T19:44:25.064386+08:00 |
| 独立父代（有决策） | 6 |
| 原始 Plan（去重） | 28 |
| 当前 formal qualified edit | 0 |
| 历史 qualified keep（非 CURRENT formal SFT） | 8 |
| pending / rejected / revoked / teacher_errors | 16 / 1 / 3 / 1 |
| CURRENT formal SFT 行 | 0 |
| MC 值（CURRENT） | 0 |
| 累计已用预算 regular / retries | 8 / 0 |
| 采集控制器 | 未运行 |
| 学生模型 | 未选定 |

**公开权威状态（已提交）：** [`records/2026-11-01/status.json`](records/2026-11-01/status.json)。

**仅部署机本地、不进入本仓库的 live 指针：** `reports/data_collection_current.json`、`reports/data_collection_progress_current.json`、`reports/grounded_edit_admission_progress_data_edit_trial_20261101_v14.json` 及 `runs/**` 原始 ledger/语料。索引说明见 `docs/reports_catalog.md`；工程 smoke 摘要见 `reports/foundation_status.md`（**不**表示训练就绪）。

## 文档地图

| 文档 | 用途 |
| --- | --- |
| `docs/code_map.md` | 研究一 data CLI 与 foundation 模块 |
| `docs/reports_catalog.md` | 公开记录 vs 服务器 live 指针 |
| `docs/data_flow_and_sources.md` | 开放数据源调查（solver 轨迹 vs 目标 G/D/U） |
| `docs/data_protocol.md` | **历史** 采集协议 |
| `docs/experiment_protocol.md` | **历史** foundation 实验协议 |

## 运行时依赖（仓库外）

ModularRSI 与 Harbor 路径约定见 `docs/environment.md`、`docs/repo_layout.md`（公开记录不含个人 home 绝对路径）。

## 许可

ModularRSI 上游含非商业等限制；外部数据集条款见 `docs/data_flow_and_sources.md` 脚注。
