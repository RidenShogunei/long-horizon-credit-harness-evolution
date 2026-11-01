# Code map（研究一 data + 保留的 foundation CLI）

## 环境

```bash
export PYTHONPATH=src
python -m research1 --help
```

## 研究一 data CLI（`src/research1/data_pipeline/cli.py`）

| CLI | 实现 | 说明 |
| --- | --- | --- |
| `data prepare` | `protocol.py`；formal 时 `scale.prepare_formal` | 冻结批次、`protocol.json`、分区与策略 |
| `data run` / `resume` | `scale.run_formal`、`edit_trial.run_grounded_edit_trial`、`initialization.run_initialization` 或 `engine.run_campaign` | 驱动控制器；结束后可 `_export` 并调用 `report.build_report` |
| `data status` | `cli.command` 读取 `{campaign}/status.json` + `ledger.status()` | 只读折叠，不写 report |
| `data export` | `cli._export` → `sft.export_dataset` | CURRENT 导出与 manifest |
| `data revoke` | `ledger.revoke` + `_export` | lineage 维护 |

版本化逻辑在 `src/research1/data_pipeline/*`（`applicability_v3.py`、`admission_*`、`modules_v3.py`、`report.py`、`ledger.py` 等）。**未删除** 旧模块；replay 依赖历史文件名，属于组织/保留而非 dead-code 清理。

## Foundation / 诊断（历史命令，仍保留）

| 命令组 | 主要模块 |
| --- | --- |
| `preflight`, `write-h0-lock`, `calibrate` | `preflight.py`, `calibrate.py`, `contracts.py` |
| `run-modularrsi`, `run-causal` | `p2a.py`, `causal.py`, `evaluation.py` |
| `distill-*`, `collect-plan-init`, `cover-parents` | `distill.py`, `collection.py`, adapters |
| `run-branches`, `label-continuations` | `p2b.py`, `returns.py`, `rollout.py` |
| `audit`, `report` | `audit.py`, reporting helpers |

## 测试

`src/research1/tests/` — 离线单元测试（C26：625 tests OK；本任务不重复全量）。

## 配置

`configs/protocol.v0.json`, `protocol.v1.json`, `providers.v0.json`, `roles.v0.json`, `h0.snapshot.v0.json` 等为 foundation 对照；研究一批次配置在各 `runs/*/protocol.json`（**不** 上传 `runs/` 树）。
