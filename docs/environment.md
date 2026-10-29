# 环境

记录时间：2026-10-28（Asia/Shanghai）。H0 冒烟已完成。

## 本仓库

- GitHub：`RidenShogunei/long-horizon-credit-harness-evolution`（private）
- 路径：`/home/jinxu/long-horizon-credit-harness-evolution`
- 协议：`research1-v0`（`configs/protocol.v0.json`、`docs/experiment_protocol.md`）
- 角色：`configs/roles.v0.json`
- H0 快照哈希：`configs/h0.snapshot.v0.json`（只存哈希，不存上游源码）
- 磁盘停止线：可用空间 < 15GB 停止。冒烟时约 122GB 可用。

## 密钥

- `/home/jinxu/.hermes/.env` → `DEEPSEEK_API_KEY`
- `scripts/load_keys.py`；冒烟脚本自行注入，不回显

## 对照基线（只读）

- `/home/jinxu/meta-harness-repro`（Harbor CLI 0.24 路径仍可用；本项目 H0 用 ModularRSI 自带 Harbor）

## ModularRSI / H0（仓库外）

- 检出：`/home/jinxu/arms_cache/modularrsi/ModularRSI`
- 提交：`ed1e545bc60f050b404474e51ee18021777a0e23`
- 独立 venv：`ModularRSI/.venv`（Python 3.12.13，`uv pip install -e .`）
- 模块根：`generations/merged_active/gen_0/modules`
- H0 作曲器：`static` → 五个 `baseline`
- 冒烟命令：`bash scripts/run_h0_smoke.sh`
- 容器出网：verifier/agent 注入 `HTTP(S)_PROXY=http://172.17.0.1:7890`（本机 clash/代理监听 7890）。无代理时评分器拉 `uv` 失败，会得到假的 0 分。

## 已验证

- `fix-git` 两次无代理：完整执行，外部 reward=0，失败层=`scorer/infrastructure`
- `fix-git` 两次有代理：外部 reward=1.0；模块轨迹完整；独立 trial
- 报告：`experiments/h0_smoke_20261028_report.md`
