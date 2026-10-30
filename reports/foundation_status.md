# Foundation status (research1-v1)

Updated: 2026-10-30 (Asia/Shanghai) — Tongji dual-track wave archived: tools smoke + verification smoke + causal edit + dual-branch (INCOMPLETE)

## Gate — FoundationReady = **YES** (dual-rail smoke)

| Rail | Status | Evidence |
| --- | --- | --- |
| Preflight | **PASS** | proxy/provider/disk/docker/harbor/H0 hashes; train dataset 120/120 |
| Calibrate | **INCOMPLETE** (probe) | `cron-env-debug` reward=1.0; `LOCKED_MODULE=tools`; full 8×3 deferred |
| P2A ModularRSI smoke | **PASS** | `runs/modularrsi_20261029_115545` PROFILE=smoke, temp=0.0, real solver logs + A13 |
| P2B branch miniloop | **PASS** | `runs/branches_20261029_123206` (1×1, official); then **2×2 Tongji** `runs/branches_20261029_134412` |
| FoundationReady | **PASS** | both rails have minimal smoke evidence with real scores |

Scale notes: P2B expanded to **2 tasks × 2 reps** on Tongji (`cron-env-debug`, `browser-playable-leaderboard`). branch_B U=0.25 > branch_A U=-0.125. Invalidated proxy-403 run: `branches_20261029_132345`.

---

## Track A — ModularRSI（Tongji）

### A1 tools-locked smoke

| 项 | 状态 |
| --- | --- |
| 流程 / reflection | **已跑通**（contrast editor 有产物） |
| 实现编辑 / 晋升加载 | **未出实现补丁** |

- 跑次：`runs/modularrsi_20261030_022626`（保留为 tools-locked 基线）
- 报告：`reports/modularrsi_smoke_repro.md`
- Profile：`smoke`；`n_promotions=0`；`discard_reason=two-lane: nothing selected`
- Contrast finding：`is_culprit=false`，归因指向 **verification**；implementation editor 未跑（无 `gen_1`）

### A2 verification-locked next smoke

| 项 | 状态 |
| --- | --- |
| 配置冻结 | **完成** |
| 官方实施编辑 + 进化模块加载 | **仍未验证**（6/6 全过，reflect 未触发） |

- 配置：`configs/research1/pilot_verification.yaml`（+ `verification.lock.json` / `verification.stamp.json`）
- 跑次：`runs/modularrsi_20261030_042127`（PASS；browser/ffmpeg 均为 1/1/1）
- 报告：`reports/modularrsi_smoke_verification_next.md`
- 未改写 `modularrsi_20261030_022626`

---

## Track B — research1 自动修改闭环（tools）

### B1 单步因果环

| 项 | 状态 |
| --- | --- |
| 失败证据 → 计划 → 功能补丁 → H1 触发 | **PASS** |

- 覆盖：`runs/causal_20261030_022523`
- 编辑：`runs/causal_edit_20261030_025402`
- 报告：`reports/research1_scaffold_causal.md`
- 补丁：`patches/attempt_causal_H1.diff`（非 comment marker）
- 触发：`BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED:…`（H1 有、H0 无）；分数 delta `0.0`

### B2 双分支

| 项 | 状态 |
| --- | --- |
| 两支轨迹依据功能候选 | **已落地**（A 压制 / B 规整空 wait） |
| 行为触发 + 第二轮续决策 | **INCOMPLETE**（拒用 marker 凑 A2/B2） |

- 跑次：`runs/causal_dual_20261030_042451`
- 报告：`reports/research1_dual_branch.md`
- 旁支 `--resume-dual` 多轮复测 **已放弃**（非主线）

注释-only 对照仍单独标控：`runs/branches_20261029_134412`（见 `reports/audit_134412.md`）。上游 ModularRSI checkout 只读。

---

## Blockers / follow-ups

1. Provider：`pilot.yaml` `provider_mode: auto`（prefer Tongji）。Harbor 宿主代理需 `NO_PROXY` 含 `agi.tongji.edu.cn`（已在 `adapters.merge_no_proxy`）；无效跑次 `branches_20261029_132345`。
2. Full calibrate still open.
3. **A next**：需要能 mixed-fail 的条件，才能验证 verification 实施编辑与 `gen_1` 加载。
4. **B next**：仅在有新的、有依据的 tools 缺陷时再开续决策；不再做 trigger 盲复测。
5. Calibrate/smoke 仍易撞上重镜像任务（ffmpeg）；优先轻量任务。

## Key paths

- Package: `src/research1/`（含 `causal.py`）
- Config: `configs/research1/pilot.yaml`, `pilot_verification.yaml`, locks
- Reports: `modularrsi_smoke_repro.md`, `modularrsi_smoke_verification_next.md`, `research1_scaffold_causal.md`, `research1_dual_branch.md`
- Runs: `modularrsi_20261030_022626`, `042127`, `causal_20261030_022523`, `causal_edit_20261030_025402`, `causal_dual_20261030_042451`

## Audit / cleanup note (2026-10-30)

- Audit of `branches_20261029_134412`: `reports/audit_134412.md`
- Removed dry-run leftover `modularrsi_20261030_042117` and `runs/_logs/`
- Machine-local `scripts/harbor_proxy_path/` kept on disk but gitignored
