# Foundation status (research1-v1)

Updated: 2026-10-30 00:14 (Asia/Shanghai) — P2B 2×2 PASS on Tongji (`branches_20261029_134412`)

## Gate — FoundationReady = **YES** (dual-rail smoke)

| Rail | Status | Evidence |
| --- | --- | --- |
| Preflight | **PASS** | proxy/provider/disk/docker/harbor/H0 hashes; train dataset 120/120 |
| Calibrate | **INCOMPLETE** (probe) | `cron-env-debug` reward=1.0; `LOCKED_MODULE=tools`; full 8×3 deferred |
| P2A ModularRSI smoke | **PASS** | `runs/modularrsi_20261029_115545` PROFILE=smoke, temp=0.0, real solver logs + A13 |
| P2B branch miniloop | **PASS** | `runs/branches_20261029_123206` (1×1, official); then **2×2 Tongji** `runs/branches_20261029_134412` |
| FoundationReady | **PASS** | both rails have minimal smoke evidence with real scores |

Scale notes: P2B expanded to **2 tasks × 2 reps** on Tongji (`cron-env-debug`, `browser-playable-leaderboard`). branch_B U=0.25 > branch_A U=-0.125. Invalidated proxy-403 run: `branches_20261029_132345`.

## Blockers / follow-ups

1. Provider: `pilot.yaml` uses `provider_mode: auto` (prefer Tongji). **Root cause of「门户看不到用量 / 全 0 分」**：Harbor 宿主进程带了 `HTTP_PROXY=:7890`，但 `NO_PROXY` 原先只有 `api.deepseek.com`，同济请求被 clash 转出后返回 **403 当前网络环境不支持**（直连 200，走代理 403）。`llm_provider.py probe` 不走代理所以一直看起来正常。已在 `adapters.merge_no_proxy` 把 `agi.tongji.edu.cn` 加入 NO_PROXY；无效跑次 `branches_20261029_132345` 已标 FAIL 并重开。
2. Dataset pool split still partial (train complete; enough for smoke).
3. Full calibrate still open: `python -m research1 calibrate --config configs/research1/pilot.yaml`
4. Expand P2B to 2×2 then nominal 60 / MC separately.
5. Calibrate probe aborted heavy `build-ffmpeg-legacy` Docker build; prefer lighter tasks for the next probe expansion.

## Key paths

- Package: `src/research1/`
- Config: `configs/research1/pilot.yaml`, `h0.lock.json`, `tasks.lock.json`
- Protocol: `configs/protocol.v1.json`
- P2A: `runs/modularrsi_20261029_115545/` (+ `A13_local_adaptations.md`)
- P2B: `runs/branches_20261029_123206/`
