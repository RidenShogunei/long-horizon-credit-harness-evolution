# Track A next: verification-locked ModularRSI smoke (Tongji)

Frozen from tools-smoke attribution. Does **not** rewrite or delete `runs/modularrsi_20261030_022626` / `reports/modularrsi_smoke_repro.md`.

## Why this lock

Prior tools-locked smoke (`modularrsi_20261030_022626`) completed reflection with:

| Field | Value |
| --- | --- |
| `n_promotions` | 0 |
| `discard_reason` | `two-lane: nothing selected` |
| `final_gen` | `gen_0` |
| Contrast finding | `is_culprit=false` for `tools` |
| Pointer | `other_module`: verification (premature completion / `consecutive_complete_signals` without inspecting recent build-error terminal output) |

Finding path: `runs/modularrsi_20261030_022626/archive/findings/gen_1_candidate_1793332383.json`.

## Frozen config paths

| Path | Role |
| --- | --- |
| `configs/research1/pilot_verification.yaml` | Pilot overlay with `LOCKED_MODULE: verification` |
| `configs/research1/verification.lock.json` | Attribution freeze + do-not-modify list |
| `configs/research1/verification.stamp.json` | Advisory stamp (humans/docs; YAML wins in CLI) |

Left untouched (tools historical):

- `configs/research1/pilot.yaml`
- `configs/research1/tasks.lock.json` (`LOCKED_MODULE=tools`)
- `configs/research1/calibrate.stamp.json`
- `runs/modularrsi_20261030_022626/`
- `reports/modularrsi_smoke_repro.md`

## Exact Tongji launch command

From repo root `/home/jinxu/long-horizon-credit-harness-evolution`:

```bash
mkdir -p /tmp/modularrsi_verification
nohup python3 -m research1 run-modularrsi \
  --config configs/research1/pilot_verification.yaml \
  --max-tasks 2 \
  > /tmp/modularrsi_verification/cli.log 2>&1 &
echo $! > /tmp/modularrsi_verification/cli.pid
```

Notes:

- `p2a.run_modularrsi` forces `provider_mode=tongji_agi` and `API_BASE=https://agi.tongji.edu.cn/v1` regardless of pilot `provider_mode`.
- Host `NO_PROXY` must include `agi.tongji.edu.cn` (adapters merge this).
- Solver temperature follows upstream Phase0 `0.0`.
- `TASK_SEED` defaults to `0` → same smoke pair as tools run: `browser-playable-leaderboard`, `build-ffmpeg-legacy`.
- CLI stdout is captured until process exit; watch `runs/modularrsi_*/archive/` and `ps -p $(cat /tmp/modularrsi_verification/cli.pid)`.

Dry-run check (already PASS):

```bash
python3 -m research1 run-modularrsi \
  --config configs/research1/pilot_verification.yaml \
  --dry-run --max-tasks 2
```

Dry-run artifact (not a live eval): `runs/modularrsi_20261030_042117` (`dry_run=true`, `LOCKED_MODULE=verification`).

## Live run status

| Item | Value |
| --- | --- |
| Started | **yes** (2026-10-30 ~12:21 Asia/Shanghai) |
| Finished | **no** (in progress at report write) |
| PID file | `/tmp/modularrsi_verification/cli.pid` |
| Run dir | `runs/modularrsi_20261030_042127` |
| `LOCKED_MODULE` in `archive/evolve_invoke.json` | `verification` |
| Profile | `smoke` |
| Provider | Tongji `openai/DeepSeek-V4.1-Flash` @ `https://agi.tongji.edu.cn/v1` |
| Expected wall | ~1.5–2.5 h (prior tools smoke ~10:26–11:55 local; ffmpeg image cached) |

## Expected checks (post-run)

Same gate shape as `reports/modularrsi_smoke_repro.md`, but with verification as the editable module:

1. **Process / provider** — `summary.json`: `provider_id=tongji_agi`, `LOCKED_MODULE=verification`, `profile=smoke`, `exit_code=0`.
2. **Valid K-rolls** — six rolls (2 tasks × 3 attempts); `k_group_invalid_reason` null; not `EnvironmentStartTimeoutError`.
3. **Reflection triggered** — ffmpeg mixed group (if score pattern recurs) or whatever router buckets produce; look at `archive/evolution_log.jsonl` / `archive/meta.json`.
4. **Official implementation editor** — contrast may again attribute; with verification locked, expect an **implementation** editor path when a fixable finding is selected (look for `editor_n_edits>0`, `editor_committed`, non-null `editor_trajectory_path`, and/or files under `archive/editor_logs/` that are not contrast-only). Contrast-only + `two-lane: nothing selected` again is a negative result, not a pass of the load check.
5. **Subsequent load of evolved module** — if `n_promotions≥1` / `final_gen≠gen_0`: later solver rolls must show `gen_used` pointing at the new gen, and composer / bundle must load `verification=` from that gen (not only `gen_0`). If no promotion, record “no evolved verification loaded” explicitly.
6. **Historical isolation** — `runs/modularrsi_20261030_022626` still present with `LOCKED_MODULE=tools`.

## Blockers

| Blocker | Status |
| --- | --- |
| Tongji probe | Clear at launch (`/v1/models` 200) |
| Disk | Clear (`free_gb≈127` > 15) |
| Competing Track B dual-branch | Not started by this track; false-concat causal already finished |
| Tools historical rewrite | Avoided |
| Heavy smoke (ffmpeg) | Accepted; same seed/queue as prior successful tools smoke |

## Out of scope

- Track B causal dual-branch
- Rewriting `foundation_status.md` (sibling owns status; this report is the Track A next record)
- Upstream ModularRSI edits (wrappers only)
- git commit / push
