# ModularRSI smoke reproduction (Tongji)

Run directory: `runs/modularrsi_20261030_022626`

Upstream Phase0 `PROFILE=smoke` completed on the Tongji gateway. Reflection was triggered. No generation was promoted.

## Setup

| Item | Value |
| --- | --- |
| Profile | `smoke` (`max_tasks=2`, `reflect_every=2`, `epochs=1`, `attempts=3`) |
| Locked module | `tools` |
| Solver temperature | `0.0` (upstream Phase0 constant) |
| Provider | `tongji_agi` |
| Model | `openai/DeepSeek-V4.1-Flash` |
| API base | `https://agi.tongji.edu.cn/v1` |
| Host `NO_PROXY` | `localhost,127.0.0.1,::1,api.deepseek.com,agi.tongji.edu.cn` |
| Container download proxy | `http://172.17.0.1:7890` via `scripts/harbor_proxy_path/harbor` (`--ae` / `--ve`) |
| Dataset | `/home/jinxu/arms_cache/modularrsi/ModularRSI_2000_Instances/tb` train split |
| Upstream | `/home/jinxu/arms_cache/modularrsi/ModularRSI` (not modified) |

A live probe to `agi.tongji.edu.cn` returned `ok` while `HTTP_PROXY` pointed at `:7890` and `NO_PROXY` included the Tongji host.

## Seed and tasks

`TASK_SEED=0`.

Upstream `run_online_evo` takes `tasks[:max_tasks]` and only then shuffles with `random.Random(task_seed)`. For smoke, `max_tasks=2`, so every seed is the same pair: the first two train-manifest tasks. Seeds 0–19 were checked; the set never changes. Seed 0 keeps manifest order.

Queued tasks:

1. `browser-playable-leaderboard`
2. `build-ffmpeg-legacy`

`build-ffmpeg-legacy` is in the queue. No seed removes it, or `build-imagemagick-legacy` once `max_tasks` reaches 3. The previous run `runs/modularrsi_20261029_115545` (also seed 0) lost the ffmpeg K-roll to `EnvironmentStartTimeoutError` (600s) and then skipped reflection because the remaining browser rolls all scored 1.0. This run's ffmpeg image layers were already in the Docker build cache, so the environment started and all six rolls were valid.

## K-rolls

All six rolls used `gen_0`, bundle `e2bb15f190b78d583acc248b3551457aa01d22456eea099e7abb074360e02a1c`, `tools=baseline`. `k_group_invalid_reason` is null on every roll. Roll 0 of each task used the dynamic composer; rolls 1 and 2 loaded that task's frozen bundle snapshot.

| Task | Roll | Reward | Episodes | Duration (s) | Generation |
| --- | --- | --- | --- | --- | --- |
| browser-playable-leaderboard | 0 | 1.0 | 38 | 504 | gen_0 |
| browser-playable-leaderboard | 1 | 1.0 | 12 | 804 | gen_0 |
| browser-playable-leaderboard | 2 | 1.0 | 7 | 280 | gen_0 |
| build-ffmpeg-legacy | 0 | 1.0 | 69 | 1325 | gen_0 |
| build-ffmpeg-legacy | 1 | 0.0 | 63 | 2036 | gen_0 |
| build-ffmpeg-legacy | 2 | 1.0 | 53 | 1712 | gen_0 |

The ffmpeg group is mixed (1.0, 0.0, 1.0), so the all-pass skip did not fire. Router buckets: browser `all_pass_efficient`, ffmpeg `mixed`.

## Reflection

Triggered on the last ffmpeg roll. `n_promotions=0`. `final_gen=gen_0`.

`discard_reason`: `two-lane: nothing selected`

What ran:

- Contrast editor for `build-ffmpeg-legacy`, with a trajectory and episode responses under `archive/editor_logs/gen_1_candidate_1793332383_contrast_build-ffmpeg-legacy/`.
- Finding `archive/findings/gen_1_candidate_1793332383.json`: `is_culprit=false`, `fixable_now=false`, `suggested_change=""`. The contrast session attributed the 0.0 roll to a premature completion decision and pointed at `verification`, which this lock does not edit.
- Backlog row `archive/backlog/findings.jsonl` records the same finding.

What did not run:

- The implementation editor. `editor_n_edits=0`, `editor_committed=false`, `files_changed=[]`, `editor_trajectory_path=null`.
- Review and sanity gates. `review_verdict` and `sanity_passed` are null. There is no `gen_1`.

`archive/evolution_log.jsonl` records `gen_1(discarded)` with `promoted: false` and the same reason. This is the attempt flow: a contrast editor artifact exists, and the two-lane window selected no implementation. It is not a rejected patch.

## Promotion and load

There is no promoted patch. Nothing to compare against a comment-only edit, and no evolved `tools` file was written.

Load evidence: every solver roll's `gen_used` is `archive/gen_0`. The frozen snapshots for rolls 1–2 are copies of that same baseline bundle. Reflection finished after the last task, produced no new generation, and did not start a sanity battery. No later evaluation loaded an evolved implementation.

## Outcome

| Check | Result |
| --- | --- |
| Process ran on Tongji, smoke workload, temp 0.0 | Yes |
| Valid K-rolls (not an environment-start timeout) | Yes, 6/6 |
| Reflection triggered | Yes |
| Contrast editor artifacts | Yes |
| Implementation patch | No |
| `n_promotions` | 0 |
| Promoted behavior change | No |
| Evolved implementation loaded by a later eval | No |
