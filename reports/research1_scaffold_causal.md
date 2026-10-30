# research1 scaffold: failure feedback to a functional tools edit

Track B only. The upstream ModularRSI checkout was not modified. Comment-marker runs stay control experiments and are not treated as evolution samples. No evolver was trained. Score movement is not the pass condition for this round.

## Why comment markers were not enough

`branches_20261029_134412` only prepended `# research1-p2b-marker` to `tools/baseline.py`. Harbor did load each version tree (`modules_root` matched the version directory, composer logged `tools=baseline`), but a comment does not change parse or execute behavior. `apply_plan_patch` then stored `outcome_class=no-effect` because the patch applied, not because anyone measured behavior.

Cron rewards still flipped under those identical markers (H0 cron reps were 0 and 1; browser was 1 on every trial). That is score noise. `branches_20261029_132345` died on Tongji HTTP 403 (`llm_call_failed`) before any tool call. The smoke archive's ffmpeg trial died in `EnvironmentStartTimeoutError` while starting the image. None of those results is a tools-module defect.

## Was there a real tools issue?

Parser replay of the archived episode responses found **0 parse errors**. The only tools signal in those traces was five warnings of the form "Command N should end with newline … Otherwise the two commands will be concatenated". In every case the warned command had **empty keystrokes** and a positive duration (a wait). Empty keystrokes do not type anything, and the following observation still showed a shell prompt, so the concatenation claim was false. `tools/baseline.py` forwarded that parser warning unchanged. The clearest archived example is H0 `cron-env-debug` rep 1 in `branches_20261029_134412`, step 9 (reward 0 on that trial; the warning is not what failed the cron tests).

That warning alone was not treated as enough before looking further. Coverage was expanded with the existing rule: train names from `manifest.json` in fixed order, the first-8 window ranked by manifest bytes, skip tasks that already had an agent trace, skip source-build images (`build-essential`, the same class as the recorded ffmpeg abort), and skip private base images that are not on disk.

`runs/causal_20261030_022523` then ran one H0 trial each:

| Task | Reward | Tools finding |
| --- | ---: | --- |
| complex-python-memory-leak-concurrency-bug | 1.0 | none |
| wasm-crackme-reverse | 0.0 | none (a task assertion failed; the parser reported no error) |
| crypto-transaction-batching-optimizer | 1.0 | none |

No parse failure, hidden execute error, or silent duration cap appeared. The empty-keystroke warning remained the evidenced tools defect, so the edit targets that and nothing else. Observation truncation and verification early-stop were not edited.

## Automatic H0 → H1

Run: `runs/causal_edit_20261030_025402`.

Frozen planner: Tongji `DeepSeek-V4.1-Flash` (`https://agi.tongji.edu.cn/v1`), temperature 0, one call. It read the step-9 excerpt and `tools/baseline.py`. Plan (`plans/plan_causal_1b66cd07f3b0.txt`):

> The parser emits a newline-concatenation warning for an empty-keystrokes wait command even though there is nothing to concatenate. `BaselineTools.parse_llm_response` will now detect empty-keystrokes commands that are followed by another command, remove only the matching false newline warning, and append a distinctive feedback line when that suppression actually occurs.

The fixed editor did not call a model. It diffed the replacement against H0, rejected comment-only edits, and applied the stored unified diff with `patch`. `apply_status` is **applied**. Diff: `patches/attempt_causal_H1.diff`.

| | |
| --- | --- |
| H0 `modules_root` | `runs/causal_edit_20261030_025402/versions/H0/modules` |
| H1 `modules_root` | `runs/causal_edit_20261030_025402/versions/H1/modules` |
| H1 `tools/baseline.py` sha256 | `db7ff9a61eebd42abc47940fd539ead381a6cf508ea0b4fc9485f3c2f4d0d1cf` |

H1's Harbor config recorded that same `modules_root`. The upstream `generations/merged_active/gen_0/modules/tools/baseline.py` does not contain the new string.

Solver for both trials: Tongji `DeepSeek-V4.1-Flash`, temperature 0.7, task `cron-env-debug`, one trial each.

The fresh H0 trial still showed the old false warning (step 9, empty keystrokes). H1 triggered the new path twice. Step 4 sent an empty-keystroke wait and then more commands; the next prompt the agent received was:

`WARNINGS: BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED: empty-keystrokes wait command does not need a trailing newline.`

Step 9 did the same with a 30s empty wait. That string is absent from the H0 trajectory. It is present in H1 `trajectory.json` and in `episode-3/prompt.txt` and `episode-8/prompt.txt`.

## Scores

| Field | Value |
| --- | --- |
| apply_status | applied |
| behavioral trigger | yes (H1 steps 4 and 9) |
| n_h0 | 1 |
| n_h1 | 1 |
| raw H0 rewards | `[1.0]` |
| raw H1 rewards | `[1.0]` |
| observed_delta (H1 mean − H0 mean) | `0.0` |
| score direction | **no score change measured** |

Equal means are no score change measured. They are not a claim that the edit had no behavioral effect. The agent-visible warning text changed. `outcome_class` is `success` here only because the functional change was loaded and triggered, not because the mean rose. With n=1, this delta also does not say the edit is score-neutral in general; cron rewards move between trials even with no code change.

Dual-branch continuation and Monte Carlo were not started.
