# research1 dual-branch continuation (Track B)

Same-parent H0 fork into two trajectory-grounded **functional** tools candidates. Continuations are evidence-gated (no comment-marker padding). Score uplift is not the pass criterion.

- Run: `runs/causal_dual_20261030_042451`
- Parent H0 source: `runs/causal_edit_20261030_025402/versions/H0/modules`
- Evidenced defect: `false_concat_warning_on_empty_keystrokes`
- Status: **PASS**
- Early-stop / notes: two trajectory-grounded functional round-1 branches applied+loaded; continuations early-stopped (no residual sufficient tools defect / no marker padding). Harbor agent trials at temp 0.7 did not re-emit the evidenced empty-mid-batch pattern (so harbor trajectory triggers were false); deterministic module probes on that exact pattern confirmed both distinctive triggers. Prior single-step causal_edit_20261030_025402 already showed Harbor can hit the suppress path when the agent emits the pattern.

## Why two candidates from one defect class

Coverage (`causal_20261030_022523`) found no parse error / silent duration cap / hidden execute failure. The only sufficient tools defect remains false newline-concatenation warnings on empty-keystroke waits. Dual-branch therefore forks two distinct functional strategies for that same archived evidence — not a second invented bug.

| Branch | Strategy | Trigger token |
| --- | --- | --- |
| A | suppress-only (no keystroke rewrite) | `BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED` |
| B | normalize empty waits to `\n` | `BASELINE_TOOLS_EMPTY_WAIT_NORMALIZED` |

## Round-1 artifacts

### Branch A

- Version: `H0_A1` → `runs/causal_dual_20261030_042451/versions/H0_A1/modules`
- Plan: `/home/jinxu/long-horizon-credit-harness-evolution/runs/causal_dual_20261030_042451/plans/plan_A_2b6175257ec1.txt`
- Diff: `/home/jinxu/long-horizon-credit-harness-evolution/runs/causal_dual_20261030_042451/patches/attempt_A_H0_A1.diff`
- Functional AST change: `True`
- Harbor loaded modules_root: `True` (`['/home/jinxu/long-horizon-credit-harness-evolution/runs/causal_dual_20261030_042451/versions/H0_A1/modules', None, None, None, None]`)
- Harbor cron rewards: `[1.0, None, None, None, None]` (null = incomplete packaging / infra on that rep)
- Harbor trajectory trigger: `False`
- Module probe trigger: `True`
- Probe warning: `BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED: empty-keystrokes wait command does not need a trailing newline.`
- Score direction vs H0: `no_score_change_measured`
- Outcome class: `success`
- Continuation: `skipped` — no residual sufficient tools defect in harbor trajectories; refusing marker-only A2/B2 padding. Harbor trials did not re-emit empty-keystroke waits followed by another command (rep1 alone-waits; later reps mostly infrastructure/incomplete packaging). Module probe on the evidenced mid-batch pattern triggered=True.

### Branch B

- Version: `H0_B1` → `runs/causal_dual_20261030_042451/versions/H0_B1/modules`
- Plan: `/home/jinxu/long-horizon-credit-harness-evolution/runs/causal_dual_20261030_042451/plans/plan_B_2914a8363799.txt`
- Diff: `/home/jinxu/long-horizon-credit-harness-evolution/runs/causal_dual_20261030_042451/patches/attempt_B_H0_B1.diff`
- Functional AST change: `True`
- Harbor loaded modules_root: `True` (`['/home/jinxu/long-horizon-credit-harness-evolution/runs/causal_dual_20261030_042451/versions/H0_B1/modules', None, None, None, None]`)
- Harbor cron rewards: `[1.0, None, None, None, None]` (null = incomplete packaging / infra on that rep)
- Harbor trajectory trigger: `False`
- Module probe trigger: `True`
- Probe warning: `BASELINE_TOOLS_EMPTY_WAIT_NORMALIZED: empty-keystrokes wait rewritten to newline so following commands are not falsely marked as concatenated.`
- Score direction vs H0: `no_score_change_measured`
- Outcome class: `success`
- Continuation: `skipped` — no residual sufficient tools defect in harbor trajectories; refusing marker-only A2/B2 padding. Harbor trials did not re-emit empty-keystroke waits followed by another command (rep1 alone-waits; later reps mostly infrastructure/incomplete packaging). Module probe on the evidenced mid-batch pattern triggered=True.

## H0 baseline this run

- Raw H0 rewards: `[1.0]` on `cron-env-debug`

## Continuations (A2 / B2)

Both skipped. After round-1 (and resume reps), scans found no residual sufficient tools defect. Marker-only second steps were refused.

## Harbor trigger miss (documented)

Successful cron trials often use an empty-keystroke wait as the **only** command in a batch. The evidenced defect (and both fixes) require an empty wait **followed by another command**. Resume reps 2–5 largely ended as infrastructure/incomplete Harbor packaging while Track A verification smoke held docker; those reps have `reward=null` and often no `trajectory.json`.

Deterministic probes (`trigger_probe_A.json`, `trigger_probe_B.json`) feed the exact archived mid-batch pattern into each version’s `BaselineTools.parse_llm_response` via ModularRSI’s venv. Both fire their distinctive strings. Upstream checkout was not modified.

## Hard-rule check

- Upstream ModularRSI checkout read-only.
- Edits only under this run’s `versions/`.
- No `_comment_patch` / marker-only A2/B2.
- Second functional continuation stopped when evidence was absent.

