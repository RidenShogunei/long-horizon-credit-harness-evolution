# 初始化采集与长程标签：修订后的入口

`collect-plan-init` 从可恢复的真实父状态采计划。`label-continuations` 只汇总已经完成的真实续跑，不启动模型、任务或84次实验。两个阶段分别记录，初始化通过不会自动生成长程标签。

## 采集一个父状态

准备 state JSON，包含 `modules_root`、`task_name`、`evidence`、可选 `history/public_config`。任务必须属于 pilot 数据集 manifest 的 train 列表。每条 evidence 需要稳定的 `id` 与非空 `text`；已有真实任务结果可放在 `task_result`。历史信息必须在该父状态决策时可见。不要提供子版本结果、隐藏测试或参考解。

```json
{
  "modules_root": "/absolute/path/to/recoverable/modules",
  "task_name": "manifest_train_task_name",
  "evidence": [{
    "id": "parent_trial_1",
    "text": "decision-time observation and module-related evidence",
    "task_result": {"status": "PASS", "reward": 0.0, "failure_layer": "model"}
  }],
  "history": [],
  "public_config": {"visible_files": ["tools/baseline.py"], "max_code_chars": 24000}
}
```

先预览冻结，不调用模型：

```sh
python -m research1 collect-plan-init --state state.json --checks checks.json
```

确认输入后执行计划采集；行为核验通过的编辑才会进入真实任务评测：

```sh
python -m research1 collect-plan-init --state state.json --checks checks.json --execute --run-task --max-candidates 2
```

完整模块与当时可见视图冻结到 `datasets/research1/states/<state_id>/`。教师先生成全部候选，固定编辑器再分别实现；候选不读另一分支结果。状态、计划、尝试和任务执行使用独立稳定ID。同输入重跑复用已有计划和执行，避免追加为多个独立经验。

checks 是采集者提供并在教师调用前冻结的行为契约。当前默认适配器支持 tools 模块的 parser contract；其他机制的计划可以采集，但没有适用且可执行的检查时保持 pending。不能让模型临时编一个会通过的检查来升格样本。受支持的编辑范围是 `tools/baseline.py`；这仍是单模块先导实现。

行为检查应覆盖修复、正常行为及混合情形，并保存 commands、error、is_task_complete 和 warning。真实评分为0是有效结果，基础设施故障null不能晋升。keep 需要引用有效父任务证据，只表示有依据的保留决定，不表示它具有最优长期价值。

## 样本资格与重新导出

`datasets/research1/admission_registry.jsonl` 是资格事件账本，状态为 verified/pending/rejected/revoked。撤销样本时保留原始计划、补丁与结果。`rebuild_verified_plans` 重建去重的训练导出；不修改历史不可变子版本。

旧 `distill-verified/observed/observed-batches/llm-review` 必须显式传 `--allow-legacy`。旧警告探针只产生 auxiliary_warning_cleanup 候选，LLM review 只产生辅助审查，不可直接导出训练正例。

## 真实长程标签

为每个真实候选保存恰好M条独立后续演化路径。每条JSONL记录按 `research1.contracts.MCPath` 定义，包含：

- `path_id/candidate_id/policy_id/continuation_seed`；
- `horizon`：首轮候选实现之后的续决策长度L；
- `first_implementation_hash`：首轮实际实现SHA256；
- `version_order`：父版本、首轮实现、L个后继；
- `score_tables/trial_counts/expected_tasks/required_repetitions/task_weights/lambda_value`；
- `path_kind=real_evolution`。

每题先对R次合法评测取均值。每一步逐题下降先取正部，再加权汇总D；G是终点与起点的加权分差。缺失题、null或不足R次保持未完成；不能跳过缺失结果后比较不同题集。R次同版本评测不等于M条后续演化。

```sh
python -m research1 label-continuations --input datasets/research1/value/continuations.jsonl --dry-run
python -m research1 label-continuations --input datasets/research1/value/continuations.jsonl
```

默认每候选恰好2条路径，所有路径的父状态、首轮实现、任务/权重、R、后续策略、L、λ和共享前缀评测必须相同。保存每条路径G/D/U、均值、样本标准差和条件元数据。注释marker对照不能作为真实演化标签。纯数据校验能检查声明的ID与seed，不能证明采样过程确实独立，仍需保留原始演化日志。

## 当前边界

新计划小批量入口使用 `fresh_json_final_v1` 教师策略与 `exact_replacements_v1` 固定编辑器。两者要求正常结束的最终 JSON，温度0；教师独立诊断，编辑器只实现传入计划，不额外提示特定修复。契约不适用的计划保持待核验，keep 仍需独立证据复核。

首批记录位于 `runs/distill_batch_20261031_152512`：先用两个未采过计划、实际评分及父模块均可恢复的历史训练状态，再执行一个之前延后的新训练任务。总计最多3次教师调用、每父状态1个候选；真实执行R=1、200轮上限、单次900秒，无额外重试。首先遇到无法评分的父/子任务即停止并保留其余任务为延后状态。完整选择规则、证据预算、源码快照和哈希在该批 `protocol.json` 中，当前进度可读 `reports/distillation_current.json` 指向的状态文件。

历史教师计划的实现验收可通过独立入口运行：

```sh
PYTHONPATH=src python -m research1.acceptance --source-run runs/plan_init_20261031_045224 --max-attempts 2 --wall-seconds 900
```

该入口完整保留历史教师输入与计划，由固定编辑器生成1—4个精确替换块。编辑器仅接受正常结束的最终JSON内容，拒绝推理字段、截断输出、歧义替换和范围外修改。六项预冻结检查覆盖空等待误报、非空缺换行、正常命令、混合警告，以及空格和制表符输入；每项同时比较命令、顺序、时长、错误和完成标记。严格空字符串才属于无输入等待。

行为通过后执行真实任务，默认200轮上限、R=1和900秒操作时限；启动异常或无评分超时保存为未完成。有效0分可用于行为修复初始化，但任务分数不会取代行为检查。资格类型为 `directed_behavior_repair_init`，来源为 `archived_teacher_plan_reimplementation`；原始教师提示不会加入事后检查或编辑器反馈。同一来源重复尝试只占一个样本ID，不算新的独立机制或长期标签。

本次工程修订不等于已经完成新的数据集、模型训练或真实M=2采样。至少先处理3—5个不同父状态，报告不同机制与实际补丁数量，再决定扩采。正式长程采样前仍需完成任务校准与噪声对照。
