# 数据流与开放数据源（调查笔记）

日期：2026-11-01。
**本次调查未下载/导入新的外部数据**；项目 **已在使用** ModularRSI 2000 任务池作为底层任务源。
性质：源码/卡片/API 调查；**未** 在本回合新增外部数据集导入或训练。

## 本仓库目标管线（概念）

- 冻结任务分区、父 harness 源码与 preservation 合约
- 运行 proposal 任务，仅暴露决策时刻源码、轨迹证据与分支历史
- DS teacher 输出结构化单步 edit 或 keep Plan
- 事实与适用性审查通过的候选，再进入独立审核
- 固定 implementer 保存单一 patch，再做 scope/behavior/preservation 检查
- 真实子代评估；独立准入与 provenance 绑定的 generic SFT 导出
- 另跑真实 K=2、M=2、L_after=2 续跑并导出 G、D、U

**已入库的 auxiliary 快照（2026-11-02 组织）：** 23 条 editorial Plan edit（20/3 train/val，keep 0），非 formal V14 qualified 链；manifest 与 raw 96 来源记录见 [`data/open_auxiliary/v2/`](../data/open_auxiliary/v2/) 与 [`open_data_collection.md`](open_data_collection.md)。Formal CURRENT SFT 仍为 0。

## 诊断摘要

- 单个任务求解失败不能证明 harness 存在可复用缺陷。
- Teacher Plan 含真实可行性错误（如 warning-only helper 缺少应检查的命令、把空白与空 wait 混淆）。
- DS 事实审查不能作为唯一可靠门：已知无效负对照可通过，而源码一致的 completion-reminder Plan 仍可能 pending。
- 最新 V14 有两条 edit Plan，但未启动 implementer、无 saved child patch、无子代评估；不能说“最新 edit 已试且失败”。
- Formal qualified edit=0、MC=0、CURRENT formal SFT=0；V14 停在 editor/child 之前。
- 反复 one-off 协议/审查修订消耗工程时间但未产出要求的 qualified edit；离线测试不证明数据有效或 harness 改进成功。

## Solver 轨迹 vs 目标 harness 价值 G/D/U

- **Solver 结果**：单次任务成败、OpenHands/SWE 轨迹、benchmark 分数 — 仅用于结果验证或 warm-start，**不能** 直接当作改 harness 的长程 G/D/U。
- **目标 G/D/U**：须在 **真实目标 harness** 上、决策前 state + Plan + **K=2,M=2,L_after=2** 独立续跑导出；公开数据集的 binary resolved 等标签 **不可** 改称为 G/D/U。

## 已核对来源（主链接）

### ModularRSI 2000 Instances
- URL: https://huggingface.co/datasets/IQuestLab/ModularRSI_2000_Instances
- 已核实要点：1000 terminal and 1000 SWE executable tasks, each domain 120 train and 880 pool; instructions, environment, tests, reference solution; already the project's underlying task source
- 可用边界：Not 2000 evolver Plans and not long-horizon labels; never expose reference solutions/hidden tests as proposal evidence
- 拟用途：Generate task execution evidence

### ModularRSI released evaluation trajectories
- URL: https://github.com/IQuestLab/ModularRSI/blob/main/trajectories/mergefinal/README.md
- 已核实要点：Three held-out Terminal-Bench repeats, explicitly not used as evolution feedback
- 可用边界：Must not turn held-out benchmark trajectories into our training/proposal input
- 拟用途：Evaluation and reproducibility only

### ModularRSI merged active library
- URL: https://github.com/IQuestLab/ModularRSI/blob/main/generations/merged_active/README.md
- 已核实要点：20 implementations with source and archive metadata; explicitly no benchmark results, trajectories or private history
- 可用边界：Cannot reconstruct authentic autonomous Plans, original rationale or long-horizon labels from final source alone
- 拟用途：Architecture/mechanism references, auxiliary reconstructed examples if clearly labeled

### SWE-Gym OpenHands sampled/verifier trajectories
- URL: https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories
- Verifier: https://huggingface.co/datasets/SWE-Gym/OpenHands-Verifier-Trajectories
- 方法: https://github.com/SWE-Gym/SWE-Gym/blob/main/docs/OpenHands.md
- 已核实要点：6055 sampled trajectories with instance/run IDs, messages and test_result; verifier release has 5272 rows across splits and messages/resolved fields
- 可用边界：Verifier split rows are not necessarily unique examples; outcome success is not the value of changing a harness; dataset-card license is not explicit, verify redistribution/training terms before ingest
- 拟用途：Outcome verification or prefix success-prediction warm-start, after respecting original splits and removing answer leakage

### Nebius SWE-rebench OpenHands trajectories
- URL: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories
- 已核实要点：67074 trajectories, including 32161 successful, from OpenHands v0.54.0 with Qwen3-Coder-480B-A35B; task IDs, patch, binary resolved and agent-test quality fields
- 可用边界：Single-task solver trajectories, not harness self-modification demonstrations; multiple trajectories per task do not imply identical decision prefixes or randomized continuation experiments
- 拟用途：Broad success/failure evidence and verifier warm-start; join task metadata and group splits by repository/issue

### SWE-smith trajectories
- URL: https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories
- 资产: https://swesmith.com/getting_started/assets/
- 已核实要点：Expert agent trajectories and executable synthetic-bug tasks; published training recipe uses about 5000 expert trajectories
- 可用边界：Different formatting subsets must be deduplicated; task-code repair is not automatically harness evolution
- 拟用途：General software diagnosis/planning auxiliary training or task pool

### Darwin Godel Machine
- URL: https://github.com/jennyzzt/dgm
- 已核实要点：Official repository links foundation-model output logs from its self-modification experiments and includes initial-agent logs
- 可用边界：Log contents and training-data completeness not yet audited; code license alone does not establish all logs' terms; adaptive tree branches are not the project's independent K2/M2/L_after2 MC labels；**未** 审计日志是否含全部所需字段，代码许可亦不等于日志可自由训练。
- 拟用途：Closest semantic source to harness evolution; audit released logs for parent source, predecision evidence, explicit plan, actual patch, lineage and comparable evaluations

### SI2CA Training Trajectories
- URL: https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories
- 已核实要点：32340 trajectories on 10780 tasks under three settings; some have two candidate actions and judge logs
- 可用边界：Only selected action is executed; judge sees gold patch; judge score is not a counterfactual return or independent MC target; gold patch and judge fields must stay out of policy inputs
- 拟用途：Optional auxiliary action-ranking or judgment research

## 建议方向

- Evolver 数据：优先审计真实 harness 自修改日志与开源 bugfix 历史；要求改前源码/证据与原始 explicit Plan；由已知 patch 合成的 Plan 须单独标注为辅助数据，不得伪造历史 rationale 或未来结果输入。
- 价值函数：现有成败轨迹仅可作 separately named 的结果验证器 warm-start 或辅助目标；真实 predecision harness 价值须在目标域 state+Plan+续跑上训练；公开 resolved/分数 **不得** 直接称为 G/D/U。
- 下一步（有界）：Audit a small stratified public sample before downloading entire datasets or constructing another ingestion framework; freeze source revision, inspect missing fields/license/dedup/split leakage and report usable yield. No new formal data quota or paid evaluation budget authorized by this research note.

## 脚注（许可，精简）

- ModularRSI 2000 Instances：汇编 CC BY-NC 4.0；ScaleSWE 部分 CC BY 4.0 及第三方条款。
- SI2CA 卡片 CC BY 4.0；底层仓库条款仍适用。
- SWE-Gym 等 HF 卡片许可不明处，**入库前须单独核实**。
- DGM：代码许可不等于全部实验日志可自由训练；日志须逐份审计，**不假定** 已含完整 plan/patch/lineage 字段。
