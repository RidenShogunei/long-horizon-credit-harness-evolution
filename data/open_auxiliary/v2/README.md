# 研究一开源辅助规划数据 v2

已审核 **23 条 edit**：20 条训练、3 条按仓库隔离的验证样本；keep 为 0。本轮新增 19 条，保留旧版 4 条，撤销旧版 1 条。

来源为 smolagents、CrewAI、DSPy、LiteLLM 的已发布问题/变更记录。每条输入含当时源码和报告证据，答案是一份统一 JSON Plan。源码、原始补丁、测试记录、来源及许可另存，可追溯和撤销。

`evolver_framework_aux.*.jsonl`：直接框架规划辅助数据，训练 14 / 验证 2。
`evolver_infrastructure_aux.*.jsonl`：工具提供商等基础设施辅助数据，训练 6 / 验证 1。
`evolver_aux.*.jsonl`：两分区合并，训练 20 / 验证 3。`messages` 为训练字段，仅 assistant 答案计算损失。

验收包括实际固定版本的源码/补丁重建、局部修改前后行为、保持性、证据与答案对应、格式及按仓库分组。96 条原始来源记录中，66 条完成补丁重建；它们不会全部进入正向数据。完整状态见 acceptance_ledger.json。

这是辅助初始化数据。学生模型尚未锁定，未开始训练。完整上游测试、TerminalBench 迁移效果尚未验证；正式自主规划 SFT 和真实 MC 价值标签均为 0。不得把公开补丁结果补成 G、D、U。原始回复/记录与失败保留，正式采集预算本轮未消耗。

许可和来源见 source_cards.json；编译数据集与原仓库代码的许可分别保存。工作脚本和验收证据见 data_build_scripts/、root_verification/、portable_proofs/。局部验证依赖 fixtures；这些记录不声称已运行完整项目环境。

另使用服务器现有 Qwen3-4B-Instruct-2507 的 tokenizer 做只读格式检查：23 条均通过，最长 13,257 tokens（16,384 上限），未截断；只监督 assistant 的答案边界已核对。未加载模型权重，训练与推理次数为 0。该检查不等于已选定正式学生模型。
