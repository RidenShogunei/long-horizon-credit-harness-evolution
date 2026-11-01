# 研究一正式采集与单计划 SFT

协议为 `research1-data-v1`，与旧 Foundation、warning-only 和定向修复记录分开。训练对象是进化器：决策时 proposal 源码、证据、本分支历史 → 单份结构化 edit/keep 计划。MC 价值不是基础 SFT 答案；本入口不训练模型。

## 准备和执行校准

在服务器项目根目录，使用 ModularRSI 的 Python 环境并设置 `PYTHONPATH=src`：

基础SFT独立于噪声/MC校准：使用 `data prepare --mode initialization --parents 6 --campaign runs/data_init_...`，再按同一 run/status/resume/export 接口执行。固定父状态队列只来自 proposal，已知900秒不兼容或未准备好的重构建任务在准备阶段排除；未知时长按冻结顺序有界执行并如实记录。单个父任务无评分不阻止下一独立父状态生成计划，全局认证、配置、资产或资源故障仍停止。教师原始计划在实现和子结果之前立即保存，每个父状态后导出，不等待24次噪声探测。

接续初始化排除先前已观察的父任务。需要少量训练及验证父状态时，可重复指定 `--parent-task TASK` 冻结精确队列；原任务分组和可运行性守卫仍适用。预热只使用原始Dockerfile与构建上下文，不替换基础镜像。当前Harbor对未声明docker_image的任务仍按试验构建，手工hb__标签只证明构建缓存准备，不证明实际任务已运行或适配900秒。

求解器也显式传入 `reasoning_effort=none`，并在实际轨迹中核对是否仍输出推理内容。只记录字符数量，不披露推理文本；服务端未遵守冻结请求时阻止入库和盲目重试。决策前已经完成的 proposal 二元结果可作为带父版本与执行来源的独立类型展示，不能混入分数、隐藏测试详情、reward/dev或未来子结果。绑定记录的PASS仅代表有有效分，0分必须映射为任务FAIL，而不是任务成功。

历史运行时预筛保存全部匹配记录，不选择较好成绩；有效0与1同等看待。噪声校准要求8个运行时已验证的debug任务，不足时先采初始化/运行剖面，不能用基础镜像缓存冒充时限认证。接续准备保留原proposal/reward/dev分区，不把已经观察过的proposal任务重新分配到dev。

```sh
python -m research1 data prepare --campaign runs/data_calibration_20261101
python -m research1 data run --campaign runs/data_calibration_20261101 --background
python -m research1 data status --campaign runs/data_calibration_20261101
python -m research1 data resume --campaign runs/data_calibration_20261101 --background
python -m research1 data export --campaign runs/data_calibration_20261101
```

prepare 不调用模型。它冻结 H0、源码、模型政策、任务清单及评分器资产哈希、五模块契约注册表、任务分区与预算。历史暴露任务不能进入新 proposal/reward/dev 分区。proposal 再划分为19个训练任务和5个验证任务；所有后代与路径继承根分区。环境可运行性排序不使用任务成绩。基础模型、Harbor 和任务评分器不参与进化。

历史定向修复会以辅助记录保留，并撤回旧训练导出资格；原始教师输出、任务记录和失败日志不改写。其余旧记录不自动晋升到新协议。

校准上限：常规任务96次（噪声24、reward44、proposal18、dev10），基础设施重试最多32次，总计128次。有效0分不重试，缺分不补0；每作业临时基础设施故障最多2次额外尝试，认证/配置异常阻止后续提交。单任务200轮、900秒，磁盘保留15GB。只清理所属进程组和精确试验容器。

SQLite 账本记录作业、预算占用和事件修订。已完成评分与轨迹会校验哈希；变化不能通过“重新找最新有分文件”修复。正在运行的作业先核对其进程身份，不能直接重启。源码、配置、评分器或已绑定镜像变化会阻止继续。

操作时限属于预算停止，不当作可重试的临时故障。若集成修正需要替换控制器，先明确停止旧 campaign 并保留已占用次数，再用 `data prepare --supersedes OLD --campaign NEW` 接续剩余授权。预算只能交给一个后继，旧账本不能再提交新作业，不能通过另起目录重置96/32/128上限。

## 修改与核验

覆盖 tools、verification、observation、context_mgmt、agent_loop，每次只改一个模块。已注册机制有真实目标与保持性测试；这不是通用语义证明。契约必须在教师调用前绑定真实父状态证据及目标重现；未知或证据不足的机制保持 pending，不能靠合成夹具失败就认定真实缺陷。模型归因与事实复核明确标为 model review，不冒充人工真值。

教师一次保存最多2个候选，然后才实施和评测；不改写教师的 verification_kind。实现器固定模型/温度，保存精确替换和补丁，重放不再生成代码。补丁、父/子树和检查结果在恢复时重新核对。

单计划指令v3要求keep也说明事实复核及证据范围；旧记录缺少版本字段时仍按v2渲染，不能静默改写旧输入。事实复核v3区分有界弃改与全局无缺陷，分别记录实际工具输出、显式执行状态与检查报告。引用须绑定解码后的精确片段及真实步骤、调用、命令哈希和观察位置；JSON响应体只证明观察到该输出，不自动证明成功。独立语义复核仍为必要条件，简单echo/printf成功文字不能替代实际检查。诊断及契约也使用同一解码引用规则，原始回复不改写。

事实复核v4将省略步骤等序列化元数据仅作为context，并允许绑定实际空按键轮询所捕获的观察；生产命令未知时显式记录未知，不将其升级为新鲜输出、执行成功或检查通过。命令回显误标为实际输出仍不通过。

当前v5核验按真实步骤观察的哈希、片段和准确位置核对，不再从完整heredoc屏幕重建生产命令。实际生产者未由记录格式提供时保持未知。命令文字为弱证据，不能独自支持keep；混合命令与真实输出不再误拒整组事实。状态与检查文本只作为观察到的报告，不升级为成功、正确性或因果证明。未被所引证据包含的独立反馈仍不能误引；新v4计划接口为已完成proposal反馈提供单独context引用，不能当作执行观察。

教师v4与诊断v3通用地允许基于具体事实及当前源码提出可实施的流程改进假设，不要求先证明实现缺陷或收益。不得以任务失败、应用错误、检查目录或缺少证据单独推出框架问题；收益与因果留待实际评测。旧v2/v3输入字节保留。固定完整父来源只做一次目标澄清实验，保留全部新keep与失败；若仍无合格功能修改，报告证据不足，不能继续加定向提示逼出edit。

`data export --recheck-factual` 显式冻结全部现有样本、旧资格与旧原始复核文件，按当前确定性核验器派生新的资格修订。只接受原独立模型复核PASS且来源、提示、计划、可见视图、行为及实际执行绑定均成立的记录；原pending/FAIL、坏引用及不完整绑定不自动入选。原教师、旧复核和旧资格保留；同指纹恢复不重复升级，撤销不能被缓存恢复。该操作不调用模型或重跑任务，也不将新增验收修订称为新增独立父状态。

需要在新协议下复用已保存的父证据时，使用 `data prepare --mode initialization --replan-from ORIGINAL --supersedes LATEST --campaign NEW`。来源与预算前驱分开，完整来源队列在兼容性校验后冻结，不挑选较好题目；保持原root与train/validation分组，不增加独立父状态数，不重新执行父求解器，也不把旧评分冒充新协议执行。新教师及独立复核原文另存，修改后的子执行才新增评测占用。所有导出入口复查原始来源哈希及撤销状态，失效时撤销派生资格并发布排除后的版本。原来的教师、复核、评分和pending记录保持不变。

机制注册表v2增加completion_requirement_recheck：只有具体可见需求覆盖遗漏被独立归因后，才核验首次完成确认是否动态传递原始任务要求与行为检查请求。探针覆盖普通轮次、两次确认、工具顺序与反馈等13项保持性，长指令新增提示最多8192字符。该检查只证明提示与上下文连接，不证明求解器遵守、任务正确或分数上升；修改样本仍须真实子版本proposal执行。

## 真实 MC

固定 K=2、M=2、L_after=2：首轮候选之后还有2次决策。每路径通过独立本地任务抽样与新的求解轨迹续演化，历史隔离；求解 API 未提供可验证种子时明确记录，不声称位级可复现。续策略只执行其保存的第一候选，不看 reward 选候选。

真实 keep 使用新决策节点引用相同代码与本路径既有评分；后续应用失败保留当前代码和失败反馈。首轮应用失败、相同实现重复候选及缺失评测不产生完整的双候选价值验收。没有真实功能首改时提前返回 INSUFFICIENT_EVIDENCE，不耗尽预算填充纯 keep 路径。

G、D、U 独立存储，λ=0.5；D 是逐题 parent-step 正下降再加权。对照叶节点另用 dev 检查，但不反馈教师。比较表默认 preference=undetermined，不能用微小或不确定差异硬造胜负。

## SFT 导出与训练就绪

`exports/sft/CURRENT.json` 原子指向一个不可变的数据集版本。edit、keep、mixed 的 train/validation 消息及元数据分开保存。混合集 keep 最多20%，没有修改样本时不以 keep 充数。模型消息只有决策时输入和单计划答案；真实分数、补丁、MC 和资格保留在 sidecar。

合格修改须事实复核、可执行行为与保持性检查及绑定实际子版本执行；keep 须独立事实复核和有效父 proposal 执行。事实错误、截断、未知机制、缺分、撤销及旧定向样本不进入正式正例。采集资格、SFT可用和目标模型训练就绪是不同状态。

可用 `data export --target-profile PROFILE.json` 指定学生模型，本地 JSON 只允许 `model_id`、`tokenizer_path`、`max_context_tokens`。使用本地 tokenizer 与 chat template 验证实际长度和 assistant-only labels，不下载模型或执行远程模型代码。未选定模板、上下文溢出、没有验证集或没有修改样本时不标记 training_ready。

`data revoke --campaign ... --reference ID --reason ...` 会撤销关联的样本、价值和比较记录，重建活跃导出；历史事件保留。原始数据永远不是把所有记录无差别拼成正向 SFT。

## 正式配额扩采

真实校准通过后，才允许 `data prepare --from-calibration CALIBRATION --scale-config APPROVED.json --campaign FORMAL`。配额 JSON 显式提供 targets（origins、verified_edits、complete_mc_labels、mechanisms、covered_modules、validation_edits）、budget（regular、retries、total）、sampling（R、reward_n、dev_n）、max_groups，可选 target_profile。这些规模与预算须由用户在成本报告后确定；没有默认无上限采集。

正式控制器串行运行登记的根状态组，累计所有子账本实际消耗；每第5组使用验证 proposal 分区。不允许直接从 CLI 启动子组绕过全局预算。达到全部配额且 SFT 模板检查通过才标记 DATA_COLLECTION_COMPLETE；否则报告缺口、证据耗尽、预算耗尽或 NEEDS_STUDENT_PROFILE。校准本身最多到 NEEDS_SCALE_LOCK。

`report.json/report.txt` 记录实际token、耗时、噪声翻转、覆盖与缺口。供应商零价格元数据不能证明免费，金额未知会保留未知。测试回调的分数只用于软件测试，不能作为真实校准证据。

## 2026-11-01 采集实现说明（C18 / 字段精度）

独立决策时复核曾排除两个 V13 pending 候选：其一在计划声明的运行时数据路径上不可行；其二对来源 case 的归因与可见证据不一致。原始教师与复核记录均保留，未改写。

当前 app3 事实引用（fact-ID）与显式 opt-in 的 plan precision 字段审计并存：后者要求独立复核 problem_hypothesis、intended_change、preserve_behaviors 与 verification_plan 中每条非空字符串；任一字段被判 incorrect 或 uncertain 时，不能给出正向 admission PASS。整字段覆盖不等于对每条原子事实的机械证明，独立语义接受仍为必要条件。新 profile 仅适用于此后新采集的计划，不追溯升级或改写既有计划与已保存复核。

通用导出仍为 `training_ready=false`，未指定 student profile。不声称已有首个合格 edit、已完成 MC 或采集完成；不在提示中写入具体专家修复方向。上述两类 pending 的失败原因仅作观测说明，不进入模型输入。
