# 基于长程信用分配的 Harness 稳定进化方法

**Stable Harness Evolution via Long-Horizon Credit Assignment**

本仓库研究如何让模块化智能体 harness 在长程任务上稳定进化。进化对象是 ModularRSI 的基础模块化 Terminus2，共五个 harness 模块。本机已经跑通的 KIRA / Meta-Harness 复现位于 `/home/jinxu/meta-harness-repro`，只作为对照基线，只读引用，不在这里修改或替换。Harbor 负责任务执行和评分。被进化的是 Terminus2 的 harness 组件，不是基座模型，也不是 Harbor 本身。

## 五个模块

下面只记录职责。本仓库现在不实现进化。

| 模块 | 职责 |
| --- | --- |
| `agent_loop` | 步数、重试、异常处理 |
| `observation` | 终端输出的截断、过滤与组织 |
| `tools` | 工具描述、调用解析、执行逻辑 |
| `context_mgmt` | 压缩、保留与历史 |
| `verification` | 智能体自己的完成判定。最终分数仍由外部任务测试给出 |

两项研究共用这一套 harness：

1. **长程价值。** 从同一个子版本做蒙特卡洛分叉，估计一次修改的长程价值。
2. **哪些组件必须一起改。** 依赖图是输入。联合方案成功，并不能证明图中每一条依赖都正确。

## 三层数据

- **第一层，任务。** ModularRSI 终端任务，共 1000 题：120 道带标签的训练题，880 道带标签的任务池。任务池是进化任务的来源，不是留出测试集。保留题目说明、环境、测试和参考解。可以改任务入口，不要把任务压成问答文本。容器和评分器保持原样。
- **第二层，执行轨迹；第三层，修改记录。** 这两层必须由我们自己生成。回路是：父代运行，修改计划（可以同时改多个组件），固定编辑器打上补丁，得到子版本，再做同配置的前后对比，并加第二批任务看泛化。完整记录要留下成功、无效果、退步和补丁应用失败，并把环境故障与方法失败分开。训练样本是计划加上这次执行的结果，不只是最终 diff。
- ModularRSI 公开的 267 条轨迹，以及固定 harness 的重复评测，都不是本项目的训练集。初始化时不下载 DGM 外部日志。

采集回路见 `docs/data_protocol.md`。

## 第一个里程碑

在 10 到 20 道终端任务上走完一轮单组件修改：基础 harness，一次编辑，前后验证，然后落下完整记录。在此之前不做进化搜索。

协议已冻结为 `research1-v0`。H0（ModularRSI `static` 五个 baseline）已在 `fix-git` 上冒烟通过；有代理时外部 reward=1.0。详见 `docs/experiment_protocol.md` 与 `experiments/h0_smoke_20261028_report.md`。

## 现在还没有的东西

本仓库还没有生成的训练数据，也不纳入 ModularRSI 源码。运行时在仓库外：`/home/jinxu/arms_cache/modularrsi/ModularRSI`。目录约定见 `docs/repo_layout.md`，本机环境见 `docs/environment.md`。

## 上游许可

ModularRSI 原始代码带有非商业使用限制。本仓库不纳入其受版权保护的源代码。使用上游代码或任务数据时须遵守该限制。这里以后写下的笔记和我们自己产生的记录，不改变上游许可。
