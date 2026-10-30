# 研究一实验协议（research1-v1）

机器可读副本：`configs/protocol.v1.json`、`configs/roles.v0.json`、`configs/research1/pilot.yaml`。  
旧版 `configs/protocol.v0.json` / 本文档 v0 叙述保留只读对照。  
每条执行记录和修改记录都必须带 `protocol_id=research1-v1`；进化器开始训练后还要额外带 `evolver_policy_id`。

## 0. 本阶段边界（Foundation）

交付共用执行/版本/记录底座，并证明双线冒烟：

- **P2A**：官方 ModularRSI `PROFILE=smoke`，真实 A13 日志
- **P2B**：同父快照 → 两候选 → 各 1 次续跑（缩小为 2 题 × 2 次，不是 nominal 60）

不做：价值模型训练、正式 4×3×5=60 / MC=84、P4 结论、五模块大规模复现。  
禁止用模拟分数宣称 PASS。

## 1. 对象与角色

| 角色 | 是否进化 | 配置 |
| --- | --- | --- |
| 求解器 | 否 | P2A：`temperature=0.0`（官方 Phase0）；P2B：DeepSeek Flash `0.7`，`thinking.disabled` |
| 初始进化器 | 训练前冻结；训练时才更新 | 同模型族，只输出修改计划 |
| 编辑器 | 否 | 固定补丁应用；重放时再打保存下来的补丁 |
| Harbor | 否 | 任务容器与外部评分 |
| H0 | 是（研究一的对象） | ModularRSI `static` 五个 `baseline`；见 `configs/h0.snapshot.v0.json` / `h0.lock.json` |
| KIRA / Meta-Harness | 否 | `/home/jinxu/meta-harness-repro`，只读对照 |

v1 基础阶段只锁一个 `LOCKED_MODULE`（校准后写入 `tasks.lock.json`），**不预锁 observation**。

## 2. 任务隔离

训练题名只从数据集 `manifest.json` / `train_120.txt` 读取，禁止手填。  
`fix-git` 若不在 train，仅作环境联通记录，不进进化集。

校准：固定顺序 8 个 train × 最多 3 trial → 筛 4 题写入 `configs/research1/tasks.lock.json`。

## 3. 评测与失败

- 最终成绩只认外部任务测试。
- `unknown ≠ 0`：官方 reward 缺失时保持 `reward=null`。
- 失败层：`model` / `harness` / `container` / `scorer` / `infrastructure`。
- 必须代理 `http://172.17.0.1:7890`；必须 `source scripts/with_llm_provider.sh`；磁盘 <15GB 停。

## 4. 长程指标（v1）

- **主退化指标**：相对 **parent 步** 的下降之和（只计正下降）
- **best_ancestor**：诊断列，不进主 U
- **λ** 工程默认 `0.5`；始终分存 `terminal_gain` 与 `cumulative_regression`
- **U** = `terminal_gain - λ * cumulative_regression`

## 5. CLI

```bash
python -m research1 preflight --config configs/research1/pilot.yaml --write-h0-lock
python -m research1 calibrate --config configs/research1/pilot.yaml
python -m research1 run-modularrsi --config configs/research1/pilot.yaml
python -m research1 run-branches --config configs/research1/pilot.yaml --continuations 1
python -m research1 audit --run RUN_ID
python -m research1 report --run RUN_ID
```

退出语义：`0=PASS`，`1=FAIL`，`2=INCOMPLETE`。

## 6. 上游与本仓库边界

- ModularRSI 检出：`/home/jinxu/arms_cache/modularrsi/ModularRSI`（不拷进本仓）
- 数据集：`/home/jinxu/arms_cache/modularrsi/ModularRSI_2000_Instances`（CC BY-NC）
- 不修改：`/home/jinxu/meta-harness-repro`
