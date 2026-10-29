# H0 冒烟报告（2026-10-28）

`protocol_id`: `research1-v0`  
上游：`IQuestLab/ModularRSI` @ `ed1e545bc60f050b404474e51ee18021777a0e23`  
H0：`generations/merged_active/gen_0/modules`，`composer_name=static`  
日志确认：`composer chose: loop=baseline obs=baseline ctx=baseline tools=baseline verify=baseline`  
求解器：`openai/deepseek-flash`，`api_base=https://api.deepseek.com`，`temperature=0.7`  
任务：`terminal-bench@2.0` / `fix-git`  
对照基线 `/home/jinxu/meta-harness-repro` 未改动。

## 验收

| 条件 | 结果 |
| --- | --- |
| 至少一个任务完整执行并外部评分 | 是 |
| 失败能落到具体一层 | 是（无代理时：`scorer/infrastructure`） |
| 五模块显式载入 | 是 |
| 独立 trial，无共享残留工作区 | 是（不同 `fix-git__*` trial 目录） |
| 代理下真实任务分 | `reward=1.0` |

## 运行对照

### A. 无代理（`experiments/h0_smoke_20261028_first`）

两次独立 trial：

| attempt | trial | reward | 失败层 | episodes | tokens in/out |
| --- | --- | ---: | --- | ---: | --- |
| 1 | `fix-git__58QZZYS` | 0.0 | scorer/infrastructure：`test.sh` 从 GitHub 拉 `uv` 失败，`uvx` 不存在 | 14 | 88482 / 8521 |
| 2 | `fix-git__RTRP9h4` | 0.0 | 同上 | 12 | 55091 / 7300 |

智能体侧 `agent_loop` 报告 `finished success=True`；外部评分仍为 0。这说明内部 `verification` 不能代替外部测试。

### B. 有代理（`experiments/h0_smoke_20261028_proxy`）

通过 `--ve/--ae` 注入 `HTTP(S)_PROXY=http://172.17.0.1:7890`（docker bridge gateway）。

| trial | reward | verifier | episodes | tokens in/out/cache |
| --- | ---: | --- | ---: | --- |
| `fix-git__*` under proxy | **1.0** | `test_about_file`、`test_layout_file` 均 PASSED | 10 | 47965 / 10265 / 40832 |

第二次代理复跑 `h0_smoke_20261028_proxy2` / trial `fix-git__S25X6kL`：**reward=1.0**（episodes=13）。两次代理复跑均为满分，且使用不同 trial 目录，未见残留工作区污染评分。

## 环境结论

1. ModularRSI 嵌入的 Harbor **没有** Harbor 0.24 的 `--extra-docker-compose`；本机用 verifier/agent 环境变量 + bridge gateway 代替。可选 overlay 仍放在 `infra/host-proxy.compose.yaml`。
2. 后续评测默认必须带代理，否则会把评分器网络故障误记成方法失败。
3. 磁盘停止线仍为 15GB；冒烟时约 122GB 可用。

## 下一步（协议第 3 步）

在代理开启的前提下，抽约 8 个轻量终端任务，各跑 3 次，估计 H0 成绩波动与单次成本。正式 24+24+24 划分先登记，不在这一周拉满。
