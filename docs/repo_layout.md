# 目录

| 路径 | 用途 |
| --- | --- |
| `src/` | 以后的 harness 适配、记录读写和编辑器调用。现在是空的。 |
| `configs/` | 以后的任务批次与前后对比配置。不放密钥。 |
| `scripts/` | 以后的采集与检查脚本。 |
| `data/tasks/` | 第一层任务的索引。任务本体保持终端任务形态，不摊成问答文本。索引尚未建立。 |
| `data/executions/` | 第二层执行轨迹。必须是我们自己跑出来的。 |
| `data/modifications/` | 第三层修改记录：计划、结果和故障类别。 |
| `experiments/` | 某一次里程碑或研究运行的配置与小结。 |
| `docs/` | 协议与环境说明。 |

对照基线不在本仓库里，路径是 `/home/jinxu/meta-harness-repro`。不要把该目录里的密钥或环境文件复制过来。

`.gitignore` 排除 `.env`、密钥、`__pycache__`、`.venv`、`data/raw/`、`datasets/`、大体积轨迹、Docker volume、`*.pt`、`jobs/` 和 `results/raw/`。这些不要提交。
