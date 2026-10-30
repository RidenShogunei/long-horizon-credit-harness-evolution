# 目录

| 路径 | 用途 |
| --- | --- |
| `src/research1/` | research1-v1 共用底座：契约、版本、评测、规划、分叉、回报、CLI |
| `configs/` | 协议与角色配置。不放密钥。 |
| `configs/protocol.v0.json` | 旧协议只读对照 |
| `configs/protocol.v1.json` | **当前冻结协议 research1-v1** |
| `configs/research1/pilot.yaml` | 试点运行配置（含 config_hash） |
| `configs/research1/h0.lock.json` | H0 哈希/路径/代理/供应商冻结 |
| `configs/research1/tasks.lock.json` | 校准后的 4 题 + LOCKED_MODULE |
| `scripts/` | provider 切换、磁盘检查、H0 冒烟、`with_llm_provider.sh` |
| `data/tasks/` | 第一层任务索引（本体在仓库外数据集） |
| `data/executions/` | 第二层执行轨迹索引约定 |
| `data/modifications/` | 第三层修改记录约定 |
| `runs/` | 运行产物：`modularrsi_*`（P2A）与 `branches_*`（P2B）隔离 |
| `reports/` | 可提交的摘要/索引 |
| `experiments/` | 早期里程碑小结（如 H0 冒烟报告） |
| `docs/` | 协议与环境说明 |

对照基线不在本仓库里，路径是 `/home/jinxu/meta-harness-repro`。不要把该目录里的密钥或环境文件复制过来。

数据集（CC BY-NC，不 vendor）：`/home/jinxu/arms_cache/modularrsi/ModularRSI_2000_Instances`。  
上游检出：`/home/jinxu/arms_cache/modularrsi/ModularRSI` @ `ed1e545`。

`.gitignore` 排除 `.env`、密钥、`__pycache__`、`.venv`、`data/raw/`、`datasets/`、大体积轨迹、Docker volume、`*.pt`、`jobs/`、`results/raw/`、`runs/**` 大轨迹。这些不要提交。
