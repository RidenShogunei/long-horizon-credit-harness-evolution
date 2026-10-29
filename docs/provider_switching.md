# LLM 供应商切换（周限额）

同济 `agi.tongji.edu.cn` 有周限额，且没有公开的用量查询接口。本仓库用**本地周账本 + 可切换/自动回退**处理。

## 两个供应商

| id | 用途 | `api_base` | Harbor `-m` |
| --- | --- | --- | --- |
| `tongji_agi` | 学校网关（有周限额） | `https://agi.tongji.edu.cn/v1` | `openai/DeepSeek-V4.1-Flash` |
| `deepseek_official` | 官方 DeepSeek（回退） | `https://api.deepseek.com` | `openai/deepseek-flash` |

密钥只在 `/home/jinxu/.hermes/.env`：`TONGJI_AGI_API_KEY`、`DEEPSEEK_API_KEY`。  
状态在 `/home/jinxu/.hermes/llm_provider_state.json`（不进 git）。

## 常用命令

```bash
# 看当前激活谁、本周已用多少
python3 scripts/llm_provider.py status

# 模式
python3 scripts/llm_provider.py use auto              # 默认：先同济，超额/报错再官方
python3 scripts/llm_provider.py use tongji_agi        # 强制同济
python3 scripts/llm_provider.py use deepseek_official # 强制官方

# 知道周限额（token 合计）后写入本机 env
python3 scripts/llm_provider.py set-weekly-limit tongji_agi 2000000

# 给当前 shell / 子进程注入 Harbor 所需环境变量
source scripts/with_llm_provider.sh
source scripts/with_llm_provider.sh tongji_agi
bash scripts/with_llm_provider.sh auto -- env | grep -E 'API_BASE|MODEL|LLM_PROVIDER'

# 小探测（会计入周账本）
python3 scripts/llm_provider.py probe

# 手动标记本周同济已用尽 / 清标记
python3 scripts/llm_provider.py exhaust tongji_agi --reason 'portal shows weekly cap'
python3 scripts/llm_provider.py clear-exhausted tongji_agi
```

## 自动回退逻辑

1. `auto` 按 `tongji_agi → deepseek_official` 选。
2. 若同济被标记 `exhausted`，或 `TONGJI_AGI_WEEKLY_TOKEN_LIMIT` 已达，则跳过同济。
3. 探测/调用若返回含 `quota` / `429` / `限额` 等，会自动 `exhaust` 同济，下次解析走官方。
4. 评测脚本应 `source scripts/with_llm_provider.sh` 后再跑 Harbor，这样 `API_BASE`、`MODEL`、`HARBOR_EVO_API_KEY` 一致切换。

## 与评测脚本

冒烟等脚本默认仍可用官方 DeepSeek。需要走同济或 `auto` 回退时，先：

```bash
source scripts/with_llm_provider.sh auto
# 再运行对应 harbor / 评测命令，使用已导出的 API_BASE / MODEL / KEY
```
