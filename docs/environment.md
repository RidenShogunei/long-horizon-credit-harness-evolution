# 环境

记录时间：2026-10-28（Asia/Shanghai）。

- GitHub 账号：`RidenShogunei`（`gh auth status` 已登录 github.com）。
- 默认 `python3 --version`：Python 3.10.12。
- 本机另有 `python3.11`：Python 3.11.15；`python3.12`：Python 3.12.13。
- 本仓库目标解释器是 Python 3.11+，见 `pyproject.toml` 的 `requires-python`。当前没有钉死补丁版本，也没有声明第三方依赖。
- 运行时栈尚未装进本仓库：Harbor、ModularRSI、模型端点都还没有。对照基线在 `/home/jinxu/meta-harness-repro`，只读；不从那里复制密钥。

本地 `git config user.email` 为空，本次没有修改 git config。初始提交的作者邮箱只在这一次提交命令里指定为 GitHub noreply 地址 `108864288+RidenShogunei@users.noreply.github.com`。
