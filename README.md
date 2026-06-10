# StockWatch — A 股个人智能盯盘机器人

> 一个可以跑在树莓派、NAS、macOS 或 Linux 服务器上的 A 股个人盯盘机器人：每天自动看自选股、公告、新闻、技术面和持仓风险，并通过飞书提醒你该关注什么。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ranking-00A35C)](https://github.com/microsoft/LightGBM)
[![AKShare](https://img.shields.io/badge/Data-AKShare-blue)](https://github.com/akfamily/akshare)
[![Feishu](https://img.shields.io/badge/Bot-Feishu%2FLark-00D6B9)](https://github.com/larksuite/oapi-sdk-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

中文 | [English](#english)

**关键词**：A股量化、股票盯盘、股票机器人、飞书机器人、A股新闻分析、股票公告解读、LightGBM 排序模型、Alpha158 因子、AKShare、macOS/Linux 部署、树莓派 24h 服务。

StockWatch 是一个面向 A 股个人投资辅助场景的轻量级系统：用 AKShare、腾讯财经等数据源获取行情/新闻/公告，结合技术面、消息面、Alpha 因子和 LightGBM 排序模型，最终通过飞书机器人给出“偏向建议 + 风险提醒 + 观察价位”的中文解释。你也可以像和 AI 聊天一样问它“贵州茅台最近怎么样”“现在行情怎么样”，它会尽量引用公告、新闻、资金流、财务和走势数据回答。

> 仅供学习、研究和家庭辅助决策使用，不构成任何投资建议。

---

## 功能亮点

- **A股盯盘推送**：按早盘前、午间、收盘后自动运行，非交易日跳过。
- **飞书自然语言问答**：支持股票代码、股票名称、买入跟踪、盯价、取消盯价，也支持“现在行情怎么样”这类自然输入。
- **深度问答**：可回答“600449 最近一周走势如何”“宁夏建材重组怎么样”“我想看看贵州茅台行情怎么样”等问题，并优先引用公告/新闻来源。
- **消息面分析**：抓取近况新闻、公司公告、研报、资金流、财务快照和市场关注信息，交给模型生成可读建议。
- **量化因子**：计算扩展版 Alpha158/Alpha300 风格因子，覆盖动量、波动、Beta、流动性冲击、相对强弱、回撤和成交量结构。
- **LightGBM 排序模型**：离线训练 A 股横截面排序模型，线上作为辅助信号参与解释。
- **本地 Dashboard**：只读展示最近运行、信号、持仓跟踪、盯价提醒和 5 日信号复盘。
- **信号复盘报告**：基于本地 SQLite 中的历史决策和 K 线生成命中率/窗口收益报告。
- **macOS/Linux 部署**：systemd/命令行常驻服务 + SQLite 本地存储；作者使用树莓派 5 低成本 24 小时运行。
- **Docker Compose 部署**：可用容器跑定时盯盘、Dashboard 和可选飞书机器人。

---

## 实测效果

自然语言提问示例：`我想看看贵州茅台行情怎么样`。机器人会返回结论、公告/新闻、最近走势、资金/基本面/关注度、中线视角、偏向建议、主要风险和资料来源。

![飞书自然语言问答上半部分](docs/assets/feishu-research-top.png)

![飞书自然语言问答下半部分](docs/assets/feishu-research-bottom.png)

---

## 快速开始

### 1. 5 分钟体验（无需飞书）

先跑一个终端 demo。没有 MiniMax API Key 时会降级输出规则化行情快照；配置了 MiniMax 后会输出完整自然语言问答。

```bash
# 克隆本仓库后进入项目目录
cd StockWatch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 个股自然语言问答
python main.py demo "600519 最近一周走势如何"

# 大盘/行情问答
python main.py demo "现在行情怎么样"
```

### 2. 配置凭证

```bash
cp .env.example .env
nano .env
```

必填项：
- `MINIMAX_API_KEY` — MiniMax API 密钥
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` — 飞书自建应用凭证
- `FEISHU_RECEIVE_ID` — 接收人 open_id/user_id/email

### 3. 修改自选股

```bash
# 编辑 .env 中的 WATCHLIST（逗号分隔代码）
WATCHLIST=600519,000858,510300,510500,159915
# 格式：6位股票代码，ETF同理（159915=创业板ETF）
# 保存后重启服务：systemctl restart stockwatch
```

### 4. 运行模式

```bash
source .venv/bin/activate

# 自检（AKShare / MiniMax / 飞书）
python main.py test

# 立即运行一次完整流程
python main.py once

# 守护进程模式（按调度自动运行）
systemctl start stockwatch
systemctl status stockwatch

# 飞书交互式查询机器人（SDK 长连接）
python main.py bot

# 本地只读 Dashboard
python main.py dashboard

# 生成信号复盘报告
python main.py report --horizon 5 --output reports/backtest.md
```

### 5. Docker Compose

```bash
cp .env.example .env
nano .env

# 定时盯盘 + Dashboard
docker compose up -d stockwatch dashboard

# 可选：启用飞书长连接机器人
docker compose --profile bot up -d

# 查看 Dashboard
open http://127.0.0.1:8765
```

容器数据会写入 `stockwatch-data` volume，对应程序内的 `/root/.stockwatch`。

---

## 日志

```bash
# 实时日志
tail -f ~/.stockwatch/logs/stockwatch_$(date +%Y%m%d).log

# 7天保留，滚动删除
~/.stockwatch/logs/
```

---

## 数据库

```bash
# 查看 SQLite 数据
sqlite3 ~/.stockwatch/db.sqlite

# 查看最近运行记录
sqlite3 ~/.stockwatch/db.sqlite "SELECT * FROM runs ORDER BY run_ts DESC LIMIT 5;"

# 查看推送记录
sqlite3 ~/.stockwatch/db.sqlite "SELECT run_id, code, name, action, confidence, pushed FROM decisions ORDER BY run_ts DESC LIMIT 20;"
```

---

## 本地 Dashboard

```bash
python main.py dashboard
```

默认地址：`http://127.0.0.1:8765`。

Dashboard 是只读页面，直接读取本地 SQLite，展示：

- 最近运行记录
- 最近信号和置信度
- 活跃持仓跟踪
- 活跃盯价提醒
- 5 日信号复盘摘要

如果用 Docker：

```bash
docker compose up -d dashboard
```

---

## 信号复盘报告

```bash
# 统计信号后 5 个交易日表现，输出到终端
python main.py report --horizon 5

# 写入 Markdown 文件
python main.py report --horizon 5 --output reports/backtest.md
```

报告会基于本地 `decisions` 和 `daily_kline` 计算 BUY/SELL/HOLD 的样本数、命中率、平均窗口收益和中位窗口收益。命中率定义很朴素：BUY 后窗口收益为正、SELL 后窗口收益为负、HOLD 后窗口收益在 ±2% 内。

这只是事后研究复盘，不是收益承诺，也不会修改数据库。

---

## 调度说明

每日自动运行时间（Asia/Shanghai）：
- **09:10** 早盘前：隔夜消息 + 当日策略
- **12:30** 午间：上午盘面回顾 + 下午建议
- **15:15** 收盘后：全天复盘 + 次日观察

非交易日（周末/节假日）自动跳过。

## 飞书交互式查询

开启 `stockwatch-bot` 后，可以在飞书里直接给机器人发消息：

```text
600519
现在行情怎么样
我想看看贵州茅台行情怎么样
600449 最近一周走势如何
买入 600519 1680
买入 600519 1680 100股
盯买 600519 1500
盯买 600519 1500 100股
取消盯价 600519
卖出 600519
停止跟踪 600519
```

`股票代码` 会即时回复单只股票分析；自然问句会走深度问答，识别到股票时优先回答个股，未识别到具体股票时会回答大盘/行情问题；`买入` 会写入持仓跟踪，后续定时盯盘会把这只股票加入分析池，触发止损、接近目标价或模型转为 SELL 时主动推送；`卖出` 会停止跟踪。

`盯买` 会设置加仓价提醒。盘中轻量监控每 5 分钟检查一次，触价时会结合五档盘口和内外盘判断卖压；如果卖压偏重，会提示先别急着加仓或考虑撤挂单。系统也会每 30 分钟扫描自选股、持仓和盯价股的重大新闻，避免重复提醒。

---

## v2 升级路径

```bash
cd ~/stockwatch && source .venv/bin/activate

# 1. 拉取新代码后先做增量迁移（可重复执行）
python scripts/migrate_v2.py

# 2. 编辑 .env，按模块逐个开启
nano ~/stockwatch/.env

# 3. 先自检，再重启服务
python main.py test
systemctl restart stockwatch
```

v2 新增开关默认关闭：

```bash
ENABLE_CALIBRATION=false
CALIBRATION_LOOKBACK_DAYS=5
CALIBRATION_MIN_SAMPLES=50

ENABLE_ALPHA158=false

ENABLE_LGBM=false
LGBM_MODEL_PATH=~/.stockwatch/models/lgbm.txt

ENABLE_REGIME=false

ENABLE_SECTOR=false
```

建议开启顺序：`ENABLE_REGIME` → `ENABLE_SECTOR` → `ENABLE_ALPHA158` → `ENABLE_CALIBRATION` → `ENABLE_LGBM`。

### LightGBM 离线训练

```bash
# 在 Mac 上
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt

python scripts/bootstrap_history.py
python scripts/build_training_set.py
python scripts/train_lgbm.py

# 输出 models/lgbm.txt 和 models/lgbm_meta.json 后，拷到部署机器
scp models/lgbm.* <user>@<host>:~/.stockwatch/models/
```

部署端只在 `ENABLE_LGBM=true` 时加载模型；模型缺失会记录日志并跳过。

当前默认训练配置使用 `stable` 因子子集，聚焦历史验证中更稳定的流动性冲击、Beta、阶段位置、相对动量和波动类因子。如需回到全量因子训练：

```bash
STOCKWATCH_LGBM_FEATURE_SET=all python scripts/train_lgbm.py
```

---

## Roadmap / 欢迎贡献

已经完成或内置的方向：

- 终端 `demo` 体验，不配置飞书也能先看输出。
- Docker Compose 部署。
- 本地只读 Dashboard。
- 基于历史决策的信号复盘报告。
- 更清楚的数据来源、服务边界和投资风险说明。

欢迎社区补充的方向：

- 企业微信、Telegram、钉钉等推送适配。我目前只长期使用并验证了飞书，所以其他通道欢迎提 issue 或 PR。
- 更漂亮的 Dashboard 视图，例如个股详情页、信号走势和持仓曲线。
- 更完整的回测维度，例如分行业、分置信度、分市场状态的表现拆解。
- 更低门槛的配置向导，例如自动检查飞书权限、接收人 ID 和模型连通性。

---

## 数据与模型说明

- 行情数据主要来自 AKShare 封装的数据接口和腾讯财经公开行情接口。
- 公告/新闻/研报等信息会优先参考巨潮资讯公告、东方财富公告/新闻/研报/资金流等公开来源。
- LightGBM 模型以离线历史数据训练，默认标签为 20 日前瞻收益并加入回撤惩罚，线上只作为辅助排序信号。
- 模型输出不是买卖指令；最终回复会结合价格、趋势、消息面、风险位和观察价位综合表达。

---

## 开源引用与致谢

StockWatch 站在这些开源项目之上。以下列出直接依赖或训练/运行中明确使用的主要仓库：

| 项目 | 用途 | 许可证 |
| --- | --- | --- |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | Alpha158 因子命名和量化研究范式的概念参考；本项目为独立 pandas 实现 | MIT |
| [AKShare](https://github.com/akfamily/akshare) | A 股行情、新闻、公告、资金流、板块和财务数据接口 | MIT |
| [LightGBM](https://github.com/microsoft/LightGBM) | LambdaRank 横截面排序模型训练与推理 | MIT |
| [pandas](https://github.com/pandas-dev/pandas) | 表格数据处理、时间序列和训练集构建 | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | 数值计算和因子计算 | BSD-3-Clause |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | NDCG 等模型评估指标 | BSD-3-Clause |
| [Apache Arrow / pyarrow](https://github.com/apache/arrow) | Parquet 训练数据读写 | Apache-2.0 |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | OpenAI 兼容接口客户端，当前用于接入 MiniMax API | Apache-2.0 |
| [Lark/Feishu OpenAPI Python SDK](https://github.com/larksuite/oapi-sdk-python) | 飞书长连接机器人和开放平台 SDK | MIT |
| [Requests](https://github.com/psf/requests) | HTTP 请求 | Apache-2.0 |
| [urllib3](https://github.com/urllib3/urllib3) | HTTP 连接池基础库 | MIT |
| [Loguru](https://github.com/Delgan/loguru) | 日志系统 | MIT |
| [Tenacity](https://github.com/jd/tenacity) | LLM/API 调用重试 | Apache-2.0 |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` 配置加载 | BSD-3-Clause |
| [tqdm](https://github.com/tqdm/tqdm) | 训练/下载进度条 | MPL-2.0 / MIT |

这里的“引用”包含直接依赖和明确概念参考；StockWatch 与上述项目维护者没有隶属、背书或商业合作关系。各项目版权和许可证归原作者所有。

## 数据与服务声明

- 腾讯财经、巨潮资讯、东方财富、飞书开放平台、MiniMax 等属于各自公司或机构提供的数据/服务来源，不是本仓库的开源组成部分。
- 本项目不会重新分发第三方行情、公告、新闻或研报原文；运行时数据获取应遵守对应平台的服务条款、robots/接口限制和适用法律法规。
- 本项目不会连接券商账户，不会下单，不包含自动交易能力。
- `.env`、SQLite 数据库、日志、模型文件和飞书/MiniMax 凭证都应保留在本地或私有部署环境，不应提交到公开仓库。
- 模型分析可能存在延迟、缺失、误判或源数据错误，不应作为自动交易或唯一投资依据。

## 许可证

本项目采用 [MIT License](LICENSE)。MIT 对个人工具、研究项目和二次开发比较友好：允许使用、复制、修改、分发和商用，但需要保留版权与许可证声明，并且软件按“原样”提供、不提供担保。

公开仓库前建议确认：

- `.env`、数据库、模型文件、日志和本地凭证没有被提交。
- 第三方数据和服务只在运行时调用，不在仓库中重新分发原始行情、公告、新闻或研报。
- README 中保留投资风险、数据服务和开源引用说明。

## 免责声明

本项目仅用于个人学习、量化研究和家庭辅助提醒。股票市场有风险，任何模型、因子、新闻摘要、回测报告或 LLM 回复都可能出错，也可能因为数据源延迟/限流/字段变化而失效。使用者应自行核验公告、财报、交易所披露和券商/交易系统信息，并自行承担投资决策后果。

---

## English

### StockWatch — A-share stock watchlist and research assistant

StockWatch is a lightweight personal stock monitoring bot for China A-share research. It fetches market data, news, announcements, research reports, fund-flow signals and financial snapshots, then combines rule-based analysis, technical indicators, Alpha-style factors, a LightGBM ranking model and an LLM to produce readable Feishu/Lark messages.

The output is designed for non-technical users: a directional view, risk reminders and observation price levels, instead of raw quantitative jargon.

> For learning, research and personal household reminders only. Nothing in this project is investment advice.

### Highlights

- Scheduled A-share watchlist analysis before market open, at noon and after close.
- Feishu/Lark bot commands for stock lookup, position tracking, buy-price alerts and natural-language stock questions.
- Natural-language market questions such as "How is the market today?".
- Research-style replies for questions such as "How is 600449 doing this week?" or "What is the restructuring status of Ningxia Building Materials?".
- Source-aware context from company announcements, exchange-style disclosures, news, research reports, fund flow, financial data and market attention.
- Alpha158/Alpha300-style pandas factors covering momentum, volatility, beta, liquidity shock, relative strength, drawdown and volume structure.
- Optional LightGBM LambdaRank model trained offline and used online as an auxiliary signal.
- Local read-only dashboard and signal quality report based on SQLite history.
- Docker Compose deployment for the daemon, dashboard and optional bot service.
- Deployable on macOS or Linux with SQLite storage; the author's 24/7 instance runs on a Raspberry Pi 5.

### Real Output

Natural-language question example: `我想看看贵州茅台行情怎么样` ("I want to check how Kweichow Moutai is doing"). The bot replies with a conclusion, announcements/news, recent price action, fund-flow/fundamental/attention context, medium-term view, cautious directional notes, key risks and source references.

![Feishu natural-language research reply, top](docs/assets/feishu-research-top.png)

![Feishu natural-language research reply, bottom](docs/assets/feishu-research-bottom.png)

### Quick Start

#### 1. Five-minute terminal demo

Run a terminal demo before setting up Feishu/Lark. Without a MiniMax API key, StockWatch falls back to a rule-based market or stock snapshot. With MiniMax configured, it returns the full natural-language research answer.

```bash
# Enter the repository after cloning it
cd StockWatch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Natural-language stock question
python main.py demo "600519 最近一周走势如何"

# General market question
python main.py demo "现在行情怎么样"
```

#### 2. Configure credentials

```bash
cp .env.example .env
nano .env
```

Required values:

- `MINIMAX_API_KEY` — MiniMax API key
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` — Feishu/Lark self-built app credentials
- `FEISHU_RECEIVE_ID` — receiver `open_id`, `user_id` or email

Edit `.env` with your MiniMax and Feishu/Lark credentials, then run:

```bash
python main.py test
python main.py once
python main.py bot
python main.py dashboard
python main.py report --horizon 5
```

#### 3. Configure the watchlist

```bash
# Edit WATCHLIST in .env. Use comma-separated six-digit stock or ETF codes.
WATCHLIST=600519,000858,510300,510500,159915
```

#### 4. Docker Compose

```bash
cp .env.example .env
nano .env

# Scheduled daemon + dashboard
docker compose up -d stockwatch dashboard

# Optional Feishu/Lark long-connection bot
docker compose --profile bot up -d

# Open the dashboard
open http://127.0.0.1:8765
```

Container data is stored in the `stockwatch-data` volume, mapped to `/root/.stockwatch` inside the containers.

#### 5. systemd deployment

```bash
scripts/install.sh
```

For systemd deployment, see `scripts/install.sh`, `scripts/stockwatch.service` and `scripts/stockwatch-bot.service`.

### Logs

```bash
tail -f ~/.stockwatch/logs/stockwatch_$(date +%Y%m%d).log
```

Logs are stored under `~/.stockwatch/logs/` and retained for seven days.

### SQLite Storage

```bash
sqlite3 ~/.stockwatch/db.sqlite
sqlite3 ~/.stockwatch/db.sqlite "SELECT * FROM runs ORDER BY run_ts DESC LIMIT 5;"
sqlite3 ~/.stockwatch/db.sqlite "SELECT run_id, code, name, action, confidence, pushed FROM decisions ORDER BY run_ts DESC LIMIT 20;"
```

### Local Dashboard

```bash
python main.py dashboard
```

Default URL: `http://127.0.0.1:8765`.

The dashboard is read-only and reads directly from the local SQLite database. It shows recent runs, recent signals, active tracked positions, active price alerts and a five-trading-day signal review summary.

With Docker:

```bash
docker compose up -d dashboard
```

### Signal Review Report

```bash
# Print a five-trading-day signal report
python main.py report --horizon 5

# Write the report to Markdown
python main.py report --horizon 5 --output reports/backtest.md
```

The report uses local `decisions` and `daily_kline` records to calculate sample count, hit rate, average forward-window return and median forward-window return for BUY/SELL/HOLD signals. The hit-rate definition is intentionally simple: BUY is counted as a hit when the forward-window return is positive, SELL when it is negative, and HOLD when the forward-window return stays within ±2%.

This is a research review only. It is not a profit claim and does not modify the database.

### Bot Examples

```text
600519
现在行情怎么样
600449 最近一周走势如何
宁夏建材重组怎么样
买入 600519 1680 100股
盯买 600519 1500
取消盯价 600519
卖出 600519
```

Sending only a stock code returns an immediate single-stock analysis. Natural-language questions enter the research flow: if a stock is recognized, StockWatch answers about that stock; if no stock is recognized, it answers a general market question. `买入` starts position tracking, `盯买` creates a buy-price alert, and `卖出` stops tracking.

### Schedule

Default scheduled run times in `Asia/Shanghai`:

- `09:10` before market open: overnight context and daily watch plan
- `12:30` midday: morning-session review and afternoon notes
- `15:15` after close: full-day review and next-day observation points

Non-trading days are skipped automatically.

### v2 Feature Flags

The v2 modules are off by default:

```bash
ENABLE_CALIBRATION=false
ENABLE_ALPHA158=false
ENABLE_LGBM=false
ENABLE_REGIME=false
ENABLE_SECTOR=false
```

Suggested enablement order:

```text
ENABLE_REGIME -> ENABLE_SECTOR -> ENABLE_ALPHA158 -> ENABLE_CALIBRATION -> ENABLE_LGBM
```

For LightGBM offline training:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt

python scripts/bootstrap_history.py
python scripts/build_training_set.py
python scripts/train_lgbm.py
```

Copy `models/lgbm.txt` and `models/lgbm_meta.json` to the deployment machine under `~/.stockwatch/models/`, then set `ENABLE_LGBM=true`.

### Roadmap / Contributions Welcome

Already included:

- Terminal `demo` mode, so users can try the project before setting up Feishu/Lark.
- Docker Compose deployment.
- Local read-only dashboard.
- Historical signal review report.
- Clearer data-source, service-boundary and investment-risk documentation.

Contribution ideas:

- WeCom, Telegram, DingTalk or other notification adapters. The current maintained and personally verified channel is Feishu/Lark, so other channels are welcome as issues or pull requests.
- Richer dashboard views, such as stock detail pages, signal trend charts and position curves.
- More report dimensions, such as sector-level, confidence-bucket and market-regime breakdowns.
- Lower-friction setup checks for Feishu/Lark permissions, receiver IDs and model connectivity.

### Data, Model and Services

- Market and research data are fetched at runtime mainly through AKShare-wrapped public endpoints, Tencent Finance-style quote endpoints, CNINFO-style announcements and Eastmoney-style public pages.
- Third-party market data, announcements, news and research reports are not redistributed in this repository.
- The LightGBM model is trained offline with historical A-share data and is only used as an auxiliary ranking signal.
- StockWatch does not connect to brokerage accounts and does not place trades.
- `.env`, SQLite databases, logs, model files and Feishu/Lark or MiniMax credentials should stay local or in private deployment environments. Do not commit them to public repositories.
- LLM output may be delayed, incomplete or wrong. Third-party data sources may be delayed, rate-limited or change fields without notice. Users should verify exchange disclosures, company filings and brokerage/trading system data independently.

### Open Source Attribution

This project directly depends on or conceptually references the following open-source projects:

| Project | Usage | License |
| --- | --- | --- |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | Conceptual reference for Alpha158 naming and quantitative research workflow; StockWatch uses an independent pandas implementation | MIT |
| [AKShare](https://github.com/akfamily/akshare) | A-share market data, news, announcements, fund flow, sectors and financial data | MIT |
| [LightGBM](https://github.com/microsoft/LightGBM) | LambdaRank model training and inference | MIT |
| [pandas](https://github.com/pandas-dev/pandas) | DataFrames, time series and training set construction | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | Numerical computation and factor calculation | BSD-3-Clause |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Model evaluation metrics such as NDCG | BSD-3-Clause |
| [Apache Arrow / pyarrow](https://github.com/apache/arrow) | Parquet training data IO | Apache-2.0 |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | OpenAI-compatible client, currently used with MiniMax API | Apache-2.0 |
| [Lark/Feishu OpenAPI Python SDK](https://github.com/larksuite/oapi-sdk-python) | Feishu/Lark bot and OpenAPI integration | MIT |
| [Requests](https://github.com/psf/requests) | HTTP requests | Apache-2.0 |
| [urllib3](https://github.com/urllib3/urllib3) | HTTP connection pooling | MIT |
| [Loguru](https://github.com/Delgan/loguru) | Logging | MIT |
| [Tenacity](https://github.com/jd/tenacity) | Retry logic for LLM/API calls | Apache-2.0 |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` configuration loading | BSD-3-Clause |
| [tqdm](https://github.com/tqdm/tqdm) | Training/download progress bars | MPL-2.0 / MIT |

StockWatch is not affiliated with, endorsed by or commercially partnered with the maintainers or providers listed above.

### License

StockWatch is released under the [MIT License](LICENSE). MIT is a practical choice for this repository because it is permissive, easy to understand, compatible with the major dependencies used here and suitable for personal tools, research prototypes and community forks.

### Disclaimer

This project is for personal learning, quantitative research and household reminders only. Stock markets are risky. Models, factors, signal reports, news summaries and LLM responses can be wrong, and data sources may fail because of delay, rate limits or field changes. Users are responsible for verifying primary sources and making their own investment decisions.

---

## 卸载

```bash
# 停止服务
systemctl stop stockwatch
systemctl disable stockwatch

# 删除文件
rm -rf ~/stockwatch ~/.stockwatch

# 删除 systemd 单元
sudo rm /etc/systemd/system/stockwatch.service
sudo systemctl daemon-reload
```

---

## 飞书权限配置

1. 打开 [飞书开放平台](https://open.feishu.cn/app) → 找到你的应用
2. 权限管理 → 开启：
   - `im:message`（发送消息）
   - `im:message:send_as_bot`（以机器人发送）
3. 版本发布（必须发布才生效）

---

## 问题排查

```bash
# 查看服务状态
systemctl status stockwatch

# 手动跑一次看完整日志
cd ~/stockwatch && source .venv/bin/activate && python main.py once

# AKShare 数据获取失败？先更新：
pip install -U akshare

# 飞书 401/99991663 错误？
# → 检查 FEISHU_APP_ID / FEISHU_APP_SECRET 是否正确
# → 检查飞书开放平台是否已发布版本
```
