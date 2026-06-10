# StockWatch — A股智能盯盘系统

> 可部署在 macOS 或 Linux 服务器上；作者当前将它跑在树莓派 5 上作为 24 小时家庭盯盘服务。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ranking-00A35C)](https://github.com/microsoft/LightGBM)
[![AKShare](https://img.shields.io/badge/Data-AKShare-blue)](https://github.com/akfamily/akshare)
[![Feishu](https://img.shields.io/badge/Bot-Feishu%2FLark-00D6B9)](https://github.com/larksuite/oapi-sdk-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

中文 | [English](#english)

**关键词**：A股量化、股票盯盘、股票机器人、飞书机器人、A股新闻分析、股票公告解读、LightGBM 排序模型、Alpha158 因子、AKShare、macOS/Linux 部署、树莓派 24h 服务。

StockWatch 是一个面向 A 股个人投资辅助场景的轻量级系统：用 AKShare、腾讯财经等数据源获取行情/新闻/公告，结合技术面、消息面、Alpha 因子和 LightGBM 排序模型，最终通过飞书机器人给出“偏向建议 + 风险提醒 + 观察价位”的中文解释。

> 仅供学习、研究和家庭辅助决策使用，不构成任何投资建议。

---

## 功能亮点

- **A股盯盘推送**：按早盘前、午间、收盘后自动运行，非交易日跳过。
- **飞书交互式查询**：支持股票代码、股票名称、买入跟踪、盯价、取消盯价等自然输入。
- **深度问答**：可回答“600449 最近一周走势如何”“宁夏建材重组怎么样”等问题，并优先引用公告/新闻来源。
- **消息面分析**：抓取近况新闻、公司公告、研报、资金流、财务快照和市场关注信息，交给模型生成可读建议。
- **量化因子**：计算扩展版 Alpha158/Alpha300 风格因子，覆盖动量、波动、Beta、流动性冲击、相对强弱、回撤和成交量结构。
- **LightGBM 排序模型**：离线训练 A 股横截面排序模型，线上作为辅助信号参与解释。
- **macOS/Linux 部署**：systemd/命令行常驻服务 + SQLite 本地存储；作者使用树莓派 5 低成本 24 小时运行。

---

## 快速开始

### 1. 配置凭证

```bash
nano ~/stockwatch/.env
```

必填项：
- `MINIMAX_API_KEY` — MiniMax API 密钥
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` — 飞书自建应用凭证
- `FEISHU_RECEIVE_ID` — 接收人 open_id/user_id/email

### 2. 修改自选股

```bash
# 编辑 .env 中的 WATCHLIST（逗号分隔代码）
WATCHLIST=600519,000858,510300,510500,159915
# 格式：6位股票代码，ETF同理（159915=创业板ETF）
# 保存后重启服务：systemctl restart stockwatch
```

### 3. 运行模式

```bash
cd ~/stockwatch && source .venv/bin/activate

# 自检（AKShare / MiniMax / 飞书）
python main.py test

# 立即运行一次完整流程
python main.py once

# 守护进程模式（按调度自动运行）
systemctl start stockwatch
systemctl status stockwatch

# 飞书交互式查询机器人（SDK 长连接）
python main.py bot
```

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
买入 600519 1680
买入 600519 1680 100股
盯买 600519 1500
盯买 600519 1500 100股
取消盯价 600519
卖出 600519
停止跟踪 600519
```

`股票代码` 会即时回复单只股票分析；`买入` 会写入持仓跟踪，后续定时盯盘会把这只股票加入分析池，触发止损、接近目标价或模型转为 SELL 时主动推送；`卖出` 会停止跟踪。

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
- 模型分析可能存在延迟、缺失、误判或源数据错误，不应作为自动交易或唯一投资依据。

## 许可证

本项目采用 [MIT License](LICENSE)。MIT 对个人工具、研究项目和二次开发比较友好：允许使用、复制、修改、分发和商用，但需要保留版权与许可证声明，并且软件按“原样”提供、不提供担保。

公开仓库前建议确认：

- `.env`、数据库、模型文件、日志和本地凭证没有被提交。
- 第三方数据和服务只在运行时调用，不在仓库中重新分发原始行情、公告、新闻或研报。
- README 中保留投资风险、数据服务和开源引用说明。

## 免责声明

本项目仅用于个人学习、量化研究和家庭辅助提醒。股票市场有风险，任何模型、因子、新闻摘要或 LLM 回复都可能出错。使用者应自行核验公告、财报、交易所披露和券商/交易系统信息，并自行承担投资决策后果。

---

## English

### StockWatch — A-share stock watchlist and research assistant

StockWatch is a lightweight personal stock monitoring system for China A-share research. It fetches market data, news, announcements, research reports, fund-flow signals and financial snapshots, then combines rule-based analysis, technical indicators, Alpha-style factors, a LightGBM ranking model and an LLM to produce readable Feishu/Lark messages.

The output is designed for non-technical users: a directional view, risk reminders and observation price levels, instead of raw quantitative jargon.

> For learning, research and personal household reminders only. Nothing in this project is investment advice.

### Highlights

- Scheduled A-share watchlist analysis before market open, at noon and after close.
- Feishu/Lark bot commands for stock lookup, position tracking, buy-price alerts and natural-language stock questions.
- Research-style replies for questions such as "How is 600449 doing this week?" or "What is the restructuring status of Ningxia Building Materials?".
- Source-aware context from company announcements, exchange-style disclosures, news, research reports, fund flow, financial data and market attention.
- Alpha158/Alpha300-style pandas factors covering momentum, volatility, beta, liquidity shock, relative strength, drawdown and volume structure.
- Optional LightGBM LambdaRank model trained offline and used online as an auxiliary signal.
- Deployable on macOS or Linux with SQLite storage; the author's 24/7 instance runs on a Raspberry Pi 5.

### Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your MiniMax and Feishu/Lark credentials, then run:

```bash
python main.py test
python main.py once
python main.py bot
```

For systemd deployment, see `scripts/install.sh`, `scripts/stockwatch.service` and `scripts/stockwatch-bot.service`.

### Bot Examples

```text
600519
600449 最近一周走势如何
宁夏建材重组怎么样
买入 600519 1680 100股
盯买 600519 1500
取消盯价 600519
卖出 600519
```

### Data, Model and Services

- Market and research data are fetched at runtime mainly through AKShare-wrapped public endpoints, Tencent Finance-style quote endpoints, CNINFO-style announcements and Eastmoney-style public pages.
- Third-party market data, announcements, news and research reports are not redistributed in this repository.
- The LightGBM model is trained offline with historical A-share data and is only used as an auxiliary ranking signal.
- LLM output may be delayed, incomplete or wrong. Users should verify exchange disclosures, company filings and brokerage/trading system data independently.

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

This project is for personal learning, quantitative research and household reminders only. Stock markets are risky. Models, factors, news summaries and LLM responses can be wrong. Users are responsible for verifying primary sources and making their own investment decisions.

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
